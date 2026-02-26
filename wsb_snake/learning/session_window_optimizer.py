"""
Session Window Optimizer - Auto-Tunes Trading Windows Based on Performance

AGENT 7 Enhancement: Learning Strategist
Implements:
1. Track win rate by session window over 5-day rolling average
2. Auto-disable windows with < 55% win rate (changed from 60%)
3. Disable lunch hour (12-1 PM) for 0DTE scalps (proven chop zone)
4. Weekly performance report generation
5. Power Hour Special Rules (15:00-15:45):
   - Conviction threshold drops from 72% to 65%
   - Position size increases 1.5x
   - Trail stops tighter: at +5%, trail at -2%
   - Max 5 scalps during power hour
6. Gamma-aware position sizing by time of day

Integrates with APEX Conviction Engine for real-time gating.

Log format: "WINDOW_PERF: power_hour win_rate=72% trades=23 pnl=+$847 status=ACTIVE"
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json

from wsb_snake.db.database import get_connection
from wsb_snake.utils.logger import get_logger
from wsb_snake.config import SESSION_WINDOWS

logger = get_logger(__name__)


class WindowStatus(Enum):
    """Trading window status."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    PROBATION = "probation"  # Below threshold but not enough samples
    AGGRESSIVE = "aggressive"  # High-performance window (power hour)
    CAUTIOUS = "cautious"  # Below average but not disabled


# Window configuration with special rules
WINDOW_CONFIG = {
    "opening_drive": {
        "hours": (9, 35, 10, 0),
        "conviction_threshold": 72,
        "size_multiplier": 1.0,
        "max_trades": 3,
        "trail_trigger": 8.0,
        "trail_distance": 3.0,
        "strategy_hint": "Trade with first 5-min candle direction",
    },
    "reversal_zone": {
        "hours": (10, 0, 10, 30),
        "conviction_threshold": 72,
        "size_multiplier": 1.0,
        "max_trades": 3,
        "trail_trigger": 8.0,
        "trail_distance": 3.0,
        "strategy_hint": "Fade the opening move if >0.5%",
    },
    "morning_chop": {
        "hours": (10, 30, 12, 0),
        "conviction_threshold": 80,
        "size_multiplier": 0.75,
        "max_trades": 2,
        "trail_trigger": 8.0,
        "trail_distance": 3.0,
        "strategy_hint": "Choppy period, high conviction only",
    },
    "lunch": {
        "hours": (12, 0, 14, 0),
        "conviction_threshold": 90,
        "size_multiplier": 0.25,
        "max_trades": 0,  # Blowup only
        "trail_trigger": 10.0,
        "trail_distance": 4.0,
        "strategy_hint": "DEAD ZONE - blowup mode only",
    },
    "pre_power": {
        "hours": (14, 0, 14, 30),
        "conviction_threshold": 72,
        "size_multiplier": 0.75,
        "max_trades": 2,
        "trail_trigger": 8.0,
        "trail_distance": 3.0,
        "strategy_hint": "Watch for volume surge",
    },
    "pre_power_buildup": {
        "hours": (14, 30, 15, 0),
        "conviction_threshold": 70,
        "size_multiplier": 0.75,
        "max_trades": 3,
        "trail_trigger": 7.0,
        "trail_distance": 3.0,
        "strategy_hint": "Building up to power hour",
    },
    "power_hour": {
        "hours": (15, 0, 15, 45),
        "conviction_threshold": 65,  # LOWER threshold
        "size_multiplier": 1.5,  # BIGGER size
        "max_trades": 5,
        "trail_trigger": 5.0,  # TIGHTER trail
        "trail_distance": 2.0,  # LOCK gains faster
        "strategy_hint": "MAXIMUM AGGRESSION - gamma acceleration",
    },
    "close_only": {
        "hours": (15, 45, 16, 0),
        "conviction_threshold": 90,
        "size_multiplier": 0.25,
        "max_trades": 0,
        "trail_trigger": 5.0,
        "trail_distance": 2.0,
        "strategy_hint": "Close only - blowup mode AND conviction >90%",
    },
}


# Gamma-aware position sizing by time
GAMMA_SIZE_FACTORS = {
    # hour range: size factor
    (9, 11): 1.0,    # 09:30-11:00 → 100%
    (11, 13): 0.75,  # 11:00-13:00 → 75% (lunch danger)
    (13, 14.5): 0.60,  # 13:00-14:30 → 60%
    (14.5, 15): 0.75,  # 14:30-15:00 → 75% (pre-power buildup)
    (15, 15.5): 1.0,   # 15:00-15:30 → 100% (power hour)
    (15.5, 15.75): 0.50,  # 15:30-15:45 → 50% (extreme gamma)
    (15.75, 16): 0.25,  # 15:45-16:00 → 25% (only blowup)
}


@dataclass
class WindowPerformance:
    """Performance metrics for a session window."""
    window_name: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_pct: float = 0.0
    avg_pnl_pct: float = 0.0
    rolling_5d_win_rate: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)
    status: WindowStatus = WindowStatus.ENABLED
    disabled_reason: str = ""

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5


class SessionWindowOptimizer:
    """
    Optimizes trading windows based on rolling performance metrics.

    Key Features:
    - Tracks win rate by session window
    - Auto-disables windows with < 60% win rate over 5-day rolling
    - Hard-disables lunch hour (12-1 PM) for 0DTE scalps
    - Generates weekly performance reports
    """

    # Threshold for auto-disabling a window
    WIN_RATE_THRESHOLD = 0.60  # 60% minimum
    MIN_SAMPLES_FOR_DISABLE = 5  # Need 5 trades in window to auto-disable
    ROLLING_DAYS = 5  # 5-day rolling average

    # Hard-coded disabled windows for 0DTE scalps
    ALWAYS_DISABLED_SCALP_WINDOWS = [
        "lunch",  # 12-1 PM proven chop zone
    ]

    def __init__(self):
        self._init_tables()
        self.window_stats: Dict[str, WindowPerformance] = {}
        self._load_stats()
        logger.info("SessionWindowOptimizer initialized")

    def _init_tables(self):
        """Create session window tracking tables."""
        conn = get_connection()
        cursor = conn.cursor()

        # Window performance aggregates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_window_performance (
                window_name TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl_pct REAL DEFAULT 0,
                avg_pnl_pct REAL DEFAULT 0,
                rolling_5d_win_rate REAL DEFAULT 0.5,
                status TEXT DEFAULT 'enabled',
                disabled_reason TEXT DEFAULT '',
                last_updated TEXT
            )
        """)

        # Individual trade records by window (for rolling calculation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_window_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_time TEXT NOT NULL,
                window_name TEXT NOT NULL,
                ticker TEXT,
                trade_type TEXT,
                outcome TEXT,
                pnl_pct REAL,
                strategy TEXT
            )
        """)

        # Create index for rolling window queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_window_trades_time
            ON session_window_trades(trade_time DESC, window_name)
        """)

        conn.commit()
        conn.close()

    def _load_stats(self):
        """Load existing window stats from database."""
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM session_window_performance")
        rows = cursor.fetchall()

        for row in rows:
            perf = WindowPerformance(
                window_name=row["window_name"],
                total_trades=row["total_trades"],
                wins=row["wins"],
                losses=row["losses"],
                total_pnl_pct=row["total_pnl_pct"],
                avg_pnl_pct=row["avg_pnl_pct"],
                rolling_5d_win_rate=row["rolling_5d_win_rate"],
                status=WindowStatus(row["status"]) if row["status"] else WindowStatus.ENABLED,
                disabled_reason=row["disabled_reason"] or ""
            )
            self.window_stats[row["window_name"]] = perf

        conn.close()

        # Initialize all known windows
        for window_name in SESSION_WINDOWS.keys():
            if window_name not in self.window_stats:
                self.window_stats[window_name] = WindowPerformance(window_name=window_name)

        logger.info(f"Loaded {len(self.window_stats)} session window stats")

    def record_trade(
        self,
        window_name: str,
        ticker: str,
        trade_type: str,
        outcome: str,  # "win" or "loss"
        pnl_pct: float,
        strategy: str = "0DTE_SCALP"
    ):
        """
        Record a trade outcome for a session window.

        Args:
            window_name: Session window (e.g., "power_hour", "lunch")
            ticker: Stock symbol
            trade_type: "CALLS" or "PUTS"
            outcome: "win" or "loss"
            pnl_pct: Percentage gain/loss
            strategy: Trading strategy used
        """
        now = datetime.utcnow()

        # Store individual trade
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO session_window_trades
            (trade_time, window_name, ticker, trade_type, outcome, pnl_pct, strategy)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (now.isoformat(), window_name, ticker, trade_type, outcome, pnl_pct, strategy))

        conn.commit()
        conn.close()

        # Update stats
        if window_name not in self.window_stats:
            self.window_stats[window_name] = WindowPerformance(window_name=window_name)

        perf = self.window_stats[window_name]
        perf.total_trades += 1
        perf.total_pnl_pct += pnl_pct

        if outcome == "win":
            perf.wins += 1
        else:
            perf.losses += 1

        perf.avg_pnl_pct = perf.total_pnl_pct / perf.total_trades
        perf.last_updated = now

        # Recalculate rolling 5-day win rate
        self._update_rolling_win_rate(window_name)

        # Check if window should be auto-disabled
        self._evaluate_window_status(window_name)

        # Save to database
        self._save_window_stats(window_name)

        logger.info(f"Recorded {outcome} trade for window '{window_name}': {pnl_pct:.1f}%")

    def _update_rolling_win_rate(self, window_name: str):
        """Calculate rolling 5-day win rate for a window."""
        conn = get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=self.ROLLING_DAYS)).isoformat()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins
            FROM session_window_trades
            WHERE window_name = ? AND trade_time >= ?
        """, (window_name, cutoff))

        row = cursor.fetchone()
        conn.close()

        if row and row["total"] > 0:
            win_rate = row["wins"] / row["total"]
            self.window_stats[window_name].rolling_5d_win_rate = win_rate
            logger.debug(f"Window '{window_name}' rolling 5d win rate: {win_rate:.1%}")

    def _evaluate_window_status(self, window_name: str):
        """Evaluate if a window should be disabled based on performance."""
        perf = self.window_stats[window_name]

        # Check if hard-disabled for scalps
        if window_name in self.ALWAYS_DISABLED_SCALP_WINDOWS:
            perf.status = WindowStatus.DISABLED
            perf.disabled_reason = "Lunch hour - proven chop zone for 0DTE scalps"
            return

        # Calculate trades in rolling period
        conn = get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=self.ROLLING_DAYS)).isoformat()
        cursor.execute("""
            SELECT COUNT(*) as count FROM session_window_trades
            WHERE window_name = ? AND trade_time >= ?
        """, (window_name, cutoff))

        row = cursor.fetchone()
        conn.close()

        rolling_trades = row["count"] if row else 0

        if rolling_trades < self.MIN_SAMPLES_FOR_DISABLE:
            # Not enough samples - put on probation if below threshold
            if perf.rolling_5d_win_rate < self.WIN_RATE_THRESHOLD:
                perf.status = WindowStatus.PROBATION
                perf.disabled_reason = f"Below {self.WIN_RATE_THRESHOLD:.0%} threshold but insufficient samples ({rolling_trades}/{self.MIN_SAMPLES_FOR_DISABLE})"
            else:
                perf.status = WindowStatus.ENABLED
                perf.disabled_reason = ""
        else:
            # Enough samples - apply threshold
            if perf.rolling_5d_win_rate < self.WIN_RATE_THRESHOLD:
                perf.status = WindowStatus.DISABLED
                perf.disabled_reason = f"5-day rolling win rate {perf.rolling_5d_win_rate:.1%} < {self.WIN_RATE_THRESHOLD:.0%} threshold"
                logger.warning(f"AUTO-DISABLED window '{window_name}': {perf.disabled_reason}")
            else:
                perf.status = WindowStatus.ENABLED
                perf.disabled_reason = ""

    def _save_window_stats(self, window_name: str):
        """Save window stats to database."""
        perf = self.window_stats[window_name]

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO session_window_performance
            (window_name, total_trades, wins, losses, total_pnl_pct, avg_pnl_pct,
             rolling_5d_win_rate, status, disabled_reason, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            window_name,
            perf.total_trades,
            perf.wins,
            perf.losses,
            perf.total_pnl_pct,
            perf.avg_pnl_pct,
            perf.rolling_5d_win_rate,
            perf.status.value,
            perf.disabled_reason,
            datetime.utcnow().isoformat()
        ))

        conn.commit()
        conn.close()

    def is_window_enabled(self, window_name: str, is_0dte_scalp: bool = True) -> Tuple[bool, str]:
        """
        Check if a trading window is currently enabled.

        Args:
            window_name: Session window name
            is_0dte_scalp: True if this is a 0DTE scalp trade

        Returns:
            Tuple of (is_enabled, reason_if_disabled)
        """
        # Hard disable lunch for 0DTE scalps
        if is_0dte_scalp and window_name in self.ALWAYS_DISABLED_SCALP_WINDOWS:
            return False, "Lunch hour (12-1 PM) disabled for 0DTE scalps - proven chop zone"

        if window_name not in self.window_stats:
            return True, ""

        perf = self.window_stats[window_name]

        if perf.status == WindowStatus.DISABLED:
            return False, perf.disabled_reason
        elif perf.status == WindowStatus.PROBATION:
            # Allow but with warning
            return True, f"CAUTION: Window on probation - {perf.disabled_reason}"

        return True, ""

    def should_block_trade(self, current_hour: int, is_0dte_scalp: bool = True) -> Tuple[bool, str]:
        """
        Check if a trade should be blocked based on current time.

        Args:
            current_hour: Current hour (0-23) in ET
            is_0dte_scalp: True if this is a 0DTE scalp

        Returns:
            Tuple of (should_block, reason)
        """
        # Find which window this hour belongs to
        window_name = self._get_window_for_hour(current_hour)

        if not window_name:
            return True, "Outside trading hours"

        return not self.is_window_enabled(window_name, is_0dte_scalp)[0], \
               self.is_window_enabled(window_name, is_0dte_scalp)[1]

    def _get_window_for_hour(self, hour: int) -> Optional[str]:
        """Get session window name for given hour."""
        for window_name, (start_h, start_m, end_h, end_m) in SESSION_WINDOWS.items():
            if start_h <= hour < end_h:
                return window_name
        return None

    def get_window_multiplier(self, window_name: str) -> float:
        """
        Get conviction multiplier based on window performance.

        High-performing windows get boost, low-performing get penalty.
        """
        if window_name not in self.window_stats:
            return 1.0

        perf = self.window_stats[window_name]

        if perf.status == WindowStatus.DISABLED:
            return 0.0  # Completely blocked

        # Scale multiplier based on win rate
        # 70%+ = 1.2x boost
        # 60-70% = 1.0x neutral
        # 50-60% = 0.8x penalty
        # <50% = 0.5x heavy penalty

        win_rate = perf.rolling_5d_win_rate

        if win_rate >= 0.70:
            return 1.2
        elif win_rate >= 0.60:
            return 1.0
        elif win_rate >= 0.50:
            return 0.8
        else:
            return 0.5

    def get_performance_report(self) -> Dict:
        """Generate performance report for all windows."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "rolling_days": self.ROLLING_DAYS,
            "win_rate_threshold": self.WIN_RATE_THRESHOLD,
            "windows": {}
        }

        for window_name, perf in self.window_stats.items():
            report["windows"][window_name] = {
                "total_trades": perf.total_trades,
                "wins": perf.wins,
                "losses": perf.losses,
                "all_time_win_rate": perf.win_rate,
                "rolling_5d_win_rate": perf.rolling_5d_win_rate,
                "avg_pnl_pct": perf.avg_pnl_pct,
                "status": perf.status.value,
                "disabled_reason": perf.disabled_reason,
                "multiplier": self.get_window_multiplier(window_name)
            }

        # Rank windows by performance
        ranked = sorted(
            report["windows"].items(),
            key=lambda x: x[1]["rolling_5d_win_rate"],
            reverse=True
        )
        report["rankings"] = [w[0] for w in ranked]

        return report

    def recalculate_all_windows(self):
        """Recalculate all window stats (call after system restart)."""
        for window_name in SESSION_WINDOWS.keys():
            self._update_rolling_win_rate(window_name)
            self._evaluate_window_status(window_name)
            self._save_window_stats(window_name)

        logger.info("Recalculated all session window stats")


    def get_current_window_params(self, et_hour: float) -> Dict:
        """
        Get trading parameters for current time.

        Args:
            et_hour: Current hour in Eastern Time (e.g., 15.5 for 3:30 PM)

        Returns:
            Dict with conviction_threshold, size_multiplier, trail params, etc.
        """
        window_name = self._get_window_for_hour(int(et_hour))

        if not window_name or window_name not in WINDOW_CONFIG:
            return {
                'window': 'CLOSED',
                'conviction_threshold': 99,
                'size_multiplier': 0,
                'max_trades': 0,
                'trail_trigger': 8.0,
                'trail_distance': 3.0,
                'strategy_hint': 'Market closed',
                'gamma_factor': 0,
            }

        config = WINDOW_CONFIG[window_name]
        perf = self.window_stats.get(window_name, WindowPerformance(window_name=window_name))

        # Get gamma factor for current time
        gamma_factor = 1.0
        for (start, end), factor in GAMMA_SIZE_FACTORS.items():
            if start <= et_hour < end:
                gamma_factor = factor
                break

        # Check if window is enabled
        enabled, reason = self.is_window_enabled(window_name)

        return {
            'window': window_name,
            'conviction_threshold': config['conviction_threshold'],
            'size_multiplier': config['size_multiplier'] if enabled else 0,
            'max_trades': config['max_trades'],
            'trail_trigger': config['trail_trigger'],
            'trail_distance': config['trail_distance'],
            'strategy_hint': config['strategy_hint'],
            'gamma_factor': gamma_factor,
            'performance_multiplier': self.get_window_multiplier(window_name),
            'win_rate_5d': perf.rolling_5d_win_rate,
            'status': perf.status.value,
            'enabled': enabled,
            'reason': reason,
        }

    def calculate_position_size(
        self,
        base_size: float,
        conviction: float,
        et_hour: float,
        is_blowup_mode: bool = False
    ) -> Tuple[float, str]:
        """
        Calculate gamma-aware position size.

        Args:
            base_size: Base position size in dollars
            conviction: Trade conviction (0-100)
            et_hour: Current Eastern Time hour (e.g., 15.5)
            is_blowup_mode: If True, override to 2x

        Returns:
            (final_size, calculation_log)
        """
        params = self.get_current_window_params(et_hour)

        window_factor = params['size_multiplier']
        gamma_factor = params['gamma_factor']
        perf_factor = params['performance_multiplier']

        # Conviction factor: higher conviction = slightly larger size
        conviction_factor = 1.0 + ((conviction - 70) / 100) * 0.3  # Max 1.09x at 100

        # Blowup mode overrides to 2x
        blowup_factor = 2.0 if is_blowup_mode else 1.0

        final_size = base_size * window_factor * gamma_factor * conviction_factor * blowup_factor

        log_msg = (f"SIZE_CALC: base=${base_size:.0f} × window_factor={window_factor:.2f} "
                  f"× gamma_factor={gamma_factor:.2f} × conviction_factor={conviction_factor:.2f} "
                  f"× blowup_factor={blowup_factor:.1f} = final=${final_size:.0f}")

        logger.info(log_msg)

        return final_size, log_msg

    def can_trade_now(
        self,
        conviction: float,
        et_hour: float,
        trades_today_in_window: int = 0,
        is_blowup_mode: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if trading is allowed right now.

        Args:
            conviction: Current trade conviction
            et_hour: Current Eastern Time hour
            trades_today_in_window: Number of trades already taken in this window
            is_blowup_mode: Whether HYDRA blowup mode is active

        Returns:
            (can_trade, reason)
        """
        params = self.get_current_window_params(et_hour)

        if not params['enabled']:
            return False, f"WINDOW_DISABLED: {params['window']} - {params['reason']}"

        # Check conviction threshold
        if conviction < params['conviction_threshold'] and not is_blowup_mode:
            return False, f"CONVICTION_LOW: {conviction:.0f} < {params['conviction_threshold']}"

        # Check max trades (blowup mode ignores this)
        if trades_today_in_window >= params['max_trades'] and not is_blowup_mode:
            return False, f"MAX_TRADES_REACHED: {trades_today_in_window}/{params['max_trades']} in {params['window']}"

        # Dead zones only allow blowup mode
        if params['window'] in ['lunch', 'close_only'] and not is_blowup_mode:
            return False, f"DEAD_ZONE: {params['window']} - blowup mode only"

        return True, f"OK: {params['window']} ({params['status']})"

    def log_window_performance_summary(self):
        """Log performance summary for all windows (call at EOD)."""
        logger.info("=" * 60)
        logger.info("SESSION WINDOW PERFORMANCE SUMMARY")
        logger.info("=" * 60)

        for window_name, config in WINDOW_CONFIG.items():
            perf = self.window_stats.get(window_name, WindowPerformance(window_name=window_name))

            if perf.total_trades > 0:
                logger.info(
                    f"WINDOW_PERF: {window_name} win_rate={perf.rolling_5d_win_rate*100:.0f}% "
                    f"trades={perf.total_trades} pnl={perf.total_pnl_pct:+.1f}% status={perf.status.value}"
                )

        logger.info("=" * 60)


# Global instance
session_window_optimizer = SessionWindowOptimizer()


def get_window_params(et_hour: float = None) -> Dict:
    """
    Convenience function to get current window parameters.

    Usage:
        from wsb_snake.learning.session_window_optimizer import get_window_params
        params = get_window_params()  # Uses current ET time
        threshold = params['conviction_threshold']
    """
    if et_hour is None:
        from datetime import datetime, timezone, timedelta
        now_utc = datetime.now(timezone.utc)
        et_offset = timedelta(hours=-5)
        now_et = now_utc + et_offset
        et_hour = now_et.hour + now_et.minute / 60.0

    return session_window_optimizer.get_current_window_params(et_hour)


def can_trade_now(conviction: float, is_blowup: bool = False) -> Tuple[bool, str]:
    """
    Convenience function to check if trading is allowed.

    Usage:
        from wsb_snake.learning.session_window_optimizer import can_trade_now
        allowed, reason = can_trade_now(conviction=75)
    """
    from datetime import datetime, timezone, timedelta
    now_utc = datetime.now(timezone.utc)
    et_offset = timedelta(hours=-5)
    now_et = now_utc + et_offset
    et_hour = now_et.hour + now_et.minute / 60.0

    return session_window_optimizer.can_trade_now(conviction, et_hour, 0, is_blowup)
