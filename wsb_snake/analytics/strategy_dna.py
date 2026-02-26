"""
Strategy DNA - Historical Data Mining & Dynamic Calibration Engine

This module mines all historical trade data to build a "Strategy DNA Profile"
that dynamically calibrates trading parameters based on actual performance.

ALL trading parameters should read from strategy_dna.json instead of hardcoded values.

Runs:
- On startup (load existing DNA or generate new)
- Nightly at market close (recalibrate based on new data)
- On-demand via CLI

Author: WSB Snake Team
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics
import math

from wsb_snake.utils.logger import get_logger
from wsb_snake.db.database import get_connection

logger = get_logger(__name__)

# Paths
DNA_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'strategy_dna.json')
DNA_ARCHIVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'config', 'dna_archive')

# Minimum trades required for statistical significance
MIN_TRADES_FOR_ANALYSIS = 10
MIN_TRADES_PER_BUCKET = 3


@dataclass
class SessionWindow:
    """Performance metrics for a 30-minute session window."""
    window: str
    trade_count: int
    win_rate: float
    avg_pnl: float
    avg_winner: float
    avg_loser: float
    expected_value: float
    status: str  # AGGRESSIVE, ACTIVE, CAUTIOUS, AVOID


@dataclass
class ExitParams:
    """Exit parameters for a conviction tier."""
    tier: str
    target: float
    stop: float
    trail_trigger: float
    trail_distance: float
    avg_winner_size: float
    avg_loser_size: float
    best_exit_reason: str


@dataclass
class LosingFingerprint:
    """Pattern that predicts losing trades."""
    conditions: List[str]
    match_threshold: int
    action: str  # BLOCK, DOWNSIZE, ALERT
    occurrence_rate: float


@dataclass
class SystemHealth:
    """System health metrics for monitoring."""
    sharpe_ratio_20d: float
    max_drawdown_20d: float
    win_rate_trend: str  # IMPROVING, STABLE, DEGRADING
    expectancy_per_trade: float
    consecutive_loss_days: int
    halt_trading: bool
    halt_reason: Optional[str]


@dataclass
class StrategyDNA:
    """Complete Strategy DNA Profile."""
    generated_at: str
    trade_count: int
    overall_win_rate: float
    overall_avg_pnl: float
    optimal_conviction_threshold: int

    # Session windows (30-min blocks)
    session_windows: Dict[str, Dict]

    # Exit params by conviction tier
    exit_params_by_tier: Dict[str, Dict]

    # Direction bias
    direction_bias: Dict[str, float]

    # Optimal hold time
    optimal_hold_seconds: int
    hold_time_analysis: Dict[str, Any]

    # Losing fingerprint
    losing_fingerprint: Dict[str, Any]

    # Day of week effects
    day_effects: Dict[str, float]

    # Kelly criterion
    kelly_fraction: float
    kelly_33pct: float

    # System health
    system_health: Dict[str, Any]

    # Metadata
    data_start_date: str
    data_end_date: str
    last_recalibration: str
    recalibration_changes: List[str]


class StrategyDNAMiner:
    """
    Mines historical trade data to build Strategy DNA Profile.
    """

    def __init__(self):
        self.conn = None
        self.trades: List[Dict] = []
        self.current_dna: Optional[StrategyDNA] = None

    def _get_connection(self):
        """Get database connection."""
        if self.conn is None:
            self.conn = get_connection()
        return self.conn

    def _close_connection(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def extract_trades(self) -> List[Dict]:
        """
        Step 1: Extract all trades from SQLite.
        Combines data from multiple tables for complete picture.
        """
        logger.info("Extracting trade data from database...")
        trades = []

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Primary source: trade_performance table
            cursor.execute("""
                SELECT
                    id,
                    trade_date,
                    symbol as ticker,
                    engine,
                    trade_type as direction,
                    entry_hour,
                    session,
                    pnl,
                    pnl_pct,
                    exit_reason,
                    holding_time_seconds,
                    signal_id,
                    created_at,
                    r_multiple,
                    event_tier
                FROM trade_performance
                ORDER BY created_at
            """)

            for row in cursor.fetchall():
                trade = {
                    'id': row[0],
                    'trade_date': row[1],
                    'ticker': row[2],
                    'engine': row[3],
                    'direction': row[4],  # CALLS or PUTS
                    'entry_hour': row[5],
                    'session': row[6],
                    'pnl': row[7] or 0,
                    'pnl_pct': row[8] or 0,
                    'exit_reason': row[9],
                    'holding_time_seconds': row[10] or 0,
                    'signal_id': row[11],
                    'created_at': row[12],
                    'r_multiple': row[13],
                    'event_tier': row[14],
                    'source': 'trade_performance'
                }
                trades.append(trade)

            # Also check paper_trades for additional data
            cursor.execute("""
                SELECT
                    id,
                    ticker,
                    direction,
                    entry_price,
                    exit_price,
                    exit_reason,
                    pnl,
                    r_multiple,
                    created_at,
                    fill_time,
                    exit_time
                FROM paper_trades
                WHERE status = 'CLOSED' AND pnl IS NOT NULL
                ORDER BY created_at
            """)

            for row in cursor.fetchall():
                # Calculate holding time from fill_time to exit_time
                hold_seconds = 0
                if row[9] and row[10]:  # fill_time and exit_time
                    try:
                        fill = datetime.fromisoformat(row[9].replace('Z', '+00:00'))
                        exit_t = datetime.fromisoformat(row[10].replace('Z', '+00:00'))
                        hold_seconds = int((exit_t - fill).total_seconds())
                    except:
                        pass

                # Determine direction from direction field
                direction = 'CALLS' if row[2] and row[2].lower() in ('long', 'call', 'calls') else 'PUTS'

                # Extract hour from created_at
                entry_hour = 0
                if row[8]:
                    try:
                        dt = datetime.fromisoformat(row[8].replace('Z', '+00:00'))
                        entry_hour = dt.hour
                    except:
                        pass

                # Calculate pnl_pct if we have entry/exit prices
                pnl_pct = 0
                if row[3] and row[4] and row[3] > 0:
                    pnl_pct = ((row[4] - row[3]) / row[3]) * 100

                trade = {
                    'id': f"paper_{row[0]}",
                    'trade_date': row[8][:10] if row[8] else None,
                    'ticker': row[1],
                    'engine': 'paper',
                    'direction': direction,
                    'entry_hour': entry_hour,
                    'session': self._get_session_from_hour(entry_hour),
                    'pnl': row[6] or 0,
                    'pnl_pct': pnl_pct,
                    'exit_reason': row[5],
                    'holding_time_seconds': hold_seconds,
                    'signal_id': None,
                    'created_at': row[8],
                    'r_multiple': row[7],
                    'event_tier': None,
                    'source': 'paper_trades'
                }
                trades.append(trade)

            logger.info(f"Extracted {len(trades)} total trades")
            self.trades = trades
            return trades

        except Exception as e:
            logger.error(f"Error extracting trades: {e}")
            return []
        finally:
            self._close_connection()

    def _get_session_from_hour(self, hour: int) -> str:
        """Convert hour (UTC) to session name."""
        # Convert UTC to ET (approximate)
        et_hour = hour - 5 if hour >= 5 else hour + 19

        if 9 <= et_hour < 10:
            return 'open'
        elif 10 <= et_hour < 12:
            return 'morning'
        elif 12 <= et_hour < 14:
            return 'midday'
        elif 14 <= et_hour < 15:
            return 'afternoon'
        elif 15 <= et_hour < 16:
            return 'power_hour'
        else:
            return 'after_hours'

    def _get_30min_window(self, created_at: str) -> str:
        """Get 30-minute window string from timestamp."""
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            # Convert to ET (approximate)
            et_hour = dt.hour - 5 if dt.hour >= 5 else dt.hour + 19
            minute_block = '00' if dt.minute < 30 else '30'
            next_block = '30' if dt.minute < 30 else '00'
            next_hour = et_hour if dt.minute < 30 else (et_hour + 1)
            return f"{et_hour:02d}:{minute_block}-{next_hour:02d}:{next_block}"
        except:
            return "unknown"

    def _get_day_of_week(self, trade_date: str) -> int:
        """Get day of week (0=Monday, 4=Friday) from trade date."""
        try:
            dt = datetime.strptime(trade_date, '%Y-%m-%d')
            return dt.weekday()
        except:
            return -1

    def compute_features(self) -> List[Dict]:
        """
        Step 2: Compute features for each trade.
        """
        logger.info("Computing trade features...")

        for trade in self.trades:
            # Session window (30-min block)
            trade['session_window'] = self._get_30min_window(trade.get('created_at', ''))

            # Day of week
            trade['day_of_week'] = self._get_day_of_week(trade.get('trade_date', ''))

            # Winner flag
            trade['winner'] = trade.get('pnl', 0) > 0

            # Conviction (placeholder - would need signal data)
            trade['conviction'] = trade.get('r_multiple', 70) or 70

            # Direction normalized
            direction = trade.get('direction', '').upper()
            trade['is_calls'] = direction in ('CALLS', 'CALL', 'LONG', 'BUY')

        return self.trades

    def analyze_session_windows(self) -> Dict[str, SessionWindow]:
        """
        Step 3A: Win rate by session window.
        """
        logger.info("Analyzing session windows...")

        windows = defaultdict(list)
        for trade in self.trades:
            window = trade.get('session_window', 'unknown')
            windows[window].append(trade)

        results = {}
        for window, trades in windows.items():
            if len(trades) < MIN_TRADES_PER_BUCKET:
                continue

            winners = [t for t in trades if t['winner']]
            losers = [t for t in trades if not t['winner']]

            win_rate = len(winners) / len(trades) if trades else 0
            avg_pnl = statistics.mean([t['pnl_pct'] for t in trades]) if trades else 0
            avg_winner = statistics.mean([t['pnl_pct'] for t in winners]) if winners else 0
            avg_loser = statistics.mean([t['pnl_pct'] for t in losers]) if losers else 0

            # Expected value
            ev = (win_rate * avg_winner) - ((1 - win_rate) * abs(avg_loser))

            # Status based on EV and win rate
            if ev > 5 and win_rate > 0.6:
                status = "AGGRESSIVE"
            elif ev > 2 and win_rate > 0.5:
                status = "ACTIVE"
            elif ev > 0:
                status = "CAUTIOUS"
            else:
                status = "AVOID"

            results[window] = SessionWindow(
                window=window,
                trade_count=len(trades),
                win_rate=round(win_rate, 3),
                avg_pnl=round(avg_pnl, 2),
                avg_winner=round(avg_winner, 2),
                avg_loser=round(avg_loser, 2),
                expected_value=round(ev, 2),
                status=status
            )

        return results

    def find_optimal_conviction_threshold(self) -> int:
        """
        Step 3B: Find conviction threshold that maximizes EV.
        """
        logger.info("Finding optimal conviction threshold...")

        if len(self.trades) < MIN_TRADES_FOR_ANALYSIS:
            logger.warning(f"Insufficient trades ({len(self.trades)}) for conviction analysis, using default 70")
            return 70

        best_threshold = 70
        best_ev = float('-inf')

        for threshold in range(60, 90, 2):
            qualifying = [t for t in self.trades if t.get('conviction', 70) >= threshold]
            if len(qualifying) < MIN_TRADES_PER_BUCKET:
                continue

            winners = [t for t in qualifying if t['winner']]
            losers = [t for t in qualifying if not t['winner']]

            win_rate = len(winners) / len(qualifying)
            avg_winner = statistics.mean([t['pnl_pct'] for t in winners]) if winners else 0
            avg_loser = abs(statistics.mean([t['pnl_pct'] for t in losers])) if losers else 0

            ev = (win_rate * avg_winner) - ((1 - win_rate) * avg_loser)

            if ev > best_ev:
                best_ev = ev
                best_threshold = threshold

        return best_threshold

    def analyze_exit_by_conviction_tier(self) -> Dict[str, ExitParams]:
        """
        Step 3C: Exit optimization by conviction tier.
        """
        logger.info("Analyzing exits by conviction tier...")

        tiers = {
            'low': (65, 72),
            'medium': (72, 80),
            'high': (80, 100)
        }

        results = {}

        for tier_name, (min_conv, max_conv) in tiers.items():
            tier_trades = [t for t in self.trades
                          if min_conv <= t.get('conviction', 70) < max_conv]

            if len(tier_trades) < MIN_TRADES_PER_BUCKET:
                # Use defaults based on tier
                if tier_name == 'high':
                    results[tier_name] = ExitParams(
                        tier=tier_name, target=16, stop=-9,
                        trail_trigger=10, trail_distance=4,
                        avg_winner_size=0, avg_loser_size=0,
                        best_exit_reason='TARGET'
                    )
                elif tier_name == 'medium':
                    results[tier_name] = ExitParams(
                        tier=tier_name, target=11, stop=-7,
                        trail_trigger=7, trail_distance=3,
                        avg_winner_size=0, avg_loser_size=0,
                        best_exit_reason='TARGET'
                    )
                else:
                    results[tier_name] = ExitParams(
                        tier=tier_name, target=7, stop=-5,
                        trail_trigger=5, trail_distance=2,
                        avg_winner_size=0, avg_loser_size=0,
                        best_exit_reason='TARGET'
                    )
                continue

            winners = [t for t in tier_trades if t['winner']]
            losers = [t for t in tier_trades if not t['winner']]

            avg_winner = statistics.mean([t['pnl_pct'] for t in winners]) if winners else 10
            avg_loser = abs(statistics.mean([t['pnl_pct'] for t in losers])) if losers else 7

            # Count exit reasons
            exit_counts = defaultdict(int)
            for t in tier_trades:
                if t['winner']:
                    exit_counts[t.get('exit_reason', 'UNKNOWN')] += 1
            best_exit = max(exit_counts, key=exit_counts.get) if exit_counts else 'TARGET'

            # Set targets based on actual performance
            target = min(max(avg_winner * 0.8, 5), 25)  # 80% of avg winner, capped
            stop = min(max(avg_loser * 1.1, 5), 15)  # 110% of avg loser, capped

            results[tier_name] = ExitParams(
                tier=tier_name,
                target=round(target, 1),
                stop=round(-stop, 1),
                trail_trigger=round(target * 0.6, 1),
                trail_distance=round(target * 0.25, 1),
                avg_winner_size=round(avg_winner, 2),
                avg_loser_size=round(avg_loser, 2),
                best_exit_reason=best_exit
            )

        return results

    def analyze_direction_bias(self) -> Dict[str, float]:
        """
        Step 3D: Directional edge analysis.
        """
        logger.info("Analyzing directional bias...")

        calls = [t for t in self.trades if t.get('is_calls', True)]
        puts = [t for t in self.trades if not t.get('is_calls', True)]

        calls_wr = len([t for t in calls if t['winner']]) / len(calls) if calls else 0.5
        puts_wr = len([t for t in puts if t['winner']]) / len(puts) if puts else 0.5

        calls_avg_pnl = statistics.mean([t['pnl_pct'] for t in calls]) if calls else 0
        puts_avg_pnl = statistics.mean([t['pnl_pct'] for t in puts]) if puts else 0

        # If significant difference (>10% win rate gap), add conviction penalty
        puts_penalty = 0
        if calls_wr - puts_wr > 0.10:
            puts_penalty = int((calls_wr - puts_wr) * 50)  # 5 points per 10% gap

        return {
            'calls_win_rate': round(calls_wr, 3),
            'puts_win_rate': round(puts_wr, 3),
            'calls_avg_pnl': round(calls_avg_pnl, 2),
            'puts_avg_pnl': round(puts_avg_pnl, 2),
            'puts_conviction_penalty': puts_penalty,
            'calls_count': len(calls),
            'puts_count': len(puts)
        }

    def analyze_hold_time(self) -> Tuple[int, Dict]:
        """
        Step 3E: Hold time analysis.
        """
        logger.info("Analyzing hold time...")

        if len(self.trades) < MIN_TRADES_FOR_ANALYSIS:
            return 300, {'optimal_range': (180, 420), 'analysis': 'insufficient_data'}

        # Group by hold time buckets
        buckets = defaultdict(list)
        for t in self.trades:
            hold = t.get('holding_time_seconds', 0)
            if hold <= 0:
                continue
            # Bucket by 60-second intervals
            bucket = (hold // 60) * 60
            buckets[bucket].append(t)

        # Find bucket with best avg P&L
        best_bucket = 300  # Default 5 minutes
        best_pnl = float('-inf')

        bucket_stats = {}
        for bucket, trades in sorted(buckets.items()):
            if len(trades) < MIN_TRADES_PER_BUCKET:
                continue
            avg_pnl = statistics.mean([t['pnl_pct'] for t in trades])
            win_rate = len([t for t in trades if t['winner']]) / len(trades)
            bucket_stats[bucket] = {
                'avg_pnl': round(avg_pnl, 2),
                'win_rate': round(win_rate, 3),
                'count': len(trades)
            }
            if avg_pnl > best_pnl:
                best_pnl = avg_pnl
                best_bucket = bucket

        # Winners vs losers hold time
        winners = [t for t in self.trades if t['winner'] and t.get('holding_time_seconds', 0) > 0]
        losers = [t for t in self.trades if not t['winner'] and t.get('holding_time_seconds', 0) > 0]

        avg_winner_hold = statistics.mean([t['holding_time_seconds'] for t in winners]) if winners else 300
        avg_loser_hold = statistics.mean([t['holding_time_seconds'] for t in losers]) if losers else 60

        analysis = {
            'optimal_bucket': best_bucket,
            'bucket_stats': bucket_stats,
            'avg_winner_hold_seconds': round(avg_winner_hold, 0),
            'avg_loser_hold_seconds': round(avg_loser_hold, 0),
            'insight': 'exit_early' if avg_loser_hold > avg_winner_hold else 'hold_longer'
        }

        return int(best_bucket), analysis

    def analyze_losing_fingerprint(self) -> LosingFingerprint:
        """
        Step 3F: Losing streak DNA analysis.
        """
        logger.info("Analyzing losing fingerprint...")

        losers = [t for t in self.trades if not t['winner']]

        if len(losers) < MIN_TRADES_FOR_ANALYSIS:
            return LosingFingerprint(
                conditions=['insufficient_data'],
                match_threshold=99,
                action='ALERT',
                occurrence_rate=0
            )

        # Analyze common features in losers
        afternoon_count = len([t for t in losers if t.get('session') in ('afternoon', 'power_hour')])
        low_conv_count = len([t for t in losers if t.get('conviction', 70) < 72])
        puts_count = len([t for t in losers if not t.get('is_calls', True)])

        total_losers = len(losers)
        conditions = []

        # Check if each condition is overrepresented in losers
        if afternoon_count / total_losers > 0.4:
            conditions.append('afternoon_session')
        if low_conv_count / total_losers > 0.4:
            conditions.append('low_conviction')
        if puts_count / total_losers > 0.6:
            conditions.append('puts_direction')

        # Find losing streaks of 3+
        sorted_trades = sorted(self.trades, key=lambda x: x.get('created_at', ''))
        streak_conditions = []
        current_streak = []

        for trade in sorted_trades:
            if not trade['winner']:
                current_streak.append(trade)
            else:
                if len(current_streak) >= 3:
                    # Extract common features from streak
                    streak_conditions.extend(self._extract_streak_features(current_streak))
                current_streak = []

        # Add most common streak conditions
        if streak_conditions:
            from collections import Counter
            common = Counter(streak_conditions).most_common(3)
            for cond, _ in common:
                if cond not in conditions:
                    conditions.append(cond)

        if not conditions:
            conditions = ['none_identified']

        return LosingFingerprint(
            conditions=conditions[:5],  # Max 5 conditions
            match_threshold=min(len(conditions), 3),
            action='BLOCK' if len(conditions) >= 3 else 'DOWNSIZE',
            occurrence_rate=round(len(losers) / len(self.trades) if self.trades else 0, 3)
        )

    def _extract_streak_features(self, streak: List[Dict]) -> List[str]:
        """Extract common features from a losing streak."""
        features = []

        sessions = [t.get('session') for t in streak]
        directions = [t.get('is_calls') for t in streak]

        # Check for common session
        from collections import Counter
        session_counts = Counter(sessions)
        if session_counts.most_common(1)[0][1] >= len(streak) * 0.7:
            features.append(f"session_{session_counts.most_common(1)[0][0]}")

        # Check for same direction
        if all(directions) or not any(directions):
            features.append('same_direction')

        return features

    def analyze_day_effects(self) -> Dict[str, float]:
        """
        Step 3G: Day of week effect analysis.
        """
        logger.info("Analyzing day of week effects...")

        day_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        day_trades = defaultdict(list)

        for t in self.trades:
            dow = t.get('day_of_week', -1)
            if 0 <= dow <= 4:
                day_trades[dow].append(t)

        # Calculate multipliers relative to overall win rate
        overall_wr = len([t for t in self.trades if t['winner']]) / len(self.trades) if self.trades else 0.5

        effects = {}
        for dow in range(5):
            trades = day_trades.get(dow, [])
            if len(trades) < MIN_TRADES_PER_BUCKET:
                effects[day_names[dow]] = 1.0  # Neutral
                continue

            day_wr = len([t for t in trades if t['winner']]) / len(trades)
            # Multiplier: 1.0 = normal, >1 = increase size, <1 = decrease size
            multiplier = day_wr / overall_wr if overall_wr > 0 else 1.0
            multiplier = max(0.5, min(1.5, multiplier))  # Cap between 0.5x and 1.5x
            effects[day_names[dow]] = round(multiplier, 2)

        return effects

    def calculate_kelly(self) -> Tuple[float, float]:
        """
        Step 3H: Fractional Kelly position sizing.

        Kelly% = W - (1-W)/R
        where W = win rate, R = win/loss ratio
        """
        logger.info("Calculating Kelly criterion...")

        if len(self.trades) < MIN_TRADES_FOR_ANALYSIS:
            return 0.08, 0.027  # Conservative defaults

        winners = [t for t in self.trades if t['winner']]
        losers = [t for t in self.trades if not t['winner']]

        W = len(winners) / len(self.trades)

        avg_win = abs(statistics.mean([t['pnl_pct'] for t in winners])) if winners else 10
        avg_loss = abs(statistics.mean([t['pnl_pct'] for t in losers])) if losers else 10

        R = avg_win / avg_loss if avg_loss > 0 else 1

        # Kelly formula
        kelly = W - (1 - W) / R if R > 0 else 0
        kelly = max(0, min(kelly, 0.25))  # Cap at 25%

        # Use 33% Kelly for conservative sizing
        kelly_33 = kelly * 0.33

        return round(kelly, 4), round(kelly_33, 4)

    def calculate_system_health(self) -> SystemHealth:
        """
        Calculate system health metrics.
        """
        logger.info("Calculating system health...")

        if len(self.trades) < MIN_TRADES_FOR_ANALYSIS:
            return SystemHealth(
                sharpe_ratio_20d=0,
                max_drawdown_20d=0,
                win_rate_trend='INSUFFICIENT_DATA',
                expectancy_per_trade=0,
                consecutive_loss_days=0,
                halt_trading=False,
                halt_reason=None
            )

        # Sort by date
        sorted_trades = sorted(self.trades, key=lambda x: x.get('created_at', ''))

        # Calculate daily P&L
        daily_pnl = defaultdict(float)
        for t in sorted_trades:
            date = t.get('trade_date', 'unknown')
            daily_pnl[date] += t.get('pnl_pct', 0)

        pnl_series = list(daily_pnl.values())

        # Sharpe ratio (annualized)
        if len(pnl_series) >= 2:
            avg_daily = statistics.mean(pnl_series)
            std_daily = statistics.stdev(pnl_series) if len(pnl_series) > 1 else 1
            sharpe = (avg_daily / std_daily) * math.sqrt(252) if std_daily > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for pnl in pnl_series:
            cumulative += pnl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        # Win rate trend (compare recent 10 vs previous 10)
        recent = sorted_trades[-10:] if len(sorted_trades) >= 10 else sorted_trades
        previous = sorted_trades[-20:-10] if len(sorted_trades) >= 20 else []

        recent_wr = len([t for t in recent if t['winner']]) / len(recent) if recent else 0
        previous_wr = len([t for t in previous if t['winner']]) / len(previous) if previous else recent_wr

        if recent_wr > previous_wr + 0.05:
            trend = 'IMPROVING'
        elif recent_wr < previous_wr - 0.05:
            trend = 'DEGRADING'
        else:
            trend = 'STABLE'

        # Expectancy
        winners = [t for t in self.trades if t['winner']]
        losers = [t for t in self.trades if not t['winner']]
        W = len(winners) / len(self.trades) if self.trades else 0
        avg_win = statistics.mean([t['pnl_pct'] for t in winners]) if winners else 0
        avg_loss = abs(statistics.mean([t['pnl_pct'] for t in losers])) if losers else 0
        expectancy = (W * avg_win) - ((1 - W) * avg_loss)

        # Consecutive loss days
        consec_loss = 0
        for date in reversed(sorted(daily_pnl.keys())):
            if daily_pnl[date] < 0:
                consec_loss += 1
            else:
                break

        # Halt conditions
        halt = False
        halt_reason = None
        if consec_loss >= 5:
            halt = True
            halt_reason = f"5 consecutive losing days"
        elif expectancy < -5:
            halt = True
            halt_reason = f"Negative expectancy ({expectancy:.2f})"
        elif max_dd > 20:
            halt = True
            halt_reason = f"Max drawdown exceeded 20% ({max_dd:.1f}%)"

        return SystemHealth(
            sharpe_ratio_20d=round(sharpe, 2),
            max_drawdown_20d=round(max_dd, 2),
            win_rate_trend=trend,
            expectancy_per_trade=round(expectancy, 2),
            consecutive_loss_days=consec_loss,
            halt_trading=halt,
            halt_reason=halt_reason
        )

    def generate_dna(self) -> StrategyDNA:
        """
        Generate complete Strategy DNA Profile.
        """
        logger.info("="*60)
        logger.info("GENERATING STRATEGY DNA PROFILE")
        logger.info("="*60)

        # Extract and compute features
        self.extract_trades()
        self.compute_features()

        # Run all analyses
        session_windows = self.analyze_session_windows()
        optimal_threshold = self.find_optimal_conviction_threshold()
        exit_params = self.analyze_exit_by_conviction_tier()
        direction_bias = self.analyze_direction_bias()
        optimal_hold, hold_analysis = self.analyze_hold_time()
        losing_fp = self.analyze_losing_fingerprint()
        day_effects = self.analyze_day_effects()
        kelly, kelly_33 = self.calculate_kelly()
        health = self.calculate_system_health()

        # Overall stats
        overall_wr = len([t for t in self.trades if t['winner']]) / len(self.trades) if self.trades else 0
        overall_pnl = statistics.mean([t['pnl_pct'] for t in self.trades]) if self.trades else 0

        # Date range
        dates = [t.get('trade_date') for t in self.trades if t.get('trade_date')]
        start_date = min(dates) if dates else 'N/A'
        end_date = max(dates) if dates else 'N/A'

        dna = StrategyDNA(
            generated_at=datetime.utcnow().isoformat(),
            trade_count=len(self.trades),
            overall_win_rate=round(overall_wr, 3),
            overall_avg_pnl=round(overall_pnl, 2),
            optimal_conviction_threshold=optimal_threshold,
            session_windows={k: asdict(v) for k, v in session_windows.items()},
            exit_params_by_tier={k: asdict(v) for k, v in exit_params.items()},
            direction_bias=direction_bias,
            optimal_hold_seconds=optimal_hold,
            hold_time_analysis=hold_analysis,
            losing_fingerprint=asdict(losing_fp),
            day_effects=day_effects,
            kelly_fraction=kelly,
            kelly_33pct=kelly_33,
            system_health=asdict(health),
            data_start_date=start_date,
            data_end_date=end_date,
            last_recalibration=datetime.utcnow().isoformat(),
            recalibration_changes=[]
        )

        self.current_dna = dna
        return dna

    def save_dna(self, dna: StrategyDNA = None) -> str:
        """Save DNA profile to JSON file."""
        if dna is None:
            dna = self.current_dna
        if dna is None:
            raise ValueError("No DNA profile to save")

        # Ensure directory exists
        os.makedirs(os.path.dirname(DNA_CONFIG_PATH), exist_ok=True)

        # Convert to dict
        dna_dict = asdict(dna)

        # Save
        with open(DNA_CONFIG_PATH, 'w') as f:
            json.dump(dna_dict, f, indent=2, default=str)

        logger.info(f"Strategy DNA saved to {DNA_CONFIG_PATH}")
        return DNA_CONFIG_PATH

    def archive_dna(self):
        """Archive current DNA config with timestamp."""
        if not os.path.exists(DNA_CONFIG_PATH):
            return

        os.makedirs(DNA_ARCHIVE_DIR, exist_ok=True)

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        archive_path = os.path.join(DNA_ARCHIVE_DIR, f'strategy_dna_{timestamp}.json')

        import shutil
        shutil.copy(DNA_CONFIG_PATH, archive_path)
        logger.info(f"Archived DNA to {archive_path}")

        # Keep only last 30 archives
        archives = sorted(os.listdir(DNA_ARCHIVE_DIR))
        while len(archives) > 30:
            os.remove(os.path.join(DNA_ARCHIVE_DIR, archives.pop(0)))

    def load_dna(self) -> Optional[StrategyDNA]:
        """Load existing DNA profile."""
        if not os.path.exists(DNA_CONFIG_PATH):
            return None

        try:
            with open(DNA_CONFIG_PATH, 'r') as f:
                data = json.load(f)

            # Convert dict back to StrategyDNA
            # Note: This is simplified - a full implementation would validate all fields
            logger.info(f"Loaded Strategy DNA from {DNA_CONFIG_PATH}")
            return data  # Return as dict for flexibility
        except Exception as e:
            logger.error(f"Error loading DNA: {e}")
            return None


class NightlyRecalibrator:
    """
    Module 7: Nightly Recalibration

    Runs at market close to update Strategy DNA based on new data.
    """

    def __init__(self):
        self.miner = StrategyDNAMiner()
        self.changes: List[str] = []

    def recalibrate(self) -> Dict[str, Any]:
        """
        Run nightly recalibration.

        1. Load previous DNA
        2. Generate new DNA from all data
        3. Compare and log changes >10%
        4. Update strategy_dna.json
        5. Archive old config
        """
        logger.info("="*60)
        logger.info("NIGHTLY RECALIBRATION STARTING")
        logger.info("="*60)

        # Load previous DNA
        previous_dna = self.miner.load_dna()

        # Archive before changes
        self.miner.archive_dna()

        # Generate new DNA
        new_dna = self.miner.generate_dna()

        # Compare and log changes
        self.changes = []
        if previous_dna:
            self._compare_and_log(previous_dna, new_dna)

        # Update recalibration changes in DNA
        new_dna.recalibration_changes = self.changes

        # Save new DNA
        self.miner.save_dna(new_dna)

        # Check system health
        health = new_dna.system_health
        if health.get('halt_trading'):
            self._send_halt_alert(health.get('halt_reason', 'Unknown'))

        logger.info("="*60)
        logger.info("NIGHTLY RECALIBRATION COMPLETE")
        logger.info(f"Changes: {len(self.changes)}")
        for change in self.changes:
            logger.info(f"  RECAL: {change}")
        logger.info("="*60)

        return {
            'status': 'complete',
            'changes': self.changes,
            'health': health,
            'trade_count': new_dna.trade_count
        }

    def _compare_and_log(self, old: Dict, new: StrategyDNA):
        """Compare old and new DNA, log significant changes."""
        new_dict = asdict(new)

        # Key parameters to track
        params_to_check = [
            ('optimal_conviction_threshold', 'conviction_threshold'),
            ('overall_win_rate', 'win_rate'),
            ('kelly_33pct', 'kelly_fraction'),
            ('optimal_hold_seconds', 'hold_time'),
        ]

        for param, label in params_to_check:
            old_val = old.get(param, 0)
            new_val = new_dict.get(param, 0)

            if old_val and new_val:
                pct_change = abs(new_val - old_val) / old_val if old_val != 0 else 0
                if pct_change > 0.10:  # >10% change
                    self.changes.append(f"{label} {old_val}→{new_val} ({pct_change*100:.1f}%)")

        # Check exit params
        old_exits = old.get('exit_params_by_tier', {})
        new_exits = new_dict.get('exit_params_by_tier', {})

        for tier in ['low', 'medium', 'high']:
            old_t = old_exits.get(tier, {})
            new_t = new_exits.get(tier, {})

            for param in ['target', 'stop']:
                old_val = old_t.get(param, 0)
                new_val = new_t.get(param, 0)

                if old_val and new_val:
                    pct_change = abs(new_val - old_val) / abs(old_val) if old_val != 0 else 0
                    if pct_change > 0.10:
                        self.changes.append(f"{tier}_{param} {old_val}→{new_val}")

    def _send_halt_alert(self, reason: str):
        """Send Telegram alert for trading halt."""
        try:
            from wsb_snake.notifications.telegram_bot import send_alert
            send_alert(
                f"🚨 **TRADING HALTED**\n\n"
                f"Reason: {reason}\n\n"
                f"System health metrics have triggered an automatic trading halt. "
                f"Review strategy_dna.json and recent performance before resuming."
            )
        except Exception as e:
            logger.error(f"Failed to send halt alert: {e}")


# Singleton instance
_dna_cache: Optional[Dict] = None


def get_strategy_dna() -> Dict:
    """
    Get current Strategy DNA profile.

    Usage:
        from wsb_snake.analytics.strategy_dna import get_strategy_dna
        dna = get_strategy_dna()
        threshold = dna.get('optimal_conviction_threshold', 70)
    """
    global _dna_cache

    if _dna_cache is None:
        miner = StrategyDNAMiner()
        _dna_cache = miner.load_dna()

        if _dna_cache is None:
            # Generate fresh DNA
            logger.info("No existing DNA found, generating fresh profile...")
            dna = miner.generate_dna()
            miner.save_dna(dna)
            _dna_cache = asdict(dna)

    return _dna_cache or {}


def refresh_dna_cache():
    """Force refresh of DNA cache."""
    global _dna_cache
    _dna_cache = None
    return get_strategy_dna()


def run_nightly_recalibration() -> Dict:
    """Run nightly recalibration (call from scheduler)."""
    recalibrator = NightlyRecalibrator()
    result = recalibrator.recalibrate()
    refresh_dna_cache()
    return result


# CLI entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--recalibrate':
        print("Running nightly recalibration...")
        result = run_nightly_recalibration()
        print(f"\nResult: {json.dumps(result, indent=2)}")
    else:
        print("Generating Strategy DNA Profile...")
        miner = StrategyDNAMiner()
        dna = miner.generate_dna()
        path = miner.save_dna(dna)

        print(f"\n{'='*60}")
        print("STRATEGY DNA PROFILE")
        print(f"{'='*60}")
        print(f"Trade Count: {dna.trade_count}")
        print(f"Overall Win Rate: {dna.overall_win_rate*100:.1f}%")
        print(f"Optimal Conviction: {dna.optimal_conviction_threshold}")
        print(f"Kelly (33%): {dna.kelly_33pct*100:.2f}%")
        print(f"Saved to: {path}")
        print(f"{'='*60}")
