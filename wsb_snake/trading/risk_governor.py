"""
Risk Governor – central risk controls for all trading engines.

- Engine separation: SCALPER (0DTE/intraday), MOMENTUM (small-cap breakout), MACRO (commodity/LEAPS)
- Max daily loss (hard stop / kill switch)
- Max concurrent positions (per-engine and global)
- Max exposure per ticker and per sector
- Position sizing: confidence- and volatility-adjusted cap
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

from wsb_snake.utils.logger import get_logger

log = get_logger(__name__)


class TradingEngine(Enum):
    """Which engine is requesting the trade. Each has its own limits."""
    SCALPER = "scalper"    # 0DTE / intraday SPY/QQQ/ETF
    MOMENTUM = "momentum"  # Small-cap breakout (ASTS, RKLB, LUNR, etc.)
    MACRO = "macro"        # Commodity/LEAPS (SLV, GLD, longer-dated)
    VOL_SELL = "vol_sell"  # Sell volatility / IV crush (credit spreads into earnings)


# Sector mapping for exposure caps (extend as needed)
SECTOR_MAP: Dict[str, str] = {
    # Index ETFs
    "SPY": "index", "QQQ": "index", "IWM": "index",
    # Commodities
    "SLV": "commodity", "GLD": "commodity", "GDX": "commodity", "GDXJ": "commodity",
    "USO": "commodity", "UNG": "commodity",
    # Energy
    "XLE": "energy",
    # Tech / mega
    "TSLA": "tech", "NVDA": "tech", "AAPL": "tech", "META": "tech",
    "AMD": "tech", "AMZN": "tech", "GOOGL": "tech", "MSFT": "tech",
    "PYPL": "tech",
    # Space / thematic
    "RKLB": "space", "ASTS": "space", "LUNR": "space", "PL": "space",
    "ONDS": "space", "SLS": "space",
    # Other small cap / thematic
    "THH": "other", "NBIS": "tech", "POET": "tech", "ENPH": "tech",
    "USAR": "other", "XLF": "financial", "TLT": "rates", "HYG": "credit",
}
DEFAULT_SECTOR = "other"


@dataclass
class GovernorConfig:
    """Configurable limits (can be overridden via env)."""
    # ═══════════════════════════════════════════════════════════════
    # DAILY P&L TARGETS AND RISK LIMITS (WEAPONIZED)
    # ═══════════════════════════════════════════════════════════════

    # Kill switch - JP MORGAN GRADE (tighter controls)
    max_daily_loss: float = -300.0  # Stop at -$300 (kill switch)
    kill_switch_manual: bool = False  # Set True to force halt

    # Daily profit targets (paper trading - adjust for live)
    daily_profit_target: float = 500.0  # Stop trading after hitting $500
    power_hour_target: float = 300.0    # Power hour contributes $300 of $500
    max_drawdown_from_peak: float = 200.0  # If we drop $200 from daily high, half size

    # Global position limits
    max_concurrent_positions_global: int = 3  # Reduce correlation risk (scalp mode)
    max_concurrent_positions_blowup: int = 2  # In blowup mode (size is 2x)
    max_daily_exposure_global: float = 4000.0  # $4k daily max
    max_total_exposure_pct: float = 0.25  # Max 25% of buying power
    max_single_position: float = 2000.0  # Max $2k even in blowup mode

    # Per-engine position limits
    max_positions_scalper: int = 2  # Focus on quality, not quantity
    max_positions_momentum: int = 1
    max_positions_macro: int = 1
    max_positions_vol_sell: int = 1

    # Per-ticker / per-sector exposure (dollars)
    max_exposure_per_ticker: float = 1000.0  # Max $1k per ticker
    max_exposure_per_sector: float = 2000.0  # Max $2k per sector

    # Position sizing: max premium per trade by engine (base before confidence/vol scaling)
    max_premium_scalper: float = 1000.0   # $1k max for 0DTE
    max_premium_momentum: float = 800.0   # $800 for momentum
    max_premium_macro: float = 1500.0     # $1.5k for LEAPS
    max_premium_vol_sell: float = 1000.0  # $1k for vol selling

    # Account cap: max % of buying power per trade
    max_pct_buying_power_per_trade: float = 0.05  # 5% (was 10%)

    # Consecutive loss cooldown - HYDRA standard
    consecutive_loss_threshold: int = 3  # 3 losses triggers cooldown
    cooldown_hours: float = 4.0  # 4 hour pause after consecutive losses

    # WIN RATE PRESERVATION - Preserve daily profits
    min_daily_win_rate: float = 0.75  # 75% - stop trading if win rate drops below
    min_trades_for_win_rate_check: int = 2  # Need at least 2 trades before enforcing
    high_vol_exception_vix: float = 25.0  # VIX > 25 allows trading even with low win rate
    preserve_profit_threshold: float = 50.0  # If daily P/L > $50, protect it more aggressively

    # ========== RISK WARDEN ENHANCEMENTS ==========
    # Conviction-based position sizing tiers
    conviction_tier_low: float = 68.0      # 68-75% conviction: half size ($500)
    conviction_tier_mid: float = 75.0      # 75-85% conviction: full size ($1,000)
    conviction_tier_high: float = 85.0     # 85%+ conviction: 1.5x size ($1,500)
    position_size_half: float = 500.0      # Half position size
    position_size_full: float = 1000.0     # Full position size
    position_size_max: float = 1500.0      # Max position size (1.5x)

    # Drawdown circuit breaker thresholds
    drawdown_half_size_threshold: float = -150.0   # -$150 daily -> half position size
    drawdown_halt_threshold: float = -200.0        # -$200 daily -> halt all new entries
    consecutive_loss_days_threshold: int = 3       # 3 consecutive losing days -> half size

    # Correlation guard
    max_correlation_threshold: float = 0.80  # Block trades > 0.8 correlated with existing

    @classmethod
    def from_env(cls) -> "GovernorConfig":
        c = cls()
        v = os.environ.get("RISK_MAX_DAILY_LOSS")
        if v is not None:
            try:
                c.max_daily_loss = float(v)
            except ValueError:
                pass
        v = os.environ.get("RISK_MAX_CONCURRENT_POSITIONS")
        if v is not None:
            try:
                c.max_concurrent_positions_global = int(v)
            except ValueError:
                pass
        v = os.environ.get("RISK_MAX_DAILY_EXPOSURE")
        if v is not None:
            try:
                c.max_daily_exposure_global = float(v)
            except ValueError:
                pass
        return c


class RiskGovernor:
    """
    Single source of truth for risk: can we trade, and how much.
    All engines must go through can_trade() and compute_position_size().
    """

    def __init__(self, config: Optional[GovernorConfig] = None):
        self.config = config or GovernorConfig.from_env()
        self._kill_switch_manual = False
        self._lock = threading.Lock()
        # HYDRA-style consecutive loss tracking
        self._consecutive_losses = 0
        self._cooldown_until: Optional[datetime] = None
        self._win_count = 0
        self._loss_count = 0
        # WIN RATE PRESERVATION - Daily tracking
        self._daily_win_count = 0
        self._daily_loss_count = 0
        self._daily_pnl_realized = 0.0
        self._last_trade_date: Optional[str] = None
        self._win_rate_pause_active = False
        # RISK WARDEN: Drawdown circuit breaker state
        self._drawdown_half_size_active = False
        self._drawdown_halt_active = False
        self._consecutive_losing_days = 0
        self._last_day_pnl: Optional[float] = None
        self._last_day_date: Optional[str] = None
        # WEAPONIZED: Daily profit target tracking
        self._daily_pnl_peak = 0.0
        self._profit_target_hit = False
        self._power_hour_pnl = 0.0
        self._power_hour_trades = 0

    def set_kill_switch(self, on: bool) -> None:
        """Manually halt all new trades."""
        with self._lock:
            self._kill_switch_manual = on
        log.warning(f"Risk governor kill switch set to: {on}")

    @property
    def kill_switch_active(self) -> bool:
        with self._lock:
            return self._kill_switch_manual

    def _sector(self, ticker: str) -> str:
        return SECTOR_MAP.get(ticker.upper(), DEFAULT_SECTOR)

    def _positions_by_ticker_and_sector(
        self,
        positions: List[Tuple[str, str, float]]  # (ticker, option_symbol, cost)
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Aggregate exposure by ticker and by sector."""
        by_ticker: Dict[str, float] = {}
        by_sector: Dict[str, float] = {}
        for ticker, _sym, cost in positions:
            t = ticker.upper()
            by_ticker[t] = by_ticker.get(t, 0) + cost
            sec = self._sector(t)
            by_sector[sec] = by_sector.get(sec, 0) + cost
        return by_ticker, by_sector

    def can_trade(
        self,
        engine: TradingEngine,
        ticker: str,
        open_positions_count: int,
        positions_with_cost: List[Tuple[str, str, float]],  # (ticker, option_symbol, cost)
        daily_pnl: float,
        daily_exposure_used: float,
    ) -> Tuple[bool, str]:
        """
        Returns (allowed, reason).
        If allowed is False, reason explains why (e.g. daily loss, kill switch, cap).
        """
        with self._lock:
            if self._kill_switch_manual:
                return False, "Kill switch active (manual)"

        # HYDRA: Check consecutive loss cooldown
        in_cooldown, cooldown_reason = self.is_in_cooldown()
        if in_cooldown:
            return False, cooldown_reason

        # WIN RATE PRESERVATION: Check if win rate pause is active
        win_rate_paused, win_rate_reason = self.is_win_rate_pause_active()
        if win_rate_paused:
            return False, win_rate_reason

        # RISK WARDEN: Drawdown circuit breaker - halt check
        self._update_drawdown_state(daily_pnl)
        if self._drawdown_halt_active:
            return False, f"CIRCUIT BREAKER: Daily loss ${daily_pnl:.0f} triggered halt (threshold: ${self.config.drawdown_halt_threshold})"

        if daily_pnl <= self.config.max_daily_loss:
            return False, f"Daily PnL ${daily_pnl:.0f} at or below max daily loss ${self.config.max_daily_loss:.0f}"

        if open_positions_count >= self.config.max_concurrent_positions_global:
            return False, f"Max concurrent positions reached ({open_positions_count})"

        if daily_exposure_used >= self.config.max_daily_exposure_global:
            return False, f"Daily exposure cap reached (${daily_exposure_used:.0f})"

        max_for_engine = {
            TradingEngine.SCALPER: self.config.max_positions_scalper,
            TradingEngine.MOMENTUM: self.config.max_positions_momentum,
            TradingEngine.MACRO: self.config.max_positions_macro,
            TradingEngine.VOL_SELL: self.config.max_positions_vol_sell,
        }.get(engine, self.config.max_positions_scalper)

        by_ticker, by_sector = self._positions_by_ticker_and_sector(positions_with_cost)
        ticker_up = ticker.upper()
        sector = self._sector(ticker_up)

        current_ticker = by_ticker.get(ticker_up, 0)
        if current_ticker >= self.config.max_exposure_per_ticker:
            return False, f"Max exposure per ticker reached for {ticker} (${current_ticker:.0f})"

        current_sector = by_sector.get(sector, 0)
        if current_sector >= self.config.max_exposure_per_sector:
            return False, f"Max exposure per sector ({sector}) reached (${current_sector:.0f})"

        # RISK WARDEN: Correlation guard - check if new trade too correlated with existing
        existing_tickers = [t for t, _, _ in positions_with_cost]
        corr_allowed, corr_reason = self.check_correlation_guard(ticker, existing_tickers)
        if not corr_allowed:
            return False, corr_reason

        # Per-engine position count: we don't track engine per position here; we only have global count.
        # So we only enforce global and exposure. Per-engine count would require executor to tag each position with engine.
        return True, "ok"

    def get_max_premium_for_engine(self, engine: TradingEngine) -> float:
        """Base max premium per trade for this engine (before confidence/vol scaling)."""
        return {
            TradingEngine.SCALPER: self.config.max_premium_scalper,
            TradingEngine.MOMENTUM: self.config.max_premium_momentum,
            TradingEngine.MACRO: self.config.max_premium_macro,
            TradingEngine.VOL_SELL: self.config.max_premium_vol_sell,
        }.get(engine, self.config.max_premium_scalper)

    def compute_conviction_position_size(
        self,
        conviction_pct: float,
        daily_pnl: float = 0.0,
    ) -> float:
        """
        RISK WARDEN: Conviction-based position sizing with drawdown circuit breaker.

        Position size tiers:
        - 68-75% conviction: $500 (half size)
        - 75-85% conviction: $1,000 (full size)
        - 85%+ conviction: $1,500 (1.5x size)

        Drawdown adjustments:
        - Daily PnL < -$150: half all sizes
        - Daily PnL < -$200: halt new entries
        - 3 consecutive losing days: half all sizes

        Args:
            conviction_pct: Conviction score (0-100)
            daily_pnl: Current daily realized P/L

        Returns:
            Maximum position size in dollars
        """
        # Check drawdown circuit breaker
        self._update_drawdown_state(daily_pnl)

        if self._drawdown_halt_active:
            log.warning("CIRCUIT BREAKER: Drawdown halt active - no new positions")
            return 0.0

        # Base position size from conviction tier
        if conviction_pct >= self.config.conviction_tier_high:
            base_size = self.config.position_size_max  # $1,500
            tier = "HIGH (1.5x)"
        elif conviction_pct >= self.config.conviction_tier_mid:
            base_size = self.config.position_size_full  # $1,000
            tier = "FULL (1x)"
        elif conviction_pct >= self.config.conviction_tier_low:
            base_size = self.config.position_size_half  # $500
            tier = "HALF (0.5x)"
        else:
            log.info(f"Conviction {conviction_pct:.0f}% below minimum {self.config.conviction_tier_low}% - no trade")
            return 0.0

        # Apply drawdown reduction if active
        if self._drawdown_half_size_active or self._consecutive_losing_days >= self.config.consecutive_loss_days_threshold:
            base_size = base_size / 2
            log.warning(f"CIRCUIT BREAKER: Position size halved to ${base_size:.0f} (drawdown protection)")

        log.info(f"Conviction sizing: {conviction_pct:.0f}% -> {tier} -> ${base_size:.0f}")
        return base_size

    def _update_drawdown_state(self, daily_pnl: float) -> None:
        """Update drawdown circuit breaker state based on daily P/L."""
        with self._lock:
            # Check daily drawdown thresholds
            if daily_pnl <= self.config.drawdown_halt_threshold:
                if not self._drawdown_halt_active:
                    self._drawdown_halt_active = True
                    log.warning(f"CIRCUIT BREAKER ACTIVATED: Daily PnL ${daily_pnl:.2f} <= ${self.config.drawdown_halt_threshold} - HALTING ALL NEW ENTRIES")
            elif daily_pnl <= self.config.drawdown_half_size_threshold:
                if not self._drawdown_half_size_active:
                    self._drawdown_half_size_active = True
                    log.warning(f"CIRCUIT BREAKER: Daily PnL ${daily_pnl:.2f} <= ${self.config.drawdown_half_size_threshold} - HALVING POSITION SIZES")
            else:
                # Reset if P/L improves
                if self._drawdown_half_size_active and daily_pnl > self.config.drawdown_half_size_threshold + 50:
                    self._drawdown_half_size_active = False
                    log.info("Circuit breaker half-size deactivated - P/L improved")

    def update_consecutive_losing_days(self, end_of_day_pnl: float) -> None:
        """
        Track consecutive losing days for circuit breaker.
        Call at end of trading day with the day's total P/L.
        """
        from datetime import date
        today = date.today().isoformat()

        with self._lock:
            if self._last_day_date != today:
                # New day - check if previous day was a loss
                if self._last_day_pnl is not None and self._last_day_pnl < 0:
                    self._consecutive_losing_days += 1
                    log.warning(f"Consecutive losing days: {self._consecutive_losing_days}")
                elif self._last_day_pnl is not None and self._last_day_pnl >= 0:
                    self._consecutive_losing_days = 0
                    log.info("Consecutive losing days reset after winning day")

                self._last_day_pnl = end_of_day_pnl
                self._last_day_date = today

    def get_drawdown_status(self) -> Dict:
        """Get current drawdown circuit breaker status."""
        with self._lock:
            return {
                "half_size_active": self._drawdown_half_size_active,
                "halt_active": self._drawdown_halt_active,
                "consecutive_losing_days": self._consecutive_losing_days,
                "daily_pnl": self._daily_pnl_realized,
            }

    def check_correlation_guard(
        self,
        ticker: str,
        existing_positions: List[str],
    ) -> Tuple[bool, str]:
        """
        RISK WARDEN: Correlation guard to prevent overlapping risk.

        Blocks trades on assets > 0.8 correlated with existing positions.

        Args:
            ticker: Ticker to check
            existing_positions: List of tickers currently held

        Returns:
            (allowed, reason) - False if correlation too high
        """
        if not existing_positions:
            return True, "ok"

        # Known high-correlation pairs (hardcoded for speed, update as needed)
        CORRELATION_MAP = {
            # Index ETFs - highly correlated
            "SPY": {"QQQ": 0.92, "IWM": 0.88, "DIA": 0.95},
            "QQQ": {"SPY": 0.92, "IWM": 0.82, "TQQQ": 0.99, "SQQQ": -0.99},
            "IWM": {"SPY": 0.88, "QQQ": 0.82},
            # Tech mega caps - correlated
            "NVDA": {"AMD": 0.85, "SMCI": 0.82, "TSM": 0.78},
            "AMD": {"NVDA": 0.85, "INTC": 0.72},
            "AAPL": {"MSFT": 0.82, "GOOGL": 0.78},
            "MSFT": {"AAPL": 0.82, "GOOGL": 0.80},
            # Precious metals - highly correlated
            "GLD": {"SLV": 0.85, "GDX": 0.82, "GDXJ": 0.80},
            "SLV": {"GLD": 0.85, "GDX": 0.75},
            "GDX": {"GLD": 0.82, "GDXJ": 0.95, "SLV": 0.75},
            "GDXJ": {"GDX": 0.95, "GLD": 0.80},
            # Energy - correlated
            "XLE": {"USO": 0.85, "XOM": 0.88, "CVX": 0.85},
            "USO": {"XLE": 0.85, "UNG": 0.45},
            # Space stocks - correlated
            "RKLB": {"ASTS": 0.75, "LUNR": 0.72},
            "ASTS": {"RKLB": 0.75, "LUNR": 0.70},
        }

        ticker_upper = ticker.upper()
        correlations = CORRELATION_MAP.get(ticker_upper, {})

        for existing_ticker in existing_positions:
            existing_upper = existing_ticker.upper()
            if existing_upper in correlations:
                corr = correlations[existing_upper]
                if abs(corr) >= self.config.max_correlation_threshold:
                    reason = f"CORRELATION GUARD: {ticker} has {corr:.0%} correlation with existing {existing_ticker}"
                    log.warning(reason)
                    return False, reason

        return True, "ok"

    def compute_position_size(
        self,
        engine: TradingEngine,
        confidence_pct: float,
        option_price: float,
        buying_power: Optional[float] = None,
        volatility_factor: float = 1.0,
    ) -> int:
        """
        Compute number of contracts for this trade.
        - confidence_pct: 0–100
        - option_price: per-share option price (e.g. 1.50)
        - buying_power: optional; if set, caps trade at max_pct_buying_power_per_trade
        - volatility_factor: >1 reduces size, <1 can allow slightly more (e.g. 0.8 for low vol)
        Returns 0 if option too expensive or size would be 0.
        """
        if option_price <= 0:
            return 0

        base_cap = self.get_max_premium_for_engine(engine)
        # Scale by confidence (e.g. 80% -> 0.8)
        confidence_scale = max(0.5, min(1.0, confidence_pct / 100.0))
        # Reduce size when vol is high
        vol_scale = 1.0 / max(0.5, min(2.0, volatility_factor))
        max_premium = base_cap * confidence_scale * vol_scale

        if buying_power is not None and buying_power > 0:
            pct_cap = buying_power * self.config.max_pct_buying_power_per_trade
            max_premium = min(max_premium, pct_cap)

        contract_cost = option_price * 100
        if contract_cost > max_premium:
            log.debug(
                f"Position size 0: contract ${contract_cost:.2f} > cap ${max_premium:.2f} "
                f"(engine={engine.value}, conf={confidence_pct:.0f}%)"
            )
            return 0

        num = int(max_premium / contract_cost)
        return max(0, num)

    def compute_kelly_position_size(
        self,
        engine: TradingEngine,
        win_probability: float,      # From APEX (0-1)
        avg_win_pct: float,          # Historical avg win % (e.g., 0.06 for 6%)
        avg_loss_pct: float,         # Historical avg loss % (e.g., 0.10 for 10%)
        option_price: float,
        buying_power: Optional[float] = None,
        volatility_factor: float = 1.0,
    ) -> int:
        """
        Half-Kelly position sizing - institutional standard.

        Kelly f* = (p*b - q) / b where:
        - p = win probability
        - q = 1 - p
        - b = win/loss ratio (avg_win / avg_loss)

        Half-Kelly = f*/2 for conservative sizing (75% growth, 50% drawdown).
        """
        if option_price <= 0 or avg_loss_pct <= 0:
            return 0

        # Clamp win probability to reasonable bounds
        p = max(0.01, min(0.99, win_probability))
        q = 1.0 - p

        # Win/loss ratio (b)
        b = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 1.0

        # Kelly fraction: f* = (p*b - q) / b
        kelly_fraction = (p * b - q) / b if b > 0 else 0

        # Half-Kelly for conservative sizing
        half_kelly = kelly_fraction / 2.0

        # Clamp to reasonable range (0% to 25% of capital)
        half_kelly = max(0.0, min(0.25, half_kelly))

        if half_kelly <= 0:
            log.debug(f"Kelly suggests no position: p={p:.2f}, b={b:.2f}, f*={kelly_fraction:.4f}")
            return 0

        # Calculate max premium based on Kelly fraction
        base_cap = self.get_max_premium_for_engine(engine)

        # Apply volatility scaling (reduce size when vol is high)
        vol_scale = 1.0 / max(0.5, min(2.0, volatility_factor))

        # Kelly-adjusted premium cap
        kelly_premium = base_cap * half_kelly * vol_scale

        # Also respect buying power limits
        if buying_power is not None and buying_power > 0:
            bp_cap = buying_power * half_kelly
            kelly_premium = min(kelly_premium, bp_cap)

        contract_cost = option_price * 100
        if contract_cost > kelly_premium:
            log.debug(
                f"Kelly position size 0: contract ${contract_cost:.2f} > kelly cap ${kelly_premium:.2f} "
                f"(engine={engine.value}, half_kelly={half_kelly:.4f})"
            )
            return 0

        num = int(kelly_premium / contract_cost)
        log.info(f"Kelly sizing: p={p:.2f}, b={b:.2f}, half_kelly={half_kelly:.4f} -> {num} contracts")
        return max(0, num)

    def record_trade_outcome(self, outcome: str, pnl: float = 0.0) -> None:
        """
        Track consecutive losses for cooldown AND daily win rate.

        Args:
            outcome: 'win' or 'loss'
            pnl: Realized P/L in dollars (optional, for daily tracking)

        HYDRA standard: 3 consecutive losses triggers a 4-hour cooldown
        to prevent emotional/revenge trading.

        WIN RATE PRESERVATION: Also tracks daily win rate and P/L.
        """
        with self._lock:
            # Track daily stats for win rate preservation
            self._reset_daily_stats_if_new_day()
            self._daily_pnl_realized += pnl

            if outcome.lower() == 'win':
                self._win_count += 1
                self._daily_win_count += 1
                self._consecutive_losses = 0  # Reset streak on win
                log.info(f"Trade outcome: WIN (+${pnl:.2f}). Consecutive losses reset. Total: {self._win_count}W / {self._loss_count}L")
            elif outcome.lower() == 'loss':
                self._loss_count += 1
                self._daily_loss_count += 1
                self._consecutive_losses += 1
                log.warning(f"Trade outcome: LOSS (${pnl:.2f}). Consecutive losses: {self._consecutive_losses}. Total: {self._win_count}W / {self._loss_count}L")

                # Check if cooldown should be triggered
                if self._consecutive_losses >= self.config.consecutive_loss_threshold:
                    self._cooldown_until = datetime.now() + timedelta(hours=self.config.cooldown_hours)
                    log.warning(
                        f"HYDRA COOLDOWN ACTIVATED: {self._consecutive_losses} consecutive losses. "
                        f"Trading paused until {self._cooldown_until.strftime('%Y-%m-%d %H:%M:%S')}"
                    )

            # Log daily stats
            total_daily = self._daily_win_count + self._daily_loss_count
            daily_wr = self._daily_win_count / total_daily if total_daily > 0 else 0
            log.info(
                f"DAILY: {self._daily_win_count}W/{self._daily_loss_count}L ({daily_wr:.0%}) | "
                f"P/L: ${self._daily_pnl_realized:.2f}"
            )

    def is_in_cooldown(self) -> Tuple[bool, str]:
        """
        Check if in consecutive loss cooldown.

        Returns (is_in_cooldown, reason_string).
        """
        with self._lock:
            if self._cooldown_until is None:
                return False, ""

            now = datetime.now()
            if now < self._cooldown_until:
                remaining = self._cooldown_until - now
                hours_remaining = remaining.total_seconds() / 3600
                reason = (
                    f"HYDRA cooldown active: {self._consecutive_losses} consecutive losses. "
                    f"Trading resumes in {hours_remaining:.1f} hours ({self._cooldown_until.strftime('%H:%M:%S')})"
                )
                return True, reason
            else:
                # Cooldown expired
                self._cooldown_until = None
                self._consecutive_losses = 0  # Reset after cooldown
                log.info("HYDRA cooldown expired. Trading enabled.")
                return False, ""

    def get_win_rate(self) -> float:
        """
        Get historical win rate for Kelly calculation.

        Returns win rate as decimal (0-1). Returns 0.5 if no trades recorded.
        """
        with self._lock:
            total = self._win_count + self._loss_count
            if total == 0:
                return 0.5  # Default 50% if no history
            return self._win_count / total

    def _reset_daily_stats_if_new_day(self) -> None:
        """Reset daily stats if it's a new trading day."""
        from datetime import date
        today = date.today().isoformat()
        if self._last_trade_date != today:
            self._daily_win_count = 0
            self._daily_loss_count = 0
            self._daily_pnl_realized = 0.0
            self._win_rate_pause_active = False
            self._last_trade_date = today
            log.info(f"New trading day {today} - daily stats reset")

    def get_daily_win_rate(self) -> float:
        """Get today's win rate."""
        with self._lock:
            self._reset_daily_stats_if_new_day()
            total = self._daily_win_count + self._daily_loss_count
            if total == 0:
                return 1.0  # No trades yet = 100% (allow trading)
            return self._daily_win_count / total

    def get_daily_stats(self) -> dict:
        """Get today's trading statistics."""
        with self._lock:
            self._reset_daily_stats_if_new_day()
            total = self._daily_win_count + self._daily_loss_count
            return {
                "wins": self._daily_win_count,
                "losses": self._daily_loss_count,
                "total_trades": total,
                "win_rate": self.get_daily_win_rate(),
                "daily_pnl": self._daily_pnl_realized,
                "pause_active": self._win_rate_pause_active,
            }

    def _get_current_vix(self) -> float:
        """Get current VIX level for high-volatility exception.

        CRITICAL: Must be non-blocking! Use cached value only.
        Network calls here block execute_scalp_entry.
        """
        try:
            from wsb_snake.collectors.vix_structure import vix_structure
            # Only use cached VIX - don't block on network calls
            import time
            cache_key = "term_structure"
            if cache_key in vix_structure.cache:
                cached = vix_structure.cache[cache_key]
                # Accept cache even if stale rather than blocking
                data = cached.get("data", {})
                return data.get("vix_spot", 20.0)
            # No cache yet - return default, background will populate later
            return 20.0
        except Exception:
            return 20.0  # Default if VIX unavailable

    def is_win_rate_pause_active(self) -> Tuple[bool, str]:
        """
        Check if win rate preservation pause is active.

        Returns (is_paused, reason).
        Allows trading if VIX > threshold (high volatility exception).
        """
        with self._lock:
            self._reset_daily_stats_if_new_day()

            total_trades = self._daily_win_count + self._daily_loss_count

            # Need minimum trades before enforcing win rate
            if total_trades < self.config.min_trades_for_win_rate_check:
                return False, ""

            daily_win_rate = self.get_daily_win_rate()

            # Check if below minimum win rate
            if daily_win_rate < self.config.min_daily_win_rate:
                # Check for high volatility exception
                vix = self._get_current_vix()
                if vix >= self.config.high_vol_exception_vix:
                    log.info(
                        f"Win rate {daily_win_rate:.0%} below {self.config.min_daily_win_rate:.0%} but "
                        f"VIX {vix:.1f} >= {self.config.high_vol_exception_vix} - HIGH VOL EXCEPTION ACTIVE"
                    )
                    return False, ""

                # Check if we have profits to protect
                if self._daily_pnl_realized >= self.config.preserve_profit_threshold:
                    self._win_rate_pause_active = True
                    reason = (
                        f"WIN RATE PAUSE: {daily_win_rate:.0%} (below {self.config.min_daily_win_rate:.0%}). "
                        f"Daily P/L: ${self._daily_pnl_realized:.2f} - PRESERVING PROFITS. "
                        f"Trades: {self._daily_win_count}W/{self._daily_loss_count}L"
                    )
                    log.warning(reason)
                    return True, reason

        return False, ""

    def record_daily_outcome(self, outcome: str, pnl: float) -> None:
        """
        Record a trade outcome for daily tracking.

        Args:
            outcome: "win", "loss", or "scratch"
            pnl: Realized P/L in dollars
        """
        with self._lock:
            self._reset_daily_stats_if_new_day()
            self._daily_pnl_realized += pnl

            if outcome.lower() == "win":
                self._daily_win_count += 1
            elif outcome.lower() == "loss":
                self._daily_loss_count += 1

            total = self._daily_win_count + self._daily_loss_count
            win_rate = self._daily_win_count / total if total > 0 else 0

            log.info(
                f"Daily stats: {self._daily_win_count}W/{self._daily_loss_count}L "
                f"({win_rate:.0%}) | P/L: ${self._daily_pnl_realized:.2f}"
            )

    def force_resume_trading(self) -> None:
        """Manually resume trading (admin override)."""
        with self._lock:
            self._win_rate_pause_active = False
            self._profit_target_hit = False
            log.warning("WIN RATE PAUSE manually disabled - trading resumed")

    # ═══════════════════════════════════════════════════════════════
    # WEAPONIZED: Daily Profit Target and Power Hour Tracking
    # ═══════════════════════════════════════════════════════════════

    def check_profit_target(self) -> Tuple[bool, str]:
        """
        Check if daily profit target has been hit.

        Returns (target_hit, reason).
        """
        with self._lock:
            self._reset_daily_stats_if_new_day()

            if self._profit_target_hit:
                return True, f"PROFIT TARGET HIT: ${self._daily_pnl_realized:.2f} >= ${self.config.daily_profit_target:.2f} — PRESERVE GAINS"

            if self._daily_pnl_realized >= self.config.daily_profit_target:
                self._profit_target_hit = True
                log.warning(
                    f"PROFIT TARGET ACHIEVED: ${self._daily_pnl_realized:.2f} | "
                    f"Stopping trading to preserve gains"
                )
                return True, f"PROFIT TARGET HIT: ${self._daily_pnl_realized:.2f}"

            return False, ""

    def check_drawdown_from_peak(self) -> Tuple[bool, str]:
        """
        Check if we've drawn down too much from daily peak.

        Returns (should_reduce_size, reason).
        """
        with self._lock:
            self._reset_daily_stats_if_new_day()

            # Update peak
            if self._daily_pnl_realized > self._daily_pnl_peak:
                self._daily_pnl_peak = self._daily_pnl_realized

            # Check drawdown from peak
            drawdown = self._daily_pnl_peak - self._daily_pnl_realized
            if drawdown >= self.config.max_drawdown_from_peak:
                reason = (
                    f"DRAWDOWN ALERT: Down ${drawdown:.2f} from peak ${self._daily_pnl_peak:.2f} | "
                    f"Current: ${self._daily_pnl_realized:.2f} — HALF SIZE MODE"
                )
                log.warning(reason)
                return True, reason

            return False, ""

    def record_power_hour_trade(self, pnl: float) -> None:
        """Track power hour specific P&L."""
        with self._lock:
            self._power_hour_pnl += pnl
            self._power_hour_trades += 1
            log.info(
                f"POWER_HOUR: Trade #{self._power_hour_trades} | "
                f"Trade P&L: ${pnl:.2f} | Power Hour Total: ${self._power_hour_pnl:.2f}"
            )

    def check_power_hour_target(self) -> Tuple[bool, str]:
        """
        Check if power hour profit target has been hit.

        Returns (target_hit, reason).
        """
        with self._lock:
            if self._power_hour_pnl >= self.config.power_hour_target:
                reason = (
                    f"POWER_HOUR TARGET HIT: ${self._power_hour_pnl:.2f} >= "
                    f"${self.config.power_hour_target:.2f} — EXIT POWER HOUR"
                )
                log.warning(reason)
                return True, reason
            return False, ""

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on current risk state.

        Returns 0.5 if drawdown from peak, 1.0 otherwise.
        """
        should_reduce, _ = self.check_drawdown_from_peak()
        if should_reduce:
            return 0.5
        return 1.0

    def can_open_new_position(self, is_blowup_mode: bool = False) -> Tuple[bool, str]:
        """
        Check all risk conditions before opening a new position.

        Args:
            is_blowup_mode: True if in blowup mode (different limits)

        Returns (can_trade, reason).
        """
        # Check kill switch
        if self._kill_switch_manual:
            return False, "Kill switch active"

        # Check profit target
        target_hit, target_reason = self.check_profit_target()
        if target_hit:
            return False, target_reason

        # Check consecutive loss cooldown
        in_cooldown, cooldown_reason = self.is_in_cooldown()
        if in_cooldown:
            return False, cooldown_reason

        # Check win rate pause
        win_rate_paused, win_rate_reason = self.is_win_rate_pause_active()
        if win_rate_paused:
            return False, win_rate_reason

        # Check drawdown halt
        if self._drawdown_halt_active:
            return False, "CIRCUIT BREAKER: Drawdown halt active"

        return True, "ok"

    def get_weaponized_status(self) -> Dict:
        """Get full weaponized risk status."""
        with self._lock:
            self._reset_daily_stats_if_new_day()
            return {
                # Daily stats
                "daily_pnl": self._daily_pnl_realized,
                "daily_pnl_peak": self._daily_pnl_peak,
                "daily_profit_target": self.config.daily_profit_target,
                "profit_target_hit": self._profit_target_hit,
                # Win/loss
                "daily_wins": self._daily_win_count,
                "daily_losses": self._daily_loss_count,
                "daily_win_rate": self.get_daily_win_rate(),
                "consecutive_losses": self._consecutive_losses,
                # Power hour
                "power_hour_pnl": self._power_hour_pnl,
                "power_hour_trades": self._power_hour_trades,
                "power_hour_target": self.config.power_hour_target,
                # Circuit breakers
                "drawdown_half_size": self._drawdown_half_size_active,
                "drawdown_halt": self._drawdown_halt_active,
                "win_rate_pause": self._win_rate_pause_active,
                "cooldown_until": self._cooldown_until.isoformat() if self._cooldown_until else None,
            }

    def sync_daily_stats_from_alpaca(self) -> dict:
        """
        Sync daily stats from Alpaca trade history on startup.

        This ensures daily win rate and P/L are accurate even after restarts.
        Fetches today's closed orders and reconstructs the stats.
        """
        from datetime import date
        import os

        try:
            import alpaca_trade_api as tradeapi

            api = tradeapi.REST(
                os.environ.get("ALPACA_API_KEY"),
                os.environ.get("ALPACA_SECRET_KEY"),
                os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            )

            # Get today's date
            today = date.today()
            today_str = today.isoformat()

            # Fetch all orders from today
            orders = api.list_orders(status="all", limit=100, after=today_str)

            # Group orders by symbol to calculate P/L
            trades = {}  # symbol -> {buys: [], sells: []}

            for order in orders:
                if order.status != "filled":
                    continue

                symbol = order.symbol
                if symbol not in trades:
                    trades[symbol] = {"buys": [], "sells": []}

                price = float(order.filled_avg_price) if order.filled_avg_price else 0
                qty = int(order.filled_qty) if order.filled_qty else 0

                if order.side == "buy":
                    trades[symbol]["buys"].append((qty, price))
                else:
                    trades[symbol]["sells"].append((qty, price))

            # Calculate P/L for each completed round trip
            wins = 0
            losses = 0
            total_pnl = 0.0

            for symbol, data in trades.items():
                buys = data["buys"]
                sells = data["sells"]

                # Simple FIFO matching
                buy_idx = 0
                sell_idx = 0

                while buy_idx < len(buys) and sell_idx < len(sells):
                    buy_qty, buy_price = buys[buy_idx]
                    sell_qty, sell_price = sells[sell_idx]

                    matched_qty = min(buy_qty, sell_qty)

                    # Options are 100 shares per contract
                    pnl = (sell_price - buy_price) * matched_qty * 100
                    total_pnl += pnl

                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

                    # Update remaining quantities
                    buys[buy_idx] = (buy_qty - matched_qty, buy_price)
                    sells[sell_idx] = (sell_qty - matched_qty, sell_price)

                    if buys[buy_idx][0] == 0:
                        buy_idx += 1
                    if sells[sell_idx][0] == 0:
                        sell_idx += 1

            # Update governor state
            with self._lock:
                self._daily_win_count = wins
                self._daily_loss_count = losses
                self._daily_pnl_realized = total_pnl
                self._last_trade_date = today_str

                total = wins + losses
                win_rate = wins / total if total > 0 else 1.0

                log.info(
                    f"SYNCED DAILY STATS FROM ALPACA: {wins}W/{losses}L ({win_rate:.0%}) | "
                    f"P/L: ${total_pnl:.2f}"
                )

                # Check if win rate pause should be active
                if total >= self.config.min_trades_for_win_rate_check:
                    if win_rate < self.config.min_daily_win_rate:
                        if total_pnl >= self.config.preserve_profit_threshold:
                            self._win_rate_pause_active = True
                            log.warning(
                                f"WIN RATE PAUSE ACTIVATED on sync: {win_rate:.0%} below "
                                f"{self.config.min_daily_win_rate:.0%} with ${total_pnl:.2f} profit"
                            )

            return {
                "wins": wins,
                "losses": losses,
                "total_trades": wins + losses,
                "win_rate": win_rate,
                "daily_pnl": total_pnl,
                "synced": True
            }

        except Exception as e:
            log.error(f"Failed to sync daily stats from Alpaca: {e}")
            return {"synced": False, "error": str(e)}


# Singleton used by executor and engines
_governor: Optional[RiskGovernor] = None
_governor_lock = threading.Lock()


def get_risk_governor(config: Optional[GovernorConfig] = None) -> RiskGovernor:
    global _governor
    with _governor_lock:
        if _governor is None:
            _governor = RiskGovernor(config=config)
        return _governor


def reset_risk_governor(config: Optional[GovernorConfig] = None) -> RiskGovernor:
    """For tests: reset singleton."""
    global _governor
    with _governor_lock:
        _governor = RiskGovernor(config=config)
        return _governor
