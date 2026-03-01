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
    # VENOM COMPOUNDING MODE - PERCENTAGE BASED EVERYTHING
    # From $5K -> $50K+ requires aggressive reinvestment
    # ═══════════════════════════════════════════════════════════════

    # Kill switch - PERCENTAGE BASED (not flat dollar)
    max_daily_loss_pct: float = 0.20  # Stop at -20% of account (not flat $300)
    kill_switch_manual: bool = False  # Set True to force halt

    # Daily profit targets - COMPOUND OR DIE
    daily_profit_target_pct: float = 0.15  # Target +15% daily (aggressive compounding)
    power_hour_target_pct: float = 0.10    # Power hour contributes 10%
    max_drawdown_from_peak_pct: float = 0.10  # If we drop 10% from daily high, half size

    # Global position limits - SWARM CONSENSUS (10/12 personas agreed)
    # Reduced from VENOM EXTREME to disciplined aggression
    max_concurrent_positions_global: int = 5  # Allow 5 positions
    max_concurrent_positions_blowup: int = 3  # In blowup mode (size is 3x)
    max_daily_exposure_pct: float = 0.60  # 60% of account can be deployed daily (DOWN from 90%)
    max_total_exposure_pct: float = 0.30  # Max 30% at any one time (DOWN from 50%)
    max_single_position_pct: float = 0.05  # Max 5% per position (DOWN from 35% - SWARM CONSENSUS)

    # Per-engine position limits
    max_positions_scalper: int = 3  # More scalps = more compounding
    max_positions_momentum: int = 2
    max_positions_macro: int = 1
    max_positions_vol_sell: int = 1

    # Per-ticker / per-sector exposure (percentage of account)
    max_exposure_per_ticker_pct: float = 0.25  # Max 25% per ticker
    max_exposure_per_sector_pct: float = 0.40  # Max 40% per sector

    # Position sizing: max premium per trade by engine (% of buying power)
    # SWARM CONSENSUS: 2-5% per trade, scale on wins (10/12 agreed)
    max_premium_scalper_pct: float = 0.05   # 5% max for 0DTE (DOWN from 20%)
    max_premium_momentum_pct: float = 0.04  # 4% for momentum (DOWN from 15%)
    max_premium_macro_pct: float = 0.05     # 5% for LEAPS (DOWN from 25%)
    max_premium_vol_sell_pct: float = 0.03  # 3% for vol selling (DOWN from 15%)

    # Account cap: max % of buying power per trade
    max_pct_buying_power_per_trade: float = 0.05  # 5% (DOWN from 20% - SWARM CONSENSUS)

    # Consecutive loss cooldown - VENOM: DON'T STOP TRADING
    consecutive_loss_threshold: int = 6  # 6 losses triggers cooldown (UP from 4 - stay in game)
    cooldown_hours: float = 1.0  # 1 hour pause (DOWN from 2 - faster restart)

    # WIN RATE PRESERVATION - VENOM: Less restrictive
    min_daily_win_rate: float = 0.50  # 50% (LOWERED from 60% - allow more action)
    min_trades_for_win_rate_check: int = 5  # Need 5 trades before enforcing (UP from 3)
    high_vol_exception_vix: float = 25.0  # VIX > 25 allows trading (LOWERED - trade more)
    preserve_profit_threshold_pct: float = 0.08  # Protect when up 8%+ (UP from 5%)

    # ========== VENOM EXTREME COMPOUNDING ==========
    # Conviction-based position sizing tiers (% of account) - AGGRESSIVE
    conviction_tier_low: float = 60.0      # LOWERED: 60-72% conviction: 15% of account
    conviction_tier_mid: float = 72.0      # LOWERED: 72-82% conviction: 22% of account
    conviction_tier_high: float = 82.0     # LOWERED: 82%+ conviction: 35% of account
    position_pct_low: float = 0.15         # 15% of account (UP from 12%)
    position_pct_mid: float = 0.22         # 22% of account (UP from 18%)
    position_pct_high: float = 0.35        # 35% of account (UP from 25%)

    # COMPOUNDING MULTIPLIERS - EXTREME: Size up FAST after wins
    win_streak_multiplier: float = 1.75    # +75% size after each win (UP from 1.50)
    loss_streak_divisor: float = 0.60      # -40% size after loss (less punishing)
    max_win_streak_multiplier: float = 4.0  # Cap at 4x (UP from 3x) - FULL SEND
    reinvest_profit_pct: float = 0.95      # Reinvest 95% of daily gains (UP from 85%)

    # GEX REGIME MULTIPLIERS - VENOM EXTREME when dealers are short gamma
    gex_negative_multiplier: float = 1.8   # 1.8x size when GEX negative (UP from 1.5)
    gex_extreme_negative_multiplier: float = 2.5  # 2.5x size when GEX very negative (UP from 2.0)

    # Drawdown circuit breaker thresholds (PERCENTAGE)
    drawdown_half_size_threshold_pct: float = 0.10  # -10% daily -> half position size
    drawdown_halt_threshold_pct: float = 0.15       # -15% daily -> halt all new entries
    consecutive_loss_days_threshold: int = 2        # 2 consecutive losing days -> half size

    # Correlation guard
    max_correlation_threshold: float = 0.85  # Allow slightly more correlation for compounding

    # ═══════════════════════════════════════════════════════════════
    # VENOM: TIME-OF-DAY AGGRESSION MULTIPLIERS
    # Different windows have different win rates and volatility patterns
    # ═══════════════════════════════════════════════════════════════
    # Opening Drive (9:30-10:00): 1.5x - High volatility, clear direction
    tod_opening_drive_multiplier: float = 1.5
    # Morning Lull (10:00-11:30): 0.8x - Low probability setups
    tod_morning_lull_multiplier: float = 0.8
    # Lunch Chop (11:30-14:00): 0.7x - Avoid choppy markets
    tod_lunch_chop_multiplier: float = 0.7
    # Afternoon Trend (14:00-15:00): 1.2x - Institutional flow
    tod_afternoon_trend_multiplier: float = 1.2
    # Power Hour (15:00-16:00): 1.8x - Maximum aggression, clear trends
    tod_power_hour_multiplier: float = 1.8

    # ========== LEGACY FLAT DOLLAR FALLBACKS (for backwards compat) ==========
    # These are calculated from percentages at runtime based on account size
    max_daily_loss: float = -1000.0  # Will be overridden by percentage
    daily_profit_target: float = 750.0
    power_hour_target: float = 500.0
    max_drawdown_from_peak: float = 500.0
    max_daily_exposure_global: float = 4000.0
    max_single_position: float = 1250.0
    max_exposure_per_ticker: float = 1250.0
    max_exposure_per_sector: float = 2000.0
    max_premium_scalper: float = 1000.0
    max_premium_momentum: float = 750.0
    max_premium_macro: float = 1250.0
    max_premium_vol_sell: float = 750.0
    position_size_half: float = 500.0
    position_size_full: float = 750.0
    position_size_max: float = 1000.0
    drawdown_half_size_threshold: float = -500.0
    drawdown_halt_threshold: float = -750.0
    preserve_profit_threshold: float = 250.0

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
        # CRITICAL: Use RLock (reentrant) because some methods call other lock-acquiring methods
        # e.g., is_win_rate_pause_active() -> get_daily_win_rate()
        self._lock = threading.RLock()
        # HYDRA-style consecutive loss tracking
        self._consecutive_losses = 0
        self._consecutive_wins = 0  # VENOM: Track win streaks for compounding
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
        # VENOM COMPOUNDING: Track account size and streak multiplier
        self._account_size: float = 5000.0  # Default starting account
        self._current_streak_multiplier: float = 1.0

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
        log.info(f"GOVERNOR: can_trade called for {ticker}")

        log.debug("GOVERNOR: acquiring lock for kill switch check")
        with self._lock:
            if self._kill_switch_manual:
                return False, "Kill switch active (manual)"
        log.debug("GOVERNOR: kill switch check passed")

        # HYDRA: Check consecutive loss cooldown
        log.debug("GOVERNOR: checking cooldown")
        in_cooldown, cooldown_reason = self.is_in_cooldown()
        if in_cooldown:
            return False, cooldown_reason
        log.debug("GOVERNOR: cooldown check passed")

        # WIN RATE PRESERVATION: Check if win rate pause is active
        log.debug("GOVERNOR: checking win rate pause")
        win_rate_paused, win_rate_reason = self.is_win_rate_pause_active()
        log.info(f"GOVERNOR: win rate check done, paused={win_rate_paused}")
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

    def compute_venom_position_size(
        self,
        conviction_pct: float,
        buying_power: float,
        daily_pnl: float = 0.0,
        gex_regime: str = "unknown",
    ) -> float:
        """
        VENOM COMPOUNDING: Percentage-based position sizing with streak multipliers.

        Position size = account % based on conviction, multiplied by streak factor.
        - After wins: size UP (1.5x per win, max 3x)
        - After losses: size DOWN (0.5x)
        - Reinvest 85% of daily profits
        - GEX REGIME: Size UP when dealers are short gamma (amplified moves)

        Args:
            conviction_pct: Conviction score (0-100)
            buying_power: Current buying power
            daily_pnl: Current daily realized P/L
            gex_regime: GEX regime ("negative", "extreme_negative", "positive", "neutral", "unknown")

        Returns:
            Maximum position size in dollars
        """
        with self._lock:
            # Update account size with daily profits (reinvest portion)
            if daily_pnl > 0:
                reinvest_amount = daily_pnl * self.config.reinvest_profit_pct
                effective_account = buying_power + reinvest_amount
            else:
                effective_account = buying_power

            # Check drawdown circuit breakers (percentage-based)
            daily_pnl_pct = daily_pnl / buying_power if buying_power > 0 else 0

            if daily_pnl_pct <= -self.config.drawdown_halt_threshold_pct:
                log.warning(f"CIRCUIT BREAKER: Daily loss {daily_pnl_pct:.1%} - HALT")
                return 0.0

            # Base position size from conviction tier (% of account)
            if conviction_pct >= self.config.conviction_tier_high:
                base_pct = self.config.position_pct_high  # 20%
                tier = "HIGH (20%)"
            elif conviction_pct >= self.config.conviction_tier_mid:
                base_pct = self.config.position_pct_mid  # 15%
                tier = "MID (15%)"
            elif conviction_pct >= self.config.conviction_tier_low:
                base_pct = self.config.position_pct_low  # 10%
                tier = "LOW (10%)"
            else:
                log.info(f"Conviction {conviction_pct:.0f}% below minimum {self.config.conviction_tier_low}% - no trade")
                return 0.0

            base_size = effective_account * base_pct

            # Apply streak multiplier (SIZE UP AFTER WINS)
            streak_adjusted = base_size * self._current_streak_multiplier

            # ═══════════════════════════════════════════════════════════════
            # GEX REGIME MULTIPLIER - Trade bigger when dealers are short gamma
            # Negative GEX = dealers short gamma = amplified moves = OPPORTUNITY
            # ═══════════════════════════════════════════════════════════════
            gex_multiplier = 1.0
            gex_regime_lower = gex_regime.lower() if gex_regime else "unknown"

            if gex_regime_lower in ("extreme_negative", "very_negative", "extreme-negative"):
                gex_multiplier = self.config.gex_extreme_negative_multiplier  # 2.0x
                log.info(f"GEX EXTREME NEGATIVE: {gex_multiplier:.1f}x size boost")
            elif gex_regime_lower in ("negative", "neg"):
                gex_multiplier = self.config.gex_negative_multiplier  # 1.5x
                log.info(f"GEX NEGATIVE: {gex_multiplier:.1f}x size boost")

            gex_adjusted = streak_adjusted * gex_multiplier

            # ═══════════════════════════════════════════════════════════════
            # TIME-OF-DAY MULTIPLIER - Power Hour gets 1.8x, Lunch gets 0.7x
            # ═══════════════════════════════════════════════════════════════
            tod_multiplier = self.get_time_of_day_multiplier()
            tod_adjusted = gex_adjusted * tod_multiplier

            # Apply drawdown reduction if active
            if daily_pnl_pct <= -self.config.drawdown_half_size_threshold_pct:
                tod_adjusted = tod_adjusted / 2
                log.warning(f"DRAWDOWN PROTECTION: Position halved to ${tod_adjusted:.0f}")

            # Cap at max single position
            max_position = effective_account * self.config.max_single_position_pct
            final_size = min(tod_adjusted, max_position)

            log.info(
                f"VENOM SIZE: {conviction_pct:.0f}% -> {tier} | "
                f"Base: ${base_size:.0f} x {self._current_streak_multiplier:.2f}x streak x {gex_multiplier:.1f}x GEX x {tod_multiplier:.1f}x TOD = ${final_size:.0f} | "
                f"Streak: {self._consecutive_wins}W / {self._consecutive_losses}L | GEX: {gex_regime}"
            )
            return final_size

    def update_streak_multiplier(self, is_win: bool) -> None:
        """
        Update the streak multiplier after a trade.
        Wins increase size, losses decrease it.
        """
        with self._lock:
            if is_win:
                self._consecutive_wins += 1
                self._consecutive_losses = 0
                # Size up: 1.25x per win, capped at 2x
                new_mult = min(
                    self.config.max_win_streak_multiplier,
                    1.0 + (self._consecutive_wins * 0.25)
                )
                self._current_streak_multiplier = new_mult
                log.info(f"🔥 WIN STREAK: {self._consecutive_wins} wins -> {new_mult:.2f}x multiplier")
            else:
                self._consecutive_losses += 1
                self._consecutive_wins = 0
                # Size down: 0.5x after loss
                self._current_streak_multiplier = self.config.loss_streak_divisor
                log.warning(f"📉 LOSS: Streak reset -> {self._current_streak_multiplier:.2f}x multiplier")

    def get_time_of_day_multiplier(self) -> float:
        """
        VENOM: Get position size multiplier based on time of day.

        Different market windows have different probabilities and volatility:
        - Opening Drive (9:30-10:00 ET): 1.5x - High vol, clear direction
        - Morning Lull (10:00-11:30 ET): 0.8x - Low probability
        - Lunch Chop (11:30-14:00 ET): 0.7x - Avoid choppy markets
        - Afternoon Trend (14:00-15:00 ET): 1.2x - Institutional flow
        - Power Hour (15:00-16:00 ET): 1.8x - Maximum aggression

        Returns multiplier (0.7 to 1.8)
        """
        try:
            import pytz
            from datetime import datetime
            et = pytz.timezone('US/Eastern')
            now_et = datetime.now(et)
            hour = now_et.hour
            minute = now_et.minute
            time_decimal = hour + minute / 60.0

            # Pre-market or after-hours: no trading
            if time_decimal < 9.5 or time_decimal >= 16.0:
                return 0.5  # Minimal size outside market hours

            # Opening Drive: 9:30-10:00 ET
            if 9.5 <= time_decimal < 10.0:
                multiplier = self.config.tod_opening_drive_multiplier
                window = "OPENING_DRIVE"

            # Morning Lull: 10:00-11:30 ET
            elif 10.0 <= time_decimal < 11.5:
                multiplier = self.config.tod_morning_lull_multiplier
                window = "MORNING_LULL"

            # Lunch Chop: 11:30-14:00 ET
            elif 11.5 <= time_decimal < 14.0:
                multiplier = self.config.tod_lunch_chop_multiplier
                window = "LUNCH_CHOP"

            # Afternoon Trend: 14:00-15:00 ET
            elif 14.0 <= time_decimal < 15.0:
                multiplier = self.config.tod_afternoon_trend_multiplier
                window = "AFTERNOON_TREND"

            # Power Hour: 15:00-16:00 ET
            else:  # 15.0 <= time_decimal < 16.0
                multiplier = self.config.tod_power_hour_multiplier
                window = "POWER_HOUR"

            log.debug(f"TIME_OF_DAY: {window} @ {now_et.strftime('%H:%M')} ET -> {multiplier:.1f}x")
            return multiplier

        except Exception as e:
            log.debug(f"Time-of-day multiplier error: {e}")
            return 1.0  # Default to no adjustment

    def sync_account_size(self, buying_power: float) -> None:
        """Sync the account size from Alpaca."""
        with self._lock:
            self._account_size = buying_power
            # Update legacy dollar limits based on percentage
            self.config.max_daily_loss = -buying_power * self.config.max_daily_loss_pct
            self.config.daily_profit_target = buying_power * self.config.daily_profit_target_pct
            self.config.max_daily_exposure_global = buying_power * self.config.max_daily_exposure_pct
            self.config.max_single_position = buying_power * self.config.max_single_position_pct
            self.config.max_exposure_per_ticker = buying_power * self.config.max_exposure_per_ticker_pct
            self.config.max_exposure_per_sector = buying_power * self.config.max_exposure_per_sector_pct
            self.config.drawdown_half_size_threshold = -buying_power * self.config.drawdown_half_size_threshold_pct
            self.config.drawdown_halt_threshold = -buying_power * self.config.drawdown_halt_threshold_pct
            log.info(
                f"VENOM: Synced account ${buying_power:.2f} | "
                f"Max loss: ${self.config.max_daily_loss:.0f} | "
                f"Target: ${self.config.daily_profit_target:.0f}"
            )

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
        VENOM: Also updates streak multiplier for compounding.

        Args:
            outcome: 'win' or 'loss'
            pnl: Realized P/L in dollars (optional, for daily tracking)

        HYDRA standard: 4 consecutive losses triggers a 2-hour cooldown
        to prevent emotional/revenge trading.

        VENOM COMPOUNDING: Updates streak multiplier for position sizing.
        """
        with self._lock:
            # Track daily stats for win rate preservation
            self._reset_daily_stats_if_new_day()
            self._daily_pnl_realized += pnl

            is_win = outcome.lower() == 'win'

            # VENOM: Update streak multiplier FIRST
            self.update_streak_multiplier(is_win)

            if is_win:
                self._win_count += 1
                self._daily_win_count += 1
                # consecutive_losses reset happens in update_streak_multiplier
                log.info(
                    f"🔥 WIN (+${pnl:.2f}) | Streak: {self._consecutive_wins}W | "
                    f"Multiplier: {self._current_streak_multiplier:.2f}x | "
                    f"Total: {self._win_count}W/{self._loss_count}L"
                )
            else:
                self._loss_count += 1
                self._daily_loss_count += 1
                # consecutive_wins reset happens in update_streak_multiplier
                log.warning(
                    f"📉 LOSS (${pnl:.2f}) | Streak: {self._consecutive_losses}L | "
                    f"Multiplier: {self._current_streak_multiplier:.2f}x | "
                    f"Total: {self._win_count}W/{self._loss_count}L"
                )

                # Check if cooldown should be triggered
                if self._consecutive_losses >= self.config.consecutive_loss_threshold:
                    self._cooldown_until = datetime.now() + timedelta(hours=self.config.cooldown_hours)
                    log.warning(
                        f"⏸️ COOLDOWN ACTIVATED: {self._consecutive_losses} consecutive losses. "
                        f"Trading paused until {self._cooldown_until.strftime('%Y-%m-%d %H:%M:%S')}"
                    )

            # Log daily stats
            total_daily = self._daily_win_count + self._daily_loss_count
            daily_wr = self._daily_win_count / total_daily if total_daily > 0 else 0
            log.info(
                f"DAILY: {self._daily_win_count}W/{self._daily_loss_count}L ({daily_wr:.0%}) | "
                f"P/L: ${self._daily_pnl_realized:.2f} | Next trade: {self._current_streak_multiplier:.2f}x size"
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

    def sync_daily_exposure_from_alpaca(self) -> dict:
        """
        TIER 1 FIX: Sync daily exposure from Alpaca positions on startup.

        Calculates current exposure from open positions to prevent
        exceeding limits after restart.
        """
        import os
        try:
            import alpaca_trade_api as tradeapi

            api = tradeapi.REST(
                os.environ.get("ALPACA_API_KEY"),
                os.environ.get("ALPACA_SECRET_KEY"),
                os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            )

            # Get current positions
            positions = api.list_positions()
            total_exposure = 0.0
            position_details = []

            for pos in positions:
                market_value = abs(float(pos.market_value))
                total_exposure += market_value
                position_details.append({
                    "symbol": pos.symbol,
                    "qty": pos.qty,
                    "market_value": market_value,
                })

            log.info(
                f"SYNCED DAILY EXPOSURE FROM ALPACA: ${total_exposure:.2f} "
                f"across {len(positions)} positions"
            )

            return {
                "total_exposure": total_exposure,
                "position_count": len(positions),
                "positions": position_details,
                "synced": True,
            }

        except Exception as e:
            log.error(f"Failed to sync daily exposure from Alpaca: {e}")
            return {"synced": False, "error": str(e)}

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
