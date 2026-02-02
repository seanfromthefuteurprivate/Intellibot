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
    # Kill switch
    max_daily_loss: float = -500.0  # Stop all new trades if daily PnL below this
    kill_switch_manual: bool = False  # Set True to force halt

    # Global position limits
    max_concurrent_positions_global: int = 5
    max_daily_exposure_global: float = 6000.0

    # Per-engine position limits
    max_positions_scalper: int = 4
    max_positions_momentum: int = 2
    max_positions_macro: int = 2
    max_positions_vol_sell: int = 2

    # Per-ticker / per-sector exposure (dollars)
    max_exposure_per_ticker: float = 2000.0
    max_exposure_per_sector: float = 4000.0

    # Position sizing: max premium per trade by engine (base before confidence/vol scaling)
    max_premium_scalper: float = 1500.0
    max_premium_momentum: float = 1200.0
    max_premium_macro: float = 2000.0
    max_premium_vol_sell: float = 1500.0

    # Account cap: max % of buying power per trade
    max_pct_buying_power_per_trade: float = 0.10  # 10%

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
