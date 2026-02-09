"""
FastAPI router for risk governor status and limits.

Provides endpoints to monitor kill switch status, cooldown state,
consecutive losses, risk limits, and historical win rate.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from wsb_snake.trading.risk_governor import get_risk_governor

router = APIRouter(prefix="/api/risk", tags=["risk"])


class KillSwitchStatus(BaseModel):
    """Kill switch and cooldown status."""
    kill_switch_active: bool
    in_cooldown: bool
    cooldown_reason: str
    cooldown_until: Optional[str]
    consecutive_losses: int


class RiskLimits(BaseModel):
    """Current risk limits configuration."""
    max_daily_loss: float
    max_concurrent_positions_global: int
    max_daily_exposure_global: float
    max_positions_scalper: int
    max_positions_momentum: int
    max_positions_macro: int
    max_positions_vol_sell: int
    max_exposure_per_ticker: float
    max_exposure_per_sector: float
    max_premium_scalper: float
    max_premium_momentum: float
    max_premium_macro: float
    max_premium_vol_sell: float
    max_pct_buying_power_per_trade: float
    consecutive_loss_threshold: int
    cooldown_hours: float


class WinRateStats(BaseModel):
    """Historical win rate statistics."""
    win_rate: float
    win_count: int
    loss_count: int
    total_trades: int


@router.get("/status", response_model=KillSwitchStatus)
async def get_risk_status() -> KillSwitchStatus:
    """
    Get current risk governor status.

    Returns kill switch state, cooldown status, and consecutive loss count.
    """
    governor = get_risk_governor()

    in_cooldown, cooldown_reason = governor.is_in_cooldown()

    # Access internal state for cooldown_until (with lock for thread safety)
    cooldown_until_str: Optional[str] = None
    with governor._lock:
        if governor._cooldown_until is not None:
            cooldown_until_str = governor._cooldown_until.isoformat()
        consecutive_losses = governor._consecutive_losses

    return KillSwitchStatus(
        kill_switch_active=governor.kill_switch_active,
        in_cooldown=in_cooldown,
        cooldown_reason=cooldown_reason,
        cooldown_until=cooldown_until_str,
        consecutive_losses=consecutive_losses,
    )


@router.get("/limits", response_model=RiskLimits)
async def get_risk_limits() -> RiskLimits:
    """
    Get current risk limits configuration.

    Returns all configurable limits including max daily loss,
    position limits per engine, exposure caps, and cooldown settings.
    """
    governor = get_risk_governor()
    config = governor.config

    return RiskLimits(
        max_daily_loss=config.max_daily_loss,
        max_concurrent_positions_global=config.max_concurrent_positions_global,
        max_daily_exposure_global=config.max_daily_exposure_global,
        max_positions_scalper=config.max_positions_scalper,
        max_positions_momentum=config.max_positions_momentum,
        max_positions_macro=config.max_positions_macro,
        max_positions_vol_sell=config.max_positions_vol_sell,
        max_exposure_per_ticker=config.max_exposure_per_ticker,
        max_exposure_per_sector=config.max_exposure_per_sector,
        max_premium_scalper=config.max_premium_scalper,
        max_premium_momentum=config.max_premium_momentum,
        max_premium_macro=config.max_premium_macro,
        max_premium_vol_sell=config.max_premium_vol_sell,
        max_pct_buying_power_per_trade=config.max_pct_buying_power_per_trade,
        consecutive_loss_threshold=config.consecutive_loss_threshold,
        cooldown_hours=config.cooldown_hours,
    )


@router.get("/win-rate", response_model=WinRateStats)
async def get_win_rate() -> WinRateStats:
    """
    Get historical win rate statistics.

    Returns win rate as decimal (0-1), win/loss counts, and total trades.
    Used for Kelly position sizing calculations.
    """
    governor = get_risk_governor()

    with governor._lock:
        win_count = governor._win_count
        loss_count = governor._loss_count

    total_trades = win_count + loss_count
    win_rate = governor.get_win_rate()

    return WinRateStats(
        win_rate=win_rate,
        win_count=win_count,
        loss_count=loss_count,
        total_trades=total_trades,
    )
