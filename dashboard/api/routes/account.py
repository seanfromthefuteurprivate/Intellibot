"""
FastAPI router for account endpoints.

Provides REST API access to Alpaca account info and session statistics.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from wsb_snake.trading.alpaca_executor import alpaca_executor


router = APIRouter(prefix="/api/account", tags=["account"])


class AccountResponse(BaseModel):
    """Response model for Alpaca account information."""
    buying_power: float
    equity: float
    cash: float
    portfolio_value: float
    currency: str
    account_number: str
    status: str
    trading_blocked: bool
    transfers_blocked: bool
    pattern_day_trader: bool
    daytrade_count: int
    last_equity: float
    multiplier: str


class SessionStatsResponse(BaseModel):
    """Response model for current trading session statistics."""
    total_trades: int
    winning_trades: int
    win_rate: float
    total_pnl: float
    daily_pnl: float
    open_positions: int
    daily_exposure_used: float
    max_daily_exposure: float
    daily_trade_count: int
    max_concurrent_positions: int


@router.get("", response_model=AccountResponse)
@router.get("/", response_model=AccountResponse)
async def get_account():
    """
    Get Alpaca account information.

    Returns buying power, equity, cash, and other account details
    from the connected Alpaca trading account.
    """
    account = alpaca_executor.get_account()

    if not account:
        raise HTTPException(
            status_code=503,
            detail="Failed to fetch account information from Alpaca"
        )

    return AccountResponse(
        buying_power=float(account.get("buying_power", 0)),
        equity=float(account.get("equity", 0)),
        cash=float(account.get("cash", 0)),
        portfolio_value=float(account.get("portfolio_value", 0)),
        currency=account.get("currency", "USD"),
        account_number=account.get("account_number", ""),
        status=account.get("status", "UNKNOWN"),
        trading_blocked=account.get("trading_blocked", False),
        transfers_blocked=account.get("transfers_blocked", False),
        pattern_day_trader=account.get("pattern_day_trader", False),
        daytrade_count=int(account.get("daytrade_count", 0)),
        last_equity=float(account.get("last_equity", 0)),
        multiplier=account.get("multiplier", "1"),
    )


@router.get("/session", response_model=SessionStatsResponse)
async def get_session_stats():
    """
    Get current trading session statistics.

    Returns trades today, win rate, P&L, and exposure information
    for the current trading session.
    """
    stats = alpaca_executor.get_session_stats()

    return SessionStatsResponse(
        total_trades=stats.get("total_trades", 0),
        winning_trades=stats.get("winning_trades", 0),
        win_rate=stats.get("win_rate", 0.0),
        total_pnl=stats.get("total_pnl", 0.0),
        daily_pnl=stats.get("daily_pnl", 0.0),
        open_positions=stats.get("open_positions", 0),
        daily_exposure_used=alpaca_executor.daily_exposure_used,
        max_daily_exposure=alpaca_executor.MAX_DAILY_EXPOSURE,
        daily_trade_count=alpaca_executor.daily_trade_count,
        max_concurrent_positions=alpaca_executor.MAX_CONCURRENT_POSITIONS,
    )
