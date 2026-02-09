"""
FastAPI router for positions endpoints.

Provides REST API access to open positions from AlpacaExecutor.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from wsb_snake.trading.alpaca_executor import alpaca_executor, PositionStatus


router = APIRouter(prefix="/api/positions", tags=["positions"])


class PositionResponse(BaseModel):
    """Response model for a trading position."""
    symbol: str
    option_symbol: str
    qty: int
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    entry_time: Optional[datetime]
    status: str
    engine: str


class PositionsListResponse(BaseModel):
    """Response model for list of positions."""
    positions: List[PositionResponse]
    count: int


def _get_current_price(option_symbol: str) -> float:
    """Get current price for an option from Alpaca quote."""
    quote = alpaca_executor.get_option_quote(option_symbol)
    if not quote:
        return 0.0

    bid = float(quote.get("bp", 0))
    ask = float(quote.get("ap", 0))

    # Use mid price if both available, otherwise use whichever is available
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    return bid or ask or 0.0


def _position_to_response(position) -> PositionResponse:
    """Convert AlpacaPosition to PositionResponse."""
    current_price = _get_current_price(position.option_symbol)

    # Calculate P&L based on current price
    if position.entry_price > 0 and current_price > 0:
        pnl = (current_price - position.entry_price) * position.qty * 100
        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
    else:
        pnl = position.pnl
        pnl_pct = position.pnl_pct

    return PositionResponse(
        symbol=position.symbol,
        option_symbol=position.option_symbol,
        qty=position.qty,
        entry_price=position.entry_price,
        current_price=current_price,
        pnl=pnl,
        pnl_pct=pnl_pct,
        entry_time=position.entry_time,
        status=position.status.value if isinstance(position.status, PositionStatus) else str(position.status),
        engine=position.engine,
    )


@router.get("", response_model=PositionsListResponse)
@router.get("/", response_model=PositionsListResponse)
async def list_positions():
    """
    List all open positions from AlpacaExecutor.

    Returns positions that are currently OPEN or PENDING.
    Includes real-time P&L calculations based on current market prices.
    """
    open_positions = [
        p for p in alpaca_executor.positions.values()
        if p.status in (PositionStatus.OPEN, PositionStatus.PENDING)
    ]

    position_responses = [_position_to_response(p) for p in open_positions]

    return PositionsListResponse(
        positions=position_responses,
        count=len(position_responses),
    )


@router.get("/{symbol}", response_model=PositionResponse)
async def get_position(symbol: str):
    """
    Get specific position details by symbol.

    The symbol can be either:
    - The underlying symbol (e.g., "SPY")
    - The option symbol (e.g., "SPY260125C00590000")

    If multiple positions exist for the same underlying, returns the first open one.
    """
    # First try exact match on option_symbol
    for position in alpaca_executor.positions.values():
        if position.option_symbol == symbol:
            if position.status in (PositionStatus.OPEN, PositionStatus.PENDING):
                return _position_to_response(position)

    # Then try match on underlying symbol
    for position in alpaca_executor.positions.values():
        if position.symbol.upper() == symbol.upper():
            if position.status in (PositionStatus.OPEN, PositionStatus.PENDING):
                return _position_to_response(position)

    raise HTTPException(
        status_code=404,
        detail=f"No open position found for symbol: {symbol}"
    )
