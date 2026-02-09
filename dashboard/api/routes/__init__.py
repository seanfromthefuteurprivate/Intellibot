"""
API Routes Package

Exports all routers for the dashboard API.
"""

from .positions import router as positions_router
from .trades import router as trades_router
from .pnl import router as pnl_router
from .signals import router as signals_router
from .market import router as market_router
from .risk import router as risk_router
from .events import router as events_router
from .account import router as account_router

__all__ = [
    "positions_router",
    "trades_router",
    "pnl_router",
    "signals_router",
    "market_router",
    "risk_router",
    "events_router",
    "account_router",
]
