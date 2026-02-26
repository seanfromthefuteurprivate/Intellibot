"""
WSB Snake Strategy Coordination Module

Central coordination layer for all trading engines to prevent conflicts,
manage exclusive ticker locks, and provide AI-powered anomaly detection.

Components:
- TickerLockManager: Exclusive locks on tickers to prevent duplicate positions
- EngineRegistry: Track all engines, states, and health
- HawkEyeMonitor: AI/rules-based anomaly detection
- StrategyCoordinator: Central hub for all trade requests
"""

from wsb_snake.coordination.ticker_lock_manager import (
    TickerLockManager,
    TickerLock,
    LockType,
    get_ticker_lock_manager,
)
from wsb_snake.coordination.engine_registry import (
    EngineRegistry,
    EngineState,
    get_engine_registry,
)
from wsb_snake.coordination.hawk_eye_monitor import (
    HawkEyeMonitor,
    HawkEyeDecision,
    get_hawk_eye_monitor,
)
from wsb_snake.coordination.strategy_coordinator import (
    StrategyCoordinator,
    TradeRequest,
    TradeResponse,
    get_strategy_coordinator,
)

__all__ = [
    # Ticker Lock Manager
    'TickerLockManager',
    'TickerLock',
    'LockType',
    'get_ticker_lock_manager',
    # Engine Registry
    'EngineRegistry',
    'EngineState',
    'get_engine_registry',
    # Hawk Eye Monitor
    'HawkEyeMonitor',
    'HawkEyeDecision',
    'get_hawk_eye_monitor',
    # Strategy Coordinator
    'StrategyCoordinator',
    'TradeRequest',
    'TradeResponse',
    'get_strategy_coordinator',
]
