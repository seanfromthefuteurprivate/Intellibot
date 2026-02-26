"""
Strategy Coordinator - Central hub for all trade requests.

All trading engines submit requests through this coordinator, which:
1. Checks for conflicts via Hawk Eye
2. Acquires ticker locks
3. Queues or executes trades based on priority
4. Ensures graceful degradation if coordinator fails
"""

import threading
import time
import uuid
import queue
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

from wsb_snake.utils.logger import get_logger
from wsb_snake.coordination.ticker_lock_manager import (
    TickerLockManager,
    get_ticker_lock_manager,
    LockType,
)
from wsb_snake.coordination.engine_registry import (
    EngineRegistry,
    get_engine_registry,
    EngineStatus,
)
from wsb_snake.coordination.hawk_eye_monitor import (
    HawkEyeMonitor,
    get_hawk_eye_monitor,
    HawkEyeDecision,
)

logger = get_logger(__name__)


@dataclass
class TradeRequest:
    """A trade request from any engine."""
    request_id: str
    engine: str  # "scalper", "momentum", "leaps", "berserker", "orchestrator", "power_hour"
    ticker: str
    direction: str  # "long" or "short"
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float  # 0-100
    pattern: str = ""
    priority: int = 5  # 1=highest (berserker), 6=lowest (leaps)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expiry_preference: str = "0dte"  # "0dte", "weekly", "monthly", "leaps"

    # Optional HYDRA/GEX context
    gex_regime: Optional[str] = None
    hydra_direction: Optional[str] = None
    flow_bias: Optional[str] = None

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "engine": self.engine,
            "ticker": self.ticker,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "confidence": self.confidence,
            "pattern": self.pattern,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "expiry_preference": self.expiry_preference,
            "gex_regime": self.gex_regime,
            "hydra_direction": self.hydra_direction,
        }


@dataclass
class TradeResponse:
    """Response from coordinator after processing request."""
    request_id: str
    approved: bool
    executed: bool = False
    queued: bool = False
    queue_position: int = 0
    reason: str = ""
    position_id: Optional[str] = None
    coordinator_error: bool = False
    hawk_eye_decision: Optional[HawkEyeDecision] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "approved": self.approved,
            "executed": self.executed,
            "queued": self.queued,
            "queue_position": self.queue_position,
            "reason": self.reason,
            "position_id": self.position_id,
            "coordinator_error": self.coordinator_error,
            "timestamp": self.timestamp.isoformat(),
        }


class StrategyCoordinator:
    """
    Central coordination hub for all trading strategies.

    Responsibilities:
    1. Receive trade requests from all engines
    2. Check for conflicts via Hawk Eye
    3. Acquire ticker locks before execution
    4. Queue conflicting requests for later processing
    5. Enforce cross-engine position limits
    6. Execute approved trades through Alpaca executor

    Engine Priorities (lower = higher):
    - berserker: 1 (GEX-edge trades)
    - power_hour: 2 (Power hour assault)
    - scalper: 3 (SPY 0DTE scalper)
    - orchestrator: 4 (Main pipeline)
    - momentum: 5 (Small-cap momentum)
    - leaps: 6 (Long-dated thesis)
    """

    # Engine priorities
    ENGINE_PRIORITIES = {
        "berserker": 1,
        "power_hour": 2,
        "scalper": 3,
        "orchestrator": 4,
        "momentum": 5,
        "leaps": 6,
    }

    def __init__(self):
        """Initialize strategy coordinator."""
        self.lock_manager = get_ticker_lock_manager()
        self.engine_registry = get_engine_registry()
        self.hawk_eye = get_hawk_eye_monitor()

        # Request queue (priority, request)
        self._request_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._pending_requests: Dict[str, TradeRequest] = {}  # request_id -> request
        self._lock = threading.RLock()

        # Execution callback (set by main.py)
        self._executor: Optional[Callable] = None

        # Stats
        self._stats = {
            "requests_total": 0,
            "requests_approved": 0,
            "requests_blocked": 0,
            "requests_queued": 0,
            "requests_executed": 0,
            "coordinator_errors": 0,
        }

        self._running = False
        self._queue_thread: Optional[threading.Thread] = None

        logger.info("COORDINATOR: Initialized - central hub for all engines")

    def start(self) -> None:
        """Start the coordinator and queue processor."""
        if self._running:
            return

        self._running = True
        self._queue_thread = threading.Thread(target=self._process_queue_loop, daemon=True)
        self._queue_thread.start()

        # Start engine registry monitoring
        self.engine_registry.start_monitoring()

        logger.info("COORDINATOR: Started - queue processor active")

    def stop(self) -> None:
        """Stop the coordinator."""
        self._running = False
        if self._queue_thread:
            self._queue_thread.join(timeout=5)
        self.engine_registry.stop_monitoring()
        logger.info("COORDINATOR: Stopped")

    def set_executor(self, executor_fn: Callable) -> None:
        """
        Set the execution callback function.

        Args:
            executor_fn: Function that takes (ticker, direction, entry_price, target, stop, confidence, metadata)
                        and returns position_id or None
        """
        self._executor = executor_fn
        logger.info("COORDINATOR: Executor callback set")

    def register_engine(self, name: str, engine_type: str, config: Optional[Dict] = None) -> None:
        """Register an engine with the coordinator."""
        self.engine_registry.register_engine(name, engine_type, config)

    def submit_trade_request(self, request: TradeRequest) -> TradeResponse:
        """
        Submit a trade request for coordination and execution.

        Args:
            request: TradeRequest from any engine

        Returns:
            TradeResponse with approval status and execution result
        """
        self._stats["requests_total"] += 1

        # Record engine activity
        self.engine_registry.heartbeat(request.engine)
        self.engine_registry.record_signal(request.engine)

        try:
            return self._process_request(request)
        except Exception as e:
            logger.error(f"COORDINATOR: Error processing request - {e}")
            self._stats["coordinator_errors"] += 1
            return TradeResponse(
                request_id=request.request_id,
                approved=False,
                reason=f"Coordinator error: {str(e)}",
                coordinator_error=True,
            )

    def _process_request(self, request: TradeRequest) -> TradeResponse:
        """Process a single trade request."""
        with self._lock:
            # Get current positions from executor
            current_positions = self._get_current_positions()

            # Get pending requests (excluding this one)
            pending = [r for r in self._pending_requests.values() if r.request_id != request.request_id]

            # Run Hawk Eye checks
            hawk_decision = self.hawk_eye.check_trade_request(
                request=request,
                current_positions=current_positions,
                pending_requests=pending,
                engine_registry=self.engine_registry,
            )

            if not hawk_decision.allowed:
                self._stats["requests_blocked"] += 1
                logger.info(f"COORDINATOR: BLOCKED {request.engine}/{request.ticker} - {hawk_decision.reason}")
                return TradeResponse(
                    request_id=request.request_id,
                    approved=False,
                    reason=hawk_decision.reason,
                    hawk_eye_decision=hawk_decision,
                )

            # Try to acquire ticker lock
            lock_acquired, lock_reason = self.lock_manager.acquire_lock(
                ticker=request.ticker,
                engine=request.engine,
                priority=request.priority,
                request_id=request.request_id,
            )

            if not lock_acquired:
                # Queue the request for later
                self._pending_requests[request.request_id] = request
                self._request_queue.put((request.priority, time.time(), request))
                self._stats["requests_queued"] += 1

                queue_size = self._request_queue.qsize()
                logger.info(f"COORDINATOR: QUEUED {request.engine}/{request.ticker} - {lock_reason} (position {queue_size})")
                return TradeResponse(
                    request_id=request.request_id,
                    approved=True,
                    queued=True,
                    queue_position=queue_size,
                    reason=f"Queued: {lock_reason}",
                    hawk_eye_decision=hawk_decision,
                )

            # Lock acquired - execute trade
            self._stats["requests_approved"] += 1
            position_id = self._execute_trade(request)

            if position_id:
                self._stats["requests_executed"] += 1
                self.engine_registry.record_trade(request.engine)
                logger.info(f"COORDINATOR: EXECUTED {request.engine}/{request.ticker} -> {position_id}")
                return TradeResponse(
                    request_id=request.request_id,
                    approved=True,
                    executed=True,
                    position_id=position_id,
                    reason="Trade executed",
                    hawk_eye_decision=hawk_decision,
                )
            else:
                # Execution failed - release lock
                self.lock_manager.release_lock(request.ticker, request.engine)
                return TradeResponse(
                    request_id=request.request_id,
                    approved=True,
                    executed=False,
                    reason="Execution failed (executor returned None)",
                    hawk_eye_decision=hawk_decision,
                )

    def _execute_trade(self, request: TradeRequest) -> Optional[str]:
        """Execute trade through the configured executor."""
        if self._executor is None:
            logger.warning("COORDINATOR: No executor configured - cannot execute trade")
            return None

        try:
            metadata = {
                "engine": request.engine,
                "pattern": request.pattern,
                "request_id": request.request_id,
                "coordinator": True,
                "gex_regime": request.gex_regime,
                "hydra_direction": request.hydra_direction,
                "flow_bias": request.flow_bias,
                **request.metadata,
            }

            position_id = self._executor(
                ticker=request.ticker,
                direction=request.direction,
                entry_price=request.entry_price,
                target_price=request.target_price,
                stop_loss=request.stop_loss,
                confidence=request.confidence,
                expiry_preference=request.expiry_preference,
                metadata=metadata,
            )

            return position_id

        except Exception as e:
            logger.error(f"COORDINATOR: Executor error - {e}")
            return None

    def _get_current_positions(self) -> List[Dict]:
        """Get current positions from Alpaca executor."""
        try:
            from wsb_snake.trading.alpaca_executor import get_alpaca_executor
            executor = get_alpaca_executor()
            positions = executor.get_all_positions()
            return [
                {
                    "ticker": p.underlying,
                    "direction": p.direction,
                    "engine": p.metadata.get("engine", "unknown") if hasattr(p, "metadata") else "unknown",
                }
                for p in positions
            ]
        except Exception as e:
            logger.warning(f"COORDINATOR: Could not get positions - {e}")
            return []

    def _process_queue_loop(self) -> None:
        """Background loop to process queued requests."""
        while self._running:
            try:
                # Check for items in queue
                try:
                    priority, timestamp, request = self._request_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Remove from pending
                self._pending_requests.pop(request.request_id, None)

                # Check if request is still valid (not too old)
                age = (datetime.now(timezone.utc) - request.timestamp).total_seconds()
                if age > 300:  # 5 minute timeout
                    logger.info(f"COORDINATOR: Dropping stale queued request {request.request_id} (age={age:.0f}s)")
                    continue

                # Try to process again
                logger.info(f"COORDINATOR: Processing queued request {request.request_id}")
                response = self._process_request(request)

                if response.queued:
                    # Still can't execute - re-queue
                    logger.debug(f"COORDINATOR: Re-queued {request.request_id}")

            except Exception as e:
                logger.error(f"COORDINATOR: Queue processing error - {e}")

            time.sleep(0.5)  # Rate limit queue processing

    def release_position_lock(self, ticker: str, engine: str) -> bool:
        """
        Release lock when position is closed.

        Should be called by executor when position exits.
        """
        return self.lock_manager.release_lock(ticker, engine)

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status for monitoring."""
        with self._lock:
            return {
                "running": self._running,
                "stats": self._stats,
                "queue_size": self._request_queue.qsize(),
                "pending_requests": len(self._pending_requests),
                "lock_manager": self.lock_manager.get_stats(),
                "hawk_eye": self.hawk_eye.get_stats(),
                "engine_registry": self.engine_registry.get_status(),
            }

    def get_pending_requests(self) -> List[Dict]:
        """Get list of pending requests."""
        with self._lock:
            return [r.to_dict() for r in self._pending_requests.values()]


# Singleton instance
_coordinator: Optional[StrategyCoordinator] = None


def get_strategy_coordinator() -> StrategyCoordinator:
    """Get singleton StrategyCoordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = StrategyCoordinator()
    return _coordinator


def create_trade_request(
    engine: str,
    ticker: str,
    direction: str,
    entry_price: float,
    target_price: float,
    stop_loss: float,
    confidence: float,
    pattern: str = "",
    expiry_preference: str = "0dte",
    **kwargs
) -> TradeRequest:
    """
    Helper function to create a TradeRequest.

    Args:
        engine: Engine name
        ticker: Symbol
        direction: "long" or "short"
        entry_price: Entry price
        target_price: Target exit price
        stop_loss: Stop loss price
        confidence: Confidence score 0-100
        pattern: Pattern name
        expiry_preference: Option expiry preference
        **kwargs: Additional metadata

    Returns:
        TradeRequest ready to submit
    """
    return TradeRequest(
        request_id=f"{engine}_{ticker}_{uuid.uuid4().hex[:8]}",
        engine=engine,
        ticker=ticker,
        direction=direction,
        entry_price=entry_price,
        target_price=target_price,
        stop_loss=stop_loss,
        confidence=confidence,
        pattern=pattern,
        priority=StrategyCoordinator.ENGINE_PRIORITIES.get(engine, 5),
        expiry_preference=expiry_preference,
        gex_regime=kwargs.get("gex_regime"),
        hydra_direction=kwargs.get("hydra_direction"),
        flow_bias=kwargs.get("flow_bias"),
        metadata=kwargs,
    )
