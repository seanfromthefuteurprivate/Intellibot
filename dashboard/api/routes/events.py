"""
Events API Router

FastAPI router for trading events including trades, signals, regime changes, and errors.
Supports both REST endpoint for historical events and SSE for real-time streaming.
"""

import asyncio
import json
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from collections import deque
import threading
import uuid

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/events", tags=["events"])


class EventType(str, Enum):
    """Types of trading events that can be emitted."""
    TRADE_ENTRY = "TRADE_ENTRY"
    TRADE_EXIT = "TRADE_EXIT"
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    REGIME_CHANGE = "REGIME_CHANGE"
    RISK_ALERT = "RISK_ALERT"
    COOLDOWN_START = "COOLDOWN_START"
    COOLDOWN_END = "COOLDOWN_END"


@dataclass
class TradingEvent:
    """Represents a trading event in the system."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "system"
    severity: str = "info"  # info, warning, error, critical

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "source": self.source,
            "severity": self.severity,
        }

    def to_sse(self) -> str:
        """Format event for Server-Sent Events."""
        data = json.dumps(self.to_dict())
        return f"event: {self.event_type.value}\ndata: {data}\n\n"


class EventManager:
    """
    Manages trading events collection and broadcasting.

    Features:
    - Thread-safe event storage with configurable history size
    - Real-time event broadcasting to SSE subscribers
    - Event filtering by type, time range, and severity
    - Singleton pattern for global access
    """

    _instance: Optional["EventManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern - only one EventManager instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_events: int = 1000):
        """Initialize the event manager."""
        if self._initialized:
            return

        self._events: deque = deque(maxlen=max_events)
        self._subscribers: List[asyncio.Queue] = []
        self._event_lock = threading.Lock()
        self._subscriber_lock = threading.Lock()
        self._initialized = True

    def emit(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: str = "system",
        severity: str = "info",
    ) -> TradingEvent:
        """
        Emit a new trading event.

        Args:
            event_type: Type of event (from EventType enum)
            data: Event payload data
            source: Source component that generated the event
            severity: Event severity level

        Returns:
            The created TradingEvent
        """
        event = TradingEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            data=data,
            source=source,
            severity=severity,
        )

        # Store event
        with self._event_lock:
            self._events.append(event)

        # Broadcast to subscribers (non-blocking)
        self._broadcast(event)

        return event

    def _broadcast(self, event: TradingEvent):
        """Broadcast event to all SSE subscribers."""
        with self._subscriber_lock:
            dead_subscribers = []
            for queue in self._subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    # Queue is full, subscriber is slow - mark for removal
                    dead_subscribers.append(queue)

            # Remove dead subscribers
            for queue in dead_subscribers:
                self._subscribers.remove(queue)

    def subscribe(self) -> asyncio.Queue:
        """
        Subscribe to real-time events.

        Returns:
            An asyncio.Queue that will receive TradingEvent objects
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._subscriber_lock:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        """Remove a subscriber."""
        with self._subscriber_lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    def get_events(
        self,
        event_types: Optional[List[EventType]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        severity: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[TradingEvent]:
        """
        Get historical events with optional filtering.

        Args:
            event_types: Filter by specific event types
            since: Only events after this timestamp
            until: Only events before this timestamp
            severity: Filter by severity level
            source: Filter by source component
            limit: Maximum number of events to return

        Returns:
            List of matching TradingEvent objects
        """
        with self._event_lock:
            events = list(self._events)

        # Apply filters
        if event_types:
            events = [e for e in events if e.event_type in event_types]

        if since:
            events = [e for e in events if e.timestamp >= since]

        if until:
            events = [e for e in events if e.timestamp <= until]

        if severity:
            events = [e for e in events if e.severity == severity]

        if source:
            events = [e for e in events if e.source == source]

        # Sort by timestamp descending (most recent first)
        events.sort(key=lambda e: e.timestamp, reverse=True)

        # Apply limit
        return events[:limit]

    def get_event_counts(self) -> Dict[str, int]:
        """Get count of events by type."""
        with self._event_lock:
            counts: Dict[str, int] = {}
            for event in self._events:
                type_name = event.event_type.value
                counts[type_name] = counts.get(type_name, 0) + 1
            return counts

    def clear(self):
        """Clear all stored events."""
        with self._event_lock:
            self._events.clear()

    # Convenience methods for common event types

    def emit_trade_entry(
        self,
        symbol: str,
        option_symbol: str,
        side: str,
        qty: int,
        price: float,
        pattern: str,
        confidence: float,
        engine: str = "scalper",
    ) -> TradingEvent:
        """Emit a trade entry event."""
        return self.emit(
            EventType.TRADE_ENTRY,
            {
                "symbol": symbol,
                "option_symbol": option_symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "pattern": pattern,
                "confidence": confidence,
                "engine": engine,
            },
            source="alpaca_executor",
            severity="info",
        )

    def emit_trade_exit(
        self,
        symbol: str,
        option_symbol: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        hold_time_seconds: int,
    ) -> TradingEvent:
        """Emit a trade exit event."""
        severity = "info" if pnl >= 0 else "warning"
        return self.emit(
            EventType.TRADE_EXIT,
            {
                "symbol": symbol,
                "option_symbol": option_symbol,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "exit_reason": exit_reason,
                "hold_time_seconds": hold_time_seconds,
            },
            source="alpaca_executor",
            severity=severity,
        )

    def emit_signal_generated(
        self,
        symbol: str,
        signal_type: str,
        direction: str,
        confidence: float,
        source_engine: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> TradingEvent:
        """Emit a signal generation event."""
        return self.emit(
            EventType.SIGNAL_GENERATED,
            {
                "symbol": symbol,
                "signal_type": signal_type,
                "direction": direction,
                "confidence": confidence,
                "source_engine": source_engine,
                "details": details or {},
            },
            source=source_engine,
            severity="info",
        )

    def emit_regime_change(
        self,
        old_regime: str,
        new_regime: str,
        trigger: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> TradingEvent:
        """Emit a market regime change event."""
        return self.emit(
            EventType.REGIME_CHANGE,
            {
                "old_regime": old_regime,
                "new_regime": new_regime,
                "trigger": trigger,
                "metrics": metrics or {},
            },
            source="session_regime",
            severity="warning",
        )

    def emit_risk_alert(
        self,
        alert_type: str,
        message: str,
        current_value: Any,
        threshold: Any,
        action_taken: Optional[str] = None,
    ) -> TradingEvent:
        """Emit a risk alert event."""
        return self.emit(
            EventType.RISK_ALERT,
            {
                "alert_type": alert_type,
                "message": message,
                "current_value": current_value,
                "threshold": threshold,
                "action_taken": action_taken,
            },
            source="risk_governor",
            severity="error",
        )

    def emit_cooldown_start(
        self,
        reason: str,
        duration_seconds: int,
        trigger_event: Optional[str] = None,
    ) -> TradingEvent:
        """Emit a cooldown start event."""
        return self.emit(
            EventType.COOLDOWN_START,
            {
                "reason": reason,
                "duration_seconds": duration_seconds,
                "trigger_event": trigger_event,
                "cooldown_until": (
                    datetime.utcnow().timestamp() + duration_seconds
                ),
            },
            source="risk_governor",
            severity="warning",
        )

    def emit_cooldown_end(
        self,
        reason: str,
        actual_duration_seconds: int,
    ) -> TradingEvent:
        """Emit a cooldown end event."""
        return self.emit(
            EventType.COOLDOWN_END,
            {
                "reason": reason,
                "actual_duration_seconds": actual_duration_seconds,
            },
            source="risk_governor",
            severity="info",
        )


# Global event manager instance
event_manager = EventManager()


def get_event_manager() -> EventManager:
    """Get the global EventManager instance."""
    return event_manager


# ============================================================================
# REST API Endpoints
# ============================================================================


@router.get("")
async def get_events(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    since: Optional[str] = Query(None, description="ISO timestamp - events after this time"),
    until: Optional[str] = Query(None, description="ISO timestamp - events before this time"),
    severity: Optional[str] = Query(None, description="Filter by severity (info/warning/error/critical)"),
    source: Optional[str] = Query(None, description="Filter by source component"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum events to return"),
) -> Dict[str, Any]:
    """
    Get all trading events with optional filtering.

    Returns events sorted by timestamp (most recent first).

    Event types:
    - TRADE_ENTRY: New position opened
    - TRADE_EXIT: Position closed
    - SIGNAL_GENERATED: Trading signal generated by an engine
    - REGIME_CHANGE: Market regime changed (bull/bear/choppy)
    - RISK_ALERT: Risk limit triggered
    - COOLDOWN_START: Trading paused due to risk event
    - COOLDOWN_END: Trading resumed after cooldown
    """
    # Parse event types
    event_types = None
    if event_type:
        try:
            event_types = [EventType(event_type)]
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid event_type: {event_type}. Valid types: {[e.value for e in EventType]}",
            }

    # Parse timestamps
    since_dt = None
    until_dt = None
    try:
        if since:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        if until:
            until_dt = datetime.fromisoformat(until.replace("Z", "+00:00"))
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid timestamp format: {e}",
        }

    events = event_manager.get_events(
        event_types=event_types,
        since=since_dt,
        until=until_dt,
        severity=severity,
        source=source,
        limit=limit,
    )

    return {
        "success": True,
        "count": len(events),
        "events": [e.to_dict() for e in events],
        "event_counts": event_manager.get_event_counts(),
    }


@router.get("/stream")
async def stream_events(
    event_type: Optional[str] = Query(None, description="Filter by event type (optional)"),
) -> StreamingResponse:
    """
    Server-Sent Events (SSE) endpoint for real-time event streaming.

    Connect to this endpoint to receive live trading events as they happen.
    Each event is sent as an SSE message with the event type as the event name.

    Example usage with JavaScript:
    ```javascript
    const evtSource = new EventSource('/api/events/stream');
    evtSource.addEventListener('TRADE_ENTRY', (e) => {
        const data = JSON.parse(e.data);
        console.log('Trade opened:', data);
    });
    ```
    """
    # Parse event type filter
    filter_type = None
    if event_type:
        try:
            filter_type = EventType(event_type)
        except ValueError:
            pass

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from the subscription queue."""
        queue = event_manager.subscribe()
        try:
            # Send initial connection event
            yield f"event: connected\ndata: {json.dumps({'message': 'Connected to event stream'})}\n\n"

            while True:
                try:
                    # Wait for next event with timeout (for keepalive)
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)

                    # Apply filter if specified
                    if filter_type and event.event_type != filter_type:
                        continue

                    yield event.to_sse()

                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            event_manager.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/counts")
async def get_event_counts() -> Dict[str, Any]:
    """
    Get count of events by type.

    Useful for dashboard widgets showing event activity.
    """
    return {
        "success": True,
        "counts": event_manager.get_event_counts(),
        "total": sum(event_manager.get_event_counts().values()),
    }


@router.get("/types")
async def get_event_types() -> Dict[str, Any]:
    """
    Get list of available event types.

    Useful for building filter dropdowns in the UI.
    """
    return {
        "success": True,
        "types": [
            {
                "value": e.value,
                "name": e.name,
                "description": _get_event_type_description(e),
            }
            for e in EventType
        ],
    }


def _get_event_type_description(event_type: EventType) -> str:
    """Get human-readable description for an event type."""
    descriptions = {
        EventType.TRADE_ENTRY: "New trading position opened",
        EventType.TRADE_EXIT: "Trading position closed",
        EventType.SIGNAL_GENERATED: "Trading signal generated by analysis engine",
        EventType.REGIME_CHANGE: "Market regime changed (bull/bear/choppy)",
        EventType.RISK_ALERT: "Risk management limit triggered",
        EventType.COOLDOWN_START: "Trading paused due to risk event",
        EventType.COOLDOWN_END: "Trading resumed after cooldown period",
    }
    return descriptions.get(event_type, "Unknown event type")
