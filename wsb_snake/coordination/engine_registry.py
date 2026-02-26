"""
Engine Registry - Track all trading engines, states, and health.

Features:
- Register engines at startup
- Track heartbeats for health monitoring
- Detect stuck/crashed engines
- Provide status for monitoring dashboard
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class EngineStatus(Enum):
    """Status of a registered engine."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    STUCK = "stuck"


@dataclass
class EngineState:
    """State of a registered engine."""
    name: str
    engine_type: str
    status: EngineStatus
    registered_at: datetime
    last_heartbeat: datetime
    last_scan_time: Optional[datetime] = None
    last_signal_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    signals_today: int = 0
    trades_today: int = 0
    current_positions: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    config: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def time_since_heartbeat(self) -> timedelta:
        """Get time since last heartbeat."""
        return datetime.now(timezone.utc) - self.last_heartbeat

    def is_healthy(self, threshold_seconds: int = 300) -> bool:
        """Check if engine is healthy (heartbeat within threshold)."""
        return self.time_since_heartbeat().total_seconds() < threshold_seconds

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "engine_type": self.engine_type,
            "status": self.status.value,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
            "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            "signals_today": self.signals_today,
            "trades_today": self.trades_today,
            "current_positions": self.current_positions,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "time_since_heartbeat_sec": self.time_since_heartbeat().total_seconds(),
            "is_healthy": self.is_healthy(),
        }


class EngineRegistry:
    """
    Registry of all trading engines for coordination and health monitoring.

    Responsibilities:
    1. Track all registered engines
    2. Monitor heartbeats for health
    3. Detect stuck/crashed engines
    4. Provide status for dashboard
    """

    # Stuck detection thresholds (seconds)
    STUCK_THRESHOLD_DEFAULT = 300  # 5 minutes
    STUCK_THRESHOLDS = {
        "scalper": 120,      # 2 minutes (scans every 30s)
        "momentum": 300,     # 5 minutes (scans every 2min)
        "leaps": 3600,       # 1 hour (scans every 30min)
        "orchestrator": 900, # 15 minutes (runs every 10min)
        "berserker": 120,    # 2 minutes (scans every 30s)
        "power_hour": 180,   # 3 minutes
    }

    def __init__(self):
        """Initialize engine registry."""
        self._engines: Dict[str, EngineState] = {}
        self._lock = threading.RLock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

        logger.info("ENGINE_REGISTRY: Initialized")

    def register_engine(
        self,
        name: str,
        engine_type: str,
        config: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Register an engine with the registry.

        Args:
            name: Unique engine name (e.g., "scalper", "momentum")
            engine_type: Type description (e.g., "SPY_0DTE_SCALPER")
            config: Engine configuration
            metadata: Additional metadata

        Returns:
            True if registered, False if already exists
        """
        with self._lock:
            if name in self._engines:
                logger.warning(f"ENGINE_REGISTRY: {name} already registered, updating")
                self._engines[name].config = config or {}
                self._engines[name].metadata = metadata or {}
                return False

            now = datetime.now(timezone.utc)
            state = EngineState(
                name=name,
                engine_type=engine_type,
                status=EngineStatus.STARTING,
                registered_at=now,
                last_heartbeat=now,
                config=config or {},
                metadata=metadata or {},
            )
            self._engines[name] = state

            logger.info(f"ENGINE_REGISTRY: {name} ({engine_type}) registered")
            return True

    def unregister_engine(self, name: str) -> bool:
        """
        Unregister an engine.

        Args:
            name: Engine name to unregister

        Returns:
            True if unregistered
        """
        with self._lock:
            if name in self._engines:
                del self._engines[name]
                logger.info(f"ENGINE_REGISTRY: {name} unregistered")
                return True
            return False

    def heartbeat(self, name: str, status: Optional[EngineStatus] = None) -> bool:
        """
        Record heartbeat from engine.

        Args:
            name: Engine name
            status: Optional status update

        Returns:
            True if recorded
        """
        with self._lock:
            if name not in self._engines:
                logger.warning(f"ENGINE_REGISTRY: heartbeat from unknown engine {name}")
                return False

            state = self._engines[name]
            state.last_heartbeat = datetime.now(timezone.utc)

            if status:
                state.status = status
            elif state.status == EngineStatus.STARTING:
                state.status = EngineStatus.RUNNING
            elif state.status == EngineStatus.STUCK:
                state.status = EngineStatus.RUNNING
                logger.info(f"ENGINE_REGISTRY: {name} recovered from STUCK state")

            return True

    def record_scan(self, name: str) -> bool:
        """Record that engine completed a scan."""
        with self._lock:
            if name not in self._engines:
                return False
            self._engines[name].last_scan_time = datetime.now(timezone.utc)
            return self.heartbeat(name)

    def record_signal(self, name: str) -> bool:
        """Record that engine generated a signal."""
        with self._lock:
            if name not in self._engines:
                return False
            state = self._engines[name]
            state.last_signal_time = datetime.now(timezone.utc)
            state.signals_today += 1
            return self.heartbeat(name)

    def record_trade(self, name: str) -> bool:
        """Record that engine executed a trade."""
        with self._lock:
            if name not in self._engines:
                return False
            state = self._engines[name]
            state.last_trade_time = datetime.now(timezone.utc)
            state.trades_today += 1
            return self.heartbeat(name)

    def record_error(self, name: str, error: str) -> bool:
        """Record an error from engine."""
        with self._lock:
            if name not in self._engines:
                return False
            state = self._engines[name]
            state.error_count += 1
            state.last_error = error
            state.status = EngineStatus.ERROR
            logger.warning(f"ENGINE_REGISTRY: {name} error #{state.error_count}: {error}")
            return True

    def update_positions(self, name: str, count: int) -> bool:
        """Update current position count for engine."""
        with self._lock:
            if name not in self._engines:
                return False
            self._engines[name].current_positions = count
            return True

    def set_status(self, name: str, status: EngineStatus) -> bool:
        """Set engine status."""
        with self._lock:
            if name not in self._engines:
                return False
            self._engines[name].status = status
            logger.info(f"ENGINE_REGISTRY: {name} status → {status.value}")
            return True

    def get_engine_state(self, name: str) -> Optional[EngineState]:
        """Get current state of an engine."""
        with self._lock:
            return self._engines.get(name)

    def get_all_engines(self) -> List[EngineState]:
        """Get all registered engines."""
        with self._lock:
            return list(self._engines.values())

    def get_running_engines(self) -> List[EngineState]:
        """Get all running engines."""
        with self._lock:
            return [e for e in self._engines.values() if e.status == EngineStatus.RUNNING]

    def detect_stuck_engines(self) -> List[str]:
        """
        Detect engines that haven't sent heartbeat within threshold.

        Returns:
            List of stuck engine names
        """
        with self._lock:
            stuck = []
            now = datetime.now(timezone.utc)

            for name, state in self._engines.items():
                if state.status in (EngineStatus.STOPPED, EngineStatus.PAUSED):
                    continue

                threshold = self.STUCK_THRESHOLDS.get(name, self.STUCK_THRESHOLD_DEFAULT)
                age = (now - state.last_heartbeat).total_seconds()

                if age > threshold:
                    stuck.append(name)
                    if state.status != EngineStatus.STUCK:
                        state.status = EngineStatus.STUCK
                        logger.error(f"ENGINE_REGISTRY: {name} STUCK - no heartbeat for {age:.0f}s")

            return stuck

    def reset_daily_stats(self) -> None:
        """Reset daily counters for all engines (call at market open)."""
        with self._lock:
            for state in self._engines.values():
                state.signals_today = 0
                state.trades_today = 0
                state.error_count = 0
                state.last_error = None

            logger.info("ENGINE_REGISTRY: Daily stats reset for all engines")

    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """
        Start background monitoring thread.

        Args:
            interval_seconds: How often to check for stuck engines
        """
        if self._monitoring:
            return

        self._monitoring = True

        def _monitor_loop():
            while self._monitoring:
                stuck = self.detect_stuck_engines()
                if stuck:
                    logger.warning(f"ENGINE_REGISTRY: Stuck engines detected: {stuck}")
                time.sleep(interval_seconds)

        self._monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info(f"ENGINE_REGISTRY: Monitoring started (interval={interval_seconds}s)")

    def stop_monitoring(self) -> None:
        """Stop background monitoring thread."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("ENGINE_REGISTRY: Monitoring stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get full registry status for monitoring."""
        with self._lock:
            stuck = self.detect_stuck_engines()

            return {
                "total_engines": len(self._engines),
                "running_engines": len([e for e in self._engines.values() if e.status == EngineStatus.RUNNING]),
                "stuck_engines": stuck,
                "engines": {name: state.to_dict() for name, state in self._engines.items()},
            }


# Singleton instance
_registry: Optional[EngineRegistry] = None


def get_engine_registry() -> EngineRegistry:
    """Get singleton EngineRegistry instance."""
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
    return _registry
