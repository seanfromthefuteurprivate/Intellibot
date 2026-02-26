"""
Ticker Lock Manager - Exclusive locks on tickers to prevent duplicate positions.

Features:
- Lock types: EXCLUSIVE (trading), SHARED (monitoring), PENDING (waiting)
- Priority-based acquisition (BERSERKER=1, Scalper=3, Momentum=5, LEAPS=6)
- Automatic timeout to prevent deadlocks
- Cleanup when position closes
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class LockType(Enum):
    """Type of lock on a ticker."""
    EXCLUSIVE = "exclusive"  # Only one holder, blocks others
    SHARED = "shared"        # Multiple holders allowed (read-only monitoring)
    PENDING = "pending"      # Lock requested but waiting


@dataclass
class TickerLock:
    """Lock on a ticker by an engine."""
    ticker: str
    engine: str
    lock_type: LockType
    priority: int  # Lower = higher priority
    acquired_at: datetime
    expires_at: Optional[datetime]
    position_id: Optional[str] = None
    request_id: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if lock has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def time_remaining(self) -> Optional[timedelta]:
        """Get time remaining on lock."""
        if self.expires_at is None:
            return None
        remaining = self.expires_at - datetime.now(timezone.utc)
        return remaining if remaining.total_seconds() > 0 else timedelta(0)


class TickerLockManager:
    """
    Manages exclusive locks on tickers to prevent duplicate positions.

    Lock Rules:
    1. Only ONE exclusive lock per ticker
    2. Higher priority (lower number) can preempt lower priority
    3. Locks have timeouts to prevent deadlocks
    4. Automatic cleanup of expired locks
    """

    # Default lock durations by engine type
    DEFAULT_LOCK_DURATIONS = {
        "berserker": timedelta(minutes=35),   # 30 min max hold + buffer
        "scalper": timedelta(minutes=20),     # 15 min max hold + buffer
        "power_hour": timedelta(minutes=30),
        "orchestrator": timedelta(minutes=15),
        "momentum": timedelta(hours=4),       # Longer holds
        "leaps": timedelta(days=1),           # Long-term
    }

    # Engine priorities (lower = higher priority)
    ENGINE_PRIORITIES = {
        "berserker": 1,
        "power_hour": 2,
        "scalper": 3,
        "orchestrator": 4,
        "momentum": 5,
        "leaps": 6,
    }

    def __init__(self):
        """Initialize lock manager."""
        self._locks: Dict[str, TickerLock] = {}  # ticker -> lock
        self._pending: Dict[str, List[TickerLock]] = {}  # ticker -> pending locks
        self._lock = threading.RLock()
        self._stats = {
            "locks_acquired": 0,
            "locks_released": 0,
            "locks_preempted": 0,
            "locks_expired": 0,
            "locks_blocked": 0,
        }

        logger.info("LOCK_MANAGER: Initialized")

    def acquire_lock(
        self,
        ticker: str,
        engine: str,
        priority: Optional[int] = None,
        lock_type: LockType = LockType.EXCLUSIVE,
        duration: Optional[timedelta] = None,
        position_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Attempt to acquire lock on ticker.

        Args:
            ticker: Symbol to lock
            engine: Engine requesting lock
            priority: Lock priority (lower = higher, defaults to engine priority)
            lock_type: Type of lock (default EXCLUSIVE)
            duration: Lock duration (defaults based on engine type)
            position_id: Associated position ID
            request_id: Associated trade request ID

        Returns:
            (success, reason)
        """
        with self._lock:
            # Clean up expired locks first
            self._cleanup_expired_locks_internal()

            # Get priority
            if priority is None:
                priority = self.ENGINE_PRIORITIES.get(engine, 10)

            # Get duration
            if duration is None:
                duration = self.DEFAULT_LOCK_DURATIONS.get(engine, timedelta(minutes=30))

            now = datetime.now(timezone.utc)
            expires_at = now + duration

            # Check existing lock
            existing = self._locks.get(ticker)

            if existing is None:
                # No existing lock - acquire immediately
                new_lock = TickerLock(
                    ticker=ticker,
                    engine=engine,
                    lock_type=lock_type,
                    priority=priority,
                    acquired_at=now,
                    expires_at=expires_at,
                    position_id=position_id,
                    request_id=request_id,
                )
                self._locks[ticker] = new_lock
                self._stats["locks_acquired"] += 1

                logger.info(f"LOCK_MANAGER: {ticker} LOCKED by {engine} (priority={priority}, expires={expires_at.strftime('%H:%M:%S')})")
                return True, "Lock acquired"

            # Lock exists - check if we can preempt
            if existing.engine == engine:
                # Same engine - refresh lock
                existing.expires_at = expires_at
                existing.position_id = position_id or existing.position_id
                logger.debug(f"LOCK_MANAGER: {ticker} lock refreshed by {engine}")
                return True, "Lock refreshed"

            if lock_type == LockType.SHARED and existing.lock_type == LockType.SHARED:
                # Both shared - allow
                logger.debug(f"LOCK_MANAGER: {ticker} shared lock for {engine}")
                return True, "Shared lock acquired"

            # Check priority preemption
            if priority < existing.priority:
                # Higher priority - preempt existing lock
                old_engine = existing.engine
                new_lock = TickerLock(
                    ticker=ticker,
                    engine=engine,
                    lock_type=lock_type,
                    priority=priority,
                    acquired_at=now,
                    expires_at=expires_at,
                    position_id=position_id,
                    request_id=request_id,
                )
                self._locks[ticker] = new_lock
                self._stats["locks_preempted"] += 1
                self._stats["locks_acquired"] += 1

                logger.warning(f"LOCK_MANAGER: {ticker} PREEMPTED from {old_engine} by {engine} (priority {priority} < {existing.priority})")
                return True, f"Lock preempted from {old_engine}"

            # Cannot acquire - lock is held by higher/equal priority
            self._stats["locks_blocked"] += 1
            remaining = existing.time_remaining()
            remaining_str = f"{remaining.total_seconds():.0f}s" if remaining else "unknown"

            logger.debug(f"LOCK_MANAGER: {ticker} BLOCKED for {engine} - held by {existing.engine} ({remaining_str} remaining)")
            return False, f"Locked by {existing.engine} (priority {existing.priority}, {remaining_str} remaining)"

    def release_lock(self, ticker: str, engine: str) -> bool:
        """
        Release lock on ticker.

        Args:
            ticker: Symbol to unlock
            engine: Engine releasing lock

        Returns:
            True if lock was released, False if not found or wrong engine
        """
        with self._lock:
            existing = self._locks.get(ticker)

            if existing is None:
                logger.debug(f"LOCK_MANAGER: {ticker} no lock to release")
                return False

            if existing.engine != engine:
                logger.warning(f"LOCK_MANAGER: {ticker} release DENIED - owned by {existing.engine}, not {engine}")
                return False

            del self._locks[ticker]
            self._stats["locks_released"] += 1

            logger.info(f"LOCK_MANAGER: {ticker} RELEASED by {engine}")
            return True

    def check_lock_status(self, ticker: str) -> Optional[TickerLock]:
        """
        Check current lock status for a ticker.

        Args:
            ticker: Symbol to check

        Returns:
            TickerLock if locked, None if free
        """
        with self._lock:
            lock = self._locks.get(ticker)
            if lock and lock.is_expired():
                # Clean up expired lock
                del self._locks[ticker]
                self._stats["locks_expired"] += 1
                return None
            return lock

    def is_locked(self, ticker: str) -> bool:
        """Check if ticker is locked."""
        return self.check_lock_status(ticker) is not None

    def get_lock_holder(self, ticker: str) -> Optional[str]:
        """Get engine holding lock on ticker."""
        lock = self.check_lock_status(ticker)
        return lock.engine if lock else None

    def get_locks_for_engine(self, engine: str) -> List[TickerLock]:
        """Get all locks held by an engine."""
        with self._lock:
            return [
                lock for lock in self._locks.values()
                if lock.engine == engine and not lock.is_expired()
            ]

    def get_all_locks(self) -> List[TickerLock]:
        """Get all active locks."""
        with self._lock:
            self._cleanup_expired_locks_internal()
            return list(self._locks.values())

    def cleanup_expired_locks(self) -> int:
        """
        Clean up expired locks.

        Returns:
            Number of locks cleaned up
        """
        with self._lock:
            return self._cleanup_expired_locks_internal()

    def _cleanup_expired_locks_internal(self) -> int:
        """Internal cleanup (must hold lock)."""
        expired = []
        for ticker, lock in self._locks.items():
            if lock.is_expired():
                expired.append(ticker)

        for ticker in expired:
            lock = self._locks.pop(ticker)
            self._stats["locks_expired"] += 1
            logger.info(f"LOCK_MANAGER: {ticker} lock EXPIRED (was held by {lock.engine})")

        return len(expired)

    def force_release(self, ticker: str) -> bool:
        """
        Force release a lock (admin use only).

        Args:
            ticker: Symbol to force unlock

        Returns:
            True if released
        """
        with self._lock:
            if ticker in self._locks:
                lock = self._locks.pop(ticker)
                self._stats["locks_released"] += 1
                logger.warning(f"LOCK_MANAGER: {ticker} FORCE RELEASED (was held by {lock.engine})")
                return True
            return False

    def get_stats(self) -> Dict:
        """Get lock manager statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_locks": len(self._locks),
                "locked_tickers": list(self._locks.keys()),
            }


# Singleton instance
_lock_manager: Optional[TickerLockManager] = None


def get_ticker_lock_manager() -> TickerLockManager:
    """Get singleton TickerLockManager instance."""
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = TickerLockManager()
    return _lock_manager
