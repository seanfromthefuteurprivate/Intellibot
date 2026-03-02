#!/usr/bin/env python3
"""
Circuit Breaker - Prevents infinite restart loops.

Tracks restart attempts and opens circuit after too many failures.

States:
- CLOSED: Normal operation, allow restarts
- OPEN: Too many failures, STOP restarting
- HALF_OPEN: Testing if system recovered

Usage:
    cb = CircuitBreaker(max_restarts=3, time_window_minutes=5)

    can_restart, message = cb.can_restart(reason="service down")
    if can_restart:
        restart_service()
        cb.record_restart(reason="service down", success=True)
    else:
        alert_human(message)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class RestartEvent:
    """A single restart attempt."""
    timestamp: datetime
    reason: str
    success: bool


class CircuitBreaker:
    """
    Prevents infinite restart loops by tracking failure patterns.

    After max_restarts failures in time_window, the circuit OPENS
    and no more restarts are allowed until cooldown period passes.
    """

    def __init__(
        self,
        max_restarts: int = 3,
        time_window_minutes: int = 5,
        cooldown_minutes: int = 30,
        state_file: Path = None
    ):
        """
        Initialize circuit breaker.

        Args:
            max_restarts: Maximum restarts allowed in time window
            time_window_minutes: Time window for counting restarts
            cooldown_minutes: How long to wait before trying again after opening
            state_file: File to persist state (default: /tmp/circuit_breaker_state.json)
        """
        self.max_restarts = max_restarts
        self.time_window = timedelta(minutes=time_window_minutes)
        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.state_file = state_file or Path("/tmp/circuit_breaker_state.json")

        self.state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self.restart_history: List[RestartEvent] = []
        self.opened_at: Optional[datetime] = None

        self._load_state()

    def _load_state(self):
        """Load persisted state from disk."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.state = data.get("state", "CLOSED")
                self.opened_at = (
                    datetime.fromisoformat(data["opened_at"])
                    if data.get("opened_at")
                    else None
                )
                self.restart_history = [
                    RestartEvent(
                        timestamp=datetime.fromisoformat(e["timestamp"]),
                        reason=e["reason"],
                        success=e["success"],
                    )
                    for e in data.get("restart_history", [])
                ]
                logger.info(
                    f"Circuit breaker loaded: state={self.state}, "
                    f"history={len(self.restart_history)} events"
                )
            except Exception as e:
                logger.warning(f"Failed to load circuit breaker state: {e}")

    def _save_state(self):
        """Persist state to disk."""
        try:
            data = {
                "state": self.state,
                "opened_at": self.opened_at.isoformat() if self.opened_at else None,
                "restart_history": [
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "reason": e.reason,
                        "success": e.success,
                    }
                    for e in self.restart_history
                ],
            }
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save circuit breaker state: {e}")

    def _clean_old_history(self):
        """Remove restart events outside the time window."""
        now = datetime.now()
        self.restart_history = [
            e for e in self.restart_history if now - e.timestamp < self.time_window
        ]

    def can_restart(self, reason: str) -> tuple[bool, str]:
        """
        Check if restart is allowed.

        Args:
            reason: Reason for restart (for logging)

        Returns:
            (allowed: bool, message: str)
        """
        now = datetime.now()
        self._clean_old_history()

        if self.state == "OPEN":
            # Check if cooldown period has passed
            if self.opened_at and (now - self.opened_at) > self.cooldown:
                # Try to transition to HALF_OPEN
                self.state = "HALF_OPEN"
                self._save_state()
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True, "Circuit breaker HALF_OPEN: Testing system recovery"
            else:
                remaining = (
                    self.cooldown - (now - self.opened_at)
                    if self.opened_at
                    else self.cooldown
                )
                remaining_minutes = remaining.total_seconds() / 60
                message = (
                    f"Circuit breaker OPEN: Wait {remaining_minutes:.1f}m before retry. "
                    f"Manual intervention required."
                )
                return False, message

        # Count recent restarts
        recent_restarts = len(self.restart_history)

        if recent_restarts >= self.max_restarts:
            # Too many restarts, OPEN the circuit
            self.state = "OPEN"
            self.opened_at = now
            self._save_state()

            window_minutes = self.time_window.total_seconds() / 60
            message = (
                f"Circuit breaker OPENED: {recent_restarts} restarts in "
                f"{window_minutes:.1f}m. System unstable."
            )
            logger.critical(message)
            return False, message

        window_minutes = self.time_window.total_seconds() / 60
        message = (
            f"Restart allowed: {recent_restarts}/{self.max_restarts} "
            f"in {window_minutes:.1f}m window"
        )
        return True, message

    def record_restart(self, reason: str, success: bool):
        """
        Record a restart attempt.

        Args:
            reason: Reason for restart
            success: Whether restart was successful
        """
        event = RestartEvent(timestamp=datetime.now(), reason=reason, success=success)
        self.restart_history.append(event)

        if self.state == "HALF_OPEN" and success:
            # Successful restart in HALF_OPEN, close the circuit
            self.state = "CLOSED"
            self.restart_history.clear()
            self.opened_at = None
            logger.info("Circuit breaker CLOSED after successful recovery")

        self._save_state()

    def reset(self):
        """Manually reset the circuit breaker (admin action)."""
        logger.info("Circuit breaker manually reset")
        self.state = "CLOSED"
        self.restart_history.clear()
        self.opened_at = None
        self._save_state()

    def get_status(self) -> dict:
        """Get current status."""
        return {
            "state": self.state,
            "recent_restarts": len(self.restart_history),
            "max_restarts": self.max_restarts,
            "time_window_minutes": self.time_window.total_seconds() / 60,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "cooldown_minutes": self.cooldown.total_seconds() / 60,
            "history": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "reason": e.reason,
                    "success": e.success,
                }
                for e in self.restart_history[-10:]  # Last 10 events
            ],
        }


def main():
    """CLI for testing and manual control."""
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python circuit_breaker.py status    - Show current status")
        print("  python circuit_breaker.py reset     - Reset circuit breaker")
        print("  python circuit_breaker.py test      - Run test scenario")
        sys.exit(1)

    cb = CircuitBreaker()

    if sys.argv[1] == "status":
        status = cb.get_status()
        print(json.dumps(status, indent=2))

    elif sys.argv[1] == "reset":
        cb.reset()
        print("Circuit breaker reset to CLOSED state")

    elif sys.argv[1] == "test":
        print("Running test scenario: 5 rapid restart attempts...")
        for i in range(5):
            allowed, msg = cb.can_restart(f"test_restart_{i}")
            print(f"\nAttempt {i + 1}:")
            print(f"  Allowed: {allowed}")
            print(f"  Message: {msg}")

            if allowed:
                cb.record_restart(f"test_restart_{i}", success=False)
            else:
                print("\n🛑 Circuit breaker OPEN - no more restarts allowed")
                break

        print("\nFinal status:")
        print(json.dumps(cb.get_status(), indent=2))

    else:
        print(f"Unknown command: {sys.argv[1]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
