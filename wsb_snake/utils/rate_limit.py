"""
Rate Limiter with Daily Budgets for API Protection

Prevents API suspension by enforcing:
1. Per-call cooldowns (seconds between calls)
2. Per-minute limits
3. Per-hour limits
4. Daily budget caps
"""

import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional
from dataclasses import dataclass, field
import threading

from wsb_snake.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class APIBudget:
    """Budget configuration for an API service."""
    cooldown_seconds: float = 1.0      # Min seconds between calls
    max_per_minute: int = 10           # Max calls per minute
    max_per_hour: int = 100            # Max calls per hour
    max_per_day: int = 500             # Max calls per day

    # Tracking
    calls_this_minute: int = 0
    calls_this_hour: int = 0
    calls_today: int = 0
    minute_reset: datetime = field(default_factory=datetime.now)
    hour_reset: datetime = field(default_factory=datetime.now)
    day_reset: datetime = field(default_factory=datetime.now)
    last_call: float = 0.0


class RateLimiter:
    """
    Comprehensive rate limiter with daily budgets.

    Prevents API abuse that could lead to suspension.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Configure budgets for each service
        self.budgets: Dict[str, APIBudget] = {
            # AI Models - STRICT LIMITS to prevent suspension
            'openai': APIBudget(
                cooldown_seconds=2.0,
                max_per_minute=5,
                max_per_hour=50,
                max_per_day=200
            ),
            'deepseek': APIBudget(
                cooldown_seconds=2.0,
                max_per_minute=5,
                max_per_hour=50,
                max_per_day=200
            ),
            'gemini': APIBudget(
                cooldown_seconds=2.0,
                max_per_minute=5,
                max_per_hour=50,
                max_per_day=200
            ),

            # Market Data - More lenient
            'polygon': APIBudget(
                cooldown_seconds=0.2,
                max_per_minute=50,
                max_per_hour=1000,
                max_per_day=10000
            ),
            'alpaca': APIBudget(
                cooldown_seconds=0.3,
                max_per_minute=30,
                max_per_hour=500,
                max_per_day=5000
            ),
            'finnhub': APIBudget(
                cooldown_seconds=1.0,
                max_per_minute=10,
                max_per_hour=100,
                max_per_day=500
            ),

            # Other Services
            'reddit': APIBudget(
                cooldown_seconds=2.0,
                max_per_minute=5,
                max_per_hour=50,
                max_per_day=300
            ),
            'telegram': APIBudget(
                cooldown_seconds=0.5,
                max_per_minute=20,
                max_per_hour=200,
                max_per_day=1000
            ),
            'benzinga': APIBudget(
                cooldown_seconds=1.0,
                max_per_minute=10,
                max_per_hour=100,
                max_per_day=500
            ),
        }

        # Default budget for unknown services
        self.default_budget = APIBudget(
            cooldown_seconds=1.0,
            max_per_minute=10,
            max_per_hour=100,
            max_per_day=500
        )

    def _get_budget(self, service: str) -> APIBudget:
        """Get or create budget for a service."""
        if service not in self.budgets:
            self.budgets[service] = APIBudget(
                cooldown_seconds=self.default_budget.cooldown_seconds,
                max_per_minute=self.default_budget.max_per_minute,
                max_per_hour=self.default_budget.max_per_hour,
                max_per_day=self.default_budget.max_per_day
            )
        return self.budgets[service]

    def _reset_counters_if_needed(self, budget: APIBudget):
        """Reset counters if time windows have passed."""
        now = datetime.now()

        # Reset minute counter
        if now - budget.minute_reset >= timedelta(minutes=1):
            budget.calls_this_minute = 0
            budget.minute_reset = now

        # Reset hour counter
        if now - budget.hour_reset >= timedelta(hours=1):
            budget.calls_this_hour = 0
            budget.hour_reset = now

        # Reset daily counter
        if now - budget.day_reset >= timedelta(days=1):
            budget.calls_today = 0
            budget.day_reset = now

    def can_call(self, service: str) -> bool:
        """
        Check if we can make a call without blocking.
        Returns False if any limit would be exceeded.
        """
        with self._lock:
            budget = self._get_budget(service)
            self._reset_counters_if_needed(budget)

            # Check cooldown
            elapsed = time.time() - budget.last_call
            if elapsed < budget.cooldown_seconds:
                return False

            # Check rate limits
            if budget.calls_this_minute >= budget.max_per_minute:
                return False
            if budget.calls_this_hour >= budget.max_per_hour:
                return False
            if budget.calls_today >= budget.max_per_day:
                return False

            return True

    def wait_if_needed(self, service: str) -> bool:
        """
        Block until the cooldown period has passed.
        Returns False if daily/hourly limit is exhausted.
        """
        with self._lock:
            budget = self._get_budget(service)
            self._reset_counters_if_needed(budget)

            # Check if we've hit hard limits
            if budget.calls_today >= budget.max_per_day:
                log.warning(f"[{service}] Daily limit reached ({budget.max_per_day}). Blocking until tomorrow.")
                return False

            if budget.calls_this_hour >= budget.max_per_hour:
                log.warning(f"[{service}] Hourly limit reached ({budget.max_per_hour}). Wait for reset.")
                return False

            if budget.calls_this_minute >= budget.max_per_minute:
                # Wait for minute reset
                wait_time = 60 - (datetime.now() - budget.minute_reset).seconds
                if wait_time > 0:
                    log.debug(f"[{service}] Minute limit reached. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    self._reset_counters_if_needed(budget)

            # Check cooldown
            elapsed = time.time() - budget.last_call
            if elapsed < budget.cooldown_seconds:
                sleep_time = budget.cooldown_seconds - elapsed
                time.sleep(sleep_time)

            # Record the call
            budget.last_call = time.time()
            budget.calls_this_minute += 1
            budget.calls_this_hour += 1
            budget.calls_today += 1

            return True

    def record_call(self, service: str):
        """Record a call was made (for external tracking)."""
        with self._lock:
            budget = self._get_budget(service)
            self._reset_counters_if_needed(budget)
            budget.last_call = time.time()
            budget.calls_this_minute += 1
            budget.calls_this_hour += 1
            budget.calls_today += 1

    def get_remaining(self, service: str) -> Dict[str, int]:
        """Get remaining calls for each time window."""
        with self._lock:
            budget = self._get_budget(service)
            self._reset_counters_if_needed(budget)

            return {
                "minute": max(0, budget.max_per_minute - budget.calls_this_minute),
                "hour": max(0, budget.max_per_hour - budget.calls_this_hour),
                "day": max(0, budget.max_per_day - budget.calls_today),
                "used_today": budget.calls_today
            }

    def get_all_stats(self) -> Dict[str, Dict]:
        """Get stats for all tracked services."""
        stats = {}
        for service in self.budgets:
            stats[service] = self.get_remaining(service)
        return stats

    def is_budget_low(self, service: str, threshold_pct: float = 0.2) -> bool:
        """Check if remaining daily budget is below threshold."""
        with self._lock:
            budget = self._get_budget(service)
            self._reset_counters_if_needed(budget)
            remaining = budget.max_per_day - budget.calls_today
            return remaining < (budget.max_per_day * threshold_pct)


# Global rate limiter instance
limiter = RateLimiter()


def get_limiter() -> RateLimiter:
    """Get the global rate limiter."""
    return limiter
