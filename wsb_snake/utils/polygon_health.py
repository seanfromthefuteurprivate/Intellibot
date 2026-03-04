"""
Polygon API Health Check & Rate Limiting System

CRITICAL SAFETY:
- Prevents trading when Polygon is down (429/403 errors)
- Enforces rate limits with exponential backoff
- Caches responses to minimize API calls
- Monitors health and provides fallback behavior

This module is imported by cpl_gate.py to validate data availability
BEFORE allowing trades to proceed.
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class PolygonPlan(Enum):
    """Polygon.io subscription tiers with rate limits."""
    FREE = "free"           # 5 calls/min
    STARTER = "starter"     # 100 calls/min
    DEVELOPER = "developer" # Unlimited (soft limit ~300/min)


@dataclass
class PolygonHealthStatus:
    """Current health status of Polygon API."""
    is_healthy: bool = True
    last_check: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    last_error_code: Optional[int] = None
    last_error_message: str = ""
    calls_this_minute: int = 0
    rate_limit_exceeded: bool = False
    plan: PolygonPlan = PolygonPlan.STARTER  # Default assumption


@dataclass
class CacheEntry:
    """Cache entry with TTL."""
    data: Any
    cached_at: datetime
    ttl_seconds: int

    def is_expired(self) -> bool:
        return (datetime.now() - self.cached_at).total_seconds() > self.ttl_seconds


class PolygonHealthMonitor:
    """
    Monitor and enforce Polygon API health and rate limits.

    Responsibilities:
    1. Track API call rate and enforce limits
    2. Detect API failures (429, 403, 5xx)
    3. Provide circuit breaker when API is unhealthy
    4. Cache responses to reduce API load
    5. Log POLYGON_DEAD when service is down
    """

    # Rate limit configuration per plan
    RATE_LIMITS = {
        PolygonPlan.FREE: 5,        # 5 calls/min
        PolygonPlan.STARTER: 100,   # 100 calls/min
        PolygonPlan.DEVELOPER: 300, # 300 calls/min (soft limit)
    }

    # Cache TTL configuration (seconds)
    CACHE_TTLS = {
        "snapshot": 30,      # 30s - price snapshots
        "bars": 30,          # 30s - OHLC bars
        "trades": 60,        # 1min - trade flow
        "quotes": 60,        # 1min - NBBO quotes
        "options": 120,      # 2min - options chain
        "technicals": 300,   # 5min - RSI/MACD/SMA
        "indicators": 300,   # 5min - technical indicators
        "reference": 600,    # 10min - static reference data
    }

    # Health check thresholds
    MAX_CONSECUTIVE_FAILURES = 3  # Circuit breaker trips after 3 failures
    CIRCUIT_BREAKER_COOLDOWN = 60  # 60 seconds before retry after circuit break

    def __init__(self, plan: PolygonPlan = PolygonPlan.STARTER):
        self.plan = plan
        self.status = PolygonHealthStatus(plan=plan)
        self._call_timestamps = []  # Track call times for rate limiting
        self._cache: Dict[str, CacheEntry] = {}
        self._circuit_breaker_until: Optional[datetime] = None

    def get_rate_limit(self) -> int:
        """Get rate limit for current plan."""
        return self.RATE_LIMITS[self.plan]

    def can_make_request(self) -> Tuple[bool, str]:
        """
        Check if a request can be made now.

        Returns (can_proceed, reason)
        """
        now = datetime.now()

        # Check circuit breaker
        if self._circuit_breaker_until and now < self._circuit_breaker_until:
            remaining = (self._circuit_breaker_until - now).total_seconds()
            return False, f"CIRCUIT_BREAKER: Polygon unhealthy, retry in {remaining:.0f}s"

        # Reset circuit breaker if cooldown expired
        if self._circuit_breaker_until and now >= self._circuit_breaker_until:
            self._circuit_breaker_until = None
            self.status.consecutive_failures = 0
            logger.info("POLYGON_CIRCUIT_BREAKER: Reset, attempting reconnection")

        # Clean old timestamps (>60 seconds ago)
        cutoff = now - timedelta(seconds=60)
        self._call_timestamps = [ts for ts in self._call_timestamps if ts > cutoff]

        # Check rate limit
        rate_limit = self.get_rate_limit()
        current_calls = len(self._call_timestamps)

        if current_calls >= rate_limit:
            self.status.rate_limit_exceeded = True
            self.status.calls_this_minute = current_calls
            return False, f"RATE_LIMIT: {current_calls}/{rate_limit} calls/min (plan={self.plan.value})"

        self.status.rate_limit_exceeded = False
        return True, "OK"

    def record_request(self):
        """Record that a request was made."""
        self._call_timestamps.append(datetime.now())
        self.status.calls_this_minute = len(self._call_timestamps)

    def record_success(self):
        """Record successful API response."""
        self.status.is_healthy = True
        self.status.consecutive_failures = 0
        self.status.last_check = datetime.now()
        self.status.last_error_code = None
        self.status.last_error_message = ""

    def record_failure(self, error_code: int, error_message: str = ""):
        """
        Record API failure and potentially trip circuit breaker.

        Args:
            error_code: HTTP status code (429, 403, 500, etc.)
            error_message: Error message from API
        """
        now = datetime.now()
        self.status.consecutive_failures += 1
        self.status.last_error_code = error_code
        self.status.last_error_message = error_message
        self.status.last_check = now

        # Log based on error type
        if error_code == 429:
            logger.warning(f"POLYGON_429: Rate limit exceeded (plan={self.plan.value})")
        elif error_code == 403:
            logger.error(f"POLYGON_403: Authentication failed - check API key")
        elif error_code >= 500:
            logger.error(f"POLYGON_5XX: Server error {error_code}")
        else:
            logger.warning(f"POLYGON_ERROR: {error_code} - {error_message}")

        # Trip circuit breaker if too many failures
        if self.status.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            self._circuit_breaker_until = now + timedelta(seconds=self.CIRCUIT_BREAKER_COOLDOWN)
            self.status.is_healthy = False
            logger.error(
                f"POLYGON_DEAD: Circuit breaker tripped after {self.status.consecutive_failures} failures. "
                f"Cooldown until {self._circuit_breaker_until.strftime('%H:%M:%S')}"
            )

    def get_cached(self, cache_key: str) -> Optional[Any]:
        """
        Get cached data if available and not expired.

        Args:
            cache_key: Unique key for cached data

        Returns:
            Cached data or None if not found/expired
        """
        entry = self._cache.get(cache_key)
        if entry and not entry.is_expired():
            logger.debug(f"CACHE_HIT: {cache_key}")
            return entry.data

        if entry:
            # Remove expired entry
            del self._cache[cache_key]

        return None

    def set_cache(self, cache_key: str, data: Any, data_type: str = "default"):
        """
        Cache data with appropriate TTL.

        Args:
            cache_key: Unique key for cached data
            data: Data to cache
            data_type: Type of data (affects TTL)
        """
        ttl = self.CACHE_TTLS.get(data_type, 120)  # Default 2min TTL
        self._cache[cache_key] = CacheEntry(
            data=data,
            cached_at=datetime.now(),
            ttl_seconds=ttl
        )
        logger.debug(f"CACHE_SET: {cache_key} (ttl={ttl}s)")

    def get_status(self) -> Dict[str, Any]:
        """Get current health status for monitoring."""
        return {
            "is_healthy": self.status.is_healthy,
            "plan": self.plan.value,
            "rate_limit": self.get_rate_limit(),
            "calls_this_minute": self.status.calls_this_minute,
            "rate_limit_exceeded": self.status.rate_limit_exceeded,
            "consecutive_failures": self.status.consecutive_failures,
            "last_error_code": self.status.last_error_code,
            "last_error_message": self.status.last_error_message,
            "circuit_breaker_active": self._circuit_breaker_until is not None,
            "circuit_breaker_until": self._circuit_breaker_until.isoformat() if self._circuit_breaker_until else None,
            "cache_entries": len(self._cache),
            "last_check": self.status.last_check.isoformat(),
        }

    def clear_cache(self):
        """Clear all cached data (use for testing or manual reset)."""
        self._cache.clear()
        logger.info("POLYGON_CACHE: Cleared")


# Global singleton
_polygon_monitor: Optional[PolygonHealthMonitor] = None


def get_polygon_monitor() -> PolygonHealthMonitor:
    """Get singleton Polygon health monitor."""
    global _polygon_monitor
    if _polygon_monitor is None:
        # Detect plan from environment or use default
        import os
        plan_str = os.environ.get("POLYGON_PLAN", "starter").lower()
        plan = PolygonPlan.STARTER  # Safe default

        if plan_str in ["free", "basic"]:
            plan = PolygonPlan.FREE
        elif plan_str in ["developer", "advanced", "unlimited"]:
            plan = PolygonPlan.DEVELOPER

        _polygon_monitor = PolygonHealthMonitor(plan=plan)
        logger.info(f"PolygonHealthMonitor initialized (plan={plan.value}, rate_limit={_polygon_monitor.get_rate_limit()}/min)")

    return _polygon_monitor


def polygon_health_check() -> Tuple[bool, str]:
    """
    Quick health check for CPL gate.

    Returns (is_healthy, reason)

    CRITICAL: CPL should REFUSE to trade when this returns False.
    """
    monitor = get_polygon_monitor()

    # Check circuit breaker
    if not monitor.status.is_healthy:
        return False, f"POLYGON_DEAD: Circuit breaker active (failures={monitor.status.consecutive_failures})"

    # Check rate limit
    can_request, reason = monitor.can_make_request()
    if not can_request:
        return False, reason

    return True, "OK"


def get_polygon_status() -> Dict[str, Any]:
    """Get Polygon health status for monitoring/debugging."""
    monitor = get_polygon_monitor()
    return monitor.get_status()
