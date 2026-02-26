"""
Hawk Eye Monitor - AI/rules-based anomaly detection for trade requests.

Checks (must complete in <50ms):
1. Duplicate ticker detection (same ticker, different engines)
2. Correlated exposure (SPY + QQQ = too much index exposure)
3. Sector concentration (>40% in one sector)
4. Direction conflict (engine A long, engine B short same ticker)
5. Velocity anomaly (too many trades in short window)
6. Engine health (detect stuck/crashed engines)
"""

import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

from wsb_snake.utils.logger import get_logger

if TYPE_CHECKING:
    from wsb_snake.coordination.strategy_coordinator import TradeRequest

logger = get_logger(__name__)


@dataclass
class HawkEyeDecision:
    """Decision from Hawk Eye monitor."""
    allowed: bool
    reason: str
    confidence: float  # 0-100
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    check_latency_ms: float = 0.0
    audit_log_id: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "check_latency_ms": self.check_latency_ms,
            "audit_log_id": self.audit_log_id,
        }


@dataclass
class TradeVelocity:
    """Track trade velocity for rate limiting."""
    ticker: str
    engine: str
    timestamp: datetime


class HawkEyeMonitor:
    """
    AI-powered monitoring and anomaly detection.

    Rules Engine (must complete in <50ms):
    - Duplicate ticker detection
    - Correlated exposure limits
    - Sector concentration limits
    - Direction conflict detection
    - Trade velocity limits
    - Engine health verification

    All checks are synchronous and fast - no blocking I/O.
    """

    # Correlation pairs that count as single exposure (ticker1, ticker2) -> correlation
    CORRELATED_PAIRS = {
        ("SPY", "QQQ"): 0.85,
        ("SPY", "IWM"): 0.78,
        ("QQQ", "IWM"): 0.72,
        ("GLD", "SLV"): 0.82,
        ("GLD", "GDX"): 0.75,
        ("GDX", "GDXJ"): 0.90,
        ("NVDA", "AMD"): 0.72,
        ("AAPL", "MSFT"): 0.65,
        ("META", "GOOGL"): 0.68,
        ("XLE", "USO"): 0.78,
    }

    # Sector mapping
    TICKER_SECTORS = {
        "SPY": "index",
        "QQQ": "index",
        "IWM": "index",
        "SPX": "index",
        "GLD": "metals",
        "SLV": "metals",
        "GDX": "metals",
        "GDXJ": "metals",
        "USO": "energy",
        "XLE": "energy",
        "NVDA": "tech",
        "AMD": "tech",
        "AAPL": "tech",
        "MSFT": "tech",
        "META": "tech",
        "GOOGL": "tech",
        "AMZN": "tech",
        "TSLA": "auto",
        "RKLB": "space",
        "ASTS": "space",
        "LUNR": "space",
        "PL": "space",
    }

    # Limits
    MAX_SECTOR_EXPOSURE_PCT = 40  # Max % of portfolio in one sector
    MAX_CORRELATED_EXPOSURE_PCT = 50  # Max % in correlated assets
    MAX_TRADES_PER_MINUTE = 5  # Rate limit per engine
    MAX_TRADES_PER_TICKER_PER_HOUR = 3  # Rate limit per ticker

    def __init__(self):
        """Initialize Hawk Eye monitor."""
        self._lock = threading.RLock()
        self._velocity_log: List[TradeVelocity] = []
        self._audit_log: List[Dict] = []
        self._stats = {
            "checks_total": 0,
            "checks_allowed": 0,
            "checks_blocked": 0,
            "avg_latency_ms": 0.0,
        }

        logger.info("HAWK_EYE: Initialized - watching for anomalies")

    def check_trade_request(
        self,
        request: "TradeRequest",
        current_positions: List[Dict],
        pending_requests: List["TradeRequest"],
        engine_registry: Optional[Any] = None,
    ) -> HawkEyeDecision:
        """
        Check if trade request passes all anomaly checks.

        Args:
            request: Trade request to check
            current_positions: Current open positions
            pending_requests: Other pending trade requests
            engine_registry: Optional engine registry for health check

        Returns:
            HawkEyeDecision with allowed/blocked and reasoning
        """
        start = time.time()
        warnings = []
        recommendations = []

        self._stats["checks_total"] += 1
        audit_id = f"hawk_{uuid.uuid4().hex[:8]}"

        # Run all checks
        checks = [
            ("duplicate_ticker", self._check_duplicate_ticker),
            ("correlated_exposure", self._check_correlated_exposure),
            ("sector_concentration", self._check_sector_concentration),
            ("direction_conflict", self._check_direction_conflict),
            ("velocity_limit", self._check_velocity_limit),
        ]

        for check_name, check_fn in checks:
            passed, reason = check_fn(request, current_positions, pending_requests)

            if not passed:
                latency_ms = (time.time() - start) * 1000
                self._stats["checks_blocked"] += 1
                self._update_avg_latency(latency_ms)

                decision = HawkEyeDecision(
                    allowed=False,
                    reason=f"{check_name}: {reason}",
                    confidence=95.0,
                    check_latency_ms=latency_ms,
                    audit_log_id=audit_id,
                )
                self._log_decision(request, decision)
                return decision

        # Check engine health (advisory only)
        if engine_registry:
            healthy, health_reason = self._check_engine_health(request, engine_registry)
            if not healthy:
                warnings.append(f"Engine health warning: {health_reason}")

        # All checks passed
        latency_ms = (time.time() - start) * 1000
        self._stats["checks_allowed"] += 1
        self._update_avg_latency(latency_ms)

        # Record velocity
        self._record_velocity(request)

        decision = HawkEyeDecision(
            allowed=True,
            reason="All checks passed",
            confidence=90.0,
            warnings=warnings,
            recommendations=recommendations,
            check_latency_ms=latency_ms,
            audit_log_id=audit_id,
        )
        self._log_decision(request, decision)
        return decision

    def _check_duplicate_ticker(
        self,
        request: "TradeRequest",
        current_positions: List[Dict],
        pending_requests: List["TradeRequest"],
    ) -> Tuple[bool, str]:
        """Block if same ticker already has active position from different engine."""
        ticker = request.ticker

        # Check current positions
        for pos in current_positions:
            pos_ticker = pos.get("ticker", pos.get("symbol", ""))
            if pos_ticker == ticker:
                pos_engine = pos.get("engine", "unknown")
                if pos_engine != request.engine:
                    return False, f"Position already exists from {pos_engine}"

        # Check pending requests
        for pending in pending_requests:
            if pending.ticker == ticker and pending.engine != request.engine:
                return False, f"Pending request from {pending.engine}"

        return True, "OK"

    def _check_correlated_exposure(
        self,
        request: "TradeRequest",
        current_positions: List[Dict],
        pending_requests: List["TradeRequest"],
    ) -> Tuple[bool, str]:
        """Block if correlated pair exposure too high."""
        ticker = request.ticker

        # Find correlated tickers
        correlated_tickers = []
        for (t1, t2), correlation in self.CORRELATED_PAIRS.items():
            if ticker == t1:
                correlated_tickers.append((t2, correlation))
            elif ticker == t2:
                correlated_tickers.append((t1, correlation))

        if not correlated_tickers:
            return True, "OK"

        # Check exposure to correlated assets
        for corr_ticker, correlation in correlated_tickers:
            for pos in current_positions:
                pos_ticker = pos.get("ticker", pos.get("symbol", ""))
                if pos_ticker == corr_ticker:
                    return False, f"Correlated with existing {corr_ticker} position (r={correlation:.2f})"

            for pending in pending_requests:
                if pending.ticker == corr_ticker:
                    return False, f"Correlated with pending {corr_ticker} request (r={correlation:.2f})"

        return True, "OK"

    def _check_sector_concentration(
        self,
        request: "TradeRequest",
        current_positions: List[Dict],
        pending_requests: List["TradeRequest"],
    ) -> Tuple[bool, str]:
        """Warn if sector concentration > limit."""
        ticker = request.ticker
        sector = self.TICKER_SECTORS.get(ticker, "other")

        if sector == "other":
            return True, "OK"

        # Count positions in same sector
        sector_count = 0
        total_count = len(current_positions) + 1  # +1 for new request

        for pos in current_positions:
            pos_ticker = pos.get("ticker", pos.get("symbol", ""))
            pos_sector = self.TICKER_SECTORS.get(pos_ticker, "other")
            if pos_sector == sector:
                sector_count += 1

        # Calculate concentration
        if total_count > 0:
            concentration_pct = ((sector_count + 1) / total_count) * 100
            if concentration_pct > self.MAX_SECTOR_EXPOSURE_PCT:
                return False, f"{sector} sector at {concentration_pct:.0f}% (limit {self.MAX_SECTOR_EXPOSURE_PCT}%)"

        return True, "OK"

    def _check_direction_conflict(
        self,
        request: "TradeRequest",
        current_positions: List[Dict],
        pending_requests: List["TradeRequest"],
    ) -> Tuple[bool, str]:
        """Block if another engine has opposite direction on same ticker."""
        ticker = request.ticker
        direction = request.direction.lower()

        for pending in pending_requests:
            if pending.ticker == ticker:
                pending_dir = pending.direction.lower()
                if (direction == "long" and pending_dir == "short") or \
                   (direction == "short" and pending_dir == "long"):
                    return False, f"Direction conflict with {pending.engine} ({pending_dir})"

        return True, "OK"

    def _check_velocity_limit(
        self,
        request: "TradeRequest",
        current_positions: List[Dict],
        pending_requests: List["TradeRequest"],
    ) -> Tuple[bool, str]:
        """Block if too many trades in short window."""
        with self._lock:
            now = datetime.now(timezone.utc)
            cutoff_minute = now - timedelta(minutes=1)
            cutoff_hour = now - timedelta(hours=1)

            # Clean old entries
            self._velocity_log = [
                v for v in self._velocity_log
                if v.timestamp > cutoff_hour
            ]

            # Check per-engine rate (per minute)
            engine_count = sum(
                1 for v in self._velocity_log
                if v.engine == request.engine and v.timestamp > cutoff_minute
            )
            if engine_count >= self.MAX_TRADES_PER_MINUTE:
                return False, f"{request.engine} at {engine_count} trades/min (limit {self.MAX_TRADES_PER_MINUTE})"

            # Check per-ticker rate (per hour)
            ticker_count = sum(
                1 for v in self._velocity_log
                if v.ticker == request.ticker and v.timestamp > cutoff_hour
            )
            if ticker_count >= self.MAX_TRADES_PER_TICKER_PER_HOUR:
                return False, f"{request.ticker} at {ticker_count} trades/hour (limit {self.MAX_TRADES_PER_TICKER_PER_HOUR})"

            return True, "OK"

    def _check_engine_health(
        self,
        request: "TradeRequest",
        engine_registry: Any,
    ) -> Tuple[bool, str]:
        """Check if requesting engine is healthy."""
        state = engine_registry.get_engine_state(request.engine)
        if state is None:
            return False, f"Engine {request.engine} not registered"

        if not state.is_healthy():
            return False, f"Engine {request.engine} last heartbeat {state.time_since_heartbeat().total_seconds():.0f}s ago"

        return True, "OK"

    def _record_velocity(self, request: "TradeRequest") -> None:
        """Record trade for velocity tracking."""
        with self._lock:
            self._velocity_log.append(TradeVelocity(
                ticker=request.ticker,
                engine=request.engine,
                timestamp=datetime.now(timezone.utc),
            ))

    def _update_avg_latency(self, latency_ms: float) -> None:
        """Update average latency stat."""
        total = self._stats["checks_total"]
        if total > 1:
            self._stats["avg_latency_ms"] = (
                (self._stats["avg_latency_ms"] * (total - 1) + latency_ms) / total
            )
        else:
            self._stats["avg_latency_ms"] = latency_ms

    def _log_decision(self, request: "TradeRequest", decision: HawkEyeDecision) -> None:
        """Log decision for audit trail."""
        with self._lock:
            self._audit_log.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "audit_id": decision.audit_log_id,
                "request_id": request.request_id,
                "engine": request.engine,
                "ticker": request.ticker,
                "direction": request.direction,
                "allowed": decision.allowed,
                "reason": decision.reason,
                "latency_ms": decision.check_latency_ms,
            })

            # Keep last 1000 entries
            if len(self._audit_log) > 1000:
                self._audit_log = self._audit_log[-1000:]

        if not decision.allowed:
            logger.warning(f"HAWK_EYE: BLOCKED {request.engine}/{request.ticker} - {decision.reason}")
        else:
            logger.debug(f"HAWK_EYE: ALLOWED {request.engine}/{request.ticker} ({decision.check_latency_ms:.1f}ms)")

    def get_exposure_summary(self, current_positions: List[Dict]) -> Dict:
        """Get current exposure by ticker, sector, and correlation group."""
        sectors = {}
        tickers = {}

        for pos in current_positions:
            ticker = pos.get("ticker", pos.get("symbol", ""))
            sector = self.TICKER_SECTORS.get(ticker, "other")

            tickers[ticker] = tickers.get(ticker, 0) + 1
            sectors[sector] = sectors.get(sector, 0) + 1

        total = len(current_positions) or 1

        return {
            "total_positions": len(current_positions),
            "by_ticker": tickers,
            "by_sector": {k: {"count": v, "pct": v / total * 100} for k, v in sectors.items()},
        }

    def get_stats(self) -> Dict:
        """Get Hawk Eye statistics."""
        with self._lock:
            return {
                **self._stats,
                "velocity_log_size": len(self._velocity_log),
                "audit_log_size": len(self._audit_log),
            }

    def get_recent_decisions(self, limit: int = 50) -> List[Dict]:
        """Get recent audit log entries."""
        with self._lock:
            return self._audit_log[-limit:]


# Singleton instance
_hawk_eye: Optional[HawkEyeMonitor] = None


def get_hawk_eye_monitor() -> HawkEyeMonitor:
    """Get singleton HawkEyeMonitor instance."""
    global _hawk_eye
    if _hawk_eye is None:
        _hawk_eye = HawkEyeMonitor()
    return _hawk_eye
