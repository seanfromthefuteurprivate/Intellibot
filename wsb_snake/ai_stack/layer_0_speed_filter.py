"""
Layer 0: Speed Filter (Nova Micro)

Purpose: Kill 60% of obvious losers in <80ms before expensive AI layers.
Cost: $0.00004 per call
Latency: ~80ms

Kill conditions based on HYDRA intelligence:
1. GEX regime mismatch (momentum play in pinned market)
2. Flow bias aggressively opposite to signal
3. Near GEX flip point (regime change imminent)
4. HYDRA recommendation is HOLD_OFF
5. VIX chaos mode (spike >20%)
6. Signal blocked by nearby GEX levels
"""

import json
from typing import Tuple, Dict, Any
from dataclasses import dataclass

from wsb_snake.utils.logger import get_logger
from wsb_snake.collectors.hydra_bridge import get_hydra_intel

logger = get_logger(__name__)


@dataclass
class FilterResult:
    """Result from speed filter."""
    passed: bool
    reason: str
    latency_ms: float = 0
    hydra_connected: bool = False


class SpeedFilter:
    """
    Nova Micro gate — reject obvious losers in <80ms.
    Saves ~$0.011 per rejected signal by avoiding full stack.
    """

    # Instant kill conditions (no AI needed)
    INSTANT_KILL_CONDITIONS = [
        "HOLD_OFF recommendation",
        "VIX chaos (>35)",
        "GEX flip imminent (<0.3%)",
        "Aggressive flow opposition",
        "Event blocking"
    ]

    def __init__(self):
        """Initialize speed filter."""
        self._filter_count = 0
        self._kill_count = 0
        self._kill_reasons: Dict[str, int] = {}

        # Try to initialize Bedrock client (optional - can work without it)
        self._bedrock = None
        try:
            from .bedrock_client import get_bedrock_client
            self._bedrock = get_bedrock_client()
        except Exception as e:
            logger.warning(f"SPEED_FILTER: Bedrock unavailable, using rule-based only: {e}")

    def filter(self, signal: Dict[str, Any]) -> FilterResult:
        """
        Filter signal - returns (should_continue, reason).
        If False, signal is killed before expensive layers.

        Args:
            signal: Dict with 'direction', 'ticker', 'price', etc.

        Returns:
            FilterResult with passed status and reason
        """
        import time
        start = time.time()

        self._filter_count += 1

        # Get HYDRA intelligence
        hydra = get_hydra_intel()

        if not hydra.connected:
            # No HYDRA = pass (can't filter without intelligence)
            return FilterResult(
                passed=True,
                reason="HYDRA disconnected - passing through",
                latency_ms=(time.time() - start) * 1000,
                hydra_connected=False
            )

        direction = signal.get('direction', '').upper()
        ticker = signal.get('ticker', 'UNKNOWN')

        # RULE 1: HYDRA recommendation is HOLD_OFF
        if hydra.recommendation == "HOLD_OFF":
            return self._kill(start, "HOLD_OFF", f"HYDRA says HOLD_OFF", hydra)

        # RULE 2: Event blocking
        if hydra.events_next_30min:
            event_names = [e.get('name', 'event') for e in hydra.events_next_30min[:2]]
            return self._kill(start, "EVENT_BLOCK", f"Events: {', '.join(event_names)}", hydra)

        # RULE 3: VIX chaos mode (>35 is extreme fear)
        if hydra.vix_level > 35:
            return self._kill(start, "VIX_CHAOS", f"VIX at {hydra.vix_level:.1f}", hydra)

        # RULE 4: GEX flip imminent (within 0.3%)
        gex_flip_distance = getattr(hydra, 'gex_flip_distance_pct', 999)
        if gex_flip_distance < 0.3:
            return self._kill(start, "GEX_FLIP", f"Flip point {gex_flip_distance:.2f}% away", hydra)

        # RULE 5: Aggressive flow opposition
        flow_bias = getattr(hydra, 'flow_bias', 'NEUTRAL')
        if direction in ['LONG', 'CALL'] and flow_bias == 'AGGRESSIVELY_BEARISH':
            return self._kill(start, "FLOW_OPPOSE", f"LONG vs AGGRESSIVELY_BEARISH flow", hydra)
        if direction in ['SHORT', 'PUT'] and flow_bias == 'AGGRESSIVELY_BULLISH':
            return self._kill(start, "FLOW_OPPOSE", f"SHORT vs AGGRESSIVELY_BULLISH flow", hydra)

        # RULE 6: GEX regime mismatch (momentum in pinned market)
        gex_regime = getattr(hydra, 'gex_regime', 'UNKNOWN')
        if gex_regime == 'POSITIVE':
            # Pinned market - check if trying momentum near resistance/support
            gex_resistance = getattr(hydra, 'gex_key_resistance', [])
            gex_support = getattr(hydra, 'gex_key_support', [])
            price = signal.get('price', 0)

            if price and gex_resistance:
                nearest_resistance = min(gex_resistance, key=lambda x: abs(x - price))
                if direction in ['LONG', 'CALL'] and (nearest_resistance - price) / price < 0.005:
                    return self._kill(start, "GEX_BLOCKED", f"CALL near GEX resistance ${nearest_resistance:.2f}", hydra)

            if price and gex_support:
                nearest_support = min(gex_support, key=lambda x: abs(x - price))
                if direction in ['SHORT', 'PUT'] and (price - nearest_support) / price < 0.005:
                    return self._kill(start, "GEX_BLOCKED", f"PUT near GEX support ${nearest_support:.2f}", hydra)

        # RULE 7: CRASH regime without PUT
        if hydra.regime == 'CRASH' and direction in ['LONG', 'CALL']:
            return self._kill(start, "CRASH_LONG", "CALL in CRASH regime", hydra)

        # All rules passed
        latency = (time.time() - start) * 1000
        logger.debug(f"SPEED_FILTER: {ticker} {direction} PASSED in {latency:.0f}ms")

        return FilterResult(
            passed=True,
            reason="All rules passed",
            latency_ms=latency,
            hydra_connected=True
        )

    def _kill(self, start: float, code: str, reason: str, hydra) -> FilterResult:
        """Record a kill and return result."""
        import time
        latency = (time.time() - start) * 1000

        self._kill_count += 1
        self._kill_reasons[code] = self._kill_reasons.get(code, 0) + 1

        logger.info(f"SPEED_FILTER: KILLED [{code}] - {reason}")

        return FilterResult(
            passed=False,
            reason=f"[{code}] {reason}",
            latency_ms=latency,
            hydra_connected=True
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        kill_rate = self._kill_count / max(self._filter_count, 1)
        return {
            'filter_count': self._filter_count,
            'kill_count': self._kill_count,
            'kill_rate': kill_rate,
            'kill_reasons': self._kill_reasons,
            'estimated_savings_usd': self._kill_count * 0.011  # $0.011 saved per kill
        }


# Singleton
_speed_filter = None

def get_speed_filter() -> SpeedFilter:
    """Get singleton SpeedFilter instance."""
    global _speed_filter
    if _speed_filter is None:
        _speed_filter = SpeedFilter()
    return _speed_filter
