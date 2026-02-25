"""
Layer H: HYDRA Intelligence Consumer

Purpose: Single source for all market structure intelligence
Weight: 20%
Cost: $0 (data comes from HYDRA bridge)
Latency: <10ms (cached)

Consumes expanded HYDRA intelligence:
- GEX regime (dealer gamma positioning)
- Institutional flow bias
- Dark pool support/resistance levels
- Temporal sequence match confidence

Replaces: Old APEX order_flow component (20% weight)
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from wsb_snake.utils.logger import get_logger
from wsb_snake.collectors.hydra_bridge import get_hydra_intel, HydraIntelligence

logger = get_logger(__name__)


@dataclass
class HydraScore:
    """Scoring result from HYDRA intelligence."""
    adjustment: float  # -0.20 to +0.20 conviction adjustment
    components: Dict[str, float] = field(default_factory=dict)
    connected: bool = False
    gex_regime: str = "UNKNOWN"
    flow_bias: str = "NEUTRAL"
    dp_support: float = 0
    dp_resistance: float = 0
    sequence_win_rate: float = 0
    latency_ms: float = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'adjustment': self.adjustment,
            'components': self.components,
            'connected': self.connected,
            'gex_regime': self.gex_regime,
            'flow_bias': self.flow_bias,
            'dp_support': self.dp_support,
            'dp_resistance': self.dp_resistance,
            'sequence_win_rate': self.sequence_win_rate,
            'latency_ms': self.latency_ms,
            'reason': self.reason
        }


class HydraLayer:
    """
    Consumes expanded HYDRA intelligence.

    Replaces: old order_flow component (20% weight)
    Adds: GEX regime, dark pool levels, sequence match
    """

    # GEX regime adjustments
    GEX_ADJUSTMENTS = {
        'POSITIVE': -0.03,  # Pinned market, harder to move
        'NEGATIVE': +0.03,  # Trending possible
        'UNKNOWN': 0
    }

    # Flow bias adjustments (directional alignment)
    FLOW_ADJUSTMENTS = {
        ('LONG', 'AGGRESSIVELY_BULLISH'): +0.08,
        ('LONG', 'BULLISH'): +0.04,
        ('LONG', 'NEUTRAL'): 0,
        ('LONG', 'BEARISH'): -0.04,
        ('LONG', 'AGGRESSIVELY_BEARISH'): -0.10,
        ('CALL', 'AGGRESSIVELY_BULLISH'): +0.08,
        ('CALL', 'BULLISH'): +0.04,
        ('CALL', 'NEUTRAL'): 0,
        ('CALL', 'BEARISH'): -0.04,
        ('CALL', 'AGGRESSIVELY_BEARISH'): -0.10,
        ('SHORT', 'AGGRESSIVELY_BEARISH'): +0.08,
        ('SHORT', 'BEARISH'): +0.04,
        ('SHORT', 'NEUTRAL'): 0,
        ('SHORT', 'BULLISH'): -0.04,
        ('SHORT', 'AGGRESSIVELY_BULLISH'): -0.10,
        ('PUT', 'AGGRESSIVELY_BEARISH'): +0.08,
        ('PUT', 'BEARISH'): +0.04,
        ('PUT', 'NEUTRAL'): 0,
        ('PUT', 'BULLISH'): -0.04,
        ('PUT', 'AGGRESSIVELY_BULLISH'): -0.10,
    }

    def __init__(self):
        """Initialize HYDRA layer."""
        self._call_count = 0

    def score(self, signal: Dict[str, Any]) -> HydraScore:
        """
        Score signal based on HYDRA intelligence.

        Args:
            signal: Dict with 'direction', 'ticker', 'price', etc.

        Returns:
            HydraScore with conviction adjustments
        """
        start = time.time()
        self._call_count += 1

        hydra = get_hydra_intel()

        if not hydra.connected:
            return HydraScore(
                adjustment=0,
                reason="HYDRA disconnected",
                latency_ms=(time.time() - start) * 1000
            )

        components = {}
        direction = signal.get('direction', 'NEUTRAL').upper()
        price = signal.get('price', 0)

        # Component 1: GEX Regime
        gex_adj = self._score_gex(direction, price, hydra)
        components['gex'] = gex_adj

        # Component 2: Flow Alignment
        flow_adj = self._score_flow(direction, hydra)
        components['flow'] = flow_adj

        # Component 3: Dark Pool Proximity
        dp_adj = self._score_dark_pool(direction, price, hydra)
        components['dark_pool'] = dp_adj

        # Component 4: Sequence Match
        seq_adj = self._score_sequence(hydra)
        components['sequence'] = seq_adj

        # Total adjustment
        total_adj = sum(components.values())
        total_adj = max(-0.20, min(0.20, total_adj))  # Clamp to ±20%

        latency = (time.time() - start) * 1000

        # Build reason string
        reasons = []
        if abs(gex_adj) > 0.01:
            gex_regime = getattr(hydra, 'gex_regime', 'UNKNOWN')
            reasons.append(f"GEX:{gex_regime}")
        if abs(flow_adj) > 0.01:
            flow_bias = getattr(hydra, 'flow_bias', 'NEUTRAL')
            reasons.append(f"Flow:{flow_bias}")
        if abs(dp_adj) > 0.01:
            reasons.append("DP_level_nearby")
        if abs(seq_adj) > 0.01:
            win_rate = getattr(hydra, 'seq_historical_win_rate', 0)
            reasons.append(f"Seq:{win_rate:.0%}WR")

        logger.info(
            f"HYDRA_LH: adj={total_adj:+.2f} "
            f"gex={gex_adj:+.2f} flow={flow_adj:+.2f} dp={dp_adj:+.2f} seq={seq_adj:+.2f} "
            f"in {latency:.0f}ms"
        )

        return HydraScore(
            adjustment=total_adj,
            components=components,
            connected=True,
            gex_regime=getattr(hydra, 'gex_regime', 'UNKNOWN'),
            flow_bias=getattr(hydra, 'flow_bias', 'NEUTRAL'),
            dp_support=getattr(hydra, 'dp_nearest_support', 0),
            dp_resistance=getattr(hydra, 'dp_nearest_resistance', 0),
            sequence_win_rate=getattr(hydra, 'seq_historical_win_rate', 0),
            latency_ms=latency,
            reason=" | ".join(reasons) if reasons else "neutral"
        )

    def _score_gex(
        self,
        direction: str,
        price: float,
        hydra: HydraIntelligence
    ) -> float:
        """Score based on GEX regime."""
        gex_regime = getattr(hydra, 'gex_regime', 'UNKNOWN')
        base_adj = self.GEX_ADJUSTMENTS.get(gex_regime, 0)

        # Extra penalty if near flip point
        flip_distance = getattr(hydra, 'gex_flip_distance_pct', 999)
        if flip_distance < 0.5:
            base_adj -= 0.05  # Penalty for regime change risk

        # Penalty if trying momentum in POSITIVE GEX (pinned market)
        if gex_regime == 'POSITIVE':
            # Check proximity to GEX levels
            resistance = getattr(hydra, 'gex_key_resistance', [])
            support = getattr(hydra, 'gex_key_support', [])

            if price and resistance:
                nearest_res = min(resistance, key=lambda x: abs(x - price), default=0)
                if nearest_res and direction in ['LONG', 'CALL']:
                    dist_pct = (nearest_res - price) / price if price else 1
                    if dist_pct < 0.005:  # Within 0.5%
                        base_adj -= 0.05  # Blocked by GEX resistance

            if price and support:
                nearest_sup = min(support, key=lambda x: abs(x - price), default=0)
                if nearest_sup and direction in ['SHORT', 'PUT']:
                    dist_pct = (price - nearest_sup) / price if price else 1
                    if dist_pct < 0.005:  # Within 0.5%
                        base_adj -= 0.05  # Blocked by GEX support

        return base_adj

    def _score_flow(self, direction: str, hydra: HydraIntelligence) -> float:
        """Score based on institutional flow alignment."""
        flow_bias = getattr(hydra, 'flow_bias', 'NEUTRAL')
        key = (direction, flow_bias)
        return self.FLOW_ADJUSTMENTS.get(key, 0)

    def _score_dark_pool(
        self,
        direction: str,
        price: float,
        hydra: HydraIntelligence
    ) -> float:
        """Score based on dark pool level proximity."""
        if not price:
            return 0

        dp_support = getattr(hydra, 'dp_nearest_support', 0)
        dp_resistance = getattr(hydra, 'dp_nearest_resistance', 0)
        support_strength = getattr(hydra, 'dp_support_strength', 'UNKNOWN')
        resistance_strength = getattr(hydra, 'dp_resistance_strength', 'UNKNOWN')

        # Near strong dark pool support + going long = good
        if dp_support and support_strength in ['HIGH', 'VERY_HIGH']:
            dist_to_support = (price - dp_support) / price
            if dist_to_support < 0.005:  # Within 0.5%
                if direction in ['LONG', 'CALL']:
                    return +0.05  # Bounce play

        # Near strong dark pool resistance + going short = good
        if dp_resistance and resistance_strength in ['HIGH', 'VERY_HIGH']:
            dist_to_resistance = (dp_resistance - price) / price
            if dist_to_resistance < 0.005:  # Within 0.5%
                if direction in ['SHORT', 'PUT']:
                    return +0.05  # Fade play

        return 0

    def _score_sequence(self, hydra: HydraIntelligence) -> float:
        """Score based on historical sequence matching."""
        similar_patterns = getattr(hydra, 'seq_similar_patterns', 0)
        win_rate = getattr(hydra, 'seq_historical_win_rate', 0)

        if similar_patterns < 2:
            return 0  # Insufficient data

        # Strong historical match
        if win_rate > 0.70:
            return +0.06
        elif win_rate > 0.55:
            return +0.03
        elif win_rate < 0.40:
            return -0.08  # Historically losing pattern
        elif win_rate < 0.30:
            return -0.12  # Strong loser pattern

        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics."""
        return {
            'call_count': self._call_count
        }


# Singleton
_hydra_layer = None

def get_hydra_layer() -> HydraLayer:
    """Get singleton HydraLayer instance."""
    global _hydra_layer
    if _hydra_layer is None:
        _hydra_layer = HydraLayer()
    return _hydra_layer
