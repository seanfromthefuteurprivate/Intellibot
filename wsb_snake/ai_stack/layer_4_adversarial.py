"""
Layer 4: Claude Sonnet 4.5 ADVERSARIAL (Thesis Destroyer)

Purpose: Try to KILL the trade. If Claude can't destroy the thesis, it survives.
Weight: 15%
Cost: ~$0.005 per call
Latency: ~300ms

The Adversarial Challenge:
- Claude receives the bullish/bearish thesis
- Claude's job is to find EVERY reason this trade will FAIL
- If Claude can't find a fatal flaw → signal survives
- If Claude finds a kill shot → instant reject

This prevents false confidence from consensus (all AIs agreeing).
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class AdversarialResult(Enum):
    """Result of adversarial challenge."""
    SURVIVE = "SURVIVE"  # Trade passed - no fatal flaw found
    KILL = "KILL"  # Trade killed - fatal flaw found
    UNCERTAIN = "UNCERTAIN"  # Claude hedged - penalty applied


@dataclass
class AdversarialVerdict:
    """Verdict from adversarial challenge."""
    result: AdversarialResult
    adjustment: float  # -0.15 to +0.05
    kill_reason: Optional[str] = None
    survive_reason: Optional[str] = None
    uncertainty_reason: Optional[str] = None
    latency_ms: float = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'result': self.result.value,
            'adjustment': self.adjustment,
            'kill_reason': self.kill_reason,
            'survive_reason': self.survive_reason,
            'uncertainty_reason': self.uncertainty_reason,
            'latency_ms': self.latency_ms,
            'error': self.error
        }


class AdversarialLayer:
    """
    Claude Sonnet thesis destroyer.

    Given a trade thesis, Claude tries to find every reason
    it will fail. If it can't find a fatal flaw, the trade survives.
    """

    ADVERSARIAL_PROMPT = """You are a THESIS DESTROYER. Your job is to KILL this trade.

CURRENT THESIS:
{thesis}

SUPPORTING DATA:
- Ticker: {ticker}
- Direction: {direction}
- Vision Signal: {vision}
- HYDRA GEX Regime: {gex_regime}
- HYDRA Flow Bias: {flow_bias}
- HYDRA Recommendation: {hydra_rec}
- Semantic Match: {semantic}
- Current Price: ${price:.2f}
- VIX Level: {vix}

YOUR MISSION: Find EVERY reason this trade will FAIL.

Consider:
1. Is this a crowded trade? (everyone sees this = priced in)
2. Is the move already exhausted?
3. What catalyst is MISSING?
4. What's the bear case they're ignoring?
5. Is the GEX regime opposing this direction?
6. Is institutional flow opposing this?
7. Is this a trap setup (bull trap, bear trap)?
8. Time of day concerns (power hour rush, end of day dumps)?
9. Historical similar setups that failed?

RESPOND WITH EXACTLY ONE OF:
- "KILL: <specific fatal flaw>" — if you found a kill shot
- "SURVIVE: <why it's valid>" — if you genuinely cannot find a fatal flaw
- "UNCERTAIN: <what's unclear>" — if you need more information

Be RUTHLESS. False positives (killing good trades) are better than losing trades.
Your kill rate should be 20-30% of trades that reach you."""

    def __init__(self):
        """Initialize adversarial layer."""
        self._bedrock = None
        self._call_count = 0
        self._kill_count = 0
        self._survive_count = 0
        self._uncertain_count = 0

        # Initialize Bedrock client
        try:
            from .bedrock_client import get_bedrock_client
            self._bedrock = get_bedrock_client()
            logger.info("ADVERSARIAL_L4: Bedrock client initialized")
        except Exception as e:
            logger.warning(f"ADVERSARIAL_L4: Bedrock unavailable: {e}")

    def challenge(
        self,
        signal: Dict[str, Any],
        thesis: str = None,
        vision_signal: Dict = None,
        hydra_data: Dict = None,
        semantic_data: Dict = None
    ) -> AdversarialVerdict:
        """
        Challenge a trade thesis.

        Args:
            signal: Base signal dict (ticker, direction, price, etc.)
            thesis: Human-readable thesis statement
            vision_signal: Output from Layer 1
            hydra_data: Output from Layer H
            semantic_data: Output from Layer 2

        Returns:
            AdversarialVerdict with result and adjustment
        """
        start = time.time()
        self._call_count += 1

        if not self._bedrock:
            return AdversarialVerdict(
                result=AdversarialResult.UNCERTAIN,
                adjustment=0,
                error="Bedrock unavailable",
                latency_ms=(time.time() - start) * 1000
            )

        ticker = signal.get('ticker', 'UNKNOWN')
        direction = signal.get('direction', 'NEUTRAL')
        price = signal.get('price', 0)

        # Build thesis if not provided
        if not thesis:
            thesis = self._build_thesis(signal, vision_signal)

        # Get HYDRA data
        gex_regime = hydra_data.get('gex_regime', 'UNKNOWN') if hydra_data else 'UNKNOWN'
        flow_bias = hydra_data.get('flow_bias', 'NEUTRAL') if hydra_data else 'NEUTRAL'
        hydra_rec = 'UNKNOWN'

        try:
            from wsb_snake.collectors.hydra_bridge import get_hydra_intel
            hydra = get_hydra_intel()
            hydra_rec = hydra.recommendation
            vix = hydra.vix_level
        except:
            vix = 0

        # Format prompt
        prompt = self.ADVERSARIAL_PROMPT.format(
            thesis=thesis,
            ticker=ticker,
            direction=direction,
            vision=str(vision_signal) if vision_signal else "N/A",
            gex_regime=gex_regime,
            flow_bias=flow_bias,
            hydra_rec=hydra_rec,
            semantic=str(semantic_data) if semantic_data else "N/A",
            price=price,
            vix=vix
        )

        try:
            response = self._bedrock.invoke_claude(
                prompt=prompt,
                model='claude-sonnet',
                max_tokens=200,
                temperature=0,
                system="You are a ruthless trade thesis destroyer. Your job is to find fatal flaws in trading ideas. Be concise and decisive."
            )

            latency = (time.time() - start) * 1000

            if not response.success:
                return AdversarialVerdict(
                    result=AdversarialResult.UNCERTAIN,
                    adjustment=-0.05,
                    error=response.error,
                    latency_ms=latency
                )

            # Parse response
            verdict = self._parse_response(response.text)
            verdict.latency_ms = latency

            # Track stats
            if verdict.result == AdversarialResult.KILL:
                self._kill_count += 1
                logger.info(f"ADVERSARIAL_L4: KILL {ticker} {direction} - {verdict.kill_reason}")
            elif verdict.result == AdversarialResult.SURVIVE:
                self._survive_count += 1
                logger.info(f"ADVERSARIAL_L4: SURVIVE {ticker} {direction} - {verdict.survive_reason}")
            else:
                self._uncertain_count += 1
                logger.info(f"ADVERSARIAL_L4: UNCERTAIN {ticker} {direction}")

            return verdict

        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"ADVERSARIAL_L4: Error - {e}")
            return AdversarialVerdict(
                result=AdversarialResult.UNCERTAIN,
                adjustment=-0.05,
                error=str(e),
                latency_ms=latency
            )

    def _build_thesis(self, signal: Dict, vision_signal: Dict = None) -> str:
        """Build thesis statement from signal data."""
        ticker = signal.get('ticker', 'UNKNOWN')
        direction = signal.get('direction', 'NEUTRAL')
        price = signal.get('price', 0)

        thesis = f"Going {direction} on {ticker} at ${price:.2f}."

        if vision_signal:
            pattern = vision_signal.get('pattern_detected', 'none')
            if pattern != 'none':
                thesis += f" Detected {pattern} pattern."

            vwap = vision_signal.get('vwap_position', 'unknown')
            if vwap != 'unknown':
                thesis += f" Price is {vwap} VWAP."

            momentum = vision_signal.get('momentum', 'flat')
            thesis += f" Momentum is {momentum}."

        return thesis

    def _parse_response(self, text: str) -> AdversarialVerdict:
        """Parse Claude's response into AdversarialVerdict."""
        text = text.strip().upper()

        if text.startswith("KILL:"):
            reason = text[5:].strip()
            return AdversarialVerdict(
                result=AdversarialResult.KILL,
                adjustment=-0.15,  # Strong penalty
                kill_reason=reason
            )

        elif text.startswith("SURVIVE:"):
            reason = text[8:].strip()
            return AdversarialVerdict(
                result=AdversarialResult.SURVIVE,
                adjustment=+0.05,  # Small boost for surviving challenge
                survive_reason=reason
            )

        elif text.startswith("UNCERTAIN:"):
            reason = text[10:].strip()
            return AdversarialVerdict(
                result=AdversarialResult.UNCERTAIN,
                adjustment=-0.08,  # Uncertainty = penalty
                uncertainty_reason=reason
            )

        else:
            # Couldn't parse - treat as uncertain
            return AdversarialVerdict(
                result=AdversarialResult.UNCERTAIN,
                adjustment=-0.05,
                uncertainty_reason=f"Unparseable response: {text[:100]}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics."""
        total = self._call_count
        return {
            'call_count': total,
            'kill_count': self._kill_count,
            'survive_count': self._survive_count,
            'uncertain_count': self._uncertain_count,
            'kill_rate': self._kill_count / max(total, 1),
            'survive_rate': self._survive_count / max(total, 1)
        }


# Singleton
_adversarial_layer = None

def get_adversarial_layer() -> AdversarialLayer:
    """Get singleton AdversarialLayer instance."""
    global _adversarial_layer
    if _adversarial_layer is None:
        _adversarial_layer = AdversarialLayer()
    return _adversarial_layer
