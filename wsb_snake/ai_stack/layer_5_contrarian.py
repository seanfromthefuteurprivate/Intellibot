"""
Layer 5: DeepSeek CONTRARIAN (Trap Detector)

Purpose: Detect crowded trades, traps, and false breakouts
Weight: 10%
Cost: ~$0.0001 per call (negligible)
Latency: ~200ms

Reassigned from confirmation to contrarian detection:
- OLD: Confirm GPT verdict (consensus seeking)
- NEW: Find traps and crowded trades (contrarian)

Types of traps detected:
- Bull trap: Price breaks resistance then reverses hard
- Bear trap: Price breaks support then reverses hard
- Crowded long: Everyone is bullish = bag holders forming
- Crowded short: Everyone is bearish = short squeeze setup
- False breakout: Low volume breakout = easy to fade
"""

import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class TrapType(Enum):
    """Types of traps detected."""
    NONE = "NONE"
    BULL_TRAP = "BULL_TRAP"
    BEAR_TRAP = "BEAR_TRAP"
    CROWDED_LONG = "CROWDED_LONG"
    CROWDED_SHORT = "CROWDED_SHORT"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    EXHAUSTION = "EXHAUSTION"


@dataclass
class ContrarianResult:
    """Result from contrarian analysis."""
    trap_detected: TrapType
    adjustment: float  # -0.25 to 0
    trap_confidence: float  # 0-1
    reason: str
    latency_ms: float = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trap_detected': self.trap_detected.value,
            'adjustment': self.adjustment,
            'trap_confidence': self.trap_confidence,
            'reason': self.reason,
            'latency_ms': self.latency_ms,
            'error': self.error
        }


class ContrarianLayer:
    """
    DeepSeek contrarian detector.

    Identifies traps and crowded trades that retail
    typically falls into.
    """

    CONTRARIAN_PROMPT = """You are a CONTRARIAN DETECTOR for 0DTE options trading.

This trade passed preliminary filters:
- Ticker: {ticker}
- Direction: {direction}
- Price: ${price:.2f}
- Vision Pattern: {pattern}
- HYDRA Flow Bias: {flow_bias}
- Recent Failed Trades: {recent_failures}

Your job: Is this a TRAP?

Check for these trap types:
1. BULL_TRAP: Price just broke resistance, but volume is low or flow is bearish
2. BEAR_TRAP: Price just broke support, but volume is low or flow is bullish
3. CROWDED_LONG: Everyone is bullish (flow strongly bullish) = bag holders forming
4. CROWDED_SHORT: Everyone is bearish = short squeeze risk
5. FALSE_BREAKOUT: Breakout pattern detected but volume/flow doesn't confirm
6. EXHAUSTION: Move has gone too far too fast, likely to reverse

RESPOND WITH EXACTLY ONE OF:
- "TRAP: <TYPE>: <reason>" — if this is likely a trap
- "CLEAR: <why>" — if contrarian signals are absent

Your goal: Protect from herd behavior losses. Be suspicious of "obvious" plays."""

    # Trap adjustments
    TRAP_ADJUSTMENTS = {
        TrapType.NONE: 0,
        TrapType.BULL_TRAP: -0.20,
        TrapType.BEAR_TRAP: -0.20,
        TrapType.CROWDED_LONG: -0.15,
        TrapType.CROWDED_SHORT: -0.15,
        TrapType.FALSE_BREAKOUT: -0.25,
        TrapType.EXHAUSTION: -0.18,
    }

    def __init__(self):
        """Initialize contrarian layer."""
        self._client = None
        self._call_count = 0
        self._trap_count = 0

        # Try to initialize DeepSeek client
        try:
            import openai
            api_key = os.environ.get('DEEPSEEK_API_KEY')
            if api_key:
                self._client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1"
                )
                logger.info("CONTRARIAN_L5: DeepSeek client initialized")
            else:
                logger.warning("CONTRARIAN_L5: No DEEPSEEK_API_KEY found")
        except Exception as e:
            logger.warning(f"CONTRARIAN_L5: DeepSeek unavailable: {e}")

    def detect(
        self,
        signal: Dict[str, Any],
        vision_signal: Dict = None,
        hydra_data: Dict = None,
        recent_failures: List[str] = None
    ) -> ContrarianResult:
        """
        Detect if signal is a trap.

        Args:
            signal: Base signal dict
            vision_signal: Output from Layer 1
            hydra_data: Output from Layer H
            recent_failures: List of recent failed trade descriptions

        Returns:
            ContrarianResult with trap detection
        """
        start = time.time()
        self._call_count += 1

        ticker = signal.get('ticker', 'UNKNOWN')
        direction = signal.get('direction', 'NEUTRAL')
        price = signal.get('price', 0)

        # First, do rule-based trap detection (free, fast)
        rule_result = self._rule_based_detection(signal, vision_signal, hydra_data)
        if rule_result.trap_detected != TrapType.NONE:
            rule_result.latency_ms = (time.time() - start) * 1000
            self._trap_count += 1
            logger.info(f"CONTRARIAN_L5: TRAP (rule) {ticker} {rule_result.trap_detected.value}")
            return rule_result

        # If no rule-based trap and DeepSeek available, do AI analysis
        if self._client:
            ai_result = self._ai_detection(signal, vision_signal, hydra_data, recent_failures)
            ai_result.latency_ms = (time.time() - start) * 1000
            if ai_result.trap_detected != TrapType.NONE:
                self._trap_count += 1
                logger.info(f"CONTRARIAN_L5: TRAP (AI) {ticker} {ai_result.trap_detected.value}")
            else:
                logger.debug(f"CONTRARIAN_L5: CLEAR {ticker}")
            return ai_result

        # No traps detected
        return ContrarianResult(
            trap_detected=TrapType.NONE,
            adjustment=0,
            trap_confidence=0,
            reason="No contrarian signals",
            latency_ms=(time.time() - start) * 1000
        )

    def _rule_based_detection(
        self,
        signal: Dict,
        vision_signal: Dict,
        hydra_data: Dict
    ) -> ContrarianResult:
        """Fast rule-based trap detection."""
        direction = signal.get('direction', 'NEUTRAL').upper()

        # Get HYDRA flow
        flow_bias = hydra_data.get('flow_bias', 'NEUTRAL') if hydra_data else 'NEUTRAL'

        # Get vision pattern
        pattern = 'none'
        if vision_signal:
            pattern = vision_signal.get('pattern_detected', 'none')

        # Rule 1: Crowded long detection
        # Going LONG when flow is AGGRESSIVELY_BULLISH = everyone agrees = trap
        if direction in ['LONG', 'CALL'] and flow_bias == 'AGGRESSIVELY_BULLISH':
            return ContrarianResult(
                trap_detected=TrapType.CROWDED_LONG,
                adjustment=self.TRAP_ADJUSTMENTS[TrapType.CROWDED_LONG],
                trap_confidence=0.7,
                reason="Flow is AGGRESSIVELY_BULLISH - everyone agrees, bag holders forming"
            )

        # Rule 2: Crowded short detection
        if direction in ['SHORT', 'PUT'] and flow_bias == 'AGGRESSIVELY_BEARISH':
            return ContrarianResult(
                trap_detected=TrapType.CROWDED_SHORT,
                adjustment=self.TRAP_ADJUSTMENTS[TrapType.CROWDED_SHORT],
                trap_confidence=0.7,
                reason="Flow is AGGRESSIVELY_BEARISH - everyone agrees, squeeze risk"
            )

        # Rule 3: Breakout with opposing flow
        if 'breakout' in pattern.lower() and direction in ['LONG', 'CALL']:
            if flow_bias in ['BEARISH', 'AGGRESSIVELY_BEARISH']:
                return ContrarianResult(
                    trap_detected=TrapType.FALSE_BREAKOUT,
                    adjustment=self.TRAP_ADJUSTMENTS[TrapType.FALSE_BREAKOUT],
                    trap_confidence=0.8,
                    reason="Breakout pattern but flow is bearish - false breakout"
                )

        if 'breakdown' in pattern.lower() and direction in ['SHORT', 'PUT']:
            if flow_bias in ['BULLISH', 'AGGRESSIVELY_BULLISH']:
                return ContrarianResult(
                    trap_detected=TrapType.FALSE_BREAKOUT,
                    adjustment=self.TRAP_ADJUSTMENTS[TrapType.FALSE_BREAKOUT],
                    trap_confidence=0.8,
                    reason="Breakdown pattern but flow is bullish - false breakdown"
                )

        return ContrarianResult(
            trap_detected=TrapType.NONE,
            adjustment=0,
            trap_confidence=0,
            reason="No rule-based traps detected"
        )

    def _ai_detection(
        self,
        signal: Dict,
        vision_signal: Dict,
        hydra_data: Dict,
        recent_failures: List[str]
    ) -> ContrarianResult:
        """AI-based trap detection using DeepSeek."""
        ticker = signal.get('ticker', 'UNKNOWN')
        direction = signal.get('direction', 'NEUTRAL')
        price = signal.get('price', 0)

        pattern = 'none'
        if vision_signal:
            pattern = vision_signal.get('pattern_detected', 'none')

        flow_bias = hydra_data.get('flow_bias', 'NEUTRAL') if hydra_data else 'NEUTRAL'

        failures_str = "\n".join(recent_failures[:5]) if recent_failures else "None recent"

        prompt = self.CONTRARIAN_PROMPT.format(
            ticker=ticker,
            direction=direction,
            price=price,
            pattern=pattern,
            flow_bias=flow_bias,
            recent_failures=failures_str
        )

        try:
            response = self._client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a contrarian trade detector. Be suspicious of obvious plays."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0
            )

            content = response.choices[0].message.content.strip().upper()
            return self._parse_response(content)

        except Exception as e:
            logger.error(f"CONTRARIAN_L5: DeepSeek error - {e}")
            return ContrarianResult(
                trap_detected=TrapType.NONE,
                adjustment=0,
                trap_confidence=0,
                reason="AI detection failed",
                error=str(e)
            )

    def _parse_response(self, text: str) -> ContrarianResult:
        """Parse DeepSeek response."""
        if text.startswith("TRAP:"):
            parts = text[5:].strip().split(":", 1)
            trap_name = parts[0].strip()
            reason = parts[1].strip() if len(parts) > 1 else "Unknown reason"

            # Map to TrapType
            trap_type = TrapType.NONE
            for tt in TrapType:
                if tt.value == trap_name or trap_name in tt.value:
                    trap_type = tt
                    break

            if trap_type == TrapType.NONE:
                # Try to infer
                if "BULL" in trap_name:
                    trap_type = TrapType.BULL_TRAP
                elif "BEAR" in trap_name:
                    trap_type = TrapType.BEAR_TRAP
                elif "CROWD" in trap_name:
                    if "LONG" in trap_name:
                        trap_type = TrapType.CROWDED_LONG
                    else:
                        trap_type = TrapType.CROWDED_SHORT
                elif "BREAKOUT" in trap_name or "FALSE" in trap_name:
                    trap_type = TrapType.FALSE_BREAKOUT
                elif "EXHAUST" in trap_name:
                    trap_type = TrapType.EXHAUSTION

            return ContrarianResult(
                trap_detected=trap_type,
                adjustment=self.TRAP_ADJUSTMENTS.get(trap_type, -0.15),
                trap_confidence=0.75,
                reason=reason
            )

        elif text.startswith("CLEAR:"):
            reason = text[6:].strip()
            return ContrarianResult(
                trap_detected=TrapType.NONE,
                adjustment=0,
                trap_confidence=0,
                reason=reason
            )

        else:
            return ContrarianResult(
                trap_detected=TrapType.NONE,
                adjustment=0,
                trap_confidence=0,
                reason=f"Unparseable: {text[:50]}"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics."""
        return {
            'call_count': self._call_count,
            'trap_count': self._trap_count,
            'trap_rate': self._trap_count / max(self._call_count, 1)
        }


# Singleton
_contrarian_layer = None

def get_contrarian_layer() -> ContrarianLayer:
    """Get singleton ContrarianLayer instance."""
    global _contrarian_layer
    if _contrarian_layer is None:
        _contrarian_layer = ContrarianLayer()
    return _contrarian_layer
