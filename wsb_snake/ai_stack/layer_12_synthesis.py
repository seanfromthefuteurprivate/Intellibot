"""
Layer 12: Claude Sonnet 4.5 SYNTHESIS (Final Arbiter)

Purpose: Final decision maker - STRIKE, ABORT, or WAIT
Weight: 5% (lightweight - most work done by prior layers)
Cost: ~$0.003 per call
Latency: ~200ms

The Synthesis layer:
- Sees ALL layer outputs
- Makes the final STRIKE/ABORT/WAIT decision
- Explains reasoning in 2 sentences for audit
- Sets position size based on conviction
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class FinalAction(Enum):
    """Final action from synthesis."""
    STRIKE = "STRIKE"  # Execute trade
    ABORT = "ABORT"  # Do not trade
    WAIT = "WAIT"  # Signal promising but wait for better entry


@dataclass
class FinalVerdict:
    """Final verdict from synthesis layer."""
    action: FinalAction
    conviction: float  # 0-100 final conviction
    position_size: int  # Number of contracts
    reasoning: str  # 2 sentences max
    entry_price: Optional[float] = None
    stop_loss: float = 0
    take_profit: float = 0
    latency_ms: float = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action.value,
            'conviction': self.conviction,
            'position_size': self.position_size,
            'reasoning': self.reasoning,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'latency_ms': self.latency_ms,
            'error': self.error
        }


class SynthesisLayer:
    """
    Claude Sonnet final arbiter.

    Makes the ultimate STRIKE/ABORT/WAIT decision based on
    all prior layer outputs.
    """

    SYNTHESIS_PROMPT = """You are the FINAL ARBITER for a 0DTE options trade.

LAYER OUTPUTS:
1. Vision (15%): Pattern={vision_pattern}, Bias={vision_bias}, Confidence={vision_conf:.0%}
2. Semantic Match (10%): Adjustment={semantic_adj:+.2f}
3. HYDRA Intelligence (20%): GEX={gex_regime}, Flow={flow_bias}, Adjustment={hydra_adj:+.2f}
4. Adversarial Challenge (15%): Result={adversarial_result}, Adjustment={adversarial_adj:+.2f}
5. Contrarian Check (10%): Trap={trap_type}, Adjustment={contrarian_adj:+.2f}
6. Strategy DNA (25%): Win Rate={dna_win_rate:.0%}, Match={dna_match_type}, Adjustment={dna_adj:+.2f}

SIGNAL DETAILS:
- Ticker: {ticker}
- Direction: {direction}
- Current Price: ${price:.2f}
- Entry Price: ${entry_price:.2f}

AGGREGATED CONVICTION: {final_conviction:.0f}%
MINIMUM THRESHOLD: {threshold}%

DECIDE:
- STRIKE: Execute trade
- ABORT: Do not trade — explain fatal flaw
- WAIT: Signal promising but wait for better entry (explain trigger)

RESPOND IN THIS FORMAT:
ACTION: <STRIKE|ABORT|WAIT>
SIZE: <1-5 contracts>
STOP: <stop loss price>
TARGET: <take profit price>
REASON: <2 sentences max>

Your response determines real money allocation. Be decisive."""

    # Conviction thresholds
    MIN_CONVICTION = 65
    HIGH_CONVICTION = 80
    ULTRA_CONVICTION = 90

    def __init__(self):
        """Initialize synthesis layer."""
        self._bedrock = None
        self._call_count = 0
        self._strike_count = 0
        self._abort_count = 0
        self._wait_count = 0

        # Initialize Bedrock client
        try:
            from .bedrock_client import get_bedrock_client
            self._bedrock = get_bedrock_client()
            logger.info("SYNTHESIS_L12: Bedrock client initialized")
        except Exception as e:
            logger.warning(f"SYNTHESIS_L12: Bedrock unavailable: {e}")

    def synthesize(
        self,
        signal: Dict[str, Any],
        vision_result: Dict = None,
        semantic_result: Dict = None,
        hydra_result: Dict = None,
        adversarial_result: Dict = None,
        contrarian_result: Dict = None,
        dna_result: Dict = None,
        base_conviction: float = 50
    ) -> FinalVerdict:
        """
        Synthesize all layer outputs into final verdict.

        Args:
            signal: Base signal dict
            vision_result: Output from Layer 1
            semantic_result: Output from Layer 2
            hydra_result: Output from Layer H
            adversarial_result: Output from Layer 4
            contrarian_result: Output from Layer 5
            dna_result: Output from Layer 6
            base_conviction: Starting conviction before adjustments

        Returns:
            FinalVerdict with action and reasoning
        """
        start = time.time()
        self._call_count += 1

        ticker = signal.get('ticker', 'UNKNOWN')
        direction = signal.get('direction', 'NEUTRAL')
        price = signal.get('price', 0)
        entry_price = signal.get('entry_price', price)

        # Calculate final conviction from all adjustments
        final_conviction = self._calculate_conviction(
            base_conviction,
            vision_result,
            semantic_result,
            hydra_result,
            adversarial_result,
            contrarian_result,
            dna_result
        )

        # Quick decision if conviction is clearly below threshold
        if final_conviction < self.MIN_CONVICTION - 10:
            self._abort_count += 1
            return FinalVerdict(
                action=FinalAction.ABORT,
                conviction=final_conviction,
                position_size=0,
                reasoning=f"Conviction {final_conviction:.0f}% below minimum {self.MIN_CONVICTION}%.",
                latency_ms=(time.time() - start) * 1000
            )

        # Quick decision if adversarial killed it
        if adversarial_result and adversarial_result.get('result') == 'KILL':
            self._abort_count += 1
            return FinalVerdict(
                action=FinalAction.ABORT,
                conviction=final_conviction,
                position_size=0,
                reasoning=f"Adversarial kill: {adversarial_result.get('kill_reason', 'fatal flaw found')}",
                latency_ms=(time.time() - start) * 1000
            )

        # Quick decision if trap detected
        if contrarian_result and contrarian_result.get('trap_detected', 'NONE') != 'NONE':
            trap = contrarian_result.get('trap_detected')
            self._abort_count += 1
            return FinalVerdict(
                action=FinalAction.ABORT,
                conviction=final_conviction,
                position_size=0,
                reasoning=f"Trap detected: {trap}. {contrarian_result.get('reason', '')}",
                latency_ms=(time.time() - start) * 1000
            )

        # If Bedrock available and conviction is borderline, use AI synthesis
        if self._bedrock and self.MIN_CONVICTION - 5 <= final_conviction <= self.MIN_CONVICTION + 15:
            ai_result = self._ai_synthesis(
                signal, vision_result, semantic_result, hydra_result,
                adversarial_result, contrarian_result, dna_result,
                final_conviction
            )
            ai_result.latency_ms = (time.time() - start) * 1000
            self._track_action(ai_result.action)
            return ai_result

        # Rule-based decision for clear cases
        position_size = self._calculate_position_size(final_conviction)
        stop_loss, take_profit = self._calculate_exits(price, direction, final_conviction)

        if final_conviction >= self.MIN_CONVICTION:
            self._strike_count += 1
            return FinalVerdict(
                action=FinalAction.STRIKE,
                conviction=final_conviction,
                position_size=position_size,
                reasoning=self._build_reasoning(
                    vision_result, hydra_result, dna_result, final_conviction
                ),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                latency_ms=(time.time() - start) * 1000
            )
        else:
            self._wait_count += 1
            return FinalVerdict(
                action=FinalAction.WAIT,
                conviction=final_conviction,
                position_size=0,
                reasoning=f"Conviction {final_conviction:.0f}% near threshold. Wait for better setup.",
                latency_ms=(time.time() - start) * 1000
            )

    def _calculate_conviction(
        self,
        base: float,
        vision: Dict,
        semantic: Dict,
        hydra: Dict,
        adversarial: Dict,
        contrarian: Dict,
        dna: Dict
    ) -> float:
        """Calculate final conviction from all adjustments."""
        # Start with base
        conviction = base

        # Apply weighted adjustments
        if vision:
            conviction += vision.get('confidence', 0) * 15  # 15% weight

        if semantic:
            conviction += semantic.get('adjustment', 0) * 100  # Convert to %

        if hydra:
            conviction += hydra.get('adjustment', 0) * 100

        if adversarial:
            conviction += adversarial.get('adjustment', 0) * 100

        if contrarian:
            conviction += contrarian.get('adjustment', 0) * 100

        if dna:
            conviction += dna.get('adjustment', 0) * 100

        # Clamp to 0-100
        return max(0, min(100, conviction))

    def _calculate_position_size(self, conviction: float) -> int:
        """Calculate position size based on conviction."""
        if conviction >= self.ULTRA_CONVICTION:
            return 5  # Max size
        elif conviction >= self.HIGH_CONVICTION:
            return 3
        elif conviction >= self.MIN_CONVICTION:
            return 2
        else:
            return 1

    def _calculate_exits(
        self,
        price: float,
        direction: str,
        conviction: float
    ) -> tuple:
        """Calculate stop loss and take profit."""
        # Tighter stops for lower conviction
        if conviction >= self.ULTRA_CONVICTION:
            stop_pct = 0.15  # 15% stop
            target_pct = 0.30  # 30% target
        elif conviction >= self.HIGH_CONVICTION:
            stop_pct = 0.12
            target_pct = 0.20
        else:
            stop_pct = 0.10
            target_pct = 0.15

        if direction.upper() in ['LONG', 'CALL']:
            stop_loss = price * (1 - stop_pct)
            take_profit = price * (1 + target_pct)
        else:
            stop_loss = price * (1 + stop_pct)
            take_profit = price * (1 - target_pct)

        return stop_loss, take_profit

    def _build_reasoning(
        self,
        vision: Dict,
        hydra: Dict,
        dna: Dict,
        conviction: float
    ) -> str:
        """Build 2-sentence reasoning."""
        parts = []

        if vision and vision.get('pattern_detected') != 'none':
            parts.append(f"{vision.get('pattern_detected')} pattern")

        if hydra and hydra.get('connected'):
            parts.append(f"{hydra.get('gex_regime', 'unknown')} GEX")

        if dna and dna.get('match_type') != 'none':
            parts.append(f"{dna.get('historical_win_rate', 0):.0%} historical WR")

        setup = ", ".join(parts) if parts else "Multiple signals aligned"

        return f"{setup}. Conviction {conviction:.0f}% exceeds threshold."

    def _ai_synthesis(
        self,
        signal: Dict,
        vision: Dict,
        semantic: Dict,
        hydra: Dict,
        adversarial: Dict,
        contrarian: Dict,
        dna: Dict,
        conviction: float
    ) -> FinalVerdict:
        """Use AI for borderline decisions."""
        ticker = signal.get('ticker', 'UNKNOWN')
        direction = signal.get('direction', 'NEUTRAL')
        price = signal.get('price', 0)
        entry_price = signal.get('entry_price', price)

        prompt = self.SYNTHESIS_PROMPT.format(
            vision_pattern=vision.get('pattern_detected', 'none') if vision else 'N/A',
            vision_bias=vision.get('raw_bias', 'NEUTRAL') if vision else 'N/A',
            vision_conf=vision.get('confidence', 0) if vision else 0,
            semantic_adj=semantic.get('adjustment', 0) if semantic else 0,
            gex_regime=hydra.get('gex_regime', 'UNKNOWN') if hydra else 'N/A',
            flow_bias=hydra.get('flow_bias', 'NEUTRAL') if hydra else 'N/A',
            hydra_adj=hydra.get('adjustment', 0) if hydra else 0,
            adversarial_result=adversarial.get('result', 'N/A') if adversarial else 'N/A',
            adversarial_adj=adversarial.get('adjustment', 0) if adversarial else 0,
            trap_type=contrarian.get('trap_detected', 'NONE') if contrarian else 'NONE',
            contrarian_adj=contrarian.get('adjustment', 0) if contrarian else 0,
            dna_win_rate=dna.get('historical_win_rate', 0) if dna else 0,
            dna_match_type=dna.get('match_type', 'none') if dna else 'none',
            dna_adj=dna.get('adjustment', 0) if dna else 0,
            ticker=ticker,
            direction=direction,
            price=price,
            entry_price=entry_price,
            final_conviction=conviction,
            threshold=self.MIN_CONVICTION
        )

        try:
            response = self._bedrock.invoke_claude(
                prompt=prompt,
                model='claude-sonnet',
                max_tokens=200,
                temperature=0,
                system="You are a decisive trading system. Make clear STRIKE/ABORT/WAIT decisions."
            )

            if not response.success:
                return FinalVerdict(
                    action=FinalAction.WAIT,
                    conviction=conviction,
                    position_size=0,
                    reasoning="AI synthesis failed, defaulting to WAIT",
                    error=response.error
                )

            return self._parse_ai_response(response.text, conviction, price, direction)

        except Exception as e:
            logger.error(f"SYNTHESIS_L12: AI error - {e}")
            return FinalVerdict(
                action=FinalAction.WAIT,
                conviction=conviction,
                position_size=0,
                reasoning="AI synthesis error, defaulting to WAIT",
                error=str(e)
            )

    def _parse_ai_response(
        self,
        text: str,
        conviction: float,
        price: float,
        direction: str
    ) -> FinalVerdict:
        """Parse Claude's synthesis response."""
        lines = text.strip().split('\n')

        action = FinalAction.WAIT
        size = 1
        stop = 0
        target = 0
        reason = "AI synthesis"

        for line in lines:
            line = line.strip().upper()
            if line.startswith("ACTION:"):
                action_str = line.split(":", 1)[1].strip()
                if "STRIKE" in action_str:
                    action = FinalAction.STRIKE
                elif "ABORT" in action_str:
                    action = FinalAction.ABORT
                else:
                    action = FinalAction.WAIT

            elif line.startswith("SIZE:"):
                try:
                    size = int(line.split(":", 1)[1].strip().split()[0])
                    size = max(1, min(5, size))
                except:
                    size = 1

            elif line.startswith("STOP:"):
                try:
                    stop = float(line.split(":", 1)[1].strip().replace("$", ""))
                except:
                    stop = price * 0.90 if direction.upper() in ['LONG', 'CALL'] else price * 1.10

            elif line.startswith("TARGET:"):
                try:
                    target = float(line.split(":", 1)[1].strip().replace("$", ""))
                except:
                    target = price * 1.15 if direction.upper() in ['LONG', 'CALL'] else price * 0.85

            elif line.startswith("REASON:"):
                reason = line.split(":", 1)[1].strip()

        return FinalVerdict(
            action=action,
            conviction=conviction,
            position_size=size if action == FinalAction.STRIKE else 0,
            reasoning=reason,
            entry_price=price,
            stop_loss=stop,
            take_profit=target
        )

    def _track_action(self, action: FinalAction):
        """Track action statistics."""
        if action == FinalAction.STRIKE:
            self._strike_count += 1
        elif action == FinalAction.ABORT:
            self._abort_count += 1
        else:
            self._wait_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics."""
        total = self._call_count
        return {
            'call_count': total,
            'strike_count': self._strike_count,
            'abort_count': self._abort_count,
            'wait_count': self._wait_count,
            'strike_rate': self._strike_count / max(total, 1),
            'abort_rate': self._abort_count / max(total, 1)
        }


# Singleton
_synthesis_layer = None

def get_synthesis_layer() -> SynthesisLayer:
    """Get singleton SynthesisLayer instance."""
    global _synthesis_layer
    if _synthesis_layer is None:
        _synthesis_layer = SynthesisLayer()
    return _synthesis_layer
