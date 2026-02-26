"""
Predator Stack v2.1 — Orchestrator

The main orchestrator that runs all layers in sequence and produces
a final PredatorVerdict for trade execution.

Pipeline:
1. L0: Speed Filter (gate) — kill obvious losers in <80ms
2. L1: Vision (15%) — chart pattern detection
3. L2: Semantic (10%) — news similarity matching
4. LH: HYDRA (20%) — market structure intelligence
5. L4: Adversarial (15%) — thesis destroyer
6. L5: Contrarian (10%) — trap detection
7. L6: DNA (25%) — your historical wins
8. L12: Synthesis (5%) — final arbiter

Total: 100% weighted conviction
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredatorVerdict:
    """Final verdict from Predator Stack."""
    # Decision
    action: str  # "STRIKE", "ABORT", "WAIT"
    conviction: float  # 0-100 final score
    position_size: int  # Number of contracts

    # Execution params
    entry_price: float = 0
    stop_loss: float = 0
    take_profit: float = 0

    # Reasoning
    reasoning: str = ""
    kill_reason: Optional[str] = None

    # Layer outputs
    layer_results: Dict[str, Any] = field(default_factory=dict)

    # Performance
    total_latency_ms: float = 0
    layers_run: List[str] = field(default_factory=list)

    # Flags
    speed_filtered: bool = False
    adversarial_killed: bool = False
    trap_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'conviction': self.conviction,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'reasoning': self.reasoning,
            'kill_reason': self.kill_reason,
            'total_latency_ms': self.total_latency_ms,
            'layers_run': self.layers_run,
            'speed_filtered': self.speed_filtered,
            'adversarial_killed': self.adversarial_killed,
            'trap_detected': self.trap_detected
        }


class PredatorStackV2:
    """
    Predator Stack v2.1 Orchestrator.

    Runs all AI layers in optimized sequence to produce
    a final trade decision with conviction score.
    """

    # Layer weights
    WEIGHTS = {
        'vision': 0.15,
        'semantic': 0.10,
        'hydra': 0.20,
        'adversarial': 0.15,
        'contrarian': 0.10,
        'dna': 0.25,
        'synthesis': 0.05
    }

    # Base conviction before adjustments
    BASE_CONVICTION = 50

    def __init__(self):
        """Initialize Predator Stack."""
        self._layers = {}
        self._call_count = 0
        self._strike_count = 0
        self._abort_count = 0
        self._filter_count = 0

        # Initialize layers lazily
        logger.info("PREDATOR_STACK: v2.1 initialized")

    def _get_layer(self, name: str):
        """Get layer instance (lazy initialization)."""
        if name not in self._layers:
            if name == 'speed_filter':
                from .layer_0_speed_filter import get_speed_filter
                self._layers[name] = get_speed_filter()
            elif name == 'vision':
                from .layer_1_vision import get_vision_layer
                self._layers[name] = get_vision_layer()
            elif name == 'semantic':
                from .layer_2_semantic import get_semantic_layer
                self._layers[name] = get_semantic_layer()
            elif name == 'hydra':
                from .layer_h_hydra import get_hydra_layer
                self._layers[name] = get_hydra_layer()
            elif name == 'adversarial':
                from .layer_4_adversarial import get_adversarial_layer
                self._layers[name] = get_adversarial_layer()
            elif name == 'contrarian':
                from .layer_5_contrarian import get_contrarian_layer
                self._layers[name] = get_contrarian_layer()
            elif name == 'dna':
                from .layer_6_dna import get_strategy_dna
                self._layers[name] = get_strategy_dna()
            elif name == 'synthesis':
                from .layer_12_synthesis import get_synthesis_layer
                self._layers[name] = get_synthesis_layer()

        return self._layers.get(name)

    def analyze(
        self,
        signal: Dict[str, Any],
        chart_image: str = None,
        news_headlines: List[str] = None,
        candles: List[Dict] = None
    ) -> PredatorVerdict:
        """
        Run full Predator Stack analysis on a signal.

        Args:
            signal: Dict with 'ticker', 'direction', 'price', etc.
            chart_image: Optional base64 chart image for vision
            news_headlines: Optional recent news for semantic matching
            candles: Optional candle data if no image

        Returns:
            PredatorVerdict with final decision
        """
        start = time.time()
        self._call_count += 1

        ticker = signal.get('ticker', 'UNKNOWN')
        direction = signal.get('direction', 'NEUTRAL')
        layers_run = []
        layer_results = {}

        logger.info(f"PREDATOR_STACK: Analyzing {ticker} {direction}")

        # ============================================
        # LAYER 0: Speed Filter (Gate)
        # ============================================
        speed_filter = self._get_layer('speed_filter')
        filter_result = speed_filter.filter(signal)
        layers_run.append('L0_speed')
        layer_results['speed_filter'] = {
            'passed': filter_result.passed,
            'reason': filter_result.reason,
            'latency_ms': filter_result.latency_ms
        }

        if not filter_result.passed:
            self._filter_count += 1
            logger.info(f"PREDATOR_STACK: FILTERED - {filter_result.reason}")
            return PredatorVerdict(
                action="ABORT",
                conviction=0,
                position_size=0,
                reasoning=f"Speed filter: {filter_result.reason}",
                kill_reason=filter_result.reason,
                layer_results=layer_results,
                total_latency_ms=(time.time() - start) * 1000,
                layers_run=layers_run,
                speed_filtered=True
            )

        # ============================================
        # PARALLEL LAYERS: Vision, Semantic, HYDRA, DNA
        # ============================================
        vision_result = None
        semantic_result = None
        hydra_result = None
        dna_result = None

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}

            # Vision (L1)
            vision_layer = self._get_layer('vision')
            if chart_image:
                futures['vision'] = executor.submit(
                    vision_layer.analyze_chart,
                    image_base64=chart_image,
                    ticker=ticker
                )
            elif candles:
                futures['vision'] = executor.submit(
                    vision_layer.analyze_from_data,
                    candles=candles,
                    ticker=ticker
                )

            # Semantic (L2)
            if news_headlines:
                semantic_layer = self._get_layer('semantic')
                futures['semantic'] = executor.submit(
                    semantic_layer.match,
                    ticker=ticker,
                    news_headlines=news_headlines,
                    direction=direction
                )

            # HYDRA (LH)
            hydra_layer = self._get_layer('hydra')
            futures['hydra'] = executor.submit(
                hydra_layer.score,
                signal=signal
            )

            # DNA (L6)
            dna_layer = self._get_layer('dna')
            # Get pattern from vision if available
            pattern = None
            regime = None
            try:
                from wsb_snake.collectors.hydra_bridge import get_hydra_intel
                hydra = get_hydra_intel()
                regime = hydra.regime
            except:
                pass

            futures['dna'] = executor.submit(
                dna_layer.score,
                signal=signal,
                pattern=pattern,
                regime=regime
            )

            # Collect results
            for name, future in futures.items():
                try:
                    result = future.result(timeout=5)
                    if name == 'vision':
                        vision_result = result.to_dict() if hasattr(result, 'to_dict') else result
                        layers_run.append('L1_vision')
                    elif name == 'semantic':
                        semantic_result = result.to_dict() if hasattr(result, 'to_dict') else result
                        layers_run.append('L2_semantic')
                    elif name == 'hydra':
                        hydra_result = result.to_dict() if hasattr(result, 'to_dict') else result
                        layers_run.append('LH_hydra')
                    elif name == 'dna':
                        dna_result = result.to_dict() if hasattr(result, 'to_dict') else result
                        layers_run.append('L6_dna')

                    layer_results[name] = result.to_dict() if hasattr(result, 'to_dict') else result

                except Exception as e:
                    logger.warning(f"PREDATOR_STACK: {name} failed - {e}")
                    layer_results[name] = {'error': str(e)}

        # ============================================
        # LAYER 4: Adversarial Challenge
        # ============================================
        adversarial_layer = self._get_layer('adversarial')
        adversarial_result = adversarial_layer.challenge(
            signal=signal,
            vision_signal=vision_result,
            hydra_data=hydra_result,
            semantic_data=semantic_result
        )
        layers_run.append('L4_adversarial')
        layer_results['adversarial'] = adversarial_result.to_dict()

        # Check for adversarial kill
        if adversarial_result.result.value == 'KILL':
            self._abort_count += 1
            logger.info(f"PREDATOR_STACK: ADVERSARIAL KILL - {adversarial_result.kill_reason}")
            return PredatorVerdict(
                action="ABORT",
                conviction=0,
                position_size=0,
                reasoning=f"Adversarial: {adversarial_result.kill_reason}",
                kill_reason=adversarial_result.kill_reason,
                layer_results=layer_results,
                total_latency_ms=(time.time() - start) * 1000,
                layers_run=layers_run,
                adversarial_killed=True
            )

        # ============================================
        # LAYER 5: Contrarian Detection
        # ============================================
        contrarian_layer = self._get_layer('contrarian')
        contrarian_result = contrarian_layer.detect(
            signal=signal,
            vision_signal=vision_result,
            hydra_data=hydra_result
        )
        layers_run.append('L5_contrarian')
        layer_results['contrarian'] = contrarian_result.to_dict()

        # Check for trap - now ADVISORY (reduces conviction) instead of HARD BLOCK
        # FIX: Contrarian was blocking ALL trades - now just reduces conviction
        trap_penalty = 0
        if contrarian_result.trap_detected.value != 'NONE':
            trap_penalty = abs(contrarian_result.adjustment) * 100  # e.g., -0.15 → -15%
            logger.warning(f"PREDATOR_STACK: TRAP WARNING - {contrarian_result.trap_detected.value} (conviction -{trap_penalty:.0f}%)")
            # Only hard block on BULL_TRAP or BEAR_TRAP (high confidence traps)
            if contrarian_result.trap_detected.value in ['BULL_TRAP', 'BEAR_TRAP'] and contrarian_result.trap_confidence > 0.8:
                self._abort_count += 1
                logger.info(f"PREDATOR_STACK: HIGH CONFIDENCE TRAP - blocking trade")
                return PredatorVerdict(
                    action="ABORT",
                    conviction=0,
                    position_size=0,
                    reasoning=f"High confidence trap: {contrarian_result.reason}",
                    kill_reason=f"{contrarian_result.trap_detected.value}: {contrarian_result.reason}",
                    layer_results=layer_results,
                    total_latency_ms=(time.time() - start) * 1000,
                    layers_run=layers_run,
                    trap_detected=True
                )

        # ============================================
        # Calculate Base Conviction
        # ============================================
        base_conviction = self.BASE_CONVICTION - trap_penalty  # Apply trap penalty

        # Apply vision confidence
        if vision_result and vision_result.get('confidence', 0) > 0:
            base_conviction += vision_result['confidence'] * self.WEIGHTS['vision'] * 100

        # ============================================
        # LAYER 12: Final Synthesis
        # ============================================
        synthesis_layer = self._get_layer('synthesis')
        final_verdict = synthesis_layer.synthesize(
            signal=signal,
            vision_result=vision_result,
            semantic_result=semantic_result,
            hydra_result=hydra_result,
            adversarial_result=layer_results.get('adversarial'),
            contrarian_result=layer_results.get('contrarian'),
            dna_result=dna_result,
            base_conviction=base_conviction
        )
        layers_run.append('L12_synthesis')
        layer_results['synthesis'] = final_verdict.to_dict()

        total_latency = (time.time() - start) * 1000

        # Track stats
        if final_verdict.action == "STRIKE":
            self._strike_count += 1
        else:
            self._abort_count += 1

        logger.info(
            f"PREDATOR_STACK: {ticker} {direction} → {final_verdict.action} "
            f"conv={final_verdict.conviction:.0f}% size={final_verdict.position_size} "
            f"in {total_latency:.0f}ms"
        )

        return PredatorVerdict(
            action=final_verdict.action.value,
            conviction=final_verdict.conviction,
            position_size=final_verdict.position_size,
            entry_price=final_verdict.entry_price,
            stop_loss=final_verdict.stop_loss,
            take_profit=final_verdict.take_profit,
            reasoning=final_verdict.reasoning,
            layer_results=layer_results,
            total_latency_ms=total_latency,
            layers_run=layers_run
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get stack statistics."""
        total = self._call_count
        layer_stats = {}

        for name, layer in self._layers.items():
            if hasattr(layer, 'get_stats'):
                layer_stats[name] = layer.get_stats()

        return {
            'call_count': total,
            'strike_count': self._strike_count,
            'abort_count': self._abort_count,
            'filter_count': self._filter_count,
            'strike_rate': self._strike_count / max(total, 1),
            'filter_rate': self._filter_count / max(total, 1),
            'layer_stats': layer_stats
        }


# Singleton
_predator_stack = None

def get_predator_stack() -> PredatorStackV2:
    """Get singleton PredatorStackV2 instance."""
    global _predator_stack
    if _predator_stack is None:
        _predator_stack = PredatorStackV2()
    return _predator_stack
