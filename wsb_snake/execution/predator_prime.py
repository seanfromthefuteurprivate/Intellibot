"""
PREDATOR PRIME — Unified Hedge Fund Grade Execution Engine

This module connects ALL the sophisticated AI layers we built:
- Predator Stack v2.1 (8 AI layers)
- HYDRA Intelligence (GEX, Flow, Dark Pool, Sequences)
- Dynamic ATR-based stops
- Entry timing windows
- Gamma-aware exits
- VIX regime adaptation

REPLACES the old disconnected execution path.
"""

import time
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class EntryWindow(Enum):
    """Trading window classifications."""
    MORNING_TREND = "MORNING_TREND"  # 10:00-12:00 ET - Best for directional
    LUNCH_LULL = "LUNCH_LULL"        # 12:00-14:00 ET - AVOID
    MOC_REBALANCE = "MOC_REBALANCE"  # 14:00-15:00 ET - Follow imbalance
    POWER_HOUR = "POWER_HOUR"        # 15:00-15:30 ET - Momentum
    THETA_ACCEL = "THETA_ACCEL"      # 15:30-16:00 ET - Sell premium
    PRE_OPEN = "PRE_OPEN"            # 09:30-10:00 ET - High spread, avoid


class VIXRegime(Enum):
    """VIX-based market regime."""
    LOW_VOL = "LOW_VOL"      # VIX < 15
    NORMAL = "NORMAL"        # VIX 15-20
    ELEVATED = "ELEVATED"    # VIX 20-28
    CRISIS = "CRISIS"        # VIX > 28


@dataclass
class PredatorPrimeVerdict:
    """Ultimate trade decision from Predator Prime."""
    # Core decision
    action: str  # "STRIKE", "ABORT", "WAIT"
    direction: str  # "CALL", "PUT"
    conviction: float  # 0-100

    # Position sizing
    contracts: int = 1
    max_risk_dollars: float = 500

    # Dynamic stops (ATR-based)
    stop_type: str = "ATR"  # "ATR" or "FIXED"
    stop_atr_multiplier: float = 2.0
    stop_price: Optional[float] = None
    stop_pct: float = 0.15  # Fallback if ATR not available

    # Targets
    target_price: Optional[float] = None
    target_pct: float = 0.20
    scale_out_at_pct: float = 0.10  # Scale out 50% at 10%

    # Timing
    max_hold_minutes: int = 15
    entry_window: str = "STANDARD"
    window_size_mult: float = 1.0  # Position size multiplier based on window

    # Intelligence used
    hydra_gex_regime: str = "UNKNOWN"
    hydra_flow_bias: str = "NEUTRAL"
    hydra_blowup_prob: float = 0.0
    predator_layers_run: List[str] = field(default_factory=list)
    speed_filter_passed: bool = True
    adversarial_survived: bool = True
    trap_detected: str = "NONE"

    # Reasoning
    reasoning: str = ""
    kill_reason: Optional[str] = None

    # Performance
    total_latency_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'direction': self.direction,
            'conviction': self.conviction,
            'contracts': self.contracts,
            'stop_type': self.stop_type,
            'stop_atr_multiplier': self.stop_atr_multiplier,
            'stop_pct': self.stop_pct,
            'target_pct': self.target_pct,
            'max_hold_minutes': self.max_hold_minutes,
            'entry_window': self.entry_window,
            'hydra_gex_regime': self.hydra_gex_regime,
            'hydra_flow_bias': self.hydra_flow_bias,
            'hydra_blowup_prob': self.hydra_blowup_prob,
            'predator_layers_run': self.predator_layers_run,
            'speed_filter_passed': self.speed_filter_passed,
            'adversarial_survived': self.adversarial_survived,
            'trap_detected': self.trap_detected,
            'reasoning': self.reasoning,
            'kill_reason': self.kill_reason,
            'total_latency_ms': self.total_latency_ms
        }


class PredatorPrime:
    """
    Predator Prime — The Ultimate 0DTE Execution Engine

    Connects:
    1. Predator Stack v2.1 (all 8 AI layers)
    2. HYDRA Intelligence (GEX, Flow, Dark Pool, Sequences)
    3. Dynamic ATR-based risk management
    4. Entry timing windows
    5. VIX regime adaptation

    This is the SINGLE entry point for all trade decisions.
    """

    # Position size tiers based on conviction
    SIZE_TIERS = {
        90: 3,   # 90%+ conviction = 3 contracts
        80: 2,   # 80-89% = 2 contracts
        70: 1,   # 70-79% = 1 contract
        0: 0     # Below 70 = no trade
    }

    # VIX regime adjustments
    VIX_ADJUSTMENTS = {
        VIXRegime.LOW_VOL: {'size_mult': 1.2, 'stop_mult': 1.5, 'target_mult': 1.0},
        VIXRegime.NORMAL: {'size_mult': 1.0, 'stop_mult': 2.0, 'target_mult': 1.0},
        VIXRegime.ELEVATED: {'size_mult': 0.75, 'stop_mult': 2.5, 'target_mult': 1.2},
        VIXRegime.CRISIS: {'size_mult': 0.5, 'stop_mult': 4.0, 'target_mult': 1.5},
    }

    # Entry window adjustments
    WINDOW_ADJUSTMENTS = {
        EntryWindow.MORNING_TREND: {'size_mult': 1.0, 'allow_trade': True},
        EntryWindow.LUNCH_LULL: {'size_mult': 0.0, 'allow_trade': False},
        EntryWindow.MOC_REBALANCE: {'size_mult': 0.75, 'allow_trade': True},
        EntryWindow.POWER_HOUR: {'size_mult': 0.75, 'allow_trade': True},
        EntryWindow.THETA_ACCEL: {'size_mult': 0.5, 'allow_trade': True},
        EntryWindow.PRE_OPEN: {'size_mult': 0.5, 'allow_trade': True},
    }

    def __init__(self):
        """Initialize Predator Prime with all subsystems."""
        self._predator_stack = None
        self._hydra_bridge = None
        self._call_count = 0
        self._strike_count = 0
        self._abort_count = 0

        logger.info("PREDATOR_PRIME: Hedge fund grade execution engine initialized")

    def _get_predator_stack(self):
        """Lazy load Predator Stack v2.1."""
        if self._predator_stack is None:
            try:
                from wsb_snake.ai_stack.predator_stack_v2 import get_predator_stack
                self._predator_stack = get_predator_stack()
                logger.info("PREDATOR_PRIME: Predator Stack v2.1 loaded")
            except Exception as e:
                logger.error(f"PREDATOR_PRIME: Failed to load Predator Stack - {e}")
        return self._predator_stack

    def _get_hydra_bridge(self):
        """Lazy load HYDRA bridge."""
        if self._hydra_bridge is None:
            try:
                from wsb_snake.collectors.hydra_bridge import get_hydra_bridge
                self._hydra_bridge = get_hydra_bridge()
                logger.info("PREDATOR_PRIME: HYDRA Bridge loaded")
            except Exception as e:
                logger.warning(f"PREDATOR_PRIME: HYDRA Bridge unavailable - {e}")
        return self._hydra_bridge

    def _get_entry_window(self) -> Tuple[EntryWindow, Dict]:
        """
        Determine current trading window.

        Returns:
            (window_enum, adjustments_dict)
        """
        now = datetime.now()
        hour = now.hour
        minute = now.minute

        # Convert to ET (assuming system is in ET or adjust as needed)
        # For simplicity, using local time

        if hour == 9 and minute >= 30:
            window = EntryWindow.PRE_OPEN
        elif 10 <= hour < 12:
            window = EntryWindow.MORNING_TREND
        elif 12 <= hour < 14:
            window = EntryWindow.LUNCH_LULL
        elif 14 <= hour < 15:
            window = EntryWindow.MOC_REBALANCE
        elif hour == 15 and minute < 30:
            window = EntryWindow.POWER_HOUR
        elif hour == 15 and minute >= 30:
            window = EntryWindow.THETA_ACCEL
        else:
            window = EntryWindow.MORNING_TREND  # Default

        return window, self.WINDOW_ADJUSTMENTS.get(window, {'size_mult': 1.0, 'allow_trade': True})

    def _get_vix_regime(self) -> Tuple[VIXRegime, Dict]:
        """
        Determine VIX regime from HYDRA or defaults.

        Returns:
            (regime_enum, adjustments_dict)
        """
        hydra = self._get_hydra_bridge()
        vix = 20.0  # Default

        if hydra:
            try:
                intel = hydra.get_intel()
                if intel and hasattr(intel, 'vix'):
                    vix = intel.vix or 20.0
            except:
                pass

        if vix < 15:
            regime = VIXRegime.LOW_VOL
        elif vix <= 20:
            regime = VIXRegime.NORMAL
        elif vix <= 28:
            regime = VIXRegime.ELEVATED
        else:
            regime = VIXRegime.CRISIS

        return regime, self.VIX_ADJUSTMENTS.get(regime, self.VIX_ADJUSTMENTS[VIXRegime.NORMAL])

    def _calculate_position_size(
        self,
        conviction: float,
        vix_adjustments: Dict,
        window_adjustments: Dict,
        hours_to_expiry: float = 6.5
    ) -> int:
        """
        Calculate position size based on conviction and conditions.

        Uses tiered sizing with VIX and time adjustments.
        """
        # Base size from conviction tier
        base_contracts = 0
        for threshold, contracts in sorted(self.SIZE_TIERS.items(), reverse=True):
            if conviction >= threshold:
                base_contracts = contracts
                break

        # Apply VIX multiplier
        vix_mult = vix_adjustments.get('size_mult', 1.0)

        # Apply window multiplier
        window_mult = window_adjustments.get('size_mult', 1.0)

        # Apply time decay multiplier (reduce late in session)
        if hours_to_expiry < 1:
            time_mult = 0.5
        elif hours_to_expiry < 2:
            time_mult = 0.75
        else:
            time_mult = 1.0

        # Final calculation
        final_contracts = int(base_contracts * vix_mult * window_mult * time_mult)

        return max(0, final_contracts)

    def _calculate_dynamic_stops(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        vix_adjustments: Dict
    ) -> Tuple[float, float, float]:
        """
        Calculate ATR-based dynamic stops.

        Returns:
            (stop_price, target_price, stop_pct)
        """
        # ATR multiplier adjusted by VIX regime
        stop_mult = vix_adjustments.get('stop_mult', 2.0)
        target_mult = vix_adjustments.get('target_mult', 1.0)

        stop_distance = atr * stop_mult
        target_distance = atr * stop_mult * 1.5 * target_mult  # 1.5:1 R:R minimum

        if direction in ("CALL", "long", "BULLISH"):
            stop_price = entry_price - stop_distance
            target_price = entry_price + target_distance
        else:
            stop_price = entry_price + stop_distance
            target_price = entry_price - target_distance

        stop_pct = stop_distance / entry_price if entry_price > 0 else 0.15

        return stop_price, target_price, stop_pct

    def analyze(
        self,
        ticker: str,
        direction: str,  # "long" / "short" / "CALL" / "PUT"
        current_price: float,
        chart_image: str = None,
        news_headlines: List[str] = None,
        candles: List[Dict] = None,
        atr: float = None,
        hours_to_expiry: float = 6.5
    ) -> PredatorPrimeVerdict:
        """
        MAIN ENTRY POINT — Full Predator Prime analysis.

        This runs ALL layers:
        1. Entry window check
        2. VIX regime detection
        3. HYDRA intelligence fetch
        4. Predator Stack v2.1 (Speed Filter → Vision → Semantic → HYDRA →
           Adversarial → Contrarian → DNA → Synthesis)
        5. Dynamic position sizing
        6. ATR-based stop calculation

        Args:
            ticker: Symbol (e.g., "SPY")
            direction: Trade direction
            current_price: Current underlying price
            chart_image: Base64 encoded chart
            news_headlines: Recent news
            candles: OHLCV data
            atr: Average True Range (if available)
            hours_to_expiry: Hours until option expiry

        Returns:
            PredatorPrimeVerdict with full decision
        """
        start = time.time()
        self._call_count += 1

        # Normalize direction
        if direction in ("long", "CALL", "BULLISH", "call"):
            norm_direction = "CALL"
        else:
            norm_direction = "PUT"

        logger.info(f"PREDATOR_PRIME: Analyzing {ticker} {norm_direction} @ ${current_price:.2f}")

        # ============================================
        # STEP 1: Entry Window Check
        # ============================================
        entry_window, window_adj = self._get_entry_window()

        if not window_adj.get('allow_trade', True):
            self._abort_count += 1
            return PredatorPrimeVerdict(
                action="ABORT",
                direction=norm_direction,
                conviction=0,
                entry_window=entry_window.value,
                kill_reason=f"Trading blocked during {entry_window.value}",
                reasoning=f"Entry window {entry_window.value} does not allow new trades (lunch lull)",
                total_latency_ms=(time.time() - start) * 1000
            )

        # ============================================
        # STEP 2: VIX Regime Detection
        # ============================================
        vix_regime, vix_adj = self._get_vix_regime()

        # ============================================
        # STEP 3: Get HYDRA Intelligence
        # ============================================
        hydra_gex_regime = "UNKNOWN"
        hydra_flow_bias = "NEUTRAL"
        hydra_blowup_prob = 0.0

        hydra = self._get_hydra_bridge()
        if hydra:
            try:
                intel = hydra.get_intel()
                if intel:
                    hydra_gex_regime = getattr(intel, 'gex_regime', 'UNKNOWN')
                    hydra_flow_bias = getattr(intel, 'flow_bias', 'NEUTRAL')
                    hydra_blowup_prob = getattr(intel, 'blowup_probability', 0.0)

                    # Check for HOLD_OFF recommendation
                    recommendation = getattr(intel, 'recommendation', '')
                    if recommendation == 'HOLD_OFF':
                        self._abort_count += 1
                        return PredatorPrimeVerdict(
                            action="ABORT",
                            direction=norm_direction,
                            conviction=0,
                            hydra_gex_regime=hydra_gex_regime,
                            hydra_flow_bias=hydra_flow_bias,
                            hydra_blowup_prob=hydra_blowup_prob,
                            entry_window=entry_window.value,
                            kill_reason="HYDRA recommends HOLD_OFF",
                            reasoning="HYDRA intelligence indicates unfavorable conditions",
                            total_latency_ms=(time.time() - start) * 1000
                        )
            except Exception as e:
                logger.warning(f"PREDATOR_PRIME: HYDRA fetch error - {e}")

        # ============================================
        # STEP 4: Run Predator Stack v2.1
        # ============================================
        predator = self._get_predator_stack()
        predator_verdict = None
        layers_run = []
        speed_filter_passed = True
        adversarial_survived = True
        trap_detected = "NONE"

        if predator:
            try:
                signal = {
                    'ticker': ticker,
                    'direction': norm_direction,
                    'price': current_price
                }

                predator_verdict = predator.analyze(
                    signal=signal,
                    chart_image=chart_image,
                    news_headlines=news_headlines,
                    candles=candles
                )

                if predator_verdict:
                    layers_run = predator_verdict.layers_run
                    speed_filter_passed = not predator_verdict.speed_filtered
                    adversarial_survived = not predator_verdict.adversarial_killed
                    trap_detected = "TRAP" if predator_verdict.trap_detected else "NONE"

                    # If Predator Stack says ABORT, respect it
                    if predator_verdict.action == "ABORT":
                        self._abort_count += 1
                        return PredatorPrimeVerdict(
                            action="ABORT",
                            direction=norm_direction,
                            conviction=predator_verdict.conviction,
                            hydra_gex_regime=hydra_gex_regime,
                            hydra_flow_bias=hydra_flow_bias,
                            hydra_blowup_prob=hydra_blowup_prob,
                            entry_window=entry_window.value,
                            predator_layers_run=layers_run,
                            speed_filter_passed=speed_filter_passed,
                            adversarial_survived=adversarial_survived,
                            trap_detected=trap_detected,
                            kill_reason=predator_verdict.kill_reason,
                            reasoning=predator_verdict.reasoning,
                            total_latency_ms=(time.time() - start) * 1000
                        )
            except Exception as e:
                logger.error(f"PREDATOR_PRIME: Predator Stack error - {e}")

        # ============================================
        # STEP 5: Calculate Final Conviction
        # ============================================
        base_conviction = 50.0

        if predator_verdict:
            base_conviction = predator_verdict.conviction

        # Adjust for blowup probability (if high, boost conviction for aligned trades)
        if hydra_blowup_prob > 60:
            blowup_dir = hydra.get_intel().direction if hydra else "NEUTRAL"
            if (norm_direction == "CALL" and blowup_dir == "BULLISH") or \
               (norm_direction == "PUT" and blowup_dir == "BEARISH"):
                base_conviction += 10  # Aligned with blowup direction
            else:
                base_conviction -= 15  # Fighting blowup direction

        # ============================================
        # STEP 6: Position Sizing
        # ============================================
        contracts = self._calculate_position_size(
            conviction=base_conviction,
            vix_adjustments=vix_adj,
            window_adjustments=window_adj,
            hours_to_expiry=hours_to_expiry
        )

        if contracts == 0:
            self._abort_count += 1
            return PredatorPrimeVerdict(
                action="ABORT",
                direction=norm_direction,
                conviction=base_conviction,
                contracts=0,
                hydra_gex_regime=hydra_gex_regime,
                hydra_flow_bias=hydra_flow_bias,
                hydra_blowup_prob=hydra_blowup_prob,
                entry_window=entry_window.value,
                predator_layers_run=layers_run,
                speed_filter_passed=speed_filter_passed,
                adversarial_survived=adversarial_survived,
                trap_detected=trap_detected,
                kill_reason=f"Conviction {base_conviction:.0f}% below threshold",
                reasoning=f"Position size calculated to 0 contracts (conviction={base_conviction:.0f}%, vix_mult={vix_adj['size_mult']}, window_mult={window_adj['size_mult']})",
                total_latency_ms=(time.time() - start) * 1000
            )

        # ============================================
        # STEP 7: Dynamic Stop Calculation
        # ============================================
        stop_price = None
        target_price = None
        stop_pct = 0.15  # Default fallback

        if atr and atr > 0:
            stop_price, target_price, stop_pct = self._calculate_dynamic_stops(
                entry_price=current_price,
                direction=norm_direction,
                atr=atr,
                vix_adjustments=vix_adj
            )
            stop_type = "ATR"
        else:
            # Fallback to VIX-adjusted fixed stops
            stop_pct = 0.07 * vix_adj.get('stop_mult', 2.0) / 2.0  # Adjust 7% base
            stop_type = "FIXED"

        # ============================================
        # STEP 8: Build Final Verdict
        # ============================================
        total_latency = (time.time() - start) * 1000
        self._strike_count += 1

        verdict = PredatorPrimeVerdict(
            action="STRIKE",
            direction=norm_direction,
            conviction=base_conviction,
            contracts=contracts,
            stop_type=stop_type,
            stop_atr_multiplier=vix_adj.get('stop_mult', 2.0),
            stop_price=stop_price,
            stop_pct=stop_pct,
            target_price=target_price,
            target_pct=stop_pct * 1.5,  # 1.5:1 R:R
            scale_out_at_pct=stop_pct * 0.7,  # Scale out before full target
            max_hold_minutes=15 if hours_to_expiry > 2 else 5,
            entry_window=entry_window.value,
            window_size_mult=window_adj.get('size_mult', 1.0),
            hydra_gex_regime=hydra_gex_regime,
            hydra_flow_bias=hydra_flow_bias,
            hydra_blowup_prob=hydra_blowup_prob,
            predator_layers_run=layers_run,
            speed_filter_passed=speed_filter_passed,
            adversarial_survived=adversarial_survived,
            trap_detected=trap_detected,
            reasoning=f"Predator Prime STRIKE: {base_conviction:.0f}% conviction, {contracts} contracts, {entry_window.value} window, VIX regime {vix_regime.value}",
            total_latency_ms=total_latency
        )

        logger.info(
            f"PREDATOR_PRIME: {ticker} {norm_direction} → STRIKE "
            f"conv={base_conviction:.0f}% size={contracts} "
            f"stop={stop_pct*100:.1f}% target={stop_pct*150:.1f}% "
            f"in {total_latency:.0f}ms"
        )

        return verdict

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            'call_count': self._call_count,
            'strike_count': self._strike_count,
            'abort_count': self._abort_count,
            'strike_rate': self._strike_count / max(self._call_count, 1),
            'abort_rate': self._abort_count / max(self._call_count, 1)
        }


# Singleton instance
_predator_prime: Optional[PredatorPrime] = None


def get_predator_prime() -> PredatorPrime:
    """Get the singleton PredatorPrime instance."""
    global _predator_prime
    if _predator_prime is None:
        _predator_prime = PredatorPrime()
    return _predator_prime
