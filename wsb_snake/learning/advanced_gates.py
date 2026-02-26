"""
Advanced Trading Gates - TIER 1, 2, 3 Intelligence Layers

This module implements sophisticated trading intelligence:
- TIER 1: Drawdown velocity circuit breaker, exposure persistence
- TIER 2: HYDRA boosting, regime-aware Kelly, test-time reasoning, trade approval gate
- TIER 3: Prompt evolution, BATS router, hierarchical memory, adaptive weighting

Based on IgorGanapolsky/trading multi-gate architecture and self-evolving-agents research.
"""

import threading
import time
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from pathlib import Path

from wsb_snake.utils.logger import get_logger
from wsb_snake.db.database import get_connection

logger = get_logger(__name__)


# ==============================================================================
# TIER 1: DRAWDOWN VELOCITY CIRCUIT BREAKER
# ==============================================================================

@dataclass
class DrawdownVelocityState:
    """Tracks rate of drawdown over time windows."""
    pnl_history: List[Tuple[datetime, float]] = field(default_factory=list)  # (timestamp, pnl)
    velocity_5min: float = 0.0  # $ lost per minute over 5min window
    velocity_15min: float = 0.0  # $ lost per minute over 15min window
    halt_triggered: bool = False
    halt_until: Optional[datetime] = None


class DrawdownVelocityMonitor:
    """
    TIER 1: Monitors the RATE of losses, not just total.

    Problem: Standard circuit breakers only trigger at absolute thresholds.
    Rapid consecutive losses (-$100 in 5 minutes) need immediate halt.

    Thresholds:
    - $50 loss in 5 minutes = half size
    - $100 loss in 5 minutes = halt 30 minutes
    - $150 loss in 15 minutes = halt 1 hour
    """

    # Velocity thresholds (dollars per minute)
    HALF_SIZE_VELOCITY = 10.0  # $10/min for 5 min = $50 loss
    HALT_VELOCITY_5MIN = 20.0  # $20/min for 5 min = $100 loss
    HALT_VELOCITY_15MIN = 10.0  # $10/min for 15 min = $150 loss

    def __init__(self):
        self._lock = threading.RLock()
        self._state = DrawdownVelocityState()
        self._init_db()
        logger.info("TIER_1: Drawdown velocity monitor initialized")

    def _init_db(self) -> None:
        """Initialize persistence table."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drawdown_velocity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    pnl REAL,
                    velocity_5min REAL,
                    velocity_15min REAL,
                    halt_triggered INTEGER
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"TIER_1: DB init failed - {e}")

    def record_pnl(self, pnl: float, trade_pnl: float = 0.0) -> None:
        """
        Record current P/L and update velocity calculations.

        Args:
            pnl: Current cumulative daily P/L
            trade_pnl: P/L from the most recent trade (for velocity calc)
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            self._state.pnl_history.append((now, pnl))

            # Keep only last 30 minutes of history
            cutoff = now - timedelta(minutes=30)
            self._state.pnl_history = [
                (t, p) for t, p in self._state.pnl_history if t > cutoff
            ]

            # Calculate velocities
            self._state.velocity_5min = self._calculate_velocity(minutes=5)
            self._state.velocity_15min = self._calculate_velocity(minutes=15)

            # Log if velocity is concerning
            if self._state.velocity_5min < -5:  # Losing $5/min
                logger.warning(
                    f"TIER_1: Drawdown velocity: ${self._state.velocity_5min:.1f}/min (5min), "
                    f"${self._state.velocity_15min:.1f}/min (15min)"
                )

            self._save_state()

    def _calculate_velocity(self, minutes: int) -> float:
        """Calculate P/L velocity over given window."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=minutes)

        relevant = [(t, p) for t, p in self._state.pnl_history if t > cutoff]
        if len(relevant) < 2:
            return 0.0

        # P/L change over window
        oldest = min(relevant, key=lambda x: x[0])
        newest = max(relevant, key=lambda x: x[0])

        time_diff = (newest[0] - oldest[0]).total_seconds() / 60  # minutes
        pnl_diff = newest[1] - oldest[1]

        if time_diff < 0.5:  # Less than 30 seconds
            return 0.0

        return pnl_diff / time_diff  # $/minute

    def check_velocity_halt(self) -> Tuple[bool, str, float]:
        """
        Check if velocity warrants halting.

        Returns:
            (should_halt, reason, recommended_size_multiplier)
            - multiplier 0.0 = halt, 0.5 = half size, 1.0 = normal
        """
        with self._lock:
            # Check if already in halt
            if self._state.halt_triggered and self._state.halt_until:
                if datetime.now(timezone.utc) < self._state.halt_until:
                    remaining = (self._state.halt_until - datetime.now(timezone.utc)).total_seconds() / 60
                    return True, f"VELOCITY HALT active - {remaining:.0f} min remaining", 0.0
                else:
                    self._state.halt_triggered = False
                    self._state.halt_until = None
                    logger.info("TIER_1: Velocity halt expired - trading resumed")

            # Check 5-minute velocity
            if self._state.velocity_5min <= -self.HALT_VELOCITY_5MIN:
                self._state.halt_triggered = True
                self._state.halt_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                reason = f"VELOCITY HALT: ${self._state.velocity_5min:.1f}/min (5min) - halting 30min"
                logger.warning(reason)
                return True, reason, 0.0

            # Check 15-minute velocity
            if self._state.velocity_15min <= -self.HALT_VELOCITY_15MIN:
                self._state.halt_triggered = True
                self._state.halt_until = datetime.now(timezone.utc) + timedelta(hours=1)
                reason = f"VELOCITY HALT: ${self._state.velocity_15min:.1f}/min (15min) - halting 1hr"
                logger.warning(reason)
                return True, reason, 0.0

            # Check half-size threshold
            if self._state.velocity_5min <= -self.HALF_SIZE_VELOCITY:
                reason = f"VELOCITY WARNING: ${self._state.velocity_5min:.1f}/min - half size"
                return False, reason, 0.5

            return False, "ok", 1.0

    def _save_state(self) -> None:
        """Persist current state."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO drawdown_velocity
                (timestamp, pnl, velocity_5min, velocity_15min, halt_triggered)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                self._state.pnl_history[-1][1] if self._state.pnl_history else 0,
                self._state.velocity_5min,
                self._state.velocity_15min,
                1 if self._state.halt_triggered else 0,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"TIER_1: State save failed - {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current velocity status."""
        with self._lock:
            return {
                "velocity_5min": self._state.velocity_5min,
                "velocity_15min": self._state.velocity_15min,
                "halt_triggered": self._state.halt_triggered,
                "halt_until": self._state.halt_until.isoformat() if self._state.halt_until else None,
                "history_points": len(self._state.pnl_history),
            }


# ==============================================================================
# TIER 2: HYDRA SIGNAL BOOSTING
# ==============================================================================

class HydraSignalBooster:
    """
    TIER 2: Boost conviction when HYDRA intelligence aligns with setup.

    HYDRA provides:
    - GEX regime (positive/negative/neutral)
    - Flow bias (bullish/bearish/neutral)
    - Dark pool levels
    - Institutional direction

    Boosting rules:
    - GEX + Flow align with direction: +15% conviction
    - GEX aligns only: +8% conviction
    - Flow aligns only: +5% conviction
    - Both contra: -20% conviction (potential ABORT)
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._cache: Dict[str, Tuple[datetime, Dict]] = {}  # ticker -> (timestamp, data)
        self._cache_ttl = 60  # seconds
        logger.info("TIER_2: HYDRA signal booster initialized")

    def get_boost(
        self,
        ticker: str,
        direction: str,  # "long" or "short"
        base_confidence: float,
    ) -> Tuple[float, str, Dict]:
        """
        Calculate confidence boost from HYDRA alignment.

        Returns:
            (boosted_confidence, reason, hydra_data)
        """
        hydra_data = self._get_hydra_intel(ticker)
        if not hydra_data.get("available"):
            return base_confidence, "No HYDRA data", {}

        boost = 0.0
        reasons = []

        is_long = direction == "long"

        # GEX alignment
        gex_regime = hydra_data.get("gex_regime", "neutral")
        gex_aligns = (
            (is_long and gex_regime in ["positive", "bullish"]) or
            (not is_long and gex_regime in ["negative", "bearish"])
        )
        gex_contra = (
            (is_long and gex_regime in ["negative", "bearish"]) or
            (not is_long and gex_regime in ["positive", "bullish"])
        )

        # Flow alignment
        flow_bias = hydra_data.get("flow_bias", "neutral")
        flow_aligns = (
            (is_long and flow_bias in ["bullish", "call_heavy"]) or
            (not is_long and flow_bias in ["bearish", "put_heavy"])
        )
        flow_contra = (
            (is_long and flow_bias in ["bearish", "put_heavy"]) or
            (not is_long and flow_bias in ["bullish", "call_heavy"])
        )

        # Calculate boost
        if gex_aligns and flow_aligns:
            boost = 15.0
            reasons.append(f"HYDRA ALIGNED: GEX={gex_regime}, Flow={flow_bias}")
        elif gex_aligns:
            boost = 8.0
            reasons.append(f"GEX aligned: {gex_regime}")
        elif flow_aligns:
            boost = 5.0
            reasons.append(f"Flow aligned: {flow_bias}")
        elif gex_contra and flow_contra:
            boost = -20.0
            reasons.append(f"HYDRA CONTRA: GEX={gex_regime}, Flow={flow_bias} - CAUTION")
        elif gex_contra:
            boost = -10.0
            reasons.append(f"GEX contra: {gex_regime}")
        elif flow_contra:
            boost = -5.0
            reasons.append(f"Flow contra: {flow_bias}")

        boosted = max(0, min(100, base_confidence + boost))
        reason = "; ".join(reasons) if reasons else "HYDRA neutral"

        logger.info(f"TIER_2: {ticker} {direction} HYDRA boost: {boost:+.0f}% -> {boosted:.0f}%")
        return boosted, reason, hydra_data

    def _get_hydra_intel(self, ticker: str) -> Dict:
        """Get HYDRA intelligence for ticker."""
        # Check cache
        if ticker in self._cache:
            ts, data = self._cache[ticker]
            if (datetime.now(timezone.utc) - ts).total_seconds() < self._cache_ttl:
                return data

        try:
            from wsb_snake.collectors.hydra_bridge import get_hydra_bridge
            hydra = get_hydra_bridge()
            intel = hydra.get_intel()

            data = {
                "available": True,
                "gex_regime": intel.gex_regime if hasattr(intel, 'gex_regime') else intel.regime,
                "flow_bias": intel.flow_bias if hasattr(intel, 'flow_bias') else "neutral",
                "direction": intel.direction if hasattr(intel, 'direction') else "NEUTRAL",
                "conviction": intel.conviction if hasattr(intel, 'conviction') else 50,
            }

            self._cache[ticker] = (datetime.now(timezone.utc), data)
            return data

        except Exception as e:
            logger.debug(f"TIER_2: HYDRA fetch failed - {e}")
            return {"available": False}


# ==============================================================================
# TIER 2: REGIME-AWARE KELLY CALIBRATION
# ==============================================================================

class RegimeAwareKelly:
    """
    TIER 2: Adjust Kelly fraction based on VIX regime.

    Standard Kelly is too aggressive in high volatility.

    Regime adjustments:
    - VIX < 15: 0.6x Kelly (low vol, markets complacent - watch for shocks)
    - VIX 15-20: 1.0x Kelly (normal conditions)
    - VIX 20-25: 0.75x Kelly (elevated vol)
    - VIX 25-30: 0.5x Kelly (high vol)
    - VIX > 30: 0.25x Kelly (crisis mode - capital preservation)
    """

    REGIME_MULTIPLIERS = {
        "low_vol": 0.6,      # VIX < 15
        "normal": 1.0,       # VIX 15-20
        "elevated": 0.75,    # VIX 20-25
        "high": 0.5,         # VIX 25-30
        "crisis": 0.25,      # VIX > 30
    }

    def __init__(self):
        self._last_vix: Optional[float] = None
        self._last_regime: str = "normal"
        logger.info("TIER_2: Regime-aware Kelly initialized")

    def get_kelly_multiplier(self) -> Tuple[float, str, float]:
        """
        Get Kelly multiplier based on current VIX regime.

        Returns:
            (multiplier, regime_name, vix_value)
        """
        vix = self._get_vix()
        self._last_vix = vix

        if vix < 15:
            regime = "low_vol"
        elif vix < 20:
            regime = "normal"
        elif vix < 25:
            regime = "elevated"
        elif vix < 30:
            regime = "high"
        else:
            regime = "crisis"

        self._last_regime = regime
        multiplier = self.REGIME_MULTIPLIERS[regime]

        logger.info(f"TIER_2: VIX={vix:.1f} -> {regime} regime -> {multiplier:.2f}x Kelly")
        return multiplier, regime, vix

    def adjust_kelly_fraction(
        self,
        raw_kelly: float,
        win_probability: float,
    ) -> float:
        """
        Apply regime adjustment to raw Kelly fraction.

        Args:
            raw_kelly: Raw half-Kelly fraction from calculation
            win_probability: Win probability (for additional safety)

        Returns:
            Adjusted Kelly fraction
        """
        multiplier, regime, vix = self.get_kelly_multiplier()

        # Additional safety: reduce more if win probability is low
        if win_probability < 0.5:
            multiplier *= 0.5  # Extra conservative
            logger.warning(f"TIER_2: Low win prob {win_probability:.0%} - extra Kelly reduction")

        adjusted = raw_kelly * multiplier

        # Hard cap at 15% regardless of calculation
        adjusted = min(adjusted, 0.15)

        logger.info(
            f"TIER_2: Kelly adjusted: {raw_kelly:.4f} * {multiplier:.2f} = {adjusted:.4f} "
            f"(VIX={vix:.1f}, regime={regime})"
        )
        return adjusted

    def _get_vix(self) -> float:
        """Get current VIX level."""
        try:
            from wsb_snake.collectors.vix_structure import vix_structure
            signal = vix_structure.get_trading_signal()
            return signal.get("vix", 20.0)
        except Exception:
            return 20.0  # Default if unavailable


# ==============================================================================
# TIER 2: TIME-TO-EXPIRY STOP ADJUSTMENT
# ==============================================================================

class ExpiryStopAdjuster:
    """
    TIER 2: Tighten stops as 0DTE options approach expiry.

    Theta decay accelerates exponentially near expiry.
    A winning position can become a loser in minutes.

    Stop adjustments (base stop = 7%):
    - 4+ hours to expiry: 7% (normal)
    - 2-4 hours: 6% (slightly tighter)
    - 1-2 hours: 5% (tighter)
    - 30min-1hr: 4% (much tighter)
    - <30min: 3% (extremely tight - theta savage)
    """

    def __init__(self):
        logger.info("TIER_2: Expiry stop adjuster initialized")

    def adjust_stop(
        self,
        base_stop_pct: float,
        hours_to_expiry: float,
    ) -> Tuple[float, str]:
        """
        Adjust stop percentage based on time to expiry.

        Args:
            base_stop_pct: Base stop loss percentage (e.g., 0.07 for 7%)
            hours_to_expiry: Hours until option expiry

        Returns:
            (adjusted_stop_pct, reason)
        """
        if hours_to_expiry >= 4:
            multiplier = 1.0
            reason = "Normal (4h+)"
        elif hours_to_expiry >= 2:
            multiplier = 0.85
            reason = "Slightly tighter (2-4h)"
        elif hours_to_expiry >= 1:
            multiplier = 0.70
            reason = "Tighter (1-2h)"
        elif hours_to_expiry >= 0.5:
            multiplier = 0.55
            reason = "Much tighter (30min-1h)"
        else:
            multiplier = 0.40
            reason = "Theta savage (<30min)"

        adjusted = base_stop_pct * multiplier

        # Minimum stop of 2% to avoid noise exits
        adjusted = max(adjusted, 0.02)

        logger.info(
            f"TIER_2: Stop adjusted: {base_stop_pct:.1%} * {multiplier:.2f} = {adjusted:.1%} "
            f"({hours_to_expiry:.1f}h to expiry - {reason})"
        )
        return adjusted, reason

    def should_close_for_theta(
        self,
        hours_to_expiry: float,
        current_pnl_pct: float,
    ) -> Tuple[bool, str]:
        """
        Check if position should close due to theta risk.

        Args:
            hours_to_expiry: Hours until expiry
            current_pnl_pct: Current P/L percentage

        Returns:
            (should_close, reason)
        """
        # Close winners early if near expiry
        if hours_to_expiry < 0.5 and current_pnl_pct > 0.03:  # 3%+ winner
            return True, f"THETA CLOSE: +{current_pnl_pct:.1%} win with <30min to expiry"

        # Close losers to prevent total wipeout
        if hours_to_expiry < 0.25 and current_pnl_pct < -0.05:  # 5%+ loser
            return True, f"THETA CLOSE: {current_pnl_pct:.1%} loss with <15min - cut losses"

        # Force close at 5 minutes regardless
        if hours_to_expiry < 0.083:  # ~5 minutes
            return True, "THETA CLOSE: <5min to expiry - mandatory close"

        return False, "ok"


# ==============================================================================
# TIER 2: TEST-TIME REASONING (TREE OF THOUGHTS)
# ==============================================================================

@dataclass
class ThoughtPath:
    """A reasoning path through the decision tree."""
    scenario: str  # "bullish", "bearish", "neutral"
    assumptions: List[str]
    predicted_outcome: str
    confidence: float  # 0-100
    risk_factors: List[str]
    expected_pnl: float  # Expected % move
    reasoning: str


class TestTimeReasoner:
    """
    TIER 2: Tree of Thoughts reasoning before committing to trade.

    Instead of single-path reasoning, explore 3 scenarios:
    1. Bullish: What if the trade works as expected?
    2. Bearish: What if the market moves against us?
    3. Neutral: What if nothing happens (theta decay)?

    Only execute if bullish path is significantly more likely.
    """

    def __init__(self):
        self._lock = threading.RLock()
        logger.info("TIER_2: Test-time reasoner initialized")

    def evaluate_paths(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        target_pct: float,
        stop_pct: float,
        hours_to_expiry: float,
        context: Dict[str, Any],
    ) -> Tuple[bool, List[ThoughtPath], str]:
        """
        Evaluate all reasoning paths before trade decision.

        Returns:
            (should_execute, paths, final_recommendation)
        """
        paths = []

        # Path 1: Bullish scenario
        bullish_path = self._reason_bullish(
            ticker, direction, entry_price, target_pct, context
        )
        paths.append(bullish_path)

        # Path 2: Bearish scenario
        bearish_path = self._reason_bearish(
            ticker, direction, entry_price, stop_pct, context
        )
        paths.append(bearish_path)

        # Path 3: Neutral scenario (theta decay)
        neutral_path = self._reason_neutral(
            ticker, direction, entry_price, hours_to_expiry, context
        )
        paths.append(neutral_path)

        # Weighted decision
        bull_weight = bullish_path.confidence / 100
        bear_weight = bearish_path.confidence / 100
        neutral_weight = neutral_path.confidence / 100

        # Expected value calculation
        ev = (
            bull_weight * bullish_path.expected_pnl +
            bear_weight * bearish_path.expected_pnl +
            neutral_weight * neutral_path.expected_pnl
        )

        # Decision logic
        should_execute = False
        recommendation = "SKIP"

        if bull_weight > bear_weight + neutral_weight and bull_weight > 0.5:
            # Bullish dominates
            if ev > 0.02:  # Need +2% expected value
                should_execute = True
                recommendation = f"EXECUTE: Bull={bull_weight:.0%}, EV={ev:+.1%}"
            else:
                recommendation = f"SKIP: Low EV ({ev:+.1%}) despite bullish lean"
        elif bear_weight > bull_weight:
            recommendation = f"ABORT: Bear scenario dominates ({bear_weight:.0%})"
        else:
            recommendation = f"SKIP: No clear edge (Bull={bull_weight:.0%}, Bear={bear_weight:.0%})"

        logger.info(f"TIER_2: ToT decision for {ticker}: {recommendation}")
        return should_execute, paths, recommendation

    def _reason_bullish(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        target_pct: float,
        context: Dict,
    ) -> ThoughtPath:
        """Generate bullish scenario reasoning."""
        # Extract context factors
        ai_confidence = context.get("ai_confidence", 50)
        hydra_aligned = context.get("hydra_aligned", False)
        pattern_wr = context.get("pattern_win_rate", 0.5)

        # Build confidence for bullish case
        confidence = 40  # Base
        assumptions = []
        risk_factors = []

        if ai_confidence > 70:
            confidence += 15
            assumptions.append(f"AI confirms setup ({ai_confidence:.0f}%)")

        if hydra_aligned:
            confidence += 10
            assumptions.append("HYDRA intelligence aligned")

        if pattern_wr > 0.6:
            confidence += 10
            assumptions.append(f"Pattern has {pattern_wr:.0%} historical win rate")

        # Risk factors that reduce bullish confidence
        if context.get("vix", 20) > 25:
            confidence -= 10
            risk_factors.append("High VIX environment")

        if context.get("near_expiry", False):
            confidence -= 5
            risk_factors.append("Near expiry theta risk")

        confidence = max(10, min(90, confidence))

        return ThoughtPath(
            scenario="bullish",
            assumptions=assumptions,
            predicted_outcome=f"Price moves {direction} to target (+{target_pct:.1%})",
            confidence=confidence,
            risk_factors=risk_factors,
            expected_pnl=target_pct,
            reasoning=f"If momentum continues and signals hold, expect +{target_pct:.1%} gain"
        )

    def _reason_bearish(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        stop_pct: float,
        context: Dict,
    ) -> ThoughtPath:
        """Generate bearish scenario reasoning."""
        # Bearish scenarios
        confidence = 30  # Base bear case probability
        assumptions = []
        risk_factors = []

        # Factors that increase bear probability
        if context.get("trap_detected"):
            confidence += 20
            assumptions.append(f"Trap detected: {context['trap_detected']}")

        if context.get("contra_flow", False):
            confidence += 15
            assumptions.append("Order flow against direction")

        if context.get("weak_volume", False):
            confidence += 10
            assumptions.append("Weak volume confirmation")

        # Market conditions
        if context.get("vix", 20) > 30:
            confidence += 10
            risk_factors.append("Crisis VIX - expect whipsaws")

        confidence = max(10, min(80, confidence))

        contra_direction = "short" if direction == "long" else "long"

        return ThoughtPath(
            scenario="bearish",
            assumptions=assumptions,
            predicted_outcome=f"Price reverses, hits stop ({stop_pct:.1%} loss)",
            confidence=confidence,
            risk_factors=risk_factors,
            expected_pnl=-stop_pct,
            reasoning=f"If setup fails, expect reversal to stop for -{stop_pct:.1%}"
        )

    def _reason_neutral(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        hours_to_expiry: float,
        context: Dict,
    ) -> ThoughtPath:
        """Generate neutral/chop scenario reasoning."""
        # Neutral (theta decay) probability
        confidence = 30  # Base
        assumptions = []
        risk_factors = []

        # Factors that increase chop probability
        if context.get("is_chop", False):
            confidence += 20
            assumptions.append("Market in chop regime")

        if context.get("low_atr", False):
            confidence += 10
            assumptions.append("Low ATR - limited movement expected")

        if hours_to_expiry < 2:
            confidence += 10
            assumptions.append(f"Only {hours_to_expiry:.1f}h to expiry")

        # Theta decay estimate
        if hours_to_expiry < 1:
            theta_loss = -0.05  # 5% loss to theta in last hour
        elif hours_to_expiry < 2:
            theta_loss = -0.03
        else:
            theta_loss = -0.02

        risk_factors.append(f"Theta decay: ~{theta_loss:.0%} if price stagnates")

        confidence = max(10, min(60, confidence))

        return ThoughtPath(
            scenario="neutral",
            assumptions=assumptions,
            predicted_outcome="Price consolidates, theta eats premium",
            confidence=confidence,
            risk_factors=risk_factors,
            expected_pnl=theta_loss,
            reasoning=f"If market chops, expect {theta_loss:.0%} theta decay"
        )


# ==============================================================================
# TIER 2: TRADE APPROVAL GATE
# ==============================================================================

@dataclass
class ApprovalDecision:
    """Final approval decision from aggregation gate."""
    approved: bool
    confidence: float  # Final aggregated confidence
    size_multiplier: float  # 0.0 to 1.5 based on conviction
    reason: str
    checks_passed: List[str]
    checks_failed: List[str]
    recommendations: List[str]


class TradeApprovalGate:
    """
    TIER 2: Final aggregation gate before execution.

    Aggregates all signals:
    - AI confidence (Predator Prime)
    - Gate 15 (Bull/Bear Debate)
    - Gate 35 (Introspection)
    - HYDRA boost
    - Test-time reasoning
    - Risk governor
    - Velocity monitor

    Returns final approval with size multiplier.
    """

    # Minimum thresholds for approval
    MIN_CONFIDENCE = 65
    MIN_SIZE_MULTIPLIER = 0.5
    REQUIRED_CHECKS = ["ai", "risk_governor"]  # Must pass these

    def __init__(self):
        self._lock = threading.RLock()
        logger.info("TIER_2: Trade approval gate initialized")

    def evaluate(
        self,
        ticker: str,
        direction: str,
        base_confidence: float,
        checks: Dict[str, Any],
    ) -> ApprovalDecision:
        """
        Evaluate all checks and return final approval decision.

        Args:
            ticker: Symbol
            direction: "long" or "short"
            base_confidence: Initial confidence from pattern detection
            checks: Dictionary of check results:
                - ai: {"confirmed": bool, "confidence": float}
                - debate: {"action": str, "multiplier": float}
                - introspection: {"health": str, "multiplier": float}
                - hydra: {"boost": float}
                - tot: {"should_execute": bool, "ev": float}
                - risk: {"can_trade": bool, "reason": str}
                - velocity: {"halt": bool, "multiplier": float}

        Returns:
            ApprovalDecision with final verdict
        """
        passed = []
        failed = []
        recommendations = []

        # Start with base confidence
        final_confidence = base_confidence
        size_multiplier = 1.0

        # === Check AI confirmation ===
        ai = checks.get("ai", {})
        if ai.get("confirmed", False):
            passed.append("AI confirmed")
            final_confidence = max(final_confidence, ai.get("confidence", 0))
        elif ai.get("confidence", 0) < 50:
            failed.append("AI not confirmed")
        else:
            passed.append("AI neutral")

        # === Check Debate (Gate 15) ===
        debate = checks.get("debate", {})
        if debate.get("action") == "ABORT":
            failed.append(f"Debate ABORT ({debate.get('reason', 'bear dominates')})")
            size_multiplier = 0.0
        elif debate.get("action") == "REDUCE_SIZE":
            passed.append("Debate REDUCE_SIZE")
            size_multiplier *= debate.get("multiplier", 0.7)
        else:
            passed.append("Debate STRIKE")

        # === Check Introspection (Gate 35) ===
        intro = checks.get("introspection", {})
        health = intro.get("health", "NORMAL")
        if health in ["FROZEN", "COLD"]:
            failed.append(f"Pattern health: {health}")
            size_multiplier = 0.0
        elif health == "COOLING":
            passed.append(f"Pattern COOLING")
            size_multiplier *= intro.get("multiplier", 0.75)
        elif health == "HOT":
            passed.append("Pattern HOT")
            final_confidence += 3
        else:
            passed.append("Pattern NORMAL")

        # === Apply HYDRA boost ===
        hydra = checks.get("hydra", {})
        boost = hydra.get("boost", 0)
        final_confidence += boost
        if boost > 5:
            passed.append(f"HYDRA boost +{boost:.0f}%")
        elif boost < -5:
            failed.append(f"HYDRA contra {boost:.0f}%")

        # === Check Test-Time Reasoning ===
        tot = checks.get("tot", {})
        if tot.get("should_execute") is False:
            ev = tot.get("ev", 0)
            if ev < 0:
                failed.append(f"ToT negative EV ({ev:.1%})")
            else:
                recommendations.append(f"ToT suggests caution (EV={ev:+.1%})")
        elif tot.get("should_execute"):
            passed.append("ToT approves")

        # === Check Risk Governor ===
        risk = checks.get("risk", {})
        if not risk.get("can_trade", True):
            failed.append(f"Risk: {risk.get('reason', 'blocked')}")
            size_multiplier = 0.0
        else:
            passed.append("Risk approved")

        # === Check Velocity Monitor ===
        velocity = checks.get("velocity", {})
        if velocity.get("halt", False):
            failed.append("Velocity halt active")
            size_multiplier = 0.0
        elif velocity.get("multiplier", 1.0) < 1.0:
            passed.append("Velocity warning - reduced size")
            size_multiplier *= velocity.get("multiplier", 1.0)
        else:
            passed.append("Velocity ok")

        # === Final decision ===
        # Check required checks
        required_failed = []
        for req in self.REQUIRED_CHECKS:
            if req == "ai" and not ai.get("confirmed") and ai.get("confidence", 0) < 60:
                required_failed.append("ai")
            elif req == "risk_governor" and not risk.get("can_trade", True):
                required_failed.append("risk_governor")

        # Clamp confidence
        final_confidence = max(0, min(100, final_confidence))

        # Decision
        if required_failed:
            approved = False
            reason = f"Required checks failed: {', '.join(required_failed)}"
        elif size_multiplier <= 0:
            approved = False
            reason = f"Size multiplier zero: {', '.join(failed)}"
        elif final_confidence < self.MIN_CONFIDENCE:
            approved = False
            reason = f"Confidence {final_confidence:.0f}% below {self.MIN_CONFIDENCE}%"
        elif len(failed) > len(passed):
            approved = False
            reason = f"More checks failed ({len(failed)}) than passed ({len(passed)})"
        else:
            approved = True
            reason = f"Approved: {final_confidence:.0f}% confidence, {size_multiplier:.1f}x size"

        logger.info(f"TIER_2: Approval gate for {ticker}: {reason}")

        return ApprovalDecision(
            approved=approved,
            confidence=final_confidence,
            size_multiplier=size_multiplier,
            reason=reason,
            checks_passed=passed,
            checks_failed=failed,
            recommendations=recommendations,
        )


# ==============================================================================
# TIER 3: PROMPT EVOLUTION ENGINE
# ==============================================================================

@dataclass
class PromptVariant:
    """A prompt variant being tested."""
    variant_id: str
    prompt_template: str
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PromptEvolutionEngine:
    """
    TIER 3: Self-improving prompts based on trade outcomes.

    Uses Thompson Sampling to select between prompt variants.
    Winning prompts get used more, losers get evolved or dropped.

    Evolution strategies:
    1. Mutation: Tweak successful prompts
    2. Crossover: Combine elements from multiple winners
    3. Extinction: Drop consistently failing prompts
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._variants: Dict[str, Dict[str, PromptVariant]] = {}  # task -> {variant_id -> variant}
        self._init_default_prompts()
        logger.info("TIER_3: Prompt evolution engine initialized")

    def _init_default_prompts(self) -> None:
        """Initialize default prompt variants."""
        # Chart analysis prompts
        self._variants["chart_analysis"] = {
            "v1_standard": PromptVariant(
                variant_id="v1_standard",
                prompt_template="Analyze this chart. What is the likely direction? Confidence 0-100."
            ),
            "v2_detailed": PromptVariant(
                variant_id="v2_detailed",
                prompt_template="Analyze this {ticker} chart in detail. Consider: 1) trend direction, 2) support/resistance, 3) volume, 4) momentum indicators. Rate confidence 0-100 and explain."
            ),
            "v3_adversarial": PromptVariant(
                variant_id="v3_adversarial",
                prompt_template="Play devil's advocate on this {ticker} chart. What could go wrong? Then give probability of success 0-100."
            ),
        }

        # Trade validation prompts
        self._variants["trade_validation"] = {
            "v1_simple": PromptVariant(
                variant_id="v1_simple",
                prompt_template="Should I {direction} {ticker}? Yes/No and confidence."
            ),
            "v2_checklist": PromptVariant(
                variant_id="v2_checklist",
                prompt_template="Checklist for {ticker} {direction}: 1) Trend alignment? 2) Volume confirms? 3) Risk/reward > 2:1? 4) No traps? Score 0-100."
            ),
        }

    def select_prompt(self, task: str) -> Tuple[str, str]:
        """
        Select prompt using Thompson Sampling.

        Returns:
            (variant_id, prompt_template)
        """
        with self._lock:
            variants = self._variants.get(task, {})
            if not variants:
                return "default", f"Analyze {task}"

            # Thompson Sampling: sample from Beta distribution for each variant
            samples = {}
            for vid, var in variants.items():
                alpha = var.wins + 1  # Prior of 1
                beta = var.losses + 1
                samples[vid] = random.betavariate(alpha, beta)

            # Select highest sample
            best_id = max(samples, key=samples.get)
            best_variant = variants[best_id]

            # Update last used
            best_variant.last_used = datetime.now(timezone.utc)

            logger.debug(f"TIER_3: Selected prompt {best_id} for {task} (win_rate={best_variant.wins/(best_variant.wins+best_variant.losses+1):.0%})")
            return best_id, best_variant.prompt_template

    def record_outcome(
        self,
        task: str,
        variant_id: str,
        outcome: str,  # "win" or "loss"
        pnl: float = 0.0,
    ) -> None:
        """Record outcome for a prompt variant."""
        with self._lock:
            if task not in self._variants or variant_id not in self._variants[task]:
                return

            var = self._variants[task][variant_id]
            if outcome == "win":
                var.wins += 1
            else:
                var.losses += 1
            var.total_pnl += pnl

            logger.info(
                f"TIER_3: Prompt {variant_id} outcome: {outcome} "
                f"(now {var.wins}W/{var.losses}L)"
            )

            # Check if evolution is needed
            self._maybe_evolve(task)

    def _maybe_evolve(self, task: str) -> None:
        """Check if we should evolve prompts."""
        variants = self._variants.get(task, {})

        for vid, var in list(variants.items()):
            total = var.wins + var.losses
            if total < 10:
                continue  # Need minimum samples

            win_rate = var.wins / total

            # Extinction: drop variants with <30% win rate
            if win_rate < 0.30 and len(variants) > 2:
                logger.warning(f"TIER_3: Dropping prompt {vid} (win_rate={win_rate:.0%})")
                del variants[vid]

            # Mutation: clone and tweak successful variants
            elif win_rate > 0.65 and total >= 20:
                # Create mutant
                mutant_id = f"{vid}_mut_{int(time.time())}"
                mutant = PromptVariant(
                    variant_id=mutant_id,
                    prompt_template=self._mutate_prompt(var.prompt_template),
                )
                variants[mutant_id] = mutant
                logger.info(f"TIER_3: Created mutant prompt {mutant_id} from {vid}")

    def _mutate_prompt(self, template: str) -> str:
        """Mutate a prompt template."""
        mutations = [
            ("Analyze", "Carefully analyze"),
            ("confidence", "probability of success"),
            ("direction", "trend and momentum direction"),
            ("0-100", "from 0 (certain fail) to 100 (certain success)"),
            ("explain", "explain your reasoning step by step"),
        ]

        result = template
        # Apply random mutation
        for old, new in mutations:
            if old in result and random.random() > 0.5:
                result = result.replace(old, new, 1)
                break

        return result


# ==============================================================================
# TIER 3: BATS MODEL ROUTER (Budget-Aware Model Selection)
# ==============================================================================

class BATSModelRouter:
    """
    TIER 3: Budget-aware model selection.

    Routes requests to appropriate model based on:
    - Task complexity
    - Time sensitivity
    - Budget remaining
    - Model strengths

    Model tiers:
    - Fast: Haiku (~$0.001/request) - Simple classifications
    - Standard: Sonnet (~$0.01/request) - Normal analysis
    - Advanced: Opus (~$0.1/request) - Complex reasoning
    """

    # Cost estimates per 1000 tokens (input+output combined)
    COSTS = {
        "haiku": 0.001,
        "sonnet": 0.01,
        "opus": 0.10,
    }

    # Task -> recommended model
    TASK_ROUTING = {
        "quick_classification": "haiku",
        "pattern_recognition": "haiku",
        "chart_analysis": "sonnet",
        "trade_validation": "sonnet",
        "complex_reasoning": "opus",
        "strategy_planning": "opus",
    }

    def __init__(self, daily_budget: float = 10.0):
        self._lock = threading.RLock()
        self._daily_budget = daily_budget
        self._daily_spent = 0.0
        self._last_reset: Optional[str] = None
        self._usage: Dict[str, int] = {"haiku": 0, "sonnet": 0, "opus": 0}
        logger.info(f"TIER_3: BATS router initialized (budget: ${daily_budget:.2f}/day)")

    def select_model(
        self,
        task: str,
        urgency: str = "normal",  # "low", "normal", "high"
        complexity: str = "normal",  # "simple", "normal", "complex"
    ) -> Tuple[str, str]:
        """
        Select appropriate model for task.

        Returns:
            (model_id, reason)
        """
        self._reset_if_new_day()

        with self._lock:
            # Base recommendation from task
            base_model = self.TASK_ROUTING.get(task, "sonnet")

            # Adjust for urgency
            if urgency == "high":
                # Use fastest available
                base_model = "haiku"

            # Adjust for complexity
            if complexity == "complex" and base_model != "opus":
                base_model = "sonnet"  # At least use sonnet
            elif complexity == "simple":
                base_model = "haiku"

            # Check budget
            remaining = self._daily_budget - self._daily_spent
            estimated_cost = self.COSTS[base_model] * 2  # Assume ~2k tokens

            if estimated_cost > remaining:
                # Downgrade to stay in budget
                if base_model == "opus":
                    base_model = "sonnet"
                    reason = "Downgraded to sonnet (budget)"
                elif base_model == "sonnet":
                    base_model = "haiku"
                    reason = "Downgraded to haiku (budget)"
                else:
                    reason = f"Using haiku (budget: ${remaining:.2f} remaining)"
            else:
                reason = f"Selected {base_model} for {task}"

            self._usage[base_model] += 1
            logger.debug(f"TIER_3: BATS selected {base_model}: {reason}")
            return base_model, reason

    def record_usage(self, model: str, tokens: int, cost: float) -> None:
        """Record actual usage for budget tracking."""
        with self._lock:
            self._daily_spent += cost
            logger.debug(f"TIER_3: BATS usage: {model} {tokens} tokens (${cost:.4f}). Day total: ${self._daily_spent:.2f}")

    def _reset_if_new_day(self) -> None:
        """Reset daily counters if new day."""
        from datetime import date
        today = date.today().isoformat()

        with self._lock:
            if self._last_reset != today:
                self._daily_spent = 0.0
                self._usage = {"haiku": 0, "sonnet": 0, "opus": 0}
                self._last_reset = today
                logger.info(f"TIER_3: BATS daily reset (new day: {today})")

    def get_status(self) -> Dict:
        """Get current budget status."""
        with self._lock:
            return {
                "daily_budget": self._daily_budget,
                "daily_spent": self._daily_spent,
                "remaining": self._daily_budget - self._daily_spent,
                "usage": self._usage.copy(),
            }


# ==============================================================================
# TIER 3: HIERARCHICAL MEMORY
# ==============================================================================

@dataclass
class MemoryItem:
    """A memory item with importance and decay."""
    item_id: str
    content: str
    importance: float  # 0-1
    created: datetime
    last_accessed: datetime
    access_count: int = 0
    category: str = "general"
    metadata: Dict = field(default_factory=dict)


class HierarchicalMemory:
    """
    TIER 3: Short/medium/long-term memory with graduated importance decay.

    Memory tiers:
    - Short-term (STM): Last hour, full detail, 100 items max
    - Medium-term (MTM): Last day, summarized, 500 items max
    - Long-term (LTM): Persistent, highly compressed, unlimited

    Importance factors:
    - Trade outcome (wins more important)
    - Recency
    - Access frequency
    - Manual tagging
    """

    STM_MAX = 100
    MTM_MAX = 500
    STM_HOURS = 1
    MTM_HOURS = 24

    def __init__(self):
        self._lock = threading.RLock()
        self._stm: List[MemoryItem] = []  # Short-term
        self._mtm: List[MemoryItem] = []  # Medium-term
        self._ltm: List[MemoryItem] = []  # Long-term
        self._init_db()
        self._load_ltm()
        logger.info("TIER_3: Hierarchical memory initialized")

    def _init_db(self) -> None:
        """Initialize persistence."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hierarchical_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT UNIQUE,
                    content TEXT,
                    importance REAL,
                    created TEXT,
                    last_accessed TEXT,
                    access_count INTEGER,
                    category TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"TIER_3: Memory DB init failed - {e}")

    def _load_ltm(self) -> None:
        """Load long-term memory from DB."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT item_id, content, importance, created, last_accessed, access_count, category, metadata
                FROM hierarchical_memory
                ORDER BY importance DESC
                LIMIT 1000
            """)
            rows = cursor.fetchall()
            conn.close()

            for row in rows:
                self._ltm.append(MemoryItem(
                    item_id=row[0],
                    content=row[1],
                    importance=row[2],
                    created=datetime.fromisoformat(row[3]) if row[3] else datetime.now(timezone.utc),
                    last_accessed=datetime.fromisoformat(row[4]) if row[4] else datetime.now(timezone.utc),
                    access_count=row[5],
                    category=row[6],
                    metadata=json.loads(row[7]) if row[7] else {},
                ))

            logger.info(f"TIER_3: Loaded {len(self._ltm)} items from LTM")
        except Exception as e:
            logger.warning(f"TIER_3: LTM load failed - {e}")

    def store(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Store new memory item.

        Returns item_id.
        """
        import uuid
        item_id = f"mem_{uuid.uuid4().hex[:8]}"

        item = MemoryItem(
            item_id=item_id,
            content=content,
            importance=importance,
            created=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            category=category,
            metadata=metadata or {},
        )

        with self._lock:
            self._stm.append(item)

            # Enforce STM limit
            if len(self._stm) > self.STM_MAX:
                self._consolidate()

        logger.debug(f"TIER_3: Stored memory {item_id} (importance={importance:.2f})")
        return item_id

    def recall(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> List[MemoryItem]:
        """
        Recall relevant memories.

        Simple keyword matching - could be upgraded to embeddings.
        """
        with self._lock:
            all_items = self._stm + self._mtm + self._ltm

            # Filter by category if specified
            if category:
                all_items = [i for i in all_items if i.category == category]

            # Simple relevance: keyword match + importance
            query_words = set(query.lower().split())
            scored = []

            for item in all_items:
                content_words = set(item.content.lower().split())
                overlap = len(query_words & content_words)
                score = overlap * 0.5 + item.importance * 0.3 + (item.access_count * 0.01)
                scored.append((score, item))

            # Sort by score
            scored.sort(key=lambda x: x[0], reverse=True)

            # Update access counts
            results = []
            for score, item in scored[:limit]:
                item.last_accessed = datetime.now(timezone.utc)
                item.access_count += 1
                results.append(item)

            return results

    def _consolidate(self) -> None:
        """Consolidate STM to MTM, MTM to LTM."""
        now = datetime.now(timezone.utc)

        # STM -> MTM (items older than 1 hour)
        stm_cutoff = now - timedelta(hours=self.STM_HOURS)
        to_mtm = [i for i in self._stm if i.created < stm_cutoff]
        self._stm = [i for i in self._stm if i.created >= stm_cutoff]

        for item in to_mtm:
            # Summarize/compress before moving to MTM
            if len(item.content) > 200:
                item.content = item.content[:200] + "..."
            self._mtm.append(item)

        # MTM -> LTM (items older than 24 hours with sufficient importance)
        mtm_cutoff = now - timedelta(hours=self.MTM_HOURS)
        to_ltm = [i for i in self._mtm if i.created < mtm_cutoff and i.importance > 0.3]
        self._mtm = [i for i in self._mtm if i.created >= mtm_cutoff or i.importance <= 0.3]

        for item in to_ltm:
            self._ltm.append(item)
            self._persist_ltm_item(item)

        # Enforce MTM limit
        if len(self._mtm) > self.MTM_MAX:
            # Drop lowest importance items
            self._mtm.sort(key=lambda x: x.importance, reverse=True)
            self._mtm = self._mtm[:self.MTM_MAX]

        logger.debug(f"TIER_3: Consolidated memory: STM={len(self._stm)}, MTM={len(self._mtm)}, LTM={len(self._ltm)}")

    def _persist_ltm_item(self, item: MemoryItem) -> None:
        """Persist LTM item to database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO hierarchical_memory
                (item_id, content, importance, created, last_accessed, access_count, category, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.item_id,
                item.content,
                item.importance,
                item.created.isoformat(),
                item.last_accessed.isoformat(),
                item.access_count,
                item.category,
                json.dumps(item.metadata),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"TIER_3: LTM persist failed - {e}")

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        with self._lock:
            return {
                "stm_count": len(self._stm),
                "mtm_count": len(self._mtm),
                "ltm_count": len(self._ltm),
                "total": len(self._stm) + len(self._mtm) + len(self._ltm),
            }


# ==============================================================================
# TIER 3: ADAPTIVE LAYER WEIGHTING
# ==============================================================================

class AdaptiveLayerWeighting:
    """
    TIER 3: Dynamic weighting of AI layers based on recent accuracy.

    Tracks accuracy of each layer and adjusts influence dynamically.

    Layers:
    - Pattern recognition
    - Chart analysis
    - HYDRA/GEX
    - Debate consensus
    - Introspection
    - Flow analysis

    Uses exponential moving average to track accuracy.
    """

    LAYERS = [
        "pattern",
        "chart",
        "hydra",
        "debate",
        "introspection",
        "flow",
    ]

    def __init__(self, decay: float = 0.95):
        self._lock = threading.RLock()
        self._decay = decay
        self._accuracy: Dict[str, float] = {l: 0.5 for l in self.LAYERS}  # Start at 50%
        self._weights: Dict[str, float] = {l: 1.0 / len(self.LAYERS) for l in self.LAYERS}
        self._sample_counts: Dict[str, int] = {l: 0 for l in self.LAYERS}
        logger.info("TIER_3: Adaptive layer weighting initialized")

    def record_layer_outcome(
        self,
        layer: str,
        predicted_correct: bool,
    ) -> None:
        """Record whether a layer's prediction was correct."""
        if layer not in self.LAYERS:
            return

        with self._lock:
            # EMA update
            current = self._accuracy[layer]
            new_val = 1.0 if predicted_correct else 0.0
            self._accuracy[layer] = self._decay * current + (1 - self._decay) * new_val
            self._sample_counts[layer] += 1

            # Recompute weights
            self._recompute_weights()

            logger.debug(
                f"TIER_3: Layer {layer} accuracy: {self._accuracy[layer]:.1%} "
                f"(weight: {self._weights[layer]:.1%})"
            )

    def _recompute_weights(self) -> None:
        """Recompute layer weights based on accuracy."""
        # Softmax-like weighting based on accuracy
        total = sum(self._accuracy.values())
        if total > 0:
            self._weights = {l: acc / total for l, acc in self._accuracy.items()}

        # Ensure minimum weight (prevent complete silencing)
        min_weight = 0.05
        for layer in self.LAYERS:
            if self._weights[layer] < min_weight:
                self._weights[layer] = min_weight

        # Renormalize
        total = sum(self._weights.values())
        self._weights = {l: w / total for l, w in self._weights.items()}

    def get_weighted_confidence(
        self,
        layer_confidences: Dict[str, float],
    ) -> float:
        """
        Compute weighted confidence from all layers.

        Args:
            layer_confidences: {layer_name: confidence_0_100}

        Returns:
            Weighted confidence 0-100
        """
        with self._lock:
            weighted_sum = 0.0
            total_weight = 0.0

            for layer, conf in layer_confidences.items():
                if layer in self._weights:
                    weight = self._weights[layer]
                    weighted_sum += conf * weight
                    total_weight += weight

            if total_weight > 0:
                return weighted_sum / total_weight
            return sum(layer_confidences.values()) / len(layer_confidences) if layer_confidences else 50.0

    def get_weights(self) -> Dict[str, float]:
        """Get current layer weights."""
        with self._lock:
            return self._weights.copy()

    def get_stats(self) -> Dict:
        """Get weighting statistics."""
        with self._lock:
            return {
                "weights": self._weights.copy(),
                "accuracy": self._accuracy.copy(),
                "sample_counts": self._sample_counts.copy(),
            }


# ==============================================================================
# TIER 3: VOLATILITY SMILE LAYER
# ==============================================================================

class VolatilitySmileLayer:
    """
    TIER 3: Options pricing intelligence using vol smile/skew analysis.

    Analyzes:
    - Put/call skew (fear gauge)
    - Term structure slope
    - ATM vs OTM IV ratio
    - Smile convexity

    Trading signals:
    - Steep put skew = fear/hedging = potential bullish reversal
    - Flat skew = complacency = watch for downside
    - Inverted term structure = near-term event expected
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._cache: Dict[str, Tuple[datetime, Dict]] = {}
        self._cache_ttl = 300  # 5 minutes
        logger.info("TIER_3: Volatility smile layer initialized")

    def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze volatility smile for ticker.

        Returns:
            {
                "available": bool,
                "put_skew": float,  # Negative = puts expensive (fear)
                "term_structure": str,  # "contango", "backwardation", "flat"
                "smile_signal": str,  # "bullish", "bearish", "neutral"
                "confidence": float,  # 0-100
                "recommendation": str,
            }
        """
        # Check cache
        if ticker in self._cache:
            ts, data = self._cache[ticker]
            if (datetime.now(timezone.utc) - ts).total_seconds() < self._cache_ttl:
                return data

        try:
            data = self._fetch_smile_data(ticker)
            self._cache[ticker] = (datetime.now(timezone.utc), data)
            return data
        except Exception as e:
            logger.debug(f"TIER_3: Vol smile analysis failed for {ticker} - {e}")
            return {"available": False}

    def _fetch_smile_data(self, ticker: str) -> Dict:
        """Fetch and analyze volatility smile data."""
        try:
            from wsb_snake.collectors.polygon_options import polygon_options

            # Get current price
            from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
            snapshot = polygon_enhanced.get_snapshot(ticker)
            current_price = snapshot.get("price", 0) if snapshot else 0

            if current_price <= 0:
                return {"available": False}

            # Get options chain
            chain = polygon_options.get_options_chain(ticker, current_price)
            if not chain:
                return {"available": False}

            # Analyze put/call skew
            put_ivs = []
            call_ivs = []
            atm_range = current_price * 0.03  # 3% from ATM

            for opt in chain:
                strike = opt.get("strike", 0)
                iv = opt.get("implied_volatility", 0)
                opt_type = opt.get("type", "").lower()

                if abs(strike - current_price) < atm_range:
                    if "put" in opt_type:
                        put_ivs.append(iv)
                    elif "call" in opt_type:
                        call_ivs.append(iv)

            if not put_ivs or not call_ivs:
                return {"available": False}

            avg_put_iv = sum(put_ivs) / len(put_ivs)
            avg_call_iv = sum(call_ivs) / len(call_ivs)

            # Put skew: negative means puts are more expensive (fear)
            put_skew = avg_call_iv - avg_put_iv

            # Analyze term structure
            # (Would need multiple expiries - simplified here)
            term_structure = "contango"  # Default assumption

            # Generate signal
            signal = "neutral"
            confidence = 50
            recommendation = "No strong signal from vol smile"

            if put_skew < -0.05:  # Puts significantly more expensive
                signal = "bullish"
                confidence = 65
                recommendation = "Put skew steep - fear hedging suggests potential bullish reversal"
            elif put_skew > 0.05:  # Calls significantly more expensive
                signal = "bearish"
                confidence = 60
                recommendation = "Call skew - euphoria/speculation suggests caution"

            return {
                "available": True,
                "put_skew": put_skew,
                "avg_put_iv": avg_put_iv,
                "avg_call_iv": avg_call_iv,
                "term_structure": term_structure,
                "smile_signal": signal,
                "confidence": confidence,
                "recommendation": recommendation,
            }

        except Exception as e:
            logger.debug(f"TIER_3: Vol smile fetch error - {e}")
            return {"available": False}

    def get_trading_signal(
        self,
        ticker: str,
        direction: str,
    ) -> Tuple[float, str]:
        """
        Get confidence adjustment based on vol smile.

        Returns:
            (adjustment, reason) - adjustment is confidence boost/penalty
        """
        analysis = self.analyze(ticker)
        if not analysis.get("available"):
            return 0.0, "Vol smile data unavailable"

        signal = analysis.get("smile_signal", "neutral")
        is_long = direction == "long"

        if signal == "bullish" and is_long:
            return 5.0, "Vol smile supports long (put skew)"
        elif signal == "bullish" and not is_long:
            return -5.0, "Vol smile contra short (put skew)"
        elif signal == "bearish" and not is_long:
            return 5.0, "Vol smile supports short (call skew)"
        elif signal == "bearish" and is_long:
            return -5.0, "Vol smile contra long (call skew)"

        return 0.0, "Vol smile neutral"


# ==============================================================================
# SINGLETON INSTANCES
# ==============================================================================

_drawdown_velocity_monitor: Optional[DrawdownVelocityMonitor] = None
_hydra_booster: Optional[HydraSignalBooster] = None
_regime_kelly: Optional[RegimeAwareKelly] = None
_expiry_adjuster: Optional[ExpiryStopAdjuster] = None
_test_time_reasoner: Optional[TestTimeReasoner] = None
_approval_gate: Optional[TradeApprovalGate] = None
_prompt_evolution: Optional[PromptEvolutionEngine] = None
_bats_router: Optional[BATSModelRouter] = None
_hierarchical_memory: Optional[HierarchicalMemory] = None
_adaptive_weighting: Optional[AdaptiveLayerWeighting] = None
_vol_smile: Optional[VolatilitySmileLayer] = None


def get_drawdown_velocity_monitor() -> DrawdownVelocityMonitor:
    global _drawdown_velocity_monitor
    if _drawdown_velocity_monitor is None:
        _drawdown_velocity_monitor = DrawdownVelocityMonitor()
    return _drawdown_velocity_monitor


def get_hydra_booster() -> HydraSignalBooster:
    global _hydra_booster
    if _hydra_booster is None:
        _hydra_booster = HydraSignalBooster()
    return _hydra_booster


def get_regime_kelly() -> RegimeAwareKelly:
    global _regime_kelly
    if _regime_kelly is None:
        _regime_kelly = RegimeAwareKelly()
    return _regime_kelly


def get_expiry_adjuster() -> ExpiryStopAdjuster:
    global _expiry_adjuster
    if _expiry_adjuster is None:
        _expiry_adjuster = ExpiryStopAdjuster()
    return _expiry_adjuster


def get_test_time_reasoner() -> TestTimeReasoner:
    global _test_time_reasoner
    if _test_time_reasoner is None:
        _test_time_reasoner = TestTimeReasoner()
    return _test_time_reasoner


def get_approval_gate() -> TradeApprovalGate:
    global _approval_gate
    if _approval_gate is None:
        _approval_gate = TradeApprovalGate()
    return _approval_gate


def get_prompt_evolution() -> PromptEvolutionEngine:
    global _prompt_evolution
    if _prompt_evolution is None:
        _prompt_evolution = PromptEvolutionEngine()
    return _prompt_evolution


def get_bats_router() -> BATSModelRouter:
    global _bats_router
    if _bats_router is None:
        _bats_router = BATSModelRouter()
    return _bats_router


def get_hierarchical_memory() -> HierarchicalMemory:
    global _hierarchical_memory
    if _hierarchical_memory is None:
        _hierarchical_memory = HierarchicalMemory()
    return _hierarchical_memory


def get_adaptive_weighting() -> AdaptiveLayerWeighting:
    global _adaptive_weighting
    if _adaptive_weighting is None:
        _adaptive_weighting = AdaptiveLayerWeighting()
    return _adaptive_weighting


def get_vol_smile() -> VolatilitySmileLayer:
    global _vol_smile
    if _vol_smile is None:
        _vol_smile = VolatilitySmileLayer()
    return _vol_smile
