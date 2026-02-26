"""
Gate 15: Bull/Bear Debate System
================================

Based on IgorGanapolsky/trading's multi-gate funnel architecture.

Two AI agents (bull & bear) evaluate the same setup from opposing perspectives.
This prevents confirmation bias and catches traps that single-perspective AI misses.

Key Research:
- Multi-agent debate improves accuracy by 15-20% (MetaGPT, ICML'24)
- Adversarial testing catches false consensus
- Bull/bear dynamics model real market participants
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

from wsb_snake.utils.logger import get_logger
from wsb_snake.db.database import get_connection

logger = get_logger(__name__)


@dataclass
class DebateRound:
    """A single round of bull/bear debate."""
    round_number: int
    bull_conviction: float
    bear_conviction: float
    bull_argument: str
    bear_argument: str
    round_winner: str  # "bull", "bear", or "draw"


@dataclass
class DebateResult:
    """Result of bull vs bear debate."""
    bull_conviction: float  # 0-100 (final)
    bear_conviction: float  # 0-100 (final)
    bull_reasoning: str
    bear_reasoning: str
    consensus_action: str  # STRIKE, REDUCE_SIZE, ABORT
    confidence_multiplier: float  # 0.5 to 1.0
    debate_quality: str  # HIGH, MEDIUM, LOW
    timestamp: datetime
    # NEW: Multi-round debate tracking (from TradingAgents pattern)
    rounds: List[DebateRound] = field(default_factory=list)
    initial_bull_conviction: float = 0.0  # First round conviction
    initial_bear_conviction: float = 0.0
    conviction_delta: float = 0.0  # Change from initial to final
    rounds_completed: int = 1


class BullBearDebate:
    """
    Gate 15: Consensus through adversarial debate.

    Bull agent: Argues why the trade will succeed
    Bear agent: Argues why the trade will fail

    If bear agent has strong conviction (>70%), reduce position size.
    If both have weak conviction (<50%), skip trade (low quality setup).

    Multi-round debate (from TradingAgents pattern):
    - Configurable max_debate_rounds (default: 3)
    - Each round refines arguments based on opponent's points
    - Tracks conviction delta (shift from initial to final)
    - Early termination if consensus is clear
    """

    # Debate thresholds
    BEAR_VETO_THRESHOLD = 75  # Bear conviction above this = size reduction
    LOW_QUALITY_THRESHOLD = 50  # Both below this = skip trade
    HIGH_QUALITY_THRESHOLD = 70  # Bull high + bear low = confident trade
    CONSENSUS_THRESHOLD = 30  # If spread > 30, no need for more rounds

    def __init__(self, max_debate_rounds: int = 3):
        """
        Initialize debate system.

        Args:
            max_debate_rounds: Maximum rounds of debate (1-5). More rounds
                               allow for more refined arguments but take longer.
        """
        self._lock = threading.RLock()
        self.max_debate_rounds = max(1, min(5, max_debate_rounds))
        self.debate_history: List[DebateResult] = []
        self._init_db()
        logger.info(f"GATE_15: Bull/Bear debate system initialized (max_rounds={self.max_debate_rounds})")

    def _init_db(self) -> None:
        """Initialize database table for debate tracking."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS debate_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    ticker TEXT,
                    pattern TEXT,
                    direction TEXT,
                    bull_conviction REAL,
                    bear_conviction REAL,
                    consensus_action TEXT,
                    confidence_multiplier REAL,
                    trade_outcome REAL,
                    bull_correct INTEGER,
                    bear_correct INTEGER
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"GATE_15: DB init failed - {e}")

    def evaluate(
        self,
        setup: Any,  # ScalpSetup or similar
        context: Dict[str, Any]
    ) -> DebateResult:
        """
        Run multi-round bull vs bear debate on trade setup.

        Based on TradingAgents pattern - each round allows agents to
        refine arguments based on opponent's previous points.

        Args:
            setup: The trade setup being evaluated
            context: Additional context (lessons, predator verdict, etc.)

        Returns:
            DebateResult with conviction scores, rounds, and conviction delta
        """
        with self._lock:
            ticker = context.get("ticker", "UNKNOWN")
            pattern = context.get("pattern", "unknown")
            direction = getattr(setup, "direction", context.get("direction", "long"))

            rounds: List[DebateRound] = []
            bull_conviction = 50.0
            bear_conviction = 50.0
            bull_reasoning = ""
            bear_reasoning = ""
            initial_bull = 0.0
            initial_bear = 0.0

            # Multi-round debate
            for round_num in range(1, self.max_debate_rounds + 1):
                # Create round context with opponent's previous arguments
                round_context = context.copy()
                if round_num > 1:
                    round_context["opponent_bull_points"] = bull_reasoning
                    round_context["opponent_bear_points"] = bear_reasoning
                    round_context["current_bull_conviction"] = bull_conviction
                    round_context["current_bear_conviction"] = bear_conviction

                # Get arguments for this round
                new_bull_conviction, new_bull_reasoning = self._get_bull_argument(
                    setup, round_context, round_num
                )
                new_bear_conviction, new_bear_reasoning = self._get_bear_argument(
                    setup, round_context, round_num
                )

                # Store initial convictions
                if round_num == 1:
                    initial_bull = new_bull_conviction
                    initial_bear = new_bear_conviction

                # Update running convictions (blend with previous)
                if round_num > 1:
                    # Weight new arguments at 60%, previous at 40%
                    bull_conviction = 0.4 * bull_conviction + 0.6 * new_bull_conviction
                    bear_conviction = 0.4 * bear_conviction + 0.6 * new_bear_conviction
                else:
                    bull_conviction = new_bull_conviction
                    bear_conviction = new_bear_conviction

                # Determine round winner
                spread = bull_conviction - bear_conviction
                if spread > 10:
                    round_winner = "bull"
                elif spread < -10:
                    round_winner = "bear"
                else:
                    round_winner = "draw"

                # Record round
                debate_round = DebateRound(
                    round_number=round_num,
                    bull_conviction=new_bull_conviction,
                    bear_conviction=new_bear_conviction,
                    bull_argument=new_bull_reasoning,
                    bear_argument=new_bear_reasoning,
                    round_winner=round_winner,
                )
                rounds.append(debate_round)

                # Update cumulative reasoning
                bull_reasoning = new_bull_reasoning
                bear_reasoning = new_bear_reasoning

                # Early termination if clear consensus
                if abs(bull_conviction - bear_conviction) > self.CONSENSUS_THRESHOLD:
                    logger.debug(
                        f"GATE_15: Early consensus after round {round_num} "
                        f"(spread={abs(bull_conviction - bear_conviction):.0f})"
                    )
                    break

            # Calculate conviction delta
            conviction_delta = abs(bull_conviction - initial_bull) + abs(bear_conviction - initial_bear)

            # Determine consensus action
            consensus_action, confidence_multiplier = self._calculate_consensus(
                bull_conviction, bear_conviction, direction
            )

            # Assess debate quality
            debate_quality = self._assess_debate_quality(bull_conviction, bear_conviction)

            result = DebateResult(
                bull_conviction=bull_conviction,
                bear_conviction=bear_conviction,
                bull_reasoning=bull_reasoning,
                bear_reasoning=bear_reasoning,
                consensus_action=consensus_action,
                confidence_multiplier=confidence_multiplier,
                debate_quality=debate_quality,
                timestamp=datetime.now(timezone.utc),
                rounds=rounds,
                initial_bull_conviction=initial_bull,
                initial_bear_conviction=initial_bear,
                conviction_delta=conviction_delta,
                rounds_completed=len(rounds),
            )

            # Log debate with round info
            logger.info(
                f"GATE_15: {ticker} {direction} debate ({len(rounds)} rounds) - "
                f"Bull={bull_conviction:.0f}% (Δ{bull_conviction - initial_bull:+.0f}) "
                f"Bear={bear_conviction:.0f}% (Δ{bear_conviction - initial_bear:+.0f}) "
                f"→ {consensus_action} ({confidence_multiplier:.0%})"
            )

            # Track history
            self.debate_history.append(result)
            self._save_debate(ticker, pattern, direction, result)

            return result

    def _get_bull_argument(
        self,
        setup: Any,
        context: Dict[str, Any],
        round_num: int = 1
    ) -> Tuple[float, str]:
        """
        Generate bullish argument for the trade.

        In production, this would call an AI model.
        For now, use rule-based heuristics based on available data.

        Multi-round enhancement:
        - Round 1: Base analysis
        - Round 2+: Counter bear's previous points, strengthen weak areas
        """
        conviction = 50.0  # Start neutral
        reasons = []

        # Multi-round: Address opponent's concerns in later rounds
        if round_num > 1:
            opponent_points = context.get("opponent_bear_points", "")
            current_bear = context.get("current_bear_conviction", 50)

            # If bear cited weak volume, emphasize other confirmations
            if "volume" in opponent_points.lower() and current_bear > 60:
                conviction += 5
                reasons.append("Volume weakness offset by other factors")

            # If bear cited counter-momentum, look for reversal signs
            if "counter-momentum" in opponent_points.lower():
                conviction += 3
                reasons.append("Counter-trend can mean reversal opportunity")

        # Pattern confidence boost
        pattern_confidence = getattr(setup, "confidence", 50)
        if pattern_confidence > 70:
            conviction += 15
            reasons.append(f"Strong pattern ({pattern_confidence:.0f}%)")
        elif pattern_confidence > 60:
            conviction += 8
            reasons.append(f"Decent pattern ({pattern_confidence:.0f}%)")

        # AI confirmation boost
        if getattr(setup, "ai_confirmed", False):
            conviction += 12
            reasons.append("AI confirmed setup")

        # Volume confirmation
        volume_ratio = getattr(setup, "volume_ratio", 1.0)
        if volume_ratio > 1.5:
            conviction += 10
            reasons.append(f"Volume surge ({volume_ratio:.1f}x)")
        elif volume_ratio > 1.2:
            conviction += 5
            reasons.append(f"Volume confirmation ({volume_ratio:.1f}x)")

        # Momentum alignment
        momentum = getattr(setup, "momentum", 0)
        direction = getattr(setup, "direction", "long")
        if (direction == "long" and momentum > 0.1) or (direction == "short" and momentum < -0.1):
            conviction += 8
            reasons.append(f"Momentum aligned ({momentum:+.2f}%)")

        # VWAP position
        vwap = getattr(setup, "vwap", 0)
        entry_price = getattr(setup, "entry_price", 0)
        if vwap > 0 and entry_price > 0:
            vwap_distance = (entry_price - vwap) / vwap * 100
            if direction == "long" and entry_price > vwap:
                conviction += 5
                reasons.append(f"Above VWAP (+{vwap_distance:.1f}%)")
            elif direction == "short" and entry_price < vwap:
                conviction += 5
                reasons.append(f"Below VWAP ({vwap_distance:.1f}%)")

        # Predator Prime verdict
        predator_verdict = context.get("predator_verdict")
        if predator_verdict:
            if predator_verdict.action == "STRIKE":
                conviction += 10
                reasons.append(f"Predator Prime STRIKE ({predator_verdict.conviction:.0f}%)")

        # Historical lessons positive
        lessons = context.get("recent_lessons", [])
        positive_lessons = [l for l in lessons if "success" in l.get("learning", "").lower()]
        if positive_lessons:
            conviction += 5 * len(positive_lessons)
            reasons.append(f"{len(positive_lessons)} positive lessons")

        # Clamp conviction
        conviction = max(0, min(100, conviction))

        reasoning = "; ".join(reasons) if reasons else "Neutral setup, no strong bullish factors"
        return conviction, reasoning

    def _get_bear_argument(
        self,
        setup: Any,
        context: Dict[str, Any],
        round_num: int = 1
    ) -> Tuple[float, str]:
        """
        Generate bearish argument against the trade.

        This agent tries to find reasons why the trade will FAIL.

        Multi-round enhancement:
        - Round 1: Base risk analysis
        - Round 2+: Counter bull's optimism, highlight overlooked risks
        """
        conviction = 50.0  # Start neutral
        reasons = []

        # Multi-round: Counter bull's arguments in later rounds
        if round_num > 1:
            opponent_points = context.get("opponent_bull_points", "")
            current_bull = context.get("current_bull_conviction", 50)

            # If bull is overconfident, remind of base rate failures
            if current_bull > 75:
                conviction += 8
                reasons.append(f"Overconfidence warning (bull at {current_bull:.0f}%)")

            # If bull cited AI confirmation, note AI limitations
            if "ai confirmed" in opponent_points.lower():
                conviction += 5
                reasons.append("AI confirmation not infallible")

            # If bull cited Predator Prime, note market conditions
            if "predator" in opponent_points.lower():
                conviction += 3
                reasons.append("Predator signal subject to regime shifts")

        # Low confidence = bear wins
        pattern_confidence = getattr(setup, "confidence", 50)
        if pattern_confidence < 60:
            conviction += 15
            reasons.append(f"Weak pattern confidence ({pattern_confidence:.0f}%)")

        # No AI confirmation = concern
        if not getattr(setup, "ai_confirmed", False):
            conviction += 10
            reasons.append("No AI confirmation")

        # Weak volume = likely to fail
        volume_ratio = getattr(setup, "volume_ratio", 1.0)
        if volume_ratio < 1.0:
            conviction += 12
            reasons.append(f"Weak volume ({volume_ratio:.1f}x)")
        elif volume_ratio < 1.2:
            conviction += 5
            reasons.append(f"Marginal volume ({volume_ratio:.1f}x)")

        # Counter-momentum
        momentum = getattr(setup, "momentum", 0)
        direction = getattr(setup, "direction", "long")
        if (direction == "long" and momentum < 0) or (direction == "short" and momentum > 0):
            conviction += 15
            reasons.append(f"Counter-momentum ({momentum:+.2f}%)")

        # VWAP resistance
        vwap = getattr(setup, "vwap", 0)
        entry_price = getattr(setup, "entry_price", 0)
        if vwap > 0 and entry_price > 0:
            vwap_distance = abs(entry_price - vwap) / vwap * 100
            if vwap_distance < 0.1:
                conviction += 8
                reasons.append("Right at VWAP - chop risk")

        # Trap detection from contrarian layer
        trap_detected = context.get("trap_detected", "NONE")
        if trap_detected and trap_detected != "NONE":
            conviction += 20
            reasons.append(f"Trap detected: {trap_detected}")

        # Historical lessons negative
        lessons = context.get("recent_lessons", [])
        negative_lessons = [l for l in lessons if "fail" in l.get("learning", "").lower() or
                          "loss" in l.get("learning", "").lower()]
        if negative_lessons:
            conviction += 8 * len(negative_lessons)
            reasons.append(f"{len(negative_lessons)} cautionary lessons")

        # Time decay concern (0DTE)
        hours_to_expiry = context.get("hours_to_expiry", 6)
        if hours_to_expiry < 1:
            conviction += 15
            reasons.append(f"Theta acceleration ({hours_to_expiry:.1f}h to expiry)")
        elif hours_to_expiry < 2:
            conviction += 8
            reasons.append(f"Late day theta risk ({hours_to_expiry:.1f}h)")

        # High VIX = harder to predict
        vix_regime = context.get("vix_regime", "NORMAL")
        if vix_regime in ["ELEVATED", "CRISIS"]:
            conviction += 10
            reasons.append(f"High volatility regime: {vix_regime}")

        # Clamp conviction
        conviction = max(0, min(100, conviction))

        reasoning = "; ".join(reasons) if reasons else "No strong bearish factors found"
        return conviction, reasoning

    def _calculate_consensus(
        self,
        bull_conviction: float,
        bear_conviction: float,
        direction: str
    ) -> Tuple[str, float]:
        """
        Calculate consensus action and confidence multiplier.

        Returns:
            (action, multiplier) where action is STRIKE/REDUCE_SIZE/ABORT
        """
        # Both weak = low quality setup
        if bull_conviction < self.LOW_QUALITY_THRESHOLD and bear_conviction < self.LOW_QUALITY_THRESHOLD:
            return "ABORT", 0.0

        # Bear has veto power
        if bear_conviction >= self.BEAR_VETO_THRESHOLD:
            if bull_conviction >= 80:
                # Bull very strong despite bear concerns
                return "REDUCE_SIZE", 0.7
            else:
                return "ABORT", 0.0

        # Bull dominant
        if bull_conviction >= self.HIGH_QUALITY_THRESHOLD and bear_conviction < 50:
            return "STRIKE", 1.0

        # Mixed signals
        if bull_conviction > bear_conviction:
            multiplier = 1.0 - (bear_conviction / 200)  # 50% bear = 0.75x
            return "STRIKE", max(0.5, multiplier)
        else:
            # Bear slightly higher but not veto level
            return "REDUCE_SIZE", 0.6

    def _assess_debate_quality(
        self,
        bull_conviction: float,
        bear_conviction: float
    ) -> str:
        """Assess the quality of the debate based on conviction spread."""
        spread = abs(bull_conviction - bear_conviction)

        if spread > 40:
            return "HIGH"  # Clear winner
        elif spread > 20:
            return "MEDIUM"  # Moderate disagreement
        else:
            return "LOW"  # Too close to call

    def _save_debate(
        self,
        ticker: str,
        pattern: str,
        direction: str,
        result: DebateResult
    ) -> None:
        """Save debate to database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO debate_history
                (timestamp, ticker, pattern, direction, bull_conviction,
                 bear_conviction, consensus_action, confidence_multiplier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.timestamp.isoformat(),
                ticker,
                pattern,
                direction,
                result.bull_conviction,
                result.bear_conviction,
                result.consensus_action,
                result.confidence_multiplier
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"GATE_15: Save failed - {e}")

    def record_outcome(
        self,
        ticker: str,
        pnl_pct: float
    ) -> None:
        """
        Record trade outcome to learn which agent was correct.

        Bull correct if trade won, Bear correct if trade lost.
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Update most recent debate for this ticker
            bull_correct = 1 if pnl_pct > 0 else 0
            bear_correct = 1 if pnl_pct <= 0 else 0

            cursor.execute("""
                UPDATE debate_history
                SET trade_outcome = ?, bull_correct = ?, bear_correct = ?
                WHERE ticker = ? AND trade_outcome IS NULL
                ORDER BY timestamp DESC
                LIMIT 1
            """, (pnl_pct, bull_correct, bear_correct, ticker))

            conn.commit()
            conn.close()

            logger.debug(f"GATE_15: Recorded outcome for {ticker}: {pnl_pct:+.1f}%")
        except Exception as e:
            logger.debug(f"GATE_15: Outcome record failed - {e}")

    def get_accuracy_stats(self) -> Dict[str, Any]:
        """Get accuracy statistics for bull and bear agents."""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(bull_correct) as bull_wins,
                    SUM(bear_correct) as bear_wins,
                    AVG(trade_outcome) as avg_pnl
                FROM debate_history
                WHERE trade_outcome IS NOT NULL
            """)

            row = cursor.fetchone()
            conn.close()

            if row and row[0] > 0:
                return {
                    "total_debates": row[0],
                    "bull_accuracy": row[1] / row[0] if row[0] > 0 else 0,
                    "bear_accuracy": row[2] / row[0] if row[0] > 0 else 0,
                    "avg_pnl": row[3] or 0
                }

            return {"total_debates": 0}
        except Exception as e:
            logger.debug(f"GATE_15: Stats query failed - {e}")
            return {"total_debates": 0}

    def set_max_rounds(self, max_rounds: int) -> None:
        """
        Configure maximum debate rounds.

        Args:
            max_rounds: 1-5 rounds. More rounds = more refined but slower.
        """
        self.max_debate_rounds = max(1, min(5, max_rounds))
        logger.info(f"GATE_15: Max debate rounds set to {self.max_debate_rounds}")

    def get_round_stats(self) -> Dict[str, Any]:
        """
        Get statistics about multi-round debates.

        Returns info about how debates typically evolve across rounds.
        """
        if not self.debate_history:
            return {"debates_analyzed": 0}

        total_rounds = sum(r.rounds_completed for r in self.debate_history)
        avg_rounds = total_rounds / len(self.debate_history)

        # Calculate average conviction delta
        deltas = [r.conviction_delta for r in self.debate_history if r.conviction_delta > 0]
        avg_delta = sum(deltas) / len(deltas) if deltas else 0

        # Count early terminations
        early_terms = sum(1 for r in self.debate_history if r.rounds_completed < self.max_debate_rounds)

        return {
            "debates_analyzed": len(self.debate_history),
            "max_rounds_config": self.max_debate_rounds,
            "avg_rounds_used": avg_rounds,
            "avg_conviction_delta": avg_delta,
            "early_terminations": early_terms,
            "early_termination_rate": early_terms / len(self.debate_history) if self.debate_history else 0,
        }


# Singleton instance
_debate_engine: Optional[BullBearDebate] = None


def get_debate_engine() -> BullBearDebate:
    """Get singleton debate engine instance."""
    global _debate_engine
    if _debate_engine is None:
        _debate_engine = BullBearDebate()
    return _debate_engine
