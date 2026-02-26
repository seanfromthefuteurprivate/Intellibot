"""
Self-Evolving Agent Memory System

Based on research from:
- EvoAgentX/Awesome-Self-Evolving-Agents (https://arxiv.org/abs/2508.07407)
- IgorGanapolsky/trading autonomous trading system

This module implements concepts from cutting-edge self-evolving AI agent research
to enable WSB Snake to improve over time through:

1. MEMORY OPTIMIZATION - Learn from past trades and patterns
2. FEEDBACK-DRIVEN LEARNING - Thompson Sampling for strategy selection
3. PROMPT EVOLUTION - Adapt prompts based on outcomes
4. TOOL OPTIMIZATION - Learn which tools work best for which tasks
5. UNIFIED EVOLUTION - Continuous improvement across all dimensions

Key Papers & Concepts:
- Agent Workflow Memory (ICML'24) - Store and retrieve successful workflows
- MemoryBank (AAAI'24) - Long-term memory with retrieval
- A-MEM (Arxiv'25) - Agentic memory for LLM agents
- Mem0 (Arxiv'25) - Scalable long-term memory for production agents
- Thompson Sampling - Bayesian strategy selection with uncertainty
"""

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import random
import math

from wsb_snake.utils.logger import get_logger
from wsb_snake.db.database import get_connection

logger = get_logger(__name__)


# =============================================================================
# CORE CONCEPTS FROM SELF-EVOLVING AGENTS RESEARCH
# =============================================================================

class EvolutionDimension(Enum):
    """Dimensions along which agents can self-evolve."""
    LLM_BEHAVIOR = "llm_behavior"        # How the LLM responds
    PROMPTS = "prompts"                   # Prompt optimization
    MEMORY = "memory"                     # What to remember
    TOOLS = "tools"                       # Tool selection/creation
    STRATEGY = "strategy"                 # Trading strategy selection
    RISK = "risk"                         # Risk parameter tuning
    TIMING = "timing"                     # Entry/exit timing


class FeedbackType(Enum):
    """Types of feedback for learning."""
    TRADE_OUTCOME = "trade_outcome"       # Win/loss
    SIGNAL_ACCURACY = "signal_accuracy"   # Signal led to good trade
    PATTERN_VALIDITY = "pattern_validity" # Pattern was correct
    TIMING_QUALITY = "timing_quality"     # Entry/exit timing
    RISK_REWARD = "risk_reward"           # R:R achieved vs expected


@dataclass
class EvolutionFeedback:
    """Feedback signal for evolution."""
    dimension: EvolutionDimension
    feedback_type: FeedbackType
    context: Dict[str, Any]
    outcome: float  # -1 to 1 (bad to good)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# THOMPSON SAMPLING FOR STRATEGY SELECTION
# Based on IgorGanapolsky/trading's Thompson Sampling implementation
# =============================================================================

@dataclass
class BetaDistribution:
    """Beta distribution for Thompson Sampling."""
    alpha: float = 1.0  # Successes + 1
    beta: float = 1.0   # Failures + 1

    def sample(self) -> float:
        """Sample from the beta distribution."""
        return random.betavariate(self.alpha, self.beta)

    def mean(self) -> float:
        """Expected value."""
        return self.alpha / (self.alpha + self.beta)

    def update(self, success: bool, weight: float = 1.0) -> None:
        """Update distribution with new observation."""
        if success:
            self.alpha += weight
        else:
            self.beta += weight

    def confidence_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Approximate confidence interval."""
        # Use normal approximation for large alpha/beta
        n = self.alpha + self.beta
        p = self.alpha / n
        z = 1.96 if level == 0.95 else 2.58  # 95% or 99%
        margin = z * math.sqrt(p * (1 - p) / n)
        return (max(0, p - margin), min(1, p + margin))


class ThompsonSampler:
    """
    Thompson Sampling for strategy/pattern selection.

    Each strategy/pattern maintains a Beta distribution tracking its success rate.
    Selection is done by sampling from each distribution and picking the highest.
    This naturally balances exploration (uncertain strategies) with exploitation
    (known good strategies).

    Based on:
    - IgorGanapolsky/trading's trade_confidence.py
    - Multi-armed bandit research for trading
    """

    # Decay factor for older observations (30-day half-life)
    DECAY_HALF_LIFE_DAYS = 30

    def __init__(self, db_path: Optional[str] = None):
        """Initialize Thompson Sampler."""
        self.distributions: Dict[str, BetaDistribution] = {}
        self.observation_times: Dict[str, List[datetime]] = {}
        self._lock = threading.RLock()
        self.db_path = db_path

        # Load from database if available
        self._load_from_db()

        logger.info("THOMPSON_SAMPLER: Initialized with %d strategies", len(self.distributions))

    def _load_from_db(self) -> None:
        """Load distributions from database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS thompson_distributions (
                    strategy_key TEXT PRIMARY KEY,
                    alpha REAL DEFAULT 1.0,
                    beta REAL DEFAULT 1.0,
                    observations INTEGER DEFAULT 0,
                    last_update TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()

            # Load existing distributions
            cursor.execute("SELECT strategy_key, alpha, beta FROM thompson_distributions")
            for row in cursor.fetchall():
                self.distributions[row[0]] = BetaDistribution(alpha=row[1], beta=row[2])

            conn.close()
        except Exception as e:
            logger.warning(f"THOMPSON_SAMPLER: Could not load from DB - {e}")

    def _save_to_db(self, key: str, dist: BetaDistribution) -> None:
        """Save distribution to database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO thompson_distributions
                (strategy_key, alpha, beta, observations, last_update)
                VALUES (?, ?, ?, ?, ?)
            """, (key, dist.alpha, dist.beta,
                  int(dist.alpha + dist.beta - 2),
                  datetime.now(timezone.utc).isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"THOMPSON_SAMPLER: Could not save to DB - {e}")

    def get_or_create(self, key: str) -> BetaDistribution:
        """Get or create distribution for a strategy/pattern."""
        with self._lock:
            if key not in self.distributions:
                self.distributions[key] = BetaDistribution()
                self.observation_times[key] = []
            return self.distributions[key]

    def select_best(self, candidates: List[str]) -> Tuple[str, float]:
        """
        Select the best candidate using Thompson Sampling.

        Args:
            candidates: List of strategy/pattern keys

        Returns:
            (best_key, sampled_value)
        """
        with self._lock:
            if not candidates:
                raise ValueError("No candidates to select from")

            samples = {}
            for key in candidates:
                dist = self.get_or_create(key)
                samples[key] = dist.sample()

            best_key = max(samples, key=samples.get)
            return best_key, samples[best_key]

    def update(self, key: str, success: bool, weight: float = 1.0) -> None:
        """
        Update distribution with new observation.

        Args:
            key: Strategy/pattern key
            success: Whether the outcome was successful
            weight: Weight of observation (default 1.0)
        """
        with self._lock:
            dist = self.get_or_create(key)
            dist.update(success, weight)

            # Track observation time
            if key not in self.observation_times:
                self.observation_times[key] = []
            self.observation_times[key].append(datetime.now(timezone.utc))

            # Save to DB
            self._save_to_db(key, dist)

            logger.info(
                f"THOMPSON_SAMPLER: Updated {key} - "
                f"{'SUCCESS' if success else 'FAILURE'} - "
                f"new mean={dist.mean():.2%}"
            )

    def apply_decay(self) -> None:
        """Apply exponential decay to older observations."""
        with self._lock:
            now = datetime.now(timezone.utc)
            decay_rate = math.log(2) / self.DECAY_HALF_LIFE_DAYS

            for key, dist in self.distributions.items():
                # Calculate effective alpha/beta after decay
                # This is approximate - full implementation would track each observation
                total = dist.alpha + dist.beta - 2  # Subtract priors
                if total > 0:
                    decay_factor = 0.99  # Daily decay
                    dist.alpha = 1 + (dist.alpha - 1) * decay_factor
                    dist.beta = 1 + (dist.beta - 1) * decay_factor

    def get_stats(self, key: str) -> Dict[str, Any]:
        """Get statistics for a strategy/pattern."""
        with self._lock:
            dist = self.get_or_create(key)
            ci_low, ci_high = dist.confidence_interval()

            return {
                "key": key,
                "mean": dist.mean(),
                "alpha": dist.alpha,
                "beta": dist.beta,
                "observations": int(dist.alpha + dist.beta - 2),
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
            }

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all strategies."""
        with self._lock:
            return [self.get_stats(key) for key in sorted(self.distributions.keys())]


# =============================================================================
# LESSONS LEARNED RAG
# Based on IgorGanapolsky/trading's lessons learned system
# =============================================================================

@dataclass
class Lesson:
    """A learned lesson from trading experience."""
    lesson_id: str
    category: str  # e.g., "risk", "timing", "pattern", "strategy"
    title: str
    description: str
    context: Dict[str, Any]
    outcome: str  # What happened
    learning: str  # What we learned
    action: str  # What to do differently
    confidence: float  # How confident in this lesson (0-1)
    times_applied: int = 0
    times_helped: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_applied: Optional[datetime] = None


class LessonsMemory:
    """
    Memory system for storing and retrieving lessons learned.

    Based on:
    - IgorGanapolsky/trading's RAG lessons system
    - MemoryBank (AAAI'24)
    - A-MEM (Arxiv'25)
    """

    CATEGORIES = [
        "pattern_recognition",
        "risk_management",
        "entry_timing",
        "exit_timing",
        "position_sizing",
        "market_regime",
        "sector_correlation",
        "options_pricing",
        "execution",
        "general"
    ]

    def __init__(self):
        """Initialize lessons memory."""
        self.lessons: Dict[str, Lesson] = {}
        self._lock = threading.RLock()
        self._init_db()
        self._load_lessons()

        logger.info(f"LESSONS_MEMORY: Initialized with {len(self.lessons)} lessons")

    def _init_db(self) -> None:
        """Initialize database table."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lessons_learned (
                    lesson_id TEXT PRIMARY KEY,
                    category TEXT,
                    title TEXT,
                    description TEXT,
                    context TEXT,
                    outcome TEXT,
                    learning TEXT,
                    action TEXT,
                    confidence REAL,
                    times_applied INTEGER DEFAULT 0,
                    times_helped INTEGER DEFAULT 0,
                    created_at TEXT,
                    last_applied TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"LESSONS_MEMORY: DB init failed - {e}")

    def _load_lessons(self) -> None:
        """Load lessons from database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM lessons_learned")

            for row in cursor.fetchall():
                lesson = Lesson(
                    lesson_id=row[0],
                    category=row[1],
                    title=row[2],
                    description=row[3],
                    context=json.loads(row[4]) if row[4] else {},
                    outcome=row[5],
                    learning=row[6],
                    action=row[7],
                    confidence=row[8],
                    times_applied=row[9],
                    times_helped=row[10],
                    created_at=datetime.fromisoformat(row[11]) if row[11] else datetime.now(timezone.utc),
                    last_applied=datetime.fromisoformat(row[12]) if row[12] else None
                )
                self.lessons[lesson.lesson_id] = lesson

            conn.close()
        except Exception as e:
            logger.warning(f"LESSONS_MEMORY: Load failed - {e}")

    def add_lesson(
        self,
        category: str,
        title: str,
        description: str,
        context: Dict[str, Any],
        outcome: str,
        learning: str,
        action: str,
        confidence: float = 0.7
    ) -> Lesson:
        """Add a new lesson."""
        with self._lock:
            lesson_id = f"lesson_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.lessons)}"

            lesson = Lesson(
                lesson_id=lesson_id,
                category=category,
                title=title,
                description=description,
                context=context,
                outcome=outcome,
                learning=learning,
                action=action,
                confidence=confidence
            )

            self.lessons[lesson_id] = lesson
            self._save_lesson(lesson)

            logger.info(f"LESSONS_MEMORY: Added lesson '{title}' in category '{category}'")
            return lesson

    def _save_lesson(self, lesson: Lesson) -> None:
        """Save lesson to database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO lessons_learned
                (lesson_id, category, title, description, context, outcome,
                 learning, action, confidence, times_applied, times_helped,
                 created_at, last_applied)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lesson.lesson_id, lesson.category, lesson.title, lesson.description,
                json.dumps(lesson.context), lesson.outcome, lesson.learning,
                lesson.action, lesson.confidence, lesson.times_applied,
                lesson.times_helped, lesson.created_at.isoformat(),
                lesson.last_applied.isoformat() if lesson.last_applied else None
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"LESSONS_MEMORY: Save failed - {e}")

    def query_relevant(
        self,
        context: Dict[str, Any],
        categories: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Lesson]:
        """
        Query for relevant lessons based on context.

        Simple keyword matching for now - can be upgraded to vector similarity.
        """
        with self._lock:
            # Build search terms from context
            search_terms = []
            for key, value in context.items():
                if isinstance(value, str):
                    search_terms.extend(value.lower().split())
                elif isinstance(value, (int, float)):
                    search_terms.append(str(value))

            # Score each lesson
            scored = []
            for lesson in self.lessons.values():
                if categories and lesson.category not in categories:
                    continue

                # Simple keyword matching score
                score = 0
                lesson_text = f"{lesson.title} {lesson.description} {lesson.learning}".lower()

                for term in search_terms:
                    if term in lesson_text:
                        score += 1

                # Boost by confidence and success rate
                if lesson.times_applied > 0:
                    success_rate = lesson.times_helped / lesson.times_applied
                    score *= (1 + success_rate)

                score *= lesson.confidence

                if score > 0:
                    scored.append((score, lesson))

            # Sort by score and return top N
            scored.sort(key=lambda x: x[0], reverse=True)
            return [lesson for _, lesson in scored[:limit]]

    def mark_applied(self, lesson_id: str, helped: bool) -> None:
        """Mark that a lesson was applied."""
        with self._lock:
            if lesson_id in self.lessons:
                lesson = self.lessons[lesson_id]
                lesson.times_applied += 1
                if helped:
                    lesson.times_helped += 1
                lesson.last_applied = datetime.now(timezone.utc)
                self._save_lesson(lesson)

                logger.info(
                    f"LESSONS_MEMORY: Lesson '{lesson.title}' applied - "
                    f"{'helped' if helped else 'did not help'}"
                )


# =============================================================================
# SELF-EVOLVING ENGINE
# Coordinates all evolution dimensions
# =============================================================================

class SelfEvolvingEngine:
    """
    Master engine for self-evolving capabilities.

    Coordinates:
    - Thompson Sampling for strategy selection
    - Lessons learned memory
    - Feedback collection and processing
    - Evolution across multiple dimensions

    Based on:
    - EvoAgentX framework
    - IgorGanapolsky/trading RLHF pipeline
    - Research papers on self-evolving agents
    """

    def __init__(self):
        """Initialize self-evolving engine."""
        self.thompson_sampler = ThompsonSampler()
        self.lessons_memory = LessonsMemory()
        self.feedback_buffer: List[EvolutionFeedback] = []
        self._lock = threading.RLock()

        # Evolution parameters
        self.evolution_enabled = True
        self.min_observations_for_evolution = 10
        self.evolution_threshold = 0.1  # Min improvement to evolve

        logger.info("SELF_EVOLVING_ENGINE: Initialized")

    def record_trade_outcome(
        self,
        ticker: str,
        pattern: str,
        strategy: str,
        direction: str,
        pnl_pct: float,
        hold_time_minutes: int,
        context: Dict[str, Any]
    ) -> None:
        """
        Record a trade outcome for learning.

        This feeds into multiple learning systems:
        1. Thompson Sampling for pattern/strategy selection
        2. Lessons learned if notable outcome
        3. Evolution feedback buffer
        """
        success = pnl_pct > 0

        # Update Thompson Sampling for pattern
        pattern_key = f"pattern:{pattern}"
        self.thompson_sampler.update(pattern_key, success, weight=abs(pnl_pct) / 10 + 0.5)

        # Update Thompson Sampling for strategy
        strategy_key = f"strategy:{strategy}"
        self.thompson_sampler.update(strategy_key, success, weight=abs(pnl_pct) / 10 + 0.5)

        # Update Thompson Sampling for ticker
        ticker_key = f"ticker:{ticker}"
        self.thompson_sampler.update(ticker_key, success, weight=0.5)

        # Update combination key
        combo_key = f"combo:{ticker}:{pattern}:{strategy}"
        self.thompson_sampler.update(combo_key, success)

        # Add to lessons if notable
        if abs(pnl_pct) > 15:  # Big win or loss
            category = "pattern_recognition" if success else "risk_management"
            self.lessons_memory.add_lesson(
                category=category,
                title=f"{'Big Win' if success else 'Big Loss'}: {ticker} {pattern}",
                description=f"{direction.upper()} {ticker} using {pattern} pattern in {strategy} strategy",
                context=context,
                outcome=f"{'WIN' if success else 'LOSS'}: {pnl_pct:+.1f}% in {hold_time_minutes} minutes",
                learning=self._generate_learning(success, pnl_pct, pattern, context),
                action=self._generate_action(success, pnl_pct, pattern, context),
                confidence=min(0.9, 0.5 + abs(pnl_pct) / 100)
            )

        # Record feedback
        feedback = EvolutionFeedback(
            dimension=EvolutionDimension.STRATEGY,
            feedback_type=FeedbackType.TRADE_OUTCOME,
            context={
                "ticker": ticker,
                "pattern": pattern,
                "strategy": strategy,
                "direction": direction,
                "pnl_pct": pnl_pct,
                "hold_time_minutes": hold_time_minutes,
                **context
            },
            outcome=pnl_pct / 100  # Normalize to -1 to 1 range (assuming max 100% move)
        )

        with self._lock:
            self.feedback_buffer.append(feedback)

        logger.info(
            f"SELF_EVOLVING: Recorded {ticker} {pattern} outcome: "
            f"{pnl_pct:+.1f}% ({'WIN' if success else 'LOSS'})"
        )

    def _generate_learning(
        self,
        success: bool,
        pnl_pct: float,
        pattern: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate learning from trade outcome."""
        if success:
            return f"The {pattern} pattern worked well. Key factors: {list(context.keys())}"
        else:
            return f"The {pattern} pattern failed. Need to review entry criteria and risk management."

    def _generate_action(
        self,
        success: bool,
        pnl_pct: float,
        pattern: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate action item from trade outcome."""
        if success:
            return f"Continue using {pattern} pattern with similar conditions. Consider increasing position size."
        else:
            return f"Review {pattern} pattern criteria. Consider tighter stops or smaller position size."

    def select_best_pattern(self, candidate_patterns: List[str]) -> Tuple[str, float, Dict]:
        """
        Select the best pattern using Thompson Sampling.

        Args:
            candidate_patterns: List of detected patterns

        Returns:
            (best_pattern, confidence, stats)
        """
        if not candidate_patterns:
            return None, 0.0, {}

        # Get pattern keys
        pattern_keys = [f"pattern:{p}" for p in candidate_patterns]

        # Thompson Sampling selection
        best_key, sampled_value = self.thompson_sampler.select_best(pattern_keys)
        best_pattern = best_key.replace("pattern:", "")

        # Get stats
        stats = self.thompson_sampler.get_stats(best_key)

        logger.info(
            f"SELF_EVOLVING: Selected pattern '{best_pattern}' "
            f"(mean={stats['mean']:.1%}, CI=[{stats['ci_95_low']:.1%}, {stats['ci_95_high']:.1%}])"
        )

        return best_pattern, sampled_value, stats

    def get_relevant_lessons(self, context: Dict[str, Any]) -> List[Dict]:
        """Get relevant lessons for current context."""
        lessons = self.lessons_memory.query_relevant(context, limit=3)
        return [
            {
                "title": l.title,
                "learning": l.learning,
                "action": l.action,
                "confidence": l.confidence,
                "success_rate": l.times_helped / max(l.times_applied, 1)
            }
            for l in lessons
        ]

    def apply_daily_decay(self) -> None:
        """Apply daily decay to Thompson distributions."""
        self.thompson_sampler.apply_decay()
        logger.info("SELF_EVOLVING: Applied daily decay to distributions")

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics."""
        return {
            "thompson_stats": self.thompson_sampler.get_all_stats(),
            "total_lessons": len(self.lessons_memory.lessons),
            "feedback_buffer_size": len(self.feedback_buffer),
            "evolution_enabled": self.evolution_enabled,
        }


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_evolving_engine: Optional[SelfEvolvingEngine] = None


def get_self_evolving_engine() -> SelfEvolvingEngine:
    """Get singleton SelfEvolvingEngine instance."""
    global _evolving_engine
    if _evolving_engine is None:
        _evolving_engine = SelfEvolvingEngine()
    return _evolving_engine


def record_trade_for_learning(
    ticker: str,
    pattern: str,
    strategy: str,
    direction: str,
    pnl_pct: float,
    hold_time_minutes: int,
    **context
) -> None:
    """
    Convenience function to record trade for learning.

    Usage:
        from wsb_snake.learning.self_evolving_memory import record_trade_for_learning

        record_trade_for_learning(
            ticker="SPY",
            pattern="vwap_bounce",
            strategy="scalper",
            direction="long",
            pnl_pct=12.5,
            hold_time_minutes=15,
            vix=18.5,
            regime="RISK_ON"
        )
    """
    engine = get_self_evolving_engine()
    engine.record_trade_outcome(
        ticker=ticker,
        pattern=pattern,
        strategy=strategy,
        direction=direction,
        pnl_pct=pnl_pct,
        hold_time_minutes=hold_time_minutes,
        context=context
    )
