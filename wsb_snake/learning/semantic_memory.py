"""
Semantic Memory Module - Vector-based Trade Memory with Auto-Reflection

Based on qrak/LLM_trader patterns:
- ChromaDB vector storage for semantic similarity search
- Recency-weighted retrieval (similarity × 0.7 + recency × 0.3)
- Auto-reflection loop every N trades
- Learned rules injection into prompts

Key Features:
1. Store trades with 15+ metadata fields
2. Find similar past setups by embedding current conditions
3. Generate positive rules from winning patterns
4. Generate anti-patterns from losing streaks
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TradeConditions:
    """Market conditions at trade entry."""

    ticker: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    rsi: float
    adx: float
    atr: float
    macd_signal: str  # "bullish", "bearish", "neutral"
    volume_ratio: float  # Current vs average
    regime: str  # "bull", "bear", "chop"
    vix: float
    gex_regime: str  # "positive_gamma", "negative_gamma"
    hydra_direction: str
    confluence_score: float
    stop_distance_pct: float
    target_distance_pct: float


@dataclass
class TradeOutcome:
    """Final trade outcome for learning."""

    trade_id: str
    conditions: TradeConditions
    entry_reasoning: str  # Full AI reasoning at entry
    pnl_dollars: float
    pnl_percent: float
    duration_minutes: int
    max_adverse_excursion_pct: float  # Worst drawdown during trade
    max_favorable_excursion_pct: float  # Best unrealized gain
    exit_reason: str  # "target", "stop", "time", "manual"
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    lessons_learned: str = ""  # Post-trade reflection


@dataclass
class SemanticRule:
    """Learned rule from trade patterns."""

    rule_id: str
    rule_type: str  # "positive" or "avoid"
    pattern_description: str
    conditions: dict  # Conditions that trigger this rule
    confidence: float
    win_rate: float
    sample_size: int
    created_at: datetime
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class SimilarTrade:
    """A similar past trade from semantic search."""

    trade: TradeOutcome
    similarity_score: float
    recency_score: float
    combined_score: float


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC MEMORY ENGINE
# ─────────────────────────────────────────────────────────────────────────────


class SemanticMemory:
    """
    Vector-based trade memory with semantic similarity search.

    Uses simple embedding approach without external dependencies.
    Can be upgraded to ChromaDB for production use.
    """

    def __init__(
        self,
        storage_path: str = "wsb_snake_data/semantic_memory.json",
        recency_half_life_days: int = 90,
        similarity_weight: float = 0.7,
        recency_weight: float = 0.3,
        reflection_threshold: int = 10,  # Reflect every N trades
    ):
        self.storage_path = Path(storage_path)
        self.recency_half_life_days = recency_half_life_days
        self.similarity_weight = similarity_weight
        self.recency_weight = recency_weight
        self.reflection_threshold = reflection_threshold

        # In-memory storage
        self.trades: list[TradeOutcome] = []
        self.rules: list[SemanticRule] = []
        self.trades_since_reflection = 0

        # Load existing data
        self._load_from_disk()

    def _load_from_disk(self):
        """Load saved memory from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            # Parse trades
            for trade_data in data.get("trades", []):
                conditions = TradeConditions(**trade_data["conditions"])
                trade_data["conditions"] = conditions
                trade_data["entry_time"] = datetime.fromisoformat(
                    trade_data["entry_time"]
                )
                trade_data["exit_time"] = datetime.fromisoformat(
                    trade_data["exit_time"]
                )
                self.trades.append(TradeOutcome(**trade_data))

            # Parse rules
            for rule_data in data.get("rules", []):
                rule_data["created_at"] = datetime.fromisoformat(
                    rule_data["created_at"]
                )
                if rule_data.get("last_triggered"):
                    rule_data["last_triggered"] = datetime.fromisoformat(
                        rule_data["last_triggered"]
                    )
                self.rules.append(SemanticRule(**rule_data))

            self.trades_since_reflection = data.get("trades_since_reflection", 0)

            log.info(
                f"Loaded {len(self.trades)} trades and {len(self.rules)} rules from memory"
            )

        except Exception as e:
            log.error(f"Error loading semantic memory: {e}")

    def _save_to_disk(self):
        """Save memory to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable format
            trades_data = []
            for trade in self.trades:
                trade_dict = {
                    "trade_id": trade.trade_id,
                    "conditions": asdict(trade.conditions),
                    "entry_reasoning": trade.entry_reasoning,
                    "pnl_dollars": trade.pnl_dollars,
                    "pnl_percent": trade.pnl_percent,
                    "duration_minutes": trade.duration_minutes,
                    "max_adverse_excursion_pct": trade.max_adverse_excursion_pct,
                    "max_favorable_excursion_pct": trade.max_favorable_excursion_pct,
                    "exit_reason": trade.exit_reason,
                    "exit_price": trade.exit_price,
                    "entry_time": trade.entry_time.isoformat(),
                    "exit_time": trade.exit_time.isoformat(),
                    "lessons_learned": trade.lessons_learned,
                }
                trades_data.append(trade_dict)

            rules_data = []
            for rule in self.rules:
                rule_dict = {
                    "rule_id": rule.rule_id,
                    "rule_type": rule.rule_type,
                    "pattern_description": rule.pattern_description,
                    "conditions": rule.conditions,
                    "confidence": rule.confidence,
                    "win_rate": rule.win_rate,
                    "sample_size": rule.sample_size,
                    "created_at": rule.created_at.isoformat(),
                    "last_triggered": (
                        rule.last_triggered.isoformat() if rule.last_triggered else None
                    ),
                    "trigger_count": rule.trigger_count,
                }
                rules_data.append(rule_dict)

            data = {
                "trades": trades_data,
                "rules": rules_data,
                "trades_since_reflection": self.trades_since_reflection,
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            log.error(f"Error saving semantic memory: {e}")

    def record_trade(self, trade: TradeOutcome):
        """
        Record a completed trade for learning.

        Automatically triggers reflection if threshold reached.
        """
        self.trades.append(trade)
        self.trades_since_reflection += 1

        log.info(
            f"Recorded trade {trade.trade_id}: {trade.conditions.ticker} "
            f"{'WIN' if trade.pnl_dollars > 0 else 'LOSS'} ${trade.pnl_dollars:.2f}"
        )

        # Check if reflection needed
        if self.trades_since_reflection >= self.reflection_threshold:
            self._auto_reflect()
            self.trades_since_reflection = 0

        self._save_to_disk()

    def _compute_embedding(self, conditions: TradeConditions) -> list[float]:
        """
        Compute a simple feature vector for similarity comparison.

        In production, use sentence-transformers for text embeddings.
        This is a lightweight numeric approach.
        """
        # Normalize features to [0, 1] range
        features = [
            conditions.rsi / 100.0,  # RSI: 0-100
            conditions.adx / 100.0,  # ADX: 0-100
            min(conditions.atr / 10.0, 1.0),  # ATR normalized
            1.0 if conditions.macd_signal == "bullish" else 0.0,
            0.5 if conditions.macd_signal == "neutral" else 0.0,
            min(conditions.volume_ratio / 3.0, 1.0),  # Volume ratio capped at 3x
            1.0 if conditions.regime == "bull" else 0.0,
            1.0 if conditions.regime == "bear" else 0.0,
            min(conditions.vix / 50.0, 1.0),  # VIX capped at 50
            1.0 if conditions.gex_regime == "positive_gamma" else 0.0,
            1.0 if conditions.hydra_direction == "BULLISH" else 0.0,
            1.0 if conditions.hydra_direction == "BEARISH" else 0.0,
            conditions.confluence_score,  # Already 0-1
            min(conditions.stop_distance_pct / 5.0, 1.0),  # Stop % capped
            min(conditions.target_distance_pct / 10.0, 1.0),  # Target % capped
        ]
        return features

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _compute_recency_score(self, trade_time: datetime) -> float:
        """
        Compute recency score with exponential decay.

        Score = 0.5^(days_ago / half_life)
        """
        days_ago = (datetime.now() - trade_time).days
        return math.pow(0.5, days_ago / self.recency_half_life_days)

    def find_similar_trades(
        self,
        current_conditions: TradeConditions,
        top_k: int = 5,
        outcome_filter: Optional[str] = None,  # "win", "loss", or None
    ) -> list[SimilarTrade]:
        """
        Find most similar past trades using semantic similarity.

        Scoring: combined = similarity × 0.7 + recency × 0.3

        Args:
            current_conditions: Current market conditions
            top_k: Number of similar trades to return
            outcome_filter: Filter to only wins or losses

        Returns:
            List of SimilarTrade objects sorted by combined score
        """
        current_embedding = self._compute_embedding(current_conditions)
        results = []

        for trade in self.trades:
            # Apply outcome filter
            if outcome_filter == "win" and trade.pnl_dollars <= 0:
                continue
            if outcome_filter == "loss" and trade.pnl_dollars > 0:
                continue

            trade_embedding = self._compute_embedding(trade.conditions)
            similarity = self._cosine_similarity(current_embedding, trade_embedding)
            recency = self._compute_recency_score(trade.entry_time)

            combined = (
                similarity * self.similarity_weight + recency * self.recency_weight
            )

            results.append(
                SimilarTrade(
                    trade=trade,
                    similarity_score=similarity,
                    recency_score=recency,
                    combined_score=combined,
                )
            )

        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)

        return results[:top_k]

    def get_relevant_rules(
        self, current_conditions: TradeConditions
    ) -> list[SemanticRule]:
        """
        Get rules that apply to current conditions.

        Returns both positive rules and anti-patterns.
        """
        relevant = []

        for rule in self.rules:
            if self._rule_matches_conditions(rule, current_conditions):
                relevant.append(rule)
                # Track trigger
                rule.last_triggered = datetime.now()
                rule.trigger_count += 1

        self._save_to_disk()
        return relevant

    def _rule_matches_conditions(
        self, rule: SemanticRule, conditions: TradeConditions
    ) -> bool:
        """Check if a rule's conditions match current market state."""
        rule_conds = rule.conditions

        # Check each specified condition
        for key, value in rule_conds.items():
            if key == "regime" and conditions.regime != value:
                return False
            if key == "rsi_above" and conditions.rsi <= value:
                return False
            if key == "rsi_below" and conditions.rsi >= value:
                return False
            if key == "adx_above" and conditions.adx <= value:
                return False
            if key == "adx_below" and conditions.adx >= value:
                return False
            if key == "vix_above" and conditions.vix <= value:
                return False
            if key == "vix_below" and conditions.vix >= value:
                return False
            if key == "gex_regime" and conditions.gex_regime != value:
                return False
            if key == "direction" and conditions.direction != value:
                return False

        return True

    def _auto_reflect(self):
        """
        Auto-reflection loop - analyze recent trades and extract patterns.

        Called automatically every N trades.
        """
        log.info(f"Running auto-reflection on last {self.reflection_threshold} trades")

        recent_trades = self.trades[-self.reflection_threshold :]
        wins = [t for t in recent_trades if t.pnl_dollars > 0]
        losses = [t for t in recent_trades if t.pnl_dollars <= 0]

        # Generate positive rule if 5+ wins share a pattern
        if len(wins) >= 5:
            positive_rule = self._synthesize_pattern(wins, "positive")
            if positive_rule:
                self.rules.append(positive_rule)
                log.info(
                    f"Generated positive rule: {positive_rule.pattern_description}"
                )

        # Generate anti-pattern if 3+ losses share traits
        if len(losses) >= 3:
            avoid_rule = self._synthesize_pattern(losses, "avoid")
            if avoid_rule:
                self.rules.append(avoid_rule)
                log.warning(f"Generated AVOID rule: {avoid_rule.pattern_description}")

    def _synthesize_pattern(
        self, trades: list[TradeOutcome], rule_type: str
    ) -> Optional[SemanticRule]:
        """
        Synthesize a pattern from a set of similar trades.

        Looks for common conditions across the trades.
        """
        if not trades:
            return None

        # Collect condition statistics
        regimes = {}
        directions = {}
        rsi_values = []
        adx_values = []
        vix_values = []
        gex_regimes = {}

        for trade in trades:
            c = trade.conditions

            regimes[c.regime] = regimes.get(c.regime, 0) + 1
            directions[c.direction] = directions.get(c.direction, 0) + 1
            gex_regimes[c.gex_regime] = gex_regimes.get(c.gex_regime, 0) + 1
            rsi_values.append(c.rsi)
            adx_values.append(c.adx)
            vix_values.append(c.vix)

        # Find dominant patterns (>60% of trades)
        threshold = len(trades) * 0.6
        conditions = {}
        description_parts = []

        # Check regime consistency
        dominant_regime = max(regimes, key=regimes.get)
        if regimes[dominant_regime] >= threshold:
            conditions["regime"] = dominant_regime
            description_parts.append(f"regime={dominant_regime}")

        # Check direction consistency
        dominant_direction = max(directions, key=directions.get)
        if directions[dominant_direction] >= threshold:
            conditions["direction"] = dominant_direction
            description_parts.append(f"direction={dominant_direction}")

        # Check GEX regime
        dominant_gex = max(gex_regimes, key=gex_regimes.get)
        if gex_regimes[dominant_gex] >= threshold:
            conditions["gex_regime"] = dominant_gex
            description_parts.append(f"gex={dominant_gex}")

        # Check RSI range
        avg_rsi = sum(rsi_values) / len(rsi_values)
        if all(r > 70 for r in rsi_values):
            conditions["rsi_above"] = 70
            description_parts.append("RSI>70")
        elif all(r < 30 for r in rsi_values):
            conditions["rsi_below"] = 30
            description_parts.append("RSI<30")

        # Check VIX range
        avg_vix = sum(vix_values) / len(vix_values)
        if all(v > 25 for v in vix_values):
            conditions["vix_above"] = 25
            description_parts.append("VIX>25")

        # Need at least one condition to make a valid rule
        if not conditions:
            return None

        # Calculate win rate
        wins = sum(1 for t in trades if t.pnl_dollars > 0)
        win_rate = wins / len(trades)

        # Generate description
        if rule_type == "positive":
            description = f"Pattern works well: {', '.join(description_parts)}"
        else:
            description = f"AVOID: {', '.join(description_parts)}"

        # Generate unique ID
        rule_id = hashlib.md5(
            f"{rule_type}_{json.dumps(conditions)}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        return SemanticRule(
            rule_id=rule_id,
            rule_type=rule_type,
            pattern_description=description,
            conditions=conditions,
            confidence=0.6 if len(trades) >= 5 else 0.4,
            win_rate=win_rate,
            sample_size=len(trades),
            created_at=datetime.now(),
        )

    def get_prompt_injection(self, current_conditions: TradeConditions) -> str:
        """
        Generate text to inject into AI prompts based on learned rules.

        Returns formatted rules and similar trade insights.
        """
        lines = []

        # Get relevant rules
        rules = self.get_relevant_rules(current_conditions)
        if rules:
            lines.append("LEARNED RULES (from past trades):")
            for rule in rules:
                if rule.rule_type == "positive":
                    lines.append(f"  ✓ {rule.pattern_description} (win rate: {rule.win_rate:.0%})")
                else:
                    lines.append(f"  ⚠ {rule.pattern_description} (avoid this pattern)")

        # Get similar trades
        similar = self.find_similar_trades(current_conditions, top_k=3)
        if similar:
            lines.append("\nSIMILAR PAST TRADES:")
            for st in similar:
                outcome = "WIN" if st.trade.pnl_dollars > 0 else "LOSS"
                lines.append(
                    f"  • {st.trade.conditions.ticker} {outcome} "
                    f"${st.trade.pnl_dollars:.2f} ({st.similarity_score:.0%} similar)"
                )
                if st.trade.lessons_learned:
                    lines.append(f"    Lesson: {st.trade.lessons_learned}")

        return "\n".join(lines) if lines else ""

    def get_stats(self) -> dict:
        """Get memory statistics."""
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t.pnl_dollars > 0)
        total_pnl = sum(t.pnl_dollars for t in self.trades)

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": total_trades - wins,
            "win_rate": wins / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "rules_count": len(self.rules),
            "positive_rules": sum(1 for r in self.rules if r.rule_type == "positive"),
            "avoid_rules": sum(1 for r in self.rules if r.rule_type == "avoid"),
            "trades_until_reflection": self.reflection_threshold
            - self.trades_since_reflection,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

_semantic_memory: Optional[SemanticMemory] = None


def get_semantic_memory() -> SemanticMemory:
    """Get singleton semantic memory instance."""
    global _semantic_memory
    if _semantic_memory is None:
        _semantic_memory = SemanticMemory()
    return _semantic_memory
