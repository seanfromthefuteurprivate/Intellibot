"""
Trade Graph Memory - Relationship-Based Trade Analysis

Based on Open-Finance-Lab/AgenticTrading patterns:
- Graph-based trade memory for relationship modeling
- Execution trace storage (full reasoning preserved)
- Pattern discovery through graph queries
- Audit pool for post-trade analysis

Unlike flat record storage, this captures relationships:
- Trade → Conditions (what triggered it)
- Trade → Outcome (what happened)
- Trade → Similar Trades (related setups)
- Trade → Lessons (what we learned)

Lightweight implementation without Neo4j dependency.
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from collections import defaultdict

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH NODE TYPES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TradeNode:
    """A trade in the graph."""
    node_id: str
    ticker: str
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl_dollars: float
    pnl_percent: float
    pattern: str
    duration_minutes: int
    entry_reasoning: str  # Full AI reasoning at entry
    exit_reasoning: str   # Why we exited
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConditionNode:
    """Market conditions at a point in time."""
    node_id: str
    timestamp: datetime
    ticker: str
    price: float
    rsi: float
    adx: float
    vix: float
    regime: str
    gex_regime: str
    hydra_direction: str
    flow_bias: str
    volume_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LessonNode:
    """A lesson learned from trade outcomes."""
    node_id: str
    lesson_type: str  # "success_pattern", "failure_pattern", "risk_warning"
    description: str
    conditions: Dict[str, Any]  # Conditions that trigger this lesson
    confidence: float
    sample_size: int
    win_rate: float
    created_at: datetime
    last_triggered: Optional[datetime] = None


@dataclass
class Edge:
    """Relationship between nodes."""
    edge_id: str
    source_id: str
    target_id: str
    relationship: str  # "triggered_by", "resulted_in", "similar_to", "learned_from"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# TRADE GRAPH
# ─────────────────────────────────────────────────────────────────────────────


class TradeGraph:
    """
    Graph-based trade memory for relationship analysis.

    Enables queries like:
    - "Find all trades that had similar conditions to current"
    - "What lessons were learned when RSI > 70 in bull regime?"
    - "Which patterns work best with HYDRA BULLISH?"
    """

    def __init__(
        self,
        storage_path: str = "wsb_snake_data/trade_graph.json",
        similarity_threshold: float = 0.7,
    ):
        self.storage_path = Path(storage_path)
        self.similarity_threshold = similarity_threshold

        # Graph storage
        self._trades: Dict[str, TradeNode] = {}
        self._conditions: Dict[str, ConditionNode] = {}
        self._lessons: Dict[str, LessonNode] = {}
        self._edges: Dict[str, Edge] = {}

        # Indexes for fast lookup
        self._ticker_index: Dict[str, Set[str]] = defaultdict(set)
        self._pattern_index: Dict[str, Set[str]] = defaultdict(set)
        self._regime_index: Dict[str, Set[str]] = defaultdict(set)
        self._outcome_index: Dict[str, Set[str]] = defaultdict(set)  # "win" or "loss"

        self._load_from_disk()
        logger.info(f"TradeGraph initialized: {len(self._trades)} trades, {len(self._lessons)} lessons")

    def _load_from_disk(self):
        """Load graph from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            # Load trades
            for trade_data in data.get("trades", []):
                trade_data["entry_time"] = datetime.fromisoformat(trade_data["entry_time"])
                trade_data["exit_time"] = datetime.fromisoformat(trade_data["exit_time"])
                trade = TradeNode(**trade_data)
                self._trades[trade.node_id] = trade
                self._index_trade(trade)

            # Load conditions
            for cond_data in data.get("conditions", []):
                cond_data["timestamp"] = datetime.fromisoformat(cond_data["timestamp"])
                cond = ConditionNode(**cond_data)
                self._conditions[cond.node_id] = cond

            # Load lessons
            for lesson_data in data.get("lessons", []):
                lesson_data["created_at"] = datetime.fromisoformat(lesson_data["created_at"])
                if lesson_data.get("last_triggered"):
                    lesson_data["last_triggered"] = datetime.fromisoformat(lesson_data["last_triggered"])
                lesson = LessonNode(**lesson_data)
                self._lessons[lesson.node_id] = lesson

            # Load edges
            for edge_data in data.get("edges", []):
                edge = Edge(**edge_data)
                self._edges[edge.edge_id] = edge

        except Exception as e:
            logger.error(f"TradeGraph: Load failed - {e}")

    def _save_to_disk(self):
        """Save graph to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "trades": [],
                "conditions": [],
                "lessons": [],
                "edges": [],
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

            for trade in self._trades.values():
                trade_dict = asdict(trade)
                trade_dict["entry_time"] = trade.entry_time.isoformat()
                trade_dict["exit_time"] = trade.exit_time.isoformat()
                data["trades"].append(trade_dict)

            for cond in self._conditions.values():
                cond_dict = asdict(cond)
                cond_dict["timestamp"] = cond.timestamp.isoformat()
                data["conditions"].append(cond_dict)

            for lesson in self._lessons.values():
                lesson_dict = asdict(lesson)
                lesson_dict["created_at"] = lesson.created_at.isoformat()
                if lesson.last_triggered:
                    lesson_dict["last_triggered"] = lesson.last_triggered.isoformat()
                data["lessons"].append(lesson_dict)

            for edge in self._edges.values():
                data["edges"].append(asdict(edge))

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"TradeGraph: Save failed - {e}")

    def _index_trade(self, trade: TradeNode):
        """Add trade to indexes."""
        self._ticker_index[trade.ticker].add(trade.node_id)
        self._pattern_index[trade.pattern].add(trade.node_id)
        outcome = "win" if trade.pnl_dollars > 0 else "loss"
        self._outcome_index[outcome].add(trade.node_id)

    def _generate_id(self, prefix: str, data: str) -> str:
        """Generate unique ID."""
        hash_input = f"{prefix}_{data}_{datetime.now().isoformat()}"
        return f"{prefix}_{hashlib.md5(hash_input.encode()).hexdigest()[:12]}"

    # ─────────────────────────────────────────────────────────────────────────
    # RECORDING METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def record_trade(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
        pnl_dollars: float,
        pattern: str,
        entry_reasoning: str,
        exit_reasoning: str,
        conditions: Dict[str, Any],
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Record a completed trade with its conditions.

        Creates:
        - TradeNode for the trade
        - ConditionNode for market state at entry
        - Edge linking trade to conditions

        Returns:
            Trade node ID
        """
        trade_id = self._generate_id("trade", f"{ticker}_{entry_time.isoformat()}")

        # Calculate derived values
        duration = int((exit_time - entry_time).total_seconds() / 60)
        pnl_percent = ((exit_price - entry_price) / entry_price * 100) if direction == "long" else (
            (entry_price - exit_price) / entry_price * 100
        )

        # Create trade node
        trade = TradeNode(
            node_id=trade_id,
            ticker=ticker,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl_dollars=pnl_dollars,
            pnl_percent=pnl_percent,
            pattern=pattern,
            duration_minutes=duration,
            entry_reasoning=entry_reasoning,
            exit_reasoning=exit_reasoning,
            metadata=metadata or {},
        )

        self._trades[trade_id] = trade
        self._index_trade(trade)

        # Create condition node
        cond_id = self._generate_id("cond", f"{ticker}_{entry_time.isoformat()}")
        condition = ConditionNode(
            node_id=cond_id,
            timestamp=entry_time,
            ticker=ticker,
            price=entry_price,
            rsi=conditions.get("rsi", 50),
            adx=conditions.get("adx", 20),
            vix=conditions.get("vix", 20),
            regime=conditions.get("regime", "unknown"),
            gex_regime=conditions.get("gex_regime", "unknown"),
            hydra_direction=conditions.get("hydra_direction", "NEUTRAL"),
            flow_bias=conditions.get("flow_bias", "NEUTRAL"),
            volume_ratio=conditions.get("volume_ratio", 1.0),
            metadata=conditions,
        )

        self._conditions[cond_id] = condition
        self._regime_index[condition.regime].add(cond_id)

        # Create edge: trade triggered_by conditions
        edge_id = self._generate_id("edge", f"{trade_id}_{cond_id}")
        edge = Edge(
            edge_id=edge_id,
            source_id=trade_id,
            target_id=cond_id,
            relationship="triggered_by",
        )
        self._edges[edge_id] = edge

        # Find similar trades and create edges
        similar = self.find_similar_trades(conditions, top_k=3)
        for sim_trade, similarity in similar:
            if sim_trade.node_id != trade_id:
                sim_edge_id = self._generate_id("edge", f"{trade_id}_{sim_trade.node_id}")
                sim_edge = Edge(
                    edge_id=sim_edge_id,
                    source_id=trade_id,
                    target_id=sim_trade.node_id,
                    relationship="similar_to",
                    weight=similarity,
                )
                self._edges[sim_edge_id] = sim_edge

        # Run audit and generate lessons
        self._audit_trade(trade, condition)

        self._save_to_disk()

        logger.info(
            f"TradeGraph: Recorded {ticker} {direction} "
            f"{'WIN' if pnl_dollars > 0 else 'LOSS'} ${pnl_dollars:.2f} "
            f"(trade_id={trade_id})"
        )

        return trade_id

    def _audit_trade(self, trade: TradeNode, condition: ConditionNode):
        """
        Audit pool pattern - analyze trade for lessons.

        Creates LessonNodes when patterns are detected.
        """
        # Check if we have enough similar trades to identify patterns
        similar_outcomes = self.query_trades(
            pattern=trade.pattern,
            regime=condition.regime,
            min_trades=3,
        )

        if len(similar_outcomes) >= 5:
            # Enough data to analyze pattern
            wins = [t for t in similar_outcomes if t.pnl_dollars > 0]
            losses = [t for t in similar_outcomes if t.pnl_dollars <= 0]
            win_rate = len(wins) / len(similar_outcomes)

            # Create or update lesson
            lesson_key = f"{trade.pattern}_{condition.regime}"

            if win_rate > 0.65:
                # Success pattern
                lesson_id = self._generate_id("lesson", f"success_{lesson_key}")
                lesson = LessonNode(
                    node_id=lesson_id,
                    lesson_type="success_pattern",
                    description=f"Pattern '{trade.pattern}' works well in '{condition.regime}' regime ({win_rate:.0%} win rate)",
                    conditions={
                        "pattern": trade.pattern,
                        "regime": condition.regime,
                    },
                    confidence=min(0.9, 0.5 + win_rate * 0.4),
                    sample_size=len(similar_outcomes),
                    win_rate=win_rate,
                    created_at=datetime.now(timezone.utc),
                )
                self._lessons[lesson_id] = lesson

                # Link lesson to trade
                edge_id = self._generate_id("edge", f"{trade.node_id}_{lesson_id}")
                self._edges[edge_id] = Edge(
                    edge_id=edge_id,
                    source_id=trade.node_id,
                    target_id=lesson_id,
                    relationship="learned_from",
                )

                logger.info(f"TradeGraph: Created success pattern lesson for {lesson_key}")

            elif win_rate < 0.35:
                # Failure pattern
                lesson_id = self._generate_id("lesson", f"failure_{lesson_key}")
                lesson = LessonNode(
                    node_id=lesson_id,
                    lesson_type="failure_pattern",
                    description=f"AVOID: Pattern '{trade.pattern}' fails in '{condition.regime}' regime ({win_rate:.0%} win rate)",
                    conditions={
                        "pattern": trade.pattern,
                        "regime": condition.regime,
                    },
                    confidence=min(0.9, 0.5 + (1 - win_rate) * 0.4),
                    sample_size=len(similar_outcomes),
                    win_rate=win_rate,
                    created_at=datetime.now(timezone.utc),
                )
                self._lessons[lesson_id] = lesson
                logger.warning(f"TradeGraph: Created failure pattern lesson for {lesson_key}")

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def find_similar_trades(
        self,
        conditions: Dict[str, Any],
        top_k: int = 5,
        outcome_filter: Optional[str] = None,  # "win" or "loss"
    ) -> List[Tuple[TradeNode, float]]:
        """
        Find trades with similar conditions.

        Args:
            conditions: Current market conditions
            top_k: Number of similar trades to return
            outcome_filter: Filter to only wins or losses

        Returns:
            List of (TradeNode, similarity_score) tuples
        """
        results = []

        for trade in self._trades.values():
            # Apply outcome filter
            if outcome_filter == "win" and trade.pnl_dollars <= 0:
                continue
            if outcome_filter == "loss" and trade.pnl_dollars > 0:
                continue

            # Find the condition node for this trade
            trade_condition = self._get_trade_condition(trade.node_id)
            if not trade_condition:
                continue

            # Calculate similarity
            similarity = self._calculate_condition_similarity(conditions, trade_condition)

            if similarity >= self.similarity_threshold:
                results.append((trade, similarity))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _get_trade_condition(self, trade_id: str) -> Optional[ConditionNode]:
        """Get the condition node linked to a trade."""
        for edge in self._edges.values():
            if edge.source_id == trade_id and edge.relationship == "triggered_by":
                return self._conditions.get(edge.target_id)
        return None

    def _calculate_condition_similarity(
        self,
        current: Dict[str, Any],
        historical: ConditionNode,
    ) -> float:
        """
        Calculate similarity between current conditions and historical.

        Uses weighted feature matching.
        """
        score = 0.0
        weights = {
            "regime": 0.25,
            "hydra_direction": 0.20,
            "rsi_bucket": 0.15,
            "flow_bias": 0.15,
            "gex_regime": 0.15,
            "volume_bucket": 0.10,
        }

        # Regime match
        if current.get("regime", "").lower() == historical.regime.lower():
            score += weights["regime"]

        # HYDRA direction match
        if current.get("hydra_direction", "") == historical.hydra_direction:
            score += weights["hydra_direction"]

        # RSI bucket match (oversold/neutral/overbought)
        current_rsi = current.get("rsi", 50)
        hist_rsi = historical.rsi
        current_bucket = "oversold" if current_rsi < 30 else ("overbought" if current_rsi > 70 else "neutral")
        hist_bucket = "oversold" if hist_rsi < 30 else ("overbought" if hist_rsi > 70 else "neutral")
        if current_bucket == hist_bucket:
            score += weights["rsi_bucket"]

        # Flow bias match
        if current.get("flow_bias", "") == historical.flow_bias:
            score += weights["flow_bias"]

        # GEX regime match
        if current.get("gex_regime", "") == historical.gex_regime:
            score += weights["gex_regime"]

        # Volume bucket match
        current_vol = current.get("volume_ratio", 1.0)
        hist_vol = historical.volume_ratio
        current_vol_bucket = "high" if current_vol > 1.5 else ("low" if current_vol < 0.8 else "normal")
        hist_vol_bucket = "high" if hist_vol > 1.5 else ("low" if hist_vol < 0.8 else "normal")
        if current_vol_bucket == hist_vol_bucket:
            score += weights["volume_bucket"]

        return score

    def query_trades(
        self,
        ticker: Optional[str] = None,
        pattern: Optional[str] = None,
        regime: Optional[str] = None,
        outcome: Optional[str] = None,  # "win" or "loss"
        min_trades: int = 0,
    ) -> List[TradeNode]:
        """
        Query trades matching criteria.

        Args:
            ticker: Filter by ticker
            pattern: Filter by pattern
            regime: Filter by regime
            outcome: Filter by outcome
            min_trades: Minimum number to return (returns empty if not met)

        Returns:
            List of matching TradeNode objects
        """
        candidates = set(self._trades.keys())

        # Apply filters
        if ticker:
            candidates &= self._ticker_index.get(ticker, set())

        if pattern:
            candidates &= self._pattern_index.get(pattern, set())

        if outcome:
            candidates &= self._outcome_index.get(outcome, set())

        if regime:
            # Need to check conditions
            regime_trades = set()
            for trade_id in candidates:
                cond = self._get_trade_condition(trade_id)
                if cond and cond.regime.lower() == regime.lower():
                    regime_trades.add(trade_id)
            candidates = regime_trades

        results = [self._trades[tid] for tid in candidates]

        if len(results) < min_trades:
            return []

        return results

    def get_applicable_lessons(
        self,
        conditions: Dict[str, Any],
    ) -> List[LessonNode]:
        """
        Get lessons that apply to current conditions.

        Args:
            conditions: Current market conditions

        Returns:
            List of applicable LessonNode objects
        """
        applicable = []

        for lesson in self._lessons.values():
            matches = True
            for key, value in lesson.conditions.items():
                if conditions.get(key) != value:
                    matches = False
                    break

            if matches:
                applicable.append(lesson)
                # Update last triggered
                lesson.last_triggered = datetime.now(timezone.utc)

        return applicable

    def get_lessons_summary(self, conditions: Dict[str, Any]) -> str:
        """Get human-readable lessons summary for current conditions."""
        lessons = self.get_applicable_lessons(conditions)

        if not lessons:
            return ""

        lines = ["GRAPH MEMORY LESSONS:"]
        for lesson in lessons:
            if lesson.lesson_type == "success_pattern":
                lines.append(f"  ✓ {lesson.description} (n={lesson.sample_size})")
            elif lesson.lesson_type == "failure_pattern":
                lines.append(f"  ⚠ {lesson.description} (n={lesson.sample_size})")
            else:
                lines.append(f"  • {lesson.description}")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def get_pattern_stats(self, pattern: str) -> Dict[str, Any]:
        """Get statistics for a specific pattern."""
        trades = self.query_trades(pattern=pattern)

        if not trades:
            return {"pattern": pattern, "trades": 0}

        wins = [t for t in trades if t.pnl_dollars > 0]
        total_pnl = sum(t.pnl_dollars for t in trades)
        avg_duration = sum(t.duration_minutes for t in trades) / len(trades)

        return {
            "pattern": pattern,
            "trades": len(trades),
            "wins": len(wins),
            "losses": len(trades) - len(wins),
            "win_rate": len(wins) / len(trades),
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(trades),
            "avg_duration_minutes": avg_duration,
        }

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get overall graph statistics."""
        total_trades = len(self._trades)
        wins = sum(1 for t in self._trades.values() if t.pnl_dollars > 0)
        total_pnl = sum(t.pnl_dollars for t in self._trades.values())

        return {
            "total_trades": total_trades,
            "total_conditions": len(self._conditions),
            "total_lessons": len(self._lessons),
            "total_edges": len(self._edges),
            "wins": wins,
            "losses": total_trades - wins,
            "win_rate": wins / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "patterns_tracked": len(self._pattern_index),
            "regimes_tracked": len(self._regime_index),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

_trade_graph: Optional[TradeGraph] = None


def get_trade_graph() -> TradeGraph:
    """Get singleton TradeGraph instance."""
    global _trade_graph
    if _trade_graph is None:
        _trade_graph = TradeGraph()
    return _trade_graph
