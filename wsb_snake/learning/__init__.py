"""
WSB Snake Learning Module

Advanced learning capabilities for detecting and capitalizing on trading opportunities.

Gate Architecture (from IgorGanapolsky/trading multi-gate funnel):
- Gate 15: Bull/Bear Debate - Adversarial AI agents catch traps and confirmation bias
- Gate 35: Introspection Engine - Self-awareness layer detects when patterns are underperforming
- Thompson Sampling - Bayesian exploration/exploitation for strategy selection
- Lessons Memory - Learns from trade outcomes to avoid repeating mistakes

Repo-Inspired Integrations:
- TauricResearch/TradingAgents: Multi-agent debate + analyst team pattern
- virattt/ai-hedge-fund: Investor persona agents (12 legendary investors)
- Open-Finance-Lab/AgenticTrading: Self-improving memory + Neo4j graph storage
- qrak/LLM_trader: Semantic memory with auto-reflection loop
- AI4Finance-Foundation/FinRL: Deep RL position sizing patterns
- GEX Tools: Gamma exposure calculation for BERSERKER activation
"""

from wsb_snake.learning.pattern_memory import pattern_memory, PatternMatch
from wsb_snake.learning.time_learning import time_learning, TimeRecommendation
from wsb_snake.learning.event_outcomes import event_outcome_db, EventExpectation
from wsb_snake.learning.stalking_mode import stalking_mode, StalkState, StalkAlert
from wsb_snake.learning.debate_consensus import get_debate_engine, BullBearDebate, DebateResult, DebateRound
from wsb_snake.learning.introspection_engine import (
    get_introspection_engine,
    IntrospectionEngine,
    IntrospectionResult,
    PatternHealth,
    RegimeStatus
)
from wsb_snake.learning.self_evolving_memory import (
    get_self_evolving_engine,
    record_trade_for_learning
)
from wsb_snake.learning.specialist_swarm import (
    get_specialist_swarm,
    SpecialistSwarm,
    SpecialistVerdict,
    SwarmConsensus,
    TradingPersona,
    PersonaAnalysis,
)
from wsb_snake.learning.advanced_gates import (
    # TIER 1
    get_drawdown_velocity_monitor,
    DrawdownVelocityMonitor,
    # TIER 2
    get_hydra_booster,
    HydraSignalBooster,
    get_regime_kelly,
    RegimeAwareKelly,
    get_expiry_adjuster,
    ExpiryStopAdjuster,
    get_test_time_reasoner,
    TestTimeReasoner,
    get_approval_gate,
    TradeApprovalGate,
    ApprovalDecision,
    # TIER 3
    get_prompt_evolution,
    PromptEvolutionEngine,
    get_bats_router,
    BATSModelRouter,
    get_hierarchical_memory,
    HierarchicalMemory,
    get_adaptive_weighting,
    AdaptiveLayerWeighting,
    get_vol_smile,
    VolatilitySmileLayer,
)
from wsb_snake.learning.gex_calculator import (
    get_gex_calculator,
    GEXCalculator,
    GEXResult,
    BerserkerSignal,
    OptionData,
    parse_hydra_gex_data,
    check_berserker_conditions_from_hydra,
)
from wsb_snake.learning.semantic_memory import (
    get_semantic_memory,
    SemanticMemory,
    TradeConditions,
    TradeOutcome,
    SemanticRule,
    SimilarTrade,
)
from wsb_snake.learning.trading_thesis import (
    TradingThesis,
    EvidenceItem,
    EvidenceSource,
    RiskFactor,
    VolatilityAdjustment,
    ConfidenceLevel,
    create_thesis_from_setup,
)
from wsb_snake.learning.trade_graph import (
    get_trade_graph,
    TradeGraph,
    TradeNode,
    ConditionNode,
    LessonNode,
    Edge,
)

__all__ = [
    # Pattern Memory
    "pattern_memory",
    "PatternMatch",
    # Time Learning
    "time_learning",
    "TimeRecommendation",
    # Event Outcomes
    "event_outcome_db",
    "EventExpectation",
    # Stalking Mode
    "stalking_mode",
    "StalkState",
    "StalkAlert",
    # Gate 15: Bull/Bear Debate (multi-round from TradingAgents)
    "get_debate_engine",
    "BullBearDebate",
    "DebateResult",
    "DebateRound",
    # Gate 35: Introspection Engine
    "get_introspection_engine",
    "IntrospectionEngine",
    "IntrospectionResult",
    "PatternHealth",
    "RegimeStatus",
    # Self-Evolving Memory (Thompson Sampling + Lessons)
    "get_self_evolving_engine",
    "record_trade_for_learning",
    # TIER 1: Drawdown Velocity Monitor
    "get_drawdown_velocity_monitor",
    "DrawdownVelocityMonitor",
    # TIER 2: HYDRA Signal Boosting
    "get_hydra_booster",
    "HydraSignalBooster",
    # TIER 2: Regime-Aware Kelly
    "get_regime_kelly",
    "RegimeAwareKelly",
    # TIER 2: Expiry Stop Adjuster
    "get_expiry_adjuster",
    "ExpiryStopAdjuster",
    # TIER 2: Test-Time Reasoning
    "get_test_time_reasoner",
    "TestTimeReasoner",
    # TIER 2: Trade Approval Gate
    "get_approval_gate",
    "TradeApprovalGate",
    "ApprovalDecision",
    # TIER 3: Prompt Evolution
    "get_prompt_evolution",
    "PromptEvolutionEngine",
    # TIER 3: BATS Model Router
    "get_bats_router",
    "BATSModelRouter",
    # TIER 3: Hierarchical Memory
    "get_hierarchical_memory",
    "HierarchicalMemory",
    # TIER 3: Adaptive Layer Weighting
    "get_adaptive_weighting",
    "AdaptiveLayerWeighting",
    # TIER 3: Volatility Smile Layer
    "get_vol_smile",
    "VolatilitySmileLayer",
    # TIER 3: Specialist Agent Swarm (with personas from ai-hedge-fund)
    "get_specialist_swarm",
    "SpecialistSwarm",
    "SpecialistVerdict",
    "SwarmConsensus",
    "TradingPersona",
    "PersonaAnalysis",
    # GEX Calculator (for BERSERKER)
    "get_gex_calculator",
    "GEXCalculator",
    "GEXResult",
    "BerserkerSignal",
    "OptionData",
    "parse_hydra_gex_data",
    "check_berserker_conditions_from_hydra",
    # Semantic Memory (Auto-Reflection)
    "get_semantic_memory",
    "SemanticMemory",
    "TradeConditions",
    "TradeOutcome",
    "SemanticRule",
    "SimilarTrade",
    # Trading Thesis (from Trading-R1)
    "TradingThesis",
    "EvidenceItem",
    "EvidenceSource",
    "RiskFactor",
    "VolatilityAdjustment",
    "ConfidenceLevel",
    "create_thesis_from_setup",
    # Trade Graph Memory (from AgenticTrading)
    "get_trade_graph",
    "TradeGraph",
    "TradeNode",
    "ConditionNode",
    "LessonNode",
    "Edge",
]
