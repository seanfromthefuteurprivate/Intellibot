# Repository Study: Integration Patterns for WSB Snake

## Executive Summary

Studied 7 key repositories to extract patterns for WSB Snake, HYDRA, and BERSERKER. This document maps each repo's innovations to actionable integration points.

---

## Repository Analysis

### 1. TauricResearch/TradingAgents
**Type:** Multi-agent debate framework
**Stars:** ~2.5k | **Framework:** LangGraph

#### Key Architecture
```
Analyst Team (4 agents)
    ├── Fundamentals Analyst (financials, metrics)
    ├── Sentiment Analyst (social, mood)
    ├── News Analyst (macro events)
    └── Technical Analyst (MACD, RSI, patterns)
              │
              ▼
Researcher Team (Debate)
    ├── Bullish Researcher (finds positives)
    └── Bearish Researcher (finds risks)
              │
              ▼ (max_debate_rounds configurable)
Trader Agent → Risk Team → Portfolio Manager
```

#### Steal for WSB Snake
1. **Analyst Team Pattern**: Our SpecialistSwarm already mirrors this with 5 specialists
2. **Debate Rounds Config**: Add `max_debate_rounds` to BullBearDebate engine
3. **Portfolio Manager Veto**: Add final approval layer after debate consensus

#### Integration Points
- File: `wsb_snake/learning/debate_consensus.py`
- Enhancement: Add configurable debate rounds, track round-by-round sentiment shift
- New metric: `debate_conviction_delta` = abs(round_1_score - final_score)

---

### 2. TauricResearch/Trading-R1
**Type:** RL-trained reasoning model
**Innovation:** Three-stage easy-to-hard curriculum training

#### Key Patterns
1. **Structured Thesis Composition**: Evidence-based investment theses
2. **Facts-Grounded Analysis**: Every claim linked to data source
3. **Volatility-Adjusted Decision Making**: Regime-aware reasoning

#### Training Data: Tauric-TR1-DB
- 100,000 samples
- 18 months history
- 14 equities
- 5 data sources

#### Steal for WSB Snake
1. **Thesis Template**: Structure AI confirmations as formal theses
2. **Evidence Linking**: Each signal component traced to data source
3. **Curriculum Learning**: Start with high-confidence setups, gradually add edge cases

#### Integration Points
- File: `wsb_snake/engines/spy_scalper.py` → `_get_ai_confirmation()`
- Enhancement: Return structured thesis object, not just confirmation
```python
@dataclass
class TradingThesis:
    direction: str
    conviction: float
    evidence: list[EvidenceItem]  # Each linked to data source
    risk_factors: list[str]
    volatility_adjustment: float
```

---

### 3. virattt/ai-hedge-fund
**Type:** Investor persona agents
**Stars:** ~26k | **Personas:** 12 legendary investors

#### Persona List
| Investor | Style | Key Metric |
|----------|-------|------------|
| Ben Graham | Deep value | Margin of safety |
| Warren Buffett | Quality at fair price | Economic moat |
| Charlie Munger | Mental models | Avoid stupidity |
| Cathie Wood | Disruptive innovation | TAM expansion |
| Michael Burry | Contrarian deep value | Sentiment divergence |
| Stanley Druckenmiller | Macro + momentum | Risk/reward asymmetry |
| Aswath Damodaran | DCF valuation | Intrinsic value gap |
| Bill Ackman | Activist catalyst | Catalyst clarity |
| Mohnish Pabrai | Low-risk high-upside | Heads I win, tails I don't lose much |
| Peter Lynch | GARP | PEG ratio |
| Phil Fisher | Scuttlebutt | Management quality |
| Rakesh Jhunjhunwala | Emerging growth | Local advantage |

#### Steal for WSB Snake
For 0DTE options, adapt personas to short-term trading:
1. **Scalper Persona** (Druckenmiller-style): Momentum + macro alignment
2. **Contrarian Persona** (Burry-style): Fade extended moves
3. **Flow Persona** (Ackman-style): Follow institutional positioning

#### Integration Points
- File: `wsb_snake/learning/specialist_swarm.py`
- Enhancement: Add persona-based reasoning to specialists
```python
class TradingPersona(Enum):
    MOMENTUM_MACRO = "druckenmiller"  # Follow trend + regime
    CONTRARIAN = "burry"              # Fade extremes
    FLOW_FOLLOWER = "ackman"          # Track smart money
```

---

### 4. Open-Finance-Lab/AgenticTrading
**Type:** Self-improving quant agents
**Innovation:** Neo4j memory + semantic search + continuous learning

#### Architecture
```
┌─────────────────────────────────────────────────────┐
│                   Memory Agent                       │
│  ┌─────────────────┐  ┌─────────────────────────┐   │
│  │   Neo4j Graph   │  │  Vector Embeddings      │   │
│  │ (relationships) │  │ (semantic similarity)   │   │
│  └─────────────────┘  └─────────────────────────┘   │
└─────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Alpha Pool     Construction Pool   Audit Pool
   (signals)      (portfolio)         (feedback)
```

#### Key Innovation: Execution Trace Memory
Every trade stores:
- Entry reasoning (full prompt/response)
- Market state at entry
- Outcome (P/L, duration, max adverse excursion)
- Exit reasoning

#### Steal for WSB Snake
1. **Graph-Based Trade Memory**: Store trade relationships (not just flat records)
2. **Semantic Search**: Find similar past setups by embedding current conditions
3. **Audit Pool Pattern**: Post-trade analysis generates learning signals

#### Integration Points
- File: `wsb_snake/learning/advanced_gates.py` → `HierarchicalMemory`
- Enhancement: Add Neo4j integration for relationship modeling
- New: `wsb_snake/learning/trade_graph.py` for graph queries

```python
# Query example: "Find all losing trades when RSI > 70 AND regime = bull"
similar_losses = memory.semantic_search(
    current_conditions,
    filter={"outcome": "loss", "rsi_above": 70, "regime": "bull"}
)
```

---

### 5. AI4Finance-Foundation/FinRL
**Type:** Deep RL trading framework
**Innovation:** Three-layer architecture (environments, agents, applications)

#### Supported RL Algorithms
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- DDPG (Deep Deterministic Policy Gradient)
- SAC (Soft Actor-Critic)
- TD3 (Twin Delayed DDPG)

#### Environment Pattern
```python
class TradingEnv(gym.Env):
    def __init__(self, df, indicators=['macd', 'rsi', 'bb']):
        self.state = [price, holdings, cash, *indicators]

    def step(self, action):
        # action ∈ [-1, 1] for each asset (sell max to buy max)
        reward = portfolio_return - transaction_cost
        return next_state, reward, done, info
```

#### Steal for WSB Snake
1. **State Representation**: Standardize market state for all engines
2. **Action Space**: Continuous sizing (-1 to +1 scale)
3. **Reward Shaping**: Include max adverse excursion penalty

#### Integration Points
- File: `wsb_snake/trading/risk_governor.py`
- Enhancement: Replace fixed position sizing with RL-trained sizer
- New: `wsb_snake/ml/position_sizer_env.py` for training

---

### 6. qrak/LLM_trader
**Type:** Semantic memory trading system
**Innovation:** ChromaDB vector memory + auto-reflection

#### Memory Architecture
```
┌─────────────────────────────────────────┐
│           ChromaDB Collections          │
│  ┌───────────┐  ┌───────────────────┐   │
│  │  Trades   │  │  Semantic Rules   │   │
│  │ (history) │  │ (learned patterns)│   │
│  └───────────┘  └───────────────────┘   │
└─────────────────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   Similarity Search     Rule Injection
   (0.7×sim + 0.3×recency)  (into prompts)
```

#### Key Innovation: Auto-Reflection Loop
Every 10 trades:
1. Analyze recent wins → Extract positive patterns
2. Analyze recent losses → Extract anti-patterns
3. Store as semantic rules with embeddings
4. Inject rules into future prompts

#### Trade Metadata (15+ fields)
- RSI, ADX, ATR at entry
- Stop-loss distance, take-profit distance
- Risk-reward ratio
- Maximum adverse excursion
- Confluence score
- Regime at entry

#### Steal for WSB Snake
1. **Recency Weighting**: `score = similarity * 0.7 + recency * 0.3`
2. **Auto-Reflection**: Generate rules every N trades
3. **Anti-Pattern Flagging**: Explicit "AVOID" rules from losses

#### Integration Points
- File: `wsb_snake/learning/self_evolving_memory.py`
- Enhancement: Add ChromaDB integration
- Add: Auto-reflection loop with configurable threshold

```python
class SemanticMemory:
    def reflect_on_recent_trades(self, n: int = 10):
        """Every N trades, synthesize patterns into rules."""
        recent = self.get_recent_trades(n)
        wins = [t for t in recent if t.pnl > 0]
        losses = [t for t in recent if t.pnl < 0]

        if len(wins) >= 5:
            positive_rule = self.llm.synthesize_pattern(wins)
            self.store_rule(positive_rule, rule_type="positive")

        if len(losses) >= 3:
            anti_pattern = self.llm.synthesize_pattern(losses)
            self.store_rule(anti_pattern, rule_type="avoid")
```

---

### 7. GEX Calculation (Multiple Sources)
**Type:** Gamma exposure analytics
**Critical for:** BERSERKER engine

#### Complete GEX Formula
```python
def calculate_gex(spot_price: float, options_chain: list) -> float:
    """
    GEX = Σ (gamma × 100 × spot² × 0.01 × OI × direction)

    Where:
    - gamma: Black-Scholes gamma for the option
    - 100: Contract multiplier
    - spot²: Spot price squared
    - 0.01: For 1% move
    - OI: Open interest
    - direction: +1 for calls, -1 for puts (dealer assumption)
    """
    total_gex = 0.0
    for opt in options_chain:
        gamma = black_scholes_gamma(
            S=spot_price,
            K=opt.strike,
            T=opt.dte / 365,
            r=0.0,  # Risk-free rate
            sigma=opt.iv
        )
        contract_gex = gamma * 100 * (spot_price ** 2) * 0.01 * opt.oi

        if opt.option_type == "call":
            total_gex += contract_gex
        else:  # put
            total_gex -= contract_gex

    return total_gex
```

#### Zero Gamma Level (Flip Point)
```python
def find_gamma_flip(options_chain: list, spot_range: tuple) -> float:
    """Find price level where net GEX crosses zero."""
    gex_profile = []
    for price in np.linspace(spot_range[0], spot_range[1], 100):
        gex = calculate_gex(price, options_chain)
        gex_profile.append((price, gex))

    # Find zero crossing
    for i in range(1, len(gex_profile)):
        if gex_profile[i-1][1] * gex_profile[i][1] < 0:  # Sign change
            # Linear interpolation
            p1, g1 = gex_profile[i-1]
            p2, g2 = gex_profile[i]
            flip_point = p1 - g1 * (p2 - p1) / (g2 - g1)
            return flip_point
    return None
```

#### BERSERKER Activation Conditions
```python
def should_activate_berserker(
    spot_price: float,
    gamma_flip: float,
    hydra_direction: str,
    vix: float
) -> bool:
    """
    BERSERKER fires when:
    1. Price within 0.3% of gamma flip point
    2. HYDRA gives clear direction (not neutral)
    3. VIX < 25 (not crisis mode)
    """
    proximity = abs(spot_price - gamma_flip) / spot_price

    return (
        proximity < 0.003 and  # Within 0.3%
        hydra_direction in ["BULLISH", "BEARISH"] and
        vix < 25
    )
```

---

## 5-Phase Integration Roadmap

### Phase 1: APEX Conviction Engine 2.0 (from TradingAgents)
**Target:** Enhanced debate + thesis generation

1. Add `max_debate_rounds` config to BullBearDebate
2. Structure AI responses as TradingThesis objects
3. Add conviction delta tracking (initial vs final)
4. Add Portfolio Manager veto layer

**Files to Modify:**
- `wsb_snake/learning/debate_consensus.py`
- `wsb_snake/engines/spy_scalper.py`

### Phase 2: Semantic Memory (from LLM_trader)
**Target:** Vector-based trade memory with auto-reflection

1. Add ChromaDB integration
2. Store trades with 15+ metadata fields
3. Implement recency-weighted similarity search
4. Add auto-reflection loop (every 10 trades)
5. Generate and inject semantic rules

**Files to Create:**
- `wsb_snake/learning/semantic_memory.py`
- `wsb_snake/learning/reflection_engine.py`

### Phase 3: GEX Integration for BERSERKER
**Target:** Real-time gamma exposure tracking

1. Add GEX calculator with Black-Scholes gamma
2. Calculate zero gamma level (flip point)
3. Add proximity alerts (0.3%, 0.5%, 1.0%)
4. Integrate with HYDRA bridge
5. BERSERKER activation logic

**Files to Create:**
- `wsb_snake/learning/gex_calculator.py`

**Files to Modify:**
- `wsb_snake/engines/berserker_engine.py`
- `wsb_snake/integrations/hydra_bridge.py`

### Phase 4: Multi-Model Ensemble (from ai-hedge-fund)
**Target:** Persona-based signal generation

1. Add trading personas (Momentum, Contrarian, Flow)
2. Each specialist adopts a persona lens
3. Aggregate persona signals with confidence weighting
4. Track persona performance separately

**Files to Modify:**
- `wsb_snake/learning/specialist_swarm.py`

### Phase 5: Self-Evolving Strategy (from AgenticTrading)
**Target:** Graph-based memory + continuous learning

1. Add Neo4j for trade relationship modeling
2. Store execution traces (full reasoning)
3. Implement audit pool for post-trade analysis
4. Generate strategy evolution signals
5. Dynamic parameter adjustment based on performance

**Files to Create:**
- `wsb_snake/learning/trade_graph.py`
- `wsb_snake/learning/audit_pool.py`
- `wsb_snake/learning/strategy_evolution.py`

---

## Priority Matrix

| Integration | Impact | Complexity | Priority |
|-------------|--------|------------|----------|
| Debate rounds config | Medium | Low | P1 |
| Trading thesis structure | High | Medium | P1 |
| Semantic memory (ChromaDB) | High | Medium | P1 |
| Auto-reflection loop | High | Medium | P1 |
| GEX calculator | Critical | Medium | P0 |
| BERSERKER GEX integration | Critical | High | P0 |
| Persona-based specialists | Medium | Medium | P2 |
| Graph memory (Neo4j) | High | High | P2 |
| Audit pool | Medium | Medium | P2 |
| Strategy evolution | High | High | P3 |

---

## Data Requirements

### For Semantic Memory
- ChromaDB (pip install chromadb)
- Sentence transformers for embeddings (pip install sentence-transformers)

### For GEX Calculation
- Options chain data (CBOE or Polygon)
- Real-time spot price
- Implied volatility per strike
- Open interest per strike

### For Graph Memory
- Neo4j database
- py2neo or neo4j Python driver

---

## Sources

- [TradingAgents](https://github.com/TauricResearch/TradingAgents)
- [Trading-R1](https://github.com/TauricResearch/Trading-R1)
- [ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)
- [AgenticTrading](https://github.com/Open-Finance-Lab/AgenticTrading)
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL)
- [LLM_trader](https://github.com/qrak/LLM_trader)
- [GEX Calculation Guide](https://perfiliev.com/blog/how-to-calculate-gamma-exposure-and-zero-gamma-level/)
- [GEX Complete Guide 2025](https://optionstradingiq.com/what-is-gex/)
