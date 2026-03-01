# Self-Evolving Agents Reference

## Source Repositories

1. **EvoAgentX/Awesome-Self-Evolving-Agents** - Research survey on self-evolving AI agents
2. **IgorGanapolsky/trading** - Production autonomous trading system with self-healing

---

## Core Concepts from Research

### 1. Single-Agent Optimization

#### 1.1 LLM Behavior Optimization
- **Training-Based**: Supervised fine-tuning, reinforcement learning
- **Test-Time**: Feedback-based (CodeT, LEVER), search-based (Tree of Thoughts, Graph of Thoughts)
- **Key Paper**: Agent Q (2024) - Advanced reasoning via autonomous learning

#### 1.2 Prompt Optimization
- **Edit-Based**: GPS, GrIPS, TEMPERA
- **Evolutionary**: EvoPrompt, Promptbreeder
- **Generative**: OPRO (LLMs as Optimizers), PromptAgent
- **Text Gradient**: TextGrad - automatic differentiation via text

#### 1.3 Memory Optimization
- **MemoryBank** (AAAI'24) - Long-term memory with retrieval
- **Agent Workflow Memory** (ICML'24) - Store successful workflows
- **A-MEM** (2025) - Agentic memory for LLM agents
- **Mem0** (2025) - Scalable production memory

#### 1.4 Tool Optimization
- **ToolLLM** - Master 16,000+ APIs
- **CREATOR** - Tool creation for reasoning
- **Learning Evolving Tools** (ICLR'25) - Tools that improve over time

### 2. Multi-Agent Optimization

- **MetaGPT** - Meta programming for multi-agent collaboration
- **AutoGen** - Multi-agent conversations
- **AgentVerse** - Multi-agent collaboration + emergent behaviors
- **GPTSwarm** - Agents as optimizable graphs
- **AFlow** - Automating agentic workflow generation
- **Self-Evolving Multi-Agent Networks** (ICLR'25) - Networks that improve together

### 3. Domain-Specific Optimization

#### Financial Decision-Making
- **FinRobot** - Open-source AI agent for finance
- **FinCon** (NeurIPS'24) - Multi-agent with verbal reinforcement
- **R&D-Agent-Quant** - Data-centric factor optimization

---

## Production Patterns from IgorGanapolsky/trading

### Architecture: Multi-Gate Funnel

```
Gate 0 - Psychology (mindset check)
Gate 1 - Momentum (math, free)
Gate 2 - RL Filter (local inference)
Gate 3 - LLM Sentiment (budgeted)
Gate 4 - Risk Sizing (hard rules)
Gate 5 - Execution (final check)
Gate 15 - Debate (bull/bear agents)
Gate 35 - Introspection (self-awareness)
```

### Thompson Sampling for Strategy Selection

```python
# Beta distribution per strategy
class BetaDistribution:
    alpha: float = 1.0  # Successes + 1
    beta: float = 1.0   # Failures + 1

    def sample(self) -> float:
        return random.betavariate(self.alpha, self.beta)

    def update(self, success: bool):
        if success:
            self.alpha += 1
        else:
            self.beta += 1

# Selection: sample from each, pick highest
# Natural exploration/exploitation balance
```

### BATS Framework (Budget-Aware Model Selection)

| Task Complexity | Model | Cost |
|-----------------|-------|------|
| Simple | DeepSeek V3 | $0.30/1M |
| Medium | Mistral Medium | $0.40/1M |
| Complex | Kimi K2 | $0.39/1M |
| Reasoning | DeepSeek-R1 | $0.70/1M |
| Critical | Claude Opus | $15/1M |

### Feedback-Driven Context Pipeline

```
Signal Capture → Thompson Sampling → Memory Storage → Context Injection
                      ↓
              Beta-Bernoulli model
              30-day exponential decay
```

### Self-Healing CI (Ralph Mode)

- 84 GitHub Actions workflows
- Self-Healing Monitor (every 15 min)
- Overnight autonomous coding sessions
- Auto-published lessons to GitHub Pages

### Lessons Learned RAG

```
rag_knowledge/lessons_learned/
├── risk_lessons.md
├── timing_lessons.md
├── pattern_lessons.md
└── execution_lessons.md

Query before every trade to avoid repeating mistakes
```

---

## Key Implementations for WSB Snake

### 1. Thompson Sampling (`self_evolving_memory.py`)

```python
from wsb_snake.learning.self_evolving_memory import get_self_evolving_engine

engine = get_self_evolving_engine()

# Record outcome
engine.record_trade_outcome(
    ticker="SPY",
    pattern="vwap_bounce",
    strategy="scalper",
    direction="long",
    pnl_pct=12.5,
    hold_time_minutes=15,
    context={"vix": 18.5, "regime": "RISK_ON"}
)

# Select best pattern using Thompson Sampling
best_pattern, confidence, stats = engine.select_best_pattern([
    "vwap_bounce", "momentum_surge", "breakout"
])
```

### 2. Lessons Memory

```python
# Query relevant lessons before trading
lessons = engine.get_relevant_lessons({
    "ticker": "SPY",
    "pattern": "vwap_bounce",
    "regime": "RISK_ON"
})

for lesson in lessons:
    print(f"- {lesson['title']}: {lesson['action']}")
```

### 3. Daily Evolution Cycle

```python
# In daily_report or EOD function:
engine = get_self_evolving_engine()
engine.apply_daily_decay()

# Log evolution stats
stats = engine.get_evolution_stats()
logger.info(f"Evolution stats: {stats['total_lessons']} lessons learned")
```

---

## Evolution Dimensions

| Dimension | What Evolves | Mechanism |
|-----------|-------------|-----------|
| Strategy | Pattern selection | Thompson Sampling |
| Risk | Position sizing | Lessons + outcomes |
| Timing | Entry/exit | Time learning |
| Prompts | AI queries | Outcome feedback |
| Tools | Which tools work | Success tracking |
| Memory | What to remember | Relevance scoring |

---

## Key Papers to Study

1. **Agent Q** (2024) - Advanced reasoning via autonomous AI agents
2. **Tree of Thoughts** (2023) - Deliberate problem solving
3. **MemoryBank** (AAAI'24) - Long-term memory
4. **EvoPrompt** (ICLR'24) - Evolutionary prompt optimization
5. **Self-Evolving Multi-Agent Networks** (ICLR'25) - Network evolution
6. **Mem0** (2025) - Production-ready memory
7. **A-MEM** (2025) - Agentic memory systems

---

## Integration Points in WSB Snake

1. **spy_scalper.py** - Use Thompson Sampling for pattern selection
2. **alpaca_executor.py** - Record outcomes for learning
3. **risk_governor.py** - Evolve risk parameters
4. **orchestrator.py** - Query lessons before pipeline
5. **daily_report** - Apply decay, log evolution stats

---

## Next Steps

1. [ ] Integrate Thompson Sampling into pattern selection
2. [ ] Record all trade outcomes for learning
3. [ ] Query lessons before major decisions
4. [ ] Add daily decay in EOD processing
5. [ ] Create evolution dashboard endpoint
6. [ ] Implement prompt evolution for AI queries
