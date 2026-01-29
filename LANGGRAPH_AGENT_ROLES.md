# WSB Snake - LangGraph Agent Roles

## AI Agent Architecture

This document describes the LangGraph-based AI agents used in WSB Snake.

---

## Overview

WSB Snake uses LangGraph for structured AI workflows with multiple specialized nodes.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH ARCHITECTURE                        │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   SCALP ANALYZER                           │  │
│  │                   (5-Node Workflow)                        │  │
│  │                                                            │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │  │
│  │  │  VWAP   │─▶│Momentum │─▶│  Trap   │─▶│ Entry   │       │  │
│  │  │Analysis │  │Analysis │  │Detection│  │ Timing  │       │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │  │
│  │                                              │              │  │
│  │                                              ▼              │  │
│  │                                        ┌─────────┐         │  │
│  │                                        │  Final  │         │  │
│  │                                        │ Verdict │         │  │
│  │                                        └─────────┘         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   PREDATOR STACK                           │  │
│  │               (Parallel Multi-Model)                       │  │
│  │                                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │                 PARALLEL EXECUTION                   │  │  │
│  │  │                                                      │  │  │
│  │  │  ┌───────────┐              ┌───────────┐           │  │  │
│  │  │  │  OpenAI   │              │ DeepSeek  │           │  │  │
│  │  │  │  GPT-4o   │              │   Chat    │           │  │  │
│  │  │  │ (Vision)  │              │  (Text)   │           │  │  │
│  │  │  └─────┬─────┘              └─────┬─────┘           │  │  │
│  │  │        │                          │                  │  │  │
│  │  │        └──────────┬───────────────┘                  │  │  │
│  │  │                   ▼                                  │  │  │
│  │  │            ┌─────────────┐                          │  │  │
│  │  │            │   Combine   │                          │  │  │
│  │  │            │   Verdicts  │                          │  │  │
│  │  │            └─────────────┘                          │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent 1: Scalp Analyzer (LangGraph)

**File:** `wsb_snake/analysis/scalp_langgraph.py`

### Purpose
Validate 0DTE scalping patterns through a structured 5-node decision workflow.

### Node Roles

#### Node 1: VWAP Analysis
**Role:** Assess VWAP relationship quality

**Input:**
- Current price
- VWAP value
- Pattern type (reclaim/rejection/bounce)
- Recent price action

**Output:**
```python
{
    "vwap_quality": "strong" | "moderate" | "weak",
    "distance_from_vwap_pct": 0.15,
    "vwap_trending": "up" | "down" | "flat",
    "recommendation": "proceed" | "caution" | "avoid"
}
```

#### Node 2: Momentum Analysis
**Role:** Confirm momentum supports entry direction

**Input:**
- Price change percentage
- Volume ratio
- Direction (long/short)
- RSI, MACD values

**Output:**
```python
{
    "momentum_confirmed": True | False,
    "momentum_strength": "strong" | "moderate" | "weak",
    "divergence_detected": False,
    "recommendation": "proceed" | "caution" | "avoid"
}
```

#### Node 3: Trap Detection
**Role:** Identify failed breakout/breakdown opportunities

**Input:**
- Recent high/low breaks
- Volume on break
- Price rejection patterns
- Trapped buyer/seller zones

**Output:**
```python
{
    "trap_detected": True | False,
    "trap_type": "bull_trap" | "bear_trap" | None,
    "trapped_participants": "buyers" | "sellers" | None,
    "squeeze_potential": "high" | "medium" | "low"
}
```

#### Node 4: Entry Timing
**Role:** Determine optimal entry point

**Input:**
- Current bid/ask spread
- Recent candle patterns
- Session time
- Time to next resistance/support

**Output:**
```python
{
    "optimal_entry": True | False,
    "entry_reason": "pullback_complete" | "breakout_confirmed" | etc,
    "wait_for": "next_candle" | "pullback" | None,
    "timing_score": 85  # 0-100
}
```

#### Node 5: Final Verdict
**Role:** Synthesize all analyses into actionable decision

**Input:**
- All previous node outputs
- Pattern confidence
- Risk parameters

**Output:**
```python
{
    "verdict": "CALLS" | "PUTS" | "NO_TRADE",
    "confidence": 78,  # 0-100
    "entry_price": 602.50,
    "target_price": 603.10,
    "stop_loss": 601.90,
    "reasoning": "Strong VWAP reclaim with momentum confirmation..."
}
```

---

## Agent 2: Predator Stack (Multi-Model)

**File:** `wsb_snake/analysis/predator_stack.py`

### Purpose
Parallel AI analysis using multiple models for speed and accuracy.

### Model Roles

#### OpenAI GPT-4o (Vision)
**Role:** Candlestick chart pattern analysis

**Capabilities:**
- Analyze candlestick chart images
- Identify visual patterns (doji, hammer, engulfing)
- Assess VWAP band relationships
- Detect volume profile anomalies

**Input:**
- Base64 encoded chart image
- Context string (ticker, pattern, price, VWAP)

**Output:**
```python
PredatorAnalysis(
    verdict="STRIKE_CALLS",
    confidence=75.0,
    entry_quality="EXCELLENT",
    vwap_analysis="Price reclaiming VWAP with volume",
    delta_analysis="Buyers in control",
    trap_risk="LOW",
    timing_score=80,
    reasoning="Clean VWAP reclaim with momentum...",
    model_used="gpt-4o"
)
```

#### DeepSeek (Text)
**Role:** News sentiment analysis

**Capabilities:**
- Analyze news headlines
- Determine sentiment (bullish/bearish/neutral)
- Identify trading bias
- Assess news urgency

**Input:**
- List of news headlines
- Ticker symbol
- Current price
- Detected pattern

**Output:**
```python
{
    "sentiment": "bullish",
    "bias": "CALLS",
    "urgency": "high",
    "confidence": 70,
    "key_catalyst": "Apple announces record iPhone sales",
    "reasoning": "Positive earnings guidance..."
}
```

### Verdict Combination Logic

```python
def combine_verdicts(chart: PredatorAnalysis, news: Dict) -> Dict:
    """
    Agreement Scenarios:
    1. Both CALLS → STRONG_CALLS (+15% confidence)
    2. Both PUTS → STRONG_PUTS (+15% confidence)
    3. Disagree → Trust chart, reduce confidence (-20%)
    4. One neutral → Use the signal
    5. Both neutral → NO_TRADE
    """
```

---

## Agent Communication

### State Management

LangGraph maintains state across nodes:

```python
class ScalpAnalyzerState(TypedDict):
    # Input
    ticker: str
    pattern: str
    price: float
    vwap: float
    bars: List[Dict]
    
    # Node outputs
    vwap_analysis: Dict
    momentum_analysis: Dict
    trap_detection: Dict
    entry_timing: Dict
    
    # Final output
    verdict: str
    confidence: float
    reasoning: str
```

### Edge Conditions

Nodes can short-circuit based on conditions:

```python
def should_continue(state: ScalpAnalyzerState) -> str:
    # If VWAP analysis says avoid, skip to verdict
    if state["vwap_analysis"]["recommendation"] == "avoid":
        return "final_verdict"
    
    # Otherwise continue to next node
    return "momentum_analysis"
```

---

## Prompt Engineering

### System Prompts

#### Chart Analysis (OpenAI)
```
You are an expert 0DTE options scalper analyzing candlestick charts.
Focus on: VWAP relationship, volume profile, price action quality.
Be DECISIVE - traders need clear signals, not hedged language.
Always respond in the exact format specified.
```

#### News Analysis (DeepSeek)
```
You are a financial news analyst for 0DTE options trading.
Analyze headlines for immediate trading impact.
Be DECISIVE - determine if news supports CALLS, PUTS, or NONE.
Consider: sentiment, urgency, and direct price impact.
```

---

## Error Handling

### Fallback Chain

```
1. OpenAI (Primary for charts)
   ↓ (if fails)
2. DeepSeek (Text-only fallback)
   ↓ (if fails)
3. Gemini (Suspended - ToS violation)
   ↓ (if fails)
4. No AI - Proceed with algorithm only
```

### Timeout Handling

```python
async with httpx.AsyncClient(timeout=30.0) as client:
    try:
        response = await client.post(url, json=payload)
    except httpx.TimeoutException:
        logger.warning("AI timeout - using fallback")
        return fallback_analysis()
```

---

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| OpenAI latency | < 3s | ~2.5s |
| DeepSeek latency | < 2s | ~1.5s |
| Parallel total | < 3s | ~2.5s |
| Accuracy | > 60% | TBD |
| Daily cost | < $5 | ~$1.50 |
