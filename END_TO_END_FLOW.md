# WSB Snake - End-to-End Flow

## Complete System Behavior

This document traces a complete trade from signal detection to exit.

---

## Phase 1: Market Scan (Every 30 Seconds)

```
┌─────────────────────────────────────────────────────────┐
│                    SPY SCALPER LOOP                      │
│                                                          │
│  for ticker in ZERO_DTE_UNIVERSE:  # 29 tickers         │
│      if cooldown_active(ticker): skip                    │
│      bars = get_5s_15s_1m_bars(ticker)                  │
│      patterns = detect_patterns(bars)                    │
│      if patterns:                                        │
│          setup = create_scalp_setup(patterns)           │
│          process_setup(setup)                            │
└─────────────────────────────────────────────────────────┘
```

**Tickers Scanned:**
```
SPY, QQQ, IWM, AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL,
AMD, NFLX, COIN, MARA, RIOT, PLTR, SOFI, NIO, BABA, SNAP,
GME, AMC, BBBY, HOOD, LCID, RIVN, F, UBER, DIS
```

---

## Phase 2: Pattern Detection

### 2.1 Patterns Detected
| Pattern | Trigger Condition |
|---------|-------------------|
| VWAP Reclaim | Price crosses above VWAP with volume > 1.3x |
| VWAP Rejection | Price rejects from VWAP with bearish candle |
| VWAP Bounce | Price bounces off VWAP support |
| Momentum Surge Long | +0.15% move with volume > 1.5x |
| Momentum Surge Short | -0.15% move with volume > 1.5x |
| Breakout | Price > 30-bar high with volume > 1.3x |
| Breakdown | Price < 30-bar low with volume > 1.3x |
| Failed Breakout | Breakout fails, traps bulls |
| Failed Breakdown | Breakdown fails, traps bears |
| Squeeze Fire | Volatility expansion after compression |

### 2.2 Base Confidence Calculation
```python
base_confidence = 60  # Starting point

# Volume boost
if volume_ratio >= 2.0: base_confidence += 10
elif volume_ratio >= 1.5: base_confidence += 5

# Momentum boost
if abs(momentum) >= 0.3: base_confidence += 8
elif abs(momentum) >= 0.2: base_confidence += 5

# VWAP alignment boost
if direction == "long" and price > vwap: base_confidence += 5
if direction == "short" and price < vwap: base_confidence += 5
```

---

## Phase 3: Learning Boosts

### 3.1 Pattern Memory
```python
# Check if this pattern worked before
similar_patterns = pattern_memory.find_similar(
    pattern_type=setup.pattern,
    price_action=recent_bars,
    volume_profile=volume_data
)

if similar_patterns:
    avg_success_rate = calculate_success_rate(similar_patterns)
    if avg_success_rate > 0.6:
        setup.pattern_memory_boost = 10
    elif avg_success_rate > 0.5:
        setup.pattern_memory_boost = 5
```

### 3.2 Time-of-Day Learning
```python
# Get performance for current hour
time_performance = time_learning.get_hour_quality(current_hour)

if time_performance.quality_score > 70:
    setup.time_quality_score = 10
elif time_performance.quality_score > 50:
    setup.time_quality_score = 5
```

---

## Phase 4: AI Analysis (Parallel)

```
┌──────────────────────────────────────────────────────────────┐
│                    PARALLEL AI ANALYSIS                       │
│                                                               │
│  ┌─────────────────────┐     ┌─────────────────────┐         │
│  │    OpenAI GPT-4o    │     │      DeepSeek       │         │
│  │   (Chart Vision)    │     │   (News Sentiment)  │         │
│  │                     │     │                     │         │
│  │ Input: Candlestick  │     │ Input: 5 recent     │         │
│  │        chart image  │     │        news headlines│         │
│  │                     │     │                     │         │
│  │ Output:             │     │ Output:             │         │
│  │ - STRIKE_CALLS      │     │ - CALLS/PUTS/NONE   │         │
│  │ - STRIKE_PUTS       │     │ - Sentiment score   │         │
│  │ - NO_TRADE          │     │ - Key catalyst      │         │
│  │ - ABORT             │     │ - Urgency level     │         │
│  └──────────┬──────────┘     └──────────┬──────────┘         │
│             │                           │                     │
│             └───────────┬───────────────┘                     │
│                         ▼                                     │
│              ┌─────────────────────┐                         │
│              │  COMBINE VERDICTS   │                         │
│              │                     │                         │
│              │ Both agree? +15%    │                         │
│              │ Disagree? -20%      │                         │
│              │ One neutral? Use    │                         │
│              │   the other         │                         │
│              └─────────────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

---

## Phase 5: Trade Decision

```python
# Calculate total confidence
total_confidence = (
    setup.confidence +           # Base pattern confidence
    setup.pattern_memory_boost + # Learning boost
    setup.time_quality_score     # Time-of-day boost
)

# Apply AI adjustments
if ai_confirmed:
    total_confidence += 10
else:
    total_confidence -= 15

if chart_and_news_agree:
    total_confidence += 10

# Decision gate
should_alert = total_confidence >= 60
should_auto_execute = total_confidence >= 70 AND ai_confirmed
```

---

## Phase 6: Trade Execution

### 6.1 Telegram Alert Sent
```
========================================
🦅 SPY 0DTE SCALP ALERT 🦅
========================================

📊 Pattern: VWAP_RECLAIM
BUY CALLS

💰 ENTRY: $602.50
🎯 TARGET: $603.10
🛑 STOP: $601.90

📈 R:R = 1:2.0
💵 Expected Gain: ~25%

📍 VWAP: $602.00
📊 Volume: 1.8x avg
🚀 Momentum: +0.18%

✅ AI CONFIRMED
🎯 Confidence: 78%
========================================
```

### 6.2 Alpaca Order Placed
```python
if should_auto_execute:
    alpaca_position = alpaca_executor.execute_scalp_entry(
        underlying="SPY",
        direction="long",
        entry_price=602.50,
        target_price=603.10,
        stop_loss=601.90,
        confidence=78,
        pattern="vwap_reclaim"
    )
    
    send_telegram_alert("🤖 AUTO-EXECUTED: SPY CALLS @ $602.50")
```

---

## Phase 7: Position Monitoring

```
┌─────────────────────────────────────────────────────────┐
│              MONITOR LOOP (Every 5 Seconds)              │
│                                                          │
│  for position in open_positions:                         │
│      current_price = get_option_quote(position.symbol)  │
│                                                          │
│      if current_price >= target_price:                  │
│          execute_exit(position, "TARGET HIT 🎯")        │
│                                                          │
│      elif current_price <= stop_loss:                   │
│          execute_exit(position, "STOP LOSS")            │
│                                                          │
│      elif elapsed_minutes >= 45:                        │
│          execute_exit(position, "TIME DECAY")           │
│                                                          │
│      elif current_time >= 3:55 PM ET:                   │
│          close_all_0dte_positions()                     │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 8: Exit Execution

### 8.1 Exit Alert Sent
```
🔴 **SELL ORDER SENDING**

**CALLS** SPY
Contracts: 2
Entry: $1.50
Current: $1.80
Reason: TARGET HIT 🎯

⏳ Closing on Alpaca...
```

### 8.2 Order Closed
```python
result = close_position(position.option_symbol)
position.exit_price = current_price
position.pnl = (current_price - entry_price) * qty * 100
position.status = PositionStatus.CLOSED

send_telegram_alert(f"""
✅ **POSITION CLOSED**
{position.trade_type} {position.symbol}
Entry: ${position.entry_price:.2f}
Exit: ${position.exit_price:.2f}
P&L: ${position.pnl:.2f}
""")
```

---

## Phase 9: Learning Update

```python
# Record outcome to database
record_outcome(
    signal_id=signal.id,
    entry_price=entry_price,
    exit_price=exit_price,
    pnl=pnl,
    outcome_type="win" if pnl > 0 else "loss"
)

# Update pattern memory
if pnl > 0:
    pattern_memory.record_success(
        pattern_type=setup.pattern,
        confidence=total_confidence,
        pnl=pnl
    )

# Update time learning
time_learning.record_trade(
    hour=entry_hour,
    outcome="win" if pnl > 0 else "loss",
    pnl=pnl
)
```

---

## Timing Summary

| Phase | Duration |
|-------|----------|
| Market Scan | Every 30 seconds |
| Pattern Detection | ~100ms per ticker |
| AI Analysis | 2-5 seconds (parallel) |
| Order Placement | ~500ms |
| Position Monitor | Every 5 seconds |
| Exit Execution | ~500ms |
| Learning Update | ~50ms |

**Total Signal-to-Trade: ~3-6 seconds**
