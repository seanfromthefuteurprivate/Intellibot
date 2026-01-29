# WSB Snake - System Invariants (Safety Rules)

## Critical Safety Invariants

These rules are NEVER violated by the system. They exist to protect capital and ensure safe operation.

---

## 1. Position Size Limits

### INV-001: Maximum Per Trade
```
INVARIANT: No single trade exceeds $1,500
ENFORCED IN: alpaca_executor.py
CHECK: if position_cost > MAX_PER_TRADE: REJECT
```

### INV-002: Maximum Daily Exposure
```
INVARIANT: Total daily exposure never exceeds $6,000
ENFORCED IN: alpaca_executor.py
CHECK: if daily_exposure + new_trade > MAX_DAILY_EXPOSURE: REJECT
```

### INV-003: Maximum Concurrent Positions
```
INVARIANT: Never more than 5 open positions at once
ENFORCED IN: alpaca_executor.py
CHECK: if open_positions >= 5: REJECT new trades
```

---

## 2. Time-Based Safety

### INV-004: 0DTE End-of-Day Close
```
INVARIANT: ALL 0DTE positions close by 3:55 PM ET
ENFORCED IN: alpaca_executor.py
TRIGGER: 3:55 PM ET → close_all_0dte_positions()
```

### INV-005: Market Hours Only
```
INVARIANT: No trades placed outside 9:30 AM - 4:00 PM ET
ENFORCED IN: session_regime.py
CHECK: is_market_open() must be True to trade
```

### INV-006: Maximum Hold Time
```
INVARIANT: No 0DTE position held longer than 45 minutes
ENFORCED IN: alpaca_executor.py
CHECK: if elapsed_minutes >= 45: execute_exit("TIME_DECAY")
```

---

## 3. Exit Enforcement

### INV-007: Target Exit
```
INVARIANT: Position exits when +20% target hit
ENFORCED IN: alpaca_executor._check_exits()
CHECK: if current_price >= target_price: execute_exit("TARGET_HIT")
```

### INV-008: Stop Loss Exit
```
INVARIANT: Position exits when -15% stop hit
ENFORCED IN: alpaca_executor._check_exits()
CHECK: if current_price <= stop_loss: execute_exit("STOP_LOSS")
```

### INV-009: Zero Greed Protocol
```
INVARIANT: No position escapes mechanical exit rules
ENFORCED IN: zero_greed_exit.py
PRINCIPLE: Target hit = IMMEDIATE EXIT, no exceptions
```

---

## 4. AI Safety

### INV-010: Rate Limiting
```
INVARIANT: AI API calls limited to prevent abuse
LIMITS:
  - 10 calls per minute
  - 60 calls per hour
  - 6 second minimum cooldown between calls
ENFORCED IN: predator_stack.py
```

### INV-011: Daily Budget Cap
```
INVARIANT: AI spending never exceeds $5/day
ENFORCED IN: predator_stack.py
CHECK: if daily_spend >= DAILY_BUDGET_USD: skip AI analysis
```

### INV-012: Sniper Mode
```
INVARIANT: OpenAI only fires on high-value setups
REQUIREMENTS:
  - Confidence >= 60%
  - Volume >= 1.5x average
  - Pattern changed from last check
ENFORCED IN: predator_stack.py
```

---

## 5. Execution Safety

### INV-013: AI Confirmation Required for Auto-Execute
```
INVARIANT: Auto-trade requires BOTH 70%+ confidence AND AI confirmation
ENFORCED IN: spy_scalper.py
CHECK:
  should_auto_execute = (
    total_confidence >= 70 AND
    setup.ai_confirmed == True
  )
```

### INV-014: Valid Stop/Target Required
```
INVARIANT: No trade without valid stop loss and target
ENFORCED IN: alpaca_executor.py
CHECK: if stop_loss <= 0 OR target_price <= 0: REJECT
```

### INV-015: Paper Trading Mode
```
INVARIANT: System operates in paper trading mode only
ENFORCED IN: alpaca_executor.py
BASE_URL: https://paper-api.alpaca.markets
```

---

## 6. Data Integrity

### INV-016: Eastern Time Zone
```
INVARIANT: All market time calculations use US/Eastern
ENFORCED IN: session_regime.py
IMPLEMENTATION: pytz.timezone('US/Eastern')
```

### INV-017: Signal Persistence
```
INVARIANT: All signals recorded to database
ENFORCED IN: spy_scalper._record_signal()
TABLE: spy_scalp_history
```

### INV-018: Position Tracking
```
INVARIANT: All positions synced between Alpaca and local state
ENFORCED IN: alpaca_executor.sync_positions()
TRIGGER: On startup and periodically during operation
```

---

## 7. Notification Safety

### INV-019: Critical Alerts Delivered
```
INVARIANT: All trade executions and exits send Telegram alerts
ENFORCED IN: alpaca_executor.py, spy_scalper.py
ALERTS:
  - Entry: "AUTO-EXECUTED: {ticker} {direction}"
  - Fill: "ORDER FILLED: {details}"
  - Exit: "TARGET HIT / STOP LOSS / TIME DECAY"
```

---

## Violation Handling

If any invariant is violated:
1. Log ERROR with full context
2. Send Telegram alert
3. Reject the violating action
4. Continue safe operation

Example:
```python
if actual_cost > MAX_DAILY_EXPOSURE * 1.5:
    logger.error(f"EMERGENCY: Position ${actual_cost} exceeds limit!")
    send_telegram_alert("🚨 EMERGENCY: Oversized position - AUTO-CLOSING!")
    self.close_position(position.option_symbol)
```
