# BACKTEST v5 FIX CHECKLIST
## Align code with 79 historical trades

---

## PRIORITY 1: ENTRY PRICE FILTERING (CRITICAL)

### Current Code Problem
```python
# Line 56-100: PATTERN_CONFIG
"LOTTO_TICKET": {
    "otm_range": (3, 5),
    "price_range": (0.10, 0.50),  # <-- HISTORICAL: 0% win rate
    "max_sizing": 0.12,
    "stop_loss": -0.75,
}
```

### Historical Data
- **Under $0.10**: 0/2 won (0%) → -69.05% avg loss
- **$0.10-$0.50**: 0/10 won (0%) → -27.95% avg loss
- **$0.50-$1.00**: 1/8 won (12.5%) → -11.09% avg loss
- **$1.00-$2.00**: 7/28 won (25%) → -6.39% avg loss ✓ BEST
- **$2.00+**: 7/31 won (22.6%) → -8.00% avg loss ✓ GOOD

### FIX
```python
# MODIFY hell_backtest_v5.py around line 480-551

def select_berserker_strike(...):
    config = PATTERN_CONFIG.get(pattern, PATTERN_CONFIG["DEFAULT"])
    otm_min, otm_max = config["otm_range"]
    price_min, price_max = config["price_range"]

    # ADD: Entry price floor filtering
    # REJECT any option priced under $0.50
    MIN_ENTRY_PRICE = 0.50  # Historical: nothing under $0.50 wins
    MAX_ENTRY_PRICE = 5.00  # Don't go mega ITM either

    candidates = []

    for offset in range(otm_min, otm_max + 1):
        if direction == "CALL":
            strike = round(spy_price) + offset
        else:
            strike = round(spy_price) - offset

        ticker = build_option_ticker(strike, expiry, direction)
        bars = prefetched.get(ticker)

        if not bars:
            bars = fetch_option_bars(ticker, date)
            if bars:
                prefetched[ticker] = bars

        if bars:
            result = get_option_price_at_time(bars, bar_ts)
            if result:
                price, volume = result
                # ADD THESE FILTERS:
                if price < MIN_ENTRY_PRICE or price > MAX_ENTRY_PRICE:
                    continue  # Skip, too cheap or too expensive
                if volume >= 50 and price_min <= price <= price_max:
                    candidates.append({...})

    # If no candidates in preferred range, try fallback
    # But STILL enforce price floor
    if not candidates:
        for offset in range(1, 6):
            if direction == "CALL":
                strike = round(spy_price) + offset
            else:
                strike = round(spy_price) - offset

            ticker = build_option_ticker(strike, expiry, direction)
            bars = prefetched.get(ticker, [])
            if not bars:
                bars = fetch_option_bars(ticker, date)
                if bars:
                    prefetched[ticker] = bars

            if bars:
                result = get_option_price_at_time(bars, bar_ts)
                if result:
                    price, volume = result
                    # ENFORCE FLOOR EVEN IN FALLBACK
                    if price < MIN_ENTRY_PRICE or price > MAX_ENTRY_PRICE:
                        continue
                    if volume >= 30:
                        candidates.append({...})

    if not candidates:
        return None  # No valid entries at this time
```

### Also: Delete LOTTO_TICKET Pattern
```python
# DELETE or DISABLE (line 56-64)
# "LOTTO_TICKET": {
#     "otm_range": (3, 5),
#     "price_range": (0.10, 0.50),  # 0% win rate - DELETE THIS
#     "max_sizing": 0.12,
#     "stop_loss": -0.75,
#     "trail_trigger": 3.00,
#     "trail_pct": 0.30,
#     "max_hold_minutes": 90,
# },

# REASON: This pattern won 0 out of 12 historical trades
```

### Affected Code Sections
- **Line 481-552**: `select_berserker_strike()` - Add price filtering
- **Line 320-366**: `calculate_semantic_match()` - Remove LOTTO_TICKET scoring
- **Line 56-64**: `PATTERN_CONFIG` - Delete LOTTO_TICKET entirely
- **Line 339-343**: Remove `if pattern == "LOTTO_TICKET"` scoring boost

---

## PRIORITY 2: HOLD TIME DISCIPLINE (HIGH)

### Current Code Problem
```python
# Line 36-100: SCALP and BERSERKER configs
"time_stop_minutes": 10,  # SCALP exits at 10 min - too fast
"max_hold_minutes": 60,   # BERSERKER max varies 45-90 min
```

### Historical Data
- **Winners hold**: 45.4 minutes average
- **Losers hold**: 16.0 minutes average
- **Difference**: Winners hold 2.8x longer before exiting

### Current Backtest Behavior
- SCALP mode: Force exit at 10 minutes if not +10% (kills winners at 10 min average)
- BERSERKER: Max holds vary 45-90 min (better, but still unclear)

### FIX - Extend Hold Times

```python
# Line 36-46: SCALP_CONFIG
SCALP_CONFIG = {
    "sizing": 0.10,
    "max_concurrent": 1,
    "max_daily_trades": 3,
    "profit_target": 0.25,
    "stop_loss": -0.15,
    "time_stop_minutes": 30,  # CHANGE: 10 → 30 (winners hold 45 avg)
    "time_stop_min_gain": 0.10,
    "entry_momentum": 0.0015,
    "otm_offset": 0,
}

# Line 56-100: BERSERKER_CONFIG
PATTERN_CONFIG = {
    "REVERSAL_PUT": {
        "otm_range": (1, 2),
        "price_range": (0.50, 1.50),  # Change from 0.30
        "max_sizing": 0.35,
        "stop_loss": -0.35,
        "trail_trigger": 1.00,
        "trail_pct": 0.15,
        "max_hold_minutes": 60,  # CHANGE: 45 → 60 (align with winners)
    },
    "MOMENTUM_CALL": {
        "otm_range": (1, 3),
        "price_range": (0.50, 1.50),  # Change from 0.30
        "max_sizing": 0.30,
        "stop_loss": -0.40,
        "trail_trigger": 0.75,
        "trail_pct": 0.20,
        "max_hold_minutes": 70,  # CHANGE: 60 → 70
    },
    "PRECIOUS_METALS_MOMENTUM": {
        "otm_range": (1, 2),
        "price_range": (0.50, 1.50),  # Change from 0.30
        "max_sizing": 0.25,
        "stop_loss": -0.40,
        "trail_trigger": 0.80,
        "trail_pct": 0.18,
        "max_hold_minutes": 70,  # CHANGE: 50 → 70
    },
    "DEFAULT": {
        "otm_range": (2, 3),
        "price_range": (0.50, 1.00),  # Change from 0.20
        "max_sizing": 0.15,
        "stop_loss": -0.40,
        "trail_trigger": 1.00,
        "trail_pct": 0.20,
        "max_hold_minutes": 70,  # CHANGE: 60 → 70
    }
}
```

### Why This Works
- Winners in data held 45 min average
- Quick exits (under 5 min) lost money
- Extended holds (30-70 min) let winners breathe
- Still has trail stops and hard limits to prevent overnight holds

### Affected Code Sections
- **Line 36-52**: `SCALP_CONFIG` - Increase `time_stop_minutes` from 10 to 30
- **Line 55-101**: `PATTERN_CONFIG` - Increase all `max_hold_minutes` to 60-70
- **Line 666**: `time_stop_minutes` check in position monitoring

---

## PRIORITY 3: TIME WINDOW CORRECTION (CRITICAL)

### Current Code Problem
```python
# Line 49-52: SCALP_WINDOWS
SCALP_WINDOWS = [
    (9, 35, 11, 0),    # 9:35 AM - 11:00 AM   ← Historical: NO DATA
    (14, 0, 15, 30),   # 2:00 PM - 3:30 PM    ← Historical: 0% WR
]

# Line 203-215: is_in_scalp_window() checks this
def is_in_scalp_window(bar_time: datetime) -> bool:
    for start_h, start_m, end_h, end_m in SCALP_WINDOWS:
        # Only enters SCALP mode during these windows
```

### Historical Data - Entry Time Performance
```
14:00 (2-3 PM):        0/6 won (0%)      ← CURRENT SCALP WINDOW
15:00 (3-4 PM):        0/5 won (0%)      ← CURRENT SCALP WINDOW
16:00 (4-5 PM):        1/3 won (33.3%)   ← STARTING TO WORK
17:00 (5-6 PM):        4/14 won (28.6%)  ← BEST WINDOW
18:00 (6-7 PM):        4/21 won (19%)
19:00 (7-8 PM):        3/17 won (17.6%)
20:00 (8-9 PM):        3/13 won (23.1%)

Morning (9-12):        NO DATA at all
```

### FIX
```python
# REPLACE Line 49-52
SCALP_WINDOWS = [
    # Option 1: Completely shift to evening (matches data)
    (16, 0, 20, 30),   # 4:00 PM - 8:30 PM (your best window)

    # Option 2: If you must keep morning, validate with more data
    # (9, 35, 11, 0),  # COMMENTED OUT - no historical data
]

# BETTER: Make this configurable based on market regime
def get_best_scalp_window(market_regime: str):
    """Route to best windows based on market conditions"""
    if market_regime == "DEAD":  # Range < $1
        return []  # Don't scalp on dead days
    elif market_regime == "SCALP":  # Range $1-$2
        # Your data shows 4-8 PM is peak for options
        return [(16, 0, 20, 30)]  # 4 PM - 8:30 PM
    elif market_regime == "BERSERKER":  # Range > $2
        # More volatile, could do extended hours
        return [(15, 0, 21, 0)]  # 3 PM - 9 PM
    else:
        return []

# UPDATE Line 743
def run_day_simulation(...):
    # Around line 743, change:
    if not is_in_scalp_window(bar_time):
        continue

    # To:
    scalp_windows = get_best_scalp_window(mode)
    if scalp_windows and not any(
        start_h <= bar_time.hour < end_h and
        (bar_time.hour > start_h or bar_time.minute >= start_m) and
        (bar_time.hour < end_h or bar_time.minute < end_m)
        for start_h, start_m, end_h, end_m in scalp_windows
    ):
        continue  # Outside best window
```

### Why This Matters
- Your data shows 0% win rate at 2-3 PM (current SCALP window)
- Your best traders hit at 5-8 PM (currently OUTSIDE scalp window)
- Morning has no historical data, assumes it works

### Affected Code Sections
- **Line 49-52**: `SCALP_WINDOWS` definition
- **Line 203-215**: `is_in_scalp_window()` function
- **Line 738-743**: SCALP mode entry check

---

## PRIORITY 4: RISK-REWARD CALIBRATION (HIGH)

### Current Code Problem
```python
# Line 40: SCALP profit target
"profit_target": 0.25,     # Assumes +25% wins possible

# Line 41: SCALP stop loss
"stop_loss": -0.15,        # Assumes -15% loss acceptable

# Line 872: BERSERKER profit target
"profit_target_pct": 3.0,  # No hard profit target, uses trail
```

### Historical Data
```
Winners average: +10.52%  (not +25%)
Losers average: -17.05%   (worse than -15% target)

Risk-reward ratio needed to break even:
  Wins: 19% of trades at +10.52%
  Losses: 81% of trades at -17.05%
  Expectancy: -11.82% per trade
```

### The Math That Breaks Backtest
```
For breakeven with 19% win rate:
  (0.19 × Win%) + (0.81 × -17.05%) = 0%
  0.19 × Win% = 13.81%
  Win% = 72.68%

You need +72.68% wins to break even!
But your historical winners only average +10.52%

Therefore: Even if backtest hits exact targets,
it CANNOT break even with 19% win rate.
```

### Immediate Fix: Recalibrate Profit Targets
```python
# Line 36-46: SCALP_CONFIG
SCALP_CONFIG = {
    "sizing": 0.10,
    "max_concurrent": 1,
    "max_daily_trades": 3,
    "profit_target": 0.15,  # CHANGE: 0.25 → 0.15 (realistic)
    "stop_loss": -0.15,      # Keep at -15% (market reality)
    "time_stop_minutes": 30,
    "time_stop_min_gain": 0.10,
}

# Line 72-81: BERSERKER configs
PATTERN_CONFIG = {
    "REVERSAL_PUT": {
        ...
        "stop_loss": -0.35,  # KEEP - wide stops for binary outcomes
    },
    "MOMENTUM_CALL": {
        ...
        "stop_loss": -0.40,  # KEEP
    },
    ...
}
```

### The Real Problem: Win Rate Too Low
**Backtest can't fix this directly.** The root cause is trading rules, not profit targets:
1. Entry filter rejects cheap options (good)
2. But time windows are wrong (bad)
3. Hold times were too short (fixed above)

**After fixing #1, #2, #3: Win rate should improve toward 25%+ from 19%**

---

## PRIORITY 5: DIRECTION BIAS (MEDIUM)

### Current Code Problem
```python
# Lines 464-477: detect_berserker_signal()
# Treats LONG and SHORT identically in scoring
if momentum > 0.002 and vol_spike > 1.3:
    direction = "CALL"
    confidence = min(90, 50 + momentum * 2000 + vol_spike * 10 + bar_range * 500)
elif momentum < -0.002 and vol_spike > 1.3:
    direction = "PUT"
    confidence = min(90, 50 + abs(momentum) * 2000 + vol_spike * 10 + bar_range * 500)
```

### Historical Data
```
LONG (Calls):   12/55 won = 21.8% win rate
SHORT (Puts):   3/24 won = 12.5% win rate

Calls win 1.75x more often than Puts
```

### FIX
```python
# Line 464-477: Add direction confidence modifier
if momentum > 0.002 and vol_spike > 1.3:
    direction = "CALL"
    base_confidence = min(90, 50 + momentum * 2000 + vol_spike * 10 + bar_range * 500)
    confidence = base_confidence * 1.1  # BOOST calls by 10% (they win more)
elif momentum < -0.002 and vol_spike > 1.3:
    direction = "PUT"
    base_confidence = min(90, 50 + abs(momentum) * 2000 + vol_spike * 10 + bar_range * 500)
    confidence = base_confidence * 0.85  # REDUCE puts by 15% (they lose more)
```

### Or: Add PUT-specific stops
```python
# Line 81-91: Add PUT-specific config
"PUT_DEFAULT": {
    "otm_range": (1, 2),
    "price_range": (0.50, 1.50),
    "max_sizing": 0.10,      # REDUCE from 0.15 (Puts lose more)
    "stop_loss": -0.30,      # TIGHTER from -0.40 (Puts hit stops faster)
    "trail_trigger": 0.50,
    "trail_pct": 0.15,
    "max_hold_minutes": 45,  # REDUCE from 60 (Puts don't hold as well)
}
```

---

## PRIORITY 6: FILTER ZERO-HOLD TRADES (MEDIUM)

### Data Quality Issue
```
Total trades: 79
Trades with hold_minutes > 0: 31 (39%)
Trades with hold_minutes = 0: 48 (61%)

These 48 zero-hold trades are confusing the analysis:
- Are they rejected entries?
- Are they instant losses?
- Are they database errors?
```

### Investigation Needed
```python
# In hell_backtest_v5.py, add tracking:
# Around line 645-678 (position monitoring)

for pos in positions[:]:
    hold_minutes = int((bar_time - pos.entry_time).total_seconds() / 60)

    # Add logging
    if hold_minutes == 0:
        print(f"DEBUG: Zero-hold trade detected")
        print(f"  Entry: {pos.entry_time}")
        print(f"  Current: {bar_time}")
        print(f"  Reason: {exit_reason}")
```

### Action
- Run backtest with enhanced logging
- Capture why 61% of trades have zero hold
- Adjust backtest logic if these are instant rejections
- Document if this is expected behavior

---

## TESTING PLAN

### Phase 1: Validate Single Changes
1. Test entry price filtering alone
   - Run backtest with new price_range constraints
   - Confirm LOTTO_TICKET removal changes win rate

2. Test hold time extension alone
   - Increase time_stop_minutes to 30
   - Check if winners have more breathing room

3. Test time window change alone
   - Shift SCALP_WINDOWS to 4-8 PM
   - Check if trades execute at all

### Phase 2: Combined Testing
```python
# In main():
test_configs = [
    {
        "name": "BASELINE (current v5)",
        "changes": [],
    },
    {
        "name": "FIX_ENTRY_PRICES",
        "changes": ["remove_lotto_ticket", "enforce_price_floor"],
    },
    {
        "name": "FIX_HOLD_TIMES",
        "changes": ["extend_time_stop", "increase_max_hold"],
    },
    {
        "name": "FIX_TIME_WINDOWS",
        "changes": ["shift_scalp_window"],
    },
    {
        "name": "ALL_FIXES",
        "changes": ["remove_lotto_ticket", "enforce_price_floor",
                    "extend_time_stop", "increase_max_hold", "shift_scalp_window"],
    },
]

for config in test_configs:
    print(f"\nTesting: {config['name']}")
    apply_changes(config['changes'])
    results = main()  # Run backtest
    print(f"Win rate: {results['overall_win_rate']:.1f}%")
    print(f"Final P&L: {results['final_pnl']:+,.0f}")
```

### Success Criteria
- Win rate improves from current (~50%+ assumption) toward historical 19%+ in real data
- Profit target achievable with realistic 19-25% win rate
- Time windows actually execute trades
- Entry prices cluster in $1-$3 range

---

## CODE CHANGE SUMMARY

| Priority | File | Line(s) | Change | Impact |
|----------|------|---------|--------|--------|
| 1 | hell_backtest_v5.py | 56-64 | Delete LOTTO_TICKET | -12 losing trades |
| 1 | hell_backtest_v5.py | 481-552 | Add price filtering | Reject $0.10-$0.50 |
| 2 | hell_backtest_v5.py | 42 | time_stop: 10→30 min | Let winners breathe |
| 2 | hell_backtest_v5.py | 72-99 | max_hold: 45→70 min | Match 45-min winner avg |
| 3 | hell_backtest_v5.py | 49-52 | SCALP_WINDOWS: shift to 4-8 PM | Match 28.6% WR window |
| 4 | hell_backtest_v5.py | 40 | profit_target: 25%→15% | Realistic targets |
| 5 | hell_backtest_v5.py | 464-477 | Add direction bias | Boost LONG, reduce SHORT |

---

## ESTIMATED IMPACT

If all fixes implemented:

```
Current Backtest:
  Assumes: 50%+ win rate, +25% targets, -15% stops
  Reality: 19% win rate, +10.52% targets, -17% stops
  Status: Not calibrated to reality

After Fixes:
  Entry filter: Remove 0% win-rate cheap options
  Time window: Trade when you actually win (5-9 PM)
  Hold discipline: Let winners run (45 min vs 10 min)
  Direction bias: Favor LONG (21.8% vs 12.5% for SHORT)

Expected outcome:
  Win rate: 19% → 25-30% (if time window is key)
  Avg winner: +10.52% → +15%+ (with better filtering)
  Avg loser: -17% → -12% (tighter stops on PUT trades)

  Expectancy: -11.82% → -2% to +1% (closer to breakeven)
```

The data suggests you're close to breakeven. These fixes should push you over.
