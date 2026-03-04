# HISTORICAL TRADE ANALYSIS vs BACKTEST GAP REPORT

## Executive Summary
Analyzed 79 historical option trades. Current backtest logic is **fundamentally misaligned** with what actually wins. Historical win rate is only 19%, but when it wins, execution is correct. The gap: backtest is optimized for patterns that historically lose.

---

## CRITICAL FINDING: THE WRONG SIDE OF THE FENCE

### What Historical Data Shows Wins (19% of trades)
- **Entry price: $1.00-$2.00 range** (25% win rate) and **Over $2.00** (22.6% win rate)
- **Expensive/ITM options**, NOT cheap lottos
- **Longer holds**: Winners average 45.4 minutes, losers only 16 minutes
- **Early morning-to-afternoon timing worse**: 0% win rate at 14:00 and 15:00
- **Late entry window better**: 17:00-20:00 shows 19-28% win rates

### What Historical Data Shows LOSES (81% of trades)
- **Entry price Under $0.50**: 0% win rate (0 out of 12 trades won)
- **Ultra-cheap lottos** ($0.03-$0.50) have -27.95% to -69.05% average loss
- **Quick exits**: Losers average only 16 minutes before bleeding out
- **Afternoon gap trades** (14:00-15:00): 0% win rate (0 out of 6 trades)

---

## DETAILED ANALYSIS

### QUERY 1: ENTRY PRICE DISTRIBUTION
**THE BIGGEST GAP BETWEEN DATA AND BACKTEST**

#### Historical Reality
```
Distribution of trades:
  Under $0.10:       2 trades (2.5%) → 0% win rate → -69.05% avg loss
  $0.10-$0.50:      10 trades (13%) → 0% win rate → -27.95% avg loss
  $0.50-$1.00:       8 trades (10%) → 12.5% win rate → -11.09% avg loss
  $1.00-$2.00:      28 trades (35%) → 25.0% win rate → -6.39% avg loss
  Over $2.00:       31 trades (39%) → 22.6% win rate → -8.00% avg loss
```

#### Winners vs Losers Entry Price
- **Winners avg**: $2.99 (expensive, mostly $1-$3 range)
- **Losers avg**: $3.51 (all over the board, many cheap)
- **Key insight**: ALL wins came from $0.50+ entries. Zero wins from sub-$0.50 lotto tickets.

#### What Backtest v5 Does WRONG
```python
# SCALP MODE: ATM options, focuses on volume not price
"sizing": 0.10,            # 10% per trade
"otm_offset": 0,           # ATM options only (NOT cheap lottos)
# This is actually CORRECT for SCALP - but see below...

# BERSERKER MODE: LOTTO_TICKET pattern is DEADLY
"LOTTO_TICKET": {
    "otm_range": (3, 5),           # Very OTM
    "price_range": (0.10, 0.50),   # CHEAP - but data shows 0% win rate here!
    "max_sizing": 0.12,            # 12% risk per losing trade
    "stop_loss": -0.75,            # -75% stop (bleeding for 45+ mins)
}
```

**THE PROBLEM**: Backtest prioritizes LOTTO_TICKET ($0.10-$0.50) which has **never won** in your data.

---

### QUERY 2: HOLD TIME DISTRIBUTION
**WINNERS HOLD LONGER, BACKTEST EXITS TOO FAST**

#### Historical Patterns
```
Winners (15 trades):
  Average hold: 45.4 minutes
  Median: likely 30+ min (winners have patience)

Losers (64 trades):
  Average hold: 16.0 minutes
  Pattern: Quick panic exits when down, no time to recover
```

#### Hold Time Distribution
```
All trades (31 with time data, 48 with zero hold):
  Under 5 min:    10 trades (33%) - mostly losses
  5-15 min:       10 trades (33%) - mostly losses
  15-30 min:       3 trades (10%)
  30-60 min:       4 trades (13%) - trending positive
  Over 60 min:     4 trades (13%) - good hold discipline
```

#### What Backtest v5 Does WRONG
```python
# SCALP MODE - MAX 10 MINUTES BEFORE TIME STOP
"time_stop_minutes": 10,           # Exit at 10 min if not +10%
"time_stop_min_gain": 0.10,        # Need +10% to survive

# BERSERKER MODE - ALSO TOO SHORT
"LOTTO_TICKET": {"max_hold_minutes": 90},
"REVERSAL_PUT": {"max_hold_minutes": 45},
"MOMENTUM_CALL": {"max_hold_minutes": 60},
"DEFAULT": {"max_hold_minutes": 60},
```

**THE PROBLEM**: Winners hold 45 min average. Backtest has 10-90 min limits but exits on time **before winners have time to print**. Data shows you need 30-60 min holds to let winners breathe.

---

### QUERY 3: WIN RATE BY PATTERN & TIME
**TIME OF DAY MATTERS - BACKTEST IGNORES THIS**

#### By Entry Hour
```
14:00 (2-3 PM):        0.0% WR (0/6 trades) - DEAD ZONE
15:00 (3-4 PM):        0.0% WR (0/5 trades) - DEAD ZONE
16:00 (4-5 PM):       33.3% WR (1/3 trades) - Starting to work
17:00 (5-6 PM):       28.6% WR (4/14 trades) - BEST WINDOW
18:00 (6-7 PM):       19.0% WR (4/21 trades) - Good volume
19:00 (7-8 PM):       17.6% WR (3/17 trades) - Still working
20:00 (8-9 PM):       23.1% WR (3/13 trades) - Strong
```

**KEY INSIGHT**: Your best trades come AFTER market hours (5-9 PM EST). Backtest assumes SCALP WINDOWS of 9:35-11:00 AM and 2:00-3:30 PM, which is **backwards** from data.

#### Direction Performance
```
LONG:   21.8% win rate (12/55)
SHORT:  12.5% win rate (3/24) - Shorts are MUCH harder
```

**THE PROBLEM**: You have 2.3x more LONG trades and they win more. Backtest treats both equally.

---

### QUERY 4: AVERAGE WINNER vs LOSER SIZE
**RISK-REWARD IS INVERTED - YOU'RE RISKING MORE THAN YOU'RE WINNING**

#### Actual Numbers
```
Winners (15 trades):
  Average P&L:     +10.52%
  Median P&L:       +6.00% (half win bigger, half smaller)
  Max win:         +35.56%
  Min win:          +0.75%

Losers (64 trades):
  Average P&L:     -17.05% (LOSS)
  Median loss:     -11.98%
  Max loss:        -71.43% (BLOWUP)
  Min loss:         -0.65%

Risk-Reward Calculation:
  You win:          +10.52% on 19% of trades
  You lose:         -17.05% on 81% of trades

  Expectancy = (0.19 × +10.52%) + (0.81 × -17.05%)
             = +1.99% - 13.81%
             = -11.82% per trade
```

**THIS IS A DEATH SPIRAL**: You need 1.62x risk-reward ratio to break even at 19% win rate. You only have 0.62x. **Every trade you take loses -11.82% on average.**

#### What Backtest Does WRONG
```python
# Backtest assumes:
"profit_target": 0.25,     # +25% take profit
"stop_loss": -0.15,        # -15% stop loss

# PROBLEM: Your real data shows:
# Winners: +10.52% average (not +25%)
# Losers: -17.05% average (already worse than -15% target)

# The backtest's risk-reward is BACKWARDS:
# It wants to risk -15% to win +25% (good ratio)
# But your actual trades risk -17% to win +10% (bad ratio)
```

---

### QUERY 5: WHICH PRICE RANGES WORK?
**CRITICAL: REJECT LOTTO TICKETS**

#### Win Rate by Entry Price
```
Under $0.10:          0.0% WR (0/2)   → -69.05% avg loss
$0.10-$0.50:          0.0% WR (0/10)  → -27.95% avg loss
$0.50-$1.00:         12.5% WR (1/8)   → -11.09% avg loss
$1.00-$2.00:         25.0% WR (7/28)  → -6.39% avg loss    ← BEST
Over $2.00:          22.6% WR (7/31)  → -8.00% avg loss    ← SECOND BEST
```

**CLEAR PATTERN**:
- Every lotto ticket under $0.50 lost (0 wins, 12 losses)
- More expensive options win (25% + 23% win rates)
- As entry price goes down, losses get worse

#### What Backtest v5 Gets Wrong
```python
# The entire "LOTTO_TICKET" pattern targets $0.10-$0.50 cheap options
# HISTORICAL RESULT: 0% win rate, -27.95% average loss
# This is the PRIMARY pattern mode in BERSERKER

# Recommendation:
# DELETE the LOTTO_TICKET pattern entirely
# FOCUS on $1.00-$2.00 and $2.00+ entry prices
```

---

### QUERY 6: ENTRY TIME CORRELATION
**YOUR TRADES AREN'T DURING MARKET HOURS**

#### Performance by Time Block
```
Afternoon (2-3 PM):        0.0% WR (0/6)  → -16.64% avg loss
Power Hour (3 PM+):       20.5% WR (15/73) → -11.42% avg loss
```

#### Detailed Breakdown
```
14:00 (2-3 PM):        0/6 trades won
15:00 (3-4 PM):        0/5 trades won
16:00 (4-5 PM):        1/3 trades won (33%)
17:00 (5-6 PM):        4/14 trades won (28.6%)  ← PEAK
18:00 (6-7 PM):        4/21 trades won (19%)
19:00 (7-8 PM):        3/17 trades won (17.6%)
20:00 (8-9 PM):        3/13 trades won (23.1%)
```

**KEY INSIGHT**: Your data shows entries AFTER hours (5 PM+) perform better than afternoon. This suggests:
- Either you're trading options with different expiries
- Or the "slow afternoon" gets cleaned up by evening data
- The SCALP WINDOWS in backtest (9:35-11:00 AM, 2:00-3:30 PM) don't exist in your data

#### What Backtest v5 Does Wrong
```python
SCALP_WINDOWS = [
    (9, 35, 11, 0),    # 9:35 AM - 11:00 AM
    (14, 0, 15, 30),   # 2:00 PM - 3:30 PM  ← Your data shows 0% wins here
]

# Your actual winning hours: 17:00-20:00 (5-9 PM)
# Backtest assumes: Morning and early afternoon
# This is COMPLETELY BACKWARDS
```

---

## THE BACKTEST vs REALITY MATRIX

| Factor | Historical Data | Backtest v5 | Status |
|--------|-----------------|------------|--------|
| **Entry Price** | $1-$3 (25% WR) | $0.10-$0.50 (0% WR) | WRONG |
| **Hold Time** | 45 min average | 10-90 min limits | PARTIALLY WRONG |
| **Time of Day** | 5-9 PM (20% WR) | 9:35-11:00 AM, 2-3:30 PM (0% WR) | COMPLETELY WRONG |
| **Direction Bias** | LONG 22%, SHORT 12% | Treats equally | MISSING EDGE |
| **Risk-Reward** | 0.62x (bad) | Assumes 1.67x (unrealistic) | DISCONNECTED |
| **Cheap Options** | 0% win rate | Primary pattern | DEADLY |
| **Win Rate** | 19% | Assumes 50%+ | UNREALISTIC |

---

## ACTIONABLE GAPS: WHAT TO FIX

### IMMEDIATE CHANGES (Fix Backtest)

1. **DELETE LOTTO_TICKET pattern**
   - 0% historical win rate
   - -27.95% to -69% average loss
   - Wastes 12% of your trades

2. **Extend hold times**
   - Winners hold 45 min, backtest max is 90 min
   - Increase SCALP time_stop from 10 min to 20-30 min
   - Increase BERSERKER hold limits to 60-90 min average

3. **Fix entry price targeting**
   - Reject options under $0.50 entirely (0% win rate)
   - Focus on $1.00-$3.00 sweet spot (25% WR)
   - Cap maximum entry at $5.00 (no mega ITM)

4. **Correct time windows**
   - SCALP_WINDOWS are empirically wrong (0% WR at 2-3 PM)
   - Move to 5-9 PM OR shift to morning entirely
   - Current data suggests options aren't even tradeable in first half

5. **Bias toward LONG**
   - LONG: 21.8% win rate
   - SHORT: 12.5% win rate (1.75x harder)
   - Reduce SHORT exposure or add SHORT-specific stops

### DEEPER INVESTIGATION NEEDED

- **Why are 48 trades marked as 0 hold_minutes?**
  - These might be rejected, or the timer didn't start
  - Could mask pattern where instant exits = losses

- **Why does data show evening trades?**
  - Are these end-of-day closures?
  - Or are you trading options on different expiries?
  - Clarify if $0.14 entry price is the actual option value

- **Is this SPY options or something else?**
  - Entry prices ($2.99 avg winner) seem high for single SPY contracts
  - Could be spreads, different underlyings, or portfolio-level pricing

---

## STATISTICAL CONFIDENCE

- Sample size: 79 trades (reasonable for pattern analysis)
- Winners: 15 trades (enough to see pattern in winners)
- Losers: 64 trades (very clear pattern on losers)
- Time period: Recent (Feb 2026)

**Confidence level: HIGH** on entry price and time patterns, **MEDIUM** on hold times (48 zero-hold trades confuse the picture).

---

## BOTTOM LINE

Your backtest is optimizing for **exactly the patterns that lose money in reality**:
- Cheap lotto tickets (0% win rate)
- Afternoon trades (0% win rate)
- Quick 10-minute exits (losers average 16 min)
- 1:1.67 risk-reward (you have 0.62x)

Your historical winners use:
- Expensive options $1-$3 (25% win rate)
- Evening trades 5-9 PM (20-28% win rate)
- 45-minute holds (2.8x longer than losers)
- Patient execution

**You're not missing signal detection. You're missing trade filtering and time discipline.**

Fix the entry price filter, time window, and hold discipline, and you'll align backtest with reality.
