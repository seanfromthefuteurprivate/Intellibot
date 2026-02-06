# Why Last Trades Lost Money – Root Causes & Fixes

## Summary

Recent trades are losing because of **how we decide to exit** (bid vs target/stop), **how tight the stop is** for 0DTE, and **forced time exit** that crystallizes theta decay. The code is consistent (option $ everywhere), but the parameters and exit logic favor getting stopped out or exiting at a loss.

---

## Root Causes

### 1. **Exit decisions use BID; entry was at ASK**

- We **enter** at the ask (or filled near ask). We **check exits** with `current_price = quote.get("bp") or quote.get("ap")` → we use **bid** for a long position.
- So right after fill we are already “underwater” by the spread (often 3–10% on 0DTE). We need the **bid** to go up 20% from our **entry** to hit target, and the bid to go down 15% to hit stop.
- In practice: we’re a few % behind from the start, so the **stop is easier to hit** and the **target is harder to hit**. That skews outcomes toward losses.

**Where:** `wsb_snake/trading/alpaca_executor.py` → `_check_exits()` uses `current_price = float(quote.get("bp", 0)) or float(quote.get("ap", 0))`.

---

### 2. **-15% stop is too tight for 0DTE**

- 0DTE options can move 20–50% in minutes on normal volatility. A -15% stop gets hit on noise before the thesis can play out.
- So we get **stopped out often** (many small losses) and rarely let winners run to +20%.

**Where:** Position is created with `stop_loss=option_price * 0.85` and again after fill: `position.stop_loss = position.entry_price * 0.85`.

---

### 3. **45-minute forced exit + theta decay**

- We exit **every** position at 45 minutes if target/stop aren’t hit. 0DTE options lose value quickly from theta.
- So many positions that don’t hit target or stop **exit at a loss** at 45 min purely from decay. That turns “hold and see” into **realized losses**.

**Where:** `_check_exits()` → `elif elapsed >= 45: self.execute_exit(position, "TIME DECAY (45min)", current_price)`.

---

### 4. **Spread cost on entry and exit**

- We pay the spread on entry (buy at ask) and on exit (sell at bid). So we need the option to move in our favor just to break even. Combined with (1)–(3), the edge is eroded and losses dominate.

---

## What we are *not* doing wrong

- **Units:** Position target/stop are in **option dollars** (entry_price × 1.20 / 0.85). Exit check uses **option quote** (bid). No underlying/option mix-up.
- **Entry:** We use `filled_avg_price` after fill for entry_price and for target/stop. That’s correct.
- **Validation:** Underlying entry/target/stop from the scalper are only used for R:R validation and strike selection; the live position uses option-based levels.

---

## Fixes (implemented or recommended)

### Fix 1: Use **mid** for exit decisions (reduce bid bias)

- **Change:** In `_check_exits()`, set  
  `current_price = (float(quote.get("bp",0)) + float(quote.get("ap",0))) / 2` when both are present, else fall back to bid or ask.
- **Why:** Decisions are less biased toward hitting the stop (we’re not measuring only the side we’re selling at).

### Fix 2: **Widen the stop** for scalper 0DTE

- **Change:** Use e.g. **-20% or -25%** instead of -15% for scalper positions (e.g. `stop_loss = entry_price * 0.80` or `0.75`).
- **Why:** Fewer stop-outs on noise; more room for the thesis to work. R:R can be kept by raising target (e.g. +30% target with -20% stop).

### Fix 3: **Shorter max hold** or **earlier time exit**

- **Change:** Reduce max hold from 45 min to e.g. **25–30 min** for 0DTE scalper, so we don’t sit in theta decay as long. Alternatively, exit at 30 min at **market** instead of 45.
- **Why:** Less time in the position → less theta drag; forces quicker “in and out” and can reduce the number of “time decay” losses.

### Fix 4 (optional): **Record actual exit fill** for PnL

- When we send the close order, we could store the order ID and, in the monitor loop, when that order is filled, set `position.exit_price = filled_avg_price` and compute PnL from that.
- **Why:** Reported PnL matches real fills; helps backtesting and tuning.

---

## Config knobs (if you add them)

- `SCALP_STOP_PCT` (e.g. 0.85 → 0.80)
- `SCALP_TARGET_PCT` (e.g. 1.20 → 1.30)
- `SCALP_MAX_HOLD_MINUTES` (e.g. 45 → 30)
- `USE_MID_FOR_EXIT_CHECK` (true = use mid; false = current behavior)

---

## Expected effect

- **Fix 1:** Slightly fewer premature stop-outs; more symmetric exit logic.
- **Fix 2:** Fewer small losses from noise; better win rate.
- **Fix 3:** Fewer large time-decay losses; better expectancy.
- **Fix 4:** Accurate PnL; no change to live behavior.

Together, these address the main structural reasons “all last trades have lost money”: **exit logic biased to the bid**, **stop too tight for 0DTE**, and **forced 45-min exit locking in theta losses**.
