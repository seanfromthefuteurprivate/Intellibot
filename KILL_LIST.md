# WSB Snake Trading System - Issue Tracker (KILL LIST)

Last Updated: 2026-02-10

---

## ALREADY FIXED (Committed):

### 1. CPL Gate on spy_scalper.py (commit 8905dd3)
- **File:** `wsb_snake/engines/spy_scalper.py`
- **Fix:** Added CPL check at line 1327 before `_send_entry_alert`

### 2. CPL Gate on run_max_mode.py (commit 8905dd3)
- **File:** `run_max_mode.py`
- **Fix:** Changed line 80 to block trades without CPL signal

### 3. CPL Gate on ALL trading paths (commit 5d839b8)
- **Created:** `wsb_snake/utils/cpl_gate.py`
- **Modified:**
  - `momentum_engine.py`
  - `power_hour_runner.py`
  - `institutional_scalper.py`
  - `leaps_engine.py`
  - `orchestrator.py` (2 locations)

### 4. Monitor interval too slow (commit d9510a4)
- **File:** `wsb_snake/trading/alpaca_executor.py:1339`
- **Fix:** Changed `time.sleep(5)` to `time.sleep(2)`

### 5. Missing service files (commit 243baf9)
- **Created:** `wsb-snake.service`, `cpl-scanner.service`

### 6. Database connection leak in CPL functions (commit 1f10c27)
- **Files:** `wsb_snake/utils/cpl_gate.py`, `wsb_snake/engines/spy_scalper.py`, `run_max_mode.py`
- **Fix:** Added `finally` block to ensure `conn.close()` is always called

---

## TO FIX (Not yet committed):

### 1. TODO: Predator stack not integrated
- **Severity:** HIGH
- **File:** `wsb_snake/execution/apex_conviction_engine.py:601`
- **Issue:** AI verdict always returns neutral (50), predator_stack not used
- **Fix:** Integrate predator_stack for visual chart analysis

### 2. Placeholder Greeks in precious metals
- **Severity:** HIGH
- **File:** `wsb_snake/engines/precious_metals_scalper.py:1452-1454`
- **Issue:** Hardcoded `gamma=0.05`, `theta=-0.10`, `vega=0.15`
- **Fix:** Fetch real Greeks from options chain or disable feature

### 3. TODO: Economic calendar not checked
- **Severity:** MEDIUM
- **File:** `wsb_snake/engines/orchestrator.py:485`
- **Issue:** `macro_event` always None, NFP/FOMC/CPI not checked
- **Fix:** Integrate with earnings_calendar or FRED collector

### 4. Silent exception swallowing
- **Severity:** MEDIUM
- **Files:** Multiple (see grep for `except.*pass`)
- **Issue:** Many try/except blocks silently swallow errors
- **Fix:** Add logging or proper error handling

### 5. Sentiment returns placeholder
- **Severity:** MEDIUM
- **File:** `wsb_snake/analysis/sentiment.py:12`
- **Issue:** Returns placeholder when `OPENAI_API_KEY` not set
- **Fix:** Verify env var is set in production

### 6. main.py doesn't start CPL scanner
- **Severity:** HIGH
- **File:** `wsb_snake/main.py`
- **Issue:** CPL scanner not started, must run separately
- **Fix:** Either integrate into main.py or document requirement

### 7. Volatility multiplier always returns 1.0
- **Severity:** MEDIUM
- **File:** `wsb_snake/engines/precious_metals_scalper.py:921-924`
- **Issue:** `_get_volatility_multiplier()` has TODO, ATR not fetched, always returns 1.0
- **Fix:** Integrate ATR calculation from data manager

### 8. AI verdict always neutral in APEX engine
- **Severity:** HIGH (affects all conviction scores)
- **File:** `wsb_snake/execution/apex_conviction_engine.py:590-616`
- **Issue:** `_get_ai_verdict_score()` always returns score=50 (neutral) even when predator is available
- **Impact:** Signals missing AI visual analysis confirmation, conviction scores potentially lower than they should be
- **Fix:** Actually call predator.analyze() when predator is available

---

## ACCEPTABLE (Not bugs):

### Silent exceptions in stalking_mode.py (lines 200-244)
- **Why acceptable:** These are ALTER TABLE statements that fail if column exists - expected behavior for database migrations

### Silent exception in risk_model.py:102
- **Why acceptable:** Fallback to "assume market open" if timezone check fails - defensive coding

### jobs_day_cpl.py has no CPL gate (line 827)
- **File:** `wsb_snake/execution/jobs_day_cpl.py:827`
- **Why acceptable:** This file IS the CPL scanner - it generates CPL signals. It cannot gate itself against its own output. This is the source of regime intelligence, not a consumer of it.

---

## BLOCKED:

### 1. VM Unreachable
- Guardian API at `46.202.156.225:8888` not responding
- Cannot deploy any fixes until VM is back online

---

## VERIFICATION NEEDED ON VM:

1. Check if CPL signals exist in database
2. Verify all services are running
3. Test stop loss with 2-second interval
4. Verify CPL gate blocks trades correctly

---

## Quick Reference - Files Modified:

| File | Status | Severity |
|------|--------|----------|
| `wsb_snake/engines/spy_scalper.py` | FIXED | - |
| `run_max_mode.py` | FIXED | - |
| `wsb_snake/utils/cpl_gate.py` | CREATED | - |
| `wsb_snake/engines/momentum_engine.py` | FIXED | - |
| `wsb_snake/engines/power_hour_runner.py` | FIXED | - |
| `wsb_snake/engines/institutional_scalper.py` | FIXED | - |
| `wsb_snake/engines/leaps_engine.py` | FIXED | - |
| `wsb_snake/engines/orchestrator.py` | FIXED (CPL), NEEDS FIX (calendar) | MEDIUM |
| `wsb_snake/trading/alpaca_executor.py` | FIXED | - |
| `wsb-snake.service` | CREATED | - |
| `cpl-scanner.service` | CREATED | - |
| `wsb_snake/execution/apex_conviction_engine.py` | NEEDS FIX (predator, ai_verdict) | HIGH |
| `wsb_snake/engines/precious_metals_scalper.py` | NEEDS FIX (Greeks, volatility) | HIGH/MEDIUM |
| `wsb_snake/analysis/sentiment.py` | NEEDS FIX (placeholder) | MEDIUM |
| `wsb_snake/main.py` | NEEDS FIX (no CPL scanner) | HIGH |

---

## Priority Order for Fixes:

1. **HIGH - apex_conviction_engine.py** - AI verdict always neutral affects all trades
2. **HIGH - main.py** - CPL scanner not started (but workaround: run as separate service)
3. **HIGH - precious_metals_scalper.py** - Placeholder Greeks used in position sizing
4. **MEDIUM - orchestrator.py** - Economic calendar not checked
5. **MEDIUM - precious_metals_scalper.py** - Volatility always 1.0
6. **MEDIUM - sentiment.py** - Verify OPENAI_API_KEY in production
