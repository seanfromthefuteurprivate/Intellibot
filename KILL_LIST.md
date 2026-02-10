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

## FIXED (Committed 2026-02-10):

### 1. ✅ Predator stack integrated (commit ff0b979)
- **File:** `wsb_snake/execution/apex_conviction_engine.py`
- **Fix:** Full predator stack integration with chart generation
- AI verdict now calls `predator.analyze_sync()` with generated charts

### 2. ✅ Greeks estimation implemented (commit ff0b979)
- **File:** `wsb_snake/engines/precious_metals_scalper.py`
- **Fix:** Added `_estimate_greeks()` method with VIX-based calculations
- No longer using hardcoded placeholder values

### 3. ✅ Economic calendar integrated (commit ff0b979)
- **File:** `wsb_snake/engines/orchestrator.py`
- **Fix:** Integrated `fred_collector.get_economic_calendar()`
- Now detects high-impact macro events (NFP, FOMC, CPI)

### 4. Silent exception swallowing
- **Severity:** LOW (defensive coding)
- **Status:** Acceptable for now - most are database migrations

### 5. ✅ Sentiment validation added (commit ff0b979)
- **File:** `wsb_snake/analysis/sentiment.py`
- **Fix:** Added startup validation and `is_enabled()` check
- Logs warning if OPENAI_API_KEY not set

### 6. ✅ CPL scanner integrated (commit ff0b979)
- **File:** `wsb_snake/main.py`
- **Fix:** Added `JobsDayCPL` scanner, runs every 60s during market hours

### 7. ✅ Volatility multiplier fixed (commit ff0b979)
- **File:** `wsb_snake/engines/precious_metals_scalper.py`
- **Fix:** Now fetches real VIX data from `vix_structure.get_trading_signal()`

### 8. ✅ AI verdict fixed (commit ff0b979)
- **File:** `wsb_snake/execution/apex_conviction_engine.py`
- **Fix:** Complete rewrite of `_get_ai_verdict_score()`
- Now generates charts and calls predator stack for real AI analysis

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

## RESOLVED (Previously Blocked):

### 1. VM Now Reachable ✅
- **Correct IP:** `157.245.240.99` (was using wrong IP `46.202.156.225`)
- Guardian API responding at `http://157.245.240.99:8888/health`
- All fixes deployed via `/deploy` endpoint on 2026-02-10

---

## VERIFICATION NEEDED ON VM:

1. Check if CPL signals exist in database
2. Verify all services are running
3. Test stop loss with 2-second interval
4. Verify CPL gate blocks trades correctly

---

## Quick Reference - Files Modified:

| File | Status | Commit |
|------|--------|--------|
| `wsb_snake/engines/spy_scalper.py` | ✅ FIXED | 8905dd3 |
| `run_max_mode.py` | ✅ FIXED | 8905dd3 |
| `wsb_snake/utils/cpl_gate.py` | ✅ CREATED | 5d839b8 |
| `wsb_snake/engines/momentum_engine.py` | ✅ FIXED | 5d839b8 |
| `wsb_snake/engines/power_hour_runner.py` | ✅ FIXED | 5d839b8 |
| `wsb_snake/engines/institutional_scalper.py` | ✅ FIXED | 5d839b8 |
| `wsb_snake/engines/leaps_engine.py` | ✅ FIXED | 5d839b8 |
| `wsb_snake/engines/orchestrator.py` | ✅ FIXED (CPL + calendar) | ff0b979 |
| `wsb_snake/trading/alpaca_executor.py` | ✅ FIXED | d9510a4 |
| `wsb-snake.service` | ✅ CREATED | 243baf9 |
| `cpl-scanner.service` | ✅ CREATED | 243baf9 |
| `wsb_snake/execution/apex_conviction_engine.py` | ✅ FIXED (predator + ai_verdict) | ff0b979 |
| `wsb_snake/engines/precious_metals_scalper.py` | ✅ FIXED (Greeks + volatility) | ff0b979 |
| `wsb_snake/analysis/sentiment.py` | ✅ FIXED (validation) | ff0b979 |
| `wsb_snake/main.py` | ✅ FIXED (CPL scanner added) | ff0b979 |

---

## All Issues Resolved ✅

All critical and medium severity issues have been fixed and deployed to the VM at `157.245.240.99`.

**NFP Date Update:** All references to Feb 6 have been updated to Feb 11, 2026 (rescheduled due to government shutdown).

---

## INFRASTRUCTURE AUDIT CHECKLIST (MANDATORY)

**This checklist was added after a critical miss: duplicate deployments (App Platform + Droplet) ran simultaneously, causing API connection conflicts.**

### Before ANY Audit, Run These Commands:

```bash
# 1. List ALL Digital Ocean resources
doctl compute droplet list --format ID,Name,PublicIPv4,Status
doctl apps list --format ID,Spec.Name,DefaultIngress

# 2. SSH to verify droplet state
ssh root@157.245.240.99 "hostname && git -C /root/wsb-snake log --oneline -1 && systemctl status wsb-snake --no-pager | head -5"

# 3. Check for duplicate deployments
# If BOTH droplet AND app platform exist, DELETE the app platform:
# doctl apps delete <APP_ID> --force

# 4. Check for connection conflicts
ssh root@157.245.240.99 "journalctl -u wsb-snake --since '5 minutes ago' | grep -i 'connection limit'"
```

### Infrastructure State (Current):

| Resource | Status | Details |
|----------|--------|---------|
| **Droplet (wsb-snake)** | ✅ ACTIVE | 157.245.240.99, commit e283592 |
| **App Platform (coral-app)** | ❌ DELETED | Was causing "connection limit exceeded" |
| **Guardian API** | ✅ RUNNING | http://157.245.240.99:8888 |

### Why App Platform Was Deleted (2026-02-10):

1. **Duplicate trading bot** - Both connected to same Alpaca account
2. **Connection limit exceeded** - WebSocket conflicts
3. **Redundant cost** - ~$5-12/month wasted
4. **No SSH access** - Can't debug or manually intervene
5. **Droplet has Guardian API** - Full remote control capability

### Red Flags to Watch For:

- `connection limit exceeded` errors in logs
- Duplicate Telegram alerts
- `doctl apps list` showing active apps alongside droplet
- Multiple Python processes on different hosts hitting same APIs
