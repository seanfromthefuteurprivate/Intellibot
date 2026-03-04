# CPL V2 HANDOFF DOCUMENT - 2026-03-04

## CONTEXT: WHY THIS CHANGE

### March 3 Failure Analysis

| Metric | Actual | Expected |
|--------|--------|----------|
| Trades | 4 | 1 (SNIPER MODE) |
| Win Rate | 0% | 55%+ |
| P&L | -$615.50 | +$1,000 to +$2,500 |
| Direction | QQQ PUT + QQQ CALL (opposing!) | ONE direction only |

**Root Causes:**
1. CPL ignored HYDRA intelligence (was connected but unused)
2. 3-candle momentum check = random noise, not predictive
3. Default pass on data failure (`return True, 50, "insufficient_data"`)
4. Alpaca API lag race condition (5-min cooldown deployed previously)
5. SPY blocked by dead V7 engine

---

## WHAT WAS IMPLEMENTED

### File: `wsb_snake/execution/jobs_day_cpl.py`

#### Change 1: `_check_entry_quality()` Rewritten (Lines 342-523)

**OLD LOGIC:**
```python
bars = polygon_enhanced.get_intraday_bars(...)
if not bars or len(bars) < 3:
    return True, 50, "insufficient_data"  # DANGEROUS: Default pass
```

**NEW LOGIC: 7 Hard Gates**

| Gate | Check | Rejection |
|------|-------|-----------|
| 0 | Polygon health | `return False, 0, "POLYGON_UNHEALTHY"` |
| 1 | HYDRA connected | `return False, 0, "HYDRA_DISCONNECTED"` |
| 2 | Direction alignment | BULLISH=calls only, BEARISH=puts only, NEUTRAL=reject |
| 3 | Regime check | UNKNOWN/CHOPPY/NEUTRAL=reject |
| 4 | Blowup probability | >70% = reject |
| 5 | GEX flip proximity | <1% from flip = reject |
| 6 | Volume confirmation | Price direction + 1.2x volume required |

**Confidence Scoring (only if all gates pass):**
- Base: 55%
- +5-10 for flow bias alignment
- +10 for negative GEX (trending)
- +5 for sweep direction alignment
- +0-10 for volume strength
- +0-10 for momentum strength
- -20 penalty if blowup 51-70%
- Final range: 55-95%

#### Change 2: SPY Block Removed (Lines 866-869)

**OLD:**
```python
if ticker.upper() == "SPY" and self.event_date == get_todays_expiry_date():
    logger.info(f"CPL_BLOCKED: SPY 0DTE now handled by V7 engine")
    continue
```

**NEW:**
```python
# NOTE: V7 was disabled - SPY 0DTE now handled by CPL with HYDRA gates
# (Block removed 2026-03-04 - CPL trades SPY with full HYDRA validation)
```

---

## CONFIGURATION CONSTANTS (Lines 84-91)

```python
SNIPER_CAPITAL = 2500               # Position sizing base
MAX_OPEN_POSITIONS = 1              # One shot at a time
DAILY_PROFIT_TARGET = 2500          # +$2,500 = halt
DAILY_MAX_LOSS = -500               # -$500 = halt
SNIPER_COOLDOWN_SECONDS = 300       # 5-min cooldown after ANY trade
```

---

## HYDRA DATA USED

From `get_hydra_intel()` (wsb_snake/collectors/hydra_bridge.py):

| Field | Used For | Gate |
|-------|----------|------|
| `direction` | BULLISH/BEARISH/NEUTRAL | Gate 2 |
| `regime` | RISK_ON/OFF/TRENDING/CHOPPY | Gate 3 |
| `blowup_probability` | 0-100% risk score | Gate 4 |
| `gex_flip_distance_pct` | Distance to GEX flip point | Gate 5 |
| `flow_bias` | Institutional flow direction | Confidence boost |
| `gex_regime` | POSITIVE/NEGATIVE dealer gamma | Confidence boost |
| `flow_sweep_direction` | CALL_HEAVY/PUT_HEAVY | Confidence boost |
| `seq_historical_win_rate` | Layer 11 pattern match | Confidence boost |

---

## DEPLOYMENT STATUS

### Local Repository
- **File modified:** `wsb_snake/execution/jobs_day_cpl.py`
- **Compilation:** PASSED
- **Ready for commit:** YES

### EC2 (i-03f3a7c46ec809a43)
- **Current state:** Running old code
- **Action needed:** git pull && systemctl restart wsb-snake

---

## VERIFICATION COMMANDS

```bash
# 1. Check HYDRA connection
journalctl -u wsb-snake --no-pager -n 20 | grep HYDRA

# 2. Check for rejections (good sign!)
journalctl -u wsb-snake --no-pager -n 100 | grep ENTRY_V2_REJECT

# 3. Check for approvals
journalctl -u wsb-snake --no-pager -n 100 | grep ENTRY_V2_APPROVED

# 4. Verify no wrong-direction trades
journalctl -u wsb-snake --no-pager --since "09:30" | grep -E "CALL_IN_BEARISH|PUT_IN_BULLISH"
```

---

## AGENT RECOMMENDATIONS IMPLEMENTED

### From Quant Strategist (Agent 1)
- [x] HYDRA direction as hard gate
- [x] Volume confirmation required (1.2x ratio)
- [x] No trade on NEUTRAL direction
- [x] Confidence scoring with multiple factors

### From Code Reviewer (Agent 2)
- [x] Fixed BUG #1: insufficient_data now rejects
- [x] Fixed BUG #2: missing_close_data now rejects
- [x] Fixed BUG #3: RSI failure logged (not silently ignored)
- [x] Fixed BUG #4: Exception catchall now rejects
- [x] Fixed BUG #5: Zero/negative close validation

### From Integration Architect (Agent 3)
- [x] HYDRA import verified (line 30)
- [x] Direction gate implemented (lines 387-398)
- [x] Regime gate implemented (lines 400-415)
- [x] Blowup gate implemented (lines 417-422)
- [x] GEX flip gate implemented (lines 424-427)

### From Risk Manager (Agent 4)
- [x] Position sizing verified (SNIPER_CAPITAL = 2500)
- [x] Max loss per day verified (DAILY_MAX_LOSS = -500)
- [x] Max positions verified (MAX_OPEN_POSITIONS = 1)
- [x] 5-min cooldown verified (SNIPER_COOLDOWN_SECONDS = 300)

### From DevOps (Agent 5)
- [x] Polygon health check module exists
- [x] Gate 0 calls polygon_health_check()
- [x] Graceful fallback if module unavailable

---

## TOMORROW'S EXPECTED BEHAVIOR

**Market Open (9:30 AM ET):**
1. CPL starts scanning watchlist
2. Each ticker goes through 7 gates
3. Most trades REJECTED (this is correct - only high-quality passes)

**Expected Log Pattern:**
```
ENTRY_V2_REJECT: SPY CALL - HYDRA NEUTRAL
ENTRY_V2_REJECT: QQQ PUT - GEX flip 0.5% - too volatile
ENTRY_V2_REJECT: IWM CALL - weak volume 0.9x
ENTRY_V2_APPROVED: SPY CALL gates_passed | conf=78%  <- TRADE EXECUTES
```

**If No Signal by 1 PM:**
- Session halts
- $0 P&L day
- THIS IS A WIN (avoided random trades)

---

## ROLLBACK PROCEDURE

If something goes wrong:

```bash
# SSH to EC2
aws ssm start-session --target i-03f3a7c46ec809a43

# Stop service
sudo systemctl stop wsb-snake

# Revert to previous commit
cd /home/ubuntu/wsb-snake
git reset --hard HEAD~1

# Restart
sudo systemctl start wsb-snake
```

---

## NEXT SESSION: READ THIS FIRST

1. `cat /home/ubuntu/wsb-snake/ops/COMPACT_SUMMARY.md`
2. `cat /home/ubuntu/wsb-snake/ops/HANDOFF.md`
3. Check logs: `journalctl -u wsb-snake --no-pager -n 100`
4. Verify HYDRA connected: `grep HYDRA_BRIDGE` in logs
