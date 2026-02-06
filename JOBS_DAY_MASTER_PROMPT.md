# JOBS DAY CPL — MASTER EXECUTION PROMPT

> **Date:** Wednesday Feb 4, 2026 8:14 PM ET  
> **Event:** Non-Farm Payrolls (NFP) — Friday Feb 6, 2026 8:30 AM ET  
> **System:** Convexity Proof Layer (CPL) v1.0  
> **Objective:** Generate and broadcast 10+ unique, execution-complete 0DTE option calls to Telegram with full deduplication, diversity enforcement, and proof capture.

---

## SYSTEM ARCHITECTURE SUMMARY

### Core Files
| File | Purpose |
|------|---------|
| `run_snake_cpl.py` | Standalone runner with `--broadcast`, `--loop`, `--max-calls`, `--test-mode` |
| `wsb_snake/execution/jobs_day_cpl.py` | CPL engine: chain scanning, liquidity gates, diversity, deduplication |
| `wsb_snake/execution/call_schema.py` | `JobsDayCall` dataclass with all required fields |
| `wsb_snake/notifications/message_templates.py` | `format_jobs_day_call()` and `format_jobs_day_sell()` |
| `wsb_snake/notifications/telegram_bot.py` | `send_alert()` with Markdown escaping |
| `wsb_snake/db/database.py` | `cpl_calls` table, `cpl_call_exists()`, `save_cpl_call()` |

### Configuration Constants (in `jobs_day_cpl.py`)
```python
CPL_EVENT_DATE = "2026-02-06"
CPL_WATCHLIST = ["SPY", "QQQ", "IWM", "TLT", "XLF", "NVDA", "TSLA"]
TARGET_BUY_CALLS = 10
LIQUIDITY_MIN_MID = 0.10
LIQUIDITY_MAX_MID = 2.50
LIQUIDITY_MAX_SPREAD_PCT = 0.12
MAX_COST_PER_CONTRACT = 250
DIVERSITY_MODE = "PROOF"
MAX_CALLS_PER_UNDERLYING = 2
COOLDOWN_MINUTES = 45
```

### Enforced Rules
1. **Liquidity Gates:** Mid price must be \$0.10–\$2.50; spread ≤12% of mid
2. **Diversity:** Max 2 calls per underlying; two-pass (new tickers first)
3. **Deduplication:** In-memory `_sent_calls` set + DB `cpl_call_exists()`
4. **Execution-Ready Assertion:** All fields must be present before broadcast
5. **SELL Completeness:** Original BUY #, exit price, exit reason, % PnL required

---

## OPERATOR COMMANDS

### PHASE 1: PRE-FLIGHT (Run Once — Wednesday Night)

```bash
cd /Users/seankuesia/Downloads/Intellibot

# 1. Verify all API keys
python script/audit_env.py

# 2. Telegram connectivity test
python -c "from wsb_snake.notifications.telegram_bot import send_alert; print('OK' if send_alert('🐍 *CPL ONLINE* — Pre-flight passed. Jobs Day T-2.') else 'FAIL')"

# 3. Dry-run: verify chain fetching + diversity
python run_snake_cpl.py --dry-run --max-calls 5
```

**Expected Output:**
- All keys show `set`
- Telegram returns `OK` and you receive the message
- Dry-run shows 5 calls with ≥3 different underlyings

---

### PHASE 2: GO-LIVE (Wednesday Night → Friday 5 PM ET)

**Option A: Background process (recommended)**
```bash
cd /Users/seankuesia/Downloads/Intellibot
nohup python run_snake_cpl.py --broadcast --loop 60 > cpl_output.log 2>&1 &
echo "CPL launched. PID: \$!"
```

**Option B: tmux session (for live monitoring)**
```bash
tmux new -s cpl
cd /Users/seankuesia/Downloads/Intellibot
python run_snake_cpl.py --broadcast --loop 60
# Detach: Ctrl+B then D
# Re-attach: tmux attach -t cpl
```

**Option C: Foreground (keep terminal open)**
```bash
cd /Users/seankuesia/Downloads/Intellibot
python run_snake_cpl.py --broadcast --loop 60
```

| Flag | Effect |
|------|--------|
| `--broadcast` | Save to DB + send to Telegram |
| `--loop 60` | Re-scan every 60 seconds until Fri 5 PM ET |

---

### PHASE 3: MONITORING (Separate Terminal)

```bash
cd /Users/seankuesia/Downloads/Intellibot

# Live log stream
tail -f cpl_output.log

# Count emitted calls by ticker
sqlite3 wsb_snake.db "SELECT ticker, COUNT(*) FROM cpl_calls GROUP BY ticker;"

# Recent calls
sqlite3 wsb_snake.db "SELECT timestamp_et, ticker, side, strike, confidence FROM cpl_calls ORDER BY timestamp_et DESC LIMIT 10;"

# Check process status
ps aux | grep run_snake_cpl
```

---

### PHASE 4: EMERGENCY STOP

```bash
# Find PID
ps aux | grep run_snake_cpl

# Kill specific process
kill <PID>

# Kill all CPL instances
pkill -f run_snake_cpl

# Verify stopped
ps aux | grep run_snake_cpl
```

---

### PHASE 5: POST-EVENT ANALYSIS (Friday After Close)

```bash
cd /Users/seankuesia/Downloads/Intellibot

# Total calls emitted
sqlite3 wsb_snake.db "SELECT COUNT(*) FROM cpl_calls;"

# Breakdown by ticker and side
sqlite3 wsb_snake.db "SELECT ticker, side, COUNT(*) FROM cpl_calls GROUP BY ticker, side;"

# Export full proof to JSON
sqlite3 wsb_snake.db "SELECT full_json FROM cpl_calls;" > jobs_day_proof.json

# Backup database
cp wsb_snake.db wsb_snake_jobs_day_backup_\$(date +%Y%m%d).db
```

---

## QUICK ONE-LINER (Copy-Paste to Go Live)

```bash
cd /Users/seankuesia/Downloads/Intellibot && python script/audit_env.py && python -c "from wsb_snake.notifications.telegram_bot import send_alert; print('OK' if send_alert('🐍 *CPL ONLINE* — Pre-flight passed') else 'FAIL')" && python run_snake_cpl.py --dry-run --max-calls 3 && echo "=== PRE-FLIGHT PASSED ===" && nohup python run_snake_cpl.py --broadcast --loop 60 > cpl_output.log 2>&1 & echo "CPL LIVE. Logging to cpl_output.log"
```

---

## TIMELINE

| When | Action |
|------|--------|
| **Wed 8:30 PM ET** | Pre-flight + Go-live |
| **Thu all day** | CPL scans every 60s; emits calls as liquidity appears |
| **Fri 8:30 AM ET** | NFP release — peak volatility expected |
| **Fri 9:30 AM–4:00 PM ET** | Active 0DTE window |
| **Fri 5:00 PM ET** | Loop auto-stops |

---

## VALIDATION CHECKLIST

Before Jobs Day, confirm:

- [ ] `python script/audit_env.py` shows all keys set
- [ ] Telegram test message received
- [ ] `--dry-run --max-calls 5` produces ≥3 different underlyings
- [ ] `--test-mode --broadcast --max-calls 5` sends 5 execution-complete messages
- [ ] No two messages have identical (ticker, side, strike, entry price)
- [ ] Each message includes: Underlying, Side, Expiry, DTE, Strike, Entry, Stop, TP1, TP2, Max Hold, Call #, Dedupe Key

---

## TROUBLESHOOTING

| Symptom | Fix |
|---------|-----|
| `POLYGON_API_KEY not set` | Check `.env` file has `POLYGON_API_KEY=...` |
| Telegram 400 Bad Request | Markdown escaping issue; should be fixed |
| 0 calls generated | All options failed liquidity gates; widen `LIQUIDITY_MAX_MID` or wait for market open |
| Single ticker only | Liquidity only found on one ticker; expected pre-market |
| Process dies | Check `cpl_output.log` for errors; restart with same command |

---

## CRITICAL REMINDERS

1. **DO NOT** run multiple instances simultaneously (duplicate alerts)
2. **DO NOT** change `DIVERSITY_MODE` unless you want single-ticker concentration
3. **DO** keep your laptop/server awake until Friday 5 PM ET
4. **DO** check Telegram periodically to confirm alerts arriving
5. **DO** back up `wsb_snake.db` after event for proof

---

*System ready. Execute pre-flight and go live.*
