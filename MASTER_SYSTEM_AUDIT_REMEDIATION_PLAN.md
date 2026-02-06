# Remediation Plan — Fix Missing Links & Run Flawlessly End-to-End

This plan addresses every finding in **MASTER_SYSTEM_AUDIT.md**. It is ordered by dependency and risk: fix execution and EOD first, then persistence and observability, then learning and hygiene. Each item states **what to do**, **where to change**, **live-safe vs requires restart**, and **how it restores end-to-end correctness**.

---

## Principles

- **Single source of truth for execution:** One AlpacaExecutor instance; all opens and all monitoring go through it.
- **Guaranteed EOD:** 0DTE close must not depend on the 10-minute pipeline; use a dedicated time-based trigger.
- **No silent failures:** Health and observability must expose "EOD ran," "all positions monitored," and "trading paused" when applicable.
- **Safe defaults:** Prefer **requires restart** for executor/import changes; **live-safe** only for config/observability where explicitly stated.

---

## Phase 1: Execution — Single Executor & Guaranteed EOD

### 1.1 Use module-level alpaca_executor everywhere (CRITICAL)

**Problem:** Orchestrator and institutional_scalper each create `AlpacaExecutor()`. Positions they open are not in the 5s monitor loop; only the module-level executor's positions get target/stop/time exits. Exposure can exceed cap across two instances.

**Goal:** One executor instance for the entire process. All `execute_scalp_entry`, `close_position`, `close_all_0dte_positions`, `should_close_for_eod`, and the monitor loop use the same object.

**Changes:**

| File | Change | Live-safe? |
|------|--------|------------|
| `wsb_snake/engines/orchestrator.py` | Replace `from wsb_snake.trading.alpaca_executor import AlpacaExecutor` and `alpaca_executor = AlpacaExecutor()` with `from wsb_snake.trading.alpaca_executor import alpaca_executor`. Remove local instantiation. | **No — requires restart** (import and object identity change). |
| `wsb_snake/engines/institutional_scalper.py` | Replace `from wsb_snake.trading.alpaca_executor import AlpacaExecutor` and `self.executor = AlpacaExecutor()` with `from wsb_snake.trading.alpaca_executor import alpaca_executor` and use `alpaca_executor` instead of `self.executor` everywhere in that class. | **No — requires restart** (import and instance reference). |

**Verification:** After deploy, only one AlpacaExecutor is ever constructed (in `alpaca_executor.py`). Grep for `AlpacaExecutor()` — the only occurrence should be the module-level `alpaca_executor = AlpacaExecutor()`.

**End-to-end:** Every position opened by scalper, orchestrator, momentum, LEAPS, or institutional scalper is added to the same `self.positions`. The 5s monitor loop in main therefore applies target/stop/time to **all** positions.

---

### 1.2 Guaranteed EOD close — dedicated 3:55 PM ET trigger

**Problem:** EOD close runs only when the 10-minute pipeline runs and calls `should_close_for_eod()` / `close_all_0dte_positions()`. If the process is blocked or the pipeline is late, 0DTE can remain open past 3:55.

**Goal:** A dedicated, time-based trigger that runs at 3:55 PM ET regardless of pipeline. Optionally record "last EOD run" for health.

**Changes:**

| File | Change | Live-safe? |
|------|--------|------------|
| `wsb_snake/main.py` | Use a schedule that runs every minute (or every 5 minutes) and, if current time ET >= 15:55 and < 17:00, call `alpaca_executor.close_all_0dte_positions()` and set a "last_eod_run" timestamp (e.g. in a module-level variable or small JSON file). Guard so this close runs at most once per calendar day (e.g. key by date). | **Yes** (add schedule + function; no import change). |
| Alternative | If you prefer not to add a per-minute job: add a dedicated schedule at 15:55. Python `schedule` uses local time; ensure the server timezone is US/Eastern, or implement a wrapper that checks `datetime.now(pytz.timezone('US/Eastern'))` and runs the close when hour==15 and minute>=55. | **Yes** (config/timezone dependent). |

**Implementation sketch (main.py):**

- Add: `schedule.every(1).minutes.do(run_eod_check)`.
- `run_eod_check()`: get now in ET; if (hour == 15 and minute >= 55) or (hour == 16 and minute == 0), and `last_eod_date != today`, then call `alpaca_executor.close_all_0dte_positions()`, set `last_eod_date = today`, send Telegram "EOD close ran (scheduled)."

**Verification:** On a test day, either change system time or temporarily set the trigger to "next minute" and confirm Telegram and that positions are closed. In production, log "EOD check ran at …" and "EOD close ran at …" for audit.

**End-to-end:** 0DTE positions are closed by 3:55 PM ET even if the 10-minute pipeline never runs in that window.

---

### 1.3 Zero Greed Exit — clarify role and optionally wire to executor

**Problem:** Docs say Zero Greed "executes" exit; in code it only sends Telegram alerts. Actual closes are done only by `alpaca_executor._check_exits()`. Risk: operators may believe the alert alone closed the position.

**Goal:** Single source of truth for "who closes": AlpacaExecutor. Zero Greed is either alert-only (documented) or calls the executor when it detects exit (one path, no double-close).

**Option A (recommended):** Keep current behavior; document clearly.

| File | Change | Live-safe? |
|------|--------|------------|
| `INVARIANTS.md` / `WSB_SNAKE_DOCUMENTATION.md` / `MASTER_SYSTEM_AUDIT.md` | State: "Zero Greed Exit sends Telegram alerts when target/stop/time conditions are met. Actual order placement to close positions is performed only by AlpacaExecutor's monitor loop (_check_exits every 5s)." | **Yes** (docs only). |

**Option B:** Have Zero Greed call `alpaca_executor.close_position(option_symbol)` when it detects exit (and then remove that position from its own tracking to avoid duplicate alerts). Then either remove the same exit logic from `_check_exits` for scalper-originated positions, or leave both (first to act wins; idempotent close). Option B is more invasive and can introduce double-close if not careful; prefer Option A unless you want a single "exit decision" component.

**End-to-end:** No silent belief that "alert = closed"; all closes go through AlpacaExecutor.

---

## Phase 2: Persistence & Data Safety

### 2.1 SQLite concurrent access

**Problem:** Multiple threads call `get_connection()` (new connection per call). Concurrent writes (e.g. save_outcome and save_signal) can cause SQLite busy/locked.

**Goal:** Serialize writes or use a single connection with a lock so only one writer at a time.

**Changes:**

| File | Change | Live-safe? |
|------|--------|------------|
| `wsb_snake/db/database.py` | Introduce a module-level `_write_lock = threading.Lock()`. In `save_signal`, `save_outcome`, `save_paper_trade`, and any other function that writes, hold `_write_lock` for the duration of the write (open connection, execute, commit, close). Optionally increase connection timeout: `sqlite3.connect(DB_PATH, timeout=30)`. | **No — requires restart** (all callers use the same module). |

**Verification:** Run a load test (e.g. many signals + outcomes in parallel); no "database is locked" or "SQLITE_BUSY."

**End-to-end:** Every signal and outcome is durably written; no lost learning or audit trail due to lock failure.

---

### 2.2 Backup of SQLite and session_learnings.json

**Problem:** No automated backup; data loss if disk or file is corrupted.

**Goal:** Document or add a script that backs up `wsb_snake.db` and `wsb_snake_data/session_learnings.json` (or configured paths) to a dated path; recommend cron or systemd timer.

**Changes:**

| Location | Change | Live-safe? |
|----------|--------|------------|
| New file: `script/backup_state.sh` (or `.py`) | Copy `wsb_snake.db` and `wsb_snake_data/session_learnings.json` to e.g. `backups/wsb_snake_$(date +%Y%m%d_%H%M).db` and same for JSON; keep last N backups. | N/A (external script). |
| `DATABASE_SCHEMA.md` or `SYSTEM_OPERATOR_HANDBOOK.md` | Add "Backup: run script/backup_state.sh daily (cron 0 18 * * * or after 16:15 ET)." | **Yes** (docs). |

**End-to-end:** Recoverability from corruption or accidental deletion.

---

### 2.3 Align session_learnings path with config

**Problem:** session_learnings uses `wsb_snake_data/session_learnings.json`; DB_PATH is `wsb_snake.db` (cwd). Inconsistent base paths; missing dir can cause runtime errors.

**Goal:** Single base path (e.g. env `WSB_SNAKE_DATA_DIR` or config `DATA_DIR = "wsb_snake_data"`). Ensure directory exists at startup.

**Changes:**

| File | Change | Live-safe? |
|------|--------|------------|
| `wsb_snake/config.py` | Add e.g. `DATA_DIR = os.getenv("WSB_SNAKE_DATA_DIR", "wsb_snake_data")` and `SESSION_LEARNINGS_PATH = os.path.join(DATA_DIR, "session_learnings.json")`. Optionally `DB_PATH = os.path.join(DATA_DIR, "wsb_snake.db")` so DB and JSON live together. | **No — requires restart** (config load). |
| `wsb_snake/learning/session_learnings.py` | Use `from wsb_snake.config import SESSION_LEARNINGS_PATH` (or DATA_DIR) and ensure `os.makedirs(DATA_DIR, exist_ok=True)` on first use. | **No — requires restart** (import config). |

**Verification:** On a fresh run, confirm directory and file are created and paths are consistent in logs.

**End-to-end:** No silent failure due to wrong or missing path.

---

## Phase 3: Observability & Health

### 3.1 Health endpoint — EOD ran, positions, trading paused

**Problem:** No single place to see "did EOD close run today?" or "how many positions are monitored?" or "is trading paused?"

**Goal:** Extend `/health` or add `/api/status` (or both) to return: last EOD run timestamp (or date), open positions count from the single executor, and optionally a `trading_paused` flag (for Phase 4 circuit breaker).

**Changes:**

| File | Change | Live-safe? |
|------|--------|------------|
| `main.py` (or wherever `/health` / `/status` live) | Read from `alpaca_executor` (module-level): open positions count, and a `last_eod_run` value (set in 1.2). Return e.g. `{ "status": "healthy", "snake_running": true, "open_positions": N, "last_eod_run_et": "2026-02-03T15:55:00", "trading_paused": false }`. | **Yes** (add fields; backward compatible). |

**Verification:** Curl `/health` or `/status` and confirm fields appear. Operator can script a check: if after 16:00 ET `last_eod_run_et` is not today, alert.

**End-to-end:** Operator can verify EOD and position count without reading logs.

---

### 3.2 Optional: simple metrics for "all positions monitored"

**Goal:** Expose that the single executor is the one and only monitor (e.g. "monitored_positions": same as open_positions from Alpaca). After Phase 1.1, "open_positions" from the one executor equals Alpaca's count; health can state "all positions monitored" implicitly.

**Changes:** In health/status, add a line like `"all_positions_monitored": true` (always true after single-executor fix, or compare executor.positions count to Alpaca get_options_positions() count and set true only if equal). **Live-safe.**

---

## Phase 4: Circuit Breaker (API Failure)

**Problem:** If Alpaca or Polygon fails repeatedly, the system keeps trying to trade or monitor with stale/missing data.

**Goal:** After N consecutive failures (e.g. get_account or get_options_positions for Alpaca; bar fetch for Polygon), set a `trading_paused` flag, send Telegram alert, and block new trades until success or cooldown. Monitor loop can continue to *close* positions (to reduce risk) even when paused; only block *new* entries.

**Changes:**

| File | Change | Live-safe? |
|------|--------|------------|
| `wsb_snake/trading/alpaca_executor.py` (or a small `circuit_breaker.py` module) | Maintain `_consecutive_alpaca_failures` and `trading_paused`. On each `get_account`, `get_options_positions`, or order call: on success set failures to 0 and trading_paused to False; on failure increment; if >= 5 (configurable), set trading_paused True and send Telegram "Trading paused: N consecutive Alpaca failures." In `execute_scalp_entry`, if trading_paused return None immediately. Optionally clear trading_paused after 15 minutes or on next success. | **No — requires restart** if new module; **Yes** if only adding state and checks inside existing executor. |
| Polygon / data layer | Similar: track consecutive failures per collector; if above threshold, optionally skip new trades that depend on that data (or just log; circuit breaker on execution is the critical one). | **Yes** (additive). |

**Verification:** Simulate Alpaca down (wrong key or network block); confirm Telegram and that no new positions open; confirm existing positions still get exit checks if you keep monitor loop calling Alpaca for quotes.

**End-to-end:** Prevents runaway trading on bad or missing API.

---

## Phase 5: Learning & Feedback (Lower Priority)

### 5.1 Record exit fill price for PnL accuracy

**Problem:** PnL may be computed from last quote (e.g. mid) instead of actual fill; learning and reports can be slightly off.

**Goal:** When closing, store the close order ID; in monitor loop, when that order is filled, set `position.exit_price = filled_avg_price` and pass that to outcome_recorder. (Already partially present if order status is polled; ensure outcome_recorder receives the actual exit price from the fill.)

**Changes:** In `execute_exit` / `close_position`, after placing the sell order, either poll for fill and then call outcome_recorder with filled price, or in `_check_order_fills` (or a similar path) when a *close* order fills, update the position's exit_price and then call outcome_recorder. Ensure `record_trade_outcome` is called with the actual exit price from the fill. **Requires restart** (behavior change in executor).

**End-to-end:** Learning and daily reports use realized fill, not quote.

---

### 5.2 Optional: regime-aware learning pause

**Problem:** Learning can overfit to a recent regime (e.g. low vol); in a new regime, weights may be wrong.

**Goal:** (Optional) Detect regime (e.g. VIX band) and reduce learning rate or pause weight updates when regime just changed. Lower priority; document as future improvement unless you have a clear regime signal.

---

## Phase 6: Documentation & Invariants

### 6.1 Update INVARIANTS and architecture docs

**Goal:** After Phase 1, docs must match code.

**Changes:**

| File | Change |
|------|--------|
| `INVARIANTS.md` | Add: "Single executor: all trading and position monitoring use the module-level alpaca_executor. No other AlpacaExecutor instances." |
| `INVARIANTS.md` or exit section | State: "Mechanical exit: AlpacaExecutor._check_exits (every 5s) and the dedicated EOD schedule (3:55 PM ET) perform all closes. Zero Greed Exit sends alerts only." |
| `ARCHITECTURE_OVERVIEW.md` / `END_TO_END_FLOW.md` | Replace any "Zero Greed executes exit" with "Zero Greed alerts; AlpacaExecutor executes exit." Describe EOD as "dedicated schedule at 3:55 PM ET plus pipeline check." |
| `MASTER_SYSTEM_AUDIT.md` | Add a short "Remediation applied" section with dates/phase numbers once you complete phases (optional). |

**Live-safe:** Yes (docs only).

---

## Implementation Order (Checklist)

Execute in this order to avoid regressions:

1. **[ ] Phase 1.1** — Single executor (orchestrator + institutional_scalper use module-level `alpaca_executor`). **Requires restart.** Then verify: only one `AlpacaExecutor()` in codebase; all positions get target/stop/time.
2. **[ ] Phase 1.2** — EOD dedicated trigger at 3:55 PM ET in main. **Live-safe.** Verify: Telegram and logs show EOD close even if pipeline doesn’t run.
3. **[ ] Phase 1.3** — Document Zero Greed as alert-only. **Live-safe.**
4. **[ ] Phase 3.1** — Health/status with last_eod_run and open_positions. **Live-safe.**
5. **[ ] Phase 2.1** — SQLite write lock. **Requires restart.**
6. **[ ] Phase 2.2** — Backup script + docs. **N/A (external).**
7. **[ ] Phase 2.3** — Config DATA_DIR / SESSION_LEARNINGS_PATH. **Requires restart.**
8. **[ ] Phase 4** — Circuit breaker for Alpaca (and optionally Polygon). **Prefer requires restart** for clean state.
9. **[ ] Phase 5.1** — Exit fill price for PnL (optional). **Requires restart.**
10. **[ ] Phase 6.1** — Update all listed docs. **Live-safe.**

---

## End-to-End Correctness After Remediation

- **One executor:** Every open and every close goes through the same AlpacaExecutor. Exposure and position count are consistent; the 5s monitor applies to all positions.
- **EOD guaranteed:** 3:55 PM ET trigger runs regardless of pipeline; no overnight 0DTE dependency on a single pipeline run.
- **Observable:** Health shows last EOD run and open positions; operator can confirm correct behavior.
- **Durable:** SQLite writes are serialized; backup and paths are defined; no silent path or lock failures.
- **Safer under failure:** Circuit breaker pauses new trades after repeated API failure; existing positions can still be closed.
- **Docs match code:** Invariants and architecture describe a single executor and explicit exit ownership.

**Result:** The system runs flawlessly end-to-end for the intended single-process, single-account, 0DTE-no-overnight design, with no missing links between open → monitor → close → learn, and with EOD and observability guaranteed.
