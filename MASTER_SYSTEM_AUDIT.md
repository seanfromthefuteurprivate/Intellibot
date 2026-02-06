# MASTER SYSTEM AUDIT — FULL RUNDOWN

**Scope:** Live production trading system on DigitalOcean. Cursor chat = control surface. No daily restarts, no sandbox.  
**Rules:** No code changes, no solution proposals, no sugarcoating. Assumptions called out where needed.  
**Date:** February 2026 (audit snapshot)

---

## EXECUTIVE SUMMARY

WSB Snake is an **autonomous 0DTE/multi-horizon options trading engine** that runs paper (or live) on Alpaca. It ingests market and alternative data, runs several signal paths (SPY scalper 30s, orchestrator 10 min, momentum 2 min, LEAPS 30 min), applies AI and learning boosts, and executes via a single Alpaca API surface. **Execution safety is undermined by two AlpacaExecutor instances:** the orchestrator uses its own instance, so positions it opens are **not** in the 5-second monitor loop; they get target/stop/time only at EOD when the pipeline runs and closes all via API. Zero_greed_exit only sends Telegram alerts and does not place close orders; real exits are done only by the module-level AlpacaExecutor's `_check_exits()`. Persistence is SQLite + JSON; in-memory state (positions, cooldowns, caches) is lost on crash; restart recovery relies on `sync_existing_positions()` which repopulates from Alpaca. Learning is outcome-based (pattern_memory, time_learning, learning_memory, session_learnings) and feeds back into confidence; there is no circuit breaker or kill switch beyond risk governor and hard limits. Docs describe a single executor and mechanical exits; code has a dual-executor gap and zero_greed_exit is alert-only. **Verdict: Needs stabilization** — coherent enough to run, but one bad day (orchestrator-heavy flow + no intraday exit) or one missed EOD run could leave risk on the table; fix the executor split and clarify exit ownership before treating as "safe to evolve."

---

## 1) SYSTEM UNDERSTANDING

### What this system does

- Scans 29+ tickers (ZERO_DTE_UNIVERSE + momentum/LEAPS) for patterns and confluence.
- **SPY scalper:** 30s loop, VWAP/candlestick/momentum patterns, Predator Stack (Gemini/DeepSeek/GPT) confirmation, 85%+ alert / 90%+ auto-execute.
- **Orchestrator:** 10 min pipeline: ignition → pressure → surge → probability fusion, chop filter, state machine (LURK→STRIKE), family classifier, inception detector, Chart Brain, alt-data/pattern boosts; A+/A + should_strike + family_alive → execute_scalp_entry.
- **Momentum / LEAPS:** 2 min and 30 min scans; trim +50% / trail +20%; use module-level alpaca_executor.
- **Execution:** One Alpaca account (paper or live); orders placed via `execute_scalp_entry`; exits by target (+12%), stop (-8%), max hold (12 min scalper), or EOD (3:55 PM ET).
- **Learning:** Outcomes written to DB and to pattern_memory, time_learning, learning_memory, session_learnings; used to adjust confidence and thresholds.

### End-to-end data flow

- **Ingest:** Polygon (bars, options), Alpaca (account, orders, news, WebSocket), Finnhub/Benzinga/Reddit/SEC/Finviz/FINRA/FRED/earnings/VIX/congressional, etc. → collectors.
- **Signal:** Scalper + orchestrator + momentum + LEAPS produce setups; confidence = base + pattern/time/AI/alt-data boosts; state machine and family classifier gate "strike."
- **Learning:** Closed trades → outcome_recorder → save_outcome (signals/outcomes/trade_performance/daily_summaries) + learning_memory + pattern_memory + time_learning; session_learnings JSON; historical_trainer at startup.
- **Storage:** SQLite (config.DB_PATH = `wsb_snake.db`), session_learnings.json (`wsb_snake_data/`), charts/artifacts as needed.
- **Execution:** execute_scalp_entry (risk governor, validation, Alpaca order) → position in executor's `self.positions` → monitor loop (only module-level executor) runs _check_exits every 5s and execute_exit/close_position; EOD close via pipeline or executor's close_all_0dte_positions (API).

### What keeps it alive

- **Process:** Single long-running Python process; `main.py` (or Replit/DO entry) runs FastAPI and starts the snake in a daemon thread (`run_snake_background` → `wsb_snake.main.main()`). Main loop: `while True: schedule.run_pending(); time.sleep(1)`.
- **Schedules:** 10 min pipeline, daily 16:15 report, 30 min idle study; scalper/momentum/LEAPS on their own timers/threads; alpaca_executor monitor every 5s.
- **Persistence:** SQLite and JSON survive process restart; Alpaca is source of truth for open positions.
- **Restart assumption:** On startup, `sync_existing_positions()` (on the module-level executor only) pulls Alpaca options positions into that executor's `self.positions` so target/stop/time/EOD can run. **Assumption:** Only one process is ever running; no multi-instance coordination.

---

## 2) ARCHITECTURE & COHERENCE CHECK

### Alignment with docs

- Docs (ARCHITECTURE_OVERVIEW, END_TO_END_FLOW, INVARIANTS, etc.): single executor, mechanical exits, EOD 3:55, position monitoring every 5s.
- **Mismatch:** Orchestrator instantiates `AlpacaExecutor()` locally (orchestrator.py line 74). So there are **two** executors: (1) module-level in `alpaca_executor.py` used by main, spy_scalper, momentum_engine, leaps_engine; (2) orchestrator's local instance used for its execute_scalp_entry and EOD close. The 5s monitor runs in main and uses only the module-level executor's `self.positions`. So **orchestrator-opened positions are never checked for target/stop/time** until the next pipeline run; the only intraday close for them is EOD when the pipeline calls that executor's `close_all_0dte_positions()` (which uses `get_options_positions()` from API).
- **Mismatch:** INVARIANTS/WSB_SNAKE_DOC say "Zero Greed Exit … execute exit"; in code, zero_greed_exit only sends Telegram alerts and does not call Alpaca. Actual exits are only in alpaca_executor.execute_exit → close_position.

### Responsibilities

- **Execution:** AlpacaExecutor is the only component that places/cancels orders. Risk governor, validation, and limits live there or in callers. Clear.
- **Coupling:** Orchestrator depends on its own AlpacaExecutor and on paper_trader (for daily report and legacy fill/exit simulation). Spy_scalper depends on module-level alpaca_executor and zero_greed_exit. Main starts everything and uses module-level alpaca_executor for sync and monitoring. The split between "who opens" (two executors) and "who closes" (only module-level's monitor) is the hidden coupling.

### State

- **Explicit:** Positions in executor.positions, daily_exposure_used, daily_pnl, etc.; DB and JSON on disk.
- **Durable:** SQLite, session_learnings.json, Alpaca. In-memory positions and caches are not durable; restart + sync repopulates from Alpaca for the module-level executor only.

---

## 3) DATA & MEMORY AUDIT

### All memory/state

| Form | Location | Contents |
|------|----------|----------|
| **SQLite** | config.DB_PATH = `wsb_snake.db` (cwd) | signals, outcomes, paper_trades, model_weights, daily_summaries, trade_performance (db/database.py init_database). |
| **JSON** | `wsb_snake_data/session_learnings.json` (session_learnings) | battle_plan, daily_learnings. |
| **SQLite (scalper)** | same DB, table spy_scalp_history | Scalper-specific history (spy_scalper _init_db). |
| **Other DB usage** | same DB | pattern_memory, time_learning, event_outcomes, stalking_mode, etc. use get_connection() / DB_PATH. |
| **In-memory** | AlpacaExecutor.positions, daily_exposure_used, daily_pnl, daily_trade_count, daily_reset_date | Per-instance; two instances = two sets. |
| **In-memory** | ZeroGreedExit.positions | Only SPY scalper adds; alert-only, no Alpaca close. |
| **In-memory** | Orchestrator alerts_sent_today, last_scan_time, caches in engines (e.g. ignition, pattern_memory cache) | Lost on crash. |
| **In-memory** | schedule job list (main) | Lost on crash. |
| **Alpaca** | Broker | Open positions, orders; source of truth for "what is open." |

### Persists across restarts

- SQLite file (if path exists and is writable).
- session_learnings.json (if path exists).
- Alpaca positions and orders (broker state).

### Lost on crash

- All in-memory positions and exposure counters for both executor instances.
- Which positions were opened by orchestrator vs scalper/momentum/LEAPS (not stored).
- Cooldowns, alert throttles, pipeline caches, schedule state.

### Corruption / race risks

- **SQLite:** Multiple threads use get_connection() (new connection per call). No single write lock; concurrent writes (e.g. save_outcome + save_signal) can cause busy/locked or transient errors. Default timeout may not be sufficient under load.
- **Two executors:** Orchestrator's executor and module-level executor each have their own daily_exposure_used / positions. Total exposure can exceed intended cap if both open trades. Risk governor is per-executor.
- **EOD:** If the process dies after 3:55 and before the pipeline's close_all_0dte_positions runs, 0DTE positions can be left open overnight (Alpaca may auto-exercise/expire; doc says no overnight 0DTE).
- **Assumption:** Single process, single writer to SQLite in practice; no distributed lock.

---

## 4) LEARNING & ADAPTATION AUDIT

### What the system learns

- **Pattern memory:** Similar bars → historical win rate → confidence boost (e.g. +15) when similarity and win rate above threshold.
- **Time learning:** Performance by hour/session → quality score and recommendation (OPTIMAL/GOOD/AVOID) → time_quality_score boost.
- **Learning memory (engine):** Feature weights (ignition, pressure, volume_spike, etc.) updated from outcomes; daily decay.
- **Session learnings:** Battle plan (thresholds, stops, EOD time) and daily_learnings from session_learnings.json; applied in scoring and behavior.
- **Event outcomes:** Macro/earnings event results stored for future expectations.
- **Historical trainer:** Runs at startup (e.g. 6 weeks); updates learning_memory and loads upcoming events.

### How often

- **Per closed trade:** outcome_recorder.record_trade_outcome → save_outcome + learning_memory.record_outcome + pattern_memory + time_learning.record_trade.
- **Startup:** init_database, run_startup_training, learning_memory._ensure_weights_initialized.
- **Pipeline:** Pattern memory and time_learning used during scoring; session_learnings and battle_plan read when building signals.
- **Daily:** learning_memory.apply_daily_decay in run_daily_report (16:15).

### From what signals

- Realized PnL, exit_reason, entry/exit time, symbol, engine; optional bars for pattern_memory. Outcome type (win/loss/timeout/scratch) from exit reason and PnL.

### How feedback loops close

- Close → outcome_recorder → DB + learning_memory + pattern_memory + time_learning → next signal uses higher/lower confidence or time/session adjustments. No automatic rollback of bad streaks; no "stop trading after N losses" in code (risk governor and daily exposure cap only).

### Weak learning signals

- Small sample per ticker/hour; pattern similarity can be noisy; session_learnings are hand-curated and may drift from current strategy.
- No explicit regime detection (e.g. high vs low vol) to down-weight or pause learning.

### Overfitting

- Pattern memory and time_learning can overfit to recent regimes; daily decay and similarity thresholds mitigate but are fixed.
- No train/validation split; all outcomes update the same model.

### Missing feedback

- No direct feedback from "order rejected" or "fill worse than quote" into learning. No sentiment from Telegram or operator. Slippage and spread not stored per trade for learning.

---

## 5) EXECUTION & SAFETY AUDIT

### Capital safety

- **Hard limits:** MAX_PER_TRADE $1,500, MAX_DAILY_EXPOSURE $6,000, MAX_CONCURRENT_POSITIONS 5 (per executor instance). Risk governor can_block by engine/ticker/exposure/PnL.
- **Validation:** Entry/target/stop/R:R and sign checks before order; oversized fill triggers alert and auto-close (e.g. > 1.5× MAX_DAILY_EXPOSURE).
- **Paper default:** LIVE_TRADING false unless ALPACA_LIVE_TRADING=true.

### Execution assumptions

- Single Alpaca account; single process; one "true" executor for monitoring (module-level). In reality, two executors; orchestrator's positions are not in the 5s monitor.
- EOD close relies on pipeline running after 3:55 or on that executor's close_all_0dte_positions (API). If process is down or pipeline blocked, 0DTE can remain open past 3:55.
- sync_existing_positions only runs on module-level executor at startup; any position opened by orchestrator and still open after restart is then synced into the single executor's state (from API), so after restart they are monitored. So the gap is **intraday** only.

### Failure modes

| Failure | Effect | Protections | Gaps |
|--------|--------|-------------|------|
| API outage (Alpaca/Polygon) | No quotes, no orders, no sync | Retries/timeouts in places; no global circuit breaker | Monitor loop keeps running; no "pause trading" on repeated failure. |
| Bad signal (wrong direction/level) | Wrong trade | Validation, AI confirmation, confidence thresholds | No ex-post kill or max loss per trade beyond stop. |
| Spike / flash crash | Slippage, wide spreads | Market orders; no explicit spread check at exit | Could sell at worse than intended. |
| Drift (thresholds, weights) | More or fewer trades | Learning updates; session_learnings | No automatic "reset to safe defaults" or versioned config. |
| Process kill / crash | In-memory state lost | sync_existing_positions on restart | If restart fails or is delayed, EOD close may not run. |
| Orchestrator opens, monitor doesn't see | Position not closed by target/stop/time | EOD close via API when pipeline runs | Intraday: position can run past target or stop until 3:55. |

### Catastrophic loss (what could cause it)

- **Live mode + bug:** Wrong side, double size, or no stop due to a bug (e.g. wrong executor, or monitor not seeing position). Mitigated by paper default and validation.
- **Two executors:** Combined exposure could exceed $6k if both executors open; risk governor is per-instance. Lower in paper; in live, could breach intended cap.
- **EOD missed:** Process down or pipeline stuck after 3:55 → 0DTE left open overnight; expiry/assignment risk. Doc says "no overnight 0DTE"; code depends on one process and one EOD path.

### Protections that exist

- Paper default, validation, hard caps, risk governor, oversized-position auto-close, EOD close via API, sync on startup, Telegram alerts.

### Protections missing

- Single executor (or shared state) so all positions are monitored by one loop.
- Circuit breaker (e.g. "pause trading after N consecutive failures" or "if Alpaca unreachable for X minutes").
- Explicit "EOD timer" (e.g. schedule at 15:55 ET) independent of pipeline run.
- Stored "last EOD close" or health check so operator can see if EOD was applied.

---

## 6) PRODUCTION READINESS

### Reliability: C

- Process and threads keep running; schedules and monitor loop are simple. But: two executors and no intraday exit for orchestrator-opened positions; EOD depends on one code path and one process; SQLite under concurrent use can busy/lock; no circuit breaker. One bad deployment or hang can leave risk on.

### Observability: C+

- Telegram for alerts, fills, exits, EOD close, daily report. Logs to stdout (and in DO to files if configured). No structured metrics (e.g. Prometheus), no dashboard, no single "health" that includes "all positions monitored" or "EOD ran." Operator must infer from Telegram and logs.

### Recoverability: B-

- Restart + sync_existing_positions repopulates from Alpaca; DB and JSON persist. No automated backup of SQLite in repo; no documented restore. Recoverable for single-node, single-process.

### Change safety: C

- No automated tests in CI for execution path; no canary. Hot-safe vs restart not documented per change. Invariants documented but dual-executor and zero_greed behavior don't match docs.

### What would break silently

- Orchestrator-opened positions not getting target/stop/time (only EOD). Exposure over cap if both executors used heavily. SQLite busy/locked under load. Learning weights or session_learnings drift.

### What would page an operator

- Telegram alerts (exit, EOD close, errors if sent). Process crash (if supervised and restarted). Alpaca/Polygon down (if alerts on failure are added).

### What would fail unnoticed

- Zero_greed_exit "exit" alerts without an actual close (human might think it closed). Pipeline not running (e.g. schedule stuck) so EOD close not called. Second executor's positions not monitored.

---

## 7) REPLIT → CURSOR MIGRATION GAP

### What Replit was implicitly handling

- **Secrets:** Replit Secrets; now env vars (e.g. .env or DO env). Manual or platform config.
- **Always-on:** Replit Keep Alive / always-on; now DO process (systemd or app platform). Manual setup.
- **Single run:** One Replit container; same as single DO process. No change.
- **Port/binding:** Replit exposed port; now PORT env (e.g. 8080). Documented.
- **Visibility:** Replit console + logs; now DO logs / console. Same idea, different tooling.

### What is now manual

- Env/secrets on DO. Restart policy and log rotation. Backups of SQLite and JSON. Monitoring (beyond Telegram). Any "run tests before deploy" or canary.

### Where the current setup is weaker

- No built-in secret management UI. No Replit-style "one click" run. Operator must know DO and process model. Dual-executor and EOD dependency are code issues, not Replit vs Cursor.

### Where it is already stronger

- Full repo in Cursor; better for refactors and audit. Explicit deployment (DO) and no Replit vendor lock-in. Docs and invariants written down; gaps are at least identifiable.

---

## 8) CRITICAL SHORTCOMINGS (TOP 10)

Ordered by risk, then leverage, then ease of fixing.

1. **Orchestrator uses its own AlpacaExecutor; those positions are not in the 5s monitor** (execution safety). **Risk:** High — intraday target/stop/time never run for pipeline-opened positions. **Leverage:** High — one fix restores intended behavior. **Ease:** Medium — use module-level alpaca_executor in orchestrator (or merge state into one executor).

2. **EOD close depends on pipeline run or one executor** (execution safety). **Risk:** High — if process or pipeline fails after 3:55, 0DTE can stay open. **Leverage:** High. **Ease:** Medium — e.g. dedicated schedule at 15:55 ET and/or health check "EOD ran."

3. **Two executors → two exposure pools; combined exposure can exceed cap** (execution safety). **Risk:** Medium in live. **Leverage:** High. **Ease:** Same as (1) — single executor.

4. **Zero_greed_exit only alerts; does not place closes** (logic / docs). **Risk:** Medium — confusion, possible belief that "exit" happened. **Leverage:** Low — alpaca_executor already closes. **Ease:** Easy — doc change and/or remove duplicate tracking or make zero_greed call executor.close_position.

5. **No circuit breaker on API failure** (execution safety / infra). **Risk:** Medium — keep trading with stale/missing data. **Leverage:** Medium. **Ease:** Medium — e.g. "pause trading after N Alpaca/Polygon failures."

6. **SQLite concurrent access** (infra). **Risk:** Medium — busy/locked or failed writes. **Leverage:** Medium. **Ease:** Medium — connection pool, serialized writes, or WAL + timeout.

7. **No backup/restore for SQLite/JSON** (infra / recoverability). **Risk:** Medium — data loss. **Leverage:** Medium. **Ease:** Easy — cron/systemd backup script.

8. **Learning can overfit; no regime-aware pause** (learning). **Risk:** Low–medium. **Leverage:** Medium. **Ease:** Hard — regime detection and gating.

9. **Observability: no metrics, no "EOD ran" / "all positions monitored"** (infra). **Risk:** Low — slow to detect issues. **Leverage:** Medium. **Ease:** Medium — health endpoint and/or simple metrics.

10. **session_learnings path `wsb_snake_data/` vs DB_PATH cwd** (infra). **Risk:** Low — wrong path or missing dir. **Leverage:** Low. **Ease:** Easy — align paths and document.

---

## 9) SYMPHONY CHECK

### Is this system currently coherent?

- **Mostly.** One Alpaca account, one process, one DB, one intended execution model in docs. But: two executors and "who closes what" (monitor vs zero_greed vs EOD) don't match the doc's mental model. So **conceptually coherent, implementation incoherent** in one critical place (executor/monitor split).

### Is it fragile or resilient?

- **Fragile.** It can run and make money in paper, but: orchestrator-opened positions are not mechanically exited intraday; EOD is single-path; no circuit breaker; SQLite and in-memory state have known risks. A single "orchestrator-heavy day" or a missed EOD run exposes the fragility.

### One refactor away from failure or strength?

- **One refactor (single executor + guaranteed EOD) away from strength.** Fix the executor/monitor split and harden EOD (e.g. scheduled 15:55 + health check), and the system becomes much closer to "safe to evolve." Leaving it as-is is one bad day or one missed run away from a visible failure (overnight 0DTE or unmonitored position).

---

## FINAL VERDICT

**Needs stabilization.**

- Do not treat as "safe to evolve" until: (1) all positions are monitored by one executor and one loop, and (2) EOD close is guaranteed (e.g. dedicated schedule + verification).
- After that, address circuit breaker, SQLite concurrency, backup, and observability for "safe to evolve."
- **Dangerous** if run in live mode with current dual-executor and EOD behavior without operator awareness and manual checks.
