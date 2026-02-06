# Remediation Deployment Plan

Execute in order. **Requires restart** after Phase 1.1, 2.1, 2.3; **live-safe** for 1.2, 1.3, 3.1, 6.

| Step | Phase | Task | Restart? | Status |
|------|--------|------|----------|--------|
| 1 | 1.1 | Orchestrator: use module-level `alpaca_executor` | Yes | Done |
| 2 | 1.1 | Institutional scalper: use module-level `alpaca_executor` | Yes | Done |
| 3 | 1.2 | main.py: dedicated EOD trigger 3:55 PM ET | No | Done |
| 4 | 1.3 | INVARIANTS: Zero Greed alert-only | No | Done |
| 5 | 3.1 | Health: last_eod_run, open_positions; fix get_positions→get_options_positions | No | Done |
| 6 | 2.1 | database.py: write lock for SQLite | Yes | Done |
| 7 | 2.2 | Backup script + handbook note | N/A | Done |
| 8 | 2.3 | config: DATA_DIR, SESSION_LEARNINGS_PATH; session_learnings use config | Yes | Done |
| 9 | 6 | Update ARCHITECTURE / INVARIANTS | No | Done |

**Deploy:** Restart process once so single executor, DB path, and write lock are active.

**DB path migration:** Default DB is now `wsb_snake_data/wsb_snake.db`. If you had `wsb_snake.db` in project root, either move it to `wsb_snake_data/wsb_snake.db` or set `WSB_SNAKE_DB_PATH=wsb_snake.db` (and ensure `wsb_snake_data/` exists for session_learnings).
