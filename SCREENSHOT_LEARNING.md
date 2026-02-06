# Google Drive Screenshot Learning System

## What it does

The system **watches a Google Drive folder** for new trade screenshots, extracts trade data with **GPT-4o vision**, and uses that to:

1. **Store** each screenshot’s extracted trade (ticker, type, entry/exit, P&L, pattern, etc.) in the DB.
2. **Learn** from **winners** (trades with >10% gain): it creates or updates **trade recipes** (e.g. “SPY CALLS breakout power_hour”).
3. **Apply** those recipes in live trading: the **Trade Learner** boosts confidence when the current setup matches a winning recipe, and the **probability generator** uses that when scoring signals.

**Deployed config:**

- **Folder:** `GOOGLE_DRIVE_FOLDER_ID` (e.g. `1EbGgR2r_0jxDjQWvlN9yuxlrzUPWvLf4`).
- **Polling:** Every `SCREENSHOT_SCAN_INTERVAL` seconds (default 300 = 5 min).
- **Auth:** Service account `intellibot-drive@intellibot-486323.iam.gserviceaccount.com` (ADC / impersonation); folder must be shared with that account.

---

## What it has “studied” so far

Everything learned lives in the **same SQLite DB** as the rest of the snake:

- **Path:** `wsb_snake_data/wsb_snake.db` (or `WSB_SNAKE_DB_PATH` on the droplet: `/root/wsb-snake/wsb_snake_data/wsb_snake.db`).

**Tables:**

| Table | Purpose |
|-------|--------|
| **screenshots** | Every ingested file: file_id, filename, status (processed/failed/error), extracted_data (JSON), learned_trade_id. |
| **learned_trades** | One row per extracted trade: ticker, trade_type, entry/exit price, profit_loss, profit_loss_pct, detected_pattern, setup_description, trade_date, entry_time, etc. |
| **trade_recipes** | Distilled “formulas” from **winning** trades only: name, ticker_pattern, trade_type, time_window, entry_conditions (JSON), win_rate, source_trade_count, avg_profit_pct. Used to boost confidence when the live setup matches. |

So “what it has studied” = the contents of **learned_trades** (individual trades from screenshots) and **trade_recipes** (patterns the system uses to adjust signals).

---

## How to see the outputs

### 1. HTTP API (easiest on the app/droplet)

If the FastAPI app is running (e.g. on the droplet on port 8080):

```bash
curl -s http://localhost:8080/screenshot-learning
```

Returns JSON:

- **collector:** total/processed/failed/pending screenshot counts.
- **learner:** total_learned_trades, winners, losers, avg_pnl_pct, active_recipes, top_recipes.
- **recipes:** list of active trade recipes (name, ticker, type, time_window, win_rate, etc.).
- **recent_learned_trades:** last 20 rows from `learned_trades` (ticker, P&L, pattern, date, etc.).

From your machine (replace with your droplet IP if different):

```bash
curl -s http://157.245.240.99:8080/screenshot-learning
```

### 2. CLI (on the machine where the app runs)

From the project root (or on the droplet, from `/root/wsb-snake`):

```bash
# Stats: processed count, learned trades, winners/losers, active recipes, top recipes
python -m wsb_snake.collectors.screenshot_system stats

# List all active recipes (name, ticker, type, time window, win rate, avg P&L)
python -m wsb_snake.collectors.screenshot_system recipes

# Insights for one ticker (trades, win rate, best type, best entry times)
python -m wsb_snake.collectors.screenshot_system insights --ticker SPY

# Manually process new screenshots once (no watcher)
python -m wsb_snake.collectors.screenshot_system process
```

### 3. Query the database directly

On the droplet (or wherever the DB lives):

```bash
sqlite3 wsb_snake_data/wsb_snake.db
```

Examples:

```sql
-- Counts
SELECT COUNT(*) FROM screenshots;
SELECT COUNT(*) FROM learned_trades;
SELECT COUNT(*) FROM trade_recipes WHERE is_active = 1;

-- Recent learned trades
SELECT id, ticker, trade_type, profit_loss_pct, detected_pattern, trade_date, created_at
FROM learned_trades ORDER BY created_at DESC LIMIT 20;

-- Active recipes (what the system uses to boost confidence)
SELECT name, ticker_pattern, trade_type, time_window, win_rate, source_trade_count, avg_profit_pct
FROM trade_recipes WHERE is_active = 1 ORDER BY win_rate DESC;
```

---

## Summary

| Question | Answer |
|----------|--------|
| What has it studied? | Every trade extracted from screenshots in **learned_trades**, and patterns distilled into **trade_recipes** (winners only). |
| Where is it stored? | `wsb_snake_data/wsb_snake.db` (tables: screenshots, learned_trades, trade_recipes). |
| How do I see it? | **API:** `GET /screenshot-learning`. **CLI:** `python -m wsb_snake.collectors.screenshot_system stats|recipes|insights --ticker SPY`. **DB:** `sqlite3 wsb_snake_data/wsb_snake.db` + the SQL above. |

If no screenshots have been added to the Drive folder (or the folder isn’t shared with the service account), **screenshots** and **learned_trades** will be empty and **trade_recipes** will have no recipe rows; the endpoint and CLI will still run and show zeros.
