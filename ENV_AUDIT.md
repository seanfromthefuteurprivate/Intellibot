# Environment keys audit

All keys the app reads from `.env` (or environment). **Values are never printed** by the audit script.

---

## How to fix "MISSING" required keys

The audit fails when required keys are absent or empty in `.env`. Fix it like this:

### 1. Create or edit `.env` in the project root

From the project root (e.g. `/Users/seankuesia/Downloads/Intellibot/`):

```bash
# If you don't have .env yet, copy the template
cp .env.example .env

# Then edit .env and replace every placeholder with real values
# (Use any editor: nano, code, vim, etc.)
```

### 2. Put real values in `.env` (no quotes needed)

Use **one line per key**, no spaces around `=`. Example shape (do **not** paste real keys here):

```bash
ALPACA_API_KEY=PKxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxx
POLYGON_API_KEY=xxxxxxxxxxxx
TELEGRAM_BOT_TOKEN=123456:ABC-xxxx
TELEGRAM_CHAT_ID=123456789
OPENAI_API_KEY=sk-xxxx
DEEPSEEK_API_KEY=sk-xxxx
```

### 3. Where to get each required key

| Key | Where to get it |
|-----|-----------------|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | [Alpaca](https://alpaca.markets) → Paper Trading → API keys |
| `POLYGON_API_KEY` | [Polygon.io](https://polygon.io) → Dashboard → API key |
| `TELEGRAM_BOT_TOKEN` | Telegram: message [@BotFather](https://t.me/BotFather) → /newbot → copy token |
| `TELEGRAM_CHAT_ID` | Message your bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` → find `"chat":{"id":...}` |
| `OPENAI_API_KEY` | [OpenAI Platform](https://platform.openai.com/api-keys) → Create key |
| `DEEPSEEK_API_KEY` | [DeepSeek](https://platform.deepseek.com) → API keys |

### 4. Re-run the audit

```bash
python3 script/audit_env.py
```

If you pass a path, the script uses that file:

```bash
python3 script/audit_env.py /path/to/your/.env
```

**On the droplet:** after fixing `.env` locally, copy it to the server (or edit on the server):

```bash
scp .env root@157.245.240.99:/root/wsb-snake/.env
```

Then on the droplet:

```bash
ssh root@157.245.240.99 "cd /root/wsb-snake && python3 script/audit_env.py"
```

---

## How to run the audit

**Local (project root):**
```bash
python3 script/audit_env.py
```

**On the droplet (after rsync):**
```bash
ssh root@157.245.240.99 "cd /root/wsb-snake && python3 script/audit_env.py"
```

Or audit a specific file:
```bash
python3 script/audit_env.py /path/to/.env
```

---

## Key list (by role)

### Required (app needs these for trading + alerts + AI)

| Key | Used by | If missing |
|-----|---------|------------|
| `ALPACA_API_KEY` | Paper/live trading, positions | No trades, no account data |
| `ALPACA_SECRET_KEY` | Alpaca auth | Same as above |
| `POLYGON_API_KEY` | Market data, options, jobs tracker | No quotes, no options chain |
| `TELEGRAM_BOT_TOKEN` | Alerts | No Telegram alerts |
| `TELEGRAM_CHAT_ID` | Where to send alerts | Alerts not delivered |
| `OPENAI_API_KEY` | Predator stack (GPT-4o) | AI falls back to DeepSeek only |
| `DEEPSEEK_API_KEY` | Predator stack fallback | No AI if OpenAI also missing |

### Optional (used by specific features)

| Key | Used by | If missing |
|-----|---------|------------|
| `ALPACA_BASE_URL` | Alpaca client | Default: paper-api.alpaca.markets |
| `ALPACA_LIVE_TRADING` | Live vs paper | Default: false (paper) |
| `FINNHUB_API_KEY` | News, earnings, deep study | Those features disabled/limited |
| `BENZINGA_API_KEY` | News | Benzinga news off |
| `GEMINI_API_KEY` | LangGraph / trade extractor | Gemini path disabled |
| `REDDIT_CLIENT_ID` | Reddit sentiment | Reddit collector off |
| `REDDIT_CLIENT_SECRET` | Reddit auth | Same |
| `GOOGLE_DRIVE_FOLDER_ID` | Screenshot learning | No screenshot ingestion |
| `GOOGLE_SERVICE_ACCOUNT` | Screenshot learning | Same |
| `SCREENSHOT_SCAN_INTERVAL` | Screenshot watcher | Default: 300 sec |
| `WSB_SNAKE_DATA_DIR` | DB/playbook path | Default: wsb_snake_data |
| `WSB_SNAKE_DB_PATH` | DB file path | Default: wsb_snake_data/wsb_snake.db |

### Optional (tuning only)

| Key | Purpose |
|-----|--------|
| `RISK_MAX_DAILY_LOSS` | Kill switch (e.g. -500) |
| `RISK_MAX_CONCURRENT_POSITIONS` | Max open positions |
| `RISK_MAX_DAILY_EXPOSURE` | Max $ deployed per day |
| `SCALP_TARGET_PCT` | Take-profit multiplier |
| `SCALP_STOP_PCT` | Stop-loss multiplier |
| `SCALP_MAX_HOLD_MINUTES` | Max hold time |
| `FRED_API_KEY` | FRED economic data |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage data |
| `GEMINI_ENABLED` | Turn Gemini on/off |

---

## Ensure keys on the droplet

1. **From your machine** (where `.env` is correct), copy env to droplet (overwrites droplet `.env` – back up first if needed):
   ```bash
   scp .env root@157.245.240.99:/root/wsb-snake/.env
   ```
2. **Or** edit on the droplet and add only missing keys:
   ```bash
   ssh root@157.245.240.99
   nano /root/wsb-snake/.env
   ```
3. **Then run the audit** to confirm:
   ```bash
   ssh root@157.245.240.99 "cd /root/wsb-snake && python3 script/audit_env.py"
   ```
4. Restart the app so it picks up env:
   ```bash
   ssh root@157.245.240.99 "systemctl restart wsb-snake"
   ```

---

**Last audit (local .env):** All 7 required keys were set. Optional missing: GEMINI_API_KEY, REDDIT_*, WSB_SNAKE_DATA_DIR/DB_PATH (defaults in code).  
**Droplet:** Run the audit command above when SSH is available to confirm droplet `.env` matches.
