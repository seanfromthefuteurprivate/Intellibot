# WSB Snake - Complete System Handover Document

**Last Updated:** 2026-02-04 22:15 UTC
**Author:** Claude Opus 4.5 (Automated Handover)
**Purpose:** Complete system documentation for Cursor Agent onboarding

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Infrastructure & Access](#2-infrastructure--access)
3. [Environment Variables](#3-environment-variables)
4. [Architecture](#4-architecture)
5. [Core Engines](#5-core-engines)
6. [Trading Logic](#6-trading-logic)
7. [API Endpoints](#7-api-endpoints)
8. [Database Schema](#8-database-schema)
9. [Dependencies](#9-dependencies)
10. [Git History & Changes](#10-git-history--changes)
11. [Common Commands](#11-common-commands)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. System Overview

### What is WSB Snake?

WSB Snake is an **autonomous 0DTE (zero days to expiration) options scalping engine** that:

- Monitors 29 tickers (SPY, QQQ, IWM, metals, mega-caps, WSB favorites)
- Detects 10+ scalping patterns (VWAP bounce, breakouts, momentum surges, etc.)
- Uses multi-model AI confirmation (GPT-4o + DeepSeek + Gemini)
- Executes paper trades automatically via Alpaca
- Sends real-time alerts to Telegram
- Learns from past trades and screenshots

### Trading Strategy

- **Target:** 12-20% gains per trade on 0DTE options
- **Hold Time:** 5-12 minutes max (theta decay kills longer holds)
- **Risk:** 8% stop loss, 12% take profit
- **Mode:** Paper trading (ALPACA_LIVE_TRADING=false)

### Operating Hours

- **Market Hours:** 9:30 AM - 4:00 PM ET
- **Scan Interval:** Every 30 seconds
- **EOD Close:** 3:55 PM ET (mandatory close all 0DTE positions)

---

## 2. Infrastructure & Access

### DigitalOcean Droplet

| Field | Value |
|-------|-------|
| **IP Address** | `157.245.240.99` |
| **User** | `root` |
| **SSH Key** | Cursor (ID: 53703493) |
| **Droplet ID** | `549450234` |
| **Region** | NYC1 |
| **Size** | s-1vcpu-1gb (~$6/mo) |
| **OS** | Ubuntu 22.04 |

### Paths on Droplet

```
/root/wsb-snake/           # Main code directory
/root/wsb-snake/.env       # API keys (DO NOT COMMIT)
/root/wsb-snake/venv/      # Python virtual environment
/root/wsb-snake/wsb_snake_data/  # Database and learnings
/etc/systemd/system/wsb-snake.service  # Systemd service
```

### GitHub Repository

```
git@github.com:seanfromthefuteurprivate/Intellibot.git
```

### Local Development Path

```
/Users/seankuesia/Downloads/Intellibot
```

---

## 3. Environment Variables

### Required API Keys (.env)

**IMPORTANT:** Actual keys are stored in `/root/wsb-snake/.env` on the droplet and `/Users/seankuesia/Downloads/Intellibot/.env` locally. Never commit real keys to git.

```bash
# Trading & Market Data
ALPACA_API_KEY=<see-local-.env-file>
ALPACA_SECRET_KEY=<see-local-.env-file>
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_LIVE_TRADING=false

POLYGON_API_KEY=<see-local-.env-file>
FINNHUB_API_KEY=<see-local-.env-file>
BENZINGA_API_KEY=<see-local-.env-file>

# AI Models
OPENAI_API_KEY=<see-local-.env-file>
DEEPSEEK_API_KEY=<see-local-.env-file>

# Notifications
TELEGRAM_BOT_TOKEN=<see-local-.env-file>
TELEGRAM_CHAT_ID=<see-local-.env-file>

# Screenshot Learning
GOOGLE_DRIVE_FOLDER_ID=<see-local-.env-file>
SCREENSHOT_SCAN_INTERVAL=300
GOOGLE_SERVICE_ACCOUNT=<see-local-.env-file>
```

**To view actual keys:** `cat /Users/seankuesia/Downloads/Intellibot/.env`

### Tunable Parameters (via env)

| Variable | Default | Description |
|----------|---------|-------------|
| `SCALP_TARGET_PCT` | 1.12 | Take profit at +12% |
| `SCALP_STOP_PCT` | 0.92 | Stop loss at -8% |
| `SCALP_MAX_HOLD_MINUTES` | 12 | Max hold time |
| `RISK_MAX_DAILY_LOSS` | -500 | Kill switch if daily PnL < this |
| `RISK_MAX_CONCURRENT_POSITIONS` | 5 | Max open positions |
| `RISK_MAX_DAILY_EXPOSURE` | 6000 | Max daily $ deployed |

---

## 4. Architecture

### Directory Structure

```
wsb_snake/
├── main.py                 # Entry point - starts all engines
├── config.py               # Configuration and environment loading
├── analysis/               # Technical analysis and AI
│   ├── candlestick_patterns.py  # 24+ candlestick patterns
│   ├── chart_generator.py       # Generate charts for AI
│   ├── predator_stack.py        # Multi-model AI (GPT + DeepSeek + Gemini)
│   ├── scalp_langgraph.py       # LangGraph for pattern analysis
│   └── scoring.py               # Signal scoring
├── collectors/             # Data collection
│   ├── polygon_enhanced.py      # Polygon.io market data
│   ├── alpaca_stream.py         # Real-time WebSocket
│   ├── finnhub_collector.py     # News, earnings, analyst data
│   ├── scalp_data_collector.py  # 5s/15s/1m bars for scalping
│   └── screenshot_system.py     # Google Drive screenshot learning
├── engines/                # Trading engines
│   ├── spy_scalper.py           # Main 0DTE scalping engine
│   ├── momentum_engine.py       # Multi-day momentum plays
│   ├── leaps_engine.py          # Long-dated options
│   ├── orchestrator.py          # Pipeline coordinator
│   └── chart_brain.py           # Background AI analysis
├── trading/                # Execution
│   ├── alpaca_executor.py       # Paper/live trade execution
│   ├── risk_governor.py         # Risk limits and kill switch
│   └── outcome_recorder.py      # Track trade outcomes
├── learning/               # ML and pattern learning
│   ├── trade_learner.py         # Learn from screenshots
│   ├── pattern_memory.py        # Pattern recognition memory
│   ├── time_learning.py         # Time-of-day quality
│   ├── zero_greed_exit.py       # Mechanical exit system
│   └── deep_study.py            # Off-hours learning
├── notifications/          # Alerts
│   ├── telegram_bot.py          # Send Telegram alerts
│   └── message_templates.py     # Alert formatting
├── db/                     # Database
│   └── database.py              # SQLite connection
└── utils/                  # Utilities
    ├── logger.py                # Logging
    ├── rate_limit.py            # API rate limiting
    └── session_regime.py        # Market hours detection
```

### Data Flow

```
1. Market Data (Polygon/Alpaca) → scalp_data_collector
2. Pattern Detection → spy_scalper._detect_patterns()
3. AI Confirmation → predator_stack.analyze_sync()
4. Risk Check → risk_governor.can_trade()
5. Execution → alpaca_executor.execute_scalp_entry()
6. Monitoring → alpaca_executor._check_exits()
7. Learning → outcome_recorder.record_trade_outcome()
```

---

## 5. Core Engines

### SPY Scalper (`wsb_snake/engines/spy_scalper.py`)

**Purpose:** Primary 0DTE scalping engine

**Key Settings:**
```python
MIN_CONFIDENCE_FOR_AI = 80       # Only AI-analyze 80%+ setups
MIN_CONFIDENCE_FOR_ALERT = 85    # Only trade 85%+ setups
REQUIRE_AI_CONFIRMATION = True   # AI must confirm
SNIPER_MODE = True               # Only AI the best setup per cycle
trade_cooldown_minutes = 20      # Cooldown between trades
```

**Patterns Detected:**
- VWAP_BOUNCE, VWAP_RECLAIM, VWAP_REJECTION
- MOMENTUM_SURGE (long/short)
- BREAKOUT, BREAKDOWN
- FAILED_BREAKDOWN (bear trap), FAILED_BREAKOUT (bull trap)
- SQUEEZE_FIRE (volatility expansion)

**Tickers Monitored:**
```python
ZERO_DTE_UNIVERSE = [
    "SPY", "QQQ", "IWM",  # Index ETFs
    "SLV", "GLD", "GDX", "GDXJ",  # Metals
    "USO", "XLE",  # Energy
    "TSLA", "NVDA", "AAPL", "META", "AMD", "AMZN", "GOOGL", "MSFT",  # Mega caps
    "RKLB", "ASTS", "NBIS", "PL", "LUNR", "ONDS", "SLS",  # WSB space/AI
    "POET", "ENPH", "USAR", "PYPL", "THH"  # WSB YOLO
]
```

### Alpaca Executor (`wsb_snake/trading/alpaca_executor.py`)

**Purpose:** Execute paper trades on Alpaca

**Key Settings:**
```python
MAX_DAILY_EXPOSURE = 6000        # $6,000 max deployed
MAX_PER_TRADE = 1500             # $1,500 per trade
MAX_CONCURRENT_POSITIONS = 5     # 5 positions max
SCALP_TARGET_PCT = 1.12          # +12% take profit
SCALP_STOP_PCT = 0.92            # -8% stop loss
SCALP_MAX_HOLD_MINUTES = 12      # 12 min max hold
```

**Position Lifecycle:**
1. `execute_scalp_entry()` - Place buy order
2. `_check_order_fills()` - Monitor for fill
3. `_check_exits()` - Monitor price vs target/stop
4. `execute_exit()` - Close position
5. `record_trade_outcome()` - Log to learning systems

### Predator Stack (`wsb_snake/analysis/predator_stack.py`)

**Purpose:** Multi-model AI confirmation

**Models Used:**
1. **GPT-4o** (primary) - Vision-enabled, analyzes chart images
2. **DeepSeek** (fallback) - Text-only analysis
3. **Gemini** (backup) - Rate-limited due to previous suspension

**Verdicts:**
- `STRIKE_CALLS` - Buy calls
- `STRIKE_PUTS` - Buy puts
- `NO_TRADE` - Skip
- `ABORT` - Cancel/close

---

## 6. Trading Logic

### Entry Conditions

All must pass:
1. **Confidence >= 85%** (after learning boosts)
2. **AI Confirmation** (STRIKE_CALLS or STRIKE_PUTS)
3. **Order Flow Agrees** (sweep direction + 8%+ sweep volume)
4. **No Earnings** (within 2 days)
5. **Sector Not Slighted** (SPY not weak)
6. **Regime Match** (no longs in downtrend, no shorts in uptrend)
7. **Risk Governor Allows** (positions < max, exposure < max)

### Exit Conditions

Any triggers exit:
1. **Target Hit** - Price >= entry * 1.12 (+12%)
2. **Stop Loss** - Price <= entry * 0.92 (-8%)
3. **Max Hold** - 12 minutes elapsed
4. **EOD Close** - 3:55 PM ET (all 0DTE)
5. **Risk Governor** - Daily loss limit hit

### Position Sizing

```python
contract_cost = option_price * 100  # Options = 100 shares
num_contracts = min(
    MAX_PER_TRADE / contract_cost,
    (MAX_DAILY_EXPOSURE - daily_exposure_used) / contract_cost
)
```

---

## 7. API Endpoints

### Web Server (`main.py`)

The snake runs a FastAPI server on port 8080.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Basic health check |
| `/health` | GET | Health status for monitoring |
| `/status` | GET | Detailed status (positions, config, account) |

### Example `/status` Response

```json
{
  "status": "online",
  "snake_running": true,
  "uptime_seconds": 3600,
  "open_positions": 2,
  "last_eod_run_date": "2026-02-04",
  "account": {
    "buying_power": 25000.00,
    "equity": 26500.00
  },
  "config": {
    "max_daily_exposure": 6000,
    "max_per_trade": 1500,
    "min_confidence": 85
  }
}
```

---

## 8. Database Schema

### SQLite Database: `wsb_snake_data/wsb_snake.db`

#### Table: `signals`
```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    ticker TEXT,
    signal_type TEXT,
    confidence REAL,
    direction TEXT,
    entry_price REAL,
    target_price REAL,
    stop_loss REAL,
    ai_model TEXT,
    pattern TEXT,
    outcome TEXT,
    pnl REAL,
    pnl_pct REAL
);
```

#### Table: `spy_scalp_history`
```sql
CREATE TABLE spy_scalp_history (
    id INTEGER PRIMARY KEY,
    pattern TEXT,
    direction TEXT,
    entry_price REAL,
    target_price REAL,
    stop_loss REAL,
    confidence REAL,
    ai_confirmed INTEGER,
    ai_confidence REAL,
    detected_at TEXT,
    alerted_at TEXT,
    outcome TEXT,
    pnl_pct REAL,
    duration_minutes INTEGER
);
```

#### Table: `trade_outcomes`
```sql
CREATE TABLE trade_outcomes (
    id INTEGER PRIMARY KEY,
    signal_id INTEGER,
    symbol TEXT,
    trade_type TEXT,
    entry_price REAL,
    exit_price REAL,
    pnl REAL,
    pnl_pct REAL,
    exit_reason TEXT,
    entry_time TEXT,
    exit_time TEXT,
    engine TEXT
);
```

---

## 9. Dependencies

### Python Requirements (`requirements.txt`)

```
# Web Framework
fastapi>=0.109.0
uvicorn[standard]>=0.27.0

# AI/LLM
openai>=1.10.0
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-core>=0.1.0
langgraph>=0.0.20
google-generativeai>=0.3.0

# Trading & Market Data
alpaca-trade-api>=3.0.0
polygon-api-client>=1.12.0
finnhub-python>=2.4.0

# HTTP & Async
requests>=2.31.0
httpx>=0.26.0
aiohttp>=3.9.0
websockets>=9.0,<11

# Data Processing
pandas>=2.1.0
numpy>=1.26.0
scipy>=1.11.0

# Charting
matplotlib>=3.8.0
mplfinance

# Scheduling & Time
schedule>=1.2.1
pytz>=2024.1
python-dateutil>=2.8.2

# Utilities
python-dotenv>=1.0.0
tenacity>=8.2.0
beautifulsoup4>=4.12.0
lxml>=5.1.0

# Telegram
python-telegram-bot>=20.7

# Reddit (optional)
praw>=7.7.0
```

---

## 10. Git History & Changes

### Recent Commits (Most Recent First)

```
ce7996e Integrate trade_learner into trading engines for screenshot-based learning
1b9bb08 Fix repeated Telegram alerts + Add Deep Study for off-market hours
271f290 Add Screenshot Learning System for trade pattern recognition
3958284 Connect trade exits to learning systems for outcome tracking
0b77a61 Add OpenAI rate limiting for scalping-focused credit conservation
822b5d8 Add Gemini with strategic rate limiting and confluence triggers
b044026 Optimize AI API usage - only call when patterns detected
96d8509 Fix missing modules causing deployment crash
a7f0104 Add GreekOptimizer and PositionSizer to precious metals scalper
740cff9 Fix port mismatch: use port 8080 for App Platform
a82b464 Fix websockets version conflict with alpaca-trade-api
8713b1f Fix mplfinance Python 3.11 compatibility
f7de8cd Fix App Platform deployment: add missing dependencies
e984b04 Fix losing trades: 0DTE theta decay was killing all positions
3f27279 Add lethal micro-scalping system with GPT-4o vision
7c19fc3 Add rate limiting and sniper mode to prevent API suspension
90caeec Add Digital Ocean deployment configuration
```

### Key Changes Made

1. **JAN 29 FIX - Theta Decay Issue:**
   - Reduced target from +25% to +12%
   - Reduced stop from -20% to -8%
   - Reduced max hold from 30min to 12min
   - Raised confidence threshold from 75% to 85%

2. **SNIPER MODE:**
   - Only AI-analyze the BEST setup per scan cycle
   - Reduced AI calls from ~29/cycle to 1/cycle
   - Prevents API rate limit issues

3. **Screenshot Learning:**
   - Added Google Drive integration
   - Learns from winning trade screenshots
   - Adjusts confidence based on learned patterns

4. **Risk Governor:**
   - Daily PnL kill switch at -$500
   - Max 5 concurrent positions
   - Max $6,000 daily exposure

---

## 11. Common Commands

### SSH Commands

```bash
# Check service status
ssh root@157.245.240.99 "systemctl status wsb-snake --no-pager"

# View live logs
ssh root@157.245.240.99 "journalctl -u wsb-snake -f"

# View recent logs
ssh root@157.245.240.99 "cat /var/log/syslog | grep wsb-snake | tail -50"

# Restart service
ssh root@157.245.240.99 "systemctl restart wsb-snake"

# Stop service
ssh root@157.245.240.99 "systemctl stop wsb-snake"

# Check .env file
ssh root@157.245.240.99 "cat /root/wsb-snake/.env"

# Check disk space
ssh root@157.245.240.99 "df -h"

# Check memory
ssh root@157.245.240.99 "free -m"

# Check running processes
ssh root@157.245.240.99 "ps aux | grep python"
```

### Deploy Changes

```bash
# Sync local code to droplet
rsync -avz --exclude '.git' --exclude 'venv' --exclude '__pycache__' --exclude '.env' \
  -e "ssh" /Users/seankuesia/Downloads/Intellibot/ root@157.245.240.99:/root/wsb-snake/

# Install new dependencies
ssh root@157.245.240.99 "cd /root/wsb-snake && source venv/bin/activate && pip install -r requirements.txt"

# Restart after deploy
ssh root@157.245.240.99 "systemctl restart wsb-snake"
```

### doctl Commands

```bash
# List droplets
doctl compute droplet list --format ID,Name,PublicIPv4,Status

# Check account
doctl account get

# SSH via doctl
doctl compute ssh 549450234 --ssh-command "systemctl status wsb-snake"
```

### Local Development

```bash
# Run locally
cd /Users/seankuesia/Downloads/Intellibot
python main.py

# Run tests
python -m pytest tests/

# Check syntax
python3 -m py_compile wsb_snake/engines/spy_scalper.py
```

---

## 12. Troubleshooting

### Service Won't Start

```bash
# Check logs for errors
ssh root@157.245.240.99 "journalctl -u wsb-snake -n 100 --no-pager"

# Check if port is in use
ssh root@157.245.240.99 "lsof -i :8080"

# Check Python version
ssh root@157.245.240.99 "python3 --version"

# Check .env exists
ssh root@157.245.240.99 "ls -la /root/wsb-snake/.env"
```

### No Trades Executing

1. **Check market hours** - Only trades 9:30 AM - 4:00 PM ET
2. **Check confidence threshold** - Must be >= 85%
3. **Check AI confirmation** - REQUIRE_AI_CONFIRMATION=True
4. **Check risk limits** - Max positions, daily exposure
5. **Check cooldown** - 20 min cooldown between trades

### API Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Rate limit exceeded` | Too many API calls | Increase cache TTL, enable SNIPER_MODE |
| `Invalid API key` | Wrong key in .env | Check .env file on droplet |
| `No option quote` | Illiquid option | Skip trade, try different strike |
| `Insufficient buying power` | Account low | Check Alpaca account |

### Telegram Not Working

```bash
# Test Telegram directly (replace with actual token from .env)
curl -X POST "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/sendMessage" \
  -d "chat_id=<TELEGRAM_CHAT_ID>" \
  -d "text=Test message"
```

---

## Quick Start for Cursor Agent

### 1. Read This Document

```bash
cat /Users/seankuesia/Downloads/Intellibot/CURSOR_AGENT_HANDOVER.md
```

### 2. Check Snake Status

```bash
ssh root@157.245.240.99 "systemctl status wsb-snake --no-pager && cat /var/log/syslog | grep wsb-snake | tail -20"
```

### 3. Key Files to Understand

1. `wsb_snake/main.py` - Entry point
2. `wsb_snake/engines/spy_scalper.py` - Main trading logic
3. `wsb_snake/trading/alpaca_executor.py` - Trade execution
4. `wsb_snake/config.py` - Configuration

### 4. Make Changes

1. Edit files locally in `/Users/seankuesia/Downloads/Intellibot`
2. Sync to droplet: `rsync -avz ...`
3. Restart: `ssh root@157.245.240.99 "systemctl restart wsb-snake"`

---

**Document generated by Claude Opus 4.5 on 2026-02-04**
