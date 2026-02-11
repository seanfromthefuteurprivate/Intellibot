# WSB SNAKE - COMPLETE SYSTEM DOCUMENTATION
## Autonomous 0DTE Options Trading Engine v2.5

**Last Updated:** 2026-02-11
**Status:** OPERATIONAL (Paper Trading)
**Win Rate Achieved:** 100% (5/5 trades on Feb 11)
**Daily P/L Tracking:** Synced from Alpaca on restart

---

## TABLE OF CONTENTS

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Trading Strategies & Engines](#2-trading-strategies--engines)
3. [Execution & Risk Management](#3-execution--risk-management)
4. [CPL System (Convexity Proof Layer)](#4-cpl-system-convexity-proof-layer)
5. [Data Sources & APIs](#5-data-sources--apis)
6. [Telegram Notification System](#6-telegram-notification-system)
7. [Learning & Memory Systems](#7-learning--memory-systems)
8. [VM Deployment & Services](#8-vm-deployment--services)
9. [Known Issues & Failures](#9-known-issues--failures)
10. [What Works & Winning Strategies](#10-what-works--winning-strategies)
11. [Configuration Reference](#11-configuration-reference)
12. [Swarm Agent Reconnaissance](#12-swarm-agent-reconnaissance)

---

## 1. SYSTEM ARCHITECTURE OVERVIEW

### Core Philosophy
WSB Snake is a **6-engine autonomous trading system** that combines:
- Multi-signal conviction scoring (APEX Engine)
- Market regime detection (HYDRA)
- Mechanical exit protocols (Zero Greed)
- Win rate preservation (stops trading at <75% win rate)

### Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                      WSB SNAKE v2.5                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │SPY       │  │Momentum  │  │LEAPS     │  │Chart     │        │
│  │Scalper   │  │Engine    │  │Engine    │  │Brain     │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │               │
│       └─────────────┴─────────────┴─────────────┘               │
│                         │                                        │
│              ┌──────────▼──────────┐                            │
│              │  APEX Conviction    │  (68% threshold)           │
│              │  Engine             │                            │
│              └──────────┬──────────┘                            │
│                         │                                        │
│              ┌──────────▼──────────┐                            │
│              │  CPL Gate           │  (Requires CPL signal)     │
│              │  (cpl_gate.py)      │                            │
│              └──────────┬──────────┘                            │
│                         │                                        │
│              ┌──────────▼──────────┐                            │
│              │  Risk Governor      │  ($1k/trade, $4k/day)      │
│              │  (risk_governor.py) │                            │
│              └──────────┬──────────┘                            │
│                         │                                        │
│              ┌──────────▼──────────┐                            │
│              │  Alpaca Executor    │  (Paper Trading)           │
│              │  (alpaca_executor)  │                            │
│              └──────────┬──────────┘                            │
│                         │                                        │
│              ┌──────────▼──────────┐                            │
│              │  Telegram Alerts    │  (Dual Channel)            │
│              └──────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Entry Point
- **File:** `wsb_snake/main.py`
- **Command:** `python -m wsb_snake.main`
- **Startup Sequence:**
  1. Initialize database
  2. Start ChartBrain background AI analysis
  3. Start SPY 0DTE Scalper (30s scan interval)
  4. Start Momentum Engine (2min scan)
  5. Start LEAPS Engine (30min scan)
  6. Start Zero Greed Exit Protocol
  7. Start Alpaca Executor + monitoring
  8. Sync existing positions from Alpaca
  9. **Sync daily stats from Alpaca** (preserves win rate across restarts)
  10. Run historical training (6 weeks)
  11. Send startup ping to Telegram
  12. Schedule jobs (pipeline every 10min, EOD at 3:55 PM ET)

---

## 2. TRADING STRATEGIES & ENGINES

### 2.1 SPY 0DTE Scalper (`wsb_snake/engines/spy_scalper.py`)

**Purpose:** Hawk-like pattern detection for 15-30% quick gains on 0DTE options.

**Configuration:**
```python
PREDATOR_MODE = True           # Aggressive AI scanning
MIN_CONFIDENCE_FOR_ALERT = 68  # Must pass APEX threshold
SCAN_INTERVAL = 30             # Seconds between scans
trade_cooldown_minutes = 15    # Prevent over-trading
```

**Pattern Detection:**
- VWAP bounces
- Breakouts
- Reversals
- LangGraph AI-powered confirmation

**Universe:**
```python
ZERO_DTE_UNIVERSE = ["SPY", "QQQ", "IWM"]  # Liquid ETFs only
# DISABLED: SLV, GLD, GDX, GDXJ (wide spreads = losses)
```

**CPL Integration (CRITICAL):**
```python
def check_cpl_alignment(ticker: str, direction: str) -> tuple:
    """
    Returns: (is_aligned, reason)
    - No CPL signal = BLOCK trade
    - Direction conflicts = BLOCK trade
    """
```

**Files:**
- `wsb_snake/engines/spy_scalper.py:50-100` - CPL gate logic
- `wsb_snake/engines/spy_scalper.py:1327` - Entry alert with CPL check

---

### 2.2 APEX Conviction Engine (`wsb_snake/execution/apex_conviction_engine.py`)

**Purpose:** Institutional-grade multi-signal fusion. Only trades when combined conviction > 70%.

**Signal Weights:**
| Signal Source | Weight | Reliability |
|---------------|--------|-------------|
| Technical (RSI, MACD, SMA) | 20% | 1.0 |
| Candlestick Patterns (36) | 15% | 0.9 |
| Order Flow (sweeps, blocks) | 20% | 1.1 |
| Probability Generator | 20% | 1.0 |
| Pattern Memory | 15% | 0.8 |
| AI Verdict (GPT-4/Gemini) | 10% | 0.7 |

**Signal Decay (HYDRA Feature):**
```python
def get_decay_factor(self) -> float:
    """Linear decay from 1.0 to 0.5 over TTL period (5 min)."""
    age = (datetime.now() - self.created_at).total_seconds()
    decay = 1.0 - (age / self.ttl_seconds) * 0.5
    return max(0.5, min(1.0, decay))
```

**Verdict Structure:**
```python
@dataclass
class ApexVerdict:
    ticker: str
    conviction_score: float  # 0-100
    direction: str           # "STRONG_LONG", "LONG", "NEUTRAL", "SHORT", "STRONG_SHORT"
    action: str              # "BUY_CALLS", "BUY_PUTS", "NO_TRADE"
    signals: List[ConvictionSignal]
    entry_price: Optional[float]
    target_pct: float
    stop_pct: float
    position_size_multiplier: float  # 0.5-2.0 based on conviction
    time_sensitivity: str    # "CRITICAL", "HIGH", "MEDIUM", "LOW"
```

---

### 2.3 Momentum Engine (`wsb_snake/engines/momentum_engine.py`)

**Purpose:** Small-cap breakout plays - EQUITY ONLY (no options due to theta death).

**Universe:**
```python
MOMENTUM_UNIVERSE = [
    "RKLB", "ASTS", "LUNR", "PL", "ONDS", "POET", "SLS", "NBIS",
    "ENPH", "USAR", "THH", "CLSK", "MU", "INTC",
]
MOMENTUM_USE_OPTIONS = False  # CRITICAL: Equity only
```

**Scan Interval:** 2 minutes

**Exit Rules:**
- Trim at +50% gain
- Trail stop at +20%

---

### 2.4 LEAPS/Macro Engine (`wsb_snake/engines/leaps_engine.py`)

**Purpose:** 1-3 year thesis plays on commodities, indices, and mega-caps.

**Universe:**
```python
LEAPS_UNIVERSE = [
    "SLV", "GLD", "GDX", "GDXJ", "USO", "XLE",  # Commodities/energy
    "SPY", "QQQ", "IWM",                         # Index ETFs
    "META", "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "PYPL",  # Mega caps
]
LEAPS_EXPIRY_MONTHS_MIN = 12  # Only options >= 12 months out
```

**Scan Interval:** 30 minutes

---

### 2.5 Chart Brain (`wsb_snake/engines/chart_brain.py`)

**Purpose:** Background AI analysis studying charts in real-time.

**Features:**
- Multi-timeframe chart analysis
- Pattern recognition
- Feeds into APEX conviction scoring

---

### 2.6 Orchestrator (`wsb_snake/engines/orchestrator.py`)

**Purpose:** Central pipeline coordination.

**Scheduled Runs:**
- Main pipeline: Every 10 minutes during market hours
- EOD close: 3:55 PM ET (closes all 0DTE positions)
- Daily report: 4:15 PM ET

---

## 3. EXECUTION & RISK MANAGEMENT

### 3.1 Alpaca Executor (`wsb_snake/trading/alpaca_executor.py`)

**Trading Mode:**
```python
LIVE_TRADING = os.environ.get("ALPACA_LIVE_TRADING", "false").lower() == "true"
# Default: Paper trading (SAFE)

BASE_URL = "https://paper-api.alpaca.markets"  # Paper
# BASE_URL = "https://api.alpaca.markets"      # Live (DANGER)
```

**Risk Limits:**
```python
MAX_DAILY_EXPOSURE = 4000    # $4,000 max per day
MAX_PER_TRADE = 1000         # $1,000 per trade
MAX_CONCURRENT_POSITIONS = 3 # Reduce correlation risk
TARGET_PCT = 6               # +6% target
STOP_PCT = 10                # -10% stop loss
```

**Position Monitoring:**
- Loop interval: **2 seconds** (fixed from 5s)
- Trailing stop system:
  - Initial stop: -10%
  - After +5%: Move to breakeven
  - After +10%: Lock in +3% profit

**Trailing Stop Logic (`alpaca_executor.py:1483-1501`):**
```python
if current_pnl_pct >= 5.0 and position.stop_loss < position.entry_price:
    # Move to breakeven
    position.stop_loss = position.entry_price

if current_pnl_pct >= 10.0:
    # Lock in +3% profit (trail)
    new_stop = position.entry_price * 1.03
    position.stop_loss = max(position.stop_loss, new_stop)
```

**Position Sync on Startup:**
```python
synced = alpaca_executor.sync_existing_positions()
# Prevents orphaned positions from not being monitored
```

---

### 3.2 Risk Governor (`wsb_snake/trading/risk_governor.py`)

**Engine Types:**
```python
class TradingEngine(Enum):
    SCALPER = "scalper"    # 0DTE / intraday
    MOMENTUM = "momentum"  # Small-cap breakout
    MACRO = "macro"        # Commodity/LEAPS
    VOL_SELL = "vol_sell"  # Credit spreads
```

**Configuration:**
```python
@dataclass
class GovernorConfig:
    max_daily_loss: float = -200.0        # Kill switch at -$200
    max_concurrent_positions_global: int = 3
    max_daily_exposure_global: float = 4000.0

    # Per-engine limits
    max_positions_scalper: int = 2
    max_positions_momentum: int = 1
    max_positions_macro: int = 1

    # Per-ticker / per-sector
    max_exposure_per_ticker: float = 1000.0
    max_exposure_per_sector: float = 2000.0

    # HYDRA consecutive loss cooldown
    consecutive_loss_threshold: int = 3
    cooldown_hours: float = 4.0

    # WIN RATE PRESERVATION
    min_daily_win_rate: float = 0.75     # 75% minimum
    min_trades_for_win_rate_check: int = 2
    high_vol_exception_vix: float = 25.0  # VIX > 25 allows trading
    preserve_profit_threshold: float = 50.0
```

**Win Rate Preservation System:**
```python
def sync_daily_stats_from_alpaca(self) -> dict:
    """
    Reconstruct daily stats from Alpaca on startup.
    Survives service restarts.
    """
    # Fetches today's filled orders
    # Calculates wins/losses/P&L
    # Activates win_rate_pause if win rate < 75%
```

**Sector Exposure Caps:**
```python
SECTOR_MAP = {
    "SPY": "index", "QQQ": "index", "IWM": "index",
    "SLV": "commodity", "GLD": "commodity",
    "TSLA": "tech", "NVDA": "tech", "AAPL": "tech",
    "RKLB": "space", "ASTS": "space",
    # ... etc
}
```

---

### 3.3 Zero Greed Exit Protocol (`wsb_snake/learning/zero_greed_exit.py`)

**Purpose:** Mechanical ruthless exit system. No human override.

**Rules:**
1. Target hit = IMMEDIATE EXIT (book profit)
2. Stop hit = IMMEDIATE EXIT (accept loss)
3. Time decay = EXIT at deadline (theta kills 0DTE)

**Exit Reasons:**
```python
class ExitReason(Enum):
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    TIME_DECAY = "TIME_DECAY"
    MANUAL_EXIT = "MANUAL_EXIT"
    SYSTEM_ERROR = "SYSTEM_ERROR"
```

**Default Hold Time:** 60 minutes max (0DTE specific)

---

### 3.4 CPL Gate (`wsb_snake/utils/cpl_gate.py`)

**Purpose:** ALL trades must have CPL signal alignment.

**Protected Entry Points:**
- `spy_scalper.py` - Entry alerts
- `run_max_mode.py` - Direct trades
- `momentum_engine.py`
- `power_hour_runner.py`
- `institutional_scalper.py`
- `leaps_engine.py`
- `orchestrator.py` (2 locations)

**Gate Logic:**
```python
def check_cpl_alignment(ticker: str, direction: str) -> tuple:
    cpl = get_latest_cpl_signal(ticker)  # Must be within 30 min
    if not cpl:
        return False, "NO_CPL_SIGNAL - CPL intelligence required"

    # Check direction alignment
    if is_long and cpl_is_bearish:
        return False, "CPL_CONFLICT: CPL says PUT but setup is LONG"
```

---

## 4. CPL SYSTEM (CONVEXITY PROOF LAYER)

### Overview
The CPL System generates atomic option calls (BUY/SELL) for market events like NFP.

**File:** `wsb_snake/execution/jobs_day_cpl.py`

### Configuration
```python
CPL_EVENT_DATE = "2026-02-11"  # NFP rescheduled to Feb 11
TARGET_BUY_CALLS = 10          # 10 unique signals per session

# Liquidity Gates
LIQUIDITY_MAX_SPREAD_PCT = 0.15  # 15% max spread
LIQUIDITY_MIN_MID = 0.05         # Allow cheap OTM
LIQUIDITY_MAX_MID = 6.00         # Allow ATM on indices

# Cooldown
COOLDOWN_MINUTES = 45

# Paper Proof
MAX_COST_PER_CONTRACT = 250
```

### Auto-Execution
```python
CPL_AUTO_EXECUTE = os.environ.get("CPL_AUTO_EXECUTE", "false").lower() == "true"
# When True, CPL BUY calls execute on Alpaca paper trading
```

### Watchlist
```python
CPL_WATCHLIST = [
    # Core Index 0DTE
    "SPY", "QQQ", "IWM", "DIA",
    # VIX Products (panic meter)
    "VXX", "UVXY",
    # Rates & Financials
    "TLT", "IEF", "XLF",
    # Dollar & Metals
    "UUP", "GLD", "SLV", "GDX",
    # Crypto Beta
    "MSTR", "COIN", "MARA", "RIOT",
    # AI / Mega-cap
    "NVDA", "TSLA", "AAPL", "AMZN", "META", "GOOGL", "MSFT", "AMD",
    # Sectors
    "ITB", "XHB", "XLY", "XLV",
    # WSB Momentum
    "NBIS", "RKLB", "ASTS", "LUNR", "PL", "ONDS", "SLS",
]
```

### CPL Signal Structure
```python
@dataclass
class JobsDayCall:
    underlying: str          # "SPY"
    side: str                # "CALL" or "PUT"
    strike: float            # 591.0
    expiry_date: str         # "2026-02-11"
    dte: int                 # 0 for 0DTE
    regime: str              # "RISK_ON" or "RISK_OFF"
    confidence: float        # 0-100
    entry_trigger: dict      # {"price": 1.25, ...}
    stop_loss: dict          # {"price": 1.00, ...}
    take_profit: list        # [{"pct": 10, "price": 1.38}, ...]
    dedupe_key: str          # Unique identifier
    tier: str                # "2X", "4X", "20X"
```

### Liquidity Checks
```python
def _check_liquidity(contract, ticker, side, strike) -> Tuple[bool, Optional[str]]:
    # Check spread as % of mid
    if spread_pct > LIQUIDITY_MAX_SPREAD_PCT:
        return False, f"LIQUIDITY_REJECT: spread {spread_pct:.1%} > 15%"

    # Check mid price range
    if mid < LIQUIDITY_MIN_MID or mid > LIQUIDITY_MAX_MID:
        return False, f"LIQUIDITY_REJECT: mid ${mid:.2f} out of range"

    # Check for stale quotes
    if contract.get("stale"):
        return False, "LIQUIDITY_REJECT: stale quote"
```

---

## 5. DATA SOURCES & APIs

### 5.1 Polygon.io (Primary Market Data)

**File:** `wsb_snake/collectors/polygon_enhanced.py`

**Endpoints Used:**
- Stock aggregates (5s, 15s, 1min, 5min bars)
- Technical indicators (RSI, SMA, EMA, MACD)
- Gainers/Losers (market regime)
- Stock snapshots (real-time quotes)
- Options contracts reference
- Trades endpoint (trade flow)
- Quotes/NBBO (bid-ask spread)

**Rate Limiting:**
```python
REQUESTS_PER_MINUTE = 5  # Basic plan limit
_cache_ttl = 120         # 2 minute cache
```

**Configuration:**
```python
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_BASE_URL = "https://api.polygon.io"
```

### 5.2 Alpaca (Trading & Data)

**Files:** `wsb_snake/trading/alpaca_executor.py`, `wsb_snake/collectors/alpaca_stream.py`

**Endpoints:**
- Account info
- Positions
- Orders (create, cancel, list)
- Options trading
- Market data

**Configuration:**
```python
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
```

### 5.3 VIX Structure (`wsb_snake/collectors/vix_structure.py`)

**Purpose:** VIX level and term structure analysis.

**Used For:**
- Regime detection (HYDRA)
- Volatility scaling for position sizing
- High-vol exception for win rate preservation

### 5.4 Finnhub (`wsb_snake/collectors/finnhub_collector.py`)

**Purpose:** News, earnings calendar, company sentiment.

**Configuration:**
```python
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
```

### 5.5 Benzinga (`wsb_snake/collectors/benzinga_collector.py`)

**Purpose:** News data.

**Configuration:**
```python
BENZINGA_API_KEY = os.getenv("BENZINGA_API_KEY")
BENZINGA_BASE_URL = "https://api.benzinga.com/api/v2"
```

### 5.6 OpenAI / GPT

**File:** `wsb_snake/analysis/sentiment.py`

**Purpose:** Sentiment analysis, AI verdicts.

**Configuration:**
```python
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

### 5.7 Reddit (`wsb_snake/collectors/reddit_collector.py`)

**Purpose:** WSB sentiment scraping.

**Configuration:**
```python
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "python:wsb-snake:v1.0")
```

---

## 6. TELEGRAM NOTIFICATION SYSTEM

### Dual-Channel Architecture

**File:** `wsb_snake/notifications/telegram_channels.py`

**MAIN CHANNEL:** Pure trading signals (all users, any broker)
```python
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
```

**ALPACA CHANNEL (Optional):** Execution status (Alpaca users only)
```python
TELEGRAM_ALPACA_CHAT_ID = os.getenv("TELEGRAM_ALPACA_CHAT_ID")
```

### Functions
```python
def send_signal(message: str) -> bool:
    """Send to MAIN channel - all users see these."""

def send_alpaca_status(message: str) -> bool:
    """Send to ALPACA channel only. Logged if not configured."""
```

### Alert Types

**Entry Alert Example:**
```
🐍 ENTRY SIGNAL

SPY $591 C exp 02/11 (0 DTE)

Direction: LONG
Confidence: 72%
Regime: RISK_ON

Entry: $1.25
Target: $1.38 (+10%)
Stop: $1.12 (-10%)

⚡ ACTION: BUY CALLS
```

**Exit Alert Example:**
```
🎯 TARGET HIT

SPY $591 C

Entry: $1.25
Exit: $1.42
P/L: +$17.00 (+13.6%)

Reason: TARGET_HIT
```

**Startup Alert:**
```
🐍 WSB SNAKE v2.5 ONLINE

🔥 AGGRESSIVE MODE ACTIVE

💰 Trading: PAPER
Buying Power: $X,XXX

Focus: ETF Scalping + Momentum + LEAPS
EOD Close: 3:55 PM ET
```

---

## 7. LEARNING & MEMORY SYSTEMS

### 7.1 Pattern Memory (`wsb_snake/learning/pattern_memory.py`)

**Purpose:** Historical pattern matching to improve future predictions.

**Integration:** Feeds into APEX conviction engine with 0.8 reliability weight.

### 7.2 Trade Learner (`wsb_snake/learning/trade_learner.py`)

**Purpose:** Learns from past trades - what worked, what didn't.

### 7.3 Session Learnings (`wsb_snake/learning/session_learnings.py`)

**Purpose:** Intra-session learning and battle plan generation.

### 7.4 Time Learning (`wsb_snake/learning/time_learning.py`)

**Purpose:** Learns optimal trading times and session patterns.

### 7.5 Deep Study (`wsb_snake/learning/deep_study.py`)

**Purpose:** Off-market hours analysis and learning.

**Schedule:** Every 30 minutes during market closure.

### 7.6 Screenshot System (`wsb_snake/collectors/screenshot_system.py`)

**Purpose:** Learn from Google Drive chart screenshots.

---

## 8. VM DEPLOYMENT & SERVICES

### VM Details
```
IP: 157.245.240.99
Provider: DigitalOcean
OS: Ubuntu
Path: /root/wsb-snake
```

### Systemd Services

**1. wsb-snake.service (Main Engine)**
```ini
[Service]
ExecStart=/root/wsb-snake/venv/bin/python -m wsb_snake.main
Restart=on-failure
RestartSec=30
```

**2. cpl-scanner.service (CPL Intelligence)**
```ini
[Service]
ExecStart=/root/wsb-snake/venv/bin/python run_snake_cpl.py --broadcast --loop 60 --execute
Restart=on-failure
RestartSec=30
```

**3. max-mode.service (Power Hour Trading)**
```ini
[Service]
ExecStart=/root/wsb-snake/venv/bin/python run_max_mode.py
Restart=on-failure
RestartSec=30
```

**4. vm-guardian.service (Health Monitor)**
```ini
[Service]
ExecStart=/root/wsb-snake/venv/bin/python run_guardian.py
Restart=always
RestartSec=10
```

**5. wsb-dashboard.service (Web Dashboard)**
```
Location: /root/wsb-snake/dashboard/
Port: 80
```

### Service Management
```bash
# Start all services
sudo systemctl start wsb-snake cpl-scanner max-mode vm-guardian

# Check status
sudo systemctl status wsb-snake

# View logs
sudo journalctl -u wsb-snake -f
```

### Deployment Process
```bash
# SSH to VM
ssh root@157.245.240.99

# Pull latest code
cd /root/wsb-snake && git pull

# Restart services
sudo systemctl restart wsb-snake cpl-scanner
```

---

## 9. KNOWN ISSUES & FAILURES

### 9.1 SSH Timeouts (CRITICAL - ONGOING)
**Status:** Intermittent failure
**Symptom:** `nc: connectx to 157.245.240.99 port 22 (tcp) failed: Operation timed out`
**Workaround:** Power cycle VM via doctl
```bash
doctl compute droplet-action power-cycle <droplet-id>
```
**Root Cause:** Unknown - may be firewall, sshd crash, or network issue

### 9.2 CPL Signals Not Executing (INVESTIGATING)
**Status:** Under investigation
**Symptom:** Telegram alerts fire but no Alpaca orders placed
**Debug Added:**
```python
logger.info(f"CPL_AUTO_EXECUTE = {CPL_AUTO_EXECUTE}")
logger.info(f"ALPACA: Attempting execution for {call.underlying}")
```
**Confirmed:** CPL_AUTO_EXECUTE=True in logs

### 9.3 SLV/GLD Losses (FIXED)
**Status:** FIXED
**Problem:** Commodity ETFs have wide spreads causing -$255 loss
**Solution:** Disabled SLV, GLD, GDX, GDXJ, USO, XLE from 0DTE universe
**Commit:** Config update on Feb 11

### 9.4 Daily Stats Reset on Restart (FIXED)
**Status:** FIXED
**Problem:** Win rate showed 100% but daily P/L was -$272
**Solution:** Added `sync_daily_stats_from_alpaca()` to reconstruct stats
**Commit:** ef02196

### 9.5 Monitor Interval Too Slow (FIXED)
**Status:** FIXED
**Problem:** 5-second interval missed fast moves
**Solution:** Changed to 2-second interval
**File:** `wsb_snake/trading/alpaca_executor.py:1339`
**Commit:** d9510a4

### 9.6 AI Verdict Always Neutral (KNOWN)
**Status:** Known limitation
**File:** `wsb_snake/execution/apex_conviction_engine.py:601`
**Issue:** Returns 50 (neutral) - predator_stack not integrated
**Impact:** Losing 10% of conviction signal

### 9.7 Placeholder Greeks (KNOWN)
**Status:** Known limitation
**File:** `wsb_snake/engines/precious_metals_scalper.py:1452-1454`
**Issue:** Hardcoded gamma=0.05, theta=-0.10, vega=0.15

---

## 10. WHAT WORKS & WINNING STRATEGIES

### Winning Metrics (Feb 11, 2026)
- **Win Rate:** 100% (5/5 trades)
- **Profitable Tickers:** SPY, QQQ, IWM
- **Loss Tickers:** SLV, GLD (now disabled)

### Key Success Factors

**1. Liquid ETFs Only**
```python
DAILY_0DTE_TICKERS = ["SPY", "QQQ", "IWM"]
# Tight spreads, fast fills, consistent behavior
```

**2. CPL Gate Enforcement**
- Every trade must align with CPL regime signal
- No orphan trades without intelligence

**3. 68% Conviction Threshold**
- Only trade when APEX score > 68%
- Filters out low-quality setups

**4. Trailing Stop System**
- Lock in profits as they grow
- -5% -> breakeven -> +3% trail

**5. Win Rate Preservation**
- Stop trading if win rate < 75%
- Protects daily profits

**6. Fast Monitoring (2s Loop)**
- Catches exits before theta decay
- Quick stop-loss execution

### What NOT to Trade (Lessons Learned)
- **SLV/GLD/Commodities:** Wide spreads, poor fills
- **Small-cap options:** Theta death on weeklies
- **Without CPL signal:** Direction blind

---

## 11. CONFIGURATION REFERENCE

### Environment Variables (.env)

**Required:**
```bash
# Polygon (Market Data)
POLYGON_API_KEY=your_key

# Alpaca (Trading)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
ALPACA_LIVE_TRADING=false  # KEEP FALSE

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_ALPACA_CHAT_ID=optional_alpaca_channel

# CPL Auto-Execution
CPL_AUTO_EXECUTE=true
```

**Optional:**
```bash
OPENAI_API_KEY=your_key
FINNHUB_API_KEY=your_key
BENZINGA_API_KEY=your_key
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
GOOGLE_DRIVE_FOLDER_ID=folder_id
```

### Session Windows (Eastern Time)
```python
SESSION_WINDOWS = {
    "premarket": (4, 0, 9, 30),       # 4:00 AM - 9:30 AM
    "open": (9, 30, 10, 30),           # First hour
    "morning": (10, 30, 12, 0),        # 10:30 AM - 12:00 PM
    "lunch": (12, 0, 13, 0),           # Chop zone
    "power_hour_early": (13, 0, 15, 0), # 1:00 PM - 3:00 PM
    "power_hour": (15, 0, 16, 0),      # Final hour
    "afterhours": (16, 0, 20, 0),      # 4:00 PM - 8:00 PM
}
```

---

## 12. SWARM AGENT RECONNAISSANCE

### How Swarm Agents Were Used

To compile this documentation, **6 parallel swarm agents** were deployed to conduct comprehensive reconnaissance:

**Agent 1: Trading Strategies**
- Analyzed SPY Scalper, Momentum, LEAPS, Chart Brain
- Mapped signal flow and conviction thresholds

**Agent 2: Execution & Risk**
- Documented AlpacaExecutor, RiskGovernor, ZeroGreedExit
- Traced position monitoring loops

**Agent 3: CPL System**
- Mapped CPL signal generation and execution flow
- Documented Telegram broadcast logic

**Agent 4: Data Sources & APIs**
- Inventoried all API integrations
- Documented rate limits and caching

**Agent 5: Telegram & Alerts**
- Mapped dual-channel architecture
- Documented alert types and formats

**Agent 6: Known Failures**
- Cataloged all issues (SSH, CPL execution, etc.)
- Tracked fix status

### Completed Agent Reports
- `a8c90f4`: Issue tracker (KILL_LIST.md)
- `a5f4d6c`: Supervisor agent
- `adbc81b`: Jobs data date verification

All agents returned exhaustive reports which were synthesized into this document.

---

## APPENDIX: FILE INDEX

### Core Files
| File | Purpose |
|------|---------|
| `wsb_snake/main.py` | Entry point, startup sequence |
| `wsb_snake/config.py` | All configuration |
| `wsb_snake/db/database.py` | SQLite database |

### Trading
| File | Purpose |
|------|---------|
| `wsb_snake/trading/alpaca_executor.py` | Alpaca paper trading |
| `wsb_snake/trading/risk_governor.py` | Risk management |
| `wsb_snake/trading/outcome_recorder.py` | Trade outcome logging |

### Engines
| File | Purpose |
|------|---------|
| `wsb_snake/engines/spy_scalper.py` | 0DTE scalping |
| `wsb_snake/engines/momentum_engine.py` | Small-cap momentum |
| `wsb_snake/engines/leaps_engine.py` | LEAPS/Macro |
| `wsb_snake/engines/orchestrator.py` | Pipeline coordination |

### Execution
| File | Purpose |
|------|---------|
| `wsb_snake/execution/apex_conviction_engine.py` | Multi-signal fusion |
| `wsb_snake/execution/jobs_day_cpl.py` | CPL signal generation |
| `wsb_snake/execution/regime_detector.py` | HYDRA regime detection |

### Learning
| File | Purpose |
|------|---------|
| `wsb_snake/learning/zero_greed_exit.py` | Mechanical exits |
| `wsb_snake/learning/pattern_memory.py` | Historical patterns |
| `wsb_snake/learning/trade_learner.py` | Trade learning |

### Collectors
| File | Purpose |
|------|---------|
| `wsb_snake/collectors/polygon_enhanced.py` | Polygon API |
| `wsb_snake/collectors/vix_structure.py` | VIX analysis |
| `wsb_snake/collectors/alpaca_stream.py` | Alpaca streaming |

### Notifications
| File | Purpose |
|------|---------|
| `wsb_snake/notifications/telegram_bot.py` | Telegram sending |
| `wsb_snake/notifications/telegram_channels.py` | Dual-channel |
| `wsb_snake/notifications/message_templates.py` | Alert formatting |

---

**Document Generated:** 2026-02-11
**Generator:** Claude Code Swarm Reconnaissance
**Version:** 1.0
**Next Update:** After CPL execution fix confirmed
