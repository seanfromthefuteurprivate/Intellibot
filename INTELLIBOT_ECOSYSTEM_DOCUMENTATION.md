# INTELLIBOT (WSB SNAKE) - Complete Trading Bot Ecosystem Documentation

**Version:** 2.0 | **Last Updated:** February 2026 | **Status:** Production (Paper Trading)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Trading Engines](#3-trading-engines)
4. [Learning Systems](#4-learning-systems)
5. [API Integrations](#5-api-integrations)
6. [Database Schema](#6-database-schema)
7. [Risk Management](#7-risk-management)
8. [Configuration Reference](#8-configuration-reference)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [Expected Outcomes & Performance](#10-expected-outcomes--performance)
11. [Recent Additions (February 2026)](#11-recent-additions-february-2026)
12. [Planned Future Additions](#12-planned-future-additions)
13. [API Tier Comparison & Upgrade Benefits](#13-api-tier-comparison--upgrade-benefits)
14. [Troubleshooting & Monitoring](#14-troubleshooting--monitoring)

---

## 1. Executive Summary

### What is Intellibot?

Intellibot (codenamed "WSB Snake") is an autonomous options trading system designed for:
- **0DTE SPY/QQQ/IWM scalping** (15-30% quick gains)
- **Small-cap momentum plays** (RKLB, ASTS, LUNR, etc.)
- **LEAPS/macro commodity thesis** (GLD, SLV, GDX)

### Core Philosophy

```
"Hunt like a predator, not a gambler"
- Wait for high-conviction setups (State Machine)
- Execute mechanically (Zero Greed Exit)
- Learn from every outcome (Adaptive Weights)
- Never hold 0DTE overnight (Mandatory EOD Close)
```

### Key Metrics (Target)

| Metric | Target | Notes |
|--------|--------|-------|
| Win Rate | 55-65% | Higher for A+ tier signals |
| Average Winner | +15-25% | Scalp targets |
| Average Loser | -8-12% | Tight stops |
| R:R Ratio | 1.5:1+ | Minimum threshold |
| Daily P&L Target | +$150-500 | Based on $6k exposure |
| Max Daily Loss | -$500 | Kill switch |

---

## 2. System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INTELLIBOT CORE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   DATA       │    │   SIGNAL     │    │   EXECUTION  │              │
│  │ COLLECTION   │───▶│  PROCESSING  │───▶│   & TRADING  │              │
│  │   LAYER      │    │    LAYER     │    │    LAYER     │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   LEARNING   │◀───│   OUTCOME    │◀───│   POSITION   │              │
│  │   SYSTEMS    │    │  RECORDING   │    │  MANAGEMENT  │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
wsb_snake/
├── main.py                    # Entry point, scheduler, pipeline orchestration
├── config.py                  # Environment variables, trading universe
├── db/
│   └── database.py            # SQLite persistence, all table schemas
├── collectors/                # Data collection adapters
│   ├── polygon_enhanced.py    # Primary market data (stocks, options)
│   ├── polygon_options.py     # Options chains, greeks, GEX
│   ├── alpaca_stream.py       # WebSocket real-time streaming
│   ├── alpaca_news.py         # News feed
│   ├── reddit_collector.py    # r/wallstreetbets trending
│   ├── finnhub_collector.py   # News + social sentiment
│   ├── benzinga_news.py       # Premium financial news
│   ├── sec_edgar_collector.py # Insider trading filings
│   ├── finviz_collector.py    # Unusual volume metrics
│   ├── finra_darkpool.py      # Dark pool volume
│   ├── congressional_trading.py # Congressional insider trades
│   ├── fred_collector.py      # Macro regime (VIX, yields)
│   ├── vix_structure.py       # VIX term structure
│   └── earnings_calendar.py   # Earnings dates, IV rank
├── engines/                   # Signal detection engines
│   ├── ignition_detector.py   # Engine 1: Early momentum
│   ├── pressure_engine.py     # Engine 2: Options flow + technicals
│   ├── surge_hunter.py        # Engine 3: Power hour surges
│   ├── probability_generator.py # Signal fusion + scoring
│   ├── learning_memory.py     # Engine 5: Adaptive weights
│   ├── paper_trader.py        # Engine 6: Simulated execution
│   ├── state_machine.py       # Predator pattern (LURK→STRIKE)
│   ├── family_classifier.py   # Setup family classification
│   ├── inception_detector.py  # Convex instability detection
│   ├── chart_brain.py         # AI vision analysis
│   ├── spy_scalper.py         # SPY 0DTE Hawk Mode
│   ├── momentum_engine.py     # Small-cap breakout
│   ├── leaps_engine.py        # Macro/commodity LEAPS
│   └── precious_metals_scalper.py # GLD/SLV specialized
├── trading/                   # Trade execution
│   ├── alpaca_executor.py     # Real paper/live trading
│   ├── risk_governor.py       # Position limits, kill switches
│   └── outcome_recorder.py    # Learning system integration
├── learning/                  # Machine learning
│   ├── pattern_memory.py      # Historical pattern matching
│   ├── time_learning.py       # Time-of-day optimization
│   ├── session_learnings.py   # Battle-tested wisdom
│   ├── stalking_mode.py       # Wait-and-trigger setups
│   ├── event_outcomes.py      # Macro event tracking
│   └── zero_greed_exit.py     # Mechanical exit protocol
├── analysis/                  # AI analysis
│   ├── langgraph_analyzer.py  # GPT-4o chart analysis
│   └── scalp_langgraph.py     # Gemini pattern confirmation
├── notifications/             # Alerts
│   ├── telegram_bot.py        # Telegram integration
│   └── message_templates.py   # Alert formatting
└── utils/                     # Utilities
    ├── logger.py              # Logging
    ├── rate_limit.py          # API rate limiting
    ├── session_regime.py      # Market session detection
    ├── sector_strength.py     # Sector momentum
    └── time_windows.py        # Trading time utilities
```

### Pipeline Execution Flow

```
main.py (every 10 minutes)
    │
    ├─▶ Stage 0: Data Collection
    │       ├─ Reddit trending tickers
    │       ├─ Finnhub news + sentiment
    │       ├─ SEC EDGAR insider activity
    │       ├─ Dark pool volume
    │       ├─ Options flow
    │       └─ Congressional trading
    │
    ├─▶ Stage 1: Engine Detection (parallel)
    │       ├─ Ignition Detector ──▶ Score + Direction
    │       ├─ Pressure Engine ───▶ Score + Direction
    │       └─ Surge Hunter ──────▶ Score + Direction (power hour)
    │
    ├─▶ Stage 2: Signal Fusion
    │       ├─ Probability Generator (weighted combination)
    │       ├─ Chop Filter (market regime kill)
    │       ├─ State Machine (LURK→STRIKE gate)
    │       ├─ Family Classification (viability)
    │       ├─ Inception Detection (phase transition)
    │       ├─ Chart Brain Validation (AI cross-check)
    │       ├─ Alt Data Boosts (Finnhub, Finviz, SEC)
    │       └─ Pattern Memory Matching
    │
    ├─▶ Stage 3: Alert & Trade Generation
    │       ├─ A+/A tier signals only
    │       ├─ Alpaca trade execution
    │       └─ Telegram alerts
    │
    └─▶ Stage 4: Position Management
            ├─ Paper trader fills/exits
            ├─ Zero Greed exit enforcement
            └─ EOD 0DTE close (3:55 PM ET)
```

---

## 3. Trading Engines

### 3.1 Engine 1: Ignition Detector

**File:** `wsb_snake/engines/ignition_detector.py`
**Purpose:** Detect early momentum before major moves

**Signals Detected:**
| Signal | Trigger | Weight |
|--------|---------|--------|
| VOLUME_EXPLOSION | 3x+ average volume | High |
| PRICE_BREAKOUT | Breaking key resistance/support | High |
| NEWS_CATALYST | News-driven price action | Medium |
| GAP_CONTINUATION | Gap and go pattern | Medium |
| MOMENTUM_ACCELERATION | Velocity increasing | Medium |

**Input Data:**
- Polygon 1-min/5-min bars
- Volume ratios
- Price velocity/acceleration
- News sentiment (Finnhub, Benzinga)

**Output:**
```python
{
    "score": 0-100,
    "direction": "long" | "short",
    "evidence": ["VOLUME_SPIKE", "BREAKOUT", ...],
    "entry_price": float,
    "stop_loss": float,
    "target": float
}
```

### 3.2 Engine 2: Pressure Engine

**File:** `wsb_snake/engines/pressure_engine.py`
**Purpose:** Analyze options flow and technical indicators

**Signals Detected:**
| Signal | Trigger | Interpretation |
|--------|---------|----------------|
| CALL_WALL | Heavy call OI above price | Resistance level |
| PUT_WALL | Heavy put OI below price | Support level |
| TECHNICAL_BULLISH | RSI + SMA + MACD align | Long bias |
| TECHNICAL_BEARISH | RSI + SMA + MACD align | Short bias |
| MOMENTUM_SURGE | Acceleration + volume | Trend continuation |
| RSI_EXTREME | RSI < 30 or > 70 | Reversal setup |

**Input Data:**
- Polygon options chains (greeks, OI, volume)
- Technical indicators (RSI, SMA, EMA, MACD)
- Market regime (gainers/losers ratio)

**Output:**
```python
{
    "score": 0-100,
    "direction": "long" | "short",
    "put_call_ratio": float,
    "support_level": float,
    "resistance_level": float,
    "indicators": {...}
}
```

### 3.3 Engine 3: Surge Hunter

**File:** `wsb_snake/engines/surge_hunter.py`
**Purpose:** Capture power hour volatility expansion

**Active Only:** 3:00 PM - 4:00 PM ET (power hour)

**Logic:**
1. Detect compression (low range period)
2. Wait for volume surge (1.5x+)
3. Confirm direction with options flow
4. Fire signal for final hour trade

**Output:**
```python
{
    "score": 0-100,
    "direction": "long" | "short",
    "surge_type": "BREAKOUT" | "REVERSAL" | "SQUEEZE",
    "time_sensitivity": "high"
}
```

### 3.4 Probability Generator (Signal Fusion)

**File:** `wsb_snake/engines/probability_generator.py`
**Purpose:** Combine all engines into unified probability

**Weighting Formula:**
```
combined_score = (
    ignition_score * ignition_weight +
    pressure_score * pressure_weight +
    surge_score * surge_weight
) / total_weights

probability_win = combined_score / 100 * confidence_multiplier
```

**Output:**
```python
{
    "combined_score": 0-100,
    "probability_win": 0.0-1.0,
    "confidence": "low" | "medium" | "high",
    "action": "STRONG_LONG" | "LONG" | "WATCH" | "SHORT" | "STRONG_SHORT" | "AVOID",
    "direction": "long" | "short",
    "entry_price": float,
    "stop_loss": float,
    "target_1": float,
    "target_2": float,
    "risk_reward_ratio": float,
    "time_sensitivity": "low" | "medium" | "high"
}
```

**Tier Classification:**
| Tier | Score | Action |
|------|-------|--------|
| A+ | 85+ | Immediate execution |
| A | 70-84 | Execute if state = STRIKE |
| B | 55-69 | Alert only, no trade |
| C | < 55 | Log only |

### 3.5 SPY 0DTE Scalper (Hawk Mode)

**File:** `wsb_snake/engines/spy_scalper.py`
**Purpose:** High-frequency SPY scalping for 15-30% gains

**Scan Frequency:** Every 30 seconds

**Patterns Detected:**
| Pattern | Description | Target |
|---------|-------------|--------|
| VWAP_BOUNCE | Price bounces off VWAP | +12-20% |
| VWAP_RECLAIM | Price reclaims VWAP from below | +15-25% |
| VWAP_REJECTION | Price rejects at VWAP | +12-20% |
| MOMENTUM_SURGE | Strong directional move | +20-30% |
| BREAKOUT | Key level break with volume | +20-30% |
| REVERSAL | Exhaustion reversal | +15-25% |
| SQUEEZE_FIRE | Bollinger squeeze breakout | +25-35% |

**Exit Rules:**
- Target: +12% (configurable via `SCALP_TARGET_PCT`)
- Stop: -8% (configurable via `SCALP_STOP_PCT`)
- Max Hold: 12 minutes (theta decay protection)

**AI Integration:**
- Gemini: Pattern confirmation (10 RPM limit)
- GPT-4o: Chart analysis (on-demand)
- DeepSeek: Fallback

### 3.6 Momentum Engine (Small-Cap Breakout)

**File:** `wsb_snake/engines/momentum_engine.py`
**Purpose:** Capture small-cap breakout moves

**Universe:** RKLB, ASTS, LUNR, PL, ONDS, POET, SLS, NBIS, ENPH, USAR, THH, CLSK, MU, INTC

**Scan Frequency:** Every 2 minutes

**Entry Criteria:**
- Volume surge: 1.4x+ 20-day average
- Price momentum: 3%+ in 5 days OR 1.5%+ today
- Catalyst: Earnings 5-14 days OR 2+ WSB mentions

**Position Sizing:** $1,200 max per trade
**Expiry:** Weekly or next-Friday options

### 3.7 LEAPS/Macro Engine

**File:** `wsb_snake/engines/leaps_engine.py`
**Purpose:** Long-term commodity and index thesis plays

**Universe:** SLV, GLD, GDX, GDXJ, USO, XLE, SPY, QQQ, IWM, META, AAPL, NVDA, TSLA, MSFT, AMZN, GOOGL, PYPL

**Scan Frequency:** Every 30 minutes

**Entry Criteria:**
- Price > 50-day SMA
- Trend confirmation (higher highs/lows)
- Macro regime supportive

**Position Sizing:** $2,000 max per trade
**Expiry:** 12+ months (true LEAPS)

---

## 4. Learning Systems

### 4.1 Learning Memory (Adaptive Weights)

**File:** `wsb_snake/engines/learning_memory.py`
**Purpose:** Self-improving feature weights based on outcomes

**Features Tracked:**
| Feature | Default Weight | Range |
|---------|----------------|-------|
| ignition | 1.0 | 0.1-3.0 |
| pressure | 1.0 | 0.1-3.0 |
| surge | 1.0 | 0.1-3.0 |
| volume_spike | 1.0 | 0.1-3.0 |
| velocity | 1.0 | 0.1-3.0 |
| news_catalyst | 1.0 | 0.1-3.0 |
| vwap_reclaim | 1.0 | 0.1-3.0 |
| range_breakout | 1.0 | 0.1-3.0 |
| call_put_ratio | 1.0 | 0.1-3.0 |
| iv_spike | 1.0 | 0.1-3.0 |

**Learning Parameters:**
- Learning Rate: 5% per outcome
- Daily Decay: 95% (weights drift toward 1.0)
- Min Weight: 0.1
- Max Weight: 3.0

**Update Formula:**
```python
if outcome == "win":
    new_weight = old_weight + LEARNING_RATE * (1 + r_multiple * 0.1)
else:
    new_weight = old_weight - LEARNING_RATE * (1 + abs(r_multiple) * 0.1)
```

### 4.2 Pattern Memory

**File:** `wsb_snake/learning/pattern_memory.py`
**Purpose:** Store and recognize successful price patterns

**Pattern Types:**
| Type | Description | Trigger |
|------|-------------|---------|
| breakout | Sudden large move with volume | 2x avg move + 1.5x volume |
| squeeze | Compression then expansion | Late range > early range * 1.5 |
| reversal | Direction change | Early/late direction opposite |
| momentum | Sustained directional move | Default |

**Matching Algorithm:**
1. Calculate current price changes (5 bars)
2. Calculate current volume ratios
3. Compare to stored patterns (same symbol)
4. Compute similarity score (0-100)
5. If similarity > 70% and samples > 3, consider match

**Boost Logic:**
- Pattern match with 60%+ win rate: +15 to signal score
- Pattern match with 70%+ win rate: +25 to signal score

### 4.3 Time Learning

**File:** `wsb_snake/learning/time_learning.py`
**Purpose:** Optimize trading by time of day

**Sessions Tracked:**
| Session | Hours (ET) | Default Quality |
|---------|------------|-----------------|
| premarket | 4:00-9:30 | 30 |
| open | 9:30-10:30 | 60 |
| morning | 10:30-12:00 | 55 |
| midday | 12:00-14:00 | 40 |
| afternoon | 14:00-15:00 | 50 |
| power_hour | 15:00-16:00 | 75 |
| after_hours | 16:00-20:00 | 20 |

**Recommendation Output:**
```python
{
    "current_hour": 15,
    "session": "power_hour",
    "quality_score": 75,
    "historical_win_rate": 0.62,
    "recommendation": "aggressive",
    "best_strategies": ["0DTE_MOMENTUM", "SQUEEZE"]
}
```

### 4.4 Trade Outcome Recorder (NEW)

**File:** `wsb_snake/trading/outcome_recorder.py`
**Purpose:** Central orchestrator for all learning systems

**Records To:**
1. **Database outcomes table** - Full trade data with MFE/MAE
2. **Database trade_performance table** - Granular analytics
3. **Learning Memory** - Weight updates
4. **Pattern Memory** - Pattern storage (if bars provided)
5. **Time Learning** - Time-of-day statistics

**Data Captured:**
```python
{
    "signal_id": int,           # Links to signals table
    "symbol": str,              # Ticker
    "trade_type": str,          # CALLS/PUTS
    "entry_price": float,
    "exit_price": float,
    "pnl": float,               # Dollar P&L
    "pnl_pct": float,           # Percentage P&L
    "exit_reason": str,         # TARGET/STOP/TIME_DECAY
    "entry_time": datetime,
    "exit_time": datetime,
    "engine": str,              # scalper/momentum/macro
    "session": str,             # power_hour/morning/etc
    "holding_time_seconds": int
}
```

### 4.5 State Machine (Predator Pattern)

**File:** `wsb_snake/engines/state_machine.py`
**Purpose:** Prevent premature alerts, ensure conviction

**States:**
```
LURK ──▶ COILED ──▶ RATTLE ──▶ STRIKE ──▶ CONSTRICT ──▶ VENOM
  │         │          │          │           │            │
  │         │          │          │           │            │
  ▼         ▼          ▼          ▼           ▼            ▼
Passive   Setup     Signal    Execute    Manage      Postmortem
monitor  building  detected   trade    position    end-of-day
```

**Transition Triggers:**
| From | To | Trigger |
|------|-----|---------|
| LURK | COILED | Compression + time alignment + options concentration |
| COILED | RATTLE | Ignition detected + headline catalyst + volume surge |
| RATTLE | STRIKE | Structure break + direction confirmed + probability high |
| STRIKE | CONSTRICT | Trade executed |
| CONSTRICT | VENOM | Position closed |
| VENOM | LURK | End of day reset |

**Gate Function:**
```python
def should_strike() -> bool:
    return current_state == "STRIKE" and signal_tier in ["A+", "A"]
```

### 4.6 Zero Greed Exit Protocol

**File:** `wsb_snake/learning/zero_greed_exit.py`
**Purpose:** Mechanical exit enforcement (no override)

**Rules:**
1. **Target Hit** = IMMEDIATE EXIT (book profit)
2. **Stop Hit** = IMMEDIATE EXIT (accept loss)
3. **Time Decay** = EXIT at deadline (theta kills 0DTE)

**No Exceptions:**
- No "letting winners run" on 0DTE
- No "hoping for recovery" on stops
- No holding past max hold time

---

## 5. API Integrations

### 5.1 Polygon.io (Primary Market Data)

**Current Plan:** Basic + Options Starter

**Endpoints Used:**
| Endpoint | Purpose | Rate Limit |
|----------|---------|------------|
| `/v2/aggs/ticker/{ticker}/range` | Stock bars (1m, 5m, 15m) | 5/min |
| `/v1/indicators/rsi` | RSI calculation | 5/min |
| `/v1/indicators/sma` | SMA calculation | 5/min |
| `/v1/indicators/ema` | EMA calculation | 5/min |
| `/v1/indicators/macd` | MACD calculation | 5/min |
| `/v2/snapshot/locale/us/markets/stocks/gainers` | Market regime | 5/min |
| `/v3/snapshot/options/{ticker}` | Options chains | 5/min |
| `/v3/quotes/{ticker}` | Real-time quotes | 5/min |
| `/v3/trades/{ticker}` | Recent trades | 5/min |

**Adapter:** `wsb_snake/collectors/polygon_enhanced.py`

### 5.2 Alpaca (Trading + News)

**Current Plan:** Paper Trading (Free)

**Endpoints Used:**
| Endpoint | Purpose |
|----------|---------|
| `POST /v2/orders` | Place option orders |
| `DELETE /v2/positions/{symbol}` | Close positions |
| `GET /v2/account` | Account info, buying power |
| `GET /v2/positions` | Current positions |
| `GET /v2/orders/{order_id}` | Order status |

**Trading Limits:**
- Max per trade: $1,500
- Max daily exposure: $6,000
- Max concurrent positions: 5

**Adapter:** `wsb_snake/trading/alpaca_executor.py`

### 5.3 Finnhub (News + Sentiment)

**Current Plan:** Free Tier

**Endpoints Used:**
| Endpoint | Purpose | Rate Limit |
|----------|---------|------------|
| `/company-news` | Company news | 60/min |
| `/news-sentiment` | Social sentiment | 60/min |
| `/stock/insider-sentiment` | Insider MSPR | 60/min |
| `/calendar/earnings` | Earnings dates | 60/min |

**Adapter:** `wsb_snake/collectors/finnhub_collector.py`

### 5.4 Google Gemini (AI Analysis)

**Current Plan:** Free Tier

**Model:** `gemini-1.5-flash`

**Rate Limits:**
- 10 requests per minute (RPM)
- 100 requests per day (RPD)

**Usage:**
- Pattern confirmation
- Scalp signal validation
- Chart analysis (secondary)

**Adapter:** `wsb_snake/analysis/scalp_langgraph.py`

### 5.5 OpenAI (AI Vision)

**Current Plan:** Pay-as-you-go

**Model:** `gpt-4o`

**Usage:**
- Chart analysis with vision
- Pattern recognition
- Market regime classification

**Rate Limiting:** Strategic use only (cost optimization)

**Adapter:** `wsb_snake/analysis/langgraph_analyzer.py`

### 5.6 Telegram (Notifications)

**API:** Bot API (Free)

**Endpoints:**
| Endpoint | Purpose |
|----------|---------|
| `/sendMessage` | Send alerts |
| `/sendPhoto` | Send charts |

**Message Types:**
- Signal alerts (A+/A tier)
- Trade execution (buy/sell)
- Position updates (fill, exit)
- Daily summaries

**Adapter:** `wsb_snake/notifications/telegram_bot.py`

### 5.7 Reddit (r/wallstreetbets)

**Method:** Public JSON endpoints (no API key)

**Endpoints:**
| Endpoint | Purpose |
|----------|---------|
| `/r/wallstreetbets/hot.json` | Hot posts |
| `/r/wallstreetbets/new.json` | New posts |
| `/r/wallstreetbets/comments/{id}.json` | Comments |

**Data Extracted:**
- Trending tickers (mention counts)
- Sentiment (bullish/bearish keywords)
- Catalyst detection

**Adapter:** `wsb_snake/collectors/reddit_collector.py`

### 5.8 Additional Data Sources

| Source | Purpose | Adapter |
|--------|---------|---------|
| Benzinga | Premium news | `benzinga_news.py` |
| SEC EDGAR | Insider filings | `sec_edgar_collector.py` |
| Finviz | Unusual volume | `finviz_collector.py` |
| FINRA | Dark pool data | `finra_darkpool.py` |
| Congressional Trading | Insider trades | `congressional_trading.py` |
| FRED | Macro regime | `fred_collector.py` |

---

## 6. Database Schema

### 6.1 signals

Primary table for all detected signals.

```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    setup_type TEXT NOT NULL,           -- BREAKOUT, REVERSAL, etc.
    score REAL NOT NULL,                -- 0-100
    tier TEXT NOT NULL,                 -- A+, A, B, C

    -- Market features
    price REAL,
    volume INTEGER,
    change_pct REAL,
    vwap REAL,
    range_pct REAL,

    -- Options features
    atm_iv REAL,
    call_put_ratio REAL,
    top_strike REAL,
    options_pressure_score REAL,

    -- Social features
    social_velocity REAL,
    sentiment_score REAL,

    -- Session context
    session_type TEXT,
    minutes_to_close REAL,

    -- Full feature blob (JSON)
    features_json TEXT,
    evidence_json TEXT,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 6.2 outcomes

Trade outcome tracking for learning.

```sql
CREATE TABLE outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL,

    -- Price outcomes
    entry_price REAL,
    max_price REAL,          -- MFE (max favorable excursion)
    min_price REAL,          -- MAE (max adverse excursion)
    exit_price REAL,

    -- Timing
    time_to_target_1 INTEGER,
    time_to_target_2 INTEGER,
    time_to_stop INTEGER,

    -- Result
    hit_target_1 INTEGER DEFAULT 0,
    hit_target_2 INTEGER DEFAULT 0,
    hit_stop INTEGER DEFAULT 0,
    r_multiple REAL,
    outcome_type TEXT,       -- "win", "loss", "scratch", "timeout"
    notes TEXT,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);
```

### 6.3 trade_performance (NEW)

Granular analytics table.

```sql
CREATE TABLE trade_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    engine TEXT,             -- scalper, momentum, macro
    trade_type TEXT,         -- CALLS, PUTS
    entry_hour INTEGER,
    session TEXT,            -- power_hour, morning, etc.
    pnl REAL,
    pnl_pct REAL,
    exit_reason TEXT,
    holding_time_seconds INTEGER,
    signal_id INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

### 6.4 model_weights

Adaptive learning weights.

```sql
CREATE TABLE model_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    feature_name TEXT NOT NULL UNIQUE,
    weight REAL DEFAULT 1.0,
    hit_count INTEGER DEFAULT 0,
    miss_count INTEGER DEFAULT 0,
    last_updated TEXT,

    -- Context-specific weights
    weight_power_hour REAL DEFAULT 1.0,
    weight_trend_regime REAL DEFAULT 1.0,
    weight_range_regime REAL DEFAULT 1.0
);
```

### 6.5 price_patterns

Pattern memory storage.

```sql
CREATE TABLE price_patterns (
    pattern_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    pattern_type TEXT NOT NULL,      -- breakout, reversal, squeeze, momentum
    direction TEXT NOT NULL,         -- bullish, bearish
    bars_before INTEGER DEFAULT 5,
    price_changes TEXT,              -- JSON array
    volume_ratios TEXT,              -- JSON array
    rsi_min REAL DEFAULT 0,
    rsi_max REAL DEFAULT 100,
    vwap_position TEXT DEFAULT 'neutral',
    win_count INTEGER DEFAULT 0,
    loss_count INTEGER DEFAULT 0,
    avg_gain REAL DEFAULT 0,
    avg_loss REAL DEFAULT 0,
    best_hour INTEGER DEFAULT 10,
    created_at TEXT,
    last_matched TEXT
);
```

### 6.6 time_performance

Time-of-day learning.

```sql
CREATE TABLE time_performance (
    slot_key TEXT PRIMARY KEY,
    hour INTEGER NOT NULL,
    day_of_week INTEGER,
    session TEXT NOT NULL,
    total_signals INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    total_gain REAL DEFAULT 0,
    total_loss REAL DEFAULT 0,
    avg_score REAL DEFAULT 0,
    best_strategy TEXT DEFAULT '',
    updated_at TEXT
);
```

### 6.7 daily_summaries

Daily performance tracking.

```sql
CREATE TABLE daily_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    total_signals INTEGER DEFAULT 0,
    alerts_sent INTEGER DEFAULT 0,
    paper_trades INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    scratches INTEGER DEFAULT 0,
    win_rate REAL,
    avg_r_multiple REAL,
    total_r REAL,
    best_ticker TEXT,
    best_r REAL,
    worst_ticker TEXT,
    worst_r REAL,
    regime_notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## 7. Risk Management

### 7.1 Risk Governor

**File:** `wsb_snake/trading/risk_governor.py`

**Global Limits:**
| Parameter | Value |
|-----------|-------|
| Max Concurrent Positions | 5 |
| Max Daily Exposure | $6,000 |
| Max Daily Loss (Kill Switch) | -$500 |
| Max Per Ticker | $2,000 |
| Max Per Sector | $4,000 |

**Per-Engine Limits:**
| Engine | Max Positions | Max Per Trade |
|--------|---------------|---------------|
| SCALPER | 4 | $1,500 |
| MOMENTUM | 2 | $1,200 |
| MACRO | 2 | $2,000 |
| VOL_SELL | 2 | $1,500 |

**Position Sizing Formula:**
```python
base_size = max_per_trade / (option_price * 100)
confidence_adj = base_size * (0.5 + confidence_pct / 200)
volatility_adj = confidence_adj / volatility_factor
final_size = min(confidence_adj, available_capital / (option_price * 100))
```

### 7.2 Kill Switches

| Trigger | Action |
|---------|--------|
| Daily P&L < -$500 | Stop all trading for day |
| Single loss > 3% of account | Review and pause |
| 3 consecutive losses | Reduce position size 50% |
| API errors > 5 in 10 min | Pause data collection |

### 7.3 0DTE Mandatory Close

**Time:** 3:55 PM ET (5 minutes before market close)

**Logic:**
1. Scan all open positions
2. Close any 0DTE options immediately
3. Send Telegram alert
4. No overnight 0DTE risk

---

## 8. Configuration Reference

### 8.1 Environment Variables

**Required API Keys:**
```bash
# Trading
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_LIVE_TRADING=false  # Set to 'true' for live trading

# Market Data
POLYGON_API_KEY=your_key

# News & Sentiment
FINNHUB_API_KEY=your_key
BENZINGA_API_KEY=your_key

# AI Models
GEMINI_API_KEY=your_key
OPENAI_API_KEY=your_key
DEEPSEEK_API_KEY=your_key

# Notifications
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# Social
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=your_agent
```

**Trading Configuration:**
```bash
# Risk
RISK_MAX_DAILY_LOSS=-500
RISK_MAX_CONCURRENT_POSITIONS=5
RISK_MAX_DAILY_EXPOSURE=6000

# Scalp Settings
SCALP_TARGET_PCT=1.12      # +12% target
SCALP_STOP_PCT=0.92        # -8% stop
SCALP_MAX_HOLD_MINUTES=12  # Max hold time
```

### 8.2 Trading Universe

**0DTE Universe (Daily Options):**
```python
DAILY_0DTE_TICKERS = ['SPY', 'QQQ', 'IWM']
```

**Full Universe:**
```python
ZERO_DTE_UNIVERSE = [
    'SPY', 'QQQ', 'IWM',           # Index ETFs
    'SLV', 'GLD', 'GDX', 'GDXJ',   # Precious metals
    'USO', 'XLE',                   # Energy
    'TSLA', 'NVDA', 'AAPL', 'META', # Mega-cap tech
    'AMD', 'AMZN', 'GOOGL', 'MSFT',
    'THH', 'RKLB', 'ASTS', 'NBIS', # Small-cap momentum
    'PL', 'LUNR', 'ONDS', 'SLS',
    'POET', 'ENPH', 'USAR', 'PYPL'
]
```

**Momentum Universe:**
```python
MOMENTUM_UNIVERSE = [
    'RKLB', 'ASTS', 'LUNR', 'PL', 'ONDS',
    'POET', 'SLS', 'NBIS', 'ENPH', 'USAR',
    'THH', 'CLSK', 'MU', 'INTC'
]
```

**LEAPS Universe:**
```python
LEAPS_UNIVERSE = [
    'SLV', 'GLD', 'GDX', 'GDXJ',   # Commodities
    'USO', 'XLE',                   # Energy
    'SPY', 'QQQ', 'IWM',           # Indices
    'META', 'AAPL', 'NVDA', 'TSLA',
    'MSFT', 'AMZN', 'GOOGL', 'PYPL'
]
```

### 8.3 Session Windows

| Session | Start (ET) | End (ET) |
|---------|------------|----------|
| premarket | 4:00 AM | 9:30 AM |
| open | 9:30 AM | 10:30 AM |
| morning | 10:30 AM | 12:00 PM |
| lunch | 12:00 PM | 1:00 PM |
| power_hour_early | 1:00 PM | 3:00 PM |
| power_hour | 3:00 PM | 4:00 PM |
| afterhours | 4:00 PM | 8:00 PM |

---

## 9. Data Flow Diagrams

### 9.1 Signal Generation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA COLLECTION LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Polygon ──┐                                                     │
│  Alpaca ───┼─────▶ Market Data (price, volume, options)         │
│  Finnhub ──┤                                                     │
│  Benzinga ─┘                                                     │
│                                                                  │
│  Reddit ───┐                                                     │
│  SEC ──────┼─────▶ Alternative Data (sentiment, filings)        │
│  FINRA ────┘                                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ENGINE DETECTION LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │  Ignition   │   │  Pressure   │   │   Surge     │           │
│  │  Detector   │   │   Engine    │   │   Hunter    │           │
│  │  (Engine 1) │   │  (Engine 2) │   │  (Engine 3) │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                 │                 │                    │
│         └────────────┬────┴────────────────┘                    │
│                      ▼                                           │
│         ┌────────────────────────┐                              │
│         │  Probability Generator │                              │
│         │     (Signal Fusion)    │                              │
│         └───────────┬────────────┘                              │
│                     │                                            │
└─────────────────────┼────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     VALIDATION LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Chop    │  │  State   │  │  Family  │  │ Inception│        │
│  │  Filter  │  │ Machine  │  │Classifier│  │ Detector │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │                │
│       └──────┬──────┴──────┬──────┴──────┬──────┘               │
│              ▼             ▼             ▼                       │
│  ┌──────────────────────────────────────────────────┐           │
│  │            Chart Brain (AI Validation)           │           │
│  └──────────────────────────────────────────────────┘           │
│              │             │             │                       │
│              ▼             ▼             ▼                       │
│  ┌──────────────────────────────────────────────────┐           │
│  │           Pattern Memory Matching                │           │
│  └──────────────────────────────────────────────────┘           │
│                          │                                       │
└──────────────────────────┼───────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EXECUTION LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │  Tier A+/A?     │───▶│ Alpaca Executor │                     │
│  │  (Alert Gate)   │    │ (Real Trading)  │                     │
│  └────────┬────────┘    └────────┬────────┘                     │
│           │                      │                               │
│           ▼                      ▼                               │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ Telegram Alert  │    │ Position Track  │                     │
│  └─────────────────┘    └─────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Learning Feedback Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRADE LIFECYCLE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Signal Generated ──▶ Trade Executed ──▶ Position Monitored     │
│         │                    │                   │               │
│         ▼                    ▼                   ▼               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │ signal_id   │     │ entry_price │     │ current_pnl │        │
│  │ stored      │     │ stored      │     │ tracked     │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│                                                 │                │
│                                                 ▼                │
│                              ┌──────────────────────────┐       │
│                              │   Exit Triggered         │       │
│                              │   (target/stop/time)     │       │
│                              └───────────┬──────────────┘       │
│                                          │                       │
└──────────────────────────────────────────┼───────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     OUTCOME RECORDING                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              TradeOutcomeRecorder                         │  │
│  │                                                            │  │
│  │   record_trade_outcome()                                  │  │
│  │       │                                                    │  │
│  │       ├──▶ Database (outcomes + trade_performance)        │  │
│  │       │                                                    │  │
│  │       ├──▶ Learning Memory (weight updates)               │  │
│  │       │                                                    │  │
│  │       ├──▶ Pattern Memory (pattern storage)               │  │
│  │       │                                                    │  │
│  │       └──▶ Time Learning (time-of-day stats)              │  │
│  │                                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     WEIGHT ADAPTATION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Next Pipeline Run Uses:                                        │
│                                                                  │
│  ├── Updated feature weights (model_weights table)              │
│  ├── Pattern boost/penalty (pattern_memory)                     │
│  ├── Time quality score (time_performance)                      │
│  └── Session learnings (encoded wisdom)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Expected Outcomes & Performance

### 10.1 Target Metrics by Engine

| Engine | Win Rate | Avg Win | Avg Loss | Expected Monthly |
|--------|----------|---------|----------|------------------|
| SPY Scalper | 55-60% | +15% | -8% | +$500-1,500 |
| Momentum | 50-55% | +25% | -12% | +$300-800 |
| LEAPS/Macro | 45-50% | +50% | -20% | +$200-500 |

### 10.2 Signal Quality by Tier

| Tier | Expected Win Rate | Avg R:R | Action |
|------|-------------------|---------|--------|
| A+ | 65-75% | 2.0+ | Immediate execution |
| A | 55-65% | 1.5+ | Execute at STRIKE |
| B | 45-55% | 1.0-1.5 | Alert only |
| C | < 45% | < 1.0 | Log only |

### 10.3 Session Performance

| Session | Quality | Best Strategies |
|---------|---------|-----------------|
| Open (9:30-10:30) | High | Gap plays, momentum |
| Morning (10:30-12:00) | Medium | Trend continuation |
| Midday (12:00-14:00) | Low | Avoid (chop) |
| Power Hour (15:00-16:00) | Highest | 0DTE scalps, surges |

### 10.4 Learning System Impact

**After 100 Trades (Expected):**
- Feature weights calibrated to market
- Pattern memory: 20-30 stored patterns
- Time learning: Best hours identified
- Estimated win rate improvement: +5-10%

**After 500 Trades (Expected):**
- Highly tuned weights
- Pattern memory: 100+ patterns
- Time learning: Full hour-by-hour profile
- Estimated win rate improvement: +10-15%

---

## 11. Recent Additions (February 2026)

### 11.1 Learning System Integration (This Update)

**Problem Solved:**
The trading bot had 199 signals captured but 0 records in the outcomes table. Trade exits were not connected to learning systems.

**Changes Made:**

1. **Added `signal_id` to `AlpacaPosition`** (`alpaca_executor.py:68`)
   - Links trades to their originating signals
   - Enables outcome tracking back to signal features

2. **Added `signal_id` parameter to `execute_scalp_entry()`** (`alpaca_executor.py:608`)
   - Callers can now pass signal ID when entering trades
   - Signal ID stored in position for exit tracking

3. **Created `TradeOutcomeRecorder`** (`outcome_recorder.py`)
   - Central orchestrator for all learning systems
   - Records to: database, learning_memory, pattern_memory, time_learning

4. **Added `trade_performance` table** (`database.py`)
   - Granular analytics: engine, session, entry_hour, holding_time
   - Enables queries like "What's my win rate on SPY PUTS in power hour?"

5. **Added `save_outcome()` function** (`database.py`)
   - Persists outcomes to database
   - Updates daily_summaries automatically

6. **Hooked recorder into `execute_exit()`** (`alpaca_executor.py:960`)
   - Every trade exit now records outcome
   - Learning systems update automatically

**Data Flow After This Update:**

```
Trade Entry ──▶ signal_id stored in position
     │
Trade Exit ──▶ outcome_recorder.record_trade_outcome()
     │
     ├──▶ outcomes table (database)
     ├──▶ trade_performance table (database)
     ├──▶ daily_summaries table (database)
     ├──▶ learning_memory (weight updates)
     ├──▶ pattern_memory (pattern storage)
     └──▶ time_learning (time-of-day stats)
```

### 11.2 Previous Updates (January 2026)

- **OpenAI Rate Limiting** - Scalping-focused credit conservation
- **Gemini Integration** - Strategic rate limiting with confluence triggers
- **AI API Optimization** - Only call when patterns detected
- **GreekOptimizer & PositionSizer** - Added to precious metals scalper

---

## 12. Planned Future Additions

### 12.1 Short-Term (Next 30 Days)

| Feature | Description | Priority |
|---------|-------------|----------|
| MFE/MAE Tracking | Track max favorable/adverse excursion during trade | High |
| Pattern Learning Enhancement | Pass recent bars to outcome recorder | High |
| Signal Score to Position | Store original signal score in position | Medium |
| Partial Fill Handling | Handle partial option fills | Medium |

### 12.2 Medium-Term (60-90 Days)

| Feature | Description | Priority |
|---------|-------------|----------|
| Live Trading Mode | Enable actual live trading with safeguards | High |
| Advanced Position Sizing | Kelly criterion-based sizing | Medium |
| Multi-Account Support | Trade across multiple Alpaca accounts | Medium |
| Backtesting Framework | Replay historical data for strategy testing | Medium |
| Greeks-Based Exit | Dynamic exit based on delta/theta decay | Low |

### 12.3 Long-Term (6+ Months)

| Feature | Description | Priority |
|---------|-------------|----------|
| Machine Learning Integration | Train ML models on outcomes data | Medium |
| Sentiment Analysis V2 | Advanced NLP on news/social | Medium |
| Sector Rotation Strategy | Automated sector momentum | Low |
| International Markets | Extend to European/Asian options | Low |
| Mobile App | Real-time alerts via dedicated app | Low |

---

## 13. API Tier Comparison & Upgrade Benefits

### 13.1 Polygon.io

**Current:** Basic + Options Starter (~$29/month)

| Tier | Price | Rate Limit | Benefits |
|------|-------|------------|----------|
| **Basic** | $29/mo | 5/min | Current usage |
| **Starter** | $79/mo | 100/min | 20x faster data refresh |
| **Developer** | $199/mo | 1000/min | Real-time streaming, tick data |
| **Advanced** | $499/mo | Unlimited | Full market depth, historical tick |

**Upgrade Impact:**
- **Starter ($79):** Refresh every 3 seconds instead of 12 seconds
- **Developer ($199):** True real-time streaming, sub-second reaction
- **Advanced ($499):** Full order book depth, institutional-grade data

**Recommendation:** Upgrade to **Starter** when trading volume justifies (10+ trades/day)

### 13.2 Alpaca

**Current:** Paper Trading (Free)

| Tier | Price | Features |
|------|-------|----------|
| **Free** | $0 | Paper trading, basic data |
| **Algo Trader Plus** | $99/mo | 200 requests/min, extended hours |
| **Algo Trader Pro** | $299/mo | 10,000 req/min, priority support |

**Upgrade Impact:**
- **Algo Trader Plus:** More API calls for aggressive scalping
- **Algo Trader Pro:** Institutional-grade execution

**Recommendation:** Stay on **Free** until live trading proven profitable

### 13.3 Finnhub

**Current:** Free Tier

| Tier | Price | Rate Limit | Features |
|------|-------|------------|----------|
| **Free** | $0 | 60/min | Basic data |
| **Premium** | $50/mo | 300/min | Full filings, transcripts |
| **Professional** | $200/mo | 600/min | Real-time filings, bulk data |

**Upgrade Impact:**
- **Premium:** 5x more requests, full SEC filings
- **Professional:** Real-time insider trading alerts

**Recommendation:** Consider **Premium** for better SEC EDGAR integration

### 13.4 Google Gemini

**Current:** Free Tier

| Tier | Price | Rate Limit | Features |
|------|-------|------------|----------|
| **Free** | $0 | 10 RPM, 100 RPD | Basic models |
| **Pay-as-you-go** | $0.001/1k tokens | 100 RPM | All models, higher limits |

**Upgrade Impact:**
- **Pay-as-you-go:** 10x RPM, unlimited daily requests
- Estimated cost: $5-20/month for current usage

**Recommendation:** Switch to **Pay-as-you-go** for more AI calls

### 13.5 OpenAI

**Current:** Pay-as-you-go

| Model | Price | Use Case |
|-------|-------|----------|
| **GPT-4o** | $5/1M tokens | Chart vision analysis |
| **GPT-4o-mini** | $0.15/1M tokens | Fast pattern confirmation |
| **GPT-4 Turbo** | $10/1M tokens | Complex analysis |

**Optimization:**
- Use **GPT-4o-mini** for quick confirmations
- Reserve **GPT-4o** for chart vision only
- Estimated savings: 50-70%

### 13.6 Cost Optimization Summary

**Current Monthly Cost:** ~$30-50
**Recommended Upgrade Path:**

| Month | Changes | New Cost | Benefit |
|-------|---------|----------|---------|
| 1 | Gemini pay-as-you-go | ~$50-70 | 10x AI calls |
| 2 | Polygon Starter | ~$130 | 20x data refresh |
| 3 | Finnhub Premium | ~$180 | Better filings data |
| 4+ | Evaluate based on P&L | - | Scale with profits |

---

## 14. Troubleshooting & Monitoring

### 14.1 Health Checks

**Database:**
```sql
-- Check signal count
SELECT COUNT(*) FROM signals;

-- Check outcomes being recorded
SELECT COUNT(*) FROM outcomes;
SELECT outcome_type, COUNT(*) FROM outcomes GROUP BY outcome_type;

-- Check learning weights
SELECT feature_name, weight, hit_count, miss_count FROM model_weights;

-- Check time performance
SELECT hour, session, wins, losses FROM time_performance;

-- Check trade performance by engine
SELECT engine, COUNT(*), AVG(pnl), AVG(pnl_pct)
FROM trade_performance
GROUP BY engine;
```

**Logs:**
```bash
# Check for outcome recording
grep "Recorded outcome" logs/wsb_snake.log

# Check for learning updates
grep "Weight update" logs/wsb_snake.log

# Check for errors
grep "ERROR" logs/wsb_snake.log | tail -50
```

### 14.2 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No outcomes recorded | Old code without recorder | Deploy latest version |
| Weights not updating | No signal_id passed | Pass signal_id to execute_scalp_entry |
| API rate limited | Too many requests | Check rate limit settings |
| Positions not closing | Exit logic bug | Check Zero Greed Exit logs |
| Database locked | Concurrent writes | Use connection pooling |

### 14.3 Performance Monitoring

**Key Metrics to Track:**
1. Win rate by tier (A+, A, B, C)
2. Average P&L by session
3. Average P&L by engine
4. Holding time distribution
5. Exit reason breakdown

**Daily Review Checklist:**
- [ ] Check daily P&L vs target
- [ ] Review any large losses
- [ ] Verify learning weights trending correctly
- [ ] Check for API errors or rate limits
- [ ] Validate pattern memory growing

---

## Appendix A: Quick Reference

### Signal Tiers
- **A+:** Score 85+, immediate execution
- **A:** Score 70-84, execute at STRIKE state
- **B:** Score 55-69, alert only
- **C:** Score < 55, log only

### Exit Triggers
- **TARGET:** +12% (scalper default)
- **STOP:** -8% (scalper default)
- **TIME_DECAY:** 12 minutes (scalper default)
- **EOD:** 3:55 PM ET (mandatory 0DTE close)

### State Machine
- **LURK:** Passive monitoring
- **COILED:** Setup building
- **RATTLE:** Signal detected
- **STRIKE:** Execute trade
- **CONSTRICT:** Manage position
- **VENOM:** End-of-day postmortem

---

*Document generated: February 2026*
*Maintainer: Intellibot Development Team*
