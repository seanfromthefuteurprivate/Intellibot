# WSB Snake - Complete Technical Documentation

## Version 2.5 | Aggressive Mode | January 2026

---

# Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Data Flow & Signal Pipeline](#data-flow--signal-pipeline)
5. [Trading Execution](#trading-execution)
6. [AI & Analysis Stack](#ai--analysis-stack)
7. [Learning System](#learning-system)
8. [Configuration & Secrets](#configuration--secrets)
9. [Deployment Guide](#deployment-guide)
10. [Debugging & Troubleshooting](#debugging--troubleshooting)
11. [API Reference](#api-reference)
12. [File Structure](#file-structure)
13. [Daily Operations](#daily-operations)
14. [Performance Metrics](#performance-metrics)
15. [Known Issues & Limitations](#known-issues--limitations)

---

# Executive Summary

## What is WSB Snake?

WSB Snake is a **production-grade autonomous 0DTE options scalping engine** that executes paper trades on Alpaca with institutional-level reliability. The system monitors a 29-ticker universe (SPY, QQQ, IWM, metals, energy, tech stocks) and uses multi-model AI confirmation (Gemini + DeepSeek + GPT) for high-confidence setups.

## Key Capabilities

| Feature | Description |
|---------|-------------|
| **Autonomous Trading** | Opens and closes positions automatically on Alpaca |
| **Multi-AI Confirmation** | Gemini 2.0 Flash + DeepSeek + GPT-4o vision analysis |
| **Real-time Alerts** | Telegram notifications for every trade signal and execution |
| **Zero Greed Protocol** | Mechanical exits at target/stop - no human override |
| **Learning System** | Adapts based on trade outcomes and patterns |
| **EOD Auto-Close** | All positions close at 3:55 PM ET - no overnight risk |

## Current Configuration (Aggressive Mode)

```
MAX_DAILY_EXPOSURE = $6,000 (cash + margin)
MAX_PER_TRADE = $1,500
MAX_CONCURRENT_POSITIONS = 5
MIN_CONFIDENCE = 70% (ETFs) / 75% (Small Caps)
TARGET = +20% option gain
STOP = -15% option loss
EOD_CLOSE = 3:55 PM ET
```

## Expected Performance

| Scenario | Daily P&L | Monthly (20 days) |
|----------|-----------|-------------------|
| Conservative (5%) | $300 | $6,000 |
| Moderate (10%) | $600 | $12,000 |
| Aggressive (15%) | $900 | $18,000 |

---

# System Architecture

## High-Level Overview

```
                    ┌─────────────────────────────────────────┐
                    │           EXTERNAL DATA SOURCES          │
                    ├─────────────────────────────────────────┤
                    │ Polygon.io │ Finnhub │ Benzinga │ Alpaca │
                    └─────────────┬───────────────────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      DATA COLLECTORS        │
                    │  (Real-time + Historical)   │
                    └─────────────┬───────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SPY SCALPER   │    │   CHART BRAIN   │    │ PREDATOR STACK  │
│  Pattern Detect │    │  AI Vision      │    │ Multi-Model AI  │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │    SIGNAL FUSION      │
                    │  (Probability Score)  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   ALPACA EXECUTOR     │
                    │  (Paper/Live Trading) │
                    └───────────┬───────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ ZERO GREED EXIT │    │    TELEGRAM     │    │ LEARNING SYSTEM │
│  Auto Exits     │    │    Alerts       │    │  Weight Adjust  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Component Interaction Flow

```
1. Market Opens (9:30 AM ET)
   │
   ├── SPY Scalper starts 30-second scan loop
   │   └── Scans 29 tickers for patterns
   │
   ├── For each ticker with pattern:
   │   ├── Collect 5-second bars (Polygon)
   │   ├── Calculate VWAP, RSI, ATR
   │   ├── Detect candlestick patterns
   │   └── Generate initial confidence score
   │
   ├── If confidence >= threshold:
   │   ├── Generate surgical chart
   │   ├── Send to Predator Stack (Gemini → DeepSeek → GPT)
   │   ├── Get AI verdict: STRIKE_CALLS / STRIKE_PUTS / NO_TRADE
   │   └── Final confidence = base + AI boost
   │
   ├── If final confidence >= MIN_CONFIDENCE:
   │   ├── Send Telegram alert
   │   ├── Execute on Alpaca (buy options)
   │   └── Start position monitoring
   │
   └── Zero Greed Exit monitors every 5 seconds:
       ├── Target hit (+20%) → CLOSE immediately
       ├── Stop hit (-15%) → CLOSE immediately
       └── Time decay (45 min) → CLOSE immediately

2. Market Close (3:55 PM ET)
   └── Force close ALL remaining positions

3. End of Day (4:15 PM ET)
   └── Send daily P&L summary to Telegram
```

---

# Core Components

## 1. SPY Scalper (`wsb_snake/engines/spy_scalper.py`)

The primary pattern detection engine that scans for scalping opportunities.

### Configuration

```python
PREDATOR_MODE = True  # Require AI confirmation
SCAN_INTERVAL = 30    # Seconds between scans
MIN_CONFIDENCE_FOR_ALERT = 70  # Minimum confidence to trade
SMALL_CAP_MIN_CONFIDENCE = 75  # Higher bar for small caps
SMALL_CAP_REQUIRES_CANDLESTICK = True  # Strict mode

TICKER_UNIVERSE = [
    # Daily 0DTE ETFs
    'SPY', 'QQQ', 'IWM',
    # Commodity ETFs
    'GLD', 'SLV', 'GDX', 'USO', 'UNG',
    # Sector ETFs
    'XLE', 'XLF', 'TLT', 'HYG',
    # Mega-cap stocks
    'TSLA', 'NVDA', 'AAPL', 'MSFT', 'META', 'AMD', 'GOOGL', 'AMZN',
    # Small caps (strict mode)
    'SLS', 'THH', 'RKLB', 'POET', 'SOFI', 'PLTR', 'MARA', 'RIOT', 'COIN'
]
```

### Pattern Types Detected

| Pattern | Description | Stop Multiplier | Target Multiplier |
|---------|-------------|-----------------|-------------------|
| VWAP_BOUNCE | Price bounces off VWAP support | 1.0x ATR | 2.5x ATR |
| VWAP_RECLAIM | Price reclaims VWAP from below | 0.8x ATR | 2.5x ATR |
| VWAP_REJECTION | Price rejected at VWAP resistance | 1.0x ATR | 2.5x ATR |
| MOMENTUM_SURGE_LONG | Strong upward momentum | 1.2x ATR | 3.0x ATR |
| MOMENTUM_SURGE_SHORT | Strong downward momentum | 1.2x ATR | 3.0x ATR |
| BREAKOUT | 30-bar high breakout | 0.8x ATR | 3.0x ATR |
| BREAKDOWN | 30-bar low breakdown | 0.8x ATR | 3.0x ATR |
| FAILED_BREAKOUT | Bull trap reversal | 1.0x ATR | 3.5x ATR |
| FAILED_BREAKDOWN | Bear trap reversal | 1.0x ATR | 3.5x ATR |
| SQUEEZE_FIRE | Volatility expansion | 1.2x ATR | 3.5x ATR |

### Candlestick Patterns (Small Cap Strict Mode)

```python
BULLISH_PATTERNS = [
    'hammer',
    'inverted_hammer',
    'bullish_engulfing',
    'piercing_line',
    'bullish_harami',
    'strong_bullish_momentum'
]

BEARISH_PATTERNS = [
    'shooting_star',
    'hanging_man',
    'bearish_engulfing',
    'dark_cloud_cover',
    'bearish_harami',
    'strong_bearish_momentum'
]
```

### Key Methods

```python
def start():
    """Start the 30-second scan loop in background thread."""

def stop():
    """Stop the scanner."""

def scan_universe():
    """Scan all tickers for patterns."""
    # Returns list of ScalpSetup objects

def _detect_pattern(ticker, bars) -> Optional[ScalpSetup]:
    """Detect patterns in price data."""

def _detect_candlestick_pattern(bars) -> Optional[str]:
    """Detect candlestick reversal/continuation patterns."""

def _send_alert(setup: ScalpSetup):
    """Send Telegram alert and execute trade."""
```

---

## 2. Alpaca Executor (`wsb_snake/trading/alpaca_executor.py`)

Handles all trading operations on Alpaca paper/live accounts.

### Configuration

```python
# Trading Mode
LIVE_TRADING = False  # Set via ALPACA_LIVE_TRADING env var

# Risk Limits
MAX_DAILY_EXPOSURE = 6000   # $6,000 total daily exposure
MAX_PER_TRADE = 1500        # $1,500 per position
MAX_CONCURRENT_POSITIONS = 5  # 5 trades max

# Time Limits
MARKET_CLOSE_HOUR = 16      # 4 PM ET
CLOSE_BEFORE_MINUTES = 5    # Close at 3:55 PM ET

# ETF Priority
ETF_TICKERS = ['SPY', 'QQQ', 'IWM', 'GLD', 'GDX', 'SLV', 'XLE', 'XLF', 'TLT', 'USO', 'UNG', 'HYG']
ETF_PRIORITY = True
```

### Key Methods

```python
def execute_scalp_entry(
    underlying: str,
    direction: str,      # 'long' or 'short'
    entry_price: float,
    target_price: float,
    stop_loss: float,
    confidence: float,
    pattern: str
) -> Optional[AlpacaPosition]:
    """
    Execute a scalp trade entry.
    - Validates parameters (stop, target, R:R ratio)
    - Finds optimal strike price
    - Calculates position size within limits
    - Places market order on Alpaca
    - Sends Telegram notifications
    """

def place_option_order(
    underlying: str,
    expiry: datetime,
    strike: float,
    option_type: str,    # 'call' or 'put'
    side: str,           # 'buy' or 'sell'
    qty: int,
    order_type: str = "market"
) -> Optional[Dict]:
    """Place option order on Alpaca."""

def close_position(option_symbol: str) -> Optional[Dict]:
    """Close an existing position."""

def sync_existing_positions() -> int:
    """
    Sync orphaned positions from Alpaca.
    Critical for restart recovery - prevents missed exits.
    """

def start_monitoring():
    """Start the position monitoring thread."""

def _monitor_positions():
    """
    Background thread that monitors every 5 seconds:
    - Check for target/stop hit
    - Check for EOD close time
    - Execute exits as needed
    """
```

### Position Sizing Logic

```python
def calculate_position_size(option_price: float) -> int:
    """
    Calculate position size based on limits:
    1. Max per trade: $1,500
    2. Must leave room for daily exposure cap
    3. Options = 100 shares per contract
    
    Example:
    - Option price: $3.50
    - Cost per contract: $350
    - Max contracts: floor($1,500 / $350) = 4 contracts
    """
```

---

## 3. Zero Greed Exit (`wsb_snake/learning/zero_greed_exit.py`)

Mechanical exit system with no human override allowed.

### Rules

```
1. TARGET HIT (+20%): IMMEDIATE EXIT
   - No "let it run"
   - No "it might go higher"
   - BOOK PROFIT NOW

2. STOP HIT (-15%): IMMEDIATE EXIT
   - No "it might recover"
   - No averaging down
   - ACCEPT THE LOSS

3. TIME DECAY (45 min): EXIT
   - 0DTE options lose value fast
   - Theta decay accelerates after 30 min
   - GET OUT

4. EOD (3:55 PM ET): FORCE CLOSE ALL
   - No overnight risk
   - Market closes at 4 PM
   - CLOSE EVERYTHING
```

### Key Methods

```python
def start():
    """Start the exit monitoring thread."""

def stop():
    """Stop the exit monitor."""

def _monitor_loop():
    """
    Every 5 seconds:
    1. Get all tracked positions
    2. Fetch current option prices
    3. Calculate P&L percentage
    4. If target/stop/time hit → execute exit
    """

def _execute_exit(position: AlpacaPosition, reason: str):
    """
    Execute exit order and send Telegram alert:
    - BOOK PROFIT NOW (target hit)
    - STOP LOSS TRIGGERED (stop hit)
    - TIME DECAY EXIT (max hold exceeded)
    - EOD CLOSE (end of day)
    """
```

---

## 4. Predator Stack (`wsb_snake/analysis/predator_stack.py`)

Multi-model AI confirmation system using Gemini, DeepSeek, and GPT.

### Model Priority

```
1. GEMINI 2.0 FLASH (Primary)
   - Fast, cost-effective
   - Excellent chart vision
   - 95% of analyses

2. DEEPSEEK (Fallback)
   - Budget backup
   - Used when Gemini unavailable
   - Good vision capabilities

3. GPT-4o (Confirmation)
   - Double-check high-confidence setups
   - Premium quality
   - Used sparingly (cost)
```

### Verdicts

```python
class PredatorVerdict(Enum):
    STRIKE_CALLS = "STRIKE_CALLS"   # Go long (buy calls)
    STRIKE_PUTS = "STRIKE_PUTS"     # Go short (buy puts)
    NO_TRADE = "NO_TRADE"           # Pass on this setup
    ABORT = "ABORT"                 # Something wrong, don't trade
```

### Key Methods

```python
def analyze_chart(
    chart_path: str,
    ticker: str,
    pattern: str,
    entry: float,
    target: float,
    stop: float
) -> PredatorAnalysis:
    """
    Send chart to AI models for confirmation.
    
    Returns:
        PredatorAnalysis with:
        - verdict: STRIKE_CALLS, STRIKE_PUTS, NO_TRADE, ABORT
        - confidence: 0-100
        - reasoning: AI explanation
        - model_used: which model gave verdict
    """

def _analyze_with_gemini(chart_base64: str, prompt: str) -> Dict:
    """Call Gemini 2.0 Flash API."""

def _analyze_with_deepseek(chart_base64: str, prompt: str) -> Dict:
    """Call DeepSeek API."""

def _analyze_with_openai(chart_base64: str, prompt: str) -> Dict:
    """Call GPT-4o Vision API."""
```

---

## 5. Telegram Bot (`wsb_snake/notifications/telegram_bot.py`)

Sends real-time alerts for all trading activity.

### Alert Types

```
1. STARTUP ALERT
   - System online notification
   - Configuration summary
   - Account status

2. SIGNAL ALERT
   - Pattern detected
   - Entry/target/stop levels
   - Confidence score
   - AI confirmation

3. BUY ALERT
   - Order placed on Alpaca
   - Position details
   - Cost basis

4. FILL ALERT
   - Order filled
   - Actual fill price
   - Position tracking started

5. EXIT ALERT
   - Position closed
   - P&L (dollar and percent)
   - Exit reason

6. DAILY SUMMARY
   - Total trades
   - Win rate
   - Net P&L
```

### Key Methods

```python
def send_alert(message: str, parse_mode: str = "Markdown"):
    """Send message to Telegram chat."""

def format_scalp_alert(setup: ScalpSetup) -> str:
    """Format a scalp signal alert."""

def format_exit_alert(position: AlpacaPosition, reason: str) -> str:
    """Format an exit alert with P&L."""
```

---

## 6. Chart Generator (`wsb_snake/analysis/chart_generator.py`)

Generates surgical precision charts for AI analysis.

### Chart Types

```
1. VWAP CHART
   - Candlesticks
   - VWAP line
   - VWAP bands (±1σ, ±2σ)
   - Volume bars

2. VOLUME PROFILE CHART
   - Price levels on Y-axis
   - Volume at each level
   - Shows trapped buyers/sellers

3. DELTA CHART
   - Net buying vs selling per candle
   - Green = net buying
   - Red = net selling

4. PREDATOR COMBINED CHART
   - All above in one view
   - Used for AI analysis
```

### Key Methods

```python
def generate_surgical_chart(
    ticker: str,
    bars: List[Dict],
    pattern: str,
    entry: float,
    target: float,
    stop: float
) -> str:
    """
    Generate a predator-style chart for AI analysis.
    Returns path to saved PNG file.
    """
```

---

# Data Flow & Signal Pipeline

## 1. Data Collection

### Polygon.io (Primary Market Data)

```python
# 5-second bars for scalping
GET /v2/aggs/ticker/{ticker}/range/5/second/{from}/{to}

# Options chain
GET /v3/reference/options/contracts?underlying_ticker={ticker}

# Option quotes
GET /v3/quotes/{option_symbol}

# Previous day data
GET /v2/aggs/ticker/{ticker}/prev
```

### Finnhub (News & Sentiment)

```python
# Company news
GET /company-news?symbol={ticker}&from={date}&to={date}

# Earnings calendar
GET /calendar/earnings

# Economic calendar
GET /calendar/economic

# Recommendation trends
GET /stock/recommendation?symbol={ticker}
```

### Alpaca (Trading & Account)

```python
# Account info
GET /v2/account

# Place order
POST /v2/orders

# Get positions
GET /v2/positions

# Close position
DELETE /v2/positions/{symbol}

# Order history
GET /v2/orders?status=closed
```

## 2. Signal Generation Flow

```
Input: 5-second bars for last 30 minutes
                │
                ▼
┌──────────────────────────────────┐
│     PATTERN DETECTION            │
│  - VWAP relationship             │
│  - Price action patterns         │
│  - Volume analysis               │
│  - Momentum indicators           │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│     TECHNICAL SCORING            │
│  - RSI overbought/oversold       │
│  - MACD crossover                │
│  - ATR for volatility            │
│  - Volume confirmation           │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│     SMALL CAP FILTER             │
│  - If small cap: require 75%     │
│  - Must have candlestick pattern │
│  - ETFs/mega caps: 70% threshold │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│     AI CONFIRMATION              │
│  - Generate surgical chart       │
│  - Send to Predator Stack        │
│  - Get STRIKE/NO_TRADE verdict   │
│  - Boost confidence if confirmed │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│     FINAL DECISION               │
│  - Check confidence threshold    │
│  - Check daily exposure limit    │
│  - Check concurrent positions    │
│  - Execute or pass               │
└──────────────────────────────────┘
```

## 3. Confidence Scoring

```python
BASE_CONFIDENCE = 50  # Starting point

# Pattern boosts
VWAP_BOUNCE_BOOST = +10
BREAKOUT_BOOST = +15
SQUEEZE_BOOST = +20

# Technical boosts
RSI_OVERSOLD_BOOST = +5  # RSI < 30
RSI_OVERBOUGHT_BOOST = +5  # RSI > 70
VOLUME_SURGE_BOOST = +10  # Volume > 2x average

# AI boosts
AI_STRIKE_BOOST = +15  # Gemini confirms
AI_HIGH_CONFIDENCE = +5  # AI confidence > 80%

# Penalties
WIDE_SPREAD_PENALTY = -10  # Bid-ask > 20%
LOW_VOLUME_PENALTY = -15  # Volume < 50% average

# Small cap adjustments
SMALL_CAP_CANDLESTICK_BOOST = +5  # Pattern detected
SMALL_CAP_NO_PATTERN = REJECT  # No candlestick = no trade
```

---

# Trading Execution

## Order Flow

```
1. SIGNAL GENERATED
   │
   ├── Validate parameters
   │   ├── Entry > 0
   │   ├── Target > Entry (long) or Target < Entry (short)
   │   ├── Stop < Entry (long) or Stop > Entry (short)
   │   └── R:R ratio >= 0.5
   │
   ├── Check limits
   │   ├── Daily exposure < $6,000
   │   ├── Concurrent positions < 5
   │   └── Per-trade cost <= $1,500
   │
   ├── Find optimal strike
   │   ├── Try ITM, ATM, slight OTM
   │   ├── Get option quotes
   │   └── Select most liquid with good spread
   │
   ├── Calculate position size
   │   └── qty = floor(MAX_PER_TRADE / (option_price * 100))
   │
   └── Place order
       ├── Market order on Alpaca
       ├── Send Telegram: "BUY ORDER SENDING"
       ├── Wait for fill
       └── Send Telegram: "BUY ORDER FILLED"

2. POSITION MONITORING (every 5 seconds)
   │
   ├── Fetch current option price
   ├── Calculate unrealized P&L
   │
   ├── If P&L >= +20%: EXIT (target)
   ├── If P&L <= -15%: EXIT (stop)
   ├── If hold_time >= 45 min: EXIT (time decay)
   └── If time >= 3:55 PM: EXIT (EOD)

3. EXIT EXECUTION
   │
   ├── Close position on Alpaca
   ├── Calculate realized P&L
   ├── Send Telegram: "POSITION CLOSED"
   ├── Update learning system
   └── Release exposure quota
```

## Option Symbol Format

Alpaca uses OCC format:

```
{UNDERLYING}{EXPIRY}{TYPE}{STRIKE}

Example: SPY260127C00695000
- SPY = underlying
- 260127 = January 27, 2026 (YYMMDD)
- C = call (P for put)
- 00695000 = $695.00 strike (8 digits, strike * 1000)
```

## Strike Selection Logic

```python
def get_strike_interval(underlying: str, price: float) -> float:
    """
    Get appropriate strike interval based on underlying and price.
    
    SPY/QQQ/IWM: $1 intervals
    Stocks > $500: $5 intervals
    Stocks $100-$500: $2.50 intervals
    Stocks < $100: $1 intervals
    Stocks < $25: $0.50 intervals
    """
```

---

# AI & Analysis Stack

## Predator Stack Architecture

```
                    ┌─────────────────────┐
                    │   CHART GENERATOR   │
                    │  (Surgical Precision)│
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   PREDATOR STACK    │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  GEMINI 2.0     │   │    DEEPSEEK     │   │    GPT-4o       │
│  (Primary)      │   │   (Fallback)    │   │ (Confirmation)  │
└─────────────────┘   └─────────────────┘   └─────────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │      VERDICT        │
                    │  STRIKE / NO_TRADE  │
                    └─────────────────────┘
```

## AI Prompt Template

```
You are an elite options scalper analyzing a {ticker} chart.

PATTERN DETECTED: {pattern}
PROPOSED ENTRY: ${entry}
TARGET: ${target} ({gain_pct}% gain)
STOP: ${stop} ({loss_pct}% loss)
R:R RATIO: {rr_ratio}

Analyze this chart and provide:
1. VERDICT: STRIKE_CALLS, STRIKE_PUTS, or NO_TRADE
2. CONFIDENCE: 0-100
3. REASONING: 2-3 sentences explaining your decision

Consider:
- VWAP relationship and momentum
- Volume profile and trapped traders
- Probability of hitting target before stop
- Current market regime

BE DECISIVE. Only recommend STRIKE if probability > 60%.
```

## Model Configuration

### Gemini 2.0 Flash

```python
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Request format
{
    "contents": [{
        "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/png", "data": chart_base64}}
        ]
    }]
}
```

### DeepSeek

```python
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

# Request format
{
    "model": "deepseek-chat",
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_base64}"}}
        ]}
    ]
}
```

### GPT-4o (OpenAI)

```python
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Request format
{
    "model": "gpt-4o",
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{chart_base64}"}}
        ]}
    ],
    "max_tokens": 500
}
```

---

# Learning System

## Components

### 1. Pattern Memory (`wsb_snake/learning/pattern_memory.py`)

Stores successful price action patterns and matches against new setups.

```python
class PatternMemory:
    """
    Stores patterns with:
    - price_action: normalized OHLC data
    - volume_profile: normalized volume
    - outcome: win/loss
    - gain_pct: realized P&L
    
    Matching uses:
    - 70% price action similarity
    - 30% volume similarity
    """
    
    def store_pattern(self, bars, outcome, gain_pct):
        """Store a pattern after trade completion."""
    
    def find_similar(self, bars) -> List[PatternMatch]:
        """Find similar historical patterns."""
```

### 2. Time-of-Day Learning (`wsb_snake/learning/time_learning.py`)

Tracks performance by hour and session.

```python
class TimeLearning:
    """
    Tracks:
    - Hourly win rates (9 AM, 10 AM, etc.)
    - Session performance (open, mid-day, power hour)
    - Day-of-week patterns
    
    Provides:
    - Quality scores (0-100) for each time slot
    - Recommendations: OPTIMAL, GOOD, AVOID
    """
    
    TIME_MULTIPLIERS = {
        9: 1.0,   # Market open - volatile
        10: 1.2,  # Good flows
        11: 1.0,  # Settling
        12: 0.8,  # Lunch chop
        13: 0.9,  # Still slow
        14: 1.1,  # Power hour buildup
        15: 1.3,  # Power hour - best volume
    }
```

### 3. Event Outcome Database (`wsb_snake/learning/event_outcomes.py`)

Records actual market moves after CPI, FOMC, earnings.

```python
class EventOutcomeDB:
    """
    Stores:
    - Event type (CPI, FOMC, earnings)
    - Expected move
    - Actual move
    - Direction (up/down)
    
    Generates expectations for future events.
    """
```

### 4. Session Learnings (`wsb_snake/learning/session_learnings.py`)

Encodes battle-tested wisdom from live sessions.

```python
# Example insight
TradingInsight(
    category="execution",
    lesson="STOPS TOO TIGHT: 0.3% ATR = noise triggers. WIDENED to 0.8-1.2% ATR.",
    weight=1.0,
    date_learned="2026-01-27",
    trade_context={
        "old_stop_pct": 0.003,
        "new_stop_pct": 0.008,
        "key_insight": "stops_need_room_to_breathe"
    }
)
```

---

# Configuration & Secrets

## Required Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ALPACA_API_KEY` | Alpaca API key | YES |
| `ALPACA_SECRET_KEY` | Alpaca secret key | YES |
| `POLYGON_API_KEY` | Polygon.io API key | YES |
| `FINNHUB_API_KEY` | Finnhub API key | YES |
| `BENZINGA_API_KEY` | Benzinga API key | YES |
| `GEMINI_API_KEY` | Google Gemini API key | YES |
| `DEEPSEEK_API_KEY` | DeepSeek API key | YES |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | YES |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | YES |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `ALPACA_LIVE_TRADING` | Set "true" for live | Optional |

## Configuration File (`wsb_snake/config.py`)

```python
# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# API Keys
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
BENZINGA_API_KEY = os.environ.get("BENZINGA_API_KEY")

# Tickers with daily 0DTE options
DAILY_0DTE_TICKERS = ['SPY', 'QQQ', 'IWM']

# Small cap tickers (strict mode)
SMALL_CAP_TICKERS = ['SLS', 'THH', 'RKLB', 'POET', 'SOFI', 'PLTR', 'MARA', 'RIOT', 'COIN']
```

---

# Deployment Guide

## Prerequisites

1. **Alpaca Account** (Paper trading recommended)
   - Create at https://alpaca.markets
   - Enable options trading (Level 2+)
   - Get API keys

2. **API Keys**
   - Polygon.io (free tier works)
   - Finnhub (free tier works)
   - Benzinga (paid)
   - Gemini (free tier)
   - DeepSeek (free tier)
   - Telegram Bot (free)

3. **Replit Account** (for deployment)

## Step-by-Step Deployment

### 1. Set Up Secrets in Replit

Go to **Secrets** tab and add:

```
ALPACA_API_KEY = your_key
ALPACA_SECRET_KEY = your_secret
POLYGON_API_KEY = your_key
FINNHUB_API_KEY = your_key
BENZINGA_API_KEY = your_key
GEMINI_API_KEY = your_key
DEEPSEEK_API_KEY = your_key
TELEGRAM_BOT_TOKEN = your_token
TELEGRAM_CHAT_ID = your_chat_id
```

### 2. Create Telegram Bot

1. Message @BotFather on Telegram
2. Send `/newbot`
3. Follow prompts to create bot
4. Copy the token
5. Message your new bot
6. Visit `https://api.telegram.org/bot<TOKEN>/getUpdates`
7. Find your chat_id in the response

### 3. Verify API Connections

Run the health check:

```bash
python3 -c "
import os
import requests

# Test each API...
# (See API Health Check section)
"
```

### 4. Deploy on Replit

1. Click **Publish** button
2. Select **Reserved VM**
3. Choose compute tier (basic works)
4. Deploy

### 5. Verify Deployment

You should receive a Telegram message:
```
🐍 WSB SNAKE v2.5 ONLINE
🔥 AGGRESSIVE MODE ACTIVE 🔥
...
```

## Post-Deployment Checklist

- [ ] Received startup Telegram message
- [ ] Account shows correct buying power
- [ ] Aggressive mode settings displayed
- [ ] Market session status correct
- [ ] No error messages in logs

---

# Debugging & Troubleshooting

## Common Issues

### 1. "No bid for option" Error

**Cause:** Option is illiquid or market is closed.

**Solution:**
- Check if market is open
- Try a different strike price
- Check if ticker has options

### 2. "Daily exposure limit reached"

**Cause:** Already deployed $6,000 today.

**Solution:**
- Wait for tomorrow
- Or increase `MAX_DAILY_EXPOSURE` in config

### 3. "Order rejected by Alpaca"

**Cause:** Various - check error message.

**Common causes:**
- Insufficient buying power
- Invalid option symbol
- Market closed
- Order size too large

**Solution:**
- Check Alpaca dashboard for specific error
- Verify option symbol format
- Check buying power

### 4. Telegram alerts not sending

**Cause:** Invalid bot token or chat ID.

**Solution:**
```python
# Test Telegram connection
import requests
token = "your_token"
chat_id = "your_chat_id"
resp = requests.post(
    f"https://api.telegram.org/bot{token}/sendMessage",
    json={"chat_id": chat_id, "text": "Test"}
)
print(resp.json())
```

### 5. "Connection limit exceeded" (Alpaca WebSocket)

**Cause:** Too many WebSocket connections.

**Solution:**
- Restart the application
- Only one instance should run at a time

## Log Locations

```
/home/runner/workspace/wsb_snake_data/
├── wsb_snake.db          # SQLite database
├── learning.db           # Learning system data
├── session_learnings.json # Battle-tested insights
└── charts/               # Generated chart images
```

## Useful Debug Commands

```bash
# Check running processes
ps aux | grep python

# View recent logs
tail -f /tmp/logs/*.log

# Check database
sqlite3 wsb_snake_data/wsb_snake.db ".tables"

# Query trades
sqlite3 wsb_snake_data/wsb_snake.db "SELECT * FROM signals ORDER BY timestamp DESC LIMIT 10"

# Check Alpaca account
python3 -c "
from wsb_snake.trading.alpaca_executor import alpaca_executor
print(alpaca_executor.get_account())
"

# Check positions
python3 -c "
from wsb_snake.trading.alpaca_executor import alpaca_executor
print(alpaca_executor.get_positions())
"
```

---

# API Reference

## Alpaca Paper Trading API

Base URL: `https://paper-api.alpaca.markets`

### Authentication Headers

```
APCA-API-KEY-ID: {api_key}
APCA-API-SECRET-KEY: {secret_key}
```

### Endpoints Used

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v2/account` | Get account info |
| GET | `/v2/positions` | Get all positions |
| POST | `/v2/orders` | Place order |
| GET | `/v2/orders` | List orders |
| DELETE | `/v2/positions/{symbol}` | Close position |

## Polygon.io API

Base URL: `https://api.polygon.io`

### Endpoints Used

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v2/aggs/ticker/{ticker}/range/{mult}/{span}/{from}/{to}` | Get bars |
| GET | `/v2/aggs/ticker/{ticker}/prev` | Previous day data |
| GET | `/v3/reference/options/contracts` | Options chain |
| GET | `/v3/quotes/{option_symbol}` | Option quote |

## Telegram Bot API

Base URL: `https://api.telegram.org/bot{token}`

### Endpoints Used

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/getMe` | Verify bot |
| POST | `/sendMessage` | Send message |
| GET | `/getUpdates` | Get chat ID |

---

# File Structure

```
wsb_snake/
├── __init__.py
├── main.py                    # Entry point
├── config.py                  # Configuration
│
├── engines/
│   ├── __init__.py
│   ├── orchestrator.py        # Main pipeline coordinator
│   ├── spy_scalper.py         # Pattern detection engine
│   ├── chart_brain.py         # AI chart analysis
│   ├── ignition_detector.py   # Early momentum detection
│   ├── pressure_engine.py     # Technical + options pressure
│   ├── surge_hunter.py        # Power hour setups
│   ├── probability_generator.py # Signal fusion
│   ├── learning_memory.py     # Trade outcome learning
│   ├── paper_trader.py        # Legacy paper trading
│   ├── strategy_classifier.py # Strategy type detection
│   ├── multi_day_scanner.py   # Longer-term setups
│   └── institutional_scalper.py # Prop desk rules
│
├── trading/
│   ├── __init__.py
│   └── alpaca_executor.py     # Alpaca trading execution
│
├── learning/
│   ├── __init__.py
│   ├── zero_greed_exit.py     # Mechanical exit system
│   ├── pattern_memory.py      # Pattern storage
│   ├── time_learning.py       # Time-of-day learning
│   ├── event_outcomes.py      # Event outcome tracking
│   ├── stalking_mode.py       # Setup monitoring
│   └── session_learnings.py   # Battle-tested wisdom
│
├── analysis/
│   ├── __init__.py
│   ├── predator_stack.py      # Multi-model AI
│   ├── chart_generator.py     # Chart creation
│   ├── scalp_langgraph.py     # LangGraph workflow
│   └── sentiment_analyzer.py  # Text sentiment
│
├── collectors/
│   ├── __init__.py
│   ├── polygon_collector.py   # Polygon.io data
│   ├── finnhub_collector.py   # Finnhub data
│   ├── benzinga_collector.py  # Benzinga news
│   ├── scalp_data_collector.py # Ultra-fast data
│   ├── alpaca_stream.py       # Real-time stream
│   └── ...                    # Other collectors
│
├── notifications/
│   ├── __init__.py
│   ├── telegram_bot.py        # Telegram alerts
│   └── message_templates.py   # Alert formatting
│
├── db/
│   ├── __init__.py
│   └── database.py            # SQLite operations
│
└── utils/
    ├── __init__.py
    ├── logger.py              # Logging setup
    └── session_regime.py      # Market hours detection
```

---

# Daily Operations

## Typical Trading Day

```
4:00 AM ET  - Pre-market data collection starts
9:25 AM ET  - System warms up, loads patterns
9:30 AM ET  - MARKET OPEN - Active scanning begins
9:30-10:30  - High volatility period (careful)
10:30-12:00 - Primary scalping window
12:00-2:00  - Lunch chop (reduced activity)
2:00-3:55   - Power hour (aggressive)
3:55 PM ET  - FORCE CLOSE all positions
4:15 PM ET  - Daily summary sent
```

## Monitoring Dashboard

Check via Telegram alerts or logs:

```
Key Metrics:
- Trades today: X/5
- Exposure used: $X/$6,000
- Win rate: X%
- Net P&L: $X
- Open positions: X
```

## Emergency Procedures

### Stop All Trading

```python
# Via Python
from wsb_snake.engines.spy_scalper import spy_scalper
from wsb_snake.trading.alpaca_executor import alpaca_executor

spy_scalper.stop()
alpaca_executor.stop_monitoring()
```

### Close All Positions Manually

```python
from wsb_snake.trading.alpaca_executor import alpaca_executor

positions = alpaca_executor.get_positions()
for pos in positions:
    alpaca_executor.close_position(pos['symbol'])
```

---

# Performance Metrics

## Key Performance Indicators

| Metric | Target | Calculation |
|--------|--------|-------------|
| Win Rate | >60% | Winning trades / Total trades |
| Avg Win | >20% | Average gain on winners |
| Avg Loss | <15% | Average loss on losers |
| Profit Factor | >1.5 | Gross profit / Gross loss |
| Daily P&L | >$300 | Net profit per day |
| Max Drawdown | <20% | Largest peak-to-trough decline |

## Historical Performance (Jan 27, 2026)

```
ETF Performance:
- GDX: +$150 (+19.5%)
- IWM: +$64 (+35.6%)
- QQQ: +$44 (+2.7%)
Total ETF: +$258 on $2,610 exposure = 9.9% ROI

Small Cap Performance:
- SLS: -$130 (-25.5%)
Total Small Cap: -$130 on $510 exposure

NET: +$128 (4.1% ROI on $3,120 total exposure)
Win Rate: 60% (3 of 5 trades profitable)
```

---

# Known Issues & Limitations

## Current Limitations

1. **0DTE Only** - System optimized for same-day expiration
2. **Paper Trading Default** - Live trading requires explicit enable
3. **US Markets Only** - Eastern timezone hardcoded
4. **Options Only** - No stock/futures support
5. **Single Instance** - Only one deployment should run

## Known Bugs

1. **AlpacaStream Connection Limit** - Sometimes hits WebSocket limit
   - Workaround: Restart application

2. **Stale Quotes** - Polygon quotes can be 60+ seconds old after hours
   - Mitigation: Check quote timestamp

3. **Small Cap Liquidity** - Some small caps have wide spreads
   - Mitigation: Strict mode + spread check

## Future Improvements

- [ ] Multi-leg options strategies (spreads)
- [ ] Pre-market/after-hours trading
- [ ] Dynamic position sizing based on confidence
- [ ] Machine learning model for pattern recognition
- [ ] Web dashboard for monitoring
- [ ] Backtesting framework

---

# Quick Reference Card

## Commands

```bash
# Start the system
python3 run_snake.py

# Health check
python3 -c "from wsb_snake.trading.alpaca_executor import alpaca_executor; print(alpaca_executor.get_account())"

# Check positions
python3 -c "from wsb_snake.trading.alpaca_executor import alpaca_executor; print(alpaca_executor.get_positions())"
```

## Key Files to Edit

| Setting | File | Variable |
|---------|------|----------|
| Daily exposure | `alpaca_executor.py` | `MAX_DAILY_EXPOSURE` |
| Per trade max | `alpaca_executor.py` | `MAX_PER_TRADE` |
| Confidence threshold | `spy_scalper.py` | `MIN_CONFIDENCE_FOR_ALERT` |
| Scan interval | `spy_scalper.py` | `SCAN_INTERVAL` |
| Target/Stop | `alpaca_executor.py` | Line ~787-788 |

## Telegram Commands (Future)

```
/status - Get current status
/positions - List open positions
/stop - Stop trading
/start - Resume trading
/pnl - Today's P&L
```

---

# Contact & Support

## Logs Location

All logs are written to stdout and can be viewed in:
- Replit Console
- `/tmp/logs/` directory

## Database

SQLite database at `wsb_snake_data/wsb_snake.db`

Tables:
- `signals` - All detected signals
- `outcomes` - Trade outcomes
- `model_weights` - Learning weights

---

**Document Version:** 2.5  
**Last Updated:** January 27, 2026  
**Author:** WSB Snake AI System

---

*"Small gains compound, big losses destroy."* - Institutional Scalper Rule
