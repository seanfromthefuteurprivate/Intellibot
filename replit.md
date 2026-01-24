# WSB Snake - 0DTE Intelligence Engine

## Overview
WSB Snake is a production-grade 0DTE options intelligence engine implementing the **"Rattlesnake" predator pattern**. It monitors social signals (Reddit/WSB), market microstructure, options chain pressure, and news catalysts to detect late-day volatility surges and "lottery ticket" setups. The system fuses multi-source signals into scored alerts sent via Telegram, with paper trading simulation and self-learning capabilities.

## Current Status
- **Phase 1:** Connectivity + Health ✅ COMPLETE
- **Phase 2:** End-to-End Signal Pipeline ✅ COMPLETE
- **Phase 3:** 0DTE Intelligence Engine ✅ COMPLETE (6 engines built)
- **Phase 4:** Enhanced Technical Analysis ✅ COMPLETE (RSI, MACD, SMA, EMA)
- **Phase 5:** Rattlesnake Pattern ✅ COMPLETE (State Machine + Probability Engine + Chop Filter)
- **Phase 6:** Setup Family Classifier ✅ COMPLETE (10 0DTE families + viability matrix)

## Architecture

```
wsb_snake/
├── main.py                    # Main entry point with scheduler
├── config.py                  # Environment variable loading
├── db/
│   └── database.py            # SQLite database for signals/outcomes
├── collectors/
│   ├── polygon_enhanced.py    # Full Polygon basic plan utilization
│   ├── polygon_options.py     # Polygon.io options chain data
│   ├── benzinga_news.py       # Benzinga news adapter
│   ├── alpaca_news.py         # Alpaca news adapter
│   ├── reddit_collector.py    # Reddit scraping (needs OAuth)
│   └── market_data.py         # Alpaca market data
├── engines/
│   ├── orchestrator.py        # Coordinates all engines
│   ├── state_machine.py       # LURK→COILED→RATTLE→STRIKE→CONSTRICT→VENOM
│   ├── probability_engine.py  # P(hit target by close) + Chop Kill
│   ├── family_classifier.py   # NEW: 10 0DTE setup families + viability matrix
│   ├── ignition_detector.py   # Engine 1: Enhanced with RSI/MACD
│   ├── pressure_engine.py     # Engine 2: Technical + strike structure
│   ├── surge_hunter.py        # Engine 3: Power hour setups
│   ├── probability_generator.py # Engine 4: Signal fusion
│   ├── learning_memory.py     # Engine 5: Self-learning weights
│   └── paper_trader.py        # Engine 6: Paper trading + reports
├── utils/
│   ├── session_regime.py      # Market session detection
│   ├── logger.py              # Centralized logging
│   └── rate_limit.py          # API rate limiting
└── notifications/
    ├── telegram_bot.py        # Send Telegram alerts
    └── message_templates.py   # Structured message formats
```

## The Rattlesnake Pattern

The engine behaves like a **predator** using a formal state machine:

### State Machine (6 States)

| State | Description | Entry Conditions |
|-------|-------------|------------------|
| **LURK** | Passive monitoring, building heat maps | Default state |
| **COILED** | Conditions building, sensitivity raised | Time alignment (approaching power hour) |
| **RATTLE** | Warning signals, publishing "watch" events | ≥2 ignition signals (volume + catalyst + momentum) |
| **STRIKE** | Attack mode, triggering alerts/paper trades | Structure break + direction confirmed + P(hit)>55% |
| **CONSTRICT** | Post-strike management | After strike executed |
| **VENOM** | End-of-day postmortem | At market close |

### Why State Machine?
- **Prevents premature alerts**: Signals must escalate through states before triggering
- **Reduces false positives**: Multiple conditions must align
- **Surgical precision**: Only strikes when high probability + structure confirmed

## Probability Engine

Calculates **P(hit target by close)** using:

| Component | Source | Description |
|-----------|--------|-------------|
| Realized Volatility | Recent price bars | Regime-adjusted annualized vol |
| Distance to Target | Key levels (day high/low, VWAP) | Distance as % of price |
| Time Remaining | Session clock | Minutes to close |
| Regime Scalar | Market classification | Trend/chop/panic multipliers |

### Hazard Curve
- P(hit in next 5 min)
- P(hit in next 10 min)
- P(hit in next 20 min)
- P(hit by close)

### Entry Quality Assessment
- **Optimal**: High near-term probability + high overall probability
- **Acceptable**: Medium probability window
- **Poor**: Low probability or time running out

## Chop Kill Filter

Blocks signals in choppy, fake-breakout conditions:

| Metric | Weight | Description |
|--------|--------|-------------|
| Range Compression | 30 | Low ATR into a level = compression |
| Trend Strength | 40 | Weak trend = chop |
| VWAP Crossings | 30 | Many crosses = whipsaw |

**Block threshold**: Score ≥60

## The 10 Setup Families

Each "family" describes a distinct 0DTE setup archetype with unique probability curves:

| # | Family | Type | Peak Hour | Description |
|---|--------|------|-----------|-------------|
| 1 | **VWAP Reclaim + Gamma Snap** | Consistent | 3pm ET | Price reclaims VWAP late-day with volume |
| 2 | **Strike Magnet Pin → Break** | Asymmetric | 3pm ET | Price escapes heavy OI pin late-day |
| 3 | **Afternoon Range Expansion** | Consistent | 2pm ET | First meaningful expansion after compression |
| 4 | **Liquidity Sweep + Reversal** | Asymmetric | 3pm ET | Stop-hunt snapback with volume |
| 5 | **News-Assisted Gamma Ignition** | Asymmetric | 3pm ET | Headline + structure alignment |
| 6 | **Power-Hour Trend Continuation** | Consistent | 3pm ET | Trend resumes after consolidation |
| 7 | **False Break Trap → Real Move** | Asymmetric | 3pm ET | Second attempt succeeds after failed first |
| 8 | **Volatility Regime Shift** | Asymmetric | 3pm ET | Dead→alive volatility expansion |
| 9 | **Crowd Ignition + Structure** | Asymmetric | 3pm ET | Social attention + technical break |
| 10 | **Mean Reversion Snap** | Consistent | 3pm ET | Extended move snaps back to mean |

### Family Viability Matrix

The engine maintains a live ranking of families based on:
- **Regime Compatibility**: Which families work in current market conditions
- **Time Alignment**: Each family has optimal time windows
- **Volatility State**: Some families thrive in low-vol, others need expansion
- **Memory Veto**: Recent failure rate affects viability
- **Saturation**: Too many signals → family enters cooldown

### Asymmetric vs Consistent Families

| Type | Hit Rate | Payoff | Strategy |
|------|----------|--------|----------|
| **Asymmetric** | Low | Extreme | Rare alignment, explosive outcomes |
| **Consistent** | Higher | Moderate | Work across many days |

### Family Lifecycle States

| State | Description |
|-------|-------------|
| DORMANT | Not currently viable |
| AWAKENING | Viability rising |
| ALIVE | Conditions building, monitoring active |
| PEAKED | All conditions met, maximum probability |
| DYING | Viability declining |
| DEAD | Time window expired or regime poisoned |
| COOLDOWN | Repetition exhaustion, waiting to reset |

**Key Insight**: Alerts only fire when:
1. State machine reaches STRIKE
2. Probability engine confirms P(hit) threshold
3. Family is ALIVE or PEAKED (viable)

## The 6 Core Engines

| Engine | Name | Purpose |
|--------|------|---------|
| 1 | Ignition Detector | Detects early momentum bursts + RSI/MACD signals |
| 2 | Pressure Engine | Technical analysis + strike structure + market regime |
| 3 | Surge Hunter | Finds power hour setups (VWAP, breakouts) |
| 4 | Probability Generator | Fuses all signals into probability scores |
| 5 | Learning Memory | Tracks outcomes and adjusts weights |
| 6 | Paper Trader | Simulates trades and generates daily reports |

## Enhanced Polygon Basic Plan Usage

The system maximizes Polygon.io basic plan with these endpoints:

### Available Data Sources
| Endpoint | Usage | Status |
|----------|-------|--------|
| Stock Aggregates (1min) | Intraday momentum detection | ✅ Working |
| Previous Day Agg | Gap analysis | ✅ Working |
| Stock Snapshot | Real-time quotes | ✅ Working |
| **RSI Indicator** | Overbought/oversold detection | ✅ Working |
| **SMA Indicator** | Trend following | ✅ Working |
| **EMA Indicator** | Fast moving average | ✅ Working |
| **MACD Indicator** | Momentum crossovers | ✅ Working |
| Gainers/Losers | Market regime detection | ✅ Working |
| Options Contracts | Strike structure analysis | ✅ Working |
| Options Snapshot | Real-time IV/volume | ❌ Requires upgrade |

### Technical Signals Detected
- RSI_OVERBOUGHT / RSI_OVERSOLD
- RSI_RISING / RSI_FALLING
- ABOVE_SMA20 / BELOW_SMA20
- EMA_ABOVE_SMA / EMA_BELOW_SMA
- MACD_BULLISH / MACD_BEARISH
- GAP_UP / GAP_DOWN
- VOLUME_SURGE / VOLUME_DRY
- NEAR_DAY_HIGH / NEAR_DAY_LOW

### Market Regime Detection
- Uses Gainers/Losers ratio to classify market:
  - strong_bullish / bullish / neutral / bearish / strong_bearish
- Boosts aligned signals, reduces counter-trend signals

## Running the System

```bash
# Run the Python backend
PYTHONPATH=/home/runner/workspace python -m wsb_snake.main
```

Or use the run script:
```bash
python run_snake.py
```

## Environment Variables

| Variable | Status | Description |
|----------|--------|-------------|
| TELEGRAM_BOT_TOKEN | ✅ Set | Telegram bot token |
| TELEGRAM_CHAT_ID | ✅ Set | Telegram chat ID |
| ALPACA_API_KEY | ✅ Set | Alpaca API key |
| ALPACA_SECRET_KEY | ✅ Set | Alpaca secret |
| POLYGON_API_KEY | ✅ Set | Stock data + technicals (basic plan) |
| BENZINGA_API_KEY | ✅ Set | Benzinga news |
| OPENAI_API_KEY | ✅ Set | AI summarization |
| REDDIT_CLIENT_ID | ❌ Missing | Reddit OAuth |
| REDDIT_CLIENT_SECRET | ❌ Missing | Reddit OAuth |

## 0DTE Universe
Monitored tickers: SPY, QQQ, IWM, TSLA, NVDA, AAPL, META, AMD, AMZN, GOOGL, MSFT

## Signal Tiers

| Tier | Score | Action |
|------|-------|--------|
| A+ | 85+ | Immediate alert + paper trade (if STRIKE state) |
| A | 70-84 | Alert + paper trade (if STRIKE state) |
| B | 50-69 | Watchlist |
| C | 30-49 | Log only |

## Session Multipliers

| Session | Multiplier | Notes |
|---------|------------|-------|
| Premarket | 0.5x | Low liquidity |
| Open | 1.0x | First hour volatility |
| Morning | 0.9x | Settling |
| Lunch | 0.5x | Chop zone - avoid |
| Power Hour Early | 1.2x | Momentum building |
| Power Hour | 1.5x | Prime 0DTE time |
| After Hours | 0.3x | Low priority |
| Closed | 0.0x | Weekend/holiday |

## Alert Format

```
🔥 WSB SNAKE ALERT — $TICKER
Score: 85/100 | Tier: A+

📈 Action: STRONG_LONG
Direction: LONG

📊 Technical Indicators
• RSI(14): 28 (oversold bounce)
• MACD: Bullish histogram
• Price vs SMA(20): +1.2%

📊 Component Scores
• Ignition: 75
• Pressure: 80
• Surge: 90

💡 Thesis
• Volume 3.5x normal
• Breaking day high +0.5%
• News catalyst detected

🎯 Levels
Entry: $150.00
Stop: $147.50
Target 1: $152.50
R:R = 2.0

⏰ Timing
Urgency: HIGH
Minutes to close: 45
```

## Database Schema

**signals** - Stores every detected signal with features
**outcomes** - Tracks what happened after each signal
**paper_trades** - Simulated trade executions
**model_weights** - Adaptive feature weights

## Recent Changes
- **2026-01-24 (Latest):** Implemented Setup Family Classifier
  - Added 10 0DTE setup families with unique probability curves
  - Built Family Viability Matrix (regime/time/volatility ranking)
  - Added family lifecycle management (death conditions, cooldowns, memory veto)
  - Integrated family classification into orchestrator pipeline
  - Alerts now require viable family + STRIKE state + P(hit) threshold
- 2026-01-24: Implemented Rattlesnake Pattern
  - Added formal State Machine (LURK→COILED→RATTLE→STRIKE→CONSTRICT→VENOM)
  - Added Probability Engine with P(hit target by close) calculations
  - Added Chop Kill filter to block fake breakout signals
  - Orchestrator now gates alerts through state machine
  - Added VENOM (end-of-day postmortem) report
- 2026-01-24: Enhanced with full Polygon basic plan utilization
- Added RSI, SMA, EMA, MACD technical indicators
- Added market regime detection (gainers/losers ratio)
- Added strike structure analysis from options contracts
- Integrated technicals into Ignition Detector and Pressure Engine
- Pipeline now detects technical signals across all 11 tickers

## Future Enhancements (Require API Upgrades)

### With Polygon Options Starter ($79/mo):
- Real-time IV analysis
- Call/put volume walls
- Gamma exposure calculations
- Open interest clustering
- Max pain calculation
- Strike clustering for gamma-magnet detection

### With Reddit OAuth:
- Live WSB mention tracking
- Social velocity signals
- Sentiment analysis on posts
- Crowd heat integration
