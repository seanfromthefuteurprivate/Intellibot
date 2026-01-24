# WSB Snake - 0DTE Intelligence Engine

## Overview
WSB Snake is a production-grade 0DTE options intelligence engine that monitors social signals (Reddit/WSB), market microstructure, options chain pressure, and news catalysts to detect late-day volatility surges and "lottery ticket" setups. The system fuses multi-source signals into scored alerts sent via Telegram, with paper trading simulation and self-learning capabilities.

## Current Status
- **Phase 1:** Connectivity + Health ✅ COMPLETE
- **Phase 2:** End-to-End Signal Pipeline ✅ COMPLETE
- **Phase 3:** 0DTE Intelligence Engine ✅ COMPLETE (6 engines built)
- **Phase 4:** Enhanced Technical Analysis ✅ COMPLETE (RSI, MACD, SMA, EMA)

## Architecture

```
wsb_snake/
├── main.py                    # Main entry point with scheduler
├── config.py                  # Environment variable loading
├── db/
│   └── database.py            # SQLite database for signals/outcomes
├── collectors/
│   ├── polygon_enhanced.py    # NEW: Full Polygon basic plan utilization
│   ├── polygon_options.py     # Polygon.io options chain data
│   ├── benzinga_news.py       # Benzinga news adapter
│   ├── alpaca_news.py         # Alpaca news adapter
│   ├── reddit_collector.py    # Reddit scraping (needs OAuth)
│   └── market_data.py         # Alpaca market data
├── engines/
│   ├── orchestrator.py        # Coordinates all 6 engines
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

## The 6 Engines

| Engine | Name | Purpose |
|--------|------|---------|
| 1 | Ignition Detector | Detects early momentum bursts + RSI/MACD signals |
| 2 | Pressure Engine | Technical analysis + strike structure + market regime |
| 3 | Surge Hunter | Finds power hour setups (VWAP, breakouts) |
| 4 | Probability Generator | Fuses all signals into probability scores |
| 5 | Learning Memory | Tracks outcomes and adjusts weights |
| 6 | Paper Trader | Simulates trades and generates daily reports |

## Enhanced Polygon Basic Plan Usage

The system now maximizes Polygon.io basic plan with these endpoints:

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
| A+ | 85+ | Immediate alert + paper trade |
| A | 70-84 | Alert + paper trade |
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

### With Reddit OAuth:
- Live WSB mention tracking
- Social velocity signals
- Sentiment analysis on posts
