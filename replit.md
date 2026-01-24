# WSB Snake - 0DTE Intelligence Engine

## Overview
WSB Snake is a production-grade 0DTE options intelligence engine that monitors social signals (Reddit/WSB), market microstructure, options chain pressure, and news catalysts to detect late-day volatility surges and "lottery ticket" setups. The system fuses multi-source signals into scored alerts sent via Telegram, with paper trading simulation and self-learning capabilities.

## Current Status
- **Phase 1:** Connectivity + Health ✅ COMPLETE
- **Phase 2:** End-to-End Signal Pipeline ✅ COMPLETE
- **Phase 3:** 0DTE Intelligence Engine ✅ COMPLETE (6 engines built)
- **Phase 4:** Learning + Optimization (ongoing)

## Architecture

```
wsb_snake/
├── main.py                    # Main entry point with scheduler
├── config.py                  # Environment variable loading
├── db/
│   └── database.py            # SQLite database for signals/outcomes
├── collectors/
│   ├── polygon_options.py     # Polygon.io options chain data
│   ├── benzinga_news.py       # Benzinga news adapter
│   ├── alpaca_news.py         # Alpaca news adapter
│   ├── reddit_collector.py    # Reddit scraping (needs OAuth)
│   └── market_data.py         # Alpaca market data
├── engines/
│   ├── orchestrator.py        # Coordinates all 6 engines
│   ├── ignition_detector.py   # Engine 1: Early momentum detection
│   ├── pressure_engine.py     # Engine 2: Options flow analysis
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
| 1 | Ignition Detector | Detects early momentum bursts (volume, velocity, news) |
| 2 | Pressure Engine | Analyzes options flow (call/put walls, IV, gamma) |
| 3 | Surge Hunter | Finds power hour setups (VWAP, breakouts) |
| 4 | Probability Generator | Fuses all signals into probability scores |
| 5 | Learning Memory | Tracks outcomes and adjusts weights |
| 6 | Paper Trader | Simulates trades and generates daily reports |

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
| POLYGON_API_KEY | ⚠️ Set (needs upgrade) | Options data requires paid plan |
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

## Alert Format

```
🔥 WSB SNAKE ALERT — $TICKER
Score: 85/100 | Tier: A+

📈 Action: STRONG_LONG
Direction: LONG

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
- 2026-01-24: Built complete 6-engine 0DTE system
- Added Polygon options adapter (needs paid plan for full access)
- Added Benzinga and Alpaca news adapters
- Added session regime detector with time-based multipliers
- Added SQLite database for signal persistence and learning
- Added self-learning weight adjustment system
- Added paper trading simulation with daily reports
