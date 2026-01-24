# WSB Snake - Market Intelligence System

## Overview
WSB Snake is a multi-sensor trading intelligence pipeline that monitors social signals (Reddit/WSB) and market data to produce actionable trading alerts via Telegram.

## Current Status
- **Phase 1:** Connectivity + Health ✅ COMPLETE
- **Phase 2:** End-to-End Signal Pipeline ✅ COMPLETE
- **Phase 3:** Options + 0DTE (pending)
- **Phase 4:** Learning + LangGraph (pending)

## Architecture

```
wsb_snake/
├── main.py                    # Main entry point with scheduler
├── config.py                  # Environment variable loading
├── collectors/
│   ├── reddit_collector.py    # Reddit scraping (needs OAuth for full access)
│   └── market_data.py         # Alpaca market data fetching
├── parsing/
│   ├── ticker_extractor.py    # Extract stock tickers from text
│   ├── text_cleaner.py        # Clean/normalize Reddit text
│   └── dedupe.py              # Deduplicate signals
├── analysis/
│   ├── scoring.py             # Score tickers based on signals
│   ├── sentiment.py           # OpenAI sentiment analysis
│   └── risk_model.py          # Risk assessment (liquidity, volatility)
├── signals/
│   ├── signal_types.py        # Signal data structures (A+/A/B/C tiers)
│   ├── signal_router.py       # Route signals to alerts vs digest
│   └── signal_store.py        # JSON ledger for traceability
├── notifications/
│   ├── telegram_bot.py        # Send Telegram alerts
│   └── message_templates.py   # Structured message formats
└── utils/
    ├── logger.py              # Centralized logging
    ├── time_windows.py        # Time-windowed momentum tracking
    └── rate_limit.py          # API rate limiting
```

## Running the System

```bash
# Run the Python backend
PYTHONPATH=/home/runner/workspace python -m wsb_snake.main
```

Or use the run script:
```bash
python run_snake.py
```

## Environment Variables Required

| Variable | Status | Description |
|----------|--------|-------------|
| TELEGRAM_BOT_TOKEN | ✅ Set | Telegram bot token |
| TELEGRAM_CHAT_ID | ✅ Set | Telegram chat ID for alerts |
| ALPACA_API_KEY | ✅ Set | Alpaca paper trading API key |
| ALPACA_SECRET_KEY | ✅ Set | Alpaca paper trading secret |
| ALPACA_BASE_URL | ✅ Set | Alpaca API base URL |
| OPENAI_API_KEY | ✅ Set | OpenAI API key for summarization |
| REDDIT_CLIENT_ID | ❌ Missing | Required for Reddit OAuth |
| REDDIT_CLIENT_SECRET | ❌ Missing | Required for Reddit OAuth |

## Signal Tiers

| Tier | Score | Action |
|------|-------|--------|
| A+ | 85+ | Immediate alert |
| A | 70-84 | Priority alert |
| B | 50-69 | Watchlist/digest |
| C | 30-49 | Log only |

## Features

### Working
- Startup "Snake Online" ping to Telegram
- Market data fetching via Alpaca
- Signal scoring and classification
- Risk assessment (liquidity, volatility, pump detection)
- Signal storage to JSON ledger
- Telegram alert formatting
- Rate limiting
- Ticker extraction from text

### Pending
- Reddit OAuth (PRAW) for reliable data collection
- Options chain data integration
- 0DTE spike detection
- LangGraph multi-agent orchestration
- Continuous learning loop
- Periodic digest messages

## Output Format

Telegram alerts follow this structure:
```
🔥 WSB SNAKE ALERT — $TICKER
Score: 85/100 | Tier: A+

Why:
• High social velocity (5.2/min)
• Strong up move (+3.5%)
• Volume spike (2.1x avg)

Market:
Price: $150.00 (+3.5%)
Volume: 5,000,000

Risk:
⚠️ High volatility

Action: ENTER
```

## Recent Changes
- 2026-01-24: Implemented complete Phase 2 signal pipeline
- Added signal routing (A+/A/B/C tiers)
- Added risk model with liquidity/volatility checks
- Added structured Telegram message templates
- Added signal ledger (JSON storage)
