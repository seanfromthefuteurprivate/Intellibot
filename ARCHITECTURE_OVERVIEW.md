# WSB Snake - Architecture Overview

## System Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN.PY                                  │
│                    (FastAPI + Orchestrator)                      │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌─────────────────┐       ┌───────────────┐
│   COLLECTORS  │       │     ENGINES     │       │   ANALYSIS    │
│               │       │                 │       │               │
│ • Polygon     │       │ • SPY Scalper   │       │ • Predator    │
│ • Alpaca News │──────▶│ • Multi-Day     │◀──────│   Stack (AI)  │
│ • Finnhub     │       │ • Surge Hunter  │       │ • LangGraph   │
│ • FRED        │       │ • Institutional │       │ • Charts      │
└───────────────┘       └────────┬────────┘       └───────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐       ┌─────────────────┐       ┌───────────────┐
│   LEARNING    │       │    TRADING      │       │ NOTIFICATIONS │
│               │       │                 │       │               │
│ • Pattern     │       │ • Alpaca        │       │ • Telegram    │
│   Memory      │◀──────│   Executor      │──────▶│   Alerts      │
│ • Time Learn  │       │ • Zero Greed    │       │               │
│ • Stalking    │       │   Exit          │       │               │
└───────────────┘       └─────────────────┘       └───────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │    DATABASE     │
                        │    (SQLite)     │
                        └─────────────────┘
```

---

## Component Breakdown

### 1. Entry Point (`main.py`)
- FastAPI web server on port 5000
- Launches WSB Snake orchestrator as background task
- Health check endpoint at `/health`

### 2. Collectors (Data Input)
| Collector | Data Source | Purpose |
|-----------|-------------|---------|
| `polygon_enhanced.py` | Polygon.io | OHLCV, options chains, GEX |
| `alpaca_news.py` | Alpaca | News headlines for sentiment |
| `alpaca_stream.py` | Alpaca WebSocket | Real-time trades/quotes |
| `finnhub_collector.py` | Finnhub | Earnings, analyst ratings |
| `scalp_data_collector.py` | Polygon | 5s/15s/1m bars for scalping |
| `fred_collector.py` | FRED | Macro economic data |

### 3. Engines (Signal Detection)
| Engine | Focus | Output |
|--------|-------|--------|
| `spy_scalper.py` | 0DTE scalp patterns | ScalpSetup objects |
| `multi_day_scanner.py` | 3-21 DTE swings | MultiDaySetup objects |
| `surge_hunter.py` | Power hour breakouts | SurgeSetup objects |
| `institutional_scalper.py` | Prop desk rules | Risk parameters |

### 4. Analysis (AI Confirmation)
| Component | Model | Purpose |
|-----------|-------|---------|
| `predator_stack.py` | GPT-4o + DeepSeek | Chart vision + news sentiment |
| `scalp_langgraph.py` | LangGraph workflow | 5-node pattern validation |
| `chart_generator.py` | Matplotlib | Candlestick chart generation |
| `scalp_chart_generator.py` | Enhanced charts | VWAP bands, volume profile |

### 5. Learning (Adaptive Improvement)
| Module | Tracks | Improves |
|--------|--------|----------|
| `pattern_memory.py` | Successful patterns | Pattern confidence |
| `time_learning.py` | Hourly performance | Time-of-day weights |
| `stalking_mode.py` | Approaching triggers | Multi-day tracking |
| `session_learnings.py` | Daily lessons | Stop widths, thresholds |

### 6. Trading (Execution)
| Component | Function |
|-----------|----------|
| `alpaca_executor.py` | Order placement, position monitoring |
| `zero_greed_exit.py` | Mechanical exit enforcement |

### 7. Notifications
| Component | Channel |
|-----------|---------|
| `telegram_bot.py` | Telegram alerts |

---

## Data Flow

```
1. COLLECT
   Polygon → 5s/15s/1m bars
   Alpaca → News headlines
   Finnhub → Earnings/ratings

2. DETECT
   SPY Scalper scans 29 tickers
   Pattern matched (VWAP reclaim, breakout, etc.)
   Base confidence calculated

3. ENHANCE
   Pattern Memory boost (+5-15%)
   Time-of-Day boost (+5-10%)
   Session learnings applied

4. CONFIRM (PARALLEL)
   OpenAI GPT-4o → Chart vision analysis
   DeepSeek → News sentiment analysis
   Combined verdict calculated

5. EXECUTE
   If confidence >= 70% AND AI confirmed:
   → Alpaca order placed
   → Telegram alert sent
   → Position monitored

6. EXIT
   Every 5 seconds check:
   → Target +20%? SELL
   → Stop -15%? SELL
   → 45 min elapsed? SELL
   → 3:55 PM? CLOSE ALL

7. LEARN
   Record outcome
   Update pattern memory
   Adjust time weights
```

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| Runtime | Python 3.11 |
| Web Framework | FastAPI + Uvicorn |
| Database | SQLite |
| AI/ML | OpenAI GPT-4o, DeepSeek, LangGraph |
| Trading | Alpaca Paper Trading API |
| Data | Polygon.io, Finnhub, FRED |
| Notifications | Telegram Bot API |
| Scheduling | Python threading |
| Timezone | pytz (US/Eastern) |

---

## Key Design Decisions

1. **Parallel AI Analysis**: OpenAI (vision) and DeepSeek (text) run simultaneously for speed
2. **Sniper Mode**: AI only fires on high-value setups to save costs
3. **Rate Limiting**: 10 calls/min, 60/hour to prevent API bans
4. **Mechanical Exits**: Zero Greed Exit removes emotional trading
5. **Learning System**: Pattern memory adapts to what works
6. **SQLite**: Simple, file-based persistence for signals/outcomes
