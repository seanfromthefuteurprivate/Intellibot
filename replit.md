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
- **Phase 7:** Inception Stack ✅ COMPLETE (Convex instability detection + 6 meta-sensors)
- **Phase 8:** ChartBrain AI ✅ COMPLETE (LangGraph + GPT-4o Vision chart analysis)
- **Phase 9:** Alternative Data Stack ✅ COMPLETE (Finnhub + SEC EDGAR + Finviz)
- **Phase 10:** Advanced Market Structure ✅ COMPLETE (Real-time streaming + Dark Pool + Options Flow + Simulated L2)

## Architecture

```
wsb_snake/
├── main.py                    # Main entry point with scheduler
├── config.py                  # Environment variable loading
├── db/
│   └── database.py            # SQLite database for signals/outcomes
├── collectors/
│   ├── polygon_enhanced.py    # Full Polygon basic plan utilization
│   ├── polygon_options.py     # Polygon.io options chain data + GEX + max pain
│   ├── benzinga_news.py       # Benzinga news adapter
│   ├── alpaca_news.py         # Alpaca news adapter
│   ├── reddit_collector.py    # Reddit scraping (needs OAuth)
│   ├── finnhub_collector.py   # News sentiment + social sentiment + insider MSPR
│   ├── sec_edgar_collector.py # Insider trading Form 4 filings
│   ├── finviz_collector.py    # Unusual volume detection
│   ├── finnhub_websocket.py   # Real-time streaming (WebSocket)
│   ├── finra_darkpool.py      # Dark pool data (100% free)
│   ├── options_flow_scraper.py # Unusual options flow detection
│   ├── level2_simulator.py    # Simulated L2 from options
│   └── market_data.py         # Alpaca market data
├── engines/
│   ├── orchestrator.py        # Coordinates all engines
│   ├── state_machine.py       # LURK→COILED→RATTLE→STRIKE→CONSTRICT→VENOM
│   ├── probability_engine.py  # P(hit target by close) + Chop Kill
│   ├── family_classifier.py   # 10 0DTE setup families + viability matrix
│   ├── inception_detector.py  # Convex instability detection (6 sensors)
│   ├── chart_brain.py         # Background AI chart analysis engine
│   ├── ignition_detector.py   # Engine 1: Enhanced with RSI/MACD
│   ├── pressure_engine.py     # Engine 2: Technical + strike structure
│   ├── surge_hunter.py        # Engine 3: Power hour setups
│   ├── probability_generator.py # Engine 4: Signal fusion
│   ├── learning_memory.py     # Engine 5: Self-learning weights
│   └── paper_trader.py        # Engine 6: Paper trading + reports
├── analysis/
│   ├── chart_generator.py     # Candlestick chart image generation
│   ├── langgraph_analyzer.py  # LangGraph GPT-4o vision chart analysis
│   └── sentiment.py           # Text sentiment analysis
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

## Inception Stack (Convex Instability Detection)

The Inception Stack detects **phase transitions** — the moment before major moves when small perturbations create outsized effects.

### The 6 Meta-Sensors

| Sensor | Weight | Description |
|--------|--------|-------------|
| **Event Horizon** | 20% | Variance of variance, correlation velocity, instrument dispersion |
| **Correlation Fracture** | 20% | SPY/VIX relationship breaks, QQQ divergence, IV/price disconnect |
| **Liquidity Elasticity** | 20% | ε = |ΔS|/|Q| (price change per volume), air pocket detection |
| **Temporal Anomaly** | 15% | Signal compression, faster-than-usual reactions |
| **Attention Surge** | 15% | News velocity without narrative coherence |
| **Options Pressure** | 10% | GEX regime, strike magnets, volume walls |

### Instability States

| State | Threshold | Meaning |
|-------|-----------|---------|
| STABLE | < 0.40 | Normal market conditions |
| WARMING | 0.40-0.65 | Conditions building |
| CRITICAL | 0.65-0.80 | High instability risk |
| INCEPTION | > 0.80 | Phase transition imminent |

### Inception Signals Detected
- PHASE_TRANSITION_RISK
- SPY_VIX_POSITIVE_CORRELATION
- QQQ_SPY_DIVERGENCE
- IV_EXPANSION_NO_MOVE
- BIDIRECTIONAL_VOLUME
- AIR_POCKET_DETECTED
- HIGH_FRAGILITY
- TIME_COMPRESSION
- ATTENTION_WITHOUT_NARRATIVE

### Alert Paths

Two paths to Telegram alert:
1. **Traditional**: State machine → STRIKE + P(hit) threshold + viable family
2. **Inception**: Instability index > 0.80 + 2+ signals detected

### Mathematical Framework

```
Effective Volatility: σ'_eff = σ_eff × κ(regime) × (1 + ι) × (1 + γ*)
Liquidity Elasticity: ε = |ΔS| / |Q|
Correlation Fracture: C = Σ |ρ_ij(t) - ρ̄_ij|
Instability Index: I = g(event_horizon, ε, C, temporal, attention, options)
```

## The 6 Core Engines

| Engine | Name | Purpose |
|--------|------|---------|
| 1 | Ignition Detector | Detects early momentum bursts + RSI/MACD signals |
| 2 | Pressure Engine | Technical analysis + strike structure + market regime |
| 3 | Surge Hunter | Finds power hour setups (VWAP, breakouts) |
| 4 | Probability Generator | Fuses all signals into probability scores |
| 5 | Learning Memory | Tracks outcomes and adjusts weights |
| 6 | Paper Trader | Simulates trades and generates daily reports |

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

## Environment Variables

| Variable | Status | Description |
|----------|--------|-------------|
| TELEGRAM_BOT_TOKEN | ✅ Set | Telegram bot token |
| TELEGRAM_CHAT_ID | ✅ Set | Telegram chat ID |
| ALPACA_API_KEY | ✅ Set | Alpaca API key |
| ALPACA_SECRET_KEY | ✅ Set | Alpaca secret |
| POLYGON_API_KEY | ✅ Set | Stock data + technicals + options |
| BENZINGA_API_KEY | ✅ Set | Benzinga news |
| OPENAI_API_KEY | ✅ Set | AI summarization |

## Alternative Data Stack

### Data Sources

| Source | API Key | Data | Boost |
|--------|---------|------|-------|
| **Finnhub** | FINNHUB_API_KEY | News sentiment, social sentiment, insider MSPR | +4 to +8 pts aligned, -2 to -5 contra |
| **SEC EDGAR** | None (free) | Insider trading Form 4 filings | +5 pts high activity |
| **Finviz** | None (scrape) | Unusual volume detection | +5 to +15 pts unusual volume |

### Signal Boost Calculation

```
Alt Data Boost = Σ(
  Finviz unusual volume boost (0 to +15),
  Finnhub sentiment alignment (+4 to +8 or -2 to -5),
  Finnhub insider buying (+5),
  SEC EDGAR high activity (+5)
)

Final Score = AI_Adjusted_Score + Alt_Data_Boost
```

### Pipeline Integration

- **Stage 0.5:** Collects alt data for priority tickers (SPY, QQQ, TSLA, NVDA, AAPL)
- **Stage 0.6:** Collects advanced market structure data (Dark Pool, Options Flow, L2 Sim)
- **Stage 2.9:** Applies alt data boosts to probability scores

## Advanced Market Structure (Phase 10)

### Data Sources (Free/Cheap)

| Feature | Source | Cost | Latency | Notes |
|---------|--------|------|---------|-------|
| **Real-time Streaming** | Finnhub WebSocket | Free | Sub-second | Trades for US stocks |
| **Dark Pool Visibility** | FINRA OTC Transparency | Free | 2-4 week delay | ATS volume by security |
| **Unusual Options Flow** | Barchart + Polygon | Free | 15-min delay | Sweeps, blocks detection |
| **Simulated Level 2** | Options-derived | Free | Real-time | GEX walls, max pain, OI |

### Level 2 Simulator

Since true Level 2 requires expensive subscriptions ($200+/mo), we simulate order book pressure using:
- Options strike concentration (where big orders cluster)
- GEX (Gamma Exposure) as support/resistance
- Max Pain as price magnet
- Volume Profile analysis

This gives ~80% of actionable insight at 0% of the cost.

### What We Can't Get Cheaply
- Sub-millisecond execution (requires co-location ~$10K+/mo)
- Real-time Level 2 for stocks (NASDAQ TotalView ~$26/mo retail)
- Real-time dark pool prints (requires $1K+/mo feeds)

## Recent Changes
- **2026-01-24 (Latest):** Implemented Alternative Data Stack
  - Finnhub collector: News sentiment + social sentiment + insider MSPR
  - SEC EDGAR collector: Insider trading Form 4 filings (free, no API key)
  - Finviz collector: Unusual volume detection (scraping, no API key)
  - Stage 0.5: Collects alt data for priority tickers
  - Stage 2.9: Applies signal boosts based on alt data alignment
  - Unusual volume gives +5 to +15 point boost
  - Sentiment alignment gives +4 to +8 boost, contra gives -2 to -5 penalty
- 2026-01-24: Implemented ChartBrain AI
  - LangGraph-based AI chart analyzer with GPT-4o vision
  - Generates candlestick charts with indicators (SMA, VWAP)
  - Background analysis thread studies all 11 tickers continuously
  - 5-node LangGraph workflow: patterns → trend → levels → recommendation → compile
  - AI validates algorithmic signals (boosts confirmed, dampens contradicted)
  - Integrated as Stage 2.8 in orchestrator pipeline
  - 5-minute cache TTL to avoid redundant API calls
- 2026-01-24: Implemented Inception Stack
  - Added 6 meta-sensors for convex instability detection
  - Event Horizon: variance of variance + instrument dispersion
  - Correlation Fracture: SPY/VIX breaks, QQQ divergence, IV/price disconnect
  - Liquidity Elasticity: price change per volume, air pocket detection
  - Temporal Anomaly: signal compression, reaction latency
  - Attention Surge: news velocity without narrative coherence
  - Options Pressure: GEX regime, strike magnets, volume walls
  - Integrated with orchestrator as Stage 2.7
  - Added dedicated Telegram alert for inception detections
- 2026-01-24: Implemented Setup Family Classifier (10 0DTE families)
- 2026-01-24: Implemented Rattlesnake Pattern (State Machine + Probability Engine)
- 2026-01-24: Enhanced with full Polygon basic plan utilization
