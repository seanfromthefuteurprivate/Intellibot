# WSB Snake - Options Intelligence Engine

## Overview
WSB Snake is a production-grade options intelligence engine designed to detect volatility setups in the market by implementing the "Rattlesnake" predator pattern. It analyzes social signals (Reddit/WSB), market microstructure, options chain pressure, news catalysts, congressional trading, and macro regime. The system supports 7 distinct trading strategy types, ranging from 0DTE to 21 DTE, by fusing multi-source signals into scored alerts delivered via Telegram. It also includes paper trading simulation and self-learning capabilities to improve its predictive accuracy. The project aims to provide surgical precision in detecting high-probability trading opportunities by preventing premature alerts and reducing false positives through a sophisticated state machine and advanced signal processing.

## User Preferences
I want iterative development. Ask before making major changes. I prefer detailed explanations for complex concepts.

## System Architecture

### Core Design
The system is built around an orchestrator that coordinates various engines and data collectors. It employs a "Rattlesnake" state machine (`LURK` → `COILED` → `RATTLE` → `STRIKE` → `CONSTRICT` → `VENOM`) to ensure signals escalate through defined stages before triggering alerts, enhancing precision and reducing false positives. A critical component is the **Inception Stack**, which uses 6 meta-sensors to detect "phase transitions" or convex instability in the market, signaling imminent major moves.

### Data Collection & Processing
A comprehensive suite of collectors gathers data from various sources:
- **Market Data:** Polygon.io (options chain, GEX, max pain), Alpaca, Finnhub (real-time streaming via WebSocket).
- **Alpaca Real-Time Stream:** WebSocket streaming for sub-second latency:
  - **Real-time trades** - Individual trade executions with price/size
  - **Real-time quotes** - Bid/ask updates for spread analysis
  - **Real-time bars** - Minute bars as they form
  - **Trading halts** - Immediate halt detection for risk management
  - **LULD bands** - Limit Up/Limit Down for volatility boundaries
- **Ultra-Fast Scalp Data:** Enhanced Polygon data for 0DTE scalping:
  - **5-second bars** - Micro-momentum detection for surgical entries
  - **15-second bars** - Trend confirmation layer
  - **1-minute bars** - VWAP context and pattern detection
  - **5-minute bars** - Broader trend alignment
  - **Recent trades** - Order flow analysis (large vs small trades)
  - **NBBO quotes** - Bid/ask spread pressure and imbalance detection
  - **Order Flow Analysis** - Buy/sell pressure scoring for AI context
  - **Trade Classification** - Sweeps, blocks, odd lots for institutional detection
- **News & Sentiment:** Benzinga, Alpaca News, Finnhub (news sentiment, social sentiment), Alpha Vantage (AI news sentiment).
- **Finnhub Ruthless Context:** Enhanced sentiment and forward-looking data:
  - **Earnings Calendar** - Upcoming earnings with estimates/actuals
  - **Economic Calendar** - CPI, FOMC, Jobs reports with impact levels
  - **Recommendation Trends** - Analyst buy/sell consensus with percentages
  - **Price Targets** - High/low/mean analyst targets
  - **Support/Resistance** - Technical levels from pattern recognition
- **Regulatory & Insider:** SEC EDGAR (insider trading Form 4), Congressional trading data.
- **Economic & Volatility:** FRED economic data, VIX term structure.
- **Market Microstructure:** FINRA Dark Pool data, unusual options flow (scraped), simulated Level 2 data derived from options.
- **Unusual Activity:** Finviz (unusual volume detection).

### Engines
The system incorporates 6 core engines:
1.  **Ignition Detector:** Identifies early momentum and technical signals (RSI, MACD).
2.  **Pressure Engine:** Integrates technical analysis with strike structure and market regime.
3.  **Surge Hunter:** Focuses on power hour setups like VWAP and breakouts.
4.  **Probability Generator:** Fuses all collected signals into probability scores for potential trades.
5.  **Learning Memory:** Tracks trade outcomes and adjusts signal weights for continuous improvement.
6.  **Paper Trader:** Simulates trades and generates performance reports.

Additionally, a **Strategy Classifier** identifies opportunities across 7 different strategy types, and a **Multi-Day Scanner** focuses on longer-term setups (3-21 DTE).

### SPY 0DTE Scalper (Primary Focus)
The **SPY Scalper** is a dedicated hawk-like engine for 0DTE SPY options scalping, targeting quick 15-30% gains:
- **Scan Frequency:** Every 30 seconds during market hours
- **Pattern Detection:** 10 scalping patterns including:
  - VWAP bounces, reclaims, and rejections
  - Momentum surges (long and short)
  - Breakouts and breakdowns (30-bar highs/lows)
  - Failed breakout/breakdown traps (bear/bull traps)
  - Squeeze fires (volatility expansion)
- **AI Confirmation:** Uses specialized LangGraph workflow (`scalp_langgraph.py`) with GPT-4o Vision for pattern validation
- **Learning Integration:** Applies Pattern Memory and Time-of-Day Learning boosts to confidence scores
- **Alerts:** Sends complete Telegram signals with entry, target, stop, R:R ratio, and expected option gain
- **Exit Tracking:** Automatically stalks positions and sends "BOOK PROFIT NOW" alerts when targets are hit

### Learning System
The system includes 4 integrated learning modules that improve over time:
1. **Pattern Memory:** Stores successful price action patterns (breakout, squeeze, reversal, momentum) and matches against new setups. Uses 70% price action + 30% volume matching for similarity scoring.
2. **Time-of-Day Learning:** Tracks hourly and session-based performance, provides quality scores (0-100) and optimal trading recommendations.
3. **Event Outcome Database:** Records actual market moves after CPI, FOMC, earnings events. Generates expectations for future events based on historical data.
4. **Stalking Mode:** Monitors up to 20 setups approaching trigger points, with 5-level urgency system (DORMANT → WATCHING → HEATING → HOT → IMMINENT). Tracks multi-day setups up to 72 hours. **Now includes complete trade signals:**
   - Entry alerts with: ticker, entry price, target price, stop loss, trade type (CALLS/PUTS), R:R ratio
   - Automatic exit alerts when target hit: "BOOK PROFIT NOW" with P&L calculation
   - Stop loss alerts for risk management
   - Full Telegram-formatted actionable signals

All modules persist to SQLite database (`wsb_snake_data/learning.db`) and integrate with paper trader for automatic outcome recording.

### AI & Analysis
-   **ChartBrain AI:** Utilizes LangGraph and GPT-4o Vision for background AI chart analysis, validating algorithmic signals and providing visual insights.
-   **Scalp LangGraph Analyzer:** Specialized 5-node LangGraph workflow for 0DTE scalp pattern confirmation:
    1. VWAP Analysis - Assesses bounce/reclaim/rejection quality
    2. Momentum Analysis - Confirms momentum is supporting entry
    3. Trap Detection - Identifies failed breakout/breakdown opportunities
    4. Entry Timing - Determines optimal entry point
    5. Final Verdict - Decisive CALLS/PUTS/NO TRADE recommendation with confidence
-   **Predator Stack (Multi-Model AI):** Apex predator vision analysis using multiple models:
    - **Primary:** Gemini 2.0 Flash (fast, cost-effective, excellent chart vision)
    - **Fallback:** DeepSeek (budget backup when Gemini unavailable)
    - **Confirmation:** GPT-4o (validates high-confidence setups for double-check)
    - Returns decisive STRIKE_CALLS/STRIKE_PUTS/NO_TRADE/ABORT verdicts with confidence scores
-   **Surgical Precision Charts:** Enhanced chart generator for 0DTE scalping:
    - VWAP Bands (±1σ, ±2σ) - Key bounce/rejection zones
    - Volume Profile - Shows trapped buyers/sellers creating squeeze fuel
    - Delta Bars - Net buying vs selling pressure per candle
    - Combined Predator View - All-in-one chart for AI analysis
-   **Sentiment Analysis:** Processes text-based sentiment from news and social feeds.
-   **Chart Generation:** Creates candlestick charts with integrated indicators for visual analysis.

### Zero Greed Exit Protocol
Mechanical ruthless exit system with NO human override:
- **Target Hit:** IMMEDIATE EXIT - "BOOK PROFIT NOW" alert sent
- **Stop Hit:** IMMEDIATE EXIT - accept the loss, no averaging down
- **Time Decay:** EXIT at 60-minute deadline - theta kills 0DTE options
- **No Exceptions:** Pure machine execution, no "let it run" or "maybe it recovers"
- Monitors all active positions every 5 seconds for real-time exit alerts

### System Flow
The orchestrator integrates these components, applying "Alt Data Boosts" and "Session Multipliers" to refine signal scores. Signals are tiered (A+, A, B, C) based on their score, determining the action (immediate alert, watchlist, or log). Alerts are sent via Telegram.

## External Dependencies

-   **Telegram API:** For sending real-time alerts.
-   **Alpaca API:** Market data and trading.
-   **Polygon.io API:** Comprehensive stock and options data, including real-time quotes, historical data, and options chains.
-   **Benzinga API:** Financial news data.
-   **OpenAI API:** For AI summarization and vision-based chart analysis (GPT-4o Vision via LangGraph).
-   **Google Gemini API:** Primary vision model for chart analysis (Gemini 2.0 Flash).
-   **DeepSeek API:** Budget fallback vision model for chart analysis.
-   **Finnhub API:** Real-time news, sentiment, insider trading, and WebSocket streaming.
-   **SEC EDGAR:** Publicly available insider trading data (Form 4 filings).
-   **Finviz:** Unusual volume data (scraped).
-   **FINRA OTC Transparency Data:** Free dark pool data.
-   **Barchart.com:** Used for unusual options flow detection.
-   **FRED (Federal Reserve Economic Data) API:** Macroeconomic data series.
-   **Alpha Vantage API:** AI-powered news sentiment.