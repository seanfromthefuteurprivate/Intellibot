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
- **News & Sentiment:** Benzinga, Alpaca News, Finnhub (news sentiment, social sentiment), Alpha Vantage (AI news sentiment).
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

### AI & Analysis
-   **ChartBrain AI:** Utilizes LangGraph and GPT-4o Vision for background AI chart analysis, validating algorithmic signals and providing visual insights.
-   **Sentiment Analysis:** Processes text-based sentiment from news and social feeds.
-   **Chart Generation:** Creates candlestick charts with integrated indicators for visual analysis.

### System Flow
The orchestrator integrates these components, applying "Alt Data Boosts" and "Session Multipliers" to refine signal scores. Signals are tiered (A+, A, B, C) based on their score, determining the action (immediate alert, watchlist, or log). Alerts are sent via Telegram.

## External Dependencies

-   **Telegram API:** For sending real-time alerts.
-   **Alpaca API:** Market data and trading.
-   **Polygon.io API:** Comprehensive stock and options data, including real-time quotes, historical data, and options chains.
-   **Benzinga API:** Financial news data.
-   **OpenAI API:** For AI summarization and vision-based chart analysis (GPT-4o Vision via LangGraph).
-   **Finnhub API:** Real-time news, sentiment, insider trading, and WebSocket streaming.
-   **SEC EDGAR:** Publicly available insider trading data (Form 4 filings).
-   **Finviz:** Unusual volume data (scraped).
-   **FINRA OTC Transparency Data:** Free dark pool data.
-   **Barchart.com:** Used for unusual options flow detection.
-   **FRED (Federal Reserve Economic Data) API:** Macroeconomic data series.
-   **Alpha Vantage API:** AI-powered news sentiment.