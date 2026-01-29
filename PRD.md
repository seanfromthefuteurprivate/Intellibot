# WSB Snake - Product Requirements Document (PRD)

## Executive Summary

WSB Snake is an autonomous 0DTE options scalping engine that combines algorithmic pattern detection with AI-powered confirmation to execute paper trades on Alpaca. The system targets 15-30% gains on quick scalp trades while maintaining strict risk management.

---

## Product Vision

**Mission:** Provide surgical precision in detecting high-probability 0DTE trading opportunities.

**Vision:** An autonomous trading system that learns from its performance and continuously improves its predictive accuracy.

---

## Target Users

1. **Primary:** Options traders seeking automated 0DTE signals
2. **Secondary:** Algorithmic trading enthusiasts
3. **Tertiary:** Quantitative researchers studying market microstructure

---

## Core Features

### F1: Multi-Ticker Scanning
- **Description:** Continuously scan 29 tickers for scalping opportunities
- **Frequency:** Every 30 seconds during market hours
- **Tickers:** SPY, QQQ, IWM, AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, AMD, NFLX, COIN, MARA, RIOT, PLTR, SOFI, NIO, BABA, SNAP, GME, AMC, BBBY, HOOD, LCID, RIVN, F, UBER, DIS

### F2: Pattern Detection
- **Description:** Detect 10 intraday scalping patterns
- **Patterns:**
  - VWAP Reclaim/Rejection/Bounce
  - Momentum Surge (Long/Short)
  - Breakout/Breakdown
  - Failed Breakout/Breakdown
  - Squeeze Fire
- **Confidence:** Calculate base confidence 0-100%

### F3: AI Confirmation (Parallel)
- **Description:** Use AI for pattern validation
- **Architecture:**
  - OpenAI GPT-4o: Candlestick chart vision analysis
  - DeepSeek: News sentiment analysis (text)
  - Both run simultaneously
- **Output:** STRIKE_CALLS, STRIKE_PUTS, NO_TRADE, or ABORT

### F4: Automated Trading
- **Description:** Execute paper trades on Alpaca
- **Triggers:**
  - Alert at 60%+ confidence
  - Auto-execute at 70%+ AND AI confirmed
- **Limits:**
  - $1,500 max per trade
  - $6,000 max daily exposure
  - 5 max concurrent positions

### F5: Position Monitoring
- **Description:** Monitor open positions for exits
- **Frequency:** Every 5 seconds
- **Exit Conditions:**
  - Target: +20%
  - Stop: -15%
  - Time: 45 minutes
  - EOD: 3:55 PM ET

### F6: Telegram Alerts
- **Description:** Real-time notifications
- **Alert Types:**
  - Entry signals
  - Auto-execution confirmations
  - Position exits
  - Session summaries

### F7: Learning System
- **Description:** Adaptive improvement
- **Components:**
  - Pattern Memory: Learn from successful patterns
  - Time Learning: Optimize for time-of-day
  - Session Learnings: Apply daily lessons

---

## Non-Functional Requirements

### NFR1: Performance
- Signal detection: < 500ms per ticker
- AI analysis: < 5 seconds
- Order execution: < 1 second
- Total signal-to-trade: < 10 seconds

### NFR2: Reliability
- System uptime: 99.5% during market hours
- Auto-recovery from API failures
- Graceful degradation without AI

### NFR3: Safety
- Paper trading only (no real money)
- Hard position limits
- Mandatory EOD close
- Rate limiting on all APIs

### NFR4: Cost Control
- AI budget: $5/day max
- OpenAI calls: 50/day max
- Sniper mode to reduce unnecessary calls

---

## Success Metrics

### KPI1: Signal Quality
- Target: 55%+ win rate
- Measure: Winning trades / Total trades

### KPI2: Risk-Adjusted Return
- Target: 1.5+ Sharpe ratio
- Measure: Return / Volatility

### KPI3: Execution Efficiency
- Target: < 5 second signal-to-trade
- Measure: Time from pattern detection to order placed

### KPI4: System Health
- Target: < 5% error rate
- Measure: Failed API calls / Total calls

---

## Roadmap

### Phase 1: Foundation (Complete)
- [x] Core pattern detection
- [x] Alpaca paper trading
- [x] Telegram alerts
- [x] Basic AI confirmation

### Phase 2: AI Enhancement (Complete)
- [x] Multi-model AI (OpenAI + DeepSeek)
- [x] Parallel analysis
- [x] Sniper mode cost control
- [x] Rate limiting

### Phase 3: Learning (In Progress)
- [x] Pattern memory
- [x] Time-of-day learning
- [ ] Reinforcement learning for thresholds
- [ ] Automatic parameter optimization

### Phase 4: Scale (Future)
- [ ] Real money trading (opt-in)
- [ ] Multi-account support
- [ ] Web dashboard
- [ ] Mobile app

---

## Risk Assessment

### R1: API Reliability
- **Risk:** External APIs may be unavailable
- **Mitigation:** Fallback chains, graceful degradation

### R2: AI Cost Overrun
- **Risk:** AI calls could exceed budget
- **Mitigation:** Hard caps, sniper mode, rate limits

### R3: False Signals
- **Risk:** Patterns may not predict actual moves
- **Mitigation:** AI confirmation, learning system

### R4: Regulatory
- **Risk:** Paper trading rules may change
- **Mitigation:** Stay informed, maintain compliance

---

## Appendix

### A1: Ticker Universe

| Category | Tickers |
|----------|---------|
| Index ETFs | SPY, QQQ, IWM |
| Mega Cap | AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL |
| Growth | AMD, NFLX, UBER |
| Crypto-Related | COIN, MARA, RIOT |
| Meme Stocks | GME, AMC, BBBY, HOOD |
| EV/Tech | NIO, LCID, RIVN, PLTR, SOFI |
| Other | BABA, SNAP, F, DIS |

### A2: Confidence Boost Breakdown

| Source | Max Boost |
|--------|-----------|
| Base pattern | 60% |
| Volume (2x+) | +10% |
| Momentum (0.3%+) | +8% |
| VWAP alignment | +5% |
| Pattern memory | +15% |
| Time-of-day | +10% |
| AI confirmation | +10% |
| Chart+News agree | +10% |
| **Max Possible** | **128%** |

### A3: Exit Priority

1. TARGET HIT (+20%) - Immediate
2. STOP LOSS (-15%) - Immediate
3. TIME DECAY (45 min) - Forced
4. EOD (3:55 PM ET) - Mandatory
