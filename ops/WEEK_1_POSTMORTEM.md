# WSB-Snake Week 1 Post-Mortem: March 2-6, 2026

## Executive Summary
- Started: $89,953 (March 2 open)
- Ended: ~$86,014 (March 6 close)
- Total Loss: -$3,939
- Win Rate: 1 winning trade out of ~15 (6.7%)
- The one winner: QQQ $609C +$1,551 (March 2, 12:13 PM)

## Daily Breakdown

### March 2 (Monday): -$1,172
- CPL running with NO filters (original code)
- Found QQQ $609C at 50% confidence — +$1,551 winner
- Then made 10+ more trades, all losers
- Net: -$1,172
- Lesson: CPL CAN find winners. It just cannot stop trading after.

### March 3 (Tuesday): -$615
- Deployed SNIPER MODE (kill switch, position cap, cooldown)
- Position cap had race condition — Alpaca API lag
- 4 trades executed instead of 1
- QQQ PUT -> IWM CALL -> QQQ CALL -> IWM PUT (straddles!)
- Net: -$615
- Lesson: Position cap in CPL scan loop does not work.
  Alpaca API is too slow. Need cooldown (deployed later).

### March 4 (Wednesday): -$462
- Polygon API key expired (403 errors all morning)
- System dead until afternoon when key was replaced
- SPY $684 CALL entered at 1:32 PM — lost on stop
- Net: -$462
- Lesson: External dependencies (Polygon) can kill the system.
  Need health checks and fallbacks.

### March 5 (Thursday): $0 (system dead)
- Beast Mode V4.0 deployed — 13-signal conviction system
- HYDRA returns garbage for 10/13 signals
- GEX flip gate blocked everything (flip_point = null = 0.0%)
- NEUTRAL direction gate blocked everything
- UNKNOWN regime gate blocked everything
- Insufficient bars gate blocked everything (Polygon returns 1 bar)
- 0 trades. 1,400+ Dead Man Switch alerts.
- Lesson: NEVER build a system assuming ideal data.
  Test with REAL data quality before deploying.

### March 6 (Friday): -$1,656
- Diagnosis revealed 13-signal system mathematically impossible
- Deployed V5 Minimal (5 signals, MIN_CONVICTION=1)
- Fixed Polygon adapter caching (direct API fallback)
- System started generating CONV_APPROVED signals
- BUT: DIA and VXX appeared in trades (NOT in plan)
- Multiple DIA buys/sells, VXX calls — all losers
- Watchlist somehow included tickers beyond SPY/QQQ
- Net: -$1,656
- Lesson: Lowering conviction to 1 = no quality filter.
  Unknown tickers in watchlist = random trading.

## Root Causes (Ranked by Impact)

### 1. NO PERSISTENT MEMORY
Claude Code compacts and loses all context. Each session starts
from scratch. Fixes from previous sessions get half-implemented
or overwritten. The HANDOFF.md protocol helps but is not reliable.

### 2. NO END-TO-END TESTING
Every change was deployed directly to production without simulation.
No mock scans, no paper trade tests, no verification that the
full pipeline works before going live.

### 3. HYDRA DATA QUALITY
2 of 4 components healthy. Direction always NEUTRAL. Flow always 0.
Dark pool always null. Any system built on HYDRA data is building
on sand.

### 4. POLYGON DATA QUALITY
Starter plan = DELAYED data, 1 bar per request, rate limited.
The adapter caches empty responses, poisoning entire scan cycles.

### 5. WATCHLIST NOT ENFORCED
CPL somehow picked up DIA and VXX despite explicit instructions
to trade only SPY and QQQ. The watchlist is either configured
wrong or another code path adds tickers.

### 6. NO EXTERNAL OVERSIGHT
The system runs unsupervised. When something breaks (Polygon
403, Beast Mode impossible gates, DIA/VXX appearing), nobody
catches it for hours. 1,400 Dead Man Switch alerts but no
intelligent monitoring.

## What Actually Works
- Execution layer: Pyramids add correctly on winners
- Trailing stops: Managed the QQQ exit correctly (+69% move)
- Kill switch: Limits daily loss (when thresholds are right)
- Cooldown: Prevents API lag race condition
- Telegram alerts: Notifications work (when not spamming)
- CPL signal discovery: It DID find the QQQ $609C winner

## What Is Broken
- Signal quality: CPL buys both sides, trades random tickers
- Data pipeline: HYDRA 50% broken, Polygon delayed
- Conviction system: Cannot work without real data
- Watchlist enforcement: DIA/VXX should never appear
- Memory: Lost on compaction, fixes get half-applied
- Monitoring: Spams instead of intelligently diagnosing

## Weekend Plan

### Saturday (DONE)
- [x] Stop wsb-snake service
- [x] Close all positions
- [x] Deploy Second Brain EC2
- [x] Verify all API endpoints
- [x] Write this post-mortem

### Sunday (TODO)
- [ ] Rebuild CPL V6 with hard-coded watchlist (SPY, QQQ ONLY)
- [ ] Signal logic: Use ONLY Polygon price data + GEX regime
- [ ] Test in simulation: Mock scan with real data, verify signals
- [ ] Test kill switch: Verify thresholds work
- [ ] Test cooldown: Verify 5-min lockout works
- [ ] Test position cap: Verify MAX_OPEN_POSITIONS=1 works
- [ ] ONLY deploy after ALL tests pass

### Monday (TODO)
- [ ] Pre-market: Run Second Brain /api/health checks
- [ ] 9:30 AM: Enable wsb-snake (V6)
- [ ] Monitor via Second Brain
- [ ] Every trade reported to Second Brain
- [ ] Every bug auto-diagnosed with historical context
- [ ] Post-market: Generate daily post-mortem

## Second Brain Deployed

### Instance Details
- Instance ID: i-04b0f930bd1e371c1
- Public IP: 98.82.24.119
- API Port: 8080
- AI Model: Claude Haiku 4.5 via Bedrock

### API Endpoints
```bash
curl http://98.82.24.119:8080/api/health
curl http://98.82.24.119:8080/api/context
curl http://98.82.24.119:8080/api/hydra/health
curl -X POST http://98.82.24.119:8080/api/ask -d '{"question":"..."}'
```

## The North Star

**MISSION:** $2,500 capital -> multiply daily via ONE lethal 0DTE trade.

The system only needs to do THREE things right:
1. PICK the right trade (conviction stacking)
2. EXECUTE it perfectly (pyramid + trailing stop)
3. STOP after one trade (kill switch + cooldown)

We dont need to be right every day.
We need to be right 3 out of 5 days and not lose big on the other 2.

Thats the edge. Thats the mission. Build it.

---

**Document Version:** 1.0
**Created:** March 6, 2026
**Author:** Claude Opus 4.5
