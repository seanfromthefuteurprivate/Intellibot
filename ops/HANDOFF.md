# BEAST MODE HANDOFF - March 4, 2026 (Updated)

## WHAT IS DEPLOYED ON EC2 (commit 0e31205)

### Beast Mode V3.1 - LIVE
- 10-signal conviction stacking system DEPLOYED
- Service running: `wsb-snake.service active (running)`
- HYDRA Bridge connected and polling

### Changes in V3.1 (this commit):
1. **Signal 10: Predator Vision** — AI pattern recognition via PredatorStackV2
2. **Kill switch updated** — $10K profit / -$750 loss (was $2.5K / -$500)
3. **IWM removed** from watchlist (illiquid)
4. Session halt REMOVED (hunts all day)
5. SPY block REMOVED

### Files Modified:
- `wsb_snake/execution/jobs_day_cpl.py` — Beast Mode V3.1 with 10 signals
- `wsb_snake/trading/alpaca_executor.py` — Updated kill switch
- `wsb_snake/ai_stack/predator_stack_v2.py` — Now wired into CPL

### 10 CONVICTION SIGNALS IMPLEMENTED:
1. HYDRA direction aligned
2. Sweep direction aligned (flow_sweep_direction)
3. Near dark pool level (dp_support/resistance within 0.5%)
4. Volume ratio > 1.5x
5. GEX regime favorable (NEGATIVE = trending)
6. Momentum > 0.3% in direction
7. Whale premium > $500K in direction
8. Charm flow favorable (afternoon only)
9. Time window optimal (9:35-10:30 AM or 2:30-3:45 PM)
10. **Predator Vision** — AI pattern recognition (STRIKE + >60% conviction)

### HARD GATES (instant rejection):
- Polygon health failing
- HYDRA disconnected/stale
- Direction conflict (CALL in BEARISH, PUT in BULLISH)
- Blowup > 70%
- GEX flip < 1%
- Regime CHOPPY/UNKNOWN
- Insufficient data
- Momentum wrong direction (>0.5% against)

### Conviction Scoring:
- Minimum conviction = 4 signals to trade
- 4-5 signals = base position size (confidence 55-69)
- 6-7 signals = 1.5x size (confidence 70-84)
- 8-10 signals = FULL SEND max $2,500 (confidence 85-95)

### Current Configuration:
- SNIPER_CAPITAL = $2,500
- MAX_OPEN_POSITIONS = 1
- DAILY_PROFIT_TARGET = $10,000
- DAILY_MAX_LOSS = -$750
- SNIPER_COOLDOWN_SECONDS = 300 (5 min)

### Watchlist (IWM removed):
SPY, QQQ, DIA, VXX, UVXY, TLT, IEF, XLF, UUP, GLD, SLV, GDX,
MSTR, COIN, MARA, RIOT, NVDA, TSLA, AAPL, AMZN, META, GOOGL, MSFT, AMD,
ITB, XHB, XLY, XLV, NBIS, RKLB, ASTS, LUNR, PL, ONDS, SLS

### Architecture:
- EC2: i-03f3a7c46ec809a43 (AWS SSM)
- Repo: github.com/seanfromthefuteurprivate/Intellibot
- Branch: main
- Services: wsb-snake, wsb-ops-monitor (systemd)
- Telegram: alerts active
- HYDRA: connected at http://54.172.22.157:8000/api/predator

### NOT YET DONE:
- [ ] Opening range breakout gate (discussed, not coded)
- [ ] Momentum acceleration (candle SIZE increasing, not just direction)
- [ ] Pre-market bias from futures
- [ ] GEX-aware strike selection
- [ ] Simulated scan showing all gates (market closed)

### Deploy Commands:
```bash
git add -A && git commit -m "update message"
git push origin main
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["export HOME=/root && git config --global --add safe.directory /home/ubuntu/wsb-snake && cd /home/ubuntu/wsb-snake && git pull && systemctl restart wsb-snake"]' \
  --region us-east-1
```

### The Goal:
$2,500 capital → multiply daily via ONE lethal 0DTE trade
10-signal conviction stacking ensures only the best setups trade
Execution layer (pyramid + trailing stop) proven and working
