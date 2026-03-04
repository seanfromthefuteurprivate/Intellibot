# BEAST MODE HANDOFF - March 4, 2026

## WHAT IS DEPLOYED ON EC2 (commit 7bd6112)

### Files Modified:
- wsb_snake/execution/jobs_day_cpl.py — BEAST MODE V3.0
  9-signal conviction stacking system DEPLOYED, compiles clean
- wsb_snake/trading/alpaca_executor.py — kill switch in place
- wsb_snake/engines/v7_scalper.py — V7 disabled
- ops/monitor_agent.py — DOWN_STATES + circuit breaker
- wsb_snake/utils/polygon_health.py — NEW health monitoring

### What's LIVE on EC2 right now:
- Session halt: REMOVED (hunts all day)
- 9 conviction signals implemented (1-9)
- Signal 10 (Predator Vision) NOT YET ADDED — was interrupted
- Hard gates: Polygon health, HYDRA connection, HYDRA direction,
  blowup >70%, GEX flip <1%, regime CHOPPY/UNKNOWN, data availability,
  momentum wrong direction
- Conviction minimum: 4 signals to trade
- Conviction sizing: 4-5=base, 6-7=1.5x, 8-9=full send
- Kill switch: $2,500 profit / -$500 loss (NEEDS UPDATE to $10K/-$750)
- Cooldown: 5 min between trades
- Max positions: 1
- IWM: NOT YET REMOVED from watchlist
- SPY block: REMOVED
- V7: DISABLED

### 9 CONVICTION SIGNALS IMPLEMENTED:
1. HYDRA direction aligned
2. Sweep direction aligned (flow_sweep_direction)
3. Near dark pool level (dp_support/resistance within 0.5%)
4. Volume ratio > 1.5x
5. GEX regime favorable (NEGATIVE = trending)
6. Momentum > 0.3% in direction
7. Whale premium > $500K in direction
8. Charm flow favorable (afternoon only)
9. Time window optimal (9:35-10:30 AM or 2:30-3:45 PM)

### What's NOT YET DONE:
- [ ] Signal 10: Predator Vision (predator_stack_v2 exists, not wired)
- [ ] Kill switch update: $2500→$10000 profit, -$500→-$750 loss
- [ ] Remove IWM from watchlist
- [ ] Opening range breakout gate (discussed, not coded)
- [ ] Momentum acceleration (candle size increasing, not just direction)
- [ ] Pre-market bias from futures
- [ ] GEX-aware strike selection
- [ ] Run simulated scan to verify all gates work

### CRITICAL: Current state
- LOCAL code has beast mode changes, compiled successfully
- EC2 has commit 7bd6112 (previous HYDRA integration)
- Need to: git commit new changes, push, pull on EC2, restart

### Account State:
- Portfolio: ~$88,034
- Daily P&L: ~-$462
- Open positions: 0

### Architecture:
- EC2: i-03f3a7c46ec809a43 (AWS SSM)
- Repo: github.com/seanfromthefuteurprivate/Intellibot
- Branch: main
- Services: wsb-snake, wsb-ops-monitor (systemd)
- Telegram: alerts active
- HYDRA: connected, direction=NEUTRAL, regime=UNKNOWN

### Key Files:
- _check_entry_quality() at jobs_day_cpl.py:342-660
- Conviction constants at jobs_day_cpl.py:84-91
- SNIPER_CAPITAL=2500, MAX_OPEN_POSITIONS=1
- predator_stack_v2.py exists at wsb_snake/ai_stack/

### The Goal:
$2,500 capital → multiply daily via ONE lethal 0DTE trade
9-signal conviction stacking ensures only the best setups trade
Execution layer (pyramid + trailing stop) proven and working
