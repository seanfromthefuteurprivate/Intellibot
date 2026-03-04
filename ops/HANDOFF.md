# BEAST MODE HANDOFF - March 4, 2026 (V4.0)

## WHAT IS DEPLOYED ON EC2 (commit 739ff93)

### Beast Mode V4.0 - LIVE
- **13-signal conviction stacking system** DEPLOYED
- Service running: `wsb-snake.service active (running)`
- HYDRA Bridge connected and polling

### 13 CONVICTION SIGNALS:
| # | Signal | Description |
|---|--------|-------------|
| 1 | HYDRA_DIR | HYDRA direction aligned (BULLISH→CALL, BEARISH→PUT) |
| 2 | SWEEP | Flow sweep direction aligned (CALL_HEAVY/PUT_HEAVY) |
| 3 | DARK_POOL | Near dark pool support/resistance (within 0.5%) |
| 4 | VOLUME | Volume ratio > 1.5x recent vs older |
| 5 | GEX_NEG | GEX regime NEGATIVE (trending market) |
| 6 | ACCEL | **Momentum ACCELERATION** - candle SIZE increasing |
| 7 | WHALE | Whale premium > $500K in direction |
| 8 | CHARM | Charm flow favorable (afternoon only) |
| 9 | TIME | Time window optimal (9:35-10:30 or 14:30-15:45) |
| 10 | PREDATOR | AI pattern recognition (PredatorStackV2) |
| 11 | OR_BRK | **Opening Range Breakout** (SPY/QQQ > high or < low) |
| 12 | PM_BIAS | **Pre-market Bias** (+1 confirm, -1 conflict) |
| 13 | GEX_PROX | **GEX Proximity** favorable to flip point |

### NEW IN V4.0:
- **Signal 6 UPGRADED**: Now checks candle SIZE acceleration, not just direction
- **Signal 11 NEW**: Opening Range Breakout (9:30-9:35 high/low for SPY/QQQ)
- **Signal 12 NEW**: Pre-market bias from /tmp/premarket_bias.txt
- **Signal 13 NEW**: GEX proximity relative to flip point

### Conviction Scoring:
- **Minimum conviction = 5** (was 4) to trade
- 5-7 signals = base position size (confidence 55-69)
- 8-10 signals = 1.5x size (confidence 70-84)
- 11-13 signals = FULL SEND max $2,500 (confidence 85-95)

### HARD GATES (instant rejection):
- Polygon API unhealthy
- HYDRA disconnected/stale
- Direction conflict (CALL in BEARISH, PUT in BULLISH)
- HYDRA NEUTRAL
- Blowup > 70%
- GEX flip < 1%
- Regime CHOPPY/UNKNOWN
- Insufficient data
- Momentum wrong direction (>0.5% against)

### Configuration (jobs_day_cpl.py):
```python
SNIPER_CAPITAL = 2500
MAX_OPEN_POSITIONS = 1
DAILY_PROFIT_TARGET = 10000  # $10K target
DAILY_MAX_LOSS = -750        # -$750 floor
SNIPER_COOLDOWN_SECONDS = 300
MIN_CONVICTION = 5           # Out of 13 signals
```

### Watchlist (IWM removed):
SPY, QQQ, DIA, VXX, UVXY, TLT, IEF, XLF, UUP, GLD, SLV, GDX,
MSTR, COIN, MARA, RIOT, NVDA, TSLA, AAPL, AMZN, META, GOOGL, MSFT, AMD,
ITB, XHB, XLY, XLV, NBIS, RKLB, ASTS, LUNR, PL, ONDS, SLS

### Key Files Modified:
- `wsb_snake/execution/jobs_day_cpl.py` — Beast Mode V4 with 13 signals
- `ops/premarket_check.sh` — Now writes SPY gap bias to /tmp/premarket_bias.txt

### Opening Range Mechanism:
```python
# After 9:35 AM, _update_opening_range() fetches:
_opening_range = {
    "SPY": {"high": 585.50, "low": 584.20, "date": "2026-03-04"},
    "QQQ": {"high": 520.10, "low": 519.30, "date": "2026-03-04"}
}
# Signal 11 checks: CALL if spot > high, PUT if spot < low
```

### Pre-market Bias Mechanism:
```bash
# ops/premarket_check.sh at 9:25 AM:
# - Fetches SPY pre-market price vs previous close
# - Gap > +0.3% → BULLISH
# - Gap < -0.3% → BEARISH
# - Else → NEUTRAL
# - Writes to /tmp/premarket_bias.txt
```

### Architecture:
- EC2: i-03f3a7c46ec809a43 (AWS SSM)
- Repo: github.com/seanfromthefuteurprivate/Intellibot
- Branch: main
- Services: wsb-snake, wsb-ops-monitor (systemd)
- Telegram: alerts active
- HYDRA: connected at http://54.172.22.157:8000/api/predator

### Deploy Commands:
```bash
git add -A && git commit -m "message"
git push origin main
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["export HOME=/root && git config --global --add safe.directory /home/ubuntu/wsb-snake && cd /home/ubuntu/wsb-snake && git pull && chown -R ubuntu:ubuntu wsb_snake_data/ && systemctl restart wsb-snake"]' \
  --region us-east-1
```

### The Goal:
$2,500 capital → multiply daily via ONE lethal 0DTE trade
13-signal conviction stacking ensures only the best setups trade
Execution layer (pyramid + trailing stop) proven and working
