# WSB SNAKE - BRUTAL AGENT HANDOVER

**READ THIS OR YOU WILL FUCK UP TRADES AND LOSE MONEY.**

This is not documentation. This is a survival guide. You are inheriting a 0DTE options scalping system that trades real money on Alpaca. Every bug costs dollars. Every misunderstanding costs the user's capital.

---

## WHAT THIS SYSTEM DOES IN 30 SECONDS

1. **Scans** SPY/QQQ/IWM for high-conviction scalp setups (68%+ confidence)
2. **Executes** 0DTE options trades on Alpaca paper trading
3. **Monitors** positions with automatic target/stop exits
4. **Alerts** via Telegram on every trade event
5. **Learns** from outcomes to improve future signals

**One sentence:** AI-powered 0DTE options scalper that only takes A+ setups and auto-exits.

---

## THE ONLY FILES THAT MATTER

| File | What It Does | Touch With Care |
|------|--------------|-----------------|
| `run_max_mode.py` | **ENTRY POINT** - Aggressive scalping mode | YES |
| `wsb_snake/trading/alpaca_executor.py` | **EXECUTES ALL TRADES** - Position lifecycle | CRITICAL |
| `wsb_snake/trading/risk_governor.py` | **RISK CONTROLS** - Kill switch, limits | CRITICAL |
| `wsb_snake/execution/apex_conviction_engine.py` | **SIGNAL FUSION** - Multi-signal scoring | YES |
| `wsb_snake/collectors/polygon_enhanced.py` | **MARKET DATA** - Price, technicals | YES |
| `wsb_snake/notifications/telegram_bot.py` | **ALERTS** - All notifications | NO |

Everything else is supporting infrastructure. These 6 files control 90% of behavior.

---

## CRITICAL NUMBERS - MEMORIZE THESE

```
MAX_PER_TRADE      = $1,000   (NOT $50,000 - that was a bug)
MAX_DAILY_EXPOSURE = $4,000   (daily limit including margin)
MAX_CONCURRENT     = 3        (positions at once)
KILL_SWITCH        = -$200    (daily loss limit, then HALT)
TRADE_THRESHOLD    = 68%      (minimum conviction to trade)
TARGET             = +6%      (take profit)
STOP               = -10%     (initial stop loss)
MAX_HOLD           = 5 min    (time decay exit)
```

If you see different numbers in the code, THE CODE IS WRONG. These are the JP Morgan grade settings.

---

## HOW TRADES ACTUALLY HAPPEN

```
Signal Detection (spy_scalper / apex_engine)
    │
    ▼
Conviction Score (0-100%)
    │
    ▼
≥68%? ──NO──> SKIP (not high enough conviction)
    │
   YES
    ▼
risk_governor.can_trade()
    │
    ▼
BLOCKED? ──YES──> SKIP (hit daily limit / kill switch)
    │
    NO
    ▼
alpaca_executor.execute_scalp_entry()
    │
    ├── Format OCC symbol (SPY260208C00600000)
    ├── Get quote from Alpaca
    ├── Calculate position size ($1k max)
    ├── POST order to Alpaca API
    ├── Create AlpacaPosition object
    └── Start monitoring thread
    │
    ▼
Background Monitor (every 5 seconds)
    │
    ├── Check if target hit (+6%) → execute_exit("TARGET HIT")
    ├── Check if stop hit (-10%) → execute_exit("STOP LOSS")
    ├── Check if time exceeded (5 min) → execute_exit("TIME DECAY")
    └── Apply trailing stop logic
    │
    ▼
Telegram Alert on every state change
```

---

## THE BUGS THAT WERE JUST FIXED (Feb 8, 2026)

These bugs existed. They are now fixed. DO NOT REINTRODUCE THEM.

### BUG 1: Direction Always "long"
**File:** `run_max_mode.py` line 230
**Was:** `direction="long"` hardcoded
**Now:** `direction = "long" if verdict.action == "BUY_CALLS" else "short"`
**Impact:** PUT signals were executing as CALLS. 100% inverted P&L.

### BUG 2: Symbol Format Mismatch
**File:** `alpaca_executor.py` line 865-874
**Was:** Using Polygon symbol format (7 digits)
**Now:** Always use `format_option_symbol()` (OCC 8 digits)
**Impact:** 404 errors on position close. Positions couldn't exit.

### BUG 3: MAX_PER_TRADE = $50,000
**File:** `alpaca_executor.py` line 158
**Was:** `MAX_PER_TRADE = 50000`
**Now:** `MAX_PER_TRADE = 1000`
**Impact:** Could deploy 50x intended capital in single trade.

### BUG 4: Kill Switch Too Loose
**File:** `risk_governor.py` line 59
**Was:** `max_daily_loss = -500.0`
**Now:** `max_daily_loss = -200.0`
**Impact:** Allowed 12.5% daily loss before halt. Now 5%.

### BUG 5: Conviction Threshold Too Low
**File:** `apex_conviction_engine.py` line 73
**Was:** `TRADE_THRESHOLD = 55`
**Now:** `TRADE_THRESHOLD = 68`
**Impact:** Trading coin-flip signals. No edge.

---

## OCC OPTION SYMBOL FORMAT

**THIS WILL BITE YOU.**

```
SPY260208C00600000
│   │     │ │
│   │     │ └── Strike × 1000, padded to 8 digits (600.00 = 00600000)
│   │     └──── C = Call, P = Put
│   └────────── YYMMDD (Feb 8, 2026 = 260208)
└────────────── Underlying ticker

CORRECT: SPY260208C00600000 (8 digits after C/P)
WRONG:   SPY260208C0060000  (7 digits - WILL CAUSE 404)
```

The `format_option_symbol()` method in `alpaca_executor.py` handles this. ALWAYS use it.

---

## API RATE LIMITS THAT WILL BREAK YOU

| API | Limit | What Happens |
|-----|-------|--------------|
| Polygon.io | 5 req/min | Returns stale cached data |
| Alpaca Trading | Generous | N/A |
| Telegram | 30 msg/sec | Unlikely to hit |
| OpenAI GPT-4 | Custom (200/day) | Falls back to Gemini |
| Reddit | 60 req/min | Returns 429, waits 30s |

The Polygon limit is the most common issue. The code caches for 120 seconds.

---

## ENVIRONMENT VARIABLES - REQUIRED

```bash
# TRADING (REQUIRED)
ALPACA_API_KEY=xxx
ALPACA_SECRET_KEY=xxx
ALPACA_LIVE_TRADING=false  # KEEP FALSE UNLESS PRODUCTION

# NOTIFICATIONS (REQUIRED)
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx

# DATA (REQUIRED)
POLYGON_API_KEY=xxx

# AI (OPTIONAL - for chart analysis)
OPENAI_API_KEY=xxx
```

Without these, the system WILL crash on startup.

---

## THE POSITION LIFECYCLE

```
PENDING  →  Order submitted, waiting for fill
    │
    ▼
OPEN     →  Order filled, monitoring for exit
    │
    ├── Target hit → CLOSED (profit)
    ├── Stop hit → CLOSED (loss)
    ├── Time exceeded → CLOSED (decay)
    └── 3:55 PM ET → CLOSED (EOD mandatory)
    │
    ▼
CLOSED   →  P&L recorded, position removed from tracking
```

**CRITICAL:** If the system crashes, positions are NOT automatically tracked on restart. The `sync_existing_positions()` method must be called to restore tracking. This happens automatically in `start_monitoring()`.

---

## TRAILING STOP LOGIC

```python
profit_pct = (current_price - entry_price) / entry_price

if profit_pct >= 0.05:   # +5% profit
    stop = entry_price * 1.03   # Lock in +3%
elif profit_pct >= 0.03: # +3% profit
    stop = entry_price * 1.00   # Move to breakeven
elif profit_pct >= 0.02: # +2% profit
    stop = entry_price * 0.95   # Reduce to -5%
```

The stop only moves UP, never down. Winners are protected progressively.

---

## SIGNAL ENGINES (6 TOTAL)

| Engine | Weight | What It Measures |
|--------|--------|------------------|
| Technical | 20% | RSI, MACD, SMA, EMA |
| Candlestick | 15% | 36 patterns + confluence |
| Order Flow | 20% | Sweeps, blocks, institutional |
| Probability | 20% | Multi-engine fusion (ignition/pressure/surge) |
| Pattern Memory | 15% | Historical pattern matching |
| AI Verdict | 10% | GPT-4 vision chart analysis |

Final score = weighted average. Trade if ≥68%.

---

## SESSION TIMING (EASTERN TIME)

| Session | Time | Signal Multiplier |
|---------|------|-------------------|
| Premarket | 4:00-9:30 AM | 0.5x (low confidence) |
| Open | 9:30-10:30 AM | 1.0x (high volatility) |
| Morning | 10:30 AM-12:00 PM | 0.9x |
| Lunch | 12:00-1:00 PM | 0.5x (CHOP ZONE - AVOID) |
| Power Hour Early | 1:00-3:00 PM | 1.2x |
| Power Hour Final | 3:00-4:00 PM | 1.5x (PRIME 0DTE TIME) |
| After Hours | 4:00-8:00 PM | 0.3x |

**3:55 PM ET:** ALL 0DTE positions force-closed. Non-negotiable.

---

## COMMON FAILURE MODES

### 1. "LIQUIDITY_REJECT" in logs
**Cause:** Option mid-price < $0.05 or > $6.00
**Fix:** This is intentional filtering. Illiquid options are skipped.

### 2. "404 Not Found" on position close
**Cause:** Symbol format mismatch (7 vs 8 digits)
**Fix:** Ensure `format_option_symbol()` is used everywhere

### 3. "Risk governor blocked trade"
**Cause:** Hit daily limit, kill switch, or max positions
**Fix:** Check `risk_governor.py` limits. Wait for next day.

### 4. No trades executing
**Cause:** Conviction threshold not met (68%)
**Fix:** Check apex_conviction_engine thresholds. Or wait for better setup.

### 5. Trades not closing at target
**Cause:** Monitor thread not running
**Fix:** Ensure `start_monitoring()` called after `execute_scalp_entry()`

---

## VM DEPLOYMENT (Digital Ocean)

```bash
# SSH to VM
ssh root@157.245.240.99

# Pull latest code
cd /root/wsb-snake && git fetch origin && git reset --hard origin/main

# Restart service
systemctl restart wsb-snake

# Check status
systemctl status wsb-snake --no-pager

# View logs
journalctl -u wsb-snake -f
```

The service auto-starts on boot via systemd.

---

## TESTING CHECKLIST

Before any deployment:

- [ ] `MAX_PER_TRADE = 1000` (not 50000)
- [ ] `MAX_DAILY_EXPOSURE = 4000` (not 100000)
- [ ] `TRADE_THRESHOLD = 68` (not 55)
- [ ] `ALPACA_LIVE_TRADING = false` (unless production)
- [ ] Direction logic handles both CALLS and PUTS
- [ ] Symbol format uses `format_option_symbol()`
- [ ] Telegram alerts configured
- [ ] Position monitoring auto-starts

---

## WHAT THE USER EXPECTS

1. **Trade only high-conviction setups** (no gambling on 55% signals)
2. **Risk max $1,000 per trade** (not $50k)
3. **Stop at -$200 daily loss** (capital preservation)
4. **Auto-exit at target/stop** (no babysitting)
5. **Close all 0DTE before expiration** (no worthless expiry)
6. **Telegram alerts on everything** (real-time visibility)

If any of these break, you will hear about it.

---

## THE PHILOSOPHY

```
PREDATOR MODE:
- Only strike on A+ setups (68%+ conviction)
- Enter fast, exit faster
- Never let winners become losers (trailing stop)
- Book profit, hunt again
- No mercy on exits
- Quality over quantity
```

This is not a day trading system. This is a sniper system. Few shots, high accuracy.

---

## FINAL WORDS

This system trades real money (paper by default, live with env var). Every line of code you touch can cause financial loss.

The Feb 6 session lost $388 because:
- Stops were too tight (-8% with 5% bid-ask spread = instant stop)
- Threshold was too low (55% = coin flip)
- Position sizes were wrong ($50k bug)

These are now fixed. DO NOT REGRESS.

When in doubt:
1. Check the risk governor limits
2. Check the symbol format
3. Check the direction logic
4. Test on paper trading first

Good luck. Don't fuck it up.

---

*Last Updated: Feb 8, 2026*
*Commit: 97fa927 - JP MORGAN GRADE SYSTEM AUDIT - 12 Critical Bug Fixes*
