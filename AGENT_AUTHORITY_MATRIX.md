# WSB Snake - Agent Authority Matrix

## Power Limits & Decision Authority

This document defines what each component of the system is authorized to do.

---

## Authority Levels

| Level | Description | Examples |
|-------|-------------|----------|
| **L0** | Read-only observation | View prices, check status |
| **L1** | Alert/notify | Send Telegram messages |
| **L2** | Recommend action | Suggest trades, flag setups |
| **L3** | Execute within limits | Place orders up to limits |
| **L4** | Execute without limits | Emergency close all |
| **L5** | Configure system | Change thresholds, limits |

---

## Component Authority

### SPY Scalper Engine

| Action | Authority | Limits |
|--------|-----------|--------|
| Scan tickers | L0 | All 29 tickers |
| Detect patterns | L0 | Unlimited |
| Calculate confidence | L0 | 0-100% |
| Request AI analysis | L1 | Rate limited |
| Send alerts | L1 | Must meet 60% threshold |
| Trigger auto-execute | L2 | Recommends only |
| Execute trades | L3 | Via Alpaca Executor |

**CANNOT:**
- Bypass confidence thresholds
- Execute without AI confirmation (for auto)
- Override position limits

---

### Predator Stack (AI)

| Action | Authority | Limits |
|--------|-----------|--------|
| Analyze charts | L0 | 50 calls/day OpenAI |
| Analyze news | L0 | 200 calls/day DeepSeek |
| Return verdict | L2 | STRIKE_CALLS/PUTS/NO_TRADE/ABORT |
| Adjust confidence | L2 | +/- 25% max |

**CANNOT:**
- Execute trades directly
- Exceed daily budget ($5)
- Bypass rate limits (10/min, 60/hour)

---

### Alpaca Executor

| Action | Authority | Limits |
|--------|-----------|--------|
| Check account | L0 | Unlimited |
| Get positions | L0 | Unlimited |
| Get quotes | L0 | Rate limited by Alpaca |
| Place orders | L3 | $1,500/trade, $6,000/day |
| Monitor positions | L0 | Every 5 seconds |
| Close positions | L3 | At target/stop/time |
| Emergency close all | L4 | EOD or manual trigger |
| Reject oversized | L4 | Auto-close if > 150% limit |

**CANNOT:**
- Trade real money (paper only)
- Exceed 5 concurrent positions
- Hold past 3:55 PM ET

---

### Zero Greed Exit

| Action | Authority | Limits |
|--------|-----------|--------|
| Track positions | L0 | Up to 20 positions |
| Calculate exit levels | L0 | Based on entry |
| Send exit alerts | L1 | At target/stop/time |
| Trigger close | L3 | Via Alpaca Executor |

**CANNOT:**
- Override exit rules
- Extend hold time
- Skip stop losses

---

### Telegram Bot

| Action | Authority | Limits |
|--------|-----------|--------|
| Send alerts | L1 | Unlimited |
| Receive commands | L0 | Read only |

**CANNOT:**
- Execute trades
- Change configuration
- Access sensitive data

---

### Learning Modules

| Module | Authority | Limits |
|--------|-----------|--------|
| Pattern Memory | L0 | Read patterns, boost confidence +15% max |
| Time Learning | L0 | Read performance, boost +10% max |
| Session Learnings | L5 | Update battle plan parameters |
| Stalking Mode | L1 | Track setups, send alerts |

**CANNOT:**
- Execute trades directly
- Override AI decisions
- Bypass safety limits

---

## Decision Flow Authority

```
┌─────────────────────────────────────────────────────────────────┐
│                    DECISION AUTHORITY FLOW                       │
│                                                                  │
│  Pattern Detected                                                │
│       │                                                          │
│       ▼                                                          │
│  [L0] Calculate base confidence                                  │
│       │                                                          │
│       ▼                                                          │
│  [L0] Apply learning boosts (max +25%)                          │
│       │                                                          │
│       ▼                                                          │
│  [L0] AI Analysis (rate limited)                                │
│       │                                                          │
│       ▼                                                          │
│  [L2] AI returns verdict (+/- 25% confidence)                   │
│       │                                                          │
│       ▼                                                          │
│  ┌────┴────┐                                                     │
│  │ >= 60%? │                                                     │
│  └────┬────┘                                                     │
│       │ YES                                                       │
│       ▼                                                          │
│  [L1] Send Telegram alert                                        │
│       │                                                          │
│       ▼                                                          │
│  ┌────┴────────────────┐                                         │
│  │ >= 70% AND AI conf? │                                         │
│  └────┬────────────────┘                                         │
│       │ YES                                                       │
│       ▼                                                          │
│  [L3] Execute trade (within limits)                              │
│       │                                                          │
│       ▼                                                          │
│  [L0] Monitor position (every 5s)                                │
│       │                                                          │
│       ▼                                                          │
│  ┌────┴─────────────────────┐                                    │
│  │ Target/Stop/Time reached? │                                   │
│  └────┬─────────────────────┘                                    │
│       │ YES                                                       │
│       ▼                                                          │
│  [L3] Execute exit                                               │
│       │                                                          │
│       ▼                                                          │
│  [L1] Send exit alert                                            │
│       │                                                          │
│       ▼                                                          │
│  [L0] Record outcome for learning                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Override Authority

### Human Operator (L5)

Can override:
- All thresholds and limits
- Stop/close any position
- Disable auto-execution
- Restart/stop system

### System (L4)

Can override:
- Individual trades (emergency close)
- EOD close (mandatory)
- Oversized position close

### No One Can Override:
- Paper trading mode (hardcoded)
- Maximum position limits (code enforced)
- Rate limits (API enforced)

---

## Audit Trail

All authority exercises are logged:

```
2026-01-28 15:30:05 | L1 | spy_scalper | Alert sent: SPY VWAP_RECLAIM 78%
2026-01-28 15:30:06 | L3 | alpaca_executor | Order placed: SPY260128C00602000 2x
2026-01-28 15:35:15 | L3 | alpaca_executor | Position closed: TARGET_HIT +$60
```
