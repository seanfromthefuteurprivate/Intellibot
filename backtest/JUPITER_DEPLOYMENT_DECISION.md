# Jupiter Direct Gap Fade — Deployment Decision

## The Truth (Verified with Real Alpaca Data)

### Claimed vs Reality
| Metric | Original Claim | Verified Reality |
|--------|----------------|------------------|
| Win Rate | 50% | **25%** |
| Total P&L (8 trades) | +391.6% | **-18.9%** |
| COIN Mar 4 | +92% | **-40%** |
| MARA Mar 4 | +93% | **-40%** |

### Why The Discrepancy?
The original analysis used **wrong stock prices**:
- COIN Mar 4: Claimed low was $180.30 (gap fill)
- COIN Mar 4: Actual low was $195.40 (gap CONTINUED UP)

The gaps did NOT fade in 6 of 8 trades. They continued.

### Individual Trade Breakdown (Real Data)
| Date | Ticker | Gap | Type | P&L | Exit | Outcome |
|------|--------|-----|------|-----|------|---------|
| 2026-02-13 | COIN | +8.9% | PUT | -40.0% | STOP | ❌ Gap continued |
| 2026-02-19 | SMCI | +5.2% | PUT | -40.0% | STOP | ❌ Gap continued |
| 2026-02-24 | AMD | +7.6% | PUT | -40.0% | STOP | ❌ Gap continued |
| 2026-02-25 | COIN | +6.0% | PUT | -40.0% | STOP | ❌ Gap continued |
| **2026-02-27** | **MARA** | **+14.6%** | **PUT** | **+146.6%** | **EOD** | ✅ **Gap faded** |
| **2026-03-03** | **HOOD** | **-6.0%** | **CALL** | **+74.4%** | **EOD** | ✅ **Gap faded** |
| 2026-03-04 | COIN | +7.4% | PUT | -40.0% | STOP | ❌ Gap continued |
| 2026-03-04 | MARA | +5.2% | PUT | -40.0% | STOP | ❌ Gap continued |

---

## The Math

### Asymmetric Risk/Reward
- Average Win: **+110.5%**
- Average Loss: **-40.0%**
- Win/Loss Ratio: **2.76x**

### Expected Value
| Win Rate | EV per Trade | After 10 Trades |
|----------|--------------|-----------------|
| 25% (observed) | -2.4% | -24% |
| 30% | +9.2% | +92% |
| 40% | +24.2% | +242% |
| 50% | +35.3% | +353% |

### Breakeven Win Rate: 26.6%

---

## What We Learned

1. **Gap UP does NOT mean gap fade** — 6 of 8 gap-up trades continued UP
2. **Big gaps work better** — MARA +14.6% gap faded successfully
3. **Gap DOWN may work better** — HOOD -6.0% gap filled
4. **The 15x leverage assumption was wrong** — Real option returns differ significantly

---

## Deployment Decision

### ❌ DO NOT DEPLOY (as-is)

The raw gap fade strategy lost money (-18.9%) on real data.

### 🔧 POTENTIAL IMPROVEMENTS

1. **Larger gap threshold**: Only trade gaps >10% (MARA +14.6% was the only gap-up winner)
2. **Focus on gap-downs**: HOOD -6.0% gap-down worked perfectly
3. **Add VIX filter**: High VIX days may fade better (uncertainty = profit-taking)
4. **Early price action filter**: Skip if gap continues in first 15 minutes

### 📊 Required Win Rate

Need **27%+ win rate** to break even. Currently at **25%**.

If filters can improve win rate to 35%, strategy becomes viable:
- EV at 35%: +14.6% per trade
- After 10 trades: +146%

---

## Next Steps

1. Backtest larger gap threshold (>10%)
2. Backtest gap-DOWN only
3. Add 15-minute price action filter
4. Paper trade for 2 weeks before live deployment

---

*Generated 2026-03-15 with verified Alpaca market data*
