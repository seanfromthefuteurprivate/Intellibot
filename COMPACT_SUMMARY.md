# WSB Snake - Compact Summary

**Date:** March 6, 2026
**Status:** OPERATIONAL

---

## TL;DR

System was dead for 4 days. Beast Mode V4.0's 13-signal conviction needed 5 signals to trade, but HYDRA only provides 3 working fields. Mathematically impossible.

**Fix:** Replaced with V5 Minimal (5 signals, MIN_CONVICTION=1). System now generating signals.

---

## Current Config

```
Engine: V5 Minimal
Conviction: 1/5 minimum (GEX_NEG only signal reliably firing)
V7 Scalper: Disabled
Monitor: Disabled
```

---

## Data Sources

| Source | Health | Notes |
|--------|--------|-------|
| HYDRA | 12.5% | Only GEX regime, blowup, charm work |
| Polygon | Limited | 1-2 bars, DELAYED status |
| Alpaca | 100% | No issues |

---

## What's Working

- CPL scans running every minute
- Conviction scoring operational
- Trade signals broadcasting
- First signal: SPY CALL 673 at 18:03 UTC

---

## What's Broken

1. **HYDRA Layers 9-11** - Flow, Dark Pool, Sequence all return zeros/nulls
2. **Polygon caching** - Health monitor caches empty responses (workaround in place)
3. **Monitor service** - Disabled to stop spam

---

## Files Changed (March 6)

- `wsb_snake/execution/jobs_day_cpl.py` - V5 conviction system
- `wsb_snake/collectors/polygon_enhanced.py` - Direct API fallback
- `ops/MARCH_6_POSTMORTEM.md` - Full post-mortem

---

## Quick Check

```bash
# Is it working?
journalctl -u wsb-snake --since "5 min ago" | grep "CONV_APPROVED\|CPL BUY"

# Expected output:
# CONV_APPROVED: SPY CALL 1/5 conf=43% [GEX_NEG]
# CPL BUY broadcast #1: SPY CALL 673
```

---

## Next Steps

1. Fix HYDRA (get all 4 layers working)
2. Raise MIN_CONVICTION to 2+ when data improves
3. Re-enable monitor with correct table checks

---

**Full details:** `ops/MARCH_6_POSTMORTEM.md`
