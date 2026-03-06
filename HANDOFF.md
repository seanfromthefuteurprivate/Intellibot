# WSB Snake Handoff Document

**Last Updated:** March 6, 2026 18:15 UTC
**System Status:** OPERATIONAL (V5 Minimal)
**Author:** Claude Opus 4.5

---

## Current Deployment State

### Engine Configuration
| Component | Status | Notes |
|-----------|--------|-------|
| **CPL Engine** | ACTIVE | V5 Minimal conviction system |
| **V7 Scalper** | DISABLED | Hard return at line 234 |
| **Beast Mode V4.0** | REPLACED | 13-signal system removed |
| **wsb-ops-monitor** | DISABLED | Stopped spam, needs rebuild |

### Conviction System: V5 Minimal
```
MIN_CONVICTION: 1 (temporary - raise when data improves)
Signals: 5 total
  1. GEX regime negative (HYDRA) - WORKING
  2. Volume ratio > 1.2x (Polygon) - needs 3+ bars
  3. Momentum aligned (Polygon) - needs 2+ bars
  4. Time window optimal (System) - 9:35-10:30 or 14:30-15:45 ET
  5. Charm flow favorable (HYDRA) - afternoon only
```

### Data Source Health
| Source | Status | Issues |
|--------|--------|--------|
| HYDRA | 2/4 components | Layers 9-11 dead (Flow, Dark Pool, Sequence) |
| Polygon | DELAYED | Returns 1-2 bars, rate limited |
| Alpaca | Healthy | No issues |

---

## What Happened This Week

### March 2-5: System Dead
- Beast Mode V4.0 deployed with 13-signal conviction
- HYDRA returned garbage for 21/24 fields
- Polygon returned insufficient bars
- Hard gates blocked everything
- 1400+ Dead Man's Switch alerts

### March 6: System Restored
- Diagnosed root cause: conviction system mathematically impossible
- Replaced with V5 Minimal using only working data
- Added Polygon direct API fallback
- Lowered MIN_CONVICTION to 1
- First signal at 18:03 UTC: SPY CALL 673

---

## Critical Files

| File | Purpose | Last Modified |
|------|---------|---------------|
| `wsb_snake/execution/jobs_day_cpl.py` | CPL engine with V5 conviction | March 6 |
| `wsb_snake/collectors/polygon_enhanced.py` | Polygon adapter with fallback | March 6 |
| `wsb_snake/collectors/hydra_bridge.py` | HYDRA API client | March 4 |
| `ops/MARCH_6_POSTMORTEM.md` | Detailed post-mortem | March 6 |

---

## Known Issues (Must Fix)

### 1. HYDRA Intelligence Broken (HIGH)
- Only 3/24 fields return real data
- Layers 9, 10, 11 completely non-functional
- Direction always NEUTRAL, regime always UNKNOWN

### 2. Polygon Rate Limiting (MEDIUM)
- Health monitor caches empty responses
- Direct API fallback added as workaround
- Consider upgrading from Starter plan

### 3. MIN_CONVICTION Too Low (MEDIUM)
- Currently set to 1 (only GEX signal needed)
- Should be 2-3 for quality trades
- Blocked by data quality issues

### 4. Monitor Disabled (LOW)
- wsb-ops-monitor service disabled
- Needs rebuild to check correct tables
- Should add rate limiting

---

## Quick Commands

```bash
# Check service status
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl status wsb-snake --no-pager"]' \
  --region us-east-1

# View recent logs
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["journalctl -u wsb-snake --no-pager -n 50"]' \
  --region us-east-1

# Check CPL activity
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["journalctl -u wsb-snake --no-pager --since \"5 min ago\" | grep -E \"CONV_|CPL run|CPL BUY\""]' \
  --region us-east-1

# Test HYDRA
curl http://54.172.22.157:8000/api/predator | python3 -m json.tool

# Test Polygon
curl "https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/$(date +%Y-%m-%d)/$(date +%Y-%m-%d)?limit=8&apiKey=YOUR_KEY"
```

---

## Weekend TODO

1. **Fix HYDRA** - Get Layers 9-11 working
2. **Test data quality** - Verify HYDRA/Polygon return usable data
3. **Raise MIN_CONVICTION** - Back to 2-3 when data improves
4. **Rebuild monitor** - Check cpl_calls table, add rate limiting
5. **Integration tests** - Test against real API responses before deploy

---

## Contacts

- **EC2 Instance:** i-03f3a7c46ec809a43
- **HYDRA API:** http://54.172.22.157:8000/api/predator
- **Region:** us-east-1

---

**Next Engineer:** Read `ops/MARCH_6_POSTMORTEM.md` for full context on what broke and why.
