# WSB-Snake Handoff Document
## Last Updated: March 6, 2026 @ 2:00 PM EST

---

## CRITICAL STATUS

| System | State | Notes |
|--------|-------|-------|
| **WSB-Snake** | STOPPED | Disabled via systemctl |
| **Second Brain** | ONLINE | http://98.82.24.119:8080 |
| **HYDRA** | DEGRADED | 50% fields working |
| **Open Positions** | NONE | All closed |

---

## This Week: -$3,939 Lost

| Day | P&L | Root Cause |
|-----|-----|------------|
| Mon | -$1,172 | CPL no filters, gave back QQQ winner |
| Tue | -$615 | Position cap race condition |
| Wed | -$462 | Polygon API 403 errors |
| Thu | $0 | Beast Mode gates blocked everything |
| Fri | -$1,656 | DIA/VXX traded (not in watchlist) |

**One Winner:** QQQ $609C +$1,551 (March 2, 12:13 PM)

---

## Second Brain Deployment

### Instance Details
- **Instance ID:** i-04b0f930bd1e371c1
- **Public IP:** 98.82.24.119
- **API Port:** 8080
- **AI Model:** Claude Haiku 4.5 via Bedrock

### API Endpoints
```bash
# Health check
curl http://98.82.24.119:8080/api/health

# Get full context (after compaction)
curl http://98.82.24.119:8080/api/context

# Check HYDRA health
curl http://98.82.24.119:8080/api/hydra/health

# Report a trade
curl -X POST http://98.82.24.119:8080/api/report/trade \
  -H "Content-Type: application/json" \
  -d '{"ticker":"SPY","side":"CALL","pnl":500}'

# Report a bug
curl -X POST http://98.82.24.119:8080/api/report/bug \
  -H "Content-Type: application/json" \
  -d '{"id":"bug_007","title":"Description","severity":"critical"}'

# Ask the brain (AI-powered with context)
curl -X POST http://98.82.24.119:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Why did the system fail today?"}'
```

---

## Root Causes Identified

1. **NO PERSISTENT MEMORY** - Claude Code compacts and loses context
2. **NO END-TO-END TESTING** - Changes deployed directly to production
3. **HYDRA DATA QUALITY** - 50% of signals return garbage
4. **POLYGON DATA QUALITY** - Delayed, 1 bar, rate limited
5. **WATCHLIST NOT ENFORCED** - DIA/VXX appeared despite SPY/QQQ only
6. **NO EXTERNAL OVERSIGHT** - 1,400+ alerts but no intelligent response

---

## What Works
- Execution layer (pyramids, trailing stops)
- Kill switch (when configured correctly)
- Cooldown (prevents race conditions)
- Telegram alerts (when not spamming)
- CPL signal discovery (found the QQQ winner)
- Second Brain memory persistence

---

## Weekend Plan

### Saturday (DONE)
- [x] Stop wsb-snake service
- [x] Close all positions
- [x] Write WEEK_1_POSTMORTEM.md
- [x] Deploy Second Brain EC2
- [x] Verify all API endpoints

### Sunday (TODO)
- [ ] Rebuild CPL V6 with hard-coded watchlist (SPY, QQQ ONLY)
- [ ] Test in simulation with real data
- [ ] Verify kill switch and cooldown
- [ ] Deploy to EC2 (disabled)

### Monday (TODO)
- [ ] Pre-market: Run /api/health checks
- [ ] 9:30 AM: Enable wsb-snake
- [ ] Monitor via Second Brain
- [ ] Report every trade to Second Brain
- [ ] Post-market: Daily post-mortem

---

## EC2 Instances

| Instance | ID | IP | Purpose |
|----------|----|----|---------|
| WSB-Snake | i-03f3a7c46ec809a43 | 54.172.22.157 | Trading engine (STOPPED) |
| Second Brain | i-04b0f930bd1e371c1 | 98.82.24.119 | AI command center (ONLINE) |

---

## Commands for Next Session

```bash
# Check Second Brain
curl http://98.82.24.119:8080/api/health

# Get full context
curl http://98.82.24.119:8080/api/context | jq .

# Check wsb-snake status
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl is-active wsb-snake"]' \
  --region us-east-1

# Enable wsb-snake (Monday)
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl enable wsb-snake && systemctl start wsb-snake"]' \
  --region us-east-1

# View logs
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["journalctl -u wsb-snake --no-pager -n 50"]' \
  --region us-east-1
```

---

## The Mission

**$2,500 capital -> multiply daily via ONE lethal 0DTE trade.**

Three things:
1. PICK the right trade (conviction stacking)
2. EXECUTE it perfectly (pyramid + trailing stop)
3. STOP after one trade (kill switch + cooldown)

Win 3 out of 5 days. Dont lose big on the other 2.

---

## Signed Off

**Week 1 Post-Mortem Complete. Second Brain Deployed.**

Updated: March 6, 2026 14:00 EST
Engineer: Claude Opus 4.5
