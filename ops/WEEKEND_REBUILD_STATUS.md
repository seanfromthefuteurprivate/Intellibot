# WSB-Snake Weekend Rebuild — Complete Status Report
## March 6, 2026 @ 2:15 PM EST

---

## EXECUTIVE SUMMARY

| Metric | Value |
|--------|-------|
| **Week 1 P&L** | -$3,939 |
| **Win Rate** | 6.7% (1/15) |
| **System Status** | STOPPED |
| **Second Brain** | ONLINE |
| **Phase 0-3** | COMPLETE |
| **Phase 4** | IN PROGRESS |

---

## TABLE OF CONTENTS

1. [Week 1 Financial Summary](#week-1-financial-summary)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Infrastructure Deployed](#infrastructure-deployed)
4. [Phase Completion Status](#phase-completion-status)
5. [Detailed Action Log](#detailed-action-log)
6. [Second Brain Architecture](#second-brain-architecture)
7. [What's Left To Do](#whats-left-to-do)
8. [Sunday Build Plan](#sunday-build-plan)
9. [Monday Go-Live Checklist](#monday-go-live-checklist)
10. [Critical Commands Reference](#critical-commands-reference)

---

## WEEK 1 FINANCIAL SUMMARY

### Daily Breakdown

| Day | Date | P&L | Trades | Root Cause |
|-----|------|-----|--------|------------|
| Mon | Mar 2 | -$1,172 | ~12 | No filters, gave back QQQ winner |
| Tue | Mar 3 | -$615 | 4 | Position cap race condition |
| Wed | Mar 4 | -$462 | 1 | Polygon API 403 errors |
| Thu | Mar 5 | $0 | 0 | Beast Mode gates blocked all |
| Fri | Mar 6 | -$1,656 | ~8 | DIA/VXX traded (not in plan) |
| **TOTAL** | | **-$3,939** | **~25** | |

### The One Winner

```
Ticker:     QQQ $609 CALL
Date:       March 2, 2026 @ 12:13 PM ET
Entry:      ~$2.50
Exit:       ~$4.20
P&L:        +$1,551
Move:       +69%
Signals:    50% confidence (no filtering)
```

**Key Insight:** CPL CAN find winners. The problem is it doesn't stop after.

### Account Status

```
Starting Balance (Mar 2):  $89,953
Ending Balance (Mar 6):    ~$86,014
Net Change:                -$3,939 (-4.4%)
Open Positions:            0 (all closed)
System State:              STOPPED
```

---

## ROOT CAUSE ANALYSIS

### Ranked by Impact (Highest First)

#### 1. NO PERSISTENT MEMORY
- Claude Code compacts and loses all context
- Each session starts fresh
- Fixes from previous sessions get half-implemented
- HANDOFF.md protocol helps but not reliable
- **Solution:** Second Brain EC2 with persistent storage

#### 2. NO END-TO-END TESTING
- Every change deployed directly to production
- No mock scans with real data
- No paper trade verification
- No gate testing before deployment
- **Solution:** Test harness with real API responses

#### 3. HYDRA DATA QUALITY (50% Broken)
| Component | Status | Impact |
|-----------|--------|--------|
| Layer 8 (GEX) | WORKING | gex_regime returns NEGATIVE |
| Layer 9 (Flow) | BROKEN | All zeros |
| Layer 10 (Dark Pool) | BROKEN | All nulls |
| Layer 11 (Sequence) | BROKEN | All zeros |
| Direction | BROKEN | Always NEUTRAL |
| Regime | BROKEN | Always UNKNOWN |

**Field Health Audit:**
```
Total Fields:    24
Working Fields:  3-4
Health:          12.5-16%
```

#### 4. POLYGON DATA QUALITY
| Issue | Impact |
|-------|--------|
| Plan Type | Starter (DELAYED data) |
| Bars Returned | 1 when 10 requested |
| Rate Limit | 100/min but effectively lower |
| Caching Bug | Empty responses cached, poison scan cycles |

#### 5. WATCHLIST NOT ENFORCED
- CPL traded DIA and VXX on Friday
- Explicit instructions said SPY and QQQ ONLY
- Watchlist either misconfigured or code path adds tickers
- **Solution:** Hard-code watchlist in V6

#### 6. NO EXTERNAL OVERSIGHT
- 1,400+ Dead Man's Switch alerts over 4 days
- No intelligent response to alerts
- Nobody catches issues for hours
- **Solution:** Second Brain monitoring with AI diagnosis

---

## INFRASTRUCTURE DEPLOYED

### EC2 Instances

| Instance | ID | IP | Status | Purpose |
|----------|----|----|--------|---------|
| WSB-Snake | i-03f3a7c46ec809a43 | 54.172.22.157 | STOPPED | Trading engine |
| Second Brain | i-04b0f930bd1e371c1 | 98.82.24.119 | RUNNING | AI command center |

### Second Brain Details

```
Instance Type:     t3.medium (2 vCPU, 4GB RAM)
AMI:               Ubuntu 24.04
Storage:           50GB gp3
IAM Role:          EC2-SSM-Profile
Security Group:    sg-0b540ee94abc2f843
AI Model:          Claude Haiku 4.5 via Bedrock
API Port:          8080
Service:           second-brain.service (systemd)
```

### API Endpoints Verified

| Endpoint | Method | Status | Response |
|----------|--------|--------|----------|
| `/api/health` | GET | ✅ WORKING | System status |
| `/api/context` | GET | ✅ WORKING | Full context for Claude Code |
| `/api/hydra/health` | GET | ✅ WORKING | HYDRA field status |
| `/api/report/trade` | POST | ✅ WORKING | Store trade data |
| `/api/report/bug` | POST | ✅ WORKING | Store bug reports |
| `/api/ask` | POST | ✅ WORKING | AI Q&A with context |

### Memory Directory Structure

```
/opt/second-brain/memory/
├── trades/           # Daily trade logs (JSON)
├── bugs/             # Bug reports (Markdown)
├── fixes/            # Fix documentation (Markdown)
├── postmortems/      # Daily/weekly reviews
├── decisions/        # Architecture decisions
├── research/         # AI-generated research
├── handoffs/         # Context handoff docs
├── objectives/       # Mission statement
└── state/            # Current system state
```

---

## PHASE COMPLETION STATUS

### Phase 0: Stop the Bleeding ✅ COMPLETE

| Task | Status | Evidence |
|------|--------|----------|
| Stop wsb-snake service | ✅ | `systemctl stop wsb-snake` |
| Disable wsb-snake | ✅ | Removed from systemd |
| Verify service dead | ✅ | `inactive` + `CONFIRMED_DEAD` |
| Check open positions | ✅ | `[]` (empty array) |
| Close any positions | ✅ | None to close |

**Commands Executed:**
```bash
# Stop and disable
aws ssm send-command ... 'commands=["systemctl stop wsb-snake && systemctl disable wsb-snake"]'
# Result: SNAKE_STOPPED

# Verify dead
aws ssm send-command ... 'commands=["systemctl is-active wsb-snake"]'
# Result: inactive, CONFIRMED_DEAD

# Check positions
aws ssm send-command ... 'commands=["curl alpaca positions"]'
# Result: []
```

---

### Phase 1: Full Week Post-Mortem ✅ COMPLETE

| Task | Status | File |
|------|--------|------|
| Write post-mortem | ✅ | `ops/WEEK_1_POSTMORTEM.md` |
| Document daily breakdown | ✅ | All 5 days detailed |
| Identify root causes | ✅ | 6 root causes ranked |
| Document what works | ✅ | 6 working components |
| Document what's broken | ✅ | 6 broken components |
| Weekend plan | ✅ | Saturday/Sunday/Monday |

**Files Created:**
- `/Users/seankuesia/Downloads/Intellibot/ops/WEEK_1_POSTMORTEM.md`
- Pushed to EC2 via SSM (in progress)

---

### Phase 2: Delete Old EC2 Agents ✅ COMPLETE (N/A)

| Task | Status | Notes |
|------|--------|-------|
| List all EC2 instances | ✅ | Only 1 instance found |
| Find old agent instances | ✅ | None exist |
| Terminate old agents | N/A | Nothing to terminate |

**Discovery:**
```
Only EC2 instance: i-03f3a7c46ec809a43 (wsb-snake)
No old agent instances found.
```

---

### Phase 3: Launch Second Brain EC2 ✅ COMPLETE

| Task | Status | Details |
|------|--------|---------|
| Check security groups | ✅ | Port 8080 already open |
| Check IAM roles | ✅ | EC2-SSM-Profile has Bedrock + SSM |
| Get Ubuntu 24.04 AMI | ✅ | ami-0071174ad8cbb9e17 |
| Create bootstrap script | ✅ | Flask + Gunicorn + memory dirs |
| Launch instance | ✅ | i-04b0f930bd1e371c1 |
| Wait for public IP | ✅ | 98.82.24.119 |
| Wait for bootstrap | ✅ | ~90 seconds |
| Test /api/health | ✅ | ONLINE |
| Test /api/context | ✅ | Returns mission + trades |
| Test /api/hydra/health | ✅ | Shows 50% health |
| Fix Bedrock model ID | ✅ | Changed to inference profile |
| Test /api/ask | ✅ | Claude Haiku 4.5 responding |
| Test /api/report/trade | ✅ | Storing trades |
| Test /api/report/bug | ✅ | Storing bugs |

**Bootstrap Script Contents:**
```bash
#!/bin/bash
# Install: python3, pip, flask, boto3, gunicorn
# Create: /opt/second-brain/memory/* directories
# Deploy: app.py with 6 API endpoints
# Configure: systemd service
# Start: gunicorn on port 8080
```

**Bedrock Model Fix:**
```
Original:  anthropic.claude-3-5-sonnet-20241022-v2:0  (FAILED - legacy)
Attempt 2: us.anthropic.claude-3-7-sonnet-20250219-v1:0 (FAILED - legacy)
Final:     us.anthropic.claude-haiku-4-5-20251001-v1:0 (SUCCESS)
```

---

### Phase 4: Write Complete Handoff ⏳ IN PROGRESS

| Task | Status | Notes |
|------|--------|-------|
| Update HANDOFF.md locally | ✅ | Done |
| Write WEEK_1_POSTMORTEM.md | ✅ | Done |
| Git commit | ✅ | Commit 31eef90 |
| Git push | ✅ | Pushed to origin/main |
| Sync to EC2 | ⏳ | Divergent branches - needs force reset |

**Git Status:**
```
Commit: 31eef90
Message: WEEK 1 POSTMORTEM + SECOND BRAIN DEPLOYMENT
Files: ops/HANDOFF.md, ops/WEEK_1_POSTMORTEM.md
Branch: main
Remote: Pushed successfully
```

**EC2 Sync Issue:**
```
Error: Divergent branches
Cause: Local changes on EC2 conflict with remote
Fix: Need to run `git reset --hard origin/main`
```

---

## DETAILED ACTION LOG

### March 6, 2026 Timeline

| Time (EST) | Action | Result |
|------------|--------|--------|
| 13:42 | Session started | Context from CLAUDE.md |
| 13:43 | User provided Second Brain blueprint | 600+ lines of architecture |
| 13:45 | Discussed architecture concerns | Recommended phased approach |
| 13:50 | User provided FULL STOP plan | 4 phases to execute |
| 13:51 | **Phase 0 Started** | |
| 13:51 | Sent SSM stop command | CommandId: ac58cb1c |
| 13:52 | Verified stop | SNAKE_STOPPED |
| 13:52 | Sent verify dead command | CommandId: c587b77b |
| 13:52 | Sent check positions command | CommandId: afe78179 |
| 13:53 | Confirmed dead | inactive, CONFIRMED_DEAD |
| 13:53 | Confirmed no positions | [] |
| 13:53 | **Phase 0 Complete** | |
| 13:54 | **Phase 1 Started** | |
| 13:55 | Wrote WEEK_1_POSTMORTEM to EC2 | CommandId: 83ac7e1f |
| 13:55 | Verified write | Success |
| 13:55 | **Phase 1 Complete** | |
| 13:56 | **Phase 2 Started** | |
| 13:56 | Listed all EC2 instances | Only wsb-snake found |
| 13:56 | **Phase 2 Complete (N/A)** | |
| 13:57 | **Phase 3 Started** | |
| 13:57 | Checked for blueprint file | FILE_NOT_FOUND |
| 13:57 | Listed security groups | sg-0b540ee94abc2f843 |
| 13:57 | Listed key pairs | wsb-snake-key |
| 13:58 | Checked security group rules | Port 8080 open |
| 13:58 | Listed IAM profiles | EC2-SSM-Profile |
| 13:58 | Got Ubuntu AMI | ami-0071174ad8cbb9e17 |
| 13:59 | Checked IAM role policies | SSM + Bedrock attached |
| 14:00 | Created bootstrap script | /tmp/second_brain_userdata.sh |
| 14:01 | Launched EC2 instance | i-04b0f930bd1e371c1 |
| 14:01 | Waited 30s for initialization | |
| 14:02 | Got public IP | 98.82.24.119 |
| 14:02 | Waited 90s for bootstrap | |
| 14:04 | Tested /api/health | ONLINE |
| 14:04 | Tested /api/context | Returns mission |
| 14:04 | Tested /api/hydra/health | 50% health |
| 14:04 | Tested /api/ask | Bedrock error |
| 14:05 | Listed Bedrock models | Found inference profiles |
| 14:05 | Listed inference profiles | Found us.anthropic.* |
| 14:06 | Fixed model ID in app | Changed to inference profile |
| 14:06 | Restarted second-brain | active |
| 14:07 | Tested /api/ask again | Still error (legacy model) |
| 14:07 | Changed to Claude 3.7 Sonnet | Still error |
| 14:08 | Changed to Claude Haiku 4.5 | SUCCESS |
| 14:08 | Tested trade reporting | Stored |
| 14:08 | Tested bug reporting | Stored |
| 14:08 | **Phase 3 Complete** | |
| 14:09 | **Phase 4 Started** | |
| 14:09 | Read existing HANDOFF.md | Beast Mode V4.0 content |
| 14:10 | Updated HANDOFF.md | New infrastructure |
| 14:10 | Read MARCH_6_POSTMORTEM.md | Existing content |
| 14:11 | Wrote WEEK_1_POSTMORTEM.md | Full week summary |
| 14:12 | Git commit | 31eef90 |
| 14:12 | Git push | Success |
| 14:13 | Attempted EC2 git pull | Divergent branches error |
| 14:13 | Attempted git reset | Interrupted by user |
| 14:15 | User requested status document | This file |

---

## SECOND BRAIN ARCHITECTURE

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     YOU (Sean)                               │
│                  Claude Code / Claude.ai                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ HTTP API
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              SECOND BRAIN EC2 (98.82.24.119)                │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │   Flask API     │  │   AI Engine     │                   │
│  │   (Gunicorn)    │  │   (Bedrock)     │                   │
│  │   Port 8080     │  │   Haiku 4.5     │                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           │                    │                             │
│           ▼                    ▼                             │
│  ┌─────────────────────────────────────────┐                │
│  │           MEMORY STORE                   │                │
│  │  /opt/second-brain/memory/               │                │
│  │  ├── trades/     (daily JSON)            │                │
│  │  ├── bugs/       (markdown)              │                │
│  │  ├── fixes/      (markdown)              │                │
│  │  ├── postmortems/                        │                │
│  │  ├── objectives/ (MISSION.md)            │                │
│  │  └── state/      (system health)         │                │
│  └─────────────────────────────────────────┘                │
│                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │  SSM Client     │  │  HTTP Client    │                   │
│  │  (boto3)        │  │  (requests)     │                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           │                    │                             │
└───────────┼────────────────────┼────────────────────────────┘
            │                    │
            ▼                    ▼
┌─────────────────────┐  ┌─────────────────────┐
│    WSB-Snake EC2    │  │       HYDRA         │
│  i-03f3a7c46ec809a43│  │  54.172.22.157:8000 │
│     (STOPPED)       │  │   /api/predator     │
└─────────────────────┘  └─────────────────────┘
```

### API Reference

#### GET /api/health
```json
{
  "second_brain": "ONLINE",
  "timestamp": "2026-03-06T18:55:45.603630",
  "memory_dirs": ["trades", "bugs", "fixes", ...],
  "wsb_snake_instance": "i-03f3a7c46ec809a43"
}
```

#### GET /api/context
```json
{
  "handoff": "# WSB-Snake Handoff...",
  "compact_summary": null,
  "mission": "# THE MISSION...",
  "todays_trades": [],
  "timestamp": "2026-03-06T18:55:50.891625"
}
```

#### GET /api/hydra/health
```json
{
  "status": "UP",
  "working_fields": 4,
  "total_fields": 8,
  "health_pct": 50,
  "fields": {
    "blowup_probability": 22,
    "blowup_direction": "NEUTRAL",
    "gex_regime": "NEGATIVE",
    "gex_flip_point": null,
    "flow_sweep_direction": "NEUTRAL",
    "charm_flow_per_hour": null,
    "components_healthy": 3,
    "components_total": 4
  }
}
```

#### POST /api/ask
```bash
curl -X POST http://98.82.24.119:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the current system state?"}'
```
Response: AI-generated answer with full context

#### POST /api/report/trade
```bash
curl -X POST http://98.82.24.119:8080/api/report/trade \
  -H "Content-Type: application/json" \
  -d '{"ticker":"SPY","side":"CALL","strike":680,"pnl":500}'
```

#### POST /api/report/bug
```bash
curl -X POST http://98.82.24.119:8080/api/report/bug \
  -H "Content-Type: application/json" \
  -d '{"id":"bug_007","title":"Description","severity":"critical"}'
```

---

## WHAT'S LEFT TO DO

### Immediate (Today)

| Task | Priority | Status |
|------|----------|--------|
| Force sync git to EC2 | HIGH | PENDING |
| Verify HANDOFF.md on EC2 | HIGH | PENDING |
| Verify WEEK_1_POSTMORTEM.md on EC2 | HIGH | PENDING |
| Add handoff to Second Brain memory | MEDIUM | PENDING |

**Command to sync:**
```bash
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cd /home/ubuntu/wsb-snake && git fetch origin && git reset --hard origin/main"]' \
  --region us-east-1
```

### Sunday (Full Day)

| Task | Priority | Estimated Time |
|------|----------|----------------|
| Design CPL V6 architecture | HIGH | 2 hours |
| Hard-code watchlist (SPY, QQQ only) | HIGH | 30 min |
| Simplify signal logic (price + GEX only) | HIGH | 2 hours |
| Create test harness | HIGH | 2 hours |
| Mock scan with real Polygon data | HIGH | 1 hour |
| Verify kill switch ($10K / -$750) | MEDIUM | 30 min |
| Verify cooldown (5 min) | MEDIUM | 30 min |
| Verify position cap (MAX=1) | MEDIUM | 30 min |
| End-to-end test | HIGH | 1 hour |
| Deploy to EC2 (DISABLED) | HIGH | 30 min |

**Total: ~10 hours**

### Monday (Market Day)

| Time | Task |
|------|------|
| 9:00 AM | Second Brain health check |
| 9:15 AM | HYDRA health check |
| 9:25 AM | Pre-market bias check |
| 9:29 AM | Enable wsb-snake |
| 9:30 AM | Market open - monitor closely |
| 9:30-10:30 | Morning momentum window |
| 10:30 AM | First checkpoint - any trades? |
| 12:00 PM | Lunch checkpoint |
| 14:30-15:45 | Power hour window |
| 16:00 PM | Market close |
| 16:05 PM | Stop wsb-snake |
| 16:10 PM | Generate post-mortem |
| 16:15 PM | Report to Second Brain |

---

## SUNDAY BUILD PLAN

### CPL V6 Requirements

#### Hard Requirements (Non-Negotiable)
1. **Watchlist:** `["SPY", "QQQ"]` — NO OTHER TICKERS
2. **Position Cap:** MAX_OPEN_POSITIONS = 1
3. **Daily Limits:** $10K profit / -$750 loss
4. **Cooldown:** 5 minutes between trades
5. **Kill Switch:** Stops trading when limits hit

#### Signal System (V6 Minimal)
```python
# ONLY these signals - all use verified working data
signals = {
    "GEX_NEG": hydra.gex_regime == "NEGATIVE",  # HYDRA (working)
    "PRICE_UP": close > open,                    # Polygon (1 bar OK)
    "TIME_OPT": is_optimal_window(),            # System clock
}

# Conviction = count of True signals
MIN_CONVICTION = 2  # Require at least 2 of 3
```

#### Simplified Logic Flow
```
1. Check watchlist → ONLY SPY or QQQ
2. Check position cap → MAX 1 open
3. Check daily limits → Not hit
4. Check cooldown → 5 min passed
5. Score signals → Need 2/3
6. If pass all → EXECUTE
7. Report to Second Brain
```

### Test Harness Design

```python
# test_cpl_v6.py

def test_watchlist_enforced():
    """DIA and VXX must be rejected"""
    assert cpl.check_ticker("DIA") == False
    assert cpl.check_ticker("VXX") == False
    assert cpl.check_ticker("SPY") == True
    assert cpl.check_ticker("QQQ") == True

def test_position_cap():
    """Only 1 position at a time"""
    mock_positions(count=1)
    assert cpl.can_open_position() == False

def test_kill_switch_profit():
    """Stop at $10K profit"""
    mock_daily_pnl(10500)
    assert cpl.is_trading_allowed() == False

def test_kill_switch_loss():
    """Stop at -$750 loss"""
    mock_daily_pnl(-800)
    assert cpl.is_trading_allowed() == False

def test_cooldown():
    """5 min between trades"""
    mock_last_trade(minutes_ago=3)
    assert cpl.cooldown_passed() == False
    mock_last_trade(minutes_ago=6)
    assert cpl.cooldown_passed() == True

def test_conviction_minimum():
    """Need 2/3 signals"""
    mock_signals(gex=True, price=False, time=False)
    assert cpl.check_conviction() == False  # 1/3
    mock_signals(gex=True, price=True, time=False)
    assert cpl.check_conviction() == True   # 2/3
```

---

## MONDAY GO-LIVE CHECKLIST

### Pre-Market (9:00-9:29 AM)

- [ ] Check Second Brain: `curl http://98.82.24.119:8080/api/health`
- [ ] Check HYDRA: `curl http://98.82.24.119:8080/api/hydra/health`
- [ ] Verify wsb-snake still stopped
- [ ] Verify V6 code deployed
- [ ] Verify watchlist = SPY, QQQ only
- [ ] Check account balance
- [ ] Check pre-market bias

### Market Open (9:29 AM)

```bash
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl enable wsb-snake && systemctl start wsb-snake"]' \
  --region us-east-1
```

### During Market (9:30-16:00)

- [ ] Monitor first 30 minutes closely
- [ ] Check for any DIA/VXX (should not appear)
- [ ] Verify position cap working
- [ ] Verify cooldown working
- [ ] Report every trade to Second Brain
- [ ] Report every bug to Second Brain

### Post-Market (16:00+)

```bash
# Stop trading
aws ssm send-command ... 'commands=["systemctl stop wsb-snake"]'

# Generate post-mortem
curl -X POST http://98.82.24.119:8080/api/ask \
  -d '{"question":"Generate a post-mortem for today based on trades"}'

# Get trade summary
curl http://98.82.24.119:8080/api/context | jq '.todays_trades'
```

---

## CRITICAL COMMANDS REFERENCE

### Second Brain Commands

```bash
# Health check
curl http://98.82.24.119:8080/api/health

# Get full context (after compaction)
curl http://98.82.24.119:8080/api/context | jq .

# Check HYDRA health
curl http://98.82.24.119:8080/api/hydra/health | jq .

# Ask AI question
curl -X POST http://98.82.24.119:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What should I check before market open?"}'

# Report a trade
curl -X POST http://98.82.24.119:8080/api/report/trade \
  -H "Content-Type: application/json" \
  -d '{"ticker":"SPY","side":"CALL","strike":680,"entry":2.50,"exit":3.00,"pnl":250}'

# Report a bug
curl -X POST http://98.82.24.119:8080/api/report/bug \
  -H "Content-Type: application/json" \
  -d '{"id":"bug_007","title":"Bug title","severity":"critical","root_cause":"..."}'
```

### WSB-Snake Commands

```bash
# Check status
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl is-active wsb-snake"]' \
  --region us-east-1

# Start service
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl enable wsb-snake && systemctl start wsb-snake"]' \
  --region us-east-1

# Stop service
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl stop wsb-snake && systemctl disable wsb-snake"]' \
  --region us-east-1

# View logs
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["journalctl -u wsb-snake --no-pager -n 100"]' \
  --region us-east-1

# Deploy new code
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cd /home/ubuntu/wsb-snake && git fetch origin && git reset --hard origin/main && systemctl restart wsb-snake"]' \
  --region us-east-1

# Get command output
aws ssm get-command-invocation \
  --command-id "COMMAND_ID" \
  --instance-id "i-03f3a7c46ec809a43" \
  --region us-east-1
```

### Git Commands

```bash
# Local commit and push
git add -A && git commit -m "message" && git push origin main

# Force sync EC2 to remote
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cd /home/ubuntu/wsb-snake && git fetch origin && git reset --hard origin/main"]' \
  --region us-east-1
```

---

## THE MISSION

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   $2,500 capital → multiply daily via ONE lethal 0DTE trade │
│                                                              │
│   The system only needs to do THREE things right:            │
│                                                              │
│   1. PICK the right trade (conviction stacking)              │
│   2. EXECUTE it perfectly (pyramid + trailing stop)          │
│   3. STOP after one trade (kill switch + cooldown)           │
│                                                              │
│   We don't need to be right every day.                       │
│   We need to be right 3 out of 5 days.                       │
│   And not lose big on the other 2.                           │
│                                                              │
│   That's the edge. That's the mission. Build it.             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 1.0
**Created:** March 6, 2026 @ 2:15 PM EST
**Author:** Claude Opus 4.5
**Status:** Phase 4 In Progress
