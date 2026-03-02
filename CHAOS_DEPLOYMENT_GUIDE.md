# CHAOS ENGINEERING DEPLOYMENT GUIDE

## TL;DR - What You're Getting

Your trading system had **175 restarts in 24 hours** costing **$1,800 in missed trades** because the monitor created an infinite restart loop.

This deployment gives you an **UNKILLABLE** system with:

1. **Circuit Breaker** - After 3 failures in 5 minutes, STOP and alert human
2. **Dead Man's Switch** - Alert if no trades for 30 minutes during market hours
3. **Systemd Limits** - Hard cap of 5 restarts in 5 minutes
4. **Full Testing** - Verify everything works before deployment
5. **Monitoring** - Real-time circuit breaker status

---

## FILES CREATED

### Core Resilience Components

- `/Users/seankuesia/Downloads/Intellibot/ops/circuit_breaker.py`
  - Prevents infinite restart loops
  - CLI: `python3 circuit_breaker.py status|reset|test`

- `/Users/seankuesia/Downloads/Intellibot/ops/dead_mans_switch.py`
  - Monitors trading activity to detect silent failures
  - CLI: `python3 dead_mans_switch.py status`

### Updated Files

- `/Users/seankuesia/Downloads/Intellibot/ops/monitor_agent.py`
  - Integrated circuit breaker and dead man's switch
  - Now imports resilience modules

- `/Users/seankuesia/Downloads/Intellibot/wsb-snake.service`
  - Added `StartLimitBurst=5` and `StartLimitIntervalSec=300`
  - Systemd will STOP trying after 5 failures in 5 minutes

### Deployment Scripts

- `/Users/seankuesia/Downloads/Intellibot/ops/deploy_resilience_phase1.sh`
  - One-command deployment script

- `/Users/seankuesia/Downloads/Intellibot/ops/test_resilience.py`
  - Test suite to verify everything works

### Documentation

- `/Users/seankuesia/Downloads/Intellibot/RESILIENCE_ARCHITECTURE.md`
  - Full architecture design with all 5 layers
  - Includes future phases (meta-monitor, chaos testing, immutable infrastructure)

---

## DEPLOYMENT STEPS

### Step 1: Review Changes Locally

```bash
cd /Users/seankuesia/Downloads/Intellibot

# View circuit breaker implementation
cat ops/circuit_breaker.py | head -50

# View dead man's switch implementation
cat ops/dead_mans_switch.py | head -50

# View updated systemd service file
cat wsb-snake.service
```

### Step 2: Test Locally (Optional)

```bash
# Test circuit breaker
python3 ops/circuit_breaker.py test

# Test dead man's switch
python3 ops/dead_mans_switch.py status
```

### Step 3: Commit to Git

```bash
git add ops/circuit_breaker.py \
        ops/dead_mans_switch.py \
        ops/monitor_agent.py \
        ops/deploy_resilience_phase1.sh \
        ops/test_resilience.py \
        wsb-snake.service \
        RESILIENCE_ARCHITECTURE.md \
        CHAOS_DEPLOYMENT_GUIDE.md

git commit -m "Add resilience architecture Phase 1: circuit breaker + dead man's switch

- Circuit breaker prevents infinite restart loops (max 3 in 5 min)
- Dead man's switch alerts if no trades for 30 minutes
- Systemd limits: max 5 restarts in 5 minutes
- Monitor agent integrated with resilience components
- Full test suite and deployment script

Fixes: 175 restarts in 24h costing $1,800 in missed trades
Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"

git push
```

### Step 4: Deploy via Guardian API

```bash
# Deploy to VM
curl -X POST http://157.245.240.99:8888/deploy

# Wait 10 seconds for pull to complete
sleep 10
```

### Step 5: Run Deployment Script on VM

```bash
# SSH to VM
ssh root@157.245.240.99

# Run Phase 1 deployment
cd /root/wsb-snake
chmod +x ops/deploy_resilience_phase1.sh
./ops/deploy_resilience_phase1.sh
```

### Step 6: Run Tests on VM

```bash
# Still on VM
cd /root/wsb-snake
python3 ops/test_resilience.py
```

### Step 7: Verify Everything Works

```bash
# Check circuit breaker status
python3 ops/circuit_breaker.py status

# Check dead man's switch
python3 ops/dead_mans_switch.py status

# Check systemd limits
systemctl show wsb-snake.service | grep StartLimit

# Check monitor is running
systemctl status wsb-ops-monitor

# Watch monitor logs
journalctl -u wsb-ops-monitor -f
```

---

## HOW IT WORKS

### Before (Broken)

```
Service crashes
  ↓
Monitor detects crash
  ↓
Monitor restarts service
  ↓
Service crashes again (same bug)
  ↓
Monitor restarts again
  ↓
[INFINITE LOOP - 175 restarts]
```

### After (Fixed)

```
Service crashes
  ↓
Circuit breaker: "1/3 restarts used"
  ↓
Monitor restarts service
  ↓
Service crashes again
  ↓
Circuit breaker: "2/3 restarts used"
  ↓
Monitor restarts service
  ↓
Service crashes third time
  ↓
Circuit breaker: "3/3 restarts used - OPENING CIRCUIT"
  ↓
Monitor attempts restart
  ↓
Circuit breaker: "RESTART BLOCKED - Alert human"
  ↓
Telegram alert: "🛑 CIRCUIT BREAKER OPEN - Manual intervention required"
  ↓
[NO MORE RESTARTS]
```

---

## MONITORING

### Circuit Breaker Status

```bash
# On VM
python3 /root/wsb-snake/ops/circuit_breaker.py status
```

Output:
```json
{
  "state": "CLOSED",
  "recent_restarts": 0,
  "max_restarts": 3,
  "time_window_minutes": 5.0,
  "opened_at": null,
  "cooldown_minutes": 30.0,
  "history": []
}
```

### Dead Man's Switch Status

```bash
# On VM
python3 /root/wsb-snake/ops/dead_mans_switch.py status
```

Output:
```json
{
  "is_market_hours": true,
  "last_trade": "2026-03-02T10:45:23",
  "last_signal": "2026-03-02T10:44:15",
  "silence_threshold_minutes": 30.0,
  "database_path": "/root/wsb-snake/wsb_snake_data/wsb_snake.db",
  "database_exists": true
}
```

### Systemd Restart Counter

```bash
# On VM
systemctl show wsb-snake.service | grep -E "Restart|NRestarts"
```

---

## RECOVERY PROCEDURES

### If Circuit Breaker Opens

**Symptoms:**
- Telegram alert: "🛑 CIRCUIT BREAKER OPEN"
- Service is down and won't restart
- Circuit breaker status shows `"state": "OPEN"`

**Recovery:**

1. **Investigate the root cause:**
   ```bash
   ssh root@157.245.240.99
   journalctl -u wsb-snake --no-pager -n 200
   ```

2. **Fix the underlying issue** (database corruption, API keys, etc.)

3. **Reset the circuit breaker:**
   ```bash
   python3 /root/wsb-snake/ops/circuit_breaker.py reset
   ```

4. **Restart the service:**
   ```bash
   systemctl restart wsb-snake
   ```

5. **Monitor for stability:**
   ```bash
   journalctl -u wsb-snake -f
   ```

### If Dead Man's Switch Alerts

**Symptoms:**
- Telegram alert: "⚠️ DEAD MAN'S SWITCH: No trades in 35m"
- Service is running but not trading

**Recovery:**

1. **Check if market is actually open:**
   ```bash
   python3 -c "
   from ops.dead_mans_switch import DeadMansSwitch
   dms = DeadMansSwitch()
   print('Market hours:', dms.is_market_hours())
   "
   ```

2. **Check for CPL signal issues:**
   ```bash
   sqlite3 /root/wsb-snake/wsb_snake_data/wsb_snake.db "SELECT * FROM cpl_signals ORDER BY timestamp DESC LIMIT 5;"
   ```

3. **Check for API connectivity:**
   ```bash
   curl http://localhost:8000/api/health
   ```

4. **If needed, restart service:**
   ```bash
   systemctl restart wsb-snake
   ```

### If Systemd Gives Up

**Symptoms:**
- Service status: `failed`
- `systemctl status wsb-snake` shows "start request repeated too quickly"

**Recovery:**

1. **Check systemd failure count:**
   ```bash
   systemctl show wsb-snake.service | grep StartLimit
   ```

2. **Reset systemd failure count:**
   ```bash
   systemctl reset-failed wsb-snake
   ```

3. **Fix underlying issue, then restart:**
   ```bash
   systemctl restart wsb-snake
   ```

---

## TESTING THE CIRCUIT BREAKER

Want to see it in action? Intentionally trigger the circuit breaker:

```bash
# On VM
cd /root/wsb-snake

# Kill service 3 times rapidly
for i in {1..3}; do
  echo "Crash $i"
  systemctl stop wsb-snake
  sleep 5
done

# Check circuit breaker status - should be OPEN
python3 ops/circuit_breaker.py status

# Try to start service - monitor should block it
systemctl start wsb-snake

# Check Telegram - should see circuit breaker alert
```

---

## METRICS TO TRACK

### Before Resilience Architecture

| Metric | Value |
|--------|-------|
| Restarts in 24h | 175 |
| MTTR | Unknown |
| False restart rate | 100% |
| Cost of downtime | $1,800 |

### After Resilience Architecture (Target)

| Metric | Target |
|--------|--------|
| Restarts in 24h | <5 |
| MTTR | <30 seconds |
| False restart rate | <10% |
| Cost of downtime | $0 |

Track these with:
```bash
# On VM - add to crontab for daily tracking
0 0 * * * journalctl -u wsb-snake --since today | grep -c "Started WSB Snake" > /tmp/restart_count_today.txt
```

---

## FUTURE PHASES

This is Phase 1. Future enhancements:

### Phase 2 (This Week)
- **Meta-monitor** - Watches the monitor itself
- **Resilient data provider** - Fallback from Polygon → Alpaca → Cache
- **Chaos test suite** - Daily pre-market resilience testing

### Phase 3 (Next Week)
- **Docker containerization** - Immutable infrastructure
- **Blue-green restarts** - Spin up new instance, kill old one
- **Automated chaos testing** - Run at 9:00 AM ET daily

See `RESILIENCE_ARCHITECTURE.md` for full details.

---

## SUPPORT

### Quick Reference

```bash
# Circuit breaker status
python3 /root/wsb-snake/ops/circuit_breaker.py status

# Reset circuit breaker
python3 /root/wsb-snake/ops/circuit_breaker.py reset

# Dead man's switch status
python3 /root/wsb-snake/ops/dead_mans_switch.py status

# Monitor logs
journalctl -u wsb-ops-monitor -f

# Trading service logs
journalctl -u wsb-snake -f

# Systemd restart limits
systemctl show wsb-snake.service | grep StartLimit
```

### Files to Know

- Circuit breaker state: `/tmp/circuit_breaker_state.json`
- Monitor state: `/root/wsb-snake/ops/state.json`
- Service logs: `journalctl -u wsb-snake`
- Monitor logs: `journalctl -u wsb-ops-monitor`

---

## CONCLUSION

You now have a **5-layer resilience architecture** that will:

1. **Stop infinite restart loops** (circuit breaker)
2. **Detect business logic failures** (dead man's switch)
3. **Limit systemd restart attempts** (StartLimitBurst)
4. **Alert you immediately** (Telegram integration)
5. **Provide full visibility** (status commands)

The result: An **UNKILLABLE** trading system that gets **stronger under stress**.

**Deploy with confidence. Break things intentionally. Learn from controlled failures.**

---

**Questions?** Review `RESILIENCE_ARCHITECTURE.md` for full technical details.

**Ready?** Run the deployment steps above and watch your system become unkillable.
