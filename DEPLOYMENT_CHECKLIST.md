# WSB Snake Deployment Checklist

## Quick Reference

| Item | Command/Check |
|------|---------------|
| SSH | `ssh trader@YOUR_VM_IP` |
| Service | `sudo systemctl status wsb-snake` |
| Logs | `tail -f /var/log/wsb-snake/output.log` |
| Health | `curl http://localhost:5000/health` |
| Restart | `sudo systemctl restart wsb-snake` |

---

## 1. Pre-Deployment Verification

### Local Testing
```bash
# Run environment audit (checks all required API keys)
python script/audit_env.py

# Test core imports
python -c "from wsb_snake.main import main; print('OK')"

# Verify Alpaca connection
python -c "
from wsb_snake.trading.alpaca_executor import alpaca_executor
acct = alpaca_executor.get_account()
print(f'Buying Power: \${float(acct.get(\"buying_power\", 0)):,.2f}')
print(f'Trading Mode: {\"LIVE\" if alpaca_executor.LIVE_TRADING else \"PAPER\"}')
"

# Test Telegram alerts
python -c "
from wsb_snake.notifications.telegram_bot import send_alert
send_alert('Deployment test message')
print('Check Telegram for message')
"
```

### Required Environment Variables
Verify these are set (values not shown for security):

**CRITICAL - Trading & Alerts:**
- [ ] `ALPACA_API_KEY` - Alpaca trading key
- [ ] `ALPACA_SECRET_KEY` - Alpaca trading secret
- [ ] `POLYGON_API_KEY` - Market data (options, 5s bars)
- [ ] `TELEGRAM_BOT_TOKEN` - Alert notifications
- [ ] `TELEGRAM_CHAT_ID` - Your Telegram chat ID
- [ ] `OPENAI_API_KEY` - GPT-4o AI confirmation
- [ ] `DEEPSEEK_API_KEY` - Backup AI model

**OPTIONAL - Enhanced Features:**
- [ ] `ALPACA_BASE_URL` - Default: `https://paper-api.alpaca.markets`
- [ ] `ALPACA_LIVE_TRADING` - Set `true` for live (DANGER)
- [ ] `FINNHUB_API_KEY` - News, earnings, sentiment
- [ ] `BENZINGA_API_KEY` - News data
- [ ] `GEMINI_API_KEY` - Primary AI vision
- [ ] `REDDIT_CLIENT_ID` - Social sentiment
- [ ] `REDDIT_CLIENT_SECRET` - Reddit auth
- [ ] `GOOGLE_DRIVE_FOLDER_ID` - Screenshot learning
- [ ] `GOOGLE_SERVICE_ACCOUNT` - Drive service account

**TUNING (optional):**
- [ ] `RISK_MAX_DAILY_LOSS` - Kill switch threshold (e.g., -500)
- [ ] `RISK_MAX_CONCURRENT_POSITIONS` - Max positions (default: 3)
- [ ] `RISK_MAX_DAILY_EXPOSURE` - Max exposure (default: $4000)
- [ ] `SCALP_TARGET_PCT` - Target exit (default: 1.06 = +6%)
- [ ] `SCALP_STOP_PCT` - Stop loss (default: 0.90 = -10%)
- [ ] `SCALP_MAX_HOLD_MINUTES` - Max hold time (default: 5)

### Code Review
```bash
# Check for uncommitted changes
git status

# Ensure on correct branch
git branch --show-current

# Pull latest
git pull origin main

# Verify requirements are up to date
pip install -r requirements.txt --dry-run
```

---

## 2. SSH Commands for VM Deployment

### Initial Connection
```bash
# SSH into VM (as trader user)
ssh trader@YOUR_VM_IP

# Or with specific key
ssh -i ~/.ssh/your_key trader@YOUR_VM_IP
```

### First-Time Setup
```bash
# Navigate to project
cd ~/apps/wsb-snake

# Activate virtual environment
source venv/bin/activate

# Pull latest code
git pull origin main

# Install/update dependencies
pip install -r requirements.txt

# Copy environment file (edit with your keys)
cp .env.example ~/.env
nano ~/.env

# Verify environment loads
source ~/.bashrc
python script/audit_env.py
```

### Deploy New Version
```bash
# SSH in
ssh trader@YOUR_VM_IP

# Pull and deploy
cd ~/apps/wsb-snake
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart wsb-snake

# Verify deployment
sudo systemctl status wsb-snake
curl http://localhost:5000/health
```

---

## 3. Service Restart Commands

### Systemd Service Control
```bash
# Check status
sudo systemctl status wsb-snake

# Start service
sudo systemctl start wsb-snake

# Stop service
sudo systemctl stop wsb-snake

# Restart service (for updates)
sudo systemctl restart wsb-snake

# Enable auto-start on boot
sudo systemctl enable wsb-snake

# Disable auto-start
sudo systemctl disable wsb-snake

# Reload systemd after config changes
sudo systemctl daemon-reload
```

### Docker Deployment (Alternative)
```bash
# If using Docker Compose
cd ~/apps/wsb-snake

# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# Rebuild and restart
docker-compose up -d --build

# View logs
docker-compose logs -f wsb-snake
```

### Manual Process Control (Debugging)
```bash
# Run manually (foreground - for debugging)
cd ~/apps/wsb-snake
source venv/bin/activate
python main.py

# Run in background with tmux
tmux new -s snake
python main.py
# Ctrl+B, D to detach
# tmux attach -t snake to reattach
```

---

## 4. Log Monitoring Commands

### Real-Time Logs
```bash
# Main output log
tail -f /var/log/wsb-snake/output.log

# Error log
tail -f /var/log/wsb-snake/error.log

# Both logs combined
tail -f /var/log/wsb-snake/*.log

# Last 100 lines
tail -100 /var/log/wsb-snake/output.log

# Search for specific patterns
grep -i "error" /var/log/wsb-snake/output.log
grep -i "trade" /var/log/wsb-snake/output.log | tail -50
grep "FILLED" /var/log/wsb-snake/output.log
```

### Systemd Journal
```bash
# Recent service logs
journalctl -u wsb-snake -n 50

# Follow logs in real-time
journalctl -u wsb-snake -f

# Logs since last boot
journalctl -u wsb-snake -b

# Logs from specific time
journalctl -u wsb-snake --since "2026-02-08 09:30:00"

# Logs from today
journalctl -u wsb-snake --since today
```

### Docker Logs (if using Docker)
```bash
# Follow logs
docker-compose logs -f wsb-snake

# Last 100 lines
docker-compose logs --tail 100 wsb-snake

# With timestamps
docker-compose logs -f -t wsb-snake
```

### Log Analysis
```bash
# Count trades today
grep "ORDER FILLED" /var/log/wsb-snake/output.log | wc -l

# Find errors
grep -E "(ERROR|Exception|Traceback)" /var/log/wsb-snake/error.log

# Check pipeline runs
grep "Running scheduled pipeline" /var/log/wsb-snake/output.log | tail -10

# Check EOD closes
grep "0DTE EOD close" /var/log/wsb-snake/output.log
```

---

## 5. Rollback Procedure

### Quick Rollback (Git)
```bash
# SSH into VM
ssh trader@YOUR_VM_IP
cd ~/apps/wsb-snake

# Stop service
sudo systemctl stop wsb-snake

# Find previous commit
git log --oneline -10

# Rollback to specific commit
git checkout <commit_hash>

# Or rollback to previous commit
git reset --hard HEAD~1

# Restart service
sudo systemctl start wsb-snake

# Verify
sudo systemctl status wsb-snake
```

### Database Rollback
```bash
# Backups are stored in wsb_snake_data/backups/
ls -la wsb_snake_data/backups/

# Restore database from backup
cp wsb_snake_data/backups/wsb_snake_YYYYMMDD_HHMM.db wsb_snake_data/wsb_snake.db

# Restart service
sudo systemctl restart wsb-snake
```

### Full Emergency Rollback
```bash
# 1. Stop everything
sudo systemctl stop wsb-snake

# 2. Backup current state
cp -r wsb_snake_data wsb_snake_data_backup_$(date +%Y%m%d_%H%M)

# 3. Hard reset to last known good state
git fetch origin
git reset --hard origin/main~3  # Go back 3 commits

# 4. Restore last good database
cp wsb_snake_data/backups/wsb_snake_YYYYMMDD_HHMM.db wsb_snake_data/wsb_snake.db

# 5. Reinstall dependencies
source venv/bin/activate
pip install -r requirements.txt

# 6. Restart
sudo systemctl start wsb-snake

# 7. Verify
curl http://localhost:5000/health
```

### Rollback Docker Deployment
```bash
# Stop current
docker-compose down

# Pull specific version/tag
git checkout v2.4.0  # or specific tag

# Rebuild and start
docker-compose up -d --build
```

---

## 6. Critical Environment Variables

### Production .env Template
```bash
# ============================================
# CRITICAL - TRADING (Required)
# ============================================
ALPACA_API_KEY=pk_xxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=sk_xxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets
# Set to "true" for LIVE trading (REAL MONEY!)
ALPACA_LIVE_TRADING=false

# ============================================
# CRITICAL - MARKET DATA (Required)
# ============================================
POLYGON_API_KEY=xxxxxxxxxxxxxxxx

# ============================================
# CRITICAL - AI MODELS (Required)
# ============================================
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
GEMINI_API_KEY=xxxxxxxxxxxxxxxx

# ============================================
# CRITICAL - NOTIFICATIONS (Required)
# ============================================
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789

# ============================================
# OPTIONAL - ENHANCED FEATURES
# ============================================
FINNHUB_API_KEY=xxxxxxxxxxxxxxxx
BENZINGA_API_KEY=xxxxxxxxxxxxxxxx
REDDIT_CLIENT_ID=xxxxxxxxxxxxxxxx
REDDIT_CLIENT_SECRET=xxxxxxxxxxxxxxxx
REDDIT_USER_AGENT=python:wsb-snake:v2.5

# ============================================
# OPTIONAL - SCREENSHOT LEARNING
# ============================================
GOOGLE_DRIVE_FOLDER_ID=1EbGgR2r_0jxDjQWvlN9yuxlrzUPWvLf4
GOOGLE_SERVICE_ACCOUNT=intellibot-drive@intellibot-486323.iam.gserviceaccount.com
SCREENSHOT_SCAN_INTERVAL=300

# ============================================
# OPTIONAL - RISK TUNING
# ============================================
# Stop all trading if daily loss exceeds this
RISK_MAX_DAILY_LOSS=-500
# Maximum concurrent positions
RISK_MAX_CONCURRENT_POSITIONS=3
# Maximum daily exposure (dollars)
RISK_MAX_DAILY_EXPOSURE=4000

# ============================================
# OPTIONAL - SCALPER TUNING
# ============================================
# Target exit (1.06 = +6%)
SCALP_TARGET_PCT=1.06
# Stop loss (0.90 = -10%)
SCALP_STOP_PCT=0.90
# Max hold time in minutes
SCALP_MAX_HOLD_MINUTES=5
```

### Verify Environment on VM
```bash
# Check all required keys are set
python script/audit_env.py

# Manually check critical vars (shows set/not set, not values)
for var in ALPACA_API_KEY ALPACA_SECRET_KEY POLYGON_API_KEY TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID; do
  if [ -z "${!var}" ]; then
    echo "$var: NOT SET"
  else
    echo "$var: SET"
  fi
done
```

---

## 7. Health Check Commands

### Quick Health Check
```bash
# HTTP health endpoint
curl http://localhost:5000/health
# Expected: {"status": "healthy", "snake_running": true}

# Detailed status
curl http://localhost:5000/status

# Root endpoint
curl http://localhost:5000/
# Expected: {"status": "running", "service": "WSB Snake v2.5", ...}
```

### Full System Health Check Script
```bash
#!/bin/bash
echo "=== WSB Snake Health Check ==="
echo ""

# 1. Service Status
echo "1. Service Status:"
systemctl is-active wsb-snake && echo "   [OK] Service running" || echo "   [FAIL] Service not running"

# 2. HTTP Health
echo "2. HTTP Health Endpoint:"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000/health)
if [ "$HTTP_STATUS" == "200" ]; then
  echo "   [OK] Health endpoint responding (HTTP $HTTP_STATUS)"
else
  echo "   [FAIL] Health endpoint not responding (HTTP $HTTP_STATUS)"
fi

# 3. Process Check
echo "3. Python Process:"
pgrep -f "python.*main.py" > /dev/null && echo "   [OK] Python process running" || echo "   [FAIL] Python process not found"

# 4. Memory Usage
echo "4. Memory Usage:"
free -h | grep Mem | awk '{print "   Used: "$3" / "$2}'

# 5. Disk Usage
echo "5. Disk Usage:"
df -h / | tail -1 | awk '{print "   Used: "$3" / "$2" ("$5")"}'

# 6. Last Log Entry
echo "6. Last Log Entry:"
tail -1 /var/log/wsb-snake/output.log

# 7. Error Count (last hour)
echo "7. Errors (last hour):"
ERROR_COUNT=$(grep -c "ERROR" /var/log/wsb-snake/error.log 2>/dev/null || echo "0")
echo "   Count: $ERROR_COUNT"

echo ""
echo "=== End Health Check ==="
```

### Alpaca Account Health
```bash
# Check Alpaca connection and account status
python -c "
from wsb_snake.trading.alpaca_executor import alpaca_executor

acct = alpaca_executor.get_account()
if not acct:
    print('[FAIL] Cannot connect to Alpaca')
    exit(1)

print('[OK] Alpaca Connected')
print(f'  Mode: {\"LIVE\" if alpaca_executor.LIVE_TRADING else \"PAPER\"}')
print(f'  Buying Power: \${float(acct.get(\"buying_power\", 0)):,.2f}')
print(f'  Equity: \${float(acct.get(\"equity\", 0)):,.2f}')

positions = alpaca_executor.get_options_positions()
print(f'  Open Positions: {len(positions)}')

stats = alpaca_executor.get_session_stats()
print(f'  Daily PnL: \${stats[\"daily_pnl\"]:+.2f}')
print(f'  Win Rate: {stats[\"win_rate\"]:.0f}%')
"
```

### Market Hours Check
```bash
# Check if market is open
python -c "
from wsb_snake.utils.session_regime import get_session_info, is_market_open

info = get_session_info()
print(f'Session: {info[\"session\"].upper()}')
print(f'Market Open: {\"YES\" if info[\"is_open\"] else \"NO\"}')
print(f'Time ET: {info[\"current_time_et\"]}')
print(f'Power Hour: {\"YES\" if info[\"is_power_hour\"] else \"NO\"}')
"
```

### Telegram Test
```bash
# Send test message to verify Telegram is working
python -c "
from wsb_snake.notifications.telegram_bot import send_alert
result = send_alert('Health check test - ignore this message')
print('[OK] Telegram message sent' if result else '[FAIL] Telegram send failed')
"
```

---

## Daily Operations Checklist

### Pre-Market (Before 9:30 AM ET)
- [ ] Verify service running: `systemctl status wsb-snake`
- [ ] Check Telegram for startup message
- [ ] Verify health endpoint: `curl http://localhost:5000/health`
- [ ] Check Alpaca buying power
- [ ] Review overnight errors: `grep ERROR /var/log/wsb-snake/error.log | tail -20`

### Market Hours (9:30 AM - 4:00 PM ET)
- [ ] Monitor Telegram for trade alerts
- [ ] Periodic health check: `curl http://localhost:5000/status`
- [ ] Watch for error patterns in logs
- [ ] Check open positions: `curl http://localhost:5000/status | jq .positions`

### Post-Market (After 4:00 PM ET)
- [ ] Confirm all positions closed (EOD auto-close at 3:55 PM)
- [ ] Review daily summary in Telegram
- [ ] Check final daily P&L
- [ ] Review any errors: `journalctl -u wsb-snake --since today | grep -i error`
- [ ] Backup database: `bash script/backup_state.sh`

---

## Emergency Procedures

### Bot Not Responding
```bash
# 1. Check if process is running
ps aux | grep python | grep main.py

# 2. Check service status
sudo systemctl status wsb-snake

# 3. Check for crash in logs
tail -100 /var/log/wsb-snake/error.log

# 4. Force restart
sudo systemctl restart wsb-snake

# 5. If still failing, run manually to see errors
sudo systemctl stop wsb-snake
cd ~/apps/wsb-snake
source venv/bin/activate
python main.py
```

### Positions Not Closing
```bash
# 1. Check current positions
python -c "
from wsb_snake.trading.alpaca_executor import alpaca_executor
positions = alpaca_executor.get_options_positions()
for p in positions:
    print(f'{p[\"symbol\"]}: {p[\"qty\"]} @ \${p[\"current_price\"]}')
"

# 2. Force close all positions
python -c "
from wsb_snake.trading.alpaca_executor import alpaca_executor
closed = alpaca_executor.close_all_0dte_positions()
print(f'Closed {closed} positions')
"
```

### Kill Switch - Stop All Trading
```bash
# Immediate stop
sudo systemctl stop wsb-snake

# Verify no orders pending
python -c "
import requests
import os
headers = {
    'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'],
    'APCA-API-SECRET-KEY': os.environ['ALPACA_SECRET_KEY']
}
resp = requests.get('https://paper-api.alpaca.markets/v2/orders', headers=headers)
orders = resp.json()
print(f'Pending orders: {len(orders)}')
for o in orders:
    print(f'  {o[\"symbol\"]}: {o[\"side\"]} {o[\"qty\"]} - {o[\"status\"]}')
"

# Cancel all pending orders if needed
python -c "
import requests
import os
headers = {
    'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'],
    'APCA-API-SECRET-KEY': os.environ['ALPACA_SECRET_KEY']
}
resp = requests.delete('https://paper-api.alpaca.markets/v2/orders', headers=headers)
print('All orders cancelled' if resp.status_code == 207 else f'Error: {resp.text}')
"
```

---

**Document Version:** 1.0
**Last Updated:** February 2026
**Target Platform:** Digital Ocean Ubuntu 24.04 LTS with systemd
