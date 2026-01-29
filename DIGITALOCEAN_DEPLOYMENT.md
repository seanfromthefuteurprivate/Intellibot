# WSB Snake - Digital Ocean Deployment Guide

## Overview

This guide covers deploying WSB Snake to a Digital Ocean Droplet for 24/7 autonomous trading operations.

**System Components:**
- Python Trading Engine (FastAPI on port 5000)
- Optional: Node.js Dashboard (React/Express on port 3000)

---

## Prerequisites

### 1. Digital Ocean Account
- Create account at https://digitalocean.com
- Add billing/payment method

### 2. API Keys (Required)
Gather these before deployment:

| API | Purpose | Get From |
|-----|---------|----------|
| `ALPACA_API_KEY` | Trading execution | https://alpaca.markets |
| `ALPACA_SECRET_KEY` | Trading auth | https://alpaca.markets |
| `POLYGON_API_KEY` | Market data (5s bars, options) | https://polygon.io |
| `FINNHUB_API_KEY` | News, earnings, sentiment | https://finnhub.io |
| `BENZINGA_API_KEY` | News data | https://benzinga.com/apis |
| `GEMINI_API_KEY` | Primary AI vision | https://aistudio.google.com |
| `DEEPSEEK_API_KEY` | Backup AI | https://deepseek.com |
| `TELEGRAM_BOT_TOKEN` | Alert notifications | @BotFather on Telegram |
| `TELEGRAM_CHAT_ID` | Your chat ID | See setup below |

### 3. Optional API Keys
| API | Purpose | Get From |
|-----|---------|----------|
| `OPENAI_API_KEY` | GPT-4o confirmation | https://platform.openai.com |
| `REDDIT_CLIENT_ID` | Social sentiment | https://reddit.com/prefs/apps |
| `REDDIT_CLIENT_SECRET` | Reddit auth | https://reddit.com/prefs/apps |

---

## Step 1: Create Digital Ocean Droplet

### Recommended Specs

| Tier | CPU | RAM | Storage | Monthly Cost | Use Case |
|------|-----|-----|---------|--------------|----------|
| **Basic** | 1 vCPU | 2 GB | 50 GB | $12/mo | Testing |
| **Recommended** | 2 vCPU | 4 GB | 80 GB | $24/mo | Production |
| **Premium** | 4 vCPU | 8 GB | 160 GB | $48/mo | + Dashboard |

### Create Droplet

1. Go to **Create** > **Droplets**
2. Choose **Ubuntu 24.04 LTS** (or 22.04)
3. Select **Basic** plan (shared CPU)
4. Choose **Regular SSD** (or NVMe for premium)
5. Pick datacenter: **New York** (NYC1/NYC3) - closest to US markets
6. Add **SSH Key** (recommended) or use password
7. Name it: `wsb-snake-prod`
8. Click **Create Droplet**

---

## Step 2: Initial Server Setup

### SSH into your server

```bash
ssh root@YOUR_DROPLET_IP
```

### Update system and install dependencies

```bash
# Update packages
apt update && apt upgrade -y

# Install Python 3.11 and essentials
apt install -y python3.11 python3.11-venv python3-pip git curl wget htop tmux

# Make Python 3.11 the default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Node.js 20 (for dashboard - optional)
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# Verify installations
python3 --version   # Should show 3.11.x
node --version      # Should show v20.x
npm --version       # Should show 10.x
```

---

## Step 3: Create Non-Root User

```bash
# Create user
adduser trader
usermod -aG sudo trader

# Set up SSH for the new user
mkdir -p /home/trader/.ssh
cp ~/.ssh/authorized_keys /home/trader/.ssh/
chown -R trader:trader /home/trader/.ssh

# Switch to trader user
su - trader
```

---

## Step 4: Clone and Setup Project

```bash
# Create project directory
mkdir -p ~/apps
cd ~/apps

# Clone your repository (replace with your repo URL)
git clone https://github.com/YOUR_USERNAME/Intellibot.git wsb-snake
cd wsb-snake

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Update requirements.txt (if needed)

The current `requirements.txt` is minimal. You may need these packages:

```bash
pip install fastapi uvicorn openai langchain requests schedule \
    python-dotenv alpaca-trade-api polygon-api-client pytz matplotlib \
    pandas numpy aiohttp websockets httpx tenacity google-generativeai
```

Save the complete requirements:
```bash
pip freeze > requirements.txt
```

---

## Step 5: Configure Environment Variables

### Create environment file

```bash
nano ~/.env
```

Add your API keys:

```bash
# Alpaca (Paper Trading)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Market Data
POLYGON_API_KEY=your_polygon_key
FINNHUB_API_KEY=your_finnhub_key
BENZINGA_API_KEY=your_benzinga_key

# AI Models
GEMINI_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key
OPENAI_API_KEY=your_openai_key

# Telegram Alerts
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Optional: Reddit
REDDIT_CLIENT_ID=your_reddit_id
REDDIT_CLIENT_SECRET=your_reddit_secret
REDDIT_USER_AGENT=python:wsb-snake:v2.5
```

### Load environment on login

```bash
echo 'set -a; source ~/.env; set +a' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 6: Set Timezone to US Eastern

```bash
sudo timedatectl set-timezone America/New_York
timedatectl
```

Verify it shows `America/New_York` or `EDT/EST`.

---

## Step 7: Create Systemd Service

This ensures WSB Snake starts automatically and restarts on failure.

```bash
sudo nano /etc/systemd/system/wsb-snake.service
```

Add this configuration:

```ini
[Unit]
Description=WSB Snake Trading Engine
After=network.target

[Service]
Type=simple
User=trader
Group=trader
WorkingDirectory=/home/trader/apps/wsb-snake
Environment="PATH=/home/trader/apps/wsb-snake/venv/bin:/usr/bin"
EnvironmentFile=/home/trader/.env
ExecStart=/home/trader/apps/wsb-snake/venv/bin/python main.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/wsb-snake/output.log
StandardError=append:/var/log/wsb-snake/error.log

[Install]
WantedBy=multi-user.target
```

### Create log directory and enable service

```bash
sudo mkdir -p /var/log/wsb-snake
sudo chown trader:trader /var/log/wsb-snake

sudo systemctl daemon-reload
sudo systemctl enable wsb-snake
sudo systemctl start wsb-snake
```

### Check status

```bash
sudo systemctl status wsb-snake
```

---

## Step 8: Configure Firewall

```bash
# Enable UFW
sudo ufw allow OpenSSH
sudo ufw allow 5000/tcp   # FastAPI health endpoint
sudo ufw enable

# Check status
sudo ufw status
```

---

## Step 9: Setup Telegram Bot

### Create Bot
1. Message `@BotFather` on Telegram
2. Send `/newbot`
3. Follow prompts to name your bot
4. Copy the bot token (looks like: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`)

### Get Your Chat ID
1. Message your new bot (send anything)
2. Visit: `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
3. Find `"chat":{"id":123456789}` - that number is your chat ID

### Test Connection
```bash
curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
  -d "chat_id=$TELEGRAM_CHAT_ID" \
  -d "text=WSB Snake test message"
```

---

## Step 10: Verify Deployment

### Check all services
```bash
# Service status
sudo systemctl status wsb-snake

# View logs
tail -f /var/log/wsb-snake/output.log

# Check health endpoint
curl http://localhost:5000/health
```

### Expected responses
- Telegram: Startup message received
- Health endpoint: `{"status": "healthy", "snake_running": true}`
- Logs: Show "WSB SNAKE v2.5 ONLINE"

---

## Monitoring & Maintenance

### View Logs
```bash
# Real-time logs
tail -f /var/log/wsb-snake/output.log

# Error logs
tail -f /var/log/wsb-snake/error.log

# Last 100 lines
tail -100 /var/log/wsb-snake/output.log
```

### Control Commands
```bash
# Stop
sudo systemctl stop wsb-snake

# Start
sudo systemctl start wsb-snake

# Restart
sudo systemctl restart wsb-snake

# View status
sudo systemctl status wsb-snake
```

### Update Code
```bash
cd ~/apps/wsb-snake
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart wsb-snake
```

---

## Optional: Setup Dashboard (Frontend)

If you want the web dashboard:

```bash
cd ~/apps/wsb-snake

# Install Node.js dependencies
npm install

# Build for production
npm run build

# Create systemd service for frontend
sudo nano /etc/systemd/system/wsb-dashboard.service
```

Dashboard service config:
```ini
[Unit]
Description=WSB Snake Dashboard
After=network.target wsb-snake.service

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/apps/wsb-snake
Environment="NODE_ENV=production"
ExecStart=/usr/bin/npm run start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable dashboard:
```bash
sudo systemctl daemon-reload
sudo systemctl enable wsb-dashboard
sudo systemctl start wsb-dashboard
sudo ufw allow 3000/tcp
```

---

## Optional: Setup Nginx Reverse Proxy

For HTTPS and domain access:

```bash
sudo apt install nginx certbot python3-certbot-nginx

sudo nano /etc/nginx/sites-available/wsb-snake
```

Nginx config:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable and get SSL:
```bash
sudo ln -s /etc/nginx/sites-available/wsb-snake /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
sudo certbot --nginx -d your-domain.com
```

---

## Log Rotation

Prevent logs from filling disk:

```bash
sudo nano /etc/logrotate.d/wsb-snake
```

Add:
```
/var/log/wsb-snake/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 trader trader
    postrotate
        systemctl restart wsb-snake
    endscript
}
```

---

## Troubleshooting

### Bot not starting
```bash
# Check logs for errors
journalctl -u wsb-snake -n 50

# Test manually
cd ~/apps/wsb-snake
source venv/bin/activate
python main.py
```

### No Telegram alerts
```bash
# Verify environment variables
echo $TELEGRAM_BOT_TOKEN
echo $TELEGRAM_CHAT_ID

# Test Telegram API
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"
```

### Alpaca connection issues
```bash
# Test Alpaca API
python3 -c "
from wsb_snake.trading.alpaca_executor import alpaca_executor
print(alpaca_executor.get_account())
"
```

### Market hours issues
```bash
# Verify timezone
timedatectl

# Check current ET time
python3 -c "
from wsb_snake.utils.session_regime import get_eastern_time
print(get_eastern_time())
"
```

---

## Security Best Practices

1. **Never commit `.env` files** - Add to `.gitignore`
2. **Use SSH keys** - Disable password authentication
3. **Keep system updated** - `apt update && apt upgrade` weekly
4. **Monitor resource usage** - Use `htop` or Digital Ocean monitoring
5. **Backup database** - Copy `wsb_snake.db` periodically
6. **Use paper trading first** - Test thoroughly before live trading

---

## Cost Optimization

- Use **Reserved IPs** ($4/mo) for stable IP
- Consider **Spaces** for log storage if needed
- Enable **Monitoring** (free) in droplet settings
- Set up **Alerts** for CPU/memory thresholds

---

## Daily Operations Checklist

### Pre-Market (Before 9:30 AM ET)
- [ ] Verify service is running: `systemctl status wsb-snake`
- [ ] Check Telegram for startup message
- [ ] Verify Alpaca account balance
- [ ] Review any overnight errors in logs

### During Market Hours
- [ ] Monitor Telegram for trade alerts
- [ ] Check `/health` endpoint periodically
- [ ] Watch for error patterns in logs

### Post-Market (After 4:00 PM ET)
- [ ] Confirm all positions closed
- [ ] Review daily summary in Telegram
- [ ] Check API usage (stay within limits)

---

## Support Resources

- **Alpaca Docs:** https://docs.alpaca.markets
- **Polygon Docs:** https://polygon.io/docs
- **Digital Ocean Docs:** https://docs.digitalocean.com
- **Project Issues:** GitHub Issues in your repo

---

**Document Version:** 1.0
**Last Updated:** January 2026
**Target Platform:** Digital Ocean Ubuntu 24.04 LTS
