#!/bin/bash
# Pre-market health check — runs at 9:25 AM ET daily
cd /home/ubuntu/wsb-snake

SNAKE=$(systemctl is-active wsb-snake)
MONITOR=$(systemctl is-active wsb-ops-monitor)
LAST_LOG=$(journalctl -u wsb-snake --no-pager --since "2 min ago" | tail -1)

# Source env for Telegram
source /home/ubuntu/wsb-snake/.env

MSG="🐍 PRE-MARKET CHECK $(date '+%Y-%m-%d %H:%M ET')
wsb-snake: $SNAKE
monitor: $MONITOR
last log: $LAST_LOG"

# If either service is down, restart and alert
if [ "$SNAKE" != "active" ]; then
    systemctl restart wsb-snake
    MSG="$MSG
⚠️ wsb-snake was DOWN — restarted"
fi

if [ "$MONITOR" != "active" ]; then
    systemctl restart wsb-ops-monitor
    MSG="$MSG
⚠️ monitor was DOWN — restarted"
fi

# Send to Telegram
curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  -d chat_id="${TELEGRAM_CHAT_ID}" \
  -d text="$MSG" > /dev/null
