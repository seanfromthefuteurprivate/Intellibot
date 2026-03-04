#!/bin/bash
# Pre-market health check — runs at 9:25 AM ET daily
cd /home/ubuntu/wsb-snake

SNAKE=$(systemctl is-active wsb-snake)
MONITOR=$(systemctl is-active wsb-ops-monitor)
LAST_LOG=$(journalctl -u wsb-snake --no-pager --since "2 min ago" | tail -1)

# Source env for Telegram and Polygon API
source /home/ubuntu/wsb-snake/.env

# ═══════════════════════════════════════════════════════════════════════════════
# PRE-MARKET BIAS CALCULATION
# Fetch SPY pre-market price and compare to previous close
# ═══════════════════════════════════════════════════════════════════════════════
PREMARKET_BIAS="NEUTRAL"
GAP_PCT="0.00"

if [ -n "$POLYGON_API_KEY" ]; then
    # Get previous day's close
    PREV_CLOSE=$(curl -s "https://api.polygon.io/v2/aggs/ticker/SPY/prev?apiKey=${POLYGON_API_KEY}" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('results',[{}])[0].get('c',0))" 2>/dev/null)

    # Get current pre-market snapshot
    CURRENT_PRICE=$(curl -s "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/SPY?apiKey=${POLYGON_API_KEY}" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); t=d.get('ticker',{}); print(t.get('lastTrade',{}).get('p',0) or t.get('day',{}).get('c',0))" 2>/dev/null)

    # Calculate gap percentage
    if [ -n "$PREV_CLOSE" ] && [ -n "$CURRENT_PRICE" ] && [ "$PREV_CLOSE" != "0" ]; then
        GAP_PCT=$(python3 -c "print(f'{(($CURRENT_PRICE - $PREV_CLOSE) / $PREV_CLOSE) * 100:.2f}')" 2>/dev/null)

        # Determine bias based on gap
        # > +0.3% = BULLISH, < -0.3% = BEARISH, else NEUTRAL
        if (( $(echo "$GAP_PCT > 0.3" | bc -l) )); then
            PREMARKET_BIAS="BULLISH"
        elif (( $(echo "$GAP_PCT < -0.3" | bc -l) )); then
            PREMARKET_BIAS="BEARISH"
        else
            PREMARKET_BIAS="NEUTRAL"
        fi
    fi
fi

# Write bias to file for CPL to read
echo "$PREMARKET_BIAS" > /tmp/premarket_bias.txt
chmod 644 /tmp/premarket_bias.txt

MSG="🐍 PRE-MARKET CHECK $(date '+%Y-%m-%d %H:%M ET')
wsb-snake: $SNAKE
monitor: $MONITOR
SPY Gap: ${GAP_PCT}% → ${PREMARKET_BIAS}
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
