# WSB Snake - System Operator Handbook

## Purpose
This handbook provides operational guidance for running the WSB Snake 0DTE options trading engine. It covers startup procedures, monitoring, troubleshooting, and emergency procedures.

---

## 1. System Overview

WSB Snake is an autonomous 0DTE options scalping engine that:
- Scans 29 tickers every 30 seconds during market hours
- Uses AI (OpenAI GPT-4o + DeepSeek) for pattern confirmation
- Auto-executes trades on Alpaca paper trading account
- Monitors positions and auto-exits at targets/stops
- Sends all alerts via Telegram

**Operating Hours:** 9:30 AM - 4:00 PM ET (market hours)
**After Hours:** System sleeps, no trading

---

## 2. Startup Procedures

### 2.1 Normal Startup
```bash
cd /home/runner/workspace
python3 main.py
```

### 2.2 Background Startup (Production)
```bash
nohup python3 main.py > /tmp/wsb_snake.log 2>&1 &
```

### 2.3 Verify Startup
```bash
# Check process is running
ps aux | grep "python3 main.py"

# Check logs
tail -50 /tmp/wsb_snake.log

# Verify ET time is correct
python3 -c "from wsb_snake.utils.session_regime import get_eastern_time; print(get_eastern_time())"
```

---

## 3. Key Operational Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Alert Threshold | 60% | spy_scalper.py |
| Auto-Execute Threshold | 70% | spy_scalper.py |
| Max Per Trade | $1,500 | alpaca_executor.py |
| Max Daily Exposure | $6,000 | alpaca_executor.py |
| Max Concurrent Positions | 5 | alpaca_executor.py |
| Target Exit | +20% | alpaca_executor.py |
| Stop Loss | -15% | alpaca_executor.py |
| Time Decay Exit | 45 minutes | alpaca_executor.py |
| EOD Close | 3:55 PM ET | alpaca_executor.py |

---

## 4. Monitoring

### 4.1 Live Log Monitoring
```bash
tail -f /tmp/wsb_snake.log
```

### 4.2 Key Log Messages to Watch
- `🎯 APEX PREDATOR STRIKE` - Trade signal detected
- `📈 Alpaca AUTO-EXECUTE` - Trade placed
- `✅ ORDER FILLED` - Trade confirmed
- `🎯 TARGET HIT` - Profit taken
- `STOP LOSS` - Loss cut
- `❌ Rate limited` - AI calls throttled

### 4.3 API Budget Check
```python
from wsb_snake.analysis.predator_stack import predator_stack
print(predator_stack.get_budget_status())
```

---

## 5. Troubleshooting

### 5.1 No Trades Firing
1. Check market is open: `is_market_open()` should return True
2. Check confidence thresholds aren't too high
3. Verify AI API keys are set in Secrets
4. Check rate limiting hasn't blocked calls

### 5.2 AI Not Working
```python
from wsb_snake.analysis.predator_stack import predator_stack
print(f"OpenAI: {predator_stack.openai_available}")
print(f"DeepSeek: {predator_stack.deepseek_available}")
```

### 5.3 Alpaca Connection Issues
```python
from wsb_snake.trading.alpaca_executor import alpaca_executor
print(alpaca_executor.get_account_info())
```

---

## 6. Emergency Procedures

### 6.1 Stop All Trading Immediately
```bash
pkill -f "python3 main.py"
```

### 6.2 Close All Positions
```python
from wsb_snake.trading.alpaca_executor import alpaca_executor
alpaca_executor.close_all_0dte_positions()
```

### 6.3 Manual Position Close
```python
alpaca_executor.close_position("SPY260128C00600000")
```

---

## 7. Daily Checklist

### Pre-Market (Before 9:30 AM ET)
- [ ] Verify system is running
- [ ] Check Alpaca account balance
- [ ] Verify API keys are valid
- [ ] Review previous day's performance

### During Market Hours
- [ ] Monitor Telegram for alerts
- [ ] Check for error messages in logs
- [ ] Verify positions are being tracked

### Post-Market (After 4:00 PM ET)
- [ ] Confirm all 0DTE positions closed
- [ ] Review session statistics
- [ ] Check API usage/budget

---

## 8. Secret Management

Required secrets in Replit Secrets:
- `ALPACA_API_KEY` - Alpaca trading API key
- `ALPACA_SECRET_KEY` - Alpaca secret key
- `OPENAI_API_KEY` - OpenAI GPT-4o for chart vision
- `DEEPSEEK_API_KEY` - DeepSeek for news analysis
- `TELEGRAM_BOT_TOKEN` - Telegram bot token
- `TELEGRAM_CHAT_ID` - Your Telegram chat ID
- `POLYGON_API_KEY` - Polygon.io market data
- `FINNHUB_API_KEY` - Finnhub news/sentiment

---

## 9. Contact & Escalation

For critical issues:
1. Stop the bot immediately
2. Close all positions manually on Alpaca
3. Review logs for root cause
4. Document the incident
