# Polygon Health Check - Quick Reference

## Health Check Function

```python
from wsb_snake.utils.polygon_health import polygon_health_check

# Quick check (used by CPL)
is_healthy, reason = polygon_health_check()
if not is_healthy:
    print(f"POLYGON DOWN: {reason}")
    # DO NOT TRADE
```

## Monitoring Endpoints

```bash
# Health status
curl http://157.245.240.99:8080/api/health/polygon

# System health (all services)
curl http://157.245.240.99:8080/api/health/system

# Basic liveness check
curl http://157.245.240.99:8080/api/health/
```

## Rate Limit Configuration

```python
# Environment variable (required)
POLYGON_PLAN=starter  # or 'free' or 'developer'

# Rate limits per plan:
# - free: 5 calls/min
# - starter: 100 calls/min
# - developer: 300 calls/min
```

## Cache TTL Configuration

```python
CACHE_TTLS = {
    "snapshot": 30,      # 30s - price snapshots (CRITICAL)
    "bars": 30,          # 30s - OHLC bars (CRITICAL)
    "trades": 60,        # 1min - trade flow
    "quotes": 60,        # 1min - NBBO quotes
    "options": 120,      # 2min - options chain
    "technicals": 300,   # 5min - RSI/MACD/SMA (SAFE TO EXTEND)
    "indicators": 300,   # 5min - technical indicators (SAFE TO EXTEND)
    "reference": 600,    # 10min - static reference data (SAFE TO EXTEND)
}
```

## Circuit Breaker Behavior

1. **Triggers:** 3 consecutive API failures (429, 403, 5xx)
2. **Cooldown:** 60 seconds before retry
3. **Effect:** ALL Polygon requests blocked
4. **CPL Response:** REFUSES to trade, logs "POLYGON_HEALTH_FAIL"

## Log Patterns

```bash
# Initialization
"PolygonHealthMonitor initialized (plan=starter, rate_limit=100/min)"

# Healthy operation
"CACHE_HIT: /v2/snapshot/SPY:..."
"CACHE_SET: /v2/snapshot/SPY:... (ttl=30s)"

# Rate limit warning
"POLYGON_429: Rate limit exceeded (plan=starter)"
"RATE_LIMIT: 100/100 calls/min (plan=starter)"

# Circuit breaker
"POLYGON_DEAD: Circuit breaker tripped after 3 failures. Cooldown until 14:35:00"
"POLYGON_CIRCUIT_BREAKER: Reset, attempting reconnection"

# CPL rejection
"POLYGON_HEALTH_FAIL: CIRCUIT_BREAKER - REFUSING TRADE SPY CALL"
"ENTRY_V2_REJECT: SPY CALL - Polygon unhealthy: RATE_LIMIT"
```

## Debugging Commands

```bash
# Check Polygon health status
curl -s http://157.245.240.99:8080/api/health/polygon | jq .

# Check rate limit usage
curl -s http://157.245.240.99:8080/api/health/polygon | jq '.status.calls_this_minute'

# Check circuit breaker status
curl -s http://157.245.240.99:8080/api/health/polygon | jq '.status.circuit_breaker_active'

# Test Polygon API directly
curl "https://api.polygon.io/v2/aggs/ticker/SPY/prev?apiKey=YOUR_KEY"

# Watch live logs
ssh root@157.245.240.99 "journalctl -u wsb-snake.service -f | grep POLYGON"
```

## Manual Intervention

```python
# Reset circuit breaker (if needed)
from wsb_snake.utils.polygon_health import get_polygon_monitor

monitor = get_polygon_monitor()
monitor.status.consecutive_failures = 0
monitor.status.is_healthy = True
monitor._circuit_breaker_until = None

# Clear cache (force refresh)
monitor.clear_cache()

# Check status
print(monitor.get_status())
```

## Production Checklist

- [ ] `POLYGON_PLAN` set in `.env`
- [ ] `POLYGON_API_KEY` is valid (test with curl)
- [ ] Dashboard health endpoint accessible
- [ ] CPL logs show "PolygonHealthMonitor initialized"
- [ ] No "POLYGON_HEALTH_FAIL" during normal operation
- [ ] Rate limit not exceeded (`calls_this_minute < rate_limit`)
- [ ] Telegram alerts configured (optional)

## Emergency Contacts

- Polygon Status: https://status.polygon.io/
- API Docs: https://polygon.io/docs
- Support: support@polygon.io
- Billing: https://polygon.io/dashboard

## Common Issues

### Issue: "POLYGON_403: Authentication failed"
**Fix:** Check API key validity, verify subscription active

### Issue: "POLYGON_429: Rate limit exceeded"
**Fix:** Increase plan tier OR reduce scan frequency OR check HYDRA overlap

### Issue: Circuit breaker stuck active
**Fix:** Wait 60s OR manually reset OR restart service

### Issue: CPL still trading with no data
**Fix:** Verify `_check_entry_quality` includes health check (should reject when Polygon down)
