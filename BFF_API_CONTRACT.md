# WSB Snake - BFF API Contract

## Backend-for-Frontend API Specification

This document defines the API contract between the WSB Snake backend and any frontend clients.

---

## Base URL

```
Development: http://localhost:5000
Production: https://wsb-snake.replit.app
```

---

## Authentication

Currently no authentication required (internal use only).

Future: Add API key header `X-API-Key: <key>`

---

## Endpoints

### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-28T16:30:00-05:00",
  "market_open": true,
  "session": "power_hour"
}
```

---

### 2. System Status

```http
GET /api/status
```

**Response:**
```json
{
  "running": true,
  "market_open": true,
  "eastern_time": "2026-01-28T16:30:00-05:00",
  "session_type": "power_hour",
  "open_positions": 2,
  "signals_today": 5,
  "entries_today": 3,
  "ai_budget": {
    "daily_budget": 5.0,
    "spent_today": 1.25,
    "remaining": 3.75,
    "openai_calls_today": 15,
    "openai_calls_max": 50
  }
}
```

---

### 3. Account Info

```http
GET /api/account
```

**Response:**
```json
{
  "equity": 98693.95,
  "buying_power": 394775.80,
  "cash": 98693.95,
  "portfolio_value": 98693.95,
  "account_status": "ACTIVE",
  "trading_blocked": false,
  "pattern_day_trader": true
}
```

---

### 4. Open Positions

```http
GET /api/positions
```

**Response:**
```json
{
  "positions": [
    {
      "symbol": "SPY",
      "option_symbol": "SPY260128C00602000",
      "trade_type": "CALLS",
      "qty": 2,
      "entry_price": 1.50,
      "current_price": 1.65,
      "target_price": 1.80,
      "stop_loss": 1.27,
      "pnl": 30.00,
      "pnl_pct": 10.0,
      "entry_time": "2026-01-28T15:30:00-05:00",
      "status": "OPEN"
    }
  ],
  "total_pnl": 30.00
}
```

---

### 5. Recent Signals

```http
GET /api/signals?limit=10
```

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| limit | int | 10 | Max signals to return |
| ticker | string | null | Filter by ticker |
| pattern | string | null | Filter by pattern |

**Response:**
```json
{
  "signals": [
    {
      "id": 123,
      "ticker": "SPY",
      "pattern": "vwap_reclaim",
      "direction": "long",
      "entry_price": 602.50,
      "target_price": 603.10,
      "stop_loss": 601.90,
      "confidence": 78,
      "ai_confirmed": true,
      "detected_at": "2026-01-28T15:25:00-05:00",
      "alerted_at": "2026-01-28T15:25:05-05:00"
    }
  ]
}
```

---

### 6. Session Statistics

```http
GET /api/stats
```

**Response:**
```json
{
  "today": {
    "total_trades": 5,
    "winning_trades": 3,
    "losing_trades": 2,
    "win_rate": 60.0,
    "total_pnl": 125.00,
    "avg_pnl_per_trade": 25.00,
    "largest_win": 85.00,
    "largest_loss": -45.00
  },
  "all_time": {
    "total_trades": 47,
    "win_rate": 55.3,
    "total_pnl": 1250.00
  }
}
```

---

### 7. Stalked Setups

```http
GET /api/stalking
```

**Response:**
```json
{
  "setups": [
    {
      "id": "spy_vwap_1706470800",
      "symbol": "SPY",
      "setup_type": "0DTE_vwap_reclaim",
      "trigger_price": 600.00,
      "current_price": 599.50,
      "distance_pct": 0.08,
      "urgency": "HOT",
      "expires_at": "2026-01-28T18:00:00-05:00",
      "entry_price": 600.00,
      "target_price": 601.00,
      "stop_loss": 599.50
    }
  ]
}
```

---

### 8. Close Position (Manual)

```http
POST /api/positions/close
```

**Request Body:**
```json
{
  "option_symbol": "SPY260128C00602000"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Position closed successfully",
  "exit_price": 1.75,
  "pnl": 50.00
}
```

---

### 9. Close All Positions (Emergency)

```http
POST /api/positions/close-all
```

**Response:**
```json
{
  "success": true,
  "positions_closed": 3,
  "total_pnl": 85.00
}
```

---

### 10. Update Configuration

```http
PATCH /api/config
```

**Request Body:**
```json
{
  "min_confidence_for_alert": 60,
  "high_confidence_auto_execute": 70,
  "max_per_trade": 1500,
  "max_daily_exposure": 6000
}
```

**Response:**
```json
{
  "success": true,
  "config": {
    "min_confidence_for_alert": 60,
    "high_confidence_auto_execute": 70,
    "max_per_trade": 1500,
    "max_daily_exposure": 6000
  }
}
```

---

## WebSocket API

### Real-time Updates

```
WS /ws/updates
```

**Message Types:**

1. **Signal Detected**
```json
{
  "type": "signal",
  "data": {
    "ticker": "SPY",
    "pattern": "vwap_reclaim",
    "confidence": 78,
    "direction": "long"
  }
}
```

2. **Trade Executed**
```json
{
  "type": "trade_executed",
  "data": {
    "ticker": "SPY",
    "option_symbol": "SPY260128C00602000",
    "direction": "long",
    "entry_price": 1.50
  }
}
```

3. **Position Update**
```json
{
  "type": "position_update",
  "data": {
    "option_symbol": "SPY260128C00602000",
    "current_price": 1.65,
    "pnl": 30.00,
    "pnl_pct": 10.0
  }
}
```

4. **Position Closed**
```json
{
  "type": "position_closed",
  "data": {
    "option_symbol": "SPY260128C00602000",
    "exit_price": 1.80,
    "pnl": 60.00,
    "reason": "TARGET_HIT"
  }
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": true,
  "code": "POSITION_NOT_FOUND",
  "message": "No position found with symbol SPY260128C00602000"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| MARKET_CLOSED | 400 | Market is not open |
| INVALID_SYMBOL | 400 | Invalid option symbol |
| POSITION_NOT_FOUND | 404 | Position does not exist |
| MAX_POSITIONS | 429 | Maximum concurrent positions reached |
| DAILY_LIMIT_EXCEEDED | 429 | Daily exposure limit reached |
| INTERNAL_ERROR | 500 | Unexpected server error |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| GET endpoints | 60/minute |
| POST endpoints | 10/minute |
| WebSocket | 1 connection per client |
