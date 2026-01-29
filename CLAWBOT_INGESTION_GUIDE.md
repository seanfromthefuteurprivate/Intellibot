# WSB Snake - ClawBot Ingestion Guide

## Data Ingestion Architecture

This guide covers how data flows into WSB Snake from external sources.

---

## Data Sources Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL DATA SOURCES                         │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Polygon  │  │  Alpaca  │  │ Finnhub  │  │   FRED   │        │
│  │   .io    │  │   News   │  │          │  │          │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │                │
│       ▼             ▼             ▼             ▼                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    COLLECTORS LAYER                      │    │
│  │                                                          │    │
│  │  polygon_enhanced.py  │  alpaca_news.py  │  finnhub.py  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    DATA NORMALIZATION                    │    │
│  │                                                          │    │
│  │  ScalpDataPacket  │  NewsArticle  │  EarningsData       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      ENGINES                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Polygon.io Integration

### Collector: `polygon_enhanced.py`

**Data Types:**
| Type | Endpoint | Frequency |
|------|----------|-----------|
| Bars (5s) | `/v2/aggs/ticker/{ticker}/range/5/second` | Real-time |
| Bars (15s) | `/v2/aggs/ticker/{ticker}/range/15/second` | Real-time |
| Bars (1m) | `/v2/aggs/ticker/{ticker}/range/1/minute` | Real-time |
| Bars (5m) | `/v2/aggs/ticker/{ticker}/range/5/minute` | Real-time |
| NBBO Quotes | `/v3/quotes/{ticker}` | On demand |
| Options Chain | `/v3/reference/options/contracts` | On demand |
| Recent Trades | `/v3/trades/{ticker}` | On demand |

**Authentication:**
```python
headers = {
    "Authorization": f"Bearer {POLYGON_API_KEY}"
}
```

**Rate Limits:**
- Free tier: 5 calls/minute
- Starter: 100 calls/minute
- Developer: Unlimited

**Normalized Output:**
```python
@dataclass
class ScalpDataPacket:
    ticker: str
    bars_5s: List[Dict]    # Last 30 bars
    bars_15s: List[Dict]   # Last 20 bars
    bars_1m: List[Dict]    # Last 60 bars
    bars_5m: List[Dict]    # Last 12 bars
    vwap: float
    volume_ratio: float
    momentum: float
    current_price: float
    bid: float
    ask: float
    spread: float
    timestamp: datetime
```

---

## 2. Alpaca Integration

### Collector: `alpaca_news.py`

**Data Types:**
| Type | Endpoint | Frequency |
|------|----------|-----------|
| News | `/v1beta1/news` | On demand |
| Ticker News | `/v1beta1/news?symbols={ticker}` | Per pattern |

**Authentication:**
```python
headers = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
}
```

**Rate Limits:**
- 200 calls/minute

**Normalized Output:**
```python
{
    "headline": "Apple announces new product line",
    "summary": "Apple Inc unveiled...",
    "author": "Reuters",
    "created_at": "2026-01-28T15:30:00Z",
    "source": "reuters",
    "symbols": ["AAPL"],
    "url": "https://..."
}
```

### Collector: `alpaca_stream.py` (WebSocket)

**Stream Types:**
| Stream | Data |
|--------|------|
| trades | Real-time trade executions |
| quotes | Bid/ask updates |
| bars | Minute bars as they form |
| news | Breaking news |

**Connection:**
```python
wss://stream.data.alpaca.markets/v2/iex
```

---

## 3. Finnhub Integration

### Collector: `finnhub_collector.py`

**Data Types:**
| Type | Endpoint | Frequency |
|------|----------|-----------|
| Quote | `/quote` | On demand |
| Company News | `/company-news` | Daily |
| Earnings Calendar | `/calendar/earnings` | Daily |
| Recommendation Trends | `/stock/recommendation` | Weekly |
| Price Targets | `/stock/price-target` | Weekly |
| Technical Indicators | `/scan/support-resistance` | On demand |
| Insider Sentiment | `/stock/insider-sentiment` | Weekly |

**Authentication:**
```python
params = {"token": FINNHUB_API_KEY}
```

**Rate Limits:**
- Free: 60 calls/minute
- Premium: 300 calls/minute

**Normalized Output:**
```python
{
    "earnings_soon": True,
    "days_until_earnings": 3,
    "expected_eps": 1.25,
    "recommendation": "buy",
    "recommendation_score": 3.8,  # 1-5 scale
    "price_target_mean": 185.00,
    "support_levels": [175.00, 170.00],
    "resistance_levels": [185.00, 190.00]
}
```

---

## 4. FRED Integration

### Collector: `fred_collector.py`

**Data Series:**
| Series ID | Description |
|-----------|-------------|
| DFF | Federal Funds Rate |
| VIXCLS | VIX Close |
| T10Y2Y | 10Y-2Y Treasury Spread |
| UNRATE | Unemployment Rate |
| CPIAUCSL | CPI All Urban |

**Authentication:**
```python
params = {"api_key": FRED_API_KEY}
```

**Rate Limits:**
- 120 calls/minute

---

## 5. Data Freshness Requirements

| Data Type | Max Staleness | Refresh Trigger |
|-----------|---------------|-----------------|
| Price bars | 5 seconds | Continuous scan |
| VWAP | 30 seconds | Each scan cycle |
| News | 5 minutes | On pattern detection |
| Earnings calendar | 24 hours | Daily pre-market |
| Macro data | 24 hours | Daily pre-market |

---

## 6. Error Handling

### Retry Strategy
```python
MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 5]  # seconds

for attempt, delay in enumerate(RETRY_DELAYS):
    try:
        response = fetch_data()
        break
    except RateLimitError:
        sleep(delay)
    except APIError as e:
        log.error(f"API error attempt {attempt}: {e}")
```

### Fallback Chain
```
Primary: Polygon.io bars
    ↓ (if fails)
Fallback: Alpaca bars
    ↓ (if fails)
Cache: Last known values
    ↓ (if stale > 5 min)
Skip: Mark ticker unavailable
```

---

## 7. Data Validation

All ingested data is validated:

```python
def validate_bar(bar: Dict) -> bool:
    required = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    
    # Check required fields
    if not all(k in bar for k in required):
        return False
    
    # Check price sanity
    if bar['high'] < bar['low']:
        return False
    
    if bar['close'] < 0 or bar['open'] < 0:
        return False
    
    # Check volume sanity
    if bar['volume'] < 0:
        return False
    
    return True
```

---

## 8. Ingestion Metrics

Track ingestion health:

```python
ingestion_metrics = {
    "polygon_calls_1m": 45,
    "polygon_errors_1m": 2,
    "polygon_latency_avg_ms": 125,
    
    "alpaca_calls_1m": 10,
    "alpaca_errors_1m": 0,
    
    "finnhub_calls_1m": 5,
    "finnhub_errors_1m": 0,
    
    "data_freshness_pct": 98.5,
    "tickers_stale": ["BBBY"]  # If any
}
```

---

## 9. Secret Management

All API keys stored in Replit Secrets:

| Secret Name | Service |
|-------------|---------|
| `POLYGON_API_KEY` | Polygon.io |
| `ALPACA_API_KEY` | Alpaca |
| `ALPACA_SECRET_KEY` | Alpaca |
| `FINNHUB_API_KEY` | Finnhub |
| `FRED_API_KEY` | FRED (optional) |

**Access:**
```python
import os
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")
```
