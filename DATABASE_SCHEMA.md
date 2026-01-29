# WSB Snake - Database Schema

## Data Truth Source

All persistent data is stored in SQLite at `wsb_snake_data/wsb_snake.db`.

---

## Tables Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATABASE SCHEMA                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  signals ──────────< outcomes                                   │
│     │                                                            │
│     │                                                            │
│  spy_scalp_history                                              │
│                                                                  │
│  pattern_memory                                                 │
│                                                                  │
│  time_performance                                               │
│                                                                  │
│  event_outcomes                                                 │
│                                                                  │
│  stalked_setups                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Table: signals

Stores every detected signal with full feature set.

```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    score REAL NOT NULL,
    tier TEXT NOT NULL,  -- 'A+', 'A', 'B', 'C'
    
    -- Market features
    price REAL,
    volume INTEGER,
    change_pct REAL,
    vwap REAL,
    range_pct REAL,
    
    -- Options features
    atm_iv REAL,
    call_put_ratio REAL,
    top_strike REAL,
    options_pressure_score REAL,
    
    -- Social features
    social_velocity REAL,
    sentiment_score REAL,
    
    -- Session context
    session_type TEXT,
    minutes_to_close REAL,
    
    -- Full feature blob (JSON)
    features_json TEXT,
    
    -- Evidence
    evidence_json TEXT,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_signals_ticker ON signals(ticker);
CREATE INDEX idx_signals_timestamp ON signals(timestamp);
CREATE INDEX idx_signals_tier ON signals(tier);
```

---

## Table: outcomes

Tracks what happened after each signal.

```sql
CREATE TABLE outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL,
    
    -- Price outcomes
    entry_price REAL,
    max_price REAL,      -- MFE (max favorable excursion)
    min_price REAL,      -- MAE (max adverse excursion)
    exit_price REAL,
    
    -- Timing
    time_to_target_1 INTEGER,   -- seconds to hit target 1
    time_to_target_2 INTEGER,   -- seconds to hit target 2
    time_to_stop INTEGER,       -- seconds to hit stop (if any)
    
    -- Result
    hit_target_1 INTEGER DEFAULT 0,
    hit_target_2 INTEGER DEFAULT 0,
    hit_stop INTEGER DEFAULT 0,
    r_multiple REAL,
    
    -- Meta
    outcome_type TEXT,  -- 'win', 'loss', 'scratch', 'timeout'
    notes TEXT,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

CREATE INDEX idx_outcomes_signal ON outcomes(signal_id);
CREATE INDEX idx_outcomes_type ON outcomes(outcome_type);
```

---

## Table: spy_scalp_history

Records SPY scalper signals and outcomes.

```sql
CREATE TABLE spy_scalp_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL,
    direction TEXT NOT NULL,  -- 'long' or 'short'
    entry_price REAL NOT NULL,
    target_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    confidence REAL NOT NULL,
    ai_confirmed INTEGER DEFAULT 0,
    ai_confidence REAL,
    detected_at TEXT NOT NULL,
    alerted_at TEXT,
    
    -- Outcome (filled later)
    exit_price REAL,
    pnl REAL,
    outcome TEXT,  -- 'win', 'loss', 'timeout'
    exit_reason TEXT,
    exited_at TEXT,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_scalp_pattern ON spy_scalp_history(pattern);
CREATE INDEX idx_scalp_outcome ON spy_scalp_history(outcome);
```

---

## Table: pattern_memory

Stores successful price action patterns for similarity matching.

```sql
CREATE TABLE pattern_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,  -- 'breakout', 'squeeze', 'reversal', 'momentum'
    
    -- Price action signature (normalized)
    price_action_json TEXT NOT NULL,  -- Array of normalized candle data
    volume_profile_json TEXT NOT NULL, -- Volume signature
    
    -- Outcome
    success INTEGER NOT NULL,  -- 1 = win, 0 = loss
    profit_pct REAL,
    
    -- Context
    ticker TEXT,
    session_type TEXT,
    timestamp TEXT,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pattern_type ON pattern_memory(pattern_type);
CREATE INDEX idx_pattern_success ON pattern_memory(success);
```

---

## Table: time_performance

Tracks performance by hour and session.

```sql
CREATE TABLE time_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hour INTEGER NOT NULL,  -- 0-23
    day_of_week INTEGER,    -- 0=Monday, 6=Sunday
    
    -- Stats
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0,
    avg_pnl REAL DEFAULT 0,
    
    -- Calculated
    win_rate REAL DEFAULT 0,
    quality_score INTEGER DEFAULT 50,  -- 0-100
    
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(hour, day_of_week)
);
```

---

## Table: event_outcomes

Records market reactions to events (earnings, CPI, FOMC).

```sql
CREATE TABLE event_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,  -- 'earnings', 'cpi', 'fomc', 'jobs'
    ticker TEXT,               -- NULL for macro events
    event_date TEXT NOT NULL,
    
    -- Pre-event
    pre_event_price REAL,
    pre_event_iv REAL,
    
    -- Post-event
    post_event_price REAL,
    post_event_iv REAL,
    move_pct REAL,
    move_direction TEXT,  -- 'up', 'down', 'flat'
    
    -- Expectations
    expected_move REAL,
    beat_expectations INTEGER,  -- 1 = beat, 0 = miss, NULL = in-line
    
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_event_type ON event_outcomes(event_type);
CREATE INDEX idx_event_ticker ON event_outcomes(ticker);
```

---

## Table: stalked_setups

Tracks setups approaching trigger points.

```sql
CREATE TABLE stalked_setups (
    id TEXT PRIMARY KEY,  -- Format: {ticker}_{type}_{timestamp}
    symbol TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    
    -- Trigger
    trigger_price REAL NOT NULL,
    trigger_condition TEXT,
    
    -- Trade params
    entry_price REAL,
    target_price REAL,
    stop_loss REAL,
    direction TEXT,
    trade_type TEXT,  -- 'CALLS' or 'PUTS'
    
    -- Status
    urgency TEXT DEFAULT 'DORMANT',  -- DORMANT, WATCHING, HEATING, HOT, IMMINENT
    entry_alerted INTEGER DEFAULT 0,
    
    -- Context
    catalyst TEXT,
    expected_move REAL,
    expires_at TEXT,
    
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_stalk_symbol ON stalked_setups(symbol);
CREATE INDEX idx_stalk_urgency ON stalked_setups(urgency);
```

---

## JSON File: session_learnings.json

Battle-tested wisdom stored in `wsb_snake_data/session_learnings.json`.

```json
{
  "battle_plan": {
    "min_confidence_for_alert": 60,
    "high_confidence_auto_execute": 70,
    "max_per_trade": 1500,
    "max_daily_exposure": 6000,
    "max_concurrent_positions": 5,
    "stop_atr_multiplier_min": 0.8,
    "stop_atr_multiplier_max": 1.2,
    "target_atr_multiplier_min": 2.5,
    "target_atr_multiplier_max": 3.5,
    "eod_close_time": "15:55"
  },
  "daily_learnings": [
    {
      "date": "2026-01-27",
      "lesson": "Stops too tight at 0.3% ATR - noise triggers. Widened to 0.8-1.2% ATR",
      "trades": ["AAPL +$35", "QQQ -$62", "SPY -$23"],
      "net_pnl": -82
    }
  ]
}
```

---

## Data Retention

| Table | Retention |
|-------|-----------|
| signals | 90 days |
| outcomes | 90 days |
| spy_scalp_history | 180 days |
| pattern_memory | 1 year |
| time_performance | Permanent |
| event_outcomes | 2 years |
| stalked_setups | Auto-purge on expiry |

---

## Backup Strategy

```bash
# Daily backup
cp wsb_snake_data/wsb_snake.db wsb_snake_data/backups/wsb_snake_$(date +%Y%m%d).db

# Keep last 30 days
find wsb_snake_data/backups -name "*.db" -mtime +30 -delete
```
