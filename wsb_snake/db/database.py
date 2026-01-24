"""
SQLite Database for Signal Persistence

Stores:
- Signals and their features
- Trade outcomes
- Model weights for learning
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from wsb_snake.config import DB_PATH
from wsb_snake.utils.logger import log


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize database schema."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Signals table - stores every alert with features
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            setup_type TEXT NOT NULL,
            score REAL NOT NULL,
            tier TEXT NOT NULL,
            
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
        )
    """)
    
    # Outcomes table - tracks what happened after each signal
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
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
            outcome_type TEXT,  -- "win", "loss", "scratch", "timeout"
            notes TEXT,
            
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (signal_id) REFERENCES signals(id)
        )
    """)
    
    # Paper trades table - simulated executions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            
            -- Trade plan
            direction TEXT NOT NULL,    -- "long" or "short"
            entry_trigger TEXT,
            entry_price REAL,
            stop_price REAL,
            target_1_price REAL,
            target_2_price REAL,
            position_size INTEGER,
            
            -- Execution
            status TEXT DEFAULT 'pending',  -- pending, open, closed
            fill_price REAL,
            fill_time TEXT,
            exit_price REAL,
            exit_time TEXT,
            exit_reason TEXT,
            
            -- P&L
            pnl REAL,
            r_multiple REAL,
            
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (signal_id) REFERENCES signals(id)
        )
    """)
    
    # Model weights table - for learning
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature_name TEXT NOT NULL UNIQUE,
            weight REAL DEFAULT 1.0,
            hit_count INTEGER DEFAULT 0,
            miss_count INTEGER DEFAULT 0,
            last_updated TEXT,
            
            -- Context-specific weights
            weight_power_hour REAL DEFAULT 1.0,
            weight_trend_regime REAL DEFAULT 1.0,
            weight_range_regime REAL DEFAULT 1.0
        )
    """)
    
    # Daily summaries table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL UNIQUE,
            
            -- Stats
            total_signals INTEGER DEFAULT 0,
            alerts_sent INTEGER DEFAULT 0,
            paper_trades INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            scratches INTEGER DEFAULT 0,
            
            -- Performance
            win_rate REAL,
            avg_r_multiple REAL,
            total_r REAL,
            
            -- Best/Worst
            best_ticker TEXT,
            best_r REAL,
            worst_ticker TEXT,
            worst_r REAL,
            
            -- Notes
            regime_notes TEXT,
            
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    log.info("Database initialized")


def save_signal(signal_data: Dict) -> int:
    """
    Save a signal to the database.
    
    Returns:
        The inserted signal ID
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO signals (
            ticker, timestamp, setup_type, score, tier,
            price, volume, change_pct, vwap, range_pct,
            atm_iv, call_put_ratio, top_strike, options_pressure_score,
            social_velocity, sentiment_score,
            session_type, minutes_to_close,
            features_json, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal_data.get("ticker"),
        signal_data.get("timestamp", datetime.utcnow().isoformat()),
        signal_data.get("setup_type", "UNKNOWN"),
        signal_data.get("score", 0),
        signal_data.get("tier", "C"),
        signal_data.get("price"),
        signal_data.get("volume"),
        signal_data.get("change_pct"),
        signal_data.get("vwap"),
        signal_data.get("range_pct"),
        signal_data.get("atm_iv"),
        signal_data.get("call_put_ratio"),
        signal_data.get("top_strike"),
        signal_data.get("options_pressure_score"),
        signal_data.get("social_velocity"),
        signal_data.get("sentiment_score"),
        signal_data.get("session_type"),
        signal_data.get("minutes_to_close"),
        json.dumps(signal_data.get("features", {})),
        json.dumps(signal_data.get("evidence", [])),
    ))
    
    signal_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    log.debug(f"Saved signal {signal_id} for {signal_data.get('ticker')}")
    return signal_id


def save_paper_trade(trade_data: Dict) -> int:
    """Save a paper trade to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO paper_trades (
            signal_id, ticker, direction,
            entry_trigger, entry_price, stop_price,
            target_1_price, target_2_price, position_size, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_data.get("signal_id"),
        trade_data.get("ticker"),
        trade_data.get("direction", "long"),
        trade_data.get("entry_trigger"),
        trade_data.get("entry_price"),
        trade_data.get("stop_price"),
        trade_data.get("target_1_price"),
        trade_data.get("target_2_price"),
        trade_data.get("position_size", 100),
        "pending",
    ))
    
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return trade_id


def get_recent_signals(limit: int = 50) -> List[Dict]:
    """Get recent signals from database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM signals
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_daily_stats(date_str: str = None) -> Dict:
    """Get stats for a specific day."""
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Count signals for the day
    cursor.execute("""
        SELECT COUNT(*) as count FROM signals
        WHERE date(timestamp) = ?
    """, (date_str,))
    signal_count = cursor.fetchone()["count"]
    
    # Count paper trades
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN r_multiple < 0 THEN 1 ELSE 0 END) as losses,
            AVG(r_multiple) as avg_r,
            SUM(r_multiple) as total_r
        FROM paper_trades
        WHERE date(created_at) = ? AND status = 'closed'
    """, (date_str,))
    
    trade_stats = cursor.fetchone()
    conn.close()
    
    return {
        "date": date_str,
        "signals": signal_count,
        "trades": trade_stats["total"] or 0,
        "wins": trade_stats["wins"] or 0,
        "losses": trade_stats["losses"] or 0,
        "avg_r": trade_stats["avg_r"] or 0,
        "total_r": trade_stats["total_r"] or 0,
        "win_rate": (trade_stats["wins"] / trade_stats["total"]) if trade_stats["total"] else 0,
    }


# Initialize on import
try:
    init_database()
except Exception as e:
    log.error(f"Failed to initialize database: {e}")
