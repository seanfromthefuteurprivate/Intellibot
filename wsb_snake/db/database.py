"""
SQLite Database for Signal Persistence

Stores:
- Signals and their features
- Trade outcomes
- Model weights for learning
"""

import sqlite3
import json
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from wsb_snake.config import DB_PATH
from wsb_snake.utils.logger import log

# Serialize writes to avoid SQLITE_BUSY under concurrent load
_write_lock = threading.Lock()
_CONNECTION_TIMEOUT = 30


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH, timeout=_CONNECTION_TIMEOUT)
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

    # Trade performance table - granular analytics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            engine TEXT,
            trade_type TEXT,
            entry_hour INTEGER,
            session TEXT,
            pnl REAL,
            pnl_pct REAL,
            r_multiple REAL,
            exit_reason TEXT,
            holding_time_seconds INTEGER,
            signal_id INTEGER,
            event_tier TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Backfill columns if table existed without them
    for col, typ in [("r_multiple", "REAL"), ("event_tier", "TEXT")]:
        try:
            cursor.execute(f"ALTER TABLE trade_performance ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass

    # CPL (Convexity Proof Layer) calls - Jobs Day atomic trade calls, dedupe by key
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cpl_calls (
            call_id TEXT PRIMARY KEY,
            timestamp_et TEXT NOT NULL,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            strike REAL NOT NULL,
            expiry TEXT NOT NULL,
            dedupe_key TEXT UNIQUE NOT NULL,
            regime TEXT,
            confidence REAL,
            alerted_at TEXT,
            full_json TEXT
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_cpl_dedupe ON cpl_calls(dedupe_key)"
    )

    # ─────────────────────────────────────────────────────────────────
    # Apex Governance Layer Tables
    # ─────────────────────────────────────────────────────────────────

    # Governance events table - tracks state transitions and decisions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS governance_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            dedupe_key TEXT,
            ticker TEXT,
            state_from TEXT,
            state_to TEXT,
            pnl_at_event REAL,
            reason TEXT,
            metadata_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_gov_dedupe ON governance_events(dedupe_key)"
    )

    # Counterfactual ledger table - what-if analysis for runners
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS counterfactual_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id TEXT NOT NULL,
            timestamp_et TEXT NOT NULL,
            ticker TEXT NOT NULL,
            checkpoint_type TEXT,
            actual_pnl_pct REAL,
            counterfactual_exit_pnl REAL,
            runner_outcome_pnl REAL,
            missed_upside_pct REAL,
            scenario_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_cf_call ON counterfactual_ledger(call_id)"
    )

    # Preregistration locks table - axiom freezing before sessions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS preregistration_locks (
            session_id TEXT PRIMARY KEY,
            locked_at TEXT NOT NULL,
            axioms_json TEXT NOT NULL,
            state_definitions_json TEXT NOT NULL,
            prohibited_behaviors_json TEXT NOT NULL,
            telemetry_schema_json TEXT NOT NULL,
            hash_signature TEXT
        )
    """)

    # ─────────────────────────────────────────────────────────────────
    # Screenshot Learning Tables
    # ─────────────────────────────────────────────────────────────────

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screenshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            mime_type TEXT,
            file_size INTEGER,
            content_hash TEXT,
            status TEXT DEFAULT 'pending',
            extracted_data TEXT,
            learned_trade_id INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learned_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            screenshot_id INTEGER,
            ticker TEXT NOT NULL,
            trade_type TEXT,
            strike REAL,
            entry_price REAL,
            exit_price REAL,
            contracts INTEGER,
            profit_loss REAL,
            profit_loss_pct REAL,
            detected_pattern TEXT,
            setup_description TEXT,
            confidence_score REAL,
            replication_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (screenshot_id) REFERENCES screenshots(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            ticker_pattern TEXT,
            trade_type TEXT,
            time_window TEXT,
            min_confidence REAL,
            entry_conditions TEXT,
            exit_conditions TEXT,
            source_trade_count INTEGER DEFAULT 0,
            total_profit REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            avg_profit_pct REAL DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            last_used TEXT,
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
    with _write_lock:
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
    with _write_lock:
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
    """Get stats for a specific day (paper_trades only)."""
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


def get_daily_stats_for_report(date_str: str = None) -> Dict:
    """
    Full daily stats for 4:15 ET report: aggregates paper_trades + trade_performance.
    Includes win rate, avg R, total R, best/worst setups, and 2X/4X/20X event counts.
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
    conn = get_connection()
    cursor = conn.cursor()

    # From trade_performance (includes CPL and scalper)
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
            AVG(r_multiple) as avg_r,
            SUM(r_multiple) as total_r
        FROM trade_performance
        WHERE trade_date = ?
    """, (date_str,))
    tp = cursor.fetchone()
    total = tp["total"] or 0
    wins = tp["wins"] or 0
    losses = tp["losses"] or 0
    avg_r = tp["avg_r"] or 0
    total_r = tp["total_r"] or 0

    # If no trade_performance, fall back to paper_trades
    if total == 0:
        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN r_multiple < 0 THEN 1 ELSE 0 END) as losses,
                   AVG(r_multiple) as avg_r, SUM(r_multiple) as total_r
            FROM paper_trades WHERE date(created_at) = ? AND status = 'closed'
        """, (date_str,))
        row = cursor.fetchone()
        total = row["total"] or 0
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        avg_r = row["avg_r"] or 0
        total_r = row["total_r"] or 0

    # Best/worst by symbol (trade_performance)
    cursor.execute("""
        SELECT symbol, SUM(pnl) as pnl_sum, AVG(r_multiple) as avg_r
        FROM trade_performance WHERE trade_date = ?
        GROUP BY symbol ORDER BY pnl_sum DESC
    """, (date_str,))
    by_symbol = [dict(r) for r in cursor.fetchall()]
    best_ticker = by_symbol[0]["symbol"] if by_symbol and by_symbol[0]["pnl_sum"] and by_symbol[0]["pnl_sum"] > 0 else None
    best_r = by_symbol[0]["avg_r"] if best_ticker else None
    worst_ticker = by_symbol[-1]["symbol"] if by_symbol and by_symbol[-1]["pnl_sum"] is not None and by_symbol[-1]["pnl_sum"] < 0 else None
    worst_r = by_symbol[-1]["avg_r"] if worst_ticker and by_symbol else None

    # Event tier counts (2X, 4X, 20X)
    cursor.execute("""
        SELECT event_tier, COUNT(*) as cnt FROM trade_performance
        WHERE trade_date = ? AND event_tier IS NOT NULL AND event_tier != ''
        GROUP BY event_tier
    """, (date_str,))
    tier_counts = {row["event_tier"]: row["cnt"] for row in cursor.fetchall()}

    conn.close()
    win_rate = (wins / total) if total else 0
    return {
        "date": date_str,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_r": avg_r,
        "total_r": total_r,
        "best_ticker": best_ticker,
        "best_r": best_r,
        "worst_ticker": worst_ticker,
        "worst_r": worst_r,
        "tier_2x": tier_counts.get("2X", 0),
        "tier_4x": tier_counts.get("4X", 0),
        "tier_20x": tier_counts.get("20X", 0),
    }


def save_outcome(outcome_data: Dict) -> int:
    """
    Save a trade outcome to the database.

    Args:
        outcome_data: Dict with keys:
            - signal_id: Optional[int]
            - entry_price: float
            - exit_price: float
            - max_price: float (MFE)
            - min_price: float (MAE)
            - outcome_type: str ("win", "loss", "scratch", "timeout")
            - pnl: float
            - pnl_pct: float
            - exit_reason: str
            - symbol: str
            - trade_type: str
            - engine: str
            - entry_time: datetime
            - exit_time: datetime
            - holding_time_seconds: int
            - session: str

    Returns:
        The inserted outcome ID
    """
    with _write_lock:
        conn = get_connection()
        cursor = conn.cursor()

        # Insert into outcomes table
        signal_id = outcome_data.get("signal_id")
        if signal_id:
            # Calculate r_multiple
            entry_price = outcome_data.get("entry_price", 0)
            exit_price = outcome_data.get("exit_price", 0)
            outcome_type = outcome_data.get("outcome_type", "scratch")

            if outcome_type == "win":
                r_multiple = abs(exit_price - entry_price) / max(abs(entry_price * 0.01), 0.01)
            elif outcome_type == "loss":
                r_multiple = -abs(exit_price - entry_price) / max(abs(entry_price * 0.01), 0.01)
            else:
                r_multiple = 0

            cursor.execute("""
                INSERT INTO outcomes (
                    signal_id, entry_price, exit_price, max_price, min_price,
                    r_multiple, outcome_type, hit_target_1, hit_stop
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id,
                outcome_data.get("entry_price"),
                outcome_data.get("exit_price"),
                outcome_data.get("max_price", outcome_data.get("exit_price")),
                outcome_data.get("min_price", outcome_data.get("exit_price")),
                r_multiple,
                outcome_type,
                1 if outcome_type == "win" else 0,
                1 if outcome_type == "loss" else 0,
            ))
            outcome_id = cursor.lastrowid
        else:
            outcome_id = 0

        # Insert into trade_performance for analytics
        entry_time = outcome_data.get("entry_time")
        entry_hour = entry_time.hour if entry_time else None
        trade_date = entry_time.strftime("%Y-%m-%d") if entry_time else datetime.utcnow().strftime("%Y-%m-%d")

        cursor.execute("""
            INSERT INTO trade_performance (
                trade_date, symbol, engine, trade_type, entry_hour, session,
                pnl, pnl_pct, exit_reason, holding_time_seconds, signal_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_date,
            outcome_data.get("symbol"),
            outcome_data.get("engine"),
            outcome_data.get("trade_type"),
            entry_hour,
            outcome_data.get("session"),
            outcome_data.get("pnl"),
            outcome_data.get("pnl_pct"),
            outcome_data.get("exit_reason"),
            outcome_data.get("holding_time_seconds"),
            signal_id,
        ))

        # Update daily_summaries
        cursor.execute("""
            INSERT INTO daily_summaries (date, paper_trades, wins, losses)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                paper_trades = paper_trades + 1,
                wins = wins + excluded.wins,
                losses = losses + excluded.losses
        """, (
            trade_date,
            1 if outcome_data.get("outcome_type") == "win" else 0,
            1 if outcome_data.get("outcome_type") == "loss" else 0,
        ))

        conn.commit()
        conn.close()

    log.info(f"Saved outcome for {outcome_data.get('symbol')}: {outcome_data.get('outcome_type')} (${outcome_data.get('pnl', 0):+.2f})")
    return outcome_id


def _r_multiple_from_prices(entry_price: float, exit_price: float, stop_price: float) -> float:
    """R = (exit - entry) / (entry - stop) per contract (option dollars)."""
    if not entry_price or not stop_price or entry_price <= stop_price:
        return 0.0
    risk = entry_price - stop_price
    return (exit_price - entry_price) / risk if risk else 0.0


def save_cpl_outcome(
    trade_date: str,
    symbol: str,
    entry_price: float,
    exit_price: float,
    stop_price: float,
    pnl_pct: float,
    exit_reason: str,
    holding_time_seconds: int = 0,
    event_tier: str = None,
    session: str = None,
) -> int:
    """
    Record a CPL trade outcome (no signal_id). Writes to trade_performance and daily_summaries.
    R-multiple = (exit - entry) / (entry - stop).
    """
    r_mult = _r_multiple_from_prices(entry_price, exit_price, stop_price)
    pnl_dollars = (exit_price - entry_price) * 100  # per contract
    outcome_type = "win" if pnl_dollars > 0 else ("loss" if pnl_dollars < 0 else "scratch")
    with _write_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trade_performance (
                trade_date, symbol, engine, trade_type, entry_hour, session,
                pnl, pnl_pct, r_multiple, exit_reason, holding_time_seconds, signal_id, event_tier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_date,
            symbol,
            "CPL",
            "CALL",  # or PUT; caller can extend if needed
            None,
            session,
            pnl_dollars,
            pnl_pct,
            r_mult,
            exit_reason,
            holding_time_seconds,
            None,
            event_tier or "",
        ))
        oid = cursor.lastrowid
        cursor.execute("""
            INSERT INTO daily_summaries (date, paper_trades, wins, losses)
            VALUES (?, 1, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                paper_trades = paper_trades + 1,
                wins = wins + excluded.wins,
                losses = losses + excluded.losses
        """, (
            trade_date,
            1 if outcome_type == "win" else 0,
            1 if outcome_type == "loss" else 0,
        ))
        conn.commit()
        conn.close()
    log.info(f"Saved CPL outcome: {symbol} {outcome_type} R={r_mult:+.2f}")
    return oid


def cpl_call_exists(dedupe_key: str) -> bool:
    """Check if a CPL call with this dedupe_key was already stored/alerted."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM cpl_calls WHERE dedupe_key = ?", (dedupe_key,))
    row = cursor.fetchone()
    conn.close()
    return row is not None


def save_cpl_call(
    call_id: str,
    timestamp_et: str,
    ticker: str,
    side: str,
    strike: float,
    expiry: str,
    dedupe_key: str,
    full_json: str,
    regime: Optional[str] = None,
    confidence: Optional[float] = None,
    alerted_at: Optional[str] = None,
) -> None:
    """Insert a CPL call into cpl_calls (after dedupe check)."""
    with _write_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO cpl_calls (
                call_id, timestamp_et, ticker, side, strike, expiry,
                dedupe_key, regime, confidence, alerted_at, full_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            call_id,
            timestamp_et,
            ticker,
            side,
            strike,
            expiry,
            dedupe_key,
            regime,
            confidence,
            alerted_at,
            full_json,
        ))
        conn.commit()
        conn.close()
    log.debug(f"Saved CPL call {call_id} ({dedupe_key})")


def get_recent_cpl_calls(limit: int = 50, days_back: Optional[int] = None) -> List[Dict]:
    """Get recent CPL calls for alert-vs-price matching. Ordered by alerted_at desc."""
    conn = get_connection()
    cursor = conn.cursor()
    if days_back:
        cursor.execute("""
            SELECT * FROM cpl_calls
            WHERE date(COALESCE(alerted_at, timestamp_et)) >= date('now', ?)
            ORDER BY COALESCE(alerted_at, timestamp_et) DESC
            LIMIT ?
        """, (f"-{days_back} days", limit))
    else:
        cursor.execute("""
            SELECT * FROM cpl_calls
            ORDER BY COALESCE(alerted_at, timestamp_et) DESC
            LIMIT ?
        """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────
# Apex Governance Layer Database Functions
# ─────────────────────────────────────────────────────────────────

def save_governance_event(
    event_type: str,
    dedupe_key: Optional[str] = None,
    ticker: Optional[str] = None,
    state_from: Optional[str] = None,
    state_to: Optional[str] = None,
    pnl_at_event: Optional[float] = None,
    reason: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> int:
    """Save a governance event to the database."""
    with _write_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO governance_events (
                event_type, dedupe_key, ticker, state_from, state_to,
                pnl_at_event, reason, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_type,
            dedupe_key,
            ticker,
            state_from,
            state_to,
            pnl_at_event,
            reason,
            json.dumps(metadata) if metadata else None,
        ))
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()
    log.debug(f"Saved governance event: {event_type}")
    return event_id


def get_governance_events(dedupe_key: Optional[str] = None, limit: int = 100) -> List[Dict]:
    """Get governance events, optionally filtered by dedupe_key."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if dedupe_key:
        cursor.execute("""
            SELECT * FROM governance_events
            WHERE dedupe_key = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (dedupe_key, limit))
    else:
        cursor.execute("""
            SELECT * FROM governance_events
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_counterfactual_checkpoint(
    call_id: str,
    timestamp_et: str,
    ticker: str,
    checkpoint_type: str,
    actual_pnl_pct: float,
    counterfactual_exit_pnl: float,
    runner_outcome_pnl: Optional[float] = None,
    missed_upside_pct: Optional[float] = None,
    scenario_json: Optional[str] = None,
) -> int:
    """Save a counterfactual checkpoint to the database."""
    with _write_lock:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO counterfactual_ledger (
                call_id, timestamp_et, ticker, checkpoint_type,
                actual_pnl_pct, counterfactual_exit_pnl, runner_outcome_pnl,
                missed_upside_pct, scenario_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            call_id,
            timestamp_et,
            ticker,
            checkpoint_type,
            actual_pnl_pct,
            counterfactual_exit_pnl,
            runner_outcome_pnl,
            missed_upside_pct,
            scenario_json,
        ))
        checkpoint_id = cursor.lastrowid
        conn.commit()
        conn.close()
    log.debug(f"Saved counterfactual checkpoint: {checkpoint_type} for {call_id}")
    return checkpoint_id


def get_counterfactual_checkpoints(call_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
    """Get counterfactual checkpoints, optionally filtered by call_id."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if call_id:
        cursor.execute("""
            SELECT * FROM counterfactual_ledger
            WHERE call_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (call_id, limit))
    else:
        cursor.execute("""
            SELECT * FROM counterfactual_ledger
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# Initialize on import
try:
    init_database()
except Exception as e:
    log.error(f"Failed to initialize database: {e}")
