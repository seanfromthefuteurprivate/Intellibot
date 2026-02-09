"""
Signal Tracker - Tracks all trading signals for reconciliation.

Provides comprehensive signal lifecycle tracking:
- New signal creation and storage
- Alert logging
- Status updates on exits
- Query functions for open/specific signals
"""

import threading
from datetime import datetime
from typing import Dict, List, Any, Optional

from wsb_snake.db.database import get_connection
from wsb_snake.utils.logger import log

# Serialize writes to avoid SQLITE_BUSY under concurrent load
_write_lock = threading.Lock()


def init_signal_tracking_tables() -> None:
    """
    Initialize signal tracking tables in the database.

    Creates:
    - tracked_signals: Main signal tracking table
    - signal_alerts: Alert logging table
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Main tracked signals table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracked_signals (
            signal_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            underlying TEXT,
            strike REAL,
            direction TEXT NOT NULL,
            contracts INTEGER,
            entry_price REAL,
            stop_loss REAL,
            pl1_target REAL,
            pl2_target REAL,
            pl3_target REAL,
            confidence REAL,
            pattern TEXT,
            session TEXT,
            max_hold_minutes INTEGER,
            hold_guidance TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT,
            status TEXT DEFAULT 'OPEN',
            pl1_hit_at TEXT,
            pl2_hit_at TEXT,
            pl3_hit_at TEXT,
            stop_hit_at TEXT,
            exit_price REAL,
            exit_reason TEXT,
            exit_at TEXT,
            final_pnl_pct REAL
        )
    """)

    # Create indices for common queries
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_tracked_signals_status ON tracked_signals(status)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_tracked_signals_ticker ON tracked_signals(ticker)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_tracked_signals_created ON tracked_signals(created_at)"
    )

    # Signal alerts table for tracking sent notifications
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER NOT NULL,
            alert_type TEXT NOT NULL,
            message TEXT,
            sent_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (signal_id) REFERENCES tracked_signals(signal_id)
        )
    """)

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_signal_alerts_signal_id ON signal_alerts(signal_id)"
    )

    conn.commit()
    conn.close()
    log.info("Signal tracking tables initialized")


def track_new_signal(signal: Dict[str, Any]) -> int:
    """
    Insert a new signal into the tracked_signals table.

    Args:
        signal: Dictionary containing signal data with keys:
            - ticker: Option symbol (e.g., "SPY250210C590")
            - underlying: Underlying ticker (e.g., "SPY")
            - strike: Strike price
            - direction: "CALL" or "PUT"
            - contracts: Number of contracts
            - entry_price: Entry price per contract
            - stop_loss: Stop loss price
            - pl1_target: First profit level target
            - pl2_target: Second profit level target
            - pl3_target: Third profit level target
            - confidence: Signal confidence score (0-100)
            - pattern: Pattern name that triggered the signal
            - session: Trading session (e.g., "POWER_HOUR", "MORNING")
            - max_hold_minutes: Maximum hold time in minutes
            - hold_guidance: Human-readable hold guidance
            - expires_at: Expiration timestamp

    Returns:
        The inserted signal_id
    """
    with _write_lock:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO tracked_signals (
                ticker, underlying, strike, direction, contracts,
                entry_price, stop_loss, pl1_target, pl2_target, pl3_target,
                confidence, pattern, session, max_hold_minutes, hold_guidance,
                expires_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
        """, (
            signal.get("ticker"),
            signal.get("underlying"),
            signal.get("strike"),
            signal.get("direction"),
            signal.get("contracts"),
            signal.get("entry_price"),
            signal.get("stop_loss"),
            signal.get("pl1_target"),
            signal.get("pl2_target"),
            signal.get("pl3_target"),
            signal.get("confidence"),
            signal.get("pattern"),
            signal.get("session"),
            signal.get("max_hold_minutes"),
            signal.get("hold_guidance"),
            signal.get("expires_at"),
        ))

        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()

    log.info(f"Tracked new signal {signal_id}: {signal.get('ticker')} {signal.get('direction')}")
    return signal_id


def record_alert_sent(signal_id: int, alert_type: str, message: str) -> int:
    """
    Log that an alert was sent for a signal.

    Args:
        signal_id: The signal ID this alert is for
        alert_type: Type of alert (e.g., "ENTRY", "PL1_HIT", "STOP_HIT", "EXIT")
        message: The alert message that was sent

    Returns:
        The inserted alert ID
    """
    with _write_lock:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO signal_alerts (signal_id, alert_type, message)
            VALUES (?, ?, ?)
        """, (signal_id, alert_type, message))

        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()

    log.debug(f"Recorded {alert_type} alert for signal {signal_id}")
    return alert_id


def update_signal_status(
    signal_id: int,
    status: str,
    exit_price: Optional[float] = None,
    exit_reason: Optional[str] = None,
    final_pnl_pct: Optional[float] = None,
) -> bool:
    """
    Update a signal's status on exit.

    Args:
        signal_id: The signal ID to update
        status: New status (e.g., "CLOSED", "STOPPED", "EXPIRED")
        exit_price: The exit price (if exiting)
        exit_reason: Reason for exit (e.g., "STOP_HIT", "PL3_HIT", "MANUAL", "EXPIRED")
        final_pnl_pct: Final P&L percentage

    Returns:
        True if update was successful, False otherwise
    """
    with _write_lock:
        conn = get_connection()
        cursor = conn.cursor()

        exit_at = datetime.utcnow().isoformat() if status != "OPEN" else None

        cursor.execute("""
            UPDATE tracked_signals
            SET status = ?,
                exit_price = COALESCE(?, exit_price),
                exit_reason = COALESCE(?, exit_reason),
                exit_at = COALESCE(?, exit_at),
                final_pnl_pct = COALESCE(?, final_pnl_pct)
            WHERE signal_id = ?
        """, (status, exit_price, exit_reason, exit_at, final_pnl_pct, signal_id))

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

    if rows_affected > 0:
        log.info(f"Updated signal {signal_id} status to {status} (exit_reason={exit_reason})")
        return True
    else:
        log.warning(f"Failed to update signal {signal_id} - not found")
        return False


def mark_target_hit(signal_id: int, target_level: int) -> bool:
    """
    Mark that a profit target was hit.

    Args:
        signal_id: The signal ID
        target_level: Which target (1, 2, or 3)

    Returns:
        True if update was successful
    """
    column_map = {
        1: "pl1_hit_at",
        2: "pl2_hit_at",
        3: "pl3_hit_at",
    }

    column = column_map.get(target_level)
    if not column:
        log.warning(f"Invalid target level: {target_level}")
        return False

    with _write_lock:
        conn = get_connection()
        cursor = conn.cursor()

        hit_time = datetime.utcnow().isoformat()
        cursor.execute(f"""
            UPDATE tracked_signals
            SET {column} = ?
            WHERE signal_id = ? AND {column} IS NULL
        """, (hit_time, signal_id))

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

    if rows_affected > 0:
        log.info(f"Marked PL{target_level} hit for signal {signal_id}")
        return True
    return False


def mark_stop_hit(signal_id: int) -> bool:
    """
    Mark that the stop was hit.

    Args:
        signal_id: The signal ID

    Returns:
        True if update was successful
    """
    with _write_lock:
        conn = get_connection()
        cursor = conn.cursor()

        hit_time = datetime.utcnow().isoformat()
        cursor.execute("""
            UPDATE tracked_signals
            SET stop_hit_at = ?
            WHERE signal_id = ? AND stop_hit_at IS NULL
        """, (hit_time, signal_id))

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

    if rows_affected > 0:
        log.info(f"Marked stop hit for signal {signal_id}")
        return True
    return False


def get_open_signals() -> List[Dict[str, Any]]:
    """
    Get all signals with status='OPEN'.

    Returns:
        List of open signal dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM tracked_signals
        WHERE status = 'OPEN'
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_signal_by_id(signal_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a single signal by its ID.

    Args:
        signal_id: The signal ID to retrieve

    Returns:
        Signal dictionary or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM tracked_signals
        WHERE signal_id = ?
    """, (signal_id,))

    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def get_signal_alerts(signal_id: int) -> List[Dict[str, Any]]:
    """
    Get all alerts for a specific signal.

    Args:
        signal_id: The signal ID

    Returns:
        List of alert dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM signal_alerts
        WHERE signal_id = ?
        ORDER BY sent_at ASC
    """, (signal_id,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_signals_by_date(date_str: str) -> List[Dict[str, Any]]:
    """
    Get all signals created on a specific date.

    Args:
        date_str: Date in YYYY-MM-DD format

    Returns:
        List of signal dictionaries
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM tracked_signals
        WHERE date(created_at) = ?
        ORDER BY created_at DESC
    """, (date_str,))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_signal_stats(date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Get signal statistics, optionally for a specific date.

    Args:
        date_str: Optional date in YYYY-MM-DD format

    Returns:
        Dictionary with signal statistics
    """
    conn = get_connection()
    cursor = conn.cursor()

    date_filter = "WHERE date(created_at) = ?" if date_str else ""
    params = (date_str,) if date_str else ()

    cursor.execute(f"""
        SELECT
            COUNT(*) as total_signals,
            SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_signals,
            SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed_signals,
            SUM(CASE WHEN pl1_hit_at IS NOT NULL THEN 1 ELSE 0 END) as pl1_hits,
            SUM(CASE WHEN pl2_hit_at IS NOT NULL THEN 1 ELSE 0 END) as pl2_hits,
            SUM(CASE WHEN pl3_hit_at IS NOT NULL THEN 1 ELSE 0 END) as pl3_hits,
            SUM(CASE WHEN stop_hit_at IS NOT NULL THEN 1 ELSE 0 END) as stop_hits,
            AVG(final_pnl_pct) as avg_pnl_pct
        FROM tracked_signals
        {date_filter}
    """, params)

    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else {}


# Initialize tables on import
try:
    init_signal_tracking_tables()
except Exception as e:
    log.error(f"Failed to initialize signal tracking tables: {e}")
