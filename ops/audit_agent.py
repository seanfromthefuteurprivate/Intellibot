#!/usr/bin/env python3
"""
AGENT 3: TRADE AUDITOR — ops/audit_agent.py
"I KNOW WHO PLACED EVERY ORDER. I TRACK EVERY DOLLAR. I CATCH EVERY GHOST."

Order attribution, direction lock enforcement, circuit breaker tracking, daily reports.
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any

import requests

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wsb_snake.notifications.telegram_bot import send_alert
from wsb_snake.utils.logger import get_logger

logger = get_logger("ops.audit")

# Config
STATE_FILE = Path(__file__).parent / "state.json"
AUDIT_DB = Path(__file__).parent / "audit.db"
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")

# Engine patterns for attribution
ENGINE_PATTERNS = [
    (r"CPL BUY broadcast", "CPL"),
    (r"V7_ENTRY:", "V7"),
    (r"V7_SIGNAL_FOUND:", "V7"),
    (r"ChartBrain", "CHARTBRAIN"),
    (r"GOVERNANCE: Position registered.*\|CPL\|", "CPL"),
    (r"GOVERNANCE: Position registered.*\|V7\|", "V7"),
    (r"EXECUTOR: execute_scalp_entry called for.*pattern=CPL", "CPL"),
    (r"EXECUTOR: execute_scalp_entry called for.*pattern=V7", "V7"),
]


def init_database():
    """Initialize SQLite audit database."""
    conn = sqlite3.connect(AUDIT_DB)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty INTEGER NOT NULL,
            price REAL,
            engine TEXT NOT NULL,
            client_order_id TEXT UNIQUE,
            pnl REAL,
            hold_minutes INTEGER,
            status TEXT DEFAULT 'OPEN',
            direction_locked TEXT,
            notes TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            resolved INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT PRIMARY KEY,
            total_pnl REAL,
            trade_count INTEGER,
            win_count INTEGER,
            loss_count INTEGER,
            crash_count INTEGER,
            ghost_trades INTEGER,
            orphan_count INTEGER,
            v7_trades INTEGER,
            cpl_trades INTEGER,
            report_sent INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Audit database initialized")


def load_state() -> Dict:
    """Load persisted state."""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load state: {e}")
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "last_order_id": None,
        "direction_locks": {},
        "consecutive_losses": 0,
        "killed_engines": [],
    }


def save_state(state: Dict):
    """Persist state."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def reset_daily_state(state: Dict) -> Dict:
    """Reset state on new day."""
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("date") != today:
        logger.info(f"New day: {today}. Resetting audit state.")
        state["date"] = today
        state["direction_locks"] = {}
        state["consecutive_losses"] = 0
    return state


def is_market_hours() -> bool:
    """Check if within extended market hours (9:25 AM - 4:15 PM ET)."""
    try:
        import zoneinfo
        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
        start = et.replace(hour=9, minute=25, second=0)
        end = et.replace(hour=16, minute=15, second=0)
        return start <= et <= end
    except Exception:
        return True


def is_report_time() -> bool:
    """Check if it's daily report time (4:05 PM ET)."""
    try:
        import zoneinfo
        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
        return et.hour == 16 and 5 <= et.minute < 10
    except Exception:
        return False


def alpaca_request(endpoint: str) -> Optional[Any]:
    """Make authenticated request to Alpaca API."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None
    try:
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        url = f"{ALPACA_BASE_URL}{endpoint}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Alpaca API error: {e}")
        return None


def get_recent_logs(since_seconds: int = 120) -> str:
    """Get recent journal logs."""
    try:
        result = subprocess.run(
            ["journalctl", "-u", "wsb-snake", "--no-pager", "-n", "500",
             "--since", f"{since_seconds}s ago"],
            capture_output=True, text=True, timeout=15
        )
        return result.stdout
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        return ""


def attribute_order(order: Dict, logs: str) -> str:
    """Determine which engine placed an order."""
    symbol = order.get("symbol", "")
    client_order_id = order.get("client_order_id", "")
    filled_at = order.get("filled_at", "")

    # Search for client_order_id in logs
    if client_order_id and client_order_id in logs:
        # Found the order ID - now find the engine
        for pattern, engine in ENGINE_PATTERNS:
            if re.search(pattern, logs):
                return engine

    # Search by symbol and timing
    symbol_base = symbol.split("2")[0] if "2" in symbol else symbol[:3]  # Extract ticker
    for pattern, engine in ENGINE_PATTERNS:
        full_pattern = f"{pattern}.*{symbol_base}|{symbol_base}.*{pattern}"
        if re.search(full_pattern, logs, re.IGNORECASE):
            return engine

    # Check source field
    source = order.get("source")
    if source == "access_key":
        # API order but unknown engine - could be any of our engines
        return "UNKNOWN_API"
    elif source is None:
        # Possibly manual or old API format
        return "UNKNOWN"

    return "UNKNOWN"


def record_trade(order: Dict, engine: str):
    """Record trade in audit database."""
    try:
        conn = sqlite3.connect(AUDIT_DB)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR IGNORE INTO trades
            (timestamp, symbol, side, qty, price, engine, client_order_id, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.get("filled_at") or order.get("submitted_at"),
            order.get("symbol"),
            order.get("side"),
            int(order.get("filled_qty", 0)),
            float(order.get("filled_avg_price", 0) or 0),
            engine,
            order.get("client_order_id"),
            "FILLED" if order.get("status") == "filled" else order.get("status"),
        ))

        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to record trade: {e}")


def check_direction_violation(state: Dict, order: Dict, engine: str) -> Optional[str]:
    """Check if order violates direction lock."""
    symbol = order.get("symbol", "")
    side = order.get("side", "")

    # Extract ticker from option symbol (e.g., QQQ260302C00609000 -> QQQ)
    ticker = symbol.split("2")[0] if "2" in symbol else symbol[:3]

    # Determine direction from option type
    if "C" in symbol[6:10]:  # Call option
        direction = "CALL"
    elif "P" in symbol[6:10]:  # Put option
        direction = "PUT"
    else:
        direction = "UNKNOWN"

    locks = state.get("direction_locks", {})

    if side == "buy":  # Only check on entries, not exits
        if ticker in locks and locks[ticker]["direction"] != direction:
            lock_time = locks[ticker]["time"]
            return (
                f"🔴 DIRECTION VIOLATION: {engine} entered {ticker} {direction} "
                f"but {locks[ticker]['direction']} was locked at {lock_time}"
            )
        else:
            # Set or update lock
            locks[ticker] = {
                "direction": direction,
                "time": datetime.now().strftime("%H:%M:%S"),
                "engine": engine,
            }
            state["direction_locks"] = locks

    return None


def check_zombie_engine(state: Dict, engine: str) -> Optional[str]:
    """Check if a killed engine placed an order."""
    killed = state.get("killed_engines", [])
    if engine in killed:
        return f"🔴🔴 ZOMBIE ENGINE: {engine} placed order but is supposed to be KILLED"
    return None


def track_consecutive_losses(state: Dict, order: Dict) -> Dict:
    """Track consecutive losses for circuit breaker verification."""
    side = order.get("side", "")
    if side != "sell":
        return state

    # This is a closing order - check P&L
    # For simplicity, check if the unrealized_pl was negative
    # In practice, we'd need to match with the opening order
    pnl = float(order.get("filled_avg_price", 0) or 0)

    # Note: This is a simplified check. Full implementation would
    # match buys with sells to calculate actual P&L
    conn = sqlite3.connect(AUDIT_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT price FROM trades WHERE symbol = ? AND side = 'buy' ORDER BY timestamp DESC LIMIT 1",
        (order.get("symbol"),)
    )
    result = cursor.fetchone()
    conn.close()

    if result:
        entry_price = result[0]
        exit_price = float(order.get("filled_avg_price", 0) or 0)
        if exit_price < entry_price:
            state["consecutive_losses"] = state.get("consecutive_losses", 0) + 1
            if state["consecutive_losses"] >= 2:
                send_alert(
                    f"🛑 CIRCUIT BREAKER: {state['consecutive_losses']} consecutive losses. "
                    "V7 should have halted."
                )
        else:
            state["consecutive_losses"] = 0

    return state


def process_new_orders(state: Dict) -> Dict:
    """Process new orders from Alpaca."""
    # Get orders from last 2 minutes
    after = (datetime.utcnow() - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    orders = alpaca_request(f"/v2/orders?status=all&limit=20&after={after}")

    if orders is None:
        return state

    logs = get_recent_logs(since_seconds=180)
    last_id = state.get("last_order_id")

    for order in orders:
        order_id = order.get("id")
        client_order_id = order.get("client_order_id")

        # Skip if already processed
        if last_id and order_id <= last_id:
            continue

        # Only process filled orders
        if order.get("status") != "filled":
            continue

        # Attribute the order
        engine = attribute_order(order, logs)

        # Record to database
        record_trade(order, engine)

        # Send attribution alert
        symbol = order.get("symbol", "")
        side = order.get("side", "")
        qty = order.get("filled_qty", 0)
        price = float(order.get("filled_avg_price", 0) or 0)

        send_alert(f"📋 TRADE [{engine}]: {side.upper()} {qty}x {symbol} @ ${price:.2f}")

        # Check for ghost trade
        if engine in ("UNKNOWN", "UNKNOWN_API"):
            send_alert(
                f"🔴 GHOST TRADE: {symbol} {side} {qty}x @ ${price:.2f} — "
                f"NO ENGINE MATCH. client_order_id={client_order_id}"
            )

        # Check direction violation
        violation = check_direction_violation(state, order, engine)
        if violation:
            send_alert(violation)

        # Check zombie engine
        zombie = check_zombie_engine(state, engine)
        if zombie:
            send_alert(zombie)

        # Track consecutive losses
        state = track_consecutive_losses(state, order)

        state["last_order_id"] = order_id

    return state


def generate_daily_report(state: Dict) -> Dict:
    """Generate end-of-day report at 4:05 PM ET."""
    today = state.get("date")

    # Check if report already sent
    conn = sqlite3.connect(AUDIT_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT report_sent FROM daily_summary WHERE date = ?", (today,))
    result = cursor.fetchone()
    if result and result[0]:
        conn.close()
        return state

    # Get today's trades
    cursor.execute("""
        SELECT engine, side, price, qty
        FROM trades
        WHERE date(timestamp) = ? AND status = 'FILLED'
    """, (today,))
    trades = cursor.fetchall()

    # Calculate stats
    engine_stats = {}
    total_trades = 0
    for engine, side, price, qty in trades:
        if engine not in engine_stats:
            engine_stats[engine] = {"count": 0, "buys": 0, "sells": 0}
        engine_stats[engine]["count"] += 1
        if side == "buy":
            engine_stats[engine]["buys"] += 1
        else:
            engine_stats[engine]["sells"] += 1
        total_trades += 1

    # Get P&L from state (set by monitor agent)
    daily_pnl = state.get("current_pnl", 0)
    portfolio_value = state.get("portfolio_value", 0)

    # Build report
    report = f"""📊 DAILY REPORT — {today}

💰 P&L: ${daily_pnl:+.2f}
📈 Portfolio: ${portfolio_value:.2f}

TRADES BY ENGINE:"""

    for engine, stats in engine_stats.items():
        report += f"\n- {engine}: {stats['count']} trades ({stats['buys']} buys, {stats['sells']} sells)"

    if not engine_stats:
        report += "\n- No trades today"

    report += f"""

SYSTEM:
- Consecutive losses: {state.get('consecutive_losses', 0)}
- Direction locks: {len(state.get('direction_locks', {}))} tickers"""

    send_alert(report)

    # Mark report as sent
    cursor.execute("""
        INSERT OR REPLACE INTO daily_summary
        (date, total_pnl, trade_count, report_sent)
        VALUES (?, ?, ?, 1)
    """, (today, daily_pnl, total_trades))
    conn.commit()
    conn.close()

    return state


def main():
    """Main audit loop."""
    logger.info("🟢 AUDIT AGENT STARTING")
    init_database()
    send_alert("🟢 AUDIT AGENT ONLINE — tracking orders, attribution, direction locks")

    state = load_state()
    state = reset_daily_state(state)

    order_interval = 60  # Check orders every 60 seconds
    save_interval = 30

    last_order_check = 0
    last_save = 0
    last_report_check = 0

    try:
        while True:
            now = time.time()
            state = reset_daily_state(state)

            # Check for new orders (market hours only)
            if is_market_hours() and now - last_order_check >= order_interval:
                state = process_new_orders(state)
                last_order_check = now

            # Check for daily report time
            if now - last_report_check >= 60:  # Check every minute
                if is_report_time():
                    state = generate_daily_report(state)
                last_report_check = now

            # Save state
            if now - last_save >= save_interval:
                save_state(state)
                last_save = now

            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("Audit agent stopped by user")
    except Exception as e:
        logger.error(f"Audit agent crashed: {e}")
        send_alert(f"🔴 AUDIT AGENT CRASHED: {e}")
        raise


if __name__ == "__main__":
    main()
