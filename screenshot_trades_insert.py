#!/usr/bin/env python3
"""
Insert extracted screenshot trades into learned_trades database.
Extracted from zip file: New Folder With Items 3.zip (53 screenshots)
"""

import sqlite3
from datetime import datetime

# Major closed trades extracted from screenshots
SCREENSHOT_TRADES = [
    # QQQ $800 -> $49K legendary trade (IMG_0307, IMG_0507)
    {
        "ticker": "QQQ", "strike": 615, "option_type": "CALL", "expiry": "2026-01-21",
        "entry_price": 0.08, "exit_price": 5.03, "quantity": 100,
        "entry_time": "2026-01-21 13:31:00", "exit_time": "2026-01-21 14:50:00",
        "pnl_dollars": 49500.00, "pnl_percent": 6187.5, "source": "screenshot_thinkorswim",
        "notes": "0DTE scalp, $800 to $49K in 79 minutes"
    },
    # USAR calls +$72K (IMG_0355)
    {
        "ticker": "USAR", "strike": None, "option_type": "CALL", "expiry": "2026-01-26",
        "entry_price": 0.86, "exit_price": 6.868, "quantity": 125,
        "entry_time": "2026-01-24 10:00:00", "exit_time": "2026-01-26 10:00:00",
        "pnl_dollars": 72249.11, "pnl_percent": 698.6, "source": "screenshot_wsb",
        "notes": "WSB YOLO play, 700%+ gains"
    },
    # SLV $100 Call massive winner (IMG_0363, IMG_0364)
    {
        "ticker": "SLV", "strike": 100, "option_type": "CALL", "expiry": "2026-01-28",
        "entry_price": 0.61, "exit_price": 8.11, "quantity": 60,
        "entry_time": "2026-01-23 09:00:00", "exit_time": "2026-01-26 10:00:00",
        "pnl_dollars": 44919.15, "pnl_percent": 1229.5, "source": "screenshot_etrade",
        "notes": "Silver breakout play, 3-day hold"
    },
    # SLV $40 Call LEAPS (IMG_0404)
    {
        "ticker": "SLV", "strike": 40, "option_type": "CALL", "expiry": "2028-01-21",
        "entry_price": 13.00, "exit_price": 56.92, "quantity": 1,
        "entry_time": "2025-10-01 10:00:00", "exit_time": "2026-01-12 08:44:00",
        "pnl_dollars": 4392.46, "pnl_percent": 337.89, "source": "screenshot_robinhood",
        "notes": "LEAPS swing trade on silver"
    },
    # GLD $515 Call (IMG_0417)
    {
        "ticker": "GLD", "strike": 515, "option_type": "CALL", "expiry": "2026-01-30",
        "entry_price": 5.48, "exit_price": None, "quantity": None,
        "entry_time": "2026-01-28 09:00:00", "exit_time": "2026-01-29 10:00:00",
        "pnl_dollars": 10600.00, "pnl_percent": None, "source": "screenshot_moomoo",
        "notes": "~11k in a day on gold calls"
    },
    # IAU (Gold) Calls batch 1 (IMG_0427)
    {
        "ticker": "IAU", "strike": 91, "option_type": "CALL", "expiry": "2027-01-15",
        "entry_price": 2.55, "exit_price": 19.89, "quantity": 4,
        "entry_time": "2025-10-07 10:00:00", "exit_time": "2026-01-29 10:00:00",
        "pnl_dollars": 6938.63, "pnl_percent": 681.0, "source": "screenshot_etrade",
        "notes": "Gold LEAPS swing"
    },
    # IAU Calls batch 2 (IMG_0427)
    {
        "ticker": "IAU", "strike": 91, "option_type": "CALL", "expiry": "2027-01-15",
        "entry_price": 2.55, "exit_price": 17.24, "quantity": 2,
        "entry_time": "2025-10-07 10:00:00", "exit_time": "2026-01-28 10:00:00",
        "pnl_dollars": 2939.31, "pnl_percent": 577.0, "source": "screenshot_etrade",
        "notes": "Gold LEAPS swing"
    },
    # IAU Calls batch 3 (IMG_0427)
    {
        "ticker": "IAU", "strike": 91, "option_type": "CALL", "expiry": "2027-01-15",
        "entry_price": 2.55, "exit_price": 16.94, "quantity": 2,
        "entry_time": "2025-10-07 10:00:00", "exit_time": "2026-01-28 10:00:00",
        "pnl_dollars": 2879.31, "pnl_percent": 565.0, "source": "screenshot_etrade",
        "notes": "Gold LEAPS swing"
    },
    # MSFT $430 Put earnings play (IMG_0420)
    {
        "ticker": "MSFT", "strike": 430, "option_type": "PUT", "expiry": "2026-01-30",
        "entry_price": 1.10, "exit_price": 4.00, "quantity": 9,
        "entry_time": "2026-01-29 09:30:00", "exit_time": "2026-01-29 16:00:00",
        "pnl_dollars": 2610.00, "pnl_percent": 263.64, "source": "screenshot_robinhood",
        "notes": "MSFT earnings play, sold too early"
    },
    # QQQ $625 Put 0DTE monster (IMG_0487)
    {
        "ticker": "QQQ", "strike": 625, "option_type": "PUT", "expiry": "2026-02-03",
        "entry_price": 0.87, "exit_price": 7.88, "quantity": 65,
        "entry_time": "2026-02-03 09:30:00", "exit_time": "2026-02-03 10:24:00",
        "pnl_dollars": 45547.00, "pnl_percent": 805.7, "source": "screenshot_robinhood",
        "notes": "0DTE put on selloff day, 65 contracts"
    },
    # SLV $86 Put (IMG_0488)
    {
        "ticker": "SLV", "strike": 86, "option_type": "PUT", "expiry": "2026-04-17",
        "entry_price": 12.32, "exit_price": 18.40, "quantity": 2,
        "entry_time": "2026-01-20 10:00:00", "exit_time": "2026-01-30 12:47:00",
        "pnl_dollars": 1215.00, "pnl_percent": 49.30, "source": "screenshot_robinhood",
        "notes": "Silver put swing"
    },
    # NAVI $10 Put (IMG_0489)
    {
        "ticker": "NAVI", "strike": 10, "option_type": "PUT", "expiry": "2026-04-17",
        "entry_price": 0.4324, "exit_price": 0.80, "quantity": 507,
        "entry_time": "2026-01-15 10:00:00", "exit_time": "2026-01-30 06:54:00",
        "pnl_dollars": 18641.00, "pnl_percent": 85.05, "source": "screenshot_robinhood",
        "notes": "507 contract put play"
    },
    # SNDK $620 Call (IMG_0490)
    {
        "ticker": "SNDK", "strike": 620, "option_type": "CALL", "expiry": "2026-01-30",
        "entry_price": 6.00, "exit_price": 50.00, "quantity": 1,
        "entry_time": "2026-01-29 10:00:00", "exit_time": "2026-01-30 10:00:00",
        "pnl_dollars": 4400.00, "pnl_percent": 733.34, "source": "screenshot_robinhood",
        "notes": "733% return single contract"
    },
    # SLV $74.5 Put (IMG_0501)
    {
        "ticker": "SLV", "strike": 74.5, "option_type": "PUT", "expiry": "2026-01-30",
        "entry_price": 0.50, "exit_price": 4.95, "quantity": 20,
        "entry_time": "2026-01-28 10:00:00", "exit_time": "2026-01-30 10:26:00",
        "pnl_dollars": 9880.00, "pnl_percent": 890.0, "source": "screenshot_robinhood",
        "notes": "Silver put scalp"
    },
    # MSTR $143 Put (IMG_0502)
    {
        "ticker": "MSTR", "strike": 143, "option_type": "PUT", "expiry": "2026-02-06",
        "entry_price": 4.05, "exit_price": 6.10, "quantity": 13,
        "entry_time": "2026-01-30 10:00:00", "exit_time": "2026-02-02 13:32:00",
        "pnl_dollars": 2665.00, "pnl_percent": 50.6, "source": "screenshot_robinhood",
        "notes": "MSTR put swing"
    },
    # UNG $12.5 Call assignment (IMG_0503)
    {
        "ticker": "UNG", "strike": 12.5, "option_type": "CALL", "expiry": "2026-01-21",
        "entry_price": 10.82, "exit_price": 13.55, "quantity": 60,
        "entry_time": "2025-12-01 10:00:00", "exit_time": "2026-01-21 16:00:00",
        "pnl_dollars": 16395.00, "pnl_percent": 25.26, "source": "screenshot_robinhood",
        "notes": "Natural gas call assignment, 6000 shares"
    },
    # UNG $15 Put (IMG_0504)
    {
        "ticker": "UNG", "strike": 15, "option_type": "PUT", "expiry": "2026-02-20",
        "entry_price": 0.78, "exit_price": 2.07, "quantity": 60,
        "entry_time": "2026-01-20 10:00:00", "exit_time": "2026-02-02 10:00:00",
        "pnl_dollars": 7740.00, "pnl_percent": 165.39, "source": "screenshot_robinhood",
        "notes": "Natural gas put swing"
    },
    # SPY 692 Put 0DTE (IMG_0419)
    {
        "ticker": "SPY", "strike": 692, "option_type": "PUT", "expiry": "2026-01-29",
        "entry_price": 1.78, "exit_price": 2.92, "quantity": 40,
        "entry_time": "2026-01-29 09:30:00", "exit_time": "2026-01-29 10:30:00",
        "pnl_dollars": 6759.14, "pnl_percent": 64.0, "source": "screenshot_wsb",
        "notes": "SPY 0DTE put scalp"
    },
    # GLD 450C closed trades (IMG_0424)
    {
        "ticker": "GLD", "strike": 450, "option_type": "CALL", "expiry": "2026-06-30",
        "entry_price": 20.00, "exit_price": 64.50, "quantity": 2,
        "entry_time": "2025-11-01 10:00:00", "exit_time": "2026-01-29 10:00:00",
        "pnl_dollars": 8900.00, "pnl_percent": 222.5, "source": "screenshot_thinkorswim",
        "notes": "Gold LEAPS swing"
    },
    {
        "ticker": "GLD", "strike": 450, "option_type": "CALL", "expiry": "2026-06-30",
        "entry_price": 20.00, "exit_price": 69.69, "quantity": 3,
        "entry_time": "2025-11-01 10:00:00", "exit_time": "2026-01-29 10:00:00",
        "pnl_dollars": 14907.00, "pnl_percent": 248.5, "source": "screenshot_thinkorswim",
        "notes": "Gold LEAPS swing"
    },
]

def insert_trades(db_path: str):
    """Insert screenshot trades into learned_trades table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='learned_trades'
    """)
    if not cursor.fetchone():
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                strike REAL,
                option_type TEXT,
                expiry TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity INTEGER,
                entry_time TEXT,
                exit_time TEXT,
                pnl_dollars REAL,
                pnl_percent REAL,
                source TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

    inserted = 0
    for trade in SCREENSHOT_TRADES:
        # Check for duplicates
        cursor.execute("""
            SELECT id FROM learned_trades
            WHERE ticker = ? AND strike = ? AND entry_time = ? AND pnl_dollars = ?
        """, (trade["ticker"], trade.get("strike"), trade["entry_time"], trade["pnl_dollars"]))

        if cursor.fetchone():
            print(f"Skipping duplicate: {trade['ticker']} {trade.get('strike')} {trade['entry_time']}")
            continue

        cursor.execute("""
            INSERT INTO learned_trades
            (ticker, strike, option_type, expiry, entry_price, exit_price,
             quantity, entry_time, exit_time, pnl_dollars, pnl_percent, source, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade["ticker"], trade.get("strike"), trade["option_type"], trade.get("expiry"),
            trade["entry_price"], trade.get("exit_price"), trade.get("quantity"),
            trade["entry_time"], trade.get("exit_time"), trade["pnl_dollars"],
            trade.get("pnl_percent"), trade["source"], trade.get("notes")
        ))
        inserted += 1
        print(f"Inserted: {trade['ticker']} ${trade.get('strike')} {trade['option_type']} -> ${trade['pnl_dollars']:,.2f}")

    conn.commit()

    # Show summary
    cursor.execute("SELECT COUNT(*) FROM learned_trades")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(pnl_dollars) FROM learned_trades WHERE pnl_dollars > 0")
    total_profit = cursor.fetchone()[0] or 0

    cursor.execute("SELECT SUM(pnl_dollars) FROM learned_trades WHERE pnl_dollars < 0")
    total_loss = cursor.fetchone()[0] or 0

    print(f"\n=== SUMMARY ===")
    print(f"Inserted: {inserted} new trades")
    print(f"Total trades in DB: {total}")
    print(f"Total profits: ${total_profit:,.2f}")
    print(f"Total losses: ${total_loss:,.2f}")
    print(f"Net P&L: ${total_profit + total_loss:,.2f}")

    conn.close()
    return inserted

if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "wsb_snake_data/wsb_snake.db"
    insert_trades(db_path)
