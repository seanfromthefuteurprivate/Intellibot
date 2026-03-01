#!/usr/bin/env python3
"""
Insert extracted screenshot trades into learned_trades database.
Schema verified from: sqlite3 wsb_snake.db ".schema learned_trades"

COLUMNS (in order):
id, ticker, trade_type, strike, expiry, direction, entry_price, exit_price,
quantity, profit_loss_dollars, profit_loss_pct, platform, trade_date, is_0dte,
notes, raw_text, confidence, source, created_at, telegram_chat_id, telegram_msg_id,
catalyst, pattern
"""

import sqlite3
import sys

# Major closed trades extracted from screenshots - MATCHING EXACT SCHEMA
SCREENSHOT_TRADES = [
    # QQQ $800 -> $49K legendary trade (IMG_0307, IMG_0507)
    {
        "ticker": "QQQ", "trade_type": "CALL", "strike": 615.0, "expiry": "2026-01-21",
        "direction": "long", "entry_price": 0.08, "exit_price": 5.03, "quantity": 100,
        "profit_loss_dollars": 49500.0, "profit_loss_pct": 6187.5, "platform": "thinkorswim",
        "trade_date": "2026-01-21", "is_0dte": 1, "notes": "0DTE scalp $800 to $49K in 79 minutes",
        "confidence": 1.0, "source": "screenshot", "pattern": "LOTTO_TICKET"
    },
    # USAR calls +$72K (IMG_0355)
    {
        "ticker": "USAR", "trade_type": "CALL", "strike": None, "expiry": "2026-01-26",
        "direction": "long", "entry_price": 0.86, "exit_price": 6.868, "quantity": 125,
        "profit_loss_dollars": 72249.11, "profit_loss_pct": 698.6, "platform": "WSB",
        "trade_date": "2026-01-26", "is_0dte": 0, "notes": "WSB YOLO play 700%+ gains",
        "confidence": 0.9, "source": "screenshot", "pattern": "LOTTO_TICKET"
    },
    # SLV $100 Call massive winner (IMG_0363, IMG_0364)
    {
        "ticker": "SLV", "trade_type": "CALL", "strike": 100.0, "expiry": "2026-01-28",
        "direction": "long", "entry_price": 0.61, "exit_price": 8.11, "quantity": 60,
        "profit_loss_dollars": 44919.15, "profit_loss_pct": 1229.5, "platform": "E-Trade",
        "trade_date": "2026-01-26", "is_0dte": 0, "notes": "Silver breakout play 3-day hold",
        "confidence": 1.0, "source": "screenshot", "pattern": "PRECIOUS_METALS_MOMENTUM"
    },
    # SLV $40 Call LEAPS (IMG_0404)
    {
        "ticker": "SLV", "trade_type": "CALL", "strike": 40.0, "expiry": "2028-01-21",
        "direction": "long", "entry_price": 13.0, "exit_price": 56.92, "quantity": 1,
        "profit_loss_dollars": 4392.46, "profit_loss_pct": 337.89, "platform": "Robinhood",
        "trade_date": "2026-01-12", "is_0dte": 0, "notes": "LEAPS swing trade on silver",
        "confidence": 1.0, "source": "screenshot", "pattern": "PRECIOUS_METALS_MOMENTUM"
    },
    # GLD $515 Call (IMG_0417)
    {
        "ticker": "GLD", "trade_type": "CALL", "strike": 515.0, "expiry": "2026-01-30",
        "direction": "long", "entry_price": 5.48, "exit_price": None, "quantity": None,
        "profit_loss_dollars": 10600.0, "profit_loss_pct": None, "platform": "moomoo",
        "trade_date": "2026-01-29", "is_0dte": 0, "notes": "~11k in a day on gold calls",
        "confidence": 0.9, "source": "screenshot", "pattern": "PRECIOUS_METALS_MOMENTUM"
    },
    # IAU (Gold) Calls batch 1 (IMG_0427)
    {
        "ticker": "IAU", "trade_type": "CALL", "strike": 91.0, "expiry": "2027-01-15",
        "direction": "long", "entry_price": 2.55, "exit_price": 19.89, "quantity": 4,
        "profit_loss_dollars": 6938.63, "profit_loss_pct": 681.0, "platform": "E-Trade",
        "trade_date": "2026-01-29", "is_0dte": 0, "notes": "Gold LEAPS swing",
        "confidence": 1.0, "source": "screenshot", "pattern": "PRECIOUS_METALS_MOMENTUM"
    },
    # IAU Calls batch 2 (IMG_0427)
    {
        "ticker": "IAU", "trade_type": "CALL", "strike": 91.0, "expiry": "2027-01-15",
        "direction": "long", "entry_price": 2.55, "exit_price": 17.24, "quantity": 2,
        "profit_loss_dollars": 2939.31, "profit_loss_pct": 577.0, "platform": "E-Trade",
        "trade_date": "2026-01-28", "is_0dte": 0, "notes": "Gold LEAPS swing",
        "confidence": 1.0, "source": "screenshot", "pattern": "PRECIOUS_METALS_MOMENTUM"
    },
    # IAU Calls batch 3 (IMG_0427)
    {
        "ticker": "IAU", "trade_type": "CALL", "strike": 91.0, "expiry": "2027-01-15",
        "direction": "long", "entry_price": 2.55, "exit_price": 16.94, "quantity": 2,
        "profit_loss_dollars": 2879.31, "profit_loss_pct": 565.0, "platform": "E-Trade",
        "trade_date": "2026-01-28", "is_0dte": 0, "notes": "Gold LEAPS swing",
        "confidence": 1.0, "source": "screenshot", "pattern": "PRECIOUS_METALS_MOMENTUM"
    },
    # MSFT $430 Put earnings play (IMG_0420)
    {
        "ticker": "MSFT", "trade_type": "PUT", "strike": 430.0, "expiry": "2026-01-30",
        "direction": "long", "entry_price": 1.10, "exit_price": 4.0, "quantity": 9,
        "profit_loss_dollars": 2610.0, "profit_loss_pct": 263.64, "platform": "Robinhood",
        "trade_date": "2026-01-29", "is_0dte": 0, "notes": "MSFT earnings play sold too early",
        "confidence": 1.0, "source": "screenshot", "pattern": "EARNINGS_PLAY"
    },
    # QQQ $625 Put 0DTE monster (IMG_0487)
    {
        "ticker": "QQQ", "trade_type": "PUT", "strike": 625.0, "expiry": "2026-02-03",
        "direction": "long", "entry_price": 0.87, "exit_price": 7.88, "quantity": 65,
        "profit_loss_dollars": 45547.0, "profit_loss_pct": 805.7, "platform": "Robinhood",
        "trade_date": "2026-02-03", "is_0dte": 1, "notes": "0DTE put on selloff day 65 contracts",
        "confidence": 1.0, "source": "screenshot", "pattern": "REVERSAL_PUT"
    },
    # SLV $86 Put (IMG_0488)
    {
        "ticker": "SLV", "trade_type": "PUT", "strike": 86.0, "expiry": "2026-04-17",
        "direction": "long", "entry_price": 12.32, "exit_price": 18.4, "quantity": 2,
        "profit_loss_dollars": 1215.0, "profit_loss_pct": 49.3, "platform": "Robinhood",
        "trade_date": "2026-01-30", "is_0dte": 0, "notes": "Silver put swing",
        "confidence": 1.0, "source": "screenshot", "pattern": "PRECIOUS_METALS_MOMENTUM"
    },
    # NAVI $10 Put (IMG_0489)
    {
        "ticker": "NAVI", "trade_type": "PUT", "strike": 10.0, "expiry": "2026-04-17",
        "direction": "long", "entry_price": 0.4324, "exit_price": 0.8, "quantity": 507,
        "profit_loss_dollars": 18641.0, "profit_loss_pct": 85.05, "platform": "Robinhood",
        "trade_date": "2026-01-30", "is_0dte": 0, "notes": "507 contract put play",
        "confidence": 1.0, "source": "screenshot", "pattern": "HIGH_VOLUME_CONVICTION"
    },
    # SNDK $620 Call (IMG_0490)
    {
        "ticker": "SNDK", "trade_type": "CALL", "strike": 620.0, "expiry": "2026-01-30",
        "direction": "long", "entry_price": 6.0, "exit_price": 50.0, "quantity": 1,
        "profit_loss_dollars": 4400.0, "profit_loss_pct": 733.34, "platform": "Robinhood",
        "trade_date": "2026-01-30", "is_0dte": 0, "notes": "733% return single contract",
        "confidence": 1.0, "source": "screenshot", "pattern": "LOTTO_TICKET"
    },
    # SLV $74.5 Put (IMG_0501)
    {
        "ticker": "SLV", "trade_type": "PUT", "strike": 74.5, "expiry": "2026-01-30",
        "direction": "long", "entry_price": 0.5, "exit_price": 4.95, "quantity": 20,
        "profit_loss_dollars": 9880.0, "profit_loss_pct": 890.0, "platform": "Robinhood",
        "trade_date": "2026-01-30", "is_0dte": 0, "notes": "Silver put scalp",
        "confidence": 1.0, "source": "screenshot", "pattern": "PRECIOUS_METALS_MOMENTUM"
    },
    # MSTR $143 Put (IMG_0502)
    {
        "ticker": "MSTR", "trade_type": "PUT", "strike": 143.0, "expiry": "2026-02-06",
        "direction": "long", "entry_price": 4.05, "exit_price": 6.1, "quantity": 13,
        "profit_loss_dollars": 2665.0, "profit_loss_pct": 50.6, "platform": "Robinhood",
        "trade_date": "2026-02-02", "is_0dte": 0, "notes": "MSTR put swing",
        "confidence": 1.0, "source": "screenshot", "pattern": None
    },
    # UNG $12.5 Call assignment (IMG_0503)
    {
        "ticker": "UNG", "trade_type": "CALL", "strike": 12.5, "expiry": "2026-01-21",
        "direction": "long", "entry_price": 10.82, "exit_price": 13.55, "quantity": 60,
        "profit_loss_dollars": 16395.0, "profit_loss_pct": 25.26, "platform": "Robinhood",
        "trade_date": "2026-01-21", "is_0dte": 0, "notes": "Natural gas call assignment 6000 shares",
        "confidence": 1.0, "source": "screenshot", "pattern": None
    },
    # UNG $15 Put (IMG_0504)
    {
        "ticker": "UNG", "trade_type": "PUT", "strike": 15.0, "expiry": "2026-02-20",
        "direction": "long", "entry_price": 0.78, "exit_price": 2.07, "quantity": 60,
        "profit_loss_dollars": 7740.0, "profit_loss_pct": 165.39, "platform": "Robinhood",
        "trade_date": "2026-02-02", "is_0dte": 0, "notes": "Natural gas put swing",
        "confidence": 1.0, "source": "screenshot", "pattern": None
    },
    # SPY 692 Put 0DTE (IMG_0419)
    {
        "ticker": "SPY", "trade_type": "PUT", "strike": 692.0, "expiry": "2026-01-29",
        "direction": "long", "entry_price": 1.78, "exit_price": 2.92, "quantity": 40,
        "profit_loss_dollars": 6759.14, "profit_loss_pct": 64.0, "platform": "WSB",
        "trade_date": "2026-01-29", "is_0dte": 1, "notes": "SPY 0DTE put scalp",
        "confidence": 0.9, "source": "screenshot", "pattern": "REVERSAL_PUT"
    },
    # GLD 450C closed trades (IMG_0424) - batch 1
    {
        "ticker": "GLD", "trade_type": "CALL", "strike": 450.0, "expiry": "2026-06-30",
        "direction": "long", "entry_price": 20.0, "exit_price": 64.5, "quantity": 2,
        "profit_loss_dollars": 8900.0, "profit_loss_pct": 222.5, "platform": "thinkorswim",
        "trade_date": "2026-01-29", "is_0dte": 0, "notes": "Gold LEAPS swing",
        "confidence": 1.0, "source": "screenshot", "pattern": "PRECIOUS_METALS_MOMENTUM"
    },
    # GLD 450C closed trades (IMG_0424) - batch 2
    {
        "ticker": "GLD", "trade_type": "CALL", "strike": 450.0, "expiry": "2026-06-30",
        "direction": "long", "entry_price": 20.0, "exit_price": 69.69, "quantity": 3,
        "profit_loss_dollars": 14907.0, "profit_loss_pct": 248.5, "platform": "thinkorswim",
        "trade_date": "2026-01-29", "is_0dte": 0, "notes": "Gold LEAPS swing",
        "confidence": 1.0, "source": "screenshot", "pattern": "PRECIOUS_METALS_MOMENTUM"
    },
    # GLD 475C (IMG_0412)
    {
        "ticker": "GLD", "trade_type": "CALL", "strike": 475.0, "expiry": "2026-03-31",
        "direction": "long", "entry_price": 10.0, "exit_price": 38.925, "quantity": 15,
        "profit_loss_dollars": 43377.59, "profit_loss_pct": 289.0, "platform": "thinkorswim",
        "trade_date": "2026-01-29", "is_0dte": 0, "notes": "WSB post 15k to 58k",
        "confidence": 1.0, "source": "screenshot", "pattern": "PRECIOUS_METALS_MOMENTUM"
    },
]


def insert_trades(db_path: str):
    """Insert screenshot trades into learned_trades table using EXACT schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    inserted = 0
    skipped = 0

    for trade in SCREENSHOT_TRADES:
        # Check for duplicates by ticker + strike + trade_date + profit_loss_dollars
        cursor.execute("""
            SELECT id FROM learned_trades
            WHERE ticker = ? AND strike = ? AND trade_date = ? AND profit_loss_dollars = ?
        """, (trade["ticker"], trade.get("strike"), trade["trade_date"], trade["profit_loss_dollars"]))

        if cursor.fetchone():
            print(f"SKIP duplicate: {trade['ticker']} {trade.get('strike')} {trade['trade_date']}")
            skipped += 1
            continue

        # INSERT using EXACT column names from schema
        cursor.execute("""
            INSERT INTO learned_trades
            (ticker, trade_type, strike, expiry, direction, entry_price, exit_price,
             quantity, profit_loss_dollars, profit_loss_pct, platform, trade_date,
             is_0dte, notes, raw_text, confidence, source, catalyst, pattern)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade["ticker"],
            trade["trade_type"],
            trade.get("strike"),
            trade.get("expiry"),
            trade["direction"],
            trade["entry_price"],
            trade.get("exit_price"),
            trade.get("quantity"),
            trade["profit_loss_dollars"],
            trade.get("profit_loss_pct"),
            trade["platform"],
            trade["trade_date"],
            trade["is_0dte"],
            trade.get("notes"),
            None,  # raw_text
            trade.get("confidence", 0.9),
            trade["source"],
            None,  # catalyst
            trade.get("pattern")
        ))
        inserted += 1
        print(f"INSERT: {trade['ticker']} ${trade.get('strike')} {trade['trade_type']} -> ${trade['profit_loss_dollars']:,.2f}")

    conn.commit()

    # Show summary
    cursor.execute("SELECT COUNT(*) FROM learned_trades")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(profit_loss_dollars) FROM learned_trades WHERE profit_loss_dollars > 0")
    total_profit = cursor.fetchone()[0] or 0

    cursor.execute("SELECT SUM(profit_loss_dollars) FROM learned_trades WHERE profit_loss_dollars < 0")
    total_loss = cursor.fetchone()[0] or 0

    cursor.execute("SELECT COUNT(DISTINCT pattern) FROM learned_trades WHERE pattern IS NOT NULL")
    pattern_count = cursor.fetchone()[0]

    print(f"\n{'='*50}")
    print(f"INSERTED: {inserted} new trades")
    print(f"SKIPPED: {skipped} duplicates")
    print(f"TOTAL IN DB: {total}")
    print(f"TOTAL PROFITS: ${total_profit:,.2f}")
    print(f"TOTAL LOSSES: ${total_loss:,.2f}")
    print(f"NET P&L: ${total_profit + total_loss:,.2f}")
    print(f"UNIQUE PATTERNS: {pattern_count}")
    print(f"{'='*50}")

    conn.close()
    return inserted


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "wsb_snake_data/wsb_snake.db"
    insert_trades(db_path)
