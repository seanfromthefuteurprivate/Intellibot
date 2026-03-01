#!/usr/bin/env python3
"""Insert historical trades from screenshots into learned_trades database."""
import sqlite3
import os

DB_PATH = os.environ.get("DB_PATH", "/home/ubuntu/wsb-snake/wsb_snake_data/wsb_snake.db")

# All trades extracted from the 19 screenshots - REAL WINNING TRADES
TRADES = [
    # IMG_0890 - Robinhood: Buy NFLX $90 Call 2/27
    {"ticker": "NFLX", "trade_type": "CALL", "strike": 90, "expiry": "2026-02-27", "direction": "long", "entry_price": 0.04, "exit_price": None, "quantity": 20, "profit_loss_dollars": None, "profit_loss_pct": None, "platform": "Robinhood", "trade_date": "2026-02-25", "is_0dte": 0, "notes": "Entry position - filled at $0.04"},

    # IMG_0891 - Robinhood: Buy NFLX $91 Call 2/27
    {"ticker": "NFLX", "trade_type": "CALL", "strike": 91, "expiry": "2026-02-27", "direction": "long", "entry_price": 0.03, "exit_price": None, "quantity": 25, "profit_loss_dollars": None, "profit_loss_pct": None, "platform": "Robinhood", "trade_date": "2026-02-26", "is_0dte": 0, "notes": "Entry position - filled at $0.03"},

    # IMG_0892 - Robinhood: Sell NFLX $90 Call 2/27 - WINNER
    {"ticker": "NFLX", "trade_type": "CALL", "strike": 90, "expiry": "2026-02-27", "direction": "long", "entry_price": 0.04, "exit_price": 2.37, "quantity": 20, "profit_loss_dollars": 4660, "profit_loss_pct": 5825, "platform": "Robinhood", "trade_date": "2026-02-27", "is_0dte": 1, "notes": "MASSIVE WINNER - 0DTE play, bought at $0.04 sold at $2.37"},

    # IMG_0894 - E*TRADE positions (stocks and LEAPS)
    {"ticker": "AMPY", "trade_type": "STOCK", "strike": None, "expiry": None, "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 5391.62, "profit_loss_pct": 26.25, "platform": "ETRADE", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "Energy stock position"},
    {"ticker": "MRNFF", "trade_type": "STOCK", "strike": None, "expiry": None, "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 4311.32, "profit_loss_pct": 15.55, "platform": "ETRADE", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "Energy stock position"},
    {"ticker": "NOG", "trade_type": "STOCK", "strike": None, "expiry": None, "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 4858.14, "profit_loss_pct": 10.80, "platform": "ETRADE", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "Northern Oil & Gas"},
    {"ticker": "OXY", "trade_type": "STOCK", "strike": None, "expiry": None, "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 1168.76, "profit_loss_pct": 11.01, "platform": "ETRADE", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "Occidental Petroleum"},
    {"ticker": "SM", "trade_type": "STOCK", "strike": None, "expiry": None, "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 13789.97, "profit_loss_pct": 23.54, "platform": "ETRADE", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "SM Energy"},
    {"ticker": "SSLVF", "trade_type": "STOCK", "strike": None, "expiry": None, "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 33510.77, "profit_loss_pct": 85.69, "platform": "ETRADE", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "Silver stock - big winner"},
    {"ticker": "XOM", "trade_type": "CALL", "strike": 125, "expiry": "2028-01-21", "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 112748.41, "profit_loss_pct": 287.23, "platform": "ETRADE", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "XOM LEAPS - massive winner"},
    {"ticker": "XOM", "trade_type": "CALL", "strike": 150, "expiry": "2028-01-21", "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 55691.59, "profit_loss_pct": 429.85, "platform": "ETRADE", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "XOM LEAPS higher strike - even bigger pct gain"},

    # IMG_0896 - Robinhood history: NFLX $95 Call 3/20 trades
    {"ticker": "NFLX", "trade_type": "CALL", "strike": 95, "expiry": "2026-03-20", "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 290, "profit_loss_pct": None, "platform": "Robinhood", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "NFLX swing trade"},
    {"ticker": "NFLX", "trade_type": "CALL", "strike": 95, "expiry": "2026-03-20", "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 24000, "profit_loss_pct": None, "platform": "Robinhood", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "NFLX swing trade - big winner"},
    {"ticker": "NFLX", "trade_type": "CALL", "strike": 95, "expiry": "2026-03-20", "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 6750, "profit_loss_pct": None, "platform": "Robinhood", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "NFLX swing trade"},
    {"ticker": "NFLX", "trade_type": "CALL", "strike": 95, "expiry": "2026-03-20", "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 24600, "profit_loss_pct": None, "platform": "Robinhood", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "NFLX swing trade - big winner"},
    {"ticker": "NFLX", "trade_type": "CALL", "strike": 95, "expiry": "2026-03-20", "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": 26875, "profit_loss_pct": None, "platform": "Robinhood", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "NFLX swing trade - biggest single trade"},
    {"ticker": "NFLX", "trade_type": "CALL", "strike": 95, "expiry": "2026-03-20", "direction": "long", "entry_price": None, "exit_price": None, "quantity": None, "profit_loss_dollars": -5600, "profit_loss_pct": None, "platform": "Robinhood", "trade_date": "2026-02-12", "is_0dte": 0, "notes": "Entry cost for NFLX position"},

    # IMG_0898 - Robinhood position: NFLX $100 Call 3/20
    {"ticker": "NFLX", "trade_type": "CALL", "strike": 100, "expiry": "2026-03-20", "direction": "long", "entry_price": 0.71, "exit_price": 2.35, "quantity": 30, "profit_loss_dollars": 4920, "profit_loss_pct": 230.99, "platform": "Robinhood", "trade_date": "2026-01-26", "is_0dte": 0, "notes": "Open position - today +6270 (+803.85pct)"},

    # IMG_0899 - Reddit WSB: SPXW trades (0DTE SPX)
    {"ticker": "SPXW", "trade_type": "CALL", "strike": 6860, "expiry": "2026-02-27", "direction": "long", "entry_price": None, "exit_price": None, "quantity": 1, "profit_loss_dollars": 2109.92, "profit_loss_pct": 342.48, "platform": "Unknown", "trade_date": "2026-02-27", "is_0dte": 1, "notes": "Reddit WSB post - 0DTE SPX play"},
    {"ticker": "SPXW", "trade_type": "CALL", "strike": 6880, "expiry": "2026-02-27", "direction": "short", "entry_price": None, "exit_price": None, "quantity": 1, "profit_loss_dollars": 87.23, "profit_loss_pct": 62.75, "platform": "Unknown", "trade_date": "2026-02-27", "is_0dte": 1, "notes": "Reddit WSB post - 0DTE SPX spread leg"},

    # IMG_0900 - Reddit WSB: NFLX Mar 20 $100 Call - MONSTER WINNER
    {"ticker": "NFLX", "trade_type": "CALL", "strike": 100, "expiry": "2026-03-20", "direction": "long", "entry_price": 0.07, "exit_price": 1.235, "quantity": 100, "profit_loss_dollars": 11650, "profit_loss_pct": 1664.29, "platform": "Unknown", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "Reddit WSB - MONSTER 16x winner bought 700 now 12350"},

    # IMG_0901 - Reddit WSB: NFLX $84 Call 3/20
    {"ticker": "NFLX", "trade_type": "CALL", "strike": 84, "expiry": "2026-03-20", "direction": "long", "entry_price": 2.42, "exit_price": 10.10, "quantity": 6, "profit_loss_dollars": 4610, "profit_loss_pct": 317.94, "platform": "Unknown", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "Reddit WSB post - closed position"},

    # IMG_0902 - IBKR: CIEN short (LOSING trade - keep for learning)
    {"ticker": "CIEN", "trade_type": "STOCK", "strike": None, "expiry": None, "direction": "short", "entry_price": 347.74, "exit_price": None, "quantity": 600, "profit_loss_dollars": -531.62, "profit_loss_pct": -0.15, "platform": "IBKR", "trade_date": "2026-02-27", "is_0dte": 0, "notes": "Short position - unrealized loss -100.73pct of portfolio"},

    # IMG_0903 & IMG_0904 - Robinhood: QQQ $614 Put 2/26
    {"ticker": "QQQ", "trade_type": "PUT", "strike": 614, "expiry": "2026-02-26", "direction": "long", "entry_price": 3.96, "exit_price": 8.23, "quantity": 17, "profit_loss_dollars": 7263, "profit_loss_pct": 107.96, "platform": "Robinhood", "trade_date": "2026-02-26", "is_0dte": 1, "notes": "0DTE QQQ put - doubled"},

    # IMG_0905 - Robinhood: NVDA $190 Put 2/27
    {"ticker": "NVDA", "trade_type": "PUT", "strike": 190, "expiry": "2026-02-27", "direction": "long", "entry_price": 3.34, "exit_price": 4.55, "quantity": 8, "profit_loss_dollars": 966, "profit_loss_pct": 36.13, "platform": "Robinhood", "trade_date": "2026-02-26", "is_0dte": 0, "notes": "NVDA put play - closed for profit"},

    # IMG_0906 - Robinhood position: UCO $28 Call 3/20
    {"ticker": "UCO", "trade_type": "CALL", "strike": 28, "expiry": "2026-03-20", "direction": "long", "entry_price": 0.75, "exit_price": 2.03, "quantity": 35, "profit_loss_dollars": 4480, "profit_loss_pct": 170.67, "platform": "Robinhood", "trade_date": "2026-02-12", "is_0dte": 0, "notes": "Oil play UCO - today +2205 (+45pct)"},
]

def insert_trades():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    inserted = 0
    for trade in TRADES:
        try:
            cursor.execute("""
                INSERT INTO learned_trades (
                    ticker, trade_type, strike, expiry, direction,
                    entry_price, exit_price, quantity, profit_loss_dollars, profit_loss_pct,
                    platform, trade_date, is_0dte, notes, raw_text, confidence,
                    source, catalyst, pattern
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade["ticker"], trade["trade_type"], trade["strike"], trade["expiry"],
                trade["direction"], trade["entry_price"], trade["exit_price"],
                trade["quantity"], trade["profit_loss_dollars"], trade["profit_loss_pct"],
                trade["platform"], trade["trade_date"], trade["is_0dte"], trade["notes"],
                "manual_screenshot_import", 1.0, "screenshot_historical",
                None, None
            ))
            inserted += 1
            print(f"Inserted: {trade['ticker']} {trade['trade_type']} ${trade['strike']} P/L: ${trade['profit_loss_dollars']}")
        except Exception as e:
            print(f"Error inserting {trade['ticker']}: {e}")

    conn.commit()
    conn.close()
    print(f"\n=== Total inserted: {inserted} trades ===")

    # Print summary
    total_pnl = sum(t["profit_loss_dollars"] or 0 for t in TRADES)
    winners = sum(1 for t in TRADES if (t["profit_loss_dollars"] or 0) > 0)
    losers = sum(1 for t in TRADES if (t["profit_loss_dollars"] or 0) < 0)
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"Winners: {winners} | Losers: {losers}")

if __name__ == "__main__":
    insert_trades()
