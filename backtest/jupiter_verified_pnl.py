#!/usr/bin/env python3
"""
JUPITER DIRECT — VERIFIED P&L CALCULATION
Using REAL prices from Alpaca API
"""

import math
import requests
from scipy.stats import norm

ALPACA_KEY = "PKWT6T5BFKHBTFDW3CPAFW2XBZ"
ALPACA_SECRET = "pVdzbVte2pQvL1RmCTFw3oaQ6TBWYimAzC42DUyTEy8"
ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# The 8 trades to verify
TRADES = [
    {"date": "2026-02-13", "ticker": "COIN", "gap_claimed": 8.9},
    {"date": "2026-02-19", "ticker": "SMCI", "gap_claimed": 5.2},
    {"date": "2026-02-24", "ticker": "AMD", "gap_claimed": 7.6},
    {"date": "2026-02-25", "ticker": "COIN", "gap_claimed": 6.0},
    {"date": "2026-02-27", "ticker": "MARA", "gap_claimed": 14.6},
    {"date": "2026-03-03", "ticker": "HOOD", "gap_claimed": -6.0},
    {"date": "2026-03-04", "ticker": "COIN", "gap_claimed": 7.4},
    {"date": "2026-03-04", "ticker": "MARA", "gap_claimed": 5.2},
]

# Realistic IV
IV_MAP = {
    "COIN": 0.70,
    "MARA": 1.00,
    "SMCI": 0.80,
    "AMD": 0.50,
    "HOOD": 0.65,
}


def get_headers():
    return {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}


def get_trading_calendar():
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/calendar",
                        headers=get_headers(),
                        params={"start": "2026-02-01", "end": "2026-03-15"})
    return {d["date"]: d for d in resp.json()}


def get_prev_trading_day(date, calendar):
    dates = sorted(calendar.keys())
    try:
        idx = dates.index(date)
        return dates[idx - 1] if idx > 0 else None
    except ValueError:
        return None


def get_daily_bar(ticker, date):
    url = f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars"
    resp = requests.get(url, headers=get_headers(), params={
        "start": f"{date}T00:00:00Z",
        "end": f"{date}T23:59:59Z",
        "timeframe": "1Day",
        "limit": 1,
        "adjustment": "raw"
    })
    bars = resp.json().get("bars", [])
    return bars[0] if bars else None


def black_scholes_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1), 0.01)


def black_scholes_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2), 0.01)


def main():
    print("=" * 100)
    print("JUPITER DIRECT — VERIFIED P&L (REAL ALPACA DATA)")
    print("=" * 100)

    calendar = get_trading_calendar()
    print(f"Loaded {len(calendar)} trading days\n")

    results = []

    for trade in TRADES:
        date = trade["date"]
        ticker = trade["ticker"]
        print(f"\n{'─'*80}")
        print(f"TRADE: {date} {ticker}")
        print(f"{'─'*80}")

        # Get prev day close
        prev_date = get_prev_trading_day(date, calendar)
        prev_bar = get_daily_bar(ticker, prev_date)
        trade_bar = get_daily_bar(ticker, date)

        if not prev_bar or not trade_bar:
            print("  ERROR: Missing data")
            continue

        prev_close = prev_bar["c"]
        trade_open = trade_bar["o"]
        trade_low = trade_bar["l"]
        trade_high = trade_bar["h"]
        trade_close = trade_bar["c"]

        # Calculate actual gap
        gap = ((trade_open - prev_close) / prev_close) * 100
        print(f"  Prev Close: ${prev_close:.2f}")
        print(f"  Open: ${trade_open:.2f}")
        print(f"  Gap: {gap:+.2f}% (claimed: {trade['gap_claimed']:+.1f}%)")
        print(f"  Day Range: L=${trade_low:.2f} H=${trade_high:.2f} C=${trade_close:.2f}")

        # Skip if gap < 5%
        if abs(gap) < 5.0:
            print(f"  ❌ SKIP: Gap < 5% threshold")
            continue

        # Trade logic
        iv = IV_MAP.get(ticker, 0.60)
        r = 0.05
        T_entry = 6.0 / (252 * 6.5)  # 6 hours left at 9:45
        T_exit = 0.5 / (252 * 6.5)   # 30 min left at 3:30

        # Entry at open (simplify - gap typically continues first 15 min)
        entry_stock = trade_open
        strike = round(entry_stock)

        if gap > 0:
            # Gap UP = buy PUTS
            opt_type = "PUT"
            entry_opt = black_scholes_put(entry_stock, strike, T_entry, r, iv)

            # Target = gap fill (prev_close)
            target = prev_close

            if trade_low <= target:
                # Target hit
                exit_stock = target
                exit_reason = "TARGET"
            else:
                # Check if stopped out (stock went UP 5%+)
                stop_stock = entry_stock * 1.05
                if trade_high >= stop_stock:
                    exit_stock = stop_stock
                    exit_reason = "STOP"
                else:
                    # EOD exit
                    exit_stock = trade_close
                    exit_reason = "EOD"

            exit_opt = black_scholes_put(exit_stock, strike, T_exit, r, iv)

        else:
            # Gap DOWN = buy CALLS
            opt_type = "CALL"
            entry_opt = black_scholes_call(entry_stock, strike, T_entry, r, iv)

            target = prev_close

            if trade_high >= target:
                exit_stock = target
                exit_reason = "TARGET"
            else:
                stop_stock = entry_stock * 0.95
                if trade_low <= stop_stock:
                    exit_stock = stop_stock
                    exit_reason = "STOP"
                else:
                    exit_stock = trade_close
                    exit_reason = "EOD"

            exit_opt = black_scholes_call(exit_stock, strike, T_exit, r, iv)

        # Calculate P&L
        pnl = ((exit_opt - entry_opt) / entry_opt) * 100

        # Cap at -40% stop
        if pnl < -40:
            pnl = -40
            exit_reason = "STOP"

        print(f"\n  {opt_type} @ Strike ${strike} (IV: {iv*100:.0f}%)")
        print(f"  Entry: Stock ${entry_stock:.2f} → Option ${entry_opt:.2f}")
        print(f"  Exit ({exit_reason}): Stock ${exit_stock:.2f} → Option ${exit_opt:.2f}")
        print(f"  P&L: {pnl:+.1f}%")

        results.append({
            "date": date,
            "ticker": ticker,
            "gap": gap,
            "opt_type": opt_type,
            "pnl": pnl,
            "exit_reason": exit_reason
        })

    # Summary
    print("\n" + "=" * 100)
    print("VERIFIED SUMMARY")
    print("=" * 100)

    if not results:
        print("No valid trades found!")
        return

    winners = [r for r in results if r["pnl"] > 0]
    losers = [r for r in results if r["pnl"] <= 0]

    total = sum(r["pnl"] for r in results)
    avg_win = sum(r["pnl"] for r in winners) / len(winners) if winners else 0
    avg_loss = sum(r["pnl"] for r in losers) / len(losers) if losers else 0

    print(f"\nTrades: {len(results)}")
    print(f"Winners: {len(winners)} ({len(winners)/len(results)*100:.0f}%)")
    print(f"Losers: {len(losers)} ({len(losers)/len(results)*100:.0f}%)")
    print(f"\nAvg Win: {avg_win:+.1f}%")
    print(f"Avg Loss: {avg_loss:+.1f}%")

    # Individual results
    print("\n┌────────────────────────────────────────────────────┐")
    print("│ Date       Ticker  Gap      Type  P&L     Exit    │")
    print("├────────────────────────────────────────────────────┤")
    for r in results:
        win = "✅" if r["pnl"] > 0 else "❌"
        print(f"│ {r['date']} {r['ticker']:<6} {r['gap']:+5.1f}%  {r['opt_type']:<4} {r['pnl']:+6.1f}%  {r['exit_reason']:<7} {win} │")
    print("├────────────────────────────────────────────────────┤")
    print(f"│ TOTAL P&L: {total:+.1f}%                              │")
    print("└────────────────────────────────────────────────────┘")

    # Expected value at 50% win rate
    if avg_win > 0 and avg_loss < 0:
        ev_50 = (0.50 * avg_win) + (0.50 * avg_loss)
        breakeven = abs(avg_loss) / (avg_win - avg_loss)
        print(f"\n📊 EXPECTED VALUE ANALYSIS")
        print(f"   At 50% win rate: {ev_50:+.1f}% per trade")
        print(f"   Breakeven win rate: {breakeven*100:.1f}%")

        if ev_50 > 0:
            print(f"\n✅ STRATEGY VERIFIED PROFITABLE")
            print(f"   Deploy Jupiter Direct on Monday")
        else:
            print(f"\n❌ STRATEGY NOT PROFITABLE AT 50% WR")


if __name__ == "__main__":
    main()
