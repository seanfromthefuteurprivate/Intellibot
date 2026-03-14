#!/usr/bin/env python3
"""
VALIDATION SCRIPT - Check if GAP FADE +1,269% is REAL or FAKE
Uses REAL option prices from Alpaca/Polygon APIs
"""

import os
import json
import requests
from datetime import datetime, timedelta

# API Keys
ALP_KEY = "PKWT6T5BFKHBTFDW3CPAFW2XBZ"
ALP_SEC = "pVdzbVte2pQvL1RmCTFw3oaQ6TBWYimAzC42DUyTEy8"
POLYGON_KEY = "oraSCKTscol9VtSNF0O6_YITBH1iQ90H"

ALP_H = {'APCA-API-KEY-ID': ALP_KEY, 'APCA-API-SECRET-KEY': ALP_SEC}

# The 4 trades from the backtest
TRADES = [
    {"date": "2026-02-25", "ticker": "COIN", "dir": "CALL", "strike": 171, "entry_stock": 171.00, "exit_stock": 185.00},
    {"date": "2026-02-26", "ticker": "NVDA", "dir": "PUT", "strike": 197, "entry_stock": 197.46, "exit_stock": 185.26},
    {"date": "2026-02-27", "ticker": "MARA", "dir": "PUT", "strike": 10, "entry_stock": 9.57, "exit_stock": 9.10},
    {"date": "2026-03-02", "ticker": "MARA", "dir": "CALL", "strike": 9, "entry_stock": 8.66, "exit_stock": 9.74},
]

# All tickers to scan for gaps
UNIVERSE = ['TSLA','NVDA','AMD','COIN','MARA','RIOT','PLTR','SOFI','GME','AMC',
            'HOOD','SMCI','ARM','SNAP','SQ','SHOP','RBLX','DKNG','META','GOOGL']

def format_option_symbol(ticker, date, cp, strike):
    """Format Alpaca option symbol: COIN260225C00171000"""
    dt = datetime.strptime(date, "%Y-%m-%d")
    date_str = dt.strftime("%y%m%d")
    strike_str = str(int(strike * 1000)).zfill(8)
    return f"{ticker}{date_str}{cp[0].upper()}{strike_str}"

def get_option_bars_alpaca(symbol, date):
    """Get option bars from Alpaca"""
    try:
        start = f"{date}T14:00:00Z"  # 10:00 AM ET = 14:00 UTC
        end = f"{date}T21:00:00Z"    # 4:00 PM ET = 20:00 UTC

        url = f"https://data.alpaca.markets/v1beta1/options/bars"
        params = {
            'symbols': symbol,
            'timeframe': '1Min',
            'start': start,
            'end': end,
            'limit': 500
        }
        r = requests.get(url, headers=ALP_H, params=params, timeout=10)
        print(f"  Alpaca bars {symbol}: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if 'bars' in data and symbol in data['bars']:
                return data['bars'][symbol]
        return None
    except Exception as e:
        print(f"  Alpaca error: {e}")
        return None

def get_option_snapshot_alpaca(symbol):
    """Get option snapshot from Alpaca"""
    try:
        url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}"
        r = requests.get(url, headers=ALP_H, timeout=10)
        print(f"  Alpaca snapshot {symbol}: {r.status_code}")
        if r.status_code == 200:
            return r.json()
        return None
    except Exception as e:
        print(f"  Alpaca snapshot error: {e}")
        return None

def get_option_bars_polygon(ticker, date, strike, cp):
    """Get option bars from Polygon"""
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        exp_date = dt.strftime("%y%m%d")
        # Polygon format: O:COIN260225C00171000
        symbol = f"O:{ticker}{exp_date}{cp[0].upper()}{str(int(strike*1000)).zfill(8)}"

        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{date}/{date}"
        params = {'apiKey': POLYGON_KEY, 'limit': 500}
        r = requests.get(url, params=params, timeout=10)
        print(f"  Polygon {symbol}: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if 'results' in data and data['results']:
                return data['results']
        return None
    except Exception as e:
        print(f"  Polygon error: {e}")
        return None

def get_stock_bars(ticker, date):
    """Get stock bars from Alpaca"""
    try:
        start = f"{date}T09:30:00-04:00"
        end = f"{date}T16:00:00-04:00"
        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
        params = {'start': start, 'end': end, 'timeframe': '1Min', 'limit': 500}
        r = requests.get(url, headers=ALP_H, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get('bars', [])
        return []
    except:
        return []

def get_prev_close(ticker, date):
    """Get previous day's close"""
    try:
        dt = datetime.strptime(date, "%Y-%m-%d")
        prev = dt - timedelta(days=1)
        # Skip weekends
        while prev.weekday() >= 5:
            prev -= timedelta(days=1)
        prev_str = prev.strftime("%Y-%m-%d")

        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
        params = {'start': f'{prev_str}T00:00:00-04:00', 'end': f'{date}T09:30:00-04:00',
                  'timeframe': '1Day', 'limit': 5}
        r = requests.get(url, headers=ALP_H, params=params, timeout=10)
        if r.status_code == 200:
            bars = r.json().get('bars', [])
            if bars:
                return bars[-1]['c']
        return None
    except:
        return None

def scan_all_gaps(start_date, end_date):
    """Scan ALL gaps in date range to check for cherry-picking"""
    print("\n" + "="*70)
    print("SCANNING ALL GAPS - CHECKING FOR CHERRY-PICKING")
    print("="*70)

    all_gaps = []
    cur = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while cur <= end:
        if cur.weekday() >= 5:
            cur += timedelta(days=1)
            continue

        ds = cur.strftime("%Y-%m-%d")
        print(f"\n{ds}:")

        for ticker in UNIVERSE:
            try:
                prev_close = get_prev_close(ticker, ds)
                if not prev_close:
                    continue

                bars = get_stock_bars(ticker, ds)
                if not bars or len(bars) < 5:
                    continue

                open_price = bars[0]['o']
                gap_pct = ((open_price - prev_close) / prev_close) * 100

                if abs(gap_pct) >= 5.0:
                    # Check if gap faded or continued
                    entry_bar = None
                    for b in bars:
                        if ':30:' in b['t'] and '10:' in b['t']:
                            entry_bar = b
                            break
                    if not entry_bar and len(bars) > 30:
                        entry_bar = bars[30]  # ~10:00 AM

                    if entry_bar:
                        entry_price = entry_bar['c']
                        close_price = bars[-1]['c']

                        # Did gap fade?
                        if gap_pct > 0:  # Gap up
                            fade_pct = ((entry_price - close_price) / entry_price) * 100
                            faded = close_price < entry_price
                        else:  # Gap down
                            fade_pct = ((close_price - entry_price) / entry_price) * 100
                            faded = close_price > entry_price

                        result = "FADED" if faded else "CONTINUED"
                        all_gaps.append({
                            'date': ds,
                            'ticker': ticker,
                            'gap': gap_pct,
                            'entry': entry_price,
                            'close': close_price,
                            'fade_pct': fade_pct,
                            'result': result
                        })
                        print(f"  {ticker}: GAP {gap_pct:+.1f}% -> {result} ({fade_pct:+.1f}% move)")

            except Exception as e:
                continue

        cur += timedelta(days=1)

    return all_gaps

def validate_trades():
    """Validate the 4 trades with real option prices"""
    print("="*70)
    print("VALIDATING GAP FADE TRADES - REAL OPTION PRICES")
    print("="*70)

    results = []

    for trade in TRADES:
        print(f"\n{'='*60}")
        print(f"Trade: {trade['ticker']} {trade['date']} {trade['dir']}")
        print(f"Stock: ${trade['entry_stock']:.2f} -> ${trade['exit_stock']:.2f}")
        print(f"{'='*60}")

        # Calculate fake return (what the backtest used)
        if trade['dir'] == 'CALL':
            stock_move = (trade['exit_stock'] - trade['entry_stock']) / trade['entry_stock']
        else:
            stock_move = (trade['entry_stock'] - trade['exit_stock']) / trade['entry_stock']

        fake_return = stock_move * 12  # The hardcoded 12x
        print(f"Fake return (12x): {fake_return*100:+.1f}%")

        # Try to get real option prices
        option_symbol = format_option_symbol(
            trade['ticker'],
            trade['date'],
            trade['dir'],
            trade['strike']
        )
        print(f"Option symbol: {option_symbol}")

        # Try Alpaca first
        bars = get_option_bars_alpaca(option_symbol, trade['date'])

        # Try Polygon if Alpaca fails
        if not bars:
            bars = get_option_bars_polygon(
                trade['ticker'],
                trade['date'],
                trade['strike'],
                trade['dir']
            )

        if bars and len(bars) > 0:
            # Get entry price (first bar after 10:00 AM)
            entry_price = bars[0]['c'] if 'c' in bars[0] else bars[0].get('close', 0)
            exit_price = bars[-1]['c'] if 'c' in bars[-1] else bars[-1].get('close', 0)

            if entry_price > 0 and exit_price > 0:
                real_return = (exit_price - entry_price) / entry_price
                print(f"REAL option prices: ${entry_price:.2f} -> ${exit_price:.2f}")
                print(f"REAL return: {real_return*100:+.1f}%")
            else:
                real_return = None
                print("Could not parse option prices")
        else:
            real_return = None
            print("NO REAL OPTION DATA AVAILABLE")

            # Estimate with Black-Scholes-ish logic
            print("\nEstimating with option math:")
            atm_premium_pct = 0.02  # ~2% of stock price for ATM 0DTE
            entry_premium = trade['entry_stock'] * atm_premium_pct

            # Intrinsic value at exit
            if trade['dir'] == 'CALL':
                intrinsic = max(0, trade['exit_stock'] - trade['strike'])
            else:
                intrinsic = max(0, trade['strike'] - trade['exit_stock'])

            exit_premium = intrinsic + 0.05  # Small extrinsic remaining

            if entry_premium > 0:
                estimated_return = (exit_premium - entry_premium) / entry_premium
                print(f"  Estimated entry premium: ${entry_premium:.2f}")
                print(f"  Estimated exit premium: ${exit_premium:.2f}")
                print(f"  Estimated return: {estimated_return*100:+.1f}%")
                real_return = estimated_return

        results.append({
            'trade': trade,
            'fake_return': fake_return,
            'real_return': real_return
        })

    return results

def main():
    print("="*70)
    print("GAP FADE VALIDATION - IS +1,269% REAL OR FAKE?")
    print("="*70)

    # Step 1: Validate the 4 trades
    trade_results = validate_trades()

    # Step 2: Scan ALL gaps for cherry-picking
    all_gaps = scan_all_gaps("2026-02-24", "2026-03-12")

    # Step 3: Print comparison
    print("\n" + "="*70)
    print("COMPARISON: FAKE vs REAL RETURNS")
    print("="*70)

    print(f"\n{'Trade':<20} {'Fake 12x':<15} {'Real/Est':<15} {'Difference':<15}")
    print("-"*65)

    fake_capital = 5000
    real_capital = 5000

    for r in trade_results:
        t = r['trade']
        name = f"{t['ticker']} {t['date']}"
        fake = r['fake_return']
        real = r['real_return'] if r['real_return'] else 0

        fake_pnl = fake_capital * fake
        real_pnl = real_capital * real if real else 0

        print(f"{name:<20} {fake*100:>+10.1f}%     {real*100:>+10.1f}%     {(fake-real)*100:>+10.1f}%")

        fake_capital += fake_pnl
        if real:
            real_capital += real_pnl

    print("-"*65)
    fake_total = ((fake_capital - 5000) / 5000) * 100
    real_total = ((real_capital - 5000) / 5000) * 100

    print(f"\n{'TOTALS':<20} {fake_total:>+10.1f}%     {real_total:>+10.1f}%")
    print(f"\nFake capital: ${fake_capital:,.0f}")
    print(f"Real capital: ${real_capital:,.0f}")

    # Step 4: Gap cherry-picking analysis
    print("\n" + "="*70)
    print("CHERRY-PICKING ANALYSIS")
    print("="*70)

    if all_gaps:
        total_gaps = len(all_gaps)
        faded = len([g for g in all_gaps if g['result'] == 'FADED'])
        continued = len([g for g in all_gaps if g['result'] == 'CONTINUED'])

        print(f"\nTotal 5%+ gaps found: {total_gaps}")
        print(f"Gaps that FADED: {faded} ({faded/total_gaps*100:.0f}%)")
        print(f"Gaps that CONTINUED: {continued} ({continued/total_gaps*100:.0f}%)")

        print("\nALL GAPS IN DATE RANGE:")
        print(f"{'Date':<12} {'Ticker':<6} {'Gap':<8} {'Result':<10} {'Move':<8}")
        print("-"*50)
        for g in all_gaps:
            print(f"{g['date']:<12} {g['ticker']:<6} {g['gap']:>+5.1f}%  {g['result']:<10} {g['fade_pct']:>+5.1f}%")

        # Check if backtest used all gaps or just winners
        backtest_dates = ['2026-02-25', '2026-02-26', '2026-02-27', '2026-03-02']
        missing_trades = [g for g in all_gaps if g['date'] not in backtest_dates]

        if missing_trades:
            print(f"\n{'!'*60}")
            print("GAPS NOT INCLUDED IN BACKTEST (POSSIBLE CHERRY-PICKING):")
            print(f"{'!'*60}")
            for g in missing_trades:
                print(f"  {g['date']} {g['ticker']}: {g['gap']:+.1f}% -> {g['result']}")

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
