"""
FRESH BLOOD — Historical Gap Checker

Check what gaps would have qualified under V2 rules
over the past N trading days.

Use this to validate the strategy without backtesting option prices.
Just checks: "Would this gap have triggered a trade?"
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict
import logging

from config import (
    ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, ALPACA_DATA_URL,
    TICKERS, GAP_DOWN_MIN, GAP_UP_MIN
)

log = logging.getLogger(__name__)


def get_headers() -> Dict:
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET
    }


def get_trading_days(days_back: int = 30) -> List[str]:
    """Get list of trading days."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back + 10)).strftime("%Y-%m-%d")

    resp = requests.get(
        f"{ALPACA_BASE_URL}/v2/calendar",
        headers=get_headers(),
        params={"start": start, "end": end}
    )

    if resp.status_code == 200:
        return [d["date"] for d in resp.json()][-days_back:]
    return []


def get_daily_bars(ticker: str, start_date: str, end_date: str) -> List[Dict]:
    """Get daily bars for date range."""
    resp = requests.get(
        f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars",
        headers=get_headers(),
        params={
            "start": f"{start_date}T00:00:00Z",
            "end": f"{end_date}T23:59:59Z",
            "timeframe": "1Day",
            "limit": 100,
            "adjustment": "raw"
        }
    )

    if resp.status_code == 200:
        return resp.json().get("bars", [])
    return []


def check_historical_gaps(days_back: int = 10) -> Dict:
    """
    Check historical gaps against V2 rules.

    Returns summary of what would have been traded.
    """
    log.info("=" * 70)
    log.info(f"HISTORICAL GAP CHECK — Last {days_back} Trading Days")
    log.info("=" * 70)
    log.info(f"V2 Rules: Gap DOWN ≤{GAP_DOWN_MIN*100}% = TRADE | Gap UP ≥{GAP_UP_MIN*100}% = TRADE")
    log.info(f"Tickers: {', '.join(TICKERS)}")

    trading_days = get_trading_days(days_back + 5)  # Get extra days for lookback

    if len(trading_days) < 2:
        log.error("Not enough trading days")
        return {}

    results = {
        "period": f"{trading_days[0]} to {trading_days[-1]}",
        "days_checked": 0,
        "total_gaps_5pct": 0,
        "v1_would_trade": 0,
        "v2_would_trade": 0,
        "v2_skipped": 0,
        "signals": []
    }

    for i in range(1, len(trading_days)):
        prev_date = trading_days[i - 1]
        trade_date = trading_days[i]

        log.info(f"\n{trade_date}:")
        results["days_checked"] += 1

        for ticker in TICKERS:
            # Get bars for both days
            bars = get_daily_bars(ticker, prev_date, trade_date)

            if len(bars) < 2:
                continue

            # Find the two days
            prev_bar = None
            trade_bar = None
            for bar in bars:
                bar_date = bar["t"][:10]
                if bar_date == prev_date:
                    prev_bar = bar
                elif bar_date == trade_date:
                    trade_bar = bar

            if not prev_bar or not trade_bar:
                continue

            prev_close = prev_bar["c"]
            trade_open = trade_bar["o"]
            trade_close = trade_bar["c"]
            trade_low = trade_bar["l"]
            trade_high = trade_bar["h"]

            gap_pct = (trade_open - prev_close) / prev_close

            # Only log 5%+ gaps
            if abs(gap_pct) < 0.05:
                continue

            results["total_gaps_5pct"] += 1

            # V1 would trade any 5%+ gap
            results["v1_would_trade"] += 1

            # Check V2 rules
            if gap_pct <= GAP_DOWN_MIN:
                # Gap DOWN - V2 trades this
                action = "TRADE (gap down fade)"
                results["v2_would_trade"] += 1

                # Check if gap filled
                if trade_high >= prev_close:
                    outcome = "FILLED"
                else:
                    move_back = (trade_high - trade_open) / (prev_close - trade_open) * 100
                    outcome = f"PARTIAL ({move_back:.0f}% fill)"

                signal = {
                    "date": trade_date,
                    "ticker": ticker,
                    "gap_pct": gap_pct,
                    "action": "CALL",
                    "v2_action": "TRADE",
                    "outcome": outcome
                }
                results["signals"].append(signal)
                log.info(f"  ✅ {ticker}: {gap_pct*100:+.2f}% → CALLS ({outcome})")

            elif gap_pct >= GAP_UP_MIN:
                # Large gap UP - V2 trades this
                action = "TRADE (large gap up fade)"
                results["v2_would_trade"] += 1

                # Check if gap filled
                if trade_low <= prev_close:
                    outcome = "FILLED"
                else:
                    move_back = (trade_open - trade_low) / (trade_open - prev_close) * 100
                    outcome = f"PARTIAL ({move_back:.0f}% fill)"

                signal = {
                    "date": trade_date,
                    "ticker": ticker,
                    "gap_pct": gap_pct,
                    "action": "PUT",
                    "v2_action": "TRADE",
                    "outcome": outcome
                }
                results["signals"].append(signal)
                log.info(f"  ✅ {ticker}: {gap_pct*100:+.2f}% → PUTS ({outcome})")

            elif gap_pct >= 0.05:
                # Gap UP 5-10% - V2 SKIPS this
                action = "SKIP (danger zone)"
                results["v2_skipped"] += 1

                # Check what would have happened if we traded
                if trade_close < trade_open:
                    would_have = "would have won (price fell)"
                else:
                    would_have = "AVOIDED LOSS (price rose)"

                signal = {
                    "date": trade_date,
                    "ticker": ticker,
                    "gap_pct": gap_pct,
                    "action": "PUT",
                    "v2_action": "SKIP",
                    "outcome": would_have
                }
                results["signals"].append(signal)
                log.info(f"  ⏭️  {ticker}: {gap_pct*100:+.2f}% → SKIPPED ({would_have})")

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"Period: {results['period']}")
    log.info(f"Days checked: {results['days_checked']}")
    log.info(f"Total 5%+ gaps found: {results['total_gaps_5pct']}")
    log.info(f"V1 would have traded: {results['v1_would_trade']}")
    log.info(f"V2 would have traded: {results['v2_would_trade']}")
    log.info(f"V2 skipped (5-10% up): {results['v2_skipped']}")

    if results["signals"]:
        log.info("\nV2 SIGNALS:")
        for s in results["signals"]:
            action = s["v2_action"]
            icon = "✅" if action == "TRADE" else "⏭️"
            log.info(f"  {icon} {s['date']} {s['ticker']}: {s['gap_pct']*100:+.1f}% → {s['action']} [{action}] - {s['outcome']}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    results = check_historical_gaps(10)
