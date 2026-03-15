"""
FRESH BLOOD — Gap Scanner

Scans for qualifying gaps using VERIFIED rules:
- Gap DOWN 5%+ → BUY CALLS (fade it)
- Gap UP 10%+ → BUY PUTS (fade it)
- Gap UP 5-10% → SKIP (these continue, don't fade)
"""

import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict
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


def get_trading_calendar(days_back: int = 14) -> List[Dict]:
    """Get recent trading calendar."""
    # Use a wider range to handle weekends
    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    resp = requests.get(
        f"{ALPACA_BASE_URL}/v2/calendar",
        headers=get_headers(),
        params={"start": start, "end": today}
    )

    if resp.status_code == 200:
        return resp.json()
    return []


def get_prev_trading_day() -> Optional[str]:
    """Get the previous trading day's date."""
    calendar = get_trading_calendar()
    if len(calendar) >= 2:
        return calendar[-2]["date"]
    return None


def get_daily_bar(ticker: str, date: str) -> Optional[Dict]:
    """Get daily OHLCV bar for a ticker."""
    resp = requests.get(
        f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars",
        headers=get_headers(),
        params={
            "start": f"{date}T00:00:00Z",
            "end": f"{date}T23:59:59Z",
            "timeframe": "1Day",
            "limit": 1,
            "adjustment": "raw"
        }
    )

    if resp.status_code == 200:
        bars = resp.json().get("bars", [])
        return bars[0] if bars else None
    return None


def get_current_price(ticker: str) -> Optional[float]:
    """Get current/latest price for a ticker."""
    resp = requests.get(
        f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/trades/latest",
        headers=get_headers()
    )

    if resp.status_code == 200:
        return resp.json().get("trade", {}).get("p")
    return None


def scan_for_gaps() -> List[Dict]:
    """
    Scan all tickers for qualifying gaps.

    Returns list of gap signals with trade direction.
    """
    log.info("=" * 60)
    log.info("SCANNING FOR GAPS")
    log.info("=" * 60)
    log.info(f"Rules: Gap DOWN ≤{GAP_DOWN_MIN*100}% = CALLS | Gap UP ≥{GAP_UP_MIN*100}% = PUTS")
    log.info(f"Tickers: {', '.join(TICKERS)}")

    prev_date = get_prev_trading_day()
    if not prev_date:
        log.error("Could not determine previous trading day")
        return []

    log.info(f"Previous trading day: {prev_date}")

    signals = []

    for ticker in TICKERS:
        # Get previous close
        prev_bar = get_daily_bar(ticker, prev_date)
        if not prev_bar:
            log.warning(f"  {ticker}: No previous day data")
            continue

        prev_close = prev_bar["c"]

        # Get current price
        current = get_current_price(ticker)
        if not current:
            log.warning(f"  {ticker}: No current price")
            continue

        # Calculate gap
        gap_pct = (current - prev_close) / prev_close

        # Determine action
        if gap_pct <= GAP_DOWN_MIN:
            # Gap DOWN 5%+ → fade with CALLS
            action = "CALLS"
            rule = "GAP_DOWN_FADE"
            signals.append({
                "ticker": ticker,
                "prev_close": prev_close,
                "current": current,
                "gap_pct": gap_pct,
                "direction": "DOWN",
                "trade_type": "CALL",
                "rule": rule,
                "target": prev_close,  # Gap fill target
                "timestamp": datetime.now().isoformat()
            })
            log.info(f"  ✅ {ticker}: {gap_pct*100:+.2f}% → {action} (gap down fade)")

        elif gap_pct >= GAP_UP_MIN:
            # Gap UP 10%+ → fade with PUTS
            action = "PUTS"
            rule = "LARGE_GAP_UP_FADE"
            signals.append({
                "ticker": ticker,
                "prev_close": prev_close,
                "current": current,
                "gap_pct": gap_pct,
                "direction": "UP",
                "trade_type": "PUT",
                "rule": rule,
                "target": prev_close,
                "timestamp": datetime.now().isoformat()
            })
            log.info(f"  ✅ {ticker}: {gap_pct*100:+.2f}% → {action} (large gap up fade)")

        elif gap_pct >= 0.05:
            # Gap UP 5-10% → DANGER ZONE, skip
            log.info(f"  ⏭️  {ticker}: {gap_pct*100:+.2f}% → SKIP (5-10% danger zone)")

        else:
            # Gap < 5% → no signal
            log.info(f"  ── {ticker}: {gap_pct*100:+.2f}% → No signal (gap < 5%)")

    log.info("=" * 60)
    log.info(f"Found {len(signals)} qualifying gap(s)")

    return signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    signals = scan_for_gaps()

    if signals:
        print("\nSIGNALS:")
        for s in signals:
            print(f"  {s['ticker']}: {s['gap_pct']*100:+.1f}% → {s['trade_type']} ({s['rule']})")
    else:
        print("\nNo qualifying gaps found.")
