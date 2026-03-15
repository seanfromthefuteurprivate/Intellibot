"""
FRESH BLOOD — Pre-Market Mover Scanner

Scans for pre-market movers with large gaps.
Adds them to the gap fade candidate list.

Run at 9:00 AM ET to catch earnings movers and news-driven gaps.
"""

import requests
from datetime import datetime
from typing import List, Dict
import logging

from config import ALPACA_KEY, ALPACA_SECRET, ALPACA_DATA_URL, TICKERS

log = logging.getLogger(__name__)

# Minimum market cap to avoid penny stocks (in millions)
MIN_MARKET_CAP = 1000  # $1B+

# Pre-market mover thresholds
PREMARKET_GAP_THRESHOLD = 0.05  # 5% move in pre-market


def get_headers() -> Dict:
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET
    }


def get_most_active_stocks() -> List[Dict]:
    """
    Get most active stocks from Alpaca screener.
    Returns top movers by volume/change.
    """
    # Alpaca doesn't have a native screener, so we check a predefined list
    # of volatile stocks + our existing tickers

    VOLATILE_UNIVERSE = [
        # Crypto proxies
        "COIN", "MARA", "RIOT", "MSTR", "CLSK", "HUT", "BITF",
        # Meme stocks
        "GME", "AMC", "BBBY", "KOSS",
        # High beta tech
        "HOOD", "SMCI", "ARM", "PLTR", "IONQ", "RGTI", "QUBT",
        # Semis
        "AMD", "NVDA", "MU", "AVGO", "TSM",
        # Mag 7
        "TSLA", "META", "GOOGL", "AMZN", "AAPL", "MSFT",
        # Leveraged ETFs
        "TQQQ", "SQQQ", "SPXL", "SPXS", "UVXY", "VXX"
    ]

    return VOLATILE_UNIVERSE


def scan_premarket_gaps() -> List[Dict]:
    """
    Scan pre-market for stocks with large gaps.

    Returns list of tickers with significant overnight moves.
    """
    log.info("=" * 60)
    log.info("PRE-MARKET MOVER SCAN")
    log.info("=" * 60)

    universe = get_most_active_stocks()
    log.info(f"Scanning {len(universe)} tickers for pre-market gaps...")

    movers = []

    for ticker in universe:
        try:
            # Get latest quote (includes pre-market)
            resp = requests.get(
                f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/quotes/latest",
                headers=get_headers()
            )

            if resp.status_code != 200:
                continue

            quote = resp.json().get("quote", {})
            bid = quote.get("bp", 0)
            ask = quote.get("ap", 0)

            if bid <= 0 or ask <= 0:
                continue

            mid = (bid + ask) / 2

            # Get previous close
            bars_resp = requests.get(
                f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars",
                headers=get_headers(),
                params={
                    "timeframe": "1Day",
                    "limit": 1,
                    "adjustment": "raw"
                }
            )

            if bars_resp.status_code != 200:
                continue

            bars = bars_resp.json().get("bars", [])
            if not bars:
                continue

            prev_close = bars[0]["c"]
            gap_pct = (mid - prev_close) / prev_close

            if abs(gap_pct) >= PREMARKET_GAP_THRESHOLD:
                mover = {
                    "ticker": ticker,
                    "prev_close": prev_close,
                    "premarket_price": mid,
                    "gap_pct": gap_pct,
                    "direction": "UP" if gap_pct > 0 else "DOWN"
                }
                movers.append(mover)
                log.info(f"  🔥 {ticker}: {gap_pct*100:+.2f}% gap")

        except Exception as e:
            log.warning(f"  Error scanning {ticker}: {e}")
            continue

    log.info("=" * 60)
    log.info(f"Found {len(movers)} pre-market movers with 5%+ gaps")

    return movers


def get_enhanced_ticker_list() -> List[str]:
    """
    Get enhanced ticker list including pre-market movers.

    Call this at 9:00 AM to update the scan list for 9:35 AM.
    """
    # Start with base tickers
    tickers = list(TICKERS)

    # Add pre-market movers
    movers = scan_premarket_gaps()
    for mover in movers:
        if mover["ticker"] not in tickers:
            tickers.append(mover["ticker"])
            log.info(f"Added {mover['ticker']} ({mover['gap_pct']*100:+.1f}%) to scan list")

    return tickers


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    movers = scan_premarket_gaps()

    print("\nPRE-MARKET MOVERS (5%+ gap):")
    print("-" * 50)
    for m in sorted(movers, key=lambda x: abs(x["gap_pct"]), reverse=True):
        print(f"  {m['ticker']}: {m['gap_pct']*100:+.2f}% ({m['direction']})")
