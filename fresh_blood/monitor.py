"""
FRESH BLOOD — Position Monitor

Monitors open positions and exits based on:
- Stop loss (-40%)
- Profit target (+100%)
- End of day (3:50 PM ET)
"""

import json
import time
import requests
from datetime import datetime
from typing import Dict, List
import logging

from config import (
    ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, ALPACA_DATA_URL,
    STOP_LOSS_PCT, PROFIT_TARGET_PCT, EXIT_TIME, TRADES_FILE
)

log = logging.getLogger(__name__)


def get_headers() -> Dict:
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET
    }


def get_positions() -> List[Dict]:
    """Get all open positions."""
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/positions", headers=get_headers())
    return resp.json() if resp.status_code == 200 else []


def get_option_positions() -> List[Dict]:
    """Get only option positions."""
    positions = get_positions()
    return [p for p in positions if p.get("asset_class") == "us_option"]


def close_position(symbol: str, qty: int) -> Dict:
    """Close position by selling."""
    order = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "sell",
        "type": "market",
        "time_in_force": "day"
    }

    resp = requests.post(
        f"{ALPACA_BASE_URL}/v2/orders",
        headers=get_headers(),
        json=order
    )

    return resp.json()


def is_market_hours() -> bool:
    """Check if market is open (simplified ET check)."""
    now = datetime.now()
    hour, minute = now.hour, now.minute

    # Before 9:30 AM
    if hour < 9 or (hour == 9 and minute < 30):
        return False
    # After 4:00 PM
    if hour >= 16:
        return False

    return True


def is_exit_time() -> bool:
    """Check if it's time to exit all positions."""
    now = datetime.now()
    exit_h, exit_m = map(int, EXIT_TIME.split(":"))

    return now.hour > exit_h or (now.hour == exit_h and now.minute >= exit_m)


def load_trades() -> List[Dict]:
    """Load active trades from file."""
    try:
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_trades(trades: List[Dict]):
    """Save trades to file."""
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)


def monitor_positions(check_interval: int = 30):
    """
    Main monitoring loop.

    Checks positions every `check_interval` seconds.
    Exits on stop loss, profit target, or EOD.
    """
    log.info("=" * 60)
    log.info("POSITION MONITOR STARTED")
    log.info("=" * 60)
    log.info(f"Stop Loss: -{STOP_LOSS_PCT*100:.0f}%")
    log.info(f"Profit Target: +{PROFIT_TARGET_PCT*100:.0f}%")
    log.info(f"EOD Exit: {EXIT_TIME}")
    log.info(f"Check Interval: {check_interval}s")

    results = []

    while True:
        # Check market hours
        if not is_market_hours():
            log.info("Market closed. Exiting monitor.")
            break

        # Get option positions
        positions = get_option_positions()

        if not positions:
            log.info("No option positions open.")
            time.sleep(check_interval)

            # If past exit time, stop monitoring
            if is_exit_time():
                log.info("Past exit time with no positions. Done.")
                break
            continue

        # Check each position
        for pos in positions:
            symbol = pos["symbol"]
            qty = abs(int(float(pos["qty"])))
            avg_entry = float(pos["avg_entry_price"])
            current = float(pos["current_price"])
            unrealized_pnl = float(pos["unrealized_pl"])

            # Calculate P&L percentage
            pnl_pct = ((current - avg_entry) / avg_entry) * 100 if avg_entry > 0 else 0

            log.info(f"{symbol}: ${current:.2f} | P&L: {pnl_pct:+.1f}% (${unrealized_pnl:+.2f})")

            exit_reason = None

            # Check stop loss
            if pnl_pct <= -STOP_LOSS_PCT * 100:
                exit_reason = "STOP_LOSS"
                log.warning(f"⛔ STOP LOSS: {symbol} at {pnl_pct:.1f}%")

            # Check profit target
            elif pnl_pct >= PROFIT_TARGET_PCT * 100:
                exit_reason = "PROFIT_TARGET"
                log.info(f"🎯 PROFIT TARGET: {symbol} at {pnl_pct:.1f}%")

            # Check EOD exit
            elif is_exit_time():
                exit_reason = "EOD"
                log.info(f"⏰ EOD EXIT: {symbol}")

            # Exit if needed
            if exit_reason:
                result = close_position(symbol, qty)
                log.info(f"   Closing: {result.get('status', 'unknown')}")

                results.append({
                    "symbol": symbol,
                    "entry": avg_entry,
                    "exit": current,
                    "pnl_pct": pnl_pct,
                    "pnl_dollars": unrealized_pnl,
                    "exit_reason": exit_reason,
                    "exit_time": datetime.now().isoformat()
                })

        # Sleep before next check
        time.sleep(check_interval)

    # Final summary
    log.info("\n" + "=" * 60)
    log.info("MONITORING COMPLETE")
    log.info("=" * 60)

    if results:
        total_pnl = sum(r["pnl_dollars"] for r in results)
        winners = sum(1 for r in results if r["pnl_pct"] > 0)

        for r in results:
            status = "✅" if r["pnl_pct"] > 0 else "❌"
            log.info(f"{status} {r['symbol']}: {r['pnl_pct']:+.1f}% (${r['pnl_dollars']:+.2f}) - {r['exit_reason']}")

        log.info(f"\nTotal P&L: ${total_pnl:+.2f}")
        log.info(f"Win Rate: {winners}/{len(results)}")

        # Save results
        save_trades(results)
        log.info(f"Results saved to {TRADES_FILE}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    monitor_positions()
