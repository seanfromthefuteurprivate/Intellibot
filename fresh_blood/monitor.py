"""
FRESH BLOOD — Position Monitor with Pyramiding

Monitors open positions and:
- Pyramids into winners (+30%, +60%)
- Moves stop to breakeven after first pyramid
- Exits on stop loss, profit target, or EOD
"""

import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional
import logging

from config import (
    ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, ALPACA_DATA_URL,
    STOP_LOSS_PCT, PROFIT_TARGET_PCT, EXIT_TIME, TRADES_FILE,
    PYRAMID_ENABLED, PYRAMID_TRIGGER_1, PYRAMID_TRIGGER_2,
    PYRAMID_SIZE_PCT, MAX_PYRAMIDS, MOVE_STOP_AFTER_PYRAMID,
    POSITION_SIZE_USD
)

log = logging.getLogger(__name__)

# Track pyramid state per position
pyramid_state = {}  # symbol -> {"pyramids": 0, "initial_entry": float, "stop_price": float}


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


def place_order(symbol: str, qty: int, side: str = "buy") -> Dict:
    """Place option order."""
    order = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day"
    }

    resp = requests.post(
        f"{ALPACA_BASE_URL}/v2/orders",
        headers=get_headers(),
        json=order
    )

    return resp.json()


def close_position(symbol: str, qty: int) -> Dict:
    """Close position by selling."""
    return place_order(symbol, qty, "sell")


def get_option_quote(symbol: str) -> Optional[Dict]:
    """Get current option quote."""
    resp = requests.get(
        f"{ALPACA_DATA_URL}/v1beta1/options/quotes/latest",
        headers=get_headers(),
        params={"symbols": symbol, "feed": "indicative"}
    )

    if resp.status_code == 200:
        quotes = resp.json().get("quotes", {})
        return quotes.get(symbol)
    return None


def is_market_hours() -> bool:
    """Check if market is open."""
    now = datetime.now()
    hour, minute = now.hour, now.minute

    if hour < 9 or (hour == 9 and minute < 30):
        return False
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


def load_pyramid_state():
    """Load pyramid state from file."""
    global pyramid_state
    try:
        with open("pyramid_state.json", "r") as f:
            pyramid_state = json.load(f)
    except FileNotFoundError:
        pyramid_state = {}


def save_pyramid_state():
    """Save pyramid state to file."""
    with open("pyramid_state.json", "w") as f:
        json.dump(pyramid_state, f, indent=2)


def init_pyramid_state(symbol: str, entry_price: float):
    """Initialize pyramid tracking for a position."""
    if symbol not in pyramid_state:
        pyramid_state[symbol] = {
            "pyramids": 0,
            "initial_entry": entry_price,
            "stop_price": entry_price * (1 - STOP_LOSS_PCT),
            "total_qty": 0
        }
        save_pyramid_state()


def should_pyramid(symbol: str, current_price: float, entry_price: float) -> bool:
    """Check if we should add to this position."""
    if not PYRAMID_ENABLED:
        return False

    state = pyramid_state.get(symbol, {})
    pyramids = state.get("pyramids", 0)

    if pyramids >= MAX_PYRAMIDS:
        return False

    pnl_pct = (current_price - entry_price) / entry_price

    # Check pyramid triggers
    if pyramids == 0 and pnl_pct >= PYRAMID_TRIGGER_1:
        return True
    if pyramids == 1 and pnl_pct >= PYRAMID_TRIGGER_2:
        return True

    return False


def execute_pyramid(symbol: str, current_price: float, initial_qty: int) -> bool:
    """Execute pyramid add."""
    state = pyramid_state.get(symbol, {})

    # Calculate pyramid size (50% of initial)
    pyramid_qty = max(1, int(initial_qty * PYRAMID_SIZE_PCT))

    log.info(f"🔺 PYRAMIDING: Adding {pyramid_qty} contracts to {symbol}")

    # Place order
    result = place_order(symbol, pyramid_qty, "buy")

    if "id" in result:
        log.info(f"   Pyramid order placed: {result['id']}")

        # Update state
        state["pyramids"] = state.get("pyramids", 0) + 1
        state["total_qty"] = state.get("total_qty", 0) + pyramid_qty

        # Move stop to breakeven after first pyramid
        if MOVE_STOP_AFTER_PYRAMID and state["pyramids"] == 1:
            state["stop_price"] = state["initial_entry"]
            log.info(f"   Stop moved to breakeven: ${state['stop_price']:.2f}")

        pyramid_state[symbol] = state
        save_pyramid_state()
        return True
    else:
        log.error(f"   Pyramid failed: {result}")
        return False


def check_pyramid_stop(symbol: str, current_price: float) -> bool:
    """Check if pyramided position hit stop."""
    state = pyramid_state.get(symbol, {})
    stop_price = state.get("stop_price", 0)

    if stop_price > 0 and current_price <= stop_price:
        return True
    return False


def monitor_positions(check_interval: int = 30):
    """
    Main monitoring loop with pyramiding.
    """
    log.info("=" * 60)
    log.info("POSITION MONITOR WITH PYRAMIDING")
    log.info("=" * 60)
    log.info(f"Stop Loss: -{STOP_LOSS_PCT*100:.0f}%")
    log.info(f"Profit Target: +{PROFIT_TARGET_PCT*100:.0f}%")
    log.info(f"Pyramid at: +{PYRAMID_TRIGGER_1*100:.0f}%, +{PYRAMID_TRIGGER_2*100:.0f}%")
    log.info(f"Pyramid size: {PYRAMID_SIZE_PCT*100:.0f}% of initial")
    log.info(f"Max pyramids: {MAX_PYRAMIDS}")
    log.info(f"EOD Exit: {EXIT_TIME}")

    # Load existing pyramid state
    load_pyramid_state()

    results = []

    while True:
        if not is_market_hours():
            log.info("Market closed. Exiting monitor.")
            break

        positions = get_option_positions()

        if not positions:
            log.info("No option positions open.")
            time.sleep(check_interval)

            if is_exit_time():
                log.info("Past exit time with no positions. Done.")
                break
            continue

        for pos in positions:
            symbol = pos["symbol"]
            qty = abs(int(float(pos["qty"])))
            avg_entry = float(pos["avg_entry_price"])
            current = float(pos["current_price"])
            unrealized_pnl = float(pos["unrealized_pl"])

            # Initialize pyramid state if new position
            init_pyramid_state(symbol, avg_entry)

            state = pyramid_state.get(symbol, {})
            pyramids = state.get("pyramids", 0)

            # Calculate P&L from initial entry (not avg which changes with pyramids)
            initial_entry = state.get("initial_entry", avg_entry)
            pnl_pct = ((current - initial_entry) / initial_entry) * 100

            pyramid_indicator = f"[P{pyramids}]" if pyramids > 0 else ""
            log.info(f"{symbol} {pyramid_indicator}: ${current:.2f} | P&L: {pnl_pct:+.1f}% (${unrealized_pnl:+.2f})")

            exit_reason = None

            # Check pyramid stop (breakeven after first add)
            if pyramids > 0 and check_pyramid_stop(symbol, current):
                exit_reason = "PYRAMID_STOP"
                log.warning(f"⛔ PYRAMID STOP (breakeven): {symbol}")

            # Check regular stop loss
            elif pnl_pct <= -STOP_LOSS_PCT * 100:
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

            # Check pyramid opportunity (before exit checks)
            elif should_pyramid(symbol, current, initial_entry):
                # Estimate initial qty from position size
                initial_qty = max(1, int(POSITION_SIZE_USD / (initial_entry * 100)))
                execute_pyramid(symbol, current, initial_qty)

            # Exit if needed
            if exit_reason:
                result = close_position(symbol, qty)
                log.info(f"   Closing {qty} contracts: {result.get('status', 'unknown')}")

                results.append({
                    "symbol": symbol,
                    "entry": initial_entry,
                    "exit": current,
                    "pnl_pct": pnl_pct,
                    "pnl_dollars": unrealized_pnl,
                    "exit_reason": exit_reason,
                    "pyramids": pyramids,
                    "exit_time": datetime.now().isoformat()
                })

                # Clear pyramid state
                if symbol in pyramid_state:
                    del pyramid_state[symbol]
                    save_pyramid_state()

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
            pyramid_info = f" [P{r['pyramids']}]" if r["pyramids"] > 0 else ""
            log.info(f"{status} {r['symbol']}{pyramid_info}: {r['pnl_pct']:+.1f}% (${r['pnl_dollars']:+.2f}) - {r['exit_reason']}")

        log.info(f"\nTotal P&L: ${total_pnl:+.2f}")
        log.info(f"Win Rate: {winners}/{len(results)}")

        save_trades(results)
        log.info(f"Results saved to {TRADES_FILE}")

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    monitor_positions()
