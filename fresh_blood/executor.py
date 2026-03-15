"""
FRESH BLOOD — Trade Executor

Executes gap fade trades with real options.
- Gets option chain from Alpaca
- Selects ATM strike
- Places market order
- Returns trade details for monitoring
"""

import requests
from datetime import datetime
from typing import Optional, Dict, List
import logging

from config import (
    ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL, ALPACA_DATA_URL,
    POSITION_SIZE_USD
)

log = logging.getLogger(__name__)


def get_headers() -> Dict:
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET
    }


def get_account() -> Dict:
    """Get account info."""
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=get_headers())
    return resp.json() if resp.status_code == 200 else {}


def get_positions() -> List[Dict]:
    """Get all open positions."""
    resp = requests.get(f"{ALPACA_BASE_URL}/v2/positions", headers=get_headers())
    return resp.json() if resp.status_code == 200 else []


def get_option_positions() -> List[Dict]:
    """Get only option positions."""
    positions = get_positions()
    return [p for p in positions if p.get("asset_class") == "us_option"]


def get_option_chain(ticker: str, opt_type: str) -> List[Dict]:
    """
    Get option chain with real quotes from Alpaca.

    Returns list of options sorted by strike.
    """
    resp = requests.get(
        f"{ALPACA_DATA_URL}/v1beta1/options/snapshots/{ticker}",
        headers=get_headers(),
        params={"feed": "indicative", "limit": 100}
    )

    if resp.status_code != 200:
        log.error(f"Failed to get options: {resp.text}")
        return []

    data = resp.json()
    snapshots = data.get("snapshots", {})

    # Get today's date for 0DTE filtering
    today = datetime.now().strftime("%y%m%d")

    results = []
    for symbol, snap in snapshots.items():
        # Filter for today's expiry (0DTE)
        if today not in symbol:
            continue

        # Filter for option type (C or P)
        opt_char = "C" if opt_type.upper() == "CALL" else "P"
        if opt_char not in symbol:
            continue

        quote = snap.get("latestQuote", {})
        if not quote:
            continue

        # Extract strike from symbol (e.g., COIN260315P00195000 -> 195.00)
        try:
            strike_str = symbol[-8:]
            strike = int(strike_str) / 1000
        except:
            continue

        bid = quote.get("bp", 0)
        ask = quote.get("ap", 0)

        # Skip if no valid quote
        if bid <= 0 or ask <= 0:
            continue

        results.append({
            "symbol": symbol,
            "strike": strike,
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
            "spread_pct": (ask - bid) / ask * 100 if ask > 0 else 100,
            "iv": snap.get("impliedVolatility", 0),
            "delta": snap.get("greeks", {}).get("delta", 0)
        })

    return sorted(results, key=lambda x: x["strike"])


def find_atm_option(chain: List[Dict], stock_price: float) -> Optional[Dict]:
    """Find ATM option from chain."""
    if not chain:
        return None

    # Find closest strike to stock price
    return min(chain, key=lambda x: abs(x["strike"] - stock_price))


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


def execute_trade(signal: Dict) -> Optional[Dict]:
    """
    Execute a gap fade trade based on signal.

    Returns trade details for monitoring, or None if failed.
    """
    ticker = signal["ticker"]
    stock_price = signal["current"]
    opt_type = signal["trade_type"]
    target = signal["target"]

    log.info(f"\n{'='*60}")
    log.info(f"EXECUTING: {ticker} {opt_type}")
    log.info(f"Gap: {signal['gap_pct']*100:+.2f}% | Target: ${target:.2f}")
    log.info(f"{'='*60}")

    # Get option chain
    chain = get_option_chain(ticker, opt_type)

    if not chain:
        log.error(f"No {opt_type} options found for {ticker}")
        return None

    log.info(f"Found {len(chain)} {opt_type} options")

    # Find ATM option
    atm = find_atm_option(chain, stock_price)

    if not atm:
        log.error("No ATM option found")
        return None

    # Check spread (skip if too wide)
    if atm["spread_pct"] > 20:
        log.warning(f"Spread too wide: {atm['spread_pct']:.1f}%")
        # Continue anyway for now, but log it

    log.info(f"Selected: {atm['symbol']}")
    log.info(f"  Strike: ${atm['strike']:.2f}")
    log.info(f"  Bid: ${atm['bid']:.2f} | Ask: ${atm['ask']:.2f}")
    log.info(f"  Spread: {atm['spread_pct']:.1f}%")

    # Calculate quantity
    option_price = atm["ask"]  # Pay the ask to get filled
    if option_price <= 0:
        log.error("Invalid option price")
        return None

    # Each contract = 100 shares
    contract_cost = option_price * 100
    qty = max(1, int(POSITION_SIZE_USD / contract_cost))
    total_cost = qty * contract_cost

    log.info(f"\nOrder: BUY {qty} x {atm['symbol']} @ ${option_price:.2f}")
    log.info(f"Total Cost: ${total_cost:.2f}")

    # Place order
    order = place_order(atm["symbol"], qty, "buy")

    if "id" in order:
        log.info(f"✅ ORDER PLACED: {order['id']}")
        log.info(f"   Status: {order.get('status')}")

        return {
            "order_id": order["id"],
            "ticker": ticker,
            "symbol": atm["symbol"],
            "strike": atm["strike"],
            "opt_type": opt_type,
            "qty": qty,
            "entry_price": option_price,
            "entry_cost": total_cost,
            "gap_pct": signal["gap_pct"],
            "target": target,
            "stop_price": option_price * 0.60,  # -40% stop
            "rule": signal["rule"],
            "entry_time": datetime.now().isoformat(),
            "status": "OPEN"
        }
    else:
        log.error(f"❌ ORDER FAILED: {order}")
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Test with a mock signal
    test_signal = {
        "ticker": "SPY",
        "current": 500.0,
        "trade_type": "CALL",
        "gap_pct": -0.06,
        "target": 530.0,
        "rule": "GAP_DOWN_FADE"
    }

    print("Testing option chain fetch...")
    chain = get_option_chain("SPY", "CALL")
    print(f"Found {len(chain)} options")
    if chain:
        print(f"Sample: {chain[0]}")
