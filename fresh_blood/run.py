#!/usr/bin/env python3
"""
FRESH BLOOD — Main Entry Point

Simple gap fade strategy with verified rules.
Run this at 9:40 AM ET on trading days.

Usage:
    python run.py scan      # Just scan for gaps (no trades)
    python run.py trade     # Scan and execute trades
    python run.py monitor   # Monitor open positions
    python run.py full      # Full cycle: scan → trade → monitor
"""

import sys
import json
import logging
from datetime import datetime

from config import (
    ALPACA_BASE_URL, ENTRY_WINDOW_START, ENTRY_WINDOW_END,
    MAX_POSITIONS, MAX_DAILY_LOSS, LOG_FILE, TRADES_FILE
)
from scanner import scan_for_gaps
from executor import execute_trade, get_account, get_option_positions
from monitor import monitor_positions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def check_entry_window() -> bool:
    """Check if we're in the entry window."""
    now = datetime.now()
    current = now.strftime("%H:%M")

    if current < ENTRY_WINDOW_START:
        log.warning(f"Too early to trade. Window starts at {ENTRY_WINDOW_START}")
        return False

    if current > ENTRY_WINDOW_END:
        log.warning(f"Too late to enter. Window ended at {ENTRY_WINDOW_END}")
        return False

    return True


def check_daily_loss() -> bool:
    """Check if daily loss limit hit."""
    account = get_account()
    equity = float(account.get("equity", 0))
    last_equity = float(account.get("last_equity", 0))
    daily_pnl = equity - last_equity

    if daily_pnl <= MAX_DAILY_LOSS:
        log.error(f"Daily loss limit hit: ${daily_pnl:.2f}")
        return False

    log.info(f"Daily P&L: ${daily_pnl:+.2f}")
    return True


def check_position_limit() -> bool:
    """Check if we have room for more positions."""
    positions = get_option_positions()
    if len(positions) >= MAX_POSITIONS:
        log.warning(f"Max positions reached: {len(positions)}/{MAX_POSITIONS}")
        return False
    return True


def cmd_scan():
    """Scan for gaps without trading."""
    log.info("\n" + "=" * 60)
    log.info("FRESH BLOOD — SCAN MODE")
    log.info("=" * 60)

    signals = scan_for_gaps()

    if signals:
        log.info(f"\n✅ {len(signals)} signal(s) found:")
        for s in signals:
            log.info(f"   {s['ticker']}: {s['gap_pct']*100:+.1f}% → {s['trade_type']} ({s['rule']})")
    else:
        log.info("\n❌ No qualifying gaps found")

    return signals


def cmd_trade():
    """Scan and execute trades."""
    log.info("\n" + "=" * 60)
    log.info("FRESH BLOOD — TRADE MODE")
    log.info("=" * 60)

    # Pre-checks
    if not check_entry_window():
        return []

    if not check_daily_loss():
        return []

    if not check_position_limit():
        return []

    # Scan
    signals = scan_for_gaps()

    if not signals:
        log.info("No qualifying gaps. Nothing to trade.")
        return []

    # Execute trades
    trades = []
    for signal in signals:
        if not check_position_limit():
            log.warning("Position limit reached. Stopping.")
            break

        trade = execute_trade(signal)
        if trade:
            trades.append(trade)

    # Save trades
    if trades:
        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2)
        log.info(f"\n✅ {len(trades)} trade(s) executed. Saved to {TRADES_FILE}")

    return trades


def cmd_monitor():
    """Monitor open positions."""
    log.info("\n" + "=" * 60)
    log.info("FRESH BLOOD — MONITOR MODE")
    log.info("=" * 60)

    return monitor_positions()


def cmd_full():
    """Full trading cycle: scan → trade → monitor."""
    log.info("\n" + "=" * 60)
    log.info("FRESH BLOOD — FULL CYCLE")
    log.info("=" * 60)
    log.info(f"Time: {datetime.now()}")

    # Account check
    account = get_account()
    log.info(f"Account: ${float(account.get('portfolio_value', 0)):,.2f}")
    log.info(f"Buying Power: ${float(account.get('options_buying_power', 0)):,.2f}")

    # Trade
    trades = cmd_trade()

    if trades:
        log.info("\nStarting position monitor...")
        cmd_monitor()
    else:
        log.info("\nNo trades executed. Nothing to monitor.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "scan":
        cmd_scan()
    elif cmd == "trade":
        cmd_trade()
    elif cmd == "monitor":
        cmd_monitor()
    elif cmd == "full":
        cmd_full()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
