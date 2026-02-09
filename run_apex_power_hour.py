#!/usr/bin/env python3
"""
APEX POWER HOUR - Institutional-Grade Multi-Signal Trading

Combines ALL available analysis systems:
- Technical Analysis (RSI, MACD, SMA, EMA, VWAP)
- Candlestick Patterns (36 patterns + confluence)
- Order Flow (sweeps, blocks, institutional activity)
- Probability Generator (multi-engine fusion)
- Pattern Memory (historical pattern matching)
- AI Verdict (GPT-4/Gemini visual analysis)

Only trades when combined conviction > 70%

Usage:
  python3 run_apex_power_hour.py                  # Live execution
  python3 run_apex_power_hour.py --dry-run        # Alerts only, no execution
  python3 run_apex_power_hour.py --single         # Single scan
  python3 run_apex_power_hour.py --max-trades 5   # Limit trades

Power Hour Settings (3-4 PM ET):
  - 30-second scan intervals
  - +6% target / -10% stop (quick exits)
  - 1.2x position sizing
  - Volume spike detection
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from wsb_snake.execution.power_hour_runner import (
    run_power_hour,
    scan_and_execute,
    POWER_HOUR_WATCHLIST,
    MIN_CONVICTION,
)


def main():
    parser = argparse.ArgumentParser(
        description="APEX Power Hour - Institutional Multi-Signal Trading"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Send alerts but don't execute trades"
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=10,
        help="Maximum trades per session (default: 10)"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run single scan instead of continuous loop"
    )
    parser.add_argument(
        "--min-conviction",
        type=int,
        default=MIN_CONVICTION,
        help=f"Minimum conviction to trade (default: {MIN_CONVICTION})"
    )
    args = parser.parse_args()

    # Override min conviction if specified
    if args.min_conviction != MIN_CONVICTION:
        import wsb_snake.execution.power_hour_runner as runner
        runner.MIN_CONVICTION = args.min_conviction

    print("=" * 60)
    print("APEX POWER HOUR - INSTITUTIONAL MULTI-SIGNAL TRADING")
    print("=" * 60)
    print()
    print(f"Mode: {'DRY RUN (alerts only)' if args.dry_run else 'LIVE EXECUTION'}")
    print(f"Watchlist: {', '.join(POWER_HOUR_WATCHLIST)}")
    print(f"Min Conviction: {args.min_conviction}%")
    print(f"Max Trades: {args.max_trades}")
    print()
    print("Signal Sources:")
    print("  • Technical (RSI, MACD, SMA, EMA, VWAP) - 20%")
    print("  • Candlestick (36 patterns + confluence) - 15%")
    print("  • Order Flow (sweeps, blocks, institutional) - 20%")
    print("  • Probability Generator (multi-engine) - 20%")
    print("  • Pattern Memory (historical match) - 15%")
    print("  • AI Verdict (GPT-4/Gemini) - 10%")
    print()
    print("=" * 60)
    print()

    if args.single:
        print("Running single scan...")
        verdicts = scan_and_execute(dry_run=args.dry_run)
        print(f"\nScan complete: {len(verdicts)} high-conviction signals")
        for v in verdicts:
            print(f"  {v.ticker}: {v.conviction_score:.0f}% {v.direction} -> {v.action}")
    else:
        print("Starting continuous power hour scanning...")
        print("Press Ctrl+C to stop")
        print()
        try:
            run_power_hour(dry_run=args.dry_run, max_trades=args.max_trades)
        except KeyboardInterrupt:
            print("\nStopped by user")


if __name__ == "__main__":
    main()
