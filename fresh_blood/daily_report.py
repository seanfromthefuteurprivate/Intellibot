"""
FRESH BLOOD — Daily Report Generator

Generates end-of-day summary of gap fade activity.
Run after market close (4:15 PM ET).
"""

import json
import os
from datetime import datetime
from typing import Dict, List
import logging

from config import TRADES_FILE, LOG_FILE

log = logging.getLogger(__name__)


def load_trades() -> List[Dict]:
    """Load today's trades from file."""
    try:
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def parse_log_file() -> Dict:
    """Parse log file for today's activity."""
    stats = {
        "tickers_scanned": 0,
        "gaps_found": 0,
        "gaps_skipped": 0,
        "trades_placed": 0,
        "errors": []
    }

    try:
        with open(LOG_FILE, "r") as f:
            today = datetime.now().strftime("%Y-%m-%d")

            for line in f:
                if today not in line:
                    continue

                if "Scanning" in line:
                    stats["tickers_scanned"] += 1
                elif "GAP DOWN" in line or "LARGE GAP UP" in line:
                    stats["gaps_found"] += 1
                elif "SKIP" in line:
                    stats["gaps_skipped"] += 1
                elif "ORDER PLACED" in line:
                    stats["trades_placed"] += 1
                elif "ERROR" in line or "Error" in line:
                    stats["errors"].append(line.strip())

    except FileNotFoundError:
        stats["errors"].append("Log file not found")

    return stats


def generate_report() -> str:
    """Generate daily report."""
    today = datetime.now().strftime("%Y-%m-%d")

    report = []
    report.append("=" * 60)
    report.append(f"FRESH BLOOD — DAILY REPORT — {today}")
    report.append("=" * 60)

    # Parse log for stats
    stats = parse_log_file()
    report.append(f"\nSCAN SUMMARY:")
    report.append(f"  Tickers scanned: {stats['tickers_scanned']}")
    report.append(f"  Qualifying gaps found: {stats['gaps_found']}")
    report.append(f"  Gaps skipped (5-10% danger zone): {stats['gaps_skipped']}")
    report.append(f"  Trades placed: {stats['trades_placed']}")

    # Load trades
    trades = load_trades()

    if trades:
        report.append(f"\nTRADES:")
        total_pnl = 0
        winners = 0

        for t in trades:
            pnl = t.get("pnl_pct", 0)
            pnl_dollars = t.get("pnl_dollars", 0)
            total_pnl += pnl_dollars

            if pnl > 0:
                winners += 1
                icon = "✅"
            else:
                icon = "❌"

            report.append(f"  {icon} {t.get('symbol', 'N/A')}: {pnl:+.1f}% (${pnl_dollars:+.2f}) - {t.get('exit_reason', 'N/A')}")

        report.append(f"\nDAILY P&L: ${total_pnl:+.2f}")
        report.append(f"Win Rate: {winners}/{len(trades)}")
    else:
        report.append(f"\nTRADES: None today")

    # Errors
    if stats["errors"]:
        report.append(f"\n⚠️ ERRORS:")
        for err in stats["errors"][:5]:  # Show max 5
            report.append(f"  {err[:80]}")

    # GO/NO-GO progress
    report.append(f"\n" + "-" * 60)
    report.append("GO/NO-GO PROGRESS (Need 2+ trades with 50%+ win rate by Friday)")
    report.append("-" * 60)

    # Load all trades from this week (simplified - just today's for now)
    week_trades = trades
    week_wins = sum(1 for t in week_trades if t.get("pnl_pct", 0) > 0)

    report.append(f"  Week trades: {len(week_trades)}")
    report.append(f"  Week wins: {week_wins}")
    if week_trades:
        report.append(f"  Week win rate: {week_wins/len(week_trades)*100:.0f}%")

    report.append("\n" + "=" * 60)

    return "\n".join(report)


def save_report(report: str):
    """Save report to file."""
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"report_{today}.txt"

    with open(filename, "w") as f:
        f.write(report)

    log.info(f"Report saved to {filename}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    report = generate_report()
    print(report)
    save_report(report)


if __name__ == "__main__":
    main()
