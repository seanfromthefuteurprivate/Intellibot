#!/usr/bin/env python3
"""
Jobs Report Tracker – Run once or continuously until Friday NFP (Feb 6, 2026).

Usage:
  python run_jobs_report_tracker.py              # one shot
  python run_jobs_report_tracker.py --loop        # run every 30 min until Fri 5 PM ET
  python run_jobs_report_tracker.py --loop 60    # run every 60 min

Requires: .env with POLYGON_API_KEY.
Outputs:
  - wsb_snake_data/jobs_report_playbook.json
  - wsb_snake_data/JOBS_REPORT_FEB6.md
  - wsb_snake_data/jobs_report_tracker.log (when running with --loop)
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env so POLYGON_API_KEY (and optional OpenAI/DeepSeek) are set
# override=True ensures .env takes precedence over shell environment
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from datetime import datetime, timezone
from typing import Optional
import pytz

from wsb_snake.event_driven.jobs_report_tracker import (
    JobsReportTracker,
    JOBS_REPORT_EVENT_DATE,
    JOBS_REPORT_WATCHLIST,
    BUDGET_WEBBULL_USD,
)
from wsb_snake.config import DATA_DIR

# Stop looping after this time on event day (Fri 5 PM ET)
EVENT_DAY_END_ET = "2026-02-06 17:00:00"
INTERVAL_SECONDS_DEFAULT = 30 * 60  # 30 min


def _now_et():
    return datetime.now(pytz.timezone("America/New_York"))


def _should_stop_loop() -> bool:
    """True if we're past Friday 5 PM ET (stop after event day)."""
    et = pytz.timezone("America/New_York")
    try:
        end = et.localize(datetime(2026, 2, 6, 17, 0, 0))
    except Exception:
        end = datetime(2026, 2, 6, 17, 0, 0).replace(tzinfo=et)
    return _now_et() >= end


def run_once(tracker: JobsReportTracker, log_path: Optional[str] = None) -> None:
    playbook = tracker.run()
    lines = []
    lines.append(f"  {len(playbook.watchlist)} tickers | {len(playbook.recommended_trades)} plays")
    for w in playbook.watchlist:
        lines.append(
            f"  {w['ticker']}: ${w['price']} ({w['change_pct']}%) "
            f"| ATM {w.get('atm_strike')} | bias={w.get('momentum_bias')}"
        )
    for t in playbook.recommended_trades:
        lines.append(
            f"  {t['ticker']} {t['direction'].upper()} @ {t['strike']} "
            f"| entry ~${t['entry_price_est']} | target +{t['exit_target_pct']}% | stop {t['stop_loss_pct']}%"
        )
    for line in lines:
        print(line)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(datetime.now(timezone.utc).isoformat() + " | " + " | ".join(lines) + "\n")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Jobs Report Tracker – NFP Feb 6, 2026")
    parser.add_argument(
        "--loop",
        nargs="?",
        const=INTERVAL_SECONDS_DEFAULT,
        metavar="SECONDS",
        help="Run every N seconds until Fri 5 PM ET (default 1800 = 30 min)",
    )
    args = parser.parse_args()
    interval = int(args.loop) if args.loop is not None else None

    print("Jobs Report Tracker – NFP Feb 6, 2026")
    print("Watchlist:", ", ".join(JOBS_REPORT_WATCHLIST))
    print("Budget (WeBull):", f"${BUDGET_WEBBULL_USD}")
    if interval:
        if interval >= 60:
            print(f"Mode: continuous (every {interval // 60} min until Fri 5 PM ET)")
        else:
            print(f"Mode: continuous (every {interval} sec until Fri 5 PM ET)")
        print("Ctrl+C to stop.")
    print()

    tracker = JobsReportTracker(
        event_date=JOBS_REPORT_EVENT_DATE,
        watchlist=JOBS_REPORT_WATCHLIST,
        budget_usd=BUDGET_WEBBULL_USD,
    )
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)
    log_path = os.path.join(data_dir, "jobs_report_tracker.log") if interval else None

    if not interval:
        run_once(tracker, log_path)
        print()
        print("Playbook written to wsb_snake_data/jobs_report_playbook.json and JOBS_REPORT_FEB6.md")
        return

    run_count = 0
    while True:
        if _should_stop_loop():
            print("Event day ended (Fri 5 PM ET). Stopping.")
            break
        run_count += 1
        now_et = _now_et()
        print(f"[{now_et.strftime('%Y-%m-%d %H:%M ET')}] Run #{run_count}")
        try:
            run_once(tracker, log_path)
        except Exception as e:
            print(f"Run failed: {e}")
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(datetime.now(timezone.utc).isoformat() + " | ERROR: " + str(e) + "\n")
        print()
        if interval >= 60:
            print(f"Next run in {interval // 60} min. Ctrl+C to stop.")
        else:
            print(f"Next run in {interval} sec. Ctrl+C to stop.")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break


if __name__ == "__main__":
    main()
