#!/usr/bin/env python3
"""
Convexity Proof Layer (CPL) — Standalone runner for Jobs Day / event-vol days.

Strike mode: full event-vol watchlist (index, VIX, rates, dollar, metals, crypto beta, AI/mega).
Sends BUY/SELL alerts to Telegram; human executes. NFP rescheduled per BLS/Reuters:
  Jan 2026 Employment Situation = Wed Feb 11, 2026 @ 8:30am ET (not Fri Feb 6).

Usage:
  python3 run_snake_cpl.py --dry-run                    # generate only, no Telegram/DB
  python3 run_snake_cpl.py --broadcast                  # one shot: DB + Telegram
  python3 run_snake_cpl.py --broadcast --loop 60        # every 60s until event day 5 PM ET
  python3 run_snake_cpl.py --broadcast --power-hour     # overtime: 30s scans until 4 PM ET, up to 20 calls/run
  python3 run_snake_cpl.py --broadcast --test-mode      # send 5 BUY alerts to Telegram (open/close test)

Requires: .env POLYGON_API_KEY; TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID for --broadcast.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)  # .env takes precedence over shell env
except ImportError:
    pass

from datetime import datetime

try:
    import pytz
except ImportError:
    pytz = None

from wsb_snake.execution.jobs_day_cpl import JobsDayCPL, _is_execution_ready
from wsb_snake.execution.call_schema import JobsDayCall
from wsb_snake.notifications.message_templates import format_jobs_day_call
from wsb_snake.notifications.telegram_bot import send_alert


def _now_et():
    if pytz:
        return datetime.now(pytz.timezone("America/New_York"))
    return datetime.utcnow()


def _should_stop_loop() -> bool:
    """True if past event day 5 PM ET. NFP rescheduled to Wed Feb 11, 2026 (BLS/Reuters)."""
    if not pytz:
        return False
    et = pytz.timezone("America/New_York")
    # Event day end: Wed Feb 11, 2026 17:00 ET (NFP = Feb 11 @ 8:30am ET)
    try:
        end = et.localize(datetime(2026, 2, 11, 17, 0, 0))
    except Exception:
        end = datetime(2026, 2, 11, 17, 0, 0).replace(tzinfo=et)
    return _now_et() >= end


def _is_friday_et() -> bool:
    """True if today is Friday ET."""
    if not pytz:
        return False
    return _now_et().weekday() == 4


def _is_market_hours_et() -> bool:
    """True if 9:30–16:00 ET (market open to close)."""
    if not pytz:
        return True
    now = _now_et()
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    if h < 9 or (h == 9 and m < 30):
        return False
    if h >= 16:
        return False
    return True


def _sleep_until_market_open_et() -> None:
    """Sleep until 9:30 AM ET (next weekday if weekend)."""
    if not pytz:
        return
    et = pytz.timezone("America/New_York")
    now = _now_et()
    # Target 9:30 today or next weekday
    target = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now >= target:
        return
    if now.weekday() >= 5:
        # Weekend: skip to next Monday 9:30
        days = 7 - now.weekday()
        from datetime import timedelta
        target = (now + timedelta(days=days)).replace(hour=9, minute=30, second=0, microsecond=0)
    secs = (target - now).total_seconds()
    if secs > 0:
        print(f"[FULL SEND] Sleeping until market open 9:30 ET ({secs:.0f}s)")
        time.sleep(min(secs, 3600))  # Cap 1h to re-check


def main():
    # CRITICAL: Prevent duplicate instances
    from wsb_snake.utils.pid_lock import acquire_lock
    _lock = acquire_lock("cpl-scanner", exit_on_fail=True)

    parser = argparse.ArgumentParser(description="CPL Jobs Day runner")
    parser.add_argument("--dry-run", action="store_true", help="Generate calls only; no DB, no Telegram")
    parser.add_argument("--broadcast", action="store_true", help="Save to DB and send alerts to Telegram")
    parser.add_argument("--loop", type=int, default=0, metavar="SEC", help="Run every SEC seconds until event day 5 PM ET (0 = single run)")
    parser.add_argument("--max-calls", type=int, default=0, metavar="N", help="Override TARGET_BUY_CALLS (0 = use default 10)")
    parser.add_argument("--test-mode", action="store_true", help="Test mode: skip DB writes, send single test message")
    parser.add_argument("--untruncated-tails", action="store_true", help="Paper trader Friday: sequential $250->5 figures, human execution via Telegram")
    parser.add_argument("--high-hitters", type=int, default=0, metavar="N", help="Send top N 20X/4X BUY alerts to Telegram (5-10 recommended); no position tracking")
    parser.add_argument("--power-hour", action="store_true", help="Overtime: 30s scans from now until 4:00 PM ET, up to 20 calls/run, find all opportunities")
    parser.add_argument("--execute", action="store_true", help="Auto-execute CPL calls on Alpaca paper trading")
    args = parser.parse_args()

    # Power hour: 30s loop until close, higher target, shorter cooldown — big moves in remaining ~4h
    power_hour = args.power_hour or (os.environ.get("CPL_POWER_HOUR", "").strip().lower() in ("1", "true", "yes"))
    if power_hour:
        if args.loop == 0:
            args = argparse.Namespace(**{**vars(args), "loop": 30})
        if args.max_calls == 0:
            args = argparse.Namespace(**{**vars(args), "max_calls": 20})
        import wsb_snake.execution.jobs_day_cpl as cpl_module
        cpl_module.CPL_POWER_HOUR = True
        print("[POWER HOUR] 30s scans until 4:00 PM ET | up to 20 calls/run | shorter cooldown — find all opportunities")

    # Untruncated tails: set CPL module flag from CLI or env
    untruncated_tails = args.untruncated_tails or (os.environ.get("CPL_UNTRUNCATED_TAILS", "").strip().lower() in ("1", "true", "yes"))
    if untruncated_tails:
        import wsb_snake.execution.jobs_day_cpl as cpl_module
        cpl_module.UNTRUNCATED_TAILS = True
        print("[PAPER TRADER] UNTRUNCATED_TAILS=1 | Sequential $250 -> 5 figures | Human executes from Telegram")

    # Auto-execute on Alpaca: set CPL module flag from CLI or env
    auto_execute = args.execute or (os.environ.get("CPL_AUTO_EXECUTE", "").strip().lower() in ("1", "true", "yes"))
    if auto_execute:
        import wsb_snake.execution.jobs_day_cpl as cpl_module
        cpl_module.CPL_AUTO_EXECUTE = True
        print("[ALPACA EXECUTE] CPL_AUTO_EXECUTE=1 | All BUY calls will be executed on Alpaca paper trading")

    if not args.broadcast and not args.dry_run and args.high_hitters <= 0:
        print("Use --dry-run (generate only, no Telegram/DB) or --broadcast (Telegram + DB) or --high-hitters N")
        sys.exit(0)

    # Override target if --max-calls specified (only when not untruncated-tails; untruncated uses sequential target)
    if args.max_calls > 0 and not untruncated_tails:
        import wsb_snake.execution.jobs_day_cpl as cpl_module
        cpl_module.TARGET_BUY_CALLS = args.max_calls
        print(f"[TEST MODE] TARGET_BUY_CALLS overridden to {args.max_calls}")

    # Use today (ET) for 0DTE so we find moves tomorrow and every day — same tickers, any movement → BUY/SELL Telegram
    today_et = _now_et()
    event_date = today_et.strftime("%Y-%m-%d")
    cpl = JobsDayCPL(event_date=event_date)
    if event_date != "2026-02-11":
        print(f"[VOL DAY] Using today's 0DTE expiry {event_date} (NFP = Wed Feb 11)")
    interval = max(args.loop, 1) if args.loop else 0

    if interval == 0:
        # High hitters: top 5-10 20X/4X BUYs to Telegram (no position tracking)
        if args.high_hitters > 0:
            n = max(1, min(args.high_hitters, 15))
            print(f"[HIGH HITTERS] Sending top {n} 20X/4X BUY alerts to Telegram")
            calls = cpl.run(broadcast=args.broadcast, dry_run=args.dry_run, untruncated_tails=False, high_hitters_batch=n)
            print(f"CPL: {len(calls)} HIGH hitters sent")
            for i, c in enumerate(calls[:15], 1):
                print(f"  {i}. {c.underlying} {c.side} @ {c.strike} | {getattr(c, 'event_tier', '')}")
            return

        # Single run
        if args.test_mode:
            # FIX 2: Test mode = full option descriptors, same path as live; send 5 execution-complete messages (no summary).
            print("[TEST MODE] Running with dry_run=True; sending 5 execution-complete BUY messages to Telegram")
            calls = cpl.run(broadcast=False, dry_run=True)
            print(f"CPL: {len(calls)} calls generated")

            if not args.broadcast:
                for i, c in enumerate(calls[:15], 1):
                    print(f"  {i}. {c.underlying} {c.side} @ {c.strike} | {c.dedupe_key}")
                return

            sent = 0
            target = min(args.max_calls or 5, 5)
            for call in calls:
                if sent >= target:
                    break
                ready, reason = _is_execution_ready(call)
                if not ready:
                    print(f"[TEST MODE] Skip (incomplete): {reason} | {call.underlying} {call.side} @ {call.strike}")
                    continue
                sent += 1
                msg = format_jobs_day_call(call, sent, test_mode=True)
                if send_alert(msg):
                    print(f"[TEST MODE] Sent {sent}/{target}: {call.underlying} {call.side} @ ${call.strike:.2f} exp {call.expiry_date}")
                else:
                    print(f"[TEST MODE] Failed to send message {sent}")
            print(f"[TEST MODE] Sent {sent} execution-complete message(s) to Telegram. No positions opened.")
            return

        calls = cpl.run(broadcast=args.broadcast, dry_run=args.dry_run, untruncated_tails=untruncated_tails)
        print(f"CPL: {len(calls)} calls generated")
        for i, c in enumerate(calls[:15], 1):
            print(f"  {i}. {c.underlying} {c.side} @ {c.strike} | {c.dedupe_key}")
        return

    # Every weekday: market open (9:30 ET) to close (16:00 ET). Power hour: skip sleep (we're already live).
    if _now_et().weekday() < 5 and not power_hour:
        print("[VOL DAY] Running from market open (9:30 ET) to close (16:00 ET) — same tickers, any movement → BUY/SELL")
        _sleep_until_market_open_et()
    elif power_hour:
        print("[POWER HOUR] Live from now until 4:00 PM ET — alerts show TICKER | DTE | ENTRY | EXIT (STOP)")

    # Loop until event day end (shorter interval when flat in untruncated mode to get next BUY fast)
    run_count = 0
    while not _should_stop_loop():
        # Weekday: stop after market close (16:00 ET) so we don't run overnight; restart next morning
        if _now_et().weekday() < 5 and not _is_market_hours_et():
            print(f"[{_now_et()}] Market close — exiting loop. Restart tomorrow 9:25 ET for next day.")
            break
        # Use today's 0DTE every run (so tomorrow we get tomorrow's expiry)
        cpl.event_date = _now_et().strftime("%Y-%m-%d")
        run_count += 1
        calls = cpl.run(broadcast=args.broadcast, dry_run=args.dry_run, untruncated_tails=untruncated_tails)
        print(f"[{_now_et()}] CPL run #{run_count}: {len(calls)} calls")
        # Power hour: 30s; weekday vol day: 60s; when untruncated and flat, 30s (max capital)
        sleep_sec = interval
        if power_hour:
            sleep_sec = 30
        elif _now_et().weekday() < 5 and interval > 60:
            sleep_sec = 60
        if untruncated_tails and sleep_sec > 30:
            import wsb_snake.execution.jobs_day_cpl as _cpl
            if len(_cpl._open_positions) == 0:
                sleep_sec = 30
        time.sleep(sleep_sec)


if __name__ == "__main__":
    main()
