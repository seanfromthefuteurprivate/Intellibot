#!/usr/bin/env python3
"""Match Telegram call alerts vs current pricing and report if they went as expected."""

import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from wsb_snake.db.database import get_recent_cpl_calls

def _get_current_price(ticker):
    try:
        from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
        if polygon_enhanced:
            snap = polygon_enhanced.get_snapshot(ticker)
            if snap and snap.get("price"):
                return float(snap["price"])
        from wsb_snake.collectors.polygon_options import polygon_options
        if polygon_options and hasattr(polygon_options, "get_quote"):
            q = polygon_options.get_quote(ticker)
            if q and q.get("price"):
                return float(q["price"])
    except Exception:
        pass
    return 0.0

def _underlying_at_alert(ticker, timestamp_et):
    try:
        import pytz
        from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
        if not polygon_enhanced or not timestamp_et:
            return 0.0
        et = pytz.timezone("America/New_York")
        parts = timestamp_et.replace("T", " ").split(".")[0].strip()
        try:
            dt = datetime.strptime(parts[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                dt = datetime.strptime(parts[:10], "%Y-%m-%d")
            except ValueError:
                return 0.0
        dt_et = et.localize(dt)
        alert_ms = int(dt_et.timestamp() * 1000)
        bars = polygon_enhanced.get_intraday_bars(ticker, timespan="minute", multiplier=1, limit=120)
        if not bars:
            return 0.0
        for b in bars:
            ts = b.get("timestamp") or b.get("t") or 0
            if ts <= alert_ms and ts + 60000 >= alert_ms:
                return float(b.get("close") or b.get("c") or 0)
        if bars:
            return float(bars[0].get("close") or bars[0].get("c") or 0)
    except Exception:
        pass
    return 0.0

def main():
    limit, days_back = 30, 7
    if len(sys.argv) > 1:
        try: limit = int(sys.argv[1])
        except ValueError: pass
    if len(sys.argv) > 2:
        try: days_back = int(sys.argv[2])
        except ValueError: pass
    calls = get_recent_cpl_calls(limit=limit, days_back=days_back)
    if not calls:
        print("No CPL calls found in DB. Run with --broadcast to log alerts.")
        return
    print("Alert vs current price (last %d calls, last %d days)" % (len(calls), days_back))
    print("=" * 90)
    for row in calls:
        ticker = row.get("ticker") or ""
        side = (row.get("side") or "").upper()
        strike = float(row.get("strike") or 0)
        timestamp_et = row.get("timestamp_et") or row.get("alerted_at") or ""
        alerted_at = row.get("alerted_at") or timestamp_et
        full = row.get("full_json")
        entry_opt = stop = 0.0
        if full:
            try:
                data = json.loads(full)
                et = data.get("entry_trigger") or {}
                sl = data.get("stop_loss") or {}
                entry_opt = float(et.get("price") or 0)
                stop = float(sl.get("price") or 0)
            except Exception: pass
        current = _get_current_price(ticker)
        at_alert = _underlying_at_alert(ticker, timestamp_et)
        if side == "CALL":
            went_ok = current > strike if current and strike else None
            direction = "underlying up"
        else:
            went_ok = current < strike if current and strike else None
            direction = "underlying down"
        if at_alert and current:
            move_pct = (current - at_alert) / at_alert * 100 if at_alert else 0
        else:
            move_pct = None
        status = "AS EXPECTED" if went_ok else ("AGAINST" if went_ok is False else "?")
        print("  %s %s @ $%.2f  |  Alert: %s" % (ticker, side, strike, timestamp_et or alerted_at))
        line2 = "    Entry(opt): $%.2f  Stop: $%.2f  |  Underlying now: $%.2f" % (entry_opt, stop, current)
        if at_alert and move_pct is not None:
            line2 += "  (at alert: $%.2f, move: %+.2f%%)" % (at_alert, move_pct)
        print(line2)
        print("    Direction wanted: %s  |  Status: %s" % (direction, status))
        print()
    print("=" * 90)

if __name__ == "__main__": main()
