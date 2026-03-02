#!/usr/bin/env python3
"""
AGENT 2: SYSTEM MONITOR — ops/monitor_agent.py
"I WATCH EVERYTHING. NOTHING CRASHES WITHOUT ME KNOWING."

Monitors: service health, V7 heartbeat, memory, error rate, positions, P&L, orphans.
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any

import requests

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wsb_snake.notifications.telegram_bot import send_alert
from wsb_snake.utils.logger import get_logger

logger = get_logger("ops.monitor")

# Config
STATE_FILE = Path(__file__).parent / "state.json"
LOG_DIR = Path(__file__).parent / "logs"
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY")

# Thresholds
MAX_CRASHES_BEFORE_UNSTABLE = 3
V7_STALE_WARN_SECONDS = 90
V7_DEAD_SECONDS = 300
MEMORY_WARN_MB = 500
MEMORY_CRITICAL_MB = 1000
ERROR_STORM_THRESHOLD = 20
RESTART_COOLDOWN_SECONDS = 120

# Market hours (ET)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 16


def load_state() -> Dict:
    """Load persisted state from JSON file."""
    try:
        if STATE_FILE.exists():
            with open(STATE_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load state: {e}")
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "crash_count": 0,
        "last_crash": None,
        "restart_cooldown_until": None,
        "position_snapshot": {},
        "daily_pnl_alerts_sent": [],
        "orphans_alerted": [],
        "memory_history": [],
        "last_v7_scan": None,
        "v7_prices": [],
    }


def save_state(state: Dict):
    """Persist state to JSON file."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def reset_daily_state(state: Dict) -> Dict:
    """Reset state if new day."""
    today = datetime.now().strftime("%Y-%m-%d")
    if state.get("date") != today:
        logger.info(f"New day detected: {today}. Resetting daily state.")
        state["date"] = today
        state["crash_count"] = 0
        state["daily_pnl_alerts_sent"] = []
        state["orphans_alerted"] = []
        state["memory_history"] = []
        state["v7_prices"] = []
    return state


def is_market_hours() -> bool:
    """Check if within market hours (9:30 AM - 4:00 PM ET)."""
    try:
        import zoneinfo
        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
        market_open = et.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MIN, second=0)
        market_close = et.replace(hour=MARKET_CLOSE_HOUR, minute=0, second=0)
        return market_open <= et <= market_close
    except Exception:
        # Fallback: assume market hours if zoneinfo fails
        return True


def get_service_status() -> str:
    """Check systemctl service status."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "wsb-snake"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Failed to check service status: {e}")
        return "unknown"


def restart_service() -> bool:
    """Restart the wsb-snake service."""
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "restart", "wsb-snake"],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to restart service: {e}")
        return False


def get_recent_logs(since: str = "60s ago", lines: int = 100) -> str:
    """Get recent journal logs for wsb-snake."""
    try:
        result = subprocess.run(
            ["journalctl", "-u", "wsb-snake", "--no-pager", "-n", str(lines), "--since", since],
            capture_output=True, text=True, timeout=15
        )
        return result.stdout
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        return ""


def get_process_memory_mb() -> Optional[float]:
    """Get wsb-snake process memory usage in MB."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "wsb_snake.main"],
            capture_output=True, text=True, timeout=10
        )
        pids = result.stdout.strip().split("\n")
        if not pids or not pids[0]:
            return None

        total_rss = 0
        for pid in pids:
            if pid:
                status_path = f"/proc/{pid}/status"
                if os.path.exists(status_path):
                    with open(status_path, "r") as f:
                        for line in f:
                            if line.startswith("VmRSS:"):
                                kb = int(line.split()[1])
                                total_rss += kb / 1024
                                break
        return total_rss if total_rss > 0 else None
    except Exception as e:
        logger.error(f"Failed to get memory: {e}")
        return None


def alpaca_request(endpoint: str) -> Optional[Dict]:
    """Make authenticated request to Alpaca API."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None
    try:
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        url = f"{ALPACA_BASE_URL}{endpoint}"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Alpaca API error: {e}")
        return None


def check_service_alive(state: Dict) -> Dict:
    """TIER 1 CHECK 1: Service alive."""
    status = get_service_status()

    if status != "active":
        state["crash_count"] = state.get("crash_count", 0) + 1
        state["last_crash"] = datetime.now().isoformat()

        # Check cooldown
        cooldown_until = state.get("restart_cooldown_until")
        if cooldown_until:
            cooldown_time = datetime.fromisoformat(cooldown_until)
            if datetime.now() < cooldown_time:
                logger.warning(f"Service down but in cooldown until {cooldown_until}")
                return state

        crash_count = state["crash_count"]
        send_alert(f"🔴 SERVICE DOWN #{crash_count} today. Restarting...")
        logger.warning(f"Service down (crash #{crash_count}). Attempting restart.")

        if restart_service():
            send_alert("✅ Service restarted successfully.")
            # Set cooldown
            state["restart_cooldown_until"] = (
                datetime.now() + timedelta(seconds=RESTART_COOLDOWN_SECONDS)
            ).isoformat()
        else:
            send_alert("❌ Service restart FAILED. Manual intervention required.")

        if crash_count >= MAX_CRASHES_BEFORE_UNSTABLE:
            send_alert(f"🛑 UNSTABLE: {crash_count} crashes today. System needs attention.")

    return state


def check_v7_heartbeat(state: Dict) -> Dict:
    """TIER 1 CHECK 2: V7 heartbeat (market hours only)."""
    if not is_market_hours():
        return state

    logs = get_recent_logs(since="5 min ago", lines=200)

    # Find most recent V7_SCAN
    v7_scans = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*V7_SCAN:", logs)

    if v7_scans:
        last_scan_str = v7_scans[-1]
        try:
            last_scan = datetime.strptime(last_scan_str, "%Y-%m-%d %H:%M:%S")
            state["last_v7_scan"] = last_scan.isoformat()
            seconds_ago = (datetime.now() - last_scan).total_seconds()

            if seconds_ago > V7_DEAD_SECONDS:
                send_alert(f"🔴 V7 DEAD: No scan in {int(seconds_ago)}s. Triggering restart.")
                restart_service()
            elif seconds_ago > V7_STALE_WARN_SECONDS:
                send_alert(f"⚠️ V7 STALE: Last scan {int(seconds_ago)}s ago")
        except Exception as e:
            logger.error(f"Failed to parse V7 scan time: {e}")
    else:
        # No V7 scans found in last 5 minutes
        if state.get("last_v7_scan"):
            send_alert("🔴 V7 DEAD: No scan in 5min. Triggering restart.")
            restart_service()

    # Check for frozen data (same price 10+ times)
    prices = re.findall(r"V7_SCAN: SPY=\$(\d+\.\d+)", logs)
    if len(prices) >= 10:
        recent_prices = prices[-10:]
        if len(set(recent_prices)) == 1:
            send_alert(f"⚠️ V7 FROZEN DATA: Same SPY price ${recent_prices[0]} for 10+ scans")

    return state


def check_memory(state: Dict) -> Dict:
    """TIER 1 CHECK 3: Process memory."""
    memory_mb = get_process_memory_mb()
    if memory_mb is None:
        return state

    # Track memory history
    history = state.get("memory_history", [])
    history.append({"time": datetime.now().isoformat(), "mb": memory_mb})
    # Keep last hour
    cutoff = (datetime.now() - timedelta(hours=1)).isoformat()
    history = [h for h in history if h["time"] > cutoff]
    state["memory_history"] = history

    # Alert deduplication: only alert once per hour for each type
    now = datetime.now()
    last_mem_alert = state.get("last_memory_alert")
    last_leak_alert = state.get("last_leak_alert")

    def can_alert(last_alert_key: str) -> bool:
        last = state.get(last_alert_key)
        if not last:
            return True
        try:
            last_time = datetime.fromisoformat(last)
            return (now - last_time).total_seconds() > 3600  # 1 hour
        except:
            return True

    if memory_mb > MEMORY_CRITICAL_MB:
        if can_alert("last_memory_critical_alert"):
            send_alert(f"🔴 MEMORY CRITICAL: {memory_mb:.0f}MB. Restarting to prevent OOM.")
            state["last_memory_critical_alert"] = now.isoformat()
        restart_service()
    elif memory_mb > MEMORY_WARN_MB:
        if can_alert("last_memory_warn_alert"):
            send_alert(f"⚠️ MEMORY: wsb-snake using {memory_mb:.0f}MB")
            state["last_memory_warn_alert"] = now.isoformat()

    # Check for leak (growing 50MB+ per hour)
    if len(history) >= 10:
        first_mb = history[0]["mb"]
        last_mb = history[-1]["mb"]
        growth = last_mb - first_mb
        if growth > 50:
            if can_alert("last_leak_alert"):
                send_alert(f"⚠️ MEMORY LEAK: Growing {growth:.0f}MB/hr")
                state["last_leak_alert"] = now.isoformat()

    return state


def check_error_rate(state: Dict) -> Dict:
    """TIER 1 CHECK 4: Error rate."""
    logs = get_recent_logs(since="60s ago", lines=500)

    errors = len(re.findall(r"ERROR|Exception|Traceback", logs))
    if errors > ERROR_STORM_THRESHOLD:
        send_alert(f"🔴 ERROR STORM: {errors} errors in 60s. Something is very wrong.")

    # Check for 404 retry loop
    failed_closes = len(re.findall(r"Failed to close.*404", logs))
    if failed_closes > 5:
        send_alert(f"🔴 404 RETRY LOOP DETECTED: {failed_closes} failed closes in 60s")

    return state


def check_positions(state: Dict) -> Dict:
    """TIER 2: Position tracking."""
    if not is_market_hours():
        return state

    positions = alpaca_request("/v2/positions")
    if positions is None:
        return state

    current = {p["symbol"]: p for p in positions}
    previous = state.get("position_snapshot", {})

    # Detect new positions
    for symbol, pos in current.items():
        if symbol not in previous:
            send_alert(
                f"📈 NEW POSITION: {symbol} {pos['qty']}x @ ${float(pos['avg_entry_price']):.2f}"
            )

    # Detect closed positions
    for symbol, prev_pos in previous.items():
        if symbol not in current:
            entry = float(prev_pos["avg_entry_price"])
            # Estimate exit from current price (not perfect but close)
            exit_price = float(prev_pos.get("current_price", entry))
            qty = int(prev_pos["qty"])
            pnl = (exit_price - entry) * qty * 100  # Options multiplier
            pnl_pct = ((exit_price / entry) - 1) * 100 if entry > 0 else 0
            send_alert(
                f"📉 CLOSED: {symbol} {qty}x | Entry: ${entry:.2f} Exit: ${exit_price:.2f} | "
                f"P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)"
            )

    state["position_snapshot"] = current
    return state


def check_daily_pnl(state: Dict) -> Dict:
    """TIER 2: Daily P&L tracking."""
    if not is_market_hours():
        return state

    account = alpaca_request("/v2/account")
    if account is None:
        return state

    portfolio_value = float(account.get("portfolio_value", 0))
    last_equity = float(account.get("last_equity", portfolio_value))
    daily_pnl = portfolio_value - last_equity

    alerts_sent = state.get("daily_pnl_alerts_sent", [])

    # Alert on threshold crossings
    thresholds = [
        (-1000, "🛑 DAILY P&L: -${:.0f} — CONSIDER HALTING"),
        (-500, "🔴 DAILY P&L: -${:.0f} — REVIEW POSITIONS"),
        (-300, "⚠️ DAILY P&L: -${:.0f}"),
        (500, "💰 DAILY P&L: +${:.0f}"),
        (1000, "🔥 DAILY P&L: +${:.0f}"),
    ]

    for threshold, msg_template in thresholds:
        threshold_key = f"pnl_{threshold}"
        if threshold < 0 and daily_pnl <= threshold and threshold_key not in alerts_sent:
            send_alert(msg_template.format(abs(daily_pnl)))
            alerts_sent.append(threshold_key)
        elif threshold > 0 and daily_pnl >= threshold and threshold_key not in alerts_sent:
            send_alert(msg_template.format(daily_pnl))
            alerts_sent.append(threshold_key)

    state["daily_pnl_alerts_sent"] = alerts_sent
    state["current_pnl"] = daily_pnl
    state["portfolio_value"] = portfolio_value
    return state


def check_api_health(state: Dict) -> Dict:
    """TIER 3: API health checks."""
    # Alpaca
    start = time.time()
    account = alpaca_request("/v2/account")
    latency_ms = (time.time() - start) * 1000

    if account is None:
        send_alert("🔴 ALPACA API DOWN")
    elif account.get("status") != "ACTIVE":
        send_alert(f"🔴 ALPACA ACCOUNT STATUS: {account.get('status')}")
    elif latency_ms > 5000:
        send_alert(f"⚠️ ALPACA SLOW: {latency_ms:.0f}ms")

    # Polygon (simple bars request)
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/2026-03-01/2026-03-02"
        params = {"apiKey": POLYGON_API_KEY, "limit": 1}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            send_alert(f"🔴 POLYGON API ERROR: {response.status_code}")
    except Exception as e:
        send_alert(f"🔴 POLYGON API DOWN: {e}")

    return state


def check_disk_space(state: Dict) -> Dict:
    """TIER 3: Disk space check."""
    try:
        result = subprocess.run(
            ["df", "-h", "/home/ubuntu"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 5:
                pct = int(parts[4].replace("%", ""))
                if pct > 95:
                    send_alert(f"🔴 DISK CRITICAL: {pct}%. Logs may stop writing.")
                elif pct > 80:
                    send_alert(f"⚠️ DISK: {pct}% used on /home/ubuntu")
    except Exception as e:
        logger.error(f"Disk check failed: {e}")

    return state


def run_tier1_checks(state: Dict) -> Dict:
    """Run TIER 1 checks (every 10 seconds)."""
    state = check_service_alive(state)
    state = check_v7_heartbeat(state)
    state = check_memory(state)
    state = check_error_rate(state)
    return state


def run_tier2_checks(state: Dict) -> Dict:
    """Run TIER 2 checks (every 30 seconds)."""
    state = check_positions(state)
    state = check_daily_pnl(state)
    return state


def run_tier3_checks(state: Dict) -> Dict:
    """Run TIER 3 checks (every 5 minutes)."""
    state = check_api_health(state)
    state = check_disk_space(state)
    return state


def main():
    """Main monitoring loop."""
    logger.info("🟢 MONITOR AGENT STARTING")
    send_alert("🟢 MONITOR AGENT ONLINE — watching service, V7, memory, positions, P&L")

    state = load_state()
    state = reset_daily_state(state)

    tier1_interval = 10  # seconds
    tier2_interval = 30
    tier3_interval = 300
    save_interval = 30

    last_tier1 = 0
    last_tier2 = 0
    last_tier3 = 0
    last_save = 0

    try:
        while True:
            now = time.time()
            state = reset_daily_state(state)

            # TIER 1: Every 10 seconds
            if now - last_tier1 >= tier1_interval:
                state = run_tier1_checks(state)
                last_tier1 = now

            # TIER 2: Every 30 seconds
            if now - last_tier2 >= tier2_interval:
                state = run_tier2_checks(state)
                last_tier2 = now

            # TIER 3: Every 5 minutes
            if now - last_tier3 >= tier3_interval:
                state = run_tier3_checks(state)
                last_tier3 = now

            # Save state every 30 seconds
            if now - last_save >= save_interval:
                save_state(state)
                last_save = now

            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Monitor agent stopped by user")
    except Exception as e:
        logger.error(f"Monitor agent crashed: {e}")
        send_alert(f"🔴 MONITOR AGENT CRASHED: {e}")
        raise


if __name__ == "__main__":
    main()
