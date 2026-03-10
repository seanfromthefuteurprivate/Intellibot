#!/usr/bin/env python3
"""
FINAL DIABOLICAL BACKTEST
=========================
The definitive backtest comparing 5 exit configurations across 3 entry windows.

ENTRY WINDOWS (Two-Stage Confirmation):
  - MORNING: 9:45 + 10:00 confirmation, 0.50% threshold
  - COBRA:  13:00 + 13:15 confirmation, 0.55% threshold
  - MAMBA:  15:00 + 15:15 confirmation, 0.80% threshold

EXIT CONFIGS:
  1. V7_CURRENT    - 15% stop, tightening trails
  2. WIDER_STOP    - 25% stop, no tightening
  3. V8_CORRECTED  - 25% stop, wider trails, 30s delay
  4. YOLO_SMART    - 40% stop, moonshot trail only
  5. YOLO_EOD      - 50% stop, hold to EOD only

Account: $2,500 starting, $500/trade, -$300 daily max loss
Pyramids: ON (+30% trigger, 50% add, 25% add)
QQQ priority over SPY, BOTH directions

Usage:
  python final_diabolical.py --start 2026-02-20 --end 2026-03-09
"""

import os
import sys
import json
import time
import math
import argparse
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from copy import deepcopy

# ============================================================
# CONFIGURATION
# ============================================================

POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")

# Account settings
INITIAL_ACCOUNT = 2500.0
TRADE_SIZE = 500.0
DAILY_MAX_LOSS = -300.0

# Entry Windows (Two-Stage Confirmation)
ENTRY_WINDOWS = {
    "MORNING": {
        "first_check_minute": 15,   # 9:45 AM (minutes after open)
        "confirm_minute": 30,       # 10:00 AM
        "threshold": 0.0050,        # 0.50%
    },
    "COBRA": {
        "first_check_minute": 210,  # 13:00 PM
        "confirm_minute": 225,      # 13:15 PM
        "threshold": 0.0055,        # 0.55%
    },
    "MAMBA": {
        "first_check_minute": 330,  # 15:00 PM
        "confirm_minute": 345,      # 15:15 PM
        "threshold": 0.0080,        # 0.80%
    },
}

# Pyramid settings
PYRAMID_TRIGGER_PCT = 0.30
PYRAMID_ADD_1_PCT = 0.50
PYRAMID_ADD_2_PCT = 0.25
PYRAMID_MAX_ADDS = 2

# EXIT CONFIGURATIONS TO TEST
CONFIGS = {
    "V7_CURRENT": {
        "stop_loss": 0.15,
        "tighten": True,
        "lock": (0.70, 0.20),      # (trigger, trail)
        "agg": (1.00, 0.15),
        "moon": (2.00, 0.10),
        "time_delay": 0,
    },
    "WIDER_STOP": {
        "stop_loss": 0.25,
        "tighten": False,
        "lock": (0.70, 0.20),
        "agg": (1.00, 0.15),
        "moon": (2.00, 0.10),
        "time_delay": 0,
    },
    "V8_CORRECTED": {
        "stop_loss": 0.25,
        "tighten": False,
        "lock": (0.80, 0.35),
        "agg": (1.50, 0.25),
        "moon": (2.50, 0.15),
        "time_delay": 30,
    },
    "YOLO_SMART": {
        "stop_loss": 0.40,
        "tighten": False,
        "lock": None,
        "agg": None,
        "moon": (2.50, 0.20),
        "time_delay": 0,
    },
    "YOLO_EOD": {
        "stop_loss": 0.50,
        "tighten": False,
        "lock": None,
        "agg": None,
        "moon": None,
        "time_delay": 0,
    },
}

# Option parameters
OTM_OFFSET = 2
MIN_OPTION_PRICE = 0.20
MAX_OPTION_PRICE = 30.00
SLIPPAGE_PCT = 0.02

# Rate limiting
API_DELAY = 12  # seconds between Polygon calls

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Position:
    entry_time: datetime
    entry_minute: int
    entry_price: float
    direction: str
    strike: float
    expiry: str
    option_ticker: str
    contracts: int
    entry_option_price: float
    position_cost: float
    peak_option_price: float = 0.0
    pyramid_adds: int = 0
    original_contracts: int = 0
    window: str = ""

    def __post_init__(self):
        if self.peak_option_price == 0.0:
            self.peak_option_price = self.entry_option_price
        if self.original_contracts == 0:
            self.original_contracts = self.contracts


@dataclass
class Trade:
    date: str
    window: str
    entry_time: str
    exit_time: str
    direction: str
    ticker: str
    strike: float
    contracts: int
    entry_option_price: float
    exit_option_price: float
    peak_option_price: float
    position_cost: float
    pnl_dollars: float
    pnl_pct: float
    exit_reason: str
    hold_minutes: int
    pyramid_adds: int
    underlying_entry: float
    underlying_exit: float
    config: str


@dataclass
class DayResult:
    date: str
    ticker: str
    config: str
    trades: List[Trade]
    daily_pnl: float
    end_account: float
    windows_triggered: Dict[str, bool]


# ============================================================
# API FUNCTIONS
# ============================================================

def polygon_request(url: str, params: dict) -> Optional[dict]:
    """Make Polygon API request with rate limiting."""
    time.sleep(API_DELAY)
    try:
        params["apiKey"] = POLYGON_KEY
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            print(f"    [RATE_LIMIT] Waiting 60s...")
            time.sleep(60)
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            print(f"    [RATE_LIMIT] Still failing: {r.status_code}")
        else:
            print(f"    [API_ERROR] Status {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"    [API_ERROR] {e}")
    return None


def fetch_underlying_bars(ticker: str, date: str) -> List[Dict]:
    """Fetch 1-minute bars for underlying."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}"
    data = polygon_request(url, {"limit": 50000, "sort": "asc"})
    if data and "results" in data:
        bars = []
        for bar in data["results"]:
            ts_ms = bar["t"]
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            # Filter to market hours only (9:30 - 16:00 ET)
            et_hour = (dt.hour - 5) % 24  # UTC to ET approximation
            if 9 <= et_hour < 16 or (et_hour == 9 and dt.minute >= 30):
                bars.append({
                    "t": ts_ms,
                    "dt": dt,
                    "o": bar["o"],
                    "h": bar["h"],
                    "l": bar["l"],
                    "c": bar["c"],
                    "v": bar["v"],
                })
        return bars
    return []


def fetch_option_bars(option_ticker: str, date: str) -> List[Dict]:
    """Fetch minute-level option bars from Polygon."""
    url = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/minute/{date}/{date}"
    data = polygon_request(url, {"limit": 50000, "sort": "asc"})
    if data and "results" in data:
        return data["results"]
    return []


def get_option_price_at_time(option_bars: List[Dict], target_ts: int, use_low: bool = False) -> Optional[float]:
    """Find option price closest to target timestamp."""
    if not option_bars:
        return None

    best_bar, best_diff = None, float('inf')
    for bar in option_bars:
        diff = abs(bar["t"] - target_ts)
        if diff < best_diff:
            best_diff, best_bar = diff, bar
        if bar["t"] > target_ts and best_bar:
            break

    if best_bar and best_diff <= 180000:  # Within 3 minutes
        return best_bar["l"] if use_low else best_bar["c"]
    return None


def build_option_ticker(ticker: str, strike: float, expiry: str, direction: str) -> str:
    """Build OCC option ticker symbol."""
    expiry_fmt = datetime.strptime(expiry, "%Y-%m-%d").strftime("%y%m%d")
    cp = "C" if direction == "CALL" else "P"
    return f"O:{ticker}{expiry_fmt}{cp}{int(strike * 1000):08d}"


# ============================================================
# MARCH 9 MOONSHOT VERIFICATION
# ============================================================

def verify_march9_moonshot():
    """Task 1: Verify the March 9 QQQ moonshot move."""
    print("\n" + "="*70)
    print("  TASK 1: MARCH 9 QQQ MOONSHOT VERIFICATION")
    print("="*70)

    date = "2026-03-09"
    option_ticker = "O:QQQ260309C00605000"

    print(f"\n  Option: {option_ticker}")
    print(f"  Date: {date}")
    print(f"  Pulling real Polygon data...")

    # Fetch option bars
    option_bars = fetch_option_bars(option_ticker, date)

    if not option_bars:
        print("  [ERROR] No option data available for March 9")
        return None

    print(f"  Retrieved {len(option_bars)} option bars")

    # Fetch underlying QQQ bars
    qqq_bars = fetch_underlying_bars("QQQ", date)
    if not qqq_bars:
        print("  [ERROR] No QQQ underlying data")
        return None

    print(f"  Retrieved {len(qqq_bars)} QQQ underlying bars")

    # Find open price (9:30 AM)
    qqq_open = qqq_bars[0]["o"] if qqq_bars else 0

    # Analyze option price action
    min_price = min(bar["l"] for bar in option_bars)
    max_price = max(bar["h"] for bar in option_bars)

    # Find when low and high occurred
    low_bar = min(option_bars, key=lambda x: x["l"])
    high_bar = max(option_bars, key=lambda x: x["h"])

    low_time = datetime.fromtimestamp(low_bar["t"] / 1000, tz=timezone.utc)
    high_time = datetime.fromtimestamp(high_bar["t"] / 1000, tz=timezone.utc)

    print(f"\n  OPTION PRICE ACTION:")
    print(f"  ---------------------")
    print(f"  Low:  ${min_price:.2f} at {low_time.strftime('%H:%M')} UTC")
    print(f"  High: ${max_price:.2f} at {high_time.strftime('%H:%M')} UTC")
    print(f"  Move: ${min_price:.2f} -> ${max_price:.2f} ({((max_price/min_price)-1)*100:+.0f}%)")

    # Check QQQ % from open at 3:00 PM and 3:15 PM ET (20:00 and 20:15 UTC)
    target_times = {
        "15:00": 330,  # minutes after 9:30
        "15:15": 345,
    }

    print(f"\n  QQQ % FROM OPEN:")
    print(f"  -----------------")
    print(f"  Open: ${qqq_open:.2f}")

    qqq_at_times = {}
    for label, minutes in target_times.items():
        if minutes < len(qqq_bars):
            price = qqq_bars[minutes]["c"]
            pct = (price - qqq_open) / qqq_open * 100
            qqq_at_times[label] = {"price": price, "pct": pct}
            print(f"  {label} PM ET: ${price:.2f} ({pct:+.2f}%)")

    # Would MAMBA have caught it?
    print(f"\n  MAMBA SIGNAL CHECK:")
    print(f"  --------------------")
    mamba_threshold = 0.80

    # MAMBA checks at 15:00 + 15:15 confirmation
    if "15:00" in qqq_at_times and "15:15" in qqq_at_times:
        first_pct = abs(qqq_at_times["15:00"]["pct"])
        confirm_pct = abs(qqq_at_times["15:15"]["pct"])

        direction = "CALL" if qqq_at_times["15:00"]["pct"] > 0 else "PUT"

        print(f"  First check (15:00):  {qqq_at_times['15:00']['pct']:+.2f}% (need >{mamba_threshold}%)")
        print(f"  Confirm (15:15):      {qqq_at_times['15:15']['pct']:+.2f}%")
        print(f"  Direction:            {direction}")

        if first_pct >= mamba_threshold and confirm_pct >= mamba_threshold:
            print(f"  RESULT: MAMBA WOULD HAVE TRIGGERED!")

            # Estimate entry price at 15:15
            entry_minute = 345
            if entry_minute < len(option_bars):
                entry_ts = qqq_bars[entry_minute]["t"] if entry_minute < len(qqq_bars) else option_bars[0]["t"]
                entry_price = get_option_price_at_time(option_bars, entry_ts)
                if entry_price:
                    print(f"  Entry price at 15:15: ${entry_price:.2f}")
                    print(f"  Peak price:           ${max_price:.2f}")
                    print(f"  Potential gain:       {((max_price/entry_price)-1)*100:+.0f}%")
        else:
            print(f"  RESULT: MAMBA would NOT have triggered (below {mamba_threshold}% threshold)")

    # Return data for report
    return {
        "date": date,
        "option_ticker": option_ticker,
        "option_low": min_price,
        "option_high": max_price,
        "low_time": low_time.strftime("%H:%M UTC"),
        "high_time": high_time.strftime("%H:%M UTC"),
        "move_pct": ((max_price/min_price)-1)*100,
        "qqq_open": qqq_open,
        "qqq_at_times": qqq_at_times,
        "mamba_triggered": first_pct >= mamba_threshold if "15:00" in qqq_at_times else False,
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================

class DiabolicalBacktest:
    """Multi-config backtest engine with three entry windows."""

    def __init__(self, config_name: str, config: Dict):
        self.config_name = config_name
        self.config = config
        self.option_cache: Dict[str, List[Dict]] = {}

    def check_two_stage_entry(
        self,
        underlying_bars: List[Dict],
        window_name: str,
        window_config: Dict,
    ) -> Optional[Tuple[str, int]]:
        """Check if two-stage entry is triggered for a window."""
        first_minute = window_config["first_check_minute"]
        confirm_minute = window_config["confirm_minute"]
        threshold = window_config["threshold"]

        if len(underlying_bars) < confirm_minute + 1:
            return None

        open_price = underlying_bars[0]["o"]

        # First check
        first_price = underlying_bars[first_minute]["c"]
        first_pct = (first_price - open_price) / open_price

        if abs(first_pct) < threshold:
            return None

        # Confirmation check
        confirm_price = underlying_bars[confirm_minute]["c"]
        confirm_pct = (confirm_price - open_price) / open_price

        if abs(confirm_pct) < threshold:
            return None

        # Both stages passed - determine direction
        direction = "CALL" if confirm_pct > 0 else "PUT"
        return (direction, confirm_minute)

    def check_exit(
        self,
        position: Position,
        current_price: float,
        minutes_held: int,
    ) -> Optional[str]:
        """Check exit conditions based on config."""
        if current_price > position.peak_option_price:
            position.peak_option_price = current_price

        pnl_pct = (current_price - position.entry_option_price) / position.entry_option_price
        drawdown = (position.peak_option_price - current_price) / position.peak_option_price if position.peak_option_price > 0 else 0

        # Time delay check (for V8_CORRECTED)
        time_delay = self.config.get("time_delay", 0)
        if minutes_held < time_delay:
            return None  # Don't exit during delay period

        # Stop loss
        stop_loss = self.config["stop_loss"]
        if pnl_pct <= -stop_loss:
            return "STOP_LOSS"

        # Moonshot trail
        moon = self.config.get("moon")
        if moon and pnl_pct >= moon[0] and drawdown > moon[1]:
            return "TRAIL_MOONSHOT"

        # Aggressive trail
        agg = self.config.get("agg")
        if agg and pnl_pct >= agg[0] and drawdown > agg[1]:
            return "TRAIL_AGGRESSIVE"

        # Lock profit trail
        lock = self.config.get("lock")
        if lock and pnl_pct >= lock[0] and drawdown > lock[1]:
            return "TRAIL_LOCK"

        # Tightening (V7_CURRENT only)
        if self.config.get("tighten") and minutes_held > 30:
            tight_stop = 0.10  # Tighten to 10% after 30 min
            if pnl_pct <= -tight_stop:
                return "STOP_TIGHTEN"

        return None

    def simulate_day(
        self,
        ticker: str,
        date: str,
        start_account: float,
    ) -> DayResult:
        """Simulate one trading day with all three windows."""
        trades: List[Trade] = []
        account = start_account
        daily_pnl = 0.0
        windows_triggered = {"MORNING": False, "COBRA": False, "MAMBA": False}

        # Fetch underlying bars
        underlying_bars = fetch_underlying_bars(ticker, date)
        if not underlying_bars or len(underlying_bars) < 50:
            return DayResult(
                date=date, ticker=ticker, config=self.config_name,
                trades=[], daily_pnl=0, end_account=account,
                windows_triggered=windows_triggered,
            )

        open_price = underlying_bars[0]["o"]
        expiry = date

        # Pre-fetch option data for likely strikes
        mid_price = (max(b["h"] for b in underlying_bars) + min(b["l"] for b in underlying_bars)) / 2
        for offset in range(-6, 7):
            strike = round(mid_price) + offset
            for direction in ["CALL", "PUT"]:
                opt_ticker = build_option_ticker(ticker, strike, expiry, direction)
                if opt_ticker not in self.option_cache:
                    opt_bars = fetch_option_bars(opt_ticker, date)
                    self.option_cache[opt_ticker] = opt_bars

        # Check each window independently
        for window_name, window_config in ENTRY_WINDOWS.items():
            # Check daily loss limit
            if daily_pnl <= DAILY_MAX_LOSS:
                break

            entry_result = self.check_two_stage_entry(underlying_bars, window_name, window_config)
            if not entry_result:
                continue

            direction, entry_minute = entry_result
            windows_triggered[window_name] = True

            # Get entry price and select strike
            entry_bar = underlying_bars[entry_minute]
            entry_underlying = entry_bar["c"]

            if direction == "CALL":
                strike = round(entry_underlying) + OTM_OFFSET
            else:
                strike = round(entry_underlying) - OTM_OFFSET

            option_ticker = build_option_ticker(ticker, strike, expiry, direction)
            option_bars = self.option_cache.get(option_ticker, [])

            if not option_bars:
                continue

            # Get entry option price
            entry_ts = entry_bar["t"]
            entry_option_price = get_option_price_at_time(option_bars, entry_ts)

            if not entry_option_price or entry_option_price < MIN_OPTION_PRICE or entry_option_price > MAX_OPTION_PRICE:
                continue

            entry_option_price *= (1 + SLIPPAGE_PCT)  # Slippage on entry

            # Calculate contracts
            contracts = max(1, int(TRADE_SIZE / (entry_option_price * 100)))
            position_cost = contracts * entry_option_price * 100

            # Create position
            position = Position(
                entry_time=entry_bar["dt"],
                entry_minute=entry_minute,
                entry_price=entry_underlying,
                direction=direction,
                strike=strike,
                expiry=expiry,
                option_ticker=option_ticker,
                contracts=contracts,
                entry_option_price=entry_option_price,
                position_cost=position_cost,
                window=window_name,
            )

            # Simulate position through rest of day
            exit_reason = None
            exit_price = entry_option_price
            exit_minute = entry_minute

            for i in range(entry_minute + 1, len(underlying_bars)):
                bar = underlying_bars[i]
                bar_ts = bar["t"]
                minutes_held = i - entry_minute

                current_option_price = get_option_price_at_time(option_bars, bar_ts)
                if not current_option_price:
                    continue

                exit_reason = self.check_exit(position, current_option_price, minutes_held)

                if exit_reason:
                    exit_price = current_option_price * (1 - SLIPPAGE_PCT)
                    exit_minute = i
                    break

                # Pyramid check
                if position.pyramid_adds < PYRAMID_MAX_ADDS:
                    pnl_pct = (current_option_price - position.entry_option_price) / position.entry_option_price
                    if pnl_pct >= PYRAMID_TRIGGER_PCT:
                        add_pct = PYRAMID_ADD_1_PCT if position.pyramid_adds == 0 else PYRAMID_ADD_2_PCT
                        add_contracts = max(1, int(position.original_contracts * add_pct))
                        add_cost = add_contracts * current_option_price * 100 * (1 + SLIPPAGE_PCT)
                        position.contracts += add_contracts
                        position.position_cost += add_cost
                        position.pyramid_adds += 1

            # EOD close if not already exited
            if not exit_reason:
                exit_reason = "EOD_CLOSE"
                last_bar = underlying_bars[-1]
                exit_price = get_option_price_at_time(option_bars, last_bar["t"], use_low=True)
                if exit_price:
                    exit_price *= (1 - SLIPPAGE_PCT)
                else:
                    exit_price = position.entry_option_price * 0.5  # Worst case fallback
                exit_minute = len(underlying_bars) - 1

            # Calculate P&L
            exit_value = position.contracts * exit_price * 100
            pnl_dollars = exit_value - position.position_cost
            pnl_pct = (exit_price - position.entry_option_price) / position.entry_option_price * 100
            hold_minutes = exit_minute - entry_minute

            # Record trade
            trade = Trade(
                date=date,
                window=window_name,
                entry_time=position.entry_time.strftime("%H:%M"),
                exit_time=underlying_bars[exit_minute]["dt"].strftime("%H:%M") if exit_minute < len(underlying_bars) else "16:00",
                direction=direction,
                ticker=ticker,
                strike=strike,
                contracts=position.contracts,
                entry_option_price=position.entry_option_price,
                exit_option_price=exit_price,
                peak_option_price=position.peak_option_price,
                position_cost=position.position_cost,
                pnl_dollars=pnl_dollars,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
                hold_minutes=hold_minutes,
                pyramid_adds=position.pyramid_adds,
                underlying_entry=entry_underlying,
                underlying_exit=underlying_bars[exit_minute]["c"] if exit_minute < len(underlying_bars) else underlying_bars[-1]["c"],
                config=self.config_name,
            )
            trades.append(trade)
            daily_pnl += pnl_dollars
            account += pnl_dollars

        return DayResult(
            date=date,
            ticker=ticker,
            config=self.config_name,
            trades=trades,
            daily_pnl=daily_pnl,
            end_account=account,
            windows_triggered=windows_triggered,
        )


# ============================================================
# METRICS & REPORTING
# ============================================================

def calculate_metrics(all_trades: List[Trade], initial_capital: float) -> Dict:
    """Calculate comprehensive metrics for a set of trades."""
    if not all_trades:
        return {
            "trades": 0, "wins": 0, "win_pct": 0, "total_pnl": 0,
            "profit_factor": 0, "max_drawdown": 0, "final_capital": initial_capital,
            "avg_winner": 0, "avg_loser": 0, "expectancy": 0,
        }

    wins = [t for t in all_trades if t.pnl_dollars > 0]
    losses = [t for t in all_trades if t.pnl_dollars <= 0]

    total_pnl = sum(t.pnl_dollars for t in all_trades)
    gross_wins = sum(t.pnl_dollars for t in wins)
    gross_losses = abs(sum(t.pnl_dollars for t in losses))

    # Calculate drawdown
    equity_curve = [initial_capital]
    for t in sorted(all_trades, key=lambda x: x.date):
        equity_curve.append(equity_curve[-1] + t.pnl_dollars)

    peak = equity_curve[0]
    max_dd = 0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return {
        "trades": len(all_trades),
        "wins": len(wins),
        "win_pct": len(wins) / len(all_trades) * 100 if all_trades else 0,
        "total_pnl": total_pnl,
        "profit_factor": gross_wins / gross_losses if gross_losses > 0 else 999.99,
        "max_drawdown": max_dd * 100,
        "final_capital": initial_capital + total_pnl,
        "avg_winner": gross_wins / len(wins) if wins else 0,
        "avg_loser": -gross_losses / len(losses) if losses else 0,
        "expectancy": total_pnl / len(all_trades) if all_trades else 0,
    }


def print_comparison_table(all_results: Dict[str, List[DayResult]]):
    """Print comparison table across all configs."""
    print("\n" + "="*90)
    print("  TASK 2B: FULL 5-CONFIG COMPARISON TABLE")
    print("="*90)

    header = f"{'Config':<14} {'Trades':>7} {'Wins':>6} {'Win%':>6} {'Total P&L':>11} {'PF':>6} {'Max DD':>8} {'Final Cap':>11}"
    print(f"\n  {header}")
    print(f"  {'-'*14} {'-'*7} {'-'*6} {'-'*6} {'-'*11} {'-'*6} {'-'*8} {'-'*11}")

    for config_name in CONFIGS.keys():
        results = all_results.get(config_name, [])
        all_trades = []
        for r in results:
            all_trades.extend(r.trades)

        metrics = calculate_metrics(all_trades, INITIAL_ACCOUNT)

        print(f"  {config_name:<14} {metrics['trades']:>7} {metrics['wins']:>6} "
              f"{metrics['win_pct']:>5.1f}% ${metrics['total_pnl']:>+10,.0f} "
              f"{metrics['profit_factor']:>5.2f} {metrics['max_drawdown']:>7.1f}% "
              f"${metrics['final_capital']:>10,.0f}")


def print_window_breakdown(all_results: Dict[str, List[DayResult]]):
    """Print P&L breakdown by window for each config."""
    print("\n" + "="*90)
    print("  TASK 2C: BY WINDOW x CONFIG")
    print("="*90)

    for config_name in CONFIGS.keys():
        results = all_results.get(config_name, [])
        all_trades = []
        for r in results:
            all_trades.extend(r.trades)

        window_pnl = {"MORNING": 0, "COBRA": 0, "MAMBA": 0}
        window_trades = {"MORNING": 0, "COBRA": 0, "MAMBA": 0}

        for t in all_trades:
            window_pnl[t.window] += t.pnl_dollars
            window_trades[t.window] += 1

        print(f"\n  {config_name}:")
        for window in ["MORNING", "COBRA", "MAMBA"]:
            print(f"    {window:<8}: {window_trades[window]:>3} trades, ${window_pnl[window]:>+8,.0f}")


def print_moonshot_capture(all_results: Dict[str, List[DayResult]]):
    """Print moonshot capture rate by config."""
    print("\n" + "="*90)
    print("  TASK 2D: MOONSHOT CAPTURE RATE")
    print("="*90)

    header = f"{'Config':<14} {'Trades +200%':>14} {'Trades +300%':>14} {'Avg Winner Peak':>17}"
    print(f"\n  {header}")
    print(f"  {'-'*14} {'-'*14} {'-'*14} {'-'*17}")

    for config_name in CONFIGS.keys():
        results = all_results.get(config_name, [])
        all_trades = []
        for r in results:
            all_trades.extend(r.trades)

        winners = [t for t in all_trades if t.pnl_dollars > 0]

        trades_200 = sum(1 for t in all_trades if (t.peak_option_price - t.entry_option_price) / t.entry_option_price >= 2.0)
        trades_300 = sum(1 for t in all_trades if (t.peak_option_price - t.entry_option_price) / t.entry_option_price >= 3.0)

        avg_peak = 0
        if winners:
            avg_peak = sum((t.peak_option_price - t.entry_option_price) / t.entry_option_price * 100 for t in winners) / len(winners)

        print(f"  {config_name:<14} {trades_200:>14} {trades_300:>14} {avg_peak:>16.1f}%")


def print_top_trades(all_results: Dict[str, List[DayResult]]):
    """Print top 10 trades across all configs."""
    print("\n" + "="*90)
    print("  TASK 2E: TOP 10 TRADES")
    print("="*90)

    all_trades = []
    for config_name, results in all_results.items():
        for r in results:
            all_trades.extend(r.trades)

    top_trades = sorted(all_trades, key=lambda x: x.pnl_dollars, reverse=True)[:10]

    print(f"\n  {'#':>3} {'Date':<12} {'Config':<14} {'Window':<8} {'Dir':<5} {'Strike':>7} {'P&L':>10} {'Peak%':>8}")
    print(f"  {'-'*3} {'-'*12} {'-'*14} {'-'*8} {'-'*5} {'-'*7} {'-'*10} {'-'*8}")

    for i, t in enumerate(top_trades, 1):
        peak_pct = (t.peak_option_price - t.entry_option_price) / t.entry_option_price * 100
        print(f"  {i:>3} {t.date:<12} {t.config:<14} {t.window:<8} {t.direction:<5} ${t.strike:>6.0f} ${t.pnl_dollars:>+9,.0f} {peak_pct:>+7.0f}%")


def print_day_by_day(results: List[DayResult], config_name: str):
    """Print day-by-day log for winning config."""
    print("\n" + "="*90)
    print(f"  TASK 2F: DAY-BY-DAY LOG ({config_name})")
    print("="*90)

    print(f"\n  {'Date':<12} {'Ticker':<6} {'Windows':<20} {'Trades':>7} {'P&L':>10} {'Account':>12}")
    print(f"  {'-'*12} {'-'*6} {'-'*20} {'-'*7} {'-'*10} {'-'*12}")

    account = INITIAL_ACCOUNT
    for r in results:
        windows_str = ",".join([w[0] for w, triggered in r.windows_triggered.items() if triggered]) or "-"
        account = r.end_account
        print(f"  {r.date:<12} {r.ticker:<6} {windows_str:<20} {len(r.trades):>7} ${r.daily_pnl:>+9,.0f} ${account:>11,.0f}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Final Diabolical Backtest")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--ticker", type=str, default="QQQ", help="Primary ticker (default: QQQ)")
    parser.add_argument("--skip-march9", action="store_true", help="Skip March 9 verification")
    args = parser.parse_args()

    if not POLYGON_KEY:
        print("ERROR: POLYGON_API_KEY not set")
        sys.exit(1)

    print("\n" + "="*70)
    print("  FINAL DIABOLICAL BACKTEST")
    print("="*70)
    print(f"  Start: {args.start}")
    print(f"  End:   {args.end}")
    print(f"  Ticker: {args.ticker}")
    print(f"  Account: ${INITIAL_ACCOUNT:,.0f}")
    print(f"  Trade Size: ${TRADE_SIZE:,.0f}")
    print(f"  Daily Max Loss: ${DAILY_MAX_LOSS:,.0f}")
    print(f"  Configs: {', '.join(CONFIGS.keys())}")
    print("="*70)

    # Task 1: March 9 Moonshot Verification
    march9_data = None
    if not args.skip_march9 and args.end >= "2026-03-09":
        march9_data = verify_march9_moonshot()

    # Generate trading dates
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    dates = []
    current = start_dt
    while current <= end_dt:
        if current.weekday() < 5:  # Weekdays only
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    print(f"\n  Trading dates: {len(dates)}")

    # Task 2: Run backtest for each config
    all_results: Dict[str, List[DayResult]] = {}

    for config_name, config in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"  RUNNING CONFIG: {config_name}")
        print(f"{'='*70}")

        backtest = DiabolicalBacktest(config_name, config)
        results = []
        account = INITIAL_ACCOUNT

        for date in dates:
            print(f"\n  [{config_name}] {date} | Account: ${account:,.0f}")
            result = backtest.simulate_day(args.ticker, date, account)
            results.append(result)
            account = result.end_account

            if result.trades:
                for t in result.trades:
                    print(f"    {t.window}: {t.direction} ${t.strike:.0f} | Entry ${t.entry_option_price:.2f} -> Exit ${t.exit_option_price:.2f} | P&L ${t.pnl_dollars:+,.0f} ({t.exit_reason})")

        all_results[config_name] = results

    # Print reports
    print_comparison_table(all_results)
    print_window_breakdown(all_results)
    print_moonshot_capture(all_results)
    print_top_trades(all_results)

    # Find winning config
    best_config = None
    best_pnl = float('-inf')
    for config_name, results in all_results.items():
        total_pnl = sum(r.daily_pnl for r in results)
        if total_pnl > best_pnl:
            best_pnl = total_pnl
            best_config = config_name

    if best_config:
        print_day_by_day(all_results[best_config], best_config)

    # Task 2G: Final Verdict
    print("\n" + "="*90)
    print("  TASK 2G: FINAL VERDICT")
    print("="*90)

    if best_config:
        config = CONFIGS[best_config]
        metrics = calculate_metrics(
            [t for r in all_results[best_config] for t in r.trades],
            INITIAL_ACCOUNT
        )

        print(f"\n  RECOMMENDED CONFIG: {best_config}")
        print(f"\n  Exact Parameters:")
        print(f"    Stop Loss:      {config['stop_loss']*100:.0f}%")
        print(f"    Tightening:     {'ON' if config.get('tighten') else 'OFF'}")
        if config.get('lock'):
            print(f"    Lock Trail:     {config['lock'][0]*100:.0f}% trigger, {config['lock'][1]*100:.0f}% trail")
        else:
            print(f"    Lock Trail:     DISABLED")
        if config.get('agg'):
            print(f"    Agg Trail:      {config['agg'][0]*100:.0f}% trigger, {config['agg'][1]*100:.0f}% trail")
        else:
            print(f"    Agg Trail:      DISABLED")
        if config.get('moon'):
            print(f"    Moon Trail:     {config['moon'][0]*100:.0f}% trigger, {config['moon'][1]*100:.0f}% trail")
        else:
            print(f"    Moon Trail:     DISABLED")
        print(f"    Time Delay:     {config.get('time_delay', 0)}s")

        print(f"\n  Performance:")
        print(f"    Total P&L:      ${metrics['total_pnl']:+,.0f}")
        print(f"    Win Rate:       {metrics['win_pct']:.1f}%")
        print(f"    Profit Factor:  {metrics['profit_factor']:.2f}")
        print(f"    Max Drawdown:   {metrics['max_drawdown']:.1f}%")
        print(f"    Final Capital:  ${metrics['final_capital']:,.0f}")

        print(f"\n  Code Changes Required:")
        print(f"    File: wsb_snake/backtest/backtest_v7_final.py")
        print(f"    Lines 76-81: Update TRAIL_* constants to match {best_config}")

    print("\n" + "="*90)

    # Save results to JSON
    output_path = os.path.join(os.path.dirname(__file__), "final_diabolical_results.json")

    output_data = {
        "run_time": datetime.now(timezone.utc).isoformat(),
        "params": {
            "start": args.start,
            "end": args.end,
            "ticker": args.ticker,
            "initial_account": INITIAL_ACCOUNT,
            "trade_size": TRADE_SIZE,
            "daily_max_loss": DAILY_MAX_LOSS,
        },
        "march9_verification": march9_data,
        "configs": {
            name: {
                "params": config,
                "metrics": calculate_metrics(
                    [t for r in all_results.get(name, []) for t in r.trades],
                    INITIAL_ACCOUNT
                ),
                "trades": [asdict(t) for r in all_results.get(name, []) for t in r.trades],
            }
            for name, config in CONFIGS.items()
        },
        "recommended_config": best_config,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
