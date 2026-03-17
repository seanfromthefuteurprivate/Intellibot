#!/usr/bin/env python3
"""
Reusable Polygon replay helpers for underlying + 0DTE option backtests.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
BACKTEST_DIR = Path(os.environ.get("WSB_BACKTEST_DIR", "/tmp/wsb_snake_backtests"))
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


def iter_weekdays(start: str, end: str) -> Iterable[str]:
    current = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    while current <= end_date:
        if current.weekday() < 5:
            yield current.isoformat()
        current += timedelta(days=1)


class PolygonClient:
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise RuntimeError("POLYGON_API_KEY is required")
        self.api_key = api_key
        self.session = requests.Session()
        self.stock_cache: Dict[str, List[dict]] = {}
        self.option_cache: Dict[str, List[dict]] = {}

    def _get_json(self, url: str) -> dict:
        last_error = None
        for attempt in range(5):
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200:
                    return response.json()
                last_error = RuntimeError(f"{response.status_code}: {response.text[:400]}")
                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise last_error
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Polygon request failed: {last_error}")

    def get_underlying_bars(self, underlying: str, trade_date: str) -> List[dict]:
        cache_key = f"{underlying}|{trade_date}"
        if cache_key in self.stock_cache:
            return self.stock_cache[cache_key]
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{underlying}/range/1/minute/{trade_date}/{trade_date}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}"
        )
        data = self._get_json(url)
        rows = []
        for row in data.get("results", []):
            ts = datetime.fromtimestamp(row["t"] / 1000, tz=timezone.utc).astimezone(ET)
            rows.append(
                {
                    "t": row["t"],
                    "et": ts,
                    "o": float(row["o"]),
                    "h": float(row["h"]),
                    "l": float(row["l"]),
                    "c": float(row["c"]),
                    "v": float(row["v"]),
                }
            )
        self.stock_cache[cache_key] = rows
        return rows

    def get_option_bars(self, option_symbol: str, trade_date: str) -> List[dict]:
        cache_key = f"{option_symbol}|{trade_date}"
        if cache_key in self.option_cache:
            return self.option_cache[cache_key]
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{option_symbol}/range/1/minute/{trade_date}/{trade_date}"
            f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}"
        )
        data = self._get_json(url)
        rows = []
        for row in data.get("results", []):
            ts = datetime.fromtimestamp(row["t"] / 1000, tz=timezone.utc).astimezone(ET)
            rows.append(
                {
                    "t": row["t"],
                    "et": ts,
                    "o": float(row["o"]),
                    "h": float(row["h"]),
                    "l": float(row["l"]),
                    "c": float(row["c"]),
                    "v": float(row["v"]),
                }
            )
        self.option_cache[cache_key] = rows
        return rows


def build_option_symbol(underlying: str, trade_date: str, strike: int, direction: str) -> str:
    expiry = datetime.strptime(trade_date, "%Y-%m-%d").strftime("%y%m%d")
    cp = "C" if direction == "CALL" else "P"
    return f"O:{underlying}{expiry}{cp}{int(strike * 1000):08d}"


def find_bar_index(rows: List[dict], target_time: datetime) -> Optional[int]:
    for idx, row in enumerate(rows):
        if row["et"] >= target_time:
            return idx
    return None


def pick_option_contract(
    client: PolygonClient,
    underlying: str,
    trade_date: str,
    direction: str,
    spot_price: float,
    target_premium: float,
    entry_bar_time: datetime,
) -> Optional[tuple[str, dict]]:
    rounded = int(round(spot_price))
    if direction == "CALL":
        strikes = list(range(rounded, rounded + 5))
    else:
        strikes = list(range(rounded, rounded - 5, -1))

    best_symbol = None
    best_bar = None
    best_diff = float("inf")

    for strike in strikes:
        symbol = build_option_symbol(underlying, trade_date, strike, direction)
        bars = client.get_option_bars(symbol, trade_date)
        if not bars:
            continue
        bar = next((row for row in bars if row["et"] >= entry_bar_time), None)
        if not bar:
            continue
        price = bar["o"] or bar["c"]
        if price <= 0:
            continue
        diff = abs(price - target_premium)
        if diff < best_diff:
            best_diff = diff
            best_symbol = symbol
            best_bar = bar

    if best_symbol and best_bar:
        return best_symbol, best_bar
    return None
