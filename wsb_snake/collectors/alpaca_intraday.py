from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

from wsb_snake.config import ALPACA_API_KEY, ALPACA_DATA_URL, ALPACA_SECRET_KEY
from wsb_snake.utils.logger import get_logger

log = get_logger(__name__)

ET = ZoneInfo("America/New_York")


@dataclass
class _CacheEntry:
    rows: List[dict]
    cached_at: datetime


def _is_live_trade_date(trade_date: str) -> bool:
    return trade_date == datetime.now(ET).date().isoformat()


def _cache_is_fresh(entry: _CacheEntry, trade_date: str, ttl_seconds: int = 15) -> bool:
    if not _is_live_trade_date(trade_date):
        return True
    return (datetime.now(timezone.utc) - entry.cached_at).total_seconds() < ttl_seconds


class AlpacaIntradayClient:
    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        data_url: str = ALPACA_DATA_URL,
    ) -> None:
        self.api_key = api_key or ALPACA_API_KEY or os.getenv("ALPACA_API_KEY", "")
        self.api_secret = api_secret or ALPACA_SECRET_KEY or os.getenv("ALPACA_SECRET_KEY", "")
        self.data_url = data_url.rstrip("/")
        self.stock_cache: Dict[str, _CacheEntry] = {}

        if not self.api_key or not self.api_secret:
            raise RuntimeError("Alpaca API credentials are required for live intraday data")

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def get_underlying_bars(
        self,
        underlying: str,
        trade_date: str,
        current: Optional[datetime] = None,
    ) -> List[dict]:
        cache_key = f"{underlying}|{trade_date}"
        cached = self.stock_cache.get(cache_key)
        if cached and _cache_is_fresh(cached, trade_date):
            return cached.rows

        session_open = datetime.strptime(f"{trade_date} 09:30", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        session_close = datetime.strptime(f"{trade_date} 16:00", "%Y-%m-%d %H:%M").replace(tzinfo=ET)

        end_time = session_close
        if _is_live_trade_date(trade_date):
            live_now = current.astimezone(ET) if current else datetime.now(ET)
            end_time = min(max(live_now, session_open), session_close)

        url = f"{self.data_url}/v2/stocks/{underlying}/bars"
        params = {
            "timeframe": "1Min",
            "start": session_open.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "end": end_time.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "limit": 1000,
            "adjustment": "raw",
            "sort": "asc",
            "feed": "iex",
        }
        resp = requests.get(url, headers=self.headers, params=params, timeout=10)
        resp.raise_for_status()

        rows: List[dict] = []
        for bar in resp.json().get("bars", []):
            raw_ts = bar.get("t")
            if not raw_ts:
                continue
            bar_time = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).astimezone(ET)
            rows.append(
                {
                    "o": float(bar.get("o", 0) or 0),
                    "h": float(bar.get("h", 0) or 0),
                    "l": float(bar.get("l", 0) or 0),
                    "c": float(bar.get("c", 0) or 0),
                    "v": float(bar.get("v", 0) or 0),
                    "t": raw_ts,
                    "et": bar_time,
                }
            )

        self.stock_cache[cache_key] = _CacheEntry(rows=rows, cached_at=datetime.now(timezone.utc))
        return rows

    def get_option_bars(
        self,
        option_symbol: str,
        trade_date: str,
    ) -> List[dict]:
        """
        Fetch 1-minute option bars from Alpaca Data API v1beta1.

        Accepts symbols in either Polygon format (O:QQQ260317C00615000)
        or bare OCC format (QQQ260317C00615000).
        """
        cache_key = f"opt|{option_symbol}|{trade_date}"
        cached = self.stock_cache.get(cache_key)
        if cached and _cache_is_fresh(cached, trade_date):
            return cached.rows

        # Strip "O:" prefix (Polygon format) for Alpaca compatibility
        alpaca_sym = option_symbol[2:] if option_symbol.startswith("O:") else option_symbol

        session_open = datetime.strptime(f"{trade_date} 09:30", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
        session_close = datetime.strptime(f"{trade_date} 16:01", "%Y-%m-%d %H:%M").replace(tzinfo=ET)

        url = f"{self.data_url}/v1beta1/options/bars"
        params = {
            "symbols": alpaca_sym,
            "timeframe": "1Min",
            "start": session_open.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "end": session_close.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "limit": 10000,
            "sort": "asc",
        }

        all_raw: List[dict] = []
        page_token = None
        while True:
            if page_token:
                params["page_token"] = page_token
            try:
                resp = requests.get(url, headers=self.headers, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                log.warning("Alpaca option bars failed for %s: %s", alpaca_sym, exc)
                break

            bars_by_symbol = data.get("bars", {})
            if isinstance(bars_by_symbol, dict):
                all_raw.extend(bars_by_symbol.get(alpaca_sym, []))
            elif isinstance(bars_by_symbol, list):
                all_raw.extend(bars_by_symbol)

            page_token = data.get("next_page_token")
            if not page_token:
                break

        rows: List[dict] = []
        for bar in all_raw:
            raw_ts = bar.get("t")
            if not raw_ts:
                continue
            bar_time = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).astimezone(ET)
            rows.append(
                {
                    "o": float(bar.get("o", 0) or 0),
                    "h": float(bar.get("h", 0) or 0),
                    "l": float(bar.get("l", 0) or 0),
                    "c": float(bar.get("c", 0) or 0),
                    "v": float(bar.get("v", 0) or 0),
                    "t": raw_ts,
                    "et": bar_time,
                }
            )

        self.stock_cache[cache_key] = _CacheEntry(rows=rows, cached_at=datetime.now(timezone.utc))
        return rows

    def get_latest_trade_price(self, underlying: str) -> Optional[float]:
        url = f"{self.data_url}/v2/stocks/{underlying}/snapshot"
        try:
            resp = requests.get(url, headers=self.headers, params={"feed": "iex"}, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("Alpaca snapshot failed for %s: %s", underlying, exc)
            return None

        latest_trade = payload.get("latestTrade") or payload.get("latest_trade") or {}
        if latest_trade.get("p"):
            return float(latest_trade["p"])

        minute_bar = payload.get("minuteBar") or payload.get("minute_bar") or {}
        if minute_bar.get("c"):
            return float(minute_bar["c"])

        daily_bar = payload.get("dailyBar") or payload.get("daily_bar") or {}
        if daily_bar.get("c"):
            return float(daily_bar["c"])

        return None
