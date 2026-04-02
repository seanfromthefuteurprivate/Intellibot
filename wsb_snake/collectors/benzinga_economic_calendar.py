"""
Benzinga Economic Calendar Adapter

Fetches and normalizes high-impact macro events for trade gating.
"""

import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import requests

from wsb_snake.config import BENZINGA_API_KEY
from wsb_snake.utils.logger import log


EASTERN_TZ = ZoneInfo("America/New_York")
BENZINGA_ECONOMICS_URL = "https://api.benzinga.com/api/v2.1/calendar/economics"


class BenzingaEconomicCalendarAdapter:
    """Adapter for Benzinga's economics calendar endpoint."""

    def __init__(self):
        self.api_key = BENZINGA_API_KEY
        self.base_url = BENZINGA_ECONOMICS_URL
        self.session = requests.Session()
        self.cache: Dict[str, Any] = {}
        self.cache_ttl_seconds = 60
        self.last_call = 0.0
        self.min_interval = 0.35

    def _rate_limit(self) -> None:
        """Keep Benzinga calls comfortably below bursty request patterns."""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()

    def _get_cached(self, key: str) -> Optional[Any]:
        cached = self.cache.get(key)
        if not cached:
            return None
        value, timestamp = cached
        if time.time() - timestamp < self.cache_ttl_seconds:
            return value
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        self.cache[key] = (value, time.time())

    def _refresh_relative_fields(self, events: List[Dict[str, Any]], now: datetime) -> List[Dict[str, Any]]:
        refreshed = []
        for event in events:
            event_dt = datetime.fromisoformat(event["scheduled_at"]).astimezone(EASTERN_TZ)
            updated = dict(event)
            updated["minutes_until"] = int(round((event_dt - now).total_seconds() / 60.0))
            refreshed.append(updated)
        return refreshed

    def _request(self, params: Dict[str, str]) -> Optional[str]:
        """Make an authenticated request to the Benzinga economics endpoint."""
        if not self.api_key:
            log.warning("BENZINGA_API_KEY not set - economics calendar unavailable")
            return None

        request_params = dict(params)
        request_params["token"] = self.api_key

        try:
            self._rate_limit()
            response = self.session.get(
                self.base_url,
                params=request_params,
                headers={"accept": "application/xml, application/json"},
                timeout=15,
            )
            if response.status_code != 200:
                log.warning(f"Benzinga economics error {response.status_code}: {response.text[:200]}")
                return None
            return response.text
        except Exception as exc:
            log.warning(f"Benzinga economics request failed: {exc}")
            return None

    def _parse_payload(self, payload: str) -> List[Dict[str, str]]:
        """Parse Benzinga XML or JSON payloads into raw dictionaries."""
        if not payload:
            return []

        payload = payload.strip()
        if not payload:
            return []

        if payload.startswith("<"):
            return self._parse_xml_payload(payload)
        return self._parse_json_payload(payload)

    def _parse_xml_payload(self, payload: str) -> List[Dict[str, str]]:
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            log.warning(f"Failed to parse Benzinga economics XML: {exc}")
            return []

        rows = []
        for item in root.findall(".//item"):
            row = {child.tag: (child.text or "").strip() for child in item}
            if row:
                rows.append(row)
        return rows

    def _parse_json_payload(self, payload: str) -> List[Dict[str, str]]:
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            log.warning(f"Failed to parse Benzinga economics JSON: {exc}")
            return []

        if isinstance(decoded, dict):
            entries = decoded.get("economics") or decoded.get("data") or []
        elif isinstance(decoded, list):
            entries = decoded
        else:
            entries = []

        rows: List[Dict[str, str]] = []
        for entry in entries:
            if isinstance(entry, dict):
                rows.append({str(k): "" if v is None else str(v) for k, v in entry.items()})
        return rows

    def _parse_importance(self, value: str) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    def _parse_event_datetime(self, date_value: str, time_value: str) -> Optional[datetime]:
        if not date_value:
            return None

        normalized_time = (time_value or "00:00:00").strip() or "00:00:00"
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                event_dt = datetime.strptime(f"{date_value} {normalized_time}", fmt)
                return event_dt.replace(tzinfo=EASTERN_TZ)
            except ValueError:
                continue
        return None

    def classify_event_name(self, name: str, category: str = "") -> str:
        """Map verbose Benzinga labels into canonical event codes."""
        label = f"{name} {category}".lower()

        if any(token in label for token in ["fomc", "fed interest rate decision", "fed monetary policy report"]):
            return "FOMC"
        if "cpi" in label:
            return "CPI"
        if "ppi" in label:
            return "PPI"
        if any(token in label for token in ["nonfarm payroll", "employment situation", "jobless claims"]):
            return "JOBS"
        if "gdp" in label:
            return "GDP"
        if "retail sales" in label:
            return "RETAIL_SALES"
        if "pce" in label or "personal consumption expenditures" in label:
            return "PCE"
        if "consumer confidence" in label or "consumer sentiment" in label:
            return "SENTIMENT"
        if "crude oil inventories" in label:
            return "OIL"
        return "OTHER"

    def _normalize_event(self, row: Dict[str, str], now: datetime) -> Optional[Dict[str, Any]]:
        event_dt = self._parse_event_datetime(row.get("date", ""), row.get("time", ""))
        if not event_dt:
            return None

        minutes_until = int(round((event_dt - now).total_seconds() / 60.0))
        event_name = row.get("event_name", "").strip() or "Unknown Event"
        event_category = row.get("event_category", "").strip()
        time_label = event_dt.strftime("%I:%M %p ET").lstrip("0")

        return {
            "id": row.get("id", ""),
            "source": "benzinga",
            "name": event_name,
            "event_name": event_name,
            "category": event_category,
            "event_category": event_category,
            "event_code": self.classify_event_name(event_name, event_category),
            "country": row.get("country", "").strip(),
            "importance": self._parse_importance(row.get("importance", "")),
            "scheduled_at": event_dt.isoformat(),
            "date": row.get("date", "").strip(),
            "time": row.get("time", "").strip(),
            "time_label": time_label,
            "scheduled_at_human": f"{event_dt.strftime('%Y-%m-%d')} {time_label}",
            "minutes_until": minutes_until,
            "consensus": row.get("consensus", "").strip(),
            "consensus_unit": row.get("consensus_t", "").strip(),
            "prior": row.get("prior", "").strip(),
            "prior_unit": row.get("prior_t", "").strip(),
            "actual": row.get("actual", "").strip(),
            "actual_unit": row.get("actual_t", "").strip(),
            "period": row.get("event_period", "").strip(),
            "period_year": row.get("period_year", "").strip(),
            "description": row.get("description", "").strip(),
            "updated": row.get("updated", "").strip(),
            "notes": row.get("notes", "").strip(),
            "is_high_impact": self._parse_importance(row.get("importance", "")) >= 3,
        }

    def get_events(
        self,
        date_from: str,
        date_to: str,
        country: str = "USA",
        importance_min: int = 3,
        page_size: int = 100,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch normalized economic events from Benzinga."""
        now_et = now.astimezone(EASTERN_TZ) if now else datetime.now(EASTERN_TZ)
        cache_key = f"{date_from}:{date_to}:{country}:{importance_min}:{page_size}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return self._refresh_relative_fields(cached, now_et)

        params = {
            "parameters[date_from]": date_from,
            "parameters[date_to]": date_to,
            "parameters[country]": country,
            "pagesize": str(page_size),
        }

        payload = self._request(params)
        if not payload:
            return []

        rows = self._parse_payload(payload)
        events: List[Dict[str, Any]] = []
        for row in rows:
            normalized = self._normalize_event(row, now_et)
            if not normalized:
                continue
            if normalized["country"].upper() != country.upper():
                continue
            if normalized["importance"] < importance_min:
                continue
            events.append(normalized)

        events.sort(key=lambda event: event["scheduled_at"])
        self._set_cache(cache_key, events)
        return self._refresh_relative_fields(events, now_et)

    def get_events_window(
        self,
        lookback_minutes: int = 15,
        lookahead_minutes: int = 24 * 60,
        importance_min: int = 3,
        country: str = "USA",
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch a filtered event window around the current time."""
        now_et = now.astimezone(EASTERN_TZ) if now else datetime.now(EASTERN_TZ)
        start = now_et - timedelta(minutes=lookback_minutes)
        end = now_et + timedelta(minutes=lookahead_minutes)

        events = self.get_events(
            date_from=start.strftime("%Y-%m-%d"),
            date_to=end.strftime("%Y-%m-%d"),
            country=country,
            importance_min=importance_min,
            now=now_et,
        )

        windowed = []
        for event in events:
            event_dt = datetime.fromisoformat(event["scheduled_at"]).astimezone(EASTERN_TZ)
            if start <= event_dt <= end:
                windowed.append(event)
        return windowed

    def get_upcoming_events(
        self,
        hours_ahead: int = 24,
        importance_min: int = 3,
        country: str = "USA",
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get upcoming high-impact events within the lookahead window."""
        return [
            event
            for event in self.get_events_window(
                lookback_minutes=0,
                lookahead_minutes=hours_ahead * 60,
                importance_min=importance_min,
                country=country,
                now=now,
            )
            if event["minutes_until"] >= 0
        ]

    def get_event_gate(
        self,
        block_before_minutes: int = 30,
        block_after_minutes: int = 15,
        importance_min: int = 3,
        country: str = "USA",
        now: Optional[datetime] = None,
        events: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Return whether entries should be blocked around major economic events."""
        now_et = now.astimezone(EASTERN_TZ) if now else datetime.now(EASTERN_TZ)
        event_rows = events if events is not None else self.get_events_window(
            lookback_minutes=block_after_minutes,
            lookahead_minutes=max(24 * 60, block_before_minutes),
            importance_min=importance_min,
            country=country,
            now=now_et,
        )

        blocked_events = []
        upcoming_events = []
        events_today = []

        for event in event_rows:
            event_dt = datetime.fromisoformat(event["scheduled_at"]).astimezone(EASTERN_TZ)
            minutes_until = int(round((event_dt - now_et).total_seconds() / 60.0))
            enriched = dict(event)
            enriched["minutes_until"] = minutes_until

            if event_dt.date() == now_et.date():
                events_today.append(enriched)

            if -block_after_minutes <= minutes_until <= block_before_minutes:
                blocked_events.append(enriched)
            elif minutes_until > block_before_minutes:
                upcoming_events.append(enriched)

        blocked_events.sort(key=lambda event: event["minutes_until"])
        upcoming_events.sort(key=lambda event: event["minutes_until"])

        reason = None
        if blocked_events:
            active = blocked_events[0]
            if active["minutes_until"] >= 0:
                reason = (
                    f"ECON_EVENT_BLOCK: {active['name']} at {active['time_label']} "
                    f"in {active['minutes_until']}m"
                )
            else:
                reason = (
                    f"ECON_EVENT_COOLDOWN: {active['name']} released "
                    f"{abs(active['minutes_until'])}m ago"
                )

        next_event = upcoming_events[0] if upcoming_events else None
        return {
            "blocked": bool(blocked_events),
            "reason": reason,
            "active_events": blocked_events,
            "next_event": next_event,
            "minutes_to_next": next_event["minutes_until"] if next_event else None,
            "has_event_today": bool(events_today),
            "events_today": events_today,
        }


_benzinga_economic_calendar: Optional[BenzingaEconomicCalendarAdapter] = None


def get_benzinga_economic_calendar() -> BenzingaEconomicCalendarAdapter:
    """Get the singleton Benzinga economic calendar adapter."""
    global _benzinga_economic_calendar
    if _benzinga_economic_calendar is None:
        _benzinga_economic_calendar = BenzingaEconomicCalendarAdapter()
    return _benzinga_economic_calendar


benzinga_economic_calendar = get_benzinga_economic_calendar()
