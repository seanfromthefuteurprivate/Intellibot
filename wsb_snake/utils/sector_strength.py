"""
Sector strength – cross-cutting "sector slighted down" check.

When SPY (or sector ETFs) are down sharply from open/prior close, pause new scalps
and optionally block momentum entries. UNHINGED: lower threshold so we react faster.
"""

from __future__ import annotations

from typing import Dict, Optional

from wsb_snake.utils.logger import get_logger
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced

log = get_logger(__name__)

# UNHINGED: react at -0.7% from open (was -1.5%). "Sector slighted down" = no new scalp/momentum.
SECTOR_SLIGHTED_DOWN_PCT = -0.7  # SPY down this much from open = pause
# Secondary: from prior close (for premarket / early open)
SECTOR_SLIGHTED_DOWN_PCT_FROM_PREV = -1.0

# Market proxy – primary gauge
MARKET_PROXY_TICKER = "SPY"
# Optional sector ETFs to average (if we want broader "sector" view)
SECTOR_ETFS = ["SPY", "QQQ"]


def get_sector_strength_score() -> Optional[float]:
    """
    Return -1 to 1: negative = sector weak, positive = strong.
    None if data unavailable.
    """
    try:
        snap = polygon_enhanced.get_snapshot(MARKET_PROXY_TICKER)
        if not snap:
            return None
        price = snap.get("price", 0) or snap.get("last_trade_price", 0)
        today_open = snap.get("today_open", 0)
        prev_close = snap.get("prev_close", 0)
        if not price or price <= 0:
            return None
        ref = today_open if today_open and today_open > 0 else prev_close
        if not ref or ref <= 0:
            return None
        pct = (price - ref) / ref * 100
        # Normalize to roughly -1..1 (e.g. -3% -> -1, +3% -> 1)
        import math
        score = math.tanh(pct / 3.0)
        return round(score, 4)
    except Exception as e:
        log.debug(f"Sector strength error: {e}")
        return None


def is_sector_slighted_down() -> bool:
    """
    True if market/sector is "slighted down" – no new scalp, consider no momentum entries.
    UNHINGED: uses -0.7% from open.
    """
    try:
        snap = polygon_enhanced.get_snapshot(MARKET_PROXY_TICKER)
        if not snap:
            return False  # Don't block on data failure
        price = snap.get("price", 0) or snap.get("last_trade_price", 0)
        today_open = snap.get("today_open", 0)
        prev_close = snap.get("prev_close", 0)
        if not price or price <= 0:
            return False
        # From open (intraday)
        if today_open and today_open > 0:
            pct_open = (price - today_open) / today_open * 100
            if pct_open <= SECTOR_SLIGHTED_DOWN_PCT:
                log.info(f"Sector slighted down: SPY {pct_open:.2f}% from open (threshold {SECTOR_SLIGHTED_DOWN_PCT}%)")
                return True
        # From prev close (early session / premarket)
        if prev_close and prev_close > 0:
            pct_prev = (price - prev_close) / prev_close * 100
            if pct_prev <= SECTOR_SLIGHTED_DOWN_PCT_FROM_PREV:
                log.info(f"Sector slighted down: SPY {pct_prev:.2f}% from prev close")
                return True
        return False
    except Exception as e:
        log.debug(f"Sector slighted check error: {e}")
        return False


def get_sector_status() -> Dict[str, object]:
    """For dashboard / debugging."""
    score = get_sector_strength_score()
    slighted = is_sector_slighted_down()
    return {
        "score": score,
        "sector_slighted_down": slighted,
        "threshold_pct": SECTOR_SLIGHTED_DOWN_PCT,
    }
