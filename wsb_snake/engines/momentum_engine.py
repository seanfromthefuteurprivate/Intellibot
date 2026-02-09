"""
Momentum Engine – small-cap / thematic. UNHINGED: full scan + execution + trim.

Replicates USAR/LUNR/ONDS style: thematic momentum, trim on sector weakness, let runners run.
Uses weekly or next-Friday options. Tag: TradingEngine.MOMENTUM.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from wsb_snake.utils.logger import get_logger
from wsb_snake.config import MOMENTUM_UNIVERSE
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.collectors.finnhub_collector import finnhub_collector
from wsb_snake.utils.sector_strength import is_sector_slighted_down
from wsb_snake.trading.alpaca_executor import alpaca_executor
from wsb_snake.trading.risk_governor import TradingEngine, get_risk_governor
from wsb_snake.notifications.telegram_bot import send_alert as send_telegram_alert

log = get_logger(__name__)

# UNHINGED: lower bar, scan every 2 min
MOMENTUM_SCAN_INTERVAL_SEC = 120
MIN_CONFIDENCE_MOMENTUM = 65
VOLUME_SURGE_MULT = 1.4  # volume > 1.4x 20-day avg
PRICE_UP_5D_PCT = 3.0    # up 3% in 5 days
PRICE_UP_1D_PCT = 1.5    # or up 1.5% today
# Catalyst filter: require earnings in 5-14d or WSB mentions (configurable)
REQUIRE_CATALYST = True
MIN_WSB_MENTIONS = 2


@dataclass
class MomentumCandidate:
    """A momentum-style setup (thematic + catalyst + technical)."""
    ticker: str
    direction: str
    confidence: float
    catalyst: str
    expiry_preference: str
    detected_at: datetime
    entry_price: float = 0.0
    notes: str = ""


def _avg_volume(bars: List[Dict], n: int = 20) -> float:
    if not bars or len(bars) < n:
        return 0.0
    vols = [b.get("v", b.get("volume", 0)) for b in bars[:n]]
    return sum(vols) / len(vols) if vols else 0.0


def _price_change_pct(bars: List[Dict], days: int) -> Optional[float]:
    if not bars or len(bars) <= days:
        return None
    c_now = bars[0].get("c", bars[0].get("close", 0))
    c_ago = bars[days].get("c", bars[days].get("close", 0))
    if not c_ago or c_ago <= 0:
        return None
    return (c_now / c_ago - 1) * 100


def _has_catalyst(ticker: str, earnings_14: Dict, wsb_mentions: Dict[str, Dict]) -> bool:
    """True if ticker has earnings in 5-14 days or WSB mention count >= MIN_WSB_MENTIONS."""
    if earnings_14.get("has_earnings") and earnings_14.get("days_away", 99) >= 5:
        return True
    mentions = wsb_mentions.get(ticker, {}).get("mentions", 0)
    return mentions >= MIN_WSB_MENTIONS


def scan_momentum_universe() -> List[MomentumCandidate]:
    """Scan MOMENTUM_UNIVERSE for volume surge + price momentum. Sector filter applied."""
    if is_sector_slighted_down():
        log.debug("Momentum scan skipped: sector slighted down")
        return []
    wsb_mentions: Dict[str, Dict] = {}
    if REQUIRE_CATALYST:
        try:
            from wsb_snake.collectors.reddit_collector import reddit_collector
            wsb_mentions = reddit_collector.get_ticker_mentions()
        except Exception as e:
            log.debug(f"Reddit mentions for catalyst: {e}")
    candidates: List[MomentumCandidate] = []
    for ticker in MOMENTUM_UNIVERSE:
        try:
            # Filter out tickers with earnings in 0-2 days (IV crush risk)
            earnings_near = finnhub_collector.is_earnings_soon(ticker, days=2)
            if earnings_near.get("has_earnings"):
                continue
            bars = polygon_enhanced.get_daily_bars(ticker, limit=30)
            if not bars or len(bars) < 10:
                continue
            vol_avg = _avg_volume(bars, 20)
            vol_today = bars[0].get("v", bars[0].get("volume", 0))
            if vol_avg <= 0:
                continue
            surge = vol_today / vol_avg if vol_avg else 0
            if surge < VOLUME_SURGE_MULT:
                continue
            chg_5 = _price_change_pct(bars, 5)
            chg_1 = _price_change_pct(bars, 1)
            price_up = (chg_5 is not None and chg_5 >= PRICE_UP_5D_PCT) or (
                chg_1 is not None and chg_1 >= PRICE_UP_1D_PCT
            )
            if not price_up:
                continue
            close = bars[0].get("c", bars[0].get("close", 0))
            confidence = MIN_CONFIDENCE_MOMENTUM + min(20, (surge - 1) * 10 + (chg_5 or 0))
            # Optional boost when earnings in 5-14 days (catalyst)
            earnings_14 = finnhub_collector.is_earnings_soon(ticker, days=14)
            if earnings_14.get("has_earnings") and earnings_14.get("days_away", 99) >= 5:
                confidence = min(95, confidence + 5)
            confidence = min(95, confidence)
            # Catalyst filter: require earnings 5-14d or WSB mentions when enabled
            if REQUIRE_CATALYST and not _has_catalyst(ticker, earnings_14, wsb_mentions):
                continue
            candidates.append(
                MomentumCandidate(
                    ticker=ticker,
                    direction="long",
                    confidence=confidence,
                    catalyst="volume_surge_momentum",
                    expiry_preference="weekly",
                    detected_at=datetime.utcnow(),
                    entry_price=close,
                    notes=f"vol {surge:.1f}x 5d {chg_5:.1f}%" if chg_5 else f"vol {surge:.1f}x",
                )
            )
        except Exception as e:
            log.debug(f"Momentum scan {ticker}: {e}")
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def execute_momentum_entry(candidate: MomentumCandidate) -> bool:
    """Execute one momentum entry via executor (weekly expiry, engine=MOMENTUM).

    CRITICAL: Small-cap options have terrible theta decay on weekly holds.
    This engine is DISABLED for options by default. Set MOMENTUM_USE_OPTIONS=True to enable.
    """
    try:
        # HYDRA FIX: Check if options trading is enabled for momentum
        from wsb_snake.config import MOMENTUM_USE_OPTIONS
        if not MOMENTUM_USE_OPTIONS:
            log.info(f"Momentum engine: Skipping {candidate.ticker} - options disabled for small caps (MOMENTUM_USE_OPTIONS=False)")
            return False
        # Earnings within 2d – skip buy (IV crush risk)
        earnings_check = finnhub_collector.is_earnings_soon(candidate.ticker, days=2)
        if earnings_check.get("has_earnings"):
            log.info(f"Earnings within 2d – skip buy on {candidate.ticker} (IV crush risk)")
            return False
        # Next Friday expiry for weekly
        from datetime import datetime as dt
        import pytz
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et)
        days_until_friday = (4 - now.weekday()) % 7
        if days_until_friday == 0 and now.hour >= 16:
            days_until_friday = 7
        expiry = (now + timedelta(days=days_until_friday)).replace(tzinfo=None)
        entry = candidate.entry_price or 0
        if entry <= 0:
            snap = polygon_enhanced.get_snapshot(candidate.ticker)
            if snap:
                entry = snap.get("price", 0) or snap.get("last_trade_price", 0)
        if entry <= 0:
            return False
        stop = entry * 0.92
        target = entry * 1.15
        pos = alpaca_executor.execute_scalp_entry(
            underlying=candidate.ticker,
            direction=candidate.direction,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            confidence=candidate.confidence,
            pattern=candidate.catalyst,
            engine=TradingEngine.MOMENTUM,
            expiry_override=expiry,
        )
        if pos:
            send_telegram_alert(
                f"🚀 **MOMENTUM** {candidate.ticker} {candidate.catalyst} @ {candidate.confidence:.0f}% "
                f"(trim at +50%, trail +20%)"
            )
            return True
        return False
    except Exception as e:
        log.error(f"Momentum execute {candidate.ticker}: {e}")
        return False


def get_momentum_engine_status() -> Dict[str, Any]:
    return {
        "engine": "momentum",
        "universe_size": len(MOMENTUM_UNIVERSE),
        "universe": MOMENTUM_UNIVERSE,
        "min_confidence": MIN_CONFIDENCE_MOMENTUM,
        "scan_interval_sec": MOMENTUM_SCAN_INTERVAL_SEC,
    }


# Background runner (UNHINGED: run every 2 min)
_running = False
_worker: Optional[threading.Thread] = None
_last_scan: Optional[datetime] = None
_cooldown_until: Optional[datetime] = None
MOMENTUM_COOLDOWN_MINUTES = 30  # don't re-enter same ticker for 30 min


def start_momentum_engine():
    """Start background momentum scanner + executor."""
    global _running, _worker
    if _running:
        return
    _running = True
    _worker = threading.Thread(target=_momentum_loop, daemon=True)
    _worker.start()
    log.info("Momentum engine started (UNHINGED: 2min scan)")


def stop_momentum_engine():
    global _running
    _running = False


def _momentum_loop():
    while _running:
        try:
            from wsb_snake.utils.session_regime import is_market_open
            if is_market_open():
                cands = scan_momentum_universe()
                if cands:
                    best = cands[0]
                    if best.confidence >= MIN_CONFIDENCE_MOMENTUM:
                        global _cooldown_until
                        if _cooldown_until and datetime.utcnow() < _cooldown_until:
                            pass
                        else:
                            if execute_momentum_entry(best):
                                _cooldown_until = datetime.utcnow() + timedelta(minutes=MOMENTUM_COOLDOWN_MINUTES)
            time.sleep(MOMENTUM_SCAN_INTERVAL_SEC)
        except Exception as e:
            log.error(f"Momentum loop: {e}")
            time.sleep(60)
