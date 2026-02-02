"""
LEAPS / Macro Engine – long-dated thesis. UNHINGED: full scan + optional AI + execution.

Replicates SLV/SPY/META LEAPS style: macro thesis, 1–3 year calls, hold through volatility.
Tag: TradingEngine.MACRO. Trim at +50%, trail +20%.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from wsb_snake.utils.logger import get_logger
from wsb_snake.config import LEAPS_UNIVERSE, LEAPS_EXPIRY_MONTHS_MIN
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.trading.alpaca_executor import alpaca_executor
from wsb_snake.trading.risk_governor import TradingEngine, get_risk_governor
from wsb_snake.notifications.telegram_bot import send_alert as send_telegram_alert

log = get_logger(__name__)

# UNHINGED: scan every 30 min, lower bar
LEAPS_SCAN_INTERVAL_SEC = 1800
MIN_CONFIDENCE_LEAPS = 62
SMA_DAYS = 50
TREND_ABOVE_SMA_PCT = 0.5  # close > SMA by at least 0.5%


@dataclass
class LEAPSCandidate:
    ticker: str
    direction: str
    confidence: float
    thesis: str
    expiry_months_min: int
    detected_at: datetime
    entry_price: float = 0.0
    notes: str = ""


def _sma(bars: List[Dict], n: int) -> Optional[float]:
    if not bars or len(bars) < n:
        return None
    closes = [b.get("c", b.get("close", 0)) for b in bars[:n]]
    if not all(closes):
        return None
    return sum(closes) / len(closes)


def scan_leaps_universe() -> List[LEAPSCandidate]:
    """Scan LEAPS_UNIVERSE for macro/trend: close > 50-day SMA. Optional AI thesis (if OPENAI)."""
    candidates: List[LEAPSCandidate] = []
    for ticker in LEAPS_UNIVERSE:
        try:
            bars = polygon_enhanced.get_daily_bars(ticker, limit=60)
            if not bars or len(bars) < SMA_DAYS:
                continue
            close = bars[0].get("c", bars[0].get("close", 0))
            sma50 = _sma(bars, SMA_DAYS)
            if not close or not sma50 or sma50 <= 0:
                continue
            above = (close - sma50) / sma50 * 100
            if above < TREND_ABOVE_SMA_PCT:
                continue
            thesis = "uptrend_50d"
            confidence = MIN_CONFIDENCE_LEAPS + min(25, above * 2)
            confidence = min(92, confidence)
            candidates.append(
                LEAPSCandidate(
                    ticker=ticker,
                    direction="long",
                    confidence=confidence,
                    thesis=thesis,
                    expiry_months_min=LEAPS_EXPIRY_MONTHS_MIN,
                    detected_at=datetime.utcnow(),
                    entry_price=close,
                    notes=f"close {above:.1f}% above 50d SMA",
                )
            )
        except Exception as e:
            log.debug(f"LEAPS scan {ticker}: {e}")
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def _optional_ai_thesis(ticker: str, thesis: str, entry: float) -> float:
    """Optional GPT-4o macro thesis boost. Returns confidence delta (-5 to +10)."""
    if not os.environ.get("OPENAI_API_KEY"):
        return 0
    try:
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a macro options analyst. Reply with ONLY a number from 0 to 10: how much to add to confidence for a LEAPS call on this ticker (0=no change, 10=strong conviction). One short line."},
                {"role": "user", "content": f"Ticker {ticker}, thesis: {thesis}, entry ~{entry:.2f}. Add 0-10 to confidence."},
            ],
            max_tokens=20,
        )
        text = (r.choices[0].message.content or "0").strip()
        return max(-5, min(10, int(text) if text.isdigit() else 0))
    except Exception as e:
        log.debug(f"LEAPS AI thesis skip: {e}")
        return 0


def execute_leaps_entry(candidate: LEAPSCandidate) -> bool:
    """Execute one LEAPS entry: expiry 12–18 months out, engine=MACRO."""
    try:
        # Expiry 12–18 months from now (UNHINGED: 12 months)
        from datetime import date
        today = date.today()
        expiry_date = today + timedelta(days=365 * (LEAPS_EXPIRY_MONTHS_MIN // 12 + 1))
        expiry = datetime(expiry_date.year, expiry_date.month, expiry_date.day)
        entry = candidate.entry_price or 0
        if entry <= 0:
            snap = polygon_enhanced.get_snapshot(candidate.ticker)
            if snap:
                entry = snap.get("price", 0) or snap.get("last_trade_price", 0)
        if entry <= 0:
            return False
        # Optional AI boost
        conf = candidate.confidence + _optional_ai_thesis(candidate.ticker, candidate.thesis, entry)
        conf = min(95, conf)
        stop = entry * 0.88
        target = entry * 1.50
        pos = alpaca_executor.execute_scalp_entry(
            underlying=candidate.ticker,
            direction=candidate.direction,
            entry_price=entry,
            target_price=target,
            stop_loss=stop,
            confidence=conf,
            pattern=candidate.thesis,
            engine=TradingEngine.MACRO,
            expiry_override=expiry,
        )
        if pos:
            send_telegram_alert(
                f"📈 **LEAPS** {candidate.ticker} {candidate.thesis} @ {conf:.0f}% "
                f"(trim +50%, trail +20%)"
            )
            return True
        return False
    except Exception as e:
        log.error(f"LEAPS execute {candidate.ticker}: {e}")
        return False


def get_leaps_engine_status() -> Dict[str, Any]:
    return {
        "engine": "leaps",
        "universe_size": len(LEAPS_UNIVERSE),
        "universe": LEAPS_UNIVERSE,
        "expiry_months_min": LEAPS_EXPIRY_MONTHS_MIN,
        "min_confidence": MIN_CONFIDENCE_LEAPS,
        "scan_interval_sec": LEAPS_SCAN_INTERVAL_SEC,
    }


# Background runner (UNHINGED: every 30 min)
_running = False
_worker: Optional[threading.Thread] = None
_leaps_cooldown_until: Optional[datetime] = None
LEAPS_COOLDOWN_MINUTES = 1440  # 24h between LEAPS entries (capital tied long)


def start_leaps_engine():
    global _running, _worker
    if _running:
        return
    _running = True
    _worker = threading.Thread(target=_leaps_loop, daemon=True)
    _worker.start()
    log.info("LEAPS engine started (UNHINGED: 30min scan)")


def stop_leaps_engine():
    global _running
    _running = False


def _leaps_loop():
    global _leaps_cooldown_until
    while _running:
        try:
            from wsb_snake.utils.session_regime import is_market_open
            if is_market_open():
                cands = scan_leaps_universe()
                if cands:
                    best = cands[0]
                    if best.confidence >= MIN_CONFIDENCE_LEAPS:
                        if _leaps_cooldown_until and datetime.utcnow() < _leaps_cooldown_until:
                            pass
                        else:
                            if execute_leaps_entry(best):
                                _leaps_cooldown_until = datetime.utcnow() + timedelta(minutes=LEAPS_COOLDOWN_MINUTES)
            time.sleep(LEAPS_SCAN_INTERVAL_SEC)
        except Exception as e:
            log.error(f"LEAPS loop: {e}")
            time.sleep(300)
