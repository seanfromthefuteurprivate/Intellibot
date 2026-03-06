"""
Convexity Proof Layer (CPL) — 0DTE Signal Generation Engine.

Generates atomic option calls (BUY/SELL) for 0DTE trading,
with deduplication (memory + DB) and Telegram broadcast.

Originally built for NFP Jobs Day events, now extended for daily 0DTE trading.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.collectors.polygon_options import polygon_options
from wsb_snake.db.database import cpl_call_exists, save_cpl_call
from wsb_snake.execution.call_schema import JobsDayCall
from wsb_snake.notifications.message_templates import (
    format_jobs_day_call,
    format_jobs_day_sell,
    format_session_header,
)
from wsb_snake.notifications.telegram_bot import send_alert
from wsb_snake.db.database import save_cpl_outcome
from wsb_snake.utils.logger import get_logger
from wsb_snake.trading.alpaca_executor import alpaca_executor
from wsb_snake.trading.risk_governor import TradingEngine
from wsb_snake.collectors.hydra_bridge import get_hydra_intel

# Apex Governance Layer (toggleable)
try:
    from wsb_snake.execution.apex_governance import ApexRunnerGovernance, GOVERNANCE_ENABLED, GovernanceState
    from wsb_snake.execution.telemetry_bus import get_telemetry_bus, TelemetryEventBus
    _telemetry_bus = get_telemetry_bus()
    _governance = ApexRunnerGovernance(enabled=GOVERNANCE_ENABLED, telemetry_bus=_telemetry_bus)
except ImportError:
    GOVERNANCE_ENABLED = False
    _governance = None
    _telemetry_bus = None

logger = get_logger(__name__)

# Direction lock: one direction per ticker per day (prevents CALL+PUT whipsaw)
_direction_lock = {}  # {ticker: "CALL" or "PUT", "_last_reset": "YYYY-MM-DD"}

# Opening Range Breakout: Track 9:30-9:35 AM high/low for SPY/QQQ
# Format: {"SPY": {"high": 585.50, "low": 584.20, "date": "2026-03-04"}, ...}
_opening_range = {}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    SECOND BRAIN INTEGRATION                                   ║
# ║              Persistent memory for cross-session context                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
SECOND_BRAIN_URL = "http://98.82.24.119:8080"

def _report_trade_to_second_brain(
    ticker: str,
    side: str,
    strike: float,
    entry_price: float,
    option_symbol: str,
    confidence: float,
    signals: List[str] = None,
    note: str = None
) -> bool:
    """
    Report a trade execution to Second Brain for persistent memory.

    This ensures trade history persists across Claude Code context resets.
    """
    import requests as req
    try:
        payload = {
            "ticker": ticker,
            "side": side,
            "strike": strike,
            "entry": entry_price,
            "option_symbol": option_symbol,
            "conviction": confidence,
            "signals": signals or [],
            "note": note or f"V6_SNIPER_{side}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        resp = req.post(
            f"{SECOND_BRAIN_URL}/api/report/trade",
            json=payload,
            timeout=5
        )
        if resp.status_code == 200:
            logger.info(f"SECOND_BRAIN: Trade reported - {ticker} {side} ${strike}")
            return True
        else:
            logger.warning(f"SECOND_BRAIN: Report failed ({resp.status_code})")
            return False
    except Exception as e:
        logger.warning(f"SECOND_BRAIN: Report error - {e}")
        return False


def _update_opening_range():
    """
    Fetch and store the 9:30-9:35 AM 5-min bar high/low for SPY and QQQ.
    Called once per day after 9:35 AM ET.
    """
    global _opening_range
    from zoneinfo import ZoneInfo
    et_now = datetime.now(ZoneInfo("America/New_York"))
    today = et_now.strftime("%Y-%m-%d")

    # Only update once per day, and only after 9:35 AM
    if et_now.hour < 9 or (et_now.hour == 9 and et_now.minute < 35):
        return

    for ticker in ["SPY", "QQQ"]:
        # Skip if already captured today
        if ticker in _opening_range and _opening_range[ticker].get("date") == today:
            continue

        try:
            # Fetch the first 5-min bar of the day (9:30-9:35)
            bars = polygon_enhanced.get_intraday_bars(
                ticker, timespan="minute", multiplier=5, limit=1,
                from_time=f"{today}T09:30:00", to_time=f"{today}T09:35:00"
            )
            if bars and len(bars) > 0:
                bar = bars[0]
                high = bar.get('high') or bar.get('h')
                low = bar.get('low') or bar.get('l')
                if high and low:
                    _opening_range[ticker] = {
                        "high": float(high),
                        "low": float(low),
                        "date": today
                    }
                    logger.info(f"OPENING_RANGE_SET: {ticker} high={high:.2f} low={low:.2f}")
        except Exception as e:
            logger.debug(f"Opening range fetch failed for {ticker}: {e}")


def get_todays_expiry_date() -> str:
    """
    Get today's date in YYYY-MM-DD format for 0DTE options.
    Uses Eastern Time since options expire based on ET market hours.
    """
    try:
        import zoneinfo
        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except Exception:
        import pytz
        et = datetime.now(pytz.timezone("US/Eastern"))
    return et.strftime("%Y-%m-%d")


# Dynamic 0DTE expiry: always use today's date for SPY/QQQ/IWM daily expirations
CPL_EVENT_DATE = get_todays_expiry_date()
# Strike mode: full event-vol watchlist to capture all macro/WSB events (NFP day + risk-off/earnings/crypto days)
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           V6 SNIPER MODE                                      ║
# ║                   SPY + QQQ ONLY — ONE TRADE PER DAY                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# LESSON LEARNED (Week 1): V1 with NO filters found QQQ $609C winner (+$1,551).
# Every version after with MORE filters performed WORSE.
# V6 strips back to basics: SPY/QQQ only, one direction, one trade, let execution work.
CPL_WATCHLIST = ["SPY", "QQQ"]  # HARD-CODED — NO OTHER TICKERS

# FIX 1: Target forcing function
TARGET_BUY_CALLS = 3

# SNIPER MODE CONFIG
SNIPER_CAPITAL = 2500               # Position sizing base (pretend cap)
MAX_OPEN_POSITIONS = 1              # One shot at a time
DAILY_PROFIT_TARGET = 10000         # +$10,000 = halt (Beast Mode)
DAILY_MAX_LOSS = -750               # -$750 = halt. Week 1 lesson: cut losses early.
SNIPER_COOLDOWN_SECONDS = 300       # 5-min cooldown after ANY trade (prevents API lag race)

# Local tracking to prevent API lag race condition (mutable dict for nested scope)
_sniper_state = {"last_trade_time": None}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      V6 SNIPER: ONE DIRECTION, ONE TRADE                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# Rule 1: ONE DIRECTION PER DAY - first 15 min of SPY determines CALL or PUT
# Rule 2: ONE TRADE THEN DONE - after first trade executes, STOP for the day
_v6_state = {
    "daily_direction": None,      # "CALL" or "PUT" - set by first 15 min SPY move
    "direction_set": False,       # True once direction is locked
    "direction_set_time": None,   # When direction was determined
    "trade_complete": False,      # True after first trade - NO MORE SCANS
    "trade_completed_time": None, # When trade was executed
    "last_reset_date": None,      # Date of last daily reset
}

# ═══════════════════════════════════════════════════════════════════════════════
# V6 TRADE_COMPLETE PERSISTENCE — Survives systemd restarts
# Week 1 bug: service crashed mid-day, restarted, trade_complete was reset,
# system placed ANOTHER trade the same day. File flag prevents this.
# ═══════════════════════════════════════════════════════════════════════════════

def _get_trade_complete_flag_path() -> str:
    """Get path for today's trade_complete persistence flag."""
    from zoneinfo import ZoneInfo
    et_now = datetime.now(ZoneInfo("America/New_York"))
    today_str = et_now.strftime("%Y-%m-%d")
    return f"/tmp/wsb_snake_trade_complete_{today_str}.flag"


def _check_persisted_trade_complete() -> bool:
    """Check if trade_complete was set before a service restart."""
    flag_path = _get_trade_complete_flag_path()
    if os.path.exists(flag_path):
        logger.info(f"TRADE_COMPLETE_PERSISTENCE: Found flag {flag_path} — trade already done today.")
        return True
    return False


def _persist_trade_complete():
    """Write trade_complete flag to disk so it survives systemd restarts."""
    flag_path = _get_trade_complete_flag_path()
    try:
        with open(flag_path, "w") as f:
            f.write(f"trade_complete=True\nset_at={datetime.now(timezone.utc).isoformat()}\n")
        logger.info(f"TRADE_COMPLETE_PERSISTENCE: Wrote flag to {flag_path}")
    except Exception as e:
        logger.error(f"TRADE_COMPLETE_PERSISTENCE: Failed to write flag: {e}")


# FIX 3: Liquidity gates — relaxed so we get 5–10 HIGH hitters (SPY/QQQ ATM often > $2.50)
LIQUIDITY_MAX_SPREAD_PCT = 0.15  # 15% of mid (slightly relaxed)
LIQUIDITY_MIN_MID = 0.30         # block sub-$0.30 lottos (IWM $0.17 lost $350)
LIQUIDITY_MAX_MID = 6.00         # allow ATM on indices (was 2.50, rejected most)

# FIX 4: Cooldown config — 0 in untruncated so we get next BUY fast (5–10 round-trips per day)
COOLDOWN_MINUTES = 45

# Paper proof constraints
MAX_COST_PER_CONTRACT = 250

# Diversity: required for proof/demo; optional for future "production alpha" (set to None to disable)
DIVERSITY_MODE = "PROOF"
MAX_CALLS_PER_UNDERLYING = 2

# ========== ALPACA AUTO-EXECUTION ==========
# When True, CPL BUY calls are automatically executed on Alpaca paper trading
# DEFAULT: True (execute trades automatically) - set CPL_AUTO_EXECUTE=false to disable
import os
CPL_AUTO_EXECUTE = os.environ.get("CPL_AUTO_EXECUTE", "true").lower() == "true"
logger.info(f"CPL_AUTO_EXECUTE = {CPL_AUTO_EXECUTE} (env: {os.environ.get('CPL_AUTO_EXECUTE', 'NOT SET - defaulting to TRUE')})")

# In-memory dedupe (session)
_sent_calls: Set[str] = set()

# FIX 4: Cooldown tracking: {ticker|side|expiry -> last_emit_timestamp}
_cooldown_tracker: Dict[str, datetime] = {}

# FIX 2: Open positions for SELL tracking
# {dedupe_key -> {entry_price, entry_time, call, buy_call_number}}
_open_positions: Dict[str, Dict[str, Any]] = {}

# Track call numbers for SELL lineage
_call_number_counter = 0

# Paper trader Friday: untruncated tails (sequential $250 -> 5 figures, human execution)
UNTRUNCATED_TAILS = False
# Power hour: 12:00–16:00 ET overtime — shorter cooldown, more calls per underlying
CPL_POWER_HOUR = False
_session_balance: float = 250.0
_trade_counter: int = 0
# Telegram message sequencing (incremented per BUY/SELL sent)
_message_sequence: int = 0
_last_session_date: Optional[str] = None


def _get_spot_price(ticker: str) -> Optional[float]:
    """Get current spot price for ticker."""
    quote = polygon_options.get_quote(ticker) if polygon_options else None
    if quote and quote.get("price"):
        return float(quote["price"])
    if polygon_enhanced:
        snap = polygon_enhanced.get_snapshot(ticker)
        if snap and snap.get("price"):
            return float(snap["price"])
    return None


def _get_hydra_size_multiplier() -> float:
    """
    Get position size multiplier based on HYDRA blowup probability.

    Returns:
        1.0 (full size) if blowup <= 50
        0.5 (half size) if blowup 51-70
        0.0 (no trade) if blowup > 70
    """
    try:
        hydra = get_hydra_intel()
        if hydra.blowup_probability > 70:
            return 0.0  # Block trade
        elif hydra.blowup_probability > 50:
            return 0.5  # Half size
        return 1.0  # Full size
    except Exception as e:
        logger.warning(f"HYDRA size multiplier failed: {e}")
        return 1.0  # Default to full size on error


def _determine_v6_direction() -> Optional[str]:
    """
    V6 SNIPER: Determine daily direction based on first 15 minutes of SPY.

    Rules:
    - Wait until 9:45 AM ET (15 min after open)
    - Check SPY price change from 9:30 open
    - If SPY +0.2% or more: CALLS only all day
    - If SPY -0.2% or more: PUTS only all day
    - If between -0.2% and +0.2%: Wait for breakout (check again next scan)

    Returns: "CALL", "PUT", or None (still waiting)
    """
    global _v6_state
    from zoneinfo import ZoneInfo
    et_now = datetime.now(ZoneInfo("America/New_York"))

    # Must be after 9:45 AM to have 15 min of data
    if et_now.hour < 9 or (et_now.hour == 9 and et_now.minute < 45):
        logger.debug(f"V6_DIRECTION: Waiting for 9:45 AM (currently {et_now.hour}:{et_now.minute:02d})")
        return None

    # Already set today
    if _v6_state["direction_set"]:
        return _v6_state["daily_direction"]

    # Get SPY snapshot with today's open and current price
    try:
        # Use snapshot which includes today_open and change_pct
        spy_snapshot = polygon_enhanced.get_snapshot("SPY")
        if not spy_snapshot:
            logger.warning("V6_DIRECTION: No snapshot for SPY")
            return None

        open_price = spy_snapshot.get("today_open")
        current_price = spy_snapshot.get("price")
        if not open_price or not current_price:
            logger.warning(f"V6_DIRECTION: Missing price data (open={open_price}, current={current_price})")
            return None

        # Calculate change percentage
        change_pct = ((current_price - open_price) / open_price) * 100

        logger.info(f"V6_DIRECTION_CHECK: SPY open={open_price:.2f} current={current_price:.2f} change={change_pct:+.2f}%")

        # Determine direction
        if change_pct >= 0.2:
            direction = "CALL"
            logger.info(f"V6_DIRECTION_SET: CALLS ONLY (SPY +{change_pct:.2f}%)")
        elif change_pct <= -0.2:
            direction = "PUT"
            logger.info(f"V6_DIRECTION_SET: PUTS ONLY (SPY {change_pct:.2f}%)")
        else:
            # Not enough move yet - keep waiting
            logger.info(f"V6_DIRECTION_WAIT: SPY {change_pct:+.2f}% (need ±0.2% for direction lock)")
            return None

        # Lock direction for the day
        _v6_state["daily_direction"] = direction
        _v6_state["direction_set"] = True
        _v6_state["direction_set_time"] = datetime.now(timezone.utc)

        # Send Telegram alert
        try:
            from wsb_snake.notifications.telegram_bot import send_alert
            send_alert(f"🎯 V6 SNIPER DIRECTION LOCKED: {direction}S ONLY\nSPY {change_pct:+.2f}% from open\nOne trade, let it ride.")
        except Exception:
            pass

        return direction

    except Exception as e:
        logger.warning(f"V6_DIRECTION_ERROR: {e}")
        return None


def _is_affordable(contract: Dict[str, Any], max_cap_override: Optional[float] = None) -> bool:
    """Contract cost (ask or last_price * 100) <= cap. When untruncated, cap = session_balance."""
    price = contract.get("ask") or contract.get("last_price") or 0
    cap = max_cap_override if max_cap_override is not None else MAX_COST_PER_CONTRACT
    return price > 0 and (price * 100) <= cap


def _check_liquidity(contract: Dict[str, Any], ticker: str, side: str, strike: float) -> Tuple[bool, Optional[str]]:
    """
    FIX 3: Enforce liquidity gates BEFORE affordability.
    Returns (is_valid, rejection_reason or None)
    """
    bid = contract.get("bid") or 0
    ask = contract.get("ask") or contract.get("last_price") or 0

    # Calculate mid price
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2
    elif ask > 0:
        mid = ask
    else:
        return False, f"LIQUIDITY_REJECT: {ticker} {side} @ {strike} - no valid quote (stale)"

    # Check mid in paper proof range
    if mid < LIQUIDITY_MIN_MID:
        return False, f"LIQUIDITY_REJECT: {ticker} {side} @ {strike} - mid ${mid:.2f} < ${LIQUIDITY_MIN_MID}"
    if mid > LIQUIDITY_MAX_MID:
        return False, f"LIQUIDITY_REJECT: {ticker} {side} @ {strike} - mid ${mid:.2f} > ${LIQUIDITY_MAX_MID}"

    # Check spread as % of mid
    if bid > 0 and ask > 0:
        spread = ask - bid
        spread_pct = spread / mid if mid > 0 else 1.0
        if spread_pct > LIQUIDITY_MAX_SPREAD_PCT:
            return False, f"LIQUIDITY_REJECT: {ticker} {side} @ {strike} - spread {spread_pct:.1%} > {LIQUIDITY_MAX_SPREAD_PCT:.0%}"

    # Check for stale quote flag
    if contract.get("stale") or contract.get("is_stale"):
        return False, f"LIQUIDITY_REJECT: {ticker} {side} @ {strike} - stale quote detected"

    return True, None


def _window() -> str:
    """AM before 10:30 ET, PM after."""
    try:
        import zoneinfo
        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
        return "AM" if et.hour < 10 or (et.hour == 10 and et.minute < 30) else "PM"
    except Exception:
        return "AM"


def _is_execution_ready(call: JobsDayCall) -> tuple[bool, Optional[str]]:
    """
    FIX 4: Assert all required fields present and non-null before BUY broadcast.
    Returns (True, None) if ready; (False, reason) if not.
    """
    if not (call.underlying and call.side and call.expiry_date):
        return False, "missing underlying/side/expiry_date"
    if call.dte is None:
        return False, "missing dte"
    if not (call.strike is not None and call.strike > 0):
        return False, "missing or invalid strike"
    entry_price = call.entry_trigger.get("price") if call.entry_trigger else None
    if entry_price is None or (isinstance(entry_price, (int, float)) and entry_price <= 0):
        return False, "missing or invalid entry_trigger.price"
    stop_price = call.stop_loss.get("price") if call.stop_loss else None
    if stop_price is None or (isinstance(stop_price, (int, float)) and stop_price <= 0):
        return False, "missing or invalid stop_loss.price"
    tp = call.take_profit or []
    if len(tp) < 2:
        return False, "take_profit must have at least TP1 and TP2"
    if not (call.dedupe_key and call.dedupe_key.strip()):
        return False, "missing dedupe_key"
    return True, None


def _event_tier(spot: float, strike: float, side: str, entry_price: float, momentum_score: float = 50) -> str:
    """
    SMART TIER: Classify with momentum weighting.
    High momentum + ATM = better than low momentum + OTM lottery.
    Prioritizes setups where momentum confirms direction.
    """
    if not spot or spot <= 0 or entry_price <= 0:
        return "2X"

    # Calculate OTM percentage
    if side.upper() == "CALL":
        otm_pct = ((strike - spot) / spot) * 100 if strike > spot else 0
    else:
        otm_pct = ((spot - strike) / spot) * 100 if strike < spot else 0

    # ATM or slight OTM with high momentum = BEST (reliable 2-4x moves)
    if otm_pct < 1.0 and momentum_score >= 65:
        return "4X"  # ATM with strong momentum
    if otm_pct < 2.0 and momentum_score >= 55:
        return "4X"  # Near-money with decent momentum

    # Deep OTM lottery only if momentum is VERY strong
    if entry_price < 0.25 and otm_pct >= 0.5 and momentum_score >= 70:
        return "20X"
    if entry_price < 0.50 and otm_pct >= 0.3 and momentum_score >= 60:
        return "20X"

    # Moderate setups need momentum confirmation
    if entry_price < 0.80 and otm_pct >= 0.2 and momentum_score >= 55:
        return "4X"
    if momentum_score >= 60:
        return "4X"

    return "2X"


def _tier_rank(tier: str) -> int:
    """Lower = higher priority for selection (20X first, then 4X, then 2X)."""
    return {"20X": 0, "4X": 1, "2X": 2}.get(tier or "2X", 3)


def _visual_key(call: JobsDayCall) -> Tuple[str, str, float, str, float]:
    """FIX 3: Key for message-level uniqueness (strike, side, expiry, entry price)."""
    entry_price = call.entry_trigger.get("price") or 0
    return (
        (call.underlying or "").upper(),
        (call.side or "").upper(),
        float(call.strike or 0),
        call.expiry_date or "",
        round(float(entry_price), 2),
    )


def _get_cooldown_key(ticker: str, side: str, expiry_date: str) -> str:
    """FIX 4: Cooldown key includes expiry_date."""
    return f"{ticker.upper()}|{side.upper()}|{expiry_date}"


def _is_cooldown_active(ticker: str, side: str, expiry_date: str, effective_cooldown_minutes: Optional[float] = None) -> bool:
    """FIX 4: Check if cooldown is active. When untruncated, effective_cooldown_minutes=0."""
    key = _get_cooldown_key(ticker, side, expiry_date)
    last_emit = _cooldown_tracker.get(key)
    if not last_emit:
        return False
    mins = effective_cooldown_minutes if effective_cooldown_minutes is not None else COOLDOWN_MINUTES
    if mins <= 0:
        return False
    elapsed = (datetime.now(timezone.utc) - last_emit).total_seconds() / 60
    return elapsed < mins


def _set_cooldown(ticker: str, side: str, expiry_date: str):
    """FIX 4: Set cooldown timestamp for ticker|side|expiry."""
    key = _get_cooldown_key(ticker, side, expiry_date)
    _cooldown_tracker[key] = datetime.now(timezone.utc)


def _get_current_option_price(option_symbol: str, ticker: str) -> Optional[float]:
    """Get current price for an option to check exit conditions."""
    if not option_symbol:
        return None
    try:
        quote = polygon_options.get_option_quote(option_symbol) if hasattr(polygon_options, 'get_option_quote') else None
        if quote and quote.get("mid"):
            return float(quote["mid"])
        if quote and quote.get("last"):
            return float(quote["last"])
    except Exception:
        pass
    return None


def _check_entry_quality(ticker: str, side: str, spot: float) -> Tuple[bool, float, str]:
    """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                           V7 SNAPSHOT MODE                                    ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    LESSON LEARNED (Week 1):
    - V1 (no filters) found QQQ $609C winner: +$1,551
    - V6 required intraday bars → Polygon returns 0 → ALL trades blocked → $0
    - The execution layer (pyramids + trailing stops) IS the edge
    - This check's job: reject obvious garbage, approve anything reasonable

    AVAILABLE DATA (Polygon Starter plan):
    - get_snapshot(): today_open, price, today_high, today_low, today_volume, change_pct ✅
    - get_previous_day(): prev open, high, low, close, volume ✅
    - get_intraday_bars(): UNRELIABLE (returns 0-1 bars) ❌

    CHECKS:
    1. Price moving in trade direction (from today's open, via snapshot)
    2. Not at session extreme (don't buy calls at day high, puts at day low)
    3. Volume sanity (today's volume > 0)
    4. Time filter (skip first 15 min — direction lock handles this, but be safe)

    Returns: (is_valid, confidence_score, reason)
    """
    side_upper = side.upper()

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 1: Get snapshot data (this WORKS on Polygon Starter)
    # ════════════════════════════════════════════════════════════════════════════
    try:
        snapshot = polygon_enhanced.get_snapshot(ticker)
    except Exception as e:
        logger.warning(f"V7_REJECT: {ticker} {side_upper} - snapshot failed: {e}")
        return False, 0, "V7_DATA: Snapshot unavailable"

    if not snapshot or not snapshot.get("price"):
        logger.info(f"V7_REJECT: {ticker} {side_upper} - no snapshot data")
        return False, 0, "V7_DATA: No snapshot"

    current_price = float(snapshot["price"])
    today_open = float(snapshot.get("today_open", 0))
    today_high = float(snapshot.get("today_high", current_price))
    today_low = float(snapshot.get("today_low", current_price))
    today_volume = int(snapshot.get("today_volume", 0))

    if today_open <= 0:
        logger.info(f"V7_REJECT: {ticker} {side_upper} - no open price in snapshot")
        return False, 0, "V7_DATA: No open price"

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 2: Direction alignment — price must be moving with the trade
    # ════════════════════════════════════════════════════════════════════════════
    # Calculate intraday change from today's open
    intraday_change_pct = ((current_price - today_open) / today_open) * 100

    # For CALLS: price should be above today's open (uptrend from open)
    # Allow flat (>= -0.1%) — don't require strong momentum, just not falling
    if side_upper == "CALL" and intraday_change_pct < -0.1:
        logger.info(
            f"V7_REJECT: {ticker} CALL - price BELOW open by {intraday_change_pct:.2f}% "
            f"(open={today_open:.2f}, now={current_price:.2f})"
        )
        return False, 0, f"V7_DIRECTION: Price below open {intraday_change_pct:+.2f}%"

    # For PUTS: price should be below today's open (downtrend from open)
    if side_upper == "PUT" and intraday_change_pct > 0.1:
        logger.info(
            f"V7_REJECT: {ticker} PUT - price ABOVE open by {intraday_change_pct:.2f}% "
            f"(open={today_open:.2f}, now={current_price:.2f})"
        )
        return False, 0, f"V7_DIRECTION: Price above open {intraday_change_pct:+.2f}%"

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 3: Don't chase extremes — avoid buying at day high/low
    # ════════════════════════════════════════════════════════════════════════════
    day_range = today_high - today_low
    if day_range > 0.01:  # Avoid division by near-zero
        position_in_range = (current_price - today_low) / day_range

        # For CALLS: don't buy if price is in top 5% of today's range
        if side_upper == "CALL" and position_in_range > 0.95:
            logger.info(
                f"V7_REJECT: {ticker} CALL - at day high "
                f"({position_in_range:.0%} of range, high={today_high:.2f})"
            )
            return False, 0, f"V7_EXTREME: At day high ({position_in_range:.0%})"

        # For PUTS: don't buy if price is in bottom 5% of today's range
        if side_upper == "PUT" and position_in_range < 0.05:
            logger.info(
                f"V7_REJECT: {ticker} PUT - at day low "
                f"({position_in_range:.0%} of range, low={today_low:.2f})"
            )
            return False, 0, f"V7_EXTREME: At day low ({position_in_range:.0%})"

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 4: Volume sanity — make sure market is actually trading
    # ════════════════════════════════════════════════════════════════════════════
    if today_volume <= 0:
        logger.info(f"V7_REJECT: {ticker} {side_upper} - no volume data")
        return False, 0, "V7_VOLUME: No trading volume"

    # ════════════════════════════════════════════════════════════════════════════
    # STEP 5: Gap analysis (bonus confidence)
    # ════════════════════════════════════════════════════════════════════════════
    gap_boost = 0
    try:
        prev_day = polygon_enhanced.get_previous_day(ticker)
        if prev_day and prev_day.get("close", 0) > 0:
            prev_close = prev_day["close"]
            gap_pct = ((today_open - prev_close) / prev_close) * 100
            # Gap in trade direction = confidence boost
            if side_upper == "CALL" and gap_pct > 0.2:
                gap_boost = 5
            elif side_upper == "PUT" and gap_pct < -0.2:
                gap_boost = 5
    except Exception:
        pass  # Previous day data is a bonus, not required

    # ════════════════════════════════════════════════════════════════════════════
    # V7 APPROVED - Calculate confidence
    # ════════════════════════════════════════════════════════════════════════════
    # Base confidence 70% - we're trading SPY/QQQ with direction lock
    confidence = 70

    # Boost for momentum in direction
    if side_upper == "CALL" and intraday_change_pct > 0.15:
        confidence += 10
    elif side_upper == "PUT" and intraday_change_pct < -0.15:
        confidence += 10

    # Boost for very strong momentum
    if side_upper == "CALL" and intraday_change_pct > 0.3:
        confidence += 5
    elif side_upper == "PUT" and intraday_change_pct < -0.3:
        confidence += 5

    # Gap boost
    confidence += gap_boost

    confidence = min(95, confidence)

    reason = (
        f"V7_APPROVED: {ticker} {side_upper} "
        f"intraday={intraday_change_pct:+.2f}% "
        f"range_pos={((current_price - today_low) / max(today_high - today_low, 0.01)):.0%} "
        f"conf={confidence}%"
    )
    logger.info(reason)

    return True, confidence, reason


def _check_exits_and_emit_sells(broadcast: bool, dry_run: bool, untruncated_tails: bool = False) -> List[JobsDayCall]:
    """
    FIX 2: Check open positions for TP/SL/TIME hits and emit SELL calls.
    When untruncated_tails: no TIME cap for runners; TP observational only; SL/TIME governed by intelligent SL.
    """
    global _session_balance, _trade_counter
    sells: List[JobsDayCall] = []
    to_close = []
    # Untruncated: no 30-min cap (runners run until structure break); else 30 min
    max_hold_minutes = None if untruncated_tails else 30

    for dedupe_key, pos in _open_positions.items():
        entry_price = pos["entry_price"]
        entry_time = pos["entry_time"]
        call = pos["call"]
        buy_call_number = pos.get("buy_call_number", 0)
        contracts = pos.get("contracts", 1)

        current_price = _get_current_option_price(call.option_symbol, call.underlying)

        # For dry-run, simulate price movement
        if dry_run and current_price is None:
            import random
            if random.random() > 0.7:
                current_price = entry_price * (1 + random.uniform(-0.20, 0.30))

        if current_price is None:
            elapsed_minutes = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
            if max_hold_minutes is not None and elapsed_minutes >= max_hold_minutes:
                exit_reason = "TIME"
                exit_price = entry_price
                pnl_pct = 0.0
            else:
                continue
        else:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            elapsed_minutes = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
            exit_reason = None
            exit_price = current_price

            # SL always checked (keep stop loss)
            if pnl_pct <= -15:
                exit_reason = "SL"
            # TP: only set when NOT untruncated (untruncated = observational only, exit via structure break)
            elif not untruncated_tails and pnl_pct >= 25:
                exit_reason = "TP"
            # TIME: only when we have a cap (untruncated has no cap)
            elif max_hold_minutes is not None and elapsed_minutes >= max_hold_minutes:
                exit_reason = "TIME"

            if not exit_reason:
                if _governance and GOVERNANCE_ENABLED:
                    decision = _governance.evaluate_position(dedupe_key, current_price)
                    if decision.state == GovernanceState.RELEASE and not decision.structure_intact:
                        exit_reason = "STRUCTURE_BREAK"
                        if _telemetry_bus:
                            gov_pos = _governance.positions.get(dedupe_key) if _governance else None
                            _telemetry_bus.emit_structure_break_alert(
                                dedupe_key,
                                decision.reason,
                                pnl_pct,
                                entry_ref_price=entry_price,
                                exit_ref_price_val=current_price,
                                peak_ref_price=gov_pos.peak_price if gov_pos else None,
                            )
                        logger.warning(f"GOVERNANCE RELEASE: {dedupe_key} | {decision.reason}")
                    else:
                        continue
                else:
                    continue

            # GOVERNANCE: Check exit permitted for ALL reasons (TP, SL, TIME) when governance enabled
            if _governance and GOVERNANCE_ENABLED:
                permitted, gov_reason = _governance.is_exit_permitted(dedupe_key, exit_reason, current_price, untruncated_tails=untruncated_tails)
                if not permitted:
                    logger.info(f"GOVERNANCE EXIT_BLOCKED: {dedupe_key} | {gov_reason} | pnl={pnl_pct:.1f}%")
                    continue

        # FIX 2: Create SELL with original call_id lineage
        sell_call = JobsDayCall.create(
            underlying=call.underlying,
            side=call.side,
            strike=call.strike,
            expiry_date=call.expiry_date,
            dte=0,
            entry_price=exit_price,
            stop_pct=0,
            tp_pcts=[],
            option_symbol=call.option_symbol,
            regime=call.regime,
            confidence=call.confidence,
            reasons=[f"EXIT: {exit_reason}", f"Original BUY #{buy_call_number}"],
            window=_window(),
            action="SELL",
            original_call_id=call.call_id,  # Preserve lineage
        )
        sells.append(sell_call)
        to_close.append(dedupe_key)

        # R-multiple and CPL outcome recording
        stop_price = float(call.stop_loss.get("price") or 0) if call.stop_loss else 0
        r_multiple = None
        if entry_price and stop_price and entry_price > stop_price:
            risk = entry_price - stop_price
            r_multiple = (exit_price - entry_price) / risk if risk else None
        event_tier = getattr(call, "event_tier", None) or ""

        # FIX 5: SELL completeness — block broadcast if any required field missing
        if broadcast and not dry_run:
            if buy_call_number is None or buy_call_number <= 0:
                logger.warning("EXECUTION_INCOMPLETE_REJECT (SELL): missing original_buy_number")
            elif exit_price is None:
                logger.warning("EXECUTION_INCOMPLETE_REJECT (SELL): missing exit_price")
            elif not exit_reason:
                logger.warning("EXECUTION_INCOMPLETE_REJECT (SELL): missing exit_reason")
            elif pnl_pct is None:
                logger.warning("EXECUTION_INCOMPLETE_REJECT (SELL): missing pnl_pct")
            else:
                peak_ref = None
                if _governance and dedupe_key in _governance.positions:
                    peak_ref = getattr(_governance.positions[dedupe_key], "peak_price", None)
                # Update session balance (untruncated paper trader)
                if untruncated_tails:
                    _trade_counter += 1
                    dollar_pnl = (exit_price - entry_price) * 100 * contracts
                    _session_balance += dollar_pnl
                # Record CPL outcome for daily report (win rate, avg R, 2X/4X/20X)
                try:
                    et = datetime.now(timezone.utc)
                    try:
                        import zoneinfo
                        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
                    except Exception:
                        pass
                    trade_date = et.strftime("%Y-%m-%d")
                    holding_sec = int((datetime.now(timezone.utc) - entry_time).total_seconds())
                    save_cpl_outcome(
                        trade_date=trade_date,
                        symbol=call.underlying or "",
                        entry_price=entry_price,
                        exit_price=exit_price,
                        stop_price=stop_price,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        holding_time_seconds=holding_sec,
                        event_tier=event_tier or None,
                        session=_window(),
                    )
                except Exception as e:
                    logger.warning("save_cpl_outcome failed: %s", e)
                global _message_sequence
                _message_sequence += 1
                msg = format_jobs_day_sell(
                    call=sell_call,
                    original_buy_number=buy_call_number,
                    exit_reason=exit_reason,
                    exit_price=exit_price,
                    entry_price=entry_price,
                    pnl_pct=pnl_pct,
                    peak_ref_price=peak_ref,
                    untruncated_tails=untruncated_tails,
                    trade_number=_trade_counter if untruncated_tails else None,
                    session_balance=_session_balance if untruncated_tails else None,
                    contracts=contracts,
                    message_sequence=_message_sequence,
                    event_tier=event_tier or None,
                    r_multiple=r_multiple,
                )
                send_alert(msg)
                logger.info(f"CPL SELL broadcast: {call.underlying} {call.side} - {exit_reason} ({pnl_pct:+.1f}%)")

                # ========== ALPACA AUTO-EXIT EXECUTION ==========
                if CPL_AUTO_EXECUTE:
                    try:
                        close_result = alpaca_executor.close_position(call.option_symbol, exit_price)
                        if close_result:
                            logger.info(f"ALPACA EXIT: {call.underlying} {exit_reason} pnl={pnl_pct:+.1f}%")
                            send_alert(f"✅ **ALPACA EXIT** CPL #{buy_call_number}\n{call.underlying} {call.side} ${call.strike}\nExit: ${exit_price:.2f} ({pnl_pct:+.1f}%)")
                        else:
                            logger.warning(f"ALPACA EXIT SKIPPED: {call.underlying} - no position found or close failed")
                    except Exception as e:
                        logger.error(f"ALPACA EXIT ERROR: {e}")

                # GOVERNANCE: Emit exit telemetry
                if _telemetry_bus and GOVERNANCE_ENABLED:
                    final_state = "RELEASE" if exit_reason == "STRUCTURE_BREAK" else (_governance.get_state(dedupe_key).name if _governance and _governance.get_state(dedupe_key) else "OBSERVE")
                    _telemetry_bus.emit_exit(
                        dedupe_key=dedupe_key,
                        exit_reason=exit_reason,
                        exit_price=exit_price,
                        entry_price=entry_price,
                        pnl_pct=pnl_pct,
                        final_state=final_state,
                        peak_price=peak_ref,
                    )
        elif dry_run:
            logger.info(f"CPL SELL (dry-run): {call.underlying} {call.side} @ {call.strike} - {exit_reason} ({pnl_pct:+.1f}%)")

    # Remove closed positions and unregister from governance
    for key in to_close:
        # GOVERNANCE: Unregister position
        if _governance and GOVERNANCE_ENABLED:
            _governance.unregister_position(key)
        del _open_positions[key]

    return sells


class JobsDayCPL:
    """
    Convexity Proof Layer: rank instruments, scan chains, build atomic calls,
    dedupe, and broadcast to Telegram.

    For 0DTE trading, uses today's date by default to scan current-day expirations.
    """

    def __init__(
        self,
        event_date: Optional[str] = None,
        watchlist: Optional[List[str]] = None,
    ):
        # Always use fresh today's date if not specified (handles overnight runs)
        self.event_date = event_date or get_todays_expiry_date()
        self.watchlist = watchlist or CPL_WATCHLIST

    def run(
        self,
        broadcast: bool = False,
        dry_run: bool = False,
        untruncated_tails: bool = False,
        high_hitters_batch: int = 0,
    ) -> List[JobsDayCall]:
        """
        FIX 1: Fetch chains, generate calls until TARGET_BUY_CALLS reached.
        When untruncated_tails: sequential only (max 1 open), target=1 when flat, cooldown=0.
        When high_hitters_batch=N: emit top N 20X/4X BUYs to Telegram only (no position tracking).
        """
        global _call_number_counter, _direction_lock, _sent_calls, _cooldown_tracker, _v6_state

        # Daily reset of direction lock and memory-leak-prone structures
        today = datetime.now().strftime("%Y-%m-%d")
        if _direction_lock.get("_last_reset") != today:
            _direction_lock.clear()
            _direction_lock["_last_reset"] = today
            # FIX: Clear _sent_calls and _cooldown_tracker daily to prevent memory leak
            _sent_calls.clear()
            _cooldown_tracker.clear()
            logger.info(f"CPL_DIRECTION_COOLDOWN: Daily reset for {today} (cleared direction locks, _sent_calls, _cooldown_tracker)")

        # ╔══════════════════════════════════════════════════════════════════════════════╗
        # ║                      V6 SNIPER: DAILY RESET + TRADE CHECK                    ║
        # ╚══════════════════════════════════════════════════════════════════════════════╝
        if _v6_state["last_reset_date"] != today:
            _v6_state = {
                "daily_direction": None,
                "direction_set": False,
                "direction_set_time": None,
                "trade_complete": False,
                "trade_completed_time": None,
                "last_reset_date": today,
            }
            logger.info(f"V6_SNIPER_RESET: New day {today} - direction unlocked, ready for ONE trade")
            # Clean up old persistence flags (keep only today's)
            import glob as _glob
            today_flag = _get_trade_complete_flag_path()
            for old_flag in _glob.glob("/tmp/wsb_snake_trade_complete_*.flag"):
                if old_flag != today_flag:
                    try:
                        os.remove(old_flag)
                        logger.debug(f"Cleaned old flag: {old_flag}")
                    except Exception:
                        pass

        # V6 PERSISTENCE RECOVERY: Check if trade was completed before a restart
        if not _v6_state["trade_complete"] and _check_persisted_trade_complete():
            _v6_state["trade_complete"] = True
            _v6_state["direction_set"] = True
            logger.info("V6_PERSISTENCE_RESTORED: trade_complete recovered from flag file. Done for today.")

        # V6 RULE 2: ONE TRADE THEN DONE - If trade already executed today, STOP
        if _v6_state["trade_complete"]:
            completed_time = _v6_state.get("trade_completed_time", "unknown")
            logger.info(f"V6_SNIPER_DONE: Trade already completed at {completed_time}. No more scans until tomorrow.")
            # Still check for exits on existing position
            sell_calls = _check_exits_and_emit_sells(broadcast, dry_run, untruncated_tails)
            return sell_calls

        # Always refresh to today's date on each run (handles overnight/multi-day runs)
        self.event_date = get_todays_expiry_date()
        logger.debug(f"CPL scanning for expiry: {self.event_date}")

        # Update Opening Range for SPY/QQQ (after 9:35 AM)
        _update_opening_range()

        # HYDRA INTELLIGENCE STATUS
        try:
            hydra = get_hydra_intel()
            logger.info(
                f"HYDRA_STATUS: dir={hydra.direction} regime={hydra.regime} "
                f"blowup={hydra.blowup_probability}% gex_regime={hydra.gex_regime} "
                f"gex_flip_dist={hydra.gex_flip_distance_pct:.2f}% flow={hydra.flow_bias} "
                f"connected={hydra.connected}"
            )
            if not hydra.connected:
                logger.warning("HYDRA_DISCONNECTED: Trading may be limited without intelligence")
        except Exception as e:
            logger.warning(f"HYDRA status check failed: {e}")

        # SNIPER MODE: Session, Position, and P&L checks
        import os
        import requests as req
        from zoneinfo import ZoneInfo

        # BEAST MODE: Session halt REMOVED - let it hunt all day
        # Kill switch (+$2,500 / -$500) still active below
        et_now = datetime.now(ZoneInfo("America/New_York"))
        logger.debug(f"CPL_BEAST_MODE: Hunting at {et_now.hour}:{et_now.minute:02d} ET")

        # SNIPER COOLDOWN: Prevent API lag race condition (March 3 bug)
        if _sniper_state["last_trade_time"] is not None:
            elapsed = (datetime.now(timezone.utc) - _sniper_state["last_trade_time"]).total_seconds()
            if elapsed < SNIPER_COOLDOWN_SECONDS:
                remaining = int(SNIPER_COOLDOWN_SECONDS - elapsed)
                logger.info(f"SNIPER_COOLDOWN: {remaining}s remaining. No new trades until cooldown expires.")
                return []

        # Max 1 position check
        try:
            pos_resp = req.get(
                f"{os.environ.get('ALPACA_BASE_URL', '')}/v2/positions",
                headers={
                    "APCA-API-KEY-ID": os.environ.get("ALPACA_API_KEY", ""),
                    "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", "")
                },
                timeout=5
            )
            if pos_resp.status_code == 200:
                positions = pos_resp.json()
                if len(positions) >= MAX_OPEN_POSITIONS:
                    logger.info(f"SNIPER_POSITION_CAP: {len(positions)}/{MAX_OPEN_POSITIONS} open. Waiting for exit.")
                    sell_calls = _check_exits_and_emit_sells(broadcast, dry_run, untruncated_tails)
                    return sell_calls
        except Exception as e:
            logger.warning(f"Position cap check failed: {e}")

        # Daily P&L check (belt + suspenders)
        try:
            acct_resp = req.get(
                f"{os.environ.get('ALPACA_BASE_URL', '')}/v2/account",
                headers={
                    "APCA-API-KEY-ID": os.environ.get("ALPACA_API_KEY", ""),
                    "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", "")
                },
                timeout=5
            )
            if acct_resp.status_code == 200:
                acct = acct_resp.json()
                daily_pnl = float(acct.get("portfolio_value", 0)) - float(acct.get("last_equity", 0))
                if daily_pnl >= DAILY_PROFIT_TARGET:
                    logger.info(f"CPL_SNIPER_HALT: +{daily_pnl:,.2f} target hit. No more scans.")
                    return []
                if daily_pnl <= DAILY_MAX_LOSS:
                    logger.info(f"CPL_SNIPER_HALT: {daily_pnl:,.2f} loss limit. No more scans.")
                    return []
                logger.info(f"CPL_SNIPER_PNL: {daily_pnl:+,.2f} (target: +{DAILY_PROFIT_TARGET} | floor: {DAILY_MAX_LOSS})")
        except Exception as e:
            logger.warning(f"CPL daily P&L check failed: {e}")

        # ╔══════════════════════════════════════════════════════════════════════════════╗
        # ║                  V6 SNIPER: DETERMINE DAILY DIRECTION                        ║
        # ╚══════════════════════════════════════════════════════════════════════════════╝
        # Rule 1: ONE DIRECTION PER DAY - wait for first 15 min, then lock direction
        v6_direction = _determine_v6_direction()
        if v6_direction is None:
            logger.info("V6_SNIPER_WAIT: Direction not yet determined. Waiting for SPY to show bias.")
            return []
        logger.info(f"V6_SNIPER_DIRECTION: {v6_direction}S ONLY today")

        untruncated_tails = untruncated_tails or UNTRUNCATED_TAILS
        generated: List[JobsDayCall] = []
        candidates: List[JobsDayCall] = []
        window = _window()

        # FIX 2: First check for exits on existing positions and emit SELLs (skip in high-hitters batch)
        if high_hitters_batch <= 0:
            sell_calls = _check_exits_and_emit_sells(broadcast, dry_run, untruncated_tails)
            generated.extend(sell_calls)

        # Collect all candidates first
        for ticker in self.watchlist:
            # NOTE: V7 was disabled - SPY 0DTE now handled by CPL with HYDRA gates
            # (Block removed 2026-03-04 - CPL trades SPY with full HYDRA validation)
            price = _get_spot_price(ticker)
            if not price or price <= 0:
                logger.debug(f"CPL: no price for {ticker}, skip")
                continue

            try:
                chain = polygon_options.get_chain_for_expiration(
                    ticker, price, self.event_date, strike_range=12  # Wide range for more candidates
                )
            except Exception as e:
                logger.warning(f"CPL: chain failed {ticker}: {e}")
                continue

            calls = chain.get("calls") or []
            puts = chain.get("puts") or []

            # V6 SNIPER: Only process the locked direction
            # Process calls and puts
            for side, contracts, regime in [
                ("CALL", calls, "RISK_ON"),
                ("PUT", puts, "RISK_OFF"),
            ]:
                # V6: Skip if not the locked direction
                if v6_direction and side != v6_direction:
                    continue
                if side == "CALL":
                    atm_list = [c for c in contracts if c.get("strike") and abs(c["strike"] - price) < 3]
                    otm_list = [c for c in contracts if c.get("strike") and c["strike"] > price]
                    otm_list.sort(key=lambda x: x["strike"])
                else:
                    atm_list = [p for p in contracts if p.get("strike") and abs(p["strike"] - price) < 3]
                    otm_list = [p for p in contracts if p.get("strike") and p["strike"] < price]
                    otm_list.sort(key=lambda x: -x["strike"])

                # Scan ATM + OTM-1 + OTM-2 + OTM-3
                all_candidates = atm_list[:2] + otm_list[:3]

                for c in all_candidates:
                    strike = c.get("strike") or round(price)

                    # FIX 3: Liquidity gate FIRST (before affordability)
                    is_liquid, reject_reason = _check_liquidity(c, ticker, side, strike)
                    if not is_liquid:
                        logger.info(reject_reason)
                        continue

                    # SMART ENTRY: Momentum validation (only buy CALLS in uptrend, PUTS in downtrend)
                    is_momentum_ok, momentum_conf, momentum_reason = _check_entry_quality(ticker, side, price)
                    if not is_momentum_ok:
                        logger.info(momentum_reason)
                        continue

                    # Affordability: untruncated = cost <= session_balance; else cost <= MAX_COST_PER_CONTRACT
                    max_cap = _session_balance if untruncated_tails else None
                    if not _is_affordable(c, max_cap_override=max_cap):
                        logger.debug(f"CPL: {ticker} {side} @ {strike} skip - not affordable")
                        continue

                    entry_price = c.get("ask") or c.get("last_price") or 0
                    if entry_price <= 0:
                        continue

                    opt_symbol = c.get("symbol")

                    job_call = JobsDayCall.create(
                        underlying=ticker,
                        side=side,
                        strike=strike,
                        expiry_date=self.event_date,
                        dte=0,
                        entry_price=entry_price,
                        stop_pct=-15,
                        tp_pcts=[25, 50, 100],
                        option_symbol=opt_symbol,
                        regime=regime,
                        confidence=momentum_conf,  # REAL confidence from momentum check
                        reasons=[
                            f"CPL {regime}",
                            f"Momentum: {momentum_conf:.0f}%",
                        ],
                        window=window,
                    )
                    job_call.spot_at_alert = price
                    job_call.momentum_confidence = momentum_conf  # Store for Alpaca
                    job_call.event_tier = _event_tier(price, strike, side, entry_price, momentum_conf)
                    candidates.append(job_call)

        # Prioritize 2X / 4X / 20X events: sort by tier (20X first, then 4X, then 2X)
        candidates.sort(key=lambda c: (_tier_rank(c.event_tier), -c.confidence))

        # High hitters batch: top N 20X/4X only, emit to Telegram (no position tracking)
        if high_hitters_batch > 0:
            high_only = [c for c in candidates if (c.event_tier or "").upper() in ("20X", "4X")]
            for i, call in enumerate(high_only[:high_hitters_batch], 1):
                ready, _ = _is_execution_ready(call)
                if not ready:
                    continue
                generated.append(call)
                if broadcast and not dry_run:
                    _call_number_counter += 1
                    msg = format_jobs_day_call(
                        call, _call_number_counter, test_mode=False,
                        untruncated_tails=False,
                        trade_number=None, session_balance=None, contracts=1,
                        message_sequence=_call_number_counter,
                    )
                    send_alert(f"🔥 HIGH HITTER {i}/{min(len(high_only), high_hitters_batch)} (20X/4X only)\n\n{msg}")
                    logger.info(f"HIGH HITTER {i}: {call.underlying} {call.side} @ {call.strike} ({call.event_tier})")

                    # ========== ALPACA AUTO-EXECUTION FOR HIGH HITTERS ==========
                    if CPL_AUTO_EXECUTE:
                        try:
                            option_premium = call.entry_trigger.get("price", 0)
                            direction = "long"
                            target_price = option_premium * 1.25
                            stop_loss = option_premium * 0.88
                            real_confidence = getattr(call, 'momentum_confidence', 50)

                            alpaca_pos = alpaca_executor.execute_scalp_entry(
                                underlying=call.underlying,
                                direction=direction,
                                entry_price=option_premium,
                                target_price=target_price,
                                stop_loss=stop_loss,
                                confidence=real_confidence,
                                pattern=f"HIGH_HITTER_{call.side}",
                                engine=TradingEngine.SCALPER,
                                strike_override=call.strike,
                                option_symbol_override=call.option_symbol,
                                option_type_override=call.side.lower(),
                            )
                            if alpaca_pos:
                                _sniper_state["last_trade_time"] = datetime.now(timezone.utc)
                                logger.info(f"ALPACA EXECUTED: {call.underlying} {call.side} ${call.strike} qty={alpaca_pos.qty}")
                                logger.info(f"SNIPER_COOLDOWN_SET: {SNIPER_COOLDOWN_SECONDS}s cooldown started")
                                send_alert(f"✅ **ALPACA EXECUTED** HIGH HITTER #{_call_number_counter}\n{call.underlying} {call.side} ${call.strike}\nOption: {alpaca_pos.option_symbol}")

                                # V6 SNIPER: ONE TRADE THEN DONE
                                _v6_state["trade_complete"] = True
                                _v6_state["trade_completed_time"] = datetime.now(timezone.utc).isoformat()
                                logger.info(f"V6_SNIPER_TRADE_COMPLETE: HIGH HITTER executed. No more scans until tomorrow.")
                                _persist_trade_complete()

                                # SECOND BRAIN: Report HIGH HITTER trade
                                _report_trade_to_second_brain(
                                    ticker=call.underlying,
                                    side=call.side,
                                    strike=call.strike,
                                    entry_price=option_premium,
                                    option_symbol=alpaca_pos.option_symbol,
                                    confidence=real_confidence,
                                    signals=["HIGH_HITTER", f"V6_{_v6_state['daily_direction']}"],
                                    note=f"HIGH_HITTER executed at {real_confidence:.0f}% confidence"
                                )
                            else:
                                logger.warning(f"ALPACA SKIPPED: {call.underlying} (max positions or limit)")
                        except Exception as e:
                            logger.error(f"ALPACA EXECUTION ERROR: {e}")
            return generated

        # Sequential (untruncated): emit at most 1 BUY and only when flat
        effective_target = (1 if len(_open_positions) == 0 else 0) if untruncated_tails else TARGET_BUY_CALLS
        # Cooldown: 0 when untruncated; 25 min in power hour (more opportunities); else 45
        effective_cooldown_minutes = 0 if untruncated_tails else (25 if CPL_POWER_HOUR else COOLDOWN_MINUTES)

        calls_per_underlying: Dict[str, int] = {}
        count = 0
        skipped_dedupe = 0
        skipped_cooldown = 0
        skipped_cap = 0
        seen_visual: Set[Tuple[str, str, float, str, float]] = set()

        for pass_num in (0, 1):
            if count >= effective_target:
                break
            for call in candidates:
                if count >= effective_target:
                    break

                ticker = (call.underlying or "").upper()
                direction = call.side.upper()  # "CALL" or "PUT"

                # DIRECTION COOLDOWN: Block opposite direction for 10 minutes after entry
                if ticker in _direction_lock and isinstance(_direction_lock[ticker], dict):
                    lock_info = _direction_lock[ticker]
                    locked_side = lock_info.get("side", "")
                    lock_time = lock_info.get("time")
                    if locked_side and lock_time and locked_side != direction:
                        elapsed_min = (datetime.now(timezone.utc) - lock_time).total_seconds() / 60
                        if elapsed_min < 10:
                            logger.warning(f"CPL_DIRECTION_COOLDOWN: {ticker} locked to {locked_side} ({elapsed_min:.0f}min ago). Blocking {direction}. Unlocks in {10 - elapsed_min:.0f}min.")
                            continue
                        else:
                            logger.info(f"CPL_DIRECTION_UNLOCK: {ticker} cooldown expired ({elapsed_min:.0f}min). Allowing {direction}.")

                # Diversity: skip when untruncated (single best setup). Power hour: allow 3 per ticker.
                max_per_underlying = 3 if CPL_POWER_HOUR else MAX_CALLS_PER_UNDERLYING
                if not untruncated_tails and DIVERSITY_MODE == "PROOF":
                    current = calls_per_underlying.get(ticker, 0)
                    if current != pass_num:
                        continue
                    if current >= max_per_underlying:
                        skipped_cap += 1
                        logger.info(f"UNDERLYING_CAP_REJECT: {ticker}")
                        continue

                vkey = _visual_key(call)
                if vkey in seen_visual:
                    logger.debug(f"UNIQUENESS_REJECT: duplicate visual key {vkey}")
                    continue

                if call.dedupe_key in _sent_calls:
                    skipped_dedupe += 1
                    logger.debug(f"DEDUPE_REJECT: {call.dedupe_key}")
                    continue
                if cpl_call_exists(call.dedupe_key):
                    _sent_calls.add(call.dedupe_key)
                    skipped_dedupe += 1
                    logger.debug(f"DEDUPE_REJECT (DB): {call.dedupe_key}")
                    continue

                # Cooldown: use effective (0 when untruncated)
                if _is_cooldown_active(call.underlying, call.side, call.expiry_date, effective_cooldown_minutes=effective_cooldown_minutes):
                    skipped_cooldown += 1
                    logger.info(f"COOLDOWN_REJECT: {call.underlying} {call.side} {call.expiry_date}")
                    continue

                # Valid candidate - emit
                count += 1
                _call_number_counter += 1
                call_number = _call_number_counter

                seen_visual.add(vkey)
                if not untruncated_tails and DIVERSITY_MODE == "PROOF":
                    calls_per_underlying[ticker] = calls_per_underlying.get(ticker, 0) + 1

                generated.append(call)
                full_json = json.dumps(call.to_dict(), default=str)

                if not dry_run:
                    save_cpl_call(
                        call_id=call.call_id,
                        timestamp_et=call.timestamp_et,
                        ticker=call.underlying,
                        side=call.side,
                        strike=call.strike,
                        expiry=call.expiry_date,
                        dedupe_key=call.dedupe_key,
                        regime=call.regime,
                        confidence=call.confidence,
                        alerted_at=datetime.now(timezone.utc).isoformat() if broadcast else None,
                        full_json=full_json,
                    )

                if broadcast and not dry_run:
                    ready, reject_reason = _is_execution_ready(call)
                    if not ready:
                        logger.warning(f"EXECUTION_INCOMPLETE_REJECT: {reject_reason} | {call.dedupe_key}")
                        continue
                    if not untruncated_tails and DIVERSITY_MODE == "PROOF":
                        if calls_per_underlying.get(ticker, 0) > max_per_underlying:
                            logger.warning("HARD_FAIL_DIVERSITY: per-underlying cap violated; not sending")
                            continue
                    # Session header and sequencing: send once per run when first message goes out
                    global _message_sequence, _last_session_date
                    try:
                        import zoneinfo
                        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
                        today_et = et.strftime("%Y-%m-%d")
                        if _last_session_date != today_et or _message_sequence == 0:
                            _message_sequence += 1
                            tiers = list(dict.fromkeys([c.event_tier for c in candidates if getattr(c, "event_tier", None)][:3]))
                            header = format_session_header(today_et, _message_sequence, tiers or ["2X", "4X", "20X"])
                            send_alert(header)
                            _last_session_date = today_et
                    except Exception:
                        pass
                    _message_sequence += 1
                    # Paper trader: contracts from balance (compound)
                    contracts = max(1, int(_session_balance / 250.0)) if untruncated_tails else 1
                    trade_num = (_trade_counter + 1) if untruncated_tails else call_number
                    msg = format_jobs_day_call(
                        call, call_number, test_mode=False,
                        untruncated_tails=untruncated_tails,
                        trade_number=trade_num,
                        session_balance=_session_balance if untruncated_tails else None,
                        contracts=contracts,
                        message_sequence=_message_sequence,
                    )
                    send_alert(msg)
                    logger.info(f"CPL BUY broadcast #{call_number}: {call.underlying} {call.side} {call.strike}" + (f" x{contracts}" if untruncated_tails else ""))
                    _set_cooldown(call.underlying, call.side, call.expiry_date)
                    _open_positions[call.dedupe_key] = {
                        "entry_price": call.entry_trigger.get("price", 0),
                        "entry_time": datetime.now(timezone.utc),
                        "call": call,
                        "buy_call_number": call_number,
                        "contracts": contracts,
                    }

                    # ========== ALPACA AUTO-EXECUTION ==========
                    logger.info(f"CPL_AUTO_EXECUTE check: {CPL_AUTO_EXECUTE}")
                    if CPL_AUTO_EXECUTE:
                        logger.info(f"ALPACA: Attempting execution for {call.underlying} {call.side} ${call.strike}")
                        logger.info(f"ALPACA: option_symbol={call.option_symbol}, entry_trigger={call.entry_trigger}")
                        try:
                            # Log HYDRA context for this trade
                            try:
                                hydra = get_hydra_intel()
                                logger.info(
                                    f"ALPACA_HYDRA_CONTEXT: dir={hydra.direction} regime={hydra.regime} "
                                    f"blowup={hydra.blowup_probability}% flow={hydra.flow_bias}"
                                )
                            except Exception:
                                pass

                            option_premium = call.entry_trigger.get("price", 0)
                            logger.info(f"ALPACA: option_premium=${option_premium:.2f}")
                            # Direction: Always "long" since we're BUYING options (calls or puts)
                            # "short" would mean shorting, but we're buying to open
                            # The option_type (call/put) determines bullish vs bearish bet
                            direction = "long"
                            # Target/stop based on option premium (not underlying price!)
                            target_price = option_premium * 1.25  # +25% target (wider for 0DTE)
                            stop_loss = option_premium * 0.88     # -12% stop (slightly wider)

                            # Real confidence from momentum check (not fake 85%)
                            real_confidence = getattr(call, 'momentum_confidence', 50)

                            # FIX: Pass strike, option_symbol, and option_type directly from CPL
                            # Previously we passed option_premium as entry_price, which the executor
                            # incorrectly used to calculate strike (e.g., $1.43 -> strike $1!)
                            alpaca_pos = alpaca_executor.execute_scalp_entry(
                                underlying=call.underlying,
                                direction=direction,
                                entry_price=option_premium,  # Still needed for validation
                                target_price=target_price,
                                stop_loss=stop_loss,
                                confidence=real_confidence,  # REAL confidence from momentum analysis
                                pattern=f"CPL_{call.side}",
                                engine=TradingEngine.SCALPER,
                                strike_override=call.strike,  # Use CPL's strike directly
                                option_symbol_override=call.option_symbol,  # Use CPL's option symbol directly
                                option_type_override=call.side.lower(),  # "call" or "put" from CPL
                            )
                            if alpaca_pos:
                                _sniper_state["last_trade_time"] = datetime.now(timezone.utc)
                                logger.info(f"ALPACA EXECUTED: {call.underlying} {call.side} ${call.strike} qty={alpaca_pos.qty}")
                                logger.info(f"SNIPER_COOLDOWN_SET: {SNIPER_COOLDOWN_SECONDS}s cooldown started")
                                send_alert(f"✅ **ALPACA EXECUTED** CPL #{call_number}\n{call.underlying} {call.side} ${call.strike}\nOption: {alpaca_pos.option_symbol}")
                                # Lock direction: 30-minute cooldown, not full day
                                _direction_lock[ticker] = {"side": call.side.upper(), "time": datetime.now(timezone.utc)}
                                logger.info(f"CPL_DIRECTION_SET: {ticker} locked to {call.side.upper()} for 30min cooldown")

                                # ═══════════════════════════════════════════════════════════════
                                # V6 SNIPER: ONE TRADE THEN DONE - Mark trade complete
                                # ═══════════════════════════════════════════════════════════════
                                _v6_state["trade_complete"] = True
                                _v6_state["trade_completed_time"] = datetime.now(timezone.utc).isoformat()
                                logger.info(f"V6_SNIPER_TRADE_COMPLETE: Trade executed. No more scans until tomorrow.")
                                _persist_trade_complete()
                                send_alert(f"🎯 V6 SNIPER COMPLETE\n{call.underlying} {call.side} ${call.strike}\nDone for today. Let execution layer manage the position.")

                                # ═══════════════════════════════════════════════════════════════
                                # SECOND BRAIN: Report trade for persistent memory
                                # ═══════════════════════════════════════════════════════════════
                                _report_trade_to_second_brain(
                                    ticker=call.underlying,
                                    side=call.side,
                                    strike=call.strike,
                                    entry_price=call.entry_trigger.get("price", 0),
                                    option_symbol=alpaca_pos.option_symbol,
                                    confidence=real_confidence,
                                    signals=[f"V6_SNIPER_{_v6_state['daily_direction']}"],
                                    note=f"V6_SNIPER executed at {real_confidence:.0f}% confidence"
                                )
                            else:
                                logger.warning(f"ALPACA SKIPPED: {call.underlying} (max positions or limit)")
                        except Exception as e:
                            logger.error(f"ALPACA EXECUTION ERROR: {e}")
                    
                    # GOVERNANCE: Register position for tracking
                    if _governance and GOVERNANCE_ENABLED:
                        entry_price = call.entry_trigger.get("price", 0)
                        _governance.register_position(call.dedupe_key, entry_price)
                        # Emit entry telemetry
                        if _telemetry_bus:
                            _telemetry_bus.emit_entry(
                                dedupe_key=call.dedupe_key,
                                ticker=call.underlying,
                                side=call.side,
                                strike=call.strike,
                                entry_price=entry_price,
                            )

                _sent_calls.add(call.dedupe_key)

        # Summary logging
        if skipped_dedupe:
            logger.info(f"CPL: {skipped_dedupe} calls skipped (dedupe)")
        if skipped_cooldown:
            logger.info(f"CPL: {skipped_cooldown} calls skipped (cooldown)")
        if skipped_cap:
            logger.info(f"CPL: {skipped_cap} calls skipped (underlying cap)")

        target_met = "YES" if count >= effective_target else "NO"
        logger.info(f"CPL run: {count} BUY broadcast, {len(sell_calls)} SELL, target={effective_target} met={target_met}")

        return generated
