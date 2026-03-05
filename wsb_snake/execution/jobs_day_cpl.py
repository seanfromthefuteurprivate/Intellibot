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
# Index: SPY, QQQ, IWM, DIA | Vol: VXX, UVXY | Rates: TLT, IEF | Dollar: UUP | Metals: GLD, SLV
# Crypto beta: MSTR, COIN, MARA, RIOT | AI/mega: NVDA, TSLA, AAPL, AMZN, META, GOOGL, MSFT, AMD
# Sectors: XLF, ITB, XHB, XLY, XLV, GDX
CPL_WATCHLIST = [
    "SPY", "QQQ", "DIA",                          # core index 0DTE (IWM removed - illiquid)
    "VXX", "UVXY",                                # panic meter (VIX products)
    "TLT", "IEF", "XLF",                          # rates, intermediate Treasuries, financials
    "UUP", "GLD", "SLV", "GDX",                   # dollar, metals, gold miners
    "MSTR", "COIN", "MARA", "RIOT",               # crypto beta (WSB focus)
    "NVDA", "TSLA", "AAPL", "AMZN", "META", "GOOGL", "MSFT", "AMD",  # AI / mega-cap
    "ITB", "XHB", "XLY", "XLV",                   # homebuilders, consumer, healthcare
    "NBIS", "RKLB", "ASTS", "LUNR", "PL", "ONDS", "SLS",  # WSB 0DTE / momentum (NBIS spot vs strike context)
]

# FIX 1: Target forcing function
TARGET_BUY_CALLS = 3

# SNIPER MODE CONFIG
SNIPER_CAPITAL = 2500               # Position sizing base (pretend cap)
MAX_OPEN_POSITIONS = 1              # One shot at a time
DAILY_PROFIT_TARGET = 10000         # +$10,000 = halt (Beast Mode)
DAILY_MAX_LOSS = -750               # -$750 = halt (wider floor)
SNIPER_COOLDOWN_SECONDS = 300       # 5-min cooldown after ANY trade (prevents API lag race)

# Local tracking to prevent API lag race condition (mutable dict for nested scope)
_sniper_state = {"last_trade_time": None}

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
    ║                           BEAST MODE V4.0                                     ║
    ║                   13-SIGNAL CONVICTION STACKING SYSTEM                        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝

    CONVICTION SIGNALS (each adds +1, some can add -1 penalty):
    ┌────────────────────────────────────────────────────────────────────────────┐
    │ 1.  HYDRA direction aligned                                                 │
    │ 2.  Sweep direction aligned (flow_sweep_direction)                         │
    │ 3.  Near dark pool level (dp_support for CALL, dp_resistance for PUT)      │
    │ 4.  Volume ratio > 1.5x                                                     │
    │ 5.  GEX regime favorable (NEGATIVE = trending)                             │
    │ 6.  Momentum ACCELERATING (candle SIZE increasing, not just direction)     │
    │ 7.  Whale premium present ($500K+ in direction)                            │
    │ 8.  Charm flow favorable (afternoon theta alignment)                       │
    │ 9.  Time window optimal (power hour / morning momentum)                    │
    │ 10. Predator Vision (AI pattern recognition)                               │
    │ 11. Opening Range Breakout (SPY/QQQ > OR high = CALL, < OR low = PUT)     │
    │ 12. Pre-market Bias (+1 if confirms, -1 if conflicts)                      │
    │ 13. GEX proximity favorable (near positive GEX for CALL, negative for PUT) │
    └────────────────────────────────────────────────────────────────────────────┘

    MINIMUM conviction = 5 to trade (out of 13)
    5-7  = base position size (confidence 55-69)
    8-10 = 1.5x position size (confidence 70-84)
    11-13 = FULL SEND max $2,500 (confidence 85-95)

    HARD GATES (instant reject, no conviction points):
    - Polygon API unhealthy
    - HYDRA disconnected/stale
    - Direction conflict
    - Blowup > 70%
    - GEX flip < 1%
    - Regime CHOPPY/UNKNOWN
    - Data unavailable
    - Strong momentum against

    Returns: (is_valid, confidence_score, reason)
    """
    side_upper = side.upper()
    from zoneinfo import ZoneInfo
    et_now = datetime.now(ZoneInfo("America/New_York"))

    # ════════════════════════════════════════════════════════════════════════════
    # HARD GATES - Instant rejection, no conviction calculation
    # ════════════════════════════════════════════════════════════════════════════

    # Gate 0: Polygon health
    try:
        from wsb_snake.utils.polygon_health import polygon_health_check
        is_healthy, health_reason = polygon_health_check()
        if not is_healthy:
            logger.error(f"BEAST_REJECT: {ticker} {side_upper} - Polygon unhealthy: {health_reason}")
            return False, 0, f"HARD_GATE_POLYGON: {health_reason}"
    except ImportError:
        pass

    # Gate 1: HYDRA connection
    try:
        hydra = get_hydra_intel()
    except Exception as e:
        logger.warning(f"BEAST_REJECT: {ticker} {side_upper} - HYDRA unavailable: {e}")
        return False, 0, "HARD_GATE_HYDRA: Intelligence unavailable"

    if not hydra.connected:
        logger.info(f"BEAST_REJECT: {ticker} {side_upper} - HYDRA disconnected")
        return False, 0, "HARD_GATE_HYDRA: Disconnected"

    if hydra.is_stale():
        logger.info(f"BEAST_REJECT: {ticker} {side_upper} - HYDRA stale >3min")
        return False, 0, "HARD_GATE_HYDRA: Stale data"

    # Gate 2: Direction conflict (hard block opposite direction)
    # NOTE: NEUTRAL is no longer a hard block - allows trading with reduced conviction
    # when HYDRA has no clear directional signal (common during low-volume periods)
    if hydra.direction == "NEUTRAL":
        logger.info(f"BEAST_WARNING: {ticker} {side_upper} - HYDRA NEUTRAL (proceeding with caution)")
        # Don't block - just log warning and continue to conviction scoring

    if side_upper == "CALL" and hydra.direction == "BEARISH":
        logger.info(f"BEAST_REJECT: {ticker} CALL blocked - HYDRA BEARISH")
        return False, 0, "HARD_GATE_DIRECTION: CALL in BEARISH"

    if side_upper == "PUT" and hydra.direction == "BULLISH":
        logger.info(f"BEAST_REJECT: {ticker} PUT blocked - HYDRA BULLISH")
        return False, 0, "HARD_GATE_DIRECTION: PUT in BULLISH"

    # Gate 3: Blowup probability
    if hydra.blowup_probability > 70:
        logger.info(f"BEAST_REJECT: {ticker} {side_upper} - blowup {hydra.blowup_probability}%")
        return False, 0, f"HARD_GATE_BLOWUP: {hydra.blowup_probability}%"

    # Gate 4: GEX flip proximity
    # NOTE: Skip gate if GEX data appears missing (flip_point=0 or flip_distance=0 indicates unavailable data)
    # Only apply gate when we have valid GEX data (flip_point > 0)
    if hydra.gex_flip_point > 0 and hydra.gex_flip_distance_pct < 1.0:
        logger.info(f"BEAST_REJECT: {ticker} {side_upper} - GEX flip {hydra.gex_flip_distance_pct:.2f}%")
        return False, 0, f"HARD_GATE_GEX_FLIP: {hydra.gex_flip_distance_pct:.1f}%"
    elif hydra.gex_flip_point == 0:
        logger.debug(f"BEAST_SKIP_GEX_GATE: {ticker} - GEX data unavailable (flip_point=0)")

    # Gate 5: Regime (hard block CHOPPY only, UNKNOWN = missing data)
    # NOTE: UNKNOWN regime is no longer a hard block - indicates HYDRA data unavailable
    # CHOPPY is a valid signal to stay out (high whipsaw risk)
    if hydra.regime == "CHOPPY":
        logger.info(f"BEAST_REJECT: {ticker} {side_upper} - regime CHOPPY")
        return False, 0, "HARD_GATE_REGIME: CHOPPY"
    elif hydra.regime == "UNKNOWN":
        logger.info(f"BEAST_WARNING: {ticker} {side_upper} - regime UNKNOWN (proceeding with caution)")

    # Gate 6: Data availability
    # NOTE: Reduced minimum from 5 to 2 bars - allows trading with limited data
    # when Polygon has temporary gaps. 2 bars is minimum for price change calculation.
    try:
        bars = polygon_enhanced.get_intraday_bars(ticker, timespan="minute", multiplier=5, limit=8)
    except Exception as e:
        logger.warning(f"BEAST_REJECT: {ticker} {side_upper} - bars failed: {e}")
        return False, 0, "HARD_GATE_DATA: Bars unavailable"

    if not bars or len(bars) < 2:
        logger.info(f"BEAST_REJECT: {ticker} {side_upper} - insufficient bars (got {len(bars) if bars else 0})")
        return False, 0, "HARD_GATE_DATA: Insufficient bars"

    # Extract price/volume data (use available bars, up to 5)
    closes = []
    volumes = []
    for b in bars[:min(5, len(bars))]:
        c = b.get('close') or b.get('c')
        v = b.get('volume') or b.get('v') or 0
        if c is None:
            return False, 0, "HARD_GATE_DATA: Missing close"
        closes.append(c)
        volumes.append(v)

    price_change_pct = (closes[0] - closes[-1]) / closes[-1] * 100 if closes[-1] > 0 else 0
    recent_vol_avg = sum(volumes[:min(2, len(volumes))]) / min(2, len(volumes))
    older_vol_avg = sum(volumes[2:]) / len(volumes[2:]) if len(volumes) > 2 else recent_vol_avg
    volume_ratio = recent_vol_avg / older_vol_avg if older_vol_avg > 0 else 1.0

    # Gate 7: Basic momentum alignment (must be in right direction)
    if side_upper == "CALL" and price_change_pct < -0.5:
        logger.info(f"BEAST_REJECT: {ticker} CALL - strong downtrend {price_change_pct:+.2f}%")
        return False, 0, f"HARD_GATE_MOMENTUM: Wrong direction {price_change_pct:+.2f}%"
    if side_upper == "PUT" and price_change_pct > 0.5:
        logger.info(f"BEAST_REJECT: {ticker} PUT - strong uptrend {price_change_pct:+.2f}%")
        return False, 0, f"HARD_GATE_MOMENTUM: Wrong direction {price_change_pct:+.2f}%"

    # ════════════════════════════════════════════════════════════════════════════
    # CONVICTION STACKING - 13 Signals, need 5+ to trade
    # ════════════════════════════════════════════════════════════════════════════

    conviction = 0
    conviction_details = []

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 1: HYDRA Direction Aligned (+1)
    # ══════════════════════════════════════════════════════════════════════════
    direction_aligned = (
        (side_upper == "CALL" and hydra.direction == "BULLISH") or
        (side_upper == "PUT" and hydra.direction == "BEARISH")
    )
    if direction_aligned:
        conviction += 1
        conviction_details.append("✓ HYDRA_DIR")
    else:
        conviction_details.append("✗ HYDRA_DIR")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 2: Sweep Direction Aligned (+1)
    # ══════════════════════════════════════════════════════════════════════════
    sweep_aligned = (
        (side_upper == "CALL" and hydra.flow_sweep_direction == "CALL_HEAVY") or
        (side_upper == "PUT" and hydra.flow_sweep_direction == "PUT_HEAVY")
    )
    if sweep_aligned:
        conviction += 1
        conviction_details.append("✓ SWEEP")
    else:
        conviction_details.append(f"✗ SWEEP({hydra.flow_sweep_direction})")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 3: Dark Pool Level (+1)
    # ══════════════════════════════════════════════════════════════════════════
    dp_support = hydra.dp_nearest_support or 0
    dp_resistance = hydra.dp_nearest_resistance or 0
    dp_proximity_pct = 0.5  # Within 0.5% of level

    near_dp_level = False
    if side_upper == "CALL" and dp_support > 0:
        dist_to_support = abs(spot - dp_support) / spot * 100
        if dist_to_support < dp_proximity_pct:
            near_dp_level = True
    if side_upper == "PUT" and dp_resistance > 0:
        dist_to_resistance = abs(spot - dp_resistance) / spot * 100
        if dist_to_resistance < dp_proximity_pct:
            near_dp_level = True

    if near_dp_level:
        conviction += 1
        conviction_details.append("✓ DARK_POOL")
    else:
        conviction_details.append("✗ DARK_POOL")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 4: Volume Confirmation (+1) - 1.5x ratio
    # ══════════════════════════════════════════════════════════════════════════
    strong_volume = volume_ratio >= 1.5
    if strong_volume:
        conviction += 1
        conviction_details.append(f"✓ VOL({volume_ratio:.1f}x)")
    else:
        conviction_details.append(f"✗ VOL({volume_ratio:.1f}x)")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 5: GEX Regime Favorable (+1) - NEGATIVE = trending
    # ══════════════════════════════════════════════════════════════════════════
    gex_favorable = hydra.gex_regime == "NEGATIVE"
    if gex_favorable:
        conviction += 1
        conviction_details.append("✓ GEX_NEG")
    else:
        conviction_details.append(f"✗ GEX({hydra.gex_regime})")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 6: Momentum ACCELERATING (+1) - Candles getting BIGGER
    # ══════════════════════════════════════════════════════════════════════════
    # Get last 3 candle sizes: abs(close - open) for each
    # Accelerating = each candle bigger than the previous
    candle_sizes = []
    for b in bars[:3]:
        o = b.get('open') or b.get('o') or 0
        c = b.get('close') or b.get('c') or 0
        candle_sizes.append(abs(c - o))

    # Check acceleration: each candle bigger than previous (most recent first)
    accelerating = False
    if len(candle_sizes) >= 3:
        # bars[0] is most recent, bars[1] is second most recent, etc.
        # Accelerating means: size[0] > size[1] > size[2]
        if candle_sizes[0] > candle_sizes[1] > candle_sizes[2] and candle_sizes[2] > 0:
            accelerating = True

    if accelerating:
        conviction += 1
        conviction_details.append(f"✓ ACCEL({candle_sizes[0]:.2f}>{candle_sizes[1]:.2f}>{candle_sizes[2]:.2f})")
    else:
        conviction_details.append(f"✗ ACCEL(flat)")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 7: Whale Premium Present (+1) - $500K+ in direction
    # ══════════════════════════════════════════════════════════════════════════
    WHALE_THRESHOLD = 500_000
    whale_present = False
    if side_upper == "CALL" and hydra.flow_net_premium_calls >= WHALE_THRESHOLD:
        whale_present = True
    if side_upper == "PUT" and hydra.flow_net_premium_puts >= WHALE_THRESHOLD:
        whale_present = True

    if whale_present:
        conviction += 1
        premium = hydra.flow_net_premium_calls if side_upper == "CALL" else hydra.flow_net_premium_puts
        conviction_details.append(f"✓ WHALE(${premium/1000:.0f}K)")
    else:
        conviction_details.append("✗ WHALE")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 8: Charm Flow Favorable (+1) - Afternoon theta alignment
    # ══════════════════════════════════════════════════════════════════════════
    charm_favorable = False
    charm_flow = hydra.charm_flow_per_hour or 0

    # After 2 PM: negative charm = MM selling puts = favor calls
    # Positive charm = MM selling calls = favor puts
    if et_now.hour >= 14:
        if side_upper == "CALL" and charm_flow < -10000:
            charm_favorable = True
        if side_upper == "PUT" and charm_flow > 10000:
            charm_favorable = True

    if charm_favorable:
        conviction += 1
        conviction_details.append(f"✓ CHARM({charm_flow/1000:.0f}K)")
    else:
        if et_now.hour >= 14:
            conviction_details.append(f"✗ CHARM({charm_flow/1000:.0f}K)")
        else:
            conviction_details.append("- CHARM(pre-2PM)")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 9: Time Window Optimal (+1) - Power hour / morning momentum
    # ══════════════════════════════════════════════════════════════════════════
    optimal_time = False
    current_hour = et_now.hour
    current_minute = et_now.minute

    # Morning momentum: 9:35 - 10:30 (after open chop settles)
    if current_hour == 9 and current_minute >= 35:
        optimal_time = True
    if current_hour == 10 and current_minute <= 30:
        optimal_time = True
    # Power hour: 2:30 - 3:45 (momentum + gamma amplification)
    if current_hour == 14 and current_minute >= 30:
        optimal_time = True
    if current_hour == 15 and current_minute <= 45:
        optimal_time = True

    if optimal_time:
        conviction += 1
        conviction_details.append(f"✓ TIME({current_hour}:{current_minute:02d})")
    else:
        conviction_details.append(f"✗ TIME({current_hour}:{current_minute:02d})")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 10: Predator Vision (+1) - AI pattern recognition
    # ══════════════════════════════════════════════════════════════════════════
    predator_bullish = False
    try:
        from wsb_snake.ai_stack.predator_stack_v2 import get_predator_stack

        # Build signal dict for Predator
        predator_signal = {
            'ticker': ticker,
            'direction': 'BULLISH' if side_upper == 'CALL' else 'BEARISH',
            'price': spot,
        }

        # Convert bars to candles format for Predator
        predator_candles = []
        for b in bars[:5]:
            predator_candles.append({
                'open': b.get('open') or b.get('o'),
                'high': b.get('high') or b.get('h'),
                'low': b.get('low') or b.get('l'),
                'close': b.get('close') or b.get('c'),
                'volume': b.get('volume') or b.get('v') or 0,
            })

        predator_stack = get_predator_stack()
        verdict = predator_stack.analyze(
            signal=predator_signal,
            candles=predator_candles
        )

        if verdict.action == "STRIKE" and verdict.conviction >= 60:
            predator_bullish = True
            conviction += 1
            conviction_details.append(f"✓ PREDATOR({verdict.conviction:.0f}%)")
        else:
            conviction_details.append(f"✗ PREDATOR({verdict.action}:{verdict.conviction:.0f}%)")

    except Exception as e:
        logger.debug(f"Predator Stack unavailable: {e}")
        conviction_details.append("- PREDATOR(n/a)")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 11: Opening Range Breakout (+1) - SPY/QQQ only
    # ══════════════════════════════════════════════════════════════════════════
    or_breakout = False
    if ticker in ["SPY", "QQQ"] and ticker in _opening_range:
        or_data = _opening_range[ticker]
        or_high = or_data.get("high", 0)
        or_low = or_data.get("low", 0)
        or_date = or_data.get("date", "")

        # Only use if captured today
        today_str = et_now.strftime("%Y-%m-%d")
        if or_date == today_str and or_high > 0 and or_low > 0:
            if side_upper == "CALL" and spot > or_high:
                or_breakout = True
                conviction += 1
                conviction_details.append(f"✓ OR_BRK(>{or_high:.2f})")
            elif side_upper == "PUT" and spot < or_low:
                or_breakout = True
                conviction += 1
                conviction_details.append(f"✓ OR_BRK(<{or_low:.2f})")
            else:
                conviction_details.append(f"✗ OR_BRK({or_low:.2f}-{or_high:.2f})")
        else:
            conviction_details.append("- OR_BRK(stale)")
    else:
        conviction_details.append("- OR_BRK(n/a)")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 12: Pre-market Bias (+1 if confirms, -1 if conflicts)
    # ══════════════════════════════════════════════════════════════════════════
    premarket_bias = None
    try:
        with open("/tmp/premarket_bias.txt", "r") as f:
            premarket_bias = f.read().strip().upper()
    except Exception:
        pass

    if premarket_bias in ["BULLISH", "BEARISH", "NEUTRAL"]:
        if side_upper == "CALL" and premarket_bias == "BULLISH":
            conviction += 1
            conviction_details.append("✓ PM_BIAS(BULL)")
        elif side_upper == "PUT" and premarket_bias == "BEARISH":
            conviction += 1
            conviction_details.append("✓ PM_BIAS(BEAR)")
        elif side_upper == "CALL" and premarket_bias == "BEARISH":
            conviction -= 1  # PENALTY
            conviction_details.append("✗ PM_BIAS(BEAR,-1)")
        elif side_upper == "PUT" and premarket_bias == "BULLISH":
            conviction -= 1  # PENALTY
            conviction_details.append("✗ PM_BIAS(BULL,-1)")
        else:
            conviction_details.append(f"- PM_BIAS({premarket_bias})")
    else:
        conviction_details.append("- PM_BIAS(n/a)")

    # ══════════════════════════════════════════════════════════════════════════
    # SIGNAL 13: GEX Proximity Favorable (+1)
    # For CALLS: price near positive GEX level (support)
    # For PUTS: price near negative GEX level (resistance)
    # ══════════════════════════════════════════════════════════════════════════
    gex_proximity_favorable = False
    gex_flip_point = hydra.gex_flip_point or 0

    if gex_flip_point > 0:
        # NEGATIVE GEX regime: price below flip point favors momentum
        # POSITIVE GEX regime: price above flip point = mean reversion
        if hydra.gex_regime == "NEGATIVE":
            # In NEGATIVE GEX, we want price BELOW flip point (trending)
            if spot < gex_flip_point:
                # Good for momentum trades - both calls and puts can work
                # But CALLs after bounce, PUTs in breakdown
                if side_upper == "CALL" and price_change_pct > 0:
                    gex_proximity_favorable = True
                if side_upper == "PUT" and price_change_pct < 0:
                    gex_proximity_favorable = True
        else:
            # In POSITIVE GEX, we want price ABOVE flip point (stable)
            if spot > gex_flip_point:
                # Good for mean reversion - fades work better
                gex_proximity_favorable = True

    if gex_proximity_favorable:
        conviction += 1
        conviction_details.append(f"✓ GEX_PROX({hydra.gex_regime}@{gex_flip_point:.0f})")
    else:
        conviction_details.append(f"✗ GEX_PROX({hydra.gex_regime})")

    # ════════════════════════════════════════════════════════════════════════════
    # CONVICTION VERDICT
    # ════════════════════════════════════════════════════════════════════════════

    MIN_CONVICTION = 5  # Updated for 13-signal system
    conviction_str = " | ".join(conviction_details)

    if conviction < MIN_CONVICTION:
        logger.info(
            f"BEAST_REJECT: {ticker} {side_upper} CONVICTION {conviction}/13 < {MIN_CONVICTION} | {conviction_str}"
        )
        return False, 0, f"CONVICTION_LOW: {conviction}/13 (need {MIN_CONVICTION}+) | {conviction_str}"

    # ════════════════════════════════════════════════════════════════════════════
    # POSITION SIZING via Confidence Score (Beast Mode V4)
    # ════════════════════════════════════════════════════════════════════════════
    # 5-7  conviction = 55-69 confidence = base size
    # 8-10 conviction = 70-84 confidence = 1.5x size
    # 11-13 conviction = 85-95 confidence = FULL SEND

    if conviction <= 7:
        confidence = 55 + (conviction - 5) * 5  # 55-65
    elif conviction <= 10:
        confidence = 70 + (conviction - 8) * 5  # 70-80
    else:
        confidence = 85 + (conviction - 11) * 5  # 85-95

    # Boost for aggressive flow
    if hydra.flow_bias in ["AGGRESSIVELY_BULLISH", "AGGRESSIVELY_BEARISH"]:
        confidence = min(95, confidence + 5)

    # Penalty for elevated blowup
    if hydra.blowup_probability > 50:
        confidence = max(55, confidence - 10)

    reason = (
        f"BEAST_APPROVED: {ticker} {side_upper} CONVICTION={conviction}/13 | "
        f"HYDRA={hydra.direction} GEX={hydra.gex_regime} regime={hydra.regime} | "
        f"{conviction_str}"
    )
    logger.info(f"{reason} | conf={confidence}%")

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
        global _call_number_counter, _direction_lock, _sent_calls, _cooldown_tracker

        # Daily reset of direction lock and memory-leak-prone structures
        today = datetime.now().strftime("%Y-%m-%d")
        if _direction_lock.get("_last_reset") != today:
            _direction_lock.clear()
            _direction_lock["_last_reset"] = today
            # FIX: Clear _sent_calls and _cooldown_tracker daily to prevent memory leak
            _sent_calls.clear()
            _cooldown_tracker.clear()
            logger.info(f"CPL_DIRECTION_COOLDOWN: Daily reset for {today} (cleared direction locks, _sent_calls, _cooldown_tracker)")

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

            # Process calls and puts
            for side, contracts, regime in [
                ("CALL", calls, "RISK_ON"),
                ("PUT", puts, "RISK_OFF"),
            ]:
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
