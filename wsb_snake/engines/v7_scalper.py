"""
V7 LIVE SCALPER: Backtest-validated signal detection for SPY 0DTE.
V2 CORE: momentum + volume + MA alignment = ENTER IMMEDIATELY
FIX 1: Circuit breaker (2 consecutive losses = stop)
FIX 2: $0.50 minimum entry (no lotto/reversal)
FIX 3: Conviction sizing (15% default, 25% high)
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pytz

from wsb_snake.utils.logger import get_logger
from wsb_snake.trading.alpaca_executor import alpaca_executor
from wsb_snake.trading.risk_governor import TradingEngine
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.collectors.polygon_options import polygon_options
from wsb_snake.notifications.telegram_bot import send_alert

logger = get_logger(__name__)

# V7 Parameters (from backtest)
OTM_OFFSET = 2
MIN_ENTRY_PRICE = 0.50
DEFAULT_SIZE_PCT = 0.15
HIGH_CONVICTION_SIZE_PCT = 0.25
MAX_TOTAL_EXPOSURE = 0.40
MAX_POSITIONS = 3
HIGH_CONVICTION_MOMENTUM = 0.005
HIGH_CONVICTION_VOLUME = 2.0

# Circuit breaker state
_consecutive_losses = 0
_circuit_breaker_active = False
_last_reset_date: Optional[str] = None
_signals_blocked = 0


@dataclass
class V7Signal:
    """Signal from V7 detection."""
    direction: str  # "CALL" or "PUT"
    confidence: float
    strike: float
    conviction: str  # "DEFAULT" or "HIGH"
    spy_price: float
    momentum: float
    vol_spike: float
    timestamp: datetime


def get_et_now() -> datetime:
    """Get current Eastern Time."""
    return datetime.now(pytz.timezone("US/Eastern"))


def reset_daily_state():
    """Reset circuit breaker and counters at start of each day."""
    global _consecutive_losses, _circuit_breaker_active, _last_reset_date, _signals_blocked
    today = get_et_now().strftime("%Y-%m-%d")
    if _last_reset_date != today:
        _consecutive_losses = 0
        _circuit_breaker_active = False
        _signals_blocked = 0
        _last_reset_date = today
        logger.info(f"V7: Daily state reset for {today}")


def record_trade_outcome(is_win: bool):
    """Record trade outcome for circuit breaker."""
    global _consecutive_losses, _circuit_breaker_active, _signals_blocked
    if is_win:
        _consecutive_losses = 0
        logger.info("V7: Win recorded, consecutive losses reset")
    else:
        _consecutive_losses += 1
        logger.warning(f"V7: Loss #{_consecutive_losses} recorded")
        if _consecutive_losses >= 2:
            _circuit_breaker_active = True
            logger.warning("V7: CIRCUIT BREAKER ACTIVATED - 2 consecutive losses")
            send_alert("🚨 **V7 CIRCUIT BREAKER**\n2 consecutive losses - trading halted for today")


def calculate_ma(bars: List[Dict], period: int) -> Optional[float]:
    """Calculate moving average from bars (most recent last)."""
    if len(bars) < period:
        return None
    recent = bars[-period:]
    closes = [b.get("close") or b.get("c") for b in recent]
    if not all(closes):
        return None
    return sum(closes) / period


def detect_signal_v7(bars: List[Dict]) -> Optional[V7Signal]:
    """
    V7 signal detection: momentum + volume + MA alignment.
    NO confirmation delay - enter immediately.

    Args:
        bars: List of minute bars, most recent LAST

    Returns:
        V7Signal if valid signal detected, None otherwise
    """
    if len(bars) < 11:
        return None

    # Get current bar (last in list)
    current_bar = bars[-1]
    bar_time = get_et_now()

    # Skip first 5 minutes after open
    market_open = bar_time.replace(hour=9, minute=30, second=0, microsecond=0)
    if (bar_time - market_open).total_seconds() / 60 < 5:
        return None

    # Use last 11 bars for analysis
    recent = bars[-11:]

    # Get current price
    current_price = recent[-1].get("close") or recent[-1].get("c")
    if not current_price:
        return None

    # Calculate momentum (% change over 10 bars)
    start_price = recent[0].get("close") or recent[0].get("c")
    if not start_price:
        return None
    momentum = (current_price - start_price) / start_price

    # Calculate volume spike
    volumes = [b.get("volume") or b.get("v") or 0 for b in recent[:-1]]
    avg_vol = sum(volumes) / len(volumes) if volumes else 1
    current_vol = recent[-1].get("volume") or recent[-1].get("v") or 0
    vol_spike = current_vol / avg_vol if avg_vol > 0 else 1

    # Calculate bar range (volatility indicator)
    high = recent[-1].get("high") or recent[-1].get("h") or current_price
    low = recent[-1].get("low") or recent[-1].get("l") or current_price
    bar_range = (high - low) / current_price if current_price > 0 else 0

    # Calculate MAs
    ma9 = calculate_ma(bars, 9)
    ma20 = calculate_ma(bars, 20)
    ma30 = calculate_ma(bars, 30)
    ma50 = calculate_ma(bars, 50)

    direction = None
    confidence = 0

    # V7 signal: momentum > 0.2% + volume spike 1.3x + MA alignment
    if momentum > 0.002 and vol_spike > 1.3:
        # MA alignment check for CALL (don't buy calls below MA30)
        if ma30 and current_price < ma30:
            logger.debug(f"V7: CALL rejected - price {current_price:.2f} < MA30 {ma30:.2f}")
            return None
        direction = "CALL"
        confidence = min(90, 50 + momentum * 2000 + vol_spike * 10 + bar_range * 500)
        strike = round(current_price) + OTM_OFFSET

    elif momentum < -0.002 and vol_spike > 1.3:
        # MA alignment check for PUT (don't buy puts above MA30)
        if ma30 and current_price > ma30:
            logger.debug(f"V7: PUT rejected - price {current_price:.2f} > MA30 {ma30:.2f}")
            return None
        direction = "PUT"
        confidence = min(90, 50 + abs(momentum) * 2000 + vol_spike * 10 + bar_range * 500)
        strike = round(current_price) - OTM_OFFSET

    if not direction or confidence < 62:
        return None

    # Conviction check for sizing
    conviction = "DEFAULT"
    if ma9 and ma20 and ma50:
        if direction == "CALL":
            ma_aligned = current_price > ma9 > ma20 > ma50
        else:
            ma_aligned = current_price < ma9 < ma20 < ma50

        if abs(momentum) > HIGH_CONVICTION_MOMENTUM and vol_spike > HIGH_CONVICTION_VOLUME and ma_aligned:
            conviction = "HIGH"

    return V7Signal(
        direction=direction,
        confidence=confidence,
        strike=strike,
        conviction=conviction,
        spy_price=current_price,
        momentum=momentum,
        vol_spike=vol_spike,
        timestamp=bar_time,
    )


def get_option_price(ticker: str, strike: float, option_type: str, expiry: str) -> Optional[float]:
    """Get current option price from Polygon."""
    try:
        chain = polygon_options.get_chain_for_expiration(ticker, strike, expiry, strike_range=3)
        contracts = chain.get("calls" if option_type.upper() == "CALL" else "puts", [])

        for c in contracts:
            if abs(c.get("strike", 0) - strike) < 0.5:
                # Return ask price for buying
                return c.get("ask") or c.get("last_price")
        return None
    except Exception as e:
        logger.warning(f"V7: Failed to get option price: {e}")
        return None


def get_option_symbol(ticker: str, strike: float, option_type: str, expiry: str) -> Optional[str]:
    """Get option symbol from Polygon chain."""
    try:
        chain = polygon_options.get_chain_for_expiration(ticker, strike, expiry, strike_range=3)
        contracts = chain.get("calls" if option_type.upper() == "CALL" else "puts", [])

        for c in contracts:
            if abs(c.get("strike", 0) - strike) < 0.5:
                return c.get("symbol")
        return None
    except Exception as e:
        logger.warning(f"V7: Failed to get option symbol: {e}")
        return None


def run_v7_scan() -> Optional[Dict]:
    """
    Run V7 signal scan on SPY.
    Returns trade details if signal found and executed, None otherwise.
    """
    global _signals_blocked

    # Reset daily state if new day
    reset_daily_state()

    # Check circuit breaker
    if _circuit_breaker_active:
        logger.debug("V7: Circuit breaker active, skipping scan")
        return None

    # Check market hours
    et_now = get_et_now()
    if et_now.hour < 9 or (et_now.hour == 9 and et_now.minute < 35):
        return None
    if et_now.hour >= 16:
        return None

    # Get SPY minute bars
    try:
        bars = polygon_enhanced.get_intraday_bars("SPY", timespan="minute", multiplier=1, limit=60)
        if not bars or len(bars) < 15:
            logger.debug(f"V7: Insufficient bars ({len(bars) if bars else 0})")
            return None
    except Exception as e:
        logger.warning(f"V7: Failed to get SPY bars: {e}")
        return None

    # Detect signal
    signal = detect_signal_v7(bars)
    if not signal:
        return None

    logger.info(
        f"V7: Signal detected - {signal.direction} @ ${signal.strike} | "
        f"Conf: {signal.confidence:.0f}% | Mom: {signal.momentum*100:.2f}% | "
        f"Vol: {signal.vol_spike:.1f}x | Conv: {signal.conviction}"
    )

    # Get today's expiry
    expiry = et_now.strftime("%Y-%m-%d")

    # Get option price
    option_price = get_option_price("SPY", signal.strike, signal.direction, expiry)
    if not option_price:
        logger.warning(f"V7: No option price for SPY {signal.direction} ${signal.strike}")
        return None

    # FIX 2: Minimum entry price check
    if option_price < MIN_ENTRY_PRICE:
        logger.info(f"V7: BLOCKED - option ${option_price:.2f} < ${MIN_ENTRY_PRICE:.2f} minimum")
        return None

    # Get option symbol
    option_symbol = get_option_symbol("SPY", signal.strike, signal.direction, expiry)
    if not option_symbol:
        logger.warning(f"V7: No option symbol for SPY {signal.direction} ${signal.strike}")
        return None

    # FIX 3: Conviction-based sizing
    size_pct = HIGH_CONVICTION_SIZE_PCT if signal.conviction == "HIGH" else DEFAULT_SIZE_PCT

    # Calculate targets
    target_price = option_price * 1.25  # +25% target
    stop_loss = option_price * 0.60     # -40% stop (as per V7 backtest)

    # Execute trade via Alpaca
    logger.info(
        f"V7: Executing {signal.direction} ${signal.strike} @ ${option_price:.2f} | "
        f"Target: ${target_price:.2f} | Stop: ${stop_loss:.2f} | {signal.conviction}"
    )

    try:
        position = alpaca_executor.execute_scalp_entry(
            underlying="SPY",
            direction="long",  # Always long options (buying calls or puts)
            entry_price=option_price,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=signal.confidence,
            pattern=f"V7_{signal.direction}_{signal.conviction}",
            engine=TradingEngine.SCALPER,
            strike_override=signal.strike,
            option_symbol_override=option_symbol,
            option_type_override=signal.direction.lower(),
        )

        if position:
            logger.info(f"V7: EXECUTED - {position.option_symbol} qty={position.qty}")
            send_alert(
                f"🎯 **V7 ENTRY**\n"
                f"SPY {signal.direction} ${signal.strike}\n"
                f"Entry: ${option_price:.2f}\n"
                f"Confidence: {signal.confidence:.0f}%\n"
                f"Conviction: {signal.conviction}\n"
                f"Mom: {signal.momentum*100:+.2f}% | Vol: {signal.vol_spike:.1f}x"
            )
            return {
                "symbol": option_symbol,
                "direction": signal.direction,
                "strike": signal.strike,
                "entry_price": option_price,
                "target": target_price,
                "stop": stop_loss,
                "conviction": signal.conviction,
                "confidence": signal.confidence,
            }
        else:
            logger.warning("V7: Trade skipped by Alpaca (position limits or risk)")
            return None

    except Exception as e:
        logger.error(f"V7: Execution failed - {e}")
        return None


def get_v7_status() -> Dict:
    """Get current V7 engine status."""
    return {
        "circuit_breaker_active": _circuit_breaker_active,
        "consecutive_losses": _consecutive_losses,
        "signals_blocked": _signals_blocked,
        "last_reset_date": _last_reset_date,
        "parameters": {
            "otm_offset": OTM_OFFSET,
            "min_entry_price": MIN_ENTRY_PRICE,
            "default_size_pct": DEFAULT_SIZE_PCT,
            "high_conviction_size_pct": HIGH_CONVICTION_SIZE_PCT,
            "max_exposure": MAX_TOTAL_EXPOSURE,
        }
    }


def force_reset_circuit_breaker():
    """Admin override to reset circuit breaker."""
    global _consecutive_losses, _circuit_breaker_active, _signals_blocked
    _consecutive_losses = 0
    _circuit_breaker_active = False
    _signals_blocked = 0
    logger.warning("V7: Circuit breaker FORCE RESET by admin")
    send_alert("⚠️ **V7 Circuit Breaker Reset**\nTrading resumed by admin override")
