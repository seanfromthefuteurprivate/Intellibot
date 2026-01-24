"""
Session Regime Detector

Determines current market session and trading regime.
Implements logic gates for different time windows.
"""

from datetime import datetime, time
from enum import Enum
from typing import Optional, Dict
import pytz

from wsb_snake.config import SESSION_WINDOWS
from wsb_snake.utils.logger import log


class SessionType(Enum):
    """Market session types."""
    PREMARKET = "premarket"
    OPEN = "open"           # First hour - high volatility
    MORNING = "morning"     # Mid-morning - settling
    LUNCH = "lunch"         # Chop zone - avoid signals
    POWER_HOUR_EARLY = "power_hour_early"  # 1pm-3pm - momentum builds
    POWER_HOUR = "power_hour"  # 3pm-4pm - final push
    AFTERHOURS = "afterhours"
    CLOSED = "closed"


class RegimeType(Enum):
    """Market regime types."""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    CHOP = "chop"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


def get_eastern_time() -> datetime:
    """Get current time in Eastern timezone."""
    eastern = pytz.timezone('US/Eastern')
    return datetime.now(eastern)


def get_session_type() -> SessionType:
    """
    Determine current session type based on Eastern time.
    
    Returns:
        SessionType enum value
    """
    now = get_eastern_time()
    current_hour = now.hour
    current_minute = now.minute
    current_time = current_hour * 60 + current_minute  # Minutes since midnight
    
    # Check if weekend
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return SessionType.CLOSED
    
    # Map session window names to enum values
    session_map = {
        "premarket": SessionType.PREMARKET,
        "open": SessionType.OPEN,
        "morning": SessionType.MORNING,
        "lunch": SessionType.LUNCH,
        "power_hour_early": SessionType.POWER_HOUR_EARLY,
        "power_hour": SessionType.POWER_HOUR,
        "afterhours": SessionType.AFTERHOURS,
    }
    
    # Convert session windows to minutes
    for session_name, (start_h, start_m, end_h, end_m) in SESSION_WINDOWS.items():
        start_mins = start_h * 60 + start_m
        end_mins = end_h * 60 + end_m
        
        if start_mins <= current_time < end_mins:
            return session_map.get(session_name, SessionType.CLOSED)
    
    return SessionType.CLOSED


def is_market_open() -> bool:
    """Check if US stock market is currently open."""
    session = get_session_type()
    return session in [
        SessionType.OPEN,
        SessionType.MORNING,
        SessionType.LUNCH,
        SessionType.POWER_HOUR_EARLY,
        SessionType.POWER_HOUR,
    ]


def is_power_hour() -> bool:
    """Check if we're in power hour (1pm-4pm ET)."""
    session = get_session_type()
    return session in [SessionType.POWER_HOUR_EARLY, SessionType.POWER_HOUR]


def is_final_hour() -> bool:
    """Check if we're in the final trading hour (3pm-4pm ET)."""
    return get_session_type() == SessionType.POWER_HOUR


def is_lunch_chop() -> bool:
    """Check if we're in lunch chop zone (avoid signals)."""
    return get_session_type() == SessionType.LUNCH


def get_session_info() -> Dict:
    """
    Get detailed session information.
    
    Returns:
        Dict with session type, flags, and time info
    """
    now = get_eastern_time()
    session = get_session_type()
    
    # Calculate time to market events
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    power_hour_start = now.replace(hour=15, minute=0, second=0, microsecond=0)
    
    minutes_to_close = (market_close - now).total_seconds() / 60 if now < market_close else 0
    minutes_to_power_hour = (power_hour_start - now).total_seconds() / 60 if now < power_hour_start else 0
    
    return {
        "session": session.value,
        "is_open": is_market_open(),
        "is_power_hour": is_power_hour(),
        "is_final_hour": is_final_hour(),
        "is_lunch_chop": is_lunch_chop(),
        "current_time_et": now.strftime("%H:%M:%S"),
        "minutes_to_close": max(0, minutes_to_close),
        "minutes_to_power_hour": max(0, minutes_to_power_hour),
        "signal_quality_multiplier": get_session_signal_multiplier(session),
    }


def get_session_signal_multiplier(session: SessionType) -> float:
    """
    Get signal quality multiplier based on session.
    
    Signals during lunch are penalized, power hour signals get boosted.
    """
    multipliers = {
        SessionType.PREMARKET: 0.5,
        SessionType.OPEN: 1.0,
        SessionType.MORNING: 0.9,
        SessionType.LUNCH: 0.5,          # Chop zone - discount signals
        SessionType.POWER_HOUR_EARLY: 1.2,  # Building momentum
        SessionType.POWER_HOUR: 1.5,      # Prime time for 0DTE
        SessionType.AFTERHOURS: 0.3,
        SessionType.CLOSED: 0.0,
    }
    return multipliers.get(session, 0.5)


def should_scan_for_signals() -> bool:
    """
    Determine if we should actively scan for signals.
    
    Returns False during lunch chop and closed hours.
    """
    session = get_session_type()
    
    # Don't scan during lunch (too choppy) or when closed
    if session in [SessionType.LUNCH, SessionType.CLOSED, SessionType.AFTERHOURS]:
        return False
    
    # For 0DTE, we especially want power hour
    return True


def get_0dte_urgency() -> str:
    """
    Get urgency level for 0DTE trades based on time remaining.
    
    Returns:
        "low", "medium", "high", "critical", or "expired"
    """
    now = get_eastern_time()
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if now >= market_close:
        return "expired"
    
    minutes_remaining = (market_close - now).total_seconds() / 60
    
    if minutes_remaining > 120:  # > 2 hours
        return "low"
    elif minutes_remaining > 60:  # 1-2 hours
        return "medium"
    elif minutes_remaining > 30:  # 30-60 min
        return "high"
    else:  # < 30 min
        return "critical"
