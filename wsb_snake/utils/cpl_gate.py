"""
CPL Gate - Centralized CPL alignment check for ALL trading paths.

CRITICAL: Every trading path MUST call cpl_gate.check() before executing trades.
If CPL is not aligned, the trade MUST be blocked.

This prevents the system from trading against market regime intelligence.
"""

from wsb_snake.db.database import get_connection
from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


def get_latest_cpl_signal(ticker: str) -> dict:
    """
    Get the latest CPL signal for a ticker (within last 30 minutes).

    Returns dict with keys: side, regime, confidence, timestamp
    Returns None if no recent signal exists.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT side, regime, confidence, timestamp_et
            FROM cpl_calls
            WHERE ticker = ?
            AND datetime(timestamp_et) > datetime('now', '-30 minutes')
            ORDER BY timestamp_et DESC
            LIMIT 1
        """, (ticker,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "side": row[0],      # "CALL" or "PUT"
                "regime": row[1],    # "RISK_ON" or "RISK_OFF"
                "confidence": row[2],
                "timestamp": row[3]
            }
    except Exception as e:
        logger.debug(f"CPL lookup failed for {ticker}: {e}")
    return None


def check(ticker: str, direction: str, allow_no_signal: bool = False) -> tuple:
    """
    Check if trade direction aligns with CPL intelligence.

    Args:
        ticker: The ticker symbol (e.g., "SPY")
        direction: "long" or "short" (or "calls"/"puts")
        allow_no_signal: If True, allow trades when no CPL signal exists.
                         Default False (conservative - block if no signal).

    Returns: (is_allowed, reason)
        is_allowed: True if trade can proceed, False if blocked
        reason: Human-readable explanation
    """
    # Normalize direction
    is_long = direction.lower() in ("long", "calls", "call", "bullish", "buy")

    cpl = get_latest_cpl_signal(ticker)

    if not cpl:
        if allow_no_signal:
            return True, "NO_CPL_SIGNAL (allowed by config)"
        else:
            return False, "NO_CPL_SIGNAL - CPL intelligence required"

    cpl_is_bullish = cpl["side"] == "CALL" and cpl["regime"] == "RISK_ON"
    cpl_is_bearish = cpl["side"] == "PUT" or cpl["regime"] == "RISK_OFF"

    # CRITICAL: Block if direction conflicts with CPL
    if is_long and cpl_is_bearish:
        return False, f"CPL_CONFLICT: CPL says {cpl['side']}/{cpl['regime']} but trade is LONG"

    if not is_long and cpl_is_bullish:
        return False, f"CPL_CONFLICT: CPL says {cpl['side']}/{cpl['regime']} but trade is SHORT"

    return True, f"CPL_ALIGNED: {cpl['side']}/{cpl['regime']} (conf={cpl['confidence']:.0f}%)"


def block_trade(ticker: str, direction: str, reason: str, send_alert: bool = True):
    """
    Log and optionally alert when a trade is blocked by CPL.

    Args:
        ticker: The ticker symbol
        direction: Trade direction
        reason: The reason for blocking
        send_alert: Whether to send a Telegram alert (default True)
    """
    logger.warning(f"🚫 TRADE BLOCKED by CPL: {ticker} {direction} - {reason}")

    if send_alert:
        try:
            from wsb_snake.notifications.telegram_bot import send_alert as telegram_alert
            telegram_alert(f"🚫 **TRADE BLOCKED BY CPL**\n\n{ticker} {direction.upper()}\nReason: {reason}")
        except Exception as e:
            logger.debug(f"Failed to send CPL block alert: {e}")
