"""
Dual-channel Telegram notification system.

MAIN CHANNEL: Pure signals for ALL users (any broker)
ALPACA CHANNEL: Execution status (optional, Alpaca users only)

If TELEGRAM_ALPACA_CHAT_ID is not configured, Alpaca status messages
are silently logged but NOT sent to the main channel.
"""

import logging
import os
import requests

from wsb_snake.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from wsb_snake.notifications.telegram_bot import send_alert

logger = logging.getLogger(__name__)

# Optional Alpaca-specific channel
TELEGRAM_ALPACA_CHAT_ID = os.getenv("TELEGRAM_ALPACA_CHAT_ID")


def send_signal(message: str) -> bool:
    """
    Send a trading signal to the MAIN channel.
    All users (any broker) see these signals.

    Args:
        message: The signal message to send

    Returns:
        True if sent successfully, False otherwise
    """
    logger.info(f"Sending signal to main channel: {message[:100]}...")
    result = send_alert(message)
    if result:
        logger.info("Signal sent to main channel successfully")
    else:
        logger.warning("Failed to send signal to main channel")
    return result


def send_alpaca_status(message: str) -> bool:
    """
    Send execution status to the ALPACA channel only.
    If TELEGRAM_ALPACA_CHAT_ID is not configured, the message is logged but not sent.

    This ensures Alpaca-specific execution details don't clutter the main signal channel.

    Args:
        message: The Alpaca execution status message

    Returns:
        True if sent successfully (or logged when channel not configured), False on error
    """
    if not TELEGRAM_ALPACA_CHAT_ID:
        logger.info(f"[Alpaca Status - Not Sent] {message}")
        return True  # Not an error - just not configured

    if not TELEGRAM_BOT_TOKEN:
        logger.warning("Telegram bot token not set. Cannot send Alpaca status.")
        return False

    logger.info(f"Sending Alpaca status: {message[:100]}...")

    # Escape for Telegram Markdown
    if isinstance(message, str):
        text = message.replace("\\", "\\\\").replace("_", "\\_")
    else:
        text = str(message)

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_ALPACA_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Alpaca status sent successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to send Alpaca status: {e}")
        return False


def send_error_after_retry(message: str, error: Exception, retries: int = 3) -> bool:
    """
    Send a user-friendly error notification after retries are exhausted.

    This sends to the MAIN channel so users know something went wrong,
    but presents a friendly message rather than raw API errors.

    Args:
        message: User-friendly context about what failed
        error: The exception that occurred (logged but not sent to users)
        retries: Number of retries that were attempted

    Returns:
        True if notification sent successfully, False otherwise
    """
    # Log the full technical error for debugging
    logger.error(f"Error after {retries} retries: {error}")
    logger.error(f"Context: {message}")

    # Craft a user-friendly message (no raw API errors)
    friendly_message = (
        f"*System Notice*\n\n"
        f"{message}\n\n"
        f"Our system encountered an issue after {retries} attempts. "
        f"We're monitoring the situation and will resume normal operations shortly."
    )

    logger.info("Sending user-friendly error notification to main channel")
    result = send_alert(friendly_message)

    if result:
        logger.info("Error notification sent successfully")
    else:
        logger.warning("Failed to send error notification")

    return result
