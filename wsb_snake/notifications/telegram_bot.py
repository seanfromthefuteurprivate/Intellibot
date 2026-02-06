import logging
import requests
from wsb_snake.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

def send_alert(message):
    """
    Sends a message to the configured Telegram chat using the HTTP API.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not set. Skipping alert.")
        return False

    logger.info(f"Sending Telegram alert...")

    # Escape for Telegram Markdown: \ and _ (avoid italic break e.g. RISK_ON, option_descriptor)
    if isinstance(message, str):
        text = message.replace("\\", "\\\\").replace("_", "\\_")
    else:
        text = str(message)

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Alert sent successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False
