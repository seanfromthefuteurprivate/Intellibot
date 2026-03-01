"""
TELEGRAM SCREENSHOT BOT - Zero Friction Trade Learning
═══════════════════════════════════════════════════════════════════

See a banger trade on WSB/Twitter/Discord? Screenshot it. Send to this bot.
The snake learns from it automatically.

Flow:
1. Screenshot trade on phone
2. Share to Telegram → Select this bot → Send
3. Bot downloads image, runs GPT-4o vision OCR
4. Extracts: ticker, strike, expiry, entry/exit, P&L, direction
5. Feeds into Semantic Memory + Trade Graph
6. Sends confirmation with extracted data

Under 5 seconds from screenshot to learned.
"""

import os
import sys
import base64
import hashlib
import threading
import time
import fcntl
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from wsb_snake.utils.logger import get_logger
from wsb_snake.db.database import get_connection
from wsb_snake.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = get_logger(__name__)

# Telegram API base URL
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Lock file to prevent multiple instances
LOCK_FILE = Path("/tmp/telegram_screenshot_bot.lock")
_lock_file_handle = None


def acquire_lock() -> bool:
    """Acquire exclusive lock to prevent multiple bot instances."""
    global _lock_file_handle
    try:
        _lock_file_handle = open(LOCK_FILE, 'w')
        fcntl.flock(_lock_file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        _lock_file_handle.write(str(os.getpid()))
        _lock_file_handle.flush()
        return True
    except (IOError, OSError):
        log.warning("Another Telegram bot instance is already running")
        return False


def release_lock():
    """Release the lock file."""
    global _lock_file_handle
    if _lock_file_handle:
        try:
            fcntl.flock(_lock_file_handle.fileno(), fcntl.LOCK_UN)
            _lock_file_handle.close()
            LOCK_FILE.unlink(missing_ok=True)
        except:
            pass
        _lock_file_handle = None


@dataclass
class TelegramPhoto:
    """A photo received from Telegram."""
    file_id: str
    file_unique_id: str
    width: int
    height: int
    file_size: int
    message_id: int
    chat_id: int
    from_user: str
    timestamp: datetime


class TelegramScreenshotBot:
    """
    Telegram bot that receives trade screenshots and feeds them into the learning system.

    Usage:
        bot = TelegramScreenshotBot()
        bot.start()  # Starts polling in background

    Or run standalone:
        python -m wsb_snake.collectors.telegram_screenshot_bot
    """

    def __init__(self, poll_interval: int = 10):
        """
        Initialize the bot.

        Args:
            poll_interval: Seconds between polling for new messages
        """
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_update_id = 0
        self._processed_photos: set = set()

        # Load last update ID from database
        self._load_state()

        # Initialize database tables
        self._ensure_tables()

        log.info(f"TelegramScreenshotBot initialized (poll interval: {poll_interval}s)")

    def _ensure_tables(self):
        """Create required database tables."""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Telegram screenshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telegram_screenshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id TEXT UNIQUE NOT NULL,
                    file_unique_id TEXT,
                    message_id INTEGER,
                    chat_id INTEGER,
                    from_user TEXT,
                    received_at TEXT,

                    -- Processing
                    status TEXT DEFAULT 'pending',
                    processed_at TEXT,
                    error_message TEXT,

                    -- Extracted data
                    extracted_data TEXT,
                    learned_trade_id INTEGER,

                    -- Learning linkage
                    semantic_id TEXT,
                    graph_id TEXT,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Bot state table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telegram_bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)

            conn.commit()
            conn.close()
        except Exception as e:
            log.debug(f"Table creation: {e}")

    def _load_state(self):
        """Load bot state from database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM telegram_bot_state WHERE key = 'last_update_id'")
            row = cursor.fetchone()
            if row:
                self._last_update_id = int(row[0])

            cursor.execute("SELECT file_unique_id FROM telegram_screenshots")
            rows = cursor.fetchall()
            self._processed_photos = set(row[0] for row in rows)

            conn.close()
            log.info(f"Loaded bot state: last_update_id={self._last_update_id}, processed={len(self._processed_photos)}")
        except Exception as e:
            log.debug(f"Could not load state: {e}")

    def _save_state(self):
        """Save bot state to database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO telegram_bot_state (key, value)
                VALUES ('last_update_id', ?)
            """, (str(self._last_update_id),))
            conn.commit()
            conn.close()
        except Exception as e:
            log.debug(f"Could not save state: {e}")

    def get_updates(self) -> List[Dict]:
        """
        Get new updates from Telegram using long polling.

        Returns:
            List of update objects
        """
        if not TELEGRAM_BOT_TOKEN:
            log.warning("TELEGRAM_BOT_TOKEN not configured")
            return []

        try:
            params = {
                "offset": self._last_update_id + 1,
                "timeout": 30,  # Long polling
                "allowed_updates": ["message"]
            }

            resp = requests.get(
                f"{TELEGRAM_API}/getUpdates",
                params=params,
                timeout=35
            )

            if resp.status_code != 200:
                log.warning(f"Telegram API error: {resp.status_code}")
                return []

            data = resp.json()
            if not data.get("ok"):
                log.warning(f"Telegram API returned error: {data}")
                return []

            return data.get("result", [])

        except requests.exceptions.Timeout:
            # Normal for long polling
            return []
        except Exception as e:
            log.error(f"Failed to get updates: {e}")
            return []

    def download_photo(self, file_id: str) -> Optional[bytes]:
        """
        Download a photo from Telegram.

        Args:
            file_id: Telegram file ID

        Returns:
            Photo bytes or None on failure
        """
        try:
            # Get file path
            resp = requests.get(
                f"{TELEGRAM_API}/getFile",
                params={"file_id": file_id},
                timeout=10
            )

            if resp.status_code != 200:
                log.error(f"Failed to get file path: {resp.status_code}")
                return None

            data = resp.json()
            if not data.get("ok"):
                return None

            file_path = data["result"]["file_path"]

            # Download file
            download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
            resp = requests.get(download_url, timeout=30)

            if resp.status_code != 200:
                log.error(f"Failed to download file: {resp.status_code}")
                return None

            return resp.content

        except Exception as e:
            log.error(f"Failed to download photo: {e}")
            return None

    def send_message(self, chat_id: int, text: str, reply_to: Optional[int] = None) -> bool:
        """
        Send a message to a Telegram chat.

        Args:
            chat_id: Target chat ID
            text: Message text
            reply_to: Message ID to reply to

        Returns:
            Success status
        """
        try:
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }

            if reply_to:
                payload["reply_to_message_id"] = reply_to

            resp = requests.post(
                f"{TELEGRAM_API}/sendMessage",
                json=payload,
                timeout=10
            )

            return resp.status_code == 200

        except Exception as e:
            log.error(f"Failed to send message: {e}")
            return False

    def process_photo(self, photo: TelegramPhoto) -> Dict[str, Any]:
        """
        Process a received photo - download, OCR, extract trade data, ingest.

        Args:
            photo: TelegramPhoto object

        Returns:
            Processing result with extracted data
        """
        result = {
            "status": "error",
            "file_id": photo.file_id,
            "message": "",
            "extracted": None,
            "learned_trade_id": None,
        }

        try:
            # 1. Download photo
            log.info(f"📸 Downloading photo from {photo.from_user}...")
            content = self.download_photo(photo.file_id)

            if not content:
                result["message"] = "Failed to download photo"
                self._save_screenshot_record(photo, "error", result["message"])
                return result

            # 2. Convert to base64
            image_base64 = base64.b64encode(content).decode("utf-8")
            content_hash = hashlib.sha256(content).hexdigest()[:16]

            # 3. OCR with TradeExtractor
            log.info(f"🔍 Running OCR extraction...")
            from wsb_snake.collectors.trade_extractor import TradeExtractor
            extractor = TradeExtractor()

            extracted = extractor.extract_trade_data(
                image_base64=image_base64,
                filename=f"telegram_{photo.file_unique_id}.jpg",
                mime_type="image/jpeg"
            )

            if not extracted:
                result["message"] = "Could not extract trade data from image"
                result["status"] = "no_trade"
                self._save_screenshot_record(photo, "no_trade", result["message"])
                return result

            result["extracted"] = extracted

            # 4. Save to screenshots table
            screenshot_id = self._save_screenshot_record(
                photo, "processed", None, extracted
            )

            # 5. Save as learned trade
            learned_trade_id = extractor.save_learned_trade(screenshot_id, extracted)
            result["learned_trade_id"] = learned_trade_id

            # 6. Ingest into Semantic Memory + Trade Graph
            semantic_id, graph_id = self._ingest_to_learning_systems(extracted, learned_trade_id)

            # 7. Update record with learning IDs
            self._update_learning_ids(screenshot_id, semantic_id, graph_id)

            result["status"] = "success"
            result["message"] = self._format_success_message(extracted)

            log.info(f"✅ Processed screenshot: {extracted.get('ticker')} {extracted.get('trade_type')} P&L: {extracted.get('profit_loss_pct')}%")

        except Exception as e:
            log.error(f"Failed to process photo: {e}")
            result["message"] = f"Processing error: {str(e)[:100]}"
            self._save_screenshot_record(photo, "error", str(e))

        return result

    def _save_screenshot_record(
        self,
        photo: TelegramPhoto,
        status: str,
        error_message: Optional[str],
        extracted: Optional[Dict] = None
    ) -> int:
        """Save screenshot record to database."""
        import json

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO telegram_screenshots (
                file_id, file_unique_id, message_id, chat_id, from_user,
                received_at, status, processed_at, error_message, extracted_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            photo.file_id,
            photo.file_unique_id,
            photo.message_id,
            photo.chat_id,
            photo.from_user,
            photo.timestamp.isoformat(),
            status,
            datetime.utcnow().isoformat(),
            error_message,
            json.dumps(extracted) if extracted else None,
        ))

        screenshot_id = cursor.lastrowid
        conn.commit()
        conn.close()

        self._processed_photos.add(photo.file_unique_id)

        return screenshot_id

    def _update_learning_ids(self, screenshot_id: int, semantic_id: str, graph_id: str):
        """Update screenshot record with learning system IDs."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE telegram_screenshots
            SET semantic_id = ?, graph_id = ?
            WHERE id = ?
        """, (semantic_id, graph_id, screenshot_id))
        conn.commit()
        conn.close()

    def _ingest_to_learning_systems(
        self,
        extracted: Dict[str, Any],
        learned_trade_id: int
    ) -> tuple:
        """
        Ingest extracted trade into Semantic Memory and Trade Graph.

        Returns:
            (semantic_id, graph_id)
        """
        semantic_id = None
        graph_id = None

        try:
            # Build ParsedTrade-like object for ingestion
            from datetime import timedelta

            ticker = extracted.get("ticker") or "SPY"
            trade_type = extracted.get("trade_type") or "CALLS"
            option_type = "CALL" if "CALL" in trade_type.upper() else "PUT"
            direction = "LONG" if option_type == "CALL" else "SHORT"

            entry_price = extracted.get("entry_price") or 1.0
            exit_price = extracted.get("exit_price") or entry_price
            pnl_dollars = extracted.get("profit_loss") or 0.0
            pnl_percent = extracted.get("profit_loss_pct") or 0.0

            # Parse times
            trade_date = extracted.get("trade_date") or datetime.now().strftime("%Y-%m-%d")
            entry_time_str = extracted.get("entry_time") or "10:00"
            exit_time_str = extracted.get("exit_time") or "10:30"

            try:
                entry_time = datetime.strptime(f"{trade_date} {entry_time_str}", "%Y-%m-%d %H:%M")
                exit_time = datetime.strptime(f"{trade_date} {exit_time_str}", "%Y-%m-%d %H:%M")
            except:
                entry_time = datetime.now(timezone.utc)
                exit_time = entry_time + timedelta(minutes=30)

            # 1. Ingest to Semantic Memory
            try:
                from wsb_snake.learning.semantic_memory import (
                    get_semantic_memory,
                    TradeOutcome,
                    TradeConditions,
                )

                semantic = get_semantic_memory()

                conditions = TradeConditions(
                    ticker=ticker,
                    direction=direction,
                    entry_price=entry_price,
                    rsi=50.0,
                    adx=25.0,
                    atr=1.0,
                    macd_signal="neutral",
                    volume_ratio=1.0,
                    regime="unknown",
                    vix=20.0,
                    gex_regime="unknown",
                    hydra_direction="NEUTRAL",
                    confluence_score=0.7,
                    stop_distance_pct=10.0,
                    target_distance_pct=15.0,
                )

                trade_id = hashlib.md5(
                    f"telegram_{learned_trade_id}_{ticker}".encode()
                ).hexdigest()[:12]

                duration = int((exit_time - entry_time).total_seconds() / 60)

                outcome = TradeOutcome(
                    trade_id=trade_id,
                    conditions=conditions,
                    entry_reasoning=f"Learned from Telegram screenshot: {extracted.get('notes', '')}",
                    pnl_dollars=pnl_dollars,
                    pnl_percent=pnl_percent,
                    duration_minutes=duration,
                    max_adverse_excursion_pct=min(0, pnl_percent),
                    max_favorable_excursion_pct=max(0, pnl_percent),
                    exit_reason="TARGET" if pnl_dollars > 0 else "STOP",
                    exit_price=exit_price,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    lessons_learned=f"WSB screenshot: {'WIN' if pnl_dollars > 0 else 'LOSS'} {pnl_percent:+.1f}%",
                )

                semantic.record_trade(outcome)
                semantic_id = trade_id
                log.info(f"📚 Ingested to Semantic Memory: {trade_id}")

            except Exception as e:
                log.warning(f"Semantic memory ingestion failed: {e}")

            # 2. Ingest to Trade Graph
            try:
                from wsb_snake.learning.trade_graph import get_trade_graph

                trade_graph = get_trade_graph()

                pattern = extracted.get("chart_pattern") or f"TELEGRAM_{option_type}_{ticker}"

                conditions_dict = {
                    "ticker": ticker,
                    "pattern": pattern,
                    "direction": direction.lower(),
                    "option_type": option_type,
                    "strike": extracted.get("strike") or 0.0,
                    "expiry": extracted.get("expiry") or "",
                    "regime": "unknown",
                    "gex_regime": "unknown",
                    "hydra_direction": "NEUTRAL",
                    "flow_bias": "NEUTRAL",
                    "volume_ratio": 1.0,
                    "platform": extracted.get("platform") or "unknown",
                }

                trade_graph.record_trade(
                    ticker=ticker,
                    direction=direction.lower(),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    pnl_dollars=pnl_dollars,
                    pattern=pattern,
                    entry_reasoning=f"Telegram screenshot: {extracted.get('notes', '')}",
                    exit_reasoning="TARGET" if pnl_dollars > 0 else "STOP",
                    conditions=conditions_dict,
                    metadata={
                        "source": "telegram_screenshot",
                        "learned_trade_id": learned_trade_id,
                        "platform": extracted.get("platform"),
                        "quantity": extracted.get("quantity"),
                    },
                )
                graph_id = f"tg_{learned_trade_id}"
                log.info(f"🕸️ Ingested to Trade Graph: {graph_id}")

            except Exception as e:
                log.warning(f"Trade graph ingestion failed: {e}")

        except Exception as e:
            log.error(f"Learning system ingestion failed: {e}")

        return semantic_id or "", graph_id or ""

    def _format_success_message(self, extracted: Dict[str, Any]) -> str:
        """Format success message for Telegram reply."""
        ticker = extracted.get("ticker") or "?"
        trade_type = extracted.get("trade_type") or "?"
        strike = extracted.get("strike")
        expiry = extracted.get("expiry")
        pnl = extracted.get("profit_loss")
        pnl_pct = extracted.get("profit_loss_pct")
        platform = extracted.get("platform") or "Unknown"

        # Emoji based on P&L
        if pnl and pnl > 0:
            emoji = "🔥" if pnl_pct and pnl_pct > 50 else "✅"
        elif pnl and pnl < 0:
            emoji = "📉"
        else:
            emoji = "📊"

        msg = f"{emoji} *Trade Learned!*\n\n"
        msg += f"*Ticker:* {ticker}\n"
        msg += f"*Type:* {trade_type}"

        if strike:
            msg += f" ${strike}"
        if expiry:
            msg += f" exp {expiry}"
        msg += "\n"

        if pnl is not None:
            sign = "+" if pnl >= 0 else ""
            msg += f"*P&L:* {sign}${pnl:,.2f}"
            if pnl_pct is not None:
                msg += f" ({sign}{pnl_pct:.1f}%)"
            msg += "\n"

        msg += f"*Platform:* {platform}\n"
        msg += f"\n_Saved to Semantic Memory \\+ Trade Graph_"

        return msg

    def poll_once(self) -> int:
        """
        Poll for updates once and process any photos.

        Returns:
            Number of photos processed
        """
        updates = self.get_updates()
        processed = 0

        for update in updates:
            update_id = update.get("update_id", 0)

            # Update last_update_id
            if update_id > self._last_update_id:
                self._last_update_id = update_id
                self._save_state()

            message = update.get("message", {})

            # Check for photo
            photos = message.get("photo", [])
            if not photos:
                continue

            # Get largest photo (last in list)
            largest_photo = photos[-1]
            file_unique_id = largest_photo.get("file_unique_id")

            # Skip if already processed
            if file_unique_id in self._processed_photos:
                continue

            # Build TelegramPhoto object
            photo = TelegramPhoto(
                file_id=largest_photo.get("file_id"),
                file_unique_id=file_unique_id,
                width=largest_photo.get("width", 0),
                height=largest_photo.get("height", 0),
                file_size=largest_photo.get("file_size", 0),
                message_id=message.get("message_id", 0),
                chat_id=message.get("chat", {}).get("id", 0),
                from_user=message.get("from", {}).get("username", "unknown"),
                timestamp=datetime.fromtimestamp(message.get("date", 0), tz=timezone.utc),
            )

            log.info(f"📷 New screenshot from @{photo.from_user}")

            # Send "processing" reaction
            self.send_message(
                photo.chat_id,
                "⏳ Processing screenshot...",
                reply_to=photo.message_id
            )

            # Process the photo
            result = self.process_photo(photo)
            processed += 1

            # Send result
            if result["status"] == "success":
                self.send_message(
                    photo.chat_id,
                    result["message"],
                    reply_to=photo.message_id
                )
            elif result["status"] == "no_trade":
                self.send_message(
                    photo.chat_id,
                    "🤔 Couldn't extract trade data from this image. Make sure it shows a trade with P&L visible.",
                    reply_to=photo.message_id
                )
            else:
                self.send_message(
                    photo.chat_id,
                    f"❌ Error: {result['message']}",
                    reply_to=photo.message_id
                )

        return processed

    def start(self):
        """Start background polling thread."""
        if self._running:
            log.warning("Bot already running")
            return

        # Acquire lock to prevent multiple instances
        if not acquire_lock():
            log.error("Cannot start: another instance is already running (409 conflict prevention)")
            return False

        self._running = True

        def _poll_loop():
            log.info(f"🤖 Telegram Screenshot Bot started (polling every {self.poll_interval}s)")

            while self._running:
                try:
                    processed = self.poll_once()
                    if processed > 0:
                        log.info(f"Processed {processed} screenshots")
                except Exception as e:
                    log.error(f"Poll error: {e}")

                time.sleep(self.poll_interval)

        self._thread = threading.Thread(target=_poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background polling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        release_lock()
        log.info("Telegram Screenshot Bot stopped")

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'processed' THEN 1 ELSE 0 END) as processed,
                SUM(CASE WHEN status = 'no_trade' THEN 1 ELSE 0 END) as no_trade,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as errors
            FROM telegram_screenshots
        """)

        row = cursor.fetchone()
        conn.close()

        return {
            "total": row[0] or 0,
            "processed": row[1] or 0,
            "no_trade": row[2] or 0,
            "errors": row[3] or 0,
            "last_update_id": self._last_update_id,
        }


# Singleton instance
_bot: Optional[TelegramScreenshotBot] = None


def get_telegram_screenshot_bot() -> TelegramScreenshotBot:
    """Get singleton bot instance."""
    global _bot
    if _bot is None:
        _bot = TelegramScreenshotBot()
    return _bot


def start_telegram_screenshot_bot():
    """Start the bot (call from main.py)."""
    bot = get_telegram_screenshot_bot()
    bot.start()
    return bot


# CLI entry point
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("TELEGRAM SCREENSHOT BOT - Trade Learning")
    print("=" * 60)
    print()
    print("Send trade screenshots to your Telegram bot.")
    print("The snake will learn from them automatically.")
    print()
    print("Press Ctrl+C to stop.")
    print()

    bot = TelegramScreenshotBot(poll_interval=5)
    bot.start()

    try:
        while True:
            time.sleep(60)
            stats = bot.get_stats()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Stats: {stats}")
    except KeyboardInterrupt:
        bot.stop()
        print("\nStopped.")
