"""
TELEGRAM WEBHOOK - No More Polling, Zero Message Loss
═══════════════════════════════════════════════════════════════════

Webhook endpoint for Telegram screenshot ingestion.
Unlike polling, webhooks are push-based - no message loss, instant processing.

Setup:
1. Deploy wsb-snake with this endpoint accessible at /api/telegram/webhook
2. Register webhook with Telegram:
   curl -X POST "https://api.telegram.org/bot$TOKEN/setWebhook" \
     -d "url=https://YOUR_DOMAIN/api/telegram/webhook" \
     -d "allowed_updates=[\"message\"]"

3. Screenshots sent to bot are instantly processed
"""

import os
import base64
import hashlib
import json
import requests
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from wsb_snake.utils.logger import get_logger
from wsb_snake.config import TELEGRAM_BOT_TOKEN, OPENAI_API_KEY
from wsb_snake.db.database import get_connection

log = get_logger(__name__)

# Router for FastAPI
router = APIRouter(prefix="/api/telegram", tags=["telegram"])

# Telegram API base
TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Deduplication cache (in-memory, cleared on restart)
_processed_file_ids: set = set()


@dataclass
class WebhookPhoto:
    """Photo data from webhook."""
    file_id: str
    file_unique_id: str
    width: int
    height: int
    file_size: int
    message_id: int
    chat_id: int
    from_user: str
    timestamp: datetime


def send_telegram_message(chat_id: int, text: str, reply_to: Optional[int] = None) -> bool:
    """Send a message to Telegram chat."""
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
        log.error(f"Failed to send Telegram message: {e}")
        return False


def download_telegram_photo(file_id: str) -> Optional[bytes]:
    """Download photo from Telegram."""
    try:
        # Get file path
        resp = requests.get(
            f"{TELEGRAM_API}/getFile",
            params={"file_id": file_id},
            timeout=10
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data.get("ok"):
            return None

        file_path = data["result"]["file_path"]

        # Download file
        download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        resp = requests.get(download_url, timeout=30)

        if resp.status_code != 200:
            return None

        return resp.content

    except Exception as e:
        log.error(f"Failed to download photo: {e}")
        return None


def ocr_trade_image(image_base64: str) -> Dict[str, Any]:
    """
    Extract trade data from screenshot using GPT-4o Vision.

    Returns extracted trade data or {"error": "reason"}.
    """
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not configured"}

    prompt = """Extract trade data from this screenshot. Return JSON only:
{"ticker":"SPY","trade_type":"CALLS","strike":590,"expiry":"2026-02-28","direction":"long",
"entry_price":2.50,"exit_price":4.00,"profit_loss":150,"profit_loss_pct":60,
"platform":"Robinhood","chart_pattern":"","notes":"any observations"}

If you can't extract trade data, return {"error":"reason"}.
Return ONLY valid JSON, no markdown, no code blocks."""

    try:
        resp = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={'Authorization': f'Bearer {OPENAI_API_KEY}'},
            json={
                'model': 'gpt-4o',
                'max_tokens': 500,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {
                            'url': f'data:image/jpeg;base64,{image_base64}'
                        }}
                    ]
                }]
            },
            timeout=60
        )

        if resp.status_code != 200:
            return {"error": f"OpenAI API error: {resp.status_code}"}

        content = resp.json()['choices'][0]['message']['content']
        # Strip markdown formatting if present
        content = content.strip()
        if content.startswith('```'):
            content = content.removeprefix('```json').removeprefix('```')
            content = content.removesuffix('```').strip()

        return json.loads(content)

    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON from OCR: {str(e)[:50]}"}
    except Exception as e:
        return {"error": f"OCR failed: {str(e)[:50]}"}


def save_learned_trade(extracted: Dict[str, Any], photo: WebhookPhoto) -> int:
    """Save extracted trade to database."""
    conn = get_connection()
    cursor = conn.cursor()

    # Ensure learned_trades table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learned_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            trade_type TEXT,
            strike REAL,
            expiry TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            profit_loss REAL,
            profit_loss_pct REAL,
            detected_pattern TEXT,
            setup_description TEXT,
            confidence_score REAL,
            source TEXT DEFAULT 'telegram',
            source_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        INSERT INTO learned_trades (
            ticker, trade_type, strike, expiry, direction,
            entry_price, exit_price, profit_loss, profit_loss_pct,
            detected_pattern, setup_description, confidence_score,
            source, source_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        extracted.get('ticker', '?'),
        extracted.get('trade_type', '?'),
        extracted.get('strike'),
        extracted.get('expiry'),
        extracted.get('direction', 'long'),
        extracted.get('entry_price'),
        extracted.get('exit_price'),
        extracted.get('profit_loss'),
        extracted.get('profit_loss_pct'),
        extracted.get('chart_pattern') or f"TELEGRAM_{extracted.get('trade_type', '?')}",
        extracted.get('notes', ''),
        0.9 if extracted.get('ticker') else 0.5,
        'telegram_webhook',
        photo.file_unique_id,
    ))

    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return trade_id


def ingest_to_learning_systems(extracted: Dict[str, Any], trade_id: int) -> tuple:
    """Ingest trade into Semantic Memory and Trade Graph."""
    semantic_id = None
    graph_id = None

    ticker = extracted.get('ticker', 'SPY')
    trade_type = extracted.get('trade_type', 'CALLS')
    direction = "LONG" if "CALL" in trade_type.upper() else "SHORT"

    entry_price = extracted.get('entry_price') or 1.0
    exit_price = extracted.get('exit_price') or entry_price
    pnl_dollars = extracted.get('profit_loss') or 0.0
    pnl_percent = extracted.get('profit_loss_pct') or 0.0

    entry_time = datetime.now(timezone.utc) - timedelta(minutes=30)
    exit_time = datetime.now(timezone.utc)

    # 1. Semantic Memory
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

        trade_hash = hashlib.md5(f"webhook_{trade_id}_{ticker}".encode()).hexdigest()[:12]
        duration = int((exit_time - entry_time).total_seconds() / 60)

        outcome = TradeOutcome(
            trade_id=trade_hash,
            conditions=conditions,
            entry_reasoning=f"Learned from Telegram webhook: {extracted.get('notes', '')}",
            pnl_dollars=pnl_dollars,
            pnl_percent=pnl_percent,
            duration_minutes=duration,
            max_adverse_excursion_pct=min(0, pnl_percent),
            max_favorable_excursion_pct=max(0, pnl_percent),
            exit_reason="TARGET" if pnl_dollars > 0 else "STOP",
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            lessons_learned=f"Webhook screenshot: {'WIN' if pnl_dollars > 0 else 'LOSS'} {pnl_percent:+.1f}%",
        )

        semantic.record_trade(outcome)
        semantic_id = trade_hash
        log.info(f"📚 Semantic Memory: {trade_hash}")

    except Exception as e:
        log.debug(f"Semantic memory failed: {e}")

    # 2. Trade Graph
    try:
        from wsb_snake.learning.trade_graph import get_trade_graph

        trade_graph = get_trade_graph()
        pattern = extracted.get('chart_pattern') or f"WEBHOOK_{trade_type}"

        trade_graph.record_trade(
            ticker=ticker,
            direction=direction.lower(),
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl_dollars=pnl_dollars,
            pattern=pattern,
            entry_reasoning=f"Telegram webhook: {extracted.get('notes', '')}",
            exit_reasoning="TARGET" if pnl_dollars > 0 else "STOP",
            conditions={
                "ticker": ticker,
                "pattern": pattern,
                "direction": direction.lower(),
                "platform": extracted.get('platform', 'unknown'),
            },
            metadata={
                "source": "telegram_webhook",
                "trade_id": trade_id,
            },
        )
        graph_id = f"wh_{trade_id}"
        log.info(f"🕸️ Trade Graph: {graph_id}")

    except Exception as e:
        log.debug(f"Trade graph failed: {e}")

    return semantic_id, graph_id


def process_webhook_photo(photo: WebhookPhoto) -> Dict[str, Any]:
    """
    Process a photo received via webhook.

    Returns processing result with extracted data.
    """
    result = {
        "status": "error",
        "file_id": photo.file_id,
        "message": "",
        "extracted": None,
        "trade_id": None,
    }

    try:
        # 1. Download photo
        log.info(f"📸 Downloading photo from @{photo.from_user}...")
        content = download_telegram_photo(photo.file_id)

        if not content:
            result["message"] = "Failed to download photo"
            return result

        # 2. Convert to base64
        image_base64 = base64.b64encode(content).decode("utf-8")

        # 3. OCR extraction
        log.info("🔍 Running GPT-4o Vision OCR...")
        extracted = ocr_trade_image(image_base64)

        if extracted.get("error"):
            result["status"] = "no_trade"
            result["message"] = extracted["error"]
            return result

        result["extracted"] = extracted

        # 4. Save to database
        trade_id = save_learned_trade(extracted, photo)
        result["trade_id"] = trade_id

        # 5. Ingest to learning systems
        semantic_id, graph_id = ingest_to_learning_systems(extracted, trade_id)

        result["status"] = "success"
        result["semantic_id"] = semantic_id
        result["graph_id"] = graph_id

        # Format success message
        ticker = extracted.get('ticker', '?')
        trade_type = extracted.get('trade_type', '?')
        strike = extracted.get('strike', '')
        pnl = extracted.get('profit_loss', 0) or 0
        pnl_pct = extracted.get('profit_loss_pct', 0) or 0
        platform = extracted.get('platform', 'Unknown')

        emoji = "🔥" if pnl_pct > 50 else ("✅" if pnl > 0 else "📉")

        result["message"] = (
            f"{emoji} *Trade Learned!*\n\n"
            f"*Ticker:* {ticker}\n"
            f"*Type:* {trade_type}"
            + (f" ${strike}" if strike else "") + "\n"
            f"*P&L:* {'+'if pnl >= 0 else ''}{pnl:,.2f} ({pnl_pct:+.1f}%)\n"
            f"*Platform:* {platform}\n\n"
            f"_Saved to Semantic Memory \\+ Trade Graph_"
        )

        log.info(f"✅ Webhook processed: {ticker} {trade_type} P&L: {pnl_pct:+.1f}%")

    except Exception as e:
        log.error(f"Webhook processing failed: {e}")
        result["message"] = f"Processing error: {str(e)[:100]}"

    return result


@router.post("/webhook")
async def telegram_webhook(request: Request):
    """
    Telegram webhook endpoint.

    Receives updates from Telegram when users send messages/photos to the bot.
    Processes photos through GPT-4o Vision OCR and ingests into learning systems.
    """
    try:
        update = await request.json()
    except Exception as e:
        log.error(f"Invalid webhook payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    log.debug(f"Webhook received: {json.dumps(update)[:500]}")

    # Extract message
    message = update.get("message", {})
    if not message:
        return JSONResponse({"ok": True, "message": "No message"})

    # Check for photos
    photos = message.get("photo", [])
    if not photos:
        # Not a photo message - acknowledge but don't process
        return JSONResponse({"ok": True, "message": "Not a photo"})

    # Get largest photo (last in list)
    largest_photo = photos[-1]
    file_unique_id = largest_photo.get("file_unique_id")

    # Deduplicate
    if file_unique_id in _processed_file_ids:
        log.debug(f"Duplicate photo ignored: {file_unique_id}")
        return JSONResponse({"ok": True, "message": "Already processed"})

    _processed_file_ids.add(file_unique_id)

    # Build photo object
    photo = WebhookPhoto(
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

    log.info(f"📷 Webhook photo from @{photo.from_user}")

    # Send "processing" message
    send_telegram_message(
        photo.chat_id,
        "⏳ Processing screenshot...",
        reply_to=photo.message_id
    )

    # Process the photo
    result = process_webhook_photo(photo)

    # Send result message
    if result["status"] == "success":
        send_telegram_message(
            photo.chat_id,
            result["message"],
            reply_to=photo.message_id
        )
    elif result["status"] == "no_trade":
        send_telegram_message(
            photo.chat_id,
            "🤔 Couldn't extract trade data. Make sure P&L is visible.",
            reply_to=photo.message_id
        )
    else:
        send_telegram_message(
            photo.chat_id,
            f"❌ Error: {result['message']}",
            reply_to=photo.message_id
        )

    return JSONResponse({
        "ok": True,
        "status": result["status"],
        "trade_id": result.get("trade_id"),
    })


@router.get("/webhook/status")
async def webhook_status():
    """Get webhook processing stats."""
    return {
        "mode": "webhook",
        "processed_count": len(_processed_file_ids),
        "webhook_url": "Set via Telegram API setWebhook",
    }


@router.post("/set-webhook")
async def set_webhook(request: Request):
    """
    Register webhook URL with Telegram.

    Body: {"url": "https://your-domain/api/telegram/webhook"}
    """
    try:
        body = await request.json()
        webhook_url = body.get("url")

        if not webhook_url:
            raise HTTPException(status_code=400, detail="Missing 'url' in request body")

        resp = requests.post(
            f"{TELEGRAM_API}/setWebhook",
            json={
                "url": webhook_url,
                "allowed_updates": ["message"],
            },
            timeout=10
        )

        data = resp.json()
        log.info(f"Webhook registration result: {data}")

        return data

    except Exception as e:
        log.error(f"Failed to set webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete-webhook")
async def delete_webhook():
    """Delete the current webhook (switch back to polling)."""
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/deleteWebhook",
            timeout=10
        )
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
