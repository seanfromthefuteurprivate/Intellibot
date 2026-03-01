#!/usr/bin/env python3
"""Telegram screenshot poller - Bedrock Claude Sonnet OCR, no OpenAI, no mercy."""
import os, json, time, sqlite3, base64, requests, boto3

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DB_PATH = os.environ.get("DB_PATH", "/home/ubuntu/wsb-snake/wsb_snake_data/wsb_snake.db")
OFFSET_FILE = "/home/ubuntu/wsb-snake/telegram_offset.txt"
POLL_INTERVAL = 30

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

OCR_PROMPT = """You are a trade data extraction expert. Extract ANY trade information visible in this screenshot. This could be from ANY brokerage platform (Robinhood, Webull, ThinkorSwim, Schwab, IBKR, Alpaca, etc.) or a cropped screenshot from Reddit/Twitter/Discord.

Extract ALL of the following that you can find. If a field isn't visible, use null — do NOT reject the screenshot:
{
  "trades": [
    {
      "ticker": "SPY",
      "trade_type": "CALL or PUT",
      "strike": 590,
      "expiry": "2026-02-28",
      "direction": "long or short",
      "entry_price": 2.50,
      "exit_price": 4.00,
      "quantity": 10,
      "total_cost": 2500,
      "total_credit": 4000,
      "profit_loss_dollars": 1500,
      "profit_loss_pct": 60,
      "platform": "Robinhood",
      "date": "2026-02-26",
      "is_0dte": true,
      "notes": "any other context visible"
    }
  ],
  "raw_text": "any other relevant text visible in the screenshot",
  "confidence": 0.85
}

CRITICAL RULES:
- If you can see ANY ticker symbol and ANY financial data, extract it. Do NOT return empty trades.
- A screenshot showing P&L IS a trade. A screenshot showing positions IS a trade. A screenshot showing filled orders IS a trade.
- Multiple trades in one screenshot? Return all of them in the trades array.
- If it's truly not financial (a meme, a text conversation about non-trading topics), ONLY THEN return {"trades": [], "raw_text": "description", "confidence": 0}
- Partial data is better than no data. If you can see a ticker and a P&L but not the strike, still extract what you can.

Return ONLY valid JSON, no markdown."""

def load_offset():
    try:
        return int(open(OFFSET_FILE).read().strip())
    except:
        return 0

def save_offset(offset):
    open(OFFSET_FILE, "w").write(str(offset))

def download_file(file_id):
    r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile", params={"file_id": file_id})
    path = r.json().get("result", {}).get("file_path", "")
    if not path:
        return None
    r2 = requests.get(f"https://api.telegram.org/file/bot{BOT_TOKEN}/{path}")
    return base64.b64encode(r2.content).decode() if r2.ok else None

def ocr_image_bedrock(b64_image):
    """OCR with AWS Bedrock Claude 3 Haiku - fast and already enabled."""
    try:
        # Claude 3 Haiku is ACTIVE and doesn't require use case form
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_image}},
                    {"type": "text", "text": OCR_PROMPT}
                ]}]
            })
        )
        result = json.loads(response['body'].read())
        txt = result['content'][0]['text']
        print(f"[BEDROCK] Raw response: {txt[:200]}...")
        txt = txt.replace("```json", "").replace("```", "").strip()
        return json.loads(txt)
    except json.JSONDecodeError as e:
        print(f"[BEDROCK] JSON parse error: {e}, raw: {txt[:300]}")
        return {"trades": [], "confidence": 0, "raw_text": txt[:500]}
    except Exception as e:
        print(f"[BEDROCK] OCR error: {e}")
        return {"trades": [], "confidence": 0}

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS learned_trades (
        id INTEGER PRIMARY KEY, ticker TEXT, trade_type TEXT, strike REAL, expiry TEXT,
        direction TEXT, entry_price REAL, exit_price REAL, quantity INTEGER,
        profit_loss_dollars REAL, profit_loss_pct REAL, platform TEXT, trade_date TEXT,
        is_0dte INTEGER, notes TEXT, raw_text TEXT, confidence REAL,
        source TEXT DEFAULT 'telegram', created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        telegram_chat_id INTEGER, telegram_msg_id INTEGER, catalyst TEXT, pattern TEXT)""")
    conn.commit()
    conn.close()

def insert_trade(trade, raw_text, confidence, chat_id, msg_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO learned_trades (ticker, trade_type, strike, expiry, direction,
        entry_price, exit_price, quantity, profit_loss_dollars, profit_loss_pct, platform,
        trade_date, is_0dte, notes, raw_text, confidence, telegram_chat_id, telegram_msg_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (trade.get("ticker"), trade.get("trade_type"), trade.get("strike"), trade.get("expiry"),
         trade.get("direction"), trade.get("entry_price"), trade.get("exit_price"),
         trade.get("quantity"), trade.get("profit_loss_dollars"), trade.get("profit_loss_pct"),
         trade.get("platform"), trade.get("date"), 1 if trade.get("is_0dte") else 0,
         trade.get("notes"), raw_text, confidence, chat_id, msg_id))
    conn.commit()
    conn.close()

def reply(chat_id, text):
    requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})

def process_update(update):
    msg = update.get("message", {})
    chat_id, msg_id = msg.get("chat", {}).get("id"), msg.get("message_id")
    file_id = None
    if "photo" in msg:
        file_id = msg["photo"][-1]["file_id"]
    elif "document" in msg and msg["document"].get("mime_type", "").startswith("image"):
        file_id = msg["document"]["file_id"]
    if not file_id:
        return
    print(f"[TELEGRAM] Processing image from chat {chat_id}")
    b64 = download_file(file_id)
    if not b64:
        reply(chat_id, "❌ Failed to download image")
        return
    data = ocr_image_bedrock(b64)
    trades = data.get("trades", [])
    confidence = data.get("confidence", 0)
    if not trades:
        reply(chat_id, f"🤔 No trades found (confidence: {confidence:.0%})\n\n{data.get('raw_text', '')[:200]}")
        return
    for trade in trades:
        insert_trade(trade, data.get("raw_text", ""), confidence, chat_id, msg_id)
        print(f"[TELEGRAM] Saved: {trade.get('ticker')} {trade.get('strike')} {trade.get('trade_type')} P/L: {trade.get('profit_loss_pct')}%")
    summary = "\n".join([f"• {t.get('ticker')} ${t.get('strike')} {t.get('trade_type')} P/L: {t.get('profit_loss_pct') or 'N/A'}%" for t in trades])
    reply(chat_id, f"✅ *{len(trades)} Trade(s) Captured*\n\n{summary}\n\n_Confidence: {confidence:.0%}_")

if __name__ == "__main__":
    print(f"[TELEGRAM POLLER] Starting - Bedrock Claude Sonnet OCR - polling every {POLL_INTERVAL}s")
    init_db()
    offset = load_offset()
    while True:
        try:
            r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                params={"offset": offset, "timeout": 25}, timeout=30)
            for update in r.json().get("result", []):
                process_update(update)
                offset = update["update_id"] + 1
                save_offset(offset)
        except Exception as e:
            print(f"[TELEGRAM] Poll error: {e}")
        time.sleep(POLL_INTERVAL)
