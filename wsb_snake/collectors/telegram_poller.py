#!/usr/bin/env python3
"""Telegram screenshot poller - GPT-4o Vision OCR, aggressive extraction, no mercy."""
import os, json, time, sqlite3, base64, requests
from openai import OpenAI

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DB_PATH = os.environ.get("DB_PATH", "/home/ubuntu/wsb-snake/wsb_snake_data/wsb_snake.db")
OFFSET_FILE = "/home/ubuntu/wsb-snake/telegram_offset.txt"
POLL_INTERVAL = 30

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

OCR_PROMPT = """You are a financial trade data extraction expert. Extract ALL trade and position information from this brokerage screenshot.

This image could be from ANY platform: Robinhood (green/white or dark mode), E*TRADE (dark purple UI), Webull (dark UI with red/green), Interactive Brokers (IBKR - gray/blue UI), ThinkorSwim, Schwab, or a Reddit/Twitter post containing a brokerage screenshot.

Extract EVERY piece of financial data you can see. Return JSON:
{
  "trades": [
    {
      "ticker": "SPY",
      "trade_type": "CALL",
      "strike": 590,
      "expiry": "2026-03-20",
      "direction": "long",
      "action": "buy",
      "entry_price": 2.50,
      "exit_price": 4.00,
      "quantity": 10,
      "total_cost": 2500,
      "total_credit": 4000,
      "profit_loss_dollars": 1500,
      "profit_loss_pct": 60.0,
      "platform": "Robinhood",
      "date": "2026-02-26",
      "is_0dte": false,
      "is_open": false,
      "notes": "any context"
    }
  ],
  "account_summary": {
    "total_value": 150608,
    "daily_gain_dollars": 67942,
    "daily_gain_pct": 82.19,
    "buying_power": 190214
  },
  "source": "robinhood",
  "confidence": 0.95
}

CRITICAL RULES:
1. If you see ANY ticker, ANY dollar amount, ANY percentage — extract it. NEVER return empty trades unless the image is truly not financial.
2. Multiple trades in one screenshot? Return ALL of them.
3. Account summary screenshots (showing total value, daily P&L) — fill account_summary even if no individual trades visible.
4. Reddit posts — extract the brokerage data visible inside the embedded screenshot. Also note the Reddit username and upvote count in notes.
5. Position views showing unrealized gains ARE trades — extract them with is_open: true.
6. Trade history lists — extract EVERY line item as a separate trade.
7. If a field isn't visible, use null — do NOT reject the entire screenshot.
8. For E*TRADE positions showing stocks (not options), set trade_type to 'STOCK' and strike to null.
9. For options, parse the strike and expiry from the contract name (e.g., 'XOM Jan 21 28 $125 Call' = ticker XOM, strike 125, expiry 2028-01-21, type CALL).
10. Confidence should reflect how much data you could extract, not whether it's a 'valid trade.'

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

def ocr_image_gpt4o(b64_image):
    """OCR with GPT-4o Vision - already authenticated and working."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                    {"type": "text", "text": OCR_PROMPT}
                ]
            }],
            max_tokens=2000
        )
        txt = response.choices[0].message.content
        print(f"[GPT-4o] Raw response: {txt[:300]}...")
        txt = txt.replace("```json", "").replace("```", "").strip()
        return json.loads(txt)
    except json.JSONDecodeError as e:
        print(f"[GPT-4o] JSON parse error: {e}, raw: {txt[:500]}")
        return {"trades": [], "confidence": 0, "raw_text": txt[:500]}
    except Exception as e:
        print(f"[GPT-4o] OCR error: {e}")
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
        reply(chat_id, "Failed to download image")
        return
    data = ocr_image_gpt4o(b64)
    trades = data.get("trades", [])
    confidence = data.get("confidence", 0)
    account = data.get("account_summary", {})

    # Handle account summary even if no trades
    if account and account.get("total_value"):
        summary = f"*Account Summary*\nTotal: ${account.get('total_value', 0):,.2f}\nDaily: ${account.get('daily_gain_dollars', 0):+,.2f} ({account.get('daily_gain_pct', 0):+.2f}%)"
        if not trades:
            reply(chat_id, f"{summary}\n\n_Confidence: {confidence:.0%}_")
            return

    if not trades:
        raw = data.get("raw_text", "")[:200]
        reply(chat_id, f"No trades found (confidence: {confidence:.0%})\n\n{raw}")
        return

    for trade in trades:
        insert_trade(trade, json.dumps(data.get("account_summary", {})), confidence, chat_id, msg_id)
        print(f"[TELEGRAM] Saved: {trade.get('ticker')} {trade.get('strike')} {trade.get('trade_type')} P/L: {trade.get('profit_loss_pct')}%")

    summary = "\n".join([f"• {t.get('ticker')} ${t.get('strike') or 'STOCK'} {t.get('trade_type') or ''} P/L: {t.get('profit_loss_pct') or 'N/A'}%" for t in trades])
    reply(chat_id, f"*{len(trades)} Trade(s) Captured*\n\n{summary}\n\n_Confidence: {confidence:.0%}_")

if __name__ == "__main__":
    print(f"[TELEGRAM POLLER] Starting - GPT-4o Vision OCR - polling every {POLL_INTERVAL}s")
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
