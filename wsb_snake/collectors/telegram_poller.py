#!/usr/bin/env python3
"""Telegram screenshot poller - bare loop, no frameworks, no mercy."""
import os, json, time, sqlite3, base64, requests

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
DB_PATH = os.environ.get("DB_PATH", "/home/ubuntu/wsb-snake/wsb_snake_data/wsb_snake.db")
OFFSET_FILE = "/tmp/telegram_offset.txt"
POLL_INTERVAL = 30

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

def ocr_image(b64_image):
    r = requests.post("https://api.openai.com/v1/chat/completions", headers={
        "Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"
    }, json={
        "model": "gpt-4o", "max_tokens": 500,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Extract trade data from this screenshot. Return JSON only: {\"ticker\": \"SPY\", \"strike\": 590, \"direction\": \"CALL\" or \"PUT\", \"entry_price\": 1.50, \"exit_price\": 2.25, \"pnl_pct\": 50.0, \"expiry\": \"2026-03-01\", \"notes\": \"any extra context\"}. If not a trade screenshot, return {\"error\": \"not a trade\"}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
        ]}]
    }, timeout=60)
    try:
        txt = r.json()["choices"][0]["message"]["content"]
        txt = txt.replace("```json", "").replace("```", "").strip()
        return json.loads(txt)
    except:
        return {"error": "parse failed"}

def insert_trade(data, chat_id, msg_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS learned_trades (
        id INTEGER PRIMARY KEY, ticker TEXT, strike REAL, direction TEXT, entry_price REAL,
        exit_price REAL, pnl_pct REAL, expiry TEXT, notes TEXT, source TEXT, created_at TEXT,
        telegram_chat_id INTEGER, telegram_msg_id INTEGER)""")
    conn.execute("""INSERT INTO learned_trades (ticker, strike, direction, entry_price, exit_price,
        pnl_pct, expiry, notes, source, created_at, telegram_chat_id, telegram_msg_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'telegram', datetime('now'), ?, ?)""",
        (data.get("ticker"), data.get("strike"), data.get("direction"), data.get("entry_price"),
         data.get("exit_price"), data.get("pnl_pct"), data.get("expiry"), data.get("notes"), chat_id, msg_id))
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
    data = ocr_image(b64)
    if "error" in data:
        reply(chat_id, f"❌ {data['error']}")
        return
    insert_trade(data, chat_id, msg_id)
    reply(chat_id, f"✅ *Trade Captured*\n`{data.get('ticker')} ${data.get('strike')} {data.get('direction')}`\nP&L: {data.get('pnl_pct', 0):+.1f}%")
    print(f"[TELEGRAM] Saved: {data.get('ticker')} {data.get('strike')} {data.get('direction')} {data.get('pnl_pct')}%")

if __name__ == "__main__":
    print(f"[TELEGRAM POLLER] Starting - polling every {POLL_INTERVAL}s")
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
            print(f"[TELEGRAM] Error: {e}")
        time.sleep(POLL_INTERVAL)
