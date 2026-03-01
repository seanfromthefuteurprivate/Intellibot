#!/usr/bin/env python3
"""Telegram screenshot ingest - raw, no framework, runs via cron."""
import os, json, base64, hashlib, sqlite3, requests
from datetime import datetime

# Load env
for line in open('.env'):
    if '=' in line and not line.startswith('#'):
        k, v = line.strip().split('=', 1)
        os.environ[k] = v

TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
OPENAI_KEY = os.environ['OPENAI_API_KEY']
DB_PATH = 'wsb_snake_data/wsb_snake.db'
STATE_FILE = '/tmp/telegram_last_update_id'

def get_last_id():
    try: return int(open(STATE_FILE).read().strip())
    except: return 0

def save_last_id(uid):
    open(STATE_FILE, 'w').write(str(uid))

def ocr_image(b64_img):
    prompt = """Extract trade data from this screenshot. Return JSON only:
{"ticker":"SPY","trade_type":"CALLS","strike":590,"expiry":"2026-02-28","direction":"long",
"entry_price":2.50,"exit_price":4.00,"profit_loss":150,"profit_loss_pct":60,"platform":"Robinhood","notes":"any observations"}
If you can't extract trade data, return {"error":"reason"}. Return ONLY valid JSON, no markdown."""
    r = requests.post('https://api.openai.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {OPENAI_KEY}'},
        json={'model': 'gpt-4o', 'max_tokens': 500, 'messages': [
            {'role': 'user', 'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64_img}'}}
            ]}
        ]}, timeout=60)
    txt = r.json()['choices'][0]['message']['content']
    txt = txt.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
    return json.loads(txt)

def insert_trade(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO learned_trades
        (ticker, trade_type, strike, entry_price, exit_price, profit_loss, profit_loss_pct,
         detected_pattern, setup_description, confidence_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (data.get('ticker','?'), data.get('trade_type','?'),
         data.get('strike'), data.get('entry_price'), data.get('exit_price'),
         data.get('profit_loss'), data.get('profit_loss_pct'),
         f"TELEGRAM_{data.get('trade_type','')}", data.get('notes',''),
         0.9 if data.get('ticker') else 0.3))
    trade_id = c.lastrowid
    conn.commit()
    conn.close()
    return trade_id

def main():
    last_id = get_last_id()
    r = requests.get(f'https://api.telegram.org/bot{TOKEN}/getUpdates',
        params={'offset': last_id + 1, 'timeout': 5}, timeout=10)
    updates = r.json().get('result', [])

    processed = 0
    for u in updates:
        uid = u['update_id']
        msg = u.get('message', {})
        photos = msg.get('photo', [])
        if not photos:
            save_last_id(uid)
            continue

        file_id = photos[-1]['file_id']
        fr = requests.get(f'https://api.telegram.org/bot{TOKEN}/getFile', params={'file_id': file_id}, timeout=10)
        file_path = fr.json()['result']['file_path']
        img = requests.get(f'https://api.telegram.org/file/bot{TOKEN}/{file_path}', timeout=30).content
        b64 = base64.b64encode(img).decode()

        try:
            data = ocr_image(b64)
            if data.get('error'):
                print(f"⚠️  Could not extract: {data['error']}")
            else:
                trade_id = insert_trade(data)
                pnl = data.get('profit_loss_pct', 0) or 0
                print(f"✅ #{trade_id} | {data.get('ticker','?')} {data.get('trade_type','?')} ${data.get('strike','')} | P&L: {pnl:+.1f}% | {data.get('platform','?')}")
                processed += 1
                chat_id = msg['chat']['id']
                requests.post(f'https://api.telegram.org/bot{TOKEN}/sendMessage',
                    json={'chat_id': chat_id, 'text': f"✅ Learned: {data.get('ticker')} {data.get('trade_type')} {pnl:+.1f}%"}, timeout=5)
        except Exception as e:
            print(f"❌ OCR failed: {e}")

        save_last_id(uid)

    print(f"\n{'='*50}\nProcessed {processed} screenshots\n{'='*50}")

if __name__ == '__main__':
    main()
