#!/usr/bin/env python3
"""One-time script to process pending Telegram screenshots."""

import os
import sys
import time

# Load env
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k] = v

from wsb_snake.collectors.telegram_screenshot_bot import TelegramScreenshotBot, release_lock
import pathlib

# Clear lock
pathlib.Path('/tmp/telegram_screenshot_bot.lock').unlink(missing_ok=True)

bot = TelegramScreenshotBot(poll_interval=2)

print('=' * 60)
print('Processing Telegram Screenshots...')
print('=' * 60)

total_processed = 0
rounds = 0
consecutive_empty = 0

while rounds < 30 and consecutive_empty < 5:
    rounds += 1
    try:
        processed = bot.poll_once()
        if processed > 0:
            total_processed += processed
            consecutive_empty = 0
            print(f'Round {rounds}: Processed {processed} screenshot(s) (total: {total_processed})')
        else:
            consecutive_empty += 1
            if total_processed > 0:
                print(f'Round {rounds}: No new screenshots (waiting...)')
        time.sleep(2)
    except Exception as e:
        print(f'Error in round {rounds}: {e}')
        import traceback
        traceback.print_exc()
        break

print()
print('=' * 60)
print(f'TOTAL PROCESSED: {total_processed}')
print('=' * 60)
print(f'Stats: {bot.get_stats()}')

# Now show what was extracted
print()
print('=' * 60)
print('EXTRACTED TRADE DATA')
print('=' * 60)

from wsb_snake.db.database import get_connection
import json

conn = get_connection()
cursor = conn.cursor()
cursor.execute("""
    SELECT id, file_id, from_user, status, extracted_data, received_at
    FROM telegram_screenshots
    ORDER BY id DESC
    LIMIT 15
""")
rows = cursor.fetchall()
conn.close()

for row in rows:
    print(f"\n--- Screenshot #{row['id']} ({row['status']}) ---")
    if row['extracted_data']:
        data = json.loads(row['extracted_data'])
        ticker = data.get('ticker', '?')
        trade_type = data.get('trade_type', '?')
        strike = data.get('strike', '')
        pnl = data.get('profit_loss', 0)
        pnl_pct = data.get('profit_loss_pct', 0)
        direction = data.get('direction', '?')
        platform = data.get('platform', '?')

        strike_str = f" ${strike}" if strike else ""
        pnl_str = f"${pnl:+,.2f}" if pnl else "N/A"
        pct_str = f"({pnl_pct:+.1f}%)" if pnl_pct else ""

        print(f"  Ticker: {ticker} {trade_type}{strike_str}")
        print(f"  Direction: {direction}")
        print(f"  P&L: {pnl_str} {pct_str}")
        print(f"  Platform: {platform}")
    else:
        print(f"  No data extracted")

release_lock()
print("\nDone.")
