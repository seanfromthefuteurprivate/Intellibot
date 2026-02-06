#!/usr/bin/env bash
# Backup WSB Snake persistent state (SQLite + session_learnings.json).
# Run daily after market close (e.g. cron 0 18 * * * or 22:15 ET).
# Keeps last 30 backups.

set -e
DATA_DIR="${WSB_SNAKE_DATA_DIR:-wsb_snake_data}"
BACKUP_DIR="${BACKUP_DIR:-$DATA_DIR/backups}"
KEEP=30

mkdir -p "$BACKUP_DIR"
TS=$(date +%Y%m%d_%H%M)

if [ -f "$DATA_DIR/wsb_snake.db" ]; then
  cp "$DATA_DIR/wsb_snake.db" "$BACKUP_DIR/wsb_snake_$TS.db"
fi
if [ -f "$DATA_DIR/session_learnings.json" ]; then
  cp "$DATA_DIR/session_learnings.json" "$BACKUP_DIR/session_learnings_$TS.json"
fi

# Keep last KEEP backups
ls -t "$BACKUP_DIR"/wsb_snake_*.db 2>/dev/null | tail -n +$((KEEP+1)) | xargs -r rm --
ls -t "$BACKUP_DIR"/session_learnings_*.json 2>/dev/null | tail -n +$((KEEP+1)) | xargs -r rm --
