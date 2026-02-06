#!/bin/bash
# Phase 3: full send at 9:30 AM (cron) — CPL runs from market open (9:30) to close (16:00).
# Loop 60s. Untruncated tails = sequential $250 → 5 figures.
# Use python3 so cron (minimal PATH) finds it.
#
# CRON: run at 9:30 AM every weekday (Mon–Fri):
#   30 9 * * 1-5 /Users/seankuesia/Downloads/Intellibot/scripts/run_phase3_go_live.sh
# To install: crontab -e  then paste the line above (adjust path if needed).
# To confirm: crontab -l
#
cd /Users/seankuesia/Downloads/Intellibot
nohup python3 run_snake_cpl.py --broadcast --loop 60 --untruncated-tails > cpl_output.log 2>&1 &
echo "$(date -Iseconds) CPL LIVE — FULL SEND STARTED pid=$!" >> cpl_output.log
