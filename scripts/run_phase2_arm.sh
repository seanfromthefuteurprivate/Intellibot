#!/bin/bash
# Phase 2 arm: dry-run, 3 calls. On Friday use for pre-open check; full send is phase3.
# Use python3 so cron (minimal PATH) finds it.
cd /Users/seankuesia/Downloads/Intellibot
python3 run_snake_cpl.py --dry-run --max-calls 3 --untruncated-tails >> cpl_phase2.log 2>&1
echo "$(date -Iseconds) phase2 exit=$?" >> cpl_phase2.log
