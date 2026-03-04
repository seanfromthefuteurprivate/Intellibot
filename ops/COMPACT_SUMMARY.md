# READ THIS FIRST AFTER COMPACTION

You are continuing BEAST MODE deployment for WSB-Snake.

Read /home/ubuntu/wsb-snake/ops/HANDOFF.md for full state.

## IMMEDIATE NEXT STEPS:
1. Add Signal 10 (Predator Vision) to _check_entry_quality
2. Remove IWM from watchlist (line ~75 in jobs_day_cpl.py)
3. Update kill switch to $10K/-$750 (lines 86-87)
4. Git commit + push + pull to EC2
5. Restart wsb-snake service
6. Run simulated scan showing all gates

## CURRENT STATE:
- Beast Mode V3.0 with 9 conviction signals LOCAL (not on EC2 yet)
- Session halt REMOVED
- SPY block REMOVED
- HYDRA integration COMPLETE
- Compiles clean

## DEPLOY COMMANDS:
```bash
git add -A && git commit -m "Beast Mode V3: 9-signal conviction"
git push origin main
aws ssm send-command --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["cd /home/ubuntu/wsb-snake && git pull && systemctl restart wsb-snake"]'
```

## KEY CONSTANTS (jobs_day_cpl.py):
- SNIPER_CAPITAL = 2500
- MAX_OPEN_POSITIONS = 1
- DAILY_PROFIT_TARGET = 2500 (change to 10000)
- DAILY_MAX_LOSS = -500 (change to -750)
- MIN_CONVICTION = 4
