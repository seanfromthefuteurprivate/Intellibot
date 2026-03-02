# WSB Snake - Claude Code Instructions

## Infrastructure (AWS EC2)

| Resource | Details |
|----------|---------|
| **Instance ID** | i-03f3a7c46ec809a43 |
| **Region** | us-east-1 |
| **Path** | /home/ubuntu/wsb-snake |
| **Access** | AWS SSM (no SSH key needed) |

## Services Running

| Service | Description |
|---------|-------------|
| wsb-snake | Main trading engine |
| wsb-ops-monitor | Health monitoring agent |
| wsb-ops-audit | Order attribution & audit agent |
| wsb-ops-deploy | Auto-deploy agent |

## AWS SSM Commands

```bash
# Run any command on EC2
aws ssm send-command \
  --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters '{"commands":["YOUR_COMMAND_HERE"]}' \
  --region us-east-1

# Deploy latest code
aws ssm send-command \
  --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters '{"commands":["cd /home/ubuntu/wsb-snake && git pull origin main && systemctl restart wsb-snake"]}' \
  --region us-east-1

# Check service status
aws ssm send-command \
  --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters '{"commands":["systemctl status wsb-snake wsb-ops-monitor wsb-ops-audit wsb-ops-deploy --no-pager"]}' \
  --region us-east-1

# View logs
aws ssm send-command \
  --instance-ids i-03f3a7c46ec809a43 \
  --document-name "AWS-RunShellScript" \
  --parameters '{"commands":["journalctl -u wsb-snake --no-pager -n 100"]}' \
  --region us-east-1

# Get command output (replace COMMAND_ID)
aws ssm get-command-invocation \
  --command-id "COMMAND_ID" \
  --instance-id "i-03f3a7c46ec809a43" \
  --region us-east-1
```

## Project Context

- **Git Repo:** wsb_snake trading bot
- **Main Entry:** `wsb_snake/main.py`
- **Config:** `.env` file (never commit)
- **Database:** SQLite at `wsb_snake_data/wsb_snake.db`

## Key Files

- `wsb_snake/main.py` - Entry point
- `wsb_snake/engines/v7_scalper.py` - V7 scalping engine
- `wsb_snake/execution/jobs_day_cpl.py` - CPL engine with direction lock
- `wsb_snake/trading/alpaca_executor.py` - Trade execution
- `ops/monitor_agent.py` - Health monitoring
- `ops/audit_agent.py` - Order attribution
- `ops/deploy_agent.py` - Auto-deploy

## Autonomous Control Plane

Three agents run 24/7 watching the system:

1. **Monitor Agent** - Health checks, P&L tracking, position snapshots
2. **Audit Agent** - Order attribution (CPL/V7), direction locks, daily reports
3. **Deploy Agent** - Auto-deploy on git push, syntax validation, auto-rollback
