# WSB Snake - Claude Code Instructions

## MANDATORY: Infrastructure Audit Rules

**BEFORE making any code changes or debugging errors, run a full infrastructure audit:**

### Audit Commands (Run ALL before any work)

```bash
# 1. List ALL App Platform deployments
doctl apps list

# 2. List ALL Droplets
doctl compute droplet list --format ID,Name,PublicIPv4,Status,Memory,VCPUs,Region

# 3. SSH into wsb-snake droplet and check
ssh root@157.245.240.99 "
  echo '=== SERVICES ===' && systemctl list-units --type=service --state=running | grep -E 'wsb|snake|guardian|dashboard'
  echo '=== PROCESSES ===' && ps aux | grep -E 'python|node' | grep -v grep
  echo '=== PORTS ===' && ss -tlnp
  echo '=== GIT ===' && cd /root/wsb-snake && git log --oneline -1
"

# 4. List managed databases
doctl databases list

# 5. Cross-reference: confirm NO duplicate services
```

### Rules

1. **NEVER** assume "VM unreachable" without checking ALL deployments (`doctl apps list` + `doctl compute droplet list`)
2. **NEVER** make code fixes before confirming infrastructure state
3. If you see `connection limit exceeded` errors → **FIRST** check for duplicate deployments, NOT fix code
4. **ALWAYS** present a full infrastructure map before proceeding with any work
5. **ONLY** touch wsb-snake droplet (157.245.240.99) - other droplets are separate projects

### Current Infrastructure State (2026-02-10)

| Resource | IP/URL | Status | Purpose | Scope |
|----------|--------|--------|---------|-------|
| **Droplet: wsb-snake** | 157.245.240.99 | ACTIVE | Trading bot + Guardian API | **IN SCOPE** |
| **Droplet: hydra-engine** | 64.23.144.49 | ACTIVE | Separate project (hydra dashboard) | OUT OF SCOPE |
| **App Platform** | - | DELETED | Was coral-app, caused duplicates | N/A |

### wsb-snake Droplet Services

| Service | Port | Process | Description |
|---------|------|---------|-------------|
| wsb-snake.service | 8081 | main.py | Trading engine |
| wsb-dashboard.service | 8080 | uvicorn dashboard | Dashboard API |
| vm-guardian.service | 8888 | run_guardian.py | Remote control API |
| nginx | 80 | nginx | Reverse proxy |

### Past Incident (2026-02-10)

**Problem:** App Platform (coral-app) + Droplet (wsb-snake) were BOTH running the same trading bot simultaneously.

**Symptoms:**
- `connection limit exceeded` errors from Alpaca WebSocket
- Potential duplicate trades
- Wasted ~$5-12/month

**Root Cause:** Audit never checked `doctl apps list` - assumed only droplet existed.

**Resolution:** Deleted App Platform deployment via `doctl apps delete <ID> --force`

**Prevention:** This CLAUDE.md file and updated KILL_LIST.md with infrastructure audit checklist.

---

## Project Context

- **Git Repo:** wsb_snake trading bot
- **Main Entry:** `wsb_snake/main.py`
- **Config:** `.env` file (never commit)
- **Database:** SQLite at `wsb_snake_data/wsb_snake.db`
- **Deployment:** Droplet 157.245.240.99 via `doctl` or Guardian API

## Guardian API Endpoints

```bash
# Health check
curl http://157.245.240.99:8888/health

# Full status
curl http://157.245.240.99:8888/status

# Git status
curl http://157.245.240.99:8888/git

# Deploy latest code
curl -X POST http://157.245.240.99:8888/deploy

# Restart service
curl -X POST http://157.245.240.99:8888/restart -d '{"service":"wsb-snake"}'

# View logs
curl "http://157.245.240.99:8888/logs?service=wsb-snake&lines=50"
```

## SSH Access

```bash
ssh root@157.245.240.99
```

## Key Files

- `KILL_LIST.md` - Issue tracker with infrastructure checklist
- `wsb_snake/main.py` - Entry point
- `wsb_snake/engines/spy_scalper.py` - Main scalping engine
- `wsb_snake/trading/alpaca_executor.py` - Trade execution
- `wsb_snake/guardian/` - Remote control system
