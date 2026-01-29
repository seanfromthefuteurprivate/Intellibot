# WSB Snake - N8N Workflow Catalog

## Automation Workflows

This document catalogs potential N8N workflows for extending WSB Snake functionality.

---

## Overview

N8N can be used to:
- Extend notification channels
- Create data pipelines
- Build custom dashboards
- Integrate with external services

---

## Workflow 1: Multi-Channel Alert Distribution

**Purpose:** Distribute alerts beyond Telegram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Telegram   │────▶│    N8N      │────▶│   Discord   │
│   Webhook   │     │   Router    │     │   Webhook   │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                           ├────────────▶ Slack
                           │
                           ├────────────▶ Email
                           │
                           └────────────▶ SMS (Twilio)
```

**Trigger:** Webhook from WSB Snake
**Nodes:**
1. Webhook trigger
2. JSON parser
3. Router (by alert type)
4. Discord HTTP Request
5. Slack HTTP Request
6. Email node
7. Twilio SMS node

---

## Workflow 2: Daily Performance Report

**Purpose:** Generate and send daily trading summary

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Cron      │────▶│  Fetch DB   │────▶│  Generate   │
│  4:30 PM ET │     │   Stats     │     │   Report    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │    Send     │
                                        │   Report    │
                                        └─────────────┘
```

**Schedule:** Daily at 4:30 PM ET
**Nodes:**
1. Cron trigger
2. HTTP Request to `/api/stats`
3. HTML template generator
4. Email send
5. Archive to Google Sheets

---

## Workflow 3: Economic Calendar Monitor

**Purpose:** Alert before major economic events

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Cron      │────▶│   Fetch     │────▶│   Filter    │
│  8:00 AM ET │     │  Calendar   │     │  High Impact│
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Alert     │
                                        │  30m before │
                                        └─────────────┘
```

**Schedule:** Daily at 8:00 AM ET
**Nodes:**
1. Cron trigger
2. Finnhub economic calendar fetch
3. Filter for CPI, FOMC, Jobs reports
4. Schedule alert 30 minutes before
5. Telegram notification

---

## Workflow 4: Position Sync Backup

**Purpose:** Backup position data to external storage

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Cron      │────▶│   Fetch     │────▶│   Write     │
│  Every 5m   │     │  Positions  │     │  to Sheet   │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Schedule:** Every 5 minutes during market hours
**Nodes:**
1. Cron trigger with market hours filter
2. HTTP Request to `/api/positions`
3. Google Sheets append row

---

## Workflow 5: Error Alert Escalation

**Purpose:** Escalate critical errors

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Webhook   │────▶│   Check     │────▶│   PagerDuty │
│   Error     │     │  Severity   │     │    Alert    │
└─────────────┘     └──────┬──────┘     └─────────────┘
                           │
                           └──────────▶ Log to Airtable
```

**Trigger:** Webhook on error
**Nodes:**
1. Webhook trigger
2. Severity router (INFO/WARN/ERROR/CRITICAL)
3. PagerDuty integration (CRITICAL only)
4. Airtable logging (all errors)

---

## Workflow 6: Weekly Learning Summary

**Purpose:** Summarize learning module insights

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Cron      │────▶│  Query DB   │────▶│  Generate   │
│  Sunday 6PM │     │  Learnings  │     │   Summary   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ OpenAI      │
                                        │ Summarize   │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Email     │
                                        │   Report    │
                                        └─────────────┘
```

**Schedule:** Sunday 6:00 PM ET
**Nodes:**
1. Cron trigger
2. SQLite query for week's patterns
3. OpenAI summarization
4. Email report

---

## Workflow 7: Unusual Volume Scanner

**Purpose:** Alert on unusual volume spikes

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Cron      │────▶│   Finviz    │────▶│   Filter    │
│  Every 15m  │     │   Scrape    │     │   > 3x Vol  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │  Telegram   │
                                        │   Alert     │
                                        └─────────────┘
```

**Schedule:** Every 15 minutes
**Nodes:**
1. Cron trigger
2. HTTP Request to Finviz
3. HTML parser
4. Volume filter (> 3x average)
5. Telegram notification

---

## Workflow 8: Options Flow Monitor

**Purpose:** Track unusual options activity

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Cron      │────▶│  Barchart   │────▶│   Filter    │
│  Every 30m  │     │   Scrape    │     │   Sweeps    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Match     │
                                        │  Universe   │
                                        └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │   Alert     │
                                        └─────────────┘
```

**Schedule:** Every 30 minutes
**Nodes:**
1. Cron trigger
2. HTTP Request to Barchart
3. Filter for sweeps > $1M
4. Match against ticker universe
5. Telegram alert

---

## Implementation Notes

### Webhook Configuration

WSB Snake can trigger N8N webhooks by adding to notification code:

```python
import requests

def trigger_n8n_workflow(workflow_url: str, data: dict):
    try:
        requests.post(workflow_url, json=data, timeout=5)
    except Exception as e:
        logger.warning(f"N8N trigger failed: {e}")
```

### Authentication

For secure N8N webhook calls:
1. Generate webhook token in N8N
2. Store in Replit Secrets as `N8N_WEBHOOK_TOKEN`
3. Include in header: `X-N8N-Token: {token}`

### Rate Limiting

Respect N8N instance limits:
- Free: 5 executions/minute
- Pro: 50 executions/minute

---

## Deployment Options

| Option | Pros | Cons |
|--------|------|------|
| N8N Cloud | Managed, reliable | Cost |
| Self-hosted | Free, customizable | Maintenance |
| Replit integration | Same platform | Resource sharing |
