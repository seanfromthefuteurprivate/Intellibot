# WSB Snake - RBAC Matrix

## Role-Based Access Control

This document defines roles and their permissions within the WSB Snake system.

---

## Roles Overview

| Role | Description | Access Level |
|------|-------------|--------------|
| **SYSTEM** | Automated system processes | Full execution |
| **OPERATOR** | Human system administrator | Full control |
| **VIEWER** | Read-only monitoring | View only |
| **API_CLIENT** | External API consumers | Limited API |

---

## Permission Matrix

### Trading Permissions

| Permission | SYSTEM | OPERATOR | VIEWER | API_CLIENT |
|------------|--------|----------|--------|------------|
| View positions | ✅ | ✅ | ✅ | ✅ |
| View signals | ✅ | ✅ | ✅ | ✅ |
| View account | ✅ | ✅ | ✅ | ❌ |
| Place orders | ✅ | ✅ | ❌ | ❌ |
| Close positions | ✅ | ✅ | ❌ | ❌ |
| Close all positions | ✅ | ✅ | ❌ | ❌ |
| Modify orders | ❌ | ✅ | ❌ | ❌ |

### Configuration Permissions

| Permission | SYSTEM | OPERATOR | VIEWER | API_CLIENT |
|------------|--------|----------|--------|------------|
| View config | ✅ | ✅ | ✅ | ❌ |
| Modify thresholds | ❌ | ✅ | ❌ | ❌ |
| Modify limits | ❌ | ✅ | ❌ | ❌ |
| Enable/disable AI | ❌ | ✅ | ❌ | ❌ |
| Modify ticker universe | ❌ | ✅ | ❌ | ❌ |

### System Permissions

| Permission | SYSTEM | OPERATOR | VIEWER | API_CLIENT |
|------------|--------|----------|--------|------------|
| Start system | ✅ | ✅ | ❌ | ❌ |
| Stop system | ❌ | ✅ | ❌ | ❌ |
| Restart system | ❌ | ✅ | ❌ | ❌ |
| View logs | ✅ | ✅ | ✅ | ❌ |
| Clear logs | ❌ | ✅ | ❌ | ❌ |
| Access secrets | ✅ | ❌ | ❌ | ❌ |
| Modify secrets | ❌ | ✅ (via Replit) | ❌ | ❌ |

### Data Permissions

| Permission | SYSTEM | OPERATOR | VIEWER | API_CLIENT |
|------------|--------|----------|--------|------------|
| Read database | ✅ | ✅ | ✅ | ❌ |
| Write database | ✅ | ✅ | ❌ | ❌ |
| Delete records | ❌ | ✅ | ❌ | ❌ |
| Export data | ❌ | ✅ | ✅ | ❌ |
| Backup database | ✅ | ✅ | ❌ | ❌ |

### API Permissions

| Permission | SYSTEM | OPERATOR | VIEWER | API_CLIENT |
|------------|--------|----------|--------|------------|
| GET /health | ✅ | ✅ | ✅ | ✅ |
| GET /api/status | ✅ | ✅ | ✅ | ✅ |
| GET /api/positions | ✅ | ✅ | ✅ | ✅ |
| GET /api/signals | ✅ | ✅ | ✅ | ✅ |
| GET /api/account | ✅ | ✅ | ✅ | ❌ |
| POST /api/positions/close | ✅ | ✅ | ❌ | ❌ |
| PATCH /api/config | ❌ | ✅ | ❌ | ❌ |
| WS /ws/updates | ✅ | ✅ | ✅ | ✅ |

---

## Role Definitions

### SYSTEM Role

The automated system process that:
- Executes trading logic
- Monitors positions
- Sends alerts
- Records data

**Cannot:**
- Stop itself permanently
- Access secrets directly (uses env vars)
- Bypass safety limits

### OPERATOR Role

A human administrator who:
- Monitors system health
- Adjusts configuration
- Handles emergencies
- Reviews performance

**Access Methods:**
- Replit console
- API endpoints
- Telegram commands (if implemented)

### VIEWER Role

A read-only observer who:
- Views dashboard
- Monitors positions
- Reviews signals
- Exports reports

**Cannot:**
- Modify any settings
- Execute trades
- Access sensitive data

### API_CLIENT Role

An external system that:
- Consumes status endpoints
- Receives position updates
- Integrates with other tools

**Cannot:**
- Execute trades
- Access account data
- Modify configuration

---

## Authentication Methods

### Current Implementation

| Method | Status |
|--------|--------|
| API Keys | 🔲 Not implemented |
| JWT Tokens | 🔲 Not implemented |
| Basic Auth | 🔲 Not implemented |
| Replit Auth | 🔲 Possible |

### Planned Implementation

```python
# API Key authentication
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401)
    return get_role_for_key(x_api_key)
```

---

## Audit Logging

All permission-requiring actions are logged:

```python
@audit_log
async def close_position(position_id: str, role: Role):
    log.info(f"[AUDIT] {role.name} closing position {position_id}")
    # ... execution
```

**Audit Log Format:**
```
2026-01-28 15:30:05 | AUDIT | OPERATOR | close_position | SPY260128C00602000 | SUCCESS
2026-01-28 15:30:10 | AUDIT | VIEWER | close_position | DENIED | insufficient_permissions
```

---

## Emergency Overrides

### OPERATOR Emergency Actions

In emergency situations, OPERATOR can:
1. **Kill Switch:** Stop all trading immediately
2. **Force Close:** Close all positions regardless of P&L
3. **Disable AI:** Turn off AI confirmation
4. **Manual Mode:** Disable auto-execution

### Override Logging

All overrides are specially logged:

```
2026-01-28 15:30:05 | EMERGENCY | OPERATOR | kill_switch | ACTIVATED
2026-01-28 15:30:05 | EMERGENCY | OPERATOR | force_close_all | 3 positions closed
```

---

## Future Enhancements

### Planned Roles

| Role | Description |
|------|-------------|
| ADMIN | Full access + user management |
| TRADER | Can place manual trades |
| ANALYST | Extended data access |

### Planned Features

- Multi-user support
- Role assignment UI
- Permission groups
- Temporary access tokens
- IP whitelisting
