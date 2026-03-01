# Telegram Refactoring Plan - Broker-Agnostic Signal System

## User Requirements Summary

| Requirement | Decision |
|-------------|----------|
| Alert Content | Super advanced + easy to understand. COPY lines + PL1/2/3 targets + case-specific hold guidance |
| Auto-Execute | Separate Channels (signals vs Alpaca status) |
| Error Handling | Retry 3x silently, then friendly message |
| Exit Alerts | Both Price + Time based alerts |

---

## Current State Problems

1. **API errors sent to users** - Raw "404 Client Error" messages
2. **Signals mixed with Alpaca status** - Confusing for Webull users
3. **No PL1/2/3 exit targets** - Users don't know when to exit
4. **No time-based exit reminders** - 0DTE positions expire worthless
5. **No trade tracking** - Can't reconcile manual trades

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION                         │
│  (Momentum, APEX, CPL, Scalper, etc.)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                 SIGNAL FORMATTER (NEW)                       │
│  - Creates broker-agnostic BUY/SELL signals                 │
│  - Adds COPY lines for manual entry                         │
│  - Calculates PL1/2/3 targets                               │
│  - Adds case-specific hold guidance                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  MAIN CHANNEL   │ │ ALPACA CHANNEL  │ │  TRADE TRACKER  │
│  (All Users)    │ │ (Optional)      │ │  (Database)     │
│                 │ │                 │ │                 │
│ - Pure signals  │ │ - Order status  │ │ - Signal ID     │
│ - COPY lines    │ │ - Fill confirms │ │ - Entry/Exit    │
│ - PL1/2/3       │ │ - API errors    │ │ - P&L tracking  │
│ - Exit alerts   │ │ - Position sync │ │ - Time alerts   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## New Signal Format (Main Channel)

### BUY Signal Template

```
🟢 **WSB SNAKE BUY SIGNAL** #2847

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **SETUP**
Ticker: SPY
Direction: CALLS (Bullish)
Confidence: 78%
Session: Power Hour

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 **COPY FOR YOUR BROKER**
```
COPY: BUY 2 SPY 590C 02/09 @ LIMIT $2.50
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **PROFIT TARGETS**
• Entry: $2.50
• PL1 (+20%): $3.00 ← Take 1/3 profit
• PL2 (+40%): $3.50 ← Take 1/3 profit
• PL3 (+60%): $4.00 ← Close remaining
• Stop Loss (-15%): $2.13

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **RISK GUIDANCE**
• Max Hold: 45 minutes (0DTE)
• R:R Ratio: 4.0
• Volatility: HIGH - Consider smaller size

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 **HOLD GUIDANCE**
This is a MOMENTUM play. If PL1 hits quickly (<5min),
HOLD for PL2/PL3. If slow grind, take PL1 and exit.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏰ Signal expires: 3:45 PM ET
🔔 Exit alerts will follow
```

---

### SELL/EXIT Signal Template

```
🔴 **WSB SNAKE EXIT ALERT** #2847

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **POSITION STATUS**
Ticker: SPY 590C
Entry: $2.50
Current: $3.05
P&L: +22% ✅

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **EXIT REASON: PL1 HIT**

📋 **COPY FOR YOUR BROKER**
```
COPY: SELL 2 SPY 590C 02/09 @ MARKET
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 **HOLD vs EXIT RECOMMENDATION**

⚡ TAKE PROFIT NOW
Momentum slowing. Price action shows resistance.
PL1 achieved in 12 minutes - book the win.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### Time-Based Warning Template

```
⏰ **TIME WARNING** #2847

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Position: SPY 590C
Time Held: 35 minutes
Max Hold: 45 minutes

Current P&L: +8%
PL1 Target: +20% (not yet hit)

⚠️ **RECOMMENDATION**
10 minutes remaining. Price stalling.
Consider exiting at current +8% to avoid
time decay into close.

📋 COPY: SELL 2 SPY 590C @ MARKET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Implementation Plan

### Phase 1: Create Signal Formatter (NEW FILE)

**File:** `wsb_snake/notifications/signal_formatter.py`

```python
"""
Broker-Agnostic Signal Formatter

Creates clean, actionable signals for ANY broker.
Includes COPY lines, PL1/2/3 targets, and hold guidance.
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
from enum import Enum

class HoldGuidance(Enum):
    TAKE_PROFIT_NOW = "take_now"      # Exit immediately
    HOLD_FOR_MORE = "hold"            # Momentum strong, wait for PL2/3
    SCALE_OUT = "scale"               # Take partial, hold rest
    TIGHT_STOP = "tight_stop"         # Raise stop, let it run

@dataclass
class SignalTargets:
    entry: float
    stop_loss: float
    pl1: float  # +20% default
    pl2: float  # +40% default
    pl3: float  # +60% default

    @classmethod
    def calculate(cls, entry: float, direction: str = "long"):
        if direction == "long":
            return cls(
                entry=entry,
                stop_loss=entry * 0.85,  # -15%
                pl1=entry * 1.20,        # +20%
                pl2=entry * 1.40,        # +40%
                pl3=entry * 1.60,        # +60%
            )
        else:  # short/puts
            return cls(
                entry=entry,
                stop_loss=entry * 1.15,  # +15% (loss for puts)
                pl1=entry * 0.80,        # -20% (profit for puts)
                pl2=entry * 0.60,        # -40%
                pl3=entry * 0.40,        # -60%
            )

@dataclass
class TradingSignal:
    signal_id: str
    ticker: str
    underlying: str
    strike: float
    expiry: str
    direction: str  # "CALLS" or "PUTS"
    contracts: int
    entry_price: float
    targets: SignalTargets
    confidence: float
    pattern: str
    session: str
    max_hold_minutes: int
    volatility: str  # "LOW", "MEDIUM", "HIGH"
    hold_guidance: HoldGuidance
    hold_reasoning: str
    created_at: datetime
    expires_at: datetime

def format_buy_signal(signal: TradingSignal) -> str:
    """Format a broker-agnostic BUY signal"""

    # Determine emoji based on confidence
    conf_emoji = "🔥" if signal.confidence >= 80 else "⚡" if signal.confidence >= 70 else "📊"

    # Build COPY line
    side = "C" if signal.direction == "CALLS" else "P"
    copy_line = f"COPY: BUY {signal.contracts} {signal.underlying} {signal.strike}{side} {signal.expiry} @ LIMIT ${signal.entry_price:.2f}"

    # Build hold guidance
    hold_map = {
        HoldGuidance.TAKE_PROFIT_NOW: "Take PL1 immediately when hit.",
        HoldGuidance.HOLD_FOR_MORE: "Momentum strong - hold for PL2/PL3 if PL1 hits quickly.",
        HoldGuidance.SCALE_OUT: "Scale out: 1/3 at PL1, 1/3 at PL2, 1/3 at PL3.",
        HoldGuidance.TIGHT_STOP: "Raise stop to breakeven after PL1, let winners run.",
    }

    msg = f"""🟢 **WSB SNAKE BUY SIGNAL** #{signal.signal_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{conf_emoji} **SETUP**
Ticker: {signal.underlying}
Direction: {signal.direction} ({'Bullish' if signal.direction == 'CALLS' else 'Bearish'})
Confidence: {signal.confidence:.0f}%
Pattern: {signal.pattern}
Session: {signal.session}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 **COPY FOR YOUR BROKER**
```
{copy_line}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **PROFIT TARGETS**
• Entry: ${signal.entry_price:.2f}
• PL1 (+20%): ${signal.targets.pl1:.2f} ← Take 1/3 profit
• PL2 (+40%): ${signal.targets.pl2:.2f} ← Take 1/3 profit
• PL3 (+60%): ${signal.targets.pl3:.2f} ← Close remaining
• Stop Loss (-15%): ${signal.targets.stop_loss:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **RISK GUIDANCE**
• Max Hold: {signal.max_hold_minutes} minutes
• Volatility: {signal.volatility}
{f'• HIGH VOL: Consider smaller size' if signal.volatility == 'HIGH' else ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 **HOLD GUIDANCE**
{hold_map.get(signal.hold_guidance, '')}
{signal.hold_reasoning}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏰ Signal expires: {signal.expires_at.strftime('%I:%M %p ET')}
🔔 Exit alerts will follow
"""
    return msg

def format_exit_signal(
    signal_id: str,
    ticker: str,
    entry_price: float,
    current_price: float,
    exit_reason: str,
    recommendation: str,
    reasoning: str,
) -> str:
    """Format a broker-agnostic EXIT signal"""

    pnl_pct = (current_price - entry_price) / entry_price * 100
    pnl_emoji = "✅" if pnl_pct > 0 else "❌"
    action_emoji = "🟢" if "HOLD" in recommendation.upper() else "🔴"

    msg = f"""🔴 **WSB SNAKE EXIT ALERT** #{signal_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 **POSITION STATUS**
Ticker: {ticker}
Entry: ${entry_price:.2f}
Current: ${current_price:.2f}
P&L: {pnl_pct:+.1f}% {pnl_emoji}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 **EXIT REASON: {exit_reason}**

📋 **COPY FOR YOUR BROKER**
```
COPY: SELL ALL {ticker} @ MARKET
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 **RECOMMENDATION**

{action_emoji} {recommendation}
{reasoning}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return msg

def format_time_warning(
    signal_id: str,
    ticker: str,
    time_held_min: int,
    max_hold_min: int,
    current_pnl_pct: float,
    pl1_target_pct: float,
    recommendation: str,
) -> str:
    """Format a time-based warning"""

    time_remaining = max_hold_min - time_held_min
    pnl_emoji = "✅" if current_pnl_pct > 0 else "⚠️"

    msg = f"""⏰ **TIME WARNING** #{signal_id}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Position: {ticker}
Time Held: {time_held_min} minutes
Max Hold: {max_hold_min} minutes
Time Remaining: {time_remaining} minutes

Current P&L: {current_pnl_pct:+.1f}% {pnl_emoji}
PL1 Target: +{pl1_target_pct:.0f}% {'✅ HIT' if current_pnl_pct >= pl1_target_pct else '❌ Not yet'}

⚠️ **RECOMMENDATION**
{recommendation}

📋 COPY: SELL ALL {ticker} @ MARKET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    return msg
```

---

### Phase 2: Create Dual Channel System

**File:** `wsb_snake/notifications/telegram_channels.py`

```python
"""
Dual Channel Telegram System

MAIN_CHANNEL: Pure signals for all users (any broker)
ALPACA_CHANNEL: Execution status (optional, Alpaca users only)
"""

import os
from wsb_snake.notifications.telegram_bot import send_alert as _send_telegram

# Channel configuration
MAIN_CHANNEL_ID = os.getenv("TELEGRAM_CHAT_ID")  # Existing
ALPACA_CHANNEL_ID = os.getenv("TELEGRAM_ALPACA_CHAT_ID")  # NEW (optional)

def send_signal(message: str) -> bool:
    """Send to MAIN channel - all users see this"""
    return _send_telegram(message, chat_id=MAIN_CHANNEL_ID)

def send_alpaca_status(message: str) -> bool:
    """Send to ALPACA channel - only if configured"""
    if not ALPACA_CHANNEL_ID:
        # Log but don't send - Alpaca channel not configured
        import logging
        logging.getLogger(__name__).debug(f"Alpaca status (not sent): {message[:100]}")
        return True
    return _send_telegram(message, chat_id=ALPACA_CHANNEL_ID)

def send_error_after_retry(message: str, error: str, retries: int = 3) -> bool:
    """Send user-friendly error after retries exhausted"""
    friendly_msg = f"""⚠️ **EXECUTION NOTICE**

{message}

Technical details logged. Signal remains valid.
You can still execute manually on your broker.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Alpaca API temporarily unavailable.
Retried {retries}x before this alert.
"""
    return send_alpaca_status(friendly_msg)
```

---

### Phase 3: Create Trade Tracker (Database)

**File:** `wsb_snake/tracking/signal_tracker.py`

```python
"""
Signal Tracker - Track all signals and their outcomes

Allows reconciliation for users on ANY broker.
Stores signal_id, entry, targets, exit alerts sent.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import sqlite3
from wsb_snake.db.database import get_connection

def init_signal_tracking_tables():
    """Create signal tracking tables"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracked_signals (
            signal_id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            underlying TEXT NOT NULL,
            strike REAL,
            direction TEXT,
            contracts INTEGER,
            entry_price REAL,
            stop_loss REAL,
            pl1_target REAL,
            pl2_target REAL,
            pl3_target REAL,
            confidence REAL,
            pattern TEXT,
            session TEXT,
            max_hold_minutes INTEGER,
            hold_guidance TEXT,
            created_at TEXT,
            expires_at TEXT,
            status TEXT DEFAULT 'OPEN',
            pl1_hit_at TEXT,
            pl2_hit_at TEXT,
            pl3_hit_at TEXT,
            stop_hit_at TEXT,
            exit_price REAL,
            exit_reason TEXT,
            exit_at TEXT,
            final_pnl_pct REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id TEXT,
            alert_type TEXT,
            message TEXT,
            sent_at TEXT,
            FOREIGN KEY (signal_id) REFERENCES tracked_signals(signal_id)
        )
    """)

    conn.commit()
    conn.close()

def track_new_signal(signal) -> str:
    """Record a new signal and return signal_id"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO tracked_signals (
            signal_id, ticker, underlying, strike, direction, contracts,
            entry_price, stop_loss, pl1_target, pl2_target, pl3_target,
            confidence, pattern, session, max_hold_minutes, hold_guidance,
            created_at, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signal.signal_id, signal.ticker, signal.underlying, signal.strike,
        signal.direction, signal.contracts, signal.entry_price,
        signal.targets.stop_loss, signal.targets.pl1, signal.targets.pl2,
        signal.targets.pl3, signal.confidence, signal.pattern, signal.session,
        signal.max_hold_minutes, signal.hold_guidance.value,
        signal.created_at.isoformat(), signal.expires_at.isoformat()
    ))

    conn.commit()
    conn.close()
    return signal.signal_id

def record_alert_sent(signal_id: str, alert_type: str, message: str):
    """Record that an alert was sent"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO signal_alerts (signal_id, alert_type, message, sent_at)
        VALUES (?, ?, ?, ?)
    """, (signal_id, alert_type, message[:500], datetime.now().isoformat()))
    conn.commit()
    conn.close()

def update_signal_status(signal_id: str, status: str, exit_price: float = None,
                         exit_reason: str = None, final_pnl_pct: float = None):
    """Update signal status when target/stop hit or expired"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE tracked_signals
        SET status = ?, exit_price = ?, exit_reason = ?, exit_at = ?, final_pnl_pct = ?
        WHERE signal_id = ?
    """, (status, exit_price, exit_reason, datetime.now().isoformat(), final_pnl_pct, signal_id))
    conn.commit()
    conn.close()
```

---

### Phase 4: Modify alpaca_executor.py

**Changes Required:**

1. **Import new modules:**
```python
from wsb_snake.notifications.telegram_channels import send_signal, send_alpaca_status, send_error_after_retry
from wsb_snake.notifications.signal_formatter import format_buy_signal, format_exit_signal, TradingSignal, SignalTargets
```

2. **Replace send_telegram_alert calls:**

| Old Code | New Code |
|----------|----------|
| `send_telegram_alert(f"⚠️ Failed to close...")` | `send_alpaca_status(f"⚠️ Alpaca: Failed to close...")` |
| `send_telegram_alert(f"✅ BUY ORDER PLACED...")` | `send_alpaca_status(f"✅ Alpaca: Order placed...")` |
| `send_telegram_alert(f"🔴 SELL ORDER SENDING...")` | `send_alpaca_status(f"🔴 Alpaca: Closing position...")` |

3. **Add retry logic for API errors:**
```python
def close_position_with_retry(self, option_symbol: str, max_retries: int = 3):
    """Close position with retry logic"""
    for attempt in range(max_retries):
        try:
            return self._close_position(option_symbol)
        except Exception as e:
            if attempt == max_retries - 1:
                # All retries exhausted - send friendly message
                send_error_after_retry(
                    f"Could not close {option_symbol} on Alpaca",
                    str(e),
                    max_retries
                )
                return None
            time.sleep(2 ** attempt)  # Exponential backoff
```

---

### Phase 5: Create Signal Monitor (Time + Price Alerts)

**File:** `wsb_snake/tracking/signal_monitor.py`

```python
"""
Signal Monitor - Watch open signals and send alerts

Monitors:
1. Price targets (PL1, PL2, PL3, Stop Loss)
2. Time remaining (warnings at 75%, 90% of max hold)
3. Volatility changes
"""

import threading
import time
from datetime import datetime, timedelta
from wsb_snake.notifications.telegram_channels import send_signal
from wsb_snake.notifications.signal_formatter import format_exit_signal, format_time_warning
from wsb_snake.tracking.signal_tracker import update_signal_status, record_alert_sent
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced

class SignalMonitor:
    def __init__(self):
        self.running = False
        self.open_signals = {}  # signal_id -> signal data

    def add_signal(self, signal):
        """Add a signal to monitor"""
        self.open_signals[signal.signal_id] = {
            "signal": signal,
            "pl1_alerted": False,
            "pl2_alerted": False,
            "pl3_alerted": False,
            "time_75_alerted": False,
            "time_90_alerted": False,
        }

    def start(self):
        """Start monitoring loop"""
        self.running = True
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def _monitor_loop(self):
        while self.running:
            for signal_id, data in list(self.open_signals.items()):
                self._check_signal(signal_id, data)
            time.sleep(5)  # Check every 5 seconds

    def _check_signal(self, signal_id: str, data: dict):
        signal = data["signal"]

        # Get current price
        current_price = self._get_option_price(signal.ticker)
        if not current_price:
            return

        entry = signal.entry_price
        pnl_pct = (current_price - entry) / entry * 100

        # Check price targets
        if pnl_pct >= 60 and not data["pl3_alerted"]:
            self._send_exit_alert(signal, current_price, "PL3 HIT (+60%)",
                                  "TAKE PROFIT", "Maximum target reached. Book the win!")
            data["pl3_alerted"] = True
            update_signal_status(signal_id, "CLOSED", current_price, "PL3_HIT", pnl_pct)

        elif pnl_pct >= 40 and not data["pl2_alerted"]:
            self._send_exit_alert(signal, current_price, "PL2 HIT (+40%)",
                                  "SCALE OUT or HOLD", "Strong momentum. Consider taking 1/3, hold rest for PL3.")
            data["pl2_alerted"] = True

        elif pnl_pct >= 20 and not data["pl1_alerted"]:
            self._send_exit_alert(signal, current_price, "PL1 HIT (+20%)",
                                  self._get_pl1_recommendation(signal),
                                  self._get_pl1_reasoning(signal))
            data["pl1_alerted"] = True

        elif pnl_pct <= -15:
            self._send_exit_alert(signal, current_price, "STOP LOSS HIT (-15%)",
                                  "EXIT NOW", "Stop loss triggered. Preserve capital.")
            update_signal_status(signal_id, "CLOSED", current_price, "STOP_LOSS", pnl_pct)
            del self.open_signals[signal_id]

        # Check time warnings
        time_held = (datetime.now() - signal.created_at).total_seconds() / 60
        time_pct = time_held / signal.max_hold_minutes * 100

        if time_pct >= 90 and not data["time_90_alerted"]:
            msg = format_time_warning(
                signal_id, signal.ticker, int(time_held), signal.max_hold_minutes,
                pnl_pct, 20, f"Only {signal.max_hold_minutes - int(time_held)} min left. Consider exit."
            )
            send_signal(msg)
            record_alert_sent(signal_id, "TIME_90", msg)
            data["time_90_alerted"] = True

        elif time_pct >= 75 and not data["time_75_alerted"]:
            msg = format_time_warning(
                signal_id, signal.ticker, int(time_held), signal.max_hold_minutes,
                pnl_pct, 20, "75% of max hold time. Monitor closely."
            )
            send_signal(msg)
            record_alert_sent(signal_id, "TIME_75", msg)
            data["time_75_alerted"] = True

    def _send_exit_alert(self, signal, current_price, reason, recommendation, reasoning):
        msg = format_exit_signal(
            signal.signal_id, signal.ticker, signal.entry_price,
            current_price, reason, recommendation, reasoning
        )
        send_signal(msg)
        record_alert_sent(signal.signal_id, f"EXIT_{reason}", msg)

    def _get_pl1_recommendation(self, signal) -> str:
        """Case-specific PL1 recommendation"""
        if signal.hold_guidance.value == "hold":
            return "HOLD FOR MORE"
        elif signal.hold_guidance.value == "take_now":
            return "TAKE PROFIT NOW"
        else:
            return "SCALE OUT (1/3)"

    def _get_pl1_reasoning(self, signal) -> str:
        """Case-specific reasoning"""
        return signal.hold_reasoning or "PL1 achieved. Review momentum before deciding."

signal_monitor = SignalMonitor()
```

---

## File Changes Summary

### NEW FILES (4):
1. `wsb_snake/notifications/signal_formatter.py` - Broker-agnostic signal formatting
2. `wsb_snake/notifications/telegram_channels.py` - Dual channel system
3. `wsb_snake/tracking/signal_tracker.py` - Signal database tracking
4. `wsb_snake/tracking/signal_monitor.py` - Price/time alert monitoring

### MODIFIED FILES (5):
1. `wsb_snake/trading/alpaca_executor.py` - Replace 39 Telegram calls with channel routing
2. `wsb_snake/engines/orchestrator.py` - Use new signal formatter
3. `wsb_snake/execution/jobs_day_cpl.py` - Use new signal formatter
4. `wsb_snake/config.py` - Add TELEGRAM_ALPACA_CHAT_ID
5. `wsb_snake/db/database.py` - Add signal tracking tables

---

## Configuration

Add to `.env`:
```bash
# Main channel - all users see signals here
TELEGRAM_CHAT_ID=existing_chat_id

# Alpaca channel (optional) - only Alpaca execution status
# Leave empty to disable Alpaca status messages
TELEGRAM_ALPACA_CHAT_ID=
```

---

## Migration Path

1. **Week 1**: Create new files, test signal formatting
2. **Week 2**: Add dual channel system, test routing
3. **Week 3**: Add signal tracker and monitor
4. **Week 4**: Migrate alpaca_executor.py calls
5. **Week 5**: Full integration testing

---

*Plan Created: 2026-02-09 | WSB Snake Telegram Refactoring*
