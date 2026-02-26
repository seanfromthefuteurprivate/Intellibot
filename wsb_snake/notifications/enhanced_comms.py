"""
Enhanced Communication System - Agent 5 (COMMS OFFICER)

Features:
1. ENHANCED EOD REPORT at 4:15 PM ET with full trade details
2. REAL-TIME P&L TICKER every 5 min during market hours
3. TRADE JOURNAL auto-generation after each trade closes
4. ERROR ALERTING - any unhandled exception sends immediate alert
5. HEARTBEAT MONITOR - if no log for 5 min, sends warning
6. EXECUTION STATUS on every alert (EXECUTED on Alpaca / ALERT ONLY)
"""

import logging
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from wsb_snake.notifications.telegram_channels import send_signal, send_alpaca_status
from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# HEARTBEAT MONITOR - Alert if no activity for 5 minutes
# ============================================================================

class HeartbeatMonitor:
    """
    Monitor system heartbeat. If no log activity for 5 minutes, send alert.

    Usage:
        heartbeat = HeartbeatMonitor()
        heartbeat.start()

        # In your code, call heartbeat.pulse("service_name") periodically
        heartbeat.pulse("orchestrator")
    """

    TIMEOUT_SECONDS = 300  # 5 minutes
    CHECK_INTERVAL = 60    # Check every 60 seconds

    def __init__(self):
        self._last_pulse: Dict[str, datetime] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._alerted_services: Dict[str, datetime] = {}  # Avoid spam

    def start(self):
        """Start heartbeat monitoring in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Heartbeat monitor started (timeout=%ds)", self.TIMEOUT_SECONDS)

    def stop(self):
        """Stop heartbeat monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Heartbeat monitor stopped")

    def pulse(self, service_name: str):
        """Record a heartbeat pulse from a service."""
        with self._lock:
            self._last_pulse[service_name] = datetime.now()

    def _monitor_loop(self):
        """Background loop to check for missed heartbeats."""
        while self._running:
            try:
                self._check_heartbeats()
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
            time.sleep(self.CHECK_INTERVAL)

    def _check_heartbeats(self):
        """Check all registered services for missed heartbeats."""
        now = datetime.now()

        with self._lock:
            for service, last_pulse in self._last_pulse.items():
                elapsed = (now - last_pulse).total_seconds()

                if elapsed > self.TIMEOUT_SECONDS:
                    # Check if we already alerted recently (avoid spam)
                    last_alert = self._alerted_services.get(service)
                    if last_alert and (now - last_alert).total_seconds() < 600:
                        continue  # Already alerted within 10 min

                    # Send heartbeat miss alert
                    minutes = int(elapsed // 60)
                    message = f"HEARTBEAT MISS: {service} - No activity for {minutes} minutes"

                    send_signal(message)
                    logger.warning(message)

                    self._alerted_services[service] = now


# Global heartbeat monitor instance
heartbeat_monitor = HeartbeatMonitor()


# ============================================================================
# ERROR ALERTING - Catch unhandled exceptions
# ============================================================================

def error_alert(service_name: str):
    """
    Decorator to catch unhandled exceptions and send Telegram alert.

    Usage:
        @error_alert("orchestrator")
        def run_pipeline():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get exception details
                exc_type = type(e).__name__
                exc_msg = str(e)[:200]  # Truncate long messages

                # Send error alert
                error_message = f"ERROR: {service_name} - {exc_type}: {exc_msg}"
                send_signal(error_message)
                logger.error(f"Unhandled exception in {service_name}: {e}")
                logger.error(traceback.format_exc())

                # Re-raise to preserve original behavior
                raise
        return wrapper
    return decorator


def send_error_alert(service_name: str, exception: Exception, context: str = ""):
    """
    Send an error alert to Telegram.

    Args:
        service_name: Name of the service/component that errored
        exception: The exception that occurred
        context: Optional context about what was happening
    """
    exc_type = type(exception).__name__
    exc_msg = str(exception)[:300]

    message = f"""ERROR: {service_name}

Type: {exc_type}
Message: {exc_msg}
"""
    if context:
        message += f"Context: {context}\n"

    message += f"Time: {datetime.now().strftime('%H:%M:%S ET')}"

    send_signal(message)
    logger.error(f"Error alert sent for {service_name}: {exc_type}")


# ============================================================================
# REAL-TIME P&L TICKER - Every 5 minutes during market hours
# ============================================================================

class PnLTicker:
    """
    Real-time P&L ticker that sends updates every 5 minutes during market hours.

    Format: P&L: +$127.50 | Win: 3/4 (75%) | Positions: 2 | Regime: RISK_ON
    """

    TICK_INTERVAL_SECONDS = 300  # 5 minutes

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start P&L ticker in background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._ticker_loop, daemon=True)
        self._thread.start()
        logger.info("P&L ticker started (interval=%ds)", self.TICK_INTERVAL_SECONDS)

    def stop(self):
        """Stop P&L ticker."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("P&L ticker stopped")

    def _ticker_loop(self):
        """Background loop for P&L updates."""
        while self._running:
            try:
                if self._is_market_hours():
                    self._send_pnl_update()
            except Exception as e:
                logger.error(f"P&L ticker error: {e}")
            time.sleep(self.TICK_INTERVAL_SECONDS)

    def _is_market_hours(self) -> bool:
        """Check if we're in market hours (9:30 AM - 4:00 PM ET)."""
        try:
            import pytz
            et = pytz.timezone('US/Eastern')
            now_et = datetime.now(et)

            # Only weekdays
            if now_et.weekday() >= 5:
                return False

            # Market hours: 9:30 - 16:00
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

            return market_open <= now_et <= market_close
        except Exception:
            return False

    def _send_pnl_update(self):
        """Send P&L ticker update to Telegram."""
        try:
            # Get stats from Alpaca executor
            from wsb_snake.trading.alpaca_executor import alpaca_executor
            stats = alpaca_executor.get_session_stats()

            total_pnl = stats.get('total_pnl', 0)
            total_trades = stats.get('total_trades', 0)
            winning_trades = stats.get('winning_trades', 0)
            win_rate = stats.get('win_rate', 0)
            open_positions = stats.get('open_positions', 0)

            # Get regime
            regime = "UNKNOWN"
            try:
                from wsb_snake.execution.regime_detector import regime_detector
                result = regime_detector.get_current_result()
                if result:
                    regime = result.regime.value.upper()
            except Exception:
                pass

            # Format message
            pnl_sign = "+" if total_pnl >= 0 else ""
            win_str = f"{winning_trades}/{total_trades}" if total_trades > 0 else "0/0"

            message = f"P&L: {pnl_sign}${total_pnl:.2f} | Win: {win_str} ({win_rate:.0f}%) | Positions: {open_positions} | Regime: {regime}"

            # Only send if there's actual activity
            if total_trades > 0 or open_positions > 0:
                send_signal(message)
                logger.info(f"P&L ticker: {message}")

        except Exception as e:
            logger.error(f"Failed to send P&L update: {e}")

    def send_immediate_update(self):
        """Send an immediate P&L update (for ad-hoc requests)."""
        self._send_pnl_update()


# Global P&L ticker instance
pnl_ticker = PnLTicker()


# ============================================================================
# TRADE JOURNAL - Auto-generate after each trade closes
# ============================================================================

def send_trade_journal_entry(
    symbol: str,
    trade_type: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    pnl: float,
    pnl_pct: float,
    exit_reason: str,
    entry_time: Optional[datetime] = None,
    exit_time: Optional[datetime] = None,
    engine: str = "scalper",
    option_symbol: Optional[str] = None,
    executed_on_alpaca: bool = True,
):
    """
    Send a trade journal entry to Telegram after each trade closes.

    Args:
        symbol: Underlying symbol (e.g., SPY)
        trade_type: CALLS or PUTS
        direction: long or short
        entry_price: Option entry price
        exit_price: Option exit price
        pnl: Dollar P&L
        pnl_pct: Percentage P&L
        exit_reason: Why the trade was closed (TARGET, STOP, TIME, etc.)
        entry_time: Trade entry timestamp
        exit_time: Trade exit timestamp
        engine: Which engine generated the trade
        option_symbol: Full OCC option symbol
        executed_on_alpaca: Whether trade was executed on Alpaca
    """
    try:
        # Calculate holding time
        holding_time = ""
        if entry_time and exit_time:
            delta = exit_time - entry_time
            minutes = int(delta.total_seconds() // 60)
            seconds = int(delta.total_seconds() % 60)
            holding_time = f"{minutes}m {seconds}s"

        # Determine emoji based on outcome
        if pnl > 0:
            emoji = ""
            outcome = "WIN"
        elif pnl < 0:
            emoji = ""
            outcome = "LOSS"
        else:
            emoji = ""
            outcome = "SCRATCH"

        # Execution status
        exec_status = "EXECUTED on Alpaca" if executed_on_alpaca else "ALERT ONLY (manual)"

        # Format option spec if available
        option_spec = ""
        if option_symbol:
            option_spec = f"\nOption: `{option_symbol}`"

        # Entry/exit times
        entry_str = entry_time.strftime('%H:%M:%S ET') if entry_time else "N/A"
        exit_str = exit_time.strftime('%H:%M:%S ET') if exit_time else "N/A"

        message = f"""{emoji} *TRADE JOURNAL - {outcome}*

*{symbol}* {trade_type} ({direction.upper()})
Engine: {engine}{option_spec}

*Entry:* ${entry_price:.2f} at {entry_str}
*Exit:* ${exit_price:.2f} at {exit_str}
*Holding Time:* {holding_time}

*P&L:* ${pnl:+.2f} ({pnl_pct:+.1f}%)
*Exit Reason:* {exit_reason}

*Status:* {exec_status}
"""

        send_signal(message)
        logger.info(f"Trade journal entry sent: {symbol} {outcome} ${pnl:+.2f}")

    except Exception as e:
        logger.error(f"Failed to send trade journal entry: {e}")


# ============================================================================
# ENHANCED EOD REPORT - 4:15 PM ET with full trade details
# ============================================================================

def send_enhanced_eod_report(date_str: Optional[str] = None) -> bool:
    """
    Send WEAPONIZED end-of-day intel report at 4:15 PM ET.

    Includes:
    - Total P&L by mode (SCALP vs BLOWUP)
    - Power Hour performance breakdown
    - HYDRA intelligence summary
    - Circuit breaker triggers
    - Each trade detail with entry/exit prices
    - Regime summary
    - Portfolio value

    Args:
        date_str: Optional date string (YYYY-MM-DD), defaults to today

    Returns:
        True if sent successfully
    """
    try:
        import pytz
        et = pytz.timezone('US/Eastern')

        if date_str is None:
            date_str = datetime.now(et).strftime("%Y-%m-%d")

        # Get daily stats from database
        from wsb_snake.db.database import get_daily_stats_for_report, get_connection
        stats = get_daily_stats_for_report(date_str)

        # Get Alpaca executor stats
        from wsb_snake.trading.alpaca_executor import alpaca_executor
        alpaca_stats = alpaca_executor.get_session_stats()

        # Get account info
        account = alpaca_executor.get_account()
        portfolio_value = float(account.get('portfolio_value', 0))
        buying_power = float(account.get('buying_power', 0))

        # Get regime info
        regime_summary = "UNKNOWN"
        try:
            from wsb_snake.execution.regime_detector import regime_detector
            result = regime_detector.get_regime_summary()
            regime_summary = f"{result.get('regime', 'unknown').upper()} (conf={result.get('confidence', 0):.0%})"
        except Exception:
            pass

        # Get HYDRA bridge status
        hydra_status = {"connected": False, "blowup_probability": 0, "regime": "UNKNOWN"}
        try:
            from wsb_snake.collectors.hydra_bridge import get_hydra_bridge
            bridge = get_hydra_bridge()
            intel = bridge.get_intel()
            hydra_status = {
                "connected": intel.connected,
                "blowup_probability": intel.blowup_probability,
                "direction": intel.direction,
                "regime": intel.regime,
                "recommendation": intel.recommendation,
            }
        except Exception:
            pass

        # Get risk governor weaponized status
        risk_status = {}
        try:
            from wsb_snake.trading.risk_governor import get_risk_governor
            governor = get_risk_governor()
            risk_status = governor.get_weaponized_status()
        except Exception:
            pass

        # Get power hour status
        power_hour_status = {}
        try:
            from wsb_snake.engines.power_hour_protocol import get_power_hour_status
            power_hour_status = get_power_hour_status()
        except Exception:
            pass

        # Get dual-mode engine status
        dual_mode_status = {}
        try:
            from wsb_snake.engines.dual_mode_engine import get_dual_mode_engine
            engine = get_dual_mode_engine()
            dual_mode_status = engine.get_status()
        except Exception:
            pass

        # Get signal counts
        signal_count = 0
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as total FROM signals WHERE date(timestamp) = ?
            """, (date_str,))
            signal_count = cursor.fetchone()['total']
            conn.close()
        except Exception:
            pass

        # Get trade details from trade_performance table
        trades = []
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, trade_type, pnl, pnl_pct, exit_reason, engine,
                       entry_hour, session, r_multiple, event_tier
                FROM trade_performance
                WHERE trade_date = ?
                ORDER BY created_at DESC
            """, (date_str,))
            trades = [dict(row) for row in cursor.fetchall()]
            conn.close()
        except Exception:
            pass

        # Calculate stats
        total_trades = stats.get('trades', 0)
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        win_rate = stats.get('win_rate', 0) * 100
        avg_r = stats.get('avg_r', 0)
        total_r = stats.get('total_r', 0)
        total_pnl = alpaca_stats.get('total_pnl', 0)

        # Scalp vs Blowup breakdown
        scalp_pnl = risk_status.get('daily_pnl', 0) - risk_status.get('power_hour_pnl', 0)
        blowup_trades = dual_mode_status.get('blowup_trades_today', 0)

        # Determine overall emoji
        if total_pnl > 100:
            day_emoji = "🔥"
        elif total_pnl > 0:
            day_emoji = "💰"
        elif total_pnl < -100:
            day_emoji = "💀"
        elif total_pnl < 0:
            day_emoji = "🛑"
        else:
            day_emoji = "➖"

        # Build message
        lines = [
            "═══════════════════════════════════════",
            f"📊 *WSB SNAKE — DAILY INTEL REPORT*",
            f"📅 *{date_str}*",
            "═══════════════════════════════════════",
            "",
            f"{day_emoji} *P&L: ${total_pnl:+.2f}*",
            "",
            "*MODE BREAKDOWN*",
            f"📈 Scalp Mode: {total_trades - blowup_trades} trades, ${scalp_pnl:+.2f}",
            f"💥 Blowup Mode: {blowup_trades} trades",
            "",
            "*POWER HOUR*",
            f"⚡ P&L: ${power_hour_status.get('power_hour_pnl', 0):+.2f}",
            f"📊 Trades: {power_hour_status.get('trades_executed', 0)}",
            f"🎯 Blowup Armed: {'YES' if power_hour_status.get('blowup_armed') else 'NO'}",
            "",
            "*PERFORMANCE*",
            f"• Trades: {total_trades} | {wins}W/{losses}L",
            f"• Win Rate: {win_rate:.1f}%",
            f"• Avg R: {avg_r:+.2f}R | Total R: {total_r:+.2f}R",
            "",
            "*HYDRA INTELLIGENCE*",
            f"• Connected: {'✅' if hydra_status.get('connected') else '❌'}",
            f"• Regime: {hydra_status.get('regime', 'UNKNOWN')}",
            f"• Blowup Score: {hydra_status.get('blowup_probability', 0)}",
            f"• Direction: {hydra_status.get('direction', 'NEUTRAL')}",
            "",
            "*CIRCUIT BREAKERS*",
            f"• Drawdown Halt: {'🛑 YES' if risk_status.get('drawdown_halt') else '✅ NO'}",
            f"• Win Rate Pause: {'⚠️ YES' if risk_status.get('win_rate_pause') else '✅ NO'}",
            f"• Consecutive Losses: {risk_status.get('consecutive_losses', 0)}",
            "",
            "*PORTFOLIO*",
            f"💵 Value: ${portfolio_value:,.2f}",
            f"💳 Buying Power: ${buying_power:,.2f}",
            "",
        ]

        # Add trade details (limit to 8)
        if trades:
            lines.append("*TRADE LOG*")
            for trade in trades[:8]:
                pnl = trade.get('pnl', 0)
                pnl_pct = trade.get('pnl_pct', 0)
                symbol = trade.get('symbol', '?')
                trade_type = trade.get('trade_type', '?')
                exit_reason = trade.get('exit_reason', '?')

                emoji = "✅" if pnl > 0 else "❌" if pnl < 0 else "➖"
                lines.append(f"{emoji} {symbol} {trade_type}: ${pnl:+.2f} ({pnl_pct:+.1f}%)")

            if len(trades) > 8:
                lines.append(f"... +{len(trades) - 8} more")
        else:
            lines.append("*No trades executed today*")

        lines.extend([
            "",
            "*REGIME*",
            f"🌡️ {regime_summary}",
            "",
            "*EVENT TIERS*",
            f"2X: {stats.get('tier_2x', 0)} | 4X: {stats.get('tier_4x', 0)} | 20X: {stats.get('tier_20x', 0)}",
            "",
            "═══════════════════════════════════════",
            "_Powered by WSB Snake v2.5 + HYDRA Bridge_",
        ])

        message = "\n".join(lines)

        result = send_signal(message)
        if result:
            logger.info("Weaponized EOD report sent successfully")

        # Also save to SQLite for tracking
        try:
            _save_daily_report_to_db(date_str, total_pnl, total_trades, wins, losses, win_rate, hydra_status, power_hour_status)
        except Exception as e:
            logger.warning(f"Failed to save report to DB: {e}")

        return result

    except Exception as e:
        logger.error(f"Failed to send enhanced EOD report: {e}")
        send_error_alert("eod_report", e, "Generating EOD report")
        return False


def _save_daily_report_to_db(
    date_str: str,
    total_pnl: float,
    total_trades: int,
    wins: int,
    losses: int,
    win_rate: float,
    hydra_status: Dict,
    power_hour_status: Dict,
):
    """Save daily report to SQLite for historical tracking."""
    try:
        from wsb_snake.db.database import get_connection
        import json

        conn = get_connection()
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT UNIQUE,
                total_pnl REAL,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                power_hour_pnl REAL,
                power_hour_trades INTEGER,
                hydra_regime TEXT,
                hydra_blowup_score INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            INSERT OR REPLACE INTO daily_reports
            (report_date, total_pnl, total_trades, wins, losses, win_rate,
             power_hour_pnl, power_hour_trades, hydra_regime, hydra_blowup_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date_str,
            total_pnl,
            total_trades,
            wins,
            losses,
            win_rate,
            power_hour_status.get('power_hour_pnl', 0),
            power_hour_status.get('trades_executed', 0),
            hydra_status.get('regime', 'UNKNOWN'),
            hydra_status.get('blowup_probability', 0),
        ))

        conn.commit()
        conn.close()
        logger.info(f"Daily report saved to DB for {date_str}")

    except Exception as e:
        logger.error(f"Failed to save daily report to DB: {e}")


# ============================================================================
# ALERT WITH EXECUTION STATUS - Includes whether executed on Alpaca
# ============================================================================

def send_signal_with_execution_status(
    message: str,
    executed_on_alpaca: bool = False,
    execution_details: Optional[Dict[str, Any]] = None,
):
    """
    Send a signal alert that includes execution status.

    Args:
        message: The signal message
        executed_on_alpaca: Whether the trade was executed on Alpaca
        execution_details: Optional dict with execution details (order_id, fill_price, etc.)
    """
    # Add execution status footer
    if executed_on_alpaca:
        status_line = "\n\n*Status:* EXECUTED on Alpaca"
        if execution_details:
            if execution_details.get('order_id'):
                status_line += f"\nOrder: `{execution_details['order_id'][:8]}...`"
            if execution_details.get('fill_price'):
                status_line += f"\nFill: ${execution_details['fill_price']:.2f}"
            if execution_details.get('qty'):
                status_line += f" x {execution_details['qty']}"
    else:
        status_line = "\n\n*Status:* ALERT ONLY (manual execution required)"

    full_message = message + status_line
    send_signal(full_message)


# ============================================================================
# STARTUP FUNCTION - Initialize all monitoring systems
# ============================================================================

def start_enhanced_comms():
    """
    Start all enhanced communication systems.

    Call this from main.py on startup.
    """
    logger.info("Starting enhanced communications systems...")

    # Start heartbeat monitor
    heartbeat_monitor.start()
    heartbeat_monitor.pulse("startup")

    # Start P&L ticker
    pnl_ticker.start()

    logger.info("Enhanced communications systems online")

    # Send startup confirmation
    send_signal("Enhanced COMMS system online - Heartbeat + P&L Ticker active")


def stop_enhanced_comms():
    """Stop all enhanced communication systems."""
    logger.info("Stopping enhanced communications systems...")
    heartbeat_monitor.stop()
    pnl_ticker.stop()
    logger.info("Enhanced communications systems stopped")
