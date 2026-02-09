"""
Signal Monitor - Watch open signals and send price/time alerts.

Monitors all tracked signals for:
1. Price targets (PL1 +20%, PL2 +40%, PL3 +60%)
2. Stop loss (-15%)
3. Time remaining warnings (75% and 90% of max hold)

Uses background thread to check every 5 seconds.
"""

import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

from wsb_snake.notifications.signal_formatter import (
    format_exit_signal,
    format_time_warning,
    format_target_hit,
    format_stop_loss_hit,
    TradingSignal,
    HoldGuidance,
)
from wsb_snake.tracking.signal_tracker import (
    get_open_signals,
    get_signal_by_id,
    update_signal_status,
    mark_target_hit,
    mark_stop_hit,
    record_alert_sent,
)
from wsb_snake.notifications.telegram_channels import send_signal
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.utils.logger import log


class SignalMonitor:
    """
    Real-time monitor for open trading signals.

    Checks prices every 5 seconds and sends alerts when:
    - PL1 hit (+20%): Case-specific recommendation based on hold_guidance
    - PL2 hit (+40%): "SCALE OUT or HOLD"
    - PL3 hit (+60%): "TAKE PROFIT - Maximum target"
    - Stop loss (-15%): "EXIT NOW - Stop loss triggered"
    - Time warnings: 75% and 90% of max_hold
    """

    def __init__(self):
        """Initialize the signal monitor."""
        self.running = False
        self.open_signals: Dict[int, Dict[str, Any]] = {}
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def add_signal(self, signal: Any) -> None:
        """
        Add a signal to the monitor.

        Args:
            signal: Either a TradingSignal object or a dict with signal data
        """
        with self._lock:
            # Handle both TradingSignal objects and dicts
            if hasattr(signal, 'signal_id'):
                signal_id = signal.signal_id
            else:
                signal_id = signal.get('signal_id')

            if signal_id is None:
                log.warning("Cannot add signal without signal_id")
                return

            self.open_signals[signal_id] = {
                "signal": signal,
                "pl1_alerted": False,
                "pl2_alerted": False,
                "pl3_alerted": False,
                "stop_alerted": False,
                "time_75_alerted": False,
                "time_90_alerted": False,
                "added_at": datetime.now(),
            }

            log.info(f"Added signal {signal_id} to monitor. Total monitored: {len(self.open_signals)}")

    def remove_signal(self, signal_id: Any) -> bool:
        """
        Remove a signal from the monitor.

        Args:
            signal_id: The signal ID to remove

        Returns:
            True if signal was removed, False if not found
        """
        with self._lock:
            if signal_id in self.open_signals:
                del self.open_signals[signal_id]
                log.info(f"Removed signal {signal_id} from monitor. Remaining: {len(self.open_signals)}")
                return True
            else:
                log.warning(f"Signal {signal_id} not found in monitor")
                return False

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self.running:
            log.warning("Signal monitor already running")
            return

        self.running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        log.info("Signal monitor started")

    def stop(self) -> None:
        """Stop the monitoring thread."""
        self.running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
        log.info("Signal monitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop - checks every 5 seconds."""
        log.info("Signal monitor loop started")

        while self.running:
            try:
                # Take snapshot of current signals to avoid holding lock during checks
                with self._lock:
                    signals_to_check = dict(self.open_signals)

                # Check each signal
                for signal_id, data in signals_to_check.items():
                    try:
                        self._check_signal(signal_id, data)
                    except Exception as e:
                        log.error(f"Error checking signal {signal_id}: {e}")

                # Sleep 5 seconds between checks
                time.sleep(5)

            except Exception as e:
                log.error(f"Error in monitor loop: {e}")
                time.sleep(5)

        log.info("Signal monitor loop ended")

    def _check_signal(self, signal_id: Any, data: Dict[str, Any]) -> None:
        """
        Check a single signal for price/time alerts.

        Args:
            signal_id: The signal ID
            data: Signal tracking data
        """
        signal = data["signal"]

        # Get signal info - handle both TradingSignal objects and dicts
        if hasattr(signal, 'ticker'):
            ticker = signal.ticker
            entry_price = signal.entry_price
            targets = signal.targets
            created_at = signal.created_at
            max_hold_minutes = signal.max_hold_minutes
            hold_guidance = signal.hold_guidance
            hold_reasoning = getattr(signal, 'hold_reasoning', '')
            underlying = signal.underlying
            strike = signal.strike
            direction = signal.direction
        else:
            ticker = signal.get('ticker', '')
            entry_price = signal.get('entry_price', 0)
            created_at_str = signal.get('created_at', '')
            if isinstance(created_at_str, str):
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    created_at = datetime.now()
            else:
                created_at = created_at_str or datetime.now()
            max_hold_minutes = signal.get('max_hold_minutes', 45)
            hold_guidance = signal.get('hold_guidance', 'scale_out')
            hold_reasoning = signal.get('hold_reasoning', '')
            underlying = signal.get('underlying', '')
            strike = signal.get('strike', 0)
            direction = signal.get('direction', '')

            # Build targets from dict
            targets = type('Targets', (), {
                'pl1': signal.get('pl1_target', entry_price * 1.20),
                'pl2': signal.get('pl2_target', entry_price * 1.40),
                'pl3': signal.get('pl3_target', entry_price * 1.60),
                'stop_loss': signal.get('stop_loss', entry_price * 0.85),
            })()

        # Get current price using polygon_enhanced
        current_price = self._get_option_price(ticker, underlying)
        if current_price is None or current_price <= 0:
            log.debug(f"Could not get price for {ticker}")
            return

        # Calculate P&L percentage
        if entry_price <= 0:
            log.warning(f"Invalid entry price for {ticker}: {entry_price}")
            return

        pnl_pct = ((current_price - entry_price) / entry_price) * 100

        # Check price targets
        self._check_price_targets(signal_id, data, signal, current_price, pnl_pct, targets, hold_guidance, hold_reasoning)

        # Check time warnings
        self._check_time_warnings(signal_id, data, signal, current_price, pnl_pct, created_at, max_hold_minutes)

    def _get_option_price(self, ticker: str, underlying: str = "") -> Optional[float]:
        """
        Get current price for an option using polygon_enhanced.

        Args:
            ticker: Option symbol (e.g., "SPY250210C590")
            underlying: Underlying ticker (e.g., "SPY") as fallback

        Returns:
            Current option price or None if unavailable
        """
        try:
            # First try to get option snapshot directly
            # The ticker might be an options symbol or a stock
            snapshot = polygon_enhanced.get_snapshot(ticker)

            if snapshot and snapshot.get('price') and snapshot['price'] > 0:
                return snapshot['price']

            # If option symbol fails, try underlying for reference
            if underlying:
                snapshot = polygon_enhanced.get_snapshot(underlying)
                if snapshot and snapshot.get('price'):
                    log.debug(f"Using underlying {underlying} price as fallback")
                    # Note: This is just for monitoring - real option price should come from options API
                    return snapshot.get('price')

            return None
        except Exception as e:
            log.error(f"Error getting price for {ticker}: {e}")
            return None

    def _check_price_targets(
        self,
        signal_id: Any,
        data: Dict[str, Any],
        signal: Any,
        current_price: float,
        pnl_pct: float,
        targets: Any,
        hold_guidance: Any,
        hold_reasoning: str,
    ) -> None:
        """Check if price targets or stop loss have been hit."""

        # Check PL3 (+60%)
        if pnl_pct >= 60 and not data["pl3_alerted"]:
            self._send_pl3_alert(signal_id, signal, current_price, pnl_pct)
            data["pl3_alerted"] = True
            with self._lock:
                if signal_id in self.open_signals:
                    self.open_signals[signal_id]["pl3_alerted"] = True

        # Check PL2 (+40%)
        elif pnl_pct >= 40 and not data["pl2_alerted"]:
            self._send_pl2_alert(signal_id, signal, current_price, pnl_pct)
            data["pl2_alerted"] = True
            with self._lock:
                if signal_id in self.open_signals:
                    self.open_signals[signal_id]["pl2_alerted"] = True

        # Check PL1 (+20%)
        elif pnl_pct >= 20 and not data["pl1_alerted"]:
            self._send_pl1_alert(signal_id, signal, current_price, pnl_pct, hold_guidance, hold_reasoning)
            data["pl1_alerted"] = True
            with self._lock:
                if signal_id in self.open_signals:
                    self.open_signals[signal_id]["pl1_alerted"] = True

        # Check stop loss (-15%)
        elif pnl_pct <= -15 and not data["stop_alerted"]:
            self._send_stop_loss_alert(signal_id, signal, current_price, pnl_pct)
            data["stop_alerted"] = True
            # Remove from monitor after stop hit
            self.remove_signal(signal_id)

    def _check_time_warnings(
        self,
        signal_id: Any,
        data: Dict[str, Any],
        signal: Any,
        current_price: float,
        pnl_pct: float,
        created_at: datetime,
        max_hold_minutes: int,
    ) -> None:
        """Check if time warnings should be sent."""
        # Calculate time held
        now = datetime.now()

        # Handle timezone-aware vs naive datetime
        if created_at.tzinfo is not None and now.tzinfo is None:
            now = now.replace(tzinfo=created_at.tzinfo)
        elif created_at.tzinfo is None and now.tzinfo is not None:
            created_at = created_at.replace(tzinfo=now.tzinfo)

        time_held_seconds = (now - created_at).total_seconds()
        time_held_minutes = time_held_seconds / 60
        remaining_minutes = max(0, max_hold_minutes - time_held_minutes)
        time_pct = (time_held_minutes / max_hold_minutes * 100) if max_hold_minutes > 0 else 0

        # Check 90% time warning (urgent)
        if time_pct >= 90 and not data["time_90_alerted"]:
            self._send_time_warning(signal_id, signal, current_price, pnl_pct, remaining_minutes, urgent=True)
            data["time_90_alerted"] = True
            with self._lock:
                if signal_id in self.open_signals:
                    self.open_signals[signal_id]["time_90_alerted"] = True

        # Check 75% time warning
        elif time_pct >= 75 and not data["time_75_alerted"]:
            self._send_time_warning(signal_id, signal, current_price, pnl_pct, remaining_minutes, urgent=False)
            data["time_75_alerted"] = True
            with self._lock:
                if signal_id in self.open_signals:
                    self.open_signals[signal_id]["time_75_alerted"] = True

    def _send_pl1_alert(
        self,
        signal_id: Any,
        signal: Any,
        current_price: float,
        pnl_pct: float,
        hold_guidance: Any,
        hold_reasoning: str,
    ) -> None:
        """Send PL1 (+20%) hit alert with case-specific recommendation."""
        log.info(f"PL1 hit for signal {signal_id}: +{pnl_pct:.1f}%")

        # Mark target hit in database
        if isinstance(signal_id, int):
            mark_target_hit(signal_id, 1)

        # Determine recommendation based on hold guidance
        if isinstance(hold_guidance, HoldGuidance):
            guidance_value = hold_guidance.value
        elif isinstance(hold_guidance, str):
            guidance_value = hold_guidance
        else:
            guidance_value = "scale_out"

        if guidance_value in ("hold_for_more", "hold"):
            recommendation = "HOLD FOR MORE - Momentum strong, wait for PL2/PL3"
        elif guidance_value in ("take_profit_now", "take_now"):
            recommendation = "TAKE PROFIT NOW - Book this win immediately"
        elif guidance_value == "tight_stop":
            recommendation = "TIGHT STOP - Raise stop to breakeven, let it run"
        else:
            recommendation = "SCALE OUT - Take 1/3 profit, hold rest for PL2"

        # Get ticker info
        if hasattr(signal, 'ticker'):
            ticker = signal.ticker
            underlying = signal.underlying
            strike = signal.strike
            direction = signal.direction
            entry_price = signal.entry_price
        else:
            ticker = signal.get('ticker', '')
            underlying = signal.get('underlying', '')
            strike = signal.get('strike', 0)
            direction = signal.get('direction', '')
            entry_price = signal.get('entry_price', 0)

        # Format and send alert
        try:
            if hasattr(signal, 'signal_id'):
                msg = format_target_hit(signal, "PL1", current_price)
            else:
                msg = (
                    f"*PL1 TARGET HIT (+20%)*\n\n"
                    f"Signal: `{signal_id}`\n"
                    f"Position: {underlying} ${strike} {direction}\n\n"
                    f"Entry: ${entry_price:.2f}\n"
                    f"Current: ${current_price:.2f}\n"
                    f"P/L: +{pnl_pct:.1f}%\n\n"
                    f"*Recommendation:*\n{recommendation}\n\n"
                    f"_{hold_reasoning}_" if hold_reasoning else ""
                )

            send_signal(msg)
            record_alert_sent(signal_id, "PL1_HIT", msg[:500])
        except Exception as e:
            log.error(f"Failed to send PL1 alert: {e}")

    def _send_pl2_alert(
        self,
        signal_id: Any,
        signal: Any,
        current_price: float,
        pnl_pct: float,
    ) -> None:
        """Send PL2 (+40%) hit alert."""
        log.info(f"PL2 hit for signal {signal_id}: +{pnl_pct:.1f}%")

        if isinstance(signal_id, int):
            mark_target_hit(signal_id, 2)

        recommendation = "SCALE OUT or HOLD - Take another 1/3, hold rest for PL3"

        try:
            if hasattr(signal, 'signal_id'):
                msg = format_target_hit(signal, "PL2", current_price)
            else:
                entry_price = signal.get('entry_price', 0)
                underlying = signal.get('underlying', '')
                strike = signal.get('strike', 0)
                direction = signal.get('direction', '')

                msg = (
                    f"*PL2 TARGET HIT (+40%)*\n\n"
                    f"Signal: `{signal_id}`\n"
                    f"Position: {underlying} ${strike} {direction}\n\n"
                    f"Entry: ${entry_price:.2f}\n"
                    f"Current: ${current_price:.2f}\n"
                    f"P/L: +{pnl_pct:.1f}%\n\n"
                    f"*Recommendation:*\n{recommendation}"
                )

            send_signal(msg)
            record_alert_sent(signal_id, "PL2_HIT", msg[:500])
        except Exception as e:
            log.error(f"Failed to send PL2 alert: {e}")

    def _send_pl3_alert(
        self,
        signal_id: Any,
        signal: Any,
        current_price: float,
        pnl_pct: float,
    ) -> None:
        """Send PL3 (+60%) hit alert."""
        log.info(f"PL3 hit for signal {signal_id}: +{pnl_pct:.1f}%")

        if isinstance(signal_id, int):
            mark_target_hit(signal_id, 3)

        # Close signal in database
        if isinstance(signal_id, int):
            entry_price = signal.get('entry_price', 0) if isinstance(signal, dict) else signal.entry_price
            update_signal_status(signal_id, "CLOSED", current_price, "PL3_HIT", pnl_pct)

        recommendation = "TAKE PROFIT - Maximum target reached. Book this excellent win!"

        try:
            if hasattr(signal, 'signal_id'):
                msg = format_target_hit(signal, "PL3", current_price)
            else:
                entry_price = signal.get('entry_price', 0)
                underlying = signal.get('underlying', '')
                strike = signal.get('strike', 0)
                direction = signal.get('direction', '')

                msg = (
                    f"*PL3 TARGET HIT (+60%)*\n\n"
                    f"Signal: `{signal_id}`\n"
                    f"Position: {underlying} ${strike} {direction}\n\n"
                    f"Entry: ${entry_price:.2f}\n"
                    f"Current: ${current_price:.2f}\n"
                    f"P/L: +{pnl_pct:.1f}%\n\n"
                    f"*Recommendation:*\n{recommendation}"
                )

            send_signal(msg)
            record_alert_sent(signal_id, "PL3_HIT", msg[:500])
        except Exception as e:
            log.error(f"Failed to send PL3 alert: {e}")

    def _send_stop_loss_alert(
        self,
        signal_id: Any,
        signal: Any,
        current_price: float,
        pnl_pct: float,
    ) -> None:
        """Send stop loss hit alert."""
        log.warning(f"STOP LOSS hit for signal {signal_id}: {pnl_pct:.1f}%")

        if isinstance(signal_id, int):
            mark_stop_hit(signal_id)
            entry_price = signal.get('entry_price', 0) if isinstance(signal, dict) else signal.entry_price
            update_signal_status(signal_id, "STOPPED", current_price, "STOP_LOSS", pnl_pct)

        recommendation = "EXIT NOW - Stop loss triggered. Preserve capital for the next trade."

        try:
            if hasattr(signal, 'signal_id'):
                msg = format_stop_loss_hit(signal, current_price)
            else:
                entry_price = signal.get('entry_price', 0)
                underlying = signal.get('underlying', '')
                strike = signal.get('strike', 0)
                direction = signal.get('direction', '')

                msg = (
                    f"*STOP LOSS TRIGGERED (-15%)*\n\n"
                    f"Signal: `{signal_id}`\n"
                    f"Position: {underlying} ${strike} {direction}\n\n"
                    f"Entry: ${entry_price:.2f}\n"
                    f"Current: ${current_price:.2f}\n"
                    f"P/L: {pnl_pct:.1f}%\n\n"
                    f"*Action Required:*\n{recommendation}"
                )

            send_signal(msg)
            record_alert_sent(signal_id, "STOP_LOSS", msg[:500])
        except Exception as e:
            log.error(f"Failed to send stop loss alert: {e}")

    def _send_time_warning(
        self,
        signal_id: Any,
        signal: Any,
        current_price: float,
        pnl_pct: float,
        remaining_minutes: float,
        urgent: bool,
    ) -> None:
        """Send time-based warning alert."""
        warning_type = "90%" if urgent else "75%"
        log.info(f"Time warning ({warning_type}) for signal {signal_id}: {remaining_minutes:.0f} min remaining")

        if urgent:
            recommendation = f"URGENT: Only {remaining_minutes:.0f} minutes remaining. Consider exiting to avoid time decay."
        else:
            recommendation = f"Monitor closely - {remaining_minutes:.0f} minutes remaining. Evaluate exit strategy."

        try:
            if hasattr(signal, 'signal_id'):
                msg = format_time_warning(signal, int(remaining_minutes), current_price)
            else:
                entry_price = signal.get('entry_price', 0)
                underlying = signal.get('underlying', '')
                strike = signal.get('strike', 0)
                direction = signal.get('direction', '')
                max_hold = signal.get('max_hold_minutes', 45)

                emoji = "*URGENT*" if urgent else "*TIME WARNING*"
                msg = (
                    f"{emoji}\n\n"
                    f"Signal: `{signal_id}`\n"
                    f"Position: {underlying} ${strike} {direction}\n\n"
                    f"Time Remaining: {remaining_minutes:.0f} minutes\n"
                    f"Max Hold: {max_hold} minutes\n\n"
                    f"Current Price: ${current_price:.2f}\n"
                    f"P/L: {'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%\n\n"
                    f"*Recommendation:*\n{recommendation}"
                )

            send_signal(msg)
            alert_type = "TIME_WARNING_90" if urgent else "TIME_WARNING_75"
            record_alert_sent(signal_id, alert_type, msg[:500])
        except Exception as e:
            log.error(f"Failed to send time warning: {e}")

    def load_open_signals(self) -> int:
        """
        Load all open signals from the database into the monitor.

        Returns:
            Number of signals loaded
        """
        try:
            open_signals = get_open_signals()
            loaded = 0

            for signal_data in open_signals:
                signal_id = signal_data.get('signal_id')
                if signal_id and signal_id not in self.open_signals:
                    self.add_signal(signal_data)
                    loaded += 1

            log.info(f"Loaded {loaded} open signals into monitor")
            return loaded
        except Exception as e:
            log.error(f"Failed to load open signals: {e}")
            return 0

    def get_status(self) -> Dict[str, Any]:
        """
        Get current monitor status.

        Returns:
            Dict with running status and monitored signals count
        """
        with self._lock:
            return {
                "running": self.running,
                "monitored_signals": len(self.open_signals),
                "signal_ids": list(self.open_signals.keys()),
            }


# Singleton instance
signal_monitor = SignalMonitor()
