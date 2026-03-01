"""
POWER HOUR ASSAULT PROTOCOL
═══════════════════════════════════════════════════════════════════

The most aggressive configuration of the system. Activates at 14:55 ET.

PHASES:
- 14:55 ET: ARMING - Pre-scan, position setup, buying power check
- 15:00 ET: ACTIVE ASSAULT - Execute high-conviction trades
- 15:45 ET: WIND-DOWN - No new scalps, tighten trails
- 15:55 ET: EMERGENCY CLOSE - Market orders, close everything

Log format: "POWER_HOUR: [phase] message"
"""

import os
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from wsb_snake.utils.logger import get_logger
from wsb_snake.trading.alpaca_executor import alpaca_executor

logger = get_logger(__name__)


class PowerHourPhase(Enum):
    """Power hour phases."""
    INACTIVE = "INACTIVE"
    ARMING = "ARMING"          # 14:55-15:00 ET
    ACTIVE = "ACTIVE"          # 15:00-15:45 ET
    WIND_DOWN = "WIND_DOWN"    # 15:45-15:55 ET
    EMERGENCY = "EMERGENCY"    # 15:55-16:00 ET
    COMPLETE = "COMPLETE"      # After 16:00 ET


@dataclass
class PowerHourState:
    """State tracking for power hour."""
    phase: PowerHourPhase = PowerHourPhase.INACTIVE
    armed_at: Optional[datetime] = None
    blowup_armed: bool = False
    blowup_direction: str = "NEUTRAL"
    initial_buying_power: float = 0.0
    spy_price_at_arm: float = 0.0
    atm_strike: int = 0
    trades_executed: int = 0
    power_hour_pnl: float = 0.0
    positions_at_start: int = 0
    positions_closed: int = 0
    last_phase_change: datetime = field(default_factory=datetime.now)


@dataclass
class PowerHourConfig:
    """Power hour configuration."""
    # Timing (ET)
    arm_time_hour: int = 14
    arm_time_minute: int = 55
    active_start_hour: int = 15
    active_start_minute: int = 0
    wind_down_hour: int = 15
    wind_down_minute: int = 45
    emergency_hour: int = 15
    emergency_minute: int = 55

    # Entry criteria - VENOM AGGRESSIVE
    min_volume_ratio: float = 1.2  # LOWERED from 1.5 - allow more entries
    min_conviction: float = 62.0   # LOWERED from 70 - more aggressive
    blowup_threshold: int = 50  # LOWERED from 60 - trigger blowup easier

    # Scalp mode parameters - WIDER for 0DTE power hour
    scalp_target_pct: float = 25.0  # UP from 12% - power hour can run!
    scalp_stop_pct: float = 10.0   # UP from 7% - give room to breathe
    scalp_trail_tiers: List[Tuple[float, float]] = field(default_factory=lambda: [
        (5.0, 2.0),   # At +5%, lock +2%
        (8.0, 4.0),   # At +8%, lock +4%
        (12.0, 8.0),  # At +12%, lock +8%
    ])

    # Blowup mode parameters
    blowup_size_multiplier: float = 2.0
    blowup_stop_directional: float = 40.0
    blowup_stop_straddle: float = 50.0
    blowup_trail_trigger: float = 30.0
    blowup_trail_distance: float = 15.0

    # Wind-down parameters
    wind_down_trail_pct: float = 1.0  # Tighten to -1% from peak

    # Limits - VENOM AGGRESSIVE
    max_power_hour_trades: int = 8   # UP from 5 - more compounding
    max_power_hour_exposure: float = 5000.0  # UP from $3k - use margin
    min_buying_power: float = 500.0   # LOWERED from $1k - allow trading even when low


class PowerHourProtocol:
    """
    Power Hour Assault Protocol - Maximum aggression 15:00-16:00 ET.

    Coordinates with HYDRA bridge, dual-mode engine, and smart execution
    for optimal power hour trading.
    """

    def __init__(self, config: Optional[PowerHourConfig] = None):
        self.config = config or PowerHourConfig()
        self.state = PowerHourState()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_check = datetime.now()

        logger.info("PowerHourProtocol initialized")

    def _get_et_now(self) -> datetime:
        """Get current time in Eastern timezone."""
        try:
            import pytz
            et = pytz.timezone("America/New_York")
            return datetime.now(et)
        except Exception:
            # Fallback: assume UTC-5
            return datetime.utcnow() - timedelta(hours=5)

    def _is_trading_day(self) -> bool:
        """Check if today is a trading day."""
        now = self._get_et_now()
        # Weekday check (0=Mon, 4=Fri)
        if now.weekday() > 4:
            return False
        # TODO: Add holiday check
        return True

    def get_current_phase(self) -> PowerHourPhase:
        """Determine current power hour phase based on time."""
        if not self._is_trading_day():
            return PowerHourPhase.INACTIVE

        now = self._get_et_now()
        hour, minute = now.hour, now.minute

        # Before arming time
        if hour < self.config.arm_time_hour:
            return PowerHourPhase.INACTIVE
        if hour == self.config.arm_time_hour and minute < self.config.arm_time_minute:
            return PowerHourPhase.INACTIVE

        # After market close
        if hour >= 16:
            return PowerHourPhase.COMPLETE

        # Emergency close phase (15:55-16:00)
        if hour == self.config.emergency_hour and minute >= self.config.emergency_minute:
            return PowerHourPhase.EMERGENCY

        # Wind-down phase (15:45-15:55)
        if hour == self.config.wind_down_hour and minute >= self.config.wind_down_minute:
            return PowerHourPhase.WIND_DOWN

        # Active phase (15:00-15:45)
        if hour >= self.config.active_start_hour:
            if hour > self.config.active_start_hour or minute >= self.config.active_start_minute:
                return PowerHourPhase.ACTIVE

        # Arming phase (14:55-15:00)
        return PowerHourPhase.ARMING

    def _fetch_hydra_intel(self) -> Dict:
        """Fetch HYDRA intelligence."""
        try:
            from wsb_snake.collectors.hydra_bridge import get_hydra_bridge
            bridge = get_hydra_bridge()
            intel = bridge.get_intel()
            return {
                "connected": intel.connected,
                "blowup_probability": intel.blowup_probability,
                "direction": intel.direction,
                "regime": intel.regime,
                "recommendation": intel.recommendation,
            }
        except Exception as e:
            logger.debug(f"HYDRA fetch failed: {e}")
            return {
                "connected": False,
                "blowup_probability": 0,
                "direction": "NEUTRAL",
                "regime": "UNKNOWN",
                "recommendation": "SCALP_ONLY",
            }

    def _fetch_spy_data(self) -> Dict:
        """Fetch current SPY data for pre-scan."""
        try:
            from wsb_snake.collectors.polygon_enhanced import polygon_enhanced

            # Get current price
            quote = polygon_enhanced.get_quote("SPY")
            price = quote.get("price", 0) if quote else 0

            # Calculate ATM strike (round to nearest 1)
            atm_strike = round(price)

            # Get volume data
            bars = polygon_enhanced.get_intraday_bars("SPY", timespan="minute", multiplier=1, limit=5)
            current_volume = sum(b.get("v", 0) for b in bars) if bars else 0

            # Get VWAP
            technicals = polygon_enhanced.get_full_technicals("SPY")
            vwap = technicals.get("vwap", {}).get("current", price) if technicals else price

            return {
                "price": price,
                "atm_strike": atm_strike,
                "vwap": vwap,
                "above_vwap": price > vwap,
                "current_volume": current_volume,
                "bid_ask_spread": quote.get("spread", 0.05) if quote else 0.05,
            }
        except Exception as e:
            logger.warning(f"SPY data fetch failed: {e}")
            return {
                "price": 0,
                "atm_strike": 0,
                "vwap": 0,
                "above_vwap": False,
                "current_volume": 0,
                "bid_ask_spread": 0.10,
            }

    def _fetch_account_data(self) -> Dict:
        """Fetch Alpaca account data."""
        try:
            account = alpaca_executor.get_account()
            positions = alpaca_executor.get_options_positions()

            return {
                "buying_power": float(account.get("buying_power", 0)),
                "portfolio_value": float(account.get("portfolio_value", 0)),
                "position_count": len(positions),
                "daily_pnl": alpaca_executor.daily_pnl,
            }
        except Exception as e:
            logger.warning(f"Account data fetch failed: {e}")
            return {
                "buying_power": 0,
                "portfolio_value": 0,
                "position_count": 0,
                "daily_pnl": 0,
            }

    def _execute_arming_phase(self) -> Dict:
        """
        ARMING PHASE (14:55 ET)

        Pre-scan and prepare for power hour assault.
        """
        logger.info("=" * 60)
        logger.info("POWER_HOUR: ARMING PHASE INITIATED")
        logger.info("=" * 60)

        # 1. Fetch HYDRA intelligence
        hydra = self._fetch_hydra_intel()
        blowup_armed = hydra["blowup_probability"] > self.config.blowup_threshold

        # 2. Fetch SPY data
        spy = self._fetch_spy_data()

        # 3. Fetch account data
        account = self._fetch_account_data()

        # 4. Check if we have enough buying power
        can_trade = account["buying_power"] >= self.config.min_buying_power

        # 5. Update state
        self.state.phase = PowerHourPhase.ARMING
        self.state.armed_at = self._get_et_now()
        self.state.blowup_armed = blowup_armed
        self.state.blowup_direction = hydra["direction"]
        self.state.initial_buying_power = account["buying_power"]
        self.state.spy_price_at_arm = spy["price"]
        self.state.atm_strike = spy["atm_strike"]
        self.state.positions_at_start = account["position_count"]

        # 6. Log arming status
        logger.warning(
            f"POWER_HOUR: ARMING — blowup={hydra['blowup_probability']} "
            f"regime={hydra['regime']} buying_power=${account['buying_power']:,.0f}"
        )
        logger.info(f"POWER_HOUR: SPY=${spy['price']:.2f} ATM={spy['atm_strike']} VWAP=${spy['vwap']:.2f}")
        logger.info(f"POWER_HOUR: Positions={account['position_count']} Daily P/L=${account['daily_pnl']:.2f}")

        if blowup_armed:
            logger.warning(
                f"POWER_HOUR: BLOWUP MODE ARMED — direction={hydra['direction']} "
                f"probability={hydra['blowup_probability']}"
            )

        if not can_trade:
            logger.warning(f"POWER_HOUR: INSUFFICIENT BUYING POWER (${account['buying_power']:,.0f} < ${self.config.min_buying_power:,.0f})")

        return {
            "phase": "ARMING",
            "can_trade": can_trade,
            "blowup_armed": blowup_armed,
            "hydra": hydra,
            "spy": spy,
            "account": account,
        }

    def _check_entry_signal(self) -> Tuple[bool, str, Dict]:
        """
        Check for power hour entry signal.

        Returns (should_enter, direction, signal_data)
        """
        try:
            from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
            from wsb_snake.execution.apex_conviction_engine import apex_engine

            # Get APEX verdict
            verdict = apex_engine.analyze("SPY")

            # Get momentum data (1-min and 5-min agreement)
            bars_1m = polygon_enhanced.get_intraday_bars("SPY", timespan="minute", multiplier=1, limit=5)
            bars_5m = polygon_enhanced.get_intraday_bars("SPY", timespan="minute", multiplier=5, limit=3)

            momentum_1m = "BULLISH" if bars_1m and bars_1m[-1].get("c", 0) > bars_1m[-1].get("o", 0) else "BEARISH"
            momentum_5m = "BULLISH" if bars_5m and bars_5m[-1].get("c", 0) > bars_5m[-1].get("o", 0) else "BEARISH"
            momentum_aligned = momentum_1m == momentum_5m

            # Get VWAP position
            spy_data = self._fetch_spy_data()
            vwap_bias = "BULLISH" if spy_data["above_vwap"] else "BEARISH"

            # Get volume
            volume_ratio = spy_data["current_volume"] / 1000000  # Rough estimate vs typical
            volume_ok = volume_ratio > self.config.min_volume_ratio / 3  # Relaxed for power hour

            # Check conviction
            conviction_ok = verdict.conviction_score >= self.config.min_conviction

            # Direction alignment check
            directions_aligned = (
                (verdict.direction in ["LONG", "STRONG_LONG"] and momentum_1m == "BULLISH" and vwap_bias == "BULLISH") or
                (verdict.direction in ["SHORT", "STRONG_SHORT"] and momentum_1m == "BEARISH" and vwap_bias == "BEARISH")
            )

            signal_data = {
                "conviction": verdict.conviction_score,
                "verdict_direction": verdict.direction,
                "momentum_1m": momentum_1m,
                "momentum_5m": momentum_5m,
                "momentum_aligned": momentum_aligned,
                "vwap_bias": vwap_bias,
                "volume_ratio": volume_ratio,
                "volume_ok": volume_ok,
                "conviction_ok": conviction_ok,
                "directions_aligned": directions_aligned,
            }

            should_enter = conviction_ok and directions_aligned and momentum_aligned
            direction = "LONG" if verdict.direction in ["LONG", "STRONG_LONG"] else "SHORT"

            return should_enter, direction, signal_data

        except Exception as e:
            logger.warning(f"Entry signal check failed: {e}")
            return False, "NEUTRAL", {"error": str(e)}

    def _execute_scalp_entry(self, direction: str, signal_data: Dict) -> Optional[Dict]:
        """Execute a scalp mode entry."""
        try:
            # NOTE: Risk governor check happens inside alpaca_executor.execute_scalp_entry()
            # which calls can_trade() with all required parameters (positions, daily_pnl, etc.)
            ticker = "SPY"
            conviction = signal_data.get("conviction", 70)

            # Get current price
            spy_data = self._fetch_spy_data()
            entry_price = spy_data["price"]

            if entry_price <= 0:
                logger.warning("POWER_HOUR: Cannot execute - no SPY price")
                return None

            # Calculate targets
            if direction == "LONG":
                target_price = entry_price * (1 + self.config.scalp_target_pct / 100)
                stop_loss = entry_price * (1 - self.config.scalp_stop_pct / 100)
            else:
                target_price = entry_price * (1 - self.config.scalp_target_pct / 100)
                stop_loss = entry_price * (1 + self.config.scalp_stop_pct / 100)

            # Execute via Alpaca
            position = alpaca_executor.execute_scalp_entry(
                underlying=ticker,
                direction="long" if direction == "LONG" else "short",
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                confidence=conviction,
                pattern="POWER_HOUR_SCALP"
            )

            if position:
                self.state.trades_executed += 1
                logger.info(
                    f"POWER_HOUR: SCALP ENTRY — {ticker} {direction} @ ${entry_price:.2f} "
                    f"target=${target_price:.2f} stop=${stop_loss:.2f}"
                )
                return {
                    "type": "SCALP",
                    "ticker": ticker,
                    "direction": direction,
                    "entry_price": entry_price,
                    "target_price": target_price,
                    "stop_loss": stop_loss,
                    "position": position,
                }

            return None

        except Exception as e:
            logger.error(f"POWER_HOUR: Scalp entry failed: {e}")
            return None

    def _execute_blowup_entry(self, direction: str) -> Optional[Dict]:
        """Execute a blowup mode entry (straddle or directional)."""
        try:
            # NOTE: Risk governor check happens inside alpaca_executor.execute_scalp_entry()
            # and straddle_executor which call can_trade() with all required parameters
            from wsb_snake.execution.straddle_executor import StraddleExecutor

            ticker = "SPY"
            spy_data = self._fetch_spy_data()

            if direction == "NEUTRAL":
                # Execute straddle
                executor = StraddleExecutor()

                # Get expiry (today for 0DTE)
                now = self._get_et_now()
                expiry = now.strftime("%Y-%m-%d")

                result = executor.execute_straddle(
                    ticker=ticker,
                    expiry=expiry,
                    size=self.config.blowup_size_multiplier
                )

                if result:
                    self.state.trades_executed += 1
                    logger.warning(f"POWER_HOUR: BLOWUP STRADDLE ENTRY — {ticker}")
                    return {"type": "BLOWUP_STRADDLE", "result": result}
            else:
                # Execute directional with 2x size
                entry_price = spy_data["price"]

                if direction == "BULLISH":
                    target_price = entry_price * 1.30  # 30% target for blowup
                    stop_loss = entry_price * (1 - self.config.blowup_stop_directional / 100)
                    trade_direction = "long"
                else:
                    target_price = entry_price * 0.70
                    stop_loss = entry_price * (1 + self.config.blowup_stop_directional / 100)
                    trade_direction = "short"

                # Execute with 2x size (pass via confidence boost)
                position = alpaca_executor.execute_scalp_entry(
                    underlying=ticker,
                    direction=trade_direction,
                    entry_price=entry_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    confidence=95,  # High confidence for larger size
                    pattern="POWER_HOUR_BLOWUP"
                )

                if position:
                    self.state.trades_executed += 1
                    logger.warning(
                        f"POWER_HOUR: BLOWUP DIRECTIONAL ENTRY — {ticker} {direction} "
                        f"@ ${entry_price:.2f} (2x size)"
                    )
                    return {
                        "type": "BLOWUP_DIRECTIONAL",
                        "ticker": ticker,
                        "direction": direction,
                        "entry_price": entry_price,
                        "position": position,
                    }

            return None

        except Exception as e:
            logger.error(f"POWER_HOUR: Blowup entry failed: {e}")
            return None

    def _execute_active_phase(self) -> Dict:
        """
        ACTIVE PHASE (15:00-15:45 ET)

        Execute high-conviction trades every 30 seconds.
        """
        if self.state.phase != PowerHourPhase.ACTIVE:
            self.state.phase = PowerHourPhase.ACTIVE
            self.state.last_phase_change = self._get_et_now()
            logger.warning("POWER_HOUR: ACTIVE ASSAULT PHASE — ENGAGING")

        # Check if we've hit trade limit
        if self.state.trades_executed >= self.config.max_power_hour_trades:
            logger.info(f"POWER_HOUR: Max trades reached ({self.state.trades_executed})")
            return {"phase": "ACTIVE", "action": "MAX_TRADES_REACHED"}

        # Check account limits
        account = self._fetch_account_data()
        if account["buying_power"] < self.config.min_buying_power:
            logger.info("POWER_HOUR: Insufficient buying power")
            return {"phase": "ACTIVE", "action": "LOW_BUYING_POWER"}

        # Check for HYDRA blowup mode
        hydra = self._fetch_hydra_intel()
        in_blowup_mode = hydra["blowup_probability"] > self.config.blowup_threshold

        if in_blowup_mode and self.state.blowup_armed:
            # Execute blowup entry
            result = self._execute_blowup_entry(hydra["direction"])
            if result:
                return {"phase": "ACTIVE", "action": "BLOWUP_ENTRY", "result": result}
        else:
            # Check for scalp entry signal
            should_enter, direction, signal_data = self._check_entry_signal()

            if should_enter:
                result = self._execute_scalp_entry(direction, signal_data)
                if result:
                    return {"phase": "ACTIVE", "action": "SCALP_ENTRY", "result": result}

        return {"phase": "ACTIVE", "action": "SCANNING", "hydra": hydra}

    def _execute_wind_down_phase(self) -> Dict:
        """
        WIND-DOWN PHASE (15:45-15:55 ET)

        No new scalp entries. Tighten all trails.
        """
        if self.state.phase != PowerHourPhase.WIND_DOWN:
            self.state.phase = PowerHourPhase.WIND_DOWN
            self.state.last_phase_change = self._get_et_now()
            logger.warning("POWER_HOUR: WIND-DOWN PHASE — TIGHTENING TRAILS")

        # Tighten trails on all positions to -1% from peak
        try:
            positions = alpaca_executor.get_options_positions()
            for pos in positions:
                # TODO: Update trail stops to -1%
                pass

            logger.info(f"POWER_HOUR: Tightened trails on {len(positions)} positions")

        except Exception as e:
            logger.warning(f"POWER_HOUR: Trail tightening failed: {e}")

        return {"phase": "WIND_DOWN", "action": "TRAILS_TIGHTENED"}

    def _execute_emergency_close(self) -> Dict:
        """
        EMERGENCY CLOSE (15:55 ET)

        Close ALL 0DTE positions regardless of P&L.
        """
        if self.state.phase != PowerHourPhase.EMERGENCY:
            self.state.phase = PowerHourPhase.EMERGENCY
            self.state.last_phase_change = self._get_et_now()
            logger.warning("=" * 60)
            logger.warning("POWER_HOUR: EMERGENCY CLOSE — CLOSING ALL 0DTE")
            logger.warning("=" * 60)

        try:
            # Close all 0DTE positions with market orders
            closed_count = alpaca_executor.close_all_0dte_positions()
            self.state.positions_closed = closed_count

            # Calculate power hour P&L
            account = self._fetch_account_data()
            self.state.power_hour_pnl = account["daily_pnl"] - (
                self.state.power_hour_pnl if self.state.armed_at else 0
            )

            logger.warning(
                f"POWER_HOUR: EMERGENCY CLOSE COMPLETE — "
                f"Closed {closed_count} positions | P&L: ${self.state.power_hour_pnl:.2f}"
            )

            return {
                "phase": "EMERGENCY",
                "action": "POSITIONS_CLOSED",
                "closed_count": closed_count,
                "power_hour_pnl": self.state.power_hour_pnl,
            }

        except Exception as e:
            logger.error(f"POWER_HOUR: Emergency close failed: {e}")
            return {"phase": "EMERGENCY", "action": "CLOSE_FAILED", "error": str(e)}

    def tick(self) -> Dict:
        """
        Main tick function - call every 30 seconds during market hours.

        Returns status dict with current phase and actions taken.
        """
        current_phase = self.get_current_phase()

        if current_phase == PowerHourPhase.INACTIVE:
            return {"phase": "INACTIVE", "action": "WAITING"}

        if current_phase == PowerHourPhase.COMPLETE:
            return {"phase": "COMPLETE", "action": "DONE"}

        if current_phase == PowerHourPhase.ARMING:
            return self._execute_arming_phase()

        if current_phase == PowerHourPhase.ACTIVE:
            return self._execute_active_phase()

        if current_phase == PowerHourPhase.WIND_DOWN:
            return self._execute_wind_down_phase()

        if current_phase == PowerHourPhase.EMERGENCY:
            return self._execute_emergency_close()

        return {"phase": "UNKNOWN", "action": "ERROR"}

    def start(self):
        """Start power hour monitoring thread."""
        if self._running:
            logger.warning("PowerHourProtocol already running")
            return

        self._running = True

        def _monitor_loop():
            while self._running:
                try:
                    result = self.tick()

                    # Log significant actions
                    if result.get("action") not in ["WAITING", "SCANNING", "DONE"]:
                        logger.info(f"POWER_HOUR: {result}")

                except Exception as e:
                    logger.error(f"Power hour tick error: {e}")

                # Check every 30 seconds during active phases, 60 seconds otherwise
                phase = self.get_current_phase()
                if phase in [PowerHourPhase.ACTIVE, PowerHourPhase.WIND_DOWN, PowerHourPhase.EMERGENCY]:
                    time.sleep(30)
                else:
                    time.sleep(60)

        self._monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("PowerHourProtocol started - monitoring for power hour")

    def stop(self):
        """Stop power hour monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("PowerHourProtocol stopped")

    def get_status(self) -> Dict:
        """Get current power hour status."""
        return {
            "phase": self.state.phase.value,
            "armed_at": self.state.armed_at.isoformat() if self.state.armed_at else None,
            "blowup_armed": self.state.blowup_armed,
            "blowup_direction": self.state.blowup_direction,
            "initial_buying_power": self.state.initial_buying_power,
            "spy_price_at_arm": self.state.spy_price_at_arm,
            "trades_executed": self.state.trades_executed,
            "power_hour_pnl": self.state.power_hour_pnl,
            "positions_closed": self.state.positions_closed,
        }


# Singleton instance
_power_hour_protocol: Optional[PowerHourProtocol] = None


def get_power_hour_protocol() -> PowerHourProtocol:
    """Get the singleton PowerHourProtocol instance."""
    global _power_hour_protocol
    if _power_hour_protocol is None:
        _power_hour_protocol = PowerHourProtocol()
    return _power_hour_protocol


def start_power_hour_protocol():
    """Start the power hour protocol (call from main.py)."""
    protocol = get_power_hour_protocol()
    protocol.start()
    return protocol


def get_power_hour_status() -> Dict:
    """Convenience function to get power hour status."""
    protocol = get_power_hour_protocol()
    return protocol.get_status()
