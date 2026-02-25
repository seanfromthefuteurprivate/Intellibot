"""
Dual-Mode Execution Engine

WSB Snake operates in two modes simultaneously, switching based on external
intelligence from HYDRA.

MODE A: SCALP MODE (default, 90% of the time)
- Entry: Slightly ITM options (0.30-0.40 delta), NOT OTM
- Order type: Limit at mid-price, 3-second timeout, then walk to ask
- Target: +12% on option price
- Stop: -7% on option price
- Trail: At +8% gain, move stop to +3% (lock profit)

MODE B: BLOWUP MODE (rare, triggered by HYDRA signal)
- Activated when HYDRA blowup_probability > 60
- Entry type depends on HYDRA direction:
  - NEUTRAL/unknown → Buy ATM straddle (both put and call at same strike)
  - BEARISH → Buy ATM put, skip the call
  - BULLISH → Buy ATM call, skip the put
- Strike selection: ATM or 1 strike ITM for better fills
- Position size: 2x normal size (this is the big bet)
- Target: +100% on the winning leg (let it run with trailing stop)
- Stop on losing leg of straddle: Let it go to zero
- Stop on directional: -40% (wider because payoff is asymmetric)
- Trail: At +50%, trail at -20% from peak
- No time stop in blowup mode — ride the wave
- Max 2 blowup trades per day

Log format: "MODE_SWITCH: SCALP → BLOWUP (probability=73, direction=BEARISH)"
"""

import os
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class TradingMode(Enum):
    SCALP = "SCALP"
    BLOWUP = "BLOWUP"


class BlowupDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"  # Triggers straddle


@dataclass
class ScalpParams:
    """Parameters for SCALP mode."""
    delta_range: Tuple[float, float] = (0.30, 0.40)  # Slightly ITM
    target_pct: float = 12.0
    stop_pct: float = -7.0
    trail_trigger_pct: float = 8.0
    trail_stop_pct: float = 3.0  # Lock at +3% when triggered
    max_hold_seconds: int = 300  # 5 minutes
    position_size_multiplier: float = 1.0


@dataclass
class BlowupParams:
    """Parameters for BLOWUP mode."""
    strike_selection: str = "ATM"  # ATM or 1_ITM
    target_pct: float = 100.0  # Let it run
    stop_pct_directional: float = -40.0  # Wider for asymmetric payoff
    stop_pct_straddle_losing_leg: float = -100.0  # Let losing leg die
    trail_trigger_pct: float = 50.0
    trail_distance_pct: float = 20.0  # Trail at -20% from peak
    max_hold_seconds: int = None  # No time stop
    position_size_multiplier: float = 2.0  # 2x size
    max_daily_blowup_trades: int = 2


@dataclass
class HydraSignal:
    """Signal from HYDRA bridge."""
    blowup_probability: float = 0.0
    direction: BlowupDirection = BlowupDirection.NEUTRAL
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_data: Dict = field(default_factory=dict)


@dataclass
class ModeState:
    """Current mode state."""
    current_mode: TradingMode = TradingMode.SCALP
    last_switch_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    blowup_trades_today: int = 0
    last_hydra_signal: Optional[HydraSignal] = None
    hysteresis_until: Optional[datetime] = None  # Don't switch until this time


class DualModeEngine:
    """
    Dual-mode execution engine that switches between SCALP and BLOWUP modes
    based on HYDRA intelligence.

    Key features:
    - Checks current mode every 30 seconds
    - Switches to BLOWUP when probability > 60%
    - Hysteresis: stay in blowup mode for minimum 10 minutes
    - Max 2 blowup trades per day
    """

    # Mode switching thresholds
    BLOWUP_THRESHOLD = 60  # Probability to trigger blowup mode
    BLOWUP_EXIT_THRESHOLD = 40  # Probability to exit blowup mode
    HYSTERESIS_MINUTES = 10  # Minimum time in blowup mode

    def __init__(self):
        self.state = ModeState()
        self.scalp_params = ScalpParams()
        self.blowup_params = BlowupParams()
        self._running = False
        self._check_thread: Optional[threading.Thread] = None

        logger.info("DualModeEngine initialized - starting in SCALP mode")

    def get_current_mode(self) -> TradingMode:
        """Get the current trading mode."""
        return self.state.current_mode

    def get_trade_params(self) -> Dict[str, Any]:
        """
        Get trading parameters based on current mode.

        Returns:
            Dict with all relevant trading parameters
        """
        mode = self.state.current_mode

        if mode == TradingMode.SCALP:
            return {
                'mode': 'SCALP',
                'delta_range': self.scalp_params.delta_range,
                'target_pct': self.scalp_params.target_pct,
                'stop_pct': self.scalp_params.stop_pct,
                'trail_trigger_pct': self.scalp_params.trail_trigger_pct,
                'trail_stop_pct': self.scalp_params.trail_stop_pct,
                'max_hold_seconds': self.scalp_params.max_hold_seconds,
                'size_multiplier': self.scalp_params.position_size_multiplier,
                'use_straddle': False,
                'direction': None,
            }
        else:
            # BLOWUP mode
            signal = self.state.last_hydra_signal
            direction = signal.direction if signal else BlowupDirection.NEUTRAL

            return {
                'mode': 'BLOWUP',
                'delta_range': (0.50, 0.60),  # ATM range
                'target_pct': self.blowup_params.target_pct,
                'stop_pct': self.blowup_params.stop_pct_directional,
                'trail_trigger_pct': self.blowup_params.trail_trigger_pct,
                'trail_distance_pct': self.blowup_params.trail_distance_pct,
                'max_hold_seconds': self.blowup_params.max_hold_seconds,
                'size_multiplier': self.blowup_params.position_size_multiplier,
                'use_straddle': direction == BlowupDirection.NEUTRAL,
                'direction': direction.value,
                'blowup_trades_today': self.state.blowup_trades_today,
                'max_blowup_trades': self.blowup_params.max_daily_blowup_trades,
            }

    def can_trade_blowup(self) -> Tuple[bool, str]:
        """Check if a blowup trade is allowed."""
        if self.state.current_mode != TradingMode.BLOWUP:
            return False, "Not in BLOWUP mode"

        if self.state.blowup_trades_today >= self.blowup_params.max_daily_blowup_trades:
            return False, f"Max blowup trades reached ({self.state.blowup_trades_today}/{self.blowup_params.max_daily_blowup_trades})"

        return True, "OK"

    def record_blowup_trade(self):
        """Record that a blowup trade was taken."""
        self.state.blowup_trades_today += 1
        logger.info(f"BLOWUP_TRADE: {self.state.blowup_trades_today}/{self.blowup_params.max_daily_blowup_trades} today")

    def update_hydra_signal(self, signal: HydraSignal):
        """
        Update with new HYDRA signal and evaluate mode switch.

        Args:
            signal: HydraSignal with blowup_probability and direction
        """
        self.state.last_hydra_signal = signal

        old_mode = self.state.current_mode
        new_mode = self._evaluate_mode(signal)

        if new_mode != old_mode:
            self._switch_mode(new_mode, signal)

    def _evaluate_mode(self, signal: HydraSignal) -> TradingMode:
        """
        Evaluate whether to switch modes based on signal.

        Includes hysteresis to prevent flip-flopping.
        """
        now = datetime.now(timezone.utc)
        current_mode = self.state.current_mode

        # Check hysteresis - don't switch if still in minimum hold period
        if self.state.hysteresis_until and now < self.state.hysteresis_until:
            logger.debug(f"Hysteresis active until {self.state.hysteresis_until}, staying in {current_mode.value}")
            return current_mode

        if current_mode == TradingMode.SCALP:
            # Switch to BLOWUP if probability exceeds threshold
            if signal.blowup_probability >= self.BLOWUP_THRESHOLD:
                return TradingMode.BLOWUP
        else:
            # Switch back to SCALP if probability drops below exit threshold
            if signal.blowup_probability < self.BLOWUP_EXIT_THRESHOLD:
                return TradingMode.SCALP

        return current_mode

    def _switch_mode(self, new_mode: TradingMode, signal: HydraSignal):
        """Execute mode switch with logging and hysteresis setup."""
        old_mode = self.state.current_mode
        now = datetime.now(timezone.utc)

        self.state.current_mode = new_mode
        self.state.last_switch_time = now

        # Set hysteresis for blowup mode
        if new_mode == TradingMode.BLOWUP:
            self.state.hysteresis_until = now + timedelta(minutes=self.HYSTERESIS_MINUTES)

            logger.warning(
                f"MODE_SWITCH: {old_mode.value} → {new_mode.value} "
                f"(probability={signal.blowup_probability:.0f}, direction={signal.direction.value})"
            )
        else:
            self.state.hysteresis_until = None

            logger.info(
                f"MODE_SWITCH: {old_mode.value} → {new_mode.value} "
                f"(probability={signal.blowup_probability:.0f})"
            )

    def check_hydra_bridge(self) -> Optional[HydraSignal]:
        """
        Check HYDRA bridge for current blowup probability.

        Returns:
            HydraSignal or None if bridge unavailable
        """
        try:
            # Try to fetch from HYDRA bridge
            # This is a placeholder - actual implementation depends on HYDRA API
            import requests

            # FIXED: Use correct HYDRA AWS endpoint (was localhost:8001)
            hydra_url = os.environ.get('HYDRA_URL', 'http://54.172.22.157:8000') + '/api/predator'

            response = requests.get(hydra_url, timeout=5)
            if response.status_code == 200:
                data = response.json()

                direction = BlowupDirection.NEUTRAL
                if data.get('direction', '').upper() == 'BULLISH':
                    direction = BlowupDirection.BULLISH
                elif data.get('direction', '').upper() == 'BEARISH':
                    direction = BlowupDirection.BEARISH

                return HydraSignal(
                    blowup_probability=data.get('blowup_probability', 0),
                    direction=direction,
                    confidence=data.get('confidence', 0),
                    raw_data=data
                )
        except Exception as e:
            logger.debug(f"HYDRA bridge unavailable: {e}")

        return None

    def start_mode_checker(self, interval_seconds: int = 30):
        """
        Start background thread to check mode every N seconds.

        Args:
            interval_seconds: How often to check (default 30s)
        """
        if self._running:
            logger.warning("Mode checker already running")
            return

        self._running = True

        def _check_loop():
            while self._running:
                try:
                    signal = self.check_hydra_bridge()
                    if signal:
                        self.update_hydra_signal(signal)
                except Exception as e:
                    logger.error(f"Mode check error: {e}")

                time.sleep(interval_seconds)

        self._check_thread = threading.Thread(target=_check_loop, daemon=True)
        self._check_thread.start()
        logger.info(f"Mode checker started (interval={interval_seconds}s)")

    def stop_mode_checker(self):
        """Stop the background mode checker."""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5)
        logger.info("Mode checker stopped")

    def reset_daily_stats(self):
        """Reset daily counters (call at market open)."""
        self.state.blowup_trades_today = 0
        logger.info("DualModeEngine daily stats reset")

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        return {
            'current_mode': self.state.current_mode.value,
            'last_switch': self.state.last_switch_time.isoformat() if self.state.last_switch_time else None,
            'blowup_trades_today': self.state.blowup_trades_today,
            'max_blowup_trades': self.blowup_params.max_daily_blowup_trades,
            'hysteresis_until': self.state.hysteresis_until.isoformat() if self.state.hysteresis_until else None,
            'last_signal': {
                'probability': self.state.last_hydra_signal.blowup_probability if self.state.last_hydra_signal else 0,
                'direction': self.state.last_hydra_signal.direction.value if self.state.last_hydra_signal else None,
            } if self.state.last_hydra_signal else None,
            'scalp_params': {
                'target': self.scalp_params.target_pct,
                'stop': self.scalp_params.stop_pct,
                'trail_trigger': self.scalp_params.trail_trigger_pct,
            },
            'blowup_params': {
                'target': self.blowup_params.target_pct,
                'stop': self.blowup_params.stop_pct_directional,
                'trail_trigger': self.blowup_params.trail_trigger_pct,
                'trail_distance': self.blowup_params.trail_distance_pct,
            }
        }


# Singleton instance
_dual_mode_engine: Optional[DualModeEngine] = None


def get_dual_mode_engine() -> DualModeEngine:
    """Get the singleton DualModeEngine instance."""
    global _dual_mode_engine
    if _dual_mode_engine is None:
        _dual_mode_engine = DualModeEngine()
    return _dual_mode_engine


def get_current_mode() -> str:
    """
    Convenience function to get current trading mode.

    Usage:
        from wsb_snake.engines.dual_mode_engine import get_current_mode
        mode = get_current_mode()  # Returns "SCALP" or "BLOWUP"
    """
    engine = get_dual_mode_engine()
    return engine.get_current_mode().value


def get_trade_params() -> Dict[str, Any]:
    """
    Convenience function to get current trade parameters.

    Usage:
        from wsb_snake.engines.dual_mode_engine import get_trade_params
        params = get_trade_params()
        target = params['target_pct']
    """
    engine = get_dual_mode_engine()
    return engine.get_trade_params()


def is_blowup_mode() -> bool:
    """Check if currently in blowup mode."""
    engine = get_dual_mode_engine()
    return engine.get_current_mode() == TradingMode.BLOWUP
