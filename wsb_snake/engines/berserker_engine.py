"""
BERSERKER Engine - Aggressive 0DTE SPX trading for high-edge GEX setups.

Activation Conditions (ALL must be true):
1. GEX flip proximity < 0.3% (price near dealer flip point)
2. HYDRA direction is BULLISH or BEARISH (not NEUTRAL)
3. Flow bias confirms direction
4. VIX < 25

Behaviors:
- SPX-only (not SPY) - larger contracts
- 3x position size (aggressive)
- Target: +50% on options
- Stop: -15% (tight because high-probability)
- Max hold: 30 minutes
- Max 3 trades per day

GEX Integration (from REPO_STUDY_INTEGRATION.md):
- Uses gex_calculator module for Black-Scholes gamma calculations
- Tracks dealer gamma exposure for directional bias
- Identifies zero gamma level (flip point) for entry timing

Log format: "BERSERKER: ACTIVATED gex_flip=0.25% dir=BULLISH flow=AGGRESSIVELY_BULLISH"
"""

import os
import threading
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from wsb_snake.utils.logger import get_logger
from wsb_snake.utils.time_utils import is_market_open
from wsb_snake.learning.gex_calculator import (
    get_gex_calculator,
    GEXResult,
    BerserkerSignal as GEXBerserkerSignal,
    check_berserker_conditions_from_hydra,
    parse_hydra_gex_data,
)

logger = get_logger(__name__)


@dataclass
class BerserkerConfig:
    """Configuration for BERSERKER mode."""
    # Entry conditions
    gex_flip_proximity_threshold: float = 0.3  # % from flip point (0.3 = 0.3%)
    min_hydra_confidence: float = 50.0
    max_vix: float = 25.0
    require_flow_confirmation: bool = True

    # Position sizing
    position_multiplier: float = 3.0  # 3x normal size
    max_position_value: float = 3000.0  # $3k max per BERSERKER trade

    # Exit rules
    target_pct: float = 50.0   # +50% on options
    stop_pct: float = -15.0    # -15% stop (tight)
    max_hold_minutes: int = 30  # 30 min max

    # Limits
    max_daily_trades: int = 3

    # Timing
    scan_interval_seconds: int = 30
    cooldown_minutes: int = 15  # Min time between BERSERKER trades


@dataclass
class BerserkerSignal:
    """A BERSERKER trading signal."""
    symbol: str  # "SPX" or "SPXW"
    direction: str  # "CALL" or "PUT"
    confidence: float
    gex_regime: str
    gex_flip_distance_pct: float
    hydra_direction: str
    flow_bias: str
    vix_level: float
    entry_price: float  # Underlying price
    strike: float
    expiry: datetime
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": self.confidence,
            "gex_regime": self.gex_regime,
            "gex_flip_distance_pct": self.gex_flip_distance_pct,
            "hydra_direction": self.hydra_direction,
            "flow_bias": self.flow_bias,
            "vix_level": self.vix_level,
            "entry_price": self.entry_price,
            "strike": self.strike,
            "expiry": self.expiry.isoformat() if self.expiry else None,
            "generated_at": self.generated_at.isoformat(),
        }


class BerserkerEngine:
    """
    BERSERKER - Aggressive 0DTE SPX engine for high-edge GEX setups.

    This engine only trades SPX (not SPY) for larger contract sizes
    when GEX conditions indicate a high-probability directional move.

    The engine polls every 30 seconds during market hours and checks
    if all BERSERKER activation conditions are met.
    """

    def __init__(self, config: Optional[BerserkerConfig] = None):
        """Initialize BERSERKER engine."""
        self.config = config or BerserkerConfig()
        self.trades_today = 0
        self.running = False
        self._coordinator = None
        self._thread: Optional[threading.Thread] = None
        self._last_signal_time: Optional[datetime] = None
        self._last_trade_time: Optional[datetime] = None
        self._activation_log: List[Dict] = []

        # Stats
        self._stats = {
            "scans": 0,
            "activations": 0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "gex_calculations": 0,
        }

        # GEX calculator integration
        self._gex_calculator = get_gex_calculator()
        self._last_gex_result: Optional[GEXResult] = None
        self._last_gex_signal: Optional[GEXBerserkerSignal] = None

        logger.info("BERSERKER: Engine initialized with GEX calculator - watching for gamma edge opportunities")

    def set_coordinator(self, coordinator) -> None:
        """Set the strategy coordinator for trade submission."""
        self._coordinator = coordinator
        if hasattr(coordinator, 'register_engine'):
            coordinator.register_engine("berserker", "GEX_AGGRESSIVE_0DTE", {
                "gex_threshold": self.config.gex_flip_proximity_threshold,
                "max_vix": self.config.max_vix,
                "position_multiplier": self.config.position_multiplier,
            })
        logger.info("BERSERKER: Registered with coordinator")

    def start(self) -> None:
        """Start BERSERKER monitoring."""
        if self.running:
            logger.warning("BERSERKER: Already running")
            return

        self.running = True
        self._thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._thread.start()

        logger.info(f"BERSERKER: Started - scanning every {self.config.scan_interval_seconds}s")

    def stop(self) -> None:
        """Stop BERSERKER monitoring."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("BERSERKER: Stopped")

    def _scan_loop(self) -> None:
        """Main scanning loop."""
        while self.running:
            try:
                if is_market_open():
                    self._stats["scans"] += 1
                    self._check_and_execute()
                else:
                    # Reset daily stats at market open
                    if self._should_reset_daily():
                        self._reset_daily_stats()

            except Exception as e:
                logger.error(f"BERSERKER: Scan error - {e}")

            time.sleep(self.config.scan_interval_seconds)

    def _check_and_execute(self) -> None:
        """Check activation conditions and execute if met."""
        # Check cooldown
        if self._last_trade_time:
            cooldown_end = self._last_trade_time + timedelta(minutes=self.config.cooldown_minutes)
            if datetime.now(timezone.utc) < cooldown_end:
                logger.debug(f"BERSERKER: In cooldown until {cooldown_end.strftime('%H:%M:%S')}")
                return

        # Check activation conditions
        activated, reason, context = self._check_activation_conditions()

        self._log_activation_check(activated, reason, context)

        if not activated:
            logger.debug(f"BERSERKER: Not activated - {reason}")
            return

        # Generate signal
        signal = self._generate_signal(context)
        if not signal:
            logger.warning("BERSERKER: Failed to generate signal")
            return

        self._stats["activations"] += 1
        logger.warning(
            f"BERSERKER ACTIVATED: gex_flip={signal.gex_flip_distance_pct:.2f}% "
            f"dir={signal.hydra_direction} flow={signal.flow_bias} "
            f"-> {signal.direction} @ {signal.strike}"
        )

        # Submit through coordinator
        self._execute_signal(signal)

    def _check_activation_conditions(self) -> Tuple[bool, str, Dict]:
        """
        Check all BERSERKER activation conditions.

        Uses enhanced GEX analysis from gex_calculator module for
        more precise gamma exposure calculations.

        Returns: (should_activate, reason, context_data)
        """
        context = {}

        try:
            # Get HYDRA intelligence
            from wsb_snake.collectors.hydra_bridge import get_hydra_intel, get_hydra_predator_raw
            intel = get_hydra_intel()

            context = {
                "gex_flip_distance_pct": intel.gex_flip_distance_pct,
                "gex_regime": intel.gex_regime,
                "gex_flip_point": intel.gex_flip_point,
                "hydra_direction": intel.direction,
                "flow_bias": intel.flow_bias,
                "vix_level": intel.vix_level,
                "blowup_probability": intel.blowup_probability,
                "confidence": intel.confidence,
                "connected": intel.connected,
            }

            # Check HYDRA connection
            if not intel.connected:
                return False, "HYDRA not connected", context

            # ENHANCED: Perform GEX calculator analysis
            try:
                hydra_raw = get_hydra_predator_raw()
                if hydra_raw:
                    gex_signal, gex_reasoning = self._perform_gex_analysis(hydra_raw)

                    if gex_signal:
                        context["gex_analysis"] = {
                            "should_activate": gex_signal.should_activate,
                            "direction": gex_signal.direction,
                            "confidence": gex_signal.confidence,
                            "flip_proximity": gex_signal.flip_proximity,
                            "regime": gex_signal.gex_regime,
                            "reasoning": gex_reasoning,
                        }

                        # Use GEX calculator's more precise flip proximity
                        if gex_signal.flip_proximity < intel.gex_flip_distance_pct / 100:
                            context["gex_flip_distance_pct"] = gex_signal.flip_proximity * 100
                            logger.debug(f"BERSERKER: Using GEX calculator proximity: {gex_signal.flip_proximity:.3%}")
            except Exception as e:
                logger.debug(f"BERSERKER: GEX calculator fallback - {e}")

            # Condition 1: GEX flip proximity
            if context["gex_flip_distance_pct"] > self.config.gex_flip_proximity_threshold:
                return False, f"GEX flip too far: {context['gex_flip_distance_pct']:.2f}% > {self.config.gex_flip_proximity_threshold}%", context

            # Condition 2: HYDRA direction not neutral
            if intel.direction == "NEUTRAL":
                return False, "HYDRA direction is NEUTRAL", context

            # Condition 3: Flow confirmation
            if self.config.require_flow_confirmation:
                flow_agrees = self._check_flow_confirmation(intel.direction, intel.flow_bias)
                if not flow_agrees:
                    return False, f"Flow bias ({intel.flow_bias}) doesn't confirm direction ({intel.direction})", context

            # Condition 4: VIX check
            if intel.vix_level >= self.config.max_vix:
                return False, f"VIX {intel.vix_level:.1f} >= {self.config.max_vix} threshold", context

            # Condition 5: Daily limit
            if self.trades_today >= self.config.max_daily_trades:
                return False, f"Max daily BERSERKER trades reached ({self.trades_today})", context

            # Condition 6: Minimum confidence
            if intel.confidence < self.config.min_hydra_confidence:
                return False, f"HYDRA confidence {intel.confidence:.0f}% < {self.config.min_hydra_confidence}% threshold", context

            # ENHANCED: Check GEX calculator signal
            if context.get("gex_analysis", {}).get("should_activate"):
                confidence_boost = context["gex_analysis"]["confidence"]
                context["gex_confidence_boost"] = confidence_boost
                logger.info(f"BERSERKER: GEX calculator confirms activation (confidence boost: {confidence_boost:.0%})")

            return True, "All BERSERKER conditions met", context

        except ImportError:
            return False, "HYDRA bridge not available", context
        except Exception as e:
            return False, f"Error checking conditions: {e}", context

    def _check_flow_confirmation(self, hydra_direction: str, flow_bias: str) -> bool:
        """Check if flow bias confirms HYDRA direction."""
        bullish_flows = ["BULLISH", "AGGRESSIVELY_BULLISH"]
        bearish_flows = ["BEARISH", "AGGRESSIVELY_BEARISH"]

        if hydra_direction == "BULLISH" and flow_bias in bullish_flows:
            return True
        if hydra_direction == "BEARISH" and flow_bias in bearish_flows:
            return True

        return False

    def _perform_gex_analysis(self, hydra_predator_response: Dict) -> Tuple[Optional[GEXBerserkerSignal], str]:
        """
        Perform enhanced GEX analysis using the gex_calculator module.

        This provides more precise gamma exposure calculations using
        Black-Scholes gamma and identifies optimal entry conditions.

        Args:
            hydra_predator_response: Response from HYDRA /api/predator

        Returns:
            (GEXBerserkerSignal, reasoning_str)
        """
        try:
            self._stats["gex_calculations"] += 1

            # Use the gex_calculator helper to check conditions
            gex_signal = check_berserker_conditions_from_hydra(hydra_predator_response)

            # Parse GEX data for detailed analysis
            gex_result = parse_hydra_gex_data(hydra_predator_response)
            if gex_result:
                self._last_gex_result = gex_result

            self._last_gex_signal = gex_signal

            # Build reasoning
            reasoning_parts = []

            if gex_result:
                reasoning_parts.append(f"GEX regime: {gex_result.regime}")
                reasoning_parts.append(f"Net GEX: ${gex_result.total_gex:.1f}B")
                if gex_result.gamma_flip:
                    reasoning_parts.append(f"Flip point: {gex_result.gamma_flip:.0f}")
                if gex_result.support_levels:
                    reasoning_parts.append(f"Supports: {gex_result.support_levels[:2]}")
                if gex_result.resistance_levels:
                    reasoning_parts.append(f"Resistances: {gex_result.resistance_levels[:2]}")

            reasoning_parts.append(f"Signal: {gex_signal.reasoning}")

            reasoning = " | ".join(reasoning_parts)

            return gex_signal, reasoning

        except Exception as e:
            logger.warning(f"BERSERKER: GEX analysis error - {e}")
            return None, f"GEX analysis failed: {e}"

    def get_gex_context(self) -> Dict[str, Any]:
        """
        Get current GEX context for external use.

        Returns detailed GEX analysis for other components that may
        want to incorporate gamma exposure data.
        """
        context = {
            "has_gex_data": self._last_gex_result is not None,
            "last_signal": None,
            "gex_result": None,
        }

        if self._last_gex_result:
            context["gex_result"] = {
                "total_gex": self._last_gex_result.total_gex,
                "call_gex": self._last_gex_result.call_gex,
                "put_gex": self._last_gex_result.put_gex,
                "gamma_flip": self._last_gex_result.gamma_flip,
                "flip_proximity": self._last_gex_result.flip_proximity,
                "regime": self._last_gex_result.regime,
                "support_levels": self._last_gex_result.support_levels,
                "resistance_levels": self._last_gex_result.resistance_levels,
            }

        if self._last_gex_signal:
            context["last_signal"] = {
                "should_activate": self._last_gex_signal.should_activate,
                "direction": self._last_gex_signal.direction,
                "confidence": self._last_gex_signal.confidence,
                "gex_regime": self._last_gex_signal.gex_regime,
                "flip_proximity": self._last_gex_signal.flip_proximity,
                "reasoning": self._last_gex_signal.reasoning,
            }

        return context

    def _generate_signal(self, context: Dict) -> Optional[BerserkerSignal]:
        """Generate BERSERKER signal from context."""
        try:
            # Get SPX current price
            spx_price = self._get_spx_price()
            if not spx_price:
                logger.warning("BERSERKER: Could not get SPX price")
                return None

            # Determine direction
            hydra_dir = context.get("hydra_direction", "NEUTRAL")
            option_direction = "CALL" if hydra_dir == "BULLISH" else "PUT"

            # Calculate ATM strike (SPX uses 5-point strikes)
            strike = round(spx_price / 5) * 5

            # Get today's 0DTE expiry
            expiry = self._get_0dte_expiry()

            # Calculate confidence
            confidence = min(95, 60 + (self.config.gex_flip_proximity_threshold - context.get("gex_flip_distance_pct", 0)) * 100)

            return BerserkerSignal(
                symbol="SPXW",  # SPX weeklies
                direction=option_direction,
                confidence=confidence,
                gex_regime=context.get("gex_regime", "UNKNOWN"),
                gex_flip_distance_pct=context.get("gex_flip_distance_pct", 999),
                hydra_direction=hydra_dir,
                flow_bias=context.get("flow_bias", "NEUTRAL"),
                vix_level=context.get("vix_level", 20),
                entry_price=spx_price,
                strike=strike,
                expiry=expiry,
            )

        except Exception as e:
            logger.error(f"BERSERKER: Signal generation error - {e}")
            return None

    def _get_spx_price(self) -> Optional[float]:
        """Get current SPX price."""
        try:
            # Try to get from HYDRA GEX flip point as proxy
            from wsb_snake.collectors.hydra_bridge import get_hydra_intel
            intel = get_hydra_intel()

            if intel.gex_flip_point > 0:
                # Estimate SPX from flip point and distance
                flip_dist_pct = intel.gex_flip_distance_pct / 100
                # If flip distance is positive, we're above flip; if negative, below
                # Approximate: price = flip_point * (1 + flip_dist_pct)
                return intel.gex_flip_point * (1 + flip_dist_pct)

            # Fallback: Use SPY price * 10 as rough estimate
            try:
                from wsb_snake.collectors.polygon_options import get_stock_quote
                spy_quote = get_stock_quote("SPY")
                if spy_quote:
                    return spy_quote.get("price", 0) * 10
            except:
                pass

            return None

        except Exception as e:
            logger.warning(f"BERSERKER: Could not get SPX price - {e}")
            return None

    def _get_0dte_expiry(self) -> datetime:
        """Get today's 0DTE expiry datetime."""
        now = datetime.now(timezone.utc)
        # SPX options expire at 4:00 PM ET
        # ET is UTC-5 (or UTC-4 during DST)
        # Approximate: set to 21:00 UTC (4 PM ET during EST)
        return now.replace(hour=21, minute=0, second=0, microsecond=0)

    def _execute_signal(self, signal: BerserkerSignal) -> None:
        """Execute BERSERKER signal through coordinator."""
        if not self._coordinator:
            logger.warning("BERSERKER: No coordinator set - cannot execute")
            return

        try:
            from wsb_snake.coordination.strategy_coordinator import TradeRequest

            # Build trade request
            request = TradeRequest(
                request_id=f"berserker_{signal.symbol}_{datetime.now().strftime('%H%M%S')}",
                engine="berserker",
                ticker=signal.symbol,
                direction="long",  # We buy calls or puts, not short
                entry_price=signal.entry_price,
                target_price=signal.entry_price * (1 + self.config.target_pct / 100) if signal.direction == "CALL" else signal.entry_price * (1 - self.config.target_pct / 100),
                stop_loss=signal.entry_price * (1 - abs(self.config.stop_pct) / 100) if signal.direction == "CALL" else signal.entry_price * (1 + abs(self.config.stop_pct) / 100),
                confidence=signal.confidence,
                pattern=f"BERSERKER_{signal.direction}",
                priority=1,  # Highest priority
                expiry_preference="0dte",
                gex_regime=signal.gex_regime,
                hydra_direction=signal.hydra_direction,
                flow_bias=signal.flow_bias,
                metadata={
                    "strike": signal.strike,
                    "option_direction": signal.direction,
                    "gex_flip_distance_pct": signal.gex_flip_distance_pct,
                    "vix_level": signal.vix_level,
                    "position_multiplier": self.config.position_multiplier,
                    "max_hold_minutes": self.config.max_hold_minutes,
                },
            )

            response = self._coordinator.submit_trade_request(request)

            if response.executed:
                self.trades_today += 1
                self._last_trade_time = datetime.now(timezone.utc)
                self._stats["trades"] += 1

                logger.warning(
                    f"BERSERKER STRIKE: {signal.direction} {signal.symbol} @ {signal.strike} "
                    f"(position_id={response.position_id})"
                )

                # Send Telegram alert
                self._send_berserker_alert(signal, response)

            elif response.queued:
                logger.info(f"BERSERKER: Trade queued at position {response.queue_position}")

            else:
                logger.warning(f"BERSERKER: Trade blocked - {response.reason}")

        except Exception as e:
            logger.error(f"BERSERKER: Execution error - {e}")

    def _send_berserker_alert(self, signal: BerserkerSignal, response) -> None:
        """Send Telegram alert for BERSERKER trade."""
        try:
            from wsb_snake.notifications.telegram_bot import send_telegram_alert

            # Build GEX context line
            gex_line = ""
            if self._last_gex_result:
                gex_line = (
                    f"GEX: ${self._last_gex_result.total_gex:.1f}B ({self._last_gex_result.regime})\n"
                )
                if self._last_gex_result.gamma_flip:
                    gex_line += f"Flip Point: {self._last_gex_result.gamma_flip:.0f}\n"
                if self._last_gex_result.support_levels:
                    gex_line += f"Supports: {self._last_gex_result.support_levels[:2]}\n"
                if self._last_gex_result.resistance_levels:
                    gex_line += f"Resistance: {self._last_gex_result.resistance_levels[:2]}\n"

            alert_msg = (
                f"BERSERKER STRIKE\n\n"
                f"{signal.direction} {signal.symbol} @ {signal.strike}\n\n"
                f"GEX Flip Proximity: {signal.gex_flip_distance_pct:.2f}%\n"
                f"{gex_line}"
                f"Direction: {signal.hydra_direction}\n"
                f"Flow: {signal.flow_bias}\n"
                f"VIX: {signal.vix_level:.1f}\n"
                f"Confidence: {signal.confidence:.0f}%\n\n"
                f"Target: +{self.config.target_pct:.0f}%\n"
                f"Stop: {self.config.stop_pct:.0f}%\n"
                f"Max Hold: {self.config.max_hold_minutes} min"
            )

            send_telegram_alert(alert_msg)

        except Exception as e:
            logger.warning(f"BERSERKER: Could not send Telegram alert - {e}")

    def _log_activation_check(self, activated: bool, reason: str, context: Dict) -> None:
        """Log activation check for debugging."""
        self._activation_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "activated": activated,
            "reason": reason,
            "context": context,
        })

        # Keep last 100 entries
        if len(self._activation_log) > 100:
            self._activation_log = self._activation_log[-100:]

    def _should_reset_daily(self) -> bool:
        """Check if daily stats should be reset."""
        if not self._last_trade_time:
            return False

        now = datetime.now(timezone.utc)
        last_trade_date = self._last_trade_time.date()
        return now.date() > last_trade_date

    def _reset_daily_stats(self) -> None:
        """Reset daily counters."""
        self.trades_today = 0
        logger.info("BERSERKER: Daily stats reset")

    def record_trade_result(self, win: bool) -> None:
        """Record trade result for stats."""
        if win:
            self._stats["wins"] += 1
        else:
            self._stats["losses"] += 1

    def get_status(self) -> Dict[str, Any]:
        """Get BERSERKER engine status."""
        status = {
            "running": self.running,
            "trades_today": self.trades_today,
            "max_daily_trades": self.config.max_daily_trades,
            "last_trade_time": self._last_trade_time.isoformat() if self._last_trade_time else None,
            "cooldown_until": (
                (self._last_trade_time + timedelta(minutes=self.config.cooldown_minutes)).isoformat()
                if self._last_trade_time else None
            ),
            "config": {
                "gex_threshold": self.config.gex_flip_proximity_threshold,
                "max_vix": self.config.max_vix,
                "position_multiplier": self.config.position_multiplier,
                "target_pct": self.config.target_pct,
                "stop_pct": self.config.stop_pct,
                "max_hold_minutes": self.config.max_hold_minutes,
            },
            "stats": self._stats,
            "recent_activation_checks": self._activation_log[-5:],
        }

        # Add GEX context from gex_calculator
        status["gex_context"] = self.get_gex_context()

        return status


# Singleton instance
_berserker: Optional[BerserkerEngine] = None


def get_berserker_engine() -> BerserkerEngine:
    """Get singleton BerserkerEngine instance."""
    global _berserker
    if _berserker is None:
        _berserker = BerserkerEngine()
    return _berserker
