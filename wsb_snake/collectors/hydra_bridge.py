"""
HYDRA Bridge - Connects WSB Snake to HYDRA Intelligence Engine

Polls HYDRA every 60 seconds for:
- blowup_probability: 0-100 score indicating likelihood of major market move
- direction: BULLISH, BEARISH, or NEUTRAL
- regime: RISK_ON, RISK_OFF, TRENDING_UP, TRENDING_DOWN, CHOPPY, etc.
- events_next_30min: Upcoming economic events
- recommendation: SCALP_ONLY, BLOWUP_READY, HOLD_OFF

Integration Points:
1. APEX Conviction Engine - regime adjusts conviction by +/-20%
2. Dual-Mode Engine - blowup_probability > 60 triggers BLOWUP mode
3. Session Windows - events block new entries
4. Trade Feedback - results sent back to HYDRA for learning

Log format: "HYDRA_BRIDGE: blowup=XX dir=XX regime=XX triggers=[...]"
"""

import os
import time
import threading
import requests
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)

# HYDRA API Configuration
HYDRA_BASE_URL = os.environ.get('HYDRA_URL', 'http://54.172.22.157:8000')
HYDRA_INTELLIGENCE_ENDPOINT = f"{HYDRA_BASE_URL}/api/predator"  # Full predator intel
HYDRA_INTELLIGENCE_FALLBACK = f"{HYDRA_BASE_URL}/api/intelligence"  # Fallback
HYDRA_TRADE_RESULT_ENDPOINT = f"{HYDRA_BASE_URL}/api/trade-result"

# Polling configuration
POLL_INTERVAL_SECONDS = 60
STALE_DATA_THRESHOLD_SECONDS = 180  # 3 minutes


@dataclass
class HydraIntelligence:
    """Intelligence data from HYDRA."""
    # Core blowup detection
    blowup_probability: int = 0
    direction: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    regime: str = "UNKNOWN"  # RISK_ON, RISK_OFF, TRENDING_UP, TRENDING_DOWN, CHOPPY, RECOVERY, CRASH
    confidence: float = 0.0
    triggers: List[str] = field(default_factory=list)
    recommendation: str = "SCALP_ONLY"  # SCALP_ONLY, BLOWUP_READY, HOLD_OFF, AGGRESSIVE
    events_next_30min: List[Dict] = field(default_factory=list)
    vix_level: float = 0.0
    vix_trend: str = "STABLE"
    connected: bool = False
    last_update: float = 0
    raw_data: Dict = field(default_factory=dict)

    # NEW: GEX Intelligence (from HYDRA Layer 8)
    gex_regime: str = "UNKNOWN"  # POSITIVE (mean-reverting), NEGATIVE (trending)
    gex_total: float = 0.0  # Net dealer gamma exposure ($)
    gex_flip_point: float = 0.0  # Price where dealer behavior flips
    gex_flip_distance_pct: float = 999.0  # % distance to flip point
    gex_key_support: List[float] = field(default_factory=list)
    gex_key_resistance: List[float] = field(default_factory=list)
    charm_flow_per_hour: float = 0.0  # Delta decay (critical for 0DTE)

    # NEW: Flow Intelligence (from HYDRA Layer 9)
    flow_bias: str = "NEUTRAL"  # AGGRESSIVELY_BULLISH, BULLISH, NEUTRAL, BEARISH, AGGRESSIVELY_BEARISH
    flow_net_premium_calls: float = 0.0
    flow_net_premium_puts: float = 0.0
    flow_sweep_direction: str = "NEUTRAL"  # CALL_HEAVY, PUT_HEAVY, BALANCED
    flow_confidence: float = 0.0

    # NEW: Dark Pool Levels (from HYDRA Layer 10)
    dp_nearest_support: float = 0.0
    dp_nearest_resistance: float = 0.0
    dp_support_strength: str = "UNKNOWN"  # LOW, MEDIUM, HIGH, VERY_HIGH
    dp_resistance_strength: str = "UNKNOWN"

    # NEW: Sequence Match (from HYDRA Layer 11)
    seq_similar_patterns: int = 0
    seq_historical_win_rate: float = 0.0
    seq_historical_avg_return: float = 0.0
    seq_nova_analysis: str = ""  # Nova Pro reasoning summary

    def is_stale(self) -> bool:
        """Check if data is stale (>3 minutes old)."""
        return time.time() - self.last_update > STALE_DATA_THRESHOLD_SECONDS

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/serialization."""
        return {
            'blowup_probability': self.blowup_probability,
            'direction': self.direction,
            'regime': self.regime,
            'confidence': self.confidence,
            'triggers': self.triggers,
            'recommendation': self.recommendation,
            'events_next_30min': self.events_next_30min,
            'connected': self.connected,
            'last_update': self.last_update,
            # GEX
            'gex_regime': self.gex_regime,
            'gex_total': self.gex_total,
            'gex_flip_point': self.gex_flip_point,
            'gex_flip_distance_pct': self.gex_flip_distance_pct,
            'gex_key_support': self.gex_key_support,
            'gex_key_resistance': self.gex_key_resistance,
            # Flow
            'flow_bias': self.flow_bias,
            'flow_net_premium_calls': self.flow_net_premium_calls,
            'flow_net_premium_puts': self.flow_net_premium_puts,
            'flow_sweep_direction': self.flow_sweep_direction,
            # Dark Pool
            'dp_nearest_support': self.dp_nearest_support,
            'dp_nearest_resistance': self.dp_nearest_resistance,
            'dp_support_strength': self.dp_support_strength,
            'dp_resistance_strength': self.dp_resistance_strength,
            # Sequence
            'seq_similar_patterns': self.seq_similar_patterns,
            'seq_historical_win_rate': self.seq_historical_win_rate,
        }


class HydraBridge:
    """
    Bridge to HYDRA Intelligence Engine.

    Runs as a background thread, polling HYDRA every 60 seconds.
    Provides methods for WSB Snake to query intelligence and make decisions.
    """

    # Blowup mode thresholds
    BLOWUP_THRESHOLD = 60  # Enter blowup mode if probability > 60
    BLOWUP_EXIT_THRESHOLD = 40  # Exit blowup mode if probability < 40

    # Regime conviction adjustments
    REGIME_ADJUSTMENTS = {
        "RISK_ON": +0.10,
        "TRENDING_UP": +0.08,
        "RECOVERY": +0.05,
        "NEUTRAL": 0.0,
        "UNKNOWN": 0.0,
        "CHOPPY": -0.05,
        "TRENDING_DOWN": -0.08,
        "RISK_OFF": -0.12,
        "CRASH": -0.20,
    }

    def __init__(self, auto_start: bool = False):
        """
        Initialize HYDRA Bridge.

        Args:
            auto_start: If True, start polling immediately
        """
        self.intel = HydraIntelligence()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._poll_count = 0
        self._error_count = 0

        if auto_start:
            self.start()

    def start(self):
        """Start background polling thread."""
        if self._running:
            logger.warning("HYDRA_BRIDGE: Already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(f"HYDRA_BRIDGE: Started — polling {HYDRA_INTELLIGENCE_ENDPOINT} every {POLL_INTERVAL_SECONDS}s")

    def stop(self):
        """Stop background polling thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("HYDRA_BRIDGE: Stopped")

    def _poll_loop(self):
        """Background polling loop."""
        # Initial poll immediately
        self._fetch_intelligence()

        while self._running:
            time.sleep(POLL_INTERVAL_SECONDS)
            if self._running:
                self._fetch_intelligence()

    def _fetch_intelligence(self):
        """Fetch latest intelligence from HYDRA."""
        self._poll_count += 1

        try:
            response = requests.get(
                HYDRA_INTELLIGENCE_ENDPOINT,
                timeout=10,
                headers={'User-Agent': 'WSBSnake/1.0'}
            )

            if response.status_code == 200:
                data = response.json()
                self._parse_intelligence(data)
                self._error_count = 0

                # Log the update
                logger.info(
                    f"HYDRA_BRIDGE: blowup={self.intel.blowup_probability} "
                    f"dir={self.intel.direction} regime={self.intel.regime} "
                    f"triggers={self.intel.triggers} rec={self.intel.recommendation}"
                )

            elif response.status_code == 404:
                # Endpoint not found - HYDRA may not have intelligence API yet
                logger.debug("HYDRA_BRIDGE: Intelligence endpoint not found (404)")
                self.intel.connected = False

            else:
                logger.warning(f"HYDRA_BRIDGE: HTTP {response.status_code}")
                self.intel.connected = False
                self._error_count += 1

        except requests.exceptions.Timeout:
            logger.warning("HYDRA_BRIDGE: Request timeout")
            self.intel.connected = False
            self._error_count += 1

        except requests.exceptions.ConnectionError:
            logger.debug("HYDRA_BRIDGE: Connection failed (HYDRA may be offline)")
            self.intel.connected = False
            self._error_count += 1

        except Exception as e:
            logger.warning(f"HYDRA_BRIDGE: Error — {e}")
            self.intel.connected = False
            self._error_count += 1

    def _parse_intelligence(self, data: Dict):
        """Parse API response into HydraIntelligence.

        Handles both:
        - /api/predator (flat fields: gex_regime, flow_institutional_bias, etc.)
        - /api/intelligence (nested: gex.regime, flow.institutional_bias, etc.)
        """
        # Check if using flat structure (predator endpoint) or nested (intelligence)
        is_flat = 'gex_regime' in data or 'flow_institutional_bias' in data

        if is_flat:
            # Flat structure from /api/predator
            self.intel = HydraIntelligence(
                # Core blowup detection
                blowup_probability=int(data.get('blowup_probability', 0)),
                direction=str(data.get('blowup_direction', data.get('direction', 'NEUTRAL'))).upper(),
                regime=str(data.get('blowup_regime', data.get('regime', 'UNKNOWN'))).upper(),
                confidence=float(data.get('confidence', 50)),
                triggers=data.get('blowup_triggers', data.get('triggers', [])) or [],
                recommendation=str(data.get('blowup_recommendation', data.get('recommendation', 'SCALP_ONLY'))).upper(),
                events_next_30min=data.get('events_next_30min', []) or [],
                vix_level=float(data.get('vix_level', 0)),
                vix_trend=str(data.get('vix_trend', 'STABLE')).upper(),
                connected=True,
                last_update=time.time(),
                raw_data=data,

                # GEX Intelligence (Layer 8) - flat fields
                gex_regime=str(data.get('gex_regime', 'UNKNOWN')).upper(),
                gex_total=float(data.get('gex_total', 0)),
                gex_flip_point=float(data.get('gex_flip_point') or 0),
                gex_flip_distance_pct=float(data.get('gex_flip_distance_pct', 999)),
                gex_key_support=data.get('gex_key_support', []) or [],
                gex_key_resistance=data.get('gex_key_resistance', []) or [],
                charm_flow_per_hour=float(data.get('gex_charm_per_hour', 0)),

                # Flow Intelligence (Layer 9) - flat fields
                flow_bias=str(data.get('flow_institutional_bias', 'NEUTRAL')).upper(),
                flow_net_premium_calls=float(data.get('flow_premium_calls', 0)),
                flow_net_premium_puts=float(data.get('flow_premium_puts', 0)),
                flow_sweep_direction=str(data.get('flow_sweep_direction', 'NEUTRAL')).upper(),
                flow_confidence=float(data.get('flow_confidence', 0)),

                # Dark Pool Levels (Layer 10) - flat fields
                dp_nearest_support=float(data.get('dp_nearest_support') or 0),
                dp_nearest_resistance=float(data.get('dp_nearest_resistance') or 0),
                dp_support_strength=str(data.get('dp_support_strength', 'UNKNOWN')).upper(),
                dp_resistance_strength=str(data.get('dp_resistance_strength', 'UNKNOWN')).upper(),

                # Sequence Match (Layer 11) - flat fields
                seq_similar_patterns=int(data.get('sequence_similar_count', 0)),
                seq_historical_win_rate=float(data.get('sequence_historical_win_rate', 0)),
                seq_historical_avg_return=float(data.get('sequence_avg_outcome', 0)),
                seq_nova_analysis=str(data.get('sequence_predicted_direction', '')),
            )
        else:
            # Nested structure from /api/intelligence (legacy)
            gex_data = data.get('gex', {}) or {}
            flow_data = data.get('flow', {}) or {}
            dp_data = data.get('dark_pool', {}) or {}
            seq_data = data.get('sequence_match', {}) or {}

            self.intel = HydraIntelligence(
                blowup_probability=int(data.get('blowup_probability', 0)),
                direction=str(data.get('direction', 'NEUTRAL')).upper(),
                regime=str(data.get('regime', 'UNKNOWN')).upper(),
                confidence=float(data.get('confidence', 0)),
                triggers=data.get('triggers', []) or [],
                recommendation=str(data.get('recommendation', 'SCALP_ONLY')).upper(),
                events_next_30min=data.get('events_next_30min', []) or [],
                vix_level=float(data.get('vix_level', 0)),
                vix_trend=str(data.get('vix_trend', 'STABLE')).upper(),
                connected=True,
                last_update=time.time(),
                raw_data=data,
                gex_regime=str(gex_data.get('regime', 'UNKNOWN')).upper(),
                gex_total=float(gex_data.get('total_gex', 0)),
                gex_flip_point=float(gex_data.get('flip_point') or 0),
                gex_flip_distance_pct=float(gex_data.get('flip_distance_pct', 999)),
                gex_key_support=gex_data.get('key_support', []) or [],
                gex_key_resistance=gex_data.get('key_resistance', []) or [],
                charm_flow_per_hour=float(gex_data.get('charm_flow_per_hour', 0)),
                flow_bias=str(flow_data.get('institutional_bias', 'NEUTRAL')).upper(),
                flow_net_premium_calls=float(flow_data.get('net_premium_calls', 0)),
                flow_net_premium_puts=float(flow_data.get('net_premium_puts', 0)),
                flow_sweep_direction=str(flow_data.get('sweep_direction', 'NEUTRAL')).upper(),
                flow_confidence=float(flow_data.get('confidence', 0)),
                dp_nearest_support=float(dp_data.get('nearest_support') or 0),
                dp_nearest_resistance=float(dp_data.get('nearest_resistance') or 0),
                dp_support_strength=str(dp_data.get('support_strength', 'UNKNOWN')).upper(),
                dp_resistance_strength=str(dp_data.get('resistance_strength', 'UNKNOWN')).upper(),
                seq_similar_patterns=int(seq_data.get('similar_patterns', 0)),
                seq_historical_win_rate=float(seq_data.get('win_rate', 0)),
                seq_historical_avg_return=float(seq_data.get('historical_outcome', 0) if not isinstance(seq_data.get('historical_outcome'), str) else 0),
                seq_nova_analysis=str(seq_data.get('nova_analysis', '')),
            )

    def get_intel(self) -> HydraIntelligence:
        """
        Get latest intelligence. Safe to call anytime.

        Returns:
            HydraIntelligence object (may be stale if HYDRA is unreachable)
        """
        # Mark as disconnected if data is stale
        if self.intel.is_stale():
            self.intel.connected = False

        return self.intel

    def is_connected(self) -> bool:
        """Check if HYDRA bridge is connected and data is fresh."""
        return self.intel.connected and not self.intel.is_stale()

    def should_enter_blowup_mode(self) -> bool:
        """
        Check if blowup mode should be activated.

        Returns:
            True if connected AND blowup_probability > 60
        """
        should_enter = (
            self.intel.connected and
            not self.intel.is_stale() and
            self.intel.blowup_probability > self.BLOWUP_THRESHOLD
        )

        # VENOM: If entering blowup mode, trigger emergency position review
        if should_enter:
            logger.warning(
                f"🚨 BLOWUP MODE ACTIVATED: probability={self.intel.blowup_probability}% "
                f"direction={self.intel.direction} regime={self.intel.regime}"
            )
            self._trigger_blowup_position_review()

        return should_enter

    def _trigger_blowup_position_review(self) -> None:
        """
        VENOM: When blowup mode triggers, review and potentially close positions.

        If HYDRA says CRASH with >70% probability, CLOSE ALL.
        If direction is opposite to our positions, CLOSE THOSE.
        """
        try:
            from wsb_snake.trading.alpaca_executor import alpaca_executor
            from wsb_snake.notifications.telegram_bot import send_alert as send_telegram_alert

            positions = alpaca_executor.get_options_positions()
            if not positions:
                logger.info("BLOWUP_MODE: No positions to review")
                return

            # CRASH MODE: Close everything if probability > 70%
            if self.intel.regime == "CRASH" and self.intel.blowup_probability > 70:
                logger.warning(f"🚨 CRASH DETECTED ({self.intel.blowup_probability}%) - CLOSING ALL POSITIONS")
                send_telegram_alert(
                    f"🚨 **CRASH MODE ACTIVATED**\n\n"
                    f"HYDRA: {self.intel.blowup_probability}% blowup probability\n"
                    f"Regime: {self.intel.regime}\n\n"
                    f"CLOSING ALL {len(positions)} POSITIONS"
                )
                closed = alpaca_executor.close_all_0dte_positions()
                logger.warning(f"BLOWUP_MODE: Emergency closed {closed} positions")
                return

            # DIRECTION CONFLICT: Close positions opposite to HYDRA direction
            for pos in positions:
                symbol = pos.get('symbol', '')
                side = pos.get('side', 'long')

                # Determine if position conflicts with HYDRA direction
                is_call = 'C' in symbol[len(symbol.split('2')[0]):][:7]  # Rough check
                is_bullish_position = (side == 'long' and is_call) or (side == 'short' and not is_call)

                should_close = False
                reason = ""

                if self.intel.direction == "BEARISH" and is_bullish_position:
                    should_close = True
                    reason = f"HYDRA says BEARISH but position is bullish"
                elif self.intel.direction == "BULLISH" and not is_bullish_position:
                    should_close = True
                    reason = f"HYDRA says BULLISH but position is bearish"

                if should_close:
                    logger.warning(f"🚨 BLOWUP CLOSE: {symbol} - {reason}")
                    try:
                        alpaca_executor.close_position(symbol)
                        send_telegram_alert(
                            f"🚨 **BLOWUP MODE CLOSE**\n\n"
                            f"Position: {symbol}\n"
                            f"Reason: {reason}\n"
                            f"HYDRA: {self.intel.direction} ({self.intel.blowup_probability}%)"
                        )
                    except Exception as e:
                        logger.error(f"Failed to close {symbol} in blowup mode: {e}")

        except Exception as e:
            logger.error(f"BLOWUP_MODE position review failed: {e}")

    def should_exit_blowup_mode(self) -> bool:
        """
        Check if blowup mode should be deactivated.

        Returns:
            True if blowup_probability < 40 OR disconnected
        """
        if not self.intel.connected or self.intel.is_stale():
            return True
        return self.intel.blowup_probability < self.BLOWUP_EXIT_THRESHOLD

    def should_block_for_event(self) -> bool:
        """
        Check if new entries should be blocked due to upcoming event.

        Returns:
            True if there are events in the next 30 minutes
        """
        return bool(self.intel.events_next_30min)

    def get_event_block_reason(self) -> Optional[str]:
        """Get reason for event block, or None if no block needed."""
        if not self.intel.events_next_30min:
            return None

        events = self.intel.events_next_30min
        event_names = [e.get('name', 'Unknown') for e in events[:3]]
        return f"Events in 30min: {', '.join(event_names)}"

    def get_conviction_adjustment(self) -> float:
        """
        Get conviction adjustment based on HYDRA regime.

        Returns:
            Adjustment factor (-0.20 to +0.10) to apply to conviction
        """
        if not self.intel.connected or self.intel.is_stale():
            return 0.0

        return self.REGIME_ADJUSTMENTS.get(self.intel.regime, 0.0)

    def get_direction_for_blowup(self) -> str:
        """
        Get direction for blowup mode trading.

        Returns:
            "BULLISH", "BEARISH", or "NEUTRAL" (for straddle)
        """
        if not self.intel.connected:
            return "NEUTRAL"
        return self.intel.direction

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status for monitoring."""
        return {
            'connected': self.is_connected(),
            'poll_count': self._poll_count,
            'error_count': self._error_count,
            'last_update': datetime.fromtimestamp(
                self.intel.last_update, tz=timezone.utc
            ).isoformat() if self.intel.last_update > 0 else None,
            'data_age_seconds': int(time.time() - self.intel.last_update) if self.intel.last_update > 0 else None,
            'intelligence': self.intel.to_dict(),
        }

    def send_trade_result(
        self,
        ticker: str,
        direction: str,
        pnl: float,
        pnl_pct: float,
        conviction: float,
        regime: str,
        mode: str,
        exit_reason: str = None,
        hold_seconds: int = None
    ) -> bool:
        """
        Send trade result back to HYDRA for learning.

        Args:
            ticker: Symbol traded (e.g., "SPY")
            direction: "LONG" or "SHORT"
            pnl: Dollar P&L
            pnl_pct: Percentage P&L
            conviction: Conviction score at entry
            regime: HYDRA regime at entry
            mode: "SCALP" or "BLOWUP"
            exit_reason: Why the trade was closed
            hold_seconds: How long the trade was held

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            payload = {
                'ticker': ticker,
                'direction': direction,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'conviction': conviction,
                'regime': regime,
                'mode': mode,
                'exit_reason': exit_reason,
                'hold_seconds': hold_seconds,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'wsb_snake'
            }

            response = requests.post(
                HYDRA_TRADE_RESULT_ENDPOINT,
                json=payload,
                timeout=3,
                headers={'User-Agent': 'WSBSnake/1.0'}
            )

            if response.status_code in (200, 201):
                logger.debug(f"HYDRA_FEEDBACK: Sent trade result for {ticker}")
                return True
            else:
                logger.debug(f"HYDRA_FEEDBACK: HTTP {response.status_code}")
                return False

        except Exception as e:
            # Non-critical, don't block trading
            logger.debug(f"HYDRA_FEEDBACK: Failed to send — {e}")
            return False


# Singleton instance
_hydra_bridge: Optional[HydraBridge] = None


def get_hydra_bridge() -> HydraBridge:
    """Get the singleton HydraBridge instance."""
    global _hydra_bridge
    if _hydra_bridge is None:
        _hydra_bridge = HydraBridge()
    return _hydra_bridge


def start_hydra_bridge():
    """Start the HYDRA bridge (call from main.py startup)."""
    bridge = get_hydra_bridge()
    bridge.start()
    return bridge


def get_hydra_intel() -> HydraIntelligence:
    """
    Convenience function to get HYDRA intelligence.

    Usage:
        from wsb_snake.collectors.hydra_bridge import get_hydra_intel
        intel = get_hydra_intel()
        if intel.blowup_probability > 60:
            enter_blowup_mode()
    """
    bridge = get_hydra_bridge()
    return bridge.get_intel()


def get_regime_adjustment() -> float:
    """
    Get conviction adjustment from HYDRA regime.

    Usage:
        from wsb_snake.collectors.hydra_bridge import get_regime_adjustment
        adjustment = get_regime_adjustment()
        adjusted_conviction = base_conviction * (1 + adjustment)
    """
    bridge = get_hydra_bridge()
    return bridge.get_conviction_adjustment()


def get_hydra_predator_raw() -> Optional[Dict]:
    """
    Get raw HYDRA /api/predator response for advanced analysis.

    Used by BERSERKER engine for GEX calculator integration.

    Usage:
        from wsb_snake.collectors.hydra_bridge import get_hydra_predator_raw
        raw = get_hydra_predator_raw()
        if raw:
            gex_data = raw.get('gex', {})
    """
    bridge = get_hydra_bridge()
    intel = bridge.get_intel()
    if intel.connected and intel.raw_data:
        return intel.raw_data
    return None


def send_trade_feedback(
    ticker: str,
    direction: str,
    pnl: float,
    conviction: float,
    mode: str = "SCALP"
) -> bool:
    """
    Convenience function to send trade result to HYDRA.

    Usage:
        from wsb_snake.collectors.hydra_bridge import send_trade_feedback
        send_trade_feedback("SPY", "LONG", 85.50, 78.5, "SCALP")
    """
    bridge = get_hydra_bridge()
    intel = bridge.get_intel()
    return bridge.send_trade_result(
        ticker=ticker,
        direction=direction,
        pnl=pnl,
        pnl_pct=0,  # Calculate if needed
        conviction=conviction,
        regime=intel.regime,
        mode=mode
    )
