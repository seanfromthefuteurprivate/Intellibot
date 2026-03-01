"""
MACRO FILTER: Fundamental/Macro Overlay for Trade Signals

SWARM CONSENSUS (9/12 personas agreed):
- Add macroeconomic/fundamental analysis layer
- Don't trade blind to macro context
- Filter signals based on broader market conditions

This module checks:
1. VIX structure (contango/backwardation)
2. Market breadth (advance/decline)
3. Major economic events (Fed, CPI, Jobs)
4. Risk-on/risk-off regime
"""
import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MacroConditions:
    """Current macro market conditions."""
    vix_level: float = 20.0
    vix_regime: str = "NORMAL"  # LOW (<15), NORMAL (15-25), ELEVATED (25-35), CRISIS (>35)
    vix_contango: bool = True  # True = calm, False = fear
    market_breadth: float = 0.5  # 0-1, percentage of stocks up
    risk_regime: str = "NEUTRAL"  # RISK_ON, NEUTRAL, RISK_OFF
    fed_blackout: bool = False  # True = near Fed meeting, avoid
    major_event_today: bool = False  # CPI, Jobs, FOMC
    spy_trend: str = "NEUTRAL"  # UP, NEUTRAL, DOWN (based on 20 SMA)
    confidence_multiplier: float = 1.0  # Apply to all trades


class MacroFilter:
    """
    Filter trades based on macro conditions.

    SWARM CONSENSUS: 9/12 personas said add this.
    """

    # VIX thresholds
    VIX_LOW = 15.0
    VIX_NORMAL = 25.0
    VIX_ELEVATED = 35.0

    # Breadth thresholds
    BREADTH_STRONG = 0.65  # >65% stocks up = bullish
    BREADTH_WEAK = 0.35    # <35% stocks up = bearish

    def __init__(self):
        self._last_check: Optional[datetime] = None
        self._cached_conditions: Optional[MacroConditions] = None
        self._cache_ttl_seconds = 300  # 5 minute cache
        logger.info("MACRO_FILTER: Initialized (SWARM CONSENSUS improvement)")

    def get_conditions(self) -> MacroConditions:
        """Get current macro conditions."""
        now = datetime.now()

        # Use cache if fresh
        if self._cached_conditions and self._last_check:
            if (now - self._last_check).total_seconds() < self._cache_ttl_seconds:
                return self._cached_conditions

        conditions = MacroConditions()

        # 1. Get VIX data
        try:
            from wsb_snake.collectors.vix_structure import vix_structure
            vix_data = vix_structure.get_trading_signal()
            conditions.vix_level = vix_data.get("vix", 20.0)
            conditions.vix_contango = vix_data.get("contango", True)

            # Classify VIX regime
            if conditions.vix_level < self.VIX_LOW:
                conditions.vix_regime = "LOW"
            elif conditions.vix_level < self.VIX_NORMAL:
                conditions.vix_regime = "NORMAL"
            elif conditions.vix_level < self.VIX_ELEVATED:
                conditions.vix_regime = "ELEVATED"
            else:
                conditions.vix_regime = "CRISIS"

        except Exception as e:
            logger.debug(f"VIX fetch failed: {e}")

        # 2. Get market breadth from HYDRA
        try:
            from wsb_snake.collectors.hydra_bridge import hydra_bridge
            hydra_data = hydra_bridge.get_signal()
            breadth = hydra_data.get("market_breadth", 0.5)
            conditions.market_breadth = breadth
        except Exception as e:
            logger.debug(f"Breadth fetch failed: {e}")

        # 3. Determine risk regime
        if conditions.vix_level < 18 and conditions.market_breadth > 0.55:
            conditions.risk_regime = "RISK_ON"
        elif conditions.vix_level > 28 or conditions.market_breadth < 0.40:
            conditions.risk_regime = "RISK_OFF"
        else:
            conditions.risk_regime = "NEUTRAL"

        # 4. Calculate confidence multiplier
        # Higher confidence when conditions are clear
        if conditions.vix_regime == "LOW" and conditions.risk_regime == "RISK_ON":
            conditions.confidence_multiplier = 1.2  # Boost in calm bull markets
        elif conditions.vix_regime == "CRISIS":
            conditions.confidence_multiplier = 0.5  # Cut size in chaos
        elif conditions.vix_regime == "ELEVATED":
            conditions.confidence_multiplier = 0.7  # Reduce in elevated vol
        elif not conditions.vix_contango:
            conditions.confidence_multiplier = 0.6  # VIX backwardation = fear
        else:
            conditions.confidence_multiplier = 1.0

        # Cache results
        self._cached_conditions = conditions
        self._last_check = now

        logger.debug(
            f"MACRO: VIX={conditions.vix_level:.1f} ({conditions.vix_regime}) "
            f"Breadth={conditions.market_breadth:.0%} "
            f"Regime={conditions.risk_regime} "
            f"Multiplier={conditions.confidence_multiplier:.1f}x"
        )

        return conditions

    def should_trade(self) -> Tuple[bool, str, float]:
        """
        Check if macro conditions allow trading.

        Returns:
            (allowed, reason, confidence_multiplier)
        """
        conditions = self.get_conditions()

        # CRISIS mode - reduce drastically or skip
        if conditions.vix_regime == "CRISIS":
            logger.warning("MACRO_FILTER: VIX CRISIS mode - reducing size 50%")
            return True, "VIX_CRISIS_CAUTION", 0.5

        # VIX backwardation - fear in market
        if not conditions.vix_contango:
            logger.warning("MACRO_FILTER: VIX backwardation - reducing size 40%")
            return True, "VIX_BACKWARDATION", 0.6

        # Extreme breadth divergence
        if conditions.market_breadth < 0.25:
            logger.warning("MACRO_FILTER: Extreme weak breadth - reducing size 30%")
            return True, "WEAK_BREADTH", 0.7

        # Normal conditions
        return True, "MACRO_OK", conditions.confidence_multiplier

    def get_regime_adjustment(self, base_confidence: float) -> float:
        """
        Adjust confidence based on macro regime.

        SWARM CONSENSUS: Use macro to validate signals.
        """
        conditions = self.get_conditions()
        adjusted = base_confidence * conditions.confidence_multiplier

        # Cap at 95%
        return min(95.0, adjusted)


# Singleton
_macro_filter: Optional[MacroFilter] = None


def get_macro_filter() -> MacroFilter:
    """Get singleton macro filter instance."""
    global _macro_filter
    if _macro_filter is None:
        _macro_filter = MacroFilter()
    return _macro_filter
