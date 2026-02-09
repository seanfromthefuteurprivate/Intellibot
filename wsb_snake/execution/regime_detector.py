"""
HYDRA-Inspired Regime Detection Engine

A 6-state market regime classifier that adapts signal weights based on
current market conditions. Inspired by institutional adaptive systems.

Regimes:
- TRENDING_UP: Strong directional bullish momentum
- TRENDING_DOWN: Strong directional bearish momentum
- MEAN_REVERTING: Range-bound, fade extremes
- HIGH_VOL: Elevated volatility (VIX > 25)
- CRASH: Crisis mode (VIX > 35 + backwardation + SPY down)
- RECOVERY: Post-crash bounce phase
- UNKNOWN: Insufficient data for classification
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import logger
try:
    from wsb_snake.utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Try to import data collectors
try:
    from wsb_snake.collectors.polygon_enhanced import get_stock_price
except ImportError:
    get_stock_price = None
    logger.warning("polygon_enhanced not available for regime detection")

try:
    from wsb_snake.collectors.vix_structure import get_vix_data, get_vix_term_structure
except ImportError:
    get_vix_data = None
    get_vix_term_structure = None
    logger.warning("vix_structure not available for regime detection")


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"       # Strong directional bullish
    TRENDING_DOWN = "trending_down"   # Strong directional bearish
    MEAN_REVERTING = "mean_reverting" # Range-bound, fade extremes
    HIGH_VOL = "high_vol"             # VIX > 25
    CRASH = "crash"                   # VIX > 35 + backwardation + SPY down
    RECOVERY = "recovery"             # Post-crash bounce
    UNKNOWN = "unknown"               # Insufficient data


@dataclass
class RegimeState:
    """
    Current market regime state with supporting metrics.

    Attributes:
        regime: The classified market regime
        confidence: Confidence in classification (0-1.0)
        vix_level: Current VIX level
        vix_structure: Term structure ("contango" or "backwardation")
        trend_strength: Directional momentum (-1 to +1)
        mean_reversion_score: Likelihood of mean reversion (0-1)
        detected_at: Timestamp of detection
    """
    regime: MarketRegime
    confidence: float
    vix_level: float
    vix_structure: str
    trend_strength: float
    mean_reversion_score: float
    detected_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate ranges after initialization."""
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.trend_strength = max(-1.0, min(1.0, self.trend_strength))
        self.mean_reversion_score = max(0.0, min(1.0, self.mean_reversion_score))


class RegimeDetector:
    """
    HYDRA-inspired adaptive market regime detector.

    Uses multi-factor analysis to classify current market conditions
    and provide signal weight adjustments for different regimes.

    Example:
        detector = RegimeDetector()
        detector.update(spy_price=450.0, vix=18.5, vix_3m=20.0, tlt_price=95.0)
        state = detector.detect_regime()
        multipliers = detector.get_regime_multipliers()
    """

    # Configuration constants
    MIN_HISTORY_SIZE = 20  # Minimum data points for classification
    MAX_HISTORY_SIZE = 100  # Rolling window size

    # VIX thresholds
    VIX_ELEVATED = 25.0
    VIX_CRISIS = 35.0
    VIX_LOW = 15.0

    # Trend thresholds
    TREND_STRONG = 0.6
    TREND_WEAK = 0.2

    # Mean reversion threshold
    MR_STRONG = 0.7

    # Regime multipliers for signal weight adjustment
    REGIME_MULTIPLIERS = {
        MarketRegime.TRENDING_UP: {
            "order_flow": 1.0,
            "technical": 1.15,
            "probability": 1.0,
            "pattern_memory": 0.9,
            "sentiment": 1.1,
        },
        MarketRegime.TRENDING_DOWN: {
            "order_flow": 1.1,
            "technical": 1.15,
            "probability": 0.95,
            "pattern_memory": 0.9,
            "sentiment": 1.2,
        },
        MarketRegime.MEAN_REVERTING: {
            "order_flow": 0.9,
            "technical": 1.0,
            "probability": 0.85,
            "pattern_memory": 1.25,
            "sentiment": 0.8,
        },
        MarketRegime.HIGH_VOL: {
            "order_flow": 1.25,
            "technical": 0.75,
            "probability": 0.7,
            "pattern_memory": 1.0,
            "sentiment": 1.3,
        },
        MarketRegime.CRASH: {
            "order_flow": 1.5,
            "technical": 0.5,
            "probability": 0.5,
            "pattern_memory": 0.7,
            "sentiment": 1.5,
        },
        MarketRegime.RECOVERY: {
            "order_flow": 1.2,
            "technical": 1.1,
            "probability": 0.8,
            "pattern_memory": 1.15,
            "sentiment": 1.0,
        },
        MarketRegime.UNKNOWN: {
            "order_flow": 1.0,
            "technical": 1.0,
            "probability": 1.0,
            "pattern_memory": 1.0,
            "sentiment": 1.0,
        },
    }

    def __init__(self, max_history: int = MAX_HISTORY_SIZE):
        """
        Initialize the regime detector with price history deques.

        Args:
            max_history: Maximum number of data points to retain
        """
        self.max_history = max_history

        # Price history deques
        self.spy_prices: deque = deque(maxlen=max_history)
        self.vix_history: deque = deque(maxlen=max_history)
        self.vix_3m_history: deque = deque(maxlen=max_history)
        self.tlt_prices: deque = deque(maxlen=max_history)
        self.timestamps: deque = deque(maxlen=max_history)

        # Current state
        self._current_state: Optional[RegimeState] = None
        self._last_regime: MarketRegime = MarketRegime.UNKNOWN
        self._regime_duration: int = 0  # How long in current regime

        # Crash detection state
        self._in_crash: bool = False
        self._crash_low: Optional[float] = None
        self._pre_crash_high: Optional[float] = None

        logger.info("RegimeDetector initialized with max_history=%d", max_history)

    def update(
        self,
        spy_price: float,
        vix: float,
        vix_3m: Optional[float] = None,
        tlt_price: Optional[float] = None
    ) -> None:
        """
        Ingest new market data point.

        Args:
            spy_price: Current SPY price
            vix: Current VIX level
            vix_3m: 3-month VIX futures (for term structure)
            tlt_price: TLT price (for flight-to-safety detection)
        """
        now = datetime.now()

        self.spy_prices.append(spy_price)
        self.vix_history.append(vix)
        self.timestamps.append(now)

        if vix_3m is not None:
            self.vix_3m_history.append(vix_3m)

        if tlt_price is not None:
            self.tlt_prices.append(tlt_price)

        logger.debug(
            "Updated regime data: SPY=%.2f, VIX=%.2f, VIX3M=%s, TLT=%s",
            spy_price, vix,
            f"{vix_3m:.2f}" if vix_3m else "N/A",
            f"{tlt_price:.2f}" if tlt_price else "N/A"
        )

    def detect_regime(self) -> RegimeState:
        """
        Classify the current market regime.

        Returns:
            RegimeState with current classification and metrics
        """
        # Check for sufficient data
        if len(self.spy_prices) < self.MIN_HISTORY_SIZE:
            logger.debug("Insufficient data for regime detection: %d/%d",
                        len(self.spy_prices), self.MIN_HISTORY_SIZE)
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                vix_level=self.vix_history[-1] if self.vix_history else 0.0,
                vix_structure="unknown",
                trend_strength=0.0,
                mean_reversion_score=0.0,
            )

        # Calculate metrics
        prices = list(self.spy_prices)
        vix = self.vix_history[-1]

        # VIX term structure
        vix_structure = self._calc_vix_structure()

        # Trend strength (-1 to +1)
        trend_strength = self._calc_trend_strength(prices)

        # Mean reversion score (0 to 1)
        mr_score = self._calc_mean_reversion(prices)

        # Classify regime
        regime, confidence = self._classify(vix, vix_structure, trend_strength, mr_score)

        # Track regime duration
        if regime == self._last_regime:
            self._regime_duration += 1
        else:
            self._regime_duration = 1
            self._last_regime = regime

        # Build state
        self._current_state = RegimeState(
            regime=regime,
            confidence=confidence,
            vix_level=vix,
            vix_structure=vix_structure,
            trend_strength=trend_strength,
            mean_reversion_score=mr_score,
        )

        logger.info(
            "Regime detected: %s (confidence=%.2f, VIX=%.1f, trend=%.2f, MR=%.2f)",
            regime.value, confidence, vix, trend_strength, mr_score
        )

        return self._current_state

    def get_regime_multipliers(self) -> Dict[str, float]:
        """
        Get signal weight adjustments for current regime.

        Returns:
            Dictionary of signal type to weight multiplier
        """
        if self._current_state is None:
            self.detect_regime()

        regime = self._current_state.regime if self._current_state else MarketRegime.UNKNOWN
        return self.REGIME_MULTIPLIERS.get(regime, self.REGIME_MULTIPLIERS[MarketRegime.UNKNOWN])

    def get_current_state(self) -> Optional[RegimeState]:
        """
        Get the most recently detected regime state.

        Returns:
            Current RegimeState or None if never detected
        """
        return self._current_state

    def _calc_vix_structure(self) -> str:
        """
        Calculate VIX term structure (contango/backwardation).

        Contango: VIX < VIX3M (normal, futures higher than spot)
        Backwardation: VIX > VIX3M (fear, spot higher than futures)

        Returns:
            "contango", "backwardation", or "unknown"
        """
        if not self.vix_history or not self.vix_3m_history:
            return "unknown"

        vix_spot = self.vix_history[-1]
        vix_3m = self.vix_3m_history[-1]

        if vix_spot < vix_3m * 0.98:  # 2% buffer
            return "contango"
        elif vix_spot > vix_3m * 1.02:
            return "backwardation"
        else:
            return "contango"  # Default to contango if flat

    def _calc_trend_strength(self, prices: List[float]) -> float:
        """
        Calculate multi-timeframe trend strength.

        Uses combination of:
        - Short-term momentum (5-period)
        - Medium-term momentum (10-period)
        - Long-term momentum (20-period)
        - Price position relative to moving averages

        Args:
            prices: List of prices

        Returns:
            Trend strength from -1 (strong down) to +1 (strong up)
        """
        if len(prices) < 20:
            return 0.0

        prices_arr = np.array(prices)

        # Calculate returns
        returns = np.diff(prices_arr) / prices_arr[:-1]

        # Multi-timeframe momentum
        mom_5 = np.mean(returns[-5:]) if len(returns) >= 5 else 0
        mom_10 = np.mean(returns[-10:]) if len(returns) >= 10 else 0
        mom_20 = np.mean(returns[-20:]) if len(returns) >= 20 else 0

        # Weighted momentum score
        weighted_mom = (mom_5 * 0.5 + mom_10 * 0.3 + mom_20 * 0.2)

        # Price position relative to SMAs
        current_price = prices_arr[-1]
        sma_10 = np.mean(prices_arr[-10:])
        sma_20 = np.mean(prices_arr[-20:])

        above_sma10 = 1.0 if current_price > sma_10 else -1.0
        above_sma20 = 1.0 if current_price > sma_20 else -1.0
        sma_position = (above_sma10 + above_sma20) / 2

        # SMA slope
        sma_slope = (sma_10 - sma_20) / sma_20 if sma_20 != 0 else 0

        # Combine factors
        # Normalize momentum to roughly -1 to 1 range (typical daily returns are <1%)
        normalized_mom = np.clip(weighted_mom * 50, -1, 1)

        # Final trend strength
        trend = (normalized_mom * 0.5 + sma_position * 0.3 + np.clip(sma_slope * 10, -1, 1) * 0.2)

        return float(np.clip(trend, -1, 1))

    def _calc_mean_reversion(self, prices: List[float]) -> float:
        """
        Calculate mean reversion likelihood using return autocorrelation.

        Negative autocorrelation suggests mean-reverting behavior.

        Args:
            prices: List of prices

        Returns:
            Mean reversion score from 0 (trending) to 1 (strongly mean-reverting)
        """
        if len(prices) < 20:
            return 0.5  # Neutral

        prices_arr = np.array(prices)
        returns = np.diff(prices_arr) / prices_arr[:-1]

        if len(returns) < 10:
            return 0.5

        # Calculate lag-1 autocorrelation
        mean_ret = np.mean(returns)
        returns_centered = returns - mean_ret

        if np.std(returns_centered) == 0:
            return 0.5

        # Autocorrelation
        autocorr = np.corrcoef(returns_centered[:-1], returns_centered[1:])[0, 1]

        if np.isnan(autocorr):
            return 0.5

        # Calculate price oscillation around mean
        sma = np.mean(prices_arr[-20:])
        deviations = (prices_arr[-20:] - sma) / sma

        # Count sign changes (more changes = more mean reverting)
        sign_changes = np.sum(np.diff(np.sign(deviations)) != 0)
        oscillation_score = min(sign_changes / 10, 1.0)  # Normalize

        # Combine: negative autocorr + high oscillation = mean reverting
        # autocorr ranges from -1 to 1, transform to 0-1 where -1 -> 1, 1 -> 0
        autocorr_score = (1 - autocorr) / 2

        mr_score = (autocorr_score * 0.6 + oscillation_score * 0.4)

        return float(np.clip(mr_score, 0, 1))

    def _classify(
        self,
        vix: float,
        vix_structure: str,
        trend_strength: float,
        mr_score: float
    ) -> Tuple[MarketRegime, float]:
        """
        Classify market regime based on calculated metrics.

        Args:
            vix: Current VIX level
            vix_structure: "contango" or "backwardation"
            trend_strength: Trend strength (-1 to +1)
            mr_score: Mean reversion score (0 to 1)

        Returns:
            Tuple of (MarketRegime, confidence)
        """
        # Calculate SPY drawdown if we have enough data
        spy_drawdown = 0.0
        if len(self.spy_prices) >= 20:
            recent_high = max(list(self.spy_prices)[-20:])
            current = self.spy_prices[-1]
            spy_drawdown = (recent_high - current) / recent_high

        # Priority 1: CRASH detection
        if vix >= self.VIX_CRISIS and vix_structure == "backwardation":
            if spy_drawdown >= 0.05:  # 5%+ drawdown
                self._in_crash = True
                self._crash_low = min(self._crash_low or float('inf'), self.spy_prices[-1])
                if not self._pre_crash_high:
                    self._pre_crash_high = max(list(self.spy_prices)[-20:])
                confidence = min(0.6 + (vix - self.VIX_CRISIS) / 30, 0.95)
                return (MarketRegime.CRASH, confidence)

        # Priority 2: RECOVERY detection (post-crash bounce)
        if self._in_crash and self._crash_low:
            current = self.spy_prices[-1]
            bounce = (current - self._crash_low) / self._crash_low
            if bounce >= 0.05 and vix < self.VIX_CRISIS:  # 5%+ bounce, VIX calming
                # Check if we've recovered enough to exit recovery state
                if self._pre_crash_high and current >= self._pre_crash_high * 0.95:
                    self._in_crash = False
                    self._crash_low = None
                    self._pre_crash_high = None
                else:
                    confidence = min(0.5 + bounce, 0.85)
                    return (MarketRegime.RECOVERY, confidence)

        # Priority 3: HIGH_VOL
        if vix >= self.VIX_ELEVATED:
            confidence = min(0.5 + (vix - self.VIX_ELEVATED) / 20, 0.9)
            return (MarketRegime.HIGH_VOL, confidence)

        # Priority 4: Strong TRENDING
        if abs(trend_strength) >= self.TREND_STRONG:
            confidence = 0.5 + abs(trend_strength) * 0.4
            if trend_strength > 0:
                return (MarketRegime.TRENDING_UP, confidence)
            else:
                return (MarketRegime.TRENDING_DOWN, confidence)

        # Priority 5: MEAN_REVERTING
        if mr_score >= self.MR_STRONG and abs(trend_strength) < self.TREND_WEAK:
            confidence = 0.5 + mr_score * 0.4
            return (MarketRegime.MEAN_REVERTING, confidence)

        # Priority 6: Weak trend detection
        if abs(trend_strength) >= self.TREND_WEAK:
            confidence = 0.4 + abs(trend_strength) * 0.3
            if trend_strength > 0:
                return (MarketRegime.TRENDING_UP, confidence)
            else:
                return (MarketRegime.TRENDING_DOWN, confidence)

        # Default: Low confidence mean-reverting (most common in quiet markets)
        if vix < self.VIX_LOW:
            return (MarketRegime.MEAN_REVERTING, 0.5)

        return (MarketRegime.UNKNOWN, 0.3)

    def fetch_and_update(self) -> bool:
        """
        Fetch current market data from collectors and update state.

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Try to get SPY price
            spy_price = None
            if get_stock_price:
                try:
                    spy_price = get_stock_price("SPY")
                except Exception as e:
                    logger.warning("Failed to fetch SPY price: %s", e)

            # Try to get VIX data
            vix = None
            vix_3m = None
            if get_vix_data:
                try:
                    vix_data = get_vix_data()
                    if vix_data:
                        vix = vix_data.get("vix") or vix_data.get("spot")
                except Exception as e:
                    logger.warning("Failed to fetch VIX data: %s", e)

            if get_vix_term_structure:
                try:
                    term_data = get_vix_term_structure()
                    if term_data:
                        vix_3m = term_data.get("vix_3m") or term_data.get("m3")
                except Exception as e:
                    logger.debug("Failed to fetch VIX term structure: %s", e)

            # Try to get TLT price
            tlt_price = None
            if get_stock_price:
                try:
                    tlt_price = get_stock_price("TLT")
                except Exception as e:
                    logger.debug("Failed to fetch TLT price: %s", e)

            # Update if we have minimum required data
            if spy_price is not None and vix is not None:
                self.update(
                    spy_price=spy_price,
                    vix=vix,
                    vix_3m=vix_3m,
                    tlt_price=tlt_price
                )
                return True
            else:
                logger.warning("Insufficient data for regime update: SPY=%s, VIX=%s",
                              spy_price, vix)
                return False

        except Exception as e:
            logger.error("Error in fetch_and_update: %s", e)
            return False

    def reset(self) -> None:
        """Reset all state and history."""
        self.spy_prices.clear()
        self.vix_history.clear()
        self.vix_3m_history.clear()
        self.tlt_prices.clear()
        self.timestamps.clear()

        self._current_state = None
        self._last_regime = MarketRegime.UNKNOWN
        self._regime_duration = 0
        self._in_crash = False
        self._crash_low = None
        self._pre_crash_high = None

        logger.info("RegimeDetector state reset")

    def get_regime_summary(self) -> Dict:
        """
        Get a summary of current regime state for logging/debugging.

        Returns:
            Dictionary with regime summary
        """
        state = self._current_state
        if state is None:
            return {"regime": "unknown", "detected": False}

        return {
            "regime": state.regime.value,
            "confidence": round(state.confidence, 3),
            "vix_level": round(state.vix_level, 2),
            "vix_structure": state.vix_structure,
            "trend_strength": round(state.trend_strength, 3),
            "mean_reversion_score": round(state.mean_reversion_score, 3),
            "regime_duration": self._regime_duration,
            "data_points": len(self.spy_prices),
            "detected_at": state.detected_at.isoformat(),
        }


# Singleton instance for global access
regime_detector = RegimeDetector()


def get_regime_detector() -> RegimeDetector:
    """Get the global regime detector instance."""
    return regime_detector


def detect_current_regime() -> RegimeState:
    """
    Convenience function to fetch data and detect current regime.

    Returns:
        Current RegimeState
    """
    regime_detector.fetch_and_update()
    return regime_detector.detect_regime()


def get_signal_multipliers() -> Dict[str, float]:
    """
    Get current signal weight multipliers based on regime.

    Returns:
        Dictionary of signal type to weight multiplier
    """
    return regime_detector.get_regime_multipliers()
