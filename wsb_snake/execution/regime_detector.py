"""
6-State Market Regime Classifier

Classifies market conditions into one of six regimes using:
- VIX level thresholds
- VIX term structure (contango/backwardation)
- SPY trend vs 20-period SMA
- Mean reversion scoring

Provides regime-specific multipliers for position sizing, conviction thresholds,
and stop-loss parameters.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class RegimeState(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOL = "high_vol"
    CRASH = "crash"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


@dataclass
class RegimeResult:
    """
    Result of regime detection with supporting metrics.

    Attributes:
        regime: The classified market regime
        confidence: Confidence in classification (0.0-1.0)
        vix_level: Current VIX level
        vix_structure: Term structure ("contango", "backwardation", or "unknown")
        trend_strength: Directional momentum (-1.0 to +1.0)
        mean_reversion_score: Likelihood of mean reversion (0.0-1.0)
        detected_at: Timestamp of detection
    """
    regime: RegimeState
    confidence: float
    vix_level: float
    vix_structure: str
    trend_strength: float
    mean_reversion_score: float
    detected_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Validate and clamp ranges after initialization."""
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.trend_strength = max(-1.0, min(1.0, self.trend_strength))
        self.mean_reversion_score = max(0.0, min(1.0, self.mean_reversion_score))


class RegimeDetector:
    """
    6-State Market Regime Classifier.

    Uses multi-factor analysis to classify current market conditions:
    - VIX level thresholds (<15 calm, 15-20 normal, 20-30 elevated, >30 crisis)
    - VIX term structure (contango/backwardation)
    - SPY trend vs 20-period SMA
    - Mean reversion scoring

    Provides regime-specific multipliers for position sizing, conviction
    thresholds, and stop-loss parameters.

    Example:
        detector = RegimeDetector()
        result = detector.detect_regime()
        multipliers = detector.get_regime_multipliers()
    """

    # History configuration
    MIN_HISTORY_SIZE: int = 20
    MAX_HISTORY_SIZE: int = 100
    SMA_PERIOD: int = 20

    # VIX level thresholds
    VIX_CALM: float = 15.0
    VIX_NORMAL_HIGH: float = 20.0
    VIX_ELEVATED_HIGH: float = 30.0

    # Trend thresholds
    TREND_STRONG: float = 0.6
    TREND_MODERATE: float = 0.3

    # Mean reversion threshold
    MR_STRONG: float = 0.65

    # Regime-specific multipliers
    REGIME_MULTIPLIERS: Dict[RegimeState, Dict[str, float]] = {
        RegimeState.TRENDING_UP: {
            "position_size": 1.2,
            "conviction_threshold": 65.0,
            "stop_loss": 0.12,
        },
        RegimeState.TRENDING_DOWN: {
            "position_size": 0.8,
            "conviction_threshold": 70.0,
            "stop_loss": 0.10,
        },
        RegimeState.MEAN_REVERTING: {
            "position_size": 1.0,
            "conviction_threshold": 65.0,
            "stop_loss": 0.10,
        },
        RegimeState.HIGH_VOL: {
            "position_size": 0.6,
            "conviction_threshold": 75.0,
            "stop_loss": 0.08,
        },
        RegimeState.CRASH: {
            "position_size": 0.5,
            "conviction_threshold": 80.0,
            "stop_loss": 0.06,
        },
        RegimeState.RECOVERY: {
            "position_size": 1.0,
            "conviction_threshold": 70.0,
            "stop_loss": 0.10,
        },
        RegimeState.UNKNOWN: {
            "position_size": 0.7,
            "conviction_threshold": 70.0,
            "stop_loss": 0.10,
        },
    }

    def __init__(self, max_history: int = MAX_HISTORY_SIZE) -> None:
        """
        Initialize the regime detector.

        Args:
            max_history: Maximum number of data points to retain in rolling window
        """
        self.max_history = max_history

        # Price and VIX history
        self._spy_prices: deque[float] = deque(maxlen=max_history)
        self._vix_history: deque[float] = deque(maxlen=max_history)
        self._timestamps: deque[datetime] = deque(maxlen=max_history)

        # Current state
        self._current_result: Optional[RegimeResult] = None
        self._last_regime: RegimeState = RegimeState.UNKNOWN
        self._regime_duration: int = 0

        # Crash tracking
        self._in_crash: bool = False
        self._crash_low: Optional[float] = None
        self._pre_crash_high: Optional[float] = None

        # VIX structure collector (lazy loaded)
        self._vix_collector: Optional[Any] = None
        self._polygon_enhanced: Optional[Any] = None

        logger.info(
            "RegimeDetector initialized with max_history=%d, SMA_period=%d",
            max_history,
            self.SMA_PERIOD,
        )

    def _get_vix_collector(self) -> Optional[Any]:
        """Lazy load VIX structure collector."""
        if self._vix_collector is None:
            try:
                from wsb_snake.collectors.vix_structure import vix_structure
                self._vix_collector = vix_structure
                logger.debug("VIX structure collector loaded successfully")
            except ImportError as e:
                logger.warning("Failed to import vix_structure: %s", e)
            except Exception as e:
                logger.error("Error loading vix_structure collector: %s", e)
        return self._vix_collector

    def _get_polygon_enhanced(self) -> Optional[Any]:
        """Lazy load Polygon enhanced data collector."""
        if self._polygon_enhanced is None:
            try:
                from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
                self._polygon_enhanced = polygon_enhanced
                logger.debug("Polygon enhanced collector loaded successfully")
            except ImportError as e:
                logger.warning("Failed to import polygon_enhanced: %s", e)
            except Exception as e:
                logger.error("Error loading polygon_enhanced collector: %s", e)
        return self._polygon_enhanced

    def _fetch_vix_data(self) -> Tuple[float, str]:
        """
        Fetch current VIX level and term structure.

        Returns:
            Tuple of (vix_level, vix_structure)
            Falls back to (20.0, "unknown") on failure
        """
        vix_level = 20.0
        vix_structure = "unknown"

        collector = self._get_vix_collector()
        if collector is None:
            logger.debug("VIX collector unavailable, using defaults")
            return vix_level, vix_structure

        try:
            # Get VIX spot price
            vix_spot = collector.get_vix_spot()
            if vix_spot and vix_spot > 0:
                vix_level = vix_spot
                logger.debug("Fetched VIX spot: %.2f", vix_level)

            # Get term structure for contango/backwardation
            term_data = collector.get_term_structure()
            if term_data:
                is_backwardation = term_data.get("is_backwardation", False)
                is_contango = term_data.get("is_contango", True)

                if is_backwardation:
                    vix_structure = "backwardation"
                elif is_contango:
                    vix_structure = "contango"
                else:
                    vix_structure = term_data.get("structure", "unknown")

                logger.debug(
                    "VIX term structure: %s (backwardation=%s, contango=%s)",
                    vix_structure,
                    is_backwardation,
                    is_contango,
                )

        except Exception as e:
            logger.warning("Error fetching VIX data: %s", e)

        return vix_level, vix_structure

    def _fetch_spy_price(self) -> Optional[float]:
        """
        Fetch current SPY price.

        Returns:
            Current SPY price or None on failure
        """
        polygon = self._get_polygon_enhanced()
        if polygon is None:
            logger.debug("Polygon collector unavailable for SPY price")
            return None

        try:
            # Try get_spot_price first
            if hasattr(polygon, "get_spot_price"):
                price = polygon.get_spot_price("SPY")
                if price and price > 0:
                    logger.debug("Fetched SPY price via get_spot_price: %.2f", price)
                    return price

            # Try get_price as fallback
            if hasattr(polygon, "get_price"):
                price = polygon.get_price("SPY")
                if price and price > 0:
                    logger.debug("Fetched SPY price via get_price: %.2f", price)
                    return price

            # Try get_quote as another fallback
            if hasattr(polygon, "get_quote"):
                quote = polygon.get_quote("SPY")
                if quote:
                    price = quote.get("price") or quote.get("last") or quote.get("mid")
                    if price and price > 0:
                        logger.debug("Fetched SPY price via get_quote: %.2f", price)
                        return price

        except Exception as e:
            logger.warning("Error fetching SPY price: %s", e)

        return None

    def _update_history(self, spy_price: float, vix_level: float) -> None:
        """
        Update internal price history.

        Args:
            spy_price: Current SPY price
            vix_level: Current VIX level
        """
        now = datetime.now()
        self._spy_prices.append(spy_price)
        self._vix_history.append(vix_level)
        self._timestamps.append(now)

        logger.debug(
            "History updated: SPY=%.2f, VIX=%.2f, points=%d",
            spy_price,
            vix_level,
            len(self._spy_prices),
        )

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """
        Calculate SPY trend strength relative to 20-period SMA.

        Uses:
        - Price position relative to SMA
        - Multi-timeframe momentum
        - SMA slope

        Args:
            prices: List of historical prices

        Returns:
            Trend strength from -1.0 (strong down) to +1.0 (strong up)
        """
        if len(prices) < self.SMA_PERIOD:
            return 0.0

        prices_arr = np.array(prices)
        current_price = prices_arr[-1]

        # 20-period SMA
        sma_20 = np.mean(prices_arr[-self.SMA_PERIOD:])

        # Price position relative to SMA (-1 to +1)
        if sma_20 > 0:
            price_deviation = (current_price - sma_20) / sma_20
        else:
            price_deviation = 0.0

        # Calculate returns for momentum
        returns = np.diff(prices_arr) / prices_arr[:-1]

        # Multi-timeframe momentum
        mom_5 = np.mean(returns[-5:]) if len(returns) >= 5 else 0.0
        mom_10 = np.mean(returns[-10:]) if len(returns) >= 10 else 0.0
        mom_20 = np.mean(returns[-20:]) if len(returns) >= 20 else 0.0

        # Weighted momentum (short-term weighted more)
        weighted_mom = mom_5 * 0.5 + mom_10 * 0.3 + mom_20 * 0.2

        # SMA slope (10-day vs 20-day SMA)
        if len(prices_arr) >= self.SMA_PERIOD:
            sma_10 = np.mean(prices_arr[-10:])
            sma_slope = (sma_10 - sma_20) / sma_20 if sma_20 != 0 else 0.0
        else:
            sma_slope = 0.0

        # Combine factors
        # Scale deviation (typical is 1-3%)
        scaled_deviation = np.clip(price_deviation * 20, -1, 1)
        # Scale momentum (typical daily returns < 1%)
        scaled_mom = np.clip(weighted_mom * 50, -1, 1)
        # Scale slope
        scaled_slope = np.clip(sma_slope * 10, -1, 1)

        # Final trend strength
        trend = scaled_deviation * 0.4 + scaled_mom * 0.4 + scaled_slope * 0.2

        return float(np.clip(trend, -1.0, 1.0))

    def _calculate_mean_reversion_score(self, prices: List[float]) -> float:
        """
        Calculate mean reversion likelihood.

        Uses:
        - Return autocorrelation (negative = mean reverting)
        - Price oscillation frequency around SMA

        Args:
            prices: List of historical prices

        Returns:
            Mean reversion score from 0.0 (trending) to 1.0 (mean reverting)
        """
        if len(prices) < self.SMA_PERIOD:
            return 0.5

        prices_arr = np.array(prices)
        returns = np.diff(prices_arr) / prices_arr[:-1]

        if len(returns) < 10:
            return 0.5

        # Lag-1 autocorrelation of returns
        mean_ret = np.mean(returns)
        returns_centered = returns - mean_ret

        if np.std(returns_centered) < 1e-10:
            return 0.5

        try:
            autocorr = np.corrcoef(returns_centered[:-1], returns_centered[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        except Exception:
            autocorr = 0.0

        # Price oscillation around SMA
        sma = np.mean(prices_arr[-self.SMA_PERIOD:])
        if sma > 0:
            deviations = (prices_arr[-self.SMA_PERIOD:] - sma) / sma
            # Count sign changes
            sign_changes = np.sum(np.diff(np.sign(deviations)) != 0)
            oscillation_score = min(sign_changes / 10.0, 1.0)
        else:
            oscillation_score = 0.5

        # Combine: negative autocorr + high oscillation = mean reverting
        # Transform autocorr from [-1, 1] to [0, 1] where -1 -> 1, 1 -> 0
        autocorr_score = (1.0 - autocorr) / 2.0

        mr_score = autocorr_score * 0.6 + oscillation_score * 0.4

        return float(np.clip(mr_score, 0.0, 1.0))

    def _classify_vix_level(self, vix: float) -> str:
        """
        Classify VIX level into categories.

        Args:
            vix: Current VIX level

        Returns:
            "calm", "normal", "elevated", or "crisis"
        """
        if vix < self.VIX_CALM:
            return "calm"
        elif vix < self.VIX_NORMAL_HIGH:
            return "normal"
        elif vix < self.VIX_ELEVATED_HIGH:
            return "elevated"
        else:
            return "crisis"

    def _classify_regime(
        self,
        vix: float,
        vix_structure: str,
        trend_strength: float,
        mr_score: float,
    ) -> Tuple[RegimeState, float]:
        """
        Classify market regime based on calculated metrics.

        Priority order:
        1. CRASH (VIX crisis + backwardation + drawdown)
        2. RECOVERY (post-crash bounce)
        3. HIGH_VOL (elevated VIX)
        4. TRENDING_UP/DOWN (strong trend)
        5. MEAN_REVERTING (high MR score, weak trend)
        6. UNKNOWN

        Args:
            vix: Current VIX level
            vix_structure: "contango", "backwardation", or "unknown"
            trend_strength: Trend strength (-1 to +1)
            mr_score: Mean reversion score (0 to 1)

        Returns:
            Tuple of (RegimeState, confidence)
        """
        vix_category = self._classify_vix_level(vix)

        # Calculate SPY drawdown if we have enough data
        spy_drawdown = 0.0
        if len(self._spy_prices) >= self.SMA_PERIOD:
            recent_high = max(list(self._spy_prices)[-self.SMA_PERIOD:])
            current = self._spy_prices[-1]
            if recent_high > 0:
                spy_drawdown = (recent_high - current) / recent_high

        logger.debug(
            "Classification inputs: vix_cat=%s, vix_struct=%s, trend=%.3f, "
            "mr=%.3f, drawdown=%.3f",
            vix_category,
            vix_structure,
            trend_strength,
            mr_score,
            spy_drawdown,
        )

        # Priority 1: CRASH detection
        if vix_category == "crisis" and vix_structure == "backwardation":
            if spy_drawdown >= 0.03:  # 3%+ drawdown
                self._in_crash = True
                if self._crash_low is None or self._spy_prices[-1] < self._crash_low:
                    self._crash_low = self._spy_prices[-1]
                if self._pre_crash_high is None and len(self._spy_prices) >= self.SMA_PERIOD:
                    self._pre_crash_high = max(list(self._spy_prices)[-self.SMA_PERIOD:])

                confidence = min(0.7 + (vix - self.VIX_ELEVATED_HIGH) / 40, 0.95)
                logger.info(
                    "CRASH detected: VIX=%.1f, drawdown=%.1f%%, confidence=%.2f",
                    vix,
                    spy_drawdown * 100,
                    confidence,
                )
                return (RegimeState.CRASH, confidence)

        # Priority 2: RECOVERY detection (post-crash bounce)
        if self._in_crash and self._crash_low is not None:
            current = self._spy_prices[-1] if self._spy_prices else 0
            if self._crash_low > 0 and current > 0:
                bounce = (current - self._crash_low) / self._crash_low

                if bounce >= 0.03 and vix < self.VIX_ELEVATED_HIGH:
                    # Check if fully recovered
                    if self._pre_crash_high and current >= self._pre_crash_high * 0.97:
                        # Exit crash/recovery state
                        self._in_crash = False
                        self._crash_low = None
                        self._pre_crash_high = None
                        logger.info("Market recovered from crash state")
                    else:
                        confidence = min(0.5 + bounce * 2, 0.85)
                        logger.info(
                            "RECOVERY detected: bounce=%.1f%%, confidence=%.2f",
                            bounce * 100,
                            confidence,
                        )
                        return (RegimeState.RECOVERY, confidence)

        # Priority 3: HIGH_VOL (elevated or higher VIX)
        if vix_category in ("elevated", "crisis"):
            confidence = min(0.5 + (vix - self.VIX_NORMAL_HIGH) / 30, 0.90)
            logger.info(
                "HIGH_VOL detected: VIX=%.1f, category=%s, confidence=%.2f",
                vix,
                vix_category,
                confidence,
            )
            return (RegimeState.HIGH_VOL, confidence)

        # Priority 4: Strong TRENDING
        if abs(trend_strength) >= self.TREND_STRONG:
            confidence = 0.5 + abs(trend_strength) * 0.4
            if trend_strength > 0:
                logger.info(
                    "TRENDING_UP detected: trend=%.3f, confidence=%.2f",
                    trend_strength,
                    confidence,
                )
                return (RegimeState.TRENDING_UP, confidence)
            else:
                logger.info(
                    "TRENDING_DOWN detected: trend=%.3f, confidence=%.2f",
                    trend_strength,
                    confidence,
                )
                return (RegimeState.TRENDING_DOWN, confidence)

        # Priority 5: MEAN_REVERTING (high MR score, weak trend)
        if mr_score >= self.MR_STRONG and abs(trend_strength) < self.TREND_MODERATE:
            confidence = 0.5 + mr_score * 0.4
            logger.info(
                "MEAN_REVERTING detected: mr_score=%.3f, trend=%.3f, confidence=%.2f",
                mr_score,
                trend_strength,
                confidence,
            )
            return (RegimeState.MEAN_REVERTING, confidence)

        # Priority 6: Moderate trending
        if abs(trend_strength) >= self.TREND_MODERATE:
            confidence = 0.4 + abs(trend_strength) * 0.3
            if trend_strength > 0:
                logger.info(
                    "TRENDING_UP (moderate) detected: trend=%.3f, confidence=%.2f",
                    trend_strength,
                    confidence,
                )
                return (RegimeState.TRENDING_UP, confidence)
            else:
                logger.info(
                    "TRENDING_DOWN (moderate) detected: trend=%.3f, confidence=%.2f",
                    trend_strength,
                    confidence,
                )
                return (RegimeState.TRENDING_DOWN, confidence)

        # Default: Calm markets tend to be mean reverting
        if vix_category == "calm":
            logger.info("MEAN_REVERTING (calm market) detected: VIX=%.1f", vix)
            return (RegimeState.MEAN_REVERTING, 0.5)

        logger.info("UNKNOWN regime: insufficient signals")
        return (RegimeState.UNKNOWN, 0.3)

    def detect_regime(self) -> RegimeState:
        """
        Main detection method - classify current market regime.

        Fetches latest market data, updates history, and classifies the regime
        using VIX levels, term structure, SPY trend, and mean reversion scoring.

        Returns:
            RegimeState enum value representing current market regime
        """
        try:
            # Fetch current data
            vix_level, vix_structure = self._fetch_vix_data()
            spy_price = self._fetch_spy_price()

            # If we can't get SPY price, try to use cached data
            if spy_price is None:
                if len(self._spy_prices) > 0:
                    spy_price = self._spy_prices[-1]
                    logger.debug("Using cached SPY price: %.2f", spy_price)
                else:
                    logger.warning(
                        "Cannot fetch SPY price and no cached data, returning UNKNOWN"
                    )
                    self._current_result = RegimeResult(
                        regime=RegimeState.UNKNOWN,
                        confidence=0.0,
                        vix_level=vix_level,
                        vix_structure=vix_structure,
                        trend_strength=0.0,
                        mean_reversion_score=0.5,
                    )
                    return RegimeState.UNKNOWN

            # Update history
            self._update_history(spy_price, vix_level)

            # Check for sufficient history
            if len(self._spy_prices) < self.MIN_HISTORY_SIZE:
                logger.debug(
                    "Insufficient history for regime detection: %d/%d points",
                    len(self._spy_prices),
                    self.MIN_HISTORY_SIZE,
                )
                self._current_result = RegimeResult(
                    regime=RegimeState.UNKNOWN,
                    confidence=0.0,
                    vix_level=vix_level,
                    vix_structure=vix_structure,
                    trend_strength=0.0,
                    mean_reversion_score=0.5,
                )
                return RegimeState.UNKNOWN

            # Calculate metrics
            prices = list(self._spy_prices)
            trend_strength = self._calculate_trend_strength(prices)
            mr_score = self._calculate_mean_reversion_score(prices)

            # Classify regime
            regime, confidence = self._classify_regime(
                vix_level, vix_structure, trend_strength, mr_score
            )

            # Track regime duration
            if regime == self._last_regime:
                self._regime_duration += 1
            else:
                self._regime_duration = 1
                self._last_regime = regime

            # Store result
            self._current_result = RegimeResult(
                regime=regime,
                confidence=confidence,
                vix_level=vix_level,
                vix_structure=vix_structure,
                trend_strength=trend_strength,
                mean_reversion_score=mr_score,
            )

            logger.info(
                "Regime detected: %s (confidence=%.2f, VIX=%.1f [%s], "
                "trend=%.3f, MR=%.3f, duration=%d)",
                regime.value,
                confidence,
                vix_level,
                self._classify_vix_level(vix_level),
                trend_strength,
                mr_score,
                self._regime_duration,
            )

            return regime

        except Exception as e:
            logger.error("Error in detect_regime: %s", e, exc_info=True)
            self._current_result = RegimeResult(
                regime=RegimeState.UNKNOWN,
                confidence=0.0,
                vix_level=20.0,
                vix_structure="unknown",
                trend_strength=0.0,
                mean_reversion_score=0.5,
            )
            return RegimeState.UNKNOWN

    def get_regime_multipliers(self) -> Dict[str, float]:
        """
        Get regime-specific trading multipliers.

        Returns dictionary with:
        - position_size: 0.5-1.5 based on regime (scale position sizing)
        - conviction_threshold: 60-80 based on regime (minimum score to trade)
        - stop_loss: tighter in high vol regimes (percentage as decimal)

        Returns:
            Dict with position_size, conviction_threshold, and stop_loss
        """
        # Ensure we have a current result
        if self._current_result is None:
            self.detect_regime()

        regime = (
            self._current_result.regime
            if self._current_result
            else RegimeState.UNKNOWN
        )

        multipliers = self.REGIME_MULTIPLIERS.get(
            regime, self.REGIME_MULTIPLIERS[RegimeState.UNKNOWN]
        )

        logger.debug(
            "Regime multipliers for %s: position_size=%.2f, "
            "conviction_threshold=%.1f, stop_loss=%.2f",
            regime.value,
            multipliers["position_size"],
            multipliers["conviction_threshold"],
            multipliers["stop_loss"],
        )

        return multipliers

    def get_current_result(self) -> Optional[RegimeResult]:
        """
        Get the most recent regime detection result.

        Returns:
            RegimeResult or None if detection hasn't run
        """
        return self._current_result

    def get_regime_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current regime state for logging/debugging.

        Returns:
            Dictionary with regime summary information
        """
        if self._current_result is None:
            return {
                "regime": "unknown",
                "detected": False,
                "data_points": len(self._spy_prices),
            }

        result = self._current_result
        return {
            "regime": result.regime.value,
            "confidence": round(result.confidence, 3),
            "vix_level": round(result.vix_level, 2),
            "vix_category": self._classify_vix_level(result.vix_level),
            "vix_structure": result.vix_structure,
            "trend_strength": round(result.trend_strength, 3),
            "mean_reversion_score": round(result.mean_reversion_score, 3),
            "regime_duration": self._regime_duration,
            "in_crash_state": self._in_crash,
            "data_points": len(self._spy_prices),
            "detected_at": result.detected_at.isoformat(),
        }

    def reset(self) -> None:
        """Reset all state and history."""
        self._spy_prices.clear()
        self._vix_history.clear()
        self._timestamps.clear()

        self._current_result = None
        self._last_regime = RegimeState.UNKNOWN
        self._regime_duration = 0
        self._in_crash = False
        self._crash_low = None
        self._pre_crash_high = None

        logger.info("RegimeDetector state reset")


# Singleton instance
regime_detector = RegimeDetector()


# Backwards compatibility aliases
MarketRegime = RegimeState


def get_regime_detector() -> RegimeDetector:
    """Get the global regime detector instance."""
    return regime_detector


def detect_current_regime() -> RegimeState:
    """
    Convenience function to detect current regime.

    Returns:
        Current RegimeState
    """
    return regime_detector.detect_regime()


def get_signal_multipliers() -> Dict[str, float]:
    """
    Get current signal weight multipliers based on regime.

    Returns:
        Dictionary of multiplier name to value
    """
    return regime_detector.get_regime_multipliers()
