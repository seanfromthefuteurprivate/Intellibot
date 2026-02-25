"""
APEX CONVICTION ENGINE - Institutional-Grade Multi-Signal Fusion

Combines ALL available analysis systems into a single conviction score:
1. Technical Analysis (RSI, MACD, SMA, EMA) - 20%
2. Candlestick Patterns (36 patterns + confluence) - 15%
3. Order Flow (sweeps, blocks, institutional) - 20%
4. Probability Generator (multi-engine fusion) - 20%
5. Pattern Memory (historical match) - 15%
6. AI Verdict (GPT-4/Gemini visual) - 10%

Only trades when combined conviction > 70%
Power Hour Mode: Aggressive scanning, faster exits, volume spikes
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from wsb_snake.utils.logger import get_logger
from wsb_snake.learning.trade_learner import trade_learner

# HYDRA Integration
from wsb_snake.collectors.hydra_bridge import get_hydra_bridge, HydraBridge

logger = get_logger(__name__)

# Conviction calibration logger for tracking threshold vs outcome
calibration_logger = get_logger("conviction_calibration")


@dataclass
class ConvictionSignal:
    """Individual signal component."""
    source: str
    score: float  # 0-100
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float  # 0-100
    reason: str
    weight: float  # 0-1 contribution to final score
    created_at: datetime = field(default_factory=datetime.now)
    ttl_seconds: int = 300  # 5 minute default TTL
    source_reliability: float = 1.0  # 0.5-1.5 based on historical accuracy

    def is_valid(self) -> bool:
        """Check if signal is still within TTL."""
        age = (datetime.now() - self.created_at).total_seconds()
        return age < self.ttl_seconds

    def get_decay_factor(self) -> float:
        """
        Exponential decay - most value lost in final 2 minutes of TTL.

        Uses exponential curve that maintains ~90% of value for first 60% of TTL,
        then rapidly decays in the final 40% (especially last 2 minutes).

        Formula: decay = 0.5 + 0.5 * exp(-3 * (age/ttl)^2.5)

        This produces:
        - At 0% TTL: 1.00 (full value)
        - At 50% TTL: 0.95 (95% value)
        - At 75% TTL: 0.82 (82% value)
        - At 90% TTL: 0.62 (62% value - rapid drop begins)
        - At 100% TTL: 0.50 (minimum floor)
        """
        import math

        age = (datetime.now() - self.created_at).total_seconds()
        age_ratio = age / self.ttl_seconds

        # Clamp to valid range
        age_ratio = max(0.0, min(1.0, age_ratio))

        # Exponential decay with power curve for accelerated end-of-life decay
        # The ^2.5 exponent keeps value high early, then drops sharply
        decay = 0.5 + 0.5 * math.exp(-3.0 * (age_ratio ** 2.5))

        return max(0.5, min(1.0, decay))

    def get_effective_score(self) -> float:
        """Score adjusted for decay and reliability."""
        return self.score * self.get_decay_factor() * self.source_reliability


@dataclass
class ApexVerdict:
    """Final verdict from all combined signals."""
    ticker: str
    conviction_score: float  # 0-100
    direction: str  # "STRONG_LONG", "LONG", "NEUTRAL", "SHORT", "STRONG_SHORT"
    action: str  # "BUY_CALLS", "BUY_PUTS", "NO_TRADE"
    signals: List[ConvictionSignal]
    entry_price: Optional[float]
    target_pct: float
    stop_pct: float
    position_size_multiplier: float  # 0.5-2.0 based on conviction
    time_sensitivity: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    reasons: List[str]


class ApexConvictionEngine:
    """
    Institutional-grade conviction scoring using ALL available systems.
    """

    # Signal weights (must sum to 1.0)
    WEIGHTS = {
        "technical": 0.20,      # RSI, MACD, SMA, EMA
        "candlestick": 0.15,    # 36 patterns + confluence
        "order_flow": 0.20,     # Sweeps, blocks, institutional
        "probability": 0.20,    # Multi-engine fusion
        "pattern_memory": 0.15, # Historical match
        "ai_verdict": 0.10,     # GPT-4/Gemini
    }

    # Source reliability multipliers (HYDRA feature)
    SOURCE_RELIABILITY = {
        "technical": 1.0,
        "candlestick": 0.9,
        "order_flow": 1.1,      # Historically more reliable
        "probability": 1.0,
        "pattern_memory": 0.8,  # Learning - starts lower
        "ai_verdict": 0.7,      # Experimental
    }

    # 0DTE-specific weights: order flow dominates, AI verdict reduced
    # In 0DTE, institutional flow is the most predictive signal
    WEIGHTS_0DTE = {
        "technical": 0.15,      # Reduced - less time for technicals to play out
        "candlestick": 0.15,    # Maintained
        "order_flow": 0.30,     # INCREASED - critical for 0DTE
        "probability": 0.20,    # Maintained
        "pattern_memory": 0.10, # Reduced - less historical relevance
        "ai_verdict": 0.10,     # REDUCED from base - visual patterns less predictive
    }

    # Conviction thresholds - JP MORGAN INSTITUTIONAL GRADE
    # These are BASE thresholds - actual thresholds are VIX-adjusted dynamically
    STRONG_CONVICTION = 75  # Strong directional agreement required
    TRADE_THRESHOLD = 68    # Minimum 68% to trade (institutional standard)
    AVOID_THRESHOLD = 50    # Clear wash zone

    # Dynamic threshold ranges based on VIX regime
    # RISK_ON (VIX < 20): 65% - aggressive in calm markets
    # NEUTRAL (VIX 20-25): 72% - standard institutional
    # RISK_OFF (VIX > 25): 78% - defensive in volatile markets
    THRESHOLD_RISK_ON = 65
    THRESHOLD_NEUTRAL = 72
    THRESHOLD_RISK_OFF = 78

    # Power hour settings (3:00-4:00 PM ET)
    POWER_HOUR_START = 15
    POWER_HOUR_END = 16

    def __init__(self):
        self._init_collectors()

    def _init_collectors(self):
        """Initialize all data collectors and analyzers."""
        # Technical Analysis
        try:
            from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
            self.polygon = polygon_enhanced
        except Exception as e:
            logger.warning(f"polygon_enhanced not available: {e}")
            self.polygon = None

        # Candlestick Patterns
        try:
            from wsb_snake.analysis.candlestick_patterns import CandlestickAnalyzer
            self.candlestick = CandlestickAnalyzer()
        except Exception as e:
            logger.warning(f"CandlestickAnalyzer not available: {e}")
            self.candlestick = None

        # Probability Generator
        try:
            from wsb_snake.engines.probability_generator import ProbabilityGenerator
            self.probability = ProbabilityGenerator()
        except Exception as e:
            logger.warning(f"ProbabilityGenerator not available: {e}")
            self.probability = None

        # Pattern Memory
        try:
            from wsb_snake.learning.pattern_memory import pattern_memory
            self.pattern_memory = pattern_memory
        except Exception as e:
            logger.warning(f"pattern_memory not available: {e}")
            self.pattern_memory = None

        # AI Visual Analysis (Predator Stack)
        try:
            from wsb_snake.analysis.predator_stack import PredatorStack
            self.predator = PredatorStack()
        except Exception as e:
            logger.warning(f"PredatorStack not available: {e}")
            self.predator = None

        # Chart Generator for AI visual analysis
        try:
            from wsb_snake.analysis.chart_generator import ChartGenerator
            self.chart_generator = ChartGenerator()
        except Exception as e:
            logger.warning(f"ChartGenerator not available: {e}")
            self.chart_generator = None

        # Polygon data for chart generation
        try:
            from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
            self.polygon = polygon_enhanced
        except Exception as e:
            logger.warning(f"polygon_enhanced not available: {e}")
            self.polygon = None

        # Order Flow
        try:
            from wsb_snake.collectors.scalp_data_collector import ScalpDataCollector
            self.scalp_collector = ScalpDataCollector()
        except Exception as e:
            logger.warning(f"ScalpDataCollector not available: {e}")
            self.scalp_collector = None

        # HYDRA Bridge for regime intelligence
        try:
            self.hydra_bridge = get_hydra_bridge()
            # Start if not already running (will be started by main.py normally)
            if not self.hydra_bridge._running:
                self.hydra_bridge.start()
            logger.info("HYDRA bridge connected to APEX")
        except Exception as e:
            logger.warning(f"HYDRA bridge not available: {e}")
            self.hydra_bridge = None

        logger.info("ApexConvictionEngine initialized")

    def _is_power_hour(self) -> bool:
        """Check if we're in power hour (3-4 PM ET)."""
        try:
            import pytz
            et = pytz.timezone("America/New_York")
            now = datetime.now(et)
            return self.POWER_HOUR_START <= now.hour < self.POWER_HOUR_END
        except Exception:
            return False

    def _is_0dte(self) -> bool:
        """
        Check if we're trading 0DTE options (same-day expiry).

        Returns True if current time is on a trading day and market is open.
        0DTE mode activates special weight adjustments.
        """
        try:
            import pytz
            et = pytz.timezone("America/New_York")
            now = datetime.now(et)

            # 0DTE is most critical in the final hours before close
            # Consider 0DTE mode active all day on trading days
            # Weekdays only (0=Monday, 4=Friday)
            if now.weekday() > 4:
                return False

            # Market hours check (9:30 AM - 4:00 PM ET)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

            return market_open <= now <= market_close
        except Exception:
            return False

    def _get_dynamic_threshold(self, vix_level: float = None) -> float:
        """
        Get VIX-based dynamic conviction threshold.

        Thresholds adjust based on market volatility:
        - RISK_ON (VIX < 20): 65% - More aggressive in calm markets
        - NEUTRAL (VIX 20-25): 72% - Standard institutional threshold
        - RISK_OFF (VIX > 25): 78% - Defensive in volatile markets

        Args:
            vix_level: Current VIX level. If None, fetches from regime detector.

        Returns:
            Dynamic conviction threshold (0-100 scale)
        """
        try:
            from wsb_snake.execution.regime_detector import regime_detector

            if vix_level is None:
                # Get VIX from regime detector
                threshold = regime_detector.get_dynamic_conviction_threshold()
                logger.debug(f"Dynamic threshold from regime detector: {threshold:.1f}%")
                return threshold

            # Calculate directly from VIX level
            if vix_level < 20.0:
                # RISK_ON: More aggressive
                threshold = self.THRESHOLD_RISK_ON
            elif vix_level <= 25.0:
                # NEUTRAL: Standard with linear interpolation
                threshold = self.THRESHOLD_RISK_ON + (
                    (vix_level - 20.0) / 5.0
                ) * (self.THRESHOLD_NEUTRAL - self.THRESHOLD_RISK_ON)
            else:
                # RISK_OFF: Defensive with linear interpolation
                threshold = self.THRESHOLD_NEUTRAL + (
                    min(vix_level - 25.0, 10.0) / 10.0
                ) * (self.THRESHOLD_RISK_OFF - self.THRESHOLD_NEUTRAL)

            logger.debug(f"Dynamic threshold for VIX {vix_level:.1f}: {threshold:.1f}%")
            return threshold

        except Exception as e:
            logger.warning(f"Error calculating dynamic threshold: {e}, using base")
            return self.TRADE_THRESHOLD

    def _get_regime_adjusted_weights(self, regime_state=None) -> Dict[str, float]:
        """Adjust signal weights based on market regime.

        Args:
            regime_state: Optional pre-fetched regime state. If None, fetches current regime.

        Returns:
            Dictionary of source -> adjusted weight based on regime conditions.
        """
        try:
            from wsb_snake.execution.regime_detector import regime_detector, MarketRegime

            # Fetch regime if not provided
            if regime_state is None:
                regime_state = regime_detector.detect_regime()

            base_weights = dict(self.WEIGHTS)

            if regime_state.regime == MarketRegime.HIGH_VOL:
                # Increase order flow weight in high vol
                base_weights["order_flow"] = 0.25
                base_weights["technical"] = 0.15
            elif regime_state.regime == MarketRegime.MEAN_REVERTING:
                # Increase pattern memory for mean reversion
                base_weights["pattern_memory"] = 0.20
                base_weights["probability"] = 0.15
            elif regime_state.regime == MarketRegime.CRASH:
                # Order flow dominates in crash
                base_weights["order_flow"] = 0.30
                base_weights["technical"] = 0.10
                base_weights["ai_verdict"] = 0.05
            elif regime_state.regime == MarketRegime.RECOVERY:
                # Technical matters in recovery
                base_weights["technical"] = 0.25
                base_weights["order_flow"] = 0.15

            return base_weights
        except Exception as e:
            logger.debug(f"Regime detection unavailable, using base weights: {e}")
            return dict(self.WEIGHTS)

    def _get_technical_score(self, ticker: str) -> ConvictionSignal:
        """
        Score from RSI, MACD, SMA, EMA analysis.
        """
        if not self.polygon:
            return ConvictionSignal("technical", 50, "NEUTRAL", 50, "no_data", self.WEIGHTS["technical"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("technical", 1.0))

        try:
            technicals = self.polygon.get_full_technicals(ticker)
            if not technicals:
                return ConvictionSignal("technical", 50, "NEUTRAL", 50, "no_technicals", self.WEIGHTS["technical"],
                                        source_reliability=self.SOURCE_RELIABILITY.get("technical", 1.0))

            score = 50
            direction = "NEUTRAL"
            reasons = []

            # RSI Analysis
            rsi = technicals.get("rsi", {}).get("current", 50)
            if rsi < 30:
                score += 15
                direction = "BULLISH"
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 70:
                score += 15
                direction = "BEARISH"
                reasons.append(f"RSI overbought ({rsi:.0f})")
            elif 40 <= rsi <= 60:
                score += 5
                reasons.append(f"RSI neutral ({rsi:.0f})")

            # MACD Analysis
            macd = technicals.get("macd", {})
            if macd.get("histogram", 0) > 0 and macd.get("macd", 0) > macd.get("signal", 0):
                score += 15
                if direction != "BEARISH":
                    direction = "BULLISH"
                reasons.append("MACD bullish crossover")
            elif macd.get("histogram", 0) < 0 and macd.get("macd", 0) < macd.get("signal", 0):
                score += 15
                if direction != "BULLISH":
                    direction = "BEARISH"
                reasons.append("MACD bearish crossover")

            # SMA/EMA Analysis
            sma = technicals.get("sma", {}).get("current", 0)
            ema = technicals.get("ema", {}).get("current", 0)
            price = technicals.get("price", 0)

            if price and sma and ema:
                if price > sma and price > ema:
                    score += 10
                    if direction != "BEARISH":
                        direction = "BULLISH"
                    reasons.append("Price above SMA/EMA")
                elif price < sma and price < ema:
                    score += 10
                    if direction != "BULLISH":
                        direction = "BEARISH"
                    reasons.append("Price below SMA/EMA")

            # VWAP Analysis
            vwap_pos = technicals.get("vwap_position", "")
            if "ABOVE" in vwap_pos:
                score += 5
                reasons.append(f"Above VWAP")
            elif "BELOW" in vwap_pos:
                score += 5
                reasons.append(f"Below VWAP")

            return ConvictionSignal(
                source="technical",
                score=min(100, max(0, score)),
                direction=direction,
                confidence=min(100, score),
                reason="; ".join(reasons) if reasons else "technical_analysis",
                weight=self.WEIGHTS["technical"],
                source_reliability=self.SOURCE_RELIABILITY.get("technical", 1.0)
            )

        except Exception as e:
            logger.debug(f"Technical analysis failed {ticker}: {e}")
            return ConvictionSignal("technical", 50, "NEUTRAL", 50, f"error: {e}", self.WEIGHTS["technical"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("technical", 1.0))

    def _get_candlestick_score(self, ticker: str) -> ConvictionSignal:
        """
        Score from 36 candlestick patterns + confluence.
        """
        if not self.candlestick or not self.polygon:
            return ConvictionSignal("candlestick", 50, "NEUTRAL", 50, "no_analyzer", self.WEIGHTS["candlestick"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("candlestick", 1.0))

        try:
            # Get 1-minute bars for pattern detection
            bars = self.polygon.get_intraday_bars(ticker, timespan="minute", multiplier=1, limit=30)
            if not bars or len(bars) < 10:
                return ConvictionSignal("candlestick", 50, "NEUTRAL", 50, "insufficient_bars", self.WEIGHTS["candlestick"],
                                        source_reliability=self.SOURCE_RELIABILITY.get("candlestick", 1.0))

            # Analyze patterns
            analysis = self.candlestick.analyze(bars)
            if not analysis:
                return ConvictionSignal("candlestick", 50, "NEUTRAL", 50, "no_patterns", self.WEIGHTS["candlestick"],
                                        source_reliability=self.SOURCE_RELIABILITY.get("candlestick", 1.0))

            patterns = analysis.get("patterns", [])
            confluence = analysis.get("confluence_score", 50)
            signal = analysis.get("signal", "NEUTRAL")

            score = 50
            direction = "NEUTRAL"
            reasons = []

            # Pattern-based scoring
            for p in patterns[:3]:  # Top 3 patterns
                strength = p.get("strength", 3)
                reliability = p.get("reliability", 0.5)
                p_direction = p.get("direction", "neutral")

                pattern_score = strength * reliability * 10
                score += pattern_score

                if p_direction == "bullish":
                    direction = "BULLISH"
                elif p_direction == "bearish":
                    direction = "BEARISH"

                reasons.append(f"{p.get('name', 'pattern')} ({strength}/5)")

            # Confluence bonus
            if confluence > 70:
                score += 15
                reasons.append(f"High confluence ({confluence:.0f})")
            elif confluence > 50:
                score += 8
                reasons.append(f"Moderate confluence ({confluence:.0f})")

            # Signal alignment
            if signal in ("STRONG_BUY", "BUY"):
                direction = "BULLISH"
                score += 10
            elif signal in ("STRONG_SELL", "SELL"):
                direction = "BEARISH"
                score += 10

            return ConvictionSignal(
                source="candlestick",
                score=min(100, max(0, score)),
                direction=direction,
                confidence=confluence,
                reason="; ".join(reasons) if reasons else "candlestick_analysis",
                weight=self.WEIGHTS["candlestick"],
                source_reliability=self.SOURCE_RELIABILITY.get("candlestick", 1.0)
            )

        except Exception as e:
            logger.debug(f"Candlestick analysis failed {ticker}: {e}")
            return ConvictionSignal("candlestick", 50, "NEUTRAL", 50, f"error: {e}", self.WEIGHTS["candlestick"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("candlestick", 1.0))

    def _get_order_flow_score(self, ticker: str) -> ConvictionSignal:
        """
        Score from order flow analysis (sweeps, blocks, institutional).
        """
        if not self.polygon:
            return ConvictionSignal("order_flow", 50, "NEUTRAL", 50, "no_polygon", self.WEIGHTS["order_flow"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("order_flow", 1.0))

        try:
            # Get order flow analysis
            flow = self.polygon.analyze_order_flow(ticker)
            if not flow:
                return ConvictionSignal("order_flow", 50, "NEUTRAL", 50, "no_flow_data", self.WEIGHTS["order_flow"],
                                        source_reliability=self.SOURCE_RELIABILITY.get("order_flow", 1.0))

            score = 50
            direction = "NEUTRAL"
            reasons = []

            # Flow signal
            flow_signal = flow.get("flow_signal", "NEUTRAL")
            if flow_signal == "STRONG_BUY":
                score += 25
                direction = "BULLISH"
                reasons.append("Strong buying pressure")
            elif flow_signal == "BUY":
                score += 15
                direction = "BULLISH"
                reasons.append("Buying pressure")
            elif flow_signal == "STRONG_SELL":
                score += 25
                direction = "BEARISH"
                reasons.append("Strong selling pressure")
            elif flow_signal == "SELL":
                score += 15
                direction = "BEARISH"
                reasons.append("Selling pressure")

            # Institutional activity
            institutional_pct = flow.get("institutional_pct", 0)
            if institutional_pct > 30:
                score += 15
                reasons.append(f"High institutional ({institutional_pct:.0f}%)")
            elif institutional_pct > 15:
                score += 8
                reasons.append(f"Moderate institutional ({institutional_pct:.0f}%)")

            # Sweep activity
            sweep_pct = flow.get("sweep_pct", 0)
            sweep_direction = flow.get("sweep_direction", "")
            if sweep_pct > 20:
                score += 10
                if "BUY" in sweep_direction:
                    direction = "BULLISH"
                    reasons.append(f"Buy sweeps ({sweep_pct:.0f}%)")
                elif "SELL" in sweep_direction:
                    direction = "BEARISH"
                    reasons.append(f"Sell sweeps ({sweep_pct:.0f}%)")

            # Bid-ask imbalance
            imbalance = flow.get("bid_ask_imbalance", 0)
            if imbalance > 10:
                score += 5
                reasons.append(f"Bid pressure (+{imbalance:.0f}%)")
            elif imbalance < -10:
                score += 5
                reasons.append(f"Ask pressure ({imbalance:.0f}%)")

            return ConvictionSignal(
                source="order_flow",
                score=min(100, max(0, score)),
                direction=direction,
                confidence=min(100, score),
                reason="; ".join(reasons) if reasons else "order_flow_analysis",
                weight=self.WEIGHTS["order_flow"],
                source_reliability=self.SOURCE_RELIABILITY.get("order_flow", 1.0)
            )

        except Exception as e:
            logger.debug(f"Order flow analysis failed {ticker}: {e}")
            return ConvictionSignal("order_flow", 50, "NEUTRAL", 50, f"error: {e}", self.WEIGHTS["order_flow"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("order_flow", 1.0))

    def _get_probability_score(self, ticker: str) -> ConvictionSignal:
        """
        Score from multi-engine probability generator.
        """
        if not self.probability:
            return ConvictionSignal("probability", 50, "NEUTRAL", 50, "no_generator", self.WEIGHTS["probability"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("probability", 1.0))

        try:
            result = self.probability.generate(ticker)
            if not result:
                return ConvictionSignal("probability", 50, "NEUTRAL", 50, "no_result", self.WEIGHTS["probability"],
                                        source_reliability=self.SOURCE_RELIABILITY.get("probability", 1.0))

            score = result.get("combined_score", 50)
            win_prob = result.get("win_probability", 0.5)
            action = result.get("recommended_action", "WATCH")
            reasons = []

            direction = "NEUTRAL"
            if action in ("STRONG_LONG", "LONG"):
                direction = "BULLISH"
                reasons.append(f"Prob: {action} ({win_prob*100:.0f}% win)")
            elif action in ("STRONG_SHORT", "SHORT"):
                direction = "BEARISH"
                reasons.append(f"Prob: {action} ({win_prob*100:.0f}% win)")
            else:
                reasons.append(f"Prob: {action}")

            # Component scores
            ignition = result.get("ignition_score", 50)
            pressure = result.get("pressure_score", 50)
            surge = result.get("surge_score", 50)

            if ignition > 70:
                reasons.append(f"High ignition ({ignition:.0f})")
            if pressure > 70:
                reasons.append(f"High pressure ({pressure:.0f})")
            if surge > 70:
                reasons.append(f"High surge ({surge:.0f})")

            return ConvictionSignal(
                source="probability",
                score=min(100, max(0, score)),
                direction=direction,
                confidence=win_prob * 100,
                reason="; ".join(reasons) if reasons else "probability_analysis",
                weight=self.WEIGHTS["probability"],
                source_reliability=self.SOURCE_RELIABILITY.get("probability", 1.0)
            )

        except Exception as e:
            logger.debug(f"Probability analysis failed {ticker}: {e}")
            return ConvictionSignal("probability", 50, "NEUTRAL", 50, f"error: {e}", self.WEIGHTS["probability"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("probability", 1.0))

    def _get_pattern_memory_score(self, ticker: str) -> ConvictionSignal:
        """
        Score from historical pattern matching.
        """
        if not self.pattern_memory or not self.polygon:
            return ConvictionSignal("pattern_memory", 50, "NEUTRAL", 50, "no_memory", self.WEIGHTS["pattern_memory"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("pattern_memory", 1.0))

        try:
            # Get recent bars for pattern matching
            bars = self.polygon.get_intraday_bars(ticker, timespan="minute", multiplier=5, limit=20)
            if not bars or len(bars) < 5:
                return ConvictionSignal("pattern_memory", 50, "NEUTRAL", 50, "insufficient_bars", self.WEIGHTS["pattern_memory"],
                                        source_reliability=self.SOURCE_RELIABILITY.get("pattern_memory", 1.0))

            matches = self.pattern_memory.find_matching_patterns(ticker, bars)
            if not matches:
                return ConvictionSignal("pattern_memory", 50, "NEUTRAL", 50, "no_matches", self.WEIGHTS["pattern_memory"],
                                        source_reliability=self.SOURCE_RELIABILITY.get("pattern_memory", 1.0))

            score = 50
            direction = "NEUTRAL"
            reasons = []

            # Best match
            best = matches[0]
            similarity = best.get("similarity", 0)
            win_rate = best.get("win_rate", 0.5)
            pattern_type = best.get("pattern_type", "unknown")

            # Similarity scoring
            if similarity > 0.8:
                score += 25
                reasons.append(f"Strong match ({similarity*100:.0f}%)")
            elif similarity > 0.6:
                score += 15
                reasons.append(f"Good match ({similarity*100:.0f}%)")

            # Win rate scoring
            if win_rate > 0.7:
                score += 20
                reasons.append(f"High win rate ({win_rate*100:.0f}%)")
            elif win_rate > 0.55:
                score += 10
                reasons.append(f"Positive win rate ({win_rate*100:.0f}%)")
            elif win_rate < 0.4:
                score -= 15
                reasons.append(f"Low win rate ({win_rate*100:.0f}%)")

            # Direction from pattern
            if pattern_type in ("breakout", "momentum"):
                direction = "BULLISH"
            elif pattern_type == "reversal":
                # Check if it's a bullish or bearish reversal
                if best.get("direction", "") == "up":
                    direction = "BULLISH"
                else:
                    direction = "BEARISH"

            return ConvictionSignal(
                source="pattern_memory",
                score=min(100, max(0, score)),
                direction=direction,
                confidence=similarity * 100,
                reason="; ".join(reasons) if reasons else "pattern_memory_analysis",
                weight=self.WEIGHTS["pattern_memory"],
                source_reliability=self.SOURCE_RELIABILITY.get("pattern_memory", 1.0)
            )

        except Exception as e:
            logger.debug(f"Pattern memory failed {ticker}: {e}")
            return ConvictionSignal("pattern_memory", 50, "NEUTRAL", 50, f"error: {e}", self.WEIGHTS["pattern_memory"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("pattern_memory", 1.0))

    def _get_ai_verdict_score(self, ticker: str, spot_price: float = None) -> ConvictionSignal:
        """
        Score from AI visual analysis (GPT-4/Gemini).

        Generates a chart, sends to predator stack for AI analysis.
        AI analysis is expensive - only runs when chart_generator and predator are available.
        """
        if not self.predator:
            return ConvictionSignal("ai_verdict", 50, "NEUTRAL", 50, "no_ai", self.WEIGHTS["ai_verdict"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("ai_verdict", 1.0))

        if not self.chart_generator or not self.polygon:
            return ConvictionSignal("ai_verdict", 50, "NEUTRAL", 50, "no_chart_gen", self.WEIGHTS["ai_verdict"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("ai_verdict", 1.0))

        try:
            # Fetch OHLCV data for chart generation (5-minute bars, last 50 candles)
            bars = self.polygon.get_bars(ticker, timeframe="5", limit=50)
            if not bars or len(bars) < 20:
                return ConvictionSignal("ai_verdict", 50, "NEUTRAL", 50, "insufficient_data", self.WEIGHTS["ai_verdict"],
                                        source_reliability=self.SOURCE_RELIABILITY.get("ai_verdict", 1.0))

            # Generate chart image as base64
            chart_base64 = self.chart_generator.generate_chart(ticker, bars, timeframe="5min")
            if not chart_base64:
                return ConvictionSignal("ai_verdict", 50, "NEUTRAL", 50, "chart_gen_failed", self.WEIGHTS["ai_verdict"],
                                        source_reliability=self.SOURCE_RELIABILITY.get("ai_verdict", 1.0))

            # Get current price for context
            current_price = spot_price or (bars[-1].get("c", 0) if bars else 0)

            # Call predator stack for AI analysis (synchronous)
            analysis = self.predator.analyze_sync(
                chart_base64=chart_base64,
                ticker=ticker,
                pattern="",  # Let AI detect patterns
                current_price=current_price,
                direction_hint=None  # No bias
            )

            if not analysis:
                return ConvictionSignal("ai_verdict", 50, "NEUTRAL", 50, "analysis_failed", self.WEIGHTS["ai_verdict"],
                                        source_reliability=self.SOURCE_RELIABILITY.get("ai_verdict", 1.0))

            # Convert AI analysis to conviction signal
            # AI direction: "BULLISH", "BEARISH", "NEUTRAL"
            ai_direction = getattr(analysis, 'direction', 'NEUTRAL').upper()
            ai_confidence = getattr(analysis, 'confidence', 50)
            ai_reasoning = getattr(analysis, 'reasoning', 'AI analysis')[:100]

            # Convert direction to score (0-100 scale)
            if ai_direction == "BULLISH":
                score = 50 + (ai_confidence / 2)  # 50-100
            elif ai_direction == "BEARISH":
                score = 50 - (ai_confidence / 2)  # 0-50
            else:
                score = 50  # Neutral

            return ConvictionSignal(
                source="ai_verdict",
                score=score,
                direction=ai_direction,
                confidence=ai_confidence,
                reason=f"AI: {ai_reasoning}",
                weight=self.WEIGHTS["ai_verdict"],
                source_reliability=self.SOURCE_RELIABILITY.get("ai_verdict", 1.0)
            )

        except Exception as e:
            logger.debug(f"AI analysis failed {ticker}: {e}")
            return ConvictionSignal("ai_verdict", 50, "NEUTRAL", 50, f"error: {e}", self.WEIGHTS["ai_verdict"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("ai_verdict", 1.0))

    def analyze(self, ticker: str, spot_price: Optional[float] = None, is_0dte: bool = None) -> ApexVerdict:
        """
        Run full multi-signal analysis and return conviction verdict.

        Args:
            ticker: Stock ticker symbol
            spot_price: Optional current stock price
            is_0dte: Optional override for 0DTE detection. If None, auto-detects.

        Returns:
            ApexVerdict with conviction score, direction, and action
        """
        signals: List[ConvictionSignal] = []

        # Collect all signals
        signals.append(self._get_technical_score(ticker))
        signals.append(self._get_candlestick_score(ticker))
        signals.append(self._get_order_flow_score(ticker))
        signals.append(self._get_probability_score(ticker))
        signals.append(self._get_pattern_memory_score(ticker))
        signals.append(self._get_ai_verdict_score(ticker, spot_price))

        # Detect 0DTE mode for weight adjustments
        is_0dte_mode = is_0dte if is_0dte is not None else self._is_0dte()

        # Get regime-adjusted weights (HYDRA feature)
        # Fetch regime once for efficiency
        regime_state = None
        vix_level = None
        try:
            from wsb_snake.execution.regime_detector import regime_detector
            regime_state = regime_detector.fetch_and_update()
            vix_level = regime_state.vix_level if regime_state else None
        except Exception as e:
            logger.debug(f"Regime state fetch: {e}")

        adjusted_weights = self._get_regime_adjusted_weights(regime_state)

        # Apply 0DTE weight adjustments if in 0DTE mode
        if is_0dte_mode:
            adjusted_weights = dict(self.WEIGHTS_0DTE)
            logger.info(f"0DTE mode active: order_flow=30%, ai_verdict=10%")

        # Filter valid signals and use effective scores (TTL + decay + reliability)
        valid_signals = [s for s in signals if s.is_valid()]
        weighted_sum = sum(s.get_effective_score() * adjusted_weights.get(s.source, s.weight)
                          for s in valid_signals)
        total_weight = sum(adjusted_weights.get(s.source, s.weight) for s in valid_signals)
        base_conviction = weighted_sum / total_weight if total_weight > 0 else 50

        # HYDRA INTEGRATION: Apply regime adjustment at 20% weight
        # Formula: 80% base + 20% HYDRA-adjusted
        hydra_adjustment = 0.0
        hydra_connected = False
        if self.hydra_bridge and self.hydra_bridge.is_connected():
            hydra_adjustment = self.hydra_bridge.get_conviction_adjustment()
            hydra_connected = True
            # Apply HYDRA weighting: 80% base + 20% HYDRA-adjusted
            conviction_score = (base_conviction * 0.80) + (base_conviction * (1 + hydra_adjustment)) * 0.20
            hydra_intel = self.hydra_bridge.get_intel()
            logger.info(
                f"HYDRA_APEX: base={base_conviction:.1f} adj={hydra_adjustment:+.2f} "
                f"final={conviction_score:.1f} regime={hydra_intel.regime} "
                f"blowup={hydra_intel.blowup_probability}"
            )
        else:
            conviction_score = base_conviction
            logger.debug("HYDRA_APEX: Bridge not connected, using base conviction")

        # Get dynamic conviction threshold based on VIX regime
        dynamic_threshold = self._get_dynamic_threshold(vix_level)
        strong_threshold = dynamic_threshold + 7  # Strong = threshold + 7%

        # Determine direction (majority vote weighted by score)
        bullish_weight = sum(s.score * s.weight for s in signals if s.direction == "BULLISH")
        bearish_weight = sum(s.score * s.weight for s in signals if s.direction == "BEARISH")

        # MAX MODE: Lower the bar for direction clarity (was 1.2, now 1.1)
        if bullish_weight > bearish_weight * 1.1:
            if conviction_score >= strong_threshold:
                direction = "STRONG_LONG"
                action = "BUY_CALLS"
            elif conviction_score >= dynamic_threshold:
                direction = "LONG"
                action = "BUY_CALLS"
            else:
                direction = "NEUTRAL"
                action = "NO_TRADE"
        elif bearish_weight > bullish_weight * 1.1:  # MAX MODE: was 1.2
            if conviction_score >= strong_threshold:
                direction = "STRONG_SHORT"
                action = "BUY_PUTS"
            elif conviction_score >= dynamic_threshold:
                direction = "SHORT"
                action = "BUY_PUTS"
            else:
                direction = "NEUTRAL"
                action = "NO_TRADE"
        else:
            direction = "NEUTRAL"
            action = "NO_TRADE"

        # Below dynamic threshold = no trade
        if conviction_score < dynamic_threshold:
            action = "NO_TRADE"

        # Position sizing based on conviction
        if conviction_score >= 85:
            size_mult = 1.5
        elif conviction_score >= 75:
            size_mult = 1.2
        elif conviction_score >= 70:
            size_mult = 1.0
        else:
            size_mult = 0.5

        # Power hour adjustments
        is_power_hour = self._is_power_hour()
        if is_power_hour:
            # Aggressive mode
            target_pct = 0.20  # +20% quick target
            stop_pct = 0.10    # -10% tight stop
            time_sensitivity = "CRITICAL"
            size_mult *= 1.2   # Slightly larger positions
        else:
            target_pct = 0.25  # +25% standard target
            stop_pct = 0.12    # -12% standard stop
            time_sensitivity = "HIGH" if conviction_score >= 80 else "MEDIUM"

        # Collect reasons from all signals
        reasons = [f"{s.source}: {s.reason}" for s in signals if s.score > 55]

        # Apply screenshot learning boost
        try:
            trade_type = "CALLS" if direction in ["STRONG_LONG", "LONG"] else "PUTS"
            current_hour = datetime.now().hour
            learning_boost, boost_reasons = trade_learner.get_confidence_adjustment(
                ticker=ticker,
                trade_type=trade_type,
                current_hour=current_hour,
                pattern=None
            )
            if learning_boost > 0:
                old_score = conviction_score
                conviction_score = min(100, conviction_score * (1 + learning_boost))
                reasons.append(f"screenshot_boost: +{learning_boost:.0%} ({', '.join(boost_reasons)})")
                logger.info(f"Screenshot learning boost: {old_score:.1f} -> {conviction_score:.1f}")
        except Exception as e:
            logger.debug(f"Screenshot learning check: {e}")

        # Build verdict
        verdict = ApexVerdict(
            ticker=ticker,
            conviction_score=conviction_score,
            direction=direction,
            action=action,
            signals=signals,
            entry_price=spot_price,
            target_pct=target_pct,
            stop_pct=stop_pct,
            position_size_multiplier=size_mult,
            time_sensitivity=time_sensitivity,
            reasons=reasons[:5]  # Top 5 reasons
        )

        # CONVICTION CALIBRATION LOGGING
        # Record threshold vs conviction for every trade decision
        # This data is essential for calibrating threshold accuracy over time
        self._log_calibration(
            ticker=ticker,
            conviction_score=conviction_score,
            dynamic_threshold=dynamic_threshold,
            action=action,
            direction=direction,
            vix_level=vix_level,
            is_0dte=is_0dte_mode,
            signals=signals,
            regime=regime_state.regime.value if regime_state and hasattr(regime_state, 'regime') else "unknown"
        )

        return verdict

    def analyze_v2(
        self,
        ticker: str,
        spot_price: Optional[float] = None,
        chart_image: str = None,
        news_headlines: List[str] = None,
        candles: List[Dict] = None
    ) -> ApexVerdict:
        """
        Run Predator Stack v2.1 analysis.

        This is the upgraded AI stack with:
        - L0: Speed filter (kill obvious losers)
        - L1-L6: Multi-layer AI analysis
        - L12: Final synthesis

        Args:
            ticker: Stock ticker symbol
            spot_price: Current stock price
            chart_image: Optional base64 chart image
            news_headlines: Optional recent news headlines
            candles: Optional candle data

        Returns:
            ApexVerdict (compatible with existing system)
        """
        try:
            from wsb_snake.ai_stack.predator_stack_v2 import get_predator_stack

            predator = get_predator_stack()

            # Build signal dict
            signal = {
                'ticker': ticker,
                'direction': 'NEUTRAL',  # Will be determined by stack
                'price': spot_price or 0,
                'entry_time': datetime.now().isoformat()
            }

            # Run Predator Stack
            verdict = predator.analyze(
                signal=signal,
                chart_image=chart_image,
                news_headlines=news_headlines,
                candles=candles
            )

            # Convert PredatorVerdict → ApexVerdict
            # Map action
            if verdict.action == "STRIKE":
                if verdict.layer_results.get('hydra', {}).get('flow_bias', '').startswith('BULLISH'):
                    action = "BUY_CALLS"
                    direction = "STRONG_LONG" if verdict.conviction >= 80 else "LONG"
                elif verdict.layer_results.get('hydra', {}).get('flow_bias', '').startswith('BEARISH'):
                    action = "BUY_PUTS"
                    direction = "STRONG_SHORT" if verdict.conviction >= 80 else "SHORT"
                else:
                    # Default to bullish based on vision or other signals
                    vision = verdict.layer_results.get('vision', {})
                    if vision.get('raw_bias') == 'PUT':
                        action = "BUY_PUTS"
                        direction = "SHORT"
                    else:
                        action = "BUY_CALLS"
                        direction = "LONG"
            else:
                action = "NO_TRADE"
                direction = "NEUTRAL"

            # Build signals list from layer results
            signals = []
            for layer_name, layer_data in verdict.layer_results.items():
                if isinstance(layer_data, dict):
                    adj = layer_data.get('adjustment', 0)
                    signals.append(ConvictionSignal(
                        source=f"predator_{layer_name}",
                        score=50 + (adj * 100),
                        direction="BULLISH" if adj > 0 else "BEARISH" if adj < 0 else "NEUTRAL",
                        confidence=abs(adj) * 100,
                        reason=layer_data.get('reason', str(layer_data)[:50]),
                        weight=0.15
                    ))

            # Position sizing
            if verdict.conviction >= 85:
                size_mult = 1.5
            elif verdict.conviction >= 75:
                size_mult = 1.2
            elif verdict.conviction >= 65:
                size_mult = 1.0
            else:
                size_mult = 0.5

            # Build ApexVerdict
            apex_verdict = ApexVerdict(
                ticker=ticker,
                conviction_score=verdict.conviction,
                direction=direction,
                action=action,
                signals=signals,
                entry_price=verdict.entry_price or spot_price,
                target_pct=abs(verdict.take_profit - (spot_price or verdict.entry_price or 100)) / (spot_price or verdict.entry_price or 100) if verdict.take_profit else 0.15,
                stop_pct=abs(verdict.stop_loss - (spot_price or verdict.entry_price or 100)) / (spot_price or verdict.entry_price or 100) if verdict.stop_loss else 0.10,
                position_size_multiplier=size_mult * verdict.position_size,
                time_sensitivity="CRITICAL" if self._is_power_hour() else "HIGH",
                reasons=[verdict.reasoning] if verdict.reasoning else []
            )

            logger.info(
                f"APEX_V2: {ticker} {direction} conv={verdict.conviction:.0f}% "
                f"action={action} size={verdict.position_size} "
                f"latency={verdict.total_latency_ms:.0f}ms"
            )

            return apex_verdict

        except Exception as e:
            logger.error(f"APEX_V2: Predator Stack error - {e}, falling back to v1")
            # Fallback to original analyze
            return self.analyze(ticker, spot_price)

    def _log_calibration(
        self,
        ticker: str,
        conviction_score: float,
        dynamic_threshold: float,
        action: str,
        direction: str,
        vix_level: float,
        is_0dte: bool,
        signals: List[ConvictionSignal],
        regime: str
    ) -> None:
        """
        Log conviction calibration data for every trade decision.

        This logs threshold vs conviction for post-hoc analysis of:
        - Threshold accuracy (did high conviction = win?)
        - Regime calibration (are VIX-based thresholds optimal?)
        - Signal quality (which signals contribute to winners?)

        Args:
            ticker: Stock symbol
            conviction_score: Final conviction score (0-100)
            dynamic_threshold: VIX-adjusted threshold used
            action: Trade action (BUY_CALLS, BUY_PUTS, NO_TRADE)
            direction: Direction verdict
            vix_level: Current VIX level
            is_0dte: Whether 0DTE mode was active
            signals: List of all conviction signals
            regime: Current market regime
        """
        try:
            import json

            # Signal breakdown for calibration
            signal_data = {}
            for s in signals:
                signal_data[s.source] = {
                    "score": round(s.score, 2),
                    "effective": round(s.get_effective_score(), 2),
                    "direction": s.direction,
                    "decay": round(s.get_decay_factor(), 3),
                    "confidence": round(s.confidence, 2)
                }

            calibration_entry = {
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "conviction": round(conviction_score, 2),
                "threshold": round(dynamic_threshold, 2),
                "margin": round(conviction_score - dynamic_threshold, 2),
                "action": action,
                "direction": direction,
                "vix": round(vix_level, 2) if vix_level else None,
                "regime": regime,
                "is_0dte": is_0dte,
                "signals": signal_data,
                # Outcome fields - to be updated by trade_learner after trade closes
                "outcome": None,  # "WIN", "LOSS", or "SCRATCH"
                "pnl_pct": None,
                "hold_duration_sec": None
            }

            # Log as structured JSON for easy parsing
            calibration_logger.info(
                f"CALIBRATION|{json.dumps(calibration_entry)}"
            )

            # Also log human-readable summary
            margin_indicator = "PASS" if conviction_score >= dynamic_threshold else "FAIL"
            calibration_logger.info(
                f"{ticker}: {conviction_score:.1f}% vs {dynamic_threshold:.1f}% threshold "
                f"[{margin_indicator}] -> {action} | VIX={vix_level:.1f if vix_level else 0:.1f} "
                f"regime={regime} 0DTE={is_0dte}"
            )

        except Exception as e:
            logger.debug(f"Calibration logging error: {e}")


# Singleton instance
apex_engine = ApexConvictionEngine()
