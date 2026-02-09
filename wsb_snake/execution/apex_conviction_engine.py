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

logger = get_logger(__name__)


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
        """Linear decay from 1.0 to 0.5 over TTL period."""
        age = (datetime.now() - self.created_at).total_seconds()
        decay = 1.0 - (age / self.ttl_seconds) * 0.5
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

    # Conviction thresholds - JP MORGAN INSTITUTIONAL GRADE
    # Higher thresholds = fewer but higher quality trades
    STRONG_CONVICTION = 75  # Strong directional agreement required
    TRADE_THRESHOLD = 68    # Minimum 68% to trade (institutional standard)
    AVOID_THRESHOLD = 50    # Clear wash zone

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

        # Order Flow
        try:
            from wsb_snake.collectors.scalp_data_collector import ScalpDataCollector
            self.scalp_collector = ScalpDataCollector()
        except Exception as e:
            logger.warning(f"ScalpDataCollector not available: {e}")
            self.scalp_collector = None

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

    def _get_regime_adjusted_weights(self) -> Dict[str, float]:
        """Adjust APEX weights based on current market regime (HYDRA feature)."""
        try:
            from wsb_snake.execution.regime_detector import regime_detector, MarketRegime
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

    def _get_ai_verdict_score(self, ticker: str) -> ConvictionSignal:
        """
        Score from AI visual analysis (GPT-4/Gemini).
        """
        if not self.predator:
            return ConvictionSignal("ai_verdict", 50, "NEUTRAL", 50, "no_ai", self.WEIGHTS["ai_verdict"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("ai_verdict", 1.0))

        try:
            # AI analysis is expensive - only use for high-conviction setups
            # For now, return neutral and let other signals drive
            # TODO: Integrate predator_stack for visual chart analysis

            return ConvictionSignal(
                source="ai_verdict",
                score=50,
                direction="NEUTRAL",
                confidence=50,
                reason="ai_pending",
                weight=self.WEIGHTS["ai_verdict"],
                source_reliability=self.SOURCE_RELIABILITY.get("ai_verdict", 1.0)
            )

        except Exception as e:
            logger.debug(f"AI analysis failed {ticker}: {e}")
            return ConvictionSignal("ai_verdict", 50, "NEUTRAL", 50, f"error: {e}", self.WEIGHTS["ai_verdict"],
                                    source_reliability=self.SOURCE_RELIABILITY.get("ai_verdict", 1.0))

    def analyze(self, ticker: str, spot_price: Optional[float] = None) -> ApexVerdict:
        """
        Run full multi-signal analysis and return conviction verdict.
        """
        signals: List[ConvictionSignal] = []

        # Collect all signals
        signals.append(self._get_technical_score(ticker))
        signals.append(self._get_candlestick_score(ticker))
        signals.append(self._get_order_flow_score(ticker))
        signals.append(self._get_probability_score(ticker))
        signals.append(self._get_pattern_memory_score(ticker))
        signals.append(self._get_ai_verdict_score(ticker))

        # Get regime-adjusted weights (HYDRA feature)
        adjusted_weights = self._get_regime_adjusted_weights()

        # Filter valid signals and use effective scores (TTL + decay + reliability)
        valid_signals = [s for s in signals if s.is_valid()]
        weighted_sum = sum(s.get_effective_score() * adjusted_weights.get(s.source, s.weight)
                          for s in valid_signals)
        total_weight = sum(adjusted_weights.get(s.source, s.weight) for s in valid_signals)
        conviction_score = weighted_sum / total_weight if total_weight > 0 else 50

        # Determine direction (majority vote weighted by score)
        bullish_weight = sum(s.score * s.weight for s in signals if s.direction == "BULLISH")
        bearish_weight = sum(s.score * s.weight for s in signals if s.direction == "BEARISH")

        # MAX MODE: Lower the bar for direction clarity (was 1.2, now 1.1)
        if bullish_weight > bearish_weight * 1.1:
            if conviction_score >= self.STRONG_CONVICTION:
                direction = "STRONG_LONG"
                action = "BUY_CALLS"
            elif conviction_score >= self.TRADE_THRESHOLD:
                direction = "LONG"
                action = "BUY_CALLS"
            else:
                direction = "NEUTRAL"
                action = "NO_TRADE"
        elif bearish_weight > bullish_weight * 1.1:  # MAX MODE: was 1.2
            if conviction_score >= self.STRONG_CONVICTION:
                direction = "STRONG_SHORT"
                action = "BUY_PUTS"
            elif conviction_score >= self.TRADE_THRESHOLD:
                direction = "SHORT"
                action = "BUY_PUTS"
            else:
                direction = "NEUTRAL"
                action = "NO_TRADE"
        else:
            direction = "NEUTRAL"
            action = "NO_TRADE"

        # Below threshold = no trade
        if conviction_score < self.TRADE_THRESHOLD:
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

        return ApexVerdict(
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


# Singleton instance
apex_engine = ApexConvictionEngine()
