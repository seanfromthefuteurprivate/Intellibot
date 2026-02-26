"""
TIER 3: Specialist Agent Swarm

Dedicated AI agents for specific analysis tasks:
- PatternSpecialist: Chart pattern recognition
- FlowSpecialist: Order flow and tape reading
- NewsSpecialist: News sentiment analysis
- TechnicalSpecialist: Technical indicator analysis
- OptionsSpecialist: Options chain analysis

Each specialist runs independently and contributes to consensus.
Based on multi-agent debate research (MetaGPT, ICML'24).

Trading Personas (from virattt/ai-hedge-fund):
- MOMENTUM (Druckenmiller): Follow trends, macro alignment
- CONTRARIAN (Burry): Fade extremes, sentiment divergence
- FLOW_FOLLOWER (Ackman): Track institutional positioning
- VALUE_HUNTER (Graham): Margin of safety, fundamental analysis
"""

import threading
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from abc import ABC, abstractmethod

from wsb_snake.utils.logger import get_logger
from wsb_snake.db.database import get_connection

logger = get_logger(__name__)


class TradingPersona(Enum):
    """
    Trading personas based on legendary investors.
    Each persona provides a unique lens for analysis.
    """
    MOMENTUM = "druckenmiller"      # Follow trend + macro alignment
    CONTRARIAN = "burry"            # Fade extremes, find divergences
    FLOW_FOLLOWER = "ackman"        # Track institutional positioning
    VALUE_HUNTER = "graham"         # Margin of safety focus


@dataclass
class PersonaAnalysis:
    """Analysis from a specific trading persona."""
    persona: TradingPersona
    direction: str                  # "long", "short", "neutral"
    conviction: float               # 0-100
    key_insight: str
    supporting_factors: List[str]
    risk_warning: Optional[str] = None


@dataclass
class SpecialistVerdict:
    """Verdict from a specialist agent."""
    specialist: str
    direction: str  # "long", "short", "neutral"
    confidence: float  # 0-100
    reasoning: str
    key_signals: List[str]
    risk_factors: List[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SwarmConsensus:
    """Aggregated consensus from all specialists."""
    direction: str
    confidence: float
    unanimity: float  # 0-1, how much specialists agree
    verdicts: List[SpecialistVerdict]
    final_recommendation: str
    dissenting_opinions: List[str]
    # NEW: Persona analysis from ai-hedge-fund pattern
    persona_analyses: List[PersonaAnalysis] = field(default_factory=list)
    dominant_persona: Optional[TradingPersona] = None


class BaseSpecialist(ABC):
    """Base class for specialist agents."""

    def __init__(self, name: str):
        self.name = name
        self._lock = threading.RLock()
        self._accuracy_history: List[Tuple[bool, datetime]] = []

    @abstractmethod
    def analyze(
        self,
        ticker: str,
        context: Dict[str, Any],
    ) -> SpecialistVerdict:
        """Perform specialist analysis."""
        pass

    def record_outcome(self, was_correct: bool) -> None:
        """Record whether this specialist's call was correct."""
        with self._lock:
            self._accuracy_history.append((was_correct, datetime.now(timezone.utc)))
            # Keep last 100 outcomes
            self._accuracy_history = self._accuracy_history[-100:]

    def get_accuracy(self) -> float:
        """Get rolling accuracy."""
        with self._lock:
            if not self._accuracy_history:
                return 0.5
            correct = sum(1 for c, _ in self._accuracy_history if c)
            return correct / len(self._accuracy_history)


class PatternSpecialist(BaseSpecialist):
    """
    Specialist in chart pattern recognition.

    Analyzes:
    - Candlestick patterns
    - Price action patterns
    - Support/resistance
    - Trend structure
    """

    BULLISH_PATTERNS = [
        "hammer", "inverted_hammer", "bullish_engulfing", "morning_star",
        "three_white_soldiers", "piercing_line", "bullish_harami",
        "double_bottom", "cup_and_handle", "ascending_triangle"
    ]

    BEARISH_PATTERNS = [
        "hanging_man", "shooting_star", "bearish_engulfing", "evening_star",
        "three_black_crows", "dark_cloud_cover", "bearish_harami",
        "double_top", "head_and_shoulders", "descending_triangle"
    ]

    def __init__(self):
        super().__init__("PatternSpecialist")
        logger.info("PatternSpecialist initialized")

    def analyze(
        self,
        ticker: str,
        context: Dict[str, Any],
    ) -> SpecialistVerdict:
        """Analyze chart patterns."""
        signals = []
        risk_factors = []
        confidence = 50
        direction = "neutral"

        # Extract pattern info from context
        detected_pattern = context.get("pattern", "")
        candlestick = context.get("candlestick_pattern", "")
        bars = context.get("bars", [])

        # Check detected patterns
        if detected_pattern:
            pattern_lower = detected_pattern.lower()

            # Check if bullish or bearish
            if any(p in pattern_lower for p in ["bounce", "reclaim", "breakout", "failed_breakdown"]):
                direction = "long"
                confidence += 15
                signals.append(f"Bullish pattern: {detected_pattern}")
            elif any(p in pattern_lower for p in ["rejection", "breakdown", "failed_breakout"]):
                direction = "short"
                confidence += 15
                signals.append(f"Bearish pattern: {detected_pattern}")

        # Check candlestick patterns
        if candlestick:
            if candlestick in self.BULLISH_PATTERNS:
                if direction == "neutral":
                    direction = "long"
                if direction == "long":
                    confidence += 10
                    signals.append(f"Bullish candle: {candlestick}")
                else:
                    risk_factors.append(f"Conflicting candle: {candlestick}")
            elif candlestick in self.BEARISH_PATTERNS:
                if direction == "neutral":
                    direction = "short"
                if direction == "short":
                    confidence += 10
                    signals.append(f"Bearish candle: {candlestick}")
                else:
                    risk_factors.append(f"Conflicting candle: {candlestick}")

        # Analyze bars for trend
        if len(bars) >= 5:
            prices = [b.get('c', b.get('close', 0)) for b in bars[-5:]]
            if all(p > 0 for p in prices):
                trend = "up" if prices[-1] > prices[0] else "down"
                trend_strength = abs(prices[-1] - prices[0]) / prices[0] * 100

                if trend_strength > 0.5:
                    signals.append(f"5-bar trend: {trend} ({trend_strength:.2f}%)")
                    if trend == "up" and direction == "long":
                        confidence += 5
                    elif trend == "down" and direction == "short":
                        confidence += 5
                    elif trend == "up" and direction == "short":
                        risk_factors.append("Counter-trend trade")
                    elif trend == "down" and direction == "long":
                        risk_factors.append("Counter-trend trade")

        # Check VWAP position
        vwap = context.get("vwap", 0)
        price = context.get("price", 0)
        if vwap > 0 and price > 0:
            if price > vwap:
                signals.append("Above VWAP")
                if direction == "long":
                    confidence += 5
            else:
                signals.append("Below VWAP")
                if direction == "short":
                    confidence += 5

        confidence = max(0, min(100, confidence))

        reasoning = f"Pattern analysis for {ticker}: "
        if signals:
            reasoning += "; ".join(signals[:3])
        else:
            reasoning += "No clear pattern signals"

        return SpecialistVerdict(
            specialist=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            key_signals=signals,
            risk_factors=risk_factors,
        )


class FlowSpecialist(BaseSpecialist):
    """
    Specialist in order flow analysis.

    Analyzes:
    - Sweep direction
    - Block trades
    - Bid/ask imbalance
    - Volume profile
    """

    def __init__(self):
        super().__init__("FlowSpecialist")
        logger.info("FlowSpecialist initialized")

    def analyze(
        self,
        ticker: str,
        context: Dict[str, Any],
    ) -> SpecialistVerdict:
        """Analyze order flow."""
        signals = []
        risk_factors = []
        confidence = 50
        direction = "neutral"

        # Get flow data from context
        sweep_direction = context.get("sweep_direction", "NONE")
        sweep_pct = context.get("sweep_pct", 0)
        block_count = context.get("block_count", 0)
        bid_ask_imbalance = context.get("bid_ask_imbalance", 0)
        volume_ratio = context.get("volume_ratio", 1.0)

        # Analyze sweep direction
        if sweep_direction == "BUY" and sweep_pct > 5:
            direction = "long"
            confidence += 15 if sweep_pct > 10 else 10
            signals.append(f"Buy sweeps: {sweep_pct:.0f}% of volume")
        elif sweep_direction == "SELL" and sweep_pct > 5:
            direction = "short"
            confidence += 15 if sweep_pct > 10 else 10
            signals.append(f"Sell sweeps: {sweep_pct:.0f}% of volume")

        # Block trades indicate institutional activity
        if block_count > 5:
            confidence += 10
            signals.append(f"Block trades: {block_count} detected")
        elif block_count > 2:
            confidence += 5
            signals.append(f"Some blocks: {block_count}")

        # Bid/ask imbalance
        if abs(bid_ask_imbalance) > 0.2:
            if bid_ask_imbalance > 0:
                if direction == "neutral":
                    direction = "long"
                signals.append(f"Bid-heavy: {bid_ask_imbalance:.2f}")
                if direction == "long":
                    confidence += 5
            else:
                if direction == "neutral":
                    direction = "short"
                signals.append(f"Ask-heavy: {bid_ask_imbalance:.2f}")
                if direction == "short":
                    confidence += 5

        # Volume confirmation
        if volume_ratio > 2.0:
            confidence += 10
            signals.append(f"Volume surge: {volume_ratio:.1f}x")
        elif volume_ratio > 1.5:
            confidence += 5
            signals.append(f"Elevated volume: {volume_ratio:.1f}x")
        elif volume_ratio < 0.8:
            confidence -= 10
            risk_factors.append(f"Low volume: {volume_ratio:.1f}x")

        confidence = max(0, min(100, confidence))

        reasoning = f"Flow analysis for {ticker}: "
        if signals:
            reasoning += "; ".join(signals[:3])
        else:
            reasoning += "No significant flow signals"

        return SpecialistVerdict(
            specialist=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            key_signals=signals,
            risk_factors=risk_factors,
        )


class TechnicalSpecialist(BaseSpecialist):
    """
    Specialist in technical indicator analysis.

    Analyzes:
    - RSI
    - MACD
    - Moving averages
    - Momentum
    """

    def __init__(self):
        super().__init__("TechnicalSpecialist")
        logger.info("TechnicalSpecialist initialized")

    def analyze(
        self,
        ticker: str,
        context: Dict[str, Any],
    ) -> SpecialistVerdict:
        """Analyze technical indicators."""
        signals = []
        risk_factors = []
        confidence = 50
        direction = "neutral"

        # Get technical data
        rsi = context.get("rsi", 50)
        momentum = context.get("momentum", 0)
        ma_cross = context.get("ma_cross", "neutral")
        macd_signal = context.get("macd_signal", "neutral")

        # RSI analysis
        if rsi < 30:
            signals.append(f"RSI oversold: {rsi:.0f}")
            if direction != "short":
                direction = "long"
                confidence += 10
        elif rsi > 70:
            signals.append(f"RSI overbought: {rsi:.0f}")
            if direction != "long":
                direction = "short"
                confidence += 10
        elif 30 <= rsi <= 40:
            signals.append(f"RSI approaching oversold: {rsi:.0f}")
            confidence += 5
        elif 60 <= rsi <= 70:
            signals.append(f"RSI approaching overbought: {rsi:.0f}")
            confidence += 5

        # Momentum analysis
        if momentum > 0.3:
            signals.append(f"Strong momentum: +{momentum:.2f}%")
            if direction == "neutral":
                direction = "long"
            if direction == "long":
                confidence += 10
            else:
                risk_factors.append("Counter-momentum trade")
        elif momentum < -0.3:
            signals.append(f"Negative momentum: {momentum:.2f}%")
            if direction == "neutral":
                direction = "short"
            if direction == "short":
                confidence += 10
            else:
                risk_factors.append("Counter-momentum trade")

        # MA cross
        if ma_cross == "bullish":
            signals.append("Bullish MA cross")
            if direction == "long":
                confidence += 8
            elif direction == "neutral":
                direction = "long"
        elif ma_cross == "bearish":
            signals.append("Bearish MA cross")
            if direction == "short":
                confidence += 8
            elif direction == "neutral":
                direction = "short"

        # MACD
        if macd_signal == "bullish":
            signals.append("MACD bullish")
            if direction == "long":
                confidence += 5
        elif macd_signal == "bearish":
            signals.append("MACD bearish")
            if direction == "short":
                confidence += 5

        confidence = max(0, min(100, confidence))

        reasoning = f"Technical analysis for {ticker}: "
        if signals:
            reasoning += "; ".join(signals[:3])
        else:
            reasoning += "Neutral technical signals"

        return SpecialistVerdict(
            specialist=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            key_signals=signals,
            risk_factors=risk_factors,
        )


class NewsSpecialist(BaseSpecialist):
    """
    Specialist in news sentiment analysis.

    Analyzes:
    - Recent headlines
    - Sentiment scores
    - Event catalysts
    """

    def __init__(self):
        super().__init__("NewsSpecialist")
        logger.info("NewsSpecialist initialized")

    def analyze(
        self,
        ticker: str,
        context: Dict[str, Any],
    ) -> SpecialistVerdict:
        """Analyze news sentiment."""
        signals = []
        risk_factors = []
        confidence = 50
        direction = "neutral"

        # Get news data
        sentiment_score = context.get("news_sentiment", 0)
        headline_count = context.get("headline_count", 0)
        catalyst = context.get("catalyst", "")
        earnings_soon = context.get("earnings_soon", False)
        analyst_rating = context.get("analyst_rating", "neutral")

        # Sentiment analysis
        if sentiment_score > 0.3:
            direction = "long"
            confidence += 15 if sentiment_score > 0.5 else 10
            signals.append(f"Positive sentiment: {sentiment_score:.2f}")
        elif sentiment_score < -0.3:
            direction = "short"
            confidence += 15 if sentiment_score < -0.5 else 10
            signals.append(f"Negative sentiment: {sentiment_score:.2f}")

        # Headline volume
        if headline_count > 10:
            signals.append(f"High news volume: {headline_count} headlines")
            confidence += 5
        elif headline_count > 5:
            signals.append(f"Elevated coverage: {headline_count} headlines")

        # Catalyst
        if catalyst:
            signals.append(f"Catalyst: {catalyst}")
            confidence += 10

        # Earnings risk
        if earnings_soon:
            risk_factors.append("Earnings within 2 days - IV crush risk")
            confidence -= 10

        # Analyst rating
        if analyst_rating == "upgrade":
            signals.append("Recent analyst upgrade")
            if direction == "long":
                confidence += 8
        elif analyst_rating == "downgrade":
            signals.append("Recent analyst downgrade")
            if direction == "short":
                confidence += 8
            else:
                risk_factors.append("Analyst downgrade")

        confidence = max(0, min(100, confidence))

        reasoning = f"News analysis for {ticker}: "
        if signals:
            reasoning += "; ".join(signals[:3])
        else:
            reasoning += "No significant news catalysts"

        return SpecialistVerdict(
            specialist=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            key_signals=signals,
            risk_factors=risk_factors,
        )


class OptionsSpecialist(BaseSpecialist):
    """
    Specialist in options chain analysis.

    Analyzes:
    - Put/call ratios
    - GEX regime
    - Max pain
    - Options walls
    """

    def __init__(self):
        super().__init__("OptionsSpecialist")
        logger.info("OptionsSpecialist initialized")

    def analyze(
        self,
        ticker: str,
        context: Dict[str, Any],
    ) -> SpecialistVerdict:
        """Analyze options chain."""
        signals = []
        risk_factors = []
        confidence = 50
        direction = "neutral"

        # Get options data
        put_call_ratio = context.get("put_call_ratio", 1.0)
        gex_regime = context.get("gex_regime", "neutral")
        max_pain = context.get("max_pain", 0)
        current_price = context.get("price", 0)
        call_wall = context.get("call_wall", 0)
        put_wall = context.get("put_wall", 0)

        # Put/call ratio
        if put_call_ratio > 1.5:
            signals.append(f"High P/C ratio: {put_call_ratio:.2f} (fear)")
            direction = "long"  # Contrarian - fear = potential bounce
            confidence += 10
        elif put_call_ratio < 0.7:
            signals.append(f"Low P/C ratio: {put_call_ratio:.2f} (complacency)")
            direction = "short"  # Contrarian - complacency = potential drop
            confidence += 10

        # GEX regime
        if gex_regime in ["positive", "bullish"]:
            signals.append(f"GEX regime: {gex_regime}")
            if direction == "long":
                confidence += 10
            elif direction == "neutral":
                direction = "long"
        elif gex_regime in ["negative", "bearish"]:
            signals.append(f"GEX regime: {gex_regime}")
            if direction == "short":
                confidence += 10
            elif direction == "neutral":
                direction = "short"

        # Max pain analysis
        if max_pain > 0 and current_price > 0:
            distance_to_max_pain = (current_price - max_pain) / max_pain * 100
            if abs(distance_to_max_pain) > 2:
                signals.append(f"Max pain ${max_pain:.0f} ({distance_to_max_pain:+.1f}% away)")
                # Price tends to gravitate to max pain on expiry
                if distance_to_max_pain > 2:
                    # Price above max pain - potential pull down
                    if direction == "short":
                        confidence += 5
                elif distance_to_max_pain < -2:
                    # Price below max pain - potential pull up
                    if direction == "long":
                        confidence += 5

        # Options walls
        if call_wall > 0 and current_price > 0:
            if current_price > call_wall * 0.98:
                risk_factors.append(f"Near call wall ${call_wall:.0f}")
                if direction == "long":
                    confidence -= 5
        if put_wall > 0 and current_price > 0:
            if current_price < put_wall * 1.02:
                risk_factors.append(f"Near put wall ${put_wall:.0f}")
                if direction == "short":
                    confidence -= 5

        confidence = max(0, min(100, confidence))

        reasoning = f"Options analysis for {ticker}: "
        if signals:
            reasoning += "; ".join(signals[:3])
        else:
            reasoning += "Neutral options positioning"

        return SpecialistVerdict(
            specialist=self.name,
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            key_signals=signals,
            risk_factors=risk_factors,
        )


class SpecialistSwarm:
    """
    TIER 3: Coordinates multiple specialist agents for consensus.

    Each specialist analyzes independently, then verdicts are aggregated
    with weighting based on recent accuracy.
    """

    def __init__(self, enable_personas: bool = True):
        self._lock = threading.RLock()
        self._specialists: Dict[str, BaseSpecialist] = {
            "pattern": PatternSpecialist(),
            "flow": FlowSpecialist(),
            "technical": TechnicalSpecialist(),
            "news": NewsSpecialist(),
            "options": OptionsSpecialist(),
        }
        self._enable_personas = enable_personas
        self._persona_stats: Dict[str, Dict] = {
            p.value: {"wins": 0, "losses": 0, "accuracy": 0.5}
            for p in TradingPersona
        }
        self._init_db()
        logger.info(f"TIER_3: Specialist swarm initialized with {len(self._specialists)} agents (personas={'ON' if enable_personas else 'OFF'})")

    def _init_db(self) -> None:
        """Initialize tracking database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS specialist_verdicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    ticker TEXT,
                    specialist TEXT,
                    direction TEXT,
                    confidence REAL,
                    was_correct INTEGER
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"TIER_3: Swarm DB init failed - {e}")

    def _analyze_momentum_persona(
        self,
        ticker: str,
        context: Dict[str, Any],
        verdicts: List[SpecialistVerdict],
    ) -> PersonaAnalysis:
        """
        Druckenmiller-style momentum analysis.

        Focus: Follow the trend, align with macro, size up on conviction.
        """
        direction = "neutral"
        conviction = 50
        factors = []
        risk = None

        # Check momentum indicators
        momentum = context.get("momentum", 0)
        trend = context.get("trend", "")
        hydra_direction = context.get("hydra_direction", "")

        if momentum > 0.2:
            direction = "long"
            conviction += 15
            factors.append(f"Positive momentum ({momentum:.2f}%)")
        elif momentum < -0.2:
            direction = "short"
            conviction += 15
            factors.append(f"Negative momentum ({momentum:.2f}%)")

        # Macro alignment (HYDRA)
        if hydra_direction == "BULLISH" and direction == "long":
            conviction += 20
            factors.append("Macro confirms (HYDRA BULLISH)")
        elif hydra_direction == "BEARISH" and direction == "short":
            conviction += 20
            factors.append("Macro confirms (HYDRA BEARISH)")
        elif hydra_direction in ["BULLISH", "BEARISH"] and direction == "neutral":
            direction = "long" if hydra_direction == "BULLISH" else "short"
            conviction += 10
            factors.append(f"Following macro: {hydra_direction}")

        # Volume confirmation
        volume_ratio = context.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            conviction += 10
            factors.append(f"Strong volume ({volume_ratio:.1f}x)")

        # Risk: counter-trend
        if trend and direction:
            if (trend == "up" and direction == "short") or (trend == "down" and direction == "long"):
                risk = "Counter-trend trade - Druckenmiller would reduce size"
                conviction -= 15

        return PersonaAnalysis(
            persona=TradingPersona.MOMENTUM,
            direction=direction,
            conviction=min(100, max(0, conviction)),
            key_insight="Follow the money, size up when conviction is high",
            supporting_factors=factors,
            risk_warning=risk,
        )

    def _analyze_contrarian_persona(
        self,
        ticker: str,
        context: Dict[str, Any],
        verdicts: List[SpecialistVerdict],
    ) -> PersonaAnalysis:
        """
        Burry-style contrarian analysis.

        Focus: Find extremes, sentiment divergences, crowded trades.
        """
        direction = "neutral"
        conviction = 50
        factors = []
        risk = None

        # Check for extremes
        rsi = context.get("rsi", 50)
        put_call_ratio = context.get("put_call_ratio", 1.0)
        sentiment = context.get("news_sentiment", 0)

        # RSI extremes (contrarian signals)
        if rsi > 75:
            direction = "short"
            conviction += 20
            factors.append(f"RSI extreme overbought ({rsi:.0f}) - fade the crowd")
        elif rsi < 25:
            direction = "long"
            conviction += 20
            factors.append(f"RSI extreme oversold ({rsi:.0f}) - buy the fear")

        # Put/call extremes (contrarian)
        if put_call_ratio > 1.8:
            # Extreme fear = contrarian long
            if direction != "short":
                direction = "long"
                conviction += 15
                factors.append(f"Extreme P/C ratio ({put_call_ratio:.2f}) - peak fear")
        elif put_call_ratio < 0.5:
            # Extreme greed = contrarian short
            if direction != "long":
                direction = "short"
                conviction += 15
                factors.append(f"Extreme low P/C ({put_call_ratio:.2f}) - peak greed")

        # Sentiment extremes
        if sentiment > 0.7:
            if direction != "long":
                direction = "short"
                conviction += 10
                factors.append("Extreme positive sentiment - contrarian short")
        elif sentiment < -0.7:
            if direction != "short":
                direction = "long"
                conviction += 10
                factors.append("Extreme negative sentiment - contrarian long")

        # Risk: fighting strong trends
        momentum = context.get("momentum", 0)
        if abs(momentum) > 0.5:
            risk = f"Strong momentum ({momentum:.2f}%) - contrarian positions are risky"
            conviction -= 10

        return PersonaAnalysis(
            persona=TradingPersona.CONTRARIAN,
            direction=direction,
            conviction=min(100, max(0, conviction)),
            key_insight="When everyone's running one way, look the other way",
            supporting_factors=factors,
            risk_warning=risk,
        )

    def _analyze_flow_persona(
        self,
        ticker: str,
        context: Dict[str, Any],
        verdicts: List[SpecialistVerdict],
    ) -> PersonaAnalysis:
        """
        Ackman-style flow analysis.

        Focus: Track institutional positioning, follow smart money.
        """
        direction = "neutral"
        conviction = 50
        factors = []
        risk = None

        # Flow data
        sweep_direction = context.get("sweep_direction", "NONE")
        block_count = context.get("block_count", 0)
        flow_bias = context.get("flow_bias", "NEUTRAL")
        dp_support = context.get("dp_nearest_support", 0)
        dp_resistance = context.get("dp_nearest_resistance", 0)

        # Sweep direction (institutional aggression)
        if sweep_direction == "BUY":
            direction = "long"
            conviction += 20
            factors.append("Buy sweeps detected - institutions accumulating")
        elif sweep_direction == "SELL":
            direction = "short"
            conviction += 20
            factors.append("Sell sweeps detected - institutions distributing")

        # Block trades
        if block_count > 5:
            conviction += 15
            factors.append(f"High block activity ({block_count}) - big players active")

        # Flow bias from HYDRA
        if flow_bias in ["BULLISH", "AGGRESSIVELY_BULLISH"]:
            if direction != "short":
                direction = "long"
                conviction += 10
                factors.append(f"Institutional flow bias: {flow_bias}")
        elif flow_bias in ["BEARISH", "AGGRESSIVELY_BEARISH"]:
            if direction != "long":
                direction = "short"
                conviction += 10
                factors.append(f"Institutional flow bias: {flow_bias}")

        # Dark pool levels
        price = context.get("price", 0)
        if dp_support > 0 and price > 0:
            if abs(price - dp_support) / price < 0.005:  # Within 0.5%
                if direction != "short":
                    direction = "long"
                    conviction += 10
                    factors.append(f"At dark pool support ${dp_support:.0f}")

        if dp_resistance > 0 and price > 0:
            if abs(price - dp_resistance) / price < 0.005:
                risk = f"Near dark pool resistance ${dp_resistance:.0f}"

        return PersonaAnalysis(
            persona=TradingPersona.FLOW_FOLLOWER,
            direction=direction,
            conviction=min(100, max(0, conviction)),
            key_insight="Follow the smart money - they know something we don't",
            supporting_factors=factors,
            risk_warning=risk,
        )

    def _analyze_value_persona(
        self,
        ticker: str,
        context: Dict[str, Any],
        verdicts: List[SpecialistVerdict],
    ) -> PersonaAnalysis:
        """
        Graham-style value analysis.

        Focus: Margin of safety, fundamental support, risk-first thinking.
        """
        direction = "neutral"
        conviction = 50
        factors = []
        risk = None

        # Price vs support/resistance
        price = context.get("price", 0)
        vwap = context.get("vwap", 0)
        support = context.get("support", 0)
        resistance = context.get("resistance", 0)

        # Margin of safety from VWAP
        if vwap > 0 and price > 0:
            vwap_distance = (price - vwap) / vwap * 100
            if vwap_distance < -0.5:  # Below VWAP = value
                direction = "long"
                conviction += 15
                factors.append(f"Below VWAP ({vwap_distance:.2f}%) - margin of safety")
            elif vwap_distance > 0.5:  # Above VWAP = expensive
                risk = f"Trading above VWAP ({vwap_distance:.2f}%) - reduced margin of safety"

        # Near support = value
        if support > 0 and price > 0:
            support_distance = (price - support) / price * 100
            if 0 < support_distance < 1:  # Within 1% of support
                direction = "long"
                conviction += 20
                factors.append(f"Near support ${support:.0f} - defined risk")

        # Risk/reward check
        if support > 0 and resistance > 0 and price > 0:
            upside = (resistance - price) / price * 100
            downside = (price - support) / price * 100
            if downside > 0:
                rr_ratio = upside / downside
                if rr_ratio > 2:
                    conviction += 15
                    factors.append(f"Favorable R/R ({rr_ratio:.1f}:1)")
                elif rr_ratio < 1:
                    conviction -= 15
                    risk = f"Poor R/R ({rr_ratio:.1f}:1) - Graham would pass"

        # Time decay risk for 0DTE
        hours_to_expiry = context.get("hours_to_expiry", 6)
        if hours_to_expiry < 2:
            risk = f"Only {hours_to_expiry:.1f}h to expiry - time is the enemy"
            conviction -= 10

        return PersonaAnalysis(
            persona=TradingPersona.VALUE_HUNTER,
            direction=direction,
            conviction=min(100, max(0, conviction)),
            key_insight="Never lose money - margin of safety is paramount",
            supporting_factors=factors,
            risk_warning=risk,
        )

    def _run_persona_analysis(
        self,
        ticker: str,
        context: Dict[str, Any],
        verdicts: List[SpecialistVerdict],
    ) -> List[PersonaAnalysis]:
        """Run all trading personas and return their analyses."""
        personas = []

        personas.append(self._analyze_momentum_persona(ticker, context, verdicts))
        personas.append(self._analyze_contrarian_persona(ticker, context, verdicts))
        personas.append(self._analyze_flow_persona(ticker, context, verdicts))
        personas.append(self._analyze_value_persona(ticker, context, verdicts))

        return personas

    def _get_dominant_persona(
        self,
        persona_analyses: List[PersonaAnalysis],
        consensus_direction: str,
    ) -> Optional[TradingPersona]:
        """Determine which persona best supports the consensus."""
        best_persona = None
        best_conviction = 0

        for pa in persona_analyses:
            if pa.direction == consensus_direction and pa.conviction > best_conviction:
                best_conviction = pa.conviction
                best_persona = pa.persona

        return best_persona

    def analyze(
        self,
        ticker: str,
        context: Dict[str, Any],
    ) -> SwarmConsensus:
        """
        Run all specialists and aggregate consensus.

        Now also includes persona analysis from ai-hedge-fund pattern.

        Args:
            ticker: Symbol to analyze
            context: Context data for specialists

        Returns:
            SwarmConsensus with aggregated verdict and persona insights
        """
        verdicts = []

        # Run each specialist
        for name, specialist in self._specialists.items():
            try:
                verdict = specialist.analyze(ticker, context)
                verdicts.append(verdict)
                logger.debug(f"TIER_3: {name} -> {verdict.direction} ({verdict.confidence:.0f}%)")
            except Exception as e:
                logger.warning(f"TIER_3: {name} failed - {e}")

        if not verdicts:
            return SwarmConsensus(
                direction="neutral",
                confidence=50,
                unanimity=0,
                verdicts=[],
                final_recommendation="No specialist verdicts available",
                dissenting_opinions=[],
            )

        # Aggregate verdicts
        consensus = self._aggregate_verdicts(verdicts, ticker, context)

        # Log consensus with persona info
        persona_str = f", dominant_persona={consensus.dominant_persona.value}" if consensus.dominant_persona else ""
        logger.info(
            f"TIER_3: Swarm consensus for {ticker}: {consensus.direction} "
            f"({consensus.confidence:.0f}%, unanimity={consensus.unanimity:.0%}{persona_str})"
        )

        # Save verdicts
        self._save_verdicts(ticker, verdicts)

        return consensus

    def _aggregate_verdicts(
        self,
        verdicts: List[SpecialistVerdict],
        ticker: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> SwarmConsensus:
        """Aggregate specialist verdicts into consensus."""
        # Count votes by direction
        long_votes = sum(1 for v in verdicts if v.direction == "long")
        short_votes = sum(1 for v in verdicts if v.direction == "short")
        neutral_votes = sum(1 for v in verdicts if v.direction == "neutral")

        total_votes = len(verdicts)

        # Weighted confidence by accuracy
        weighted_long = sum(
            v.confidence * self._specialists[v.specialist.lower().replace("specialist", "")].get_accuracy()
            for v in verdicts if v.direction == "long"
        )
        weighted_short = sum(
            v.confidence * self._specialists[v.specialist.lower().replace("specialist", "")].get_accuracy()
            for v in verdicts if v.direction == "short"
        )

        # Determine direction
        if long_votes > short_votes and long_votes > neutral_votes:
            direction = "long"
            base_confidence = weighted_long / long_votes if long_votes > 0 else 50
        elif short_votes > long_votes and short_votes > neutral_votes:
            direction = "short"
            base_confidence = weighted_short / short_votes if short_votes > 0 else 50
        else:
            direction = "neutral"
            base_confidence = 50

        # Calculate unanimity (0 = split, 1 = unanimous)
        max_votes = max(long_votes, short_votes, neutral_votes)
        unanimity = max_votes / total_votes if total_votes > 0 else 0

        # Adjust confidence based on unanimity
        confidence = base_confidence * (0.5 + 0.5 * unanimity)

        # Collect dissenting opinions
        dissenting = []
        for v in verdicts:
            if v.direction != direction and v.confidence > 60:
                dissenting.append(f"{v.specialist}: {v.direction} ({v.confidence:.0f}%) - {v.reasoning}")

        # Generate recommendation
        if unanimity >= 0.8 and confidence >= 70:
            recommendation = f"STRONG {direction.upper()} - High consensus ({unanimity:.0%})"
        elif unanimity >= 0.6 and confidence >= 60:
            recommendation = f"{direction.upper()} - Moderate consensus ({unanimity:.0%})"
        elif len(dissenting) > 2:
            recommendation = f"CAUTION - Split opinions ({long_votes}L/{short_votes}S/{neutral_votes}N)"
        else:
            recommendation = f"WEAK {direction.upper()} - Low conviction"

        # Run persona analysis if enabled
        persona_analyses = []
        dominant_persona = None

        if self._enable_personas and ticker and context:
            try:
                persona_analyses = self._run_persona_analysis(ticker, context, verdicts)
                dominant_persona = self._get_dominant_persona(persona_analyses, direction)

                # Log persona insights
                for pa in persona_analyses:
                    if pa.direction == direction:
                        logger.debug(
                            f"TIER_3: {pa.persona.value.upper()} supports {direction} "
                            f"({pa.conviction:.0f}%): {pa.key_insight}"
                        )
            except Exception as e:
                logger.warning(f"TIER_3: Persona analysis failed - {e}")

        return SwarmConsensus(
            direction=direction,
            confidence=confidence,
            unanimity=unanimity,
            verdicts=verdicts,
            final_recommendation=recommendation,
            dissenting_opinions=dissenting,
            persona_analyses=persona_analyses,
            dominant_persona=dominant_persona,
        )

    def record_outcome(
        self,
        ticker: str,
        actual_direction: str,
    ) -> None:
        """
        Record actual outcome to update specialist accuracy.

        Args:
            ticker: Symbol
            actual_direction: What actually happened ("long" = price went up)
        """
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Get recent verdicts for this ticker
            cursor.execute("""
                SELECT id, specialist, direction FROM specialist_verdicts
                WHERE ticker = ? AND was_correct IS NULL
                ORDER BY timestamp DESC
                LIMIT 10
            """, (ticker,))

            rows = cursor.fetchall()

            for row in rows:
                verdict_id, specialist, predicted = row
                was_correct = (predicted == actual_direction)

                # Update DB
                cursor.execute("""
                    UPDATE specialist_verdicts
                    SET was_correct = ?
                    WHERE id = ?
                """, (1 if was_correct else 0, verdict_id))

                # Update specialist accuracy
                specialist_key = specialist.lower().replace("specialist", "")
                if specialist_key in self._specialists:
                    self._specialists[specialist_key].record_outcome(was_correct)

            conn.commit()
            conn.close()

            logger.info(f"TIER_3: Recorded outcome for {ticker}: {actual_direction}")

        except Exception as e:
            logger.warning(f"TIER_3: Outcome record failed - {e}")

    def _save_verdicts(
        self,
        ticker: str,
        verdicts: List[SpecialistVerdict],
    ) -> None:
        """Save verdicts to database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            for v in verdicts:
                cursor.execute("""
                    INSERT INTO specialist_verdicts
                    (timestamp, ticker, specialist, direction, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    v.timestamp.isoformat(),
                    ticker,
                    v.specialist,
                    v.direction,
                    v.confidence,
                ))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"TIER_3: Verdict save failed - {e}")

    def get_specialist_stats(self) -> Dict[str, Any]:
        """Get accuracy statistics for all specialists."""
        return {
            name: {
                "accuracy": spec.get_accuracy(),
                "sample_count": len(spec._accuracy_history),
            }
            for name, spec in self._specialists.items()
        }


# Singleton instance
_swarm: Optional[SpecialistSwarm] = None


def get_specialist_swarm() -> SpecialistSwarm:
    """Get singleton specialist swarm instance."""
    global _swarm
    if _swarm is None:
        _swarm = SpecialistSwarm()
    return _swarm
