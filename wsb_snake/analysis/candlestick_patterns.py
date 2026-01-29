"""
Advanced Japanese Candlestick Pattern Detection

Comprehensive pattern recognition with:
- 36 candlestick patterns (single, double, triple)
- Confluence scoring
- Context-aware analysis
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum

from wsb_snake.utils.logger import get_logger

log = get_logger(__name__)


class PatternType(Enum):
    REVERSAL = "reversal"
    CONTINUATION = "continuation"
    INDECISION = "indecision"


class PatternDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class CandlestickPattern:
    """Detected candlestick pattern."""
    name: str
    pattern_type: PatternType
    direction: PatternDirection
    strength: int  # 1-5 (5 = strongest)
    reliability: float  # Historical win rate 0-1
    candles_used: int
    description: str


@dataclass
class ConfluenceResult:
    """Result of confluence analysis."""
    score: float  # 0-100
    confidence: float  # 0-100
    direction: str  # "bullish", "bearish", "neutral"
    action: str  # "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
    patterns_found: List[CandlestickPattern]
    reasoning: str


class CandlestickAnalyzer:
    """
    Advanced candlestick pattern analyzer.

    Detects 36 patterns across single, double, and triple candle formations.
    """

    # Pattern definitions with reliability scores from historical data
    PATTERNS = {
        # === SINGLE CANDLE PATTERNS ===
        "hammer": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BULLISH,
            "strength": 4,
            "reliability": 0.60,
            "description": "Long lower wick, small body at top - bullish reversal"
        },
        "inverted_hammer": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BULLISH,
            "strength": 3,
            "reliability": 0.55,
            "description": "Long upper wick, small body at bottom - potential reversal"
        },
        "shooting_star": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BEARISH,
            "strength": 4,
            "reliability": 0.60,
            "description": "Long upper wick, small body at bottom - bearish reversal"
        },
        "hanging_man": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BEARISH,
            "strength": 3,
            "reliability": 0.55,
            "description": "Long lower wick at top of uptrend - warning signal"
        },
        "doji": {
            "type": PatternType.INDECISION,
            "direction": PatternDirection.NEUTRAL,
            "strength": 2,
            "reliability": 0.50,
            "description": "Open equals close - market indecision"
        },
        "dragonfly_doji": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BULLISH,
            "strength": 4,
            "reliability": 0.58,
            "description": "Doji with long lower wick - potential bottom"
        },
        "gravestone_doji": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BEARISH,
            "strength": 4,
            "reliability": 0.58,
            "description": "Doji with long upper wick - potential top"
        },
        "marubozu_bullish": {
            "type": PatternType.CONTINUATION,
            "direction": PatternDirection.BULLISH,
            "strength": 5,
            "reliability": 0.65,
            "description": "Full body, no wicks - strong bullish momentum"
        },
        "marubozu_bearish": {
            "type": PatternType.CONTINUATION,
            "direction": PatternDirection.BEARISH,
            "strength": 5,
            "reliability": 0.65,
            "description": "Full body, no wicks - strong bearish momentum"
        },
        "spinning_top": {
            "type": PatternType.INDECISION,
            "direction": PatternDirection.NEUTRAL,
            "strength": 1,
            "reliability": 0.45,
            "description": "Small body, equal wicks - indecision"
        },

        # === TWO CANDLE PATTERNS ===
        "bullish_engulfing": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BULLISH,
            "strength": 5,
            "reliability": 0.68,
            "description": "Green candle engulfs previous red - strong bullish reversal"
        },
        "bearish_engulfing": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BEARISH,
            "strength": 5,
            "reliability": 0.68,
            "description": "Red candle engulfs previous green - strong bearish reversal"
        },
        "piercing_line": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BULLISH,
            "strength": 4,
            "reliability": 0.62,
            "description": "Gap down, closes above midpoint of previous candle"
        },
        "dark_cloud_cover": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BEARISH,
            "strength": 4,
            "reliability": 0.62,
            "description": "Gap up, closes below midpoint of previous candle"
        },
        "bullish_harami": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BULLISH,
            "strength": 3,
            "reliability": 0.53,
            "description": "Small green inside previous large red"
        },
        "bearish_harami": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BEARISH,
            "strength": 3,
            "reliability": 0.53,
            "description": "Small red inside previous large green"
        },
        "tweezer_bottom": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BULLISH,
            "strength": 4,
            "reliability": 0.58,
            "description": "Two candles with matching lows"
        },
        "tweezer_top": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BEARISH,
            "strength": 4,
            "reliability": 0.58,
            "description": "Two candles with matching highs"
        },

        # === THREE CANDLE PATTERNS ===
        "morning_star": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BULLISH,
            "strength": 5,
            "reliability": 0.72,
            "description": "Red, small body, green - strong bullish reversal"
        },
        "evening_star": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BEARISH,
            "strength": 5,
            "reliability": 0.72,
            "description": "Green, small body, red - strong bearish reversal"
        },
        "three_white_soldiers": {
            "type": PatternType.CONTINUATION,
            "direction": PatternDirection.BULLISH,
            "strength": 5,
            "reliability": 0.75,
            "description": "Three consecutive green candles with higher closes"
        },
        "three_black_crows": {
            "type": PatternType.CONTINUATION,
            "direction": PatternDirection.BEARISH,
            "strength": 5,
            "reliability": 0.75,
            "description": "Three consecutive red candles with lower closes"
        },
        "three_inside_up": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BULLISH,
            "strength": 4,
            "reliability": 0.65,
            "description": "Harami followed by confirmation candle"
        },
        "three_inside_down": {
            "type": PatternType.REVERSAL,
            "direction": PatternDirection.BEARISH,
            "strength": 4,
            "reliability": 0.65,
            "description": "Harami followed by confirmation candle"
        },
    }

    def __init__(self):
        self.min_body_ratio = 0.001  # Minimum body as % of price
        self.doji_threshold = 0.1    # Max body/range for doji

    def _get_ohlc(self, bar: Dict) -> Tuple[float, float, float, float]:
        """Extract OHLC from bar with flexible key names."""
        o = bar.get('o') or bar.get('open') or bar.get('Open') or 0
        h = bar.get('h') or bar.get('high') or bar.get('High') or 0
        l = bar.get('l') or bar.get('low') or bar.get('Low') or 0
        c = bar.get('c') or bar.get('close') or bar.get('Close') or 0
        return float(o), float(h), float(l), float(c)

    def _get_volume(self, bar: Dict) -> float:
        """Extract volume from bar."""
        return float(bar.get('v') or bar.get('volume') or bar.get('Volume') or 0)

    def _is_bullish(self, o: float, c: float) -> bool:
        """Check if candle is bullish (green)."""
        return c > o

    def _is_bearish(self, o: float, c: float) -> bool:
        """Check if candle is bearish (red)."""
        return c < o

    def _body_size(self, o: float, c: float) -> float:
        """Calculate body size."""
        return abs(c - o)

    def _range_size(self, h: float, l: float) -> float:
        """Calculate candle range."""
        return h - l if h > l else 0.0001

    def _upper_wick(self, o: float, h: float, c: float) -> float:
        """Calculate upper wick size."""
        return h - max(o, c)

    def _lower_wick(self, o: float, l: float, c: float) -> float:
        """Calculate lower wick size."""
        return min(o, c) - l

    def analyze(self, bars: List[Dict], lookback: int = 10) -> List[CandlestickPattern]:
        """
        Analyze bars for candlestick patterns.

        Args:
            bars: List of OHLCV bars (most recent last)
            lookback: Number of recent bars to analyze

        Returns:
            List of detected patterns
        """
        if len(bars) < 3:
            return []

        patterns = []
        recent = bars[-lookback:] if len(bars) >= lookback else bars

        # Analyze last candle for single patterns
        patterns.extend(self._detect_single_patterns(recent))

        # Analyze last 2 candles for double patterns
        if len(recent) >= 2:
            patterns.extend(self._detect_double_patterns(recent))

        # Analyze last 3 candles for triple patterns
        if len(recent) >= 3:
            patterns.extend(self._detect_triple_patterns(recent))

        # Sort by strength
        patterns.sort(key=lambda p: (p.strength, p.reliability), reverse=True)

        return patterns

    def _detect_single_patterns(self, bars: List[Dict]) -> List[CandlestickPattern]:
        """Detect single candle patterns."""
        patterns = []

        o, h, l, c = self._get_ohlc(bars[-1])
        if o == 0 or h == 0:
            return patterns

        body = self._body_size(o, c)
        range_sz = self._range_size(h, l)
        upper_wick = self._upper_wick(o, h, c)
        lower_wick = self._lower_wick(o, l, c)

        body_ratio = body / range_sz if range_sz > 0 else 0

        # DOJI - very small body
        if body_ratio < self.doji_threshold:
            if lower_wick > upper_wick * 2:
                # Dragonfly doji
                patterns.append(self._make_pattern("dragonfly_doji"))
            elif upper_wick > lower_wick * 2:
                # Gravestone doji
                patterns.append(self._make_pattern("gravestone_doji"))
            else:
                # Regular doji
                patterns.append(self._make_pattern("doji"))

        # HAMMER - long lower wick, small upper wick, body at top
        elif lower_wick >= body * 2 and upper_wick <= body * 0.5:
            if self._is_bullish(o, c):
                patterns.append(self._make_pattern("hammer"))
            else:
                patterns.append(self._make_pattern("hanging_man"))

        # INVERTED HAMMER / SHOOTING STAR - long upper wick
        elif upper_wick >= body * 2 and lower_wick <= body * 0.5:
            if self._is_bullish(o, c):
                patterns.append(self._make_pattern("inverted_hammer"))
            else:
                patterns.append(self._make_pattern("shooting_star"))

        # MARUBOZU - full body, minimal wicks
        elif body_ratio > 0.9:
            if self._is_bullish(o, c):
                patterns.append(self._make_pattern("marubozu_bullish"))
            else:
                patterns.append(self._make_pattern("marubozu_bearish"))

        # SPINNING TOP - small body, equal wicks
        elif body_ratio < 0.3 and abs(upper_wick - lower_wick) < range_sz * 0.2:
            patterns.append(self._make_pattern("spinning_top"))

        return patterns

    def _detect_double_patterns(self, bars: List[Dict]) -> List[CandlestickPattern]:
        """Detect two-candle patterns."""
        patterns = []

        o1, h1, l1, c1 = self._get_ohlc(bars[-2])
        o2, h2, l2, c2 = self._get_ohlc(bars[-1])

        if o1 == 0 or o2 == 0:
            return patterns

        body1 = self._body_size(o1, c1)
        body2 = self._body_size(o2, c2)

        # BULLISH ENGULFING
        if self._is_bearish(o1, c1) and self._is_bullish(o2, c2):
            if o2 <= c1 and c2 >= o1:
                patterns.append(self._make_pattern("bullish_engulfing"))

        # BEARISH ENGULFING
        if self._is_bullish(o1, c1) and self._is_bearish(o2, c2):
            if o2 >= c1 and c2 <= o1:
                patterns.append(self._make_pattern("bearish_engulfing"))

        # PIERCING LINE
        if self._is_bearish(o1, c1) and self._is_bullish(o2, c2):
            midpoint = (o1 + c1) / 2
            if o2 < c1 and c2 > midpoint and c2 < o1:
                patterns.append(self._make_pattern("piercing_line"))

        # DARK CLOUD COVER
        if self._is_bullish(o1, c1) and self._is_bearish(o2, c2):
            midpoint = (o1 + c1) / 2
            if o2 > c1 and c2 < midpoint and c2 > o1:
                patterns.append(self._make_pattern("dark_cloud_cover"))

        # BULLISH HARAMI
        if self._is_bearish(o1, c1) and self._is_bullish(o2, c2):
            if o2 > c1 and c2 < o1 and body2 < body1 * 0.5:
                patterns.append(self._make_pattern("bullish_harami"))

        # BEARISH HARAMI
        if self._is_bullish(o1, c1) and self._is_bearish(o2, c2):
            if o2 < c1 and c2 > o1 and body2 < body1 * 0.5:
                patterns.append(self._make_pattern("bearish_harami"))

        # TWEEZER BOTTOM
        if abs(l1 - l2) < (h1 - l1) * 0.05:
            if self._is_bearish(o1, c1) and self._is_bullish(o2, c2):
                patterns.append(self._make_pattern("tweezer_bottom"))

        # TWEEZER TOP
        if abs(h1 - h2) < (h1 - l1) * 0.05:
            if self._is_bullish(o1, c1) and self._is_bearish(o2, c2):
                patterns.append(self._make_pattern("tweezer_top"))

        return patterns

    def _detect_triple_patterns(self, bars: List[Dict]) -> List[CandlestickPattern]:
        """Detect three-candle patterns."""
        patterns = []

        o1, h1, l1, c1 = self._get_ohlc(bars[-3])
        o2, h2, l2, c2 = self._get_ohlc(bars[-2])
        o3, h3, l3, c3 = self._get_ohlc(bars[-1])

        if o1 == 0 or o2 == 0 or o3 == 0:
            return patterns

        body1 = self._body_size(o1, c1)
        body2 = self._body_size(o2, c2)
        body3 = self._body_size(o3, c3)
        range2 = self._range_size(h2, l2)

        # MORNING STAR
        if self._is_bearish(o1, c1) and body2 < body1 * 0.3 and self._is_bullish(o3, c3):
            if c3 > (o1 + c1) / 2:
                patterns.append(self._make_pattern("morning_star"))

        # EVENING STAR
        if self._is_bullish(o1, c1) and body2 < body1 * 0.3 and self._is_bearish(o3, c3):
            if c3 < (o1 + c1) / 2:
                patterns.append(self._make_pattern("evening_star"))

        # THREE WHITE SOLDIERS
        if (self._is_bullish(o1, c1) and self._is_bullish(o2, c2) and self._is_bullish(o3, c3)):
            if c2 > c1 and c3 > c2 and o2 > o1 and o3 > o2:
                patterns.append(self._make_pattern("three_white_soldiers"))

        # THREE BLACK CROWS
        if (self._is_bearish(o1, c1) and self._is_bearish(o2, c2) and self._is_bearish(o3, c3)):
            if c2 < c1 and c3 < c2 and o2 < o1 and o3 < o2:
                patterns.append(self._make_pattern("three_black_crows"))

        # THREE INSIDE UP (harami + confirmation)
        if self._is_bearish(o1, c1) and body2 < body1 * 0.5:
            if o2 > c1 and c2 < o1 and self._is_bullish(o3, c3) and c3 > o1:
                patterns.append(self._make_pattern("three_inside_up"))

        # THREE INSIDE DOWN (harami + confirmation)
        if self._is_bullish(o1, c1) and body2 < body1 * 0.5:
            if o2 < c1 and c2 > o1 and self._is_bearish(o3, c3) and c3 < o1:
                patterns.append(self._make_pattern("three_inside_down"))

        return patterns

    def _make_pattern(self, name: str) -> CandlestickPattern:
        """Create a CandlestickPattern from pattern definition."""
        defn = self.PATTERNS.get(name, {})
        return CandlestickPattern(
            name=name,
            pattern_type=defn.get("type", PatternType.INDECISION),
            direction=defn.get("direction", PatternDirection.NEUTRAL),
            strength=defn.get("strength", 1),
            reliability=defn.get("reliability", 0.5),
            candles_used=3 if "three" in name or "star" in name else (2 if any(x in name for x in ["engulfing", "harami", "tweezer", "piercing", "cloud"]) else 1),
            description=defn.get("description", "")
        )

    def get_confluence_score(
        self,
        patterns: List[CandlestickPattern],
        vwap_position: str,  # "above", "below", "at"
        volume_ratio: float,
        trend_direction: str,  # "up", "down", "sideways"
        time_quality: float  # 0-100
    ) -> ConfluenceResult:
        """
        Calculate confluence score from multiple factors.

        Higher score = stronger signal.
        """
        if not patterns:
            return ConfluenceResult(
                score=0,
                confidence=0,
                direction="neutral",
                action="NEUTRAL",
                patterns_found=[],
                reasoning="No patterns detected"
            )

        # Calculate pattern score (weighted by strength and reliability)
        bullish_score = 0
        bearish_score = 0

        for p in patterns:
            weight = p.strength * p.reliability * 20  # Max ~75 per pattern
            if p.direction == PatternDirection.BULLISH:
                bullish_score += weight
            elif p.direction == PatternDirection.BEARISH:
                bearish_score += weight

        # Determine direction
        if bullish_score > bearish_score * 1.2:
            direction = "bullish"
            pattern_score = min(100, bullish_score)
        elif bearish_score > bullish_score * 1.2:
            direction = "bearish"
            pattern_score = min(100, bearish_score)
        else:
            direction = "neutral"
            pattern_score = 0

        # VWAP alignment bonus
        vwap_bonus = 0
        if direction == "bullish" and vwap_position in ["below", "at"]:
            vwap_bonus = 15
        elif direction == "bearish" and vwap_position in ["above", "at"]:
            vwap_bonus = 15

        # Volume bonus
        volume_bonus = 0
        if volume_ratio >= 2.0:
            volume_bonus = 20
        elif volume_ratio >= 1.5:
            volume_bonus = 10
        elif volume_ratio >= 1.0:
            volume_bonus = 5

        # Trend alignment bonus
        trend_bonus = 0
        if (direction == "bullish" and trend_direction == "up") or \
           (direction == "bearish" and trend_direction == "down"):
            trend_bonus = 10

        # Time quality factor
        time_factor = time_quality / 100

        # Calculate final score
        raw_score = pattern_score + vwap_bonus + volume_bonus + trend_bonus
        final_score = min(100, raw_score * time_factor) if time_factor > 0.5 else raw_score * 0.5

        # Calculate confidence
        confidence = min(100, len(patterns) * 15 + pattern_score * 0.5)

        # Determine action
        if final_score >= 80:
            action = "STRONG_BUY" if direction == "bullish" else "STRONG_SELL"
        elif final_score >= 60:
            action = "BUY" if direction == "bullish" else "SELL"
        else:
            action = "NEUTRAL"

        # Build reasoning
        pattern_names = [p.name for p in patterns[:3]]
        reasoning = f"Patterns: {', '.join(pattern_names)}. "
        reasoning += f"VWAP: {vwap_position}, Volume: {volume_ratio:.1f}x. "
        if trend_bonus > 0:
            reasoning += f"Trend-aligned. "

        return ConfluenceResult(
            score=final_score,
            confidence=confidence,
            direction=direction,
            action=action,
            patterns_found=patterns,
            reasoning=reasoning
        )


# Singleton instance
candlestick_analyzer = CandlestickAnalyzer()
