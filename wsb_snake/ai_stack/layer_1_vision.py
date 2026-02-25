"""
Layer 1: GPT-4o Vision (DEMOTED)

Purpose: Chart pattern detection only - NO verdict
Weight: 15%
Cost: ~$0.01 per call
Latency: ~400ms

Change from original:
- OLD: Returns STRIKE_CALLS/PUTS/NO_TRADE with confidence (was verdict)
- NEW: Returns structured pattern detection only (raw signal)

Let downstream layers (adversarial, DNA) determine if trade is valid.
"""

import os
import json
import time
import base64
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class VisionSignal:
    """Structured output from vision analysis."""
    pattern_detected: str = "none"  # "bullish_engulfing", "bear_flag", "breakout", "none"
    pattern_strength: int = 0  # 1-5 scale
    vwap_position: str = "unknown"  # "above", "below", "touching"
    volume_surge: bool = False
    momentum: str = "flat"  # "accelerating", "decelerating", "flat"
    raw_bias: str = "NEUTRAL"  # "CALL", "PUT", "NEUTRAL"
    key_levels: List[float] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_detected': self.pattern_detected,
            'pattern_strength': self.pattern_strength,
            'vwap_position': self.vwap_position,
            'volume_surge': self.volume_surge,
            'momentum': self.momentum,
            'raw_bias': self.raw_bias,
            'key_levels': self.key_levels,
            'confidence': self.confidence,
            'latency_ms': self.latency_ms
        }


class VisionLayer:
    """
    GPT-4o Vision layer - chart pattern detection.

    Demoted from verdict generator to signal extractor.
    Returns raw patterns, not trade decisions.
    """

    VISION_PROMPT = """You are a chart pattern detector for 0DTE options trading.
Analyze this chart and extract ONLY the following information.
DO NOT give a trade recommendation - just detect patterns.

Output JSON format:
{
    "pattern_detected": "pattern name or none",
    "pattern_strength": 1-5,
    "vwap_position": "above|below|touching",
    "volume_surge": true/false,
    "momentum": "accelerating|decelerating|flat",
    "raw_bias": "CALL|PUT|NEUTRAL",
    "key_levels": [support1, resistance1],
    "confidence": 0.0-1.0
}

Pattern names: bullish_engulfing, bearish_engulfing, hammer, shooting_star,
doji, morning_star, evening_star, breakout_above, breakdown_below,
bull_flag, bear_flag, double_bottom, double_top, none

Be precise. This is for 0DTE - small patterns matter."""

    def __init__(self):
        """Initialize vision layer."""
        self._openai_client = None
        self._call_count = 0
        self._total_latency = 0

        # Try to initialize OpenAI client
        try:
            import openai
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                self._openai_client = openai.OpenAI(api_key=api_key)
                logger.info("VISION_LAYER: OpenAI client initialized")
            else:
                logger.warning("VISION_LAYER: No OPENAI_API_KEY found")
        except Exception as e:
            logger.warning(f"VISION_LAYER: OpenAI unavailable: {e}")

    def analyze_chart(
        self,
        image_base64: str = None,
        image_path: str = None,
        ticker: str = "SPY",
        timeframe: str = "5min"
    ) -> VisionSignal:
        """
        Analyze chart image and extract patterns.

        Args:
            image_base64: Base64 encoded image
            image_path: Path to image file
            ticker: Symbol being analyzed
            timeframe: Chart timeframe

        Returns:
            VisionSignal with detected patterns
        """
        start = time.time()

        if not self._openai_client:
            return VisionSignal(
                error="OpenAI client not available",
                latency_ms=(time.time() - start) * 1000
            )

        # Get image data
        if image_path and not image_base64:
            try:
                with open(image_path, 'rb') as f:
                    image_base64 = base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                return VisionSignal(
                    error=f"Failed to read image: {e}",
                    latency_ms=(time.time() - start) * 1000
                )

        if not image_base64:
            return VisionSignal(
                error="No image provided",
                latency_ms=(time.time() - start) * 1000
            )

        try:
            response = self._openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Ticker: {ticker}, Timeframe: {timeframe}\n\n{self.VISION_PROMPT}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0
            )

            latency = (time.time() - start) * 1000
            self._call_count += 1
            self._total_latency += latency

            # Parse response
            content = response.choices[0].message.content

            # Extract JSON from response
            signal = self._parse_response(content)
            signal.latency_ms = latency

            logger.info(
                f"VISION_L1: {ticker} pattern={signal.pattern_detected} "
                f"bias={signal.raw_bias} conf={signal.confidence:.0%} "
                f"in {latency:.0f}ms"
            )

            return signal

        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"VISION_L1: Error - {e}")
            return VisionSignal(
                error=str(e),
                latency_ms=latency
            )

    def analyze_from_data(
        self,
        candles: List[Dict],
        ticker: str = "SPY",
        indicators: Dict = None
    ) -> VisionSignal:
        """
        Analyze from raw candle data (when no chart image available).
        Uses text-based analysis as fallback.

        Args:
            candles: List of OHLCV dicts
            ticker: Symbol
            indicators: Optional dict with RSI, MACD, etc.

        Returns:
            VisionSignal
        """
        start = time.time()

        if not self._openai_client:
            return VisionSignal(
                error="OpenAI client not available",
                latency_ms=(time.time() - start) * 1000
            )

        if not candles:
            return VisionSignal(
                error="No candle data provided",
                latency_ms=(time.time() - start) * 1000
            )

        # Format candle data
        recent_candles = candles[-20:]  # Last 20 candles
        candle_text = self._format_candles(recent_candles)

        prompt = f"""Analyze these candles for {ticker} and detect patterns.

{candle_text}

{f"Indicators: {json.dumps(indicators)}" if indicators else ""}

{self.VISION_PROMPT}"""

        try:
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use mini for text-only
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0
            )

            latency = (time.time() - start) * 1000
            self._call_count += 1
            self._total_latency += latency

            content = response.choices[0].message.content
            signal = self._parse_response(content)
            signal.latency_ms = latency

            logger.debug(f"VISION_L1_TEXT: {ticker} pattern={signal.pattern_detected}")
            return signal

        except Exception as e:
            return VisionSignal(
                error=str(e),
                latency_ms=(time.time() - start) * 1000
            )

    def _format_candles(self, candles: List[Dict]) -> str:
        """Format candles for text analysis."""
        lines = ["Recent candles (oldest to newest):"]
        for i, c in enumerate(candles):
            o, h, l, close = c.get('o', 0), c.get('h', 0), c.get('l', 0), c.get('c', 0)
            v = c.get('v', 0)
            body = "GREEN" if close > o else "RED" if close < o else "DOJI"
            lines.append(f"{i+1}. O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{close:.2f} V:{v:,} [{body}]")
        return "\n".join(lines)

    def _parse_response(self, content: str) -> VisionSignal:
        """Parse GPT response into VisionSignal."""
        try:
            # Try to extract JSON
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            elif "{" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
            else:
                json_str = content

            data = json.loads(json_str)

            return VisionSignal(
                pattern_detected=data.get('pattern_detected', 'none'),
                pattern_strength=int(data.get('pattern_strength', 0)),
                vwap_position=data.get('vwap_position', 'unknown'),
                volume_surge=bool(data.get('volume_surge', False)),
                momentum=data.get('momentum', 'flat'),
                raw_bias=data.get('raw_bias', 'NEUTRAL').upper(),
                key_levels=data.get('key_levels', []),
                confidence=float(data.get('confidence', 0))
            )

        except Exception as e:
            logger.debug(f"VISION_L1: Parse error - {e}")
            return VisionSignal(error=f"Parse error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics."""
        return {
            'call_count': self._call_count,
            'total_latency_ms': self._total_latency,
            'avg_latency_ms': self._total_latency / max(self._call_count, 1)
        }


# Singleton
_vision_layer = None

def get_vision_layer() -> VisionLayer:
    """Get singleton VisionLayer instance."""
    global _vision_layer
    if _vision_layer is None:
        _vision_layer = VisionLayer()
    return _vision_layer
