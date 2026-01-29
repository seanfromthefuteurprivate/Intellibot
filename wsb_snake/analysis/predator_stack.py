"""
Predator Stack - Multi-model AI analyzer for apex predator scalping.

Uses GPT-4o (PRIMARY with VISION) + DeepSeek (SECONDARY fallback) for
surgical precision pattern recognition on 0DTE setups.

GPT-4o can SEE charts and identify:
- Candlestick patterns visually
- VWAP bands and price position
- Volume profiles
- Support/resistance levels

RATE LIMITED: Enforces strict daily budgets to prevent API suspension.
"""
import os
import asyncio
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import httpx

from wsb_snake.utils.logger import get_logger
from wsb_snake.utils.rate_limit import limiter

logger = get_logger(__name__)


class PredatorVerdict(Enum):
    """Decisive verdicts from the predator stack."""
    STRIKE_CALLS = "STRIKE_CALLS"
    STRIKE_PUTS = "STRIKE_PUTS"
    NO_TRADE = "NO_TRADE"
    ABORT = "ABORT"


@dataclass
class PredatorAnalysis:
    """Complete analysis from the predator stack."""
    verdict: PredatorVerdict
    confidence: float  # 0-100
    entry_quality: str  # "EXCELLENT", "GOOD", "MARGINAL", "POOR"
    vwap_analysis: str
    delta_analysis: str
    trap_risk: str
    timing_score: int  # 0-100
    reasoning: str
    model_used: str
    candlestick_patterns: str = ""
    confirmed_by: Optional[str] = None


class ResultCache:
    """Simple cache for AI analysis results to reduce API calls."""

    def __init__(self, ttl_seconds: int = 90):
        self.cache: Dict[str, tuple] = {}
        self.ttl = ttl_seconds

    def _make_key(self, ticker: str, pattern: str, price: float, direction_hint: str) -> str:
        price_rounded = round(price, 1)
        key_str = f"{ticker}:{pattern}:{price_rounded}:{direction_hint}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, ticker: str, pattern: str, price: float, direction_hint: str) -> Optional[PredatorAnalysis]:
        key = self._make_key(ticker, pattern, price, direction_hint)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                logger.debug(f"Cache HIT for {ticker} {pattern}")
                return result
            else:
                del self.cache[key]
        return None

    def set(self, ticker: str, pattern: str, price: float, direction_hint: str, result: PredatorAnalysis):
        key = self._make_key(ticker, pattern, price, direction_hint)
        self.cache[key] = (result, datetime.now())
        self._cleanup()

    def _cleanup(self):
        now = datetime.now()
        expired = [k for k, (_, ts) in self.cache.items()
                   if now - ts >= timedelta(seconds=self.ttl)]
        for k in expired:
            del self.cache[k]


class PredatorStack:
    """
    Multi-model apex predator analyzer.

    PRIMARY: GPT-4o with VISION (can analyze chart images)
    SECONDARY: DeepSeek (text-only fallback)

    GPT-4o sees the actual chart and identifies patterns visually.
    """

    def __init__(self):
        self.openai_key = os.environ.get('OPENAI_API_KEY')
        self.deepseek_key = os.environ.get('DEEPSEEK_API_KEY')

        self.openai_available = bool(self.openai_key)
        self.deepseek_available = bool(self.deepseek_key)

        self.cache = ResultCache(ttl_seconds=90)
        self.calls_today = {'openai': 0, 'deepseek': 0}

        logger.info(f"Predator Stack initialized - OpenAI (PRIMARY): {self.openai_available}, "
                   f"DeepSeek (SECONDARY): {self.deepseek_available}")
        logger.info("GPT-4o VISION enabled for chart analysis")

        # Vision-enabled prompt for GPT-4o
        self.vision_system_prompt = """You are an APEX PREDATOR 0DTE options scalper analyzing a real-time chart.

LOOK AT THE CHART IMAGE CAREFULLY. You can see:
- Candlestick price action with OHLC bars
- VWAP line (if shown)
- Volume bars
- Any indicators present

ANALYZE WHAT YOU SEE:

1. CANDLESTICK PATTERNS (look at the actual candles):
   - Single: Hammer, Shooting Star, Doji, Marubozu
   - Double: Engulfing, Harami, Tweezer
   - Triple: Morning/Evening Star, Three Soldiers/Crows

2. PRICE ACTION:
   - Is price trending up, down, or consolidating?
   - Any support/resistance levels?
   - Is the current candle strong or weak?

3. VOLUME ANALYSIS:
   - Is volume increasing or decreasing?
   - Any volume spikes?

4. ENTRY TIMING:
   - Is this the optimal entry point?
   - Should we wait for confirmation?
   - Is the move already extended?

YOUR VERDICT MUST BE ONE OF:
- STRIKE_CALLS: High confidence LONG entry (buy calls)
- STRIKE_PUTS: High confidence SHORT entry (buy puts)
- NO_TRADE: No clear edge, stay flat
- ABORT: Conditions too risky, avoid

RESPOND IN THIS EXACT FORMAT:
VERDICT: [STRIKE_CALLS/STRIKE_PUTS/NO_TRADE/ABORT]
CONFIDENCE: [0-100]
ENTRY_QUALITY: [EXCELLENT/GOOD/MARGINAL/POOR]
VWAP_ANALYSIS: [Your VWAP/price position analysis]
DELTA_ANALYSIS: [Your momentum analysis]
CANDLESTICK_PATTERNS: [List the candlestick patterns you see]
TRAP_RISK: [LOW/MEDIUM/HIGH]
TIMING_SCORE: [0-100]
REASONING: [2-3 sentences explaining your decision]

BE DECISIVE. If you see a clear setup, call it. If not, say NO_TRADE."""

        # Text-only prompt for DeepSeek fallback
        self.text_system_prompt = """You are an APEX PREDATOR scalping analyst analyzing a trading setup.

Based on the data provided (no chart image), analyze the setup and provide your verdict.

ANALYSIS FRAMEWORK:
1. VWAP ANALYSIS: Is price at a high-probability zone?
2. PATTERN ANALYSIS: Does the pattern suggest direction?
3. MOMENTUM: Is there clear directional momentum?
4. RISK: What are the trap/reversal risks?

YOUR VERDICT MUST BE ONE OF:
- STRIKE_CALLS: High confidence LONG entry
- STRIKE_PUTS: High confidence SHORT entry
- NO_TRADE: No clear edge
- ABORT: Too risky

RESPOND IN THIS EXACT FORMAT:
VERDICT: [STRIKE_CALLS/STRIKE_PUTS/NO_TRADE/ABORT]
CONFIDENCE: [0-100]
ENTRY_QUALITY: [EXCELLENT/GOOD/MARGINAL/POOR]
VWAP_ANALYSIS: [Analysis]
DELTA_ANALYSIS: [Analysis]
CANDLESTICK_PATTERNS: [Inferred patterns]
TRAP_RISK: [LOW/MEDIUM/HIGH]
TIMING_SCORE: [0-100]
REASONING: [2-3 sentences]"""

    async def _call_openai_vision(
        self,
        chart_base64: str,
        context: str
    ) -> Optional[str]:
        """
        PRIMARY: Call GPT-4o with VISION to analyze chart image.

        This is the preferred method as it can actually SEE the chart.
        """
        if not self.openai_key:
            return None

        if not chart_base64:
            logger.warning("No chart image provided for GPT-4o vision")
            return await self._call_openai_text(context)

        if not limiter.wait_if_needed('openai'):
            logger.warning("OpenAI daily limit reached - skipping")
            return None

        try:
            url = "https://api.openai.com/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gpt-4o",  # Full GPT-4o with vision
                "messages": [
                    {"role": "system", "content": self.vision_system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"CONTEXT: {context}\n\nAnalyze the chart below:"},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/png;base64,{chart_base64}",
                            "detail": "high"
                        }}
                    ]}
                ],
                "temperature": 0.1,
                "max_tokens": 600
            }

            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()

                data = response.json()
                if "choices" in data and data["choices"]:
                    self.calls_today['openai'] += 1
                    logger.info(f"GPT-4o VISION call successful (today: {self.calls_today['openai']})")
                    return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("OpenAI rate limited (429)")
            else:
                logger.error(f"OpenAI HTTP error {e.response.status_code}")
        except Exception as e:
            logger.error(f"OpenAI Vision API error: {e}")

        return None

    async def _call_openai_text(
        self,
        context: str
    ) -> Optional[str]:
        """Text-only OpenAI call (backup when no chart)."""
        if not self.openai_key:
            return None

        if not limiter.wait_if_needed('openai'):
            return None

        try:
            url = "https://api.openai.com/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": self.text_system_prompt},
                    {"role": "user", "content": context}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()

                data = response.json()
                if "choices" in data and data["choices"]:
                    self.calls_today['openai'] += 1
                    return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"OpenAI text API error: {e}")

        return None

    async def _call_deepseek(
        self,
        context: str
    ) -> Optional[str]:
        """
        SECONDARY: DeepSeek text-only analysis.

        Used when GPT-4o is unavailable or rate limited.
        """
        if not self.deepseek_key:
            return None

        if not limiter.wait_if_needed('deepseek'):
            logger.warning("DeepSeek daily limit reached - skipping")
            return None

        try:
            url = "https://api.deepseek.com/v1/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.deepseek_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": self.text_system_prompt},
                    {"role": "user", "content": context}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()

                data = response.json()
                if "choices" in data and data["choices"]:
                    self.calls_today['deepseek'] += 1
                    logger.debug(f"DeepSeek call successful (today: {self.calls_today['deepseek']})")
                    return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            logger.error(f"DeepSeek HTTP error {e.response.status_code}")
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")

        return None

    def _parse_response(self, response: str, model: str) -> PredatorAnalysis:
        """Parse the AI response into structured analysis."""
        if not response or not isinstance(response, str):
            return PredatorAnalysis(
                verdict=PredatorVerdict.NO_TRADE,
                confidence=0.0,
                entry_quality="POOR",
                vwap_analysis="Parse error",
                delta_analysis="Parse error",
                trap_risk="UNKNOWN",
                timing_score=0,
                reasoning="Failed to parse AI response",
                model_used=model,
                candlestick_patterns=""
            )

        lines = response.strip().split('\n')
        result = {
            'verdict': PredatorVerdict.NO_TRADE,
            'confidence': 0.0,
            'entry_quality': 'POOR',
            'vwap_analysis': '',
            'delta_analysis': '',
            'candlestick_patterns': '',
            'trap_risk': 'UNKNOWN',
            'timing_score': 0,
            'reasoning': ''
        }

        for line in lines:
            line = line.strip()
            if line.startswith('VERDICT:'):
                verdict_str = line.replace('VERDICT:', '').strip().upper()
                if 'CALLS' in verdict_str:
                    result['verdict'] = PredatorVerdict.STRIKE_CALLS
                elif 'PUTS' in verdict_str:
                    result['verdict'] = PredatorVerdict.STRIKE_PUTS
                elif 'ABORT' in verdict_str:
                    result['verdict'] = PredatorVerdict.ABORT
                else:
                    result['verdict'] = PredatorVerdict.NO_TRADE

            elif line.startswith('CONFIDENCE:'):
                try:
                    conf_str = line.replace('CONFIDENCE:', '').strip().replace('%', '')
                    result['confidence'] = float(conf_str)
                except:
                    result['confidence'] = 0.0

            elif line.startswith('ENTRY_QUALITY:'):
                result['entry_quality'] = line.replace('ENTRY_QUALITY:', '').strip().upper()

            elif line.startswith('VWAP_ANALYSIS:'):
                result['vwap_analysis'] = line.replace('VWAP_ANALYSIS:', '').strip()

            elif line.startswith('DELTA_ANALYSIS:'):
                result['delta_analysis'] = line.replace('DELTA_ANALYSIS:', '').strip()

            elif line.startswith('CANDLESTICK_PATTERNS:'):
                result['candlestick_patterns'] = line.replace('CANDLESTICK_PATTERNS:', '').strip()

            elif line.startswith('TRAP_RISK:'):
                result['trap_risk'] = line.replace('TRAP_RISK:', '').strip()

            elif line.startswith('TIMING_SCORE:'):
                try:
                    timing_str = line.replace('TIMING_SCORE:', '').strip().replace('%', '')
                    result['timing_score'] = int(float(timing_str))
                except:
                    result['timing_score'] = 0

            elif line.startswith('REASONING:'):
                result['reasoning'] = line.replace('REASONING:', '').strip()

        return PredatorAnalysis(
            verdict=result['verdict'],
            confidence=result['confidence'],
            entry_quality=result['entry_quality'],
            vwap_analysis=result['vwap_analysis'],
            delta_analysis=result['delta_analysis'],
            trap_risk=result['trap_risk'],
            timing_score=result['timing_score'],
            reasoning=result['reasoning'],
            model_used=model,
            candlestick_patterns=result['candlestick_patterns']
        )

    async def analyze(
        self,
        chart_base64: str,
        ticker: str = "SPY",
        pattern: str = "",
        current_price: float = 0.0,
        vwap: float = 0.0,
        extra_context: str = "",
        require_confirmation: bool = False
    ) -> PredatorAnalysis:
        """
        Analyze setup with multi-model predator stack.

        PRIMARY: GPT-4o with VISION (sends actual chart image)
        SECONDARY: DeepSeek (text-only fallback)
        """
        # Check cache first
        direction_hint = "long" if "bounce" in pattern.lower() or "reclaim" in pattern.lower() else "short"
        cached = self.cache.get(ticker, pattern, current_price, direction_hint)
        if cached:
            logger.info(f"Using cached analysis for {ticker} {pattern}")
            return cached

        # Build context
        vwap_position = "above" if current_price > vwap else "below"
        vwap_distance_pct = abs(current_price - vwap) / vwap * 100 if vwap > 0 else 0

        context = f"""Ticker: {ticker}
Pattern: {pattern}
Price: ${current_price:.2f}
VWAP: ${vwap:.2f} (price is {vwap_position} by {vwap_distance_pct:.2f}%)

{extra_context}"""

        # PRIMARY: Try GPT-4o with vision
        primary_response = None
        model_used = "none"

        if self.openai_available and chart_base64:
            logger.info(f"Calling GPT-4o VISION for {ticker}...")
            primary_response = await self._call_openai_vision(chart_base64, context)
            model_used = "gpt-4o-vision"

        # Fallback to text-only OpenAI if vision fails
        if not primary_response and self.openai_available:
            logger.info("GPT-4o vision unavailable, trying text...")
            primary_response = await self._call_openai_text(context)
            model_used = "gpt-4o-mini"

        # SECONDARY: DeepSeek fallback
        if not primary_response and self.deepseek_available:
            logger.info("OpenAI unavailable, falling back to DeepSeek")
            primary_response = await self._call_deepseek(context)
            model_used = "deepseek"

        if not primary_response:
            logger.warning("All AI models unavailable or rate limited")
            return PredatorAnalysis(
                verdict=PredatorVerdict.NO_TRADE,
                confidence=0.0,
                entry_quality="POOR",
                vwap_analysis="AI unavailable",
                delta_analysis="AI unavailable",
                trap_risk="UNKNOWN",
                timing_score=0,
                reasoning="AI models unavailable - defaulting to NO_TRADE",
                model_used="none"
            )

        analysis = self._parse_response(primary_response, model_used)

        # Get confirmation if requested and high confidence
        if require_confirmation and analysis.confidence >= 70 and analysis.verdict in [
            PredatorVerdict.STRIKE_CALLS, PredatorVerdict.STRIKE_PUTS
        ]:
            logger.info(f"Seeking DeepSeek confirmation for {analysis.verdict.value}")
            if self.deepseek_available and model_used != "deepseek":
                confirm_response = await self._call_deepseek(context)
                if confirm_response:
                    confirm_analysis = self._parse_response(confirm_response, "deepseek")
                    if confirm_analysis.verdict == analysis.verdict:
                        analysis.confirmed_by = "deepseek"
                        analysis.confidence = min(100, analysis.confidence + 5)
                        logger.info(f"DeepSeek CONFIRMED: {analysis.verdict.value}")
                    else:
                        analysis.confidence = max(0, analysis.confidence - 15)
                        logger.warning(f"DeepSeek DISAGREED: {confirm_analysis.verdict.value}")

        # Cache result
        self.cache.set(ticker, pattern, current_price, direction_hint, analysis)

        return analysis

    def analyze_sync(
        self,
        chart_base64: str,
        ticker: str = "SPY",
        pattern: str = "",
        current_price: float = 0.0,
        vwap: float = 0.0,
        extra_context: str = "",
        require_confirmation: bool = False
    ) -> PredatorAnalysis:
        """Synchronous wrapper for analyze."""
        try:
            return asyncio.run(
                self.analyze(chart_base64, ticker, pattern, current_price, vwap, extra_context, require_confirmation)
            )
        except Exception as e:
            logger.error(f"Sync analyze error: {e}")
            return PredatorAnalysis(
                verdict=PredatorVerdict.NO_TRADE,
                confidence=0.0,
                entry_quality="POOR",
                vwap_analysis="Error",
                delta_analysis="Error",
                trap_risk="UNKNOWN",
                timing_score=0,
                reasoning=f"Analysis error: {str(e)}",
                model_used="none"
            )

    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            "calls_today": self.calls_today,
            "rate_limits": {
                "openai": limiter.get_remaining("openai"),
                "deepseek": limiter.get_remaining("deepseek")
            }
        }


predator_stack = PredatorStack()
