"""
Predator Stack - Multi-model AI analyzer for apex predator scalping.

Uses DeepSeek (primary) + OpenAI GPT (confirmation) for
surgical precision pattern recognition on 0DTE setups.

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
    confirmed_by: Optional[str] = None


class ResultCache:
    """Simple cache for AI analysis results to reduce API calls."""

    def __init__(self, ttl_seconds: int = 120):
        self.cache: Dict[str, tuple] = {}  # hash -> (result, timestamp)
        self.ttl = ttl_seconds

    def _make_key(self, ticker: str, pattern: str, price: float, direction_hint: str) -> str:
        """Create cache key from setup parameters."""
        # Round price to reduce cache misses from small fluctuations
        price_rounded = round(price, 1)
        key_str = f"{ticker}:{pattern}:{price_rounded}:{direction_hint}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, ticker: str, pattern: str, price: float, direction_hint: str) -> Optional[PredatorAnalysis]:
        """Get cached result if valid."""
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
        """Cache a result."""
        key = self._make_key(ticker, pattern, price, direction_hint)
        self.cache[key] = (result, datetime.now())

        # Cleanup old entries
        self._cleanup()

    def _cleanup(self):
        """Remove expired entries."""
        now = datetime.now()
        expired = [k for k, (_, ts) in self.cache.items()
                   if now - ts >= timedelta(seconds=self.ttl)]
        for k in expired:
            del self.cache[k]


class PredatorStack:
    """
    Multi-model apex predator analyzer with STRICT RATE LIMITING.

    Primary: DeepSeek (cost-effective, no vision but good analysis)
    Confirmation: GPT-4o (for high-stakes validation)

    NOTE: Gemini removed due to suspension issues. Using text-based analysis only.
    """

    def __init__(self):
        self.deepseek_key = os.environ.get('DEEPSEEK_API_KEY')
        self.openai_key = os.environ.get('OPENAI_API_KEY')

        self.deepseek_available = bool(self.deepseek_key)
        self.openai_available = bool(self.openai_key)

        # Result cache to reduce API calls
        self.cache = ResultCache(ttl_seconds=120)

        # Track API usage
        self.calls_today = {'deepseek': 0, 'openai': 0}

        logger.info(f"Predator Stack initialized - DeepSeek: {self.deepseek_available}, "
                   f"OpenAI: {self.openai_available}")
        logger.info("Rate limiting ENABLED: Max 200 AI calls/day per service")

        self.scalp_system_prompt = """You are an APEX PREDATOR scalping analyst. Your job is to analyze setups
for 0DTE options scalping opportunities with SURGICAL PRECISION.

ANALYSIS FRAMEWORK:
1. VWAP ANALYSIS: Is price at a high-probability bounce/rejection zone?
   - At VWAP = neutral zone, wait for confirmation
   - At +1σ/-1σ band = potential reversal zone
   - At +2σ/-2σ band = extreme zone, high reversal probability

2. DELTA ANALYSIS: Who is in control?
   - Positive delta = buyers dominating
   - Negative delta = sellers dominating
   - Cumulative delta trend matters more than single bars

3. TRAP DETECTION: Is this a failed breakout/breakdown?
   - Failed breakouts (bull traps) = SHORT opportunity
   - Failed breakdowns (bear traps) = LONG opportunity

4. ENTRY TIMING: Is this the optimal moment?
   - Don't chase moves already extended
   - Wait for pullback in momentum moves
   - Enter on confirmation, not anticipation

YOUR VERDICT MUST BE ONE OF:
- STRIKE_CALLS: High confidence LONG entry (buy calls)
- STRIKE_PUTS: High confidence SHORT entry (buy puts)
- NO_TRADE: No clear edge, stay flat
- ABORT: Conditions too risky, avoid

RESPOND IN THIS EXACT FORMAT:
VERDICT: [STRIKE_CALLS/STRIKE_PUTS/NO_TRADE/ABORT]
CONFIDENCE: [0-100]
ENTRY_QUALITY: [EXCELLENT/GOOD/MARGINAL/POOR]
VWAP_ANALYSIS: [Your VWAP analysis in 1-2 sentences]
DELTA_ANALYSIS: [Your delta/momentum analysis in 1-2 sentences]
TRAP_RISK: [LOW/MEDIUM/HIGH - explain briefly]
TIMING_SCORE: [0-100]
REASONING: [2-3 sentences max explaining your decision]

Be DECISIVE. No wishy-washy answers. Either it's a trade or it's not."""

    def _check_rate_limit(self, service: str) -> bool:
        """Check if we can make an API call."""
        if not limiter.can_call(service):
            remaining = limiter.get_remaining(service)
            logger.warning(f"{service} rate limit: {remaining['day']} calls left today")
            return False
        return True

    async def _call_deepseek(
        self,
        context: str
    ) -> Optional[str]:
        """Call DeepSeek API for text analysis with rate limiting."""
        if not self.deepseek_key:
            return None

        # Check rate limit BEFORE calling
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
                    {"role": "system", "content": self.scalp_system_prompt},
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
            if e.response.status_code == 429:
                logger.warning("DeepSeek rate limited (429)")
            else:
                logger.error(f"DeepSeek HTTP error {e.response.status_code}")
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")

        return None

    async def _call_openai(
        self,
        context: str
    ) -> Optional[str]:
        """Call OpenAI GPT-4o for confirmation with rate limiting."""
        if not self.openai_key:
            return None

        # Check rate limit BEFORE calling
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
                "model": "gpt-4o-mini",  # Use mini for cost savings
                "messages": [
                    {"role": "system", "content": self.scalp_system_prompt},
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
                    logger.debug(f"OpenAI call successful (today: {self.calls_today['openai']})")
                    return data["choices"][0]["message"]["content"]

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("OpenAI rate limited (429)")
            else:
                logger.error(f"OpenAI HTTP error {e.response.status_code}")
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")

        return None

    def _parse_response(self, response: str, model: str) -> PredatorAnalysis:
        """Parse the AI response into structured analysis with defensive handling."""
        if not response or not isinstance(response, str):
            logger.warning(f"Invalid response from {model}")
            return PredatorAnalysis(
                verdict=PredatorVerdict.NO_TRADE,
                confidence=0.0,
                entry_quality="POOR",
                vwap_analysis="Parse error",
                delta_analysis="Parse error",
                trap_risk="UNKNOWN",
                timing_score=0,
                reasoning="Failed to parse AI response - defaulting to NO_TRADE",
                model_used=model
            )

        lines = response.strip().split('\n')
        result = {
            'verdict': PredatorVerdict.NO_TRADE,
            'confidence': 0.0,
            'entry_quality': 'POOR',
            'vwap_analysis': '',
            'delta_analysis': '',
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
            model_used=model
        )

    async def analyze(
        self,
        chart_base64: str,  # Kept for compatibility but not used
        ticker: str = "SPY",
        pattern: str = "",
        current_price: float = 0.0,
        vwap: float = 0.0,
        extra_context: str = "",
        require_confirmation: bool = False
    ) -> PredatorAnalysis:
        """
        Analyze setup with multi-model predator stack.

        NOTE: Vision analysis removed to prevent API suspension.
        Uses text-based analysis with enhanced context instead.
        """
        # Check cache first
        direction_hint = "long" if "bounce" in pattern.lower() or "reclaim" in pattern.lower() else "short"
        cached = self.cache.get(ticker, pattern, current_price, direction_hint)
        if cached:
            logger.info(f"Using cached analysis for {ticker} {pattern}")
            return cached

        # Build comprehensive text context (no image)
        vwap_position = "above" if current_price > vwap else "below"
        vwap_distance_pct = abs(current_price - vwap) / vwap * 100 if vwap > 0 else 0

        context = f"""SETUP ANALYSIS REQUEST:

Ticker: {ticker}
Pattern Detected: {pattern}
Current Price: ${current_price:.2f}
VWAP: ${vwap:.2f}
Price vs VWAP: {vwap_position} by {vwap_distance_pct:.2f}%

{extra_context}

Based on this setup data, provide your analysis.
If the pattern suggests bullish momentum (vwap_bounce, vwap_reclaim, breakout, failed_breakdown), consider STRIKE_CALLS.
If the pattern suggests bearish momentum (vwap_rejection, breakdown, failed_breakout), consider STRIKE_PUTS.
If unclear or risky, say NO_TRADE or ABORT.

Respond in the exact format required."""

        # Try DeepSeek first (primary)
        primary_response = None
        model_used = "none"

        if self.deepseek_available:
            primary_response = await self._call_deepseek(context)
            model_used = "deepseek"

        # Fallback to OpenAI if DeepSeek fails
        if not primary_response and self.openai_available:
            logger.info("DeepSeek unavailable, falling back to OpenAI")
            primary_response = await self._call_openai(context)
            model_used = "openai"

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
                reasoning="AI models unavailable or rate limited - defaulting to NO_TRADE for safety",
                model_used="none"
            )

        analysis = self._parse_response(primary_response, model_used)

        # Get confirmation from second model for high-confidence setups
        if require_confirmation and analysis.confidence >= 70 and analysis.verdict in [
            PredatorVerdict.STRIKE_CALLS, PredatorVerdict.STRIKE_PUTS
        ]:
            logger.info(f"Seeking confirmation for {analysis.verdict.value}")

            # Use the other model for confirmation
            confirm_response = None
            if model_used == "deepseek" and self.openai_available:
                confirm_response = await self._call_openai(context)
            elif model_used == "openai" and self.deepseek_available:
                confirm_response = await self._call_deepseek(context)

            if confirm_response:
                confirm_analysis = self._parse_response(
                    confirm_response,
                    "openai" if model_used == "deepseek" else "deepseek"
                )

                if confirm_analysis.verdict == analysis.verdict:
                    analysis.confirmed_by = confirm_analysis.model_used
                    analysis.confidence = min(100, analysis.confidence + 10)
                    logger.info(f"CONFIRMED by {confirm_analysis.model_used}: {analysis.verdict.value}")
                else:
                    analysis.confidence = max(0, analysis.confidence - 20)
                    analysis.reasoning += f" [Disagreement: {confirm_analysis.verdict.value}]"
                    logger.warning(f"DISAGREEMENT: {confirm_analysis.model_used} says {confirm_analysis.verdict.value}")

        # Cache the result
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
        """Synchronous wrapper for analyze. Thread-safe for worker threads."""
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
                reasoning=f"Analysis error: {str(e)} - defaulting to NO_TRADE",
                model_used="none"
            )

    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            "calls_today": self.calls_today,
            "rate_limits": {
                "deepseek": limiter.get_remaining("deepseek"),
                "openai": limiter.get_remaining("openai")
            }
        }


# Singleton instance
predator_stack = PredatorStack()
