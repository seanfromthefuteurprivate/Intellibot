"""
LangGraph AI Chart Analyzer - Uses GPT-4o vision (primary) or Gemini (fallback)
to analyze candlestick charts with pattern recognition and trade recommendations.

RATE LIMITING: Gemini has strict rate limits. We enforce:
- Max 10 requests per minute
- Max 100 requests per day
- Only call when candlestick patterns + news confluence detected
"""
import asyncio
import os
import httpx
import time
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime, date
import operator
import threading

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from wsb_snake.config import OPENAI_API_KEY
from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
GEMINI_ENABLED = os.environ.get('GEMINI_ENABLED', 'false').lower() == 'true'


class GeminiRateLimiter:
    """Rate limiter for Gemini API to prevent bans.

    Free tier limits:
    - 15 RPM (requests per minute)
    - 1,500 RPD (requests per day)
    - 1M tokens per minute

    We use conservative limits to stay safe:
    - 10 RPM
    - 100 RPD (to leave headroom)
    """

    def __init__(self, rpm_limit: int = 10, rpd_limit: int = 100):
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.minute_requests: List[float] = []
        self.daily_requests: int = 0
        self.daily_reset_date: date = date.today()
        self.lock = threading.Lock()

    def can_make_request(self) -> bool:
        """Check if we can make a request without exceeding limits."""
        with self.lock:
            now = time.time()
            today = date.today()

            # Reset daily counter if new day
            if today > self.daily_reset_date:
                self.daily_requests = 0
                self.daily_reset_date = today
                logger.info("Gemini rate limiter: Daily counter reset")

            # Check daily limit
            if self.daily_requests >= self.rpd_limit:
                logger.warning(f"Gemini daily limit reached ({self.rpd_limit} requests)")
                return False

            # Clean old minute requests
            self.minute_requests = [t for t in self.minute_requests if now - t < 60]

            # Check minute limit
            if len(self.minute_requests) >= self.rpm_limit:
                logger.warning(f"Gemini minute limit reached ({self.rpm_limit} RPM)")
                return False

            return True

    def record_request(self):
        """Record that a request was made."""
        with self.lock:
            self.minute_requests.append(time.time())
            self.daily_requests += 1
            logger.info(f"Gemini API call recorded: {self.daily_requests}/{self.rpd_limit} today, {len(self.minute_requests)}/{self.rpm_limit} this minute")

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        with self.lock:
            now = time.time()
            recent = [t for t in self.minute_requests if now - t < 60]
            return {
                "daily_used": self.daily_requests,
                "daily_limit": self.rpd_limit,
                "minute_used": len(recent),
                "minute_limit": self.rpm_limit,
                "can_request": self.can_make_request()
            }


# Global rate limiter instances
gemini_rate_limiter = GeminiRateLimiter()


class OpenAIRateLimiter:
    """Rate limiter for OpenAI API to conserve credits.

    SCALPING PHILOSOPHY: Only call AI for high-conviction setups.
    Don't burn credits on noise - focus on sure-shot opportunities.

    Conservative limits:
    - 15 RPM (requests per minute)
    - 200 RPD (requests per day) - preserve credits for actual opportunities
    """

    def __init__(self, rpm_limit: int = 15, rpd_limit: int = 200):
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.minute_requests: List[float] = []
        self.daily_requests: int = 0
        self.daily_reset_date: date = date.today()
        self.lock = threading.Lock()

    def can_make_request(self) -> bool:
        """Check if we can make a request without exceeding limits."""
        with self.lock:
            now = time.time()
            today = date.today()

            # Reset daily counter if new day
            if today > self.daily_reset_date:
                self.daily_requests = 0
                self.daily_reset_date = today
                logger.info("OpenAI rate limiter: Daily counter reset")

            # Check daily limit
            if self.daily_requests >= self.rpd_limit:
                logger.warning(f"OpenAI daily limit reached ({self.rpd_limit} requests) - conserving credits")
                return False

            # Clean old minute requests
            self.minute_requests = [t for t in self.minute_requests if now - t < 60]

            # Check minute limit
            if len(self.minute_requests) >= self.rpm_limit:
                logger.warning(f"OpenAI minute limit reached ({self.rpm_limit} RPM)")
                return False

            return True

    def record_request(self):
        """Record that a request was made."""
        with self.lock:
            self.minute_requests.append(time.time())
            self.daily_requests += 1
            logger.info(f"OpenAI API call: {self.daily_requests}/{self.rpd_limit} today")

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        with self.lock:
            now = time.time()
            recent = [t for t in self.minute_requests if now - t < 60]
            return {
                "daily_used": self.daily_requests,
                "daily_limit": self.rpd_limit,
                "minute_used": len(recent),
                "minute_limit": self.rpm_limit,
                "can_request": self.can_make_request()
            }


openai_rate_limiter = OpenAIRateLimiter()


class ChartAnalysisState(TypedDict):
    """State for the chart analysis workflow."""
    ticker: str
    chart_base64: str
    timeframe: str
    current_price: float
    signals: Dict[str, Any]
    pattern_analysis: str
    trend_analysis: str
    support_resistance: str
    trade_recommendation: str
    confidence_score: float
    final_analysis: str


CHART_ANALYST_PROMPT = """You are an expert technical analyst specializing in 0DTE (zero days to expiration) options trading. 
You're analyzing charts for potential same-day moves on high-volume stocks.

Your focus areas:
1. PATTERN RECOGNITION: Identify chart patterns (flags, wedges, head & shoulders, double tops/bottoms, triangles)
2. TREND ANALYSIS: Determine the intraday trend (bullish, bearish, choppy)
3. KEY LEVELS: Identify support/resistance, VWAP position, moving average tests
4. MOMENTUM: Assess if momentum is building or fading
5. VOLUME: Note if volume confirms price action

For 0DTE trading, we care about:
- Quick moves that can happen in the next 30-60 minutes
- Whether price is likely to continue or reverse
- Clear directional bias vs choppy conditions to avoid

Be concise and actionable. This is for same-day options, not swing trades."""


class LangGraphChartAnalyzer:
    """
    LangGraph-based chart analyzer that uses GPT-4o vision for pattern recognition.
    Runs as a background analysis layer to enhance signal quality.
    """
    
    def __init__(self):
        if not OPENAI_API_KEY:
            logger.warning("OpenAI API key not set - chart analysis disabled")
            self.enabled = False
            return
            
        self.enabled = True
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=1000
        )
        
        self.graph = self._build_graph()
        self.analysis_cache: Dict[str, Dict] = {}
        self.cache_ttl = 300
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for chart analysis."""
        
        workflow = StateGraph(ChartAnalysisState)
        
        workflow.add_node("analyze_patterns", self._analyze_patterns)
        workflow.add_node("analyze_trend", self._analyze_trend)
        workflow.add_node("identify_levels", self._identify_levels)
        workflow.add_node("generate_recommendation", self._generate_recommendation)
        workflow.add_node("compile_analysis", self._compile_analysis)
        
        workflow.set_entry_point("analyze_patterns")
        workflow.add_edge("analyze_patterns", "analyze_trend")
        workflow.add_edge("analyze_trend", "identify_levels")
        workflow.add_edge("identify_levels", "generate_recommendation")
        workflow.add_edge("generate_recommendation", "compile_analysis")
        workflow.add_edge("compile_analysis", END)
        
        return workflow.compile()
    
    async def _call_vision(self, image_base64: str, prompt: str) -> str:
        """Call GPT-4o vision with Gemini fallback.

        SCALPING STRATEGY: Only call for high-conviction setups.
        Rate limited to conserve credits for actual opportunities.
        """
        # Check OpenAI rate limits first
        if not openai_rate_limiter.can_make_request():
            status = openai_rate_limiter.get_status()
            logger.warning(f"OpenAI rate limited - {status['daily_used']}/{status['daily_limit']} today. Using Gemini fallback.")
            return await self._call_gemini_fallback(image_base64, prompt)

        # Try OpenAI first
        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            )
            
            response = await self.llm.ainvoke([
                SystemMessage(content=CHART_ANALYST_PROMPT),
                message
            ])

            openai_rate_limiter.record_request()
            return response.content
            
        except Exception as e:
            logger.warning(f"OpenAI Vision failed, trying Gemini fallback: {e}")
            return await self._call_gemini_fallback(image_base64, prompt)
    
    async def _call_gemini_fallback(self, image_base64: str, prompt: str) -> str:
        """Fallback to Gemini 2.0 Flash when OpenAI fails.

        RATE LIMITED: Uses conservative limits to prevent API bans:
        - 10 requests per minute
        - 100 requests per day

        Only called when significant patterns are detected (enforced by ChartBrain).
        """
        if not GEMINI_API_KEY or not GEMINI_ENABLED:
            logger.info("Gemini disabled or unavailable, using DeepSeek fallback")
            return await self._call_deepseek_fallback(image_base64, prompt)

        # Check rate limits before making request
        if not gemini_rate_limiter.can_make_request():
            status = gemini_rate_limiter.get_status()
            logger.warning(f"Gemini rate limited - daily: {status['daily_used']}/{status['daily_limit']}, minute: {status['minute_used']}/{status['minute_limit']}")
            return await self._call_deepseek_fallback(image_base64, prompt)
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": f"{CHART_ANALYST_PROMPT}\n\n{prompt}"},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_base64
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 1000
                }
            }
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()
                
                if "candidates" in result and result["candidates"]:
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                    gemini_rate_limiter.record_request()
                    logger.info("Gemini fallback successful")
                    return text
                    
            # Gemini returned no content - try DeepSeek
            logger.warning("Gemini returned no content, trying DeepSeek fallback")
            return await self._call_deepseek_fallback(image_base64, prompt)
            
        except Exception as e:
            logger.warning(f"Gemini fallback failed, trying DeepSeek: {e}")
            return await self._call_deepseek_fallback(image_base64, prompt)
    
    async def _call_deepseek_fallback(self, image_base64: str, prompt: str) -> str:
        """Third fallback to DeepSeek when both OpenAI and Gemini fail."""
        if not DEEPSEEK_API_KEY:
            logger.error("DeepSeek API key not available for fallback")
            return "Analysis unavailable: All AI APIs unavailable"
        
        try:
            url = "https://api.deepseek.com/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": CHART_ANALYST_PROMPT},
                    {"role": "user", "content": f"{prompt}\n\n[Note: Image analysis requested but DeepSeek text-only mode - provide general technical guidance based on the context provided.]"}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and result["choices"]:
                    text = result["choices"][0]["message"]["content"]
                    logger.info("DeepSeek fallback successful")
                    return text
                    
            return "Analysis unavailable: DeepSeek returned no content"
            
        except Exception as e:
            logger.error(f"DeepSeek fallback also failed: {e}")
            return f"Analysis unavailable: All AI models failed (OpenAI, Gemini, DeepSeek)"
    
    async def _analyze_patterns(self, state: ChartAnalysisState) -> Dict:
        """Node 1: Identify chart patterns."""
        prompt = f"""Analyze this {state['timeframe']} chart for {state['ticker']}.

IDENTIFY PATTERNS:
- What chart patterns do you see? (flags, triangles, wedges, double tops/bottoms, etc.)
- Are patterns complete or still forming?
- What do the patterns suggest for the next 30-60 minutes?

Be specific and concise. Focus on 0DTE relevance."""

        analysis = await self._call_vision(state['chart_base64'], prompt)
        return {"pattern_analysis": analysis}
    
    async def _analyze_trend(self, state: ChartAnalysisState) -> Dict:
        """Node 2: Analyze trend and momentum."""
        prompt = f"""Looking at this {state['timeframe']} chart for {state['ticker']}:

TREND ANALYSIS:
- What is the intraday trend? (Strong bullish, weak bullish, neutral/choppy, weak bearish, strong bearish)
- Is momentum building or fading?
- Are candles showing conviction (large bodies) or indecision (dojis, spinning tops)?

Keep it brief and actionable."""

        analysis = await self._call_vision(state['chart_base64'], prompt)
        return {"trend_analysis": analysis}
    
    async def _identify_levels(self, state: ChartAnalysisState) -> Dict:
        """Node 3: Identify support/resistance levels."""
        prompt = f"""For this {state['ticker']} chart:

KEY LEVELS:
- Where is immediate support?
- Where is immediate resistance?
- Where is price relative to VWAP (purple dashed line if visible)?
- Are there any clear breakout/breakdown levels?

Provide approximate price levels if visible."""

        analysis = await self._call_vision(state['chart_base64'], prompt)
        return {"support_resistance": analysis}
    
    async def _generate_recommendation(self, state: ChartAnalysisState) -> Dict:
        """Node 4: Generate trade recommendation."""
        context = f"""
Pattern Analysis: {state.get('pattern_analysis', 'N/A')}
Trend Analysis: {state.get('trend_analysis', 'N/A')}
Support/Resistance: {state.get('support_resistance', 'N/A')}
"""
        
        prompt = f"""Based on my analysis of {state['ticker']}:
{context}

TRADE RECOMMENDATION for 0DTE options:
1. Direction: CALLS, PUTS, or NO TRADE (too choppy)
2. Confidence: LOW, MEDIUM, or HIGH
3. Reasoning: One sentence explaining why
4. Risk: What would invalidate this setup?

Be decisive. If the chart is choppy or unclear, recommend NO TRADE."""

        analysis = await self._call_vision(state['chart_base64'], prompt)
        
        confidence = 0.5
        analysis_lower = analysis.lower()
        if "high confidence" in analysis_lower or "high" in analysis_lower:
            confidence = 0.8
        elif "medium confidence" in analysis_lower or "medium" in analysis_lower:
            confidence = 0.6
        elif "low confidence" in analysis_lower or "low" in analysis_lower:
            confidence = 0.4
        elif "no trade" in analysis_lower:
            confidence = 0.2
            
        return {
            "trade_recommendation": analysis,
            "confidence_score": confidence
        }
    
    async def _compile_analysis(self, state: ChartAnalysisState) -> Dict:
        """Node 5: Compile final analysis."""
        final = f"""
📊 AI CHART ANALYSIS: {state['ticker']} ({state['timeframe']})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 PATTERNS:
{state.get('pattern_analysis', 'N/A')}

📈 TREND:
{state.get('trend_analysis', 'N/A')}

📍 KEY LEVELS:
{state.get('support_resistance', 'N/A')}

💡 RECOMMENDATION:
{state.get('trade_recommendation', 'N/A')}

🎯 AI Confidence: {state.get('confidence_score', 0.5) * 100:.0f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return {"final_analysis": final}
    
    async def analyze_chart(
        self,
        ticker: str,
        chart_base64: str,
        timeframe: str = "5min",
        current_price: float = 0.0,
        signals: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze a chart image using the LangGraph workflow.
        
        Returns:
            Dict with pattern_analysis, trend_analysis, support_resistance,
            trade_recommendation, confidence_score, and final_analysis
        """
        if not self.enabled:
            return {
                "final_analysis": "Chart analysis disabled (no API key)",
                "confidence_score": 0.0,
                "trade_recommendation": "NO TRADE - AI analysis unavailable"
            }
        
        cache_key = f"{ticker}_{timeframe}"
        if cache_key in self.analysis_cache:
            cached = self.analysis_cache[cache_key]
            if (datetime.now().timestamp() - cached['timestamp']) < self.cache_ttl:
                logger.info(f"Using cached analysis for {ticker}")
                return cached['result']
        
        try:
            initial_state: ChartAnalysisState = {
                "ticker": ticker,
                "chart_base64": chart_base64,
                "timeframe": timeframe,
                "current_price": current_price,
                "signals": signals or {},
                "pattern_analysis": "",
                "trend_analysis": "",
                "support_resistance": "",
                "trade_recommendation": "",
                "confidence_score": 0.0,
                "final_analysis": ""
            }
            
            logger.info(f"Starting LangGraph analysis for {ticker}...")
            
            result = await self.graph.ainvoke(initial_state)
            
            self.analysis_cache[cache_key] = {
                "timestamp": datetime.now().timestamp(),
                "result": result
            }
            
            logger.info(f"Completed analysis for {ticker}, confidence: {result.get('confidence_score', 0):.0%}")
            
            return result
            
        except Exception as e:
            logger.error(f"LangGraph analysis error for {ticker}: {e}")
            return {
                "final_analysis": f"Analysis error: {e}",
                "confidence_score": 0.0,
                "trade_recommendation": "NO TRADE - analysis failed"
            }
    
    def analyze_chart_sync(
        self,
        ticker: str,
        chart_base64: str,
        timeframe: str = "5min",
        current_price: float = 0.0,
        signals: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for analyze_chart."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.analyze_chart(ticker, chart_base64, timeframe, current_price, signals)
        )


analyzer_instance: Optional[LangGraphChartAnalyzer] = None

def get_chart_analyzer() -> LangGraphChartAnalyzer:
    """Get or create the singleton chart analyzer instance."""
    global analyzer_instance
    if analyzer_instance is None:
        analyzer_instance = LangGraphChartAnalyzer()
    return analyzer_instance
