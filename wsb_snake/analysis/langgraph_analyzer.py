"""
LangGraph AI Chart Analyzer - Uses GPT-4o vision to analyze candlestick charts
and provide pattern recognition, trend analysis, and trade recommendations.
"""
import asyncio
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from wsb_snake.config import OPENAI_API_KEY
from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


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
        """Call GPT-4o vision with the chart image."""
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
            
            return response.content
            
        except Exception as e:
            logger.error(f"Vision API error: {e}")
            return f"Analysis unavailable: {e}"
    
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
