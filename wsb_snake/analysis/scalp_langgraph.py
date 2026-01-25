"""
LangGraph Scalping Pattern Analyzer

Specialized LangGraph workflow for detecting 0DTE scalping patterns.
Focuses on:
- VWAP interactions (bounces, reclaims, rejections)
- Momentum surges and exhaustion
- Failed breakout/breakdown traps
- Squeeze fires and volatility expansion
- Quick reversal patterns

Designed for sub-hourly trading windows with 15-30% option gain targets.
"""

import asyncio
from typing import Dict, Any, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from wsb_snake.config import OPENAI_API_KEY
from wsb_snake.utils.logger import get_logger

log = get_logger(__name__)


class ScalpAnalysisState(TypedDict):
    """State for scalp pattern analysis workflow."""
    ticker: str
    chart_base64: str
    timeframe: str
    current_price: float
    vwap: float
    momentum: float
    volume_ratio: float
    detected_pattern: str
    direction: str
    
    vwap_analysis: str
    momentum_analysis: str
    trap_analysis: str
    entry_timing: str
    scalp_recommendation: str
    confidence_score: float
    final_verdict: str


SCALP_ANALYST_PROMPT = """You are an elite 0DTE SPY options scalper with years of experience. 
You specialize in quick intraday trades for 15-30% gains on options within 1-hour windows.

Your expertise areas:
1. VWAP TRADING: Bounces, reclaims, and rejections at VWAP
2. MOMENTUM: Identifying surge entries and exhaustion exits
3. TRAP DETECTION: Failed breakouts/breakdowns that reverse quickly
4. TIMING: Optimal entry points for maximum R:R

For SPY 0DTE scalps:
- A 0.3-0.5% SPY move = ~15-25% option gain
- Entry timing is CRITICAL - seconds matter
- Volume confirmation separates real moves from fakes
- VWAP is the intraday "fair value" - respect it

Be decisive. If the setup is there, call it. If not, say NO TRADE.
Your job is to find the edge, not to guess."""


class ScalpLangGraphAnalyzer:
    """
    Specialized LangGraph analyzer for 0DTE scalping patterns.
    Uses GPT-4o vision for rapid pattern confirmation.
    """
    
    def __init__(self):
        if not OPENAI_API_KEY:
            log.warning("OpenAI API key not set - scalp analysis disabled")
            self.enabled = False
            return
        
        self.enabled = True
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=OPENAI_API_KEY,
            temperature=0.2,  # Lower temp for more decisive answers
            max_tokens=600  # Shorter responses for speed
        )
        
        self.graph = self._build_graph()
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 60  # Shorter cache for scalps
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for scalp analysis."""
        workflow = StateGraph(ScalpAnalysisState)
        
        workflow.add_node("analyze_vwap", self._analyze_vwap)
        workflow.add_node("analyze_momentum", self._analyze_momentum)
        workflow.add_node("detect_traps", self._detect_traps)
        workflow.add_node("timing_entry", self._timing_entry)
        workflow.add_node("final_verdict", self._final_verdict)
        
        workflow.set_entry_point("analyze_vwap")
        workflow.add_edge("analyze_vwap", "analyze_momentum")
        workflow.add_edge("analyze_momentum", "detect_traps")
        workflow.add_edge("detect_traps", "timing_entry")
        workflow.add_edge("timing_entry", "final_verdict")
        workflow.add_edge("final_verdict", END)
        
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
                SystemMessage(content=SCALP_ANALYST_PROMPT),
                message
            ])
            
            return response.content
            
        except Exception as e:
            log.error(f"Scalp vision API error: {e}")
            return f"Analysis unavailable: {e}"
    
    async def _analyze_vwap(self, state: ScalpAnalysisState) -> Dict:
        """Node 1: Analyze VWAP relationship for scalp setup."""
        prompt = f"""SPY {state['timeframe']} chart for scalp analysis.

VWAP ANALYSIS (critical for scalps):
1. Where is price vs VWAP (purple dashed line)?
2. Is this a BOUNCE, RECLAIM, or REJECTION setup?
3. Has price tested VWAP cleanly or multiple times?
4. Is the VWAP slope bullish, bearish, or flat?

Current info: Price ${state['current_price']:.2f}, VWAP ${state['vwap']:.2f}

Answer in 2-3 sentences. Focus on the VWAP interaction quality."""

        analysis = await self._call_vision(state['chart_base64'], prompt)
        return {"vwap_analysis": analysis}
    
    async def _analyze_momentum(self, state: ScalpAnalysisState) -> Dict:
        """Node 2: Analyze momentum for scalp confirmation."""
        prompt = f"""Looking at this SPY chart:

MOMENTUM CHECK:
1. Is momentum building or fading?
2. Are candles getting bigger (conviction) or smaller (exhaustion)?
3. Volume: Is it confirming the move? (Volume ratio: {state['volume_ratio']:.1f}x)
4. Any divergence between price and momentum?

Recent momentum: {state['momentum']:+.2f}%

Answer in 2-3 sentences. Is momentum supporting entry or warning against it?"""

        analysis = await self._call_vision(state['chart_base64'], prompt)
        return {"momentum_analysis": analysis}
    
    async def _detect_traps(self, state: ScalpAnalysisState) -> Dict:
        """Node 3: Detect potential trap setups (failed breakouts/breakdowns)."""
        prompt = f"""Scanning for TRAP patterns on SPY:

TRAP DETECTION:
1. Any recent failed breakout (bull trap)?
2. Any recent failed breakdown (bear trap)?
3. Is price showing rejection candles (wicks)?
4. Any potential stop run that reversed?

These traps create the BEST scalp opportunities when caught early.

Answer in 2-3 sentences. Is there a trap setting up?"""

        analysis = await self._call_vision(state['chart_base64'], prompt)
        return {"trap_analysis": analysis}
    
    async def _timing_entry(self, state: ScalpAnalysisState) -> Dict:
        """Node 4: Optimal entry timing assessment."""
        context = f"""
VWAP: {state.get('vwap_analysis', 'N/A')}
Momentum: {state.get('momentum_analysis', 'N/A')}
Traps: {state.get('trap_analysis', 'N/A')}
Detected pattern: {state['detected_pattern']}
Direction: {state['direction']}
"""
        
        prompt = f"""Based on my analysis: {context}

ENTRY TIMING:
1. Is NOW the right time to enter, or wait for confirmation?
2. What would be the optimal entry trigger?
3. Risk level: LOW/MEDIUM/HIGH?

For 0DTE scalps, timing is everything. We want to enter at the inflection point.

Answer in 2-3 sentences. ENTER NOW, WAIT, or NO TRADE?"""

        analysis = await self._call_vision(state['chart_base64'], prompt)
        return {"entry_timing": analysis}
    
    async def _final_verdict(self, state: ScalpAnalysisState) -> Dict:
        """Node 5: Final scalp verdict with confidence."""
        context = f"""
Pattern: {state['detected_pattern']} ({state['direction']})
VWAP: {state.get('vwap_analysis', '')}
Momentum: {state.get('momentum_analysis', '')}
Traps: {state.get('trap_analysis', '')}
Timing: {state.get('entry_timing', '')}
"""
        
        prompt = f"""FINAL SCALP VERDICT for SPY:

{context}

DECISION:
1. SCALP CALLS, SCALP PUTS, or NO TRADE?
2. CONFIDENCE: LOW (40-60%), MEDIUM (60-80%), HIGH (80%+)?
3. One sentence: Why this trade or why not?

Be decisive. This is for 0DTE options - we need conviction."""

        analysis = await self._call_vision(state['chart_base64'], prompt)
        
        # Extract confidence from response
        confidence = 0.5
        analysis_lower = analysis.lower()
        if "high" in analysis_lower and "confidence" in analysis_lower:
            confidence = 0.85
        elif "medium" in analysis_lower and "confidence" in analysis_lower:
            confidence = 0.65
        elif "low" in analysis_lower and "confidence" in analysis_lower:
            confidence = 0.45
        elif "no trade" in analysis_lower:
            confidence = 0.2
        
        return {
            "scalp_recommendation": analysis,
            "confidence_score": confidence,
            "final_verdict": analysis
        }
    
    async def analyze_scalp(
        self,
        ticker: str,
        chart_base64: str,
        timeframe: str,
        current_price: float,
        vwap: float,
        momentum: float,
        volume_ratio: float,
        detected_pattern: str,
        direction: str
    ) -> Dict[str, Any]:
        """
        Analyze a potential scalp setup using the LangGraph workflow.
        
        Returns:
            Dict with vwap_analysis, momentum_analysis, trap_analysis,
            entry_timing, scalp_recommendation, confidence_score
        """
        if not self.enabled:
            return {
                "scalp_recommendation": "Analysis disabled (no API key)",
                "confidence_score": 0.0,
                "confirms_direction": False
            }
        
        # Check cache
        cache_key = f"{ticker}_{datetime.now().strftime('%H%M')}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if (datetime.now().timestamp() - cached['timestamp']) < self.cache_ttl:
                return cached['result']
        
        try:
            initial_state: ScalpAnalysisState = {
                "ticker": ticker,
                "chart_base64": chart_base64,
                "timeframe": timeframe,
                "current_price": current_price,
                "vwap": vwap,
                "momentum": momentum,
                "volume_ratio": volume_ratio,
                "detected_pattern": detected_pattern,
                "direction": direction,
                "vwap_analysis": "",
                "momentum_analysis": "",
                "trap_analysis": "",
                "entry_timing": "",
                "scalp_recommendation": "",
                "confidence_score": 0.0,
                "final_verdict": ""
            }
            
            log.info(f"Starting LangGraph scalp analysis for {ticker}...")
            
            result = await self.graph.ainvoke(initial_state)
            
            # Determine if AI confirms our direction
            rec = result.get('scalp_recommendation', '').lower()
            confirms = (
                (direction == "long" and "calls" in rec) or
                (direction == "short" and "puts" in rec)
            )
            
            result['confirms_direction'] = confirms
            
            self.cache[cache_key] = {
                "timestamp": datetime.now().timestamp(),
                "result": result
            }
            
            log.info(f"Scalp analysis complete: confidence {result.get('confidence_score', 0):.0%}")
            
            return result
            
        except Exception as e:
            log.error(f"Scalp LangGraph error: {e}")
            return {
                "scalp_recommendation": f"Analysis error: {e}",
                "confidence_score": 0.0,
                "confirms_direction": False
            }
    
    def analyze_scalp_sync(
        self,
        ticker: str,
        chart_base64: str,
        timeframe: str,
        current_price: float,
        vwap: float,
        momentum: float,
        volume_ratio: float,
        detected_pattern: str,
        direction: str
    ) -> Dict[str, Any]:
        """Synchronous wrapper for analyze_scalp."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.analyze_scalp(
                ticker, chart_base64, timeframe, current_price,
                vwap, momentum, volume_ratio, detected_pattern, direction
            )
        )


# Singleton instance
_scalp_analyzer: Optional[ScalpLangGraphAnalyzer] = None


def get_scalp_analyzer() -> ScalpLangGraphAnalyzer:
    """Get the singleton scalp analyzer instance."""
    global _scalp_analyzer
    if _scalp_analyzer is None:
        _scalp_analyzer = ScalpLangGraphAnalyzer()
    return _scalp_analyzer
