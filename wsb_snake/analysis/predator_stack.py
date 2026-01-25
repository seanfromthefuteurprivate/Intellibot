"""
Predator Stack - Multi-model AI analyzer for apex predator scalping.

Uses Gemini (primary) + DeepSeek (fallback) + GPT (confirmation) for
surgical precision pattern recognition on 0DTE setups.
"""
import os
import asyncio
import base64
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import httpx

from wsb_snake.utils.logger import get_logger

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


class PredatorStack:
    """
    Multi-model apex predator analyzer.
    
    Primary: Gemini 2.0 Flash (fast, cost-effective, excellent vision)
    Fallback: DeepSeek (budget backup)
    Confirmation: GPT-4o (for high-stakes validation)
    """
    
    def __init__(self):
        self.gemini_key = os.environ.get('GEMINI_API_KEY')
        self.deepseek_key = os.environ.get('DEEPSEEK_API_KEY')
        self.openai_key = os.environ.get('OPENAI_API_KEY')
        
        self.gemini_available = bool(self.gemini_key)
        self.deepseek_available = bool(self.deepseek_key)
        self.openai_available = bool(self.openai_key)
        
        logger.info(f"Predator Stack initialized - Gemini: {self.gemini_available}, "
                   f"DeepSeek: {self.deepseek_available}, OpenAI: {self.openai_available}")
        
        self.scalp_system_prompt = """You are an APEX PREDATOR scalping analyst. Your job is to analyze charts 
for 0DTE SPY options scalping opportunities with SURGICAL PRECISION.

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

    async def _call_gemini(
        self,
        image_base64: str,
        context: str
    ) -> Optional[str]:
        """Call Gemini API for vision analysis with robust error handling."""
        if not self.gemini_key:
            logger.debug("Gemini API key not available")
            return None
        
        if not image_base64:
            logger.warning("Empty image provided to Gemini")
            return None
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_key}"
            
            payload = {
                "contents": [{
                    "parts": [
                        {"text": f"{self.scalp_system_prompt}\n\nCONTEXT: {context}"},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_base64
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 500
                }
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                if "candidates" in data and data["candidates"]:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                    
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
        
        return None
    
    async def _call_deepseek(
        self,
        image_base64: str,
        context: str
    ) -> Optional[str]:
        """Call DeepSeek API for vision analysis (fallback)."""
        if not self.deepseek_key:
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
                    {"role": "user", "content": [
                        {"type": "text", "text": f"CONTEXT: {context}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                data = response.json()
                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"]
                    
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
        
        return None
    
    async def _call_openai(
        self,
        image_base64: str,
        context: str
    ) -> Optional[str]:
        """Call OpenAI GPT-4o for high-stakes confirmation."""
        if not self.openai_key:
            return None
        
        try:
            url = "https://api.openai.com/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": self.scalp_system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"CONTEXT: {context}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                
                data = response.json()
                if "choices" in data and data["choices"]:
                    return data["choices"][0]["message"]["content"]
                    
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
        chart_base64: str,
        ticker: str = "SPY",
        pattern: str = "",
        current_price: float = 0.0,
        vwap: float = 0.0,
        extra_context: str = "",
        require_confirmation: bool = False
    ) -> PredatorAnalysis:
        """
        Analyze chart with multi-model predator stack.
        
        Args:
            chart_base64: Base64 encoded chart image
            ticker: Stock ticker
            pattern: Detected pattern name
            current_price: Current price
            vwap: Current VWAP value
            extra_context: Additional context (order flow, etc.)
            require_confirmation: If True, get second opinion from GPT
            
        Returns:
            PredatorAnalysis with verdict and reasoning
        """
        context = f"Ticker: {ticker} | Pattern: {pattern} | Price: ${current_price:.2f} | VWAP: ${vwap:.2f}"
        if extra_context:
            context += f"\n{extra_context}"
        
        primary_response = await self._call_gemini(chart_base64, context)
        model_used = "gemini-2.0-flash"
        
        if not primary_response and self.deepseek_available:
            logger.info("Gemini unavailable, falling back to DeepSeek")
            primary_response = await self._call_deepseek(chart_base64, context)
            model_used = "deepseek"
        
        if not primary_response:
            logger.warning("All primary models failed")
            return PredatorAnalysis(
                verdict=PredatorVerdict.ABORT,
                confidence=0.0,
                entry_quality="POOR",
                vwap_analysis="Analysis unavailable",
                delta_analysis="Analysis unavailable",
                trap_risk="UNKNOWN",
                timing_score=0,
                reasoning="AI models unavailable - aborting for safety",
                model_used="none"
            )
        
        analysis = self._parse_response(primary_response, model_used)
        
        if require_confirmation and analysis.confidence >= 70 and analysis.verdict in [
            PredatorVerdict.STRIKE_CALLS, PredatorVerdict.STRIKE_PUTS
        ]:
            logger.info(f"Seeking GPT confirmation for {analysis.verdict.value}")
            confirm_response = await self._call_openai(chart_base64, context)
            
            if confirm_response:
                confirm_analysis = self._parse_response(confirm_response, "gpt-4o")
                
                if confirm_analysis.verdict == analysis.verdict:
                    analysis.confirmed_by = "gpt-4o"
                    analysis.confidence = min(100, analysis.confidence + 10)
                    logger.info(f"GPT CONFIRMED: {analysis.verdict.value}")
                else:
                    analysis.confidence = max(0, analysis.confidence - 20)
                    analysis.reasoning += f" [GPT disagrees: {confirm_analysis.verdict.value}]"
                    logger.warning(f"GPT DISAGREEMENT: GPT says {confirm_analysis.verdict.value}")
        
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
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.analyze(chart_base64, ticker, pattern, current_price, vwap, extra_context, require_confirmation)
                    )
                    return future.result(timeout=60)
            else:
                return loop.run_until_complete(
                    self.analyze(chart_base64, ticker, pattern, current_price, vwap, extra_context, require_confirmation)
                )
        except Exception as e:
            logger.error(f"Sync analyze error: {e}")
            return PredatorAnalysis(
                verdict=PredatorVerdict.ABORT,
                confidence=0.0,
                entry_quality="POOR",
                vwap_analysis="Error",
                delta_analysis="Error",
                trap_risk="UNKNOWN",
                timing_score=0,
                reasoning=f"Analysis error: {str(e)}",
                model_used="none"
            )


predator_stack = PredatorStack()
