"""
Chart Brain - Background AI chart analysis engine
Runs continuously in a separate thread to analyze charts in real-time.
"""
import threading
import queue
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from wsb_snake.analysis.chart_generator import ChartGenerator
from wsb_snake.analysis.langgraph_analyzer import get_chart_analyzer
from wsb_snake.collectors.polygon_enhanced import get_polygon_bars
from wsb_snake.config import UNIVERSE
from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class ChartBrain:
    """
    Background AI chart analysis engine.
    Continuously analyzes charts for all universe tickers and caches results.
    Provides AI-enhanced signal validation when requested.
    """
    
    def __init__(self):
        self.chart_generator = ChartGenerator()
        self.analyzer = get_chart_analyzer()
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = threading.Lock()
        self.analysis_queue = queue.Queue()
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.last_full_scan = 0
        self.scan_interval = 120
        
    def start(self):
        """Start the background analysis thread."""
        if self.running:
            logger.warning("ChartBrain already running")
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._background_loop, daemon=True)
        self.worker_thread.start()
        logger.info("🧠 ChartBrain started - background AI analysis active")
    
    def stop(self):
        """Stop the background analysis thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("ChartBrain stopped")
    
    def _background_loop(self):
        """Main background loop that continuously analyzes charts."""
        logger.info("ChartBrain background loop started")
        
        while self.running:
            try:
                now = time.time()
                if now - self.last_full_scan >= self.scan_interval:
                    self._run_full_scan()
                    self.last_full_scan = now
                
                try:
                    ticker = self.analysis_queue.get(timeout=1)
                    self._analyze_ticker(ticker)
                except queue.Empty:
                    pass
                    
            except Exception as e:
                logger.error(f"ChartBrain loop error: {e}")
                time.sleep(5)
        
        logger.info("ChartBrain background loop ended")
    
    def _run_full_scan(self):
        """Run a full scan of all universe tickers."""
        logger.info(f"ChartBrain: Starting full universe scan ({len(UNIVERSE)} tickers)")
        
        for ticker in UNIVERSE:
            if not self.running:
                break
            try:
                self._analyze_ticker(ticker)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
        
        logger.info("ChartBrain: Full scan complete")
    
    def _analyze_ticker(self, ticker: str) -> Optional[Dict]:
        """Analyze a single ticker's chart."""
        try:
            bars = get_polygon_bars(ticker, timespan="minute", multiplier=5, limit=60)
            
            if not bars or len(bars) < 10:
                logger.warning(f"Insufficient data for {ticker} chart")
                return None
            
            chart_base64 = self.chart_generator.generate_chart(
                ticker=ticker,
                ohlcv_data=bars,
                timeframe="5min"
            )
            
            if not chart_base64:
                return None
            
            analysis = self.analyzer.analyze_chart_sync(
                ticker=ticker,
                chart_base64=chart_base64,
                timeframe="5min",
                current_price=bars[-1].get('c', 0) if bars else 0
            )
            
            with self.cache_lock:
                self.analysis_cache[ticker] = {
                    "timestamp": datetime.now(),
                    "analysis": analysis,
                    "confidence": analysis.get("confidence_score", 0.0),
                    "recommendation": self._extract_direction(analysis.get("trade_recommendation", ""))
                }
            
            logger.info(f"ChartBrain: Analyzed {ticker} - confidence {analysis.get('confidence_score', 0):.0%}")
            return analysis
            
        except Exception as e:
            logger.error(f"ChartBrain analysis error for {ticker}: {e}")
            return None
    
    def _extract_direction(self, recommendation: str) -> str:
        """Extract direction from recommendation text."""
        rec_lower = recommendation.lower()
        if "calls" in rec_lower and "no trade" not in rec_lower:
            return "LONG"
        elif "puts" in rec_lower and "no trade" not in rec_lower:
            return "SHORT"
        else:
            return "NEUTRAL"
    
    def get_analysis(self, ticker: str) -> Optional[Dict]:
        """Get cached analysis for a ticker."""
        with self.cache_lock:
            return self.analysis_cache.get(ticker)
    
    def get_ai_confidence(self, ticker: str) -> float:
        """Get AI confidence score for a ticker (0.0 to 1.0)."""
        analysis = self.get_analysis(ticker)
        if analysis:
            return analysis.get("confidence", 0.0)
        return 0.5
    
    def get_ai_direction(self, ticker: str) -> str:
        """Get AI recommended direction for a ticker."""
        analysis = self.get_analysis(ticker)
        if analysis:
            return analysis.get("recommendation", "NEUTRAL")
        return "NEUTRAL"
    
    def validate_signal(
        self,
        ticker: str,
        signal_direction: str,
        signal_score: float
    ) -> Dict[str, Any]:
        """
        Validate an algorithmic signal against AI chart analysis.
        
        Returns:
            Dict with:
                - ai_agrees: bool
                - ai_confidence: float
                - ai_direction: str
                - adjusted_score: float
                - analysis_summary: str
        """
        analysis = self.get_analysis(ticker)
        
        if not analysis:
            return {
                "ai_agrees": True,
                "ai_confidence": 0.5,
                "ai_direction": "NEUTRAL",
                "adjusted_score": signal_score,
                "analysis_summary": "No AI analysis available"
            }
        
        ai_direction = analysis.get("recommendation", "NEUTRAL")
        ai_confidence = analysis.get("confidence", 0.5)
        
        ai_agrees = (
            (signal_direction == "LONG" and ai_direction == "LONG") or
            (signal_direction == "SHORT" and ai_direction == "SHORT") or
            ai_direction == "NEUTRAL"
        )
        
        if ai_agrees and ai_confidence > 0.7:
            adjusted_score = min(100, signal_score * 1.15)
        elif ai_agrees:
            adjusted_score = signal_score
        elif ai_confidence > 0.7:
            adjusted_score = signal_score * 0.7
        else:
            adjusted_score = signal_score * 0.85
        
        full_analysis = analysis.get("analysis", {})
        summary = full_analysis.get("trade_recommendation", "N/A") if isinstance(full_analysis, dict) else str(full_analysis)[:200]
        
        return {
            "ai_agrees": ai_agrees,
            "ai_confidence": ai_confidence,
            "ai_direction": ai_direction,
            "adjusted_score": adjusted_score,
            "analysis_summary": summary
        }
    
    def request_priority_analysis(self, ticker: str):
        """Request priority analysis for a ticker (adds to front of queue)."""
        self.analysis_queue.put(ticker)
    
    def get_all_analyses(self) -> Dict[str, Dict]:
        """Get all cached analyses."""
        with self.cache_lock:
            return dict(self.analysis_cache)
    
    def get_high_confidence_tickers(self, min_confidence: float = 0.7) -> List[str]:
        """Get tickers with high AI confidence."""
        with self.cache_lock:
            return [
                ticker for ticker, data in self.analysis_cache.items()
                if data.get("confidence", 0) >= min_confidence
            ]


chart_brain_instance: Optional[ChartBrain] = None

def get_chart_brain() -> ChartBrain:
    """Get or create the singleton ChartBrain instance."""
    global chart_brain_instance
    if chart_brain_instance is None:
        chart_brain_instance = ChartBrain()
    return chart_brain_instance
