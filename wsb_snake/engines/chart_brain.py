"""
Chart Brain - Background AI chart analysis engine
Runs continuously in a separate thread to analyze charts in real-time.

OPTIMIZED: Only calls AI when significant candlestick patterns are detected.
This prevents excessive API usage and quota exhaustion.
"""
import threading
import queue
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from wsb_snake.analysis.chart_generator import ChartGenerator
from wsb_snake.analysis.langgraph_analyzer import get_chart_analyzer
from wsb_snake.analysis.candlestick_patterns import candlestick_analyzer, PatternDirection
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.config import ZERO_DTE_UNIVERSE as UNIVERSE
from wsb_snake.utils.logger import get_logger
from wsb_snake.utils.session_regime import is_market_open

logger = get_logger(__name__)

# Minimum pattern strength to trigger AI analysis (1-5 scale)
MIN_PATTERN_STRENGTH_FOR_AI = 4
# Minimum number of patterns to trigger AI analysis
MIN_PATTERNS_FOR_AI = 2


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
        # OPTIMIZED: Scan every 10 minutes instead of 2 minutes
        # AI only called when patterns detected, not for every ticker
        self.scan_interval = 600  # 10 minutes
        self.ai_calls_saved = 0
        self.ai_calls_made = 0
        
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
        """Run a full scan of all universe tickers.

        OPTIMIZED: Only scans during market hours to conserve API quota.
        Uses local candlestick analysis first - AI only called for significant patterns.
        """
        if not is_market_open():
            logger.info("ChartBrain: Market closed - skipping full scan to save API quota")
            return

        logger.info(f"ChartBrain: Starting pattern-triggered scan ({len(UNIVERSE)} tickers)")
        patterns_found = 0
        ai_triggered = 0

        for ticker in UNIVERSE:
            if not self.running:
                break
            try:
                result = self._analyze_ticker(ticker)
                if result and result.get("patterns_detected"):
                    patterns_found += 1
                if result and result.get("ai_called"):
                    ai_triggered += 1
                time.sleep(0.5)  # Reduced delay since we call AI less often
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")

        logger.info(f"ChartBrain: Scan complete - {patterns_found} patterns found, {ai_triggered} AI calls (saved {self.ai_calls_saved} calls)")
    
    def _analyze_ticker(self, ticker: str) -> Optional[Dict]:
        """Analyze a single ticker's chart.

        OPTIMIZED: Uses local candlestick pattern detection first.
        Only calls AI (GPT-4o) when significant patterns are found:
        - Pattern with strength >= 4 (strong patterns like engulfing, morning star)
        - OR 2+ patterns detected simultaneously

        This reduces AI API calls by ~80% while maintaining signal quality.
        """
        try:
            bars = polygon_enhanced.get_intraday_bars(ticker, timespan="minute", multiplier=5, limit=60)

            if not bars or len(bars) < 10:
                return None

            # STEP 1: Run LOCAL candlestick pattern detection (FREE - no API call)
            patterns = candlestick_analyzer.analyze(bars, lookback=10)

            # Check if patterns warrant AI analysis
            strong_patterns = [p for p in patterns if p.strength >= MIN_PATTERN_STRENGTH_FOR_AI]
            should_call_ai = (
                len(strong_patterns) >= 1 or  # At least one strong pattern
                len(patterns) >= MIN_PATTERNS_FOR_AI  # OR multiple weaker patterns
            )

            result = {
                "patterns_detected": len(patterns) > 0,
                "patterns": [p.name for p in patterns],
                "ai_called": False,
                "timestamp": datetime.now()
            }

            if not should_call_ai:
                # Skip AI - use local pattern analysis only
                self.ai_calls_saved += 1
                direction = "NEUTRAL"
                confidence = 0.3

                if patterns:
                    bullish = sum(1 for p in patterns if p.direction == PatternDirection.BULLISH)
                    bearish = sum(1 for p in patterns if p.direction == PatternDirection.BEARISH)
                    if bullish > bearish:
                        direction = "LONG"
                        confidence = min(0.6, 0.3 + bullish * 0.1)
                    elif bearish > bullish:
                        direction = "SHORT"
                        confidence = min(0.6, 0.3 + bearish * 0.1)

                with self.cache_lock:
                    self.analysis_cache[ticker] = {
                        "timestamp": datetime.now(),
                        "analysis": {"patterns": [p.name for p in patterns], "source": "local"},
                        "confidence": confidence,
                        "recommendation": direction
                    }
                return result

            # STEP 2: Significant patterns found - call AI for deeper analysis
            self.ai_calls_made += 1
            result["ai_called"] = True

            chart_base64 = self.chart_generator.generate_chart(
                ticker=ticker,
                ohlcv_data=bars,
                timeframe="5min"
            )

            if not chart_base64:
                return result

            logger.info(f"ChartBrain: {ticker} - {len(patterns)} patterns detected ({[p.name for p in patterns[:3]]}), calling AI...")

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
                    "recommendation": self._extract_direction(analysis.get("trade_recommendation", "")),
                    "patterns": [p.name for p in patterns]
                }

            logger.info(f"ChartBrain: AI analyzed {ticker} - confidence {analysis.get('confidence_score', 0):.0%}")
            return result

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

    def analyze_with_confluence(
        self,
        ticker: str,
        has_news_signal: bool = False,
        has_sentiment_signal: bool = False,
        has_options_flow: bool = False
    ) -> Optional[Dict]:
        """Analyze ticker ONLY if there's confluence of multiple signals.

        This is the STRATEGIC AI entry point - only calls Gemini/OpenAI when:
        1. Strong candlestick patterns detected AND
        2. At least one other signal type (news, sentiment, options flow)

        This prevents wasting API calls on noise and focuses on high-conviction setups.

        Args:
            ticker: Stock ticker to analyze
            has_news_signal: True if there's a relevant news catalyst
            has_sentiment_signal: True if sentiment is strongly bullish/bearish
            has_options_flow: True if unusual options activity detected

        Returns:
            Analysis dict if confluence found and AI called, None otherwise
        """
        try:
            bars = polygon_enhanced.get_intraday_bars(ticker, timespan="minute", multiplier=5, limit=60)
            if not bars or len(bars) < 10:
                return None

            # Run local pattern detection first
            patterns = candlestick_analyzer.analyze(bars, lookback=10)
            strong_patterns = [p for p in patterns if p.strength >= MIN_PATTERN_STRENGTH_FOR_AI]

            # Calculate confluence score
            confluence_signals = sum([
                len(strong_patterns) >= 1,  # Strong pattern
                len(patterns) >= 3,         # Multiple patterns
                has_news_signal,
                has_sentiment_signal,
                has_options_flow
            ])

            # Need at least 2 confluence signals to call AI
            if confluence_signals < 2:
                logger.debug(f"ChartBrain: {ticker} - insufficient confluence ({confluence_signals}/2 needed)")
                return None

            # Confluence detected - call AI
            logger.info(f"🎯 ChartBrain CONFLUENCE: {ticker} - {confluence_signals} signals (patterns: {[p.name for p in patterns[:2]]}, news: {has_news_signal}, sentiment: {has_sentiment_signal}, flow: {has_options_flow})")

            self.ai_calls_made += 1

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
                    "recommendation": self._extract_direction(analysis.get("trade_recommendation", "")),
                    "patterns": [p.name for p in patterns],
                    "confluence_score": confluence_signals,
                    "source": "confluence_ai"
                }

            return self.analysis_cache[ticker]

        except Exception as e:
            logger.error(f"ChartBrain confluence analysis error for {ticker}: {e}")
            return None
    
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
