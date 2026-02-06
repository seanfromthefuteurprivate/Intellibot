"""
SPY 0DTE Scalper - Hawk-like pattern detection for quick 15-30% gains

This engine focuses exclusively on SPY 0DTE scalping opportunities by:
1. Monitoring SPY every 30 seconds during market hours
2. Detecting intraday scalping patterns (VWAP bounces, breakouts, reversals)
3. Using LangGraph for AI-powered pattern confirmation
4. Learning from past trades to improve pattern recognition
5. Sending immediate entry/exit alerts to Telegram

Target: 15-30% quick gains on 0DTE options, even in 1-hour windows
"""

import threading
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from wsb_snake.utils.logger import get_logger
from wsb_snake.utils.session_regime import is_market_open, get_session_info
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.collectors.scalp_data_collector import scalp_data_collector, ScalpDataPacket
from wsb_snake.collectors.finnhub_collector import finnhub_collector
from wsb_snake.collectors.alpaca_stream import alpaca_stream
from wsb_snake.analysis.scalp_langgraph import get_scalp_analyzer
from wsb_snake.analysis.chart_generator import ChartGenerator
from wsb_snake.analysis.scalp_chart_generator import scalp_chart_generator
from wsb_snake.analysis.predator_stack import predator_stack, PredatorVerdict
from wsb_snake.learning.pattern_memory import pattern_memory
from wsb_snake.learning.time_learning import time_learning
from wsb_snake.learning.stalking_mode import stalking_mode, StalkState
from wsb_snake.learning.zero_greed_exit import zero_greed_exit
from wsb_snake.learning.session_learnings import session_learnings, battle_plan
from wsb_snake.learning.trade_learner import trade_learner
from wsb_snake.notifications.telegram_bot import send_alert as send_telegram_alert
from wsb_snake.db.database import get_connection
from wsb_snake.trading.alpaca_executor import alpaca_executor
from wsb_snake.config import ZERO_DTE_UNIVERSE, DAILY_0DTE_TICKERS
from wsb_snake.utils.sector_strength import is_sector_slighted_down
from wsb_snake.utils.session_regime import (
    get_market_regime_info,
    RegimeType,
)

log = get_logger(__name__)


class ScalpPattern(Enum):
    """Types of scalping patterns we detect"""
    VWAP_BOUNCE = "vwap_bounce"           # Price bounces off VWAP
    VWAP_RECLAIM = "vwap_reclaim"         # Price reclaims VWAP from below
    VWAP_REJECTION = "vwap_rejection"      # Price rejected at VWAP from above
    MOMENTUM_SURGE = "momentum_surge"      # Strong directional move with volume
    BREAKOUT = "breakout"                  # Breaks above resistance
    BREAKDOWN = "breakdown"                # Breaks below support
    REVERSAL = "reversal"                  # Trend reversal pattern
    SQUEEZE_FIRE = "squeeze_fire"          # Volatility squeeze release
    FAILED_BREAKDOWN = "failed_breakdown"  # Bear trap - quick recovery
    FAILED_BREAKOUT = "failed_breakout"    # Bull trap - quick rejection


@dataclass
class ScalpSetup:
    """A detected scalping opportunity"""
    pattern: ScalpPattern
    direction: str  # "long" or "short"
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float  # 0-100
    ai_confirmed: bool = False
    ai_confidence: float = 0.0
    pattern_memory_boost: float = 0.0
    time_quality_score: float = 0.0
    detected_at: datetime = field(default_factory=datetime.utcnow)
    vwap: float = 0.0
    volume_ratio: float = 1.0
    momentum: float = 0.0
    notes: str = ""


class SPYScalper:
    """
    Apex predator 0DTE SPY scalper - strikes only on highest-conviction setups.
    Runs continuously during market hours, scanning every 30 seconds.
    
    PREDATOR MODE: No failed attempts. Only strike when prey is in sight.
    Swoop in, execute, exit smoothly. No waiting for volatility.
    """
    
    # ========== APEX PREDATOR CONFIGURATION - JAN 29 FIX ==========
    # PROBLEM DIAGNOSED: 25% win rate, 0DTE theta killing all positions
    #
    # ROOT CAUSES:
    # 1. 0DTE theta decay destroys gains even on winning direction
    # 2. Win rate too low - taking too many marginal setups
    # 3. Stops too wide, targets unreachable before theta kills
    #
    # FIX: QUALITY OVER QUANTITY
    # - Raise confidence thresholds SIGNIFICANTLY
    # - Require AI confirmation to filter bad setups
    # - Only trade A+ setups (85%+ confidence)
    PREDATOR_MODE = True  # Enable apex predator behavior
    SNIPER_MODE = True  # Only call AI for the BEST setup per scan cycle
    MIN_CONFIDENCE_FOR_AI = 80  # RAISED TO 80% - Only analyze VERY HIGH quality setups
    MIN_CONFIDENCE_FOR_ALERT = 85  # RAISED TO 85% - Only trade A+ setups (was 75)
    MIN_SWEEP_PCT_FOR_FLOW = 8  # RAISED - Require stronger order flow agreement
    REQUIRE_AI_CONFIRMATION = True  # RE-ENABLED - Filter out bad setups
    REQUIRE_PREDATOR_STRIKE = False  # Keep disabled - AI confirmation enough
    HIGH_CONFIDENCE_AUTO_EXECUTE = 90  # Only auto-execute at 90%+ (near perfect setup)
    MAX_AI_CALLS_PER_HOUR = 30  # Limit AI calls to prevent abuse
    # =================================================
    
    # ========== SMALL CAP STRICT MODE - JAN 29 FIX ==========
    # Small caps are CHOPPY and unpredictable - most losses came from small caps
    # PROBLEM: Small cap 0DTE options = double theta + volatility risk
    #
    # FIX: VERY HIGH confidence required for small caps
    # Prefer EQUITIES over options for small caps (no theta decay!)
    SMALL_CAP_TICKERS = [
        "THH", "RKLB", "ASTS", "NBIS", "PL", "LUNR", "ONDS", "SLS",
        "POET", "ENPH", "USAR", "PYPL"
    ]
    MIN_CONFIDENCE_SMALL_CAP = 88  # RAISED TO 88% for small caps (was 75)
    SMALL_CAP_REQUIRE_CANDLESTICK = True  # Must have clear candlestick pattern
    SMALL_CAP_PREFER_EQUITY = True  # NEW: Prefer stock over options for small caps
    
    # Candlestick patterns we trust for small cap rallies
    BULLISH_CANDLESTICK_PATTERNS = [
        "hammer", "inverted_hammer", "bullish_engulfing", "morning_star",
        "three_white_soldiers", "piercing_line", "bullish_harami"
    ]
    BEARISH_CANDLESTICK_PATTERNS = [
        "hanging_man", "shooting_star", "bearish_engulfing", "evening_star",
        "three_black_crows", "dark_cloud_cover", "bearish_harami"
    ]
    # ============================================
    
    def __init__(self):
        self.symbol = "SPY"
        self.scan_interval = 30  # seconds between scans
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.chart_generator = ChartGenerator()
        self.scalp_chart_generator = scalp_chart_generator  # Surgical precision charts
        self.scalp_analyzer = get_scalp_analyzer()  # LangGraph analyzer
        self.predator_stack = predator_stack  # Multi-model AI (Gemini + DeepSeek + GPT)
        self.scalp_data = scalp_data_collector  # Ultra-fast data collector
        
        # Recent price data cache
        self.price_cache: List[Dict] = []
        self.last_vwap = 0.0
        self.last_price = 0.0
        self.last_data_packet: Optional[ScalpDataPacket] = None
        
        # Pattern detection state
        self.active_setup: Optional[ScalpSetup] = None
        self.cooldown_until: Optional[datetime] = None
        self.trade_cooldown_minutes = 20  # JAN 29 FIX: 20 min cooldown - stop overtrading!
        
        # Statistics
        self.signals_today = 0
        self.entries_today = 0
        self.exits_today = 0
        
        # Alpaca stream for real-time data
        self.alpaca_stream = alpaca_stream
        self._stream_started = False
        
        # Cached classified trades to avoid rate limit issues
        self._classified_trades_cache: Optional[Dict] = None
        self._classified_trades_time: Optional[datetime] = None
        self._classified_trades_ttl = 30  # 30 second cache
        
        # Initialize database table
        self._init_db()
        
        log.info("SPY 0DTE Scalper initialized - hawk mode activated")
    
    def _init_db(self):
        """Initialize scalper tracking table."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spy_scalp_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT,
                direction TEXT,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                confidence REAL,
                ai_confirmed INTEGER,
                ai_confidence REAL,
                detected_at TEXT,
                alerted_at TEXT,
                outcome TEXT,
                pnl_pct REAL,
                duration_minutes INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def start(self):
        """Start the scalper background thread."""
        if self.running:
            log.warning("SPY Scalper already running")
            return
        
        self.running = True
        
        # Start Alpaca WebSocket for real-time data (UNHINGED: SPY + QQQ + IWM 0DTE)
        if not self._stream_started:
            try:
                stream_symbols = list(set([self.symbol] + [t for t in DAILY_0DTE_TICKERS if t in ("SPY", "QQQ", "IWM")]))
                self.alpaca_stream.subscribe(stream_symbols, trades=True, quotes=True, bars=True)
                self.alpaca_stream.on_halt(self._on_halt_callback)
                self.alpaca_stream.on_luld(self._on_luld_callback)
                self.alpaca_stream.start()
                self._stream_started = True
                log.info(f"Started Alpaca real-time stream for {stream_symbols}")
            except Exception as e:
                log.warning(f"Alpaca stream start failed (will use polling): {e}")
        
        self.worker_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.worker_thread.start()
        log.info("🦅 SPY Scalper started - watching for 0DTE opportunities")
    
    def _on_halt_callback(self, halt_info: Dict):
        """Handle trading halt alerts."""
        if halt_info.get("is_halted"):
            log.warning(f"⚠️ TRADING HALT: {halt_info.get('symbol')} - {halt_info.get('status_message')}")
            # Could send Telegram alert here for halt
    
    def _on_luld_callback(self, luld_info: Dict):
        """Handle LULD band updates for volatility detection."""
        log.debug(f"LULD update: {luld_info.get('symbol')} up={luld_info.get('limit_up')} down={luld_info.get('limit_down')}")
    
    def stop(self):
        """Stop the scalper."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        if self._stream_started:
            try:
                self.alpaca_stream.stop()
            except:
                pass
        log.info("SPY Scalper stopped")
    
    def _get_classified_trades_cached(self) -> Dict:
        """Get classified trades with caching to avoid rate limits."""
        now = datetime.now()
        
        if (self._classified_trades_cache and self._classified_trades_time and 
            (now - self._classified_trades_time).total_seconds() < self._classified_trades_ttl):
            return self._classified_trades_cache
        
        try:
            classified = polygon_enhanced.get_classified_trades(self.symbol, limit=100)
            self._classified_trades_cache = classified
            self._classified_trades_time = now
            return classified
        except Exception as e:
            log.debug(f"Classified trades fetch error: {e}")
            return self._classified_trades_cache or {"available": False}
    
    def _flow_agrees_with_setup(self, ticker: str, direction: str) -> bool:
        """Order flow hard filter: require sweep_direction to agree with setup and sweep_pct >= MIN_SWEEP_PCT."""
        try:
            flow = polygon_enhanced.get_classified_trades(ticker, limit=100)
            if not flow.get("available"):
                return True  # Allow when no flow data (avoid blocking on API failure)
            sweep_dir = flow.get("sweep_direction", "NONE")
            sweep_pct = flow.get("sweep_pct", 0) or 0
            if sweep_dir == "NONE" or sweep_pct < self.MIN_SWEEP_PCT_FOR_FLOW:
                return False
            if direction == "long" and sweep_dir != "BUY":
                return False
            if direction == "short" and sweep_dir != "SELL":
                return False
            return True
        except Exception as e:
            log.debug(f"Flow check error for {ticker}: {e}")
            return True  # Allow on error to avoid blocking
    
    def _scan_loop(self):
        """Main scanning loop - runs every 30 seconds during market hours.

        SNIPER MODE: Collects all setups first, then only calls AI for the BEST one.
        This reduces API calls from potentially 29 per cycle to 1.
        """
        log.info("SPY Scalper scan loop started")
        log.info(f"Monitoring {len(ZERO_DTE_UNIVERSE)} tickers: {', '.join(ZERO_DTE_UNIVERSE)}")
        log.info(f"SNIPER MODE: {'ENABLED' if self.SNIPER_MODE else 'DISABLED'} - AI calls limited")

        while self.running:
            try:
                if is_market_open():
                    if self.SNIPER_MODE:
                        # SNIPER MODE: Collect all setups, only AI analyze the best
                        self._run_sniper_scan()
                    else:
                        # Legacy mode: Scan each ticker individually
                        for ticker in ZERO_DTE_UNIVERSE:
                            try:
                                self._run_scan_for_ticker(ticker)
                            except Exception as te:
                                log.debug(f"Error scanning {ticker}: {te}")
                else:
                    # During off-hours, check less frequently
                    time.sleep(60)
                    continue

                time.sleep(self.scan_interval)

            except Exception as e:
                log.error(f"Scalper scan error: {e}")
                time.sleep(10)

        log.info("Scalper scan loop ended")

    def _run_sniper_scan(self):
        """SNIPER MODE: Scan all tickers, but only call AI for the single best setup.

        This dramatically reduces API calls while still finding the best trades.
        """
        all_setups = []

        # Phase 1: Quick scan all tickers WITHOUT AI
        for ticker in ZERO_DTE_UNIVERSE:
            try:
                setup = self._quick_scan_ticker(ticker)
                if setup:
                    all_setups.append((ticker, setup))
            except Exception as e:
                log.debug(f"Quick scan error for {ticker}: {e}")

        if not all_setups:
            log.debug("No setups found this cycle")
            return

        # Phase 2: Sort by confidence and only AI-analyze the top 1-2
        all_setups.sort(key=lambda x: x[1].confidence, reverse=True)

        # Only process top setup(s) that meet threshold
        top_setups = [(t, s) for t, s in all_setups[:2] if s.confidence >= self.MIN_CONFIDENCE_FOR_AI]

        if not top_setups:
            log.debug(f"Best setup {all_setups[0][0]}@{all_setups[0][1].confidence:.0f}% below threshold ({self.MIN_CONFIDENCE_FOR_AI}%)")
            return

        log.info(f"🎯 SNIPER: {len(top_setups)} setup(s) qualify for AI analysis (of {len(all_setups)} detected)")

        # Phase 3: AI analyze only the best setup(s)
        for ticker, setup in top_setups:
            # Set context for AI
            self.last_vwap = setup.vwap
            self.last_price = setup.entry_price

            # Get AI confirmation
            setup = self._get_ai_confirmation(setup, ticker)

            # Apply learning boosts
            bars = self.price_cache
            setup = self._apply_learning_boosts(setup, bars)

            final_confidence = setup.confidence + setup.pattern_memory_boost + setup.time_quality_score

            # Sector slighted down? – skip new scalp (UNHINGED: pause when SPY weak)
            if is_sector_slighted_down():
                log.info(f"⏸️ Sector slighted down – skipping scalp on {ticker}")
                return
            # Earnings within 2d – skip buy (IV crush risk)
            earnings_check = finnhub_collector.is_earnings_soon(ticker, days=2)
            if earnings_check.get("has_earnings"):
                log.info(f"⏸️ Earnings within 2d – skip buy on {ticker} (IV crush risk)")
                return
            # Order flow hard filter: sweep direction must agree with setup
            if not self._flow_agrees_with_setup(ticker, setup.direction):
                log.info(f"⏸️ Flow disagrees – skipping scalp on {ticker} (sweep direction or pct)")
                return
            # Regime gate: skip or penalize in chop; only allow long in trend_up, short in trend_down
            regime_info = get_market_regime_info(ticker)
            if regime_info.get("is_chop"):
                final_confidence -= regime_info.get("confidence_penalty_in_chop", 10)
            if setup.direction == "long" and regime_info.get("regime") == RegimeType.TREND_DOWN.value:
                log.info(f"⏸️ Regime trend_down – skipping long scalp on {ticker}")
                return
            if setup.direction == "short" and regime_info.get("regime") == RegimeType.TREND_UP.value:
                log.info(f"⏸️ Regime trend_up – skipping short scalp on {ticker}")
                return
            # Check if it meets alert threshold
            if final_confidence >= self.MIN_CONFIDENCE_FOR_ALERT:
                log.info(f"🎯 SNIPER STRIKE on {ticker}: {setup.pattern.value} @ {final_confidence:.0f}%")
                self._send_entry_alert(setup, ticker)
                self.signals_today += 1
                # Set cooldown
                setattr(self, f"cooldown_{ticker}", datetime.utcnow() + timedelta(minutes=self.trade_cooldown_minutes))
            else:
                log.info(f"❌ {ticker} {setup.pattern.value} @ {final_confidence:.0f}% below alert threshold")

    def _quick_scan_ticker(self, ticker: str) -> Optional[ScalpSetup]:
        """Quick scan a ticker WITHOUT AI - just pattern detection.

        Returns the best setup for this ticker, or None.
        """
        # Check cooldown
        cooldown_key = f"cooldown_{ticker}"
        ticker_cooldown = getattr(self, cooldown_key, None)
        if ticker_cooldown and datetime.utcnow() < ticker_cooldown:
            return None

        # Get data
        data_packet = self.scalp_data.get_scalp_data(ticker)
        bars = data_packet.bars_1m if data_packet.is_valid else polygon_enhanced.get_intraday_bars(
            ticker, timespan="minute", multiplier=1, limit=60
        )

        if not bars or len(bars) < 20:
            return None

        current_price = data_packet.current_price if data_packet.current_price > 0 else bars[-1].get('c', 0)
        vwap = self._calculate_vwap(bars)

        # Save for later AI use
        self.price_cache = bars

        # Detect patterns (NO AI)
        setups = self._detect_patterns_for_ticker(bars, ticker, current_price, vwap)

        if not setups:
            return None

        # Return best setup
        best = max(setups, key=lambda s: s.confidence)
        log.debug(f"{ticker}: {best.pattern.value} @ {best.confidence:.0f}%")
        return best
    
    def _run_scan(self):
        """Execute a single scan for SPY (legacy method for compatibility)."""
        self._run_scan_for_ticker(self.symbol)
    
    def _run_scan_for_ticker(self, ticker: str):
        """Execute a single scan for scalping opportunities on a specific ticker."""
        # Check cooldown per ticker
        cooldown_key = f"cooldown_{ticker}"
        ticker_cooldown = getattr(self, cooldown_key, None)
        if ticker_cooldown and datetime.utcnow() < ticker_cooldown:
            log.debug(f"{ticker} on cooldown")
            return
        
        # Get comprehensive scalp data packet (includes 5s, 15s, 1m, 5m bars + order flow)
        data_packet = self.scalp_data.get_scalp_data(ticker)
        
        # Use 1-minute bars for pattern detection (most reliable)
        bars = data_packet.bars_1m if data_packet.is_valid else polygon_enhanced.get_intraday_bars(
            ticker, 
            timespan="minute", 
            multiplier=1, 
            limit=60
        )
        
        if not bars or len(bars) < 20:
            log.debug(f"{ticker}: insufficient bars ({len(bars) if bars else 0})")
            return
        
        current_price = data_packet.current_price if data_packet.current_price > 0 else bars[-1].get('c', 0)
        
        # Calculate VWAP
        vwap = self._calculate_vwap(bars)
        
        log.info(f"🔍 {ticker}: ${current_price:.2f} | VWAP ${vwap:.2f} | {len(bars)} bars")
        
        # Set price cache for this ticker so AI confirmation can use it
        self.price_cache = bars
        self.last_vwap = vwap
        self.last_price = current_price
        
        # Detect patterns for this ticker
        setups = self._detect_patterns_for_ticker(bars, ticker, current_price, vwap)
        
        if not setups:
            log.debug(f"{ticker}: no patterns detected")
            return
        
        log.info(f"🎯 {ticker}: {len(setups)} patterns detected")
        
        # Get best setup
        best_setup = max(setups, key=lambda s: s.confidence)
        
        # Apply learning boosts
        best_setup = self._apply_learning_boosts(best_setup, bars)
        
        # Calculate final confidence score
        final_confidence = best_setup.confidence + best_setup.pattern_memory_boost + best_setup.time_quality_score
        
        # ========== APEX PREDATOR DECISION LOGIC ==========
        # Only proceed if base confidence is high enough
        if final_confidence < self.MIN_CONFIDENCE_FOR_AI:
            log.debug(f"{ticker} prey too weak ({final_confidence:.0f}% < {self.MIN_CONFIDENCE_FOR_AI}%) - passing")
            return
        
        log.info(f"🔍 {ticker} detected {best_setup.pattern.value} @ {final_confidence:.0f}% - getting AI confirmation...")
        
        # Get AI confirmation (required in predator mode) - price_cache is already set above
        best_setup = self._get_ai_confirmation(best_setup, ticker)
        
        # Predator mode requirements:
        # 1. Final confidence >= 65%
        # 2. AI must confirm (if REQUIRE_AI_CONFIRMATION is True)
        # 3. Predator Stack must give STRIKE verdict (checked in _get_ai_confirmation)
        
        ai_passed = (not self.REQUIRE_AI_CONFIRMATION) or best_setup.ai_confirmed
        
        # ========== SMALL CAP STRICT MODE ==========
        # Small caps require HIGHER confidence + CANDLESTICK confirmation
        is_small_cap = ticker in self.SMALL_CAP_TICKERS
        
        if is_small_cap:
            # Higher confidence threshold for small caps
            min_confidence = self.MIN_CONFIDENCE_SMALL_CAP
            confidence_passed = final_confidence >= min_confidence
            
            # Detect candlestick patterns for small caps
            candlestick_pattern = self._detect_candlestick_pattern(bars)
            candlestick_passed = candlestick_pattern is not None
            
            if candlestick_pattern:
                log.info(f"🕯️ {ticker} CANDLESTICK: {candlestick_pattern} detected")
                # Boost confidence if strong candlestick pattern
                final_confidence += 5
            else:
                log.info(f"⚠️ {ticker} SMALL CAP: No clear candlestick pattern - STRICT mode requires confirmation")
            
            should_alert = ai_passed and confidence_passed and candlestick_passed
        else:
            # Standard ETF/mega-cap rules
            min_confidence = self.MIN_CONFIDENCE_FOR_ALERT
            confidence_passed = final_confidence >= min_confidence
            candlestick_passed = True  # Not required for ETFs
            candlestick_pattern = None
            should_alert = ai_passed and confidence_passed
        # =============================================
        
        if should_alert:
            if is_sector_slighted_down():
                log.info(f"⏸️ Sector slighted down – skipping scalp on {ticker}")
                return
            earnings_check = finnhub_collector.is_earnings_soon(ticker, days=2)
            if earnings_check.get("has_earnings"):
                log.info(f"⏸️ Earnings within 2d – skip buy on {ticker} (IV crush risk)")
                return
            if not self._flow_agrees_with_setup(ticker, best_setup.direction):
                log.info(f"⏸️ Flow disagrees – skipping scalp on {ticker} (sweep direction or pct)")
                return
            regime_info = get_market_regime_info(ticker)
            if regime_info.get("is_chop"):
                final_confidence -= regime_info.get("confidence_penalty_in_chop", 10)
            if best_setup.direction == "long" and regime_info.get("regime") == RegimeType.TREND_DOWN.value:
                log.info(f"⏸️ Regime trend_down – skipping long scalp on {ticker}")
                return
            if best_setup.direction == "short" and regime_info.get("regime") == RegimeType.TREND_UP.value:
                log.info(f"⏸️ Regime trend_up – skipping short scalp on {ticker}")
                return
            if final_confidence < self.MIN_CONFIDENCE_FOR_ALERT:
                log.info(f"⏸️ Confidence after regime penalty {final_confidence:.0f}% below threshold")
                return
            pattern_note = f" [Candlestick: {candlestick_pattern}]" if candlestick_pattern else ""
            log.info(f"🎯 APEX PREDATOR STRIKE on {ticker}: {best_setup.pattern.value} @ {final_confidence:.0f}% confidence{pattern_note}")
            self._send_entry_alert(best_setup, ticker)
            # Set per-ticker cooldown
            setattr(self, f"cooldown_{ticker}", datetime.utcnow() + timedelta(minutes=self.trade_cooldown_minutes))
            self.signals_today += 1
        else:
            reason = []
            if not confidence_passed:
                reason.append(f"confidence {final_confidence:.0f}% < {min_confidence}%")
            if not ai_passed:
                reason.append("AI not confirmed")
            if is_small_cap and not candlestick_passed:
                reason.append("no candlestick pattern (SMALL CAP STRICT)")
            log.info(f"❌ {ticker} passed on {best_setup.pattern.value}: {', '.join(reason)}")
    
    def _detect_candlestick_pattern(self, bars: List[Dict]) -> Optional[str]:
        """
        Detect candlestick patterns for small cap strict mode.
        
        Analyzes the last 3-5 candles for reversal/continuation patterns
        that signal high-probability moves in small caps.
        
        Returns pattern name or None if no clear pattern.
        """
        if len(bars) < 3:
            return None
        
        # Get last 5 candles (or fewer if not available)
        recent = bars[-5:] if len(bars) >= 5 else bars[-3:]
        
        # Helper to get OHLC
        def get_ohlc(bar):
            o = bar.get('o') or bar.get('open') or bar.get('Open') or 0
            h = bar.get('h') or bar.get('high') or bar.get('High') or 0
            l = bar.get('l') or bar.get('low') or bar.get('Low') or 0
            c = bar.get('c') or bar.get('close') or bar.get('Close') or 0
            return o, h, l, c
        
        # Current and previous candles
        curr_o, curr_h, curr_l, curr_c = get_ohlc(recent[-1])
        prev_o, prev_h, prev_l, prev_c = get_ohlc(recent[-2])
        
        if curr_o == 0 or prev_o == 0:
            return None
        
        body = abs(curr_c - curr_o)
        upper_wick = curr_h - max(curr_c, curr_o)
        lower_wick = min(curr_c, curr_o) - curr_l
        range_size = curr_h - curr_l if curr_h > curr_l else 0.01
        
        prev_body = abs(prev_c - prev_o)
        
        # ========== BULLISH PATTERNS ==========
        
        # HAMMER: Small body at top, long lower wick (2x+ body), little upper wick
        if lower_wick >= body * 2 and upper_wick <= body * 0.5 and curr_c > curr_o:
            return "hammer"
        
        # INVERTED HAMMER: Small body at bottom, long upper wick
        if upper_wick >= body * 2 and lower_wick <= body * 0.5:
            return "inverted_hammer"
        
        # BULLISH ENGULFING: Current green candle engulfs previous red candle
        if prev_c < prev_o and curr_c > curr_o:  # Previous red, current green
            if curr_o <= prev_c and curr_c >= prev_o:  # Current engulfs previous
                return "bullish_engulfing"
        
        # PIERCING LINE: Gap down open, closes above 50% of previous red candle
        if prev_c < prev_o and curr_c > curr_o:  # Previous red, current green
            prev_midpoint = (prev_o + prev_c) / 2
            if curr_o < prev_c and curr_c > prev_midpoint:
                return "piercing_line"
        
        # BULLISH HARAMI: Small green inside previous large red
        if prev_c < prev_o and curr_c > curr_o:  # Previous red, current green
            if curr_o > prev_c and curr_c < prev_o and body < prev_body * 0.5:
                return "bullish_harami"
        
        # ========== BEARISH PATTERNS ==========
        
        # SHOOTING STAR: Small body at bottom, long upper wick
        if upper_wick >= body * 2 and lower_wick <= body * 0.5 and curr_c < curr_o:
            return "shooting_star"
        
        # HANGING MAN: Small body at top, long lower wick (like hammer but in uptrend)
        if lower_wick >= body * 2 and upper_wick <= body * 0.5 and curr_c < curr_o:
            return "hanging_man"
        
        # BEARISH ENGULFING: Current red candle engulfs previous green candle
        if prev_c > prev_o and curr_c < curr_o:  # Previous green, current red
            if curr_o >= prev_c and curr_c <= prev_o:  # Current engulfs previous
                return "bearish_engulfing"
        
        # DARK CLOUD COVER: Gap up open, closes below 50% of previous green candle
        if prev_c > prev_o and curr_c < curr_o:  # Previous green, current red
            prev_midpoint = (prev_o + prev_c) / 2
            if curr_o > prev_c and curr_c < prev_midpoint:
                return "dark_cloud_cover"
        
        # BEARISH HARAMI: Small red inside previous large green
        if prev_c > prev_o and curr_c < curr_o:  # Previous green, current red
            if curr_o < prev_c and curr_c > prev_o and body < prev_body * 0.5:
                return "bearish_harami"
        
        # ========== MOMENTUM PATTERNS ==========
        
        # STRONG MOMENTUM CANDLE: Large body with small wicks (>80% body)
        if body / range_size > 0.8:
            if curr_c > curr_o:
                return "strong_bullish_momentum"
            else:
                return "strong_bearish_momentum"
        
        # DOJI: Very small body, indicates indecision but can precede reversal
        if body / range_size < 0.1 and range_size > 0:
            return "doji"
        
        return None
    
    def _calculate_vwap(self, bars: List[Dict]) -> float:
        """Calculate VWAP from bars."""
        total_pv = 0
        total_vol = 0
        
        if bars and len(bars) > 0:
            sample = bars[0]
            log.debug(f"VWAP calc - sample bar keys: {list(sample.keys())}")
        
        for bar in bars:
            high = bar.get('h') or bar.get('high') or bar.get('High') or 0
            low = bar.get('l') or bar.get('low') or bar.get('Low') or 0
            close = bar.get('c') or bar.get('close') or bar.get('Close') or 0
            volume = bar.get('v') or bar.get('volume') or bar.get('Volume') or 0
            
            typical_price = (high + low + close) / 3 if (high + low + close) > 0 else 0
            total_pv += typical_price * volume
            total_vol += volume
        
        vwap = total_pv / total_vol if total_vol > 0 else 0
        if vwap == 0 and bars:
            log.warning(f"VWAP=0! Sample bar: {bars[0] if bars else 'none'}")
        return vwap
    
    def _detect_patterns_for_ticker(self, bars: List[Dict], ticker: str, current_price: float, vwap: float) -> List[ScalpSetup]:
        """Detect patterns for a specific ticker."""
        # Temporarily set context for pattern detection
        old_vwap = self.last_vwap
        old_price = self.last_price
        old_cache = self.price_cache
        
        self.last_vwap = vwap
        self.last_price = current_price
        self.price_cache = bars
        
        try:
            return self._detect_patterns(bars)
        finally:
            # Restore context
            self.last_vwap = old_vwap
            self.last_price = old_price
            self.price_cache = old_cache
    
    def _get_bar_value(self, bar: Dict, key: str) -> float:
        """Get value from bar with flexible key lookup."""
        key_map = {
            'c': ['c', 'close', 'Close'],
            'h': ['h', 'high', 'High'],
            'l': ['l', 'low', 'Low'],
            'o': ['o', 'open', 'Open'],
            'v': ['v', 'volume', 'Volume', 'vol'],
        }
        for k in key_map.get(key, [key]):
            val = bar.get(k)
            if val is not None:
                return float(val)
        return 0.0
    
    def _detect_patterns(self, bars: List[Dict]) -> List[ScalpSetup]:
        """Detect all scalping patterns in current price action."""
        setups = []
        
        if len(bars) < 10:
            return setups
        
        current = bars[-1]
        prev = bars[-2]
        prev2 = bars[-3] if len(bars) > 2 else prev
        
        price = self._get_bar_value(current, 'c')
        high = self._get_bar_value(current, 'h')
        low = self._get_bar_value(current, 'l')
        volume = self._get_bar_value(current, 'v')
        
        prev_close = self._get_bar_value(prev, 'c')
        prev_volume = self._get_bar_value(prev, 'v')
        
        vwap = self.last_vwap
        
        # Calculate averages
        avg_volume = sum(self._get_bar_value(b, 'v') for b in bars[-20:]) / 20
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        # Calculate momentum (price change rate)
        prices_5min = [self._get_bar_value(b, 'c') for b in bars[-5:]]
        if len(prices_5min) >= 2:
            momentum = (prices_5min[-1] - prices_5min[0]) / prices_5min[0] * 100 if prices_5min[0] > 0 else 0
        else:
            momentum = 0
        
        # ATR for stops/targets (simplified)
        atr = self._calculate_atr(bars[-14:])
        
        # === Pattern Detection ===
        
        # 1. VWAP Bounce (Long)
        # JAN 29 FIX: Tighter targets/stops for 0DTE - quick in, quick out
        if low <= vwap * 1.001 and price > vwap and prev_close < vwap:
            # Price tested VWAP from below and bounced
            setups.append(ScalpSetup(
                pattern=ScalpPattern.VWAP_BOUNCE,
                direction="long",
                entry_price=price,
                target_price=price + (atr * 1.2),  # 1.2 ATR target (TIGHTENED for 0DTE)
                stop_loss=vwap - (atr * 0.4),      # 0.4 ATR Below VWAP (TIGHTENED - cut fast)
                confidence=65 + (volume_ratio * 5),
                vwap=vwap,
                volume_ratio=volume_ratio,
                momentum=momentum,
                notes="VWAP bounce - long setup"
            ))
        
        # 2. VWAP Reclaim (Long) - stronger signal
        # JAN 29 FIX: Tighter stops/targets for 0DTE
        if prev_close < vwap and price > vwap * 1.001 and volume_ratio > 1.3:
            setups.append(ScalpSetup(
                pattern=ScalpPattern.VWAP_RECLAIM,
                direction="long",
                entry_price=price,
                target_price=price + (atr * 1.5),  # TIGHTENED: 1.5 ATR target
                stop_loss=vwap - (atr * 0.3),       # TIGHTENED: 0.3 ATR stop
                confidence=70 + (volume_ratio * 5),
                vwap=vwap,
                volume_ratio=volume_ratio,
                momentum=momentum,
                notes="VWAP reclaim with volume - bullish"
            ))
        
        # 3. VWAP Rejection (Short)
        # JAN 29 FIX: Tighter stops/targets for 0DTE
        if high >= vwap * 0.999 and price < vwap and prev_close > vwap:
            setups.append(ScalpSetup(
                pattern=ScalpPattern.VWAP_REJECTION,
                direction="short",
                entry_price=price,
                target_price=price - (atr * 1.2),  # TIGHTENED: 1.2 ATR target
                stop_loss=vwap + (atr * 0.4),      # TIGHTENED: 0.4 ATR stop
                confidence=65 + (volume_ratio * 5),
                vwap=vwap,
                volume_ratio=volume_ratio,
                momentum=momentum,
                notes="VWAP rejection - short setup"
            ))
        
        # 4. Momentum Surge (Long)
        # JAN 29 FIX: Require STRONGER momentum + tighter stops
        if momentum > 0.20 and volume_ratio > 1.8:  # RAISED thresholds
            setups.append(ScalpSetup(
                pattern=ScalpPattern.MOMENTUM_SURGE,
                direction="long",
                entry_price=price,
                target_price=price + (atr * 1.5),  # TIGHTENED: 1.5 ATR target
                stop_loss=price - (atr * 0.5),     # TIGHTENED: 0.5 ATR stop
                confidence=68 + (momentum * 10) + (volume_ratio * 3),
                vwap=vwap,
                volume_ratio=volume_ratio,
                momentum=momentum,
                notes=f"Momentum surge +{momentum:.2f}% with volume"
            ))

        # 5. Momentum Surge (Short)
        # JAN 29 FIX: Require STRONGER momentum + tighter stops
        if momentum < -0.20 and volume_ratio > 1.8:  # RAISED thresholds
            setups.append(ScalpSetup(
                pattern=ScalpPattern.MOMENTUM_SURGE,
                direction="short",
                entry_price=price,
                target_price=price - (atr * 1.5),  # TIGHTENED: 1.5 ATR target
                stop_loss=price + (atr * 0.5),     # TIGHTENED: 0.5 ATR stop
                confidence=68 + (abs(momentum) * 10) + (volume_ratio * 3),
                vwap=vwap,
                volume_ratio=volume_ratio,
                momentum=momentum,
                notes=f"Momentum dump {momentum:.2f}% with volume"
            ))
        
        # 6. Breakout - price makes new high of last 30 bars
        # JAN 29 FIX: Tighter stops/targets, require stronger volume
        highs_30 = [b.get('h', 0) for b in bars[-30:-1] if b.get('h', 0) > 0]
        if highs_30 and price > max(highs_30) and volume_ratio > 1.6:  # RAISED volume requirement
            max_high = max(highs_30)
            # TIGHTENED stop for 0DTE
            stop = max(max_high - (atr * 0.3), price * 0.996)  # 0.4% stop (tightened)
            if stop > 0 and stop < price:
                setups.append(ScalpSetup(
                    pattern=ScalpPattern.BREAKOUT,
                    direction="long",
                    entry_price=price,
                    target_price=price + (atr * 1.5),  # TIGHTENED: 1.5 ATR target
                    stop_loss=stop,
                    confidence=72 + (volume_ratio * 4),
                    vwap=vwap,
                    volume_ratio=volume_ratio,
                    momentum=momentum,
                    notes="30-bar high breakout"
                ))

        # 7. Breakdown
        # JAN 29 FIX: Tighter stops/targets, require stronger volume
        lows_30 = [b.get('l', 0) for b in bars[-30:-1] if b.get('l', 0) > 0]
        if lows_30 and price < min(lows_30) and volume_ratio > 1.6:  # RAISED volume requirement
            min_low = min(lows_30)
            # TIGHTENED stop for 0DTE
            stop = min(min_low + (atr * 0.3), price * 1.004)  # 0.4% stop (tightened)
            if stop > 0 and stop > price:
                setups.append(ScalpSetup(
                    pattern=ScalpPattern.BREAKDOWN,
                    direction="short",
                    entry_price=price,
                    target_price=price - (atr * 1.5),  # TIGHTENED: 1.5 ATR target
                    stop_loss=stop,
                    confidence=72 + (volume_ratio * 4),
                    vwap=vwap,
                    volume_ratio=volume_ratio,
                    momentum=momentum,
                    notes="30-bar low breakdown"
                ))
        
        # 8. Failed Breakdown (Bear Trap - Long)
        # JAN 29 FIX: These are high-quality reversal patterns - tighter stops for 0DTE
        lows_10 = [b.get('l', 0) for b in bars[-10:-3] if b.get('l', 0) > 0]
        if lows_10:
            recent_low = min(lows_10)
            prev_low = prev.get('l', 0)
            if prev_low > 0 and prev_low < recent_low and price > recent_low * 1.002:
                # TIGHTENED stop for 0DTE quick scalps
                stop = max(prev_low - (atr * 0.25), price * 0.995)  # 0.5% stop
                if stop > 0 and stop < price:
                    setups.append(ScalpSetup(
                        pattern=ScalpPattern.FAILED_BREAKDOWN,
                        direction="long",
                        entry_price=price,
                        target_price=price + (atr * 1.5),  # TIGHTENED: 1.5 ATR target
                        stop_loss=stop,
                        confidence=75 + (volume_ratio * 3),
                        vwap=vwap,
                        volume_ratio=volume_ratio,
                        momentum=momentum,
                        notes="Bear trap - failed breakdown recovery"
                    ))

        # 9. Failed Breakout (Bull Trap - Short)
        # JAN 29 FIX: These are high-quality reversal patterns - tighter stops for 0DTE
        highs_10 = [b.get('h', 0) for b in bars[-10:-3] if b.get('h', 0) > 0]
        if highs_10:
            recent_high = max(highs_10)
            prev_high = prev.get('h', 0)
            if prev_high > 0 and prev_high > recent_high and price < recent_high * 0.998:
                # TIGHTENED stop for 0DTE quick scalps
                stop = min(prev_high + (atr * 0.25), price * 1.005)  # 0.5% stop
                if stop > 0 and stop > price:
                    setups.append(ScalpSetup(
                        pattern=ScalpPattern.FAILED_BREAKOUT,
                        direction="short",
                        entry_price=price,
                        target_price=price - (atr * 1.5),  # TIGHTENED: 1.5 ATR target
                        stop_loss=stop,
                        confidence=75 + (volume_ratio * 3),
                        vwap=vwap,
                        volume_ratio=volume_ratio,
                        momentum=momentum,
                        notes="Bull trap - failed breakout rejection"
                    ))
        
        # 10. Squeeze Fire - detect low volatility followed by expansion
        # JAN 29 FIX: Require STRONGER volatility expansion + tighter params
        volatility_recent = self._calculate_volatility(bars[-5:])
        volatility_older = self._calculate_volatility(bars[-15:-5])

        if volatility_older > 0 and volatility_recent / volatility_older > 2.0:  # RAISED: 2x expansion
            # Volatility expanding - these can be powerful moves
            if momentum > 0.1:  # Require minimum momentum
                setups.append(ScalpSetup(
                    pattern=ScalpPattern.SQUEEZE_FIRE,
                    direction="long",
                    entry_price=price,
                    target_price=price + (atr * 2.0),  # TIGHTENED: 2.0 ATR (squeeze can run)
                    stop_loss=price - (atr * 0.5),     # TIGHTENED: 0.5 ATR stop
                    confidence=70 + (momentum * 15),
                    vwap=vwap,
                    volume_ratio=volume_ratio,
                    momentum=momentum,
                    notes="Squeeze fire - volatility expanding bullish"
                ))
            elif momentum < -0.1:  # Require minimum momentum
                setups.append(ScalpSetup(
                    pattern=ScalpPattern.SQUEEZE_FIRE,
                    direction="short",
                    entry_price=price,
                    target_price=price - (atr * 2.0),  # TIGHTENED: 2.0 ATR
                    stop_loss=price + (atr * 0.5),     # TIGHTENED: 0.5 ATR stop
                    confidence=70 + (abs(momentum) * 15),
                    vwap=vwap,
                    volume_ratio=volume_ratio,
                    momentum=momentum,
                    notes="Squeeze fire - volatility expanding bearish"
                ))
        
        return setups
    
    def _calculate_atr(self, bars: List[Dict]) -> float:
        """Calculate Average True Range."""
        if len(bars) < 2:
            return 0.5  # Default for SPY
        
        trs = []
        for i in range(1, len(bars)):
            high = self._get_bar_value(bars[i], 'h')
            low = self._get_bar_value(bars[i], 'l')
            prev_close = self._get_bar_value(bars[i-1], 'c')
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            trs.append(tr)
        
        return sum(trs) / len(trs) if trs else 0.5
    
    def _calculate_volatility(self, bars: List[Dict]) -> float:
        """Calculate price volatility (standard deviation of returns)."""
        if len(bars) < 2:
            return 0
        
        returns = []
        for i in range(1, len(bars)):
            prev_c = bars[i-1].get('c', 1)
            curr_c = bars[i].get('c', 1)
            if prev_c > 0:
                returns.append((curr_c - prev_c) / prev_c)
        
        if not returns:
            return 0
        
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        return variance ** 0.5
    
    def _apply_learning_boosts(self, setup: ScalpSetup, bars: List[Dict]) -> ScalpSetup:
        """Apply learning system boosts to the setup."""
        # 1. Pattern Memory boost
        if len(bars) >= 5:
            price_bars = [
                {
                    'open': b.get('o', 0),
                    'high': b.get('h', 0),
                    'low': b.get('l', 0),
                    'close': b.get('c', 0),
                    'volume': b.get('v', 0)
                }
                for b in bars[-10:]
            ]
            
            # Map scalp pattern to pattern memory type
            pattern_type_map = {
                ScalpPattern.BREAKOUT: "breakout",
                ScalpPattern.SQUEEZE_FIRE: "squeeze",
                ScalpPattern.REVERSAL: "reversal",
                ScalpPattern.FAILED_BREAKDOWN: "reversal",
                ScalpPattern.FAILED_BREAKOUT: "reversal",
                ScalpPattern.MOMENTUM_SURGE: "momentum",
                ScalpPattern.VWAP_BOUNCE: "momentum",
                ScalpPattern.VWAP_RECLAIM: "momentum",
            }
            
            # Find matching patterns from memory
            matches = pattern_memory.find_matching_patterns(
                symbol=self.symbol,
                bars=price_bars,
                rsi=50,  # Simplified - could calculate actual RSI
                vwap_position="above" if setup.entry_price > setup.vwap else "below"
            )
            
            if matches:
                best_match = matches[0]
                similarity = best_match.similarity_score
                
                if similarity >= 60:
                    boost = min(15, similarity * 0.2)  # Max 15 point boost
                    setup.pattern_memory_boost = boost
                    setup.notes += f" | Pattern match: {similarity:.0f}%"
        
        # 2. Time-of-day quality score
        tq = time_learning.get_recommendation()
        quality_score = tq.quality_score if tq else 50
        recommendation = tq.recommendation if tq else "Trade with caution"
        
        if quality_score >= 60:
            setup.time_quality_score = min(10, (quality_score - 50) * 0.2)
            if "Avoid" not in recommendation:
                setup.notes += f" | Time quality: {quality_score:.0f}/100"
        elif quality_score < 40:
            # Penalize during poor trading hours
            setup.confidence -= 10
            setup.notes += f" | Poor trading hour (quality {quality_score:.0f})"

        # 3. Screenshot learning boost (from learned winning trades)
        try:
            trade_type = "CALLS" if setup.direction == "long" else "PUTS"
            current_hour = datetime.now().hour
            learning_boost, boost_reasons = trade_learner.get_confidence_adjustment(
                ticker=self.symbol,
                trade_type=trade_type,
                current_hour=current_hour,
                pattern=setup.pattern.value if setup.pattern else None
            )
            if learning_boost > 0:
                # Add to pattern_memory_boost (they stack)
                setup.pattern_memory_boost += learning_boost * 100  # Convert 0.15 to 15 points
                if boost_reasons:
                    setup.notes += f" | Screenshot: {boost_reasons[0][:50]}"
                    log.info(f"Screenshot learning boost +{learning_boost:.0%}: {boost_reasons[0]}")
        except Exception as e:
            log.debug(f"Screenshot learning check failed: {e}")

        return setup
    
    def _get_ai_confirmation(self, setup: ScalpSetup, ticker: str = "SPY") -> ScalpSetup:
        """Get AI confirmation via multi-model Predator Stack (Gemini + DeepSeek + GPT)."""
        try:
            # Use faster 5s/15s bars if available for more granular charts
            chart_data = self.price_cache
            using_fast_bars = False
            if self.last_data_packet and len(self.last_data_packet.bars_15s) >= 30:
                chart_data = self.last_data_packet.bars_15s
                using_fast_bars = True
                log.debug("Using 15s bars for chart generation")
            
            # Generate surgical precision predator chart with VWAP bands, delta, volume profile
            chart_base64 = self.scalp_chart_generator.generate_predator_chart(
                ticker=ticker,
                ohlcv_data=chart_data,
                timeframe="15s" if using_fast_bars else "1min"
            )
            
            if not chart_base64:
                # Fallback to basic chart
                chart_base64 = self.chart_generator.generate_chart(
                    ticker=ticker,
                    ohlcv_data=self.price_cache,
                    timeframe="1min"
                )
            
            if not chart_base64:
                return setup
            
            # Build RUTHLESS enhanced context with ALL data sources
            extra_context = ""
            
            # Order flow data
            if self.last_data_packet and self.last_data_packet.order_flow.get("available"):
                flow = self.last_data_packet.order_flow
                extra_context += (
                    f"\nORDER FLOW: {flow.get('flow_signal', 'NEUTRAL')} "
                    f"(score: {flow.get('flow_score', 0)}) | "
                    f"Bid/Ask Imbalance: {flow.get('bid_ask_imbalance', 0):.2f} | "
                    f"Large trades: {flow.get('large_volume_pct', 0):.0f}% of volume"
                )
            
            # Classified trades (sweeps, blocks) for institutional detection - CACHED
            classified = self._get_classified_trades_cached()
            if classified.get("available"):
                extra_context += (
                    f"\nINSTITUTIONAL: Sweeps={classified.get('sweep_count', 0)} "
                    f"({classified.get('sweep_pct', 0):.1f}%), "
                    f"Blocks={classified.get('block_count', 0)} "
                    f"({classified.get('block_pct', 0):.1f}%), "
                    f"Sweep Direction: {classified.get('sweep_direction', 'NONE')}"
                )
            
            # Alpaca real-time order flow if stream is active
            if self._stream_started:
                try:
                    stream_flow = self.alpaca_stream.get_order_flow_summary(self.symbol)
                    if stream_flow.get("available"):
                        extra_context += (
                            f"\nREAL-TIME FLOW: {stream_flow.get('flow_signal', 'NEUTRAL')} "
                            f"(buy_ratio: {stream_flow.get('buy_ratio', 0.5):.1%})"
                        )
                        if stream_flow.get("is_halted"):
                            extra_context += " ⚠️ HALTED"
                except:
                    pass
            
            # Finnhub ruthless context - ALL available data
            try:
                # Analyst consensus
                recs = finnhub_collector.get_recommendation_trends(self.symbol)
                if recs.get("available"):
                    extra_context += (
                        f"\nANALYST CONSENSUS: {recs.get('consensus', 'HOLD')} "
                        f"({recs.get('total_analysts', 0)} analysts, "
                        f"{recs.get('bullish_pct', 0.5)*100:.0f}% bullish)"
                    )
                
                # Price targets for context
                targets = finnhub_collector.get_price_target(self.symbol)
                if targets.get("available") and targets.get("target_mean"):
                    target_vs_current = ((targets["target_mean"] - self.last_price) / self.last_price * 100) if self.last_price else 0
                    extra_context += f"\nPRICE TARGET: ${targets.get('target_mean', 0):.2f} ({target_vs_current:+.1f}% from current)"
                
                # Support/resistance levels
                sr_levels = finnhub_collector.get_support_resistance(self.symbol)
                if sr_levels.get("available") and sr_levels.get("levels"):
                    levels = sr_levels["levels"][:5]  # Top 5 levels
                    extra_context += f"\nS/R LEVELS: {', '.join(f'${l:.2f}' for l in levels)}"
            except Exception as e:
                log.debug(f"Finnhub context fetch error: {e}")
            
            # Earnings warning check
            try:
                earnings_check = finnhub_collector.is_earnings_soon(self.symbol, days=2)
                if earnings_check.get("has_earnings"):
                    extra_context += f"\n⚠️ EARNINGS WARNING: {earnings_check.get('warning', '')}"
            except:
                pass
            
            # Use Predator Stack for multi-model AI confirmation
            # Gemini (primary) -> DeepSeek (fallback) -> GPT (confirmation for high confidence)
            analysis = self.predator_stack.analyze_sync(
                chart_base64=chart_base64,
                ticker=self.symbol,
                pattern=setup.pattern.value,
                current_price=self.last_price,
                vwap=setup.vwap,
                extra_context=extra_context,
                require_confirmation=(setup.confidence >= 70)  # Get GPT confirmation for high-confidence setups
            )
            
            if analysis:
                # Map predator verdict to setup
                if analysis.verdict == PredatorVerdict.STRIKE_CALLS:
                    if setup.direction == "long":
                        setup.ai_confirmed = True
                        setup.confidence += 10
                    else:
                        # AI disagrees with direction
                        setup.ai_confirmed = False
                        setup.confidence -= 20
                        
                elif analysis.verdict == PredatorVerdict.STRIKE_PUTS:
                    if setup.direction == "short":
                        setup.ai_confirmed = True
                        setup.confidence += 10
                    else:
                        setup.ai_confirmed = False
                        setup.confidence -= 20
                        
                elif analysis.verdict == PredatorVerdict.ABORT:
                    setup.ai_confirmed = False
                    setup.confidence -= 25
                    
                else:  # NO_TRADE
                    setup.ai_confirmed = False
                    setup.confidence -= 15
                
                setup.ai_confidence = analysis.confidence / 100
                
                # Extra boost if GPT confirmed
                if analysis.confirmed_by:
                    setup.confidence += 5
                    setup.notes += f" | GPT CONFIRMED"
                
                if setup.ai_confirmed:
                    setup.notes += f" | {analysis.model_used.upper()} CONFIRMS ({analysis.confidence:.0f}%)"
                    setup.notes += f" | Entry: {analysis.entry_quality} | Timing: {analysis.timing_score}/100"
                else:
                    setup.notes += f" | AI: {analysis.verdict.value} ({analysis.confidence:.0f}%)"
                    if analysis.trap_risk:
                        setup.notes += f" | Trap risk: {analysis.trap_risk}"
        
        except Exception as e:
            log.error(f"Predator Stack AI confirmation error: {e}")
        
        return setup
    
    def _send_entry_alert(self, setup: ScalpSetup, ticker: str = "SPY"):
        """Send entry alert to Telegram and add to stalking mode."""
        trade_type = "CALLS" if setup.direction == "long" else "PUTS"
        
        # Calculate R:R
        risk = abs(setup.entry_price - setup.stop_loss)
        reward = abs(setup.target_price - setup.entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Expected option move (simplified: 3x leverage)
        underlying_move_pct = reward / setup.entry_price * 100
        expected_option_gain = underlying_move_pct * 3  # Rough 0DTE delta
        
        # Format alert
        alert_msg = f"""
{'='*40}
🦅 {ticker} 0DTE SCALP ALERT 🦅
{'='*40}

📊 Pattern: {setup.pattern.value.upper()}
{'BUY' if setup.direction == 'long' else 'SELL'} {trade_type}

💰 ENTRY: ${setup.entry_price:.2f}
🎯 TARGET: ${setup.target_price:.2f}
🛑 STOP: ${setup.stop_loss:.2f}

📈 R:R = 1:{rr_ratio:.1f}
💵 Expected Gain: ~{expected_option_gain:.0f}%

📍 VWAP: ${setup.vwap:.2f}
📊 Volume: {setup.volume_ratio:.1f}x avg
{'🚀' if setup.momentum > 0 else '📉'} Momentum: {setup.momentum:+.2f}%

{'✅ AI CONFIRMED' if setup.ai_confirmed else '⚠️ No AI confirmation'}
🎯 Confidence: {setup.confidence + setup.pattern_memory_boost + setup.time_quality_score:.0f}%

{setup.notes}

{'='*40}
⏰ SCALP WINDOW: 15-60 min
💡 Exit at target or stop - no excuses!
{'='*40}
"""
        
        send_telegram_alert(alert_msg)
        log.info(f"🦅 {ticker} Scalp alert: {setup.pattern.value} {setup.direction} @ ${setup.entry_price:.2f}")
        
        # Add to stalking mode for exit tracking
        stalk_setup_id = stalking_mode.add_setup(
            symbol=ticker,
            setup_type=f"0DTE_{setup.pattern.value}",
            trigger_price=setup.entry_price,
            trigger_condition=f"Scalp {setup.direction}",
            catalyst=setup.notes,
            expected_move=underlying_move_pct,
            expires_hours=2,  # Short expiry for scalps
            entry_price=setup.entry_price,
            target_price=setup.target_price,
            stop_loss=setup.stop_loss,
            direction=setup.direction,
            trade_type=trade_type
        )
        
        # Mark as entry alerted for exit tracking
        if stalk_setup_id in stalking_mode.stalked:
            stalking_mode.stalked[stalk_setup_id].entry_alerted = True
            stalking_mode._save_setup(stalking_mode.stalked[stalk_setup_id])
            log.info(f"Stalking mode tracking exit for {stalk_setup_id}")
        
        # NOTE: We do NOT add Alpaca-executed option scalps to Zero Greed Exit.
        # Reason: Zero Greed uses underlying prices, so its alerts would show stock price
        # (e.g. CLOSE @ $82.39) instead of option premium. Alpaca executor monitors the
        # same position with option quotes and sends alerts with option spec (strike, DTE)
        # and Entry/Exit (option premium), so the user can match to Webull's options chain.

        # Execute REAL paper trade on Alpaca
        total_confidence = setup.confidence + setup.pattern_memory_boost + setup.time_quality_score
        try:
            alpaca_position = alpaca_executor.execute_scalp_entry(
                underlying=ticker,
                direction=setup.direction,
                entry_price=setup.entry_price,
                target_price=setup.target_price,
                stop_loss=setup.stop_loss,
                confidence=total_confidence,
                pattern=setup.pattern.value
            )
            if alpaca_position:
                log.info(f"📈 Alpaca paper trade placed for {ticker}: {alpaca_position.option_symbol}")
            else:
                log.warning(f"Alpaca paper trade not placed for {ticker} (max positions or error)")
        except Exception as e:
            log.error(f"Alpaca execution error: {e}")
        
        # Record in history
        self._record_signal(setup)
        
        self.entries_today += 1
    
    def _record_signal(self, setup: ScalpSetup):
        """Record signal to database for learning."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO spy_scalp_history 
            (pattern, direction, entry_price, target_price, stop_loss, 
             confidence, ai_confirmed, ai_confidence, detected_at, alerted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            setup.pattern.value,
            setup.direction,
            setup.entry_price,
            setup.target_price,
            setup.stop_loss,
            setup.confidence + setup.pattern_memory_boost + setup.time_quality_score,
            1 if setup.ai_confirmed else 0,
            setup.ai_confidence,
            setup.detected_at.isoformat(),
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _get_current_price(self, ticker: str) -> Optional[float]:
        """Get current price for a ticker (used by zero greed exit)."""
        try:
            if self.last_price > 0:
                return self.last_price
            
            # Fetch fresh price if cache is stale
            bars = polygon_enhanced.get_intraday_bars(ticker, "1", limit=1)
            if bars:
                return bars[-1].get('c', bars[-1].get('close'))
        except Exception as e:
            log.error(f"Error getting current price for {ticker}: {e}")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scalper statistics."""
        return {
            "symbol": self.symbol,
            "running": self.running,
            "signals_today": self.signals_today,
            "entries_today": self.entries_today,
            "exits_today": self.exits_today,
            "last_price": self.last_price,
            "last_vwap": self.last_vwap,
            "cooldown_active": self.cooldown_until and datetime.utcnow() < self.cooldown_until,
            "active_setup": self.active_setup.pattern.value if self.active_setup else None,
            "zero_greed_stats": zero_greed_exit.get_stats()
        }
    
    def force_scan(self) -> Optional[ScalpSetup]:
        """Force an immediate scan (for testing/manual trigger)."""
        self._run_scan()
        return self.active_setup


# Singleton instance
spy_scalper = SPYScalper()


def get_spy_scalper() -> SPYScalper:
    """Get the singleton SPY scalper instance."""
    return spy_scalper
