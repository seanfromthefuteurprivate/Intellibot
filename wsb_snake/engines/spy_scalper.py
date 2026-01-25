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
from wsb_snake.analysis.scalp_langgraph import get_scalp_analyzer
from wsb_snake.analysis.chart_generator import ChartGenerator
from wsb_snake.analysis.scalp_chart_generator import scalp_chart_generator
from wsb_snake.analysis.predator_stack import predator_stack, PredatorVerdict
from wsb_snake.learning.pattern_memory import pattern_memory
from wsb_snake.learning.time_learning import time_learning
from wsb_snake.learning.stalking_mode import stalking_mode, StalkState
from wsb_snake.learning.zero_greed_exit import zero_greed_exit
from wsb_snake.notifications.telegram_bot import send_alert as send_telegram_alert
from wsb_snake.db.database import get_connection

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
    Hawk-like 0DTE SPY scalper that monitors for quick profit opportunities.
    Runs continuously during market hours, scanning every 30 seconds.
    """
    
    def __init__(self):
        self.symbol = "SPY"
        self.scan_interval = 30  # seconds between scans
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.chart_generator = ChartGenerator()
        self.scalp_chart_generator = scalp_chart_generator  # Surgical precision charts
        self.scalp_analyzer = get_scalp_analyzer()  # LangGraph analyzer
        self.predator_stack = predator_stack  # Multi-model AI (Gemini + DeepSeek + GPT)
        
        # Recent price data cache
        self.price_cache: List[Dict] = []
        self.last_vwap = 0.0
        self.last_price = 0.0
        
        # Pattern detection state
        self.active_setup: Optional[ScalpSetup] = None
        self.cooldown_until: Optional[datetime] = None
        self.trade_cooldown_minutes = 10  # Cooldown after entry signal
        
        # Statistics
        self.signals_today = 0
        self.entries_today = 0
        self.exits_today = 0
        
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
        self.worker_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.worker_thread.start()
        log.info("🦅 SPY Scalper started - watching for 0DTE opportunities")
    
    def stop(self):
        """Stop the scalper."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        log.info("SPY Scalper stopped")
    
    def _scan_loop(self):
        """Main scanning loop - runs every 30 seconds during market hours."""
        log.info("SPY Scalper scan loop started")
        
        while self.running:
            try:
                if is_market_open():
                    self._run_scan()
                else:
                    # During off-hours, check less frequently
                    time.sleep(60)
                    continue
                
                time.sleep(self.scan_interval)
                
            except Exception as e:
                log.error(f"SPY Scalper scan error: {e}")
                time.sleep(10)
        
        log.info("SPY Scalper scan loop ended")
    
    def _run_scan(self):
        """Execute a single scan for scalping opportunities."""
        # Check cooldown
        if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
            return
        
        # Get fresh 1-minute bars
        bars = polygon_enhanced.get_intraday_bars(
            self.symbol, 
            timespan="minute", 
            multiplier=1, 
            limit=60  # Last 60 minutes
        )
        
        if not bars or len(bars) < 20:
            return
        
        self.price_cache = bars
        self.last_price = bars[-1].get('c', 0)
        
        # Calculate VWAP
        self.last_vwap = self._calculate_vwap(bars)
        
        # Detect patterns
        setups = self._detect_patterns(bars)
        
        if not setups:
            return
        
        # Get best setup
        best_setup = max(setups, key=lambda s: s.confidence)
        
        # Apply learning boosts
        best_setup = self._apply_learning_boosts(best_setup, bars)
        
        # AI confirmation if confidence is borderline
        if best_setup.confidence >= 60:
            best_setup = self._get_ai_confirmation(best_setup)
        
        # Final decision: alert if confidence >= 70 or (>= 60 and AI confirms)
        final_confidence = best_setup.confidence + best_setup.pattern_memory_boost + best_setup.time_quality_score
        
        should_alert = (
            final_confidence >= 70 or 
            (final_confidence >= 60 and best_setup.ai_confirmed)
        )
        
        if should_alert:
            self._send_entry_alert(best_setup)
            self.cooldown_until = datetime.utcnow() + timedelta(minutes=self.trade_cooldown_minutes)
            self.signals_today += 1
    
    def _calculate_vwap(self, bars: List[Dict]) -> float:
        """Calculate VWAP from bars."""
        total_pv = 0
        total_vol = 0
        
        for bar in bars:
            typical_price = (bar.get('h', 0) + bar.get('l', 0) + bar.get('c', 0)) / 3
            volume = bar.get('v', 0)
            total_pv += typical_price * volume
            total_vol += volume
        
        return total_pv / total_vol if total_vol > 0 else 0
    
    def _detect_patterns(self, bars: List[Dict]) -> List[ScalpSetup]:
        """Detect all scalping patterns in current price action."""
        setups = []
        
        if len(bars) < 10:
            return setups
        
        current = bars[-1]
        prev = bars[-2]
        prev2 = bars[-3] if len(bars) > 2 else prev
        
        price = current.get('c', 0)
        high = current.get('h', 0)
        low = current.get('l', 0)
        volume = current.get('v', 0)
        
        prev_close = prev.get('c', 0)
        prev_volume = prev.get('v', 0)
        
        vwap = self.last_vwap
        
        # Calculate averages
        avg_volume = sum(b.get('v', 0) for b in bars[-20:]) / 20
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        # Calculate momentum (price change rate)
        prices_5min = [b.get('c', 0) for b in bars[-5:]]
        if len(prices_5min) >= 2:
            momentum = (prices_5min[-1] - prices_5min[0]) / prices_5min[0] * 100 if prices_5min[0] > 0 else 0
        else:
            momentum = 0
        
        # ATR for stops/targets (simplified)
        atr = self._calculate_atr(bars[-14:])
        
        # === Pattern Detection ===
        
        # 1. VWAP Bounce (Long)
        if low <= vwap * 1.001 and price > vwap and prev_close < vwap:
            # Price tested VWAP from below and bounced
            setups.append(ScalpSetup(
                pattern=ScalpPattern.VWAP_BOUNCE,
                direction="long",
                entry_price=price,
                target_price=price + (atr * 1.5),  # 1.5 ATR target
                stop_loss=vwap - (atr * 0.5),      # Below VWAP
                confidence=65 + (volume_ratio * 5),
                vwap=vwap,
                volume_ratio=volume_ratio,
                momentum=momentum,
                notes="VWAP bounce - long setup"
            ))
        
        # 2. VWAP Reclaim (Long) - stronger signal
        if prev_close < vwap and price > vwap * 1.001 and volume_ratio > 1.3:
            setups.append(ScalpSetup(
                pattern=ScalpPattern.VWAP_RECLAIM,
                direction="long",
                entry_price=price,
                target_price=price + (atr * 2.0),
                stop_loss=vwap - (atr * 0.3),
                confidence=70 + (volume_ratio * 5),
                vwap=vwap,
                volume_ratio=volume_ratio,
                momentum=momentum,
                notes="VWAP reclaim with volume - bullish"
            ))
        
        # 3. VWAP Rejection (Short)
        if high >= vwap * 0.999 and price < vwap and prev_close > vwap:
            setups.append(ScalpSetup(
                pattern=ScalpPattern.VWAP_REJECTION,
                direction="short",
                entry_price=price,
                target_price=price - (atr * 1.5),
                stop_loss=vwap + (atr * 0.5),
                confidence=65 + (volume_ratio * 5),
                vwap=vwap,
                volume_ratio=volume_ratio,
                momentum=momentum,
                notes="VWAP rejection - short setup"
            ))
        
        # 4. Momentum Surge (Long)
        if momentum > 0.15 and volume_ratio > 1.5:  # Strong up move with volume
            setups.append(ScalpSetup(
                pattern=ScalpPattern.MOMENTUM_SURGE,
                direction="long",
                entry_price=price,
                target_price=price + (atr * 2.0),
                stop_loss=price - (atr * 0.7),
                confidence=68 + (momentum * 10) + (volume_ratio * 3),
                vwap=vwap,
                volume_ratio=volume_ratio,
                momentum=momentum,
                notes=f"Momentum surge +{momentum:.2f}% with volume"
            ))
        
        # 5. Momentum Surge (Short)
        if momentum < -0.15 and volume_ratio > 1.5:
            setups.append(ScalpSetup(
                pattern=ScalpPattern.MOMENTUM_SURGE,
                direction="short",
                entry_price=price,
                target_price=price - (atr * 2.0),
                stop_loss=price + (atr * 0.7),
                confidence=68 + (abs(momentum) * 10) + (volume_ratio * 3),
                vwap=vwap,
                volume_ratio=volume_ratio,
                momentum=momentum,
                notes=f"Momentum dump {momentum:.2f}% with volume"
            ))
        
        # 6. Breakout - price makes new high of last 30 bars
        highs_30 = [b.get('h', 0) for b in bars[-30:-1]]
        if highs_30 and price > max(highs_30) and volume_ratio > 1.3:
            setups.append(ScalpSetup(
                pattern=ScalpPattern.BREAKOUT,
                direction="long",
                entry_price=price,
                target_price=price + (atr * 2.5),
                stop_loss=max(highs_30) - (atr * 0.3),
                confidence=72 + (volume_ratio * 4),
                vwap=vwap,
                volume_ratio=volume_ratio,
                momentum=momentum,
                notes="30-bar high breakout"
            ))
        
        # 7. Breakdown
        lows_30 = [b.get('l', 0) for b in bars[-30:-1]]
        if lows_30 and price < min(lows_30) and volume_ratio > 1.3:
            setups.append(ScalpSetup(
                pattern=ScalpPattern.BREAKDOWN,
                direction="short",
                entry_price=price,
                target_price=price - (atr * 2.5),
                stop_loss=min(lows_30) + (atr * 0.3),
                confidence=72 + (volume_ratio * 4),
                vwap=vwap,
                volume_ratio=volume_ratio,
                momentum=momentum,
                notes="30-bar low breakdown"
            ))
        
        # 8. Failed Breakdown (Bear Trap - Long)
        lows_10 = [b.get('l', 0) for b in bars[-10:-3]]
        if lows_10:
            recent_low = min(lows_10)
            if prev.get('l', 0) < recent_low and price > recent_low * 1.002:
                setups.append(ScalpSetup(
                    pattern=ScalpPattern.FAILED_BREAKDOWN,
                    direction="long",
                    entry_price=price,
                    target_price=price + (atr * 2.0),
                    stop_loss=prev.get('l', 0) - (atr * 0.2),
                    confidence=75 + (volume_ratio * 3),
                    vwap=vwap,
                    volume_ratio=volume_ratio,
                    momentum=momentum,
                    notes="Bear trap - failed breakdown recovery"
                ))
        
        # 9. Failed Breakout (Bull Trap - Short)
        highs_10 = [b.get('h', 0) for b in bars[-10:-3]]
        if highs_10:
            recent_high = max(highs_10)
            if prev.get('h', 0) > recent_high and price < recent_high * 0.998:
                setups.append(ScalpSetup(
                    pattern=ScalpPattern.FAILED_BREAKOUT,
                    direction="short",
                    entry_price=price,
                    target_price=price - (atr * 2.0),
                    stop_loss=prev.get('h', 0) + (atr * 0.2),
                    confidence=75 + (volume_ratio * 3),
                    vwap=vwap,
                    volume_ratio=volume_ratio,
                    momentum=momentum,
                    notes="Bull trap - failed breakout rejection"
                ))
        
        # 10. Squeeze Fire - detect low volatility followed by expansion
        volatility_recent = self._calculate_volatility(bars[-5:])
        volatility_older = self._calculate_volatility(bars[-15:-5])
        
        if volatility_older > 0 and volatility_recent / volatility_older > 1.5:
            # Volatility expanding
            if momentum > 0:
                setups.append(ScalpSetup(
                    pattern=ScalpPattern.SQUEEZE_FIRE,
                    direction="long",
                    entry_price=price,
                    target_price=price + (atr * 2.5),
                    stop_loss=price - (atr * 0.8),
                    confidence=70 + (momentum * 15),
                    vwap=vwap,
                    volume_ratio=volume_ratio,
                    momentum=momentum,
                    notes="Squeeze fire - volatility expanding bullish"
                ))
            elif momentum < 0:
                setups.append(ScalpSetup(
                    pattern=ScalpPattern.SQUEEZE_FIRE,
                    direction="short",
                    entry_price=price,
                    target_price=price - (atr * 2.5),
                    stop_loss=price + (atr * 0.8),
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
            high = bars[i].get('h', 0)
            low = bars[i].get('l', 0)
            prev_close = bars[i-1].get('c', 0)
            
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
        
        return setup
    
    def _get_ai_confirmation(self, setup: ScalpSetup) -> ScalpSetup:
        """Get AI confirmation via multi-model Predator Stack (Gemini + DeepSeek + GPT)."""
        try:
            # Generate surgical precision predator chart with VWAP bands, delta, volume profile
            chart_base64 = self.scalp_chart_generator.generate_predator_chart(
                ticker=self.symbol,
                ohlcv_data=self.price_cache,
                timeframe="1min"
            )
            
            if not chart_base64:
                # Fallback to basic chart
                chart_base64 = self.chart_generator.generate_chart(
                    ticker=self.symbol,
                    ohlcv_data=self.price_cache,
                    timeframe="1min"
                )
            
            if not chart_base64:
                return setup
            
            # Use Predator Stack for multi-model AI confirmation
            # Gemini (primary) -> DeepSeek (fallback) -> GPT (confirmation for high confidence)
            analysis = self.predator_stack.analyze_sync(
                chart_base64=chart_base64,
                ticker=self.symbol,
                pattern=setup.pattern.value,
                current_price=self.last_price,
                vwap=setup.vwap,
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
    
    def _send_entry_alert(self, setup: ScalpSetup):
        """Send entry alert to Telegram and add to stalking mode."""
        trade_type = "CALLS" if setup.direction == "long" else "PUTS"
        
        # Calculate R:R
        risk = abs(setup.entry_price - setup.stop_loss)
        reward = abs(setup.target_price - setup.entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Expected option move (simplified: 3x leverage on SPY)
        underlying_move_pct = reward / setup.entry_price * 100
        expected_option_gain = underlying_move_pct * 3  # Rough 0DTE delta
        
        # Format alert
        alert_msg = f"""
{'='*40}
🦅 SPY 0DTE SCALP ALERT 🦅
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
        log.info(f"🦅 SPY Scalp alert: {setup.pattern.value} {setup.direction} @ ${setup.entry_price:.2f}")
        
        # Add to stalking mode for exit tracking
        stalk_setup_id = stalking_mode.add_setup(
            symbol=self.symbol,
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
        
        # Add to Zero Greed Exit for mechanical ruthless exit tracking
        position_id = f"spy_scalp_{datetime.utcnow().strftime('%H%M%S')}"
        zero_greed_exit.add_position(
            position_id=position_id,
            ticker=self.symbol,
            direction=setup.direction,
            trade_type=trade_type,
            entry_price=setup.entry_price,
            target_price=setup.target_price,
            stop_loss=setup.stop_loss,
            max_hold_minutes=60,  # 1 hour max for 0DTE
            price_getter=self._get_current_price
        )
        log.info(f"🔪 Zero Greed Exit tracking: {position_id}")
        
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
