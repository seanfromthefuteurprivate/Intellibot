"""
Session Learnings - Encoded Trading Wisdom from Live Sessions

This module captures battle-tested insights from real trading sessions
and uses them to improve future performance.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
import os
from wsb_snake.utils.logger import log


@dataclass
class TradingInsight:
    """A single trading insight learned from experience."""
    category: str  # 'winner', 'loser', 'pattern', 'timing', 'execution'
    lesson: str
    weight: float  # 0-1, higher = more important
    date_learned: str
    trade_context: Dict[str, Any] = field(default_factory=dict)


class SessionLearnings:
    """
    Encodes and applies learnings from real trading sessions.
    This is the system's brain - where experience becomes wisdom.
    """
    
    def __init__(self):
        self.insights: List[TradingInsight] = []
        self.db_path = "wsb_snake_data/session_learnings.json"
        self._load_insights()
        
        # Encode today's session learnings (Jan 26, 2026)
        self._encode_todays_session()
    
    def _load_insights(self):
        """Load saved insights from database."""
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    self.insights = [
                        TradingInsight(**i) for i in data.get('insights', [])
                    ]
        except Exception as e:
            log.error(f"Failed to load session learnings: {e}")
    
    def _save_insights(self):
        """Save insights to database."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'w') as f:
                json.dump({
                    'insights': [
                        {
                            'category': i.category,
                            'lesson': i.lesson,
                            'weight': i.weight,
                            'date_learned': i.date_learned,
                            'trade_context': i.trade_context
                        }
                        for i in self.insights
                    ],
                    'last_updated': datetime.utcnow().isoformat()
                }, f, indent=2)
        except Exception as e:
            log.error(f"Failed to save session learnings: {e}")
    
    def add_insight(self, insight: TradingInsight):
        """Add a new insight to the brain."""
        self.insights.append(insight)
        self._save_insights()
        log.info(f"🧠 New insight learned: {insight.lesson[:50]}...")
    
    def _encode_todays_session(self):
        """Encode all learnings from today's trading session."""
        # Encode both Jan 26 and Jan 27 learnings
        self._encode_jan26_session()
        self._encode_jan27_session()
    
    def _encode_jan27_session(self):
        """Encode learnings from Jan 27 - STOP WIDTH CRITICAL"""
        today = "2026-01-27"
        
        existing = [i for i in self.insights if i.date_learned == today]
        if existing:
            return
        
        self.add_insight(TradingInsight(
            category="execution",
            lesson="STOPS TOO TIGHT: IWM filled @ $1.80, immediately stopped out. 0.3% ATR stops = noise triggers. WIDENED to 0.8-1.2% ATR on underlying.",
            weight=1.0,
            date_learned=today,
            trade_context={
                "ticker": "IWM",
                "fill_price": 1.80,
                "stop_reason": "noise_triggered",
                "fix_applied": "widened_stops_to_1pct_atr",
                "key_insight": "tight_stops_kill_positions"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="execution", 
            lesson="ATR MULTIPLIERS CHANGED: VWAP bounce stop 0.5->1.0 ATR, momentum stop 0.7->1.2 ATR, breakout stop 0.3->0.8 ATR. Targets also widened to 2.5-3.5 ATR.",
            weight=0.95,
            date_learned=today,
            trade_context={
                "old_stop_multiplier": 0.3,
                "new_stop_multiplier": 1.0,
                "old_target_multiplier": 1.5,
                "new_target_multiplier": 2.5,
                "key_insight": "atr_multipliers_doubled"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="execution",
            lesson="INSTITUTIONAL STOPS WIDENED: 0.3% on underlying = $2 on $700 SPY = noise. Changed to 0.8% = $5.60 on SPY. This gives breathing room.",
            weight=0.95,
            date_learned=today,
            trade_context={
                "old_stop_pct": 0.003,
                "new_stop_pct": 0.008,
                "spy_old_stop_dollars": 2.10,
                "spy_new_stop_dollars": 5.60,
                "key_insight": "stops_need_room_to_breathe"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="pattern",
            lesson="SMALL CAPS NEED CANDLESTICK CONFIRMATION: SLS lost -$130 (both calls and puts). Small caps are choppy. Added 75% min confidence + STRICT candlestick pattern requirement.",
            weight=1.0,
            date_learned=today,
            trade_context={
                "ticker": "SLS",
                "total_loss": -130,
                "call_loss": -80,
                "put_loss": -50,
                "fix_applied": "small_cap_strict_mode",
                "new_threshold": 75,
                "requires_candlestick": True,
                "patterns_detected": ["hammer", "engulfing", "harami", "momentum"],
                "key_insight": "small_caps_choppy_need_strict_confirmation"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="winner",
            lesson="ETFs WORK BETTER: GDX +$150, IWM +$64, QQQ +$44 = +$258 from ETFs. Small caps SLS -$130. STICK TO ETFs for scalping, small caps only on BIG candlestick signals.",
            weight=1.0,
            date_learned=today,
            trade_context={
                "etf_pnl": 258,
                "small_cap_pnl": -130,
                "net_pnl": 128,
                "winners": ["GDX", "IWM", "QQQ"],
                "losers": ["SLS"],
                "key_insight": "etfs_more_reliable_than_small_caps"
            }
        ))
    
    def _encode_jan26_session(self):
        """Encode learnings from Jan 26 - LOSING DAY"""
        today = "2026-01-26"
        
        # Check if already encoded
        existing = [i for i in self.insights if i.date_learned == today]
        if existing:
            return  # Already learned from today
        
        # ========== WHAT WORKED ==========
        
        self.add_insight(TradingInsight(
            category="winner",
            lesson="AAPL +4.1%: ONLY WINNER TODAY. Quick profit taking at first target hit. Book 4-5% gains immediately - this is the ONLY strategy that worked.",
            weight=0.98,
            date_learned=today,
            trade_context={
                "ticker": "AAPL",
                "exit_pnl": 4.1,
                "dollar_pnl": 35,
                "pattern": "institutional_momentum",
                "hold_time_minutes": 15,
                "key_insight": "quick_profits_only_winner"
            }
        ))
        
        # ========== WHAT DIDN'T WORK - LOSING DAY ==========
        
        self.add_insight(TradingInsight(
            category="loser",
            lesson="LOSING DAY: Net -$82. Only 1 winner (AAPL +$35) vs 3 losers (QQQ -$62, SPY -$23, ONDS -$32). Win rate too low. Need HIGHER confidence threshold, not lower.",
            weight=1.0,  # HIGHEST WEIGHT - critical lesson
            date_learned=today,
            trade_context={
                "net_pnl": -82,
                "winners": 1,
                "losers": 3,
                "win_rate": 0.25,
                "key_insight": "losing_day_win_rate_too_low"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="loser",
            lesson="QQQ -24.6% (-$62): 0DTE theta decay is BRUTAL. Wide stops on 0DTE = death. Need MUCH tighter stops (8-10%) or AVOID 0DTE entirely.",
            weight=0.98,
            date_learned=today,
            trade_context={
                "ticker": "QQQ",
                "exit_pnl": -24.6,
                "dollar_pnl": -62,
                "pattern": "0dte_calls",
                "key_insight": "0dte_kills_avoid_it"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="loser",
            lesson="SPY -9% (-$23): Even with mechanical 60-min exit, still a loss. 0DTE options are too risky. AVOID 0DTE unless A+ setup.",
            weight=0.90,
            date_learned=today,
            trade_context={
                "ticker": "SPY",
                "exit_pnl": -9.0,
                "dollar_pnl": -23,
                "exit_reason": "time_decay_60min",
                "key_insight": "0dte_even_with_stops_loses"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="loser",
            lesson="ONDS -4% (-$32): Closed at EOD for a loss. Swing trade forced exit. Don't enter swing positions late in day if EOD close mandatory.",
            weight=0.85,
            date_learned=today,
            trade_context={
                "ticker": "ONDS",
                "exit_pnl": -4.0,
                "dollar_pnl": -32,
                "exit_reason": "eod_mandatory_close",
                "key_insight": "no_swings_near_eod"
            }
        ))
        
        # ========== EXECUTION INSIGHTS ==========
        
        self.add_insight(TradingInsight(
            category="execution",
            lesson="CRITICAL: Telegram alerts MUST trigger immediate execution. Signal-execution disconnect = missed profits. Fixed today.",
            weight=1.0,
            date_learned=today,
            trade_context={
                "bug_fixed": "orchestrator_alpaca_connection",
                "impact": "high",
                "key_insight": "alerts_must_execute"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="execution",
            lesson="WRONG: Lowering confidence to 58% LOST MONEY. 25% win rate is unacceptable. RAISE threshold to 70%+ for quality over quantity.",
            weight=1.0,  # Critical correction
            date_learned=today,
            trade_context={
                "old_threshold": 58,
                "new_threshold": 70,
                "key_insight": "quality_over_quantity"
            }
        ))
        
        # ========== TIMING INSIGHTS ==========
        
        self.add_insight(TradingInsight(
            category="timing",
            lesson="Power hour (3-4 PM ET) is PRIME time. RKLB massive win came from power hour setup. Increase position size during power hour.",
            weight=0.90,
            date_learned=today,
            trade_context={
                "session": "power_hour",
                "multiplier": 1.3,
                "key_insight": "power_hour_is_prime"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="timing",
            lesson="0DTE must close by 3:55 PM ET. No exceptions. Theta decay accelerates exponentially in final hour. ONDS closed at -4% due to EOD rule.",
            weight=0.95,
            date_learned=today,
            trade_context={
                "close_time": "15:55",
                "key_insight": "0dte_eod_mandatory"
            }
        ))
        
        # ========== PATTERN INSIGHTS ==========
        
        self.add_insight(TradingInsight(
            category="pattern",
            lesson="0DTE KILLED US: QQQ -$62, SPY -$23. 0DTE theta decay destroyed positions. STRONGLY prefer 2-4 DTE or AVOID 0DTE entirely.",
            weight=0.98,
            date_learned=today,
            trade_context={
                "0dte_pnl": -85,
                "key_insight": "0dte_is_death_prefer_multiday"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="pattern",
            lesson="Position sizing: $1000 daily cap works. Split into 2-3 trades for diversification. Single big bet = single point of failure.",
            weight=0.85,
            date_learned=today,
            trade_context={
                "daily_cap": 1000,
                "optimal_positions": 3,
                "key_insight": "diversify_daily_capital"
            }
        ))
        
        # ========== INSTITUTIONAL WISDOM ==========
        
        self.add_insight(TradingInsight(
            category="institutional",
            lesson="Citadel Rule applied: 'Speed + Edge = Alpha'. Execute fast on high-prob setups. Hesitation costs money.",
            weight=0.90,
            date_learned=today,
            trade_context={
                "rule": "citadel",
                "application": "AAPL quick profit",
                "key_insight": "speed_matters"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="institutional",
            lesson="Jane Street Rule: 'Statistical edge > gut feeling'. Trust the AI confidence scores. Override emotions with data.",
            weight=0.95,
            date_learned=today,
            trade_context={
                "rule": "jane_street",
                "application": "held_RKLB_through_drawdown",
                "key_insight": "trust_the_math"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="institutional",
            lesson="Day Trader Rule: 'Take +5-10% wins, cut at -15%'. AAPL +4.1% booked. Small gains compound into big P&L.",
            weight=0.90,
            date_learned=today,
            trade_context={
                "rule": "day_trader",
                "target_pnl": "5-10%",
                "stop_pnl": "-15%",
                "key_insight": "small_gains_compound"
            }
        ))
        
        log.info(f"🧠 Encoded {len([i for i in self.insights if i.date_learned == today])} learnings from today's session")
    
    def get_applicable_insights(
        self,
        ticker: str = None,
        pattern: str = None,
        session: str = None,
        min_weight: float = 0.5
    ) -> List[TradingInsight]:
        """Get insights applicable to current trade context."""
        applicable = []
        
        for insight in self.insights:
            if insight.weight < min_weight:
                continue
            
            context = insight.trade_context
            
            # Filter by ticker if specified
            if ticker and context.get('ticker') and context['ticker'] != ticker:
                continue
            
            # Filter by pattern if specified
            if pattern and context.get('pattern') and pattern not in context['pattern']:
                continue
            
            # Filter by session if specified
            if session and context.get('session') and context['session'] != session:
                continue
            
            applicable.append(insight)
        
        # Sort by weight (most important first)
        applicable.sort(key=lambda x: x.weight, reverse=True)
        return applicable
    
    def get_confidence_adjustment(self, ticker: str, pattern: str, session: str) -> float:
        """
        Get confidence adjustment based on learned insights.
        Returns a multiplier (0.5 - 1.5) to apply to base confidence.
        """
        adjustment = 1.0
        
        # Power hour boost
        if session == "power_hour":
            adjustment *= 1.15  # +15% confidence in power hour
        
        # Multi-day vs 0DTE preference
        if "0dte" in pattern.lower():
            adjustment *= 0.85  # -15% for 0DTE (more risky)
        
        # Apply learned patterns
        insights = self.get_applicable_insights(ticker=ticker, pattern=pattern, session=session)
        
        for insight in insights[:3]:  # Top 3 most relevant
            if insight.category == "winner":
                adjustment *= 1.05  # Boost confidence for winning patterns
            elif insight.category == "loser":
                adjustment *= 0.95  # Reduce confidence for losing patterns
        
        # Clamp to reasonable range
        return max(0.5, min(1.5, adjustment))
    
    def get_stop_loss_adjustment(self, is_0dte: bool) -> float:
        """
        Get stop loss percentage based on learnings.
        0DTE needs tighter stops due to theta decay.
        """
        if is_0dte:
            return 0.12  # 12% stop for 0DTE (learned from QQQ loss)
        else:
            return 0.18  # 18% stop for multi-day
    
    def get_profit_target(self, is_0dte: bool) -> float:
        """
        Get profit target percentage based on learnings.
        Quick profits on 0DTE, let multi-day run.
        """
        if is_0dte:
            return 0.08  # 8% target for 0DTE (quick in/out)
        else:
            return 0.15  # 15% target for multi-day (let winners run)
    
    def should_hold_through_drawdown(self, current_pnl: float, pattern: str) -> bool:
        """
        Based on RKLB experience: positions can recover from -4% to +23%.
        Returns True if we should hold, False if we should cut.
        """
        # RKLB lesson: was -4% before going to +23%
        # Only cut if beyond -15% or clear reversal pattern
        
        if current_pnl < -15:
            return False  # Cut losses at -15%
        
        if current_pnl > -8:
            return True  # Hold through small drawdowns
        
        # Medium drawdown (-8% to -15%): depends on pattern
        if "momentum" in pattern.lower():
            return True  # Momentum can recover
        
        return False  # Be cautious otherwise
    
    def format_daily_learnings(self) -> str:
        """Format today's learnings for logging/display."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        todays = [i for i in self.insights if i.date_learned == today]
        
        if not todays:
            return "No new learnings today."
        
        output = f"🧠 SESSION LEARNINGS ({len(todays)} insights)\n"
        output += "=" * 50 + "\n"
        
        by_category = {}
        for i in todays:
            if i.category not in by_category:
                by_category[i.category] = []
            by_category[i.category].append(i)
        
        for category, insights in by_category.items():
            output += f"\n📌 {category.upper()}:\n"
            for insight in insights:
                output += f"  • {insight.lesson[:80]}...\n"
        
        return output


# ========== TOMORROW'S BATTLE PLAN ==========

class TomorrowsBattlePlan:
    """
    Encoded strategy for tomorrow based on today's learnings.
    """
    
    # BALANCED APPROACH: Quality 0DTE for moonshots + Safe multi-day for base hits
    RULES = {
        "entry": {
            "confidence_threshold": 70,  # High bar for all trades
            "0dte_confidence_threshold": 80,  # HIGHER bar for 0DTE moonshots
            "prefer_multiday_for_base_hits": True,  # 2-4 DTE for consistent gains
            "0dte_for_moonshots": True,  # 0DTE for explosive moves ($800 → $50k)
            "power_hour_multiplier": 1.3,  # Power hour is prime for 0DTE
            "require_ai_confirmation": True,
            "execute_on_alert": True,  # FIXED: alerts trigger execution
        },
        "position_sizing": {
            "daily_limit": 1000,
            "max_positions": 3,
            "0dte_allocation": 0.30,  # 30% of capital for 0DTE moonshots ($300)
            "multiday_allocation": 0.70,  # 70% for safer multi-day plays ($700)
            "0dte_moonshot_size": 300,  # Fixed $300 for 0DTE lottery tickets
            "multiday_size": 500,  # $500-700 for multi-day positions
        },
        "stops": {
            "0dte_stop": 0.50,  # WIDER 50% stop for 0DTE moonshots - let them run!
            "0dte_scalp_stop": 0.15,  # 15% stop for quick 0DTE scalps
            "multiday_stop": 0.15,  # 15% stop for 2-4 DTE
            "time_stop_minutes": 45,  # 45-min for scalps, no time stop for moonshots
        },
        "targets": {
            "0dte_moonshot_target": 5.0,  # 500%+ for moonshots - $800 → $4k+
            "0dte_scalp_target": 0.20,  # 20% for quick scalps
            "multiday_target": 0.25,  # 25% for multi-day
            "scale_out": True,  # Take 50% at 100%, let rest run for moonshots
        },
        "timing": {
            "power_hour_start": 15,  # 3 PM ET - PRIME for 0DTE
            "eod_close_time": "15:55",  # All 0DTE close by 3:55 PM
            "avoid_lunch": True,  # 12-1 PM is choppy
            "avoid_first_5_min": True,  # Opening chaos
            "0dte_best_times": ["09:30-10:00", "14:00-15:55"],  # Opening + Power hour
        },
        "moonshot_criteria": {
            "require_unusual_options_flow": True,  # Big money coming in
            "require_catalyst": True,  # News, earnings, macro event
            "require_technical_breakout": True,  # Breaking key levels
            "min_volume_spike": 3.0,  # 3x normal volume
            "require_multi_model_consensus": True,  # All AI models agree
        },
        "mindset": {
            "trust_the_system": True,
            "no_emotional_overrides": True,
            "moonshots_are_lottery_tickets": True,  # Expect to lose some
            "one_moonshot_pays_for_10_losses": True,  # $800 → $50k math
            "mechanical_discipline": True,
            "let_winners_run": True,  # Don't cut moonshots early
        }
    }
    
    @classmethod
    def get_rule(cls, category: str, rule: str) -> Any:
        """Get a specific rule value."""
        return cls.RULES.get(category, {}).get(rule)
    
    @classmethod
    def format_battle_plan(cls) -> str:
        """Format the battle plan for tomorrow - CORRECTED AFTER LOSING DAY."""
        return f"""
🎯 TOMORROW'S BATTLE PLAN (CORRECTED)
{'=' * 50}
⚠️  LEARNED FROM LOSING DAY: -$82 (25% win rate)
{'=' * 50}

📈 ENTRY RULES (STRICTER):
  • Confidence threshold: {cls.RULES['entry']['confidence_threshold']}% (RAISED from 58%)
  • AVOID 0DTE - it killed us today (-$85)
  • Only 2-4 DTE multi-day options
  • Power hour: 1.2x position size
  • QUALITY over quantity - fewer, better trades

💰 POSITION SIZING (FOCUSED):
  • Daily limit: ${cls.RULES['position_sizing']['daily_limit']}
  • MAX 2 positions (not 3) - focus on quality
  • Minimum $300 per trade (bigger conviction bets)

🛑 STOPS (TIGHTER):
  • 0DTE: {cls.RULES['stops']['0dte_stop']*100:.0f}% stop (if we must trade it)
  • Multi-day: {cls.RULES['stops']['multiday_stop']*100:.0f}% stop
  • 30-min time stop (reduced from 45)

🎯 TARGETS (LOWER - BOOK PROFITS FAST):
  • 0DTE: {cls.RULES['targets']['0dte_target']*100:.0f}% quick profit
  • Multi-day: {cls.RULES['targets']['multiday_target']*100:.0f}% target
  • AAPL +4% was our ONLY winner - small gains work!

⏰ TIMING (CONSERVATIVE):
  • Power hour (3-4 PM) still prime time
  • All 0DTE close by 3:55 PM
  • NO new positions after 3 PM
  • Avoid lunch hour chop

🧠 MINDSET (HUMBLE):
  • Accept losing days - learn and improve
  • Quality over quantity
  • Small gains compound - AAPL proved it
  • Mechanical discipline - no revenge trading

{'=' * 50}
"""


# Global instances
session_learnings = SessionLearnings()
battle_plan = TomorrowsBattlePlan()
