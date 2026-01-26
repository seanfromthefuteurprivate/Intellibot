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
        today = "2026-01-26"
        
        # Check if already encoded
        existing = [i for i in self.insights if i.date_learned == today]
        if existing:
            return  # Already learned from today
        
        # ========== WHAT WORKED ==========
        
        self.add_insight(TradingInsight(
            category="winner",
            lesson="RKLB +23.2%: Patience through drawdown pays. Position was -4% before recovering to +23%. Don't panic sell on initial losses.",
            weight=0.95,
            date_learned=today,
            trade_context={
                "ticker": "RKLB",
                "entry_pnl": -4.0,
                "exit_pnl": 23.2,
                "pattern": "bearish_momentum",
                "hold_time_minutes": 45,
                "key_insight": "winners_run_after_initial_drawdown"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="winner",
            lesson="AAPL +4.1%: Quick profit taking at first target hit. Book 4-5% gains immediately - don't wait for 10%+. Small gains compound.",
            weight=0.90,
            date_learned=today,
            trade_context={
                "ticker": "AAPL",
                "exit_pnl": 4.1,
                "pattern": "institutional_momentum",
                "hold_time_minutes": 15,
                "key_insight": "quick_profits_compound"
            }
        ))
        
        # ========== WHAT DIDN'T WORK ==========
        
        self.add_insight(TradingInsight(
            category="loser",
            lesson="QQQ -24.6%: 0DTE theta decay is BRUTAL. Wide stops on 0DTE = death. Need tighter stops (10-15%) on 0DTE specifically.",
            weight=0.95,
            date_learned=today,
            trade_context={
                "ticker": "QQQ",
                "exit_pnl": -24.6,
                "pattern": "0dte_calls",
                "key_insight": "0dte_needs_tight_stops"
            }
        ))
        
        self.add_insight(TradingInsight(
            category="loser",
            lesson="SPY -9%: Time decay exit (60 min rule) saved us from bigger loss. Mechanical exits work. Never override the machine.",
            weight=0.85,
            date_learned=today,
            trade_context={
                "ticker": "SPY",
                "exit_pnl": -9.0,
                "exit_reason": "time_decay_60min",
                "key_insight": "mechanical_exits_save_capital"
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
            lesson="Confidence threshold 65% too conservative. Lowered to 58%. More trades = more opportunities. Edge compounds over volume.",
            weight=0.85,
            date_learned=today,
            trade_context={
                "old_threshold": 65,
                "new_threshold": 58,
                "key_insight": "more_trades_better"
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
            lesson="Multi-day options (RKLB, ONDS) outperformed 0DTE (QQQ, SPY). Less theta decay = more room to be right. Prefer 2-4 DTE.",
            weight=0.90,
            date_learned=today,
            trade_context={
                "0dte_pnl": -85,
                "multiday_pnl": 198,
                "key_insight": "prefer_multiday_over_0dte"
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
    
    RULES = {
        "entry": {
            "confidence_threshold": 58,  # Lowered from 65
            "prefer_multiday": True,  # 2-4 DTE over 0DTE
            "power_hour_multiplier": 1.3,  # Increase size during power hour
            "require_ai_confirmation": True,
            "execute_on_alert": True,  # FIXED: alerts trigger execution
        },
        "position_sizing": {
            "daily_limit": 1000,
            "max_positions": 3,
            "min_position": 200,  # At least $200 per trade
            "diversify": True,  # Split capital across trades
        },
        "stops": {
            "0dte_stop": 0.12,  # 12% stop for 0DTE
            "multiday_stop": 0.18,  # 18% stop for 2-4 DTE
            "time_stop_minutes": 45,  # 45 min max hold for 0DTE
        },
        "targets": {
            "0dte_target": 0.08,  # 8% quick profit for 0DTE
            "multiday_target": 0.15,  # 15% for multi-day
            "scale_out": False,  # For now, all-or-nothing exits
        },
        "timing": {
            "power_hour_start": 15,  # 3 PM ET
            "eod_close_time": "15:55",  # All 0DTE close by 3:55 PM
            "avoid_lunch": True,  # 12-1 PM is choppy
            "avoid_first_5_min": True,  # Opening chaos
        },
        "mindset": {
            "trust_the_system": True,
            "no_emotional_overrides": True,
            "small_gains_compound": True,
            "patience_through_drawdown": True,  # RKLB lesson
            "mechanical_discipline": True,
        }
    }
    
    @classmethod
    def get_rule(cls, category: str, rule: str) -> Any:
        """Get a specific rule value."""
        return cls.RULES.get(category, {}).get(rule)
    
    @classmethod
    def format_battle_plan(cls) -> str:
        """Format the battle plan for tomorrow."""
        return f"""
🎯 TOMORROW'S BATTLE PLAN
{'=' * 50}

📈 ENTRY RULES:
  • Confidence threshold: {cls.RULES['entry']['confidence_threshold']}%
  • Prefer 2-4 DTE over 0DTE
  • Power hour: 1.3x position size
  • Execute immediately on alert

💰 POSITION SIZING:
  • Daily limit: ${cls.RULES['position_sizing']['daily_limit']}
  • Split into 2-3 positions
  • Minimum $200 per trade

🛑 STOPS:
  • 0DTE: {cls.RULES['stops']['0dte_stop']*100:.0f}% stop
  • Multi-day: {cls.RULES['stops']['multiday_stop']*100:.0f}% stop
  • 45-min time stop on 0DTE

🎯 TARGETS:
  • 0DTE: {cls.RULES['targets']['0dte_target']*100:.0f}% quick profit
  • Multi-day: {cls.RULES['targets']['multiday_target']*100:.0f}% let run

⏰ TIMING:
  • Power hour (3-4 PM) is prime time
  • All 0DTE close by 3:55 PM
  • Avoid lunch hour chop

🧠 MINDSET:
  • Trust the system
  • No emotional overrides
  • Small gains compound
  • Patience through drawdowns
  • Mechanical discipline

{'=' * 50}
"""


# Global instances
session_learnings = SessionLearnings()
battle_plan = TomorrowsBattlePlan()
