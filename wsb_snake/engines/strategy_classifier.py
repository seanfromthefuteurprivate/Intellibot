"""
Strategy Classifier Engine

Expands beyond 0DTE to include all 7 trading strategy types:
1. 0DTE Directional Gamma Plays
2. 0DTE/1DTE Volatility Expansion (Non-Directional)
3. Earnings OTM Directional (7-10 DTE)
4. Earnings Volatility Plays (Straddles/Strangles, 7-10 DTE)
5. Post-Earnings Trend Continuation (3-10 DTE)
6. News-Window Options (FDA/Legal/M&A, 7-21 DTE)
7. Macro Reaction Options (Indexes, 3-10 DTE)
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from wsb_snake.utils.logger import log


class StrategyType(Enum):
    """All 7 strategy types"""
    ZERO_DTE_DIRECTIONAL = "0DTE_DIRECTIONAL"
    ZERO_DTE_VOLATILITY = "0DTE_VOLATILITY_EXPANSION"
    EARNINGS_DIRECTIONAL = "EARNINGS_OTM_DIRECTIONAL"
    EARNINGS_VOLATILITY = "EARNINGS_VOLATILITY_PLAY"
    POST_EARNINGS_TREND = "POST_EARNINGS_CONTINUATION"
    NEWS_WINDOW = "NEWS_WINDOW_OPTIONS"
    MACRO_REACTION = "MACRO_REACTION_OPTIONS"


@dataclass
class StrategySignal:
    """Signal for a specific strategy type"""
    strategy: StrategyType
    symbol: str
    direction: str
    confidence: float
    dte_range: str
    structure: str
    entry_trigger: str
    key_skill: str
    risk_level: str
    score: float
    metadata: Dict


class StrategyClassifier:
    """
    Classifies setups into the 7 strategy types.
    
    Each strategy has:
    - Specific entry conditions
    - Recommended DTE range
    - Structure (calls, puts, straddles, etc.)
    - Key skills required
    """
    
    STRATEGY_CONFIGS = {
        StrategyType.ZERO_DTE_DIRECTIONAL: {
            "name": "0DTE Directional Gamma",
            "dte_range": "0",
            "structures": ["OTM_CALLS", "OTM_PUTS"],
            "events": ["CPI", "FOMC", "FED_SPEAKERS", "INDEX_REBALANCE"],
            "key_skill": "Timing + exits",
            "risk_level": "EXTREME",
            "what_training": ["Gamma sensitivity", "Dealer hedging flows", "Intraday momentum vs fade"],
        },
        StrategyType.ZERO_DTE_VOLATILITY: {
            "name": "0DTE Volatility Expansion",
            "dte_range": "0-1",
            "structures": ["ATM_STRADDLE", "OTM_STRANGLE"],
            "events": ["CPI", "JOBS_REPORT", "SURPRISE_PRESS_CONF"],
            "key_skill": "IV vs RV analysis, speed of exit",
            "risk_level": "HIGH",
            "what_training": ["Implied vs realized volatility", "IV collapse speed", "Exit timing"],
        },
        StrategyType.EARNINGS_DIRECTIONAL: {
            "name": "Earnings OTM Directional",
            "dte_range": "7-10",
            "structures": ["OTM_CALLS", "OTM_PUTS"],
            "events": ["EARNINGS"],
            "key_skill": "Discipline > bravery",
            "risk_level": "HIGH",
            "what_training": ["Earnings gap behavior", "IV crush vs trend", "Crowd positioning errors"],
        },
        StrategyType.EARNINGS_VOLATILITY: {
            "name": "Earnings Volatility Play",
            "dte_range": "7-10",
            "structures": ["ATM_STRADDLE", "OTM_STRANGLE"],
            "events": ["EARNINGS"],
            "key_skill": "IV math, expected move accuracy",
            "risk_level": "MEDIUM",
            "what_training": ["IV math", "Expected move accuracy", "Exit one leg, let other die"],
        },
        StrategyType.POST_EARNINGS_TREND: {
            "name": "Post-Earnings Continuation",
            "dte_range": "3-10",
            "structures": ["OTM_CALLS", "OTM_PUTS"],
            "events": ["POST_EARNINGS"],
            "key_skill": "Patience over adrenaline",
            "risk_level": "MEDIUM",
            "what_training": ["IV staying elevated", "Which stocks trend vs fade", "Multi-day momentum"],
        },
        StrategyType.NEWS_WINDOW: {
            "name": "News-Window Options",
            "dte_range": "7-21",
            "structures": ["OTM_CALLS", "OTM_PUTS"],
            "events": ["FDA", "COURT_RULING", "MERGER", "REGULATORY"],
            "key_skill": "Risk asymmetry, sizing discipline",
            "risk_level": "HIGH",
            "what_training": ["Binary outcome pricing", "Emotional control", "Asymmetric warfare"],
        },
        StrategyType.MACRO_REACTION: {
            "name": "Macro Reaction Options",
            "dte_range": "3-10",
            "structures": ["OTM_CALLS", "OTM_PUTS"],
            "events": ["CPI", "FED_DECISION", "RATE_DECISION"],
            "key_skill": "Market psychology, fake first moves",
            "risk_level": "MEDIUM",
            "what_training": ["Second-order reaction", "Narrative shift", "Trend vs chop regimes"],
        },
    }
    
    def __init__(self):
        self.active_signals: Dict[str, List[StrategySignal]] = {}
        log.info("Strategy Classifier initialized (7 strategy types)")
    
    def classify_opportunity(
        self,
        symbol: str,
        context: Dict[str, Any]
    ) -> List[StrategySignal]:
        """
        Classify all applicable strategies for a given opportunity.
        
        Args:
            symbol: Stock ticker
            context: Dict with market data, earnings, news, etc.
            
        Returns:
            List of applicable strategy signals
        """
        signals = []
        
        if self._check_zero_dte_directional(symbol, context):
            signals.append(self._build_signal(
                StrategyType.ZERO_DTE_DIRECTIONAL, symbol, context
            ))
        
        if self._check_zero_dte_volatility(symbol, context):
            signals.append(self._build_signal(
                StrategyType.ZERO_DTE_VOLATILITY, symbol, context
            ))
        
        if self._check_earnings_directional(symbol, context):
            signals.append(self._build_signal(
                StrategyType.EARNINGS_DIRECTIONAL, symbol, context
            ))
        
        if self._check_earnings_volatility(symbol, context):
            signals.append(self._build_signal(
                StrategyType.EARNINGS_VOLATILITY, symbol, context
            ))
        
        if self._check_post_earnings(symbol, context):
            signals.append(self._build_signal(
                StrategyType.POST_EARNINGS_TREND, symbol, context
            ))
        
        if self._check_news_window(symbol, context):
            signals.append(self._build_signal(
                StrategyType.NEWS_WINDOW, symbol, context
            ))
        
        if self._check_macro_reaction(symbol, context):
            signals.append(self._build_signal(
                StrategyType.MACRO_REACTION, symbol, context
            ))
        
        return signals
    
    def _check_zero_dte_directional(self, symbol: str, ctx: Dict) -> bool:
        """Check if 0DTE directional play is valid"""
        is_trading_day = ctx.get("is_market_open", False)
        has_catalyst = ctx.get("has_macro_event", False) or ctx.get("has_fed_speaker", False)
        has_momentum = ctx.get("momentum_score", 0) > 50
        
        return is_trading_day and (has_catalyst or has_momentum)
    
    def _check_zero_dte_volatility(self, symbol: str, ctx: Dict) -> bool:
        """Check if 0DTE volatility expansion play is valid"""
        is_trading_day = ctx.get("is_market_open", False)
        has_volatility_event = ctx.get("is_cpi_day", False) or ctx.get("is_jobs_day", False)
        iv_underpriced = ctx.get("iv_percentile", 50) < 40
        
        return is_trading_day and has_volatility_event and iv_underpriced
    
    def _check_earnings_directional(self, symbol: str, ctx: Dict) -> bool:
        """Check if earnings directional play is valid"""
        days_to_earnings = ctx.get("days_to_earnings")
        if days_to_earnings is None:
            return False
        
        return 5 <= days_to_earnings <= 14
    
    def _check_earnings_volatility(self, symbol: str, ctx: Dict) -> bool:
        """Check if earnings volatility play is valid"""
        days_to_earnings = ctx.get("days_to_earnings")
        if days_to_earnings is None:
            return False
        
        expected_move = ctx.get("expected_move", 0)
        historical_move = ctx.get("historical_earnings_move", 0)
        
        underpriced = historical_move > expected_move * 1.2 if expected_move > 0 else False
        
        return 5 <= days_to_earnings <= 14 and (underpriced or days_to_earnings <= 7)
    
    def _check_post_earnings(self, symbol: str, ctx: Dict) -> bool:
        """Check if post-earnings continuation play is valid"""
        days_since_earnings = ctx.get("days_since_earnings")
        if days_since_earnings is None:
            return False
        
        earnings_reaction = ctx.get("earnings_reaction")
        has_trend = abs(ctx.get("post_earnings_drift", 0)) > 2
        
        return 0 < days_since_earnings <= 5 and has_trend
    
    def _check_news_window(self, symbol: str, ctx: Dict) -> bool:
        """Check if news window play is valid"""
        has_pending_event = (
            ctx.get("has_fda_date", False) or
            ctx.get("has_court_ruling", False) or
            ctx.get("has_merger_vote", False) or
            ctx.get("has_regulatory_decision", False)
        )
        
        days_to_event = ctx.get("days_to_binary_event", 999)
        
        return has_pending_event and 7 <= days_to_event <= 21
    
    def _check_macro_reaction(self, symbol: str, ctx: Dict) -> bool:
        """Check if macro reaction play is valid"""
        if symbol not in ["SPY", "QQQ", "IWM"]:
            return False
        
        days_since_macro = ctx.get("days_since_macro_event", 999)
        has_second_order = ctx.get("narrative_shift", False)
        
        return 0 < days_since_macro <= 3 and has_second_order
    
    def _build_signal(
        self,
        strategy: StrategyType,
        symbol: str,
        ctx: Dict
    ) -> StrategySignal:
        """Build a strategy signal from context"""
        config = self.STRATEGY_CONFIGS[strategy]
        
        if ctx.get("trend_direction") == "bullish":
            direction = "CALLS"
        elif ctx.get("trend_direction") == "bearish":
            direction = "PUTS"
        else:
            direction = "NEUTRAL"
        
        if "STRADDLE" in config["structures"][0] or "STRANGLE" in config["structures"][0]:
            structure = config["structures"][0]
        else:
            structure = f"OTM_{direction}"
        
        confidence = ctx.get("confidence", 50)
        score = ctx.get("score", 50)
        
        return StrategySignal(
            strategy=strategy,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            dte_range=config["dte_range"],
            structure=structure,
            entry_trigger=config["events"][0] if config["events"] else "TECHNICAL",
            key_skill=config["key_skill"],
            risk_level=config["risk_level"],
            score=score,
            metadata={
                "name": config["name"],
                "what_training": config["what_training"],
                "context": ctx,
            }
        )
    
    def get_strategy_summary(self, signals: List[StrategySignal]) -> Dict[str, Any]:
        """Get summary of classified strategies"""
        by_type = {}
        for sig in signals:
            type_name = sig.strategy.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append({
                "symbol": sig.symbol,
                "direction": sig.direction,
                "confidence": sig.confidence,
                "dte": sig.dte_range,
                "risk": sig.risk_level,
            })
        
        return {
            "total_signals": len(signals),
            "by_strategy": by_type,
            "highest_confidence": max(signals, key=lambda x: x.confidence) if signals else None,
        }


strategy_classifier = StrategyClassifier()
