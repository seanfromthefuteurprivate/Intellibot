"""
0DTE Volatility Engine (Phase 1a)

Focus: Volatility expansion, NOT direction
Strategy: ATM straddles/strangles on days where IV is underpriced

Key Concepts:
- IV vs Realized Volatility (RV) gap
- VIX term structure (contango = complacency, backwardation = fear)
- Economic event days (CPI, Jobs, FOMC) where volatility is underpriced
- Gamma scalping opportunities

This is NOT directional trading - we profit from volatility expansion regardless of direction.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, date
from dataclasses import dataclass

from wsb_snake.utils.logger import log


@dataclass
class VolatilitySetup:
    """A 0DTE volatility expansion opportunity"""
    symbol: str
    signal_type: str
    structure: str
    iv_percentile: float
    expected_move: float
    historical_move: float
    vol_edge: float
    catalyst: str
    confidence: float
    risk_level: str
    entry_timing: str
    exit_plan: str
    timestamp: str


class ZeroDTEVolatilityEngine:
    """
    Detects 0DTE volatility expansion opportunities.
    
    NOT directional - we're betting on volatility being underpriced.
    
    Key triggers:
    1. CPI Day - often most volatile, IV frequently underpriced
    2. Jobs Report (NFP) - big moves, volatility crush after
    3. FOMC Days - whipsaw potential
    4. VIX term structure showing complacency
    """
    
    MAJOR_MACRO_EVENTS = {
        "CPI": {"typical_move": 1.5, "iv_premium": 0.3},
        "JOBS": {"typical_move": 1.0, "iv_premium": 0.2},
        "FOMC": {"typical_move": 1.2, "iv_premium": 0.4},
        "GDP": {"typical_move": 0.8, "iv_premium": 0.15},
        "RETAIL_SALES": {"typical_move": 0.6, "iv_premium": 0.1},
    }
    
    def __init__(self):
        self.active_setups: List[VolatilitySetup] = []
        self.last_scan = None
        log.info("0DTE Volatility Engine initialized (Phase 1a)")
    
    def analyze_volatility_setup(
        self,
        symbol: str,
        current_iv: float,
        iv_percentile: float,
        vix_level: float,
        vix_structure: str,
        macro_event: Optional[str] = None,
        options_data: Optional[Dict] = None
    ) -> Optional[VolatilitySetup]:
        """
        Analyze if there's a volatility expansion opportunity.
        
        Args:
            symbol: Ticker symbol (SPY, QQQ, etc.)
            current_iv: Current implied volatility
            iv_percentile: IV percentile (0-100)
            vix_level: Current VIX level
            vix_structure: contango/backwardation/flat
            macro_event: Type of macro event if any
            options_data: ATM option prices/greeks
            
        Returns:
            VolatilitySetup if opportunity detected, None otherwise
        """
        
        has_catalyst = macro_event is not None
        event_config = self.MAJOR_MACRO_EVENTS.get(macro_event, {}) if macro_event else {}
        
        typical_move = event_config.get("typical_move", 0.5)
        iv_premium = event_config.get("iv_premium", 0.1)
        
        iv_cheap = iv_percentile < 35
        vix_complacent = vix_structure in ["steep_contango", "contango"] and vix_level < 18
        
        expected_move = self._calculate_expected_move(current_iv, symbol)
        historical_move = self._get_historical_event_move(symbol, macro_event)
        
        vol_edge = 0
        if historical_move > 0 and expected_move > 0:
            vol_edge = (historical_move - expected_move) / expected_move * 100
        
        should_trade = False
        confidence = 0
        
        if has_catalyst:
            if iv_cheap and vol_edge > 20:
                should_trade = True
                confidence = min(75, 50 + vol_edge / 2)
            elif vix_complacent and macro_event in ["CPI", "FOMC", "JOBS"]:
                should_trade = True
                confidence = 60
        else:
            if iv_percentile < 20 and vix_complacent:
                should_trade = True
                confidence = 50
        
        if not should_trade:
            return None
        
        structure = self._select_structure(iv_percentile, vix_level, macro_event)
        
        return VolatilitySetup(
            symbol=symbol,
            signal_type="0DTE_VOLATILITY_EXPANSION",
            structure=structure,
            iv_percentile=iv_percentile,
            expected_move=expected_move,
            historical_move=historical_move,
            vol_edge=vol_edge,
            catalyst=macro_event or "LOW_IV_ENVIRONMENT",
            confidence=confidence,
            risk_level="HIGH" if macro_event else "MEDIUM",
            entry_timing=self._get_entry_timing(macro_event),
            exit_plan=self._get_exit_plan(macro_event),
            timestamp=datetime.now().isoformat(),
        )
    
    def _calculate_expected_move(self, iv: float, symbol: str) -> float:
        """Calculate expected move based on IV"""
        daily_vol = iv / 16
        return daily_vol
    
    def _get_historical_event_move(self, symbol: str, event: Optional[str]) -> float:
        """Get average historical move for event type"""
        historical_moves = {
            "SPY": {"CPI": 1.3, "JOBS": 0.9, "FOMC": 1.1, "GDP": 0.7},
            "QQQ": {"CPI": 1.6, "JOBS": 1.1, "FOMC": 1.3, "GDP": 0.8},
            "IWM": {"CPI": 1.8, "JOBS": 1.3, "FOMC": 1.5, "GDP": 0.9},
        }
        
        if not event:
            return 0.8
        
        symbol_data = historical_moves.get(symbol, historical_moves.get("SPY", {}))
        if event:
            return symbol_data.get(event, 0.8)
        return 0.8
    
    def _select_structure(
        self, 
        iv_percentile: float, 
        vix: float, 
        event: Optional[str]
    ) -> str:
        """Select optimal structure based on conditions"""
        if iv_percentile < 25:
            return "ATM_STRADDLE"
        elif iv_percentile < 40 and event:
            return "ATM_STRADDLE"
        else:
            return "WIDE_STRANGLE"
    
    def _get_entry_timing(self, event: Optional[str]) -> str:
        """Get recommended entry timing"""
        timing_map = {
            "CPI": "Enter 15-30 min before 8:30 ET release",
            "JOBS": "Enter 15-30 min before 8:30 ET release",
            "FOMC": "Enter 30-60 min before 2:00 PM ET",
            "GDP": "Enter 15 min before 8:30 ET release",
        }
        if event and event in timing_map:
            return timing_map[event]
        return "Enter during first 30 min of session"
    
    def _get_exit_plan(self, event: Optional[str]) -> str:
        """Get recommended exit plan"""
        if event in ["CPI", "JOBS", "GDP"]:
            return "Exit winning leg within 30-60 min, let losing leg expire worthless"
        elif event == "FOMC":
            return "Exit before Powell speaks OR after initial spike settles"
        else:
            return "Exit when position doubles OR at 2:00 PM ET"
    
    def get_macro_calendar(self) -> List[Dict]:
        """Get upcoming macro events from FRED/economic calendar"""
        import calendar
        from datetime import date, timedelta
        
        today = date.today()
        events = []
        
        if today.weekday() == 3 and 10 <= today.day <= 16:
            events.append({
                "date": today.isoformat(),
                "event": "CPI",
                "importance": "HIGH",
                "typical_move": 1.5,
            })
        
        if today.weekday() == 4 and today.day <= 7:
            events.append({
                "date": today.isoformat(),
                "event": "JOBS",
                "importance": "HIGH",
                "typical_move": 1.0,
            })
        
        return events
    
    def scan_volatility_opportunities(self, market_data: Dict) -> List[VolatilitySetup]:
        """
        Scan for all volatility opportunities.
        
        Args:
            market_data: Dict with current market conditions
            
        Returns:
            List of volatility setups
        """
        setups = []
        
        symbols = ["SPY", "QQQ", "IWM"]
        
        vix_level = market_data.get("vix", 15.0)
        vix_structure = market_data.get("vix_structure", "contango")
        macro_event = market_data.get("macro_event")
        
        for symbol in symbols:
            current_iv = market_data.get(f"{symbol}_iv", 0.20)
            iv_percentile = market_data.get(f"{symbol}_iv_percentile", 50)
            
            setup = self.analyze_volatility_setup(
                symbol=symbol,
                current_iv=current_iv,
                iv_percentile=iv_percentile,
                vix_level=vix_level,
                vix_structure=vix_structure,
                macro_event=macro_event,
            )
            
            if setup:
                setups.append(setup)
                log.info(f"0DTE Vol setup: {symbol} | {setup.structure} | Edge: {setup.vol_edge:.1f}%")
        
        self.active_setups = setups
        self.last_scan = datetime.now()
        
        return setups
    
    def format_telegram_alert(self, setup: VolatilitySetup) -> str:
        """Format setup for Telegram alert"""
        return f"""
*[0DTE VOLATILITY PLAY]*

*{setup.symbol}* | {setup.structure}
━━━━━━━━━━━━━━━━━━

*Volatility Edge:* {setup.vol_edge:.1f}%
*Expected Move:* {setup.expected_move:.2f}%
*Historical Move:* {setup.historical_move:.2f}%
*IV Percentile:* {setup.iv_percentile:.0f}%

*Catalyst:* {setup.catalyst}
*Entry:* {setup.entry_timing}
*Exit:* {setup.exit_plan}

*Risk:* {setup.risk_level}
*Confidence:* {setup.confidence:.0f}%

_This is NOT a directional trade - profit from volatility expansion_
"""


zero_dte_volatility = ZeroDTEVolatilityEngine()
