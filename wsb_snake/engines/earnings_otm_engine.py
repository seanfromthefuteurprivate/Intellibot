"""
Earnings OTM Directional Engine (Phase 1b)

Focus: Cheap OTM "lotto" plays 7-10 DTE before earnings
Strategy: Asymmetric risk/reward - lose small, win big

Key Concepts:
- Target stocks with history of large earnings moves
- Look for OTM strikes where IV is relatively cheap
- Focus on 7-10 DTE to capture pre-earnings IV expansion
- Exit BEFORE earnings or ride through if conviction is high

This is a directional bet using cheap OTM options for asymmetric payoffs.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, date, timedelta
from dataclasses import dataclass

from wsb_snake.utils.logger import log


@dataclass 
class EarningsOTMSetup:
    """An earnings OTM directional opportunity"""
    symbol: str
    direction: str
    strike_type: str
    dte: int
    earnings_date: str
    expected_move: float
    historical_move: float
    surprise_history: str
    iv_rank: float
    recommendation: str
    entry_price_range: str
    target_return: str
    max_loss: str
    confidence: float
    timestamp: str


class EarningsOTMEngine:
    """
    Detects OTM directional opportunities before earnings.
    
    The "lotto" approach:
    - Buy cheap OTM calls/puts 7-10 days before earnings
    - Risk small amount for potential 3-10x return
    - IV expansion alone can double position value
    - If earnings gap is big, can 10x+
    
    Key filters:
    1. Stocks with history of large earnings moves (>5%)
    2. Current IV rank not too high (<60)
    3. Historical surprise rate (beat/miss pattern)
    4. Sector momentum alignment
    """
    
    EARNINGS_MOVERS = {
        "TSLA": {"avg_move": 8.5, "surprise_rate": 0.65, "direction_bias": "volatile"},
        "NVDA": {"avg_move": 7.0, "surprise_rate": 0.75, "direction_bias": "bullish"},
        "NFLX": {"avg_move": 10.0, "surprise_rate": 0.55, "direction_bias": "volatile"},
        "META": {"avg_move": 8.0, "surprise_rate": 0.70, "direction_bias": "bullish"},
        "AMZN": {"avg_move": 6.5, "surprise_rate": 0.65, "direction_bias": "bullish"},
        "GOOGL": {"avg_move": 5.5, "surprise_rate": 0.60, "direction_bias": "neutral"},
        "AAPL": {"avg_move": 4.0, "surprise_rate": 0.75, "direction_bias": "neutral"},
        "MSFT": {"avg_move": 4.5, "surprise_rate": 0.80, "direction_bias": "bullish"},
        "AMD": {"avg_move": 9.0, "surprise_rate": 0.55, "direction_bias": "volatile"},
        "CRM": {"avg_move": 7.5, "surprise_rate": 0.65, "direction_bias": "bullish"},
        "SHOP": {"avg_move": 12.0, "surprise_rate": 0.50, "direction_bias": "volatile"},
        "SQ": {"avg_move": 10.0, "surprise_rate": 0.55, "direction_bias": "volatile"},
        "COIN": {"avg_move": 15.0, "surprise_rate": 0.50, "direction_bias": "volatile"},
        "SNOW": {"avg_move": 12.0, "surprise_rate": 0.55, "direction_bias": "volatile"},
        "PLTR": {"avg_move": 10.0, "surprise_rate": 0.60, "direction_bias": "volatile"},
    }
    
    def __init__(self):
        self.active_setups: List[EarningsOTMSetup] = []
        self.last_scan = None
        log.info("Earnings OTM Engine initialized (Phase 1b)")
    
    def analyze_earnings_setup(
        self,
        symbol: str,
        days_to_earnings: int,
        earnings_date: str,
        current_iv_rank: float,
        expected_move: float,
        price: float,
        sector_trend: str = "neutral",
        congressional_signal: Optional[str] = None
    ) -> Optional[EarningsOTMSetup]:
        """
        Analyze if there's an OTM earnings opportunity.
        
        Args:
            symbol: Ticker symbol
            days_to_earnings: Days until earnings
            earnings_date: Date of earnings
            current_iv_rank: Current IV rank (0-100)
            expected_move: Market expected move %
            price: Current stock price
            sector_trend: Sector momentum (bullish/bearish/neutral)
            congressional_signal: If congress is trading this stock
            
        Returns:
            EarningsOTMSetup if opportunity detected, None otherwise
        """
        
        if days_to_earnings < 5 or days_to_earnings > 14:
            return None
        
        stock_profile = self.EARNINGS_MOVERS.get(symbol)
        if not stock_profile:
            return None
        
        historical_move = stock_profile["avg_move"]
        surprise_rate = stock_profile["surprise_rate"]
        direction_bias = stock_profile["direction_bias"]
        
        iv_cheap = current_iv_rank < 60
        move_underpriced = historical_move > expected_move * 1.15
        
        if not (iv_cheap or move_underpriced):
            return None
        
        direction = self._determine_direction(
            direction_bias, 
            sector_trend, 
            surprise_rate,
            congressional_signal
        )
        
        confidence = self._calculate_confidence(
            iv_cheap, 
            move_underpriced, 
            surprise_rate,
            sector_trend,
            direction_bias
        )
        
        if confidence < 45:
            return None
        
        dte = min(days_to_earnings + 2, 14)
        
        strike_otm_pct = self._calculate_strike_distance(
            historical_move, 
            expected_move, 
            direction
        )
        
        if direction == "CALLS":
            strike_price = price * (1 + strike_otm_pct / 100)
            strike_type = f"${strike_price:.0f} Call ({strike_otm_pct:.1f}% OTM)"
        else:
            strike_price = price * (1 - strike_otm_pct / 100)
            strike_type = f"${strike_price:.0f} Put ({strike_otm_pct:.1f}% OTM)"
        
        return EarningsOTMSetup(
            symbol=symbol,
            direction=direction,
            strike_type=strike_type,
            dte=dte,
            earnings_date=earnings_date,
            expected_move=expected_move,
            historical_move=historical_move,
            surprise_history=f"{surprise_rate*100:.0f}% beat rate",
            iv_rank=current_iv_rank,
            recommendation=self._get_recommendation(confidence, direction),
            entry_price_range=self._get_entry_price_range(price, strike_otm_pct, dte),
            target_return="3-5x on moderate move, 10x+ on big gap",
            max_loss="100% of premium (size accordingly)",
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
        )
    
    def _determine_direction(
        self,
        historical_bias: str,
        sector_trend: str,
        surprise_rate: float,
        congress_signal: Optional[str]
    ) -> str:
        """Determine optimal direction for OTM play"""
        
        if congress_signal in ["STRONG_BUY", "BUY"]:
            return "CALLS"
        elif congress_signal in ["STRONG_SELL", "SELL"]:
            return "PUTS"
        
        if historical_bias == "bullish" and sector_trend != "bearish":
            return "CALLS"
        elif historical_bias == "bearish" and sector_trend != "bullish":
            return "PUTS"
        
        if sector_trend == "bullish" and surprise_rate >= 0.6:
            return "CALLS"
        elif sector_trend == "bearish":
            return "PUTS"
        
        return "CALLS" if surprise_rate >= 0.55 else "PUTS"
    
    def _calculate_confidence(
        self,
        iv_cheap: bool,
        move_underpriced: bool,
        surprise_rate: float,
        sector_trend: str,
        direction_bias: str
    ) -> float:
        """Calculate confidence score for the setup"""
        confidence = 40
        
        if iv_cheap:
            confidence += 10
        if move_underpriced:
            confidence += 15
        if surprise_rate >= 0.7:
            confidence += 10
        elif surprise_rate >= 0.6:
            confidence += 5
        
        if direction_bias == "bullish" and sector_trend == "bullish":
            confidence += 10
        elif direction_bias == "bearish" and sector_trend == "bearish":
            confidence += 10
        elif sector_trend != "neutral" and direction_bias == sector_trend:
            confidence += 5
        
        return min(confidence, 85)
    
    def _calculate_strike_distance(
        self,
        historical_move: float,
        expected_move: float,
        direction: str
    ) -> float:
        """Calculate how far OTM to go"""
        avg_gap = (historical_move + expected_move) / 2
        
        strike_distance = avg_gap * 0.6
        
        strike_distance = max(3.0, min(strike_distance, 15.0))
        
        return strike_distance
    
    def _get_recommendation(self, confidence: float, direction: str) -> str:
        """Get trade recommendation"""
        if confidence >= 70:
            return f"STRONG {direction} - High conviction lotto"
        elif confidence >= 60:
            return f"{direction} - Good risk/reward"
        else:
            return f"SPECULATIVE {direction} - Small position only"
    
    def _get_entry_price_range(
        self, 
        stock_price: float, 
        otm_pct: float, 
        dte: int
    ) -> str:
        """Estimate entry price range for the options"""
        base_premium_pct = 0.02
        
        time_factor = dte / 14
        otm_factor = max(0.3, 1 - otm_pct / 20)
        
        estimated_pct = base_premium_pct * time_factor * otm_factor
        estimated_price = stock_price * estimated_pct
        
        low = max(0.5, estimated_price * 0.7)
        high = estimated_price * 1.5
        
        return f"${low:.2f} - ${high:.2f} per contract"
    
    def scan_earnings_opportunities(
        self, 
        earnings_calendar: List[Dict],
        market_data: Dict
    ) -> List[EarningsOTMSetup]:
        """
        Scan for all earnings OTM opportunities.
        
        Args:
            earnings_calendar: List of upcoming earnings
            market_data: Current market conditions
            
        Returns:
            List of earnings OTM setups
        """
        setups = []
        
        for earnings in earnings_calendar:
            symbol = earnings.get("symbol", "")
            days_to = earnings.get("days_until", 0)
            
            if symbol not in self.EARNINGS_MOVERS:
                continue
            
            if days_to < 5 or days_to > 14:
                continue
            
            expected_move = earnings.get("expected_move", 5.0)
            iv_rank = earnings.get("iv_rank", 50.0)
            price = earnings.get("price", 100.0)
            earnings_date = earnings.get("date", "")
            sector_trend = market_data.get(f"{symbol}_sector_trend", "neutral")
            congress_signal = market_data.get(f"{symbol}_congress_signal")
            
            setup = self.analyze_earnings_setup(
                symbol=symbol,
                days_to_earnings=days_to,
                earnings_date=earnings_date,
                current_iv_rank=iv_rank,
                expected_move=expected_move,
                price=price,
                sector_trend=sector_trend,
                congressional_signal=congress_signal,
            )
            
            if setup:
                setups.append(setup)
                log.info(
                    f"Earnings OTM: {symbol} | {setup.direction} | "
                    f"DTE: {setup.dte} | Conf: {setup.confidence:.0f}%"
                )
        
        self.active_setups = setups
        self.last_scan = datetime.now()
        
        return setups
    
    def format_telegram_alert(self, setup: EarningsOTMSetup) -> str:
        """Format setup for Telegram alert"""
        direction_label = "BULLISH" if setup.direction == "CALLS" else "BEARISH"
        
        return f"""
*[EARNINGS OTM PLAY]*

*{setup.symbol}* | {setup.direction} ({direction_label})
━━━━━━━━━━━━━━━━━━

*Earnings:* {setup.earnings_date}
*DTE:* {setup.dte} days
*Strike:* {setup.strike_type}

*Expected Move:* {setup.expected_move:.1f}%
*Historical Move:* {setup.historical_move:.1f}%
*IV Rank:* {setup.iv_rank:.0f}%
*History:* {setup.surprise_history}

*Entry Range:* {setup.entry_price_range}
*Target:* {setup.target_return}
*Max Loss:* {setup.max_loss}

*{setup.recommendation}*
*Confidence:* {setup.confidence:.0f}%

_Cheap lotto play - size for total loss_
"""


earnings_otm_engine = EarningsOTMEngine()
