"""
Engine 2: 0DTE Pressure Engine

Monitors options flow and unusual activity for 0DTE contracts.
Detects put/call pressure, strike clustering, and gamma walls.

Note: Full functionality requires Polygon.io paid plan for options data.
Falls back to stock-based momentum when options data unavailable.
"""

import requests
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from wsb_snake.config import (
    POLYGON_API_KEY, POLYGON_BASE_URL,
    ZERO_DTE_UNIVERSE
)
from wsb_snake.collectors.polygon_options import polygon_options
from wsb_snake.utils.logger import log
from wsb_snake.utils.session_regime import get_session_info


class PressureType(Enum):
    """Types of options pressure."""
    CALL_WALL = "call_wall"          # Heavy call OI at strike
    PUT_WALL = "put_wall"            # Heavy put OI at strike
    GAMMA_SQUEEZE = "gamma_squeeze"  # Dealers hedging drives momentum
    IV_SPIKE = "iv_spike"            # Volatility explosion
    FLOW_IMBALANCE = "flow_imbalance"  # Heavy directional flow


@dataclass
class PressureSignal:
    """A pressure detection signal."""
    ticker: str
    pressure_type: PressureType
    score: float  # 0-100
    
    # Market context
    spot_price: float
    atm_strike: float
    
    # Options metrics (when available)
    call_put_ratio: float
    total_volume: int
    avg_iv: float
    
    # Key levels
    resistance_strike: float
    support_strike: float
    max_pain: float
    
    # Evidence
    evidence: List[str]
    
    # Timestamps
    detected_at: datetime
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "pressure_type": self.pressure_type.value,
            "score": self.score,
            "spot_price": self.spot_price,
            "atm_strike": self.atm_strike,
            "call_put_ratio": self.call_put_ratio,
            "total_volume": self.total_volume,
            "avg_iv": self.avg_iv,
            "resistance_strike": self.resistance_strike,
            "support_strike": self.support_strike,
            "max_pain": self.max_pain,
            "evidence": self.evidence,
            "detected_at": self.detected_at.isoformat(),
        }


class PressureEngine:
    """
    Engine 2: 0DTE Pressure Engine
    
    Analyzes options flow for unusual activity that could drive 0DTE moves.
    """
    
    # Thresholds
    CALL_PUT_RATIO_BULLISH = 1.5   # Bullish above this
    CALL_PUT_RATIO_BEARISH = 0.67  # Bearish below this
    IV_SPIKE_THRESHOLD = 1.3       # 30% above average IV
    MIN_SCORE_TO_SIGNAL = 40
    
    def __init__(self):
        self.polygon_key = POLYGON_API_KEY
        self.options_available = self._check_options_access()
        
    def _check_options_access(self) -> bool:
        """Check if we have options data access."""
        # Try a simple options request
        try:
            chain = polygon_options.get_options_chain("SPY", limit=1)
            return len(chain) > 0
        except:
            return False
    
    def scan_universe(self) -> List[PressureSignal]:
        """
        Scan the 0DTE universe for options pressure.
        
        Returns:
            List of PressureSignal objects
        """
        signals = []
        session_info = get_session_info()
        
        log.info(f"Pressure scan starting | Options access: {self.options_available}")
        
        for ticker in ZERO_DTE_UNIVERSE:
            try:
                signal = self._analyze_ticker(ticker)
                if signal and signal.score >= self.MIN_SCORE_TO_SIGNAL:
                    signal.score *= session_info["signal_quality_multiplier"]
                    signals.append(signal)
                    log.info(f"Pressure detected: {ticker} | Score: {signal.score:.0f} | Type: {signal.pressure_type.value}")
            except Exception as e:
                log.warning(f"Pressure analysis failed for {ticker}: {e}")
        
        signals.sort(key=lambda s: s.score, reverse=True)
        log.info(f"Pressure scan complete | Found {len(signals)} signals")
        
        return signals
    
    def _analyze_ticker(self, ticker: str) -> Optional[PressureSignal]:
        """Analyze a single ticker for options pressure."""
        
        # Get spot price first
        quote = self._get_spot_price(ticker)
        if not quote:
            return None
        
        spot_price = quote["price"]
        
        if self.options_available:
            # Full analysis with options data
            return self._analyze_with_options(ticker, spot_price)
        else:
            # Fallback: infer pressure from price action
            return self._analyze_without_options(ticker, quote)
    
    def _analyze_with_options(self, ticker: str, spot_price: float) -> Optional[PressureSignal]:
        """Full analysis with options data."""
        chain = polygon_options.get_0dte_chain(ticker, spot_price, strike_range=10)
        
        if not chain or not chain.get("metrics"):
            return None
        
        metrics = chain["metrics"]
        
        # Calculate pressure score
        score, pressure_type, evidence = self._score_pressure(
            call_put_volume_ratio=metrics.get("call_put_volume_ratio", 1),
            call_put_oi_ratio=metrics.get("call_put_oi_ratio", 1),
            avg_iv=metrics.get("avg_iv", 0),
            top_volume_strikes=metrics.get("top_volume_strikes", []),
            spot_price=spot_price,
            atm_strike=chain.get("atm_strike", spot_price),
        )
        
        if score < self.MIN_SCORE_TO_SIGNAL:
            return None
        
        # Find key levels
        resistance_strike, support_strike = self._find_key_levels(
            chain.get("calls", []),
            chain.get("puts", []),
            spot_price
        )
        
        return PressureSignal(
            ticker=ticker,
            pressure_type=pressure_type,
            score=score,
            spot_price=spot_price,
            atm_strike=chain.get("atm_strike", spot_price),
            call_put_ratio=metrics.get("call_put_volume_ratio", 1),
            total_volume=metrics.get("total_call_volume", 0) + metrics.get("total_put_volume", 0),
            avg_iv=metrics.get("avg_iv", 0),
            resistance_strike=resistance_strike,
            support_strike=support_strike,
            max_pain=self._calculate_max_pain(chain.get("calls", []), chain.get("puts", [])),
            evidence=evidence,
            detected_at=datetime.utcnow(),
        )
    
    def _analyze_without_options(self, ticker: str, quote: Dict) -> Optional[PressureSignal]:
        """Fallback analysis using price action as options proxy."""
        
        # Use price momentum as proxy for options pressure
        change_pct = quote.get("change_pct", 0) * 100
        volume = quote.get("volume", 0)
        
        # Simple heuristic: big moves with high volume suggest options activity
        score = 0
        evidence = []
        pressure_type = PressureType.FLOW_IMBALANCE
        
        if abs(change_pct) >= 2.0:
            score += min(40, abs(change_pct) * 10)
            direction = "bullish" if change_pct > 0 else "bearish"
            evidence.append(f"Strong {direction} move ({change_pct:+.1f}%)")
            
            if change_pct > 0:
                pressure_type = PressureType.CALL_WALL
            else:
                pressure_type = PressureType.PUT_WALL
        
        if volume > 0:
            evidence.append(f"Volume: {volume:,.0f}")
            score += 10
        
        if score < self.MIN_SCORE_TO_SIGNAL:
            return None
        
        spot_price = quote.get("price", 0)
        
        return PressureSignal(
            ticker=ticker,
            pressure_type=pressure_type,
            score=score,
            spot_price=spot_price,
            atm_strike=spot_price,  # Approximate
            call_put_ratio=1.0,  # Unknown
            total_volume=0,  # Unknown
            avg_iv=0,  # Unknown
            resistance_strike=spot_price * 1.02,  # 2% above
            support_strike=spot_price * 0.98,  # 2% below
            max_pain=spot_price,
            evidence=evidence + ["⚠️ Limited data - options access needed"],
            detected_at=datetime.utcnow(),
        )
    
    def _score_pressure(
        self,
        call_put_volume_ratio: float,
        call_put_oi_ratio: float,
        avg_iv: float,
        top_volume_strikes: List[Dict],
        spot_price: float,
        atm_strike: float,
    ) -> tuple:
        """Score options pressure."""
        score = 0
        evidence = []
        pressure_type = PressureType.FLOW_IMBALANCE
        
        # Call/Put ratio imbalance (0-30 points)
        if call_put_volume_ratio >= self.CALL_PUT_RATIO_BULLISH:
            ratio_score = min(30, (call_put_volume_ratio - 1) * 15)
            score += ratio_score
            evidence.append(f"Call/Put ratio: {call_put_volume_ratio:.2f}x (bullish)")
            pressure_type = PressureType.CALL_WALL
        elif call_put_volume_ratio <= self.CALL_PUT_RATIO_BEARISH:
            ratio_score = min(30, (1 - call_put_volume_ratio) * 30)
            score += ratio_score
            evidence.append(f"Call/Put ratio: {call_put_volume_ratio:.2f}x (bearish)")
            pressure_type = PressureType.PUT_WALL
        
        # IV spike (0-25 points)
        if avg_iv >= 0.5:  # 50%+ IV is elevated
            iv_score = min(25, avg_iv * 25)
            score += iv_score
            evidence.append(f"IV: {avg_iv*100:.0f}%")
            if avg_iv >= 0.8:
                pressure_type = PressureType.IV_SPIKE
        
        # Strike clustering - high volume near ATM (0-25 points)
        atm_volume = 0
        for s in top_volume_strikes:
            if abs(s.get("strike", 0) - atm_strike) <= 2:
                atm_volume += s.get("volume", 0)
        
        if atm_volume > 10000:
            cluster_score = min(25, atm_volume / 1000)
            score += cluster_score
            evidence.append(f"ATM volume clustering: {atm_volume:,.0f}")
            pressure_type = PressureType.GAMMA_SQUEEZE
        
        # Bonus for OI imbalance matching volume imbalance (0-20 points)
        if (call_put_volume_ratio > 1 and call_put_oi_ratio > 1) or \
           (call_put_volume_ratio < 1 and call_put_oi_ratio < 1):
            score += 20
            evidence.append("Volume/OI direction aligned")
        
        return score, pressure_type, evidence
    
    def _find_key_levels(self, calls: List[Dict], puts: List[Dict], spot: float) -> tuple:
        """Find resistance (call wall) and support (put wall) levels."""
        
        # Find highest call OI above spot (resistance)
        resistance = spot * 1.05  # Default 5% above
        calls_above = [c for c in calls if c.get("strike", 0) > spot]
        if calls_above:
            max_call = max(calls_above, key=lambda c: c.get("open_interest", 0))
            resistance = max_call.get("strike", resistance)
        
        # Find highest put OI below spot (support)
        support = spot * 0.95  # Default 5% below
        puts_below = [p for p in puts if p.get("strike", 0) < spot]
        if puts_below:
            max_put = max(puts_below, key=lambda p: p.get("open_interest", 0))
            support = max_put.get("strike", support)
        
        return resistance, support
    
    def _calculate_max_pain(self, calls: List[Dict], puts: List[Dict]) -> float:
        """Calculate max pain strike."""
        if not calls and not puts:
            return 0
        
        # Get unique strikes
        all_strikes = set()
        for c in calls:
            all_strikes.add(c.get("strike", 0))
        for p in puts:
            all_strikes.add(p.get("strike", 0))
        
        if not all_strikes:
            return 0
        
        # For each strike, calculate total pain (ITM OI value)
        min_pain = float('inf')
        max_pain_strike = 0
        
        for strike in sorted(all_strikes):
            pain = 0
            
            # Pain from calls: calls with strike < test_strike are ITM
            for c in calls:
                if c.get("strike", 0) < strike:
                    pain += c.get("open_interest", 0) * (strike - c["strike"])
            
            # Pain from puts: puts with strike > test_strike are ITM
            for p in puts:
                if p.get("strike", 0) > strike:
                    pain += p.get("open_interest", 0) * (p["strike"] - strike)
            
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = strike
        
        return max_pain_strike
    
    def _get_spot_price(self, ticker: str) -> Optional[Dict]:
        """Get current spot price."""
        quote = polygon_options.get_quote(ticker)
        if quote:
            return quote
        
        # Fallback to aggregates
        if not self.polygon_key:
            return None
        
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/prev"
        params = {"apiKey": self.polygon_key}
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    r = data["results"][0]
                    return {
                        "price": r.get("c", 0),
                        "volume": r.get("v", 0),
                        "change_pct": 0,  # Can't calculate without current day
                    }
        except Exception as e:
            log.warning(f"Failed to get spot for {ticker}: {e}")
        
        return None


# Global instance
pressure_engine = PressureEngine()


def run_pressure_scan() -> List[Dict]:
    """Run pressure scan and return signals as dicts."""
    signals = pressure_engine.scan_universe()
    return [s.to_dict() for s in signals]
