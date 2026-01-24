"""
Engine 2: Enhanced 0DTE Pressure Engine

Monitors options flow and unusual activity for 0DTE contracts.
Now enhanced with full technical analysis using Polygon basic plan:
- RSI, SMA, EMA, MACD indicators
- Strike structure analysis from options contracts
- Market regime detection
- Intraday momentum signals
"""

from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from wsb_snake.config import ZERO_DTE_UNIVERSE
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced, get_full_analysis
from wsb_snake.utils.logger import log
from wsb_snake.utils.session_regime import get_session_info


class PressureType(Enum):
    """Types of pressure signals."""
    CALL_WALL = "call_wall"              # Heavy call strikes above
    PUT_WALL = "put_wall"                # Heavy put strikes below
    TECHNICAL_BULLISH = "tech_bullish"   # Technical indicators bullish
    TECHNICAL_BEARISH = "tech_bearish"   # Technical indicators bearish
    MOMENTUM_SURGE = "momentum_surge"    # Momentum accelerating
    RSI_EXTREME = "rsi_extreme"          # RSI overbought/oversold
    BREAKOUT = "breakout"                # Breaking key levels


@dataclass
class PressureSignal:
    """A pressure detection signal."""
    ticker: str
    pressure_type: PressureType
    score: float  # 0-100
    
    # Market context
    spot_price: float
    change_pct: float
    
    # Technical indicators
    rsi: float
    sma_20: float
    ema_9: float
    
    # Options structure (when available)
    support_level: float
    resistance_level: float
    put_call_contract_ratio: float
    
    # Signals detected
    signals: List[tuple]  # (name, strength)
    direction: str  # LONG, SHORT, NEUTRAL
    
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
            "change_pct": self.change_pct,
            "rsi": self.rsi,
            "sma_20": self.sma_20,
            "ema_9": self.ema_9,
            "support_level": self.support_level,
            "resistance_level": self.resistance_level,
            "put_call_contract_ratio": self.put_call_contract_ratio,
            "signals": self.signals,
            "direction": self.direction,
            "evidence": self.evidence,
            "detected_at": self.detected_at.isoformat(),
        }


class PressureEngine:
    """
    Engine 2: Enhanced 0DTE Pressure Engine
    
    Uses all available Polygon basic plan data:
    - Technical indicators (RSI, SMA, EMA, MACD)
    - Strike structure from options contracts
    - Intraday momentum analysis
    - Market regime context
    """
    
    # Thresholds
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    RSI_EXTREME_HIGH = 80
    RSI_EXTREME_LOW = 20
    MIN_SCORE_TO_SIGNAL = 35
    
    def __init__(self):
        self._market_regime = None
        self._regime_updated = None
        
    def scan_universe(self) -> List[PressureSignal]:
        """
        Scan the 0DTE universe for pressure signals.
        
        Returns:
            List of PressureSignal objects
        """
        signals = []
        session_info = get_session_info()
        
        # Update market regime (cache for 5 minutes)
        self._update_market_regime()
        
        log.info(f"Pressure scan starting | Regime: {self._market_regime.get('regime', 'unknown') if self._market_regime else 'unknown'}")
        
        # Skip scanning if session multiplier is 0 (market closed)
        if session_info["signal_quality_multiplier"] == 0:
            log.info("Pressure scan skipped - market closed")
            return signals
        
        for ticker in ZERO_DTE_UNIVERSE:
            try:
                signal = self._analyze_ticker(ticker)
                if signal and signal.score >= self.MIN_SCORE_TO_SIGNAL:
                    # Apply session multiplier
                    signal.score *= session_info["signal_quality_multiplier"]
                    # Only append if score is still above threshold after multiplier
                    if signal.score >= self.MIN_SCORE_TO_SIGNAL:
                        signals.append(signal)
                        log.info(f"Pressure detected: {ticker} | Score: {signal.score:.0f} | Type: {signal.pressure_type.value}")
            except Exception as e:
                log.warning(f"Pressure analysis failed for {ticker}: {e}")
        
        signals.sort(key=lambda s: s.score, reverse=True)
        log.info(f"Pressure scan complete | Found {len(signals)} signals")
        
        return signals
    
    def _update_market_regime(self):
        """Update market regime with caching."""
        now = datetime.now()
        if self._regime_updated and (now - self._regime_updated).seconds < 300:
            return  # Use cached
        
        try:
            self._market_regime = polygon_enhanced.get_market_regime()
            self._regime_updated = now
        except Exception as e:
            log.warning(f"Failed to get market regime: {e}")
            self._market_regime = {"regime": "unknown", "score": 0}
    
    def _analyze_ticker(self, ticker: str) -> Optional[PressureSignal]:
        """Analyze a single ticker using all available data."""
        
        # Get comprehensive analysis
        analysis = get_full_analysis(ticker)
        
        if not analysis.get("price"):
            return None
        
        price = analysis["price"]
        technicals = analysis.get("technicals", {})
        momentum = analysis.get("momentum", {})
        options = analysis.get("options_structure", {})
        all_signals = analysis.get("all_signals", [])
        
        # Calculate score from signals
        score, pressure_type, evidence = self._score_signals(
            all_signals,
            technicals,
            momentum,
            options,
            price
        )
        
        if score < self.MIN_SCORE_TO_SIGNAL:
            return None
        
        # Apply market regime modifier
        if self._market_regime:
            regime_score = self._market_regime.get("score", 0)
            direction = analysis.get("direction", "NEUTRAL")
            
            # Boost score if aligned with regime
            if (direction == "LONG" and regime_score > 5) or \
               (direction == "SHORT" and regime_score < -5):
                score *= 1.2
                evidence.append(f"Aligned with market regime: {self._market_regime.get('regime', 'unknown')}")
        
        # Extract indicator values
        rsi_data = technicals.get("rsi", {})
        sma_data = technicals.get("sma_20", {})
        ema_data = technicals.get("ema_9", {})
        snapshot = technicals.get("snapshot", {})
        
        return PressureSignal(
            ticker=ticker,
            pressure_type=pressure_type,
            score=min(100, score),
            spot_price=price,
            change_pct=snapshot.get("change_pct", 0),
            rsi=rsi_data.get("current", 50) if rsi_data else 50,
            sma_20=sma_data.get("current", 0) if sma_data else 0,
            ema_9=ema_data.get("current", 0) if ema_data else 0,
            support_level=options.get("support_levels", [0])[0] if options.get("support_levels") else 0,
            resistance_level=options.get("resistance_levels", [0])[0] if options.get("resistance_levels") else 0,
            put_call_contract_ratio=options.get("put_call_ratio", 1),
            signals=all_signals,
            direction=analysis.get("direction", "NEUTRAL"),
            evidence=evidence,
            detected_at=datetime.now(),
        )
    
    def _score_signals(
        self,
        signals: List[tuple],
        technicals: Dict,
        momentum: Dict,
        options: Dict,
        price: float
    ) -> tuple:
        """Score all signals and determine pressure type."""
        score = 0
        evidence = []
        pressure_type = PressureType.TECHNICAL_BULLISH
        
        # Sum up signal scores (each signal is (name, strength))
        net_direction = 0
        for name, strength in signals:
            score += abs(strength) * 10  # Convert to 0-100 scale
            net_direction += strength
            evidence.append(f"{name}: {'+' if strength > 0 else ''}{strength:.1f}")
        
        # Determine pressure type based on signals
        rsi = technicals.get("rsi", {})
        if rsi:
            rsi_val = rsi.get("current", 50)
            if rsi_val >= self.RSI_EXTREME_HIGH or rsi_val <= self.RSI_EXTREME_LOW:
                pressure_type = PressureType.RSI_EXTREME
                score += 15
                evidence.append(f"RSI extreme: {rsi_val:.0f}")
        
        # Check for breakout
        snapshot = technicals.get("snapshot", {})
        if snapshot:
            high = snapshot.get("today_high", 0)
            low = snapshot.get("today_low", 0)
            if high and low:
                range_pct = (high - low) / low * 100 if low else 0
                if range_pct > 2:  # 2% range is significant
                    score += 10
                    evidence.append(f"Wide range: {range_pct:.1f}%")
                    pressure_type = PressureType.BREAKOUT
        
        # Momentum surge detection
        if momentum.get("volume_ratio", 1) > 1.5:
            score += 15
            evidence.append(f"Volume surge: {momentum['volume_ratio']:.1f}x")
            pressure_type = PressureType.MOMENTUM_SURGE
        
        # Options structure analysis
        if options.get("available"):
            pc_ratio = options.get("put_call_ratio", 1)
            if pc_ratio > 1.3:
                score += 10
                evidence.append(f"More puts than calls: {pc_ratio:.2f} P/C")
                pressure_type = PressureType.PUT_WALL
            elif pc_ratio < 0.7:
                score += 10
                evidence.append(f"More calls than puts: {pc_ratio:.2f} P/C")
                pressure_type = PressureType.CALL_WALL
            
            # Support/resistance proximity
            if options.get("resistance_levels"):
                nearest_resistance = options["resistance_levels"][0]
                dist_to_resistance = (nearest_resistance - price) / price * 100
                if 0 < dist_to_resistance < 1:  # Within 1% of resistance
                    score += 10
                    evidence.append(f"Near resistance: ${nearest_resistance}")
            
            if options.get("support_levels"):
                nearest_support = options["support_levels"][0]
                dist_to_support = (price - nearest_support) / price * 100
                if 0 < dist_to_support < 1:  # Within 1% of support
                    score += 10
                    evidence.append(f"Near support: ${nearest_support}")
        
        # Final direction determination
        if net_direction > 2:
            if pressure_type not in [PressureType.RSI_EXTREME, PressureType.BREAKOUT]:
                pressure_type = PressureType.TECHNICAL_BULLISH
        elif net_direction < -2:
            if pressure_type not in [PressureType.RSI_EXTREME, PressureType.BREAKOUT]:
                pressure_type = PressureType.TECHNICAL_BEARISH
        
        return score, pressure_type, evidence
    
    def get_market_context(self) -> Dict:
        """Get current market regime context."""
        self._update_market_regime()
        return self._market_regime or {"regime": "unknown", "score": 0}


# Global instance
pressure_engine = PressureEngine()


def run_pressure_scan() -> List[Dict]:
    """Run pressure scan and return signals as dicts."""
    signals = pressure_engine.scan_universe()
    return [s.to_dict() for s in signals]


def get_market_regime() -> Dict:
    """Get current market regime."""
    return pressure_engine.get_market_context()
