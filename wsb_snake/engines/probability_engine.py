"""
WSB Snake Probability Engine - P(Hit Target by Close)

This implements the mathematical probability layer that estimates:
1. P(underlying tags target level before close)
2. Time-to-hit distribution (hazard curve)
3. Optimal entry window detection
4. Chop Kill filter scoring

Uses realized volatility since we don't have real-time IV without the Options upgrade.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import numpy as np

from wsb_snake.config import POLYGON_API_KEY, POLYGON_BASE_URL, ZERO_DTE_UNIVERSE
from wsb_snake.utils.logger import log
from wsb_snake.utils.session_regime import get_session_info, get_eastern_time
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced


@dataclass
class ProbabilityEstimate:
    """Probability estimate for hitting a target."""
    ticker: str
    current_price: float
    target_level: float
    direction: str  # "up" or "down"
    
    # Core probabilities
    p_hit_by_close: float           # P(hit target before close)
    p_hit_5min: float               # P(hit in next 5 min)
    p_hit_10min: float              # P(hit in next 10 min)
    p_hit_20min: float              # P(hit in next 20 min)
    
    # Hazard curve (conditional probabilities)
    hazard_curve: Dict[str, float]  # {"0-5": p, "5-10": p, ...}
    
    # Supporting metrics
    effective_volatility: float     # Regime-adjusted volatility
    distance_pct: float             # Distance to target as %
    time_remaining_min: float       # Minutes to close
    regime_scalar: float            # Regime multiplier
    
    # Quality metrics
    confidence: str                 # "low", "medium", "high"
    entry_quality: str              # "optimal", "acceptable", "poor"
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "current_price": self.current_price,
            "target_level": self.target_level,
            "direction": self.direction,
            "p_hit_by_close": self.p_hit_by_close,
            "p_hit_5min": self.p_hit_5min,
            "p_hit_10min": self.p_hit_10min,
            "p_hit_20min": self.p_hit_20min,
            "hazard_curve": self.hazard_curve,
            "effective_volatility": self.effective_volatility,
            "distance_pct": self.distance_pct,
            "time_remaining_min": self.time_remaining_min,
            "confidence": self.confidence,
            "entry_quality": self.entry_quality,
        }


@dataclass 
class ChopScore:
    """Chop detection score."""
    ticker: str
    score: float                    # 0-100 (higher = more chop)
    vwap_crossings: int             # Number of VWAP crosses in window
    range_compression: float        # ATR compression ratio
    trend_strength: float           # 0-1 trend indicator
    mean_reversion_freq: float      # Frequency of reversals
    is_chop: bool                   # True if should block signals
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "score": self.score,
            "vwap_crossings": self.vwap_crossings,
            "range_compression": self.range_compression,
            "trend_strength": self.trend_strength,
            "is_chop": self.is_chop,
        }


class ProbabilityEngine:
    """
    The Probability Engine.
    
    Calculates P(hit target by close) using:
    - Realized volatility from recent bars
    - Distance to target level
    - Time remaining
    - Regime multipliers (trend/chop/panic)
    """
    
    # Trading hours (ET)
    MARKET_OPEN_HOUR = 9
    MARKET_OPEN_MINUTE = 30
    MARKET_CLOSE_HOUR = 16
    MARKET_CLOSE_MINUTE = 0
    
    # Regime scalars
    REGIME_SCALARS = {
        "strong_bullish": 1.3,
        "bullish": 1.1,
        "neutral": 1.0,
        "bearish": 1.1,  # Bearish can be trendy too
        "strong_bearish": 1.3,
        "chop": 0.7,
        "panic": 1.5,
    }
    
    # Chop thresholds
    CHOP_SCORE_BLOCK_THRESHOLD = 60
    VWAP_CROSSING_THRESHOLD = 5  # Too many crosses = chop
    
    def __init__(self):
        self._vol_cache: Dict[str, Tuple[datetime, float]] = {}
        self._chop_cache: Dict[str, Tuple[datetime, ChopScore]] = {}
    
    def _get_time_remaining(self) -> float:
        """Get minutes remaining until market close (ET timezone)."""
        now = get_eastern_time()
        # Create close time in ET
        close_time = now.replace(
            hour=self.MARKET_CLOSE_HOUR, 
            minute=self.MARKET_CLOSE_MINUTE,
            second=0,
            microsecond=0
        )
        
        if now >= close_time:
            return 0.0
        
        delta = close_time - now
        return delta.total_seconds() / 60
    
    def _get_realized_volatility(self, ticker: str, window_minutes: int = 30) -> float:
        """
        Calculate realized volatility from recent price bars.
        
        Returns annualized volatility as decimal (e.g., 0.25 = 25%).
        """
        # Check cache (valid for 5 min)
        cache_key = f"{ticker}_{window_minutes}"
        if cache_key in self._vol_cache:
            cached_time, cached_vol = self._vol_cache[cache_key]
            if (datetime.now() - cached_time).seconds < 300:
                return cached_vol
        
        try:
            # Get recent bars from Polygon
            momentum = polygon_enhanced.get_momentum_signals(ticker)
            
            if not momentum.get("returns"):
                # Fallback to typical volatility by ticker class
                if ticker in ["SPY", "QQQ", "IWM"]:
                    return 0.15  # ~15% annual for indices
                else:
                    return 0.40  # ~40% for single stocks
            
            returns = momentum.get("returns", {})
            
            # Use the shortest window return to estimate minute-level vol
            ret_1m = abs(returns.get("1m", 0))
            ret_5m = abs(returns.get("5m", 0)) / math.sqrt(5)  # Scale down
            ret_15m = abs(returns.get("15m", 0)) / math.sqrt(15)
            
            # Average the estimates
            avg_1m_return = (ret_1m + ret_5m + ret_15m) / 3
            
            # Annualize: sqrt(252 trading days * 390 minutes/day)
            annual_factor = math.sqrt(252 * 390)
            realized_vol = avg_1m_return * annual_factor
            
            # Clamp to reasonable range
            realized_vol = max(0.10, min(2.0, realized_vol))
            
            # Cache it
            self._vol_cache[cache_key] = (datetime.now(), realized_vol)
            
            return realized_vol
            
        except Exception as e:
            log.debug(f"Volatility calc failed for {ticker}: {e}")
            return 0.30  # Default fallback
    
    def _get_regime_scalar(self, ticker: str = "") -> Tuple[float, str]:
        """Get regime multiplier and regime name."""
        try:
            regime = polygon_enhanced.get_market_regime()
            if regime:
                regime_name = regime.get("regime", "neutral")
            else:
                regime_name = "neutral"
            scalar = self.REGIME_SCALARS.get(regime_name, 1.0)
            return scalar, regime_name
        except Exception:
            return 1.0, "neutral"
    
    def calculate_probability(
        self,
        ticker: str,
        target_level: float,
        current_price: Optional[float] = None,
    ) -> ProbabilityEstimate:
        """
        Calculate probability of hitting target by close.
        
        Uses barrier hitting approximation with drifted Brownian motion.
        """
        # Get current price if not provided
        if current_price is None or current_price == 0:
            momentum = polygon_enhanced.get_momentum_signals(ticker)
            if momentum:
                current_price = float(momentum.get("price", 0) or 0)
            else:
                current_price = 0.0
            if current_price == 0:
                # Fallback
                snapshot = polygon_enhanced.get_snapshot(ticker)
                if snapshot:
                    current_price = float(snapshot.get("price", 100) or 100)
                else:
                    current_price = 100.0
        
        # Ensure current_price is valid
        current_price = float(current_price) if current_price else 100.0
        
        # Calculate distance
        distance = target_level - current_price
        direction = "up" if distance > 0 else "down"
        distance_pct = abs(distance) / current_price if current_price > 0 else 0
        
        # Get volatility and regime
        realized_vol = self._get_realized_volatility(ticker)
        regime_scalar, regime_name = self._get_regime_scalar(ticker)
        
        # Time remaining
        time_remaining = self._get_time_remaining()
        tau_years = time_remaining / (252 * 390)  # Convert to year fraction
        
        if time_remaining <= 0 or tau_years <= 0:
            # Market closed
            return ProbabilityEstimate(
                ticker=ticker,
                current_price=float(current_price),
                target_level=target_level,
                direction=direction,
                p_hit_by_close=0.0,
                p_hit_5min=0.0,
                p_hit_10min=0.0,
                p_hit_20min=0.0,
                hazard_curve={"0-5": 0.0, "5-10": 0.0, "10-20": 0.0, "20+": 0.0},
                effective_volatility=realized_vol,
                distance_pct=distance_pct,
                time_remaining_min=0.0,
                regime_scalar=regime_scalar,
                confidence="low",
                entry_quality="poor",
            )
        
        # Effective volatility (regime-adjusted)
        effective_vol = realized_vol * regime_scalar
        
        # Drift proxy (use recent momentum)
        try:
            momentum = polygon_enhanced.get_momentum_signals(ticker)
            ret_5m = momentum.get("returns", {}).get("5m", 0)
            drift = ret_5m / 5  # Per-minute drift
            # Annualize drift for model
            drift_annual = drift * 252 * 390
        except Exception:
            drift_annual = 0
        
        # Normalized distance (z-score)
        if effective_vol * math.sqrt(tau_years) > 0:
            z = abs(distance_pct) / (effective_vol * math.sqrt(tau_years))
        else:
            z = float('inf')
        
        # P(hit by close) using normal CDF approximation
        # For barrier hitting: P ≈ Φ(-z + μ*√τ)
        drift_adjustment = drift_annual * math.sqrt(tau_years) if direction == "up" else -drift_annual * math.sqrt(tau_years)
        p_hit_by_close = float(stats.norm.cdf(-z + drift_adjustment))
        
        # Calculate time-bucketed probabilities
        def p_hit_by_time(minutes: float) -> float:
            if minutes <= 0:
                return 0.0
            tau_t = minutes / (252 * 390)
            if effective_vol * math.sqrt(tau_t) > 0:
                z_t = abs(distance_pct) / (effective_vol * math.sqrt(tau_t))
                drift_adj = drift_annual * math.sqrt(tau_t)
                return float(stats.norm.cdf(-z_t + drift_adj))
            return 0.0
        
        p_5 = p_hit_by_time(min(5, time_remaining))
        p_10 = p_hit_by_time(min(10, time_remaining))
        p_20 = p_hit_by_time(min(20, time_remaining))
        
        # Hazard curve (conditional probabilities)
        # P(hit in window | hasn't hit yet)
        hazard_curve = {
            "0-5": p_5,
            "5-10": (p_10 - p_5) / (1 - p_5) if p_5 < 1 else 0,
            "10-20": (p_20 - p_10) / (1 - p_10) if p_10 < 1 else 0,
            "20+": (p_hit_by_close - p_20) / (1 - p_20) if p_20 < 1 else 0,
        }
        
        # Confidence based on data quality
        if time_remaining > 30 and p_hit_by_close > 0.4:
            confidence = "high"
        elif time_remaining > 15 and p_hit_by_close > 0.3:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Entry quality
        # Best entries: high near-term probability, low theta burn risk
        if p_5 > 0.2 and p_hit_by_close > 0.5:
            entry_quality = "optimal"
        elif p_10 > 0.2 and p_hit_by_close > 0.4:
            entry_quality = "acceptable"
        else:
            entry_quality = "poor"
        
        return ProbabilityEstimate(
            ticker=ticker,
            current_price=float(current_price),
            target_level=target_level,
            direction=direction,
            p_hit_by_close=float(p_hit_by_close),
            p_hit_5min=float(p_5),
            p_hit_10min=float(p_10),
            p_hit_20min=float(p_20),
            hazard_curve=hazard_curve,
            effective_volatility=effective_vol,
            distance_pct=distance_pct,
            time_remaining_min=time_remaining,
            regime_scalar=regime_scalar,
            confidence=confidence,
            entry_quality=entry_quality,
        )
    
    def calculate_chop_score(self, ticker: str) -> ChopScore:
        """
        Calculate chop score to detect fake breakout conditions.
        
        High chop score = avoid trading, signals are likely fake.
        """
        # Check cache (valid for 2 min)
        if ticker in self._chop_cache:
            cached_time, cached_score = self._chop_cache[ticker]
            if (datetime.now() - cached_time).seconds < 120:
                return cached_score
        
        try:
            technicals = polygon_enhanced.get_full_technicals(ticker)
            momentum = polygon_enhanced.get_momentum_signals(ticker)
            
            # Get components
            signals = technicals.get("signals", [])
            
            # 1. Range compression (low ATR = compression)
            # Check if we're near day range extremes
            price = momentum.get("price", 0)
            day_high = momentum.get("day_high", price)
            day_low = momentum.get("day_low", price)
            day_range = day_high - day_low if day_high > day_low else 1
            
            # Position in day range (0.5 = middle = potentially choppy)
            if day_range > 0:
                range_position = (price - day_low) / day_range
            else:
                range_position = 0.5
            
            # Middle of range = more choppy
            range_compression = 1 - abs(range_position - 0.5) * 2  # 0-1, higher = more compressed
            
            # 2. Trend strength from technicals
            trend_signals = 0
            total_signals = len(signals)
            
            for sig_name, sig_score in signals:
                if "BULLISH" in sig_name or "ABOVE" in sig_name or "RISING" in sig_name:
                    trend_signals += sig_score
                elif "BEARISH" in sig_name or "BELOW" in sig_name or "FALLING" in sig_name:
                    trend_signals -= sig_score
            
            # Normalize trend strength (strong either direction = good)
            trend_strength = min(1.0, abs(trend_signals) / 5) if total_signals > 0 else 0
            
            # 3. VWAP crossings (estimate from returns volatility)
            # High short-term vs longer-term vol = whipsaw
            returns = momentum.get("returns", {})
            ret_1m = abs(returns.get("1m", 0))
            ret_5m = abs(returns.get("5m", 0))
            
            # If 1m moves are relatively large vs 5m, suggests whipsaw
            if ret_5m > 0:
                whipsaw_ratio = (ret_1m * 5) / ret_5m  # Should be ~1 if smooth
            else:
                whipsaw_ratio = 1.0
            
            vwap_crossings = max(0, int((whipsaw_ratio - 1) * 5))  # Estimate
            
            # 4. Mean reversion frequency
            mean_reversion_freq = max(0, whipsaw_ratio - 1)
            
            # Calculate final chop score
            # Higher = more chop
            chop_score = (
                range_compression * 30 +           # 0-30 from compression
                (1 - trend_strength) * 40 +        # 0-40 from weak trend
                min(vwap_crossings, 5) * 6         # 0-30 from crossings
            )
            
            is_chop = chop_score >= self.CHOP_SCORE_BLOCK_THRESHOLD
            
            result = ChopScore(
                ticker=ticker,
                score=chop_score,
                vwap_crossings=vwap_crossings,
                range_compression=range_compression,
                trend_strength=trend_strength,
                mean_reversion_freq=mean_reversion_freq,
                is_chop=is_chop,
            )
            
            self._chop_cache[ticker] = (get_eastern_time().replace(tzinfo=None), result)
            return result
            
        except Exception as e:
            log.debug(f"Chop score calc failed for {ticker}: {e}")
            return ChopScore(
                ticker=ticker,
                score=50,  # Neutral default
                vwap_crossings=0,
                range_compression=0.5,
                trend_strength=0.5,
                mean_reversion_freq=0,
                is_chop=False,
            )
    
    def should_block_signal(self, ticker: str) -> Tuple[bool, str]:
        """
        Check if signals for this ticker should be blocked.
        
        Returns (should_block, reason).
        """
        chop = self.calculate_chop_score(ticker)
        
        if chop.is_chop:
            return True, f"Chop score {chop.score:.0f} (threshold: {self.CHOP_SCORE_BLOCK_THRESHOLD})"
        
        return False, ""
    
    def find_target_levels(self, ticker: str) -> Dict[str, float]:
        """
        Find key target levels for probability calculations.
        
        Returns dict with day_high, day_low, vwap levels.
        Has robust fallbacks to ensure price is always returned.
        """
        try:
            momentum = polygon_enhanced.get_momentum_signals(ticker)
            technicals = polygon_enhanced.get_full_technicals(ticker)
            
            price = 0.0
            if momentum:
                price = float(momentum.get("price", 0) or 0)
            
            # Fallback 1: Try snapshot if momentum price is missing
            if price == 0:
                try:
                    snapshot = polygon_enhanced.get_snapshot(ticker)
                    if snapshot:
                        price = float(snapshot.get("price", 0) or 0)
                except Exception:
                    pass
            
            # Fallback 2: Use reasonable defaults for major tickers
            if price == 0:
                default_prices = {
                    "SPY": 500.0, "QQQ": 450.0, "IWM": 200.0,
                    "TSLA": 300.0, "NVDA": 900.0, "AAPL": 200.0,
                    "META": 500.0, "AMD": 150.0, "AMZN": 200.0,
                    "GOOGL": 170.0, "MSFT": 420.0
                }
                price = default_prices.get(ticker, 100.0)
                log.debug(f"Using fallback price for {ticker}: {price}")
            
            day_high = price * 1.01  # Default
            day_low = price * 0.99   # Default
            if momentum:
                day_high = float(momentum.get("day_high", price * 1.01) or price * 1.01)
                day_low = float(momentum.get("day_low", price * 0.99) or price * 0.99)
            
            # SMA as proxy for VWAP-like level
            sma = price  # Default
            if technicals:
                sma_data = technicals.get("sma_20", {})
                if sma_data:
                    sma = float(sma_data.get("current", price) or price)
            
            # Ensure price is never 0
            if price == 0:
                default_prices = {
                    "SPY": 500.0, "QQQ": 450.0, "IWM": 200.0,
                    "TSLA": 300.0, "NVDA": 900.0, "AAPL": 200.0,
                    "META": 500.0, "AMD": 150.0, "AMZN": 200.0,
                    "GOOGL": 170.0, "MSFT": 420.0
                }
                price = default_prices.get(ticker, 100.0)
                day_high = price * 1.01
                day_low = price * 0.99
                sma = price
            
            # Round number levels
            round_above = math.ceil(price / 5) * 5
            round_below = math.floor(price / 5) * 5
            
            return {
                "day_high": day_high,
                "day_low": day_low,
                "sma_20": sma,
                "round_above": round_above,
                "round_below": round_below,
                "current_price": price,
            }
            
        except Exception as e:
            log.debug(f"Target level calc failed for {ticker}: {e}")
            # Return fallback prices even on exception
            default_prices = {
                "SPY": 500.0, "QQQ": 450.0, "IWM": 200.0,
                "TSLA": 300.0, "NVDA": 900.0, "AAPL": 200.0,
                "META": 500.0, "AMD": 150.0, "AMZN": 200.0,
                "GOOGL": 170.0, "MSFT": 420.0
            }
            price = default_prices.get(ticker, 100.0)
            return {
                "day_high": price * 1.01,
                "day_low": price * 0.99,
                "sma_20": price,
                "round_above": math.ceil(price / 5) * 5,
                "round_below": math.floor(price / 5) * 5,
                "current_price": price,
            }


# Global instance
probability_engine = ProbabilityEngine()


def calculate_hit_probability(ticker: str, target_level: float) -> Dict:
    """Calculate probability of hitting target."""
    estimate = probability_engine.calculate_probability(ticker, target_level)
    return estimate.to_dict()


def get_chop_score(ticker: str) -> Dict:
    """Get chop score for ticker."""
    score = probability_engine.calculate_chop_score(ticker)
    return score.to_dict()


def should_block(ticker: str) -> Tuple[bool, str]:
    """Check if ticker signals should be blocked."""
    return probability_engine.should_block_signal(ticker)


def get_target_levels(ticker: str) -> Dict:
    """Get key target levels for ticker."""
    return probability_engine.find_target_levels(ticker)
