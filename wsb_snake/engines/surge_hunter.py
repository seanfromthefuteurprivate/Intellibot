"""
Engine 3: Late-Day Surge Hunter

Specialized for detecting momentum surges during power hour (3pm-4pm ET).
Optimized for 0DTE timing when theta decay accelerates.

Key patterns:
- VWAP reclaim/rejection
- Range breakout with volume
- News catalyst + momentum alignment
- Final hour gamma acceleration
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from wsb_snake.config import (
    POLYGON_API_KEY, POLYGON_BASE_URL,
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_DATA_URL,
    ZERO_DTE_UNIVERSE
)
from wsb_snake.utils.logger import log
from wsb_snake.utils.session_regime import (
    get_session_info, is_power_hour, is_final_hour, get_0dte_urgency
)
from wsb_snake.collectors.alpaca_news import alpaca_news


class SurgeType(Enum):
    """Types of surge patterns."""
    VWAP_RECLAIM = "vwap_reclaim"       # Price reclaiming VWAP with volume
    RANGE_BREAKOUT = "range_breakout"    # Breaking day's high/low
    MOMENTUM_BURST = "momentum_burst"    # Sudden velocity increase
    NEWS_SURGE = "news_surge"            # News-driven power hour move
    GAMMA_ACCELERATION = "gamma_accel"   # Dealers hedging accelerating move


@dataclass
class SurgeSignal:
    """A surge detection signal."""
    ticker: str
    surge_type: SurgeType
    score: float  # 0-100
    direction: str  # "long" or "short"
    
    # Price data
    current_price: float
    vwap: float
    day_high: float
    day_low: float
    
    # Momentum metrics
    velocity: float      # Change per minute
    volume_ratio: float  # vs average
    range_pct: float     # Day range as %
    
    # Timing
    minutes_to_close: float
    urgency: str
    
    # Evidence
    evidence: List[str]
    
    # Entry/Exit
    entry_zone: float
    stop_loss: float
    target_1: float
    target_2: float
    
    detected_at: datetime
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "surge_type": self.surge_type.value,
            "score": self.score,
            "direction": self.direction,
            "current_price": self.current_price,
            "vwap": self.vwap,
            "day_high": self.day_high,
            "day_low": self.day_low,
            "velocity": self.velocity,
            "volume_ratio": self.volume_ratio,
            "range_pct": self.range_pct,
            "minutes_to_close": self.minutes_to_close,
            "urgency": self.urgency,
            "evidence": self.evidence,
            "entry_zone": self.entry_zone,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "detected_at": self.detected_at.isoformat(),
        }


class SurgeHunter:
    """
    Engine 3: Late-Day Surge Hunter
    
    Specialized for power hour 0DTE opportunities.
    """
    
    # Thresholds
    VOLUME_SURGE_THRESHOLD = 2.0   # 2x volume required
    VELOCITY_THRESHOLD = 0.3       # 0.3% per minute
    BREAKOUT_THRESHOLD = 0.5       # 0.5% beyond range
    MIN_SCORE_TO_SIGNAL = 45
    
    def __init__(self):
        self.polygon_key = POLYGON_API_KEY
        self.alpaca_key = ALPACA_API_KEY
        self.alpaca_secret = ALPACA_SECRET_KEY
    
    def scan_universe(self) -> List[SurgeSignal]:
        """
        Scan for power hour surge setups.
        
        Returns:
            List of SurgeSignal objects
        """
        signals = []
        session_info = get_session_info()
        
        # Surge hunting is most effective during power hour
        is_ph = is_power_hour()
        urgency = get_0dte_urgency()
        
        log.info(f"Surge hunt starting | Power Hour: {is_ph} | Urgency: {urgency}")
        
        for ticker in ZERO_DTE_UNIVERSE:
            try:
                signal = self._hunt_surge(ticker, session_info)
                if signal and signal.score >= self.MIN_SCORE_TO_SIGNAL:
                    # Boost score during power hour
                    if is_ph:
                        signal.score *= 1.3
                    signals.append(signal)
                    log.info(f"Surge detected: {ticker} | Score: {signal.score:.0f} | Type: {signal.surge_type.value}")
            except Exception as e:
                log.warning(f"Surge hunt failed for {ticker}: {e}")
        
        signals.sort(key=lambda s: s.score, reverse=True)
        log.info(f"Surge hunt complete | Found {len(signals)} signals")
        
        return signals
    
    def _hunt_surge(self, ticker: str, session_info: Dict) -> Optional[SurgeSignal]:
        """Hunt for surge setup on a single ticker."""
        
        # Get intraday data
        bars = self._get_intraday_bars(ticker)
        if not bars or len(bars) < 10:
            return None
        
        # Calculate metrics
        current_bar = bars[-1]
        current_price = current_bar.get("c", 0)
        
        if current_price <= 0:
            return None
        
        # VWAP calculation
        vwap = self._calculate_vwap(bars)
        
        # Day high/low
        day_high = max(b.get("h", 0) for b in bars)
        day_low = min(b.get("l", float('inf')) for b in bars)
        
        # Recent velocity (last 5 bars)
        recent_bars = bars[-5:]
        price_change = (recent_bars[-1].get("c", 0) - recent_bars[0].get("c", 0))
        velocity = (price_change / recent_bars[0].get("c", 1)) * 100 / len(recent_bars)
        
        # Volume analysis
        avg_volume = sum(b.get("v", 0) for b in bars[:-5]) / max(len(bars) - 5, 1)
        recent_volume = sum(b.get("v", 0) for b in recent_bars) / len(recent_bars)
        volume_ratio = recent_volume / max(avg_volume, 1)
        
        # Range calculation
        range_pct = ((day_high - day_low) / current_price) * 100
        
        # Check for news
        news = self._check_recent_news(ticker)
        
        # Score the surge
        score, surge_type, evidence, direction = self._score_surge(
            ticker=ticker,
            current_price=current_price,
            vwap=vwap,
            day_high=day_high,
            day_low=day_low,
            velocity=velocity,
            volume_ratio=volume_ratio,
            range_pct=range_pct,
            has_news=len(news) > 0,
        )
        
        if score < self.MIN_SCORE_TO_SIGNAL:
            return None
        
        # Calculate trade levels
        entry_zone, stop_loss, target_1, target_2 = self._calculate_levels(
            direction=direction,
            current_price=current_price,
            vwap=vwap,
            day_high=day_high,
            day_low=day_low,
        )
        
        return SurgeSignal(
            ticker=ticker,
            surge_type=surge_type,
            score=score,
            direction=direction,
            current_price=current_price,
            vwap=vwap,
            day_high=day_high,
            day_low=day_low,
            velocity=velocity,
            volume_ratio=volume_ratio,
            range_pct=range_pct,
            minutes_to_close=session_info.get("minutes_to_close", 0),
            urgency=get_0dte_urgency(),
            evidence=evidence + news[:2],
            entry_zone=entry_zone,
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            detected_at=datetime.utcnow(),
        )
    
    def _score_surge(
        self,
        ticker: str,
        current_price: float,
        vwap: float,
        day_high: float,
        day_low: float,
        velocity: float,
        volume_ratio: float,
        range_pct: float,
        has_news: bool,
    ) -> tuple:
        """Score surge signal."""
        score = 0
        evidence = []
        surge_type = SurgeType.MOMENTUM_BURST
        direction = "long" if velocity > 0 else "short"
        
        # VWAP position (0-25 points)
        vwap_distance = (current_price - vwap) / vwap * 100
        if abs(vwap_distance) < 0.2:
            # Near VWAP - potential reclaim
            score += 15
            evidence.append(f"At VWAP (±{abs(vwap_distance):.2f}%)")
        elif vwap_distance > 0.5 and velocity > 0:
            # Above VWAP with momentum
            score += 25
            evidence.append(f"Above VWAP +{vwap_distance:.2f}%")
            surge_type = SurgeType.VWAP_RECLAIM
        elif vwap_distance < -0.5 and velocity < 0:
            # Below VWAP with momentum
            score += 25
            evidence.append(f"Below VWAP {vwap_distance:.2f}%")
            surge_type = SurgeType.VWAP_RECLAIM
        
        # Range breakout (0-30 points)
        high_distance = (current_price - day_high) / day_high * 100
        low_distance = (day_low - current_price) / current_price * 100
        
        if high_distance > 0:
            # Breaking day high
            score += min(30, high_distance * 30)
            evidence.append(f"Breaking day high +{high_distance:.2f}%")
            surge_type = SurgeType.RANGE_BREAKOUT
            direction = "long"
        elif low_distance > 0:
            # Breaking day low
            score += min(30, low_distance * 30)
            evidence.append(f"Breaking day low -{low_distance:.2f}%")
            surge_type = SurgeType.RANGE_BREAKOUT
            direction = "short"
        
        # Velocity (0-25 points)
        if abs(velocity) >= self.VELOCITY_THRESHOLD:
            vel_score = min(25, abs(velocity) * 40)
            score += vel_score
            evidence.append(f"Velocity {velocity:.2f}%/bar")
            if vel_score >= 20:
                surge_type = SurgeType.MOMENTUM_BURST
        
        # Volume surge (0-15 points)
        if volume_ratio >= self.VOLUME_SURGE_THRESHOLD:
            vol_score = min(15, (volume_ratio - 1) * 10)
            score += vol_score
            evidence.append(f"Volume {volume_ratio:.1f}x avg")
        
        # News catalyst (0-20 points)
        if has_news:
            score += 20
            evidence.append("News catalyst")
            surge_type = SurgeType.NEWS_SURGE
        
        # Range expansion bonus (0-10 points)
        if range_pct >= 2.0:
            range_score = min(10, range_pct * 2)
            score += range_score
            evidence.append(f"Range {range_pct:.1f}%")
        
        return score, surge_type, evidence, direction
    
    def _calculate_levels(
        self,
        direction: str,
        current_price: float,
        vwap: float,
        day_high: float,
        day_low: float,
    ) -> tuple:
        """Calculate entry, stop, and target levels."""
        
        if direction == "long":
            # Entry at current, stop below VWAP or day low
            entry_zone = current_price
            stop_loss = min(vwap * 0.995, day_low * 0.998)
            
            # Targets: 1R and 2R from risk
            risk = current_price - stop_loss
            target_1 = current_price + risk
            target_2 = current_price + (risk * 2)
        else:
            # Short: entry at current, stop above VWAP or day high
            entry_zone = current_price
            stop_loss = max(vwap * 1.005, day_high * 1.002)
            
            risk = stop_loss - current_price
            target_1 = current_price - risk
            target_2 = current_price - (risk * 2)
        
        return entry_zone, stop_loss, target_1, target_2
    
    def _get_intraday_bars(self, ticker: str) -> List[Dict]:
        """Get 5-minute intraday bars."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Try Polygon first
        if self.polygon_key:
            url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/5/minute/{today}/{today}"
            params = {"apiKey": self.polygon_key, "limit": 100}
            
            try:
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("results", [])
            except:
                pass
        
        # Fallback to Alpaca
        if self.alpaca_key and self.alpaca_secret:
            headers = {
                "APCA-API-KEY-ID": self.alpaca_key,
                "APCA-API-SECRET-KEY": self.alpaca_secret,
            }
            
            url = f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/bars"
            params = {
                "timeframe": "5Min",
                "start": f"{today}T09:30:00Z",
                "limit": 100,
            }
            
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    bars = data.get("bars", [])
                    # Convert Alpaca format to Polygon format
                    return [
                        {
                            "o": b.get("o"),
                            "h": b.get("h"),
                            "l": b.get("l"),
                            "c": b.get("c"),
                            "v": b.get("v"),
                            "t": b.get("t"),
                        }
                        for b in bars
                    ]
            except:
                pass
        
        return []
    
    def _calculate_vwap(self, bars: List[Dict]) -> float:
        """Calculate VWAP from intraday bars."""
        if not bars:
            return 0
        
        cumulative_tpv = 0  # Typical Price * Volume
        cumulative_volume = 0
        
        for bar in bars:
            typical_price = (bar.get("h", 0) + bar.get("l", 0) + bar.get("c", 0)) / 3
            volume = bar.get("v", 0)
            
            cumulative_tpv += typical_price * volume
            cumulative_volume += volume
        
        return cumulative_tpv / cumulative_volume if cumulative_volume > 0 else 0
    
    def _check_recent_news(self, ticker: str) -> List[str]:
        """Check for recent news on ticker."""
        try:
            news = alpaca_news.get_ticker_news(ticker, limit=3)
            return [
                f"📰 {n['headline'][:50]}..."
                for n in news
                if n.get("importance") in ["high", "medium"]
            ][:2]
        except:
            return []


# Global instance
surge_hunter = SurgeHunter()


def run_surge_hunt() -> List[Dict]:
    """Run surge hunt and return signals as dicts."""
    signals = surge_hunter.scan_universe()
    return [s.to_dict() for s in signals]
