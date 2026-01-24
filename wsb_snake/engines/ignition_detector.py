"""
Engine 1: Ignition Detector

Detects early momentum ignition signals - the first signs that a ticker
is "waking up" before a larger move.

Key signals:
- Volume spike vs average
- Price acceleration (rate of change increasing)
- Range expansion (breaking out of consolidation)
- News catalyst detection
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
    get_session_info, get_session_signal_multiplier, SessionType
)
from wsb_snake.collectors.alpaca_news import alpaca_news
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.collectors.reddit_collector import get_wsb_sentiment, get_wsb_trending


class IgnitionType(Enum):
    """Types of ignition patterns."""
    VOLUME_EXPLOSION = "volume_explosion"     # Volume > 3x average
    PRICE_BREAKOUT = "price_breakout"         # Breaking key level
    NEWS_CATALYST = "news_catalyst"           # News-driven move
    GAP_CONTINUATION = "gap_continuation"     # Gap and go pattern
    MOMENTUM_ACCELERATION = "momentum_accel"  # Rate of change increasing


@dataclass
class IgnitionSignal:
    """An ignition detection signal."""
    ticker: str
    ignition_type: IgnitionType
    score: float  # 0-100
    
    # Market data
    price: float
    change_pct: float
    volume: int
    volume_ratio: float  # vs average
    range_pct: float
    
    # Computed metrics
    velocity: float  # price change per minute
    acceleration: float  # change in velocity
    
    # Evidence
    evidence: List[str]
    news_headlines: List[str]
    
    # Timestamps
    detected_at: datetime
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "ignition_type": self.ignition_type.value,
            "score": self.score,
            "price": self.price,
            "change_pct": self.change_pct,
            "volume": self.volume,
            "volume_ratio": self.volume_ratio,
            "range_pct": self.range_pct,
            "velocity": self.velocity,
            "acceleration": self.acceleration,
            "evidence": self.evidence,
            "news_headlines": self.news_headlines,
            "detected_at": self.detected_at.isoformat(),
        }


class IgnitionDetector:
    """
    Engine 1: Detects early momentum ignition.
    
    Scans the 0DTE universe for tickers showing early signs of momentum,
    before the main move happens.
    """
    
    # Thresholds
    VOLUME_EXPLOSION_THRESHOLD = 2.5  # 2.5x normal volume
    RANGE_EXPANSION_THRESHOLD = 1.5   # 1.5x normal range
    VELOCITY_THRESHOLD = 0.5  # 0.5% per minute
    MIN_SCORE_TO_SIGNAL = 50  # Minimum score to emit signal
    
    def __init__(self):
        self.polygon_key = POLYGON_API_KEY
        self.alpaca_key = ALPACA_API_KEY
        self.alpaca_secret = ALPACA_SECRET_KEY
        
        # Cache for previous data points
        self._price_cache: Dict[str, Any] = {}  # Can hold List[Dict] or List[float]
        self._volume_cache: Dict[str, float] = {}  # Average volumes
        
    def scan_universe(self) -> List[IgnitionSignal]:
        """
        Scan the 0DTE universe for ignition signals.
        
        Returns:
            List of IgnitionSignal objects for tickers showing ignition
        """
        signals = []
        session_info = get_session_info()
        
        log.info(f"Ignition scan starting | Session: {session_info['session']}")
        
        # Skip scanning if session multiplier is 0 (market closed)
        if session_info["signal_quality_multiplier"] == 0:
            log.info("Ignition scan skipped - market closed")
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
                        log.info(f"Ignition detected: {ticker} | Score: {signal.score:.0f} | Type: {signal.ignition_type.value}")
            except Exception as e:
                log.warning(f"Failed to analyze {ticker}: {e}")
        
        # Sort by score
        signals.sort(key=lambda s: s.score, reverse=True)
        
        log.info(f"Ignition scan complete | Found {len(signals)} signals")
        return signals
    
    def _analyze_ticker(self, ticker: str) -> Optional[IgnitionSignal]:
        """
        Analyze a single ticker for ignition patterns.
        """
        # Get current data
        current = self._get_current_bar(ticker)
        if not current:
            return None
        
        # Get historical data for comparison
        prev_bars = self._get_historical_bars(ticker, lookback_days=5)
        
        # Calculate metrics
        avg_volume = self._calculate_avg_volume(prev_bars)
        volume_ratio = current["volume"] / max(avg_volume, 1)
        
        avg_range = self._calculate_avg_range(prev_bars)
        current_range = ((current["high"] - current["low"]) / current["close"]) * 100
        range_ratio = current_range / max(avg_range, 0.01)
        
        # Calculate velocity (price change per minute since open)
        minutes_since_open = max(1, self._minutes_since_market_open())
        velocity = current["change_pct"] / minutes_since_open
        
        # Calculate acceleration (compare to previous velocity)
        acceleration = self._calculate_acceleration(ticker, velocity)
        
        # Check for news catalyst
        news = self._check_news_catalyst(ticker)
        
        # Score the ignition
        score, ignition_type, evidence = self._score_ignition(
            ticker=ticker,
            volume_ratio=volume_ratio,
            range_ratio=range_ratio,
            velocity=velocity,
            acceleration=acceleration,
            change_pct=current["change_pct"],
            has_news=len(news) > 0,
        )
        
        if score > 0:
            return IgnitionSignal(
                ticker=ticker,
                ignition_type=ignition_type,
                score=score,
                price=current["close"],
                change_pct=current["change_pct"],
                volume=current["volume"],
                volume_ratio=volume_ratio,
                range_pct=current_range,
                velocity=velocity,
                acceleration=acceleration,
                evidence=evidence,
                news_headlines=news,
                detected_at=datetime.utcnow(),
            )
        
        return None
    
    def _score_ignition(
        self,
        ticker: str,
        volume_ratio: float,
        range_ratio: float,
        velocity: float,
        acceleration: float,
        change_pct: float,
        has_news: bool,
    ) -> tuple:
        """
        Score the ignition signal based on multiple factors.
        Enhanced with RSI and technical indicators from Polygon.
        
        Returns:
            (score, ignition_type, evidence list)
        """
        score = 0
        evidence = []
        ignition_type = IgnitionType.MOMENTUM_ACCELERATION
        
        # Volume explosion (0-30 points)
        if volume_ratio >= self.VOLUME_EXPLOSION_THRESHOLD:
            vol_score = min(30, (volume_ratio - 1) * 10)
            score += vol_score
            evidence.append(f"Volume {volume_ratio:.1f}x normal")
            ignition_type = IgnitionType.VOLUME_EXPLOSION
        
        # Range expansion (0-20 points)
        if range_ratio >= self.RANGE_EXPANSION_THRESHOLD:
            range_score = min(20, (range_ratio - 1) * 10)
            score += range_score
            evidence.append(f"Range {range_ratio:.1f}x normal")
            if ignition_type != IgnitionType.VOLUME_EXPLOSION:
                ignition_type = IgnitionType.PRICE_BREAKOUT
        
        # Velocity (0-25 points)
        if abs(velocity) >= self.VELOCITY_THRESHOLD:
            vel_score = min(25, abs(velocity) * 20)
            score += vel_score
            direction = "up" if velocity > 0 else "down"
            evidence.append(f"Velocity {velocity:.2f}%/min ({direction})")
        
        # Acceleration (0-15 points) - bonus for increasing speed
        if acceleration > 0:
            accel_score = min(15, acceleration * 30)
            score += accel_score
            evidence.append(f"Accelerating +{acceleration:.2f}")
        
        # Directional move (0-10 points)
        if abs(change_pct) >= 1.0:
            dir_score = min(10, abs(change_pct) * 2)
            score += dir_score
            evidence.append(f"Move {change_pct:+.2f}%")
        
        # News catalyst bonus (0-15 points)
        if has_news:
            score += 15
            evidence.append("News catalyst detected")
            ignition_type = IgnitionType.NEWS_CATALYST
        
        # Enhanced with RSI from Polygon (0-20 points)
        try:
            rsi = polygon_enhanced.get_rsi(ticker, window=14, timespan="minute")
            if rsi:
                rsi_val = rsi.get("current", 50)
                rsi_prev = rsi.get("previous", 50)
                
                # Oversold bounce or overbought extension
                if rsi_val < 30 and rsi_val > rsi_prev:
                    score += 15
                    evidence.append(f"RSI oversold bounce: {rsi_val:.0f}")
                elif rsi_val > 70 and rsi_val > rsi_prev:
                    score += 10
                    evidence.append(f"RSI momentum extension: {rsi_val:.0f}")
                elif 40 < rsi_val < 60 and abs(rsi_val - rsi_prev) > 5:
                    score += 5
                    evidence.append(f"RSI breakout from neutral: {rsi_val:.0f}")
        except Exception:
            pass  # Continue without RSI if unavailable
        
        # MACD crossover detection (0-15 points)
        try:
            macd = polygon_enhanced.get_macd(ticker, timespan="minute")
            if macd:
                histogram = macd.get("histogram", 0)
                if histogram > 0 and change_pct > 0:
                    score += 10
                    evidence.append("MACD bullish histogram")
                elif histogram < 0 and change_pct < 0:
                    score += 10
                    evidence.append("MACD bearish histogram")
        except Exception:
            pass
        
        # WSB Reddit Sentiment Boost (0-20 points)
        try:
            wsb_data = get_wsb_sentiment(ticker)
            if wsb_data and wsb_data.get("mentions", 0) > 0:
                mentions = wsb_data.get("mentions", 0)
                sentiment = wsb_data.get("overall_sentiment", "mixed")
                heat_score = wsb_data.get("heat_score", 0)
                bullish_ratio = wsb_data.get("bullish_ratio", 0.5)
                
                # Mentions boost (0-10 points)
                mention_boost = min(10, mentions * 2)
                score += mention_boost
                evidence.append(f"WSB mentions: {mentions}")
                
                # Sentiment alignment boost (0-10 points)
                if sentiment == "bullish" and change_pct > 0:
                    score += 10
                    evidence.append(f"WSB bullish sentiment ({bullish_ratio:.0%})")
                elif sentiment == "bearish" and change_pct < 0:
                    score += 10
                    evidence.append(f"WSB bearish sentiment ({1-bullish_ratio:.0%})")
                elif sentiment != "no_data":
                    score += 3
                    evidence.append(f"WSB active: {sentiment}")
                
                # DD post boost
                if wsb_data.get("dd_posts", 0) > 0:
                    score += 5
                    evidence.append(f"WSB DD posts: {wsb_data.get('dd_posts')}")
        except Exception as e:
            log.debug(f"WSB sentiment unavailable for {ticker}: {e}")
        
        return score, ignition_type, evidence
    
    def _get_current_bar(self, ticker: str) -> Optional[Dict]:
        """Get current bar data from Polygon."""
        if not self.polygon_key:
            return self._get_current_bar_alpaca(ticker)
        
        url = f"{POLYGON_BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        params = {"apiKey": self.polygon_key}
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                ticker_data = data.get("ticker", {})
                day = ticker_data.get("day", {})
                prev = ticker_data.get("prevDay", {})
                
                close = day.get("c") or day.get("vw") or prev.get("c", 0)
                prev_close = prev.get("c", close)
                change_pct = ((close - prev_close) / prev_close * 100) if prev_close else 0
                
                return {
                    "close": close,
                    "high": day.get("h", close),
                    "low": day.get("l", close),
                    "volume": day.get("v", 0),
                    "change_pct": change_pct,
                }
        except Exception as e:
            log.warning(f"Polygon snapshot failed for {ticker}: {e}")
        
        return self._get_current_bar_alpaca(ticker)
    
    def _get_current_bar_alpaca(self, ticker: str) -> Optional[Dict]:
        """Fallback to Alpaca for current data."""
        if not self.alpaca_key or not self.alpaca_secret:
            return None
        
        headers = {
            "APCA-API-KEY-ID": self.alpaca_key,
            "APCA-API-SECRET-KEY": self.alpaca_secret,
        }
        
        url = f"{ALPACA_DATA_URL}/v2/stocks/{ticker}/snapshot"
        
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                daily = data.get("dailyBar", {})
                prev = data.get("prevDailyBar", {})
                
                close = daily.get("c", 0)
                prev_close = prev.get("c", close)
                change_pct = ((close - prev_close) / prev_close * 100) if prev_close else 0
                
                return {
                    "close": close,
                    "high": daily.get("h", close),
                    "low": daily.get("l", close),
                    "volume": daily.get("v", 0),
                    "change_pct": change_pct,
                }
        except Exception as e:
            log.warning(f"Alpaca snapshot failed for {ticker}: {e}")
        
        return None
    
    def _get_historical_bars(self, ticker: str, lookback_days: int = 5) -> List[Dict]:
        """Get historical bars for baseline calculation."""
        if not self.polygon_key:
            return []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days + 3)  # Extra days for weekends
        
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {"apiKey": self.polygon_key, "limit": lookback_days}
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                return [
                    {
                        "close": r.get("c", 0),
                        "high": r.get("h", 0),
                        "low": r.get("l", 0),
                        "volume": r.get("v", 0),
                    }
                    for r in results
                ]
        except Exception as e:
            log.warning(f"Failed to get historical bars for {ticker}: {e}")
        
        return []
    
    def _calculate_avg_volume(self, bars: List[Dict]) -> float:
        """Calculate average volume from historical bars."""
        if not bars:
            return 1_000_000  # Default fallback
        volumes = [b["volume"] for b in bars if b.get("volume")]
        return sum(volumes) / len(volumes) if volumes else 1_000_000
    
    def _calculate_avg_range(self, bars: List[Dict]) -> float:
        """Calculate average daily range % from historical bars."""
        if not bars:
            return 1.0  # Default 1% range
        ranges = []
        for b in bars:
            if b.get("close") and b.get("high") and b.get("low"):
                daily_range = ((b["high"] - b["low"]) / b["close"]) * 100
                ranges.append(daily_range)
        return sum(ranges) / len(ranges) if ranges else 1.0
    
    def _calculate_acceleration(self, ticker: str, current_velocity: float) -> float:
        """Calculate velocity change (acceleration)."""
        # Store velocity for future comparison
        cache_key = f"velocity_{ticker}"
        prev_velocities = self._price_cache.get(cache_key, [])
        
        if prev_velocities:
            prev_velocity = prev_velocities[-1]
            acceleration = current_velocity - prev_velocity
        else:
            acceleration = 0.0
        
        # Update cache
        prev_velocities.append(current_velocity)
        if len(prev_velocities) > 5:
            prev_velocities.pop(0)
        self._price_cache[cache_key] = prev_velocities
        
        return acceleration
    
    def _check_news_catalyst(self, ticker: str) -> List[str]:
        """Check for recent news on a ticker."""
        try:
            news = alpaca_news.get_ticker_news(ticker, limit=5)
            # Filter to high importance news
            catalysts = [
                n["headline"] 
                for n in news 
                if n.get("importance") == "high"
            ]
            return catalysts[:3]  # Return top 3
        except Exception as e:
            log.debug(f"News check failed for {ticker}: {e}")
            return []
    
    def _minutes_since_market_open(self) -> int:
        """Calculate minutes since 9:30 AM ET."""
        from wsb_snake.utils.session_regime import get_eastern_time
        now = get_eastern_time()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        
        if now < market_open:
            return 1  # Pre-market
        
        delta = now - market_open
        return max(1, int(delta.total_seconds() / 60))


# Global instance
ignition_detector = IgnitionDetector()


def run_ignition_scan() -> List[Dict]:
    """
    Run an ignition scan and return signals as dicts.
    
    Convenience function for the main scheduler.
    """
    signals = ignition_detector.scan_universe()
    return [s.to_dict() for s in signals]
