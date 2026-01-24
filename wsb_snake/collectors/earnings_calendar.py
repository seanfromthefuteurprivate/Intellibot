"""
Earnings Calendar Collector

Tracks upcoming earnings with estimates and historical surprises.
Uses Finnhub free tier for earnings data.

Key edge:
- Earnings dates = volatility catalysts
- Expected move vs implied move
- Surprise direction prediction
"""

import os
import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from wsb_snake.utils.logger import log


class EarningsCalendarCollector:
    """
    Collects earnings calendar and surprise data.
    
    Features:
    - Upcoming earnings dates
    - EPS estimates vs actuals
    - Revenue estimates vs actuals
    - Historical surprise patterns
    """
    
    FINNHUB_URL = "https://finnhub.io/api/v1"
    
    def __init__(self):
        self.api_key = os.environ.get("FINNHUB_API_KEY", "")
        self.session = requests.Session()
        
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 1800
        self.last_call = 0
        self.min_interval = 1.0
        
        if not self.api_key:
            log.warning("FINNHUB_API_KEY not set - earnings calendar limited")
        else:
            log.info("Earnings Calendar collector initialized")
    
    def _rate_limit(self):
        """Respect Finnhub rate limits (60 calls/min free tier)"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    def get_earnings_calendar(self, from_date: str = None, to_date: str = None) -> List[Dict]:
        """
        Get upcoming earnings calendar.
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of upcoming earnings
        """
        if not from_date:
            from_date = datetime.now().strftime("%Y-%m-%d")
        if not to_date:
            to_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")
        
        cache_key = f"calendar:{from_date}:{to_date}"
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["data"]
        
        if not self.api_key:
            return []
        
        self._rate_limit()
        
        try:
            url = f"{self.FINNHUB_URL}/calendar/earnings"
            params = {
                "from": from_date,
                "to": to_date,
                "token": self.api_key,
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                earnings = data.get("earningsCalendar", [])
                
                result = []
                for e in earnings:
                    result.append({
                        "symbol": e.get("symbol", ""),
                        "date": e.get("date", ""),
                        "hour": e.get("hour", ""),
                        "eps_estimate": e.get("epsEstimate"),
                        "eps_actual": e.get("epsActual"),
                        "revenue_estimate": e.get("revenueEstimate"),
                        "revenue_actual": e.get("revenueActual"),
                        "quarter": e.get("quarter"),
                        "year": e.get("year"),
                    })
                
                self.cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": result
                }
                
                return result
            else:
                log.debug(f"Finnhub earnings returned {response.status_code}")
                return []
                
        except Exception as e:
            log.debug(f"Earnings calendar error: {e}")
            return []
    
    def get_earnings_for_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get earnings info for a specific ticker.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dict with next earnings date and estimates
        """
        calendar = self.get_earnings_calendar()
        
        symbol_upper = symbol.upper()
        matches = [e for e in calendar if e.get("symbol", "").upper() == symbol_upper]
        
        if not matches:
            return {
                "symbol": symbol,
                "has_upcoming_earnings": False,
                "next_earnings_date": None,
                "days_until_earnings": None,
            }
        
        next_earnings = matches[0]
        
        try:
            earnings_date = datetime.strptime(next_earnings["date"], "%Y-%m-%d")
            days_until = (earnings_date - datetime.now()).days
        except:
            days_until = None
        
        return {
            "symbol": symbol,
            "has_upcoming_earnings": True,
            "next_earnings_date": next_earnings["date"],
            "earnings_hour": next_earnings.get("hour"),
            "days_until_earnings": days_until,
            "eps_estimate": next_earnings.get("eps_estimate"),
            "revenue_estimate": next_earnings.get("revenue_estimate"),
            "is_within_week": days_until is not None and days_until <= 7,
            "is_tomorrow": days_until == 1,
            "is_today": days_until == 0,
        }
    
    def get_earnings_surprises(self, symbol: str, limit: int = 4) -> List[Dict]:
        """
        Get historical earnings surprises for a ticker.
        
        Args:
            symbol: Stock ticker
            limit: Number of quarters to look back
            
        Returns:
            List of historical earnings with surprises
        """
        if not self.api_key:
            return []
        
        cache_key = f"surprises:{symbol}:{limit}"
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["data"]
        
        self._rate_limit()
        
        try:
            url = f"{self.FINNHUB_URL}/stock/earnings"
            params = {
                "symbol": symbol,
                "limit": limit,
                "token": self.api_key,
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                earnings = response.json()
                
                result = []
                for e in earnings:
                    actual = e.get("actual")
                    estimate = e.get("estimate")
                    
                    if actual is not None and estimate is not None and estimate != 0:
                        surprise_pct = ((actual - estimate) / abs(estimate)) * 100
                        beat = actual > estimate
                    else:
                        surprise_pct = None
                        beat = None
                    
                    result.append({
                        "period": e.get("period"),
                        "actual": actual,
                        "estimate": estimate,
                        "surprise_pct": round(surprise_pct, 2) if surprise_pct else None,
                        "beat": beat,
                        "quarter": e.get("quarter"),
                        "year": e.get("year"),
                    })
                
                self.cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": result
                }
                
                return result
            else:
                return []
                
        except Exception as e:
            log.debug(f"Earnings surprises error for {symbol}: {e}")
            return []
    
    def get_earnings_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Get actionable earnings signal for a ticker.
        
        Returns:
            Dict with earnings-based trading recommendation
        """
        upcoming = self.get_earnings_for_ticker(symbol)
        surprises = self.get_earnings_surprises(symbol)
        
        beat_count = sum(1 for s in surprises if s.get("beat") is True)
        miss_count = sum(1 for s in surprises if s.get("beat") is False)
        
        if len(surprises) >= 3:
            if beat_count >= 3:
                historical_pattern = "consistent_beater"
                pattern_bias = "bullish"
            elif miss_count >= 3:
                historical_pattern = "consistent_misser"
                pattern_bias = "bearish"
            elif beat_count > miss_count:
                historical_pattern = "tends_to_beat"
                pattern_bias = "slight_bullish"
            else:
                historical_pattern = "mixed"
                pattern_bias = "neutral"
        else:
            historical_pattern = "insufficient_data"
            pattern_bias = "neutral"
        
        days_until = upcoming.get("days_until_earnings")
        
        if days_until is not None:
            if days_until == 0:
                strategy = "0DTE_VOLATILITY_EXPANSION"
                urgency = "critical"
            elif days_until <= 2:
                strategy = "EARNINGS_STRADDLE"
                urgency = "high"
            elif days_until <= 7:
                strategy = "EARNINGS_OTM_DIRECTIONAL"
                urgency = "medium"
            elif days_until <= 14:
                strategy = "EARNINGS_POSITION_BUILDING"
                urgency = "low"
            else:
                strategy = "MONITOR"
                urgency = "none"
        else:
            strategy = "NO_EARNINGS_CATALYST"
            urgency = "none"
        
        return {
            "symbol": symbol,
            "has_earnings": upcoming.get("has_upcoming_earnings", False),
            "earnings_date": upcoming.get("next_earnings_date"),
            "days_until": days_until,
            "earnings_hour": upcoming.get("earnings_hour"),
            "eps_estimate": upcoming.get("eps_estimate"),
            "historical_pattern": historical_pattern,
            "pattern_bias": pattern_bias,
            "beat_rate": f"{beat_count}/{len(surprises)}" if surprises else "N/A",
            "recommended_strategy": strategy,
            "urgency": urgency,
        }
    
    def get_this_week_earnings(self, symbols: List[str] = None) -> List[Dict]:
        """
        Get earnings happening this week for universe.
        
        Args:
            symbols: List of symbols to check (defaults to 0DTE universe)
            
        Returns:
            List of tickers with earnings this week
        """
        if symbols is None:
            from wsb_snake.config import ZERO_DTE_UNIVERSE
            symbols = ZERO_DTE_UNIVERSE
        
        this_week = []
        
        for symbol in symbols:
            signal = self.get_earnings_signal(symbol)
            days = signal.get("days_until")
            
            if days is not None and 0 <= days <= 7:
                this_week.append(signal)
        
        this_week.sort(key=lambda x: x.get("days_until", 999))
        
        return this_week


earnings_calendar = EarningsCalendarCollector()
