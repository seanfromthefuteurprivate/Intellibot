"""
FINRA Dark Pool Collector - OTC Transparency Data

100% FREE - No API key required!
Tracks dark pool / ATS volume for institutional flow analysis.

Data includes:
- Weekly ATS volume by security
- Market Participant Identifier (MPID) for each dark pool
- Share volume and trade counts
- Non-ATS OTC volume

Delay: 2 weeks for Tier 1 (S&P 500), 4 weeks for other securities
"""

import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict

from wsb_snake.utils.logger import log


class FINRADarkPoolCollector:
    """
    Collects dark pool (ATS) transparency data from FINRA.
    100% free, no API key required.
    
    Use cases:
    - Track institutional liquidity flows
    - Compare dark pool vs lit exchange volumes
    - Detect unusual dark pool activity
    """
    
    BASE_URL = "https://api.finra.org/data/group/otcMarket/name"
    OTC_PORTAL = "https://otctransparency.finra.org"
    
    ATS_NAMES = {
        "UBSS": "UBS ATS",
        "CODA": "Coda Markets",
        "DBAX": "Deutsche Bank ATS",
        "BAML": "BofA Securities",
        "JPMX": "JPMorgan ATS",
        "MSPL": "Morgan Stanley ATS",
        "SGMA": "Sigma X2 (Goldman)",
        "KCGM": "Virtu Americas",
        "IEXG": "IEX",
        "LTSE": "Long-Term Stock Exchange",
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "WSBSnake/1.0 (Financial Research)",
            "Accept": "application/json",
        })
        
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 3600
        self.last_call = 0
        self.min_interval = 1.0
        
        log.info("FINRA Dark Pool collector initialized (free)")
    
    def _rate_limit(self):
        """Respect rate limits"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """Generate cache key"""
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{endpoint}?{param_str}"
    
    def get_ats_volume(self, symbol: str) -> Dict[str, Any]:
        """
        Get dark pool (ATS) volume data for a symbol.
        
        Returns:
            Dict with dark pool volume breakdown by ATS
        """
        cache_key = f"ats_volume:{symbol}"
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["data"]
        
        self._rate_limit()
        
        try:
            url = f"{self.BASE_URL}/weeklySummary"
            params = {
                "symbol": symbol,
                "limit": 100,
                "offset": 0,
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                result = self._parse_ats_data(symbol, data)
                
                self.cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": result
                }
                
                return result
            else:
                log.debug(f"FINRA API returned {response.status_code} for {symbol}")
                return self._get_fallback_data(symbol)
                
        except Exception as e:
            log.debug(f"FINRA Dark Pool error for {symbol}: {e}")
            return self._get_fallback_data(symbol)
    
    def _parse_ats_data(self, symbol: str, raw_data: List[Dict]) -> Dict[str, Any]:
        """Parse raw FINRA data into useful format"""
        if not raw_data:
            return self._get_fallback_data(symbol)
        
        ats_volumes = defaultdict(int)
        total_ats_volume = 0
        total_trades = 0
        weeks_of_data = 0
        latest_week = None
        
        for record in raw_data:
            mpid = record.get("mpId", "")
            volume = int(record.get("totalWeeklyShareQuantity", 0))
            trades = int(record.get("totalWeeklyTradeCount", 0))
            week_start = record.get("weekStartDate", "")
            
            ats_volumes[mpid] += volume
            total_ats_volume += volume
            total_trades += trades
            weeks_of_data += 1
            
            if not latest_week or week_start > latest_week:
                latest_week = week_start
        
        top_ats = sorted(
            ats_volumes.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        top_ats_named = []
        for mpid, vol in top_ats:
            name = self.ATS_NAMES.get(mpid, mpid)
            pct = (vol / total_ats_volume * 100) if total_ats_volume > 0 else 0
            top_ats_named.append({
                "mpid": mpid,
                "name": name,
                "volume": vol,
                "percentage": round(pct, 1)
            })
        
        return {
            "symbol": symbol,
            "total_ats_volume": total_ats_volume,
            "total_trades": total_trades,
            "weeks_of_data": weeks_of_data,
            "latest_week": latest_week,
            "top_ats": top_ats_named,
            "ats_count": len(ats_volumes),
            "has_data": True,
            "data_source": "FINRA OTC Transparency",
        }
    
    def _get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Return fallback when API fails"""
        return {
            "symbol": symbol,
            "total_ats_volume": 0,
            "total_trades": 0,
            "weeks_of_data": 0,
            "latest_week": None,
            "top_ats": [],
            "ats_count": 0,
            "has_data": False,
            "data_source": "FINRA OTC Transparency",
            "note": "Data delayed 2-4 weeks per FINRA policy"
        }
    
    def get_dark_pool_ratio(self, symbol: str) -> Dict[str, Any]:
        """
        Estimate dark pool ratio (ATS volume vs total).
        
        Returns:
            Dict with dark pool metrics and signals
        """
        ats_data = self.get_ats_volume(symbol)
        
        if not ats_data.get("has_data"):
            return {
                "symbol": symbol,
                "dark_pool_ratio": None,
                "signal": "unknown",
                "ats_volume": 0,
            }
        
        ats_volume = ats_data.get("total_ats_volume", 0)
        
        signal = "normal"
        if ats_volume > 50_000_000:
            signal = "high_institutional"
        elif ats_volume > 20_000_000:
            signal = "moderate_institutional"
        elif ats_volume > 5_000_000:
            signal = "some_institutional"
        else:
            signal = "low_institutional"
        
        return {
            "symbol": symbol,
            "ats_volume": ats_volume,
            "ats_trades": ats_data.get("total_trades", 0),
            "top_ats": ats_data.get("top_ats", []),
            "signal": signal,
            "weeks_analyzed": ats_data.get("weeks_of_data", 0),
            "data_delay": "2-4 weeks",
        }
    
    def get_unusual_dark_pool_activity(self, symbols: List[str]) -> List[Dict]:
        """
        Scan multiple symbols for unusual dark pool activity.
        
        Returns:
            List of symbols with high dark pool signals
        """
        unusual = []
        
        for symbol in symbols:
            try:
                ratio = self.get_dark_pool_ratio(symbol)
                
                if ratio.get("signal") in ["high_institutional", "moderate_institutional"]:
                    unusual.append({
                        "symbol": symbol,
                        "ats_volume": ratio.get("ats_volume", 0),
                        "signal": ratio.get("signal"),
                        "top_ats": ratio.get("top_ats", [])[:3],
                    })
                    
            except Exception as e:
                log.debug(f"Error checking dark pool for {symbol}: {e}")
        
        unusual.sort(key=lambda x: x.get("ats_volume", 0), reverse=True)
        
        return unusual
    
    def get_ats_leaderboard(self) -> List[Dict]:
        """
        Get top ATSes by overall volume (market-wide).
        Useful for understanding which dark pools are most active.
        """
        return [
            {"mpid": mpid, "name": name}
            for mpid, name in self.ATS_NAMES.items()
        ]


finra_darkpool = FINRADarkPoolCollector()
