"""
SEC EDGAR Collector - Insider Trading Form 4 Filings

100% Free, no API key required.
Tracks insider buying/selling (Form 4) for smart money signals.
"""

import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from wsb_snake.utils.logger import log


CIK_MAPPING = {
    "SPY": "0001222333",
    "QQQ": "0001067839", 
    "IWM": "0001159154",
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    "TSLA": "0001318605",
    "NVDA": "0001045810",
    "AMD": "0000002488",
}


class SECEdgarCollector:
    """
    Collects insider trading data from SEC EDGAR.
    No API key required - just need User-Agent.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "WSBSnake Trading Bot (wsb_snake@replit.com)",
            "Accept": "application/json",
        })
        self.base_url = "https://data.sec.gov"
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 300
        self.last_call = 0
        self.min_interval = 0.15
        
        log.info("SEC EDGAR collector initialized")
    
    def _rate_limit(self):
        """SEC requires max 10 requests/second"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if still valid"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[key] = (data, time.time())
    
    def get_company_filings(self, ticker: str) -> Dict:
        """
        Get recent filings for a company.
        Returns Form 4 (insider trading) filings.
        """
        cik = CIK_MAPPING.get(ticker.upper())
        if not cik:
            return {"filings": [], "error": "CIK not found"}
        
        cache_key = f"filings_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            
            url = f"{self.base_url}/submissions/CIK{cik}.json"
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                log.warning(f"SEC EDGAR returned {response.status_code} for {ticker}")
                return {"filings": [], "error": f"HTTP {response.status_code}"}
            
            data = response.json()
            
            recent_filings = data.get("filings", {}).get("recent", {})
            
            forms = recent_filings.get("form", [])
            dates = recent_filings.get("filingDate", [])
            accessions = recent_filings.get("accessionNumber", [])
            
            form4_filings = []
            for i, form in enumerate(forms[:50]):
                if form in ["4", "4/A"]:
                    form4_filings.append({
                        "form": form,
                        "date": dates[i] if i < len(dates) else None,
                        "accession": accessions[i] if i < len(accessions) else None,
                    })
            
            result = {
                "ticker": ticker,
                "company": data.get("name", ticker),
                "filings": form4_filings[:10],
                "total_form4": len(form4_filings),
            }
            
            self._set_cache(cache_key, result)
            log.debug(f"SEC EDGAR found {len(form4_filings)} Form 4 filings for {ticker}")
            return result
            
        except Exception as e:
            log.warning(f"SEC EDGAR error for {ticker}: {e}")
            return {"filings": [], "error": str(e)}
    
    def get_insider_activity(self, ticker: str) -> Dict:
        """
        Analyze recent insider trading activity.
        Returns buying/selling pressure signals.
        """
        filings = self.get_company_filings(ticker)
        
        if filings.get("error") or not filings.get("filings"):
            return {
                "ticker": ticker,
                "recent_filings": 0,
                "signal": "neutral",
                "strength": 0,
                "source": "sec_edgar"
            }
        
        recent = filings["filings"]
        
        week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        filings_this_week = sum(1 for f in recent if f.get("date", "") >= week_ago)
        filings_this_month = sum(1 for f in recent if f.get("date", "") >= month_ago)
        
        if filings_this_week >= 3:
            signal = "high_activity"
            strength = min(filings_this_week / 5, 1.0)
        elif filings_this_month >= 5:
            signal = "moderate_activity"
            strength = min(filings_this_month / 10, 0.7)
        else:
            signal = "low_activity"
            strength = 0.2
        
        return {
            "ticker": ticker,
            "recent_filings": len(recent),
            "filings_this_week": filings_this_week,
            "filings_this_month": filings_this_month,
            "signal": signal,
            "strength": round(strength, 2),
            "latest_filing": recent[0] if recent else None,
            "source": "sec_edgar"
        }
    
    def calculate_signal_boost(self, ticker: str) -> float:
        """
        Calculate boost based on insider activity.
        High insider activity = potential catalyst = small boost.
        """
        try:
            activity = self.get_insider_activity(ticker)
            
            if activity.get("signal") == "high_activity":
                return 0.08
            elif activity.get("signal") == "moderate_activity":
                return 0.04
            else:
                return 0
                
        except Exception as e:
            log.warning(f"SEC EDGAR boost calculation error: {e}")
            return 0


sec_edgar_collector = SECEdgarCollector()
