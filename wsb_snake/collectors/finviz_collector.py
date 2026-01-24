"""
Finviz Collector - Unusual Volume Detection

Free tier: Web scraping (no API required)
Detects unusual volume spikes which often precede big moves.
"""

import requests
import re
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)


class FinvizCollector:
    """
    Scrapes Finviz for unusual volume and key metrics.
    Free - no API key required.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.base_url = "https://finviz.com"
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 120
        self.last_call = 0
        self.min_interval = 2.0
        
        logger.info("Finviz collector initialized")
    
    def _rate_limit(self):
        """Be respectful with scraping"""
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
    
    def get_stock_data(self, ticker: str) -> Dict:
        """
        Get stock overview data from Finviz.
        Extracts: price, volume, relative volume, volatility, RSI, etc.
        """
        cache_key = f"stock_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            
            url = f"{self.base_url}/quote.ashx?t={ticker.upper()}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                logger.warning(f"Finviz returned {response.status_code} for {ticker}")
                return {"error": f"HTTP {response.status_code}"}
            
            html = response.text
            
            data = self._parse_stock_page(html, ticker)
            
            self._set_cache(cache_key, data)
            return data
            
        except Exception as e:
            logger.warning(f"Finviz error for {ticker}: {e}")
            return {"error": str(e)}
    
    def _parse_stock_page(self, html: str, ticker: str) -> Dict:
        """Parse key metrics from Finviz stock page."""
        data = {
            "ticker": ticker,
            "source": "finviz",
            "timestamp": datetime.now().isoformat()
        }
        
        patterns = {
            "price": r'<td[^>]*class="snapshot-td2"[^>]*><b>([\d.]+)</b>',
            "change": r'<td[^>]*class="snapshot-td2"[^>]*><b[^>]*style="color:#[^"]*">([-\d.]+%)</b>',
            "volume": r'Volume</td><td[^>]*>([\d,]+)</td>',
            "avg_volume": r'Avg Volume</td><td[^>]*>([\d.]+[MK]?)</td>',
            "rel_volume": r'Rel Volume</td><td[^>]*>([\d.]+)</td>',
            "volatility": r'Volatility</td><td[^>]*>([\d.]+%)\s*([\d.]+%)?</td>',
            "rsi": r'RSI \(14\)</td><td[^>]*>([\d.]+)</td>',
            "short_float": r'Short Float</td><td[^>]*>([\d.]+%)</td>',
            "target_price": r'Target Price</td><td[^>]*>([\d.]+)</td>',
            "perf_week": r'Perf Week</td><td[^>]*>([-\d.]+%)</td>',
            "perf_month": r'Perf Month</td><td[^>]*>([-\d.]+%)</td>',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                value = match.group(1)
                if key == "volatility" and match.lastindex >= 2 and match.group(2):
                    data["volatility_week"] = value
                    data["volatility_month"] = match.group(2)
                else:
                    data[key] = self._parse_value(value)
        
        return data
    
    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        if not value:
            return None
        
        value = value.strip()
        
        if value.endswith("%"):
            try:
                return float(value.rstrip("%"))
            except:
                return value
        
        if value.endswith("M"):
            try:
                return float(value.rstrip("M")) * 1_000_000
            except:
                return value
        
        if value.endswith("K"):
            try:
                return float(value.rstrip("K")) * 1_000
            except:
                return value
        
        value_clean = value.replace(",", "")
        try:
            if "." in value_clean:
                return float(value_clean)
            return int(value_clean)
        except:
            return value
    
    def get_unusual_volume(self, ticker: str) -> Dict:
        """
        Check if ticker has unusual volume (rel_volume > 1.5).
        High relative volume often precedes big moves.
        """
        data = self.get_stock_data(ticker)
        
        if data.get("error"):
            return {
                "ticker": ticker,
                "unusual": False,
                "rel_volume": 1.0,
                "signal": "unknown",
                "source": "finviz"
            }
        
        rel_volume = data.get("rel_volume", 1.0)
        if isinstance(rel_volume, str):
            try:
                rel_volume = float(rel_volume)
            except:
                rel_volume = 1.0
        
        if rel_volume >= 3.0:
            signal = "extreme_volume"
            unusual = True
        elif rel_volume >= 2.0:
            signal = "very_high_volume"
            unusual = True
        elif rel_volume >= 1.5:
            signal = "elevated_volume"
            unusual = True
        elif rel_volume >= 1.2:
            signal = "slightly_elevated"
            unusual = False
        else:
            signal = "normal"
            unusual = False
        
        return {
            "ticker": ticker,
            "unusual": unusual,
            "rel_volume": rel_volume,
            "signal": signal,
            "volume": data.get("volume"),
            "avg_volume": data.get("avg_volume"),
            "source": "finviz"
        }
    
    def get_technical_snapshot(self, ticker: str) -> Dict:
        """
        Get quick technical snapshot for signal validation.
        """
        data = self.get_stock_data(ticker)
        
        if data.get("error"):
            return {"ticker": ticker, "error": data.get("error")}
        
        rsi = data.get("rsi")
        if rsi:
            if rsi >= 70:
                rsi_signal = "overbought"
            elif rsi <= 30:
                rsi_signal = "oversold"
            else:
                rsi_signal = "neutral"
        else:
            rsi_signal = "unknown"
        
        return {
            "ticker": ticker,
            "price": data.get("price"),
            "change": data.get("change"),
            "rsi": rsi,
            "rsi_signal": rsi_signal,
            "volatility": data.get("volatility_week"),
            "short_float": data.get("short_float"),
            "perf_week": data.get("perf_week"),
            "source": "finviz"
        }
    
    def calculate_signal_boost(self, ticker: str) -> float:
        """
        Calculate boost based on unusual volume.
        High relative volume = momentum = boost.
        """
        try:
            volume_data = self.get_unusual_volume(ticker)
            
            if volume_data.get("signal") == "extreme_volume":
                return 0.20
            elif volume_data.get("signal") == "very_high_volume":
                return 0.15
            elif volume_data.get("signal") == "elevated_volume":
                return 0.10
            else:
                return 0
                
        except Exception as e:
            logger.warning(f"Finviz boost calculation error: {e}")
            return 0


finviz_collector = FinvizCollector()
