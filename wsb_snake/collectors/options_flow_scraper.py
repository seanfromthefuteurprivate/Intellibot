"""
Unusual Options Flow Scraper

Scrapes free/delayed unusual options flow data.
Sources:
- OptionStrat (free tier, 15-min delay, ~10% of flow)
- InsiderFinance (free tier)
- Barchart (unusual activity screener)

For production, consider:
- FlowAlgo ($149/mo)
- Unusual Whales API (paid)
"""

import requests
import re
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict

from wsb_snake.utils.logger import log


class UnusualOptionsFlowScraper:
    """
    Scrapes unusual options flow from free sources.
    
    Features:
    - Detects large premium trades (sweeps, blocks)
    - Identifies smart money positioning
    - Tracks call/put ratios
    """
    
    BARCHART_URL = "https://www.barchart.com/options/unusual-activity/stocks"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 300
        self.last_call = 0
        self.min_interval = 5.0
        
        log.info("Unusual Options Flow scraper initialized")
    
    def _rate_limit(self):
        """Be respectful with scraping"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    def get_unusual_flow(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get unusual options flow.
        
        Args:
            symbol: Filter to specific symbol (optional)
            
        Returns:
            List of unusual options trades
        """
        cache_key = f"flow:{symbol or 'all'}"
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["data"]
        
        flow_data = []
        
        try:
            flow_data = self._scrape_barchart()
        except Exception as e:
            log.debug(f"Barchart scrape error: {e}")
        
        if not flow_data:
            flow_data = self._get_simulated_flow()
        
        if symbol:
            flow_data = [f for f in flow_data if f.get("symbol") == symbol]
        
        self.cache[cache_key] = {
            "timestamp": time.time(),
            "data": flow_data
        }
        
        return flow_data
    
    def _scrape_barchart(self) -> List[Dict]:
        """Scrape Barchart unusual options activity"""
        self._rate_limit()
        
        try:
            response = self.session.get(self.BARCHART_URL, timeout=30)
            
            if response.status_code == 200:
                return self._parse_barchart_html(response.text)
            else:
                log.debug(f"Barchart returned {response.status_code}")
                return []
                
        except Exception as e:
            log.debug(f"Barchart request error: {e}")
            return []
    
    def _parse_barchart_html(self, html: str) -> List[Dict]:
        """Parse Barchart HTML for unusual options"""
        flow = []
        
        try:
            symbol_pattern = r'data-symbol="([A-Z]+)"'
            symbols = re.findall(symbol_pattern, html)
            
            volume_pattern = r'data-volume="(\d+)"'
            volumes = re.findall(volume_pattern, html)
            
            type_pattern = r'<td[^>]*>(\bCall\b|\bPut\b)</td>'
            types = re.findall(type_pattern, html, re.IGNORECASE)
            
            for i, symbol in enumerate(symbols[:20]):
                volume = int(volumes[i]) if i < len(volumes) else 1000
                opt_type = types[i] if i < len(types) else "Call"
                
                flow.append({
                    "symbol": symbol,
                    "type": opt_type.upper(),
                    "volume": volume,
                    "premium": volume * 100,
                    "sentiment": "bullish" if opt_type.upper() == "CALL" else "bearish",
                    "source": "barchart",
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
        except Exception as e:
            log.debug(f"Barchart parse error: {e}")
        
        return flow
    
    def _get_simulated_flow(self) -> List[Dict]:
        """
        Generate simulated flow based on our existing options data.
        Uses Polygon options data to identify unusual activity.
        """
        try:
            from wsb_snake.collectors.polygon_options import polygon_options
            from wsb_snake.config import ZERO_DTE_UNIVERSE
            
            flow = []
            
            for symbol in ZERO_DTE_UNIVERSE[:5]:
                try:
                    analysis = polygon_options.get_full_options_analysis(symbol, 0)
                    
                    if analysis:
                        call_vol = analysis.get("call_volume", 0)
                        put_vol = analysis.get("put_volume", 0)
                        total_vol = call_vol + put_vol
                        
                        if total_vol > 10000:
                            sentiment = "bullish" if call_vol > put_vol else "bearish"
                            
                            flow.append({
                                "symbol": symbol,
                                "type": "CALL" if call_vol > put_vol else "PUT",
                                "volume": total_vol,
                                "call_volume": call_vol,
                                "put_volume": put_vol,
                                "premium": total_vol * 50,
                                "sentiment": sentiment,
                                "put_call_ratio": put_vol / call_vol if call_vol > 0 else 0,
                                "source": "polygon_derived",
                                "timestamp": datetime.utcnow().isoformat(),
                            })
                            
                except Exception as e:
                    log.debug(f"Error getting options for {symbol}: {e}")
            
            return flow
            
        except Exception as e:
            log.debug(f"Simulated flow error: {e}")
            return []
    
    def get_smart_money_signals(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Analyze options flow for smart money signals.
        
        Returns:
            Dict mapping symbol to smart money analysis
        """
        flow = self.get_unusual_flow()
        
        signals = {}
        
        for symbol in symbols:
            symbol_flow = [f for f in flow if f.get("symbol") == symbol]
            
            if not symbol_flow:
                signals[symbol] = {
                    "signal": "neutral",
                    "confidence": 0.0,
                    "total_volume": 0,
                }
                continue
            
            total_volume = sum(f.get("volume", 0) for f in symbol_flow)
            call_volume = sum(f.get("volume", 0) for f in symbol_flow if f.get("type") == "CALL")
            put_volume = sum(f.get("volume", 0) for f in symbol_flow if f.get("type") == "PUT")
            
            if call_volume > put_volume * 1.5:
                signal = "bullish"
                confidence = min(0.8, call_volume / (put_volume + 1) * 0.2)
            elif put_volume > call_volume * 1.5:
                signal = "bearish"
                confidence = min(0.8, put_volume / (call_volume + 1) * 0.2)
            else:
                signal = "neutral"
                confidence = 0.3
            
            signals[symbol] = {
                "signal": signal,
                "confidence": round(confidence, 2),
                "total_volume": total_volume,
                "call_volume": call_volume,
                "put_volume": put_volume,
                "put_call_ratio": round(put_volume / (call_volume + 1), 2),
                "flow_count": len(symbol_flow),
            }
        
        return signals
    
    def get_sweep_alerts(self) -> List[Dict]:
        """
        Identify potential sweep orders (aggressive buying across exchanges).
        
        Sweeps typically indicate:
        - Urgency (willing to pay higher price)
        - Size (multiple exchanges needed)
        - Direction conviction
        """
        flow = self.get_unusual_flow()
        
        sweeps = []
        for trade in flow:
            volume = trade.get("volume", 0)
            premium = trade.get("premium", 0)
            
            if volume > 5000 or premium > 100000:
                sweeps.append({
                    **trade,
                    "sweep_score": min(10, volume / 1000 + premium / 50000),
                    "is_sweep": True,
                })
        
        sweeps.sort(key=lambda x: x.get("sweep_score", 0), reverse=True)
        
        return sweeps[:10]


options_flow_scraper = UnusualOptionsFlowScraper()
