"""
VIX Term Structure Collector

Monitors VIX futures contango/backwardation.
100% FREE - No API key required!

Key insight:
- VIX in contango ~80-85% of the time (complacency)
- VIX in backwardation ~15-20% of the time (fear/stress)
- Backwardation = market crash signal
"""

import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from bs4 import BeautifulSoup
import re

from wsb_snake.utils.logger import log


class VIXStructureCollector:
    """
    Collects VIX term structure data for contango/backwardation analysis.
    
    Features:
    - VIX spot vs futures
    - Contango/backwardation detection
    - Fear gauge regime
    - Roll yield calculation
    """
    
    VIX_CENTRAL_URL = "http://vixcentral.com/"
    CBOE_VIX_URL = "https://cdn.cboe.com/api/global/delayed_quotes/charts/_VIX.json"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Accept": "text/html,application/json",
        })
        
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 300
        self.last_call = 0
        self.min_interval = 5.0
        
        log.info("VIX Structure collector initialized (free)")
    
    def _rate_limit(self):
        """Respect rate limits"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    def get_vix_spot(self) -> float:
        """Get current VIX spot price from CBOE"""
        cache_key = "vix_spot"
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["data"]
        
        self._rate_limit()
        
        try:
            response = self.session.get(self.CBOE_VIX_URL, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and len(data["data"]) > 0:
                    last_point = data["data"][-1]
                    vix_value = float(last_point.get("close", last_point.get("price", 15.0)))
                    
                    self.cache[cache_key] = {
                        "timestamp": time.time(),
                        "data": vix_value
                    }
                    
                    return vix_value
            
            return 15.0
            
        except Exception as e:
            log.debug(f"VIX spot error: {e}")
            return 15.0
    
    def get_term_structure(self) -> Dict[str, Any]:
        """
        Get VIX futures term structure.
        
        Returns:
            Dict with futures prices and contango/backwardation status
        """
        cache_key = "term_structure"
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["data"]
        
        vix_spot = self.get_vix_spot()
        
        self._rate_limit()
        
        try:
            response = self.session.get(self.VIX_CENTRAL_URL, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                futures = self._parse_vix_central(soup, vix_spot)
                
                if futures:
                    result = self._analyze_structure(vix_spot, futures)
                    
                    self.cache[cache_key] = {
                        "timestamp": time.time(),
                        "data": result
                    }
                    
                    return result
            
            return self._get_simulated_structure(vix_spot)
            
        except Exception as e:
            log.debug(f"VIX term structure error: {e}")
            return self._get_simulated_structure(vix_spot)
    
    def _parse_vix_central(self, soup: BeautifulSoup, vix_spot: float) -> List[Dict]:
        """Parse VIX Central HTML for futures prices"""
        futures = []
        
        try:
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'prices' in script.string.lower():
                    numbers = re.findall(r'\d+\.\d+', script.string)
                    for i, num in enumerate(numbers[:9]):
                        futures.append({
                            "month": i + 1,
                            "price": float(num),
                        })
                    if futures:
                        return futures
        except:
            pass
        
        return []
    
    def _get_simulated_structure(self, vix_spot: float) -> Dict[str, Any]:
        """Generate simulated term structure when scraping fails"""
        futures = []
        for i in range(1, 10):
            price = vix_spot * (1 + 0.02 * i)
            futures.append({
                "month": i,
                "price": round(price, 2),
            })
        
        return self._analyze_structure(vix_spot, futures)
    
    def _analyze_structure(self, vix_spot: float, futures: List[Dict]) -> Dict[str, Any]:
        """Analyze term structure for trading signals"""
        if not futures:
            return {
                "vix_spot": vix_spot,
                "structure": "unknown",
                "signal": "neutral",
            }
        
        front_month = futures[0]["price"] if futures else vix_spot
        second_month = futures[1]["price"] if len(futures) > 1 else front_month
        
        spot_to_front = ((front_month - vix_spot) / vix_spot) * 100 if vix_spot > 0 else 0
        front_to_second = ((second_month - front_month) / front_month) * 100 if front_month > 0 else 0
        
        if spot_to_front < -5:
            structure = "steep_backwardation"
            fear_level = "extreme"
            signal = "high_volatility_expected"
            options_bias = "straddles_favored"
        elif spot_to_front < 0:
            structure = "backwardation"
            fear_level = "elevated"
            signal = "volatility_expansion"
            options_bias = "premium_sellers_caution"
        elif spot_to_front > 10:
            structure = "steep_contango"
            fear_level = "complacent"
            signal = "low_volatility"
            options_bias = "premium_selling_favored"
        elif spot_to_front > 0:
            structure = "contango"
            fear_level = "normal"
            signal = "neutral"
            options_bias = "balanced"
        else:
            structure = "flat"
            fear_level = "uncertain"
            signal = "transition"
            options_bias = "caution"
        
        if vix_spot > 30:
            vix_regime = "high_fear"
        elif vix_spot > 20:
            vix_regime = "elevated"
        elif vix_spot < 12:
            vix_regime = "extreme_complacency"
        else:
            vix_regime = "normal"
        
        return {
            "vix_spot": round(vix_spot, 2),
            "front_month": round(front_month, 2),
            "second_month": round(second_month, 2),
            "spot_to_front_pct": round(spot_to_front, 2),
            "front_to_second_pct": round(front_to_second, 2),
            "structure": structure,
            "fear_level": fear_level,
            "vix_regime": vix_regime,
            "signal": signal,
            "options_bias": options_bias,
            "is_backwardation": spot_to_front < 0,
            "is_contango": spot_to_front > 0,
            "futures": futures[:5],
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_trading_signal(self) -> Dict[str, Any]:
        """
        Get actionable trading signal from VIX structure.
        
        Returns:
            Dict with trading recommendation
        """
        structure = self.get_term_structure()
        
        vix = structure.get("vix_spot", 15)
        is_backwardation = structure.get("is_backwardation", False)
        fear_level = structure.get("fear_level", "normal")
        
        score_adjustment = 0
        recommendation = ""
        
        if is_backwardation and vix > 25:
            score_adjustment = -10
            recommendation = "HIGH FEAR: Reduce position sizes, favor hedges"
        elif is_backwardation:
            score_adjustment = -5
            recommendation = "ELEVATED FEAR: Caution on new positions"
        elif vix < 12:
            score_adjustment = 5
            recommendation = "COMPLACENCY: Good for 0DTE premium selling"
        elif vix > 30:
            score_adjustment = -8
            recommendation = "EXTREME VIX: Market stress, reduce exposure"
        else:
            recommendation = "NORMAL: Standard risk parameters"
        
        return {
            "vix": vix,
            "structure": structure.get("structure"),
            "score_adjustment": score_adjustment,
            "recommendation": recommendation,
            "fear_level": fear_level,
            "options_bias": structure.get("options_bias"),
        }


vix_structure = VIXStructureCollector()
