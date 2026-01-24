"""
FRED Economic Data Collector

Federal Reserve Economic Data - 840,000+ economic time series.
100% FREE with API key registration.

Key indicators for trading:
- GDP growth
- Unemployment rate
- CPI/Inflation
- Fed Funds Rate
- Treasury yields
- Consumer sentiment
"""

import os
import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from wsb_snake.utils.logger import log


class FREDCollector:
    """
    Collects macroeconomic data from FRED (Federal Reserve Economic Data).
    
    Features:
    - Real-time economic indicators
    - Regime detection (expansion/contraction)
    - Rate environment analysis
    - Inflation monitoring
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    KEY_SERIES = {
        "GDP": "Real Gross Domestic Product",
        "UNRATE": "Unemployment Rate",
        "CPIAUCSL": "Consumer Price Index (All Urban)",
        "FEDFUNDS": "Federal Funds Effective Rate",
        "DGS10": "10-Year Treasury Yield",
        "DGS2": "2-Year Treasury Yield",
        "UMCSENT": "Consumer Sentiment",
        "VIXCLS": "VIX Close",
        "T10Y2Y": "10Y-2Y Treasury Spread",
        "BAMLH0A0HYM2": "High Yield Spread",
    }
    
    def __init__(self):
        self.api_key = os.environ.get("FRED_API_KEY", "")
        self.session = requests.Session()
        
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 3600
        self.last_call = 0
        self.min_interval = 0.5
        
        if not self.api_key:
            log.warning("FRED_API_KEY not set - using limited access")
        else:
            log.info("FRED collector initialized")
    
    def _rate_limit(self):
        """Respect rate limits"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    def get_series(self, series_id: str, limit: int = 10) -> List[Dict]:
        """
        Get observations for a FRED series.
        
        Args:
            series_id: FRED series ID (e.g., 'GDP', 'UNRATE')
            limit: Number of observations
            
        Returns:
            List of observations with date and value
        """
        cache_key = f"series:{series_id}:{limit}"
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["data"]
        
        if not self.api_key:
            return self._get_fallback_data(series_id)
        
        self._rate_limit()
        
        try:
            url = f"{self.BASE_URL}/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "limit": limit,
                "sort_order": "desc",
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                observations = data.get("observations", [])
                
                result = []
                for obs in observations:
                    try:
                        value = float(obs.get("value", 0))
                        result.append({
                            "date": obs.get("date"),
                            "value": value,
                        })
                    except:
                        continue
                
                self.cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": result
                }
                
                return result
            else:
                log.debug(f"FRED API returned {response.status_code}")
                return self._get_fallback_data(series_id)
                
        except Exception as e:
            log.debug(f"FRED error for {series_id}: {e}")
            return self._get_fallback_data(series_id)
    
    def _get_fallback_data(self, series_id: str) -> List[Dict]:
        """Return estimated fallback data when API unavailable"""
        fallbacks = {
            "FEDFUNDS": 5.25,
            "DGS10": 4.5,
            "DGS2": 4.8,
            "UNRATE": 4.0,
            "VIXCLS": 15.0,
            "T10Y2Y": -0.3,
        }
        
        return [{
            "date": datetime.now().strftime("%Y-%m-%d"),
            "value": fallbacks.get(series_id, 0),
            "is_fallback": True,
        }]
    
    def get_macro_regime(self) -> Dict[str, Any]:
        """
        Determine current macroeconomic regime.
        
        Returns:
            Dict with regime classification and key metrics
        """
        fed_funds = self.get_series("FEDFUNDS", 2)
        dgs10 = self.get_series("DGS10", 2)
        dgs2 = self.get_series("DGS2", 2)
        vix = self.get_series("VIXCLS", 5)
        spread = self.get_series("T10Y2Y", 2)
        
        current_rate = fed_funds[0]["value"] if fed_funds else 5.0
        current_10y = dgs10[0]["value"] if dgs10 else 4.5
        current_2y = dgs2[0]["value"] if dgs2 else 4.8
        current_vix = vix[0]["value"] if vix else 15.0
        current_spread = spread[0]["value"] if spread else -0.3
        
        if current_rate > 4.0:
            rate_regime = "restrictive"
        elif current_rate > 2.0:
            rate_regime = "neutral"
        else:
            rate_regime = "accommodative"
        
        if current_spread < -0.5:
            yield_curve = "inverted"
            recession_signal = True
        elif current_spread < 0:
            yield_curve = "flat"
            recession_signal = False
        else:
            yield_curve = "normal"
            recession_signal = False
        
        if current_vix > 30:
            volatility_regime = "high_fear"
        elif current_vix > 20:
            volatility_regime = "elevated"
        elif current_vix < 12:
            volatility_regime = "complacent"
        else:
            volatility_regime = "normal"
        
        if rate_regime == "restrictive" and volatility_regime in ["high_fear", "elevated"]:
            overall_regime = "risk_off"
            options_bias = "puts_favored"
        elif rate_regime == "accommodative" and volatility_regime in ["normal", "complacent"]:
            overall_regime = "risk_on"
            options_bias = "calls_favored"
        else:
            overall_regime = "mixed"
            options_bias = "neutral"
        
        return {
            "rate_regime": rate_regime,
            "yield_curve": yield_curve,
            "volatility_regime": volatility_regime,
            "overall_regime": overall_regime,
            "options_bias": options_bias,
            "recession_signal": recession_signal,
            "metrics": {
                "fed_funds": current_rate,
                "10y_yield": current_10y,
                "2y_yield": current_2y,
                "vix": current_vix,
                "yield_spread": current_spread,
            },
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_rate_environment(self) -> Dict[str, Any]:
        """
        Analyze current rate environment for options trading.
        
        Returns:
            Dict with rate analysis and trading implications
        """
        regime = self.get_macro_regime()
        
        fed_funds = regime["metrics"]["fed_funds"]
        spread = regime["metrics"]["yield_spread"]
        
        if fed_funds > 5.0 and spread < 0:
            environment = "late_cycle_stress"
            implication = "Favor defensive, expect volatility"
        elif fed_funds > 4.0:
            environment = "tight_conditions"
            implication = "Reduced risk appetite, quality focus"
        elif fed_funds < 1.0:
            environment = "easy_money"
            implication = "Risk-on, growth favored"
        else:
            environment = "normal"
            implication = "Standard risk assessment"
        
        return {
            "environment": environment,
            "implication": implication,
            "fed_funds_rate": fed_funds,
            "yield_curve_spread": spread,
            "is_inverted": spread < 0,
        }


    def get_cpi_release_dates(self, months: int = 3) -> List[str]:
        """
        Get recent CPI release dates.
        
        Args:
            months: Number of months to look back
            
        Returns:
            List of CPI release dates (YYYY-MM-DD)
        """
        cpi_dates = []
        
        try:
            cpi_data = self.get_series("CPIAUCSL", limit=months + 1)
            for obs in cpi_data:
                if obs.get("date"):
                    cpi_dates.append(obs["date"])
        except Exception as e:
            log.warning(f"Error fetching CPI dates: {e}")
        
        if not cpi_dates:
            current = datetime.now()
            for i in range(months):
                release_date = current - timedelta(days=30 * i)
                release_date = release_date.replace(day=13)
                cpi_dates.append(release_date.strftime("%Y-%m-%d"))
        
        return cpi_dates
    
    def get_economic_calendar(self, weeks: int = 4) -> List[Dict[str, Any]]:
        """
        Get upcoming economic events calendar.
        
        Args:
            weeks: Number of weeks to look ahead
            
        Returns:
            List of upcoming economic events
        """
        events = []
        current = datetime.now()
        end_date = current + timedelta(weeks=weeks)
        
        cpi_day = 13
        current_month = current.replace(day=1)
        for i in range(weeks // 4 + 2):
            month = current_month + timedelta(days=30 * i)
            cpi_date = month.replace(day=cpi_day)
            if current <= cpi_date <= end_date:
                events.append({
                    "date": cpi_date.strftime("%Y-%m-%d"),
                    "event_type": "cpi",
                    "name": "CPI Report",
                    "impact": "high",
                })
        
        fomc_dates = [
            "2026-01-29", "2026-03-19", "2026-05-07", "2026-06-18",
            "2026-07-29", "2026-09-17", "2026-11-05", "2026-12-16",
        ]
        for date_str in fomc_dates:
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d")
                if current <= event_date <= end_date:
                    events.append({
                        "date": date_str,
                        "event_type": "fomc",
                        "name": "FOMC Meeting",
                        "impact": "high",
                    })
            except ValueError:
                continue
        
        for week in range(weeks):
            friday = current + timedelta(days=(4 - current.weekday() + 7 * week) % 7 + 7 * (week > 0 or current.weekday() > 4))
            first_friday = friday.replace(day=1)
            first_friday = first_friday + timedelta(days=(4 - first_friday.weekday()) % 7)
            
            if friday.day <= 7 and current <= friday <= end_date:
                events.append({
                    "date": friday.strftime("%Y-%m-%d"),
                    "event_type": "jobs",
                    "name": "Jobs Report (NFP)",
                    "impact": "high",
                })
        
        events.sort(key=lambda x: x["date"])
        return events


fred_collector = FREDCollector()
