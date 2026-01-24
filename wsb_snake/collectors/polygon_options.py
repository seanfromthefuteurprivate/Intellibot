"""
Polygon.io Options Chain Data Adapter

Fetches options chain snapshots for 0DTE analysis:
- ATM ± strikes
- IV, volume, OI
- Strike clustering
"""

import requests
from datetime import datetime, date
from typing import Dict, List, Any, Optional
from wsb_snake.config import POLYGON_API_KEY, POLYGON_BASE_URL
from wsb_snake.utils.logger import log


class PolygonOptionsAdapter:
    """Adapter for Polygon.io options data."""
    
    def __init__(self):
        self.api_key = POLYGON_API_KEY
        self.base_url = POLYGON_BASE_URL
        
    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make authenticated request to Polygon API."""
        if not self.api_key:
            log.error("POLYGON_API_KEY not set")
            return None
            
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            else:
                log.error(f"Polygon API error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            log.error(f"Polygon request failed: {e}")
            return None
    
    def get_options_chain(
        self,
        ticker: str,
        expiration_date: str = None,
        strike_price_gte: float = None,
        strike_price_lte: float = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get options chain snapshot for a ticker.
        
        Args:
            ticker: Underlying symbol (e.g., "SPY")
            expiration_date: YYYY-MM-DD format, defaults to today (0DTE)
            strike_price_gte: Minimum strike price
            strike_price_lte: Maximum strike price
            limit: Max contracts to return
            
        Returns:
            List of options contract data
        """
        if expiration_date is None:
            expiration_date = date.today().strftime("%Y-%m-%d")
        
        endpoint = f"/v3/snapshot/options/{ticker}"
        params = {
            "expiration_date": expiration_date,
            "limit": limit,
            "order": "asc",
            "sort": "strike_price"
        }
        
        if strike_price_gte:
            params["strike_price.gte"] = strike_price_gte
        if strike_price_lte:
            params["strike_price.lte"] = strike_price_lte
        
        data = self._request(endpoint, params)
        
        if data and "results" in data:
            log.info(f"Got {len(data['results'])} options for {ticker} exp {expiration_date}")
            return data["results"]
        
        return []
    
    def get_0dte_chain(self, ticker: str, spot_price: float, strike_range: int = 5) -> Dict[str, Any]:
        """
        Get 0DTE options chain centered around spot price.
        
        Args:
            ticker: Underlying symbol
            spot_price: Current price of underlying
            strike_range: Number of strikes above/below ATM
            
        Returns:
            Dict with calls, puts, and computed metrics
        """
        today = date.today().strftime("%Y-%m-%d")
        
        # Calculate strike bounds (assume $1 strikes for SPY, adjust for others)
        strike_step = 1 if spot_price < 200 else 5 if spot_price < 500 else 10
        strike_gte = spot_price - (strike_range * strike_step)
        strike_lte = spot_price + (strike_range * strike_step)
        
        contracts = self.get_options_chain(
            ticker=ticker,
            expiration_date=today,
            strike_price_gte=strike_gte,
            strike_price_lte=strike_lte,
            limit=100
        )
        
        # Separate calls and puts
        calls = []
        puts = []
        
        for contract in contracts:
            details = contract.get("details", {})
            greeks = contract.get("greeks", {})
            day = contract.get("day", {})
            
            parsed = {
                "symbol": details.get("ticker", ""),
                "strike": details.get("strike_price", 0),
                "expiration": details.get("expiration_date", ""),
                "type": details.get("contract_type", "").lower(),
                "iv": contract.get("implied_volatility", 0),
                "delta": greeks.get("delta", 0),
                "gamma": greeks.get("gamma", 0),
                "theta": greeks.get("theta", 0),
                "volume": day.get("volume", 0),
                "open_interest": contract.get("open_interest", 0),
                "last_price": day.get("close", 0),
                "bid": contract.get("last_quote", {}).get("bid", 0),
                "ask": contract.get("last_quote", {}).get("ask", 0),
            }
            
            if parsed["type"] == "call":
                calls.append(parsed)
            elif parsed["type"] == "put":
                puts.append(parsed)
        
        # Compute metrics
        total_call_volume = sum(c["volume"] for c in calls)
        total_put_volume = sum(p["volume"] for p in puts)
        total_call_oi = sum(c["open_interest"] for c in calls)
        total_put_oi = sum(p["open_interest"] for p in puts)
        
        # Find ATM strike
        atm_strike = min(
            [c["strike"] for c in calls] + [p["strike"] for p in puts],
            key=lambda s: abs(s - spot_price),
            default=spot_price
        )
        
        # Find top volume strikes
        all_contracts = calls + puts
        top_volume_strikes = sorted(all_contracts, key=lambda x: x["volume"], reverse=True)[:5]
        top_oi_strikes = sorted(all_contracts, key=lambda x: x["open_interest"], reverse=True)[:5]
        
        # Average IV
        all_ivs = [c["iv"] for c in all_contracts if c["iv"] > 0]
        avg_iv = sum(all_ivs) / len(all_ivs) if all_ivs else 0
        
        return {
            "ticker": ticker,
            "spot_price": spot_price,
            "expiration": today,
            "atm_strike": atm_strike,
            "calls": calls,
            "puts": puts,
            "metrics": {
                "total_call_volume": total_call_volume,
                "total_put_volume": total_put_volume,
                "call_put_volume_ratio": total_call_volume / max(total_put_volume, 1),
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
                "call_put_oi_ratio": total_call_oi / max(total_put_oi, 1),
                "avg_iv": avg_iv,
                "top_volume_strikes": [{"strike": c["strike"], "type": c["type"], "volume": c["volume"]} for c in top_volume_strikes],
                "top_oi_strikes": [{"strike": c["strike"], "type": c["type"], "oi": c["open_interest"]} for c in top_oi_strikes],
            }
        }
    
    def get_quote(self, ticker: str) -> Optional[Dict]:
        """Get current quote for underlying."""
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        data = self._request(endpoint)
        
        if data and "ticker" in data:
            ticker_data = data["ticker"]
            return {
                "symbol": ticker,
                "price": ticker_data.get("lastTrade", {}).get("p", 0),
                "bid": ticker_data.get("lastQuote", {}).get("p", 0),
                "ask": ticker_data.get("lastQuote", {}).get("P", 0),
                "volume": ticker_data.get("day", {}).get("v", 0),
                "vwap": ticker_data.get("day", {}).get("vw", 0),
                "change_pct": ticker_data.get("todaysChangePerc", 0) / 100,
            }
        return None


# Global instance
polygon_options = PolygonOptionsAdapter()
