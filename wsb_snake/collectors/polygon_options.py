"""
Polygon.io Options Chain Data Adapter - Enhanced with Options Starter Plan

Now includes:
- Full options chain snapshots with real-time greeks
- Gamma exposure (GEX) calculation for gamma-magnet detection
- Max pain calculation for strike pinning analysis
- Call/put volume walls detection
- IV surface analysis
"""

import requests
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
from wsb_snake.config import POLYGON_API_KEY, POLYGON_BASE_URL
from wsb_snake.utils.logger import log


class PolygonOptionsAdapter:
    """Enhanced adapter for Polygon.io options data with Options Starter plan."""
    
    def __init__(self):
        self.api_key = POLYGON_API_KEY
        self.base_url = POLYGON_BASE_URL
        
    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make authenticated request to Polygon API."""
        if not self.api_key:
            log.error("POLYGON_API_KEY not set")
            return None
            
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            else:
                log.warning(f"Polygon API error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            log.error(f"Polygon request failed: {e}")
            return None
    
    def get_options_chain(
        self,
        ticker: str,
        expiration_date: Optional[str] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        limit: int = 250
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
            List of options contract data with greeks
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
    
    def get_chain_for_expiration(
        self,
        ticker: str,
        spot_price: float,
        expiration_date: str,
        strike_range: int = 10,
    ) -> Dict[str, Any]:
        """
        Get options chain for a specific expiration date (e.g. jobs report Friday).
        
        Args:
            ticker: Underlying symbol
            spot_price: Current price of underlying
            expiration_date: YYYY-MM-DD (e.g. "2026-02-06")
            strike_range: Number of strikes above/below ATM
            
        Returns:
            Dict with calls, puts, metrics, GEX, max pain, walls (same shape as get_0dte_chain).
        """
        strike_step = 1 if spot_price < 200 else 5 if spot_price < 500 else 10
        strike_gte = spot_price - (strike_range * strike_step)
        strike_lte = spot_price + (strike_range * strike_step)
        contracts = self.get_options_chain(
            ticker=ticker,
            expiration_date=expiration_date,
            strike_price_gte=strike_gte,
            strike_price_lte=strike_lte,
            limit=200,
        )
        return self._build_chain_result(ticker, spot_price, expiration_date, contracts)

    def get_0dte_chain(self, ticker: str, spot_price: float, strike_range: int = 10) -> Dict[str, Any]:
        """
        Get 0DTE options chain centered around spot price with full analysis.
        
        Args:
            ticker: Underlying symbol
            spot_price: Current price of underlying
            strike_range: Number of strikes above/below ATM
            
        Returns:
            Dict with calls, puts, computed metrics, GEX, max pain, and walls
        """
        today = date.today().strftime("%Y-%m-%d")
        strike_step = 1 if spot_price < 200 else 5 if spot_price < 500 else 10
        strike_gte = spot_price - (strike_range * strike_step)
        strike_lte = spot_price + (strike_range * strike_step)
        contracts = self.get_options_chain(
            ticker=ticker,
            expiration_date=today,
            strike_price_gte=strike_gte,
            strike_price_lte=strike_lte,
            limit=200,
        )
        return self._build_chain_result(ticker, spot_price, today, contracts)

    def _build_chain_result(
        self,
        ticker: str,
        spot_price: float,
        expiration_date: str,
        contracts: List[Dict],
    ) -> Dict[str, Any]:
        """Parse raw contract list into calls/puts and compute GEX, max pain, walls."""
        calls = []
        puts = []
        
        for contract in contracts:
            details = contract.get("details", {})
            greeks = contract.get("greeks", {})
            day = contract.get("day", {})
            underlying = contract.get("underlying_asset", {})
            
            parsed = {
                "symbol": details.get("ticker", ""),
                "strike": details.get("strike_price", 0),
                "expiration": details.get("expiration_date", ""),
                "type": details.get("contract_type", "").lower(),
                "iv": contract.get("implied_volatility", 0),
                "delta": greeks.get("delta", 0),
                "gamma": greeks.get("gamma", 0),
                "theta": greeks.get("theta", 0),
                "vega": greeks.get("vega", 0),
                "volume": day.get("volume", 0),
                "open_interest": contract.get("open_interest", 0),
                "last_price": day.get("close", 0),
                "bid": contract.get("last_quote", {}).get("bid", 0),
                "ask": contract.get("last_quote", {}).get("ask", 0),
                "change": day.get("change", 0),
                "change_pct": day.get("change_percent", 0),
            }
            
            if parsed["type"] == "call":
                calls.append(parsed)
            elif parsed["type"] == "put":
                puts.append(parsed)
        
        total_call_volume = sum(c["volume"] for c in calls)
        total_put_volume = sum(p["volume"] for p in puts)
        total_call_oi = sum(c["open_interest"] for c in calls)
        total_put_oi = sum(p["open_interest"] for p in puts)
        
        all_strikes = [c["strike"] for c in calls] + [p["strike"] for p in puts]
        atm_strike = min(all_strikes, key=lambda s: abs(s - spot_price)) if all_strikes else spot_price
        
        all_contracts = calls + puts
        top_volume_strikes = sorted(all_contracts, key=lambda x: x["volume"], reverse=True)[:5]
        top_oi_strikes = sorted(all_contracts, key=lambda x: x["open_interest"], reverse=True)[:5]
        
        all_ivs = [c["iv"] for c in all_contracts if c["iv"] and c["iv"] > 0]
        avg_iv = sum(all_ivs) / len(all_ivs) if all_ivs else 0
        
        gex_data = self._calculate_gex(calls, puts, spot_price)
        max_pain = self._calculate_max_pain(calls, puts, spot_price)
        volume_walls = self._detect_volume_walls(calls, puts, spot_price)
        oi_walls = self._detect_oi_walls(calls, puts, spot_price)
        
        return {
            "ticker": ticker,
            "spot_price": spot_price,
            "expiration": expiration_date,
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
            },
            "gex": gex_data,
            "max_pain": max_pain,
            "volume_walls": volume_walls,
            "oi_walls": oi_walls,
        }
    
    def _calculate_gex(self, calls: List[Dict], puts: List[Dict], spot_price: float) -> Dict[str, Any]:
        """
        Calculate Gamma Exposure (GEX) across strikes.
        
        GEX = Gamma × Open Interest × 100 × Spot Price²
        - Positive GEX: Market makers hedge by selling when price rises (dampens moves)
        - Negative GEX: Market makers hedge by buying when price rises (amplifies moves)
        """
        gex_by_strike: Dict[float, float] = {}
        total_gex = 0.0
        
        for call in calls:
            strike = call["strike"]
            gamma = call.get("gamma", 0) or 0
            oi = call.get("open_interest", 0) or 0
            
            call_gex = gamma * oi * 100 * spot_price
            gex_by_strike[strike] = gex_by_strike.get(strike, 0) + call_gex
            total_gex += call_gex
        
        for put in puts:
            strike = put["strike"]
            gamma = put.get("gamma", 0) or 0
            oi = put.get("open_interest", 0) or 0
            
            put_gex = -gamma * oi * 100 * spot_price
            gex_by_strike[strike] = gex_by_strike.get(strike, 0) + put_gex
            total_gex += put_gex
        
        sorted_strikes = sorted(gex_by_strike.items(), key=lambda x: abs(x[1]), reverse=True)
        top_gex_strikes = sorted_strikes[:5] if sorted_strikes else []
        
        positive_gex = sum(v for v in gex_by_strike.values() if v > 0)
        negative_gex = sum(v for v in gex_by_strike.values() if v < 0)
        
        gex_regime = "positive" if total_gex > 0 else "negative"
        
        flip_point = None
        sorted_by_strike = sorted(gex_by_strike.items())
        for i in range(len(sorted_by_strike) - 1):
            strike1, gex1 = sorted_by_strike[i]
            strike2, gex2 = sorted_by_strike[i + 1]
            
            if (gex1 > 0 and gex2 < 0) or (gex1 < 0 and gex2 > 0):
                flip_point = (strike1 + strike2) / 2
                break
        
        return {
            "total_gex": total_gex,
            "gex_regime": gex_regime,
            "positive_gex": positive_gex,
            "negative_gex": negative_gex,
            "gex_flip_point": flip_point,
            "top_gex_strikes": [{"strike": s, "gex": g} for s, g in top_gex_strikes],
            "gex_by_strike": {str(k): v for k, v in sorted_strikes[:10]},
        }
    
    def _calculate_max_pain(self, calls: List[Dict], puts: List[Dict], spot_price: float) -> Dict[str, Any]:
        """
        Calculate max pain (strike where options expire worthless = minimum payout).
        
        At each strike, calculate total $ value of in-the-money options.
        Max pain is the strike with minimum total value.
        """
        all_strikes = set(c["strike"] for c in calls) | set(p["strike"] for p in puts)
        
        if not all_strikes:
            return {"max_pain_strike": spot_price, "distance_from_spot": 0, "pain_curve": []}
        
        pain_by_strike = {}
        
        for test_strike in all_strikes:
            total_pain = 0.0
            
            for call in calls:
                if test_strike > call["strike"]:
                    itm_value = (test_strike - call["strike"]) * call.get("open_interest", 0) * 100
                    total_pain += itm_value
            
            for put in puts:
                if test_strike < put["strike"]:
                    itm_value = (put["strike"] - test_strike) * put.get("open_interest", 0) * 100
                    total_pain += itm_value
            
            pain_by_strike[test_strike] = total_pain
        
        max_pain_strike = min(pain_by_strike.items(), key=lambda x: x[1])[0] if pain_by_strike else spot_price
        distance_from_spot = ((max_pain_strike - spot_price) / spot_price) * 100
        
        sorted_pain = sorted(pain_by_strike.items())
        pain_curve = [{"strike": s, "pain": p} for s, p in sorted_pain]
        
        return {
            "max_pain_strike": max_pain_strike,
            "distance_from_spot": distance_from_spot,
            "distance_pct": abs(distance_from_spot),
            "direction": "above" if max_pain_strike > spot_price else "below" if max_pain_strike < spot_price else "at_spot",
            "pain_curve": pain_curve[:20],
        }
    
    def _detect_volume_walls(self, calls: List[Dict], puts: List[Dict], spot_price: float) -> Dict[str, Any]:
        """
        Detect volume walls (strikes with unusually high trading volume).
        
        Volume walls often act as support/resistance as dealers hedge.
        """
        call_volumes = [(c["strike"], c["volume"]) for c in calls if c["volume"] > 0]
        put_volumes = [(p["strike"], p["volume"]) for p in puts if p["volume"] > 0]
        
        avg_call_volume = sum(v for _, v in call_volumes) / len(call_volumes) if call_volumes else 0
        avg_put_volume = sum(v for _, v in put_volumes) / len(put_volumes) if put_volumes else 0
        
        call_walls = [
            {"strike": s, "volume": v, "ratio": v / avg_call_volume if avg_call_volume else 0}
            for s, v in call_volumes
            if v > avg_call_volume * 2
        ]
        call_walls.sort(key=lambda x: x["volume"], reverse=True)
        
        put_walls = [
            {"strike": s, "volume": v, "ratio": v / avg_put_volume if avg_put_volume else 0}
            for s, v in put_volumes
            if v > avg_put_volume * 2
        ]
        put_walls.sort(key=lambda x: x["volume"], reverse=True)
        
        resistance_levels = [w["strike"] for w in call_walls if w["strike"] > spot_price][:3]
        support_levels = [w["strike"] for w in put_walls if w["strike"] < spot_price][:3]
        
        return {
            "call_walls": call_walls[:5],
            "put_walls": put_walls[:5],
            "resistance_levels": resistance_levels,
            "support_levels": support_levels,
            "nearest_resistance": min(resistance_levels, key=lambda x: x - spot_price) if resistance_levels else None,
            "nearest_support": max(support_levels, key=lambda x: spot_price - x) if support_levels else None,
        }
    
    def _detect_oi_walls(self, calls: List[Dict], puts: List[Dict], spot_price: float) -> Dict[str, Any]:
        """
        Detect open interest walls (strikes with heavy positioned OI).
        
        OI walls are more persistent than volume walls and often indicate
        major support/resistance levels where dealers have significant exposure.
        """
        call_oi = [(c["strike"], c["open_interest"]) for c in calls if c["open_interest"] > 0]
        put_oi = [(p["strike"], p["open_interest"]) for p in puts if p["open_interest"] > 0]
        
        avg_call_oi = sum(oi for _, oi in call_oi) / len(call_oi) if call_oi else 0
        avg_put_oi = sum(oi for _, oi in put_oi) / len(put_oi) if put_oi else 0
        
        call_walls = [
            {"strike": s, "oi": oi, "ratio": oi / avg_call_oi if avg_call_oi else 0}
            for s, oi in call_oi
            if oi > avg_call_oi * 1.5
        ]
        call_walls.sort(key=lambda x: x["oi"], reverse=True)
        
        put_walls = [
            {"strike": s, "oi": oi, "ratio": oi / avg_put_oi if avg_put_oi else 0}
            for s, oi in put_oi
            if oi > avg_put_oi * 1.5
        ]
        put_walls.sort(key=lambda x: x["oi"], reverse=True)
        
        gamma_magnets = []
        for call in calls:
            for put in puts:
                if call["strike"] == put["strike"]:
                    combined_oi = call["open_interest"] + put["open_interest"]
                    combined_gamma = abs(call.get("gamma", 0) or 0) + abs(put.get("gamma", 0) or 0)
                    if combined_oi > (avg_call_oi + avg_put_oi):
                        gamma_magnets.append({
                            "strike": call["strike"],
                            "combined_oi": combined_oi,
                            "combined_gamma": combined_gamma,
                            "distance_from_spot": abs(call["strike"] - spot_price),
                        })
        gamma_magnets.sort(key=lambda x: x["combined_oi"], reverse=True)
        
        return {
            "call_oi_walls": call_walls[:5],
            "put_oi_walls": put_walls[:5],
            "gamma_magnets": gamma_magnets[:5],
            "primary_magnet": gamma_magnets[0]["strike"] if gamma_magnets else None,
        }
    
    def get_full_options_analysis(self, ticker: str, spot_price: float) -> Dict[str, Any]:
        """
        Get comprehensive options analysis for 0DTE trading.
        
        Returns:
            Complete analysis including GEX, max pain, walls, and IV surface.
        """
        chain_data = self.get_0dte_chain(ticker, spot_price)
        
        if not chain_data.get("calls") and not chain_data.get("puts"):
            return {
                "ticker": ticker,
                "spot_price": spot_price,
                "error": "No options data available",
                "has_data": False,
            }
        
        calls = chain_data["calls"]
        puts = chain_data["puts"]
        
        atm_calls = [c for c in calls if abs(c["strike"] - spot_price) < 2]
        atm_puts = [p for p in puts if abs(p["strike"] - spot_price) < 2]
        atm_iv = sum(c["iv"] for c in atm_calls + atm_puts if c["iv"]) / max(len(atm_calls + atm_puts), 1)
        
        otm_call_iv = sum(c["iv"] for c in calls if c["strike"] > spot_price and c["iv"]) / max(len([c for c in calls if c["strike"] > spot_price]), 1) if calls else 0
        otm_put_iv = sum(p["iv"] for p in puts if p["strike"] < spot_price and p["iv"]) / max(len([p for p in puts if p["strike"] < spot_price]), 1) if puts else 0
        
        iv_skew = otm_put_iv - otm_call_iv if otm_put_iv and otm_call_iv else 0
        
        gex = chain_data["gex"]
        max_pain = chain_data["max_pain"]
        volume_walls = chain_data["volume_walls"]
        oi_walls = chain_data["oi_walls"]
        
        signals = []
        
        if gex["gex_regime"] == "negative":
            signals.append(("NEGATIVE_GEX", "Amplified moves likely"))
        else:
            signals.append(("POSITIVE_GEX", "Dampened moves likely"))
        
        if max_pain["distance_pct"] > 0.5:
            if max_pain["direction"] == "above":
                signals.append(("MAX_PAIN_ABOVE", f"Max pain ${max_pain['max_pain_strike']:.0f} above spot"))
            else:
                signals.append(("MAX_PAIN_BELOW", f"Max pain ${max_pain['max_pain_strike']:.0f} below spot"))
        
        if volume_walls["nearest_resistance"]:
            resistance = volume_walls["nearest_resistance"]
            if (resistance - spot_price) / spot_price < 0.01:
                signals.append(("NEAR_CALL_WALL", f"Approaching call wall at ${resistance:.0f}"))
        
        if volume_walls["nearest_support"]:
            support = volume_walls["nearest_support"]
            if (spot_price - support) / spot_price < 0.01:
                signals.append(("NEAR_PUT_WALL", f"Approaching put wall at ${support:.0f}"))
        
        if oi_walls["primary_magnet"]:
            magnet = oi_walls["primary_magnet"]
            if abs(magnet - spot_price) / spot_price < 0.005:
                signals.append(("AT_GAMMA_MAGNET", f"Pinned near gamma magnet ${magnet:.0f}"))
        
        if iv_skew > 0.05:
            signals.append(("PUT_SKEW", "Elevated put IV skew"))
        elif iv_skew < -0.05:
            signals.append(("CALL_SKEW", "Elevated call IV skew"))
        
        return {
            "ticker": ticker,
            "spot_price": spot_price,
            "has_data": True,
            "atm_strike": chain_data["atm_strike"],
            "iv_surface": {
                "atm_iv": atm_iv,
                "otm_call_iv": otm_call_iv,
                "otm_put_iv": otm_put_iv,
                "iv_skew": iv_skew,
            },
            "gex": gex,
            "max_pain": max_pain,
            "volume_walls": volume_walls,
            "oi_walls": oi_walls,
            "metrics": chain_data["metrics"],
            "signals": signals,
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


polygon_options = PolygonOptionsAdapter()
