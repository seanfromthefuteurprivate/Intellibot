"""
Enhanced Polygon.io Data Adapter

Maximizes the Polygon plan by using ALL available endpoints:
- Stock aggregates (5s, 15s, 1min, 5min bars for multi-timeframe analysis)
- Technical indicators (RSI, SMA, EMA, MACD)
- Gainers/Losers (market regime)
- Stock snapshots (real-time quotes)
- Options contracts reference (strike structure)
- Trades endpoint (recent trade flow)
- Quotes/NBBO endpoint (bid-ask spread analysis)

Optimized for 0DTE SPY scalping with surgical precision data.
"""

import requests
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from wsb_snake.config import POLYGON_API_KEY, POLYGON_BASE_URL
from wsb_snake.utils.logger import log


class EnhancedPolygonAdapter:
    """Enhanced adapter maximizing Polygon basic plan capabilities."""
    
    # Rate limiting: Polygon basic plan allows 5 requests/min
    REQUESTS_PER_MINUTE = 5
    
    def __init__(self):
        self.api_key = POLYGON_API_KEY
        self.base_url = POLYGON_BASE_URL
        self._cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = 120  # Increased to 2 minutes to reduce API calls
        self._request_times: List[datetime] = []
        
        # Per-scan cache for batch efficiency
        self._scan_cache: Dict[str, Any] = {}
        self._scan_cache_time: Optional[datetime] = None
        self._scan_cache_ttl = 60  # Per-scan cache valid for 60s
        
    def _request(self, endpoint: str, params: Dict = None, cache_ttl_override: int = None) -> Optional[Dict]:
        """Make authenticated request to Polygon API with caching and rate limiting."""
        if not self.api_key:
            log.error("POLYGON_API_KEY not set")
            return None
            
        if params is None:
            params = {}
        params["apiKey"] = self.api_key
        
        # Check cache first (use override TTL if provided)
        cache_ttl = cache_ttl_override if cache_ttl_override is not None else self._cache_ttl
        cache_key = f"{endpoint}:{str(sorted(params.items()))}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < cache_ttl:
                return cached_data
        
        # Rate limiting - track requests and throttle if needed
        now = datetime.now()
        self._request_times = [t for t in self._request_times if (now - t).seconds < 60]
        
        if len(self._request_times) >= self.REQUESTS_PER_MINUTE:
            # Too many requests - use cache even if expired, or wait
            if cache_key in self._cache:
                log.debug(f"Rate limited - using stale cache for {endpoint}")
                return self._cache[cache_key][1]
            else:
                # Wait for rate limit to clear
                oldest = min(self._request_times)
                wait_time = 61 - (now - oldest).seconds
                if wait_time > 0 and wait_time < 30:
                    log.debug(f"Rate limit - waiting {wait_time}s")
                    import time
                    time.sleep(wait_time)
                    self._request_times = []
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            self._request_times.append(datetime.now())
            
            if resp.status_code == 200:
                data = resp.json()
                self._cache[cache_key] = (datetime.now(), data)
                return data
            elif resp.status_code == 429:
                # Rate limited by API - use stale cache if available
                log.warning(f"Polygon 429 rate limited: {endpoint}")
                if cache_key in self._cache:
                    return self._cache[cache_key][1]
                return None
            else:
                log.warning(f"Polygon API {resp.status_code}: {endpoint}")
                return None
        except Exception as e:
            log.error(f"Polygon request failed: {e}")
            # Return stale cache on network error
            if cache_key in self._cache:
                return self._cache[cache_key][1]
            return None
    
    # ========================
    # STOCK DATA
    # ========================
    
    def get_intraday_bars(
        self, 
        ticker: str, 
        timespan: str = "minute",
        multiplier: int = 1,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get intraday price bars for momentum analysis.
        
        Args:
            ticker: Stock symbol
            timespan: minute, hour, day
            multiplier: Bar size multiplier
            limit: Number of bars
            
        Returns:
            List of OHLCV bars
        """
        today = date.today().strftime("%Y-%m-%d")
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{yesterday}/{today}"
        data = self._request(endpoint, {"limit": limit, "sort": "desc"})
        
        if data and "results" in data:
            bars = []
            for bar in data["results"]:
                bars.append({
                    "timestamp": bar.get("t", 0),
                    "open": bar.get("o", 0),
                    "high": bar.get("h", 0),
                    "low": bar.get("l", 0),
                    "close": bar.get("c", 0),
                    "volume": bar.get("v", 0),
                    "vwap": bar.get("vw", 0),
                    "trades": bar.get("n", 0),
                })
            return bars
        return []
    
    def get_ultra_fast_bars(
        self, 
        ticker: str, 
        seconds: int = 5,
        limit: int = 120
    ) -> List[Dict]:
        """
        Get ultra-fast sub-minute bars for 0DTE scalping.
        
        Args:
            ticker: Stock symbol
            seconds: Bar size in seconds (5, 15, 30)
            limit: Number of bars (default 120 = 10 minutes of 5s bars)
            
        Returns:
            List of OHLCV bars with sub-minute granularity
        """
        now = datetime.now()
        today = date.today().strftime("%Y-%m-%d")
        
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{seconds}/second/{today}/{today}"
        data = self._request(endpoint, {"limit": limit, "sort": "desc"})
        
        if data and "results" in data:
            bars = []
            for bar in data["results"]:
                bars.append({
                    "t": bar.get("t", 0),
                    "timestamp": bar.get("t", 0),
                    "open": bar.get("o", 0),
                    "high": bar.get("h", 0),
                    "low": bar.get("l", 0),
                    "close": bar.get("c", 0),
                    "volume": bar.get("v", 0),
                    "vwap": bar.get("vw", 0),
                    "trades": bar.get("n", 0),
                })
            log.debug(f"Got {len(bars)} ultra-fast {seconds}s bars for {ticker}")
            return bars
        return []
    
    def get_recent_trades(
        self, 
        ticker: str, 
        limit: int = 100
    ) -> List[Dict]:
        """
        Get recent trades for order flow analysis.
        Shows actual executed trades with size, price, and exchange.
        
        Args:
            ticker: Stock symbol
            limit: Number of recent trades
            
        Returns:
            List of recent trades
        """
        endpoint = f"/v3/trades/{ticker}"
        data = self._request(endpoint, {"limit": limit, "order": "desc"})
        
        if data and "results" in data:
            trades = []
            for t in data["results"]:
                trades.append({
                    "timestamp": t.get("sip_timestamp", 0),
                    "price": t.get("price", 0),
                    "size": t.get("size", 0),
                    "exchange": t.get("exchange", 0),
                    "conditions": t.get("conditions", []),
                    "tape": t.get("tape", ""),
                })
            log.debug(f"Got {len(trades)} recent trades for {ticker}")
            return trades
        return []
    
    def get_nbbo_quotes(
        self, 
        ticker: str, 
        limit: int = 50
    ) -> List[Dict]:
        """
        Get recent NBBO quotes for bid-ask spread analysis.
        Shows best bid/ask across all exchanges.
        
        Args:
            ticker: Stock symbol
            limit: Number of recent quotes
            
        Returns:
            List of NBBO quotes with bid/ask
        """
        endpoint = f"/v3/quotes/{ticker}"
        data = self._request(endpoint, {"limit": limit, "order": "desc"})
        
        if data and "results" in data:
            quotes = []
            for q in data["results"]:
                bid = q.get("bid_price", 0)
                ask = q.get("ask_price", 0)
                spread = ask - bid if bid and ask else 0
                quotes.append({
                    "timestamp": q.get("sip_timestamp", 0),
                    "bid": bid,
                    "bid_size": q.get("bid_size", 0),
                    "ask": ask,
                    "ask_size": q.get("ask_size", 0),
                    "spread": spread,
                    "spread_pct": (spread / bid * 100) if bid else 0,
                })
            log.debug(f"Got {len(quotes)} NBBO quotes for {ticker}")
            return quotes
        return []
    
    # Trade condition codes for ruthless classification
    CONDITION_CODES = {
        # Regular trades
        0: "regular",
        # Odd lot
        37: "odd_lot",
        # Average price trade
        52: "average_price",
        # Cash trade
        53: "cash",
        # Intermarket sweep
        12: "intermarket_sweep",
        15: "intermarket_sweep",
        # Opening trade
        17: "opening",
        # Closing trade
        6: "closing",
        # Contingent trade
        40: "contingent",
        # Derivatively priced
        30: "derivative",
        # Form T (after hours)
        29: "form_t",
        14: "form_t",
    }
    
    def classify_trade(self, trade: Dict) -> Dict:
        """
        Classify a trade based on its condition codes.
        
        Intermarket sweeps = aggressive institutional orders
        Large odd lots = retail panic
        Blocks = big institutional interest
        """
        conditions = trade.get("conditions", [])
        size = trade.get("size", 0)
        price = trade.get("price", 0)
        
        classifications = []
        for cond in conditions:
            if cond in self.CONDITION_CODES:
                classifications.append(self.CONDITION_CODES[cond])
        
        if not classifications:
            classifications = ["regular"]
        
        # Determine trade type
        trade_type = "regular"
        is_sweep = "intermarket_sweep" in classifications
        is_odd_lot = size < 100
        is_block = size >= 10000
        is_large = size >= 1000
        
        if is_sweep:
            trade_type = "SWEEP"  # Aggressive institutional
        elif is_block:
            trade_type = "BLOCK"  # Large institutional
        elif is_large:
            trade_type = "LARGE"  # Notable size
        elif is_odd_lot:
            trade_type = "ODD_LOT"  # Retail
        
        notional = price * size
        
        return {
            **trade,
            "trade_type": trade_type,
            "classifications": classifications,
            "is_sweep": is_sweep,
            "is_block": is_block,
            "is_large": is_large,
            "is_odd_lot": is_odd_lot,
            "notional": notional,
        }
    
    def get_classified_trades(self, ticker: str, limit: int = 200) -> Dict[str, Any]:
        """
        Get trades with classification for order flow analysis.
        Returns breakdown of sweeps, blocks, and retail activity.
        """
        trades = self.get_recent_trades(ticker, limit)
        
        if not trades:
            return {"available": False}
        
        classified = [self.classify_trade(t) for t in trades]
        
        sweeps = [t for t in classified if t["is_sweep"]]
        blocks = [t for t in classified if t["is_block"]]
        large = [t for t in classified if t["is_large"] and not t["is_block"]]
        odd_lots = [t for t in classified if t["is_odd_lot"]]
        
        total_volume = sum(t["size"] for t in classified)
        sweep_volume = sum(t["size"] for t in sweeps)
        block_volume = sum(t["size"] for t in blocks)
        
        sweep_pct = sweep_volume / max(total_volume, 1) * 100
        block_pct = block_volume / max(total_volume, 1) * 100
        institutional_pct = (sweep_volume + block_volume) / max(total_volume, 1) * 100
        
        # Determine institutional bias
        if sweeps:
            avg_sweep_price = sum(t["price"] for t in sweeps) / len(sweeps)
            recent_price = classified[0]["price"] if classified else 0
            sweep_direction = "BUY" if avg_sweep_price <= recent_price else "SELL"
        else:
            sweep_direction = "NONE"
        
        return {
            "available": True,
            "ticker": ticker,
            "total_trades": len(classified),
            "total_volume": total_volume,
            "sweep_count": len(sweeps),
            "sweep_volume": sweep_volume,
            "sweep_pct": sweep_pct,
            "sweep_direction": sweep_direction,
            "block_count": len(blocks),
            "block_volume": block_volume,
            "block_pct": block_pct,
            "large_count": len(large),
            "odd_lot_count": len(odd_lots),
            "institutional_pct": institutional_pct,
            "is_institutional_active": institutional_pct > 20,
            "trades": classified[:50],  # Return first 50 for context
        }
        return []
    
    def analyze_order_flow(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze order flow using trades and quotes for directional bias.
        
        Returns:
            Order flow analysis with buy/sell pressure indicators
        """
        trades = self.get_recent_trades(ticker, limit=100)
        quotes = self.get_nbbo_quotes(ticker, limit=50)
        
        if not trades:
            return {"available": False, "error": "No trade data"}
        
        total_volume = sum(t["size"] for t in trades)
        total_value = sum(t["price"] * t["size"] for t in trades)
        avg_price = total_value / total_volume if total_volume else 0
        
        large_trades = [t for t in trades if t["size"] >= 100]
        large_volume = sum(t["size"] for t in large_trades)
        
        prices = [t["price"] for t in trades]
        price_direction = (prices[0] - prices[-1]) if len(prices) >= 2 else 0
        
        bid_ask_imbalance = 0
        avg_spread = 0
        if quotes:
            total_bid_size = sum(q["bid_size"] for q in quotes)
            total_ask_size = sum(q["ask_size"] for q in quotes)
            bid_ask_imbalance = (total_bid_size - total_ask_size) / max(total_bid_size + total_ask_size, 1)
            avg_spread = sum(q["spread"] for q in quotes) / len(quotes)
        
        if price_direction > 0 and bid_ask_imbalance > 0.1:
            flow_signal = "STRONG_BUY"
            flow_score = 2
        elif price_direction > 0:
            flow_signal = "BUY"
            flow_score = 1
        elif price_direction < 0 and bid_ask_imbalance < -0.1:
            flow_signal = "STRONG_SELL"
            flow_score = -2
        elif price_direction < 0:
            flow_signal = "SELL"
            flow_score = -1
        else:
            flow_signal = "NEUTRAL"
            flow_score = 0
        
        return {
            "available": True,
            "ticker": ticker,
            "total_trades": len(trades),
            "total_volume": total_volume,
            "avg_price": avg_price,
            "large_trades": len(large_trades),
            "large_volume_pct": large_volume / max(total_volume, 1) * 100,
            "price_direction": price_direction,
            "bid_ask_imbalance": bid_ask_imbalance,
            "avg_spread": avg_spread,
            "flow_signal": flow_signal,
            "flow_score": flow_score,
        }
    
    def get_previous_day(self, ticker: str) -> Optional[Dict]:
        """Get previous day's aggregates for gap analysis."""
        endpoint = f"/v2/aggs/ticker/{ticker}/prev"
        data = self._request(endpoint)
        
        if data and "results" in data and data["results"]:
            bar = data["results"][0]
            return {
                "open": bar.get("o", 0),
                "high": bar.get("h", 0),
                "low": bar.get("l", 0),
                "close": bar.get("c", 0),
                "volume": bar.get("v", 0),
                "vwap": bar.get("vw", 0),
            }
        return None
    
    def get_snapshot(self, ticker: str) -> Optional[Dict]:
        """Get real-time quote snapshot."""
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        data = self._request(endpoint)
        
        if data and "ticker" in data:
            t = data["ticker"]
            day = t.get("day", {})
            prev = t.get("prevDay", {})
            
            return {
                "symbol": ticker,
                "price": t.get("lastTrade", {}).get("p", 0) or day.get("c", 0),
                "today_open": day.get("o", 0),
                "today_high": day.get("h", 0),
                "today_low": day.get("l", 0),
                "today_close": day.get("c", 0),
                "today_volume": day.get("v", 0),
                "today_vwap": day.get("vw", 0),
                "prev_close": prev.get("c", 0),
                "prev_volume": prev.get("v", 0),
                "change_pct": t.get("todaysChangePerc", 0),
                "updated": t.get("updated", 0),
            }
        return None
    
    # ========================
    # TECHNICAL INDICATORS
    # ========================
    
    def get_rsi(
        self, 
        ticker: str, 
        window: int = 14, 
        timespan: str = "minute"
    ) -> Optional[Dict]:
        """
        Get RSI indicator values.
        
        Args:
            ticker: Stock symbol
            window: RSI period (default 14)
            timespan: minute, hour, day
            
        Returns:
            RSI data with current and historical values
        """
        endpoint = f"/v1/indicators/rsi/{ticker}"
        data = self._request(endpoint, {
            "timespan": timespan,
            "window": window,
            "limit": 10,
            "order": "desc"
        })
        
        if data and "results" in data and "values" in data["results"]:
            values = data["results"]["values"]
            if values:
                return {
                    "current": values[0].get("value", 50),
                    "previous": values[1].get("value", 50) if len(values) > 1 else 50,
                    "history": [v.get("value", 50) for v in values],
                    "window": window,
                    "timespan": timespan,
                }
        return None
    
    def get_sma(
        self, 
        ticker: str, 
        window: int = 20, 
        timespan: str = "minute"
    ) -> Optional[Dict]:
        """Get Simple Moving Average."""
        endpoint = f"/v1/indicators/sma/{ticker}"
        data = self._request(endpoint, {
            "timespan": timespan,
            "window": window,
            "limit": 10,
            "order": "desc"
        })
        
        if data and "results" in data and "values" in data["results"]:
            values = data["results"]["values"]
            if values:
                return {
                    "current": values[0].get("value", 0),
                    "previous": values[1].get("value", 0) if len(values) > 1 else 0,
                    "history": [v.get("value", 0) for v in values],
                    "window": window,
                }
        return None
    
    def get_ema(
        self, 
        ticker: str, 
        window: int = 9, 
        timespan: str = "minute"
    ) -> Optional[Dict]:
        """Get Exponential Moving Average."""
        endpoint = f"/v1/indicators/ema/{ticker}"
        data = self._request(endpoint, {
            "timespan": timespan,
            "window": window,
            "limit": 10,
            "order": "desc"
        })
        
        if data and "results" in data and "values" in data["results"]:
            values = data["results"]["values"]
            if values:
                return {
                    "current": values[0].get("value", 0),
                    "previous": values[1].get("value", 0) if len(values) > 1 else 0,
                    "history": [v.get("value", 0) for v in values],
                    "window": window,
                }
        return None
    
    def get_macd(
        self, 
        ticker: str, 
        timespan: str = "minute",
        short_window: int = 12,
        long_window: int = 26,
        signal_window: int = 9
    ) -> Optional[Dict]:
        """Get MACD indicator."""
        endpoint = f"/v1/indicators/macd/{ticker}"
        data = self._request(endpoint, {
            "timespan": timespan,
            "short_window": short_window,
            "long_window": long_window,
            "signal_window": signal_window,
            "limit": 10,
            "order": "desc"
        })
        
        if data and "results" in data and "values" in data["results"]:
            values = data["results"]["values"]
            if values:
                current = values[0]
                return {
                    "macd": current.get("value", 0),
                    "signal": current.get("signal", 0),
                    "histogram": current.get("histogram", 0),
                    "history": values,
                }
        return None
    
    def get_full_technicals(self, ticker: str) -> Dict[str, Any]:
        """Get all technical indicators for a ticker."""
        snapshot = self.get_snapshot(ticker)
        price = snapshot.get("price", 0) if snapshot else 0
        
        rsi = self.get_rsi(ticker, window=14, timespan="minute")
        sma_20 = self.get_sma(ticker, window=20, timespan="minute")
        ema_9 = self.get_ema(ticker, window=9, timespan="minute")
        macd = self.get_macd(ticker, timespan="minute")
        
        # Compute derived signals
        signals = []
        
        if rsi:
            rsi_val = rsi["current"]
            if rsi_val > 70:
                signals.append(("RSI_OVERBOUGHT", -1))
            elif rsi_val < 30:
                signals.append(("RSI_OVERSOLD", 1))
            elif rsi_val > rsi["previous"]:
                signals.append(("RSI_RISING", 0.5))
            else:
                signals.append(("RSI_FALLING", -0.5))
        
        if sma_20 and price:
            if price > sma_20["current"] * 1.02:
                signals.append(("ABOVE_SMA20", 1))
            elif price < sma_20["current"] * 0.98:
                signals.append(("BELOW_SMA20", -1))
        
        if ema_9 and sma_20:
            if ema_9["current"] > sma_20["current"]:
                signals.append(("EMA_ABOVE_SMA", 1))
            else:
                signals.append(("EMA_BELOW_SMA", -1))
        
        if macd:
            if macd["histogram"] > 0 and macd["macd"] > macd["signal"]:
                signals.append(("MACD_BULLISH", 1))
            elif macd["histogram"] < 0:
                signals.append(("MACD_BEARISH", -1))
        
        return {
            "ticker": ticker,
            "price": price,
            "snapshot": snapshot,
            "rsi": rsi,
            "sma_20": sma_20,
            "ema_9": ema_9,
            "macd": macd,
            "signals": signals,
            "net_signal": sum(s[1] for s in signals),
        }
    
    # ========================
    # MARKET REGIME
    # ========================
    
    def get_market_movers(self, direction: str = "gainers") -> List[Dict]:
        """
        Get top market movers for regime detection.
        
        Args:
            direction: "gainers" or "losers"
            
        Returns:
            List of top moving stocks
        """
        endpoint = f"/v2/snapshot/locale/us/markets/stocks/{direction}"
        data = self._request(endpoint)
        
        if data and "tickers" in data:
            movers = []
            for t in data["tickers"][:20]:
                day = t.get("day", {})
                movers.append({
                    "ticker": t.get("ticker", ""),
                    "change_pct": t.get("todaysChangePerc", 0),
                    "price": day.get("c", 0),
                    "volume": day.get("v", 0),
                })
            return movers
        return []
    
    def get_market_regime(self) -> Dict[str, Any]:
        """
        Analyze market regime using gainers/losers ratio.
        
        Returns:
            Market regime classification
        """
        gainers = self.get_market_movers("gainers")
        losers = self.get_market_movers("losers")
        
        if not gainers and not losers:
            return {"regime": "unknown", "score": 0}
        
        # Compute breadth metrics
        avg_gainer_pct = sum(g["change_pct"] for g in gainers) / len(gainers) if gainers else 0
        avg_loser_pct = sum(l["change_pct"] for l in losers) / len(losers) if losers else 0
        
        total_gainer_vol = sum(g["volume"] for g in gainers)
        total_loser_vol = sum(l["volume"] for l in losers)
        
        # Regime score: positive = bullish, negative = bearish
        regime_score = (avg_gainer_pct + avg_loser_pct) / 2  # Net change
        volume_tilt = (total_gainer_vol - total_loser_vol) / max(total_gainer_vol + total_loser_vol, 1)
        
        combined_score = regime_score * 10 + volume_tilt * 20
        
        if combined_score > 15:
            regime = "strong_bullish"
        elif combined_score > 5:
            regime = "bullish"
        elif combined_score > -5:
            regime = "neutral"
        elif combined_score > -15:
            regime = "bearish"
        else:
            regime = "strong_bearish"
        
        return {
            "regime": regime,
            "score": combined_score,
            "avg_gainer_pct": avg_gainer_pct,
            "avg_loser_pct": avg_loser_pct,
            "volume_tilt": volume_tilt,
            "top_gainers": [g["ticker"] for g in gainers[:5]],
            "top_losers": [l["ticker"] for l in losers[:5]],
        }
    
    # ========================
    # OPTIONS STRUCTURE
    # ========================
    
    def get_options_contracts(
        self, 
        ticker: str, 
        expiration_date: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get options contracts reference data (available on basic plan).
        
        This gives us strike structure without real-time pricing.
        """
        if expiration_date is None:
            expiration_date = date.today().strftime("%Y-%m-%d")
        
        endpoint = "/v3/reference/options/contracts"
        params = {
            "underlying_ticker": ticker,
            "expiration_date": expiration_date,
            "limit": limit,
            "order": "asc",
            "sort": "strike_price"
        }
        
        data = self._request(endpoint, params)
        
        if data and "results" in data:
            contracts = []
            for c in data["results"]:
                contracts.append({
                    "ticker": c.get("ticker", ""),
                    "underlying": c.get("underlying_ticker", ""),
                    "strike": c.get("strike_price", 0),
                    "expiration": c.get("expiration_date", ""),
                    "type": c.get("contract_type", "").lower(),
                    "style": c.get("exercise_style", ""),
                    "shares_per_contract": c.get("shares_per_contract", 100),
                })
            return contracts
        return []
    
    def analyze_strike_structure(self, ticker: str, spot_price: float) -> Dict[str, Any]:
        """
        Analyze options strike structure to infer support/resistance levels.
        
        Even without volume data, strike clustering tells us where market makers
        have concentrated their hedging - these become support/resistance.
        """
        today = date.today().strftime("%Y-%m-%d")
        contracts = self.get_options_contracts(ticker, today, limit=200)
        
        if not contracts:
            return {"available": False}
        
        calls = [c for c in contracts if c["type"] == "call"]
        puts = [c for c in contracts if c["type"] == "put"]
        
        call_strikes = [c["strike"] for c in calls]
        put_strikes = [p["strike"] for p in puts]
        all_strikes = sorted(set(call_strikes + put_strikes))
        
        # Find ATM strike
        atm_strike = min(all_strikes, key=lambda s: abs(s - spot_price)) if all_strikes else spot_price
        
        # Strike density analysis (more strikes = more liquidity/hedging)
        strike_step = 1 if spot_price < 200 else 5 if spot_price < 500 else 10
        
        # Identify key levels based on strike clustering
        # Round strikes (like 600, 610, 620 for SPY) typically have more activity
        round_call_strikes = [s for s in call_strikes if s % (strike_step * 5) == 0]
        round_put_strikes = [s for s in put_strikes if s % (strike_step * 5) == 0]
        
        # Support = puts below spot, Resistance = calls above spot
        support_levels = sorted([s for s in put_strikes if s < spot_price], reverse=True)[:3]
        resistance_levels = sorted([s for s in call_strikes if s > spot_price])[:3]
        
        return {
            "available": True,
            "ticker": ticker,
            "spot_price": spot_price,
            "atm_strike": atm_strike,
            "total_calls": len(calls),
            "total_puts": len(puts),
            "put_call_ratio": len(puts) / max(len(calls), 1),
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "key_round_calls": round_call_strikes[:5],
            "key_round_puts": round_put_strikes[:5],
            "strike_range": (min(all_strikes), max(all_strikes)) if all_strikes else (0, 0),
        }
    
    # ========================
    # MOMENTUM ANALYSIS
    # ========================
    
    def get_momentum_signals(self, ticker: str) -> Dict[str, Any]:
        """
        Comprehensive momentum analysis using all available data.
        """
        # Get intraday bars
        bars = self.get_intraday_bars(ticker, timespan="minute", limit=30)
        prev_day = self.get_previous_day(ticker)
        snapshot = self.get_snapshot(ticker)
        
        if not snapshot:
            return {"available": False}
        
        price = snapshot.get("price", 0)
        today_open = snapshot.get("today_open", 0)
        today_volume = snapshot.get("today_volume", 0)
        prev_close = snapshot.get("prev_close", 0)
        prev_volume = prev_day.get("volume", 0) if prev_day else 0
        
        signals = []
        
        # Gap analysis
        if prev_close and today_open:
            gap_pct = (today_open - prev_close) / prev_close * 100
            if gap_pct > 1:
                signals.append(("GAP_UP", gap_pct / 2))
            elif gap_pct < -1:
                signals.append(("GAP_DOWN", gap_pct / 2))
        
        # Volume analysis
        if prev_volume and today_volume:
            vol_ratio = today_volume / prev_volume
            if vol_ratio > 1.5:
                signals.append(("VOLUME_SURGE", min(vol_ratio - 1, 2)))
            elif vol_ratio < 0.5:
                signals.append(("VOLUME_DRY", -1))
        
        # Intraday momentum from bars
        if len(bars) >= 5:
            recent_bars = bars[:5]
            older_bars = bars[5:10] if len(bars) >= 10 else []
            
            recent_avg_vol = sum(b["volume"] for b in recent_bars) / len(recent_bars)
            
            if older_bars:
                older_avg_vol = sum(b["volume"] for b in older_bars) / len(older_bars)
                if recent_avg_vol > older_avg_vol * 1.5:
                    signals.append(("VOLUME_ACCELERATING", 1))
            
            # Price momentum
            if recent_bars[0]["close"] > recent_bars[-1]["close"]:
                price_change = (recent_bars[0]["close"] - recent_bars[-1]["close"]) / recent_bars[-1]["close"] * 100
                if price_change > 0.5:
                    signals.append(("PRICE_MOMENTUM_UP", price_change))
            else:
                price_change = (recent_bars[-1]["close"] - recent_bars[0]["close"]) / recent_bars[0]["close"] * 100
                if price_change > 0.5:
                    signals.append(("PRICE_MOMENTUM_DOWN", -price_change))
        
        # Day range position
        if snapshot.get("today_high") and snapshot.get("today_low"):
            day_range = snapshot["today_high"] - snapshot["today_low"]
            if day_range > 0:
                range_position = (price - snapshot["today_low"]) / day_range
                if range_position > 0.9:
                    signals.append(("NEAR_DAY_HIGH", 1))
                elif range_position < 0.1:
                    signals.append(("NEAR_DAY_LOW", -1))
        
        return {
            "available": True,
            "ticker": ticker,
            "price": price,
            "change_pct": snapshot.get("change_pct", 0),
            "volume_ratio": today_volume / max(prev_volume, 1) if prev_volume else 1,
            "signals": signals,
            "net_signal": sum(s[1] for s in signals),
            "bars_analyzed": len(bars),
        }


# Global instance
polygon_enhanced = EnhancedPolygonAdapter()


def get_full_analysis(ticker: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Get comprehensive analysis combining all data sources.
    
    Uses scan cache to avoid redundant API calls within the same scan cycle.
    """
    # Check scan cache first
    now = datetime.now()
    if use_cache and polygon_enhanced._scan_cache_time:
        cache_age = (now - polygon_enhanced._scan_cache_time).seconds
        if cache_age < polygon_enhanced._scan_cache_ttl:
            if ticker in polygon_enhanced._scan_cache:
                return polygon_enhanced._scan_cache[ticker]
    
    # Reset scan cache if expired
    if not polygon_enhanced._scan_cache_time or \
       (now - polygon_enhanced._scan_cache_time).seconds >= polygon_enhanced._scan_cache_ttl:
        polygon_enhanced._scan_cache = {}
        polygon_enhanced._scan_cache_time = now
    
    technicals = polygon_enhanced.get_full_technicals(ticker)
    momentum = polygon_enhanced.get_momentum_signals(ticker)
    strike_structure = polygon_enhanced.analyze_strike_structure(
        ticker, 
        momentum.get("price", 0) or technicals.get("price", 0)
    )
    
    # Combine all signals
    all_signals = []
    if technicals.get("signals"):
        all_signals.extend(technicals["signals"])
    if momentum.get("signals"):
        all_signals.extend(momentum["signals"])
    
    total_score = sum(s[1] for s in all_signals)
    
    result = {
        "ticker": ticker,
        "price": momentum.get("price") or technicals.get("price", 0),
        "technicals": technicals,
        "momentum": momentum,
        "options_structure": strike_structure,
        "all_signals": all_signals,
        "combined_score": total_score,
        "direction": "LONG" if total_score > 2 else "SHORT" if total_score < -2 else "NEUTRAL",
    }
    
    # Store in scan cache
    polygon_enhanced._scan_cache[ticker] = result
    
    return result
