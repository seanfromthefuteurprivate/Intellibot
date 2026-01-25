"""
Ultra-Fast Scalp Data Collector

Combines all available Polygon data endpoints for surgical 0DTE scalping:
- 5-second bars for micro-momentum
- 15-second bars for trend confirmation  
- 1-minute bars for VWAP context
- Recent trades for order flow
- NBBO quotes for bid-ask pressure

Provides comprehensive data packet for AI chart analysis.
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScalpDataPacket:
    """Complete data packet for 0DTE scalping analysis."""
    ticker: str
    timestamp: datetime
    
    current_price: float = 0.0
    vwap: float = 0.0
    today_open: float = 0.0
    today_high: float = 0.0
    today_low: float = 0.0
    prev_close: float = 0.0
    
    bars_5s: List[Dict] = field(default_factory=list)
    bars_15s: List[Dict] = field(default_factory=list)
    bars_1m: List[Dict] = field(default_factory=list)
    bars_5m: List[Dict] = field(default_factory=list)
    
    recent_trades: List[Dict] = field(default_factory=list)
    nbbo_quotes: List[Dict] = field(default_factory=list)
    
    order_flow: Dict[str, Any] = field(default_factory=dict)
    momentum: Dict[str, Any] = field(default_factory=dict)
    technicals: Dict[str, Any] = field(default_factory=dict)
    
    is_valid: bool = False
    data_quality: str = "unknown"


class ScalpDataCollector:
    """
    Collects and combines all available data for 0DTE scalping.
    Optimized for SPY with sub-minute granularity.
    """
    
    def __init__(self):
        self.polygon = polygon_enhanced
        self._last_packet: Dict[str, ScalpDataPacket] = {}
        self._last_fetch_time: Dict[str, datetime] = {}
        self._min_fetch_interval = 10  # Minimum seconds between fetches (rate limit friendly)
        self._full_fetch_interval = 30  # Full data refresh interval
        self._last_full_fetch: Dict[str, datetime] = {}
        
        logger.info("ScalpDataCollector initialized - ultra-fast mode")
    
    def get_scalp_data(self, ticker: str = "SPY", force_refresh: bool = False) -> ScalpDataPacket:
        """
        Get comprehensive scalp data packet for a ticker.
        
        Args:
            ticker: Stock symbol (default SPY)
            force_refresh: Force fresh data fetch
            
        Returns:
            ScalpDataPacket with all available data
        """
        now = datetime.now()
        
        # Quick return for cached data within min interval
        if not force_refresh and ticker in self._last_fetch_time:
            elapsed = (now - self._last_fetch_time[ticker]).total_seconds()
            if elapsed < self._min_fetch_interval and ticker in self._last_packet:
                return self._last_packet[ticker]
        
        # Determine if this is a full refresh or lightweight update
        do_full_fetch = force_refresh
        if ticker not in self._last_full_fetch:
            do_full_fetch = True
        elif (now - self._last_full_fetch[ticker]).total_seconds() >= self._full_fetch_interval:
            do_full_fetch = True
        
        packet = ScalpDataPacket(ticker=ticker, timestamp=now)
        
        try:
            # ESSENTIAL DATA (always fetch - uses cache internally)
            snapshot = self.polygon.get_snapshot(ticker)
            if snapshot:
                packet.current_price = snapshot.get("price", 0)
                packet.vwap = snapshot.get("today_vwap", 0)
                packet.today_open = snapshot.get("today_open", 0)
                packet.today_high = snapshot.get("today_high", 0)
                packet.today_low = snapshot.get("today_low", 0)
                packet.prev_close = snapshot.get("prev_close", 0)
            
            # PRIMARY DATA - 1-minute bars (most important for pattern detection)
            packet.bars_1m = self.polygon.get_intraday_bars(ticker, timespan="minute", limit=60)
            
            if do_full_fetch:
                # FULL FETCH - all granular data (rate limit conscious)
                packet.bars_5s = self.polygon.get_ultra_fast_bars(ticker, seconds=5, limit=120)
                packet.bars_15s = self.polygon.get_ultra_fast_bars(ticker, seconds=15, limit=60)
                packet.bars_5m = self.polygon.get_intraday_bars(ticker, timespan="minute", multiplier=5, limit=30)
                packet.recent_trades = self.polygon.get_recent_trades(ticker, limit=100)
                packet.nbbo_quotes = self.polygon.get_nbbo_quotes(ticker, limit=50)
                packet.order_flow = self._compute_order_flow(packet.recent_trades, packet.nbbo_quotes, ticker)
                packet.momentum = self.polygon.get_momentum_signals(ticker)
                packet.technicals = self.polygon.get_full_technicals(ticker)
                self._last_full_fetch[ticker] = now
            else:
                # LIGHTWEIGHT UPDATE - reuse previous packet's supplementary data
                if ticker in self._last_packet:
                    prev = self._last_packet[ticker]
                    packet.bars_5s = prev.bars_5s
                    packet.bars_15s = prev.bars_15s
                    packet.bars_5m = prev.bars_5m
                    packet.recent_trades = prev.recent_trades
                    packet.nbbo_quotes = prev.nbbo_quotes
                    packet.order_flow = prev.order_flow
                    packet.momentum = prev.momentum
                    packet.technicals = prev.technicals
            
            packet.is_valid = self._validate_packet(packet)
            packet.data_quality = self._assess_quality(packet)
            
            self._last_packet[ticker] = packet
            self._last_fetch_time[ticker] = now
            
            logger.debug(
                f"ScalpData for {ticker}: price=${packet.current_price:.2f}, "
                f"5s_bars={len(packet.bars_5s)}, trades={len(packet.recent_trades)}, "
                f"quality={packet.data_quality}"
            )
            
        except Exception as e:
            logger.error(f"Error collecting scalp data for {ticker}: {e}")
            packet.is_valid = False
            packet.data_quality = "error"
        
        return packet
    
    def _validate_packet(self, packet: ScalpDataPacket) -> bool:
        """Check if packet has minimum required data."""
        if packet.current_price <= 0:
            return False
        if len(packet.bars_1m) < 10:
            return False
        return True
    
    def _assess_quality(self, packet: ScalpDataPacket) -> str:
        """Assess data quality level."""
        score = 0
        
        if packet.current_price > 0:
            score += 1
        if len(packet.bars_5s) >= 60:
            score += 2
        elif len(packet.bars_5s) >= 30:
            score += 1
        if len(packet.bars_15s) >= 30:
            score += 1
        if len(packet.bars_1m) >= 30:
            score += 1
        if len(packet.recent_trades) >= 50:
            score += 2
        elif len(packet.recent_trades) >= 20:
            score += 1
        if len(packet.nbbo_quotes) >= 20:
            score += 1
        if packet.order_flow.get("available"):
            score += 1
        
        if score >= 8:
            return "excellent"
        elif score >= 6:
            return "good"
        elif score >= 4:
            return "fair"
        else:
            return "poor"
    
    def get_multi_timeframe_bars(self, ticker: str = "SPY") -> Dict[str, List[Dict]]:
        """
        Get bars across multiple timeframes for comprehensive analysis.
        
        Returns:
            Dict with 5s, 15s, 1m, 5m bars
        """
        return {
            "5s": self.polygon.get_ultra_fast_bars(ticker, seconds=5, limit=120),
            "15s": self.polygon.get_ultra_fast_bars(ticker, seconds=15, limit=60),
            "1m": self.polygon.get_intraday_bars(ticker, timespan="minute", limit=60),
            "5m": self.polygon.get_intraday_bars(ticker, timespan="minute", multiplier=5, limit=30),
        }
    
    def get_order_flow_summary(self, ticker: str = "SPY") -> Dict[str, Any]:
        """
        Get order flow summary with trade and quote analysis.
        
        Returns:
            Order flow analysis dict
        """
        trades = self.polygon.get_recent_trades(ticker, limit=100)
        quotes = self.polygon.get_nbbo_quotes(ticker, limit=50)
        flow = self.polygon.analyze_order_flow(ticker)
        
        large_buys = []
        large_sells = []
        
        if trades and len(trades) >= 2:
            avg_price = sum(t["price"] for t in trades) / len(trades)
            for t in trades:
                if t["size"] >= 100:
                    if t["price"] >= avg_price:
                        large_buys.append(t)
                    else:
                        large_sells.append(t)
        
        current_spread = 0
        spread_widening = False
        if quotes and len(quotes) >= 2:
            current_spread = quotes[0].get("spread", 0)
            old_spread = quotes[-1].get("spread", 0)
            spread_widening = current_spread > old_spread * 1.5
        
        return {
            "flow_signal": flow.get("flow_signal", "NEUTRAL"),
            "flow_score": flow.get("flow_score", 0),
            "large_buys": len(large_buys),
            "large_sells": len(large_sells),
            "buy_sell_ratio": len(large_buys) / max(len(large_sells), 1),
            "bid_ask_imbalance": flow.get("bid_ask_imbalance", 0),
            "current_spread": current_spread,
            "spread_widening": spread_widening,
            "total_volume": flow.get("total_volume", 0),
        }
    
    def get_vwap_context(self, ticker: str = "SPY") -> Dict[str, Any]:
        """
        Get VWAP-specific context for scalping.
        
        Returns:
            VWAP analysis with bands and position
        """
        bars = self.polygon.get_intraday_bars(ticker, timespan="minute", limit=60)
        snapshot = self.polygon.get_snapshot(ticker)
        
        if not bars or not snapshot:
            return {"available": False}
        
        current_price = snapshot.get("price", 0)
        vwap = snapshot.get("today_vwap", 0)
        
        if not vwap or vwap == 0:
            cum_vol = 0
            cum_pv = 0
            for bar in reversed(bars):
                vol = bar.get("volume", 0)
                typical = (bar.get("high", 0) + bar.get("low", 0) + bar.get("close", 0)) / 3
                cum_vol += vol
                cum_pv += typical * vol
            vwap = cum_pv / cum_vol if cum_vol > 0 else current_price
        
        deviations = []
        cum_vol = 0
        cum_pv = 0
        for bar in reversed(bars):
            vol = bar.get("volume", 0)
            typical = (bar.get("high", 0) + bar.get("low", 0) + bar.get("close", 0)) / 3
            cum_vol += vol
            cum_pv += typical * vol
            bar_vwap = cum_pv / cum_vol if cum_vol > 0 else typical
            deviations.append((typical - bar_vwap) ** 2)
        
        std_dev = (sum(deviations) / len(deviations)) ** 0.5 if deviations else 0
        
        vwap_1_upper = vwap + std_dev
        vwap_1_lower = vwap - std_dev
        vwap_2_upper = vwap + 2 * std_dev
        vwap_2_lower = vwap - 2 * std_dev
        
        if current_price > vwap_2_upper:
            position = "EXTENDED_ABOVE"
        elif current_price > vwap_1_upper:
            position = "ABOVE_1STD"
        elif current_price > vwap:
            position = "ABOVE_VWAP"
        elif current_price > vwap_1_lower:
            position = "BELOW_VWAP"
        elif current_price > vwap_2_lower:
            position = "BELOW_1STD"
        else:
            position = "EXTENDED_BELOW"
        
        distance_pct = (current_price - vwap) / vwap * 100 if vwap else 0
        
        return {
            "available": True,
            "current_price": current_price,
            "vwap": vwap,
            "std_dev": std_dev,
            "vwap_1_upper": vwap_1_upper,
            "vwap_1_lower": vwap_1_lower,
            "vwap_2_upper": vwap_2_upper,
            "vwap_2_lower": vwap_2_lower,
            "position": position,
            "distance_pct": distance_pct,
        }


    def _compute_order_flow(
        self, 
        trades: List[Dict], 
        quotes: List[Dict],
        ticker: str
    ) -> Dict[str, Any]:
        """
        Compute order flow from already-fetched trades and quotes.
        Avoids duplicate API calls by reusing existing data.
        """
        if not trades:
            return {"available": False, "error": "No trade data"}
        
        total_volume = sum(t.get("size", 0) for t in trades)
        total_value = sum(t.get("price", 0) * t.get("size", 0) for t in trades)
        avg_price = total_value / total_volume if total_volume else 0
        
        large_trades = [t for t in trades if t.get("size", 0) >= 100]
        large_volume = sum(t.get("size", 0) for t in large_trades)
        
        prices = [t.get("price", 0) for t in trades if t.get("price", 0) > 0]
        price_direction = (prices[0] - prices[-1]) if len(prices) >= 2 else 0
        
        bid_ask_imbalance = 0
        avg_spread = 0
        if quotes:
            total_bid_size = sum(q.get("bid_size", 0) for q in quotes)
            total_ask_size = sum(q.get("ask_size", 0) for q in quotes)
            bid_ask_imbalance = (total_bid_size - total_ask_size) / max(total_bid_size + total_ask_size, 1)
            avg_spread = sum(q.get("spread", 0) for q in quotes) / len(quotes) if quotes else 0
        
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


scalp_data_collector = ScalpDataCollector()
