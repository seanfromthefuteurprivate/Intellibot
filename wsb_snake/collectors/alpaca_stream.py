"""
Alpaca Real-Time WebSocket Streaming

TRUE real-time data with sub-second latency for ruthless 0DTE scalping.
Provides:
- Real-time trades (individual executions)
- Real-time quotes (bid/ask updates)
- Real-time minute bars
- Trading halts and LULD bands for risk management
"""

import os
import json
import asyncio
import threading
import websocket
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from collections import deque
import logging

logger = logging.getLogger(__name__)


class AlpacaStream:
    """
    Real-time streaming from Alpaca for surgical 0DTE precision.
    
    Features:
    - Sub-second trade data
    - Real-time bid/ask pressure
    - Auto minute bar generation
    - Trading halt detection
    - LULD band tracking
    """
    
    STREAM_URL = "wss://stream.data.alpaca.markets/v2/iex"
    
    def __init__(self):
        self.api_key = os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
        
        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.authenticated = False
        self.thread: Optional[threading.Thread] = None
        
        self.subscribed_symbols: set = set()
        
        self.trade_callbacks: List[Callable] = []
        self.quote_callbacks: List[Callable] = []
        self.bar_callbacks: List[Callable] = []
        self.halt_callbacks: List[Callable] = []
        self.luld_callbacks: List[Callable] = []
        
        self._recent_trades: Dict[str, deque] = {}
        self._recent_quotes: Dict[str, deque] = {}
        self._current_bars: Dict[str, Dict] = {}
        self._halted_symbols: set = set()
        self._luld_bands: Dict[str, Dict] = {}
        
        self._trade_counts: Dict[str, int] = {}
        self._volume_totals: Dict[str, int] = {}
        
        if self.api_key and self.api_secret:
            logger.info("AlpacaStream initialized - real-time mode ready")
        else:
            logger.warning("Alpaca credentials not set - streaming disabled")
    
    def on_trade(self, callback: Callable):
        """Register callback for real-time trades."""
        self.trade_callbacks.append(callback)
    
    def on_quote(self, callback: Callable):
        """Register callback for real-time quotes."""
        self.quote_callbacks.append(callback)
    
    def on_bar(self, callback: Callable):
        """Register callback for minute bars."""
        self.bar_callbacks.append(callback)
    
    def on_halt(self, callback: Callable):
        """Register callback for trading halts."""
        self.halt_callbacks.append(callback)
    
    def on_luld(self, callback: Callable):
        """Register callback for LULD band updates."""
        self.luld_callbacks.append(callback)
    
    def subscribe(self, symbols: List[str], trades: bool = True, quotes: bool = True, bars: bool = True):
        """Subscribe to real-time data for symbols."""
        for symbol in symbols:
            self.subscribed_symbols.add(symbol.upper())
            if symbol not in self._recent_trades:
                self._recent_trades[symbol] = deque(maxlen=500)
            if symbol not in self._recent_quotes:
                self._recent_quotes[symbol] = deque(maxlen=200)
        
        if self.ws and self.authenticated:
            self._send_subscription(symbols, trades, quotes, bars)
    
    def _send_subscription(self, symbols: List[str], trades: bool, quotes: bool, bars: bool):
        """Send subscription message to WebSocket."""
        msg = {"action": "subscribe"}
        
        if trades:
            msg["trades"] = symbols
        if quotes:
            msg["quotes"] = symbols
        if bars:
            msg["bars"] = symbols
        
        msg["statuses"] = symbols
        msg["lulds"] = symbols
        
        try:
            self.ws.send(json.dumps(msg))
            logger.info(f"Subscribed to {len(symbols)} symbols: {symbols[:5]}...")
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
    
    def start(self):
        """Start the streaming connection in background thread."""
        if not self.api_key or not self.api_secret:
            logger.warning("Cannot start stream - no credentials")
            return
        
        if self.running:
            logger.warning("Stream already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_forever, daemon=True)
        self.thread.start()
        logger.info("AlpacaStream started in background")
    
    def stop(self):
        """Stop the streaming connection."""
        self.running = False
        if self.ws:
            self.ws.close()
        logger.info("AlpacaStream stopped")
    
    def _run_forever(self):
        """Run WebSocket connection with auto-reconnect."""
        while self.running:
            try:
                self._connect()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.running:
                    import time
                    time.sleep(5)
    
    def _connect(self):
        """Establish WebSocket connection."""
        self.ws = websocket.WebSocketApp(
            self.STREAM_URL,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        self.ws.run_forever()
    
    def _on_open(self, ws):
        """Handle connection open."""
        logger.info("AlpacaStream connected, authenticating...")
        auth_msg = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret
        }
        ws.send(json.dumps(auth_msg))
    
    def _on_message(self, ws, message):
        """Process incoming messages."""
        try:
            data = json.loads(message)
            
            for msg in data:
                msg_type = msg.get("T", "")
                
                if msg_type == "success":
                    if msg.get("msg") == "authenticated":
                        self.authenticated = True
                        logger.info("AlpacaStream authenticated!")
                        if self.subscribed_symbols:
                            self._send_subscription(
                                list(self.subscribed_symbols),
                                trades=True, quotes=True, bars=True
                            )
                
                elif msg_type == "t":
                    self._handle_trade(msg)
                
                elif msg_type == "q":
                    self._handle_quote(msg)
                
                elif msg_type == "b":
                    self._handle_bar(msg)
                
                elif msg_type == "s":
                    self._handle_status(msg)
                
                elif msg_type == "l":
                    self._handle_luld(msg)
                
                elif msg_type == "error":
                    logger.error(f"AlpacaStream error: {msg}")
                
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    def _handle_trade(self, msg: Dict):
        """Process real-time trade."""
        symbol = msg.get("S", "")
        
        trade = {
            "symbol": symbol,
            "price": msg.get("p", 0),
            "size": msg.get("s", 0),
            "timestamp": msg.get("t", ""),
            "exchange": msg.get("x", ""),
            "conditions": msg.get("c", []),
            "tape": msg.get("z", ""),
        }
        
        if symbol in self._recent_trades:
            self._recent_trades[symbol].append(trade)
        
        self._trade_counts[symbol] = self._trade_counts.get(symbol, 0) + 1
        self._volume_totals[symbol] = self._volume_totals.get(symbol, 0) + trade["size"]
        
        for callback in self.trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")
    
    def _handle_quote(self, msg: Dict):
        """Process real-time quote."""
        symbol = msg.get("S", "")
        
        quote = {
            "symbol": symbol,
            "bid": msg.get("bp", 0),
            "bid_size": msg.get("bs", 0),
            "ask": msg.get("ap", 0),
            "ask_size": msg.get("as", 0),
            "timestamp": msg.get("t", ""),
            "spread": (msg.get("ap", 0) - msg.get("bp", 0)) if msg.get("bp") and msg.get("ap") else 0,
        }
        
        if symbol in self._recent_quotes:
            self._recent_quotes[symbol].append(quote)
        
        for callback in self.quote_callbacks:
            try:
                callback(quote)
            except Exception as e:
                logger.error(f"Quote callback error: {e}")
    
    def _handle_bar(self, msg: Dict):
        """Process minute bar."""
        symbol = msg.get("S", "")
        
        bar = {
            "symbol": symbol,
            "open": msg.get("o", 0),
            "high": msg.get("h", 0),
            "low": msg.get("l", 0),
            "close": msg.get("c", 0),
            "volume": msg.get("v", 0),
            "vwap": msg.get("vw", 0),
            "trade_count": msg.get("n", 0),
            "timestamp": msg.get("t", ""),
        }
        
        self._current_bars[symbol] = bar
        
        for callback in self.bar_callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"Bar callback error: {e}")
    
    def _handle_status(self, msg: Dict):
        """Process trading status (halts)."""
        symbol = msg.get("S", "")
        status_code = msg.get("sc", "")
        status_message = msg.get("sm", "")
        reason_code = msg.get("rc", "")
        reason_message = msg.get("rm", "")
        
        halt_info = {
            "symbol": symbol,
            "status_code": status_code,
            "status_message": status_message,
            "reason_code": reason_code,
            "reason_message": reason_message,
            "timestamp": msg.get("t", ""),
            "is_halted": status_code != "T",
        }
        
        if halt_info["is_halted"]:
            self._halted_symbols.add(symbol)
            logger.warning(f"TRADING HALT: {symbol} - {status_message}")
        else:
            self._halted_symbols.discard(symbol)
            logger.info(f"Trading resumed: {symbol}")
        
        for callback in self.halt_callbacks:
            try:
                callback(halt_info)
            except Exception as e:
                logger.error(f"Halt callback error: {e}")
    
    def _handle_luld(self, msg: Dict):
        """Process LULD band update."""
        symbol = msg.get("S", "")
        
        luld = {
            "symbol": symbol,
            "limit_up": msg.get("u", 0),
            "limit_down": msg.get("d", 0),
            "indicator": msg.get("i", ""),
            "timestamp": msg.get("t", ""),
        }
        
        self._luld_bands[symbol] = luld
        
        for callback in self.luld_callbacks:
            try:
                callback(luld)
            except Exception as e:
                logger.error(f"LULD callback error: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"AlpacaStream WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        self.authenticated = False
        logger.info(f"AlpacaStream closed: {close_status_code} - {close_msg}")
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades from buffer."""
        if symbol in self._recent_trades:
            trades = list(self._recent_trades[symbol])
            return trades[-limit:]
        return []
    
    def get_recent_quotes(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get recent quotes from buffer."""
        if symbol in self._recent_quotes:
            quotes = list(self._recent_quotes[symbol])
            return quotes[-limit:]
        return []
    
    def get_current_bar(self, symbol: str) -> Optional[Dict]:
        """Get current minute bar."""
        return self._current_bars.get(symbol)
    
    def is_halted(self, symbol: str) -> bool:
        """Check if symbol is currently halted."""
        return symbol in self._halted_symbols
    
    def get_luld_bands(self, symbol: str) -> Optional[Dict]:
        """Get current LULD bands for symbol."""
        return self._luld_bands.get(symbol)
    
    def get_order_flow_summary(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time order flow summary from streaming data.
        More accurate than polling since it captures every trade.
        """
        trades = self.get_recent_trades(symbol, limit=500)
        quotes = self.get_recent_quotes(symbol, limit=100)
        
        if not trades:
            return {"available": False}
        
        total_volume = sum(t["size"] for t in trades)
        total_value = sum(t["price"] * t["size"] for t in trades)
        avg_price = total_value / total_volume if total_volume else 0
        
        large_trades = [t for t in trades if t["size"] >= 100]
        large_volume = sum(t["size"] for t in large_trades)
        
        first_price = trades[0]["price"] if trades else 0
        last_price = trades[-1]["price"] if trades else 0
        price_direction = last_price - first_price
        
        buy_pressure = 0
        sell_pressure = 0
        
        for trade in trades:
            if quotes:
                mid = (quotes[-1]["bid"] + quotes[-1]["ask"]) / 2 if quotes[-1]["bid"] and quotes[-1]["ask"] else trade["price"]
                if trade["price"] >= mid:
                    buy_pressure += trade["size"]
                else:
                    sell_pressure += trade["size"]
        
        total_pressure = buy_pressure + sell_pressure
        buy_ratio = buy_pressure / total_pressure if total_pressure else 0.5
        
        bid_ask_imbalance = 0
        if quotes:
            total_bid = sum(q["bid_size"] for q in quotes[-20:])
            total_ask = sum(q["ask_size"] for q in quotes[-20:])
            bid_ask_imbalance = (total_bid - total_ask) / max(total_bid + total_ask, 1)
        
        if buy_ratio > 0.6 and price_direction > 0:
            flow_signal = "STRONG_BUY"
            flow_score = 2
        elif buy_ratio > 0.55:
            flow_signal = "BUY"
            flow_score = 1
        elif buy_ratio < 0.4 and price_direction < 0:
            flow_signal = "STRONG_SELL"
            flow_score = -2
        elif buy_ratio < 0.45:
            flow_signal = "SELL"
            flow_score = -1
        else:
            flow_signal = "NEUTRAL"
            flow_score = 0
        
        return {
            "available": True,
            "symbol": symbol,
            "total_trades": len(trades),
            "total_volume": total_volume,
            "avg_price": avg_price,
            "large_trades": len(large_trades),
            "large_volume_pct": large_volume / max(total_volume, 1) * 100,
            "price_direction": price_direction,
            "buy_ratio": buy_ratio,
            "bid_ask_imbalance": bid_ask_imbalance,
            "flow_signal": flow_signal,
            "flow_score": flow_score,
            "is_halted": self.is_halted(symbol),
            "luld_bands": self.get_luld_bands(symbol),
        }


alpaca_stream = AlpacaStream()
