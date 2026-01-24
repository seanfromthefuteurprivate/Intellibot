"""
Finnhub WebSocket Collector - Real-Time Streaming

Free tier: Real-time trades for US stocks via WebSocket
Replaces 30-second polling with true streaming updates.
"""

import os
import json
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from collections import defaultdict

from wsb_snake.utils.logger import log

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    log.warning("websocket-client not installed - run: pip install websocket-client")


class FinnhubWebSocket:
    """
    Real-time streaming via Finnhub WebSocket.
    Free tier includes real-time trades for US stocks.
    
    Features:
    - True real-time streaming (sub-second latency)
    - Automatic reconnection
    - Trade aggregation for volume spikes
    - Callback system for signal integration
    """
    
    WEBSOCKET_URL = "wss://ws.finnhub.io"
    
    def __init__(self):
        self.api_key = os.environ.get("FINNHUB_API_KEY", "")
        self.ws: Optional[websocket.WebSocketApp] = None
        self.subscribed_symbols: set = set()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self.trade_callbacks: List[Callable] = []
        self.volume_callbacks: List[Callable] = []
        
        self.trade_buffer: Dict[str, List[Dict]] = defaultdict(list)
        self.volume_window: Dict[str, int] = defaultdict(int)
        self.last_prices: Dict[str, float] = {}
        self.last_update: Dict[str, datetime] = {}
        
        self.buffer_window = 1.0
        self.last_flush = time.time()
        
        self.reconnect_delay = 5
        self.max_reconnect_delay = 60
        self.current_reconnect_delay = self.reconnect_delay
        
        if not self.api_key:
            log.warning("FINNHUB_API_KEY not set - WebSocket streaming disabled")
        elif not WEBSOCKET_AVAILABLE:
            log.warning("websocket-client not installed")
        else:
            log.info("Finnhub WebSocket collector initialized")
    
    def on_trade(self, callback: Callable[[str, float, int, datetime], None]):
        """Register callback for trade events: (symbol, price, volume, timestamp)"""
        self.trade_callbacks.append(callback)
    
    def on_volume_spike(self, callback: Callable[[str, int, float], None]):
        """Register callback for volume spikes: (symbol, volume, price_change_pct)"""
        self.volume_callbacks.append(callback)
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if data.get("type") == "trade":
                trades = data.get("data", [])
                for trade in trades:
                    symbol = trade.get("s", "")
                    price = float(trade.get("p", 0))
                    volume = int(trade.get("v", 0))
                    timestamp = datetime.fromtimestamp(trade.get("t", 0) / 1000)
                    
                    self.trade_buffer[symbol].append({
                        "price": price,
                        "volume": volume,
                        "timestamp": timestamp
                    })
                    
                    self.volume_window[symbol] += volume
                    self.last_prices[symbol] = price
                    self.last_update[symbol] = timestamp
                    
                    for cb in self.trade_callbacks:
                        try:
                            cb(symbol, price, volume, timestamp)
                        except Exception as e:
                            log.debug(f"Trade callback error: {e}")
                
                if time.time() - self.last_flush > self.buffer_window:
                    self._flush_volume_spikes()
                    
            elif data.get("type") == "ping":
                pass
                
        except Exception as e:
            log.debug(f"WebSocket message error: {e}")
    
    def _flush_volume_spikes(self):
        """Check for volume spikes and trigger callbacks"""
        self.last_flush = time.time()
        
        for symbol, volume in self.volume_window.items():
            if volume > 10000:
                trades = self.trade_buffer.get(symbol, [])
                if len(trades) >= 2:
                    first_price = trades[0]["price"]
                    last_price = trades[-1]["price"]
                    if first_price > 0:
                        pct_change = ((last_price - first_price) / first_price) * 100
                    else:
                        pct_change = 0.0
                    
                    for cb in self.volume_callbacks:
                        try:
                            cb(symbol, volume, pct_change)
                        except Exception as e:
                            log.debug(f"Volume callback error: {e}")
        
        self.volume_window.clear()
        self.trade_buffer.clear()
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        log.warning(f"Finnhub WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        log.info(f"Finnhub WebSocket closed: {close_status_code} - {close_msg}")
        
        if self.running:
            log.info(f"Reconnecting in {self.current_reconnect_delay}s...")
            time.sleep(self.current_reconnect_delay)
            self.current_reconnect_delay = min(
                self.current_reconnect_delay * 2,
                self.max_reconnect_delay
            )
            self._connect()
    
    def _on_open(self, ws):
        """Handle WebSocket open - subscribe to symbols"""
        log.info("Finnhub WebSocket connected")
        self.current_reconnect_delay = self.reconnect_delay
        
        for symbol in self.subscribed_symbols:
            subscribe_msg = json.dumps({"type": "subscribe", "symbol": symbol})
            ws.send(subscribe_msg)
            log.debug(f"Subscribed to {symbol}")
    
    def _connect(self):
        """Establish WebSocket connection"""
        if not self.api_key or not WEBSOCKET_AVAILABLE:
            return
        
        url = f"{self.WEBSOCKET_URL}?token={self.api_key}"
        
        self.ws = websocket.WebSocketApp(
            url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        self.ws.run_forever()
    
    def subscribe(self, symbols: List[str]):
        """Subscribe to symbols for real-time trades"""
        for symbol in symbols:
            self.subscribed_symbols.add(symbol)
            
            if self.ws and self.running:
                try:
                    subscribe_msg = json.dumps({"type": "subscribe", "symbol": symbol})
                    self.ws.send(subscribe_msg)
                except Exception as e:
                    log.debug(f"Subscribe error for {symbol}: {e}")
    
    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        for symbol in symbols:
            self.subscribed_symbols.discard(symbol)
            
            if self.ws and self.running:
                try:
                    unsubscribe_msg = json.dumps({"type": "unsubscribe", "symbol": symbol})
                    self.ws.send(unsubscribe_msg)
                except Exception as e:
                    log.debug(f"Unsubscribe error for {symbol}: {e}")
    
    def start(self, symbols: Optional[List[str]] = None):
        """Start WebSocket streaming in background thread"""
        if not self.api_key:
            log.warning("Cannot start WebSocket - no API key")
            return
        
        if not WEBSOCKET_AVAILABLE:
            log.warning("Cannot start WebSocket - websocket-client not installed")
            return
        
        if self.running:
            log.warning("WebSocket already running")
            return
        
        if symbols:
            self.subscribe(symbols)
        
        self.running = True
        self.thread = threading.Thread(target=self._connect, daemon=True)
        self.thread.start()
        log.info(f"Finnhub WebSocket started - streaming {len(self.subscribed_symbols)} symbols")
    
    def stop(self):
        """Stop WebSocket streaming"""
        self.running = False
        if self.ws:
            self.ws.close()
        log.info("Finnhub WebSocket stopped")
    
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last known price for symbol"""
        return self.last_prices.get(symbol)
    
    def get_streaming_status(self) -> Dict:
        """Get current streaming status"""
        return {
            "running": self.running,
            "connected": self.ws is not None and self.running,
            "subscribed_symbols": list(self.subscribed_symbols),
            "last_prices": dict(self.last_prices),
            "api_key_set": bool(self.api_key),
            "websocket_available": WEBSOCKET_AVAILABLE,
        }


finnhub_websocket = FinnhubWebSocket()
