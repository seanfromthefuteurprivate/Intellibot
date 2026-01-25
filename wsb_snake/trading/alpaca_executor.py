"""
Alpaca Paper Trading Executor

Executes real paper trades on Alpaca for SPY 0DTE options.
Max $1,000 per trade with margin utilization for maximum scalping efficiency.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

from wsb_snake.utils.logger import get_logger
from wsb_snake.notifications.telegram_bot import send_alert as send_telegram_alert

logger = get_logger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class PositionStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class AlpacaPosition:
    """Tracked position in Alpaca paper trading."""
    position_id: str
    symbol: str
    option_symbol: str
    side: str  # 'long' or 'short'
    trade_type: str  # 'CALLS' or 'PUTS'
    qty: int
    entry_price: float
    target_price: float
    stop_loss: float
    status: PositionStatus = PositionStatus.PENDING
    entry_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    alpaca_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None


class AlpacaExecutor:
    """
    Trading executor for Alpaca (Paper or Live).
    
    Features:
    - $1,000 max per trade
    - Options trading with margin
    - Automatic position monitoring
    - Real-time exit execution
    - Telegram notifications for fills
    
    Set ALPACA_LIVE_TRADING=true environment variable to switch to live trading.
    Default is paper trading (safe).
    """
    
    # Toggle between paper and live trading via environment variable
    LIVE_TRADING = os.environ.get("ALPACA_LIVE_TRADING", "false").lower() == "true"
    
    # API endpoints - switches based on LIVE_TRADING flag
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    BASE_URL = LIVE_URL if LIVE_TRADING else PAPER_URL
    DATA_URL = "https://data.alpaca.markets"
    
    MAX_POSITION_VALUE = 1000  # $1,000 max per trade
    MAX_CONCURRENT_POSITIONS = 3  # Max 3 positions at once
    
    def __init__(self):
        self.api_key = os.environ.get("ALPACA_API_KEY", "")
        self.api_secret = os.environ.get("ALPACA_SECRET_KEY", "")
        
        self.positions: Dict[str, AlpacaPosition] = {}
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        mode = "LIVE" if self.LIVE_TRADING else "Paper"
        logger.info(f"AlpacaExecutor initialized - {mode} trading mode")
        
        if self.LIVE_TRADING:
            logger.warning("⚠️ LIVE TRADING MODE ACTIVE - REAL MONEY AT RISK ⚠️")
        
    @property
    def headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }
    
    def get_account(self) -> Dict:
        """Get account info including buying power."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/v2/account",
                headers=self.headers,
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Failed to get account: {e}")
            return {}
    
    def get_buying_power(self) -> float:
        """Get available buying power."""
        account = self.get_account()
        return float(account.get("buying_power", 0))
    
    def get_options_positions(self) -> List[Dict]:
        """Get current options positions from Alpaca."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/v2/positions",
                headers=self.headers,
                timeout=10
            )
            resp.raise_for_status()
            positions = resp.json()
            return [p for p in positions if p.get("asset_class") == "us_option"]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def format_option_symbol(
        self,
        underlying: str,
        expiry: datetime,
        strike: float,
        option_type: str  # 'C' or 'P'
    ) -> str:
        """
        Format option symbol in OCC format.
        Example: SPY240125C00590000 (SPY Jan 25 2024 $590 Call)
        """
        date_str = expiry.strftime("%y%m%d")
        strike_str = f"{int(strike * 1000):08d}"
        return f"{underlying}{date_str}{option_type}{strike_str}"
    
    def get_option_quote(self, option_symbol: str) -> Dict:
        """Get latest quote for an option."""
        try:
            resp = requests.get(
                f"{self.DATA_URL}/v1beta1/options/quotes/latest",
                headers=self.headers,
                params={"symbols": option_symbol},
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("quotes", {}).get(option_symbol, {})
        except Exception as e:
            logger.debug(f"Option quote error: {e}")
            return {}
    
    def calculate_position_size(
        self,
        option_price: float,
        max_value: float = None
    ) -> int:
        """
        Calculate number of contracts to buy with max $1,000.
        Options are 100 shares per contract.
        HARD CAP: Never exceed max_value.
        """
        if max_value is None:
            max_value = self.MAX_POSITION_VALUE
        
        if option_price <= 0:
            return 0
        
        contract_cost = option_price * 100  # 100 shares per contract
        
        # Hard cap: if single contract > max_value, return 0 (skip trade)
        if contract_cost > max_value:
            logger.warning(f"Option price ${option_price:.2f} too expensive (${contract_cost:.2f}/contract > ${max_value})")
            return 0
        
        num_contracts = int(max_value / contract_cost)
        return num_contracts
    
    def place_option_order(
        self,
        underlying: str,
        expiry: datetime,
        strike: float,
        option_type: str,  # 'call' or 'put'
        side: str,  # 'buy' or 'sell'
        qty: int,
        order_type: str = "market",
        limit_price: float = None
    ) -> Optional[Dict]:
        """
        Place an option order on Alpaca paper trading.
        """
        try:
            option_symbol = self.format_option_symbol(
                underlying,
                expiry,
                strike,
                "C" if option_type.lower() == "call" else "P"
            )
            
            order_data = {
                "symbol": option_symbol,
                "qty": str(qty),
                "side": side,
                "type": order_type,
                "time_in_force": "day"
            }
            
            if order_type == "limit" and limit_price:
                order_data["limit_price"] = str(limit_price)
            
            logger.info(f"Placing order: {side} {qty}x {option_symbol}")
            
            resp = requests.post(
                f"{self.BASE_URL}/v2/orders",
                headers=self.headers,
                json=order_data,
                timeout=10
            )
            resp.raise_for_status()
            
            order = resp.json()
            logger.info(f"Order placed: {order.get('id')} status={order.get('status')}")
            
            return order
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"Order failed: {e.response.text if e.response else e}")
            return None
        except Exception as e:
            logger.error(f"Order error: {e}")
            return None
    
    def close_position(self, option_symbol: str) -> Optional[Dict]:
        """Close an existing position by selling."""
        try:
            resp = requests.delete(
                f"{self.BASE_URL}/v2/positions/{option_symbol}",
                headers=self.headers,
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Failed to close position {option_symbol}: {e.response.text if e.response else e}")
            send_telegram_alert(f"⚠️ Failed to close {option_symbol}: {str(e)[:100]}")
            return None
        except Exception as e:
            logger.error(f"Failed to close position {option_symbol}: {e}")
            send_telegram_alert(f"⚠️ Failed to close {option_symbol}: {str(e)[:100]}")
            return None
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get order status."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/v2/orders/{order_id}",
                headers=self.headers,
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug(f"Order status error: {e}")
            return None
    
    def execute_scalp_entry(
        self,
        underlying: str,
        direction: str,  # 'long' or 'short'
        entry_price: float,
        target_price: float,
        stop_loss: float,
        confidence: float,
        pattern: str
    ) -> Optional[AlpacaPosition]:
        """
        Execute a scalp trade entry.
        
        For 0DTE SPY options:
        - direction='long' -> buy CALLS
        - direction='short' -> buy PUTS (we don't short options, we buy puts)
        """
        with self._lock:
            if len([p for p in self.positions.values() if p.status == PositionStatus.OPEN]) >= self.MAX_CONCURRENT_POSITIONS:
                logger.warning("Max concurrent positions reached, skipping entry")
                return None
        
        trade_type = "CALLS" if direction == "long" else "PUTS"
        option_type = "call" if direction == "long" else "put"
        
        expiry = datetime.now()
        if expiry.hour >= 16:
            expiry = expiry + timedelta(days=1)
        while expiry.weekday() >= 5:
            expiry = expiry + timedelta(days=1)
        
        if direction == "long":
            strike = round(entry_price - 1, 0)
        else:
            strike = round(entry_price + 1, 0)
        
        option_symbol = self.format_option_symbol(
            underlying, expiry, strike,
            "C" if option_type == "call" else "P"
        )
        
        quote = self.get_option_quote(option_symbol)
        option_price = float(quote.get("ap", 0)) or 2.0
        
        qty = self.calculate_position_size(option_price)
        
        # Skip if position size is 0 (too expensive)
        if qty == 0:
            logger.warning(f"Skipping trade - option too expensive for $1K max")
            send_telegram_alert(f"⚠️ Trade skipped: {option_symbol} too expensive (>${self.MAX_POSITION_VALUE} per contract)")
            return None
        
        logger.info(f"Executing {trade_type} entry: {qty}x {option_symbol} @ ~${option_price:.2f}")
        
        order = self.place_option_order(
            underlying=underlying,
            expiry=expiry,
            strike=strike,
            option_type=option_type,
            side="buy",
            qty=qty,
            order_type="market"
        )
        
        if not order:
            logger.error("Failed to place entry order")
            return None
        
        position_id = f"{underlying}_{datetime.now().strftime('%H%M%S')}"
        
        position = AlpacaPosition(
            position_id=position_id,
            symbol=underlying,
            option_symbol=option_symbol,
            side=direction,
            trade_type=trade_type,
            qty=qty,
            entry_price=option_price,
            target_price=option_price * 1.20,
            stop_loss=option_price * 0.85,
            status=PositionStatus.PENDING,
            alpaca_order_id=order.get("id")
        )
        
        with self._lock:
            self.positions[position_id] = position
        
        message = f"""🦅 **ALPACA PAPER TRADE PLACED**

**{trade_type}** {underlying}
Strike: ${strike:.0f} | Exp: {expiry.strftime('%m/%d')}
Contracts: {qty}
Est. Entry: ${option_price:.2f}
Target: ${position.target_price:.2f} (+20%)
Stop: ${position.stop_loss:.2f} (-15%)

Pattern: {pattern}
Confidence: {confidence:.0f}%

Order ID: `{order.get('id', 'N/A')[:8]}...`
"""
        send_telegram_alert(message)
        
        return position
    
    def execute_exit(self, position: AlpacaPosition, reason: str, current_price: float):
        """Execute exit for a position."""
        logger.info(f"Executing exit for {position.option_symbol}: {reason}")
        
        result = self.close_position(position.option_symbol)
        
        # Only mark closed if close was successful
        if result is None:
            logger.warning(f"Exit order may have failed for {position.option_symbol}")
            return None
        
        position.exit_price = current_price
        position.exit_time = datetime.now()
        position.status = PositionStatus.CLOSED
        
        if position.entry_price > 0:
            position.pnl = (current_price - position.entry_price) * position.qty * 100
            position.pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
        
        self.total_trades += 1
        self.total_pnl += position.pnl
        if position.pnl > 0:
            self.winning_trades += 1
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        emoji = "💰" if position.pnl > 0 else "🛑"
        
        message = f"""{emoji} **ALPACA PAPER TRADE CLOSED**

**{position.trade_type}** {position.symbol}
Exit Reason: {reason}
Contracts: {position.qty}

Entry: ${position.entry_price:.2f}
Exit: ${current_price:.2f}
P&L: ${position.pnl:+.2f} ({position.pnl_pct:+.1f}%)

**Session Stats:**
Trades: {self.total_trades} | Win Rate: {win_rate:.0f}%
Total P&L: ${self.total_pnl:+.2f}
"""
        send_telegram_alert(message)
        
        return result
    
    def start_monitoring(self):
        """Start background position monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Alpaca position monitoring started")
    
    def stop_monitoring(self):
        """Stop position monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Alpaca position monitoring stopped")
    
    def _monitor_loop(self):
        """Monitor positions for exits."""
        while self.running:
            try:
                self._check_order_fills()
                self._check_exits()
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            
            time.sleep(5)
    
    def _check_order_fills(self):
        """Check if pending orders have filled."""
        with self._lock:
            for position in list(self.positions.values()):
                if position.status != PositionStatus.PENDING:
                    continue
                
                if not position.alpaca_order_id:
                    continue
                
                order = self.get_order_status(position.alpaca_order_id)
                if not order:
                    continue
                
                if order.get("status") == "filled":
                    position.status = PositionStatus.OPEN
                    position.entry_time = datetime.now()
                    position.entry_price = float(order.get("filled_avg_price", position.entry_price))
                    position.target_price = position.entry_price * 1.20
                    position.stop_loss = position.entry_price * 0.85
                    
                    logger.info(f"Order filled: {position.option_symbol} @ ${position.entry_price:.2f}")
                    
                    message = f"""✅ **ORDER FILLED**

**{position.trade_type}** {position.symbol}
Filled: {position.qty}x @ ${position.entry_price:.2f}
Total Cost: ${position.entry_price * position.qty * 100:.2f}

Target: ${position.target_price:.2f} (+20%)
Stop: ${position.stop_loss:.2f} (-15%)
"""
                    send_telegram_alert(message)
                
                elif order.get("status") in ["cancelled", "expired", "rejected"]:
                    position.status = PositionStatus.CANCELLED
                    logger.warning(f"Order {order.get('status')}: {position.option_symbol}")
                    
                    send_telegram_alert(f"""⚠️ **ORDER {order.get('status').upper()}**
                    
{position.trade_type} {position.symbol}
Symbol: {position.option_symbol}
Reason: Order was {order.get('status')}
""")
    
    def _check_exits(self):
        """Check open positions for exit conditions."""
        with self._lock:
            for position in list(self.positions.values()):
                if position.status != PositionStatus.OPEN:
                    continue
                
                quote = self.get_option_quote(position.option_symbol)
                if not quote:
                    continue
                
                current_price = float(quote.get("bp", 0)) or float(quote.get("ap", 0))
                if current_price <= 0:
                    continue
                
                if current_price >= position.target_price:
                    self.execute_exit(position, "TARGET HIT 🎯", current_price)
                
                elif current_price <= position.stop_loss:
                    self.execute_exit(position, "STOP LOSS", current_price)
                
                elif position.entry_time:
                    elapsed = (datetime.now() - position.entry_time).total_seconds() / 60
                    if elapsed >= 45:
                        self.execute_exit(position, "TIME DECAY (45min)", current_price)
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        open_positions = len([p for p in self.positions.values() if p.status == PositionStatus.OPEN])
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "open_positions": open_positions
        }


alpaca_executor = AlpacaExecutor()
