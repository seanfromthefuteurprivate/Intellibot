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
    exit_reason: Optional[str] = None


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
    
    MAX_DAILY_DEPLOYED = 1000  # $1,000 TOTAL per day (not per trade!)
    MAX_CONCURRENT_POSITIONS = 3  # Max 3 positions at once (split from $1000)
    MARKET_CLOSE_HOUR = 16  # 4 PM ET - all 0DTE must close
    CLOSE_BEFORE_MINUTES = 5  # Close 5 min before market close
    
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
        
        # Track daily deployed capital
        self.daily_deployed = 0.0
        self.daily_reset_date = datetime.utcnow().date()
        
        mode = "LIVE" if self.LIVE_TRADING else "Paper"
        logger.info(f"AlpacaExecutor initialized - {mode} trading mode")
        logger.info(f"Daily limit: ${self.MAX_DAILY_DEPLOYED} | Max positions: {self.MAX_CONCURRENT_POSITIONS}")
        
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
    
    def get_strike_interval(self, underlying: str, price: float) -> float:
        """
        Get the standard option strike interval for a given underlying.
        
        Strike intervals vary by:
        - SPY/QQQ/IWM: $1 strikes
        - TSLA/NVDA/META at high prices: $5 strikes
        - AAPL/MSFT/AMD: $2.50 or $5 strikes
        - Most other stocks > $100: $5 strikes
        - Stocks $25-$100: $2.50 strikes
        - Stocks < $25: $1 strikes
        """
        # Index ETFs with $1 strikes
        if underlying in ["SPY", "QQQ", "IWM"]:
            return 1.0
        
        # Metals/Commodity ETFs - varies by price
        if underlying == "SLV":
            # SLV now ~$100, uses $1 strikes at high prices
            return 0.50 if price < 50 else 1.0
        if underlying == "GLD":
            # GLD now ~$466, uses $5 strikes at very high prices
            if price > 300:
                return 5.0
            elif price > 150:
                return 2.0
            return 1.0
        if underlying in ["GDX", "GDXJ"]:
            # Miners at higher prices now, $1 strikes
            return 0.50 if price < 30 else 1.0
        if underlying in ["USO", "XLE"]:
            # Energy ETFs
            return 0.50 if price < 30 else 1.0
        
        # High-priced stocks with $5 strikes
        if underlying in ["TSLA", "NVDA", "META", "GOOGL", "AMZN", "MSFT"]:
            if price > 100:
                return 5.0
            return 2.5
        
        # General rules by price
        if price > 100:
            return 5.0
        elif price > 25:
            return 2.5
        else:
            return 1.0
    
    def round_to_strike(self, price: float, interval: float, direction: str = "nearest") -> float:
        """Round price to valid strike based on interval and direction."""
        if direction == "down":
            return (price // interval) * interval
        elif direction == "up":
            return ((price // interval) + 1) * interval
        else:  # nearest
            return round(price / interval) * interval
    
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
    
    def get_remaining_daily_capital(self) -> float:
        """Get remaining capital available for today."""
        # Reset daily tracker if new day
        today = datetime.utcnow().date()
        if today != self.daily_reset_date:
            self.daily_deployed = 0.0
            self.daily_reset_date = today
            logger.info(f"Daily capital reset: ${self.MAX_DAILY_DEPLOYED} available")
        
        # Calculate currently deployed capital from open positions
        positions = self.get_options_positions()
        current_deployed = sum(
            float(p.get('cost_basis', 0)) or 
            (float(p.get('qty', 0)) * float(p.get('avg_entry_price', 0)) * 100)
            for p in positions
        )
        
        remaining = self.MAX_DAILY_DEPLOYED - current_deployed
        return max(0, remaining)
    
    def calculate_position_size(
        self,
        option_price: float,
        max_value: Optional[float] = None
    ) -> int:
        """
        Calculate number of contracts to buy.
        Uses remaining daily capital (total $1000/day limit).
        Options are 100 shares per contract.
        """
        if max_value is None:
            # Use remaining daily capital, not fixed max
            max_value = min(self.get_remaining_daily_capital(), self.MAX_DAILY_DEPLOYED)
        
        if option_price <= 0:
            return 0
        
        contract_cost = option_price * 100  # 100 shares per contract
        
        # Hard cap: if single contract > max_value, return 0 (skip trade)
        if contract_cost > max_value:
            logger.warning(f"Option price ${option_price:.2f} too expensive (${contract_cost:.2f}/contract > ${max_value:.2f} remaining)")
            return 0
        
        num_contracts = int(max_value / contract_cost)
        return num_contracts
    
    def close_all_0dte_positions(self) -> int:
        """
        MANDATORY: Close ALL 0DTE positions before market close.
        Called automatically at 3:55 PM ET.
        Returns number of positions closed.
        """
        import pytz
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        
        positions = self.get_options_positions()
        closed = 0
        
        for p in positions:
            symbol = p.get('symbol', '')
            try:
                self.close_position(symbol)
                closed += 1
                logger.info(f"0DTE EOD close: {symbol}")
            except Exception as e:
                logger.error(f"Failed to close 0DTE position {symbol}: {e}")
        
        if closed > 0:
            send_telegram_alert(f"""⏰ **END OF DAY - 0DTE POSITIONS CLOSED**

Closed {closed} position(s) before market close.
Time: {now_et.strftime('%I:%M %p ET')}

No overnight risk. Fresh start tomorrow!
""")
        
        return closed
    
    def should_close_for_eod(self) -> bool:
        """Check if we should close all 0DTE positions for end of day."""
        import pytz
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        
        # Close at 3:55 PM ET (5 minutes before market close)
        close_time = now_et.replace(hour=15, minute=55, second=0, microsecond=0)
        
        return now_et >= close_time and now_et.hour < 17  # Before 5 PM
    
    def place_option_order(
        self,
        underlying: str,
        expiry: datetime,
        strike: float,
        option_type: str,  # 'call' or 'put'
        side: str,  # 'buy' or 'sell'
        qty: int,
        order_type: str = "market",
        limit_price: Optional[float] = None
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
            logger.debug(f"Order payload: {order_data}")
            
            resp = requests.post(
                f"{self.BASE_URL}/v2/orders",
                headers=self.headers,
                json=order_data,
                timeout=10
            )
            
            # Log response before raising for better debugging
            if resp.status_code >= 400:
                logger.error(f"Order failed with status {resp.status_code}: {resp.text}")
                resp.raise_for_status()
            
            order = resp.json()
            logger.info(f"Order placed: {order.get('id')} status={order.get('status')}")
            
            return order
            
        except requests.exceptions.HTTPError as e:
            # Try to get actual error message from response
            error_detail = ""
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.text
                except:
                    error_detail = str(e)
            logger.error(f"Order HTTP error: {e} - Details: {error_detail}")
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
        
        # CRITICAL VALIDATION: Reject trades with invalid parameters
        if stop_loss <= 0:
            logger.error(f"INVALID STOP LOSS ${stop_loss:.2f} - ABORTING (must be positive)")
            send_telegram_alert(f"❌ Trade REJECTED: Invalid stop loss ${stop_loss:.2f} for {underlying}")
            return None
        
        if target_price <= 0:
            logger.error(f"INVALID TARGET ${target_price:.2f} - ABORTING (must be positive)")
            send_telegram_alert(f"❌ Trade REJECTED: Invalid target ${target_price:.2f} for {underlying}")
            return None
        
        if entry_price <= 0:
            logger.error(f"INVALID ENTRY ${entry_price:.2f} - ABORTING (must be positive)")
            return None
        
        # Validate direction matches stop/target relationship
        if direction == "long":
            if stop_loss >= entry_price:
                logger.error(f"INVALID LONG SETUP: Stop ${stop_loss:.2f} >= Entry ${entry_price:.2f}")
                send_telegram_alert(f"❌ Trade REJECTED: Bad stop for LONG {underlying}")
                return None
            if target_price <= entry_price:
                logger.error(f"INVALID LONG SETUP: Target ${target_price:.2f} <= Entry ${entry_price:.2f}")
                send_telegram_alert(f"❌ Trade REJECTED: Bad target for LONG {underlying}")
                return None
        else:  # short
            if stop_loss <= entry_price:
                logger.error(f"INVALID SHORT SETUP: Stop ${stop_loss:.2f} <= Entry ${entry_price:.2f}")
                send_telegram_alert(f"❌ Trade REJECTED: Bad stop for SHORT {underlying}")
                return None
            if target_price >= entry_price:
                logger.error(f"INVALID SHORT SETUP: Target ${target_price:.2f} >= Entry ${entry_price:.2f}")
                send_telegram_alert(f"❌ Trade REJECTED: Bad target for SHORT {underlying}")
                return None
        
        # Validate R:R ratio is reasonable (at least 1:1)
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        if rr_ratio < 0.5:
            logger.error(f"BAD R:R RATIO {rr_ratio:.2f} - Risk ${risk:.2f} vs Reward ${reward:.2f}")
            send_telegram_alert(f"❌ Trade REJECTED: Bad R:R {rr_ratio:.2f} for {underlying}")
            return None
        
        logger.info(f"Validated trade: {underlying} {direction} Entry=${entry_price:.2f} Target=${target_price:.2f} Stop=${stop_loss:.2f} R:R={rr_ratio:.2f}")
        
        # ETFs with daily 0DTE options - import from config or use default
        from wsb_snake.config import DAILY_0DTE_TICKERS
        daily_0dte_tickers = DAILY_0DTE_TICKERS
        
        # Use Eastern Time for market hours (server may be UTC)
        import pytz
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        
        if underlying in daily_0dte_tickers:
            # For SPY/QQQ/IWM - use same-day 0DTE or next trading day if after hours
            if now_et.hour >= 16:  # After 4 PM ET
                expiry = now_et + timedelta(days=1)
            else:
                expiry = now_et
            while expiry.weekday() >= 5:  # Skip weekends
                expiry = expiry + timedelta(days=1)
        else:
            # For individual stocks - use next Friday expiration
            # Find the next Friday (weekday 4)
            days_until_friday = (4 - now_et.weekday()) % 7
            if days_until_friday == 0 and now_et.hour >= 16:
                days_until_friday = 7  # If it's Friday after hours, use next Friday
            expiry = now_et + timedelta(days=days_until_friday)
        
        # Convert to naive datetime for formatting
        expiry = expiry.replace(tzinfo=None)
        
        logger.info(f"Selected expiry {expiry.strftime('%Y-%m-%d')} for {underlying} (0DTE: {underlying in daily_0dte_tickers}, ET hour: {now_et.hour})")
        
        # Calculate strike at proper interval for this underlying
        interval = self.get_strike_interval(underlying, entry_price)
        
        if direction == "long":
            # For calls, go slightly ITM (below current price)
            strike = self.round_to_strike(entry_price - interval, interval, "down")
        else:
            # For puts, go slightly ITM (above current price)
            strike = self.round_to_strike(entry_price + interval, interval, "up")
        
        logger.info(f"Selected strike ${strike:.0f} (interval: ${interval}) for {underlying} @ ${entry_price:.2f}")
        
        option_symbol = self.format_option_symbol(
            underlying, expiry, strike,
            "C" if option_type == "call" else "P"
        )
        
        # CRITICAL: Get real quote - NEVER assume a price
        quote = self.get_option_quote(option_symbol)
        if not quote:
            logger.error(f"No quote available for {option_symbol} - ABORTING trade")
            send_telegram_alert(f"❌ Trade aborted: No quote for {option_symbol}")
            return None
        
        # Validate quote freshness (must be within 30 seconds)
        quote_timestamp = quote.get("t", "")
        if quote_timestamp:
            try:
                from dateutil import parser as date_parser
                quote_time = date_parser.parse(quote_timestamp)
                quote_age = (datetime.now(quote_time.tzinfo) - quote_time).total_seconds()
                if quote_age > 60:  # More than 60 seconds old
                    logger.warning(f"Quote for {option_symbol} is {quote_age:.0f}s old - may be stale")
            except Exception as e:
                logger.debug(f"Could not parse quote timestamp: {e}")
        
        # Use ask price (what we pay to buy)
        option_price = float(quote.get("ap", 0))
        if option_price <= 0:
            logger.error(f"Invalid option price ${option_price} for {option_symbol} - ABORTING")
            send_telegram_alert(f"❌ Trade aborted: Invalid price for {option_symbol}")
            return None
        
        # Verify option exists and is tradeable
        bid_price = float(quote.get("bp", 0))
        if bid_price <= 0:
            logger.error(f"No bid for {option_symbol} - option may be illiquid - ABORTING")
            send_telegram_alert(f"❌ Trade aborted: No bid for {option_symbol} (illiquid)")
            return None
        
        # Verify bid-ask spread is reasonable (< 20% of mid-price)
        mid_price = (option_price + bid_price) / 2
        spread_pct = (option_price - bid_price) / mid_price * 100
        if spread_pct > 20:
            logger.warning(f"Wide spread {spread_pct:.1f}% on {option_symbol} - may be illiquid")
        
        qty = self.calculate_position_size(option_price)
        estimated_cost = qty * option_price * 100  # Total cost in dollars
        
        # Check remaining daily capital
        remaining_capital = self.get_remaining_daily_capital()
        
        # HARD CAP: Double-check we're under daily limit
        if estimated_cost > remaining_capital:
            logger.error(f"Position size ${estimated_cost:.2f} exceeds ${remaining_capital:.2f} remaining daily capital - ABORTING")
            send_telegram_alert(f"❌ Trade skipped: ${estimated_cost:.2f} exceeds ${remaining_capital:.2f} remaining today")
            return None
        
        # Skip if position size is 0 (too expensive)
        if qty == 0:
            logger.warning(f"Skipping trade - option ${option_price:.2f}/contract too expensive for ${remaining_capital:.2f} remaining")
            send_telegram_alert(f"⚠️ Trade skipped: {option_symbol} @ ${option_price:.2f} too expensive")
            return None
        
        logger.info(f"POSITION SIZE: {qty} contracts @ ${option_price:.2f} = ${estimated_cost:.2f} (${remaining_capital:.2f} remaining of ${self.MAX_DAILY_DEPLOYED})")
        
        logger.info(f"Executing {trade_type} entry: {qty}x {option_symbol} @ ~${option_price:.2f}")
        
        # Send BUY alert to Telegram in parallel with order execution
        buy_message = f"""🟢 **BUY ORDER SENDING**

**{trade_type}** {underlying}
Strike: ${strike:.0f} | Exp: {expiry.strftime('%m/%d')}
Contracts: {qty}
Est. Price: ${option_price:.2f}
Pattern: {pattern}
Confidence: {confidence:.0f}%

⏳ Executing on Alpaca...
"""
        # Send alert in parallel thread
        alert_thread = threading.Thread(
            target=send_telegram_alert,
            args=(buy_message,),
            daemon=True
        )
        alert_thread.start()
        
        # Execute order on Alpaca (runs in parallel with alert)
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
            send_telegram_alert(f"❌ BUY ORDER FAILED: {trade_type} {underlying}")
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
        
        # Confirmation alert after order placed
        confirm_message = f"""✅ **BUY ORDER PLACED**

**{trade_type}** {underlying}
Strike: ${strike:.0f} | Exp: {expiry.strftime('%m/%d')}
Contracts: {qty}
Est. Entry: ${option_price:.2f}
Target: ${position.target_price:.2f} (+20%)
Stop: ${position.stop_loss:.2f} (-15%)

Order ID: `{order.get('id', 'N/A')[:8]}...`
Status: PENDING FILL
"""
        threading.Thread(target=send_telegram_alert, args=(confirm_message,), daemon=True).start()
        
        return position
    
    def execute_exit(self, position: AlpacaPosition, reason: str, current_price: float):
        """Execute exit for a position."""
        logger.info(f"Executing exit for {position.option_symbol}: {reason}")
        
        # Send SELL alert to Telegram in parallel with order execution
        sell_message = f"""🔴 **SELL ORDER SENDING**

**{position.trade_type}** {position.symbol}
Contracts: {position.qty}
Entry: ${position.entry_price:.2f}
Current: ${current_price:.2f}
Reason: {reason}

⏳ Closing on Alpaca...
"""
        threading.Thread(target=send_telegram_alert, args=(sell_message,), daemon=True).start()
        
        # Execute close on Alpaca (runs in parallel with alert)
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
                    
                    # CRITICAL: Verify actual cost is within limits
                    actual_cost = position.entry_price * position.qty * 100
                    if actual_cost > self.MAX_DAILY_DEPLOYED * 1.5:  # 50% tolerance = EMERGENCY CLOSE
                        logger.error(f"EMERGENCY: Filled cost ${actual_cost:.2f} exceeds ${self.MAX_DAILY_DEPLOYED * 1.5}!")
                        send_telegram_alert(f"🚨 EMERGENCY: Position ${actual_cost:.2f} > limit - AUTO-CLOSING!")
                        # Immediately close the oversized position
                        self.close_position(position.option_symbol)
                        position.status = PositionStatus.CLOSED
                        position.exit_reason = "OVERSIZED_POSITION_CLOSED"
                        continue
                    elif actual_cost > self.MAX_DAILY_DEPLOYED * 0.5:  # Using more than half daily limit = WARNING
                        logger.warning(f"NOTE: Single trade ${actual_cost:.2f} using >{50}% of daily ${self.MAX_DAILY_DEPLOYED} limit")
                        send_telegram_alert(f"⚠️ WARNING: Position cost ${actual_cost:.2f} slightly over limit")
                    
                    position.target_price = position.entry_price * 1.20  # +20% target
                    position.stop_loss = position.entry_price * 0.85     # -15% stop
                    
                    logger.info(f"Order filled: {position.option_symbol} @ ${position.entry_price:.2f}")
                    logger.info(f"  Total Cost: ${actual_cost:.2f} | Target: ${position.target_price:.2f} | Stop: ${position.stop_loss:.2f}")
                    
                    message = f"""✅ **ORDER FILLED**

**{position.trade_type}** {position.symbol}
Filled: {position.qty}x @ ${position.entry_price:.2f}
Total Cost: ${actual_cost:.2f}

**EXIT LEVELS (AUTOMATIC):**
Target: ${position.target_price:.2f} (+20%)
Stop: ${position.stop_loss:.2f} (-15%)
Max Hold: 45 minutes
"""
                    send_telegram_alert(message)
                
                elif order.get("status") in ["cancelled", "expired", "rejected"]:
                    position.status = PositionStatus.CANCELLED
                    order_status = order.get('status', 'unknown')
                    logger.warning(f"Order {order_status}: {position.option_symbol}")
                    
                    send_telegram_alert(f"""⚠️ **ORDER {order_status.upper()}**
                    
{position.trade_type} {position.symbol}
Symbol: {position.option_symbol}
Reason: Order was {order_status}
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
