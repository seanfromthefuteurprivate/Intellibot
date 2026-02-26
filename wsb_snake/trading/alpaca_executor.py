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
from wsb_snake.notifications.telegram_channels import send_signal, send_alpaca_status
from wsb_snake.trading.risk_governor import (
    get_risk_governor,
    TradingEngine,
)
from wsb_snake.trading.outcome_recorder import outcome_recorder
from wsb_snake.collectors.vix_structure import vix_structure
from wsb_snake.learning.self_evolving_memory import record_trade_for_learning
from wsb_snake.learning.introspection_engine import get_introspection_engine
from wsb_snake.learning.debate_consensus import get_debate_engine

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
    engine: str = "scalper"  # scalper | momentum | macro – for trim-and-hold
    trimmed: bool = False  # True after partial exit at +50%
    signal_id: Optional[int] = None  # Links to signals table for learning
    pattern: str = ""  # Pattern that triggered this trade (for self-evolving learning)

    def option_spec_line(self) -> str:
        """
        Return a human-readable option spec line for Telegram alerts.
        Example: "SPY $590 C exp 01/25 (0 DTE)"
        """
        parsed = _parse_option_symbol(self.option_symbol)
        if not parsed:
            return f"{self.symbol} {self.trade_type}"

        underlying, expiry_str, opt_type, strike = parsed
        # Parse expiry to compute DTE
        try:
            expiry_date = datetime.strptime(expiry_str, "%y%m%d").date()
            today = datetime.now().date()
            dte = (expiry_date - today).days
            dte_str = f"{dte} DTE" if dte >= 0 else "EXPIRED"
        except:
            dte_str = "? DTE"

        opt_char = "C" if opt_type == "C" else "P"
        exp_formatted = f"{expiry_str[2:4]}/{expiry_str[4:6]}"  # MMDD from YYMMDD
        return f"{underlying} ${strike:.0f} {opt_char} exp {exp_formatted} ({dte_str})"


def _parse_option_symbol(option_symbol: str) -> Optional[tuple]:
    """
    Parse OCC option symbol into components.
    Example: SPY260125C00590000 -> ("SPY", "260125", "C", 590.0)
    Returns: (underlying, expiry_str, option_type, strike) or None
    """
    if not option_symbol or len(option_symbol) < 15:
        return None

    # Find where digits start (end of underlying)
    underlying_end = 0
    for i, c in enumerate(option_symbol):
        if c.isdigit():
            underlying_end = i
            break

    if underlying_end == 0:
        return None

    underlying = option_symbol[:underlying_end]
    rest = option_symbol[underlying_end:]

    # Format: YYMMDD + C/P + 8-digit strike (strike * 1000)
    if len(rest) < 15:
        return None

    expiry_str = rest[:6]  # YYMMDD
    opt_type = rest[6]     # C or P
    strike_str = rest[7:15]  # 8 digits

    try:
        strike = int(strike_str) / 1000.0
    except:
        return None

    return (underlying, expiry_str, opt_type, strike)


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
    
    # ========== RISK CONTROLS - JP MORGAN GRADE ==========
    MAX_DAILY_EXPOSURE = 4000   # $4k max daily (margin-aware)
    MAX_PER_TRADE = 1000        # $1k max per trade
    MAX_CONCURRENT_POSITIONS = 3  # Max 3 positions (reduce correlation)
    MARKET_CLOSE_HOUR = 16  # 4 PM ET - all 0DTE must close
    CLOSE_BEFORE_MINUTES = 5  # Close 5 min before market close
    
    # ETF Priority - focus on liquid, predictable instruments
    ETF_TICKERS = ['SPY', 'QQQ', 'IWM', 'GLD', 'GDX', 'SLV', 'XLE', 'XLF', 'TLT', 'USO', 'UNG', 'HYG']
    ETF_PRIORITY = True  # Prioritize ETFs for scalping (higher win rate)

    # ========== MAX MODE - AGGRESSIVE LAST HOUR TRADING - FEB 6 ==========
    # QUICK PROFIT SYSTEM:
    # - Take profit FAST at +10% (don't wait, book it!)
    # - Trailing stop kicks in at +5% (move to breakeven)
    # - At +8% profit: move stop to +5%
    # - Initial stop: -8% (tight risk control)
    # - Max hold: 8 minutes (quick scalps only)
    #
    # MAX MODE PHILOSOPHY:
    # 1. Enter with conviction, exit with profit
    # 2. Don't let winners turn to losers
    # 3. Quick rotations - capture volatility, move on
    # 4. Predator mode - strike fast, book profit, hunt again
    #
    # Scalper exit defaults (overridable via env: SCALP_TARGET_PCT, SCALP_STOP_PCT, SCALP_MAX_HOLD_MINUTES)
    # RISK WARDEN: Tighter 0DTE stops for better risk control
    _SCALP_TARGET_PCT_DEFAULT = 1.06   # +6% target (achievable in 0DTE with decay)
    _SCALP_STOP_PCT_DEFAULT = 0.93     # -7% initial stop (RISK WARDEN: tighter than -10%)
    _SCALP_MAX_HOLD_MINUTES_DEFAULT = 5  # 5 MIN - exit before theta acceleration

    # RISK WARDEN: Trailing stop tiers for locking profits
    TRAIL_BREAKEVEN_TRIGGER = 0.03   # Move to breakeven at +3% profit
    TRAIL_LOCK_PROFIT_TRIGGER = 0.08  # At +8% profit, lock in +5%
    TRAIL_LOCK_PROFIT_LEVEL = 0.05   # Lock +5% when at +8%+
    TRAIL_TIME_TIGHTEN_MINUTES = 30  # After 30 min, trail to -3% from peak
    TRAIL_TIME_TIGHTEN_PCT = 0.03    # -3% trailing stop after 30 min

    # ========== LIMIT ORDER MODE - PRICE-MATCHED EXECUTION ==========
    # When USE_LIMIT_ORDERS=true, entry/exit orders use limit prices
    # so Telegram price = order price (price-matched execution).
    # Default: false (market orders) for faster fills.
    # Set ALPACA_USE_LIMIT_ORDERS=true to enable.
    USE_LIMIT_ORDERS = os.environ.get("ALPACA_USE_LIMIT_ORDERS", "false").lower() == "true"
    # For limit buys: add small buffer above ask to improve fill probability
    LIMIT_BUY_BUFFER_PCT = 0.02  # 2% above ask
    # For limit sells: subtract small buffer below bid for faster exit
    LIMIT_SELL_BUFFER_PCT = 0.02  # 2% below bid

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
        
        # Track daily exposure and daily PnL (for risk governor kill switch)
        self.daily_exposure_used = 0.0  # Running total of exposure deployed today
        self.daily_pnl = 0.0  # Realized PnL today; reset each day
        self.daily_trade_count = 0
        self.daily_reset_date = datetime.utcnow().date()

        # Scalper exit levels (tunable via env)
        _v = os.environ.get("SCALP_TARGET_PCT")
        self.SCALP_TARGET_PCT = float(_v) if _v is not None else self._SCALP_TARGET_PCT_DEFAULT
        _v = os.environ.get("SCALP_STOP_PCT")
        self.SCALP_STOP_PCT = float(_v) if _v is not None else self._SCALP_STOP_PCT_DEFAULT
        _v = os.environ.get("SCALP_MAX_HOLD_MINUTES")
        self.SCALP_MAX_HOLD_MINUTES = int(_v) if _v is not None else self._SCALP_MAX_HOLD_MINUTES_DEFAULT

        mode = "LIVE" if self.LIVE_TRADING else "Paper"
        order_mode = "LIMIT (price-matched)" if self.USE_LIMIT_ORDERS else "MARKET"
        logger.info(f"AlpacaExecutor initialized - {mode} trading mode")
        logger.info(f"Order mode: {order_mode}")
        logger.info(f"Scalper exit: target +{(self.SCALP_TARGET_PCT-1)*100:.0f}% | stop {(self.SCALP_STOP_PCT-1)*100:.0f}% | max hold {self.SCALP_MAX_HOLD_MINUTES}min")
        logger.info(f"Max daily exposure: ${self.MAX_DAILY_EXPOSURE} ($1k cash + $3k margin)")
        logger.info(f"Max per trade: ${self.MAX_PER_TRADE} | Max concurrent: {self.MAX_CONCURRENT_POSITIONS}")

        if self.LIVE_TRADING:
            logger.warning("⚠️ LIVE TRADING MODE ACTIVE - REAL MONEY AT RISK ⚠️")
        if self.USE_LIMIT_ORDERS:
            logger.info("📍 LIMIT ORDERS ENABLED: Entry/exit prices will match Telegram alerts")

    def _get_current_volatility_factor(self, ticker: str = "SPY") -> float:
        """
        Get volatility scaling factor based on current VIX.

        Returns multiplier:
        - VIX < 15: 0.8 (calm, can size up slightly)
        - VIX 15-20: 1.0 (normal)
        - VIX 20-25: 1.3 (elevated, widen stops)
        - VIX 25-35: 1.6 (high vol, much wider stops)
        - VIX > 35: 2.0 (crisis, maximum caution)

        This factor is used for:
        1. Position sizing - higher VIX = smaller positions
        2. Stop-loss width - higher VIX = wider stops to avoid noise
        """
        try:
            vix_data = vix_structure.get_trading_signal()
            vix = vix_data.get("vix", 20.0)

            logger.debug(f"Current VIX: {vix:.2f} for {ticker}")

            if vix < 15:
                return 0.8
            elif vix < 20:
                return 1.0
            elif vix < 25:
                return 1.3
            elif vix < 35:
                return 1.6
            else:
                return 2.0
        except Exception as e:
            logger.debug(f"VIX fetch failed, using default: {e}")
            return 1.0

    def _get_volatility_adjusted_stop(self, base_stop_pct: float, volatility_factor: float) -> float:
        """
        Adjust stop-loss percentage based on volatility.

        Higher volatility = wider stops to avoid getting stopped out by noise.

        Args:
            base_stop_pct: Base stop percentage (e.g., 0.90 for -10% stop)
            volatility_factor: VIX-based factor (0.8 to 2.0)

        Returns:
            Adjusted stop percentage (e.g., 0.85 for -15% stop in high vol)
        """
        # Calculate the loss percentage from base (e.g., 0.90 -> 0.10 = 10% loss)
        base_loss = 1.0 - base_stop_pct

        # Widen stop proportionally to volatility factor
        # At VIX 0.8 (calm): 10% * 0.8 = 8% stop (tighter)
        # At VIX 1.0 (normal): 10% * 1.0 = 10% stop (unchanged)
        # At VIX 1.3 (elevated): 10% * 1.3 = 13% stop (wider)
        # At VIX 1.6 (high): 10% * 1.6 = 16% stop (much wider)
        # At VIX 2.0 (crisis): 10% * 2.0 = 20% stop (maximum width)
        adjusted_loss = base_loss * volatility_factor

        # Cap at -25% max stop to limit risk even in extreme vol
        adjusted_loss = min(adjusted_loss, 0.25)

        adjusted_stop_pct = 1.0 - adjusted_loss

        logger.debug(f"Stop adjustment: base={base_stop_pct:.2%} -> adjusted={adjusted_stop_pct:.2%} (vol_factor={volatility_factor})")

        return adjusted_stop_pct

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
    
    def sync_existing_positions(self) -> int:
        """
        Sync existing Alpaca positions on startup.
        
        This ensures positions opened before a restart get tracked
        and monitored for exit conditions (target, stop, time decay).
        
        Returns number of positions synced.
        """
        alpaca_positions = self.get_options_positions()
        synced_count = 0
        
        for pos in alpaca_positions:
            option_symbol = pos.get("symbol", "")
            
            # Skip if already tracking this position
            if option_symbol in self.positions:
                logger.debug(f"Already tracking {option_symbol}")
                continue
            
            # Parse position details
            qty = int(pos.get("qty", 0))
            entry_price = float(pos.get("avg_entry_price", 0))
            current_price = float(pos.get("current_price", 0))
            unrealized_pnl = float(pos.get("unrealized_pl", 0))
            side = pos.get("side", "long")
            
            # Extract underlying from option symbol (e.g., IWM260127C00262000 -> IWM)
            underlying = ""
            for i, c in enumerate(option_symbol):
                if c.isdigit():
                    underlying = option_symbol[:i]
                    break
            
            # Determine if calls or puts from symbol
            trade_type = "CALLS" if "C" in option_symbol[len(underlying):len(underlying)+7] else "PUTS"
            
            # Set conservative targets/stops for orphaned positions
            # Default: +20% target, -15% stop (standard scalp settings)
            target_price = entry_price * self.SCALP_TARGET_PCT
            stop_loss = entry_price * self.SCALP_STOP_PCT
            
            # Create tracked position
            position_id = f"sync_{option_symbol}_{int(datetime.utcnow().timestamp())}"
            new_pos = AlpacaPosition(
                position_id=position_id,
                symbol=underlying,
                option_symbol=option_symbol,
                side=side,
                trade_type=trade_type,
                qty=qty,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                status=PositionStatus.OPEN,
                entry_time=datetime.utcnow(),  # Approximate since we don't know exact entry
            )
            
            with self._lock:
                self.positions[option_symbol] = new_pos
            
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
            
            logger.info(f"🔄 SYNCED: {option_symbol} {qty}x @ ${entry_price:.2f} -> ${current_price:.2f} ({pnl_pct:+.1f}%)")
            logger.info(f"   Target: ${target_price:.2f} (+{(self.SCALP_TARGET_PCT-1)*100:.0f}%) | Stop: ${stop_loss:.2f} ({(self.SCALP_STOP_PCT-1)*100:.0f}%)")
            
            synced_count += 1

            # Get option spec for alert
            sync_option_spec = new_pos.option_spec_line()

            # Alert about synced positions - Alpaca channel only (execution status)
            send_alpaca_status(f"""🔄 **SYNCED POSITION**

**Option:** {sync_option_spec}
Entry (option): ${entry_price:.2f}
Current (option): ${current_price:.2f} ({pnl_pct:+.1f}%)
Target (option): ${target_price:.2f} (+{(self.SCALP_TARGET_PCT-1)*100:.0f}%)
Stop (option): ${stop_loss:.2f} ({(self.SCALP_STOP_PCT-1)*100:.0f}%)

_Position picked up from restart - now monitoring_""")
        
        if synced_count > 0:
            logger.info(f"✅ Synced {synced_count} existing position(s) from Alpaca")
        else:
            logger.info("No existing Alpaca positions to sync")
        
        return synced_count
    
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
    
    def _reset_daily_count_if_needed(self):
        """Reset daily trade count, exposure, and daily PnL if new trading day."""
        today = datetime.utcnow().date()
        if today != self.daily_reset_date:
            self.daily_trade_count = 0
            self.daily_exposure_used = 0.0  # Reset margin utilization
            self.daily_pnl = 0.0  # Reset daily PnL for risk governor
            self.daily_reset_date = today
            logger.info(f"Daily reset: 0/{self.MAX_CONCURRENT_POSITIONS} trades, $0/${self.MAX_DAILY_EXPOSURE} exposure")
    
    def get_remaining_daily_capital(self) -> float:
        """Get remaining capital available for today (legacy - no longer used for limiting)."""
        # Reset daily tracker if new day
        today = datetime.utcnow().date()
        if today != self.daily_reset_date:
            self.daily_trade_count = 0
            self.daily_reset_date = today
            logger.info(f"Daily trade count reset: 0/{self.MAX_CONCURRENT_POSITIONS} trades")
        
        # Calculate currently deployed capital from open positions
        positions = self.get_options_positions()
        current_deployed = sum(
            float(p.get('cost_basis', 0)) or 
            (float(p.get('qty', 0)) * float(p.get('avg_entry_price', 0)) * 100)
            for p in positions
        )
        
        remaining = self.MAX_DAILY_EXPOSURE - current_deployed
        return max(0, remaining)
    
    def calculate_position_size(
        self,
        option_price: float,
        max_value: Optional[float] = None
    ) -> int:
        """
        Calculate number of contracts to buy.
        Uses $1,000 max per trade (not daily cap).
        Options are 100 shares per contract.
        """
        if max_value is None:
            # Use MAX_PER_TRADE for each individual trade
            max_value = self.MAX_PER_TRADE
        
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

        HYDRA FIX: Now records outcomes to risk governor for consecutive loss tracking.
        """
        import pytz
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)

        positions = self.get_options_positions()
        closed = 0

        for p in positions:
            symbol = p.get('symbol', '')
            try:
                # HYDRA FIX: Get P/L before closing to record outcome
                pnl = 0.0
                try:
                    unrealized_pl = p.get('unrealized_pl')
                    if unrealized_pl is not None:
                        pnl = float(unrealized_pl)
                except:
                    pass

                self.close_position(symbol)
                closed += 1
                logger.info(f"0DTE EOD close: {symbol} (P/L: ${pnl:.2f})")

                # HYDRA FIX: Record outcome to risk governor for consecutive loss tracking + win rate preservation
                try:
                    from wsb_snake.trading.risk_governor import get_risk_governor
                    governor = get_risk_governor()
                    outcome = "win" if pnl > 0 else "loss"
                    governor.record_trade_outcome(outcome, pnl=pnl)
                    logger.info(f"EOD outcome recorded: {outcome} (${pnl:.2f}) for {symbol}")
                except Exception as gov_err:
                    logger.debug(f"Failed to record EOD outcome: {gov_err}")

            except Exception as e:
                logger.error(f"Failed to close 0DTE position {symbol}: {e}")
        
        if closed > 0:
            # Send to main channel - important for all users
            send_signal(f"""⏰ **END OF DAY - 0DTE POSITIONS CLOSED**

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
    
    def sell_option_by_symbol(
        self, option_symbol: str, qty: int, limit_price: Optional[float] = None
    ) -> Optional[Dict]:
        """Sell (reduce) option position by symbol and qty. For partial closes.

        If limit_price is provided, uses a limit order at that price.
        Otherwise uses market order.
        """
        if qty <= 0:
            return None
        try:
            order_type = "limit" if limit_price else "market"
            order_data = {
                "symbol": option_symbol,
                "qty": str(qty),
                "side": "sell",
                "type": order_type,
                "time_in_force": "day",
            }
            if limit_price:
                order_data["limit_price"] = str(round(limit_price, 2))

            resp = requests.post(
                f"{self.BASE_URL}/v2/orders",
                headers=self.headers,
                json=order_data,
                timeout=10,
            )
            if resp.status_code >= 400:
                logger.error(f"Partial sell failed {option_symbol} qty={qty}: {resp.status_code} {resp.text}")
                return None
            return resp.json()
        except Exception as e:
            logger.error(f"Partial sell error {option_symbol}: {e}")
            return None

    def close_position(self, option_symbol: str, limit_price: Optional[float] = None) -> Optional[Dict]:
        """Close an existing position by selling.

        If limit_price is provided and USE_LIMIT_ORDERS is enabled, uses a limit sell.
        Otherwise uses the DELETE /positions endpoint (market close).
        """
        # If limit order mode and we have a price, use sell_option_by_symbol with limit
        if self.USE_LIMIT_ORDERS and limit_price:
            # Get current position qty
            try:
                resp = requests.get(
                    f"{self.BASE_URL}/v2/positions/{option_symbol}",
                    headers=self.headers,
                    timeout=10
                )
                if resp.status_code == 200:
                    pos_data = resp.json()
                    qty = int(pos_data.get("qty", 0))
                    if qty > 0:
                        # Apply buffer: sell slightly below bid for faster fill
                        adjusted_price = round(limit_price * (1 - self.LIMIT_SELL_BUFFER_PCT), 2)
                        logger.info(f"Closing {option_symbol} with LIMIT @ ${adjusted_price:.2f} (bid: ${limit_price:.2f})")
                        return self.sell_option_by_symbol(option_symbol, qty, adjusted_price)
            except Exception as e:
                logger.warning(f"Could not get position qty for limit close, falling back to market: {e}")

        # Default: market close via DELETE
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
            send_alpaca_status(f"⚠️ Failed to close {option_symbol}: {str(e)[:100]}")
            return None
        except Exception as e:
            logger.error(f"Failed to close position {option_symbol}: {e}")
            send_alpaca_status(f"⚠️ Failed to close {option_symbol}: {str(e)[:100]}")
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
        pattern: str,
        engine: TradingEngine = TradingEngine.SCALPER,
        expiry_override: Optional[datetime] = None,
        signal_id: Optional[int] = None,
        strike_override: Optional[float] = None,
        option_symbol_override: Optional[str] = None,
        option_type_override: Optional[str] = None,  # 'call' or 'put' - overrides direction-based default
    ) -> Optional[AlpacaPosition]:
        """
        Execute a scalp trade entry.
        
        For 0DTE SPY options:
        - direction='long' -> buy CALLS
        - direction='short' -> buy PUTS (we don't short options, we buy puts)
        
        engine: SCALPER (default), MOMENTUM, or MACRO – used for risk governor limits and position sizing.
        """
        # DIAGNOSTIC: Log entry to help trace execution flow
        logger.info(f"EXECUTOR: execute_scalp_entry called for {underlying} {direction} @ ${entry_price:.2f}")

        self._reset_daily_count_if_needed()
        logger.debug(f"EXECUTOR: daily count reset done")

        governor = get_risk_governor()
        logger.debug(f"EXECUTOR: got risk governor")

        open_positions = [p for p in self.positions.values() if p.status in (PositionStatus.OPEN, PositionStatus.PENDING)]
        open_count = len(open_positions)
        logger.info(f"EXECUTOR: {open_count} open positions")

        positions_with_cost = [
            (p.symbol, p.option_symbol, p.entry_price * p.qty * 100)
            for p in open_positions
        ]
        logger.debug(f"EXECUTOR: calling can_trade")

        allowed, reason = governor.can_trade(
            engine=engine,
            ticker=underlying,
            open_positions_count=open_count,
            positions_with_cost=positions_with_cost,
            daily_pnl=self.daily_pnl,
            daily_exposure_used=self.daily_exposure_used,
        )
        logger.info(f"EXECUTOR: can_trade returned allowed={allowed}, reason={reason}")

        if not allowed:
            logger.warning(f"Risk governor blocked trade: {reason}")
            send_alpaca_status(f"⏸️ Risk governor: {reason}")
            return None

        logger.info(f"EXECUTOR: acquiring lock")
        with self._lock:
            if open_count >= self.MAX_CONCURRENT_POSITIONS:
                logger.warning("Max concurrent positions reached, skipping entry")
                return None
        logger.info(f"EXECUTOR: lock released")

        logger.info(f"EXECUTOR: Passed risk checks, proceeding with {underlying} {direction}")

        # Option type: use override if provided (for CPL), else infer from direction
        if option_type_override:
            option_type = option_type_override.lower()
            trade_type = "CALLS" if option_type == "call" else "PUTS"
        else:
            trade_type = "CALLS" if direction == "long" else "PUTS"
            option_type = "call" if direction == "long" else "put"
        
        # CRITICAL VALIDATION: Reject trades with invalid parameters
        if stop_loss <= 0:
            logger.error(f"INVALID STOP LOSS ${stop_loss:.2f} - ABORTING (must be positive)")
            send_alpaca_status(f"❌ Trade REJECTED: Invalid stop loss ${stop_loss:.2f} for {underlying}")
            return None

        if target_price <= 0:
            logger.error(f"INVALID TARGET ${target_price:.2f} - ABORTING (must be positive)")
            send_alpaca_status(f"❌ Trade REJECTED: Invalid target ${target_price:.2f} for {underlying}")
            return None
        
        if entry_price <= 0:
            logger.error(f"INVALID ENTRY ${entry_price:.2f} - ABORTING (must be positive)")
            return None
        
        # Validate direction matches stop/target relationship
        if direction == "long":
            if stop_loss >= entry_price:
                logger.error(f"INVALID LONG SETUP: Stop ${stop_loss:.2f} >= Entry ${entry_price:.2f}")
                send_alpaca_status(f"❌ Trade REJECTED: Bad stop for LONG {underlying}")
                return None
            if target_price <= entry_price:
                logger.error(f"INVALID LONG SETUP: Target ${target_price:.2f} <= Entry ${entry_price:.2f}")
                send_alpaca_status(f"❌ Trade REJECTED: Bad target for LONG {underlying}")
                return None
        else:  # short
            if stop_loss <= entry_price:
                logger.error(f"INVALID SHORT SETUP: Stop ${stop_loss:.2f} <= Entry ${entry_price:.2f}")
                send_alpaca_status(f"❌ Trade REJECTED: Bad stop for SHORT {underlying}")
                return None
            if target_price >= entry_price:
                logger.error(f"INVALID SHORT SETUP: Target ${target_price:.2f} >= Entry ${entry_price:.2f}")
                send_alpaca_status(f"❌ Trade REJECTED: Bad target for SHORT {underlying}")
                return None
        
        # Validate R:R ratio is reasonable (at least 1:1)
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0
        if rr_ratio < 0.5:
            logger.error(f"BAD R:R RATIO {rr_ratio:.2f} - Risk ${risk:.2f} vs Reward ${reward:.2f}")
            send_alpaca_status(f"❌ Trade REJECTED: Bad R:R {rr_ratio:.2f} for {underlying}")
            return None
        
        logger.info(f"Validated trade: {underlying} {direction} Entry=${entry_price:.2f} Target=${target_price:.2f} Stop=${stop_loss:.2f} R:R={rr_ratio:.2f}")
        
        # Expiry: override (LEAPS/momentum) or compute
        from wsb_snake.config import DAILY_0DTE_TICKERS
        daily_0dte_tickers = DAILY_0DTE_TICKERS
        
        import pytz
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        
        if expiry_override is not None:
            expiry = expiry_override.replace(tzinfo=None) if hasattr(expiry_override, 'replace') else expiry_override
            logger.info(f"Using expiry override: {expiry.strftime('%Y-%m-%d')} (engine={engine.value})")
        elif underlying in daily_0dte_tickers:
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

        # ========== STRIKE & OPTION SYMBOL RESOLUTION ==========
        # If CPL passes strike_override and option_symbol_override, use them directly
        # This fixes the bug where entry_price (option premium) was used to calculate strike
        max_contract_cost = self.MAX_PER_TRADE  # $1000 max per trade
        option_price = 0
        bid_price = 0
        strike = 0
        option_symbol = ""
        quote = {}

        if strike_override and option_symbol_override:
            # CPL/MAX MODE provided strike - use it, but generate proper OCC symbol
            strike = strike_override
            # CRITICAL FIX: Always use format_option_symbol for consistent OCC format
            # Polygon symbols may differ slightly from OCC standard (e.g., different padding)
            # This ensures the symbol we store matches what Alpaca actually receives
            option_symbol = self.format_option_symbol(
                underlying, expiry, strike,
                "C" if option_type == "call" else "P"
            )
            quote = self.get_option_quote(option_symbol)
            if quote:
                option_price = float(quote.get("ap", 0))
                bid_price = float(quote.get("bp", 0))
            logger.info(f"Using strike ${strike:.0f} with OCC symbol {option_symbol}")
        else:
            # Original logic: calculate strike from underlying price
            # NOTE: entry_price here should be the UNDERLYING price, not option premium
            # For CPL calls, we now pass strike_override so this branch won't be used
            interval = self.get_strike_interval(underlying, entry_price)
            strikes_to_try = 5  # Try up to 5 different strikes

            # Strike offsets: [ITM, ATM, slight OTM, moderate OTM, far OTM]
            strike_offsets = [-1, 0, 1, 2, 3]  # Number of intervals from ATM

            for offset in strike_offsets:
                if direction == "long":
                    # For calls: negative offset = ITM (cheaper delta), positive = OTM
                    strike = self.round_to_strike(entry_price - interval * offset, interval, "nearest")
                else:
                    # For puts: negative offset = ITM, positive = OTM (cheaper)
                    strike = self.round_to_strike(entry_price + interval * offset, interval, "nearest")

                option_symbol = self.format_option_symbol(
                    underlying, expiry, strike,
                    "C" if option_type == "call" else "P"
                )

                quote = self.get_option_quote(option_symbol)
                if not quote:
                    continue

                option_price = float(quote.get("ap", 0))
                bid_price = float(quote.get("bp", 0))

                if option_price > 0 and bid_price > 0:
                    contract_cost = option_price * 100
                    if contract_cost <= max_contract_cost:
                        logger.info(f"Selected strike ${strike:.0f} (interval: ${interval}) for {underlying} @ ${entry_price:.2f}")
                        break
                    else:
                        logger.info(f"Strike ${strike:.0f} too expensive (${contract_cost:.0f}/contract), trying further OTM...")
        
        # Check if we found an affordable option
        if not quote or option_price <= 0:
            logger.error(f"No valid quote found for {underlying} options - ABORTING trade")
            send_alpaca_status(f"❌ Trade aborted: No valid {underlying} options available")
            return None

        # Explicit check: did we find an affordable option?
        contract_cost = option_price * 100
        if contract_cost > max_contract_cost:
            logger.error(f"No affordable {underlying} option found (cheapest: ${contract_cost:.0f}/contract > ${max_contract_cost:.0f} limit)")
            send_alpaca_status(f"❌ Trade aborted: {underlying} options too expensive (cheapest ${contract_cost:.0f}/contract)")
            return None
        
        # Validate quote freshness (must be within 60 seconds)
        quote_timestamp = quote.get("t", "")
        if quote_timestamp:
            try:
                from dateutil import parser as date_parser
                quote_time = date_parser.parse(quote_timestamp)
                quote_age = (datetime.now(quote_time.tzinfo) - quote_time).total_seconds()
                if quote_age > 60:
                    logger.warning(f"Quote for {option_symbol} is {quote_age:.0f}s old - may be stale")
            except Exception as e:
                logger.debug(f"Could not parse quote timestamp: {e}")
        
        if bid_price <= 0:
            logger.error(f"No bid for {option_symbol} - option may be illiquid - ABORTING")
            send_alpaca_status(f"❌ Trade aborted: No bid for {option_symbol} (illiquid)")
            return None
        
        # Verify bid-ask spread is reasonable (< 20% of mid-price)
        mid_price = (option_price + bid_price) / 2
        spread_pct = (option_price - bid_price) / mid_price * 100
        if spread_pct > 20:
            logger.warning(f"Wide spread {spread_pct:.1f}% on {option_symbol} - may be illiquid")
        
        # Position size: risk governor (confidence + vol) or executor fallback
        buying_power = self.get_buying_power()
        governor = get_risk_governor()
        volatility_factor = self._get_current_volatility_factor(underlying)

        # Use Kelly sizing when we have win probability data
        try:
            historical_win_rate = governor.get_win_rate()
            if historical_win_rate > 0.4:  # Only use Kelly if we have enough data
                qty = governor.compute_kelly_position_size(
                    engine=engine,
                    win_probability=min(confidence / 100, historical_win_rate),
                    avg_win_pct=0.06,  # +6% target
                    avg_loss_pct=0.10,  # -10% stop
                    option_price=option_price,
                    buying_power=buying_power if buying_power > 0 else None,
                    volatility_factor=volatility_factor,
                )
                logger.info(f"Kelly sizing: {qty} contracts (win_rate={historical_win_rate:.1%})")
            else:
                # Fall back to confidence-based sizing
                qty = governor.compute_position_size(
                    engine=engine,
                    confidence_pct=confidence,
                    option_price=option_price,
                    buying_power=buying_power if buying_power > 0 else None,
                    volatility_factor=volatility_factor,
                )
        except Exception as e:
            logger.warning(f"Kelly sizing failed, using confidence-based: {e}")
            qty = governor.compute_position_size(
                engine=engine,
                confidence_pct=confidence,
                option_price=option_price,
                buying_power=buying_power if buying_power > 0 else None,
                volatility_factor=volatility_factor,
            )
        if qty <= 0:
            qty = self.calculate_position_size(option_price)
        estimated_cost = qty * option_price * 100  # Total cost in dollars
        
        # Skip if position size is 0 (option too expensive for $1000 per trade limit)
        if qty == 0:
            logger.warning(f"Skipping trade - option ${option_price:.2f}/contract too expensive (>${self.MAX_PER_TRADE}/trade limit)")
            send_alpaca_status(f"⚠️ Trade skipped: {option_symbol} @ ${option_price:.2f} exceeds ${self.MAX_PER_TRADE}/trade max")
            return None

        # Check daily exposure limit ($4,000 = $1k cash + $3k margin)
        remaining_exposure = self.MAX_DAILY_EXPOSURE - self.daily_exposure_used
        if estimated_cost > remaining_exposure:
            logger.warning(f"Daily exposure limit reached: Used ${self.daily_exposure_used:.0f}/${self.MAX_DAILY_EXPOSURE} - only ${remaining_exposure:.0f} remaining")
            send_alpaca_status(f"⏸️ Exposure cap: ${self.daily_exposure_used:.0f}/${self.MAX_DAILY_EXPOSURE} used (margin limit)")
            return None
        
        logger.info(f"POSITION SIZE: {qty} contracts @ ${option_price:.2f} = ${estimated_cost:.2f}")
        logger.info(f"EXPOSURE: Trade {self.daily_trade_count + 1}/{self.MAX_CONCURRENT_POSITIONS} | ${self.daily_exposure_used + estimated_cost:.0f}/${self.MAX_DAILY_EXPOSURE} used")
        
        logger.info(f"Executing {trade_type} entry: {qty}x {option_symbol} @ ~${option_price:.2f}")
        
        # Compute DTE for alerts
        today = datetime.now().date()
        dte = (expiry.date() - today).days if hasattr(expiry, 'date') else 0
        opt_char = "C" if option_type == "call" else "P"
        option_spec_str = f"{underlying} ${strike:.0f} {opt_char} exp {expiry.strftime('%m/%d')} ({dte} DTE)"

        # Determine order type and limit price
        if self.USE_LIMIT_ORDERS:
            # Limit buy: use ask + small buffer for better fill probability
            limit_price = round(option_price * (1 + self.LIMIT_BUY_BUFFER_PCT), 2)
            entry_order_type = "limit"
            order_type_label = f"LIMIT @ ${limit_price:.2f}"
        else:
            limit_price = None
            entry_order_type = "market"
            order_type_label = "MARKET"

        # Send BUY alert to Alpaca channel (execution status)
        buy_message = f"""🟢 **BUY ORDER SENDING**

**Option:** {option_spec_str}
Contracts: {qty}
Entry (option): ${option_price:.2f}
Order Type: {order_type_label}
Pattern: {pattern}
Confidence: {confidence:.0f}%

⏳ Executing on Alpaca...
"""
        # Send to Alpaca channel in parallel thread
        alert_thread = threading.Thread(
            target=send_alpaca_status,
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
            order_type=entry_order_type,
            limit_price=limit_price
        )
        
        if not order:
            logger.error("Failed to place entry order")
            send_alpaca_status(f"❌ BUY ORDER FAILED: {trade_type} {underlying}")
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
            target_price=option_price * self.SCALP_TARGET_PCT,
            stop_loss=option_price * self.SCALP_STOP_PCT,
            status=PositionStatus.PENDING,
            alpaca_order_id=order.get("id"),
            engine=engine.value,
            signal_id=signal_id,
            pattern=pattern,  # Store pattern for self-evolving learning
        )
        
        with self._lock:
            self.positions[position_id] = position
            self.daily_trade_count += 1  # Increment daily trade counter
            self.daily_exposure_used += estimated_cost  # Track margin utilization

        logger.info(f"Trade {self.daily_trade_count}/{self.MAX_CONCURRENT_POSITIONS} | Exposure: ${self.daily_exposure_used:.0f}/${self.MAX_DAILY_EXPOSURE}")

        # AUTO-START MONITORING: Ensure exit monitoring is running
        # This fixes the bug where positions were opened but never closed
        if not self.running:
            # First sync any existing positions from Alpaca (from previous runs)
            self.sync_existing_positions()
            self.start_monitoring()
            logger.info("Position monitoring auto-started with existing positions synced")
        
        # Confirmation alert after order placed - goes to Alpaca channel
        confirm_message = f"""✅ **BUY ORDER PLACED**

**Option:** {option_spec_str}
Contracts: {qty}
Entry (option): ${option_price:.2f}
Order Type: {order_type_label}
Target (option): ${position.target_price:.2f} (+{(self.SCALP_TARGET_PCT-1)*100:.0f}%)
Stop (option): ${position.stop_loss:.2f} ({(self.SCALP_STOP_PCT-1)*100:.0f}%)

Order ID: `{order.get('id', 'N/A')[:8]}...`
Status: PENDING FILL
"""
        threading.Thread(target=send_alpaca_status, args=(confirm_message,), daemon=True).start()
        
        return position
    
    def execute_exit(self, position: AlpacaPosition, reason: str, current_price: float):
        """Execute exit for a position."""
        logger.info(f"Executing exit for {position.option_symbol}: {reason}")

        # Get option spec for alert
        option_spec_str = position.option_spec_line()

        # Determine order type for exit
        if self.USE_LIMIT_ORDERS:
            exit_limit_price = round(current_price * (1 - self.LIMIT_SELL_BUFFER_PCT), 2)
            exit_order_type_label = f"LIMIT @ ${exit_limit_price:.2f}"
        else:
            exit_limit_price = None
            exit_order_type_label = "MARKET"

        # Send SELL alert to Alpaca channel (execution status)
        sell_message = f"""🔴 **SELL ORDER SENDING**

**Option:** {option_spec_str}
Contracts: {position.qty}
Entry (option): ${position.entry_price:.2f}
Exit (option): ${current_price:.2f}
Order Type: {exit_order_type_label}
Reason: {reason}

⏳ Closing on Alpaca...
"""
        threading.Thread(target=send_alpaca_status, args=(sell_message,), daemon=True).start()

        # Execute close on Alpaca (runs in parallel with alert)
        # Pass current_price for limit order mode
        result = self.close_position(position.option_symbol, limit_price=current_price)
        
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
        self.daily_pnl += position.pnl  # For risk governor kill switch
        if position.pnl > 0:
            self.winning_trades += 1

        # Record outcome to risk governor for consecutive loss tracking + win rate preservation
        try:
            from wsb_snake.trading.risk_governor import get_risk_governor
            governor = get_risk_governor()
            governor.record_trade_outcome("win" if position.pnl > 0 else "loss", pnl=position.pnl)
        except Exception as e:
            logger.debug(f"Failed to record outcome to governor: {e}")

        # Record outcome to all learning systems
        position.exit_reason = reason
        try:
            outcome_recorder.record_trade_outcome(
                signal_id=position.signal_id,
                symbol=position.symbol,
                trade_type=position.trade_type,
                entry_price=position.entry_price,
                exit_price=current_price,
                pnl=position.pnl,
                pnl_pct=position.pnl_pct,
                exit_reason=reason,
                entry_time=position.entry_time or datetime.now(),
                exit_time=position.exit_time or datetime.now(),
                engine=position.engine,
                bars=None,  # Could fetch bars here if needed for pattern learning
            )
        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")

        # SELF-EVOLVING MEMORY: Record trade for Thompson Sampling + Lessons Learning
        try:
            hold_minutes = int((position.exit_time - position.entry_time).total_seconds() / 60) if position.entry_time else 0
            direction = "long" if position.trade_type == "CALLS" else "short"
            # Use stored pattern or fall back to engine name
            pattern = position.pattern if position.pattern else position.engine

            record_trade_for_learning(
                ticker=position.symbol,
                pattern=pattern,
                strategy=position.engine,
                direction=direction,
                pnl_pct=position.pnl_pct,
                hold_time_minutes=hold_minutes,
                exit_reason=reason,
                entry_price=position.entry_price,
                exit_price=current_price,
            )
            logger.debug(f"SELF_EVOLVING: Recorded {position.symbol} trade for learning")
        except Exception as e:
            logger.debug(f"Self-evolving memory record failed: {e}")

        # GATE 35: INTROSPECTION - Record outcome for pattern health tracking
        try:
            introspection = get_introspection_engine()
            pattern = position.pattern if position.pattern else position.engine
            introspection.record_trade_outcome(
                pattern=pattern,
                ticker=position.symbol,
                pnl_pct=position.pnl_pct
            )
            logger.debug(f"GATE_35: Recorded {position.symbol} outcome for introspection")
        except Exception as e:
            logger.debug(f"Introspection record failed: {e}")

        # GATE 15: DEBATE - Record outcome to track bull/bear accuracy
        try:
            debate = get_debate_engine()
            debate.record_outcome(
                ticker=position.symbol,
                pnl_pct=position.pnl_pct
            )
            logger.debug(f"GATE_15: Recorded {position.symbol} outcome for debate learning")
        except Exception as e:
            logger.debug(f"Debate record failed: {e}")

        # HYDRA FEEDBACK: Send trade result to HYDRA for learning
        try:
            from wsb_snake.collectors.hydra_bridge import get_hydra_bridge
            hydra = get_hydra_bridge()
            intel = hydra.get_intel()
            hold_seconds = int((position.exit_time - position.entry_time).total_seconds()) if position.entry_time else None
            hydra.send_trade_result(
                ticker=position.symbol,
                direction="LONG" if position.trade_type == "CALLS" else "SHORT",
                pnl=position.pnl,
                pnl_pct=position.pnl_pct,
                conviction=position.conviction if hasattr(position, 'conviction') else 70.0,
                regime=intel.regime,
                mode="SCALP",  # Could be BLOWUP if in blowup mode
                exit_reason=reason,
                hold_seconds=hold_seconds
            )
            logger.debug(f"HYDRA_FEEDBACK: Sent result for {position.symbol}")
        except Exception as e:
            logger.debug(f"HYDRA feedback skipped: {e}")

        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        emoji = "💰" if position.pnl > 0 else "🛑"

        # Trade result - send to MAIN channel for all users
        message = f"""{emoji} **TRADE CLOSED**

**Option:** {option_spec_str}
Exit Reason: {reason}
Contracts: {position.qty}

Entry (option): ${position.entry_price:.2f}
Exit (option): ${current_price:.2f}
P&L: ${position.pnl:+.2f} ({position.pnl_pct:+.1f}%)

**Session Stats:**
Trades: {self.total_trades} | Win Rate: {win_rate:.0f}%
Total P&L: ${self.total_pnl:+.2f}
"""
        send_signal(message)
        
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
            
            time.sleep(2)  # Check every 2 seconds for faster stop loss reaction on 0DTE
    
    def _check_order_fills(self):
        """Check if pending orders have filled.

        CRITICAL: Do NOT hold lock during network calls to avoid blocking execute_scalp_entry.
        """
        # Step 1: Collect pending positions WITHOUT holding lock during network I/O
        pending_to_check = []
        with self._lock:
            for position in list(self.positions.values()):
                if position.status == PositionStatus.PENDING and position.alpaca_order_id:
                    pending_to_check.append((position.position_id, position.alpaca_order_id))

        if not pending_to_check:
            return

        # Step 2: Make network calls WITHOUT holding lock
        order_results = {}
        for pos_id, order_id in pending_to_check:
            order = self.get_order_status(order_id)
            if order:
                order_results[pos_id] = order

        if not order_results:
            return

        # Step 3: Update positions WITH lock held (no network calls here)
        with self._lock:
            for pos_id, order in order_results.items():
                position = self.positions.get(pos_id)
                if not position or position.status != PositionStatus.PENDING:
                    continue  # Position may have changed while we were checking
                if not order:
                    continue
                
                if order.get("status") == "filled":
                    position.status = PositionStatus.OPEN
                    position.entry_time = datetime.now()
                    position.entry_price = float(order.get("filled_avg_price", position.entry_price))
                    
                    # CRITICAL: Verify actual cost is within limits
                    actual_cost = position.entry_price * position.qty * 100
                    if actual_cost > self.MAX_DAILY_EXPOSURE * 1.5:  # 50% tolerance = EMERGENCY CLOSE
                        logger.error(f"EMERGENCY: Filled cost ${actual_cost:.2f} exceeds ${self.MAX_DAILY_EXPOSURE * 1.5}!")
                        send_alpaca_status(f"🚨 EMERGENCY: Position ${actual_cost:.2f} > limit - AUTO-CLOSING!")
                        # Immediately close the oversized position
                        self.close_position(position.option_symbol)
                        position.status = PositionStatus.CLOSED
                        position.exit_reason = "OVERSIZED_POSITION_CLOSED"
                        # HYDRA FIX: Record emergency close as loss to risk governor
                        try:
                            from wsb_snake.trading.risk_governor import get_risk_governor
                            governor = get_risk_governor()
                            governor.record_trade_outcome("loss")  # Emergency close = loss
                            logger.info(f"Emergency close recorded as loss for {position.option_symbol}")
                        except Exception as gov_err:
                            logger.debug(f"Failed to record emergency outcome: {gov_err}")
                        continue
                    elif actual_cost > self.MAX_DAILY_EXPOSURE * 0.5:  # Using more than half daily limit = WARNING
                        logger.warning(f"NOTE: Single trade ${actual_cost:.2f} using >{50}% of daily ${self.MAX_DAILY_EXPOSURE} limit")
                        send_alpaca_status(f"⚠️ WARNING: Position cost ${actual_cost:.2f} slightly over limit")
                    
                    position.target_price = position.entry_price * self.SCALP_TARGET_PCT
                    position.stop_loss = position.entry_price * self.SCALP_STOP_PCT
                    
                    logger.info(f"Order filled: {position.option_symbol} @ ${position.entry_price:.2f}")
                    logger.info(f"  Total Cost: ${actual_cost:.2f} | Target: ${position.target_price:.2f} | Stop: ${position.stop_loss:.2f}")
                    
                    # Get option spec for alert
                    fill_option_spec = position.option_spec_line()

                    # Order filled - send to Alpaca channel (execution status)
                    message = f"""✅ **ORDER FILLED**

**Option:** {fill_option_spec}
Filled: {position.qty}x
Entry (option): ${position.entry_price:.2f}
Total Cost: ${actual_cost:.2f}

**EXIT LEVELS (AUTOMATIC):**
Target (option): ${position.target_price:.2f} (+{(self.SCALP_TARGET_PCT-1)*100:.0f}%)
Stop (option): ${position.stop_loss:.2f} ({(self.SCALP_STOP_PCT-1)*100:.0f}%)
Max Hold: {self.SCALP_MAX_HOLD_MINUTES} minutes
"""
                    send_alpaca_status(message)
                
                elif order.get("status") in ["cancelled", "expired", "rejected"]:
                    position.status = PositionStatus.CANCELLED
                    order_status = order.get('status', 'unknown')
                    logger.warning(f"Order {order_status}: {position.option_symbol}")

                    # Get option spec for alert
                    cancel_option_spec = position.option_spec_line()

                    send_alpaca_status(f"""⚠️ **ORDER {order_status.upper()}**

**Option:** {cancel_option_spec}
Reason: Order was {order_status}
""")
    
    def execute_partial_exit(self, position: AlpacaPosition, fraction: float, current_price: float) -> bool:
        """Trim-and-hold: close fraction of position (e.g. 0.5 = half). Returns True if done."""
        sell_qty = max(1, int(position.qty * fraction))
        if sell_qty >= position.qty:
            self.execute_exit(position, "TARGET HIT (full trim)", current_price)
            return True
        result = self.sell_option_by_symbol(position.option_symbol, sell_qty)
        if result:
            position.qty -= sell_qty
            position.trimmed = True
            trim_option_spec = position.option_spec_line()
            # Trim notification - goes to MAIN channel as it's an action signal
            send_signal(
                f"✂️ **TRIM**\n\n**Option:** {trim_option_spec}\n"
                f"Sold {sell_qty} contracts\n"
                f"Entry (option): ${position.entry_price:.2f}\n"
                f"Exit (option): ${current_price:.2f} (+{(current_price/position.entry_price-1)*100:.0f}%)\n"
                f"Remaining: {position.qty} contracts – letting rest run"
            )
            logger.info(f"Trimmed {position.option_symbol}: sold {sell_qty}, {position.qty} left")
            return True
        return False

    def _check_exits(self):
        """Check open positions for exit conditions. Trim-and-hold for momentum/macro.

        CRITICAL: Do NOT hold lock during network calls to avoid blocking execute_scalp_entry.
        """
        # Step 1: Collect open positions WITHOUT holding lock during network I/O
        open_to_check = []
        with self._lock:
            for position in list(self.positions.values()):
                if position.status == PositionStatus.OPEN:
                    open_to_check.append(position.option_symbol)

        if not open_to_check:
            return

        # Step 2: Fetch all quotes WITHOUT holding lock
        quotes = {}
        for option_symbol in open_to_check:
            quote = self.get_option_quote(option_symbol)
            if quote:
                bp = float(quote.get("bp", 0))
                ap = float(quote.get("ap", 0))
                current_price = (bp + ap) / 2 if (bp > 0 and ap > 0) else (bp or ap)
                if current_price > 0:
                    quotes[option_symbol] = current_price

        if not quotes:
            return

        # Step 3: Process exit logic WITH lock held (no network calls)
        with self._lock:
            for position in list(self.positions.values()):
                if position.status != PositionStatus.OPEN:
                    continue

                current_price = quotes.get(position.option_symbol)
                if not current_price:
                    continue

                pnl_pct = (current_price / position.entry_price - 1) * 100 if position.entry_price else 0

                # Trim-and-hold: momentum/macro – at +50% trim half, trail rest at +20%
                if position.engine in ("momentum", "macro"):
                    if not position.trimmed and pnl_pct >= 50:
                        self.execute_partial_exit(position, 0.5, current_price)
                        continue
                    if position.trimmed and current_price < position.entry_price * (self.SCALP_TARGET_PCT * 0.96):
                        self.execute_exit(position, "TRAIL STOP", current_price)
                        continue
                    # No 45min exit for momentum/macro – hold longer
                    if current_price >= position.target_price:
                        self.execute_exit(position, "TARGET HIT 🎯", current_price)
                    elif current_price <= position.stop_loss:
                        self.execute_exit(position, "STOP LOSS", current_price)
                    continue
                
                # Scalper: RISK WARDEN TRAILING STOP - tighter 0DTE risk control
                # Initial: -7% stop (tighter than old -10%)
                # Breakeven at +3% (trigger)
                # Lock +5% at +8% profit
                # After 30 min, trail to -3% from peak

                # Calculate current profit percentage
                profit_pct = (current_price - position.entry_price) / position.entry_price if position.entry_price > 0 else 0

                # Track peak price for time-based trailing (store as attribute)
                if not hasattr(position, '_peak_price') or current_price > getattr(position, '_peak_price', 0):
                    position._peak_price = current_price

                # Calculate hold time
                hold_minutes = 0
                if position.entry_time:
                    hold_minutes = (datetime.now() - position.entry_time).total_seconds() / 60

                # RISK WARDEN: Time-based stop tightening after 30 minutes
                if hold_minutes >= self.TRAIL_TIME_TIGHTEN_MINUTES:
                    # Trail to -3% from peak price (not entry)
                    peak_price = getattr(position, '_peak_price', current_price)
                    time_trail_stop = peak_price * (1 - self.TRAIL_TIME_TIGHTEN_PCT)
                    if position.stop_loss < time_trail_stop:
                        position.stop_loss = time_trail_stop
                        logger.info(f"TIME TRAIL: {position.option_symbol} stop to ${time_trail_stop:.2f} (-3% from peak ${peak_price:.2f}) after {hold_minutes:.0f}min")

                # RISK WARDEN: Profit-based trailing stops (tighter than before)
                if profit_pct >= self.TRAIL_LOCK_PROFIT_TRIGGER:  # +8% profit
                    # Lock in +5% profit (RISK WARDEN: was +3% at +5%)
                    new_stop = position.entry_price * (1 + self.TRAIL_LOCK_PROFIT_LEVEL)
                    if position.stop_loss < new_stop:
                        position.stop_loss = new_stop
                        logger.info(f"TRAIL: {position.option_symbol} LOCK +5% profit at ${new_stop:.2f} (profit: +{profit_pct*100:.1f}%)")
                elif profit_pct >= self.TRAIL_BREAKEVEN_TRIGGER:  # +3% profit
                    # Move stop to breakeven (RISK WARDEN: was +5% trigger)
                    new_stop = position.entry_price * 1.0
                    if position.stop_loss < new_stop:
                        position.stop_loss = new_stop
                        logger.info(f"TRAIL: {position.option_symbol} to BREAKEVEN (${new_stop:.2f})")
                elif profit_pct >= 0.02:  # +2% profit
                    # Reduce initial risk to -5%
                    new_stop = position.entry_price * 0.95
                    if position.stop_loss < new_stop:
                        position.stop_loss = new_stop
                        logger.info(f"TRAIL: {position.option_symbol} stop to -5% (${new_stop:.2f})")

                # Check exits
                if current_price >= position.target_price:
                    self.execute_exit(position, "TARGET HIT 🎯", current_price)
                elif current_price <= position.stop_loss:
                    exit_reason = "STOP LOSS" if position.stop_loss <= position.entry_price else f"TRAIL STOP (+{(position.stop_loss/position.entry_price-1)*100:.0f}%)"
                    self.execute_exit(position, exit_reason, current_price)
                elif position.entry_time:
                    elapsed = (datetime.now() - position.entry_time).total_seconds() / 60
                    if elapsed >= self.SCALP_MAX_HOLD_MINUTES:
                        self.execute_exit(position, f"TIME DECAY ({self.SCALP_MAX_HOLD_MINUTES}min)", current_price)
    
    def get_session_stats(self) -> Dict:
        """Get current session statistics."""
        open_positions = len([p for p in self.positions.values() if p.status == PositionStatus.OPEN])
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": win_rate,
            "total_pnl": self.total_pnl,
            "daily_pnl": self.daily_pnl,
            "open_positions": open_positions
        }


alpaca_executor = AlpacaExecutor()
