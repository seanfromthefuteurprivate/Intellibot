"""
Straddle Executor - For BLOWUP Mode

Implements straddle execution capability for blowup mode when HYDRA direction is NEUTRAL.

execute_straddle(ticker, expiry, size):
1. Find ATM strike (closest to current price)
2. Buy the ATM call at that strike
3. Buy the ATM put at that strike
4. Same size on both legs
5. Track as a paired position (straddle_id links both)
6. P&L = (call_pnl + put_pnl) — one leg dies, other leg moons
7. Exit rules:
   - Combined +80% → close both legs
   - Combined -50% → close both legs
   - Trail on combined: at +40%, trail at -15% from peak
   - Time stop: 30 min max

Log format:
- "STRADDLE: SPY {strike} call@${call_price} put@${put_price} total_cost=${total}"
- "STRADDLE_CLOSE: call_pnl=${call_pnl} put_pnl=${put_pnl} net=${net} return=${pct}%"
"""

import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class StraddleStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"  # One leg filled
    CLOSED = "CLOSED"
    FAILED = "FAILED"


@dataclass
class StraddleLeg:
    """Individual leg of a straddle."""
    symbol: str  # Full option symbol
    side: str  # "call" or "put"
    strike: float
    quantity: int
    entry_price: float = 0.0
    current_price: float = 0.0
    exit_price: float = 0.0
    order_id: Optional[str] = None
    filled: bool = False
    closed: bool = False


@dataclass
class StraddlePosition:
    """Complete straddle position."""
    straddle_id: str
    ticker: str
    expiry: str
    strike: float
    call_leg: StraddleLeg
    put_leg: StraddleLeg
    total_cost: float = 0.0
    current_value: float = 0.0
    peak_value: float = 0.0
    status: StraddleStatus = StraddleStatus.PENDING
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    exit_reason: Optional[str] = None


class StraddleExecutor:
    """
    Executes and manages straddle positions for blowup mode.

    Exit Rules:
    - Combined position +80% → close both legs
    - Combined position -50% → close both legs
    - Trail: at +40%, trail at -15% from peak
    - Time stop: 30 min max even in blowup mode
    """

    # Exit parameters
    TARGET_PCT = 80.0
    STOP_PCT = -50.0
    TRAIL_TRIGGER_PCT = 40.0
    TRAIL_DISTANCE_PCT = 15.0
    MAX_HOLD_SECONDS = 1800  # 30 minutes

    def __init__(self, alpaca_executor=None):
        """
        Initialize StraddleExecutor.

        Args:
            alpaca_executor: Instance of AlpacaExecutor for order placement
        """
        self.executor = alpaca_executor
        self.active_straddles: Dict[str, StraddlePosition] = {}
        self.closed_straddles: List[StraddlePosition] = []

    def set_executor(self, executor):
        """Set the Alpaca executor instance."""
        self.executor = executor

    def find_atm_strike(self, ticker: str, spot_price: float) -> float:
        """
        Find ATM strike closest to current spot price.

        Args:
            ticker: Underlying ticker (e.g., "SPY")
            spot_price: Current underlying price

        Returns:
            ATM strike price
        """
        # Round to nearest strike
        # SPY has $1 strikes, most ETFs have $1 or $0.50 strikes
        if ticker in ('SPY', 'QQQ', 'IWM'):
            strike_interval = 1.0
        else:
            strike_interval = 2.5  # Default for stocks

        atm_strike = round(spot_price / strike_interval) * strike_interval
        return atm_strike

    def build_option_symbol(self, ticker: str, expiry: str, strike: float, option_type: str) -> str:
        """
        Build OCC option symbol.

        Args:
            ticker: Underlying ticker
            expiry: Expiry date (YYYY-MM-DD)
            strike: Strike price
            option_type: "C" for call, "P" for put

        Returns:
            Full option symbol (e.g., "O:SPY260225C00700000")
        """
        # Parse expiry
        exp_date = datetime.strptime(expiry, '%Y-%m-%d')
        exp_str = exp_date.strftime('%y%m%d')

        # Format strike (multiply by 1000, pad to 8 digits)
        strike_str = f"{int(strike * 1000):08d}"

        # Build symbol
        return f"O:{ticker}{exp_str}{option_type}{strike_str}"

    def execute_straddle(
        self,
        ticker: str,
        expiry: str,
        size: float,
        spot_price: Optional[float] = None
    ) -> Tuple[bool, str, Optional[StraddlePosition]]:
        """
        Execute a straddle (buy both ATM call and put).

        Args:
            ticker: Underlying ticker (e.g., "SPY")
            expiry: Expiry date (YYYY-MM-DD format)
            size: Dollar amount to allocate (split between legs)
            spot_price: Current spot price (will fetch if not provided)

        Returns:
            (success, message, position)
        """
        straddle_id = str(uuid.uuid4())[:8]

        logger.info(f"STRADDLE_INIT: {ticker} size=${size:.0f} expiry={expiry}")

        # Get spot price if not provided
        if spot_price is None:
            spot_price = self._get_spot_price(ticker)
            if spot_price is None:
                return False, "Failed to get spot price", None

        # Find ATM strike
        strike = self.find_atm_strike(ticker, spot_price)

        # Build option symbols
        call_symbol = self.build_option_symbol(ticker, expiry, strike, "C")
        put_symbol = self.build_option_symbol(ticker, expiry, strike, "P")

        logger.info(f"STRADDLE: {ticker} strike={strike} call={call_symbol} put={put_symbol}")

        # Get option prices
        call_price = self._get_option_price(call_symbol)
        put_price = self._get_option_price(put_symbol)

        if call_price is None or put_price is None:
            return False, "Failed to get option prices", None

        # Calculate quantities (equal allocation)
        total_per_contract = (call_price + put_price) * 100
        quantity = max(1, int(size / total_per_contract))

        total_cost = (call_price + put_price) * 100 * quantity

        logger.info(f"STRADDLE: {ticker} {strike} call@${call_price:.2f} put@${put_price:.2f} "
                   f"qty={quantity} total_cost=${total_cost:.2f}")

        # Create position object
        position = StraddlePosition(
            straddle_id=straddle_id,
            ticker=ticker,
            expiry=expiry,
            strike=strike,
            call_leg=StraddleLeg(
                symbol=call_symbol,
                side="call",
                strike=strike,
                quantity=quantity,
                entry_price=call_price
            ),
            put_leg=StraddleLeg(
                symbol=put_symbol,
                side="put",
                strike=strike,
                quantity=quantity,
                entry_price=put_price
            ),
            total_cost=total_cost
        )

        # Execute both legs
        call_success = self._execute_leg(position.call_leg)
        put_success = self._execute_leg(position.put_leg)

        if call_success and put_success:
            position.status = StraddleStatus.OPEN
            position.current_value = total_cost
            position.peak_value = total_cost
            self.active_straddles[straddle_id] = position

            logger.info(f"STRADDLE_OPEN: {straddle_id} {ticker} {strike} "
                       f"call@${position.call_leg.entry_price:.2f} "
                       f"put@${position.put_leg.entry_price:.2f} "
                       f"total_cost=${total_cost:.2f}")

            return True, f"Straddle opened: {straddle_id}", position

        elif call_success or put_success:
            position.status = StraddleStatus.PARTIAL
            self.active_straddles[straddle_id] = position
            return False, "Partial fill - one leg failed", position

        else:
            position.status = StraddleStatus.FAILED
            return False, "Both legs failed to fill", position

    def _execute_leg(self, leg: StraddleLeg) -> bool:
        """Execute a single leg order."""
        if not self.executor:
            logger.warning("No executor configured, simulating fill")
            leg.filled = True
            leg.order_id = f"SIM_{leg.symbol}"
            return True

        try:
            # Use smart order execution if available
            from wsb_snake.execution.smart_order_executor import get_smart_executor, OrderSide

            smart_exec = get_smart_executor()
            smart_exec.set_executor(self.executor)

            result = smart_exec.smart_entry(leg.symbol, OrderSide.BUY, leg.quantity)

            if result.success:
                leg.filled = True
                leg.entry_price = result.fill_price
                leg.order_id = result.order_id
                return True

            return False

        except Exception as e:
            logger.error(f"Leg execution failed: {e}")
            return False

    def _get_spot_price(self, ticker: str) -> Optional[float]:
        """Get current spot price for ticker."""
        try:
            if self.executor and hasattr(self.executor, 'trading_client'):
                quote = self.executor.trading_client.get_latest_quote(ticker)
                return float(quote.ask_price + quote.bid_price) / 2
        except Exception as e:
            logger.error(f"Failed to get spot price: {e}")

        return None

    def _get_option_price(self, symbol: str) -> Optional[float]:
        """Get current mid price for option."""
        try:
            if self.executor and hasattr(self.executor, 'trading_client'):
                quote = self.executor.trading_client.get_latest_quote(symbol)
                return float(quote.ask_price + quote.bid_price) / 2
        except Exception as e:
            logger.debug(f"Failed to get option price for {symbol}: {e}")

        # Return a placeholder for testing
        return None

    def update_prices(self, straddle_id: str) -> Optional[Dict]:
        """
        Update current prices and check exit conditions.

        Returns:
            Dict with current P&L info or None if not found
        """
        if straddle_id not in self.active_straddles:
            return None

        position = self.active_straddles[straddle_id]

        # Get current prices
        call_price = self._get_option_price(position.call_leg.symbol)
        put_price = self._get_option_price(position.put_leg.symbol)

        if call_price and put_price:
            position.call_leg.current_price = call_price
            position.put_leg.current_price = put_price

            current_value = (call_price + put_price) * 100 * position.call_leg.quantity
            position.current_value = current_value
            position.peak_value = max(position.peak_value, current_value)

        # Calculate P&L
        pnl_dollars = position.current_value - position.total_cost
        pnl_pct = (pnl_dollars / position.total_cost * 100) if position.total_cost > 0 else 0

        # Check exit conditions
        exit_reason = self._check_exit_conditions(position, pnl_pct)

        return {
            'straddle_id': straddle_id,
            'ticker': position.ticker,
            'strike': position.strike,
            'call_price': position.call_leg.current_price,
            'put_price': position.put_leg.current_price,
            'total_cost': position.total_cost,
            'current_value': position.current_value,
            'pnl_dollars': pnl_dollars,
            'pnl_pct': pnl_pct,
            'peak_value': position.peak_value,
            'exit_reason': exit_reason,
            'should_exit': exit_reason is not None
        }

    def _check_exit_conditions(self, position: StraddlePosition, pnl_pct: float) -> Optional[str]:
        """
        Check if position should be closed.

        Returns:
            Exit reason string or None if should hold
        """
        # Target hit
        if pnl_pct >= self.TARGET_PCT:
            return f"TARGET ({pnl_pct:.1f}% >= {self.TARGET_PCT}%)"

        # Stop hit
        if pnl_pct <= self.STOP_PCT:
            return f"STOP ({pnl_pct:.1f}% <= {self.STOP_PCT}%)"

        # Trailing stop
        if position.current_value < position.peak_value:
            peak_pnl_pct = ((position.peak_value - position.total_cost) / position.total_cost * 100)
            current_from_peak = ((position.current_value - position.peak_value) / position.peak_value * 100)

            if peak_pnl_pct >= self.TRAIL_TRIGGER_PCT and current_from_peak <= -self.TRAIL_DISTANCE_PCT:
                return f"TRAIL (peak={peak_pnl_pct:.1f}%, dropped {abs(current_from_peak):.1f}%)"

        # Time stop
        hold_seconds = (datetime.now(timezone.utc) - position.opened_at).total_seconds()
        if hold_seconds >= self.MAX_HOLD_SECONDS:
            return f"TIME ({hold_seconds/60:.0f} min)"

        return None

    def close_straddle(self, straddle_id: str, reason: str = "MANUAL") -> Tuple[bool, Dict]:
        """
        Close a straddle position.

        Args:
            straddle_id: ID of the straddle to close
            reason: Reason for closing

        Returns:
            (success, result_dict)
        """
        if straddle_id not in self.active_straddles:
            return False, {'error': 'Straddle not found'}

        position = self.active_straddles[straddle_id]

        # Close both legs
        call_success = self._close_leg(position.call_leg)
        put_success = self._close_leg(position.put_leg)

        # Calculate final P&L
        call_pnl = (position.call_leg.exit_price - position.call_leg.entry_price) * 100 * position.call_leg.quantity
        put_pnl = (position.put_leg.exit_price - position.put_leg.entry_price) * 100 * position.put_leg.quantity
        net_pnl = call_pnl + put_pnl
        return_pct = (net_pnl / position.total_cost * 100) if position.total_cost > 0 else 0

        position.status = StraddleStatus.CLOSED
        position.closed_at = datetime.now(timezone.utc)
        position.exit_reason = reason

        # Log the close
        logger.info(f"STRADDLE_CLOSE: {straddle_id} call_pnl=${call_pnl:.2f} "
                   f"put_pnl=${put_pnl:.2f} net=${net_pnl:.2f} return={return_pct:.1f}%")

        # Move to closed list
        self.closed_straddles.append(position)
        del self.active_straddles[straddle_id]

        return True, {
            'straddle_id': straddle_id,
            'ticker': position.ticker,
            'strike': position.strike,
            'call_pnl': call_pnl,
            'put_pnl': put_pnl,
            'net_pnl': net_pnl,
            'return_pct': return_pct,
            'exit_reason': reason,
            'hold_seconds': (position.closed_at - position.opened_at).total_seconds()
        }

    def _close_leg(self, leg: StraddleLeg) -> bool:
        """Close a single leg."""
        if not leg.filled or leg.closed:
            return True

        try:
            if self.executor:
                from wsb_snake.execution.smart_order_executor import get_smart_executor, OrderSide

                smart_exec = get_smart_executor()
                smart_exec.set_executor(self.executor)

                result = smart_exec.smart_exit_profit(leg.symbol, leg.quantity)

                if result.success:
                    leg.exit_price = result.fill_price
                    leg.closed = True
                    return True

            # Simulation fallback
            leg.exit_price = leg.current_price or leg.entry_price
            leg.closed = True
            return True

        except Exception as e:
            logger.error(f"Failed to close leg: {e}")
            return False

    def monitor_all(self) -> List[Dict]:
        """
        Monitor all active straddles and close any that hit exit conditions.

        Returns:
            List of closed position results
        """
        closed_results = []

        for straddle_id in list(self.active_straddles.keys()):
            status = self.update_prices(straddle_id)

            if status and status['should_exit']:
                success, result = self.close_straddle(straddle_id, status['exit_reason'])
                if success:
                    closed_results.append(result)

        return closed_results

    def get_active_positions(self) -> List[Dict]:
        """Get summary of all active straddle positions."""
        positions = []
        for straddle_id, pos in self.active_straddles.items():
            status = self.update_prices(straddle_id)
            if status:
                positions.append(status)
        return positions


# Singleton instance
_straddle_executor: Optional[StraddleExecutor] = None


def get_straddle_executor() -> StraddleExecutor:
    """Get the singleton StraddleExecutor instance."""
    global _straddle_executor
    if _straddle_executor is None:
        _straddle_executor = StraddleExecutor()
    return _straddle_executor


def execute_straddle(ticker: str, expiry: str, size: float) -> Tuple[bool, str]:
    """
    Convenience function to execute a straddle.

    Usage:
        from wsb_snake.execution.straddle_executor import execute_straddle
        success, msg = execute_straddle("SPY", "2026-02-25", 1000)
    """
    executor = get_straddle_executor()
    success, msg, _ = executor.execute_straddle(ticker, expiry, size)
    return success, msg
