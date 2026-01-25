"""
Zero Greed Exit Protocol - Mechanical ruthless exit system.

No human override. No "let it run". No "maybe it recovers".
Pure machine execution for 0DTE scalping.

Rules:
1. Target hit = IMMEDIATE EXIT (book the profit)
2. Stop hit = IMMEDIATE EXIT (accept the loss)
3. Time decay = EXIT at deadline (theta kills 0DTE)
"""
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from wsb_snake.utils.logger import get_logger
from wsb_snake.notifications.telegram_bot import send_alert as send_telegram_alert

logger = get_logger(__name__)


class ExitReason(Enum):
    """Why we exited the trade."""
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    TIME_DECAY = "TIME_DECAY"
    MANUAL_EXIT = "MANUAL_EXIT"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass
class TrackedPosition:
    """A position being tracked for automatic exit."""
    position_id: str
    ticker: str
    direction: str  # 'long' or 'short'
    trade_type: str  # 'CALLS' or 'PUTS'
    entry_price: float
    target_price: float
    stop_loss: float
    entry_time: datetime
    max_hold_minutes: int = 60
    price_getter: Optional[Callable] = None
    alerted_target: bool = False
    alerted_stop: bool = False
    alerted_time: bool = False
    exit_reason: Optional[ExitReason] = None
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    
    @property
    def is_active(self) -> bool:
        """Position is still being tracked."""
        return self.exit_reason is None
    
    @property
    def time_remaining_minutes(self) -> float:
        """Minutes until time decay exit."""
        elapsed = (datetime.utcnow() - self.entry_time).total_seconds() / 60
        return max(0, self.max_hold_minutes - elapsed)
    
    @property
    def pnl_percent(self) -> float:
        """Calculate current P&L percent if exit_price is set."""
        if self.exit_price is None:
            return 0.0
        
        if self.direction == 'long':
            return ((self.exit_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.exit_price) / self.entry_price) * 100


class ZeroGreedExit:
    """
    Mechanical exit system with zero human override.
    
    Monitors positions and sends EXIT alerts the moment:
    - Target price is hit
    - Stop loss is hit
    - Time decay threshold reached
    """
    
    def __init__(self):
        self.positions: Dict[str, TrackedPosition] = {}
        self.check_interval = 5  # seconds between checks
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self.time_decay_warning_pct = 0.75
        self.time_decay_exit_pct = 1.0
        
        logger.info("Zero Greed Exit Protocol initialized - NO MERCY MODE ACTIVE")
    
    def add_position(
        self,
        position_id: str,
        ticker: str,
        direction: str,
        trade_type: str,
        entry_price: float,
        target_price: float,
        stop_loss: float,
        max_hold_minutes: int = 60,
        price_getter: Optional[Callable] = None
    ) -> bool:
        """
        Add a position to track for automatic exit.
        
        Args:
            position_id: Unique identifier for this position
            ticker: Stock ticker
            direction: 'long' or 'short'
            trade_type: 'CALLS' or 'PUTS'
            entry_price: Entry price
            target_price: Target exit price for profit
            stop_loss: Stop loss price
            max_hold_minutes: Maximum time to hold before time decay exit
            price_getter: Callable that returns current price for ticker
            
        Returns:
            True if added successfully
        """
        with self._lock:
            if position_id in self.positions:
                logger.warning(f"Position {position_id} already being tracked")
                return False
            
            position = TrackedPosition(
                position_id=position_id,
                ticker=ticker,
                direction=direction,
                trade_type=trade_type,
                entry_price=entry_price,
                target_price=target_price,
                stop_loss=stop_loss,
                entry_time=datetime.utcnow(),
                max_hold_minutes=max_hold_minutes,
                price_getter=price_getter
            )
            
            self.positions[position_id] = position
            
            logger.info(f"🎯 Tracking position {position_id}: {ticker} {direction} @ ${entry_price:.2f}")
            logger.info(f"   Target: ${target_price:.2f} | Stop: ${stop_loss:.2f} | Max hold: {max_hold_minutes}min")
            
            return True
    
    def remove_position(self, position_id: str) -> Optional[TrackedPosition]:
        """Remove a position from tracking."""
        with self._lock:
            return self.positions.pop(position_id, None)
    
    def _check_position(self, position: TrackedPosition, current_price: float) -> Optional[ExitReason]:
        """
        Check if position should exit.
        
        Returns ExitReason if should exit, None otherwise.
        """
        if position.direction == 'long':
            if current_price >= position.target_price:
                return ExitReason.TARGET_HIT
            if current_price <= position.stop_loss:
                return ExitReason.STOP_HIT
        else:  # short
            if current_price <= position.target_price:
                return ExitReason.TARGET_HIT
            if current_price >= position.stop_loss:
                return ExitReason.STOP_HIT
        
        time_used_pct = 1 - (position.time_remaining_minutes / position.max_hold_minutes)
        if time_used_pct >= self.time_decay_exit_pct:
            return ExitReason.TIME_DECAY
        
        return None
    
    def _send_exit_alert(self, position: TrackedPosition, current_price: float, reason: ExitReason):
        """Send ruthless exit alert via Telegram."""
        position.exit_price = current_price
        position.exit_time = datetime.utcnow()
        position.exit_reason = reason
        
        pnl = position.pnl_percent
        pnl_emoji = "💰" if pnl > 0 else "🔻"
        
        if reason == ExitReason.TARGET_HIT:
            header = "🎯 TARGET HIT - BOOK PROFIT NOW 🎯"
            action = "TAKE THE MONEY AND RUN"
        elif reason == ExitReason.STOP_HIT:
            header = "🛑 STOP HIT - EXIT NOW 🛑"
            action = "ACCEPT THE LOSS - NO AVERAGING DOWN"
        elif reason == ExitReason.TIME_DECAY:
            header = "⏰ TIME DECAY EXIT ⏰"
            action = "THETA IS KILLING YOU - EXIT NOW"
        else:
            header = "⚠️ SYSTEM EXIT ⚠️"
            action = "EXIT IMMEDIATELY"
        
        hold_time = (position.exit_time - position.entry_time).total_seconds() / 60
        
        alert_msg = f"""
{'='*40}
{header}
{'='*40}

📊 {position.ticker} {position.trade_type}
{'📈' if position.direction == 'long' else '📉'} Direction: {position.direction.upper()}

💵 ENTRY: ${position.entry_price:.2f}
💵 EXIT: ${current_price:.2f}
{pnl_emoji} P&L: {pnl:+.1f}%

⏱️ Hold Time: {hold_time:.1f} min

{'='*40}
🚨 {action} 🚨
{'='*40}

NO GREED. NO FEAR. EXECUTE.
"""
        
        send_telegram_alert(alert_msg)
        
        logger.info(f"EXIT ALERT: {position.position_id} - {reason.value} @ ${current_price:.2f} ({pnl:+.1f}%)")
    
    def _send_time_warning(self, position: TrackedPosition):
        """Send time decay warning (75% of max hold reached)."""
        remaining = position.time_remaining_minutes
        
        alert_msg = f"""
⏰ TIME DECAY WARNING ⏰

{position.ticker} {position.trade_type}
Entry: ${position.entry_price:.2f}

⚠️ Only {remaining:.0f} min remaining!

Consider exiting if not near target.
"""
        
        send_telegram_alert(alert_msg)
        logger.info(f"Time warning sent for {position.position_id}: {remaining:.0f}min left")
    
    def _monitor_loop(self):
        """Main monitoring loop - runs every check_interval seconds."""
        logger.info("Zero Greed Exit monitoring started")
        
        while self.running:
            try:
                with self._lock:
                    active_positions = [p for p in self.positions.values() if p.is_active]
                
                for position in active_positions:
                    try:
                        if position.price_getter:
                            current_price = position.price_getter(position.ticker)
                        else:
                            continue
                        
                        if current_price is None:
                            continue
                        
                        exit_reason = self._check_position(position, current_price)
                        
                        if exit_reason:
                            self._send_exit_alert(position, current_price, exit_reason)
                        else:
                            time_used_pct = 1 - (position.time_remaining_minutes / position.max_hold_minutes)
                            if time_used_pct >= self.time_decay_warning_pct and not position.alerted_time:
                                self._send_time_warning(position)
                                position.alerted_time = True
                                
                    except Exception as e:
                        logger.error(f"Error checking position {position.position_id}: {e}")
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
            
            time.sleep(self.check_interval)
        
        logger.info("Zero Greed Exit monitoring stopped")
    
    def start(self):
        """Start the monitoring thread."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔪 Zero Greed Exit Protocol ACTIVATED")
    
    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("Zero Greed Exit Protocol stopped")
    
    def get_active_positions(self) -> List[TrackedPosition]:
        """Get all currently tracked positions."""
        with self._lock:
            return [p for p in self.positions.values() if p.is_active]
    
    def get_stats(self) -> Dict:
        """Get exit system statistics."""
        with self._lock:
            all_positions = list(self.positions.values())
            exited = [p for p in all_positions if not p.is_active]
            
            wins = [p for p in exited if p.pnl_percent > 0]
            losses = [p for p in exited if p.pnl_percent <= 0]
            
            target_exits = [p for p in exited if p.exit_reason == ExitReason.TARGET_HIT]
            stop_exits = [p for p in exited if p.exit_reason == ExitReason.STOP_HIT]
            time_exits = [p for p in exited if p.exit_reason == ExitReason.TIME_DECAY]
            
            return {
                'active_positions': len([p for p in all_positions if p.is_active]),
                'total_trades': len(exited),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': len(wins) / len(exited) * 100 if exited else 0,
                'avg_win_pct': sum(p.pnl_percent for p in wins) / len(wins) if wins else 0,
                'avg_loss_pct': sum(p.pnl_percent for p in losses) / len(losses) if losses else 0,
                'target_exits': len(target_exits),
                'stop_exits': len(stop_exits),
                'time_exits': len(time_exits)
            }
    
    def manual_exit(self, position_id: str, exit_price: float) -> bool:
        """
        Manually trigger exit for a position.
        Use sparingly - defeats the purpose of zero greed!
        """
        with self._lock:
            if position_id not in self.positions:
                return False
            
            position = self.positions[position_id]
            if not position.is_active:
                return False
            
            self._send_exit_alert(position, exit_price, ExitReason.MANUAL_EXIT)
            logger.warning(f"MANUAL EXIT triggered for {position_id} - review if this was necessary")
            return True


zero_greed_exit = ZeroGreedExit()
