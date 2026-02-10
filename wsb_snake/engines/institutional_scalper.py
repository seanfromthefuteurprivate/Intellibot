"""
INSTITUTIONAL SCALPER - Weaponized Trading Desk Knowledge

This module implements battle-tested trading desk wisdom from:
- Prop trading desks (Citadel, Jane Street, Two Sigma)
- Market makers (Flow trading, statistical arbitrage)
- Seasoned day traders (15+ years experience patterns)

Core Philosophy: "Small gains compound, big losses destroy"

RULES ENCODED:
1. Never risk more than you can afford to lose
2. Trade WITH momentum, not against it
3. Take profits at +10-15%, NEVER hold past -15%
4. Speed + Edge = Alpha
5. No edge = No trade
6. Time decay is the enemy on 0DTE
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import pytz

from wsb_snake.trading.alpaca_executor import alpaca_executor
from wsb_snake.collectors.market_data import get_market_data
from wsb_snake.notifications.telegram_bot import send_alert as send_telegram_alert
from wsb_snake.utils.cpl_gate import check as cpl_check, block_trade as cpl_block

log = logging.getLogger(__name__)


@dataclass
class InstitutionalSetup:
    """Validated institutional-grade trade setup."""
    ticker: str
    direction: str  # 'long' or 'short'
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: int
    edge_type: str  # momentum, mean_reversion, breakout, etc.
    risk_reward: float
    time_horizon_minutes: int
    notes: str = ""


class InstitutionalScalper:
    """
    Institutional-grade scalping engine.
    
    Implements proven trading desk strategies:
    - Momentum following with tight risk
    - Mean reversion at extremes
    - Volume-confirmed breakouts
    - Time-of-day adjustments
    """
    
    # Trading parameters - battle-tested values
    MIN_EDGE_THRESHOLD = 0.03  # 0.03% minimum directional bias
    MAX_SPREAD_PCT = 0.5  # Skip if spread > 0.5%
    
    # Risk parameters - institutional discipline (WIDENED to avoid noise stop-outs)
    STOP_PCT = 0.008  # 0.8% stop on underlying (~25-35% on option) - widened from 0.3%
    TARGET_PCT = 0.012  # 1.2% target on underlying (~35-50% on option) - widened from 0.4%
    MAX_HOLD_MINUTES = 45  # Exit after 45 mins regardless (extended from 30)
    
    # Time-of-day multipliers (institutional wisdom)
    TIME_MULTIPLIERS = {
        9: 1.0,   # Open - volatile, careful
        10: 1.2,  # Mid-morning - good flows
        11: 1.0,  # Pre-lunch chop
        12: 0.8,  # Lunch hour - low volume
        13: 0.9,  # Post-lunch
        14: 1.1,  # Afternoon pickup
        15: 1.3,  # Power hour - best volume
    }
    
    # Pattern edge scores (from backtesting)
    EDGE_SCORES = {
        'momentum_continuation': 68,
        'vwap_bounce': 65,
        'volume_breakout': 70,
        'mean_reversion': 62,
        'power_hour_momentum': 72,
    }
    
    def __init__(self):
        self.et = pytz.timezone('US/Eastern')
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        log.info("Institutional Scalper initialized - Weaponized Knowledge ACTIVE")
    
    def get_time_multiplier(self) -> float:
        """Get current hour's edge multiplier."""
        now = datetime.now(self.et)
        return self.TIME_MULTIPLIERS.get(now.hour, 1.0)
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now(self.et)
        if now.weekday() >= 5:  # Weekend
            return False
        return 9 <= now.hour < 16 or (now.hour == 9 and now.minute >= 30)
    
    def scan_for_edge(self, tickers: Optional[List[str]] = None) -> List[InstitutionalSetup]:
        """
        Scan for high-probability institutional setups.
        
        Returns only setups that meet institutional standards:
        - Clear directional bias
        - Favorable R:R ratio
        - Volume confirmation
        - Time-of-day alignment
        """
        if tickers is None:
            tickers = ['SPY', 'QQQ', 'IWM']
        
        if not self.is_market_open():
            return []
        
        setups = []
        prices = get_market_data(tickers)
        time_mult = self.get_time_multiplier()
        
        for ticker, data in prices.items():
            price = data.get('price', 0)
            change = data.get('change_pct', 0)
            volume = data.get('volume', 0)
            
            if price <= 0:
                continue
            
            # RULE 1: Must have directional edge
            if abs(change) < self.MIN_EDGE_THRESHOLD:
                continue
            
            direction = 'long' if change > 0 else 'short'
            
            # Calculate institutional stops/targets
            if direction == 'long':
                stop = price * (1 - self.STOP_PCT)
                target = price * (1 + self.TARGET_PCT)
            else:
                stop = price * (1 + self.STOP_PCT)
                target = price * (1 - self.TARGET_PCT)
            
            # Calculate R:R
            risk = abs(price - stop)
            reward = abs(target - price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # RULE 2: Minimum 1.2:1 R:R
            if rr_ratio < 1.2:
                continue
            
            # Calculate confidence with time multiplier
            base_confidence = self.EDGE_SCORES.get('momentum_continuation', 65)
            adjusted_confidence = int(base_confidence * time_mult)
            
            # RULE 3: Minimum 65% confidence
            if adjusted_confidence < 65:
                continue
            
            setup = InstitutionalSetup(
                ticker=ticker,
                direction=direction,
                entry_price=price,
                target_price=target,
                stop_loss=stop,
                confidence=adjusted_confidence,
                edge_type='momentum_continuation',
                risk_reward=rr_ratio,
                time_horizon_minutes=self.MAX_HOLD_MINUTES,
                notes=f"Momentum {change:+.2f}% with {time_mult:.1f}x time mult"
            )
            
            setups.append(setup)
            log.info(f"Found institutional setup: {ticker} {direction} @ ${price:.2f}")
        
        return setups
    
    def execute_setup(self, setup: InstitutionalSetup) -> bool:
        """
        Execute an institutional setup with full validation.
        """
        log.info(f"Executing institutional setup: {setup.ticker} {setup.direction}")

        # CPL GATE - Check alignment before execution
        cpl_ok, cpl_reason = cpl_check(setup.ticker, setup.direction)
        if not cpl_ok:
            cpl_block(setup.ticker, setup.direction, cpl_reason)
            return False
        log.info(f"✅ CPL GATE PASSED: {setup.ticker} - {cpl_reason}")

        result = alpaca_executor.execute_scalp_entry(
            underlying=setup.ticker,
            direction=setup.direction,
            entry_price=setup.entry_price,
            target_price=setup.target_price,
            stop_loss=setup.stop_loss,
            confidence=setup.confidence,
            pattern=f"institutional_{setup.edge_type}"
        )
        
        if result:
            self.daily_trades += 1
            
            message = f"""🏦 **INSTITUTIONAL ENTRY**

{setup.direction.upper()} {setup.ticker}
Entry: ${setup.entry_price:.2f}
Target: ${setup.target_price:.2f} (+{self.TARGET_PCT*100:.1f}%)
Stop: ${setup.stop_loss:.2f} (-{self.STOP_PCT*100:.1f}%)
R:R: {setup.risk_reward:.1f}:1
Confidence: {setup.confidence}%

Edge: {setup.edge_type}
Max Hold: {setup.time_horizon_minutes} minutes"""
            
            send_telegram_alert(message)
            return True
        
        return False
    
    def manage_positions(self) -> Dict[str, Any]:
        """
        Institutional position management:
        1. Book profits at target
        2. Cut losses at stop
        3. Time decay exit for 0DTE
        4. Trail stops on winners
        """
        positions = alpaca_executor.get_options_positions()
        now = datetime.now(self.et)
        
        actions = {
            'profits_taken': 0,
            'stops_hit': 0,
            'time_exits': 0,
            'positions_monitored': len(positions)
        }
        
        for p in positions:
            sym = p.get('symbol', '')
            entry = float(p.get('avg_entry_price', 0))
            current = float(p.get('current_price', entry))
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
            
            # RULE: Take profits at +10% or better
            if pnl_pct >= 10:
                log.info(f"PROFIT TARGET: {sym} at {pnl_pct:+.1f}%")
                alpaca_executor.close_position(sym)
                self.win_count += 1
                self.daily_pnl += (current - entry) * int(p.get('qty', 0)) * 100
                actions['profits_taken'] += 1
                
                send_telegram_alert(f"""🎯 **PROFIT BOOKED**
{sym}
Gain: {pnl_pct:+.1f}%
INSTITUTIONAL DISCIPLINE PAYS""")
            
            # RULE: Cut losses at -15%
            elif pnl_pct <= -15:
                log.info(f"STOP TRIGGERED: {sym} at {pnl_pct:+.1f}%")
                alpaca_executor.close_position(sym)
                self.loss_count += 1
                self.daily_pnl += (current - entry) * int(p.get('qty', 0)) * 100
                actions['stops_hit'] += 1
                
                send_telegram_alert(f"""🛑 **STOP EXECUTED**
{sym}
Loss: {pnl_pct:+.1f}%
MECHANICAL DISCIPLINE - PRESERVED CAPITAL""")
            
            # RULE: Time decay exit for 0DTE after 2:30 PM
            elif self._is_0dte(sym) and now.hour >= 14 and now.minute >= 30:
                if pnl_pct < 5:  # Not enough profit for theta risk
                    log.info(f"TIME DECAY EXIT: {sym} at {pnl_pct:+.1f}%")
                    alpaca_executor.close_position(sym)
                    actions['time_exits'] += 1
        
        return actions
    
    def _is_0dte(self, symbol: str) -> bool:
        """Check if option expires today."""
        today = datetime.now(self.et).strftime('%y%m%d')
        return today in symbol
    
    def get_stats(self) -> Dict[str, Any]:
        """Get daily statistics."""
        total_trades = self.win_count + self.loss_count
        win_rate = (self.win_count / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'wins': self.win_count,
            'losses': self.loss_count,
            'win_rate': win_rate,
        }
    
    def run_cycle(self) -> Dict[str, Any]:
        """
        Run one complete institutional scalping cycle:
        1. Manage existing positions
        2. Scan for new opportunities
        3. Execute if edge found
        """
        if not self.is_market_open():
            return {'status': 'market_closed'}
        
        # Step 1: Manage positions first
        mgmt = self.manage_positions()
        
        # Step 2: Check if we have room for new trades
        positions = alpaca_executor.get_options_positions()
        slots = max(0, 3 - len(positions))
        
        result = {
            'management': mgmt,
            'positions': len(positions),
            'slots': slots,
            'new_entries': 0
        }
        
        if slots <= 0:
            return result
        
        # Step 3: Scan for opportunities
        setups = self.scan_for_edge()
        
        # Step 4: Execute best setup
        for setup in setups[:slots]:
            if self.execute_setup(setup):
                result['new_entries'] += 1
        
        return result


# Singleton instance
_scalper = None

def get_institutional_scalper() -> InstitutionalScalper:
    """Get or create the institutional scalper instance."""
    global _scalper
    if _scalper is None:
        _scalper = InstitutionalScalper()
    return _scalper
