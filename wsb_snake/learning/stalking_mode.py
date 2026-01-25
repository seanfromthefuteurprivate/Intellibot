"""
Stalking Mode - Pre-positions engine to watch specific setups before trigger.

The engine "lurks" on setups that are building, waiting for the right moment.
Tracks:
- Setups that are X% away from triggering
- Pre-alerts when setup is forming
- Historical timing of triggers
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum
import json

from wsb_snake.db.database import get_connection
from wsb_snake.utils.logger import log


class StalkState(str, Enum):
    """States of a stalked setup."""
    WATCHING = "watching"      # Just observing
    COILING = "coiling"        # Building pressure
    READY = "ready"            # About to trigger
    TRIGGERED = "triggered"    # Has triggered
    EXPIRED = "expired"        # Setup died


@dataclass
class StalkedSetup:
    """A setup being watched."""
    setup_id: str
    symbol: str
    setup_type: str  # "breakout", "earnings_play", "macro_event", etc.
    
    # Trigger conditions
    trigger_price: Optional[float] = None
    trigger_condition: str = ""  # "above_resistance", "below_support", etc.
    trigger_time: Optional[str] = None  # Time-based trigger
    
    # Current state
    state: StalkState = StalkState.WATCHING
    distance_to_trigger: float = 100.0  # Percentage away
    
    # Build-up tracking
    coil_score: float = 0.0  # How "wound up" is the setup
    pressure_direction: str = "neutral"  # "bullish", "bearish"
    volume_building: bool = False
    
    # Alerts
    alert_at_pct: float = 5.0  # Alert when X% away
    alerted: bool = False
    
    # Timing
    created_at: str = ""
    expires_at: str = ""  # When to stop watching
    last_updated: str = ""
    
    # Context
    catalyst: str = ""  # What's the catalyst
    notes: str = ""
    expected_move: float = 0.0
    
    # Trade Details (for actionable signals)
    entry_price: Optional[float] = None  # Exact entry price
    target_price: Optional[float] = None  # Take profit price
    stop_loss: Optional[float] = None  # Stop loss price
    direction: str = "long"  # "long" or "short"
    trade_type: str = "CALLS"  # "CALLS", "PUTS", "STOCK"
    position_size_pct: float = 2.0  # Suggested position size %
    max_risk_pct: float = 1.0  # Max risk %
    
    # Entry/Exit tracking
    entry_alerted: bool = False  # Entry signal sent
    entry_filled_price: Optional[float] = None  # Actual fill price
    entry_filled_at: Optional[str] = None  # When filled
    exit_alerted: bool = False  # Exit signal sent
    trailing_stop: bool = False  # Use trailing stop


@dataclass
class StalkAlert:
    """Alert for a stalked setup."""
    setup_id: str
    symbol: str
    alert_type: str  # "approaching", "ready", "triggered", "entry", "exit"
    message: str
    urgency: int  # 1-5
    timestamp: str
    
    # Trade details (for actionable alerts)
    ticker: str = ""
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    direction: str = ""  # "long" or "short"
    trade_type: str = ""  # "CALLS", "PUTS", "STOCK"
    action: str = ""  # "BUY", "SELL"
    
    def format_telegram_message(self) -> str:
        """Format as complete Telegram alert."""
        if self.alert_type == "entry":
            risk_reward = 0
            entry = self.entry_price or 0
            target = self.target_price or 0
            stop = self.stop_loss or 0
            
            if entry and target and stop:
                reward = abs(target - entry)
                risk = abs(entry - stop)
                risk_reward = reward / risk if risk > 0 else 0
            
            return (
                f"{'=' * 40}\n"
                f"{'BUY' if self.direction == 'long' else 'SELL'} {self.trade_type or 'STOCK'} - {self.symbol}\n"
                f"{'=' * 40}\n"
                f"ENTRY: ${entry:.2f}\n"
                f"TARGET: ${target:.2f}\n"
                f"STOP: ${stop:.2f}\n"
                f"R:R = 1:{risk_reward:.1f}\n"
                f"{'=' * 40}\n"
                f"{self.message}\n"
                f"{'=' * 40}"
            )
        elif self.alert_type == "exit":
            exit_price = self.target_price or 0
            return (
                f"{'=' * 40}\n"
                f"EXIT {self.trade_type or 'STOCK'} - {self.symbol}\n"
                f"{'=' * 40}\n"
                f"CLOSE @ ${exit_price:.2f}\n"
                f"ACTION: BOOK PROFIT NOW\n"
                f"{'=' * 40}\n"
                f"{self.message}\n"
                f"{'=' * 40}"
            )
        else:
            return self.message


class StalkingMode:
    """
    Watches setups and alerts when they're about to trigger.
    """
    
    MAX_STALKED = 20  # Maximum setups to watch at once
    
    def __init__(self):
        self._init_tables()
        self.stalked: Dict[str, StalkedSetup] = {}
        self.alerts: List[StalkAlert] = []
        self._load_stalked()
        log.info("Stalking Mode initialized")
    
    def _init_tables(self):
        """Create stalking tables."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stalked_setups (
                setup_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                setup_type TEXT NOT NULL,
                trigger_price REAL,
                trigger_condition TEXT,
                trigger_time TEXT,
                state TEXT DEFAULT 'watching',
                distance_to_trigger REAL DEFAULT 100,
                coil_score REAL DEFAULT 0,
                pressure_direction TEXT DEFAULT 'neutral',
                volume_building INTEGER DEFAULT 0,
                alert_at_pct REAL DEFAULT 5,
                alerted INTEGER DEFAULT 0,
                created_at TEXT,
                expires_at TEXT,
                last_updated TEXT,
                catalyst TEXT,
                notes TEXT,
                expected_move REAL DEFAULT 0,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                direction TEXT DEFAULT 'long',
                trade_type TEXT DEFAULT 'CALLS',
                position_size_pct REAL DEFAULT 2.0,
                max_risk_pct REAL DEFAULT 1.0,
                entry_alerted INTEGER DEFAULT 0,
                entry_filled_price REAL,
                entry_filled_at TEXT,
                exit_alerted INTEGER DEFAULT 0,
                trailing_stop INTEGER DEFAULT 0
            )
        """)
        
        # Add new columns if they don't exist (for migration)
        try:
            cursor.execute("ALTER TABLE stalked_setups ADD COLUMN entry_price REAL")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE stalked_setups ADD COLUMN target_price REAL")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE stalked_setups ADD COLUMN stop_loss REAL")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE stalked_setups ADD COLUMN direction TEXT DEFAULT 'long'")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE stalked_setups ADD COLUMN trade_type TEXT DEFAULT 'CALLS'")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE stalked_setups ADD COLUMN entry_alerted INTEGER DEFAULT 0")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE stalked_setups ADD COLUMN entry_filled_price REAL")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE stalked_setups ADD COLUMN entry_filled_at TEXT")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE stalked_setups ADD COLUMN exit_alerted INTEGER DEFAULT 0")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE stalked_setups ADD COLUMN position_size_pct REAL DEFAULT 2.0")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE stalked_setups ADD COLUMN max_risk_pct REAL DEFAULT 1.0")
        except:
            pass
        try:
            cursor.execute("ALTER TABLE stalked_setups ADD COLUMN trailing_stop INTEGER DEFAULT 0")
        except:
            pass
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stalk_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                setup_id TEXT,
                symbol TEXT,
                setup_type TEXT,
                trigger_time TEXT,
                outcome TEXT,
                pnl_pct REAL,
                stalk_duration_mins INTEGER
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_stalked(self):
        """Load active stalked setups."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM stalked_setups
            WHERE state IN ('watching', 'coiling', 'ready')
        """)
        rows = cursor.fetchall()
        
        for row in rows:
            setup = StalkedSetup(
                setup_id=row["setup_id"],
                symbol=row["symbol"],
                setup_type=row["setup_type"],
                trigger_price=row["trigger_price"],
                trigger_condition=row["trigger_condition"],
                trigger_time=row["trigger_time"],
                state=StalkState(row["state"]),
                distance_to_trigger=row["distance_to_trigger"],
                coil_score=row["coil_score"],
                pressure_direction=row["pressure_direction"],
                volume_building=bool(row["volume_building"]),
                alert_at_pct=row["alert_at_pct"],
                alerted=bool(row["alerted"]),
                created_at=row["created_at"],
                expires_at=row["expires_at"],
                last_updated=row["last_updated"],
                catalyst=row["catalyst"] or "",
                notes=row["notes"] or "",
                expected_move=row["expected_move"],
                entry_price=row["entry_price"] if "entry_price" in row.keys() else None,
                target_price=row["target_price"] if "target_price" in row.keys() else None,
                stop_loss=row["stop_loss"] if "stop_loss" in row.keys() else None,
                direction=row["direction"] if "direction" in row.keys() else "long",
                trade_type=row["trade_type"] if "trade_type" in row.keys() else "CALLS",
                entry_alerted=bool(row["entry_alerted"]) if "entry_alerted" in row.keys() else False,
                entry_filled_price=row["entry_filled_price"] if "entry_filled_price" in row.keys() else None,
                entry_filled_at=row["entry_filled_at"] if "entry_filled_at" in row.keys() else None,
                exit_alerted=bool(row["exit_alerted"]) if "exit_alerted" in row.keys() else False,
                position_size_pct=row["position_size_pct"] if "position_size_pct" in row.keys() else 2.0,
                max_risk_pct=row["max_risk_pct"] if "max_risk_pct" in row.keys() else 1.0,
                trailing_stop=bool(row["trailing_stop"]) if "trailing_stop" in row.keys() else False
            )
            self.stalked[setup.setup_id] = setup
        
        conn.close()
        log.info(f"Loaded {len(self.stalked)} stalked setups")
    
    def add_setup(
        self,
        symbol: str,
        setup_type: str,
        trigger_price: Optional[float] = None,
        trigger_condition: str = "",
        trigger_time: Optional[str] = None,
        catalyst: str = "",
        expected_move: float = 0.0,
        alert_at_pct: float = 5.0,
        expires_hours: int = 24,
        entry_price: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        direction: str = "long",
        trade_type: str = "CALLS",
        position_size_pct: float = 2.0,
        max_risk_pct: float = 1.0
    ) -> str:
        """
        Add a setup to stalk with complete trade details.
        
        Args:
            symbol: Ticker symbol
            setup_type: Type of setup
            trigger_price: Price level to watch
            trigger_condition: Condition description
            trigger_time: Time-based trigger
            catalyst: What's the catalyst
            expected_move: Expected % move if triggered
            alert_at_pct: Alert when X% away from trigger
            expires_hours: Hours until setup expires
            entry_price: Exact entry price for trade
            target_price: Take profit target
            stop_loss: Stop loss price
            direction: "long" or "short"
            trade_type: "CALLS", "PUTS", or "STOCK"
            position_size_pct: Suggested position size
            max_risk_pct: Maximum risk percentage
            
        Returns:
            setup_id
        """
        # Check capacity
        if len(self.stalked) >= self.MAX_STALKED:
            self._cleanup_oldest()
        
        setup_id = f"{symbol}_{setup_type}_{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        now = datetime.utcnow()
        
        setup = StalkedSetup(
            setup_id=setup_id,
            symbol=symbol,
            setup_type=setup_type,
            trigger_price=trigger_price,
            trigger_condition=trigger_condition,
            trigger_time=trigger_time,
            state=StalkState.WATCHING,
            distance_to_trigger=100.0,
            alert_at_pct=alert_at_pct,
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=expires_hours)).isoformat(),
            last_updated=now.isoformat(),
            catalyst=catalyst,
            expected_move=expected_move,
            entry_price=entry_price or trigger_price,
            target_price=target_price,
            stop_loss=stop_loss,
            direction=direction,
            trade_type=trade_type,
            position_size_pct=position_size_pct,
            max_risk_pct=max_risk_pct
        )
        
        self.stalked[setup_id] = setup
        self._save_setup(setup)
        
        log.info(f"Stalking new setup: {setup_id}")
        return setup_id
    
    def update_setup(
        self,
        setup_id: str,
        current_price: float,
        volume_ratio: float = 1.0,
        momentum: float = 0.0
    ) -> Optional[StalkAlert]:
        """
        Update a stalked setup with current market data.
        
        Args:
            setup_id: Setup to update
            current_price: Current price
            volume_ratio: Volume vs average
            momentum: Price momentum indicator
            
        Returns:
            StalkAlert if alert triggered
        """
        if setup_id not in self.stalked:
            return None
        
        setup = self.stalked[setup_id]
        
        # Check expiry
        if setup.expires_at:
            if datetime.utcnow() > datetime.fromisoformat(setup.expires_at):
                setup.state = StalkState.EXPIRED
                self._save_setup(setup)
                return None
        
        # Calculate distance to trigger
        if setup.trigger_price and setup.trigger_price > 0:
            distance = abs(current_price - setup.trigger_price) / setup.trigger_price * 100
            setup.distance_to_trigger = distance
            
            # Update state based on distance
            if distance <= 1.0:
                setup.state = StalkState.READY
            elif distance <= 5.0:
                setup.state = StalkState.COILING
            else:
                setup.state = StalkState.WATCHING
            
            # Check if triggered
            if setup.trigger_condition == "above" and current_price > setup.trigger_price:
                setup.state = StalkState.TRIGGERED
            elif setup.trigger_condition == "below" and current_price < setup.trigger_price:
                setup.state = StalkState.TRIGGERED
        
        # Update coil score
        setup.volume_building = volume_ratio > 1.3
        setup.coil_score = min(100, setup.coil_score + (volume_ratio - 1) * 10 + abs(momentum) * 5)
        
        # Pressure direction
        if momentum > 0.1:
            setup.pressure_direction = "bullish"
        elif momentum < -0.1:
            setup.pressure_direction = "bearish"
        else:
            setup.pressure_direction = "neutral"
        
        setup.last_updated = datetime.utcnow().isoformat()
        self._save_setup(setup)
        
        # Generate alert if needed
        alert = None
        
        if setup.state == StalkState.TRIGGERED and not setup.entry_alerted:
            # ENTRY SIGNAL - complete trade details
            alert = StalkAlert(
                setup_id=setup_id,
                symbol=setup.symbol,
                alert_type="entry",
                message=f"ENTRY SIGNAL: {setup.symbol} {setup.setup_type} - {setup.catalyst}",
                urgency=5,
                timestamp=datetime.utcnow().isoformat(),
                ticker=setup.symbol,
                entry_price=setup.entry_price,
                target_price=setup.target_price,
                stop_loss=setup.stop_loss,
                direction=setup.direction,
                trade_type=setup.trade_type,
                action="BUY" if setup.direction == "long" else "SELL"
            )
            setup.entry_alerted = True
            setup.alerted = True
            self._save_setup(setup)
            
        elif setup.state == StalkState.READY and not setup.alerted:
            # IMMINENT - give full trade plan
            alert = StalkAlert(
                setup_id=setup_id,
                symbol=setup.symbol,
                alert_type="ready",
                message=f"IMMINENT: {setup.symbol} {setup.distance_to_trigger:.1f}% from trigger - PREPARE TO ENTER",
                urgency=4,
                timestamp=datetime.utcnow().isoformat(),
                ticker=setup.symbol,
                entry_price=setup.entry_price,
                target_price=setup.target_price,
                stop_loss=setup.stop_loss,
                direction=setup.direction,
                trade_type=setup.trade_type,
                action="WATCH"
            )
            
        elif setup.distance_to_trigger <= setup.alert_at_pct and not setup.alerted:
            alert = StalkAlert(
                setup_id=setup_id,
                symbol=setup.symbol,
                alert_type="approaching",
                message=f"APPROACHING: {setup.symbol} {setup.distance_to_trigger:.1f}% from trigger",
                urgency=3,
                timestamp=datetime.utcnow().isoformat(),
                ticker=setup.symbol,
                entry_price=setup.entry_price,
                target_price=setup.target_price,
                stop_loss=setup.stop_loss,
                direction=setup.direction,
                trade_type=setup.trade_type,
                action="WATCH"
            )
        
        if alert:
            self.alerts.append(alert)
        
        return alert
    
    def check_all_setups(self, prices: Dict[str, float], volume_ratios: Optional[Dict[str, float]] = None) -> List[StalkAlert]:
        """
        Update all stalked setups with current prices.
        
        Args:
            prices: Dict of symbol -> current price
            volume_ratios: Optional dict of symbol -> volume ratio
            
        Returns:
            List of triggered alerts
        """
        alerts = []
        volume_ratios = volume_ratios or {}
        
        for setup_id, setup in list(self.stalked.items()):
            if setup.symbol in prices:
                vol_ratio = volume_ratios.get(setup.symbol, 1.0)
                alert = self.update_setup(
                    setup_id,
                    prices[setup.symbol],
                    volume_ratio=vol_ratio
                )
                if alert:
                    alerts.append(alert)
        
        return alerts
    
    def confirm_entry(self, setup_id: str, fill_price: float) -> None:
        """
        Confirm that entry was filled for a setup.
        
        Args:
            setup_id: Setup ID
            fill_price: Actual fill price
        """
        if setup_id not in self.stalked:
            return
        
        setup = self.stalked[setup_id]
        setup.entry_filled_price = fill_price
        setup.entry_filled_at = datetime.utcnow().isoformat()
        self._save_setup(setup)
        
        log.info(f"Entry confirmed for {setup.symbol} @ ${fill_price:.2f}")
    
    def check_exits(self, prices: Dict[str, float]) -> List[StalkAlert]:
        """
        Check if any filled positions should exit (hit target).
        
        Args:
            prices: Dict of symbol -> current price
            
        Returns:
            List of exit alerts
        """
        exit_alerts = []
        
        for setup_id, setup in list(self.stalked.items()):
            # Check setups that have entry alert sent (use entry_price if fill not confirmed)
            if not setup.entry_alerted or setup.exit_alerted:
                continue
            
            # Use filled price if available, otherwise use entry price
            reference_price = setup.entry_filled_price or setup.entry_price
            if not reference_price:
                continue
            
            if setup.symbol not in prices:
                continue
            
            current_price = prices[setup.symbol]
            
            # Check if target hit
            should_exit = False
            exit_reason = ""
            
            if setup.direction == "long":
                if setup.target_price and current_price >= setup.target_price:
                    should_exit = True
                    exit_reason = "TARGET HIT - BOOK PROFIT"
                elif setup.stop_loss and current_price <= setup.stop_loss:
                    should_exit = True
                    exit_reason = "STOP HIT - EXIT NOW"
            else:  # short
                if setup.target_price and current_price <= setup.target_price:
                    should_exit = True
                    exit_reason = "TARGET HIT - BOOK PROFIT"
                elif setup.stop_loss and current_price >= setup.stop_loss:
                    should_exit = True
                    exit_reason = "STOP HIT - EXIT NOW"
            
            if should_exit and not setup.exit_alerted:
                # Calculate P&L using reference price
                if setup.direction == "long":
                    pnl_pct = ((current_price - reference_price) / reference_price) * 100
                else:
                    pnl_pct = ((reference_price - current_price) / reference_price) * 100
                
                alert = StalkAlert(
                    setup_id=setup_id,
                    symbol=setup.symbol,
                    alert_type="exit",
                    message=f"{exit_reason} | P&L: {pnl_pct:+.1f}%",
                    urgency=5,
                    timestamp=datetime.utcnow().isoformat(),
                    ticker=setup.symbol,
                    entry_price=setup.entry_filled_price,
                    target_price=current_price,
                    stop_loss=setup.stop_loss,
                    direction=setup.direction,
                    trade_type=setup.trade_type,
                    action="CLOSE"
                )
                
                setup.exit_alerted = True
                self._save_setup(setup)
                exit_alerts.append(alert)
                
                log.info(f"EXIT SIGNAL: {setup.symbol} @ ${current_price:.2f} | {exit_reason}")
        
        return exit_alerts
    
    def get_active_positions(self) -> List[StalkedSetup]:
        """Get all setups that have been entered but not exited."""
        return [
            setup for setup in self.stalked.values()
            if setup.entry_filled_price and not setup.exit_alerted
        ]
    
    def record_outcome(self, setup_id: str, outcome: str, pnl_pct: float):
        """Record the outcome of a triggered setup."""
        if setup_id not in self.stalked:
            return
        
        setup = self.stalked[setup_id]
        
        # Calculate stalk duration
        created = datetime.fromisoformat(setup.created_at)
        duration_mins = int((datetime.utcnow() - created).total_seconds() / 60)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO stalk_history
            (setup_id, symbol, setup_type, trigger_time, outcome, pnl_pct, stalk_duration_mins)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            setup_id,
            setup.symbol,
            setup.setup_type,
            datetime.utcnow().isoformat(),
            outcome,
            pnl_pct,
            duration_mins
        ))
        
        conn.commit()
        conn.close()
        
        # Remove from active stalking
        del self.stalked[setup_id]
    
    def _save_setup(self, setup: StalkedSetup):
        """Save setup to database."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO stalked_setups
            (setup_id, symbol, setup_type, trigger_price, trigger_condition,
             trigger_time, state, distance_to_trigger, coil_score,
             pressure_direction, volume_building, alert_at_pct, alerted,
             created_at, expires_at, last_updated, catalyst, notes, expected_move,
             entry_price, target_price, stop_loss, direction, trade_type,
             entry_alerted, entry_filled_price, entry_filled_at, exit_alerted,
             position_size_pct, max_risk_pct, trailing_stop)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            setup.setup_id,
            setup.symbol,
            setup.setup_type,
            setup.trigger_price,
            setup.trigger_condition,
            setup.trigger_time,
            setup.state.value,
            setup.distance_to_trigger,
            setup.coil_score,
            setup.pressure_direction,
            1 if setup.volume_building else 0,
            setup.alert_at_pct,
            1 if setup.alerted else 0,
            setup.created_at,
            setup.expires_at,
            setup.last_updated,
            setup.catalyst,
            setup.notes,
            setup.expected_move,
            setup.entry_price,
            setup.target_price,
            setup.stop_loss,
            setup.direction,
            setup.trade_type,
            1 if setup.entry_alerted else 0,
            setup.entry_filled_price,
            setup.entry_filled_at,
            1 if setup.exit_alerted else 0,
            setup.position_size_pct,
            setup.max_risk_pct,
            1 if setup.trailing_stop else 0
        ))
        
        conn.commit()
        conn.close()
    
    def _cleanup_oldest(self):
        """Remove oldest expired/triggered setups."""
        # Sort by last_updated
        sorted_setups = sorted(
            self.stalked.items(),
            key=lambda x: x[1].last_updated
        )
        
        # Remove oldest 5
        for setup_id, setup in sorted_setups[:5]:
            if setup.state in [StalkState.EXPIRED, StalkState.TRIGGERED]:
                del self.stalked[setup_id]
    
    def get_active_setups(self) -> List[StalkedSetup]:
        """Get all active stalked setups."""
        return [s for s in self.stalked.values() if s.state not in [StalkState.EXPIRED]]
    
    def get_ready_setups(self) -> List[StalkedSetup]:
        """Get setups that are ready to trigger."""
        return [s for s in self.stalked.values() if s.state in [StalkState.READY, StalkState.COILING]]
    
    def get_stats_summary(self) -> Dict:
        """Get stalking statistics."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT outcome, COUNT(*) as cnt, AVG(pnl_pct) as avg_pnl
            FROM stalk_history
            GROUP BY outcome
        """)
        
        by_outcome = {}
        for row in cursor.fetchall():
            by_outcome[row["outcome"]] = {
                "count": row["cnt"],
                "avg_pnl": row["avg_pnl"]
            }
        
        cursor.execute("SELECT AVG(stalk_duration_mins) as avg FROM stalk_history")
        avg_duration = cursor.fetchone()["avg"] or 0
        
        conn.close()
        
        active = len(self.get_active_setups())
        ready = len(self.get_ready_setups())
        
        return {
            "active_setups": active,
            "ready_setups": ready,
            "by_outcome": by_outcome,
            "avg_stalk_duration_mins": avg_duration
        }


# Global instance
stalking_mode = StalkingMode()
