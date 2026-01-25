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


@dataclass
class StalkAlert:
    """Alert for a stalked setup."""
    setup_id: str
    symbol: str
    alert_type: str  # "approaching", "ready", "triggered"
    message: str
    urgency: int  # 1-5
    timestamp: str


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
                expected_move REAL DEFAULT 0
            )
        """)
        
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
                expected_move=row["expected_move"]
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
        expires_hours: int = 24
    ) -> str:
        """
        Add a setup to stalk.
        
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
            expected_move=expected_move
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
        
        if setup.state == StalkState.TRIGGERED and not setup.alerted:
            alert = StalkAlert(
                setup_id=setup_id,
                symbol=setup.symbol,
                alert_type="triggered",
                message=f"TRIGGERED: {setup.symbol} {setup.setup_type} - {setup.catalyst}",
                urgency=5,
                timestamp=datetime.utcnow().isoformat()
            )
            setup.alerted = True
            self._save_setup(setup)
            
        elif setup.state == StalkState.READY and not setup.alerted:
            alert = StalkAlert(
                setup_id=setup_id,
                symbol=setup.symbol,
                alert_type="ready",
                message=f"READY: {setup.symbol} {setup.distance_to_trigger:.1f}% from trigger",
                urgency=4,
                timestamp=datetime.utcnow().isoformat()
            )
            
        elif setup.distance_to_trigger <= setup.alert_at_pct and not setup.alerted:
            alert = StalkAlert(
                setup_id=setup_id,
                symbol=setup.symbol,
                alert_type="approaching",
                message=f"APPROACHING: {setup.symbol} {setup.distance_to_trigger:.1f}% from trigger",
                urgency=3,
                timestamp=datetime.utcnow().isoformat()
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
             created_at, expires_at, last_updated, catalyst, notes, expected_move)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            setup.expected_move
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
