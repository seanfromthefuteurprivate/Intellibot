"""
Time-of-Day Learning - Tracks which hours produce the best signals.

Learns:
- Best hours for each strategy type
- Session-specific patterns (opening, midday, power hour)
- Day-of-week patterns
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from wsb_snake.db.database import get_connection
from wsb_snake.utils.logger import log


@dataclass
class TimeSlot:
    """Performance stats for a specific time slot."""
    hour: int
    day_of_week: Optional[int]  # 0=Monday, None=all days
    session: str  # "premarket", "open", "midday", "power_hour", "close"
    
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    
    total_gain: float = 0.0
    total_loss: float = 0.0
    
    avg_score: float = 0.0
    best_strategy: str = ""


@dataclass
class TimeRecommendation:
    """Recommendation based on time patterns."""
    current_hour: int
    session: str
    
    quality_score: float  # 0-100, how good is this time for signals
    historical_win_rate: float
    avg_move: float
    
    recommendation: str  # "aggressive", "normal", "cautious", "avoid"
    best_strategies: List[str]
    notes: str


class TimeLearning:
    """
    Tracks and learns from time-of-day patterns.
    """
    
    # Session definitions (ET hours)
    SESSIONS = {
        "premarket": (4, 9),
        "open": (9, 10),
        "morning": (10, 12),
        "midday": (12, 14),
        "afternoon": (14, 15),
        "power_hour": (15, 16),
        "after_hours": (16, 20)
    }
    
    def __init__(self):
        self._init_tables()
        self.time_stats: Dict[str, TimeSlot] = {}
        self._load_stats()
        log.info("Time Learning initialized")
    
    def _init_tables(self):
        """Create time learning tables."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS time_performance (
                slot_key TEXT PRIMARY KEY,
                hour INTEGER NOT NULL,
                day_of_week INTEGER,
                session TEXT NOT NULL,
                total_signals INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_gain REAL DEFAULT 0,
                total_loss REAL DEFAULT 0,
                avg_score REAL DEFAULT 0,
                best_strategy TEXT DEFAULT '',
                updated_at TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS time_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_time TEXT,
                hour INTEGER,
                day_of_week INTEGER,
                session TEXT,
                symbol TEXT,
                strategy TEXT,
                score REAL,
                outcome TEXT,
                pnl_pct REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_stats(self):
        """Load existing time stats from database."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM time_performance")
        rows = cursor.fetchall()
        
        for row in rows:
            slot = TimeSlot(
                hour=row["hour"],
                day_of_week=row["day_of_week"],
                session=row["session"],
                total_signals=row["total_signals"],
                wins=row["wins"],
                losses=row["losses"],
                total_gain=row["total_gain"],
                total_loss=row["total_loss"],
                avg_score=row["avg_score"],
                best_strategy=row["best_strategy"]
            )
            self.time_stats[row["slot_key"]] = slot
        
        conn.close()
        log.info(f"Loaded {len(self.time_stats)} time performance slots")
    
    def _get_session(self, hour: int) -> str:
        """Get session name for given hour."""
        for session, (start, end) in self.SESSIONS.items():
            if start <= hour < end:
                return session
        return "closed"
    
    def _get_slot_key(self, hour: int, day_of_week: Optional[int] = None) -> str:
        """Generate slot key."""
        if day_of_week is not None:
            return f"h{hour}_d{day_of_week}"
        return f"h{hour}"
    
    def record_signal(
        self,
        signal_time: datetime,
        symbol: str,
        strategy: str,
        score: float,
        outcome: str,
        pnl_pct: float
    ):
        """
        Record a signal outcome for time learning.
        
        Args:
            signal_time: When the signal was generated
            symbol: Ticker symbol
            strategy: Strategy type
            score: Signal score
            outcome: "win" or "loss"
            pnl_pct: Percentage gain/loss
        """
        hour = signal_time.hour
        day_of_week = signal_time.weekday()
        session = self._get_session(hour)
        
        # Update hourly stats
        slot_key = self._get_slot_key(hour)
        self._update_slot(slot_key, hour, None, session, strategy, score, outcome, pnl_pct)
        
        # Update hour+day stats
        slot_key_day = self._get_slot_key(hour, day_of_week)
        self._update_slot(slot_key_day, hour, day_of_week, session, strategy, score, outcome, pnl_pct)
        
        # Store individual signal
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO time_signals (signal_time, hour, day_of_week, session,
                                       symbol, strategy, score, outcome, pnl_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_time.isoformat(),
            hour,
            day_of_week,
            session,
            symbol,
            strategy,
            score,
            outcome,
            pnl_pct
        ))
        
        conn.commit()
        conn.close()
    
    def _update_slot(
        self,
        slot_key: str,
        hour: int,
        day_of_week: Optional[int],
        session: str,
        strategy: str,
        score: float,
        outcome: str,
        pnl_pct: float
    ):
        """Update a time slot with new signal data."""
        if slot_key not in self.time_stats:
            self.time_stats[slot_key] = TimeSlot(
                hour=hour,
                day_of_week=day_of_week,
                session=session
            )
        
        slot = self.time_stats[slot_key]
        slot.total_signals += 1
        
        if outcome == "win":
            slot.wins += 1
            slot.total_gain += pnl_pct
        else:
            slot.losses += 1
            slot.total_loss += abs(pnl_pct)
        
        # Update average score
        slot.avg_score = (slot.avg_score * (slot.total_signals - 1) + score) / slot.total_signals
        
        # Track best strategy
        if outcome == "win" and pnl_pct > 5:  # Big winner
            slot.best_strategy = strategy
        
        # Save to database
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO time_performance
            (slot_key, hour, day_of_week, session, total_signals, wins, losses,
             total_gain, total_loss, avg_score, best_strategy, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            slot_key,
            hour,
            day_of_week,
            session,
            slot.total_signals,
            slot.wins,
            slot.losses,
            slot.total_gain,
            slot.total_loss,
            slot.avg_score,
            slot.best_strategy,
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_recommendation(self, current_time: Optional[datetime] = None) -> TimeRecommendation:
        """
        Get trading recommendation based on current time.
        
        Args:
            current_time: Time to check (defaults to now ET)
            
        Returns:
            TimeRecommendation with quality score and advice
        """
        if current_time is None:
            from wsb_snake.utils.session_regime import get_session_info
            session_info = get_session_info()
            current_time = datetime.now()
        
        hour = current_time.hour
        day_of_week = current_time.weekday()
        session = self._get_session(hour)
        
        # Get stats for this hour
        slot_key = self._get_slot_key(hour)
        slot_key_day = self._get_slot_key(hour, day_of_week)
        
        slot = self.time_stats.get(slot_key)
        slot_day = self.time_stats.get(slot_key_day)
        
        # Use day-specific if available, otherwise hour-only
        active_slot = slot_day if slot_day and slot_day.total_signals >= 5 else slot
        
        if not active_slot or active_slot.total_signals < 3:
            # Not enough data - use session defaults
            return self._get_default_recommendation(hour, session)
        
        # Calculate metrics
        total = active_slot.wins + active_slot.losses
        win_rate = active_slot.wins / total if total > 0 else 0.5
        avg_move = (active_slot.total_gain - active_slot.total_loss) / total if total > 0 else 0
        
        # Quality score based on win rate and sample size
        quality = win_rate * 100 * min(1.0, total / 20)
        
        # Determine recommendation
        if quality >= 70:
            rec = "aggressive"
        elif quality >= 50:
            rec = "normal"
        elif quality >= 30:
            rec = "cautious"
        else:
            rec = "avoid"
        
        # Get best strategies for this time
        best_strategies = self._get_best_strategies_for_hour(hour)
        
        notes = f"{session.upper()}: {active_slot.total_signals} signals tracked"
        if active_slot.best_strategy:
            notes += f", best: {active_slot.best_strategy}"
        
        return TimeRecommendation(
            current_hour=hour,
            session=session,
            quality_score=quality,
            historical_win_rate=win_rate,
            avg_move=avg_move,
            recommendation=rec,
            best_strategies=best_strategies,
            notes=notes
        )
    
    def _get_default_recommendation(self, hour: int, session: str) -> TimeRecommendation:
        """Get default recommendation when no data available."""
        # Default quality by session
        session_quality = {
            "premarket": 30,
            "open": 60,  # Opening volatility
            "morning": 55,
            "midday": 40,  # Lunch doldrums
            "afternoon": 50,
            "power_hour": 75,  # Best for 0DTE
            "after_hours": 20,
            "closed": 0
        }
        
        quality = session_quality.get(session, 50)
        
        if quality >= 60:
            rec = "normal"
        elif quality >= 40:
            rec = "cautious"
        else:
            rec = "avoid"
        
        return TimeRecommendation(
            current_hour=hour,
            session=session,
            quality_score=quality,
            historical_win_rate=0.5,
            avg_move=0,
            recommendation=rec,
            best_strategies=["0DTE_MOMENTUM"] if session == "power_hour" else [],
            notes=f"No historical data for {session}"
        )
    
    def _get_best_strategies_for_hour(self, hour: int) -> List[str]:
        """Get best performing strategies for given hour."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT strategy, COUNT(*) as cnt, 
                   SUM(CASE WHEN outcome='win' THEN 1 ELSE 0 END) as wins
            FROM time_signals
            WHERE hour = ?
            GROUP BY strategy
            HAVING cnt >= 3
            ORDER BY (wins * 1.0 / cnt) DESC
            LIMIT 3
        """, (hour,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [row["strategy"] for row in rows]
    
    def get_best_hours(self, min_signals: int = 5) -> List[Dict]:
        """Get ranking of best trading hours."""
        results = []
        
        for slot_key, slot in self.time_stats.items():
            if slot.total_signals < min_signals:
                continue
            if slot.day_of_week is not None:
                continue  # Only hour-level stats
            
            total = slot.wins + slot.losses
            if total == 0:
                continue
            
            win_rate = slot.wins / total
            avg_return = (slot.total_gain - slot.total_loss) / total
            
            results.append({
                "hour": slot.hour,
                "session": slot.session,
                "signals": slot.total_signals,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "score": win_rate * 100 * min(1.0, total / 20)
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def get_stats_summary(self) -> Dict:
        """Get summary of time learning data."""
        total_slots = len(self.time_stats)
        total_signals = sum(s.total_signals for s in self.time_stats.values())
        
        best_hours = self.get_best_hours()
        
        return {
            "total_time_slots": total_slots,
            "total_signals_tracked": total_signals,
            "best_hours": best_hours[:3] if best_hours else [],
            "worst_hours": best_hours[-3:] if len(best_hours) > 3 else []
        }


# Global instance
time_learning = TimeLearning()
