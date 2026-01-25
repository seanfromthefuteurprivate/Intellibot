"""
Event Outcome Database - Tracks actual moves after CPI, FOMC, earnings events.

Learns:
- Actual price moves after each event type
- Which symbols react most to which events
- Pattern of moves (immediate, delayed, fade)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json

from wsb_snake.db.database import get_connection
from wsb_snake.utils.logger import log


@dataclass
class EventOutcome:
    """Recorded outcome of an event."""
    event_id: str
    event_type: str  # "CPI", "FOMC", "JOBS", "EARNINGS", "GDP"
    event_date: str
    symbol: str
    
    # Pre-event data
    price_before: float
    iv_before: float
    
    # Immediate reaction (first 30 min)
    move_30min: float
    move_direction: str  # "up", "down", "flat"
    
    # Intraday reaction
    move_1hr: float
    move_eod: float  # End of day
    
    # Next day
    move_next_day: float
    
    # IV crush
    iv_after: float
    iv_crush_pct: float
    
    # Did fade occur?
    faded: bool  # If initial move reversed
    fade_pct: float


@dataclass
class EventExpectation:
    """Expected outcome for upcoming event."""
    event_type: str
    symbol: str
    
    expected_move: float
    expected_direction: str  # "bullish", "bearish", "neutral"
    expected_iv_crush: float
    
    fade_probability: float
    confidence: float
    
    sample_size: int
    notes: str


class EventOutcomeDB:
    """
    Tracks and learns from event outcomes.
    """
    
    EVENT_TYPES = ["CPI", "FOMC", "JOBS", "GDP", "EARNINGS"]
    
    def __init__(self):
        self._init_tables()
        self.outcomes: Dict[str, EventOutcome] = {}
        self._load_outcomes()
        log.info("Event Outcome DB initialized")
    
    def _init_tables(self):
        """Create event outcome tables."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_outcomes (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                event_date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                price_before REAL,
                iv_before REAL,
                move_30min REAL DEFAULT 0,
                move_direction TEXT DEFAULT 'flat',
                move_1hr REAL DEFAULT 0,
                move_eod REAL DEFAULT 0,
                move_next_day REAL DEFAULT 0,
                iv_after REAL DEFAULT 0,
                iv_crush_pct REAL DEFAULT 0,
                faded INTEGER DEFAULT 0,
                fade_pct REAL DEFAULT 0,
                recorded_at TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_stats (
                stat_key TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                sample_count INTEGER DEFAULT 0,
                avg_move REAL DEFAULT 0,
                avg_iv_crush REAL DEFAULT 0,
                bullish_pct REAL DEFAULT 0,
                fade_rate REAL DEFAULT 0,
                best_strategy TEXT DEFAULT '',
                updated_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_outcomes(self):
        """Load recent outcomes from database."""
        conn = get_connection()
        cursor = conn.cursor()
        
        # Load last 100 outcomes
        cursor.execute("""
            SELECT * FROM event_outcomes
            ORDER BY event_date DESC
            LIMIT 100
        """)
        rows = cursor.fetchall()
        
        for row in rows:
            outcome = EventOutcome(
                event_id=row["event_id"],
                event_type=row["event_type"],
                event_date=row["event_date"],
                symbol=row["symbol"],
                price_before=row["price_before"],
                iv_before=row["iv_before"],
                move_30min=row["move_30min"],
                move_direction=row["move_direction"],
                move_1hr=row["move_1hr"],
                move_eod=row["move_eod"],
                move_next_day=row["move_next_day"],
                iv_after=row["iv_after"],
                iv_crush_pct=row["iv_crush_pct"],
                faded=bool(row["faded"]),
                fade_pct=row["fade_pct"]
            )
            self.outcomes[outcome.event_id] = outcome
        
        conn.close()
        log.info(f"Loaded {len(self.outcomes)} event outcomes")
    
    def record_event_start(
        self,
        event_type: str,
        event_date: str,
        symbol: str,
        price_before: float,
        iv_before: float
    ) -> str:
        """
        Record the start of an event (before release).
        
        Returns:
            event_id for later update
        """
        event_id = f"{event_type}_{symbol}_{event_date}"
        
        outcome = EventOutcome(
            event_id=event_id,
            event_type=event_type,
            event_date=event_date,
            symbol=symbol,
            price_before=price_before,
            iv_before=iv_before,
            move_30min=0,
            move_direction="flat",
            move_1hr=0,
            move_eod=0,
            move_next_day=0,
            iv_after=iv_before,
            iv_crush_pct=0,
            faded=False,
            fade_pct=0
        )
        
        self.outcomes[event_id] = outcome
        self._save_outcome(outcome)
        
        log.info(f"Recording event start: {event_id}")
        return event_id
    
    def update_event_outcome(
        self,
        event_id: str,
        price_30min: Optional[float] = None,
        price_1hr: Optional[float] = None,
        price_eod: Optional[float] = None,
        price_next_day: Optional[float] = None,
        iv_after: Optional[float] = None
    ):
        """Update an event with outcome data."""
        if event_id not in self.outcomes:
            log.warning(f"Unknown event_id: {event_id}")
            return
        
        outcome = self.outcomes[event_id]
        price_before = outcome.price_before
        
        if price_before <= 0:
            return
        
        if price_30min is not None:
            outcome.move_30min = ((price_30min - price_before) / price_before) * 100
            if outcome.move_30min > 0.3:
                outcome.move_direction = "up"
            elif outcome.move_30min < -0.3:
                outcome.move_direction = "down"
            else:
                outcome.move_direction = "flat"
        
        if price_1hr is not None:
            outcome.move_1hr = ((price_1hr - price_before) / price_before) * 100
        
        if price_eod is not None:
            outcome.move_eod = ((price_eod - price_before) / price_before) * 100
            
            # Check for fade
            if abs(outcome.move_30min) > 0.5:
                if (outcome.move_30min > 0 and outcome.move_eod < outcome.move_30min * 0.5) or \
                   (outcome.move_30min < 0 and outcome.move_eod > outcome.move_30min * 0.5):
                    outcome.faded = True
                    outcome.fade_pct = outcome.move_30min - outcome.move_eod
        
        if price_next_day is not None:
            outcome.move_next_day = ((price_next_day - price_before) / price_before) * 100
        
        if iv_after is not None:
            outcome.iv_after = iv_after
            if outcome.iv_before > 0:
                outcome.iv_crush_pct = ((outcome.iv_before - iv_after) / outcome.iv_before) * 100
        
        self._save_outcome(outcome)
        self._update_stats(outcome)
    
    def _save_outcome(self, outcome: EventOutcome):
        """Save outcome to database."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO event_outcomes
            (event_id, event_type, event_date, symbol, price_before, iv_before,
             move_30min, move_direction, move_1hr, move_eod, move_next_day,
             iv_after, iv_crush_pct, faded, fade_pct, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            outcome.event_id,
            outcome.event_type,
            outcome.event_date,
            outcome.symbol,
            outcome.price_before,
            outcome.iv_before,
            outcome.move_30min,
            outcome.move_direction,
            outcome.move_1hr,
            outcome.move_eod,
            outcome.move_next_day,
            outcome.iv_after,
            outcome.iv_crush_pct,
            1 if outcome.faded else 0,
            outcome.fade_pct,
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _update_stats(self, outcome: EventOutcome):
        """Update aggregated stats for this event type + symbol."""
        stat_key = f"{outcome.event_type}_{outcome.symbol}"
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get all outcomes for this combo
        cursor.execute("""
            SELECT move_30min, move_direction, faded
            FROM event_outcomes
            WHERE event_type = ? AND symbol = ?
        """, (outcome.event_type, outcome.symbol))
        
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            return
        
        sample_count = len(rows)
        avg_move = sum(abs(r["move_30min"]) for r in rows) / sample_count
        bullish_count = sum(1 for r in rows if r["move_direction"] == "up")
        fade_count = sum(1 for r in rows if r["faded"])
        
        cursor.execute("""
            INSERT OR REPLACE INTO event_stats
            (stat_key, event_type, symbol, sample_count, avg_move,
             bullish_pct, fade_rate, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stat_key,
            outcome.event_type,
            outcome.symbol,
            sample_count,
            avg_move,
            bullish_count / sample_count if sample_count > 0 else 0.5,
            fade_count / sample_count if sample_count > 0 else 0,
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_expectation(self, event_type: str, symbol: str) -> EventExpectation:
        """
        Get expected outcome for an upcoming event.
        
        Args:
            event_type: Type of event
            symbol: Ticker symbol
            
        Returns:
            EventExpectation with predictions
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get historical outcomes
        cursor.execute("""
            SELECT * FROM event_outcomes
            WHERE event_type = ? AND symbol = ?
            ORDER BY event_date DESC
            LIMIT 20
        """, (event_type, symbol))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            # Use defaults
            return self._get_default_expectation(event_type, symbol)
        
        sample_size = len(rows)
        
        # Calculate metrics
        moves = [r["move_30min"] for r in rows]
        avg_move = sum(abs(m) for m in moves) / sample_size
        
        bullish_count = sum(1 for r in rows if r["move_direction"] == "up")
        bullish_pct = bullish_count / sample_size
        
        fade_count = sum(1 for r in rows if r["faded"])
        fade_rate = fade_count / sample_size
        
        iv_crushes = [r["iv_crush_pct"] for r in rows if r["iv_crush_pct"] > 0]
        avg_iv_crush = sum(iv_crushes) / len(iv_crushes) if iv_crushes else 30
        
        # Determine direction
        if bullish_pct > 0.6:
            direction = "bullish"
        elif bullish_pct < 0.4:
            direction = "bearish"
        else:
            direction = "neutral"
        
        # Confidence based on sample size and consistency
        consistency = max(bullish_pct, 1 - bullish_pct)  # How one-sided
        confidence = min(1.0, sample_size / 10) * consistency * 100
        
        notes = f"Based on {sample_size} past {event_type} events"
        if fade_rate > 0.5:
            notes += f" | HIGH FADE RISK ({fade_rate*100:.0f}%)"
        
        return EventExpectation(
            event_type=event_type,
            symbol=symbol,
            expected_move=avg_move,
            expected_direction=direction,
            expected_iv_crush=avg_iv_crush,
            fade_probability=fade_rate,
            confidence=confidence,
            sample_size=sample_size,
            notes=notes
        )
    
    def _get_default_expectation(self, event_type: str, symbol: str) -> EventExpectation:
        """Get default expectation when no data available."""
        # Default expected moves by event type
        defaults = {
            "CPI": {"move": 1.2, "iv_crush": 40},
            "FOMC": {"move": 1.0, "iv_crush": 35},
            "JOBS": {"move": 0.8, "iv_crush": 30},
            "GDP": {"move": 0.6, "iv_crush": 25},
            "EARNINGS": {"move": 5.0, "iv_crush": 50}
        }
        
        d = defaults.get(event_type, {"move": 1.0, "iv_crush": 30})
        
        return EventExpectation(
            event_type=event_type,
            symbol=symbol,
            expected_move=d["move"],
            expected_direction="neutral",
            expected_iv_crush=d["iv_crush"],
            fade_probability=0.3,
            confidence=30,
            sample_size=0,
            notes=f"No historical data for {symbol} {event_type}, using defaults"
        )
    
    def get_all_expectations(self, symbols: List[str]) -> Dict[str, List[EventExpectation]]:
        """Get expectations for all event types for given symbols."""
        result = {}
        
        for symbol in symbols:
            result[symbol] = []
            for event_type in self.EVENT_TYPES:
                exp = self.get_expectation(event_type, symbol)
                if exp.sample_size > 0 or event_type in ["CPI", "FOMC", "JOBS"]:
                    result[symbol].append(exp)
        
        return result
    
    def get_stats_summary(self) -> Dict:
        """Get summary statistics."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as cnt FROM event_outcomes")
        total_outcomes = cursor.fetchone()["cnt"]
        
        cursor.execute("""
            SELECT event_type, COUNT(*) as cnt, AVG(ABS(move_30min)) as avg_move
            FROM event_outcomes
            GROUP BY event_type
        """)
        
        by_type = {}
        for row in cursor.fetchall():
            by_type[row["event_type"]] = {
                "count": row["cnt"],
                "avg_move": row["avg_move"]
            }
        
        conn.close()
        
        return {
            "total_outcomes": total_outcomes,
            "by_event_type": by_type
        }


# Global instance
event_outcome_db = EventOutcomeDB()
