"""
Trade Outcome Recorder

Central orchestrator for recording trade outcomes to all learning systems.
Connects trade exits to:
- Database outcomes table
- learning_memory (weight updates)
- pattern_memory (pattern storage)
- time_learning (time-of-day stats)
"""

from datetime import datetime
from typing import Dict, List, Optional
import pytz

from wsb_snake.utils.logger import get_logger
from wsb_snake.db.database import save_outcome
from wsb_snake.engines.learning_memory import learning_memory
from wsb_snake.learning.pattern_memory import pattern_memory
from wsb_snake.learning.time_learning import time_learning

logger = get_logger(__name__)


class TradeOutcomeRecorder:
    """Records trade outcomes to all learning systems."""

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
        logger.info("TradeOutcomeRecorder initialized")

    def _get_session(self, hour: int) -> str:
        """Get session name for given hour."""
        for session, (start, end) in self.SESSIONS.items():
            if start <= hour < end:
                return session
        return "closed"

    def _determine_outcome_type(self, pnl: float, pnl_pct: float, exit_reason: str) -> str:
        """Determine outcome type from PnL and exit reason."""
        if "TARGET" in exit_reason.upper():
            return "win"
        elif "STOP" in exit_reason.upper():
            return "loss"
        elif "TIME" in exit_reason.upper() or "DECAY" in exit_reason.upper():
            return "timeout"
        elif pnl > 0:
            return "win"
        elif pnl < 0:
            return "loss"
        else:
            return "scratch"

    def record_trade_outcome(
        self,
        signal_id: Optional[int],
        symbol: str,
        trade_type: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        entry_time: datetime,
        exit_time: datetime,
        engine: str,
        bars: Optional[List[Dict]] = None,
    ) -> None:
        """
        Record outcome to all learning systems.

        Args:
            signal_id: ID of the original signal (if available)
            symbol: Ticker symbol (e.g., "SPY")
            trade_type: "CALLS" or "PUTS"
            entry_price: Option entry price
            exit_price: Option exit price
            pnl: Dollar P&L
            pnl_pct: Percentage P&L
            exit_reason: Why the trade was closed
            entry_time: When the trade was opened
            exit_time: When the trade was closed
            engine: Trading engine ("scalper", "momentum", "macro")
            bars: Optional list of OHLCV bars for pattern learning
        """
        try:
            # Calculate holding time
            holding_time_seconds = int((exit_time - entry_time).total_seconds())

            # Determine session (use ET timezone)
            et = pytz.timezone('US/Eastern')
            entry_time_et = entry_time.astimezone(et) if entry_time.tzinfo else et.localize(entry_time)
            session = self._get_session(entry_time_et.hour)

            # Determine outcome type
            outcome_type = self._determine_outcome_type(pnl, pnl_pct, exit_reason)

            logger.info(f"Recording outcome: {symbol} {trade_type} | {outcome_type} | ${pnl:+.2f} ({pnl_pct:+.1f}%)")

            # 1. Save to database (outcomes + trade_performance tables)
            outcome_data = {
                "signal_id": signal_id,
                "symbol": symbol,
                "trade_type": trade_type,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "max_price": exit_price,  # MFE - ideally tracked during trade
                "min_price": exit_price,  # MAE - ideally tracked during trade
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "outcome_type": outcome_type,
                "exit_reason": exit_reason,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "engine": engine,
                "session": session,
                "holding_time_seconds": holding_time_seconds,
            }
            save_outcome(outcome_data)
            logger.debug("Saved to database outcomes table")

            # 2. Update learning_memory weights (if we have signal_id)
            if signal_id:
                try:
                    learning_memory.record_outcome(
                        signal_id=signal_id,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        max_price=exit_price,
                        min_price=exit_price,
                        outcome_type=outcome_type,
                    )
                    logger.debug("Updated learning_memory weights")
                except Exception as e:
                    logger.warning(f"Failed to update learning_memory: {e}")

            # 3. Store pattern in pattern_memory (if we have bars)
            if bars and len(bars) >= 3:
                try:
                    pattern_id = pattern_memory.store_pattern(
                        symbol=symbol,
                        bars=bars,
                        outcome=outcome_type if outcome_type in ("win", "loss") else "loss",
                        pnl_pct=pnl_pct,
                        rsi=50,  # Default RSI if not available
                        vwap_position="neutral",
                    )
                    if pattern_id:
                        logger.debug(f"Stored pattern: {pattern_id}")
                except Exception as e:
                    logger.warning(f"Failed to store pattern: {e}")

            # 4. Record to time_learning
            try:
                # Map trade_type to strategy
                strategy = f"{engine}_{trade_type}"
                time_learning.record_signal(
                    signal_time=entry_time_et.replace(tzinfo=None),
                    symbol=symbol,
                    strategy=strategy,
                    score=50.0,  # Default score if not available
                    outcome=outcome_type if outcome_type in ("win", "loss") else "loss",
                    pnl_pct=pnl_pct,
                )
                logger.debug("Recorded to time_learning")
            except Exception as e:
                logger.warning(f"Failed to record to time_learning: {e}")

            logger.info(f"Recorded outcome for {symbol}: {outcome_type} to all learning systems")

        except Exception as e:
            logger.error(f"Failed to record trade outcome: {e}")


# Global instance
outcome_recorder = TradeOutcomeRecorder()
