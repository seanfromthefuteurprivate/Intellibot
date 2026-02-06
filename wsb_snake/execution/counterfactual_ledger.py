"""
Counterfactual Replay Ledger — What-if analysis for Apex Governance.

Logs checkpoints and computes missed upside for:
- 2x shock-and-fade
- 4x secondary expansion
- 10x sustained dominance
- 20x+ tail cascades
- No-runner baseline
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)

# Global toggle
COUNTERFACTUAL_ENABLED = True


@dataclass
class Checkpoint:
    """A single counterfactual checkpoint."""
    call_id: str
    timestamp_et: str
    ticker: str
    checkpoint_type: str  # TP1, TP2, TP3, RUNNER_LOCK, STRUCTURE_BREAK, etc.
    actual_pnl_pct: float
    counterfactual_exit_pnl: float  # What PnL would have been if exited here
    runner_outcome_pnl: Optional[float] = None  # Final outcome if held
    missed_upside_pct: Optional[float] = None
    scenario_json: Optional[str] = None


class CounterfactualLedger:
    """
    Logs counterfactual checkpoints for what-if analysis.
    
    Tracks:
    - TP checkpoints (what if we exited at TP1/TP2/TP3?)
    - RUNNER_LOCK entry (what if we held vs exited?)
    - Final outcomes for comparison
    """
    
    def __init__(self, db_enabled: bool = True):
        self.enabled = COUNTERFACTUAL_ENABLED
        self.db_enabled = db_enabled
        self.session_checkpoints: Dict[str, List[Checkpoint]] = {}
        self._db = None
        logger.info(f"CounterfactualLedger initialized (enabled={self.enabled})")
    
    def _get_db(self):
        """Lazy load database connection."""
        if self._db is None:
            try:
                from wsb_snake.db.database import get_connection
                self._db = get_connection()
            except ImportError:
                self._db = None
        return self._db
    
    def log_checkpoint(
        self,
        call_id: str,
        ticker: str,
        checkpoint_type: str,
        actual_pnl_pct: float,
        counterfactual_exit_pnl: float,
        runner_outcome_pnl: Optional[float] = None,
        scenario: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Log a counterfactual checkpoint.
        
        Args:
            call_id: Unique identifier for the call
            ticker: Underlying ticker
            checkpoint_type: TP1, TP2, TP3, RUNNER_LOCK, STRUCTURE_BREAK, etc.
            actual_pnl_pct: Actual PnL at this point
            counterfactual_exit_pnl: What PnL would have been if exited here
            runner_outcome_pnl: Final outcome if held (optional, filled later)
            scenario: Additional context as dict
        
        Returns:
            True if logged successfully
        """
        if not self.enabled:
            return False
        
        timestamp_et = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        checkpoint = Checkpoint(
            call_id=call_id,
            timestamp_et=timestamp_et,
            ticker=ticker,
            checkpoint_type=checkpoint_type,
            actual_pnl_pct=actual_pnl_pct,
            counterfactual_exit_pnl=counterfactual_exit_pnl,
            runner_outcome_pnl=runner_outcome_pnl,
            scenario_json=json.dumps(scenario) if scenario else None,
        )
        
        # Store in memory
        if call_id not in self.session_checkpoints:
            self.session_checkpoints[call_id] = []
        self.session_checkpoints[call_id].append(checkpoint)
        
        # Store in database if enabled
        if self.db_enabled:
            try:
                self._save_to_db(checkpoint)
            except Exception as e:
                logger.warning(f"Failed to save checkpoint to DB: {e}")
        
        logger.info(f"COUNTERFACTUAL | {checkpoint_type} | {ticker} | actual={actual_pnl_pct:.1f}% cf_exit={counterfactual_exit_pnl:.1f}%")
        return True
    
    def update_runner_outcome(self, call_id: str, final_pnl_pct: float) -> None:
        """
        Update all checkpoints for a call with the final runner outcome.
        
        Called when position is closed to compute missed upside.
        """
        if call_id not in self.session_checkpoints:
            return
        
        for checkpoint in self.session_checkpoints[call_id]:
            checkpoint.runner_outcome_pnl = final_pnl_pct
            # Compute missed upside: final - counterfactual
            checkpoint.missed_upside_pct = final_pnl_pct - checkpoint.counterfactual_exit_pnl
            
            # Update in DB if enabled
            if self.db_enabled:
                self._update_outcome_in_db(checkpoint)
        
        logger.info(f"COUNTERFACTUAL | OUTCOME_UPDATED | {call_id} | final={final_pnl_pct:.1f}%")
    
    def compute_missed_upside(self, call_id: str) -> Optional[float]:
        """
        Compute total missed upside for a call.
        
        Returns the maximum missed upside across all checkpoints.
        """
        if call_id not in self.session_checkpoints:
            return None
        
        max_missed = 0.0
        for checkpoint in self.session_checkpoints[call_id]:
            if checkpoint.missed_upside_pct is not None:
                max_missed = max(max_missed, checkpoint.missed_upside_pct)
        
        return max_missed if max_missed > 0 else None
    
    def generate_replay_report(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a replay report for what-if analysis.
        
        Returns:
            Dict with:
            - total_checkpoints: int
            - total_missed_upside: float (sum of all positive missed upside)
            - worst_early_exit: dict (checkpoint where early exit would have been worst)
            - best_runner_outcome: dict (best RUNNER_LOCK outcome)
            - scenarios: list of all checkpoints
        """
        all_checkpoints = []
        for call_id, checkpoints in self.session_checkpoints.items():
            all_checkpoints.extend(checkpoints)
        
        if not all_checkpoints:
            return {
                "total_checkpoints": 0,
                "total_missed_upside": 0.0,
                "worst_early_exit": None,
                "best_runner_outcome": None,
                "scenarios": [],
            }
        
        # Calculate metrics
        total_missed = sum(
            cp.missed_upside_pct for cp in all_checkpoints
            if cp.missed_upside_pct is not None and cp.missed_upside_pct > 0
        )
        
        # Find worst early exit (highest missed upside)
        worst_early = max(
            (cp for cp in all_checkpoints if cp.missed_upside_pct is not None),
            key=lambda x: x.missed_upside_pct or 0,
            default=None,
        )
        
        # Find best runner outcome
        runner_locks = [cp for cp in all_checkpoints if cp.checkpoint_type == "RUNNER_LOCK"]
        best_runner = max(
            runner_locks,
            key=lambda x: x.runner_outcome_pnl or 0,
            default=None,
        ) if runner_locks else None
        
        return {
            "total_checkpoints": len(all_checkpoints),
            "total_missed_upside": total_missed,
            "worst_early_exit": {
                "call_id": worst_early.call_id,
                "checkpoint_type": worst_early.checkpoint_type,
                "missed_upside_pct": worst_early.missed_upside_pct,
            } if worst_early else None,
            "best_runner_outcome": {
                "call_id": best_runner.call_id,
                "runner_outcome_pnl": best_runner.runner_outcome_pnl,
            } if best_runner else None,
            "scenarios": [
                {
                    "call_id": cp.call_id,
                    "ticker": cp.ticker,
                    "checkpoint_type": cp.checkpoint_type,
                    "actual_pnl_pct": cp.actual_pnl_pct,
                    "counterfactual_exit_pnl": cp.counterfactual_exit_pnl,
                    "runner_outcome_pnl": cp.runner_outcome_pnl,
                    "missed_upside_pct": cp.missed_upside_pct,
                }
                for cp in all_checkpoints
            ],
        }
    
    def log_scenario_comparison(
        self,
        call_id: str,
        ticker: str,
        scenario_name: str,
        tp_exit_pnl: float,
        runner_pnl: float,
    ) -> None:
        """
        Log a scenario comparison (e.g., 2x shock-and-fade vs 10x sustained trend).
        """
        scenario = {
            "scenario_name": scenario_name,
            "tp_exit_pnl": tp_exit_pnl,
            "runner_pnl": runner_pnl,
            "runner_advantage": runner_pnl - tp_exit_pnl,
        }
        self.log_checkpoint(
            call_id=call_id,
            ticker=ticker,
            checkpoint_type=f"SCENARIO_{scenario_name.upper().replace(' ', '_')}",
            actual_pnl_pct=runner_pnl,
            counterfactual_exit_pnl=tp_exit_pnl,
            scenario=scenario,
        )
    
    def _save_to_db(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to database."""
        db = self._get_db()
        if db is None:
            return
        
        try:
            cursor = db.cursor()
            cursor.execute("""
                INSERT INTO counterfactual_ledger (
                    call_id, timestamp_et, ticker, checkpoint_type,
                    actual_pnl_pct, counterfactual_exit_pnl, runner_outcome_pnl,
                    missed_upside_pct, scenario_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                checkpoint.call_id,
                checkpoint.timestamp_et,
                checkpoint.ticker,
                checkpoint.checkpoint_type,
                checkpoint.actual_pnl_pct,
                checkpoint.counterfactual_exit_pnl,
                checkpoint.runner_outcome_pnl,
                checkpoint.missed_upside_pct,
                checkpoint.scenario_json,
            ))
            db.commit()
        except Exception as e:
            logger.warning(f"DB save error: {e}")
    
    def _update_outcome_in_db(self, checkpoint: Checkpoint) -> None:
        """Update checkpoint outcome in database."""
        db = self._get_db()
        if db is None:
            return
        
        try:
            cursor = db.cursor()
            cursor.execute("""
                UPDATE counterfactual_ledger
                SET runner_outcome_pnl = ?, missed_upside_pct = ?
                WHERE call_id = ? AND checkpoint_type = ? AND timestamp_et = ?
            """, (
                checkpoint.runner_outcome_pnl,
                checkpoint.missed_upside_pct,
                checkpoint.call_id,
                checkpoint.checkpoint_type,
                checkpoint.timestamp_et,
            ))
            db.commit()
        except Exception as e:
            logger.warning(f"DB update error: {e}")


# Singleton instance
_counterfactual_ledger: Optional[CounterfactualLedger] = None


def get_counterfactual_ledger() -> CounterfactualLedger:
    """Get or create the singleton counterfactual ledger."""
    global _counterfactual_ledger
    if _counterfactual_ledger is None:
        _counterfactual_ledger = CounterfactualLedger()
    return _counterfactual_ledger
