"""
Engine 5: Self-Learning Memory

Tracks signal outcomes and adjusts model weights to improve future predictions.
Implements online learning with exponential decay.

Key features:
- Tracks every signal's outcome
- Updates feature weights based on hit/miss
- Context-aware weights (power hour vs regular)
- Decay old data to adapt to changing markets
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from wsb_snake.db.database import get_connection, save_signal
from wsb_snake.utils.logger import log


@dataclass
class LearningUpdate:
    """Record of a weight update."""
    feature_name: str
    old_weight: float
    new_weight: float
    reason: str
    updated_at: datetime


class LearningMemory:
    """
    Engine 5: Self-Learning Memory
    
    Tracks outcomes and updates weights to improve predictions.
    """
    
    # Learning parameters
    LEARNING_RATE = 0.05  # How much to adjust weights per update
    DECAY_FACTOR = 0.95   # Daily decay for old data
    MIN_WEIGHT = 0.1      # Minimum weight for any feature
    MAX_WEIGHT = 3.0      # Maximum weight for any feature
    
    def __init__(self):
        self._ensure_weights_initialized()
    
    def _ensure_weights_initialized(self):
        """Initialize default weights if not present."""
        default_features = [
            # Engine weights
            ("ignition", 1.0, "Ignition detector engine"),
            ("pressure", 1.0, "Options pressure engine"),
            ("surge", 1.0, "Surge hunter engine"),
            
            # Feature weights
            ("volume_spike", 1.0, "Volume explosion signal"),
            ("velocity", 1.0, "Price velocity signal"),
            ("news_catalyst", 1.0, "News-driven moves"),
            ("vwap_reclaim", 1.0, "VWAP reclaim pattern"),
            ("range_breakout", 1.0, "Range breakout pattern"),
            ("call_put_ratio", 1.0, "Options call/put ratio"),
            ("iv_spike", 1.0, "IV spike detection"),
        ]
        
        conn = get_connection()
        cursor = conn.cursor()
        
        for feature_name, default_weight, _ in default_features:
            cursor.execute("""
                INSERT OR IGNORE INTO model_weights (feature_name, weight, last_updated)
                VALUES (?, ?, ?)
            """, (feature_name, default_weight, datetime.utcnow().isoformat()))
        
        conn.commit()
        conn.close()
    
    def record_outcome(
        self,
        signal_id: int,
        entry_price: float,
        exit_price: float,
        max_price: float,
        min_price: float,
        outcome_type: str,  # "win", "loss", "scratch", "timeout"
    ) -> None:
        """
        Record the outcome of a signal.
        
        Args:
            signal_id: ID of the original signal
            entry_price: Actual entry price
            exit_price: Actual exit price
            max_price: Maximum favorable excursion
            min_price: Maximum adverse excursion
            outcome_type: Result category
        """
        conn = get_connection()
        cursor = conn.cursor()
        
        # Calculate R-multiple
        cursor.execute("SELECT * FROM signals WHERE id = ?", (signal_id,))
        signal = cursor.fetchone()
        
        if not signal:
            log.warning(f"Signal {signal_id} not found for outcome recording")
            conn.close()
            return
        
        # Simple R-multiple based on outcome
        if outcome_type == "win":
            r_multiple = abs(exit_price - entry_price) / max(abs(entry_price * 0.01), 0.01)
        elif outcome_type == "loss":
            r_multiple = -abs(exit_price - entry_price) / max(abs(entry_price * 0.01), 0.01)
        else:
            r_multiple = 0
        
        cursor.execute("""
            INSERT INTO outcomes (
                signal_id, entry_price, exit_price, max_price, min_price,
                r_multiple, outcome_type, hit_target_1, hit_stop
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_id, entry_price, exit_price, max_price, min_price,
            r_multiple, outcome_type,
            1 if outcome_type == "win" else 0,
            1 if outcome_type == "loss" else 0,
        ))
        
        conn.commit()
        conn.close()
        
        log.info(f"Recorded outcome for signal {signal_id}: {outcome_type} (R={r_multiple:.2f})")
        
        # Update weights based on outcome
        self._update_weights_from_outcome(signal, outcome_type, r_multiple)
    
    def _update_weights_from_outcome(
        self,
        signal: Dict,
        outcome_type: str,
        r_multiple: float,
    ) -> List[LearningUpdate]:
        """Update weights based on signal outcome."""
        updates = []
        
        # Determine if this was a hit (win) or miss (loss)
        is_hit = outcome_type == "win"
        
        # Get features from the signal
        features_json = signal["features_json"] if signal["features_json"] else "{}"
        try:
            features = json.loads(features_json)
        except:
            features = {}
        
        # Update weights for active features
        active_features = []
        
        # Check which engines contributed
        if signal["score"] >= 50:
            setup_type = signal.get("setup_type", "")
            if "IGNITION" in setup_type or signal.get("social_velocity", 0) > 0:
                active_features.append("ignition")
            if signal.get("options_pressure_score", 0) > 0:
                active_features.append("pressure")
            if "SURGE" in setup_type or "BREAKOUT" in setup_type:
                active_features.append("surge")
        
        # Update feature weights based on the features dict
        for feature_name in features.get("active_features", []):
            if feature_name in ["volume_spike", "velocity", "news_catalyst", 
                               "vwap_reclaim", "range_breakout", "call_put_ratio", "iv_spike"]:
                active_features.append(feature_name)
        
        # Apply learning updates
        conn = get_connection()
        cursor = conn.cursor()
        
        for feature in set(active_features):
            # Get current weight
            cursor.execute(
                "SELECT weight, hit_count, miss_count FROM model_weights WHERE feature_name = ?",
                (feature,)
            )
            row = cursor.fetchone()
            
            if not row:
                continue
            
            old_weight = row["weight"]
            hit_count = row["hit_count"]
            miss_count = row["miss_count"]
            
            # Calculate weight adjustment
            if is_hit:
                # Reward: increase weight
                adjustment = self.LEARNING_RATE * (1 + r_multiple * 0.1)
                new_weight = old_weight + adjustment
                hit_count += 1
            else:
                # Penalty: decrease weight
                adjustment = self.LEARNING_RATE * (1 + abs(r_multiple) * 0.1)
                new_weight = old_weight - adjustment
                miss_count += 1
            
            # Clamp weight
            new_weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight))
            
            # Update database
            cursor.execute("""
                UPDATE model_weights
                SET weight = ?, hit_count = ?, miss_count = ?, last_updated = ?
                WHERE feature_name = ?
            """, (new_weight, hit_count, miss_count, datetime.utcnow().isoformat(), feature))
            
            updates.append(LearningUpdate(
                feature_name=feature,
                old_weight=old_weight,
                new_weight=new_weight,
                reason=f"{'Hit' if is_hit else 'Miss'} with R={r_multiple:.2f}",
                updated_at=datetime.utcnow(),
            ))
            
            log.debug(f"Weight update: {feature} {old_weight:.3f} -> {new_weight:.3f}")
        
        conn.commit()
        conn.close()
        
        return updates
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get all current feature weights."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT feature_name, weight FROM model_weights")
        rows = cursor.fetchall()
        conn.close()
        
        return {row["feature_name"]: row["weight"] for row in rows}
    
    def get_feature_stats(self) -> List[Dict]:
        """Get performance stats for each feature."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT feature_name, weight, hit_count, miss_count, last_updated
            FROM model_weights
            ORDER BY weight DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        stats = []
        for row in rows:
            total = row["hit_count"] + row["miss_count"]
            win_rate = row["hit_count"] / total if total > 0 else 0
            
            stats.append({
                "feature": row["feature_name"],
                "weight": row["weight"],
                "hits": row["hit_count"],
                "misses": row["miss_count"],
                "total": total,
                "win_rate": win_rate,
                "last_updated": row["last_updated"],
            })
        
        return stats
    
    def apply_daily_decay(self) -> None:
        """Apply daily decay to weights to adapt to changing markets."""
        conn = get_connection()
        cursor = conn.cursor()
        
        # Decay weights toward 1.0 (neutral)
        cursor.execute("SELECT feature_name, weight FROM model_weights")
        rows = cursor.fetchall()
        
        for row in rows:
            weight = row["weight"]
            # Decay toward 1.0
            new_weight = 1.0 + (weight - 1.0) * self.DECAY_FACTOR
            
            cursor.execute("""
                UPDATE model_weights
                SET weight = ?, last_updated = ?
                WHERE feature_name = ?
            """, (new_weight, datetime.utcnow().isoformat(), row["feature_name"]))
        
        conn.commit()
        conn.close()
        
        log.info("Applied daily weight decay")
    
    def get_learning_summary(self) -> Dict:
        """Get summary of learning performance."""
        stats = self.get_feature_stats()
        
        total_signals = sum(s["total"] for s in stats) // max(len(stats), 1)
        avg_win_rate = sum(s["win_rate"] for s in stats) / max(len(stats), 1)
        
        best_feature = max(stats, key=lambda x: x["weight"]) if stats else None
        worst_feature = min(stats, key=lambda x: x["weight"]) if stats else None
        
        return {
            "total_training_signals": total_signals,
            "average_win_rate": avg_win_rate,
            "best_feature": best_feature["feature"] if best_feature else None,
            "best_weight": best_feature["weight"] if best_feature else 1.0,
            "worst_feature": worst_feature["feature"] if worst_feature else None,
            "worst_weight": worst_feature["weight"] if worst_feature else 1.0,
            "feature_stats": stats,
        }


# Global instance
learning_memory = LearningMemory()
