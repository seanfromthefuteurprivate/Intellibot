"""
Trade Learner - Applies Learned Trade Patterns to Live Trading

Takes patterns learned from screenshots and applies them to boost
or filter trading signals in real-time.

Integration points:
- Called by probability_generator before signal output
- Boosts confidence for setups matching winning patterns
- Warns against setups matching losing patterns
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from wsb_snake.utils.logger import get_logger
from wsb_snake.db.database import get_connection

logger = get_logger(__name__)


class TradeLearner:
    """
    Applies learned trade patterns to improve signal quality.

    Learns from:
    - Screenshot-extracted trades (via trade_extractor)
    - Live trade outcomes (via outcome_recorder)

    Applies to:
    - Signal confidence boosting
    - Pattern matching
    - Time-of-day optimization
    """

    # Boost/penalty multipliers
    STRONG_RECIPE_BOOST = 0.15      # +15% confidence for matching high-WR recipe
    WEAK_RECIPE_BOOST = 0.08        # +8% for matching moderate recipe
    LOSING_PATTERN_PENALTY = -0.10  # -10% for matching losing patterns

    def __init__(self):
        self._load_active_recipes()
        logger.info(f"TradeLearner initialized with {len(self._recipes)} active recipes")

    def _load_active_recipes(self):
        """Load active trade recipes from database."""
        conn = get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM trade_recipes
                WHERE is_active = 1 AND source_trade_count >= 2
                ORDER BY win_rate DESC
            """)
            rows = cursor.fetchall()
            self._recipes = [dict(row) for row in rows]
        except Exception as e:
            logger.warning(f"Could not load trade recipes: {e}")
            self._recipes = []

        conn.close()

    def reload_recipes(self):
        """Reload recipes from database (call after new learning)."""
        self._load_active_recipes()
        logger.info(f"Reloaded {len(self._recipes)} trade recipes")

    def get_confidence_adjustment(
        self,
        ticker: str,
        trade_type: str,
        current_hour: int,
        pattern: Optional[str] = None
    ) -> Tuple[float, List[str]]:
        """
        Get confidence adjustment based on learned patterns.

        Args:
            ticker: Stock symbol (e.g., "SPY")
            trade_type: "CALLS" or "PUTS"
            current_hour: Current hour (ET)
            pattern: Detected pattern if any

        Returns:
            Tuple of (adjustment_multiplier, list_of_reasons)
            e.g., (0.15, ["Matches SPY_CALLS_breakout_power_hour recipe (WR: 75%)"])
        """
        time_window = self._get_time_window(current_hour)
        total_adjustment = 0.0
        reasons = []

        for recipe in self._recipes:
            # Check ticker match
            if recipe["ticker_pattern"] != ticker and recipe["ticker_pattern"] != "ANY":
                continue

            # Check trade type match
            if recipe["trade_type"] and recipe["trade_type"] != trade_type:
                continue

            # Check time window match
            if recipe["time_window"] and recipe["time_window"] != time_window and recipe["time_window"] != "any":
                continue

            # Check pattern match if specified
            if pattern and recipe.get("entry_conditions"):
                try:
                    conditions = json.loads(recipe["entry_conditions"])
                    if conditions.get("pattern") and conditions["pattern"] != pattern:
                        continue
                except:
                    pass

            # Calculate boost based on recipe quality
            win_rate = recipe.get("win_rate", 0.5)
            trade_count = recipe.get("source_trade_count", 0)

            if win_rate >= 0.7 and trade_count >= 3:
                boost = self.STRONG_RECIPE_BOOST
                reasons.append(f"Strong match: {recipe['name']} (WR: {win_rate:.0%}, n={trade_count})")
            elif win_rate >= 0.6:
                boost = self.WEAK_RECIPE_BOOST
                reasons.append(f"Moderate match: {recipe['name']} (WR: {win_rate:.0%})")
            else:
                boost = 0
                continue

            total_adjustment += boost

            # Only apply top 2 matching recipes
            if len(reasons) >= 2:
                break

        # Cap total adjustment
        total_adjustment = min(0.25, max(-0.15, total_adjustment))

        return total_adjustment, reasons

    def should_replicate_trade(
        self,
        ticker: str,
        trade_type: str,
        current_hour: int
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if current conditions match a high-confidence learned trade.
        Used for proactive trade suggestions.

        Returns:
            Tuple of (should_trade, matching_recipe_or_none)
        """
        time_window = self._get_time_window(current_hour)

        for recipe in self._recipes:
            # Must have strong stats
            if recipe.get("win_rate", 0) < 0.7:
                continue
            if recipe.get("source_trade_count", 0) < 3:
                continue

            # Check matches
            if recipe["ticker_pattern"] != ticker:
                continue
            if recipe["trade_type"] and recipe["trade_type"] != trade_type:
                continue
            if recipe["time_window"] and recipe["time_window"] != time_window:
                continue

            logger.info(f"Replication candidate: {recipe['name']} matches current setup")
            return True, recipe

        return False, None

    def record_replication_result(
        self,
        recipe_id: int,
        was_successful: bool,
        pnl_pct: float
    ):
        """
        Record the result of attempting to replicate a learned trade.
        Updates recipe statistics.
        """
        conn = get_connection()
        cursor = conn.cursor()

        if was_successful:
            cursor.execute("""
                UPDATE trade_recipes
                SET replication_count = replication_count + 1,
                    replication_success = replication_success + 1,
                    last_used = ?,
                    updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), recipe_id))
        else:
            cursor.execute("""
                UPDATE trade_recipes
                SET replication_count = replication_count + 1,
                    last_used = ?,
                    updated_at = ?
                WHERE id = ?
            """, (datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), recipe_id))

        conn.commit()
        conn.close()

        logger.info(f"Recorded replication result for recipe #{recipe_id}: {'success' if was_successful else 'failure'}")

    def get_learned_trade_stats(self) -> Dict:
        """Get statistics about learned trades."""
        conn = get_connection()
        cursor = conn.cursor()

        stats = {}

        # Total learned trades
        cursor.execute("SELECT COUNT(*) as count FROM learned_trades")
        stats["total_learned_trades"] = cursor.fetchone()["count"]

        # Winning vs losing
        cursor.execute("""
            SELECT
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END) as losers,
                AVG(profit_loss_pct) as avg_pnl_pct,
                SUM(profit_loss) as total_pnl
            FROM learned_trades
            WHERE profit_loss IS NOT NULL
        """)
        row = cursor.fetchone()
        stats["winners"] = row["winners"] or 0
        stats["losers"] = row["losers"] or 0
        stats["avg_pnl_pct"] = row["avg_pnl_pct"] or 0
        stats["total_pnl"] = row["total_pnl"] or 0

        # Active recipes
        cursor.execute("SELECT COUNT(*) as count FROM trade_recipes WHERE is_active = 1")
        stats["active_recipes"] = cursor.fetchone()["count"]

        # Top recipes
        cursor.execute("""
            SELECT name, ticker_pattern, trade_type, win_rate, source_trade_count, avg_profit_pct
            FROM trade_recipes
            WHERE is_active = 1 AND source_trade_count >= 2
            ORDER BY win_rate DESC, source_trade_count DESC
            LIMIT 5
        """)
        stats["top_recipes"] = [dict(row) for row in cursor.fetchall()]

        # By platform
        cursor.execute("""
            SELECT platform, COUNT(*) as count, AVG(profit_loss_pct) as avg_pnl
            FROM learned_trades
            WHERE platform IS NOT NULL
            GROUP BY platform
        """)
        stats["by_platform"] = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return stats

    def get_ticker_insights(self, ticker: str) -> Dict:
        """Get learned insights for a specific ticker."""
        conn = get_connection()
        cursor = conn.cursor()

        insights = {"ticker": ticker}

        # Trade history for this ticker
        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winners,
                AVG(profit_loss_pct) as avg_pnl_pct,
                AVG(CASE WHEN profit_loss > 0 THEN profit_loss_pct ELSE NULL END) as avg_win_pct,
                AVG(CASE WHEN profit_loss <= 0 THEN profit_loss_pct ELSE NULL END) as avg_loss_pct
            FROM learned_trades
            WHERE ticker = ?
        """, (ticker,))

        row = cursor.fetchone()
        insights["total_trades"] = row["total_trades"] or 0
        insights["winners"] = row["winners"] or 0
        insights["win_rate"] = (row["winners"] or 0) / max(1, row["total_trades"] or 1)
        insights["avg_pnl_pct"] = row["avg_pnl_pct"] or 0
        insights["avg_win_pct"] = row["avg_win_pct"] or 0
        insights["avg_loss_pct"] = row["avg_loss_pct"] or 0

        # Best trade type
        cursor.execute("""
            SELECT trade_type, COUNT(*) as count, AVG(profit_loss_pct) as avg_pnl
            FROM learned_trades
            WHERE ticker = ? AND profit_loss > 0
            GROUP BY trade_type
            ORDER BY avg_pnl DESC
            LIMIT 1
        """, (ticker,))

        row = cursor.fetchone()
        if row:
            insights["best_trade_type"] = row["trade_type"]
            insights["best_type_avg_pnl"] = row["avg_pnl"]

        # Best time of day
        cursor.execute("""
            SELECT entry_time, profit_loss_pct
            FROM learned_trades
            WHERE ticker = ? AND entry_time IS NOT NULL AND profit_loss > 0
            ORDER BY profit_loss_pct DESC
            LIMIT 3
        """, (ticker,))

        insights["best_entry_times"] = [row["entry_time"] for row in cursor.fetchall()]

        # Matching recipes
        cursor.execute("""
            SELECT name, win_rate, avg_profit_pct
            FROM trade_recipes
            WHERE ticker_pattern = ? AND is_active = 1
            ORDER BY win_rate DESC
        """, (ticker,))

        insights["recipes"] = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return insights

    def _get_time_window(self, hour: int) -> str:
        """Convert hour to time window name."""
        if hour < 10:
            return "open"
        elif hour < 12:
            return "morning"
        elif hour < 14:
            return "midday"
        elif hour < 15:
            return "afternoon"
        else:
            return "power_hour"


# Global instance
trade_learner = TradeLearner()
