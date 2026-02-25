"""
Layer 6: Strategy DNA (Your Historical Wins)

Purpose: Match current signal to YOUR winning trade patterns
Weight: 25% (HIGHEST)
Cost: $0 (local SQLite)
Latency: <5ms

Why highest weight:
- External AI doesn't know YOUR edge
- Your historical wins are the best predictor of YOUR future wins
- This layer learns from your specific style, timing, patterns

Fingerprint components:
- Ticker (SPY, QQQ, etc.)
- Hour of day (9-16 ET)
- Pattern type (breakout, reversal, etc.)
- Regime (RISK_ON, RISK_OFF, etc.)
- Direction (CALL, PUT)
"""

import os
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import pytz

from wsb_snake.utils.logger import get_logger
from wsb_snake.config import DATA_DIR

logger = get_logger(__name__)


@dataclass
class DNAMatch:
    """Result from DNA matching."""
    adjustment: float  # -0.25 to +0.25
    match_type: str  # "exact", "fuzzy", "none"
    matched_trades: int
    historical_win_rate: float
    historical_avg_pnl: float
    best_match_pnl: float
    latency_ms: float = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'adjustment': self.adjustment,
            'match_type': self.match_type,
            'matched_trades': self.matched_trades,
            'historical_win_rate': self.historical_win_rate,
            'historical_avg_pnl': self.historical_avg_pnl,
            'best_match_pnl': self.best_match_pnl,
            'latency_ms': self.latency_ms,
            'reason': self.reason
        }


class StrategyDNA:
    """
    Strategy DNA layer - matches to your historical winning patterns.

    Highest weight (25%) because YOUR data is the best predictor
    of YOUR future performance.
    """

    DB_PATH = os.path.join(DATA_DIR, "wsb_snake.db")

    def __init__(self):
        """Initialize Strategy DNA layer."""
        self._db_conn = None
        self._recipe_cache: Dict[str, Dict] = {}
        self._call_count = 0
        self._match_count = 0

        self._init_db()
        self._build_recipe_index()

    def _init_db(self):
        """Initialize database connection."""
        try:
            if os.path.exists(self.DB_PATH):
                self._db_conn = sqlite3.connect(self.DB_PATH, check_same_thread=False)
                self._db_conn.row_factory = sqlite3.Row
                logger.info(f"DNA_L6: Connected to {self.DB_PATH}")
            else:
                logger.warning(f"DNA_L6: Database not found at {self.DB_PATH}")
        except Exception as e:
            logger.error(f"DNA_L6: DB connection failed: {e}")

    def _build_recipe_index(self):
        """Build fingerprint → outcome mapping from trade history."""
        if not self._db_conn:
            return

        try:
            # Query closed trades from the last 60 days
            cutoff = (datetime.now() - timedelta(days=60)).isoformat()

            cursor = self._db_conn.execute("""
                SELECT
                    symbol,
                    entry_time,
                    direction,
                    pnl,
                    pnl_percent,
                    exit_reason
                FROM trades
                WHERE status = 'CLOSED'
                AND entry_time > ?
            """, (cutoff,))

            for row in cursor:
                try:
                    symbol = row['symbol']
                    entry_time = row['entry_time']
                    direction = row['direction']
                    pnl = row['pnl'] or 0
                    pnl_pct = row['pnl_percent'] or 0

                    # Create fingerprint
                    fingerprint = self._create_fingerprint(
                        symbol, entry_time, direction
                    )

                    # Store in cache
                    if fingerprint not in self._recipe_cache:
                        self._recipe_cache[fingerprint] = {
                            'wins': 0,
                            'losses': 0,
                            'total_pnl': 0,
                            'trades': []
                        }

                    recipe = self._recipe_cache[fingerprint]
                    if pnl > 0:
                        recipe['wins'] += 1
                    else:
                        recipe['losses'] += 1
                    recipe['total_pnl'] += pnl_pct
                    recipe['trades'].append(pnl_pct)

                except Exception as e:
                    logger.debug(f"DNA_L6: Row parse error: {e}")
                    continue

            logger.info(f"DNA_L6: Built {len(self._recipe_cache)} recipes from history")

        except Exception as e:
            logger.error(f"DNA_L6: Recipe build failed: {e}")

    def _create_fingerprint(
        self,
        symbol: str,
        entry_time: str,
        direction: str,
        pattern: str = None,
        regime: str = None
    ) -> str:
        """
        Create matchable fingerprint from trade attributes.

        Format: SYMBOL|HOUR|DIRECTION|PATTERN|REGIME
        """
        # Extract hour from entry time
        try:
            if isinstance(entry_time, str):
                dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
            else:
                dt = entry_time

            # Convert to ET
            et = pytz.timezone("US/Eastern")
            dt_et = dt.astimezone(et) if dt.tzinfo else et.localize(dt)
            hour = dt_et.hour
        except:
            hour = 12  # Default to midday

        # Normalize direction
        direction = direction.upper()
        if direction in ['LONG', 'BUY']:
            direction = 'CALL'
        elif direction in ['SHORT', 'SELL']:
            direction = 'PUT'

        parts = [
            symbol.upper() if symbol else 'SPY',
            str(hour),
            direction,
            (pattern or 'ANY').upper(),
            (regime or 'ANY').upper()
        ]

        return "|".join(parts)

    def score(
        self,
        signal: Dict[str, Any],
        pattern: str = None,
        regime: str = None
    ) -> DNAMatch:
        """
        Score signal based on DNA match to historical wins.

        Args:
            signal: Dict with 'ticker', 'direction', 'entry_time', etc.
            pattern: Optional pattern name from vision
            regime: Optional HYDRA regime

        Returns:
            DNAMatch with conviction adjustment
        """
        start = time.time()
        self._call_count += 1

        ticker = signal.get('ticker', signal.get('symbol', 'SPY'))
        direction = signal.get('direction', 'NEUTRAL')
        entry_time = signal.get('entry_time', datetime.now().isoformat())

        # Create current fingerprint
        current_fp = self._create_fingerprint(
            ticker, entry_time, direction, pattern, regime
        )

        # Try exact match first
        if current_fp in self._recipe_cache:
            recipe = self._recipe_cache[current_fp]
            result = self._score_recipe(recipe, "exact")
            result.latency_ms = (time.time() - start) * 1000
            if result.matched_trades > 0:
                self._match_count += 1
                logger.info(
                    f"DNA_L6: EXACT {ticker} {direction} adj={result.adjustment:+.2f} "
                    f"WR={result.historical_win_rate:.0%} trades={result.matched_trades}"
                )
            return result

        # Try fuzzy matches
        fuzzy_matches = self._fuzzy_match(current_fp)
        if fuzzy_matches:
            result = self._score_fuzzy_matches(fuzzy_matches)
            result.latency_ms = (time.time() - start) * 1000
            self._match_count += 1
            logger.debug(
                f"DNA_L6: FUZZY {ticker} {direction} adj={result.adjustment:+.2f} "
                f"WR={result.historical_win_rate:.0%}"
            )
            return result

        # No match
        return DNAMatch(
            adjustment=0,
            match_type="none",
            matched_trades=0,
            historical_win_rate=0,
            historical_avg_pnl=0,
            best_match_pnl=0,
            latency_ms=(time.time() - start) * 1000,
            reason="No historical match"
        )

    def _score_recipe(self, recipe: Dict, match_type: str) -> DNAMatch:
        """Score a single recipe match."""
        wins = recipe['wins']
        losses = recipe['losses']
        total = wins + losses

        if total == 0:
            return DNAMatch(
                adjustment=0,
                match_type=match_type,
                matched_trades=0,
                historical_win_rate=0,
                historical_avg_pnl=0,
                best_match_pnl=0,
                reason="Empty recipe"
            )

        win_rate = wins / total
        avg_pnl = recipe['total_pnl'] / total
        best_pnl = max(recipe['trades']) if recipe['trades'] else 0

        # Calculate adjustment based on win rate and magnitude
        # Scale: 25% max weight
        if win_rate >= 0.75:
            adjustment = min(avg_pnl / 100, 0.25)  # Cap at +25%
            reason = f"{win_rate:.0%} win rate across {total} trades"
        elif win_rate >= 0.60:
            adjustment = min(avg_pnl / 200, 0.15)  # Cap at +15%
            reason = f"{win_rate:.0%} win rate (moderate)"
        elif win_rate >= 0.50:
            adjustment = (win_rate - 0.5) * 0.25  # Scale 0-12.5%
            reason = f"{win_rate:.0%} win rate (break-even+)"
        elif win_rate >= 0.40:
            adjustment = (win_rate - 0.5) * 0.25  # Scale -2.5% to 0
            reason = f"{win_rate:.0%} win rate (below average)"
        else:
            adjustment = max(avg_pnl / 100, -0.25)  # Cap at -25%
            reason = f"{win_rate:.0%} win rate - AVOID THIS SETUP"

        return DNAMatch(
            adjustment=adjustment,
            match_type=match_type,
            matched_trades=total,
            historical_win_rate=win_rate,
            historical_avg_pnl=avg_pnl,
            best_match_pnl=best_pnl,
            reason=reason
        )

    def _fuzzy_match(self, fingerprint: str) -> List[Dict]:
        """Find fuzzy matches to fingerprint."""
        parts = fingerprint.split("|")
        if len(parts) < 3:
            return []

        symbol, hour, direction = parts[:3]
        matches = []

        for fp, recipe in self._recipe_cache.items():
            fp_parts = fp.split("|")
            if len(fp_parts) < 3:
                continue

            fp_symbol, fp_hour, fp_direction = fp_parts[:3]

            # Match criteria (any 2 of 3)
            symbol_match = fp_symbol == symbol
            hour_match = abs(int(fp_hour) - int(hour)) <= 1  # Within 1 hour
            direction_match = fp_direction == direction

            match_count = sum([symbol_match, hour_match, direction_match])

            if match_count >= 2:
                matches.append({
                    'recipe': recipe,
                    'match_strength': match_count / 3,
                    'fingerprint': fp
                })

        return matches

    def _score_fuzzy_matches(self, matches: List[Dict]) -> DNAMatch:
        """Score fuzzy matches."""
        total_wins = 0
        total_losses = 0
        total_pnl = 0
        all_trades = []
        total_weight = 0

        for match in matches:
            recipe = match['recipe']
            weight = match['match_strength']

            total_wins += recipe['wins'] * weight
            total_losses += recipe['losses'] * weight
            total_pnl += recipe['total_pnl'] * weight
            all_trades.extend(recipe['trades'])
            total_weight += weight

        if total_weight == 0:
            return DNAMatch(
                adjustment=0,
                match_type="fuzzy",
                matched_trades=0,
                historical_win_rate=0,
                historical_avg_pnl=0,
                best_match_pnl=0,
                reason="No fuzzy matches"
            )

        total = total_wins + total_losses
        win_rate = total_wins / total if total > 0 else 0
        avg_pnl = total_pnl / total_weight

        # Reduce adjustment for fuzzy matches (less confident)
        adjustment = (win_rate - 0.5) * 0.15  # Max ±7.5% for fuzzy

        return DNAMatch(
            adjustment=adjustment,
            match_type="fuzzy",
            matched_trades=len(matches),
            historical_win_rate=win_rate,
            historical_avg_pnl=avg_pnl,
            best_match_pnl=max(all_trades) if all_trades else 0,
            reason=f"Fuzzy match to {len(matches)} similar setups"
        )

    def record_trade_result(
        self,
        ticker: str,
        direction: str,
        entry_time: str,
        pnl_pct: float,
        pattern: str = None,
        regime: str = None
    ):
        """
        Record a trade result for future DNA matching.
        Call this after each trade closes.
        """
        fingerprint = self._create_fingerprint(
            ticker, entry_time, direction, pattern, regime
        )

        if fingerprint not in self._recipe_cache:
            self._recipe_cache[fingerprint] = {
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'trades': []
            }

        recipe = self._recipe_cache[fingerprint]
        if pnl_pct > 0:
            recipe['wins'] += 1
        else:
            recipe['losses'] += 1
        recipe['total_pnl'] += pnl_pct
        recipe['trades'].append(pnl_pct)

        logger.debug(f"DNA_L6: Recorded {ticker} {direction} {pnl_pct:+.1f}%")

    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics."""
        total_recipes = len(self._recipe_cache)
        winning_recipes = sum(
            1 for r in self._recipe_cache.values()
            if r['wins'] > r['losses']
        )

        return {
            'call_count': self._call_count,
            'match_count': self._match_count,
            'match_rate': self._match_count / max(self._call_count, 1),
            'total_recipes': total_recipes,
            'winning_recipes': winning_recipes,
            'recipe_win_rate': winning_recipes / max(total_recipes, 1)
        }


# Singleton
_strategy_dna = None

def get_strategy_dna() -> StrategyDNA:
    """Get singleton StrategyDNA instance."""
    global _strategy_dna
    if _strategy_dna is None:
        _strategy_dna = StrategyDNA()
    return _strategy_dna
