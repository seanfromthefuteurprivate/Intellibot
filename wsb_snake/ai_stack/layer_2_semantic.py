"""
Layer 2: Titan Embeddings V2 (Semantic News Matching)

Purpose: Compare current news context to historical winning/losing setups
Weight: 10%
Cost: ~$0.0001 per call
Latency: ~100ms

Different from HYDRA Layer 11 (which matches trading day fingerprints).
This layer focuses on NEWS CONTEXT similarity:
- "Fed hawkish" news → similar to past setups with that context
- "CPI hot" news → match to historical CPI reaction trades
"""

import json
import time
import sqlite3
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from wsb_snake.utils.logger import get_logger
from wsb_snake.config import DATA_DIR

logger = get_logger(__name__)


@dataclass
class SemanticMatch:
    """Result from semantic matching."""
    adjustment: float  # -0.15 to +0.15 conviction adjustment
    best_winner_similarity: float  # 0-1
    best_loser_similarity: float  # 0-1
    matched_setups: int
    match_reason: str
    latency_ms: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'adjustment': self.adjustment,
            'best_winner_similarity': self.best_winner_similarity,
            'best_loser_similarity': self.best_loser_similarity,
            'matched_setups': self.matched_setups,
            'match_reason': self.match_reason,
            'latency_ms': self.latency_ms
        }


class SemanticLayer:
    """
    Titan Embeddings V2 layer for news semantic matching.

    Compares current news context to historical trade setups
    to find similar winning/losing patterns.
    """

    # Database path for storing embeddings
    DB_PATH = os.path.join(DATA_DIR, "semantic_embeddings.db")

    def __init__(self):
        """Initialize semantic layer."""
        self._bedrock = None
        self._db_conn = None
        self._embedding_cache: Dict[str, List[float]] = {}

        # Initialize Bedrock client
        try:
            from .bedrock_client import get_bedrock_client
            self._bedrock = get_bedrock_client()
            logger.info("SEMANTIC_L2: Bedrock client initialized")
        except Exception as e:
            logger.warning(f"SEMANTIC_L2: Bedrock unavailable: {e}")

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize embeddings database."""
        try:
            os.makedirs(os.path.dirname(self.DB_PATH), exist_ok=True)
            self._db_conn = sqlite3.connect(self.DB_PATH, check_same_thread=False)

            self._db_conn.execute("""
                CREATE TABLE IF NOT EXISTS setup_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    timestamp TEXT,
                    news_context TEXT,
                    embedding BLOB,
                    outcome TEXT,  -- WIN or LOSS
                    pnl_pct REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self._db_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_outcome ON setup_embeddings(outcome)
            """)

            self._db_conn.commit()
            logger.debug("SEMANTIC_L2: Database initialized")

        except Exception as e:
            logger.error(f"SEMANTIC_L2: DB init failed: {e}")

    def match(
        self,
        ticker: str,
        news_headlines: List[str],
        direction: str = "NEUTRAL"
    ) -> SemanticMatch:
        """
        Match current news context to historical setups.

        Args:
            ticker: Symbol being traded
            news_headlines: Recent news headlines (last 50)
            direction: CALL, PUT, or NEUTRAL

        Returns:
            SemanticMatch with conviction adjustment
        """
        start = time.time()

        if not self._bedrock:
            return SemanticMatch(
                adjustment=0,
                best_winner_similarity=0,
                best_loser_similarity=0,
                matched_setups=0,
                match_reason="Bedrock unavailable",
                latency_ms=(time.time() - start) * 1000
            )

        if not news_headlines:
            return SemanticMatch(
                adjustment=0,
                best_winner_similarity=0,
                best_loser_similarity=0,
                matched_setups=0,
                match_reason="No news context",
                latency_ms=(time.time() - start) * 1000
            )

        # Create news context string
        news_context = f"Ticker: {ticker}, Direction: {direction}\n"
        news_context += "\n".join(news_headlines[:20])  # Top 20 headlines

        # Get embedding for current context
        current_embedding = self._get_embedding(news_context)
        if not current_embedding:
            return SemanticMatch(
                adjustment=0,
                best_winner_similarity=0,
                best_loser_similarity=0,
                matched_setups=0,
                match_reason="Embedding failed",
                latency_ms=(time.time() - start) * 1000
            )

        # Find similar historical setups
        winner_sims, loser_sims = self._find_similar_setups(current_embedding, ticker)

        latency = (time.time() - start) * 1000

        # Calculate adjustment
        best_winner = max(winner_sims) if winner_sims else 0
        best_loser = max(loser_sims) if loser_sims else 0

        # Net similarity difference → conviction adjustment
        # If matches winners more than losers → positive adjustment
        adjustment = (best_winner - best_loser) * 0.15  # Scale to ±15%

        # Determine reason
        if best_winner > 0.85:
            reason = f"Strong match to {len(winner_sims)} winning setups"
        elif best_loser > 0.85:
            reason = f"Warning: matches {len(loser_sims)} losing setups"
        elif len(winner_sims) + len(loser_sims) == 0:
            reason = "No historical matches found"
        else:
            reason = f"Partial match ({len(winner_sims)}W/{len(loser_sims)}L setups)"

        logger.info(
            f"SEMANTIC_L2: {ticker} adj={adjustment:+.2f} "
            f"winner_sim={best_winner:.2f} loser_sim={best_loser:.2f} "
            f"in {latency:.0f}ms"
        )

        return SemanticMatch(
            adjustment=adjustment,
            best_winner_similarity=best_winner,
            best_loser_similarity=best_loser,
            matched_setups=len(winner_sims) + len(loser_sims),
            match_reason=reason,
            latency_ms=latency
        )

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text, with caching."""
        # Check cache first
        cache_key = text[:100]  # Use first 100 chars as key
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            embedding = self._bedrock.embed(text)
            if embedding:
                self._embedding_cache[cache_key] = embedding
                # Limit cache size
                if len(self._embedding_cache) > 1000:
                    # Remove oldest entries
                    keys = list(self._embedding_cache.keys())[:500]
                    for k in keys:
                        del self._embedding_cache[k]
            return embedding
        except Exception as e:
            logger.error(f"SEMANTIC_L2: Embedding error: {e}")
            return None

    def _find_similar_setups(
        self,
        current_embedding: List[float],
        ticker: str
    ) -> Tuple[List[float], List[float]]:
        """Find similar historical setups."""
        winner_sims = []
        loser_sims = []

        if not self._db_conn:
            return winner_sims, loser_sims

        try:
            # Get historical setups
            cursor = self._db_conn.execute("""
                SELECT embedding, outcome, pnl_pct
                FROM setup_embeddings
                WHERE ticker = ? OR ticker = 'ALL'
                ORDER BY created_at DESC
                LIMIT 100
            """, (ticker,))

            for row in cursor:
                try:
                    embedding_blob = row[0]
                    outcome = row[1]

                    # Deserialize embedding
                    historical_embedding = np.frombuffer(embedding_blob, dtype=np.float32).tolist()

                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(current_embedding, historical_embedding)

                    if similarity > 0.5:  # Only consider meaningful matches
                        if outcome == 'WIN':
                            winner_sims.append(similarity)
                        else:
                            loser_sims.append(similarity)

                except Exception as e:
                    logger.debug(f"SEMANTIC_L2: Setup parse error: {e}")
                    continue

        except Exception as e:
            logger.error(f"SEMANTIC_L2: DB query error: {e}")

        return winner_sims, loser_sims

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a_arr = np.array(a)
            b_arr = np.array(b)

            dot_product = np.dot(a_arr, b_arr)
            norm_a = np.linalg.norm(a_arr)
            norm_b = np.linalg.norm(b_arr)

            if norm_a == 0 or norm_b == 0:
                return 0

            return dot_product / (norm_a * norm_b)

        except Exception:
            return 0

    def store_setup(
        self,
        ticker: str,
        news_context: str,
        outcome: str,
        pnl_pct: float
    ):
        """
        Store a trade setup for future matching.
        Call this after trade closes.

        Args:
            ticker: Symbol traded
            news_context: News headlines at time of trade
            outcome: 'WIN' or 'LOSS'
            pnl_pct: Percentage P&L
        """
        if not self._bedrock or not self._db_conn:
            return

        try:
            embedding = self._get_embedding(news_context)
            if not embedding:
                return

            # Serialize embedding
            embedding_blob = np.array(embedding, dtype=np.float32).tobytes()

            self._db_conn.execute("""
                INSERT INTO setup_embeddings (ticker, news_context, embedding, outcome, pnl_pct)
                VALUES (?, ?, ?, ?, ?)
            """, (ticker, news_context[:500], embedding_blob, outcome, pnl_pct))

            self._db_conn.commit()
            logger.debug(f"SEMANTIC_L2: Stored {outcome} setup for {ticker}")

        except Exception as e:
            logger.error(f"SEMANTIC_L2: Store failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get layer statistics."""
        setup_count = 0
        if self._db_conn:
            try:
                cursor = self._db_conn.execute("SELECT COUNT(*) FROM setup_embeddings")
                setup_count = cursor.fetchone()[0]
            except:
                pass

        return {
            'stored_setups': setup_count,
            'cache_size': len(self._embedding_cache),
            'bedrock_stats': self._bedrock.get_stats() if self._bedrock else {}
        }


# Singleton
_semantic_layer = None

def get_semantic_layer() -> SemanticLayer:
    """Get singleton SemanticLayer instance."""
    global _semantic_layer
    if _semantic_layer is None:
        _semantic_layer = SemanticLayer()
    return _semantic_layer
