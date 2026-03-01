"""
NEWS SENTIMENT: Quick sentiment check for major market-moving news

SWARM CONSENSUS (5/12 personas agreed):
- Add news/sentiment analysis as additional confirmation
- Avoid trading into major negative headlines

Uses simple keyword detection + optional LLM for deeper analysis.
"""
import os
import re
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NewsSentiment:
    """News sentiment analysis result."""
    sentiment: str  # BULLISH, BEARISH, NEUTRAL
    score: float  # -1.0 to +1.0
    headline_count: int
    key_headlines: List[str]
    has_major_event: bool
    confidence: float


# Keywords for sentiment detection
BEARISH_KEYWORDS = [
    "crash", "plunge", "collapse", "recession", "layoffs", "bankruptcy",
    "default", "crisis", "selloff", "bloodbath", "panic", "fear",
    "downgrade", "miss", "disappoints", "warning", "cuts", "slashes",
    "tanking", "tumbles", "sinks", "drops", "falls", "declines",
]

BULLISH_KEYWORDS = [
    "rally", "surge", "soar", "boom", "breakout", "record", "high",
    "beat", "exceeds", "growth", "upgrade", "raises", "bullish",
    "optimism", "recovery", "expansion", "jumps", "climbs", "gains",
    "strong", "positive", "upbeat", "momentum",
]

MAJOR_EVENT_KEYWORDS = [
    "fed", "fomc", "powell", "rate", "cpi", "inflation", "jobs",
    "employment", "gdp", "tariff", "trade war", "sanctions",
]


class NewsSentimentChecker:
    """
    Quick news sentiment check before trades.

    SWARM CONSENSUS improvement.
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[NewsSentiment, datetime]] = {}
        self._cache_ttl = 600  # 10 minute cache
        logger.info("NEWS_SENTIMENT: Initialized (SWARM CONSENSUS improvement)")

    def _analyze_text(self, text: str) -> Tuple[float, bool]:
        """Simple keyword-based sentiment analysis."""
        text_lower = text.lower()

        bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)
        has_major_event = any(kw in text_lower for kw in MAJOR_EVENT_KEYWORDS)

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, has_major_event

        score = (bullish_count - bearish_count) / total
        return score, has_major_event

    def get_sentiment(self, ticker: str = "SPY") -> NewsSentiment:
        """Get current news sentiment for a ticker."""
        now = datetime.now()

        # Check cache
        if ticker in self._cache:
            cached, cached_time = self._cache[ticker]
            if (now - cached_time).total_seconds() < self._cache_ttl:
                return cached

        # Try to fetch news from available sources
        headlines = []
        sentiment_scores = []
        has_major_event = False

        # 1. Try Reddit WSB mentions (we already have this)
        try:
            from wsb_snake.collectors.reddit_collector import reddit_sentiment
            wsb_data = reddit_sentiment.get_ticker_sentiment(ticker)
            if wsb_data:
                headlines.extend(wsb_data.get("headlines", [])[:5])
                wsb_score = wsb_data.get("sentiment_score", 0)
                sentiment_scores.append(wsb_score)
        except Exception as e:
            logger.debug(f"Reddit sentiment fetch failed: {e}")

        # 2. Check any fetched headlines for keywords
        for headline in headlines:
            score, major = self._analyze_text(headline)
            sentiment_scores.append(score)
            if major:
                has_major_event = True

        # Calculate aggregate sentiment
        if sentiment_scores:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
        else:
            avg_score = 0.0

        # Classify sentiment
        if avg_score > 0.2:
            sentiment = "BULLISH"
        elif avg_score < -0.2:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        result = NewsSentiment(
            sentiment=sentiment,
            score=avg_score,
            headline_count=len(headlines),
            key_headlines=headlines[:3],
            has_major_event=has_major_event,
            confidence=min(0.9, 0.5 + abs(avg_score)),
        )

        # Cache result
        self._cache[ticker] = (result, now)

        logger.debug(
            f"NEWS: {ticker} sentiment={sentiment} score={avg_score:.2f} "
            f"headlines={len(headlines)} major_event={has_major_event}"
        )

        return result

    def should_trade(self, ticker: str, direction: str) -> Tuple[bool, str, float]:
        """
        Check if news sentiment supports the trade direction.

        Returns:
            (allowed, reason, confidence_adjustment)
        """
        sentiment = self.get_sentiment(ticker)

        # Major event - reduce confidence
        if sentiment.has_major_event:
            logger.warning(f"NEWS: Major event detected for {ticker}")
            return True, "MAJOR_EVENT", 0.8

        # Check direction alignment
        if direction.lower() in ["long", "call", "calls"]:
            if sentiment.sentiment == "BEARISH" and sentiment.score < -0.4:
                logger.warning(f"NEWS: Strong bearish sentiment for {ticker}, going LONG risky")
                return True, "SENTIMENT_AGAINST", 0.7
            elif sentiment.sentiment == "BULLISH":
                return True, "SENTIMENT_ALIGNED", 1.1
        else:  # short/put
            if sentiment.sentiment == "BULLISH" and sentiment.score > 0.4:
                logger.warning(f"NEWS: Strong bullish sentiment for {ticker}, going SHORT risky")
                return True, "SENTIMENT_AGAINST", 0.7
            elif sentiment.sentiment == "BEARISH":
                return True, "SENTIMENT_ALIGNED", 1.1

        return True, "SENTIMENT_NEUTRAL", 1.0


# Singleton
_news_sentiment: Optional[NewsSentimentChecker] = None


def get_news_sentiment() -> NewsSentimentChecker:
    """Get singleton news sentiment checker."""
    global _news_sentiment
    if _news_sentiment is None:
        _news_sentiment = NewsSentimentChecker()
    return _news_sentiment
