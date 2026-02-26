"""
Benzinga News Adapter

Fetches real-time news for ticker-specific and macro events.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from wsb_snake.config import BENZINGA_API_KEY, BENZINGA_BASE_URL
from wsb_snake.utils.logger import log


class BenzingaNewsAdapter:
    """Adapter for Benzinga news API."""
    
    def __init__(self):
        self.api_key = BENZINGA_API_KEY
        self.base_url = BENZINGA_BASE_URL
        
    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make authenticated request to Benzinga API."""
        if not self.api_key:
            log.error("BENZINGA_API_KEY not set")
            return None
            
        if params is None:
            params = {}
        params["token"] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        headers = {"accept": "application/json"}
        
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                # Handle both JSON and potential XML responses
                content_type = resp.headers.get('content-type', '')
                if 'json' in content_type:
                    return resp.json()
                else:
                    log.warning(f"Benzinga returned non-JSON: {content_type}")
                    return None
            else:
                log.error(f"Benzinga API error {resp.status_code}: {resp.text[:200]}")
                return None
        except Exception as e:
            log.error(f"Benzinga request failed: {e}")
            return None
    
    def get_news(
        self,
        tickers: List[str] = None,
        page_size: int = 20,
        hours_back: int = 4
    ) -> List[Dict]:
        """
        Get recent news headlines.
        
        Args:
            tickers: List of tickers to filter (None = all)
            page_size: Number of articles to fetch
            hours_back: How far back to look
            
        Returns:
            List of news articles
        """
        # Calculate date range - Benzinga expects YYYY-MM-DD format
        now = datetime.utcnow()
        date_from = (now - timedelta(hours=hours_back)).strftime("%Y-%m-%d")
        
        params = {
            "pageSize": page_size,
            "date_from": date_from,
            "sort": "created:desc",
            "accept": "application/json",
        }
        
        if tickers:
            params["tickers"] = ",".join(tickers)
        
        # Use news endpoint (base URL already includes /api/v2)
        data = self._request("/news", params)
        
        if data:
            articles = []
            for article in data if isinstance(data, list) else data.get("data", []):
                parsed = self._parse_article(article)
                if parsed:
                    articles.append(parsed)
            
            log.info(f"Got {len(articles)} news articles from Benzinga")
            return articles
        
        return []
    
    def _parse_article(self, article: Dict) -> Optional[Dict]:
        """Parse a Benzinga article into standardized format."""
        try:
            created_at = article.get("created", "")

            # === DATA PLUMBER: Calculate news age and decay factor ===
            age_minutes = self._calculate_news_age_minutes(created_at)
            decay_factor = self._calculate_news_decay(age_minutes)

            return {
                "id": article.get("id", ""),
                "headline": article.get("title", ""),
                "summary": article.get("teaser", ""),
                "tickers": article.get("stocks", []),
                "created_at": created_at,
                "updated_at": article.get("updated", ""),
                "url": article.get("url", ""),
                "source": "benzinga",
                "channels": article.get("channels", []),
                "importance": self._score_importance(article),
                # === DATA PLUMBER: News freshness metadata ===
                "age_minutes": age_minutes,
                "decay_factor": decay_factor,
                "is_stale": age_minutes > 30,  # News > 30 min is stale
            }
        except Exception as e:
            log.warning(f"Failed to parse article: {e}")
            return None

    def _calculate_news_age_minutes(self, created_at: str) -> float:
        """Calculate how many minutes old the news is."""
        if not created_at:
            return 999  # Treat missing timestamp as very old

        try:
            # Benzinga format: "2024-01-15T14:30:00-05:00" or similar
            # Try multiple formats
            formats = [
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
            ]

            created_dt = None
            for fmt in formats:
                try:
                    created_dt = datetime.strptime(created_at[:26], fmt[:len(created_at)])
                    break
                except ValueError:
                    continue

            if created_dt is None:
                return 999

            # Make timezone-naive for comparison
            if created_dt.tzinfo is not None:
                created_dt = created_dt.replace(tzinfo=None)

            age = datetime.utcnow() - created_dt
            return age.total_seconds() / 60.0

        except Exception:
            return 999

    def _calculate_news_decay(self, age_minutes: float) -> float:
        """
        Calculate impact decay factor for stale news.

        === DATA PLUMBER: Stale news filter ===
        - News < 15 min: 100% impact (no decay)
        - News 15-30 min: Linear decay from 100% to 50%
        - News > 30 min: 50% impact (heavily discounted)
        - News > 60 min: 25% impact
        - News > 120 min: 10% impact (effectively ignored)
        """
        if age_minutes <= 15:
            return 1.0  # Fresh news - full impact
        elif age_minutes <= 30:
            # Linear decay from 1.0 to 0.5 over 15 minutes
            return 1.0 - (age_minutes - 15) / 30.0
        elif age_minutes <= 60:
            return 0.5  # Stale - half impact
        elif age_minutes <= 120:
            return 0.25  # Very stale - quarter impact
        else:
            return 0.10  # Old news - minimal impact
    
    def _score_importance(self, article: Dict) -> str:
        """
        Score article importance for trading.
        
        Returns: "high", "medium", or "low"
        """
        headline = article.get("title", "").lower()
        channels = [c.get("name", "").lower() for c in article.get("channels", [])]
        
        # High importance keywords
        high_keywords = [
            "earnings", "fda", "sec", "merger", "acquisition", "buyout",
            "upgrade", "downgrade", "halt", "resume", "guidance", "forecast",
            "beat", "miss", "surprise", "breaking", "alert"
        ]
        
        # Check for high importance
        for kw in high_keywords:
            if kw in headline:
                return "high"
        
        # Medium importance
        medium_keywords = [
            "analyst", "price target", "rating", "outlook", "revenue",
            "profit", "loss", "growth", "decline", "market"
        ]
        
        for kw in medium_keywords:
            if kw in headline:
                return "medium"
        
        return "low"
    
    def get_ticker_news(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get news for a specific ticker."""
        return self.get_news(tickers=[ticker], page_size=limit)
    
    def classify_headline(self, headline: str) -> Dict:
        """
        Classify a headline for trading relevance.
        
        Returns:
            Dict with category, sentiment, urgency
        """
        headline_lower = headline.lower()
        
        # Category detection
        category = "general"
        if any(kw in headline_lower for kw in ["earnings", "revenue", "profit", "eps"]):
            category = "earnings"
        elif any(kw in headline_lower for kw in ["fda", "approval", "trial", "drug"]):
            category = "fda"
        elif any(kw in headline_lower for kw in ["merger", "acquisition", "buyout", "deal"]):
            category = "ma"
        elif any(kw in headline_lower for kw in ["sec", "lawsuit", "investigation", "fine"]):
            category = "regulatory"
        elif any(kw in headline_lower for kw in ["fed", "fomc", "rate", "inflation", "cpi", "jobs"]):
            category = "macro"
        elif any(kw in headline_lower for kw in ["upgrade", "downgrade", "price target", "rating"]):
            category = "analyst"
        
        # Sentiment detection
        sentiment = "neutral"
        bullish = ["beat", "surge", "rally", "jump", "soar", "upgrade", "buy", "outperform", "positive"]
        bearish = ["miss", "plunge", "crash", "fall", "drop", "downgrade", "sell", "underperform", "negative"]
        
        if any(kw in headline_lower for kw in bullish):
            sentiment = "bullish"
        elif any(kw in headline_lower for kw in bearish):
            sentiment = "bearish"
        
        # Urgency detection
        urgency = "normal"
        if any(kw in headline_lower for kw in ["breaking", "alert", "just", "now", "halt"]):
            urgency = "high"
        
        return {
            "category": category,
            "sentiment": sentiment,
            "urgency": urgency,
        }


# Global instance
benzinga_news = BenzingaNewsAdapter()
