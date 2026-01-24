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
        
        # Use v2.1 endpoint for JSON response
        data = self._request("/v2.1/news", params)
        
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
            return {
                "id": article.get("id", ""),
                "headline": article.get("title", ""),
                "summary": article.get("teaser", ""),
                "tickers": article.get("stocks", []),
                "created_at": article.get("created", ""),
                "updated_at": article.get("updated", ""),
                "url": article.get("url", ""),
                "source": "benzinga",
                "channels": article.get("channels", []),
                "importance": self._score_importance(article),
            }
        except Exception as e:
            log.warning(f"Failed to parse article: {e}")
            return None
    
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
