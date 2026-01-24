"""
Alpaca News Adapter

Fetches news from Alpaca's data API.
Uses the v1beta1/news endpoint.
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from wsb_snake.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_NEWS_URL
from wsb_snake.utils.logger import log


class AlpacaNewsAdapter:
    """Adapter for Alpaca news API."""
    
    def __init__(self):
        self.api_key = ALPACA_API_KEY
        self.secret_key = ALPACA_SECRET_KEY
        self.base_url = ALPACA_NEWS_URL
        
    def _get_headers(self) -> Dict:
        """Get authentication headers."""
        return {
            "APCA-API-KEY-ID": self.api_key or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
        }
    
    def get_news(
        self,
        symbols: List[str] = None,
        limit: int = 20,
        include_content: bool = False,
    ) -> List[Dict]:
        """
        Get recent news from Alpaca.
        
        Args:
            symbols: List of symbols to filter (None = all)
            limit: Max articles to fetch
            include_content: Include full article content
            
        Returns:
            List of news articles
        """
        if not self.api_key or not self.secret_key:
            log.error("Alpaca API keys not set")
            return []
        
        params = {
            "limit": limit,
            "include_content": str(include_content).lower(),
            "sort": "desc",
        }
        
        if symbols:
            params["symbols"] = ",".join(symbols)
        
        try:
            resp = requests.get(
                self.base_url,
                headers=self._get_headers(),
                params=params,
                timeout=10
            )
            
            if resp.status_code == 200:
                data = resp.json()
                articles = []
                
                for article in data.get("news", []):
                    parsed = self._parse_article(article)
                    if parsed:
                        articles.append(parsed)
                
                log.info(f"Got {len(articles)} news articles from Alpaca")
                return articles
            else:
                log.error(f"Alpaca news API error {resp.status_code}: {resp.text[:200]}")
                return []
                
        except Exception as e:
            log.error(f"Alpaca news request failed: {e}")
            return []
    
    def _parse_article(self, article: Dict) -> Optional[Dict]:
        """Parse an Alpaca article into standardized format."""
        try:
            return {
                "id": article.get("id", ""),
                "headline": article.get("headline", ""),
                "summary": article.get("summary", ""),
                "tickers": article.get("symbols", []),
                "created_at": article.get("created_at", ""),
                "updated_at": article.get("updated_at", ""),
                "url": article.get("url", ""),
                "source": article.get("source", "alpaca"),
                "author": article.get("author", ""),
                "importance": self._score_importance(article),
            }
        except Exception as e:
            log.warning(f"Failed to parse Alpaca article: {e}")
            return None
    
    def _score_importance(self, article: Dict) -> str:
        """Score article importance."""
        headline = article.get("headline", "").lower()
        
        high_keywords = [
            "earnings", "fda", "sec", "merger", "acquisition",
            "upgrade", "downgrade", "halt", "guidance", "breaking"
        ]
        
        for kw in high_keywords:
            if kw in headline:
                return "high"
        
        medium_keywords = ["analyst", "price target", "revenue", "profit"]
        
        for kw in medium_keywords:
            if kw in headline:
                return "medium"
        
        return "low"
    
    def get_ticker_news(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get news for a specific ticker."""
        return self.get_news(symbols=[ticker], limit=limit)


# Global instance
alpaca_news = AlpacaNewsAdapter()
