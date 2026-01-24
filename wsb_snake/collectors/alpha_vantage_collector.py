"""
Alpha Vantage Collector

AI-powered news sentiment and market intelligence.
Free tier: 500 API calls/day.

Key features:
- News sentiment scores (AI-powered)
- Market-moving news detection
- Ticker-specific sentiment
"""

import os
import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from wsb_snake.utils.logger import log


class AlphaVantageCollector:
    """
    Collects AI-powered news sentiment from Alpha Vantage.
    
    Features:
    - News sentiment analysis
    - Topic-based sentiment
    - Ticker sentiment tracking
    """
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self):
        self.api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self.session = requests.Session()
        
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 600
        self.last_call = 0
        self.min_interval = 12.0
        
        if not self.api_key:
            log.warning("ALPHA_VANTAGE_API_KEY not set - sentiment limited")
        else:
            log.info("Alpha Vantage collector initialized")
    
    def _rate_limit(self):
        """Respect rate limits (5 calls/min on free tier)"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    def get_news_sentiment(self, tickers: str = None, topics: str = None, limit: int = 50) -> List[Dict]:
        """
        Get AI-powered news sentiment.
        
        Args:
            tickers: Comma-separated tickers (e.g., "AAPL,TSLA")
            topics: Topics to filter (e.g., "earnings", "ipo", "mergers")
            limit: Number of articles
            
        Returns:
            List of news articles with sentiment scores
        """
        cache_key = f"sentiment:{tickers}:{topics}:{limit}"
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["data"]
        
        if not self.api_key:
            return []
        
        self._rate_limit()
        
        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": self.api_key,
                "limit": limit,
            }
            
            if tickers:
                params["tickers"] = tickers
            if topics:
                params["topics"] = topics
            
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if "Note" in data:
                    log.warning("Alpha Vantage rate limit reached")
                    return []
                
                feed = data.get("feed", [])
                
                result = []
                for article in feed:
                    ticker_sentiments = {}
                    for ts in article.get("ticker_sentiment", []):
                        ticker_sentiments[ts.get("ticker", "")] = {
                            "relevance": float(ts.get("relevance_score", 0)),
                            "sentiment_score": float(ts.get("ticker_sentiment_score", 0)),
                            "sentiment_label": ts.get("ticker_sentiment_label", ""),
                        }
                    
                    result.append({
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "time_published": article.get("time_published", ""),
                        "source": article.get("source", ""),
                        "overall_sentiment_score": float(article.get("overall_sentiment_score", 0)),
                        "overall_sentiment_label": article.get("overall_sentiment_label", ""),
                        "ticker_sentiments": ticker_sentiments,
                        "topics": [t.get("topic") for t in article.get("topics", [])],
                    })
                
                self.cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": result
                }
                
                return result
            else:
                log.debug(f"Alpha Vantage returned {response.status_code}")
                return []
                
        except Exception as e:
            log.debug(f"Alpha Vantage error: {e}")
            return []
    
    def get_ticker_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get aggregated sentiment for a specific ticker.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Dict with sentiment summary
        """
        news = self.get_news_sentiment(tickers=symbol, limit=20)
        
        if not news:
            return {
                "symbol": symbol,
                "has_data": False,
                "sentiment_score": 0,
                "sentiment_label": "neutral",
            }
        
        scores = []
        for article in news:
            ticker_data = article.get("ticker_sentiments", {}).get(symbol)
            if ticker_data:
                scores.append(ticker_data.get("sentiment_score", 0))
        
        if not scores:
            for article in news:
                scores.append(article.get("overall_sentiment_score", 0))
        
        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 0
        
        if avg_score > 0.25:
            label = "bullish"
            boost = 5
        elif avg_score > 0.1:
            label = "somewhat_bullish"
            boost = 3
        elif avg_score < -0.25:
            label = "bearish"
            boost = -5
        elif avg_score < -0.1:
            label = "somewhat_bearish"
            boost = -3
        else:
            label = "neutral"
            boost = 0
        
        return {
            "symbol": symbol,
            "has_data": True,
            "sentiment_score": round(avg_score, 3),
            "sentiment_label": label,
            "score_boost": boost,
            "article_count": len(news),
            "scores_analyzed": len(scores),
        }
    
    def get_market_sentiment(self) -> Dict[str, Any]:
        """
        Get overall market sentiment from financial news.
        
        Returns:
            Dict with market-wide sentiment
        """
        news = self.get_news_sentiment(topics="financial_markets", limit=30)
        
        if not news:
            return {
                "has_data": False,
                "market_sentiment": "neutral",
            }
        
        scores = [a.get("overall_sentiment_score", 0) for a in news]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        bullish = sum(1 for s in scores if s > 0.1)
        bearish = sum(1 for s in scores if s < -0.1)
        neutral = len(scores) - bullish - bearish
        
        if avg_score > 0.15:
            market_sentiment = "bullish"
        elif avg_score > 0.05:
            market_sentiment = "slightly_bullish"
        elif avg_score < -0.15:
            market_sentiment = "bearish"
        elif avg_score < -0.05:
            market_sentiment = "slightly_bearish"
        else:
            market_sentiment = "neutral"
        
        return {
            "has_data": True,
            "market_sentiment": market_sentiment,
            "sentiment_score": round(avg_score, 3),
            "bullish_articles": bullish,
            "bearish_articles": bearish,
            "neutral_articles": neutral,
            "total_articles": len(scores),
        }
    
    def get_breaking_news(self, symbols: List[str] = None) -> List[Dict]:
        """
        Get breaking/recent news that could move markets.
        
        Args:
            symbols: List of tickers to check
            
        Returns:
            List of recent high-impact news
        """
        if symbols is None:
            from wsb_snake.config import ZERO_DTE_UNIVERSE
            symbols = ZERO_DTE_UNIVERSE[:5]
        
        tickers_str = ",".join(symbols)
        news = self.get_news_sentiment(tickers=tickers_str, limit=30)
        
        high_impact = []
        for article in news:
            score = abs(article.get("overall_sentiment_score", 0))
            
            if score > 0.3:
                high_impact.append({
                    "title": article.get("title"),
                    "source": article.get("source"),
                    "sentiment_score": article.get("overall_sentiment_score"),
                    "sentiment_label": article.get("overall_sentiment_label"),
                    "tickers": list(article.get("ticker_sentiments", {}).keys()),
                    "time": article.get("time_published"),
                    "impact": "high" if score > 0.5 else "medium",
                })
        
        return high_impact


alpha_vantage = AlphaVantageCollector()
