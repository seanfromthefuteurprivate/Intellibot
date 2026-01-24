"""
Reddit WSB Collector - Scrapes r/wallstreetbets for real-time sentiment and mentions
Uses public Reddit JSON endpoints (no API key required)
"""
import re
import time
import requests
from datetime import datetime, timedelta
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from wsb_snake.utils.logger import get_logger
from wsb_snake.config import ZERO_DTE_UNIVERSE

logger = get_logger(__name__)

COMMON_WORDS = {
    'THE', 'AND', 'FOR', 'THIS', 'THAT', 'WITH', 'FROM', 'HAVE', 'HAS',
    'WAS', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
    'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY',
    'NEW', 'NOW', 'OLD', 'SEE', 'WAY', 'WHO', 'BOY', 'DID', 'OWN', 'SAY',
    'SHE', 'TOO', 'USE', 'WSB', 'YOLO', 'DD', 'POV', 'CEO', 'IPO', 'WTF',
    'IRS', 'USA', 'GDP', 'FED', 'SEC', 'ATH', 'ATL', 'EOD', 'EOW', 'ITM',
    'OTM', 'ATM', 'DTE', 'IV', 'RSI', 'EPS', 'PE', 'PB', 'PS', 'ETF',
    'IMO', 'IMHO', 'TBH', 'FOMO', 'FUD', 'LOL', 'LMAO', 'RIP', 'GG',
    'TLDR', 'EDIT', 'PSA', 'FYI', 'AKA', 'MOASS', 'HODL', 'APES', 'DRS',
    'GME', 'AMC', 'BBBY', 'BB'
}

VALID_TICKERS = set(ZERO_DTE_UNIVERSE) | {
    'SPY', 'QQQ', 'IWM', 'TSLA', 'NVDA', 'AAPL', 'META', 'AMD', 'AMZN',
    'GOOGL', 'GOOG', 'MSFT', 'NFLX', 'DIS', 'BA', 'JPM', 'GS', 'MS',
    'V', 'MA', 'PYPL', 'SQ', 'COIN', 'ROKU', 'SNAP', 'UBER', 'LYFT',
    'PLTR', 'SOFI', 'HOOD', 'RIVN', 'LCID', 'NIO', 'BABA', 'JD', 'PDD',
    'TSM', 'INTC', 'MU', 'QCOM', 'AVGO', 'CRM', 'ORCL', 'ADBE', 'NOW',
    'SNOW', 'NET', 'DDOG', 'ZS', 'CRWD', 'PANW', 'OKTA', 'TWLO', 'U',
    'RBLX', 'ABNB', 'DASH', 'DKNG', 'PENN', 'MGM', 'WYNN', 'LVS',
    'XOM', 'CVX', 'OXY', 'DVN', 'FANG', 'SLB', 'HAL', 'BKR',
    'COST', 'WMT', 'TGT', 'HD', 'LOW', 'NKE', 'LULU', 'GPS', 'ANF',
    'F', 'GM', 'TM', 'HMC', 'STLA', 'LI', 'XPEV',
    'PFE', 'MRNA', 'JNJ', 'UNH', 'CVS', 'WBA', 'LLY', 'ABBV', 'MRK',
    'VIX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXS', 'SPXL', 'TZA', 'TNA'
}

BULLISH_WORDS = {
    'moon', 'rocket', 'calls', 'bull', 'bullish', 'long', 'buy', 'buying',
    'squeeze', 'breakout', 'rip', 'pump', 'green', 'tendies', 'gains',
    'up', 'higher', 'rally', 'surge', 'soar', 'explode', 'lambo', 'yolo',
    'send', 'printing', 'free money', 'to the moon', 'lfg', 'lets go'
}

BEARISH_WORDS = {
    'puts', 'bear', 'bearish', 'short', 'sell', 'selling', 'dump', 'crash',
    'tank', 'drill', 'red', 'loss', 'losses', 'down', 'lower', 'drop',
    'fall', 'collapse', 'rekt', 'rug', 'rugged', 'bag', 'bagholder',
    'blood', 'drilling', 'capitulate', 'fear', 'panic'
}


class RedditWSBCollector:
    """
    Real-time Reddit WSB scraper using public JSON endpoints.
    No API key required - uses browser-like requests.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
        })
        self.base_url = "https://old.reddit.com"
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 120
        self.last_fetch = 0
        
    def _fetch_json(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Fetch JSON from Reddit with rate limiting."""
        try:
            now = time.time()
            if now - self.last_fetch < 2:
                time.sleep(2 - (now - self.last_fetch))
            
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=15)
            self.last_fetch = time.time()
            
            if response.status_code == 429:
                logger.warning("Reddit rate limited, waiting 30s...")
                time.sleep(30)
                return None
            
            if response.status_code != 200:
                logger.warning(f"Reddit returned {response.status_code} for {endpoint}")
                return None
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Reddit fetch error: {e}")
            return None
    
    def get_hot_posts(self, limit: int = 50) -> List[Dict]:
        """Get hot posts from r/wallstreetbets."""
        cache_key = f"hot_{limit}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['time'] < self.cache_ttl:
                return cached['data']
        
        data = self._fetch_json("/r/wallstreetbets/hot.json", {"limit": limit})
        if not data:
            return []
        
        posts = self._parse_posts(data)
        self.cache[cache_key] = {'time': time.time(), 'data': posts}
        
        logger.info(f"Fetched {len(posts)} hot posts from WSB")
        return posts
    
    def get_new_posts(self, limit: int = 25) -> List[Dict]:
        """Get newest posts from r/wallstreetbets."""
        cache_key = f"new_{limit}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['time'] < self.cache_ttl:
                return cached['data']
        
        data = self._fetch_json("/r/wallstreetbets/new.json", {"limit": limit})
        if not data:
            return []
        
        posts = self._parse_posts(data)
        self.cache[cache_key] = {'time': time.time(), 'data': posts}
        
        logger.info(f"Fetched {len(posts)} new posts from WSB")
        return posts
    
    def get_rising_posts(self, limit: int = 25) -> List[Dict]:
        """Get rising posts from r/wallstreetbets."""
        data = self._fetch_json("/r/wallstreetbets/rising.json", {"limit": limit})
        if not data:
            return []
        
        posts = self._parse_posts(data)
        logger.info(f"Fetched {len(posts)} rising posts from WSB")
        return posts
    
    def _parse_posts(self, data: Dict) -> List[Dict]:
        """Parse Reddit API response into structured posts."""
        posts = []
        children = data.get("data", {}).get("children", [])
        
        for child in children:
            post_data = child.get("data", {})
            
            post = {
                "id": post_data.get("id"),
                "title": post_data.get("title", ""),
                "selftext": post_data.get("selftext", ""),
                "author": post_data.get("author", ""),
                "score": post_data.get("score", 0),
                "upvote_ratio": post_data.get("upvote_ratio", 0),
                "num_comments": post_data.get("num_comments", 0),
                "created_utc": post_data.get("created_utc", 0),
                "flair": post_data.get("link_flair_text", ""),
                "url": f"https://reddit.com{post_data.get('permalink', '')}",
                "is_dd": post_data.get("link_flair_text", "").upper() in ["DD", "DUE DILIGENCE"],
                "is_yolo": post_data.get("link_flair_text", "").upper() == "YOLO",
                "is_gain": "gain" in post_data.get("link_flair_text", "").lower(),
                "is_loss": "loss" in post_data.get("link_flair_text", "").lower(),
            }
            
            full_text = f"{post['title']} {post['selftext']}"
            post["tickers"] = self._extract_tickers(full_text)
            post["sentiment"] = self._analyze_sentiment(full_text)
            
            posts.append(post)
        
        return posts
    
    def _extract_tickers(self, text: str) -> List[str]:
        """Extract valid stock tickers from text."""
        patterns = [
            r'\$([A-Z]{1,5})\b',
            r'\b([A-Z]{2,5})\b'
        ]
        
        tickers = set()
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                ticker = match.upper()
                if ticker in VALID_TICKERS and ticker not in COMMON_WORDS:
                    tickers.add(ticker)
        
        return list(tickers)
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        text_lower = text.lower()
        words = set(text_lower.split())
        
        bullish_count = len(words & BULLISH_WORDS)
        bearish_count = len(words & BEARISH_WORDS)
        
        total = bullish_count + bearish_count
        if total == 0:
            sentiment_score = 0.5
            direction = "neutral"
        else:
            sentiment_score = bullish_count / total
            if sentiment_score > 0.6:
                direction = "bullish"
            elif sentiment_score < 0.4:
                direction = "bearish"
            else:
                direction = "neutral"
        
        return {
            "score": sentiment_score,
            "direction": direction,
            "bullish_signals": bullish_count,
            "bearish_signals": bearish_count
        }
    
    def get_ticker_mentions(self) -> Dict[str, Dict]:
        """Get aggregated ticker mentions with sentiment."""
        hot_posts = self.get_hot_posts(50)
        new_posts = self.get_new_posts(25)
        
        all_posts = hot_posts + new_posts
        
        ticker_data: Dict[str, Dict] = {}
        
        for post in all_posts:
            for ticker in post.get("tickers", []):
                if ticker not in ticker_data:
                    ticker_data[ticker] = {
                        "ticker": ticker,
                        "mentions": 0,
                        "total_score": 0,
                        "total_comments": 0,
                        "bullish_count": 0,
                        "bearish_count": 0,
                        "neutral_count": 0,
                        "dd_posts": 0,
                        "yolo_posts": 0,
                        "posts": []
                    }
                
                ticker_data[ticker]["mentions"] += 1
                ticker_data[ticker]["total_score"] += post.get("score", 0)
                ticker_data[ticker]["total_comments"] += post.get("num_comments", 0)
                
                sentiment = post.get("sentiment", {})
                direction = sentiment.get("direction", "neutral")
                if direction == "bullish":
                    ticker_data[ticker]["bullish_count"] += 1
                elif direction == "bearish":
                    ticker_data[ticker]["bearish_count"] += 1
                else:
                    ticker_data[ticker]["neutral_count"] += 1
                
                if post.get("is_dd"):
                    ticker_data[ticker]["dd_posts"] += 1
                if post.get("is_yolo"):
                    ticker_data[ticker]["yolo_posts"] += 1
                
                ticker_data[ticker]["posts"].append({
                    "title": post.get("title", "")[:100],
                    "score": post.get("score", 0),
                    "sentiment": direction,
                    "flair": post.get("flair", "")
                })
        
        for ticker in ticker_data:
            data = ticker_data[ticker]
            total = data["bullish_count"] + data["bearish_count"] + data["neutral_count"]
            if total > 0:
                data["bullish_ratio"] = data["bullish_count"] / total
                data["bearish_ratio"] = data["bearish_count"] / total
            else:
                data["bullish_ratio"] = 0.5
                data["bearish_ratio"] = 0.5
            
            if data["bullish_ratio"] > 0.6:
                data["overall_sentiment"] = "bullish"
            elif data["bearish_ratio"] > 0.6:
                data["overall_sentiment"] = "bearish"
            else:
                data["overall_sentiment"] = "mixed"
            
            data["heat_score"] = (
                data["mentions"] * 10 +
                min(data["total_score"], 1000) / 10 +
                min(data["total_comments"], 500) / 5 +
                data["dd_posts"] * 50 +
                data["yolo_posts"] * 30
            )
        
        return ticker_data
    
    def get_trending_tickers(self, min_mentions: int = 2) -> List[Dict]:
        """Get trending tickers sorted by heat score."""
        ticker_data = self.get_ticker_mentions()
        
        trending = [
            data for data in ticker_data.values()
            if data["mentions"] >= min_mentions
        ]
        
        trending.sort(key=lambda x: x["heat_score"], reverse=True)
        
        return trending[:10]
    
    def get_ticker_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Get sentiment data for a specific ticker."""
        ticker_data = self.get_ticker_mentions()
        
        if ticker in ticker_data:
            return ticker_data[ticker]
        
        return {
            "ticker": ticker,
            "mentions": 0,
            "overall_sentiment": "no_data",
            "heat_score": 0,
            "bullish_ratio": 0.5,
            "bearish_ratio": 0.5
        }
    
    def scan_for_catalysts(self) -> List[Dict]:
        """Scan for potential catalyst posts (DD, breaking news)."""
        hot_posts = self.get_hot_posts(50)
        new_posts = self.get_new_posts(25)
        
        catalysts = []
        
        for post in hot_posts + new_posts:
            is_catalyst = False
            catalyst_type = None
            
            if post.get("is_dd"):
                is_catalyst = True
                catalyst_type = "DD"
            elif post.get("score", 0) > 1000:
                is_catalyst = True
                catalyst_type = "viral"
            elif any(word in post.get("title", "").lower() for word in ["breaking", "just announced", "earnings", "fda", "merger", "buyout"]):
                is_catalyst = True
                catalyst_type = "breaking_news"
            
            if is_catalyst and post.get("tickers"):
                catalysts.append({
                    "type": catalyst_type,
                    "tickers": post["tickers"],
                    "title": post["title"],
                    "score": post["score"],
                    "comments": post["num_comments"],
                    "sentiment": post["sentiment"],
                    "url": post["url"],
                    "created_utc": post["created_utc"]
                })
        
        return catalysts


reddit_collector = RedditWSBCollector()


def collect_mentions() -> List[str]:
    """Legacy function - get top mentioned tickers."""
    trending = reddit_collector.get_trending_tickers()
    return [t["ticker"] for t in trending[:5]] if trending else ["SPY", "QQQ", "NVDA"]


def get_wsb_sentiment(ticker: str) -> Dict[str, Any]:
    """Get WSB sentiment for a specific ticker."""
    return reddit_collector.get_ticker_sentiment(ticker)


def get_wsb_catalysts() -> List[Dict]:
    """Get potential catalyst posts from WSB."""
    return reddit_collector.scan_for_catalysts()


def get_wsb_trending() -> List[Dict]:
    """Get trending tickers from WSB."""
    return reddit_collector.get_trending_tickers()
