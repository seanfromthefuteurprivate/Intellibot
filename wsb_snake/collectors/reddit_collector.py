"""
Reddit WSB Collector - Scrapes r/wallstreetbets for real-time sentiment and mentions

Supports two modes:
1. OAuth Authentication (recommended) - set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET
2. Public JSON endpoints (fallback) - may get 403 blocked

To get Reddit OAuth credentials:
1. Go to https://www.reddit.com/prefs/apps
2. Click "create another app..." at the bottom
3. Select "script" type
4. Fill in name and redirect URI (http://localhost:8080)
5. Copy client_id (under app name) and client_secret

Set environment variables:
  REDDIT_CLIENT_ID=your_client_id
  REDDIT_CLIENT_SECRET=your_client_secret
"""
import os
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

# Master kill switch - set REDDIT_ENABLED=false to disable all Reddit API calls
REDDIT_ENABLED = os.environ.get("REDDIT_ENABLED", "false").lower() in ("true", "1", "yes")

# OAuth configuration
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_OAUTH_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_OAUTH_API = "https://oauth.reddit.com"

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
    Real-time Reddit WSB scraper with OAuth support.

    Supports two modes:
    1. OAuth Authentication (if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET set)
       - Uses oauth.reddit.com API
       - Much more reliable, higher rate limits
    2. Public JSON endpoints (fallback)
       - Uses www.reddit.com/.json
       - May get 403 blocked by Reddit

    === DATA PLUMBER FIX: Enhanced resilience for 403 errors ===
    - OAuth authentication (preferred)
    - Multiple user agent rotation (fallback)
    - Exponential backoff on failures
    - Graceful degradation with empty results
    """

    # Multiple user agents to rotate (mimics different browsers)
    USER_AGENTS = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
    ]

    # Base URLs to try (old.reddit often blocked, www works better)
    BASE_URLS = [
        "https://www.reddit.com",   # Try www first (more permissive)
        "https://old.reddit.com",   # Fallback to old
    ]

    def __init__(self):
        # Check kill switch first
        self._enabled = REDDIT_ENABLED
        if not self._enabled:
            logger.info("Reddit collector DISABLED (REDDIT_ENABLED=false)")
            self.cache: Dict[str, Any] = {}
            self.cache_ttl = 120
            self.last_fetch = 0
            return

        self.session = requests.Session()
        self._ua_index = 0
        self._base_url_index = 0
        self._consecutive_failures = 0
        self._last_403_time = 0
        self._backoff_until = 0

        # OAuth state
        self._oauth_enabled = bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0

        if self._oauth_enabled:
            logger.info("Reddit OAuth enabled - using authenticated API")
            self._refresh_oauth_token()
        else:
            logger.warning(
                "Reddit OAuth not configured (no REDDIT_CLIENT_ID/SECRET). "
                "Using public endpoints which may get 403 blocked. "
                "See reddit_collector.py header for setup instructions."
            )

        self._update_session_headers()

        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 120
        self.last_fetch = 0

    def _refresh_oauth_token(self) -> bool:
        """
        Get or refresh OAuth access token using client credentials flow.

        Reddit OAuth app-only flow:
        POST https://www.reddit.com/api/v1/access_token
        - Basic auth with client_id:client_secret
        - Body: grant_type=client_credentials

        Returns True if successful, False otherwise.
        """
        if not self._oauth_enabled:
            return False

        try:
            response = requests.post(
                REDDIT_OAUTH_URL,
                auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": "wsb_snake/1.0"},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                self._access_token = data.get("access_token")
                expires_in = data.get("expires_in", 3600)
                # Refresh 5 minutes before expiry
                self._token_expires_at = time.time() + expires_in - 300
                logger.info(f"Reddit OAuth token refreshed, expires in {expires_in}s")
                return True
            else:
                logger.error(f"Reddit OAuth failed: {response.status_code} - {response.text}")
                self._oauth_enabled = False
                return False

        except Exception as e:
            logger.error(f"Reddit OAuth error: {e}")
            self._oauth_enabled = False
            return False

    def _ensure_valid_token(self) -> bool:
        """Ensure we have a valid OAuth token, refreshing if needed."""
        if not self._oauth_enabled:
            return False

        if time.time() >= self._token_expires_at:
            return self._refresh_oauth_token()

        return bool(self._access_token)

    def _update_session_headers(self):
        """Update session with current user agent and headers."""
        ua = self.USER_AGENTS[self._ua_index % len(self.USER_AGENTS)]
        self.base_url = self.BASE_URLS[self._base_url_index % len(self.BASE_URLS)]

        self.session.headers.update({
            'User-Agent': ua,
            'Accept': 'application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'DNT': '1',
            'Cache-Control': 'max-age=0',
        })
        
    def _fetch_json(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Fetch JSON from Reddit with OAuth support and 403 resilience.

        === PRIORITY ===
        1. OAuth API (oauth.reddit.com) - if credentials configured
        2. Public endpoints (www.reddit.com/.json) - fallback

        === DATA PLUMBER FIX ===
        - OAuth authentication (preferred, much more reliable)
        - Handles 403 with UA rotation and backoff (fallback)
        - Graceful degradation instead of hard failure
        """
        # Kill switch - return None when disabled
        if not getattr(self, '_enabled', False):
            return None

        try:
            now = time.time()

            # === Check if we're in backoff period ===
            if now < self._backoff_until:
                logger.debug(f"Reddit in backoff, {self._backoff_until - now:.0f}s remaining")
                return None

            # Rate limiting between requests
            if now - self.last_fetch < 3:
                time.sleep(3 - (now - self.last_fetch))

            # === TRY OAUTH FIRST (if enabled) ===
            if self._oauth_enabled and self._ensure_valid_token():
                oauth_url = f"{REDDIT_OAUTH_API}{endpoint}"
                headers = {
                    "Authorization": f"Bearer {self._access_token}",
                    "User-Agent": "wsb_snake/1.0 (trading bot)",
                }
                response = requests.get(oauth_url, params=params, headers=headers, timeout=15)
                self.last_fetch = time.time()

                if response.status_code == 200:
                    self._consecutive_failures = 0
                    return response.json()
                elif response.status_code == 401:
                    # Token expired or invalid, refresh and retry once
                    logger.warning("Reddit OAuth 401 - refreshing token")
                    if self._refresh_oauth_token():
                        headers["Authorization"] = f"Bearer {self._access_token}"
                        response = requests.get(oauth_url, params=params, headers=headers, timeout=15)
                        if response.status_code == 200:
                            self._consecutive_failures = 0
                            return response.json()
                    # OAuth failed, fall through to public endpoint
                    logger.warning("Reddit OAuth failed, falling back to public endpoints")
                elif response.status_code == 429:
                    logger.warning("Reddit OAuth rate limited, waiting 60s...")
                    self._backoff_until = time.time() + 60
                    return None
                else:
                    logger.warning(f"Reddit OAuth returned {response.status_code}, falling back to public")

            # === FALLBACK: PUBLIC ENDPOINTS ===
            url = f"{self.base_url}{endpoint}"
            response = self.session.get(url, params=params, timeout=15)
            self.last_fetch = time.time()

            if response.status_code == 429:
                logger.warning("Reddit rate limited, waiting 60s...")
                self._backoff_until = time.time() + 60
                return None

            if response.status_code == 403:
                # === DATA PLUMBER: 403 error handling ===
                self._consecutive_failures += 1
                self._last_403_time = time.time()

                # Rotate user agent
                self._ua_index += 1

                # After 2 failures, try different base URL
                if self._consecutive_failures >= 2:
                    self._base_url_index += 1
                    logger.warning(f"Reddit 403 - switching to {self.BASE_URLS[self._base_url_index % len(self.BASE_URLS)]}")

                # Exponential backoff: 30s, 60s, 120s, 300s max
                backoff_time = min(30 * (2 ** (self._consecutive_failures - 1)), 300)
                self._backoff_until = time.time() + backoff_time
                logger.warning(f"Reddit 403 forbidden - backoff {backoff_time}s (failure #{self._consecutive_failures})")

                # Update headers for next attempt
                self._update_session_headers()
                return None

            if response.status_code != 200:
                logger.warning(f"Reddit returned {response.status_code} for {endpoint}")
                return None

            # Success - reset failure counter
            self._consecutive_failures = 0
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning("Reddit request timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.warning("Reddit connection error")
            self._consecutive_failures += 1
            return None
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
