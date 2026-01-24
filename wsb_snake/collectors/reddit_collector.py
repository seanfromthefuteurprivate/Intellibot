import logging
import requests
from collections import Counter

logger = logging.getLogger(__name__)

def collect_mentions():
    """
    Scrapes r/wallstreetbets for ticker mentions using the public JSON API.
    Returns a list of popular tickers.
    """
    logger.info("Collecting mentions from Reddit (JSON API)...")
    
    # Use a standard browser User-Agent to avoid 403 on public JSON
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    url = "https://www.reddit.com/r/wallstreetbets/hot.json?limit=25"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"Reddit API returned {response.status_code}")
            return ["SPY", "QQQ", "NVDA"] # Fallback
            
        data = response.json()
        posts = data.get("data", {}).get("children", [])
        
        text_blob = ""
        for post in posts:
            post_data = post.get("data", {})
            title = post_data.get("title", "")
            selftext = post_data.get("selftext", "")
            text_blob += f"{title} {selftext} "
            
        # Very naive ticker extraction (uppercase words, 2-5 chars)
        # In production, use a dictionary of known tickers
        words = text_blob.split()
        potential_tickers = [w.strip("$.,!") for w in words if w.isupper() and 2 <= len(w) <= 5 and w.isalpha()]
        
        # Filter out common words (THIS IS CRITICAL for naive extraction)
        COMMON_WORDS = {'THE', 'AND', 'FOR', 'THIS', 'THAT', 'WITH', 'WSB', 'YOLO', 'DD', 'POV', 'CEO', 'IPO', 'WTF', 'IRS', 'USA', 'GDP'}
        filtered_tickers = [t for t in potential_tickers if t not in COMMON_WORDS]
        
        counts = Counter(filtered_tickers)
        top_tickers = [t for t, count in counts.most_common(5)]
        
        logger.info(f"Top mentions: {top_tickers}")
        return top_tickers if top_tickers else ["SPY"]

    except Exception as e:
        logger.error(f"Error collecting from Reddit: {e}")
        return ["SPY", "NVDA"] # Fallback
