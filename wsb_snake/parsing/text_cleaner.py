import re
from typing import Optional

def clean_reddit_text(text: str) -> str:
    """
    Clean and normalize Reddit post/comment text.
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove Reddit-specific formatting
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [text](url) -> text
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&nbsp;', ' ', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep $ for tickers
    text = re.sub(r'[^\w\s$.,!?\'-]', '', text)
    
    return text.strip()


def extract_sentiment_keywords(text: str) -> dict:
    """
    Extract WSB-specific sentiment keywords.
    
    Returns:
        Dict with bullish_count, bearish_count, and keyword lists
    """
    text_lower = text.lower()
    
    bullish_keywords = [
        'moon', 'rocket', 'calls', 'bull', 'buy', 'long', 'yolo', 'diamond hands',
        'tendies', 'gain', 'green', 'pump', 'squeeze', 'breakout', 'rip', 'lambo',
        'print', 'printing', 'all in', 'to the moon', 'ath', 'new high', 'bullish',
        'undervalued', 'cheap', 'discount', 'dip', 'btfd', 'buy the dip',
    ]
    
    bearish_keywords = [
        'puts', 'bear', 'short', 'sell', 'dump', 'crash', 'red', 'loss', 'bag',
        'bagholder', 'rug', 'rugpull', 'scam', 'overvalued', 'bubble', 'top',
        'bearish', 'drill', 'drilling', 'tanking', 'plunge', 'collapse',
    ]
    
    bullish_found = [kw for kw in bullish_keywords if kw in text_lower]
    bearish_found = [kw for kw in bearish_keywords if kw in text_lower]
    
    return {
        'bullish_count': len(bullish_found),
        'bearish_count': len(bearish_found),
        'bullish_keywords': bullish_found,
        'bearish_keywords': bearish_found,
        'net_sentiment': len(bullish_found) - len(bearish_found),
    }


def detect_intent_tags(text: str) -> list:
    """
    Detect trading intent tags from text.
    
    Returns:
        List of detected intents (e.g., ['0DTE', 'EARNINGS', 'SQUEEZE'])
    """
    text_lower = text.lower()
    
    intent_patterns = {
        '0DTE': ['0dte', '0 dte', 'zero dte', 'same day', 'expiring today'],
        'EARNINGS': ['earnings', 'er play', 'earnings play', 'quarterly report'],
        'SQUEEZE': ['squeeze', 'short squeeze', 'gamma squeeze', 'gamma ramp'],
        'CATALYST': ['catalyst', 'fda', 'approval', 'merger', 'acquisition'],
        'MOMENTUM': ['breakout', 'breaking out', 'ripping', 'running', 'parabolic'],
        'OPTIONS': ['calls', 'puts', 'options', 'strike', 'expiry', 'premium'],
        'DD': ['dd', 'due diligence', 'research', 'thesis', 'analysis'],
    }
    
    detected = []
    for intent, patterns in intent_patterns.items():
        if any(p in text_lower for p in patterns):
            detected.append(intent)
    
    return detected
