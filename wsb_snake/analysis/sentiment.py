import logging
import os
from openai import OpenAI
from wsb_snake.config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

# Startup validation - log warning if API key missing
_SENTIMENT_ENABLED = bool(OPENAI_API_KEY)
if not _SENTIMENT_ENABLED:
    logger.warning("⚠️ OPENAI_API_KEY not set - sentiment analysis disabled (will return placeholders)")


def is_enabled() -> bool:
    """Check if sentiment analysis is available."""
    return _SENTIMENT_ENABLED


def summarize_setup(ticker):
    """
    Uses OpenAI to summarize the setup for a ticker based on recent posts.
    """
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not set. Returning placeholder.")
        return f"Setup for {ticker} (OpenAI key missing)"

    logger.info(f"Summarizing setup for {ticker}...")
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # In a real app, you would pass the actual Reddit post content here.
        # Since we are just scraping titles/tickers, we'll ask it to generate a generic summary
        # or use the context we have (which is limited in this simplified version).
        
        prompt = f"Provide a brief, 2-sentence financial sentiment summary for the stock ticker {ticker} as if you were a WallStreetBets trader. Be risky and confident."
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Using 3.5 for cost/speed in this demo
            messages=[
                {"role": "system", "content": "You are a financial analyst on r/wallstreetbets."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"Could not summarize {ticker}."
