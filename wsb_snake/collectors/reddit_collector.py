import logging

logger = logging.getLogger(__name__)

def collect_mentions():
    """
    Scrapes r/wallstreetbets for ticker mentions.
    Returns a list of tickers (str).
    """
    logger.info("Collecting mentions from Reddit...")
    # Placeholder logic
    # TODO: Implement PRAW integration
    
    # For testing, return dummy data
    return ["AAPL", "TSLA", "NVDA", "SPY"]
