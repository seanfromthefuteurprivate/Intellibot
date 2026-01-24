import logging

logger = logging.getLogger(__name__)

def get_market_data(tickers):
    """
    Fetches market data from Alpaca for the given tickers.
    Returns a dictionary of ticker -> data.
    """
    logger.info(f"Fetching market data for: {tickers}")
    # Placeholder logic
    # TODO: Implement Alpaca integration
    
    data = {}
    for ticker in tickers:
        data[ticker] = {
            "price": 100.0,
            "volume": 1000000,
            "change": 0.05
        }
    return data
