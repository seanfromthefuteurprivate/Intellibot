import logging
import alpaca_trade_api as tradeapi
from wsb_snake.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

logger = logging.getLogger(__name__)

def get_market_data(tickers):
    """
    Fetches market data from Alpaca for the given tickers.
    Returns a dictionary of ticker -> data.
    """
    if not tickers:
        return {}
        
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.warning("Alpaca credentials not set. Returning empty data.")
        return {}

    logger.info(f"Fetching market data for: {tickers}")
    
    try:
        api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
        
        data = {}
        # Alpaca's get_latest_trade or get_snapshot might be better for real-time
        # For simplicity, we'll try to get a snapshot or latest bar
        
        for ticker in tickers:
            try:
                # Fetching snapshot for more complete data
                snapshot = api.get_snapshot(ticker)
                
                price = snapshot.latest_trade.price
                prev_close = snapshot.prev_daily_bar.close
                
                # Calculate percent change
                if prev_close and prev_close > 0:
                    change = (price - prev_close) / prev_close
                else:
                    change = 0.0

                data[ticker] = {
                    "price": float(price),
                    "volume": snapshot.daily_bar.volume if snapshot.daily_bar else 0,
                    "change": float(change)
                }
            except Exception as e:
                logger.warning(f"Could not fetch data for {ticker}: {e}")
                
        return data

    except Exception as e:
        logger.error(f"Alpaca API error: {e}")
        return {}
