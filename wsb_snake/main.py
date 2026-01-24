import time
import schedule
import logging
from wsb_snake.config import *
from wsb_snake.collectors.reddit_collector import collect_mentions
from wsb_snake.collectors.market_data import get_market_data
from wsb_snake.analysis.scoring import score_tickers
from wsb_snake.notifications.telegram_bot import send_alert
from wsb_snake.analysis.sentiment import summarize_setup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def job():
    logger.info("Starting scheduled job...")
    
    # 1. Collect mentions
    tickers = collect_mentions()
    if not tickers:
        logger.info("No tickers found.")
        return

    # 2. Get market data
    market_data = get_market_data(tickers)
    
    # 3. Score tickers
    ranked_tickers = score_tickers(tickers, market_data)
    
    # 4. Filter and alert
    if ranked_tickers:
        logger.info(f"Top tickers: {ranked_tickers[:3]}")
        for ticker, score in ranked_tickers[:3]: # Top 3
            summary = summarize_setup(ticker)
            send_alert(f"Ticker: {ticker}\nScore: {score}\n\nSummary:\n{summary}")
    else:
        logger.info("No tickers ranked.")
        
    logger.info("Job finished.")

def main():
    logger.info("WSB Snake Backend Started")
    
    # Schedule job every 15 minutes
    schedule.every(15).minutes.do(job)
    
    # Run once on startup
    job()
    
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
