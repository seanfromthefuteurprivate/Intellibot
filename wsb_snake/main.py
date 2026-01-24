import time
import schedule
from datetime import datetime

from wsb_snake.config import *
from wsb_snake.utils.logger import log
from wsb_snake.utils.rate_limit import limiter

from wsb_snake.collectors.reddit_collector import collect_mentions
from wsb_snake.collectors.market_data import get_market_data

from wsb_snake.parsing.ticker_extractor import extract_tickers
from wsb_snake.parsing.dedupe import deduplicator

from wsb_snake.analysis.scoring import score_tickers
from wsb_snake.analysis.sentiment import summarize_setup

from wsb_snake.signals.signal_router import route_signals, should_alert_immediately
from wsb_snake.signals.signal_store import save_signals_batch

from wsb_snake.notifications.telegram_bot import send_alert
from wsb_snake.notifications.message_templates import (
    format_startup_message,
    format_alert_message,
    format_digest_message,
)


def send_startup_ping():
    """Send the 'Snake Online' startup message to Telegram."""
    log.info("Sending startup ping...")
    message = format_startup_message()
    send_alert(message)
    log.info("Startup ping sent.")


def run_signal_pipeline():
    """
    Main signal pipeline:
    1. Collect Reddit mentions
    2. Extract and dedupe tickers
    3. Fetch market data
    4. Score and classify signals
    5. Route to alerts/watchlist/log
    6. Send alerts
    7. Store signals
    """
    log.info("=" * 50)
    log.info("Starting signal pipeline...")
    
    # 1. Collect mentions from Reddit
    raw_data = collect_mentions()
    if not raw_data:
        log.info("No data collected. Skipping cycle.")
        return
    
    # raw_data is a list of tickers from the simplified collector
    tickers = raw_data if isinstance(raw_data, list) else []
    
    # 2. Filter already-alerted tickers
    fresh_tickers = deduplicator.filter_tickers(tickers)
    log.info(f"Fresh tickers after dedupe: {fresh_tickers}")
    
    if not fresh_tickers:
        log.info("All tickers recently alerted. Skipping.")
        return
    
    # 3. Fetch market data
    market_data = get_market_data(fresh_tickers)
    
    # 4. Build social metrics (simplified for now)
    # In a full implementation, this would come from time_windows tracker
    social_data = {}
    for ticker in fresh_tickers:
        social_data[ticker] = {
            'count': 1,
            'velocity': 0.5,  # Placeholder
            'authors': 3,
            'sentiment': 0.3,
            'intents': [],
        }
    
    # 5. Score and classify
    signals = score_tickers(fresh_tickers, market_data, social_data)
    
    # 6. Route signals
    alerts, watchlist, logged = route_signals(signals)
    
    # 7. Send immediate alerts
    for signal in alerts[:3]:  # Max 3 alerts per cycle
        # Generate AI summary
        limiter.wait_if_needed('openai')
        signal.summary = summarize_setup(signal.ticker)
        
        # Format and send
        message = format_alert_message(signal)
        limiter.wait_if_needed('telegram')
        send_alert(message)
        
        # Mark as alerted
        deduplicator.mark_alerted(signal.ticker)
        
        log.info(f"Alert sent for {signal.ticker} (Score: {signal.score:.1f})")
    
    # 8. Store all signals
    if signals:
        save_signals_batch(signals)
    
    # 9. Log summary
    log.info(f"Pipeline complete. Alerts: {len(alerts)}, Watchlist: {len(watchlist)}, Logged: {len(logged)}")
    log.info("=" * 50)


def run_digest():
    """
    Send periodic digest of watchlist items.
    Called less frequently than the main pipeline.
    """
    log.info("Generating digest...")
    # For now, just send a placeholder digest
    # In full implementation, this would aggregate B-tier signals
    # send_alert(format_digest_message([]))
    log.info("Digest sent (placeholder).")


def main():
    log.info("=" * 50)
    log.info("WSB SNAKE BACKEND STARTING")
    log.info("=" * 50)
    
    # Send startup ping
    send_startup_ping()
    
    # Schedule jobs
    schedule.every(10).minutes.do(run_signal_pipeline)
    schedule.every(4).hours.do(run_digest)
    
    # Run immediately on startup
    log.info("Running initial pipeline...")
    run_signal_pipeline()
    
    # Main loop
    log.info("Entering main loop. Monitoring active.")
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
