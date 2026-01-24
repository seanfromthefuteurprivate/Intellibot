"""
WSB Snake - 0DTE Intelligence Engine

Main entry point for the 6-engine trading intelligence pipeline.
Runs continuously, scanning for signals and sending Telegram alerts.
"""

import time
import schedule
from datetime import datetime

from wsb_snake.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from wsb_snake.utils.logger import log

from wsb_snake.notifications.telegram_bot import send_alert
from wsb_snake.notifications.message_templates import format_startup_message

from wsb_snake.utils.session_regime import (
    get_session_info, is_market_open, should_scan_for_signals
)
from wsb_snake.db.database import init_database

from wsb_snake.engines.orchestrator import run_pipeline, send_daily_summary
from wsb_snake.engines.learning_memory import learning_memory
from wsb_snake.engines.chart_brain import get_chart_brain


def send_startup_ping():
    """Send the 'Snake Online' startup message to Telegram."""
    log.info("Sending startup ping...")
    
    session_info = get_session_info()
    
    message = f"""🐍 **WSB SNAKE v2.1 ONLINE**

**0DTE Intelligence Engine Activated**

🔧 **Engines Loaded:**
• Engine 1: Ignition Detector
• Engine 2: 0DTE Pressure Engine  
• Engine 3: Late-Day Surge Hunter
• Engine 4: Probability Generator
• Engine 5: Self-Learning Memory
• Engine 6: Paper Shadow Trader
• 🧠 ChartBrain: LangGraph AI (GPT-4o Vision)

📊 **Session Status:**
• Session: {session_info['session'].upper()}
• Market Open: {'Yes' if session_info['is_open'] else 'No'}
• Power Hour: {'Yes' if session_info['is_power_hour'] else 'No'}
• Time ET: {session_info['current_time_et']}

🎯 **Monitoring:** SPY, QQQ, IWM + TSLA, NVDA, AAPL, META, AMD, AMZN, GOOGL, MSFT

⚡ Pipeline + AI chart analysis running continuously.
"""
    
    send_alert(message)
    log.info("Startup ping sent.")


def run_scheduled_pipeline():
    """
    Run the pipeline on schedule.
    Only runs during active market hours when signals are valuable.
    """
    if should_scan_for_signals():
        log.info("Running scheduled pipeline...")
        results = run_pipeline()
        
        # Log summary
        log.info(
            f"Pipeline results: {len(results.get('probabilities', []))} signals, "
            f"{results.get('alerts_sent', 0)} alerts, "
            f"{results.get('paper_trades', 0)} paper trades"
        )
    else:
        session_info = get_session_info()
        log.info(f"Skipping pipeline - session: {session_info['session']}")


def run_daily_report():
    """Send end-of-day report at market close."""
    log.info("Sending daily report...")
    send_daily_summary()
    
    # Apply weight decay for next day
    learning_memory.apply_daily_decay()


def main():
    log.info("=" * 50)
    log.info("WSB SNAKE 0DTE ENGINE STARTING")
    log.info("=" * 50)
    
    # Initialize database
    log.info("Initializing database...")
    init_database()
    
    # Start ChartBrain background AI analysis
    log.info("Starting ChartBrain AI background analysis...")
    chart_brain = get_chart_brain()
    chart_brain.start()
    log.info("🧠 ChartBrain active - studying charts in real-time")
    
    # Send startup ping
    send_startup_ping()
    
    # Schedule jobs
    log.info("Setting up scheduler...")
    
    # Main pipeline: every 10 minutes during market hours
    schedule.every(10).minutes.do(run_scheduled_pipeline)
    
    # Daily report: 4:15 PM ET (after market close)
    schedule.every().day.at("16:15").do(run_daily_report)
    
    # Run immediately on startup if market-appropriate
    log.info("Running initial pipeline check...")
    session_info = get_session_info()
    
    if should_scan_for_signals():
        log.info("Market active - running full pipeline...")
        run_pipeline()
    else:
        log.info(f"Market session: {session_info['session']} - pipeline on standby")
        # Still run once to initialize
        results = run_pipeline()
        log.info(f"Initial scan complete: {len(results.get('probabilities', []))} signals")
    
    # Main loop
    log.info("=" * 50)
    log.info("Entering main loop. 0DTE monitoring active.")
    log.info("=" * 50)
    
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
