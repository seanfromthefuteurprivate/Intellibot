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

from wsb_snake.engines.orchestrator import run_pipeline, send_daily_summary, run_startup_training
from wsb_snake.engines.learning_memory import learning_memory
from wsb_snake.engines.chart_brain import get_chart_brain
from wsb_snake.engines.spy_scalper import spy_scalper
from wsb_snake.learning.zero_greed_exit import zero_greed_exit
from wsb_snake.trading.alpaca_executor import alpaca_executor


def send_startup_ping():
    """Send the 'Snake Online' startup message to Telegram."""
    log.info("Sending startup ping...")
    
    session_info = get_session_info()
    
    account = alpaca_executor.get_account()
    buying_power = float(account.get("buying_power", 0))
    trading_mode = "🔴 LIVE" if alpaca_executor.LIVE_TRADING else "📝 PAPER"
    
    from wsb_snake.engines.spy_scalper import spy_scalper
    predator_status = "🔥 ACTIVE" if spy_scalper.PREDATOR_MODE else "OFF"
    
    message = f"""🐍 **WSB SNAKE v2.5 ONLINE**

🔥 **AGGRESSIVE MODE ACTIVE** 🔥

🦅 **Predator Configuration:**
• Min Confidence: {spy_scalper.MIN_CONFIDENCE_FOR_ALERT}%
• AI Stack: Gemini + DeepSeek + GPT
• Small Cap Filter: 75% + Candlestick
• Cooldown: {spy_scalper.trade_cooldown_minutes} min

💰 **AGGRESSIVE TRADING:**
• Mode: {trading_mode}
• Buying Power: ${buying_power:,.2f}
• Daily Exposure: ${alpaca_executor.MAX_DAILY_EXPOSURE:,}
• Per Trade: ${alpaca_executor.MAX_PER_TRADE:,}
• Max Concurrent: {alpaca_executor.MAX_CONCURRENT_POSITIONS}
• Target: +20% | Stop: -15%

📊 **Session:**
• {session_info['session'].upper()} | {'OPEN' if session_info['is_open'] else 'CLOSED'}
• Time ET: {session_info['current_time_et']}
• Power Hour: {'YES' if session_info['is_power_hour'] else 'No'}

🎯 **Focus:** ETF Scalping (SPY, QQQ, IWM, GDX)
📍 **EOD Close:** 3:55 PM ET (automatic)

⚡ Hunting for 15%+ daily returns. Maximum aggression.
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
    log.info("ChartBrain active - studying charts in real-time")
    
    # Start SPY 0DTE Scalper - hawk-like monitoring for quick gains
    log.info("Starting SPY 0DTE Scalper (hawk mode)...")
    spy_scalper.start()
    log.info("SPY Scalper active - watching for 15-30% scalp opportunities")
    
    # Start Zero Greed Exit Protocol - mechanical ruthless exit system
    log.info("Starting Zero Greed Exit Protocol...")
    zero_greed_exit.start()
    log.info("Zero Greed Exit active - no mercy mode for exits")
    
    # Start Alpaca Paper Trading Executor - real paper trades
    log.info("Starting Alpaca Paper Trading Executor...")
    alpaca_executor.start_monitoring()
    
    # CRITICAL: Sync any existing Alpaca positions from previous session
    # This prevents orphaned positions that don't get exit-monitored
    synced = alpaca_executor.sync_existing_positions()
    log.info(f"Position sync complete: {synced} existing position(s) now tracked")
    
    account = alpaca_executor.get_account()
    buying_power = float(account.get("buying_power", 0))
    log.info(f"Alpaca Paper Trading active - Buying Power: ${buying_power:,.2f}")
    
    # Run historical training to calibrate engine weights
    log.info("Running historical training (6 weeks)...")
    training_summary = run_startup_training(weeks=6)
    log.info(f"Training complete: {training_summary.get('total_events', 0)} events analyzed")
    
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
