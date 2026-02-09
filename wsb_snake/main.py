"""
WSB Snake - 0DTE Intelligence Engine

Main entry point for the 6-engine trading intelligence pipeline.
Runs continuously, scanning for signals and sending Telegram alerts.
"""

import time
import schedule
from datetime import datetime

# EOD close: run once per day when >= 3:55 PM ET (dedicated trigger, not pipeline-dependent)
_last_eod_run_date = None


def get_last_eod_run_date():
    """Return the date (ET) when EOD close last ran, or None. Used by health/status."""
    return _last_eod_run_date

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
from wsb_snake.engines.momentum_engine import start_momentum_engine
from wsb_snake.engines.leaps_engine import start_leaps_engine
from wsb_snake.learning.zero_greed_exit import zero_greed_exit
from wsb_snake.trading.alpaca_executor import alpaca_executor
from wsb_snake.learning.deep_study import run_idle_study
from wsb_snake.collectors.screenshot_system import screenshot_system
from wsb_snake.execution.regime_detector import regime_detector


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
• Target: +6% | Stop: -10%

📊 **Session:**
• {session_info['session'].upper()} | {'OPEN' if session_info['is_open'] else 'CLOSED'}
• Time ET: {session_info['current_time_et']}
• Power Hour: {'YES' if session_info['is_power_hour'] else 'No'}

🎯 **Focus:** ETF Scalping + Momentum + LEAPS (UNHINGED)
📍 **EOD Close:** 3:55 PM ET (automatic)
🚀 **Momentum:** 2min scan | trim +50% trail +20%
📈 **LEAPS:** 30min scan | trim +50% trail +20%

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


def run_eod_check():
    """Dedicated 3:55 PM ET trigger: close all 0DTE positions once per day. Does not depend on pipeline."""
    global _last_eod_run_date
    if not alpaca_executor.should_close_for_eod():
        return
    import pytz
    et = pytz.timezone("US/Eastern")
    now_et = datetime.now(et)
    today_et = now_et.date()
    if _last_eod_run_date == today_et:
        return
    _last_eod_run_date = today_et
    closed = alpaca_executor.close_all_0dte_positions()
    log.info(f"EOD scheduled close ran: {closed} position(s) closed at {now_et.strftime('%H:%M')} ET")


def run_daily_report():
    """Send end-of-day report at 4:15 PM ET: win rate, avg R, top/worst, 2X/4X/20X tiers."""
    log.info("Sending daily report...")
    try:
        from wsb_snake.notifications.daily_report import send_daily_report as send_daily_report_telegram
        send_daily_report_telegram()
    except Exception as e:
        log.warning("Daily report (new) failed: %s", e)
    send_daily_summary()
    # Apply weight decay for next day
    learning_memory.apply_daily_decay()


def _jobs_report_tracker_should_run() -> bool:
    """True if we should still run the jobs report tracker (until Fri Feb 6, 5 PM ET)."""
    try:
        import pytz
        et = pytz.timezone("America/New_York")
        now_et = datetime.now(et)
        from datetime import date
        end_date = date(2026, 2, 6)
        if now_et.date() > end_date:
            return False
        if now_et.date() == end_date and now_et.hour >= 17:
            return False
        return True
    except Exception:
        return True


def run_jobs_report_tracker_once():
    """Refresh NFP Feb 6 playbook (watchlist + option plays). Runs every 30 min until Fri 5 PM ET."""
    if not _jobs_report_tracker_should_run():
        return
    try:
        from wsb_snake.event_driven.jobs_report_tracker import (
            JobsReportTracker,
            JOBS_REPORT_EVENT_DATE,
            JOBS_REPORT_WATCHLIST,
            BUDGET_WEBBULL_USD,
        )
        from wsb_snake.config import DATA_DIR
        import os
        tracker = JobsReportTracker(
            event_date=JOBS_REPORT_EVENT_DATE,
            watchlist=JOBS_REPORT_WATCHLIST,
            budget_usd=BUDGET_WEBBULL_USD,
        )
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), DATA_DIR)
        tracker.run(output_dir=out_dir)
        log.info("Jobs report playbook updated (NFP Feb 6)")
    except Exception as e:
        log.warning(f"Jobs report tracker skip: {e}")


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
    
    # UNHINGED: Momentum + LEAPS engines (WSB Wilder Plan)
    log.info("Starting Momentum Engine (2min scan)...")
    start_momentum_engine()
    log.info("Starting LEAPS/Macro Engine (30min scan)...")
    start_leaps_engine()
    
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
    
    # EOD close: every minute check; at 3:55 PM ET close all 0DTE (guaranteed, not pipeline-dependent)
    schedule.every(1).minutes.do(run_eod_check)
    
    # Daily report: 4:15 PM ET (after market close)
    schedule.every().day.at("16:15").do(run_daily_report)

    # Screenshot learning: Start background watcher
    log.info("Starting Screenshot Learning System...")
    screenshot_system.start()
    log.info("Screenshot watcher active - learning from Google Drive")

    # Deep study: Run during off-market hours (every 30 min)
    schedule.every(30).minutes.do(run_idle_study)

    # HYDRA: Update regime detector every 5 minutes
    def update_regime():
        try:
            state = regime_detector.fetch_and_update()
            log.info(f"Regime update: {state.regime.value} (confidence={state.confidence:.0%})")
        except Exception as e:
            log.debug(f"Regime update skipped: {e}")
    schedule.every(5).minutes.do(update_regime)
    
    # Jobs report tracker: refresh NFP Feb 6 playbook every 30 min until Fri 5 PM ET
    schedule.every(30).minutes.do(run_jobs_report_tracker_once)
    if _jobs_report_tracker_should_run():
        log.info("Running jobs report tracker (NFP Feb 6)...")
        run_jobs_report_tracker_once()
    
    # HYDRA: Warm up regime detector with initial data
    log.info("Warming up HYDRA regime detector...")
    try:
        regime_state = regime_detector.fetch_and_update()
        log.info(f"Regime: {regime_state.regime.value} (confidence={regime_state.confidence:.0%})")
    except Exception as e:
        log.warning(f"Regime warmup failed (will retry): {e}")

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
