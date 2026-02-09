"""
WSB Snake - Web Server Wrapper for Replit Deployment
Runs FastAPI server on port 5000 and starts the trading bot as a background task.
"""

import os
import sys
import asyncio
import threading
from datetime import datetime
from contextlib import asynccontextmanager

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Global state
snake_thread = None
snake_running = False
start_time = None


def run_snake_background():
    """Run the WSB Snake in a background thread."""
    global snake_running
    try:
        from wsb_snake.main import main as snake_main
        snake_running = True
        snake_main()
    except Exception as e:
        print(f"Snake error: {e}")
        snake_running = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the snake when server starts."""
    global snake_thread, start_time
    
    print("Starting WSB Snake as background task...")
    start_time = datetime.now()
    
    # Start snake in background thread
    snake_thread = threading.Thread(target=run_snake_background, daemon=True)
    snake_thread.start()
    
    # Give it a moment to initialize
    await asyncio.sleep(2)
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down WSB Snake...")


app = FastAPI(
    title="WSB Snake Trading Bot",
    description="Autonomous 0DTE options scalping engine",
    version="2.5",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "WSB Snake v2.5",
        "mode": "AGGRESSIVE",
        "snake_active": snake_running,
        "uptime_seconds": (datetime.now() - start_time).total_seconds() if start_time else 0
    }


@app.get("/health")
async def health():
    """Health check for Replit."""
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "snake_running": snake_running}
    )


@app.get("/status")
async def status():
    """Get detailed status. Includes last EOD run, open positions, and why-no-trades diagnostics."""
    try:
        from wsb_snake.trading.alpaca_executor import alpaca_executor
        from wsb_snake.engines.spy_scalper import spy_scalper
        from wsb_snake.utils.session_regime import is_market_open, get_session_info

        account = alpaca_executor.get_account()
        options_positions = alpaca_executor.get_options_positions()
        open_count = len(options_positions) if options_positions else 0

        last_eod = None
        try:
            from wsb_snake import main as snake_main
            last_eod = snake_main.get_last_eod_run_date()
        except Exception:
            pass

        # Why-no-trades diagnostics
        market_open = is_market_open()
        session_info = get_session_info()
        active_recipes = 0
        try:
            from wsb_snake.learning.trade_learner import trade_learner
            active_recipes = len(trade_learner._recipes)
        except Exception:
            pass
        why_no_trades = (
            "Market closed (no scans run)." if not market_open else
            "No setup cleared all gates: need confidence >= 85%, AI confirm, order flow agree, sector not weak, no earnings soon, regime match. "
            f"Screenshot recipes boosting confidence: {active_recipes} (0 = screenshot learning not affecting trades yet)."
        )

        return {
            "status": "online",
            "snake_running": snake_running,
            "uptime_seconds": (datetime.now() - start_time).total_seconds() if start_time else 0,
            "open_positions": open_count,
            "last_eod_run_date": str(last_eod) if last_eod else None,
            "account": {
                "buying_power": float(account.get("buying_power", 0)) if account else 0,
                "equity": float(account.get("equity", 0)) if account else 0
            },
            "positions": open_count,
            "config": {
                "max_daily_exposure": alpaca_executor.MAX_DAILY_EXPOSURE,
                "max_per_trade": alpaca_executor.MAX_PER_TRADE,
                "min_confidence": spy_scalper.MIN_CONFIDENCE_FOR_ALERT
            },
            "diagnostics": {
                "market_open": market_open,
                "session": session_info.get("session", "unknown"),
                "min_confidence_for_alert": spy_scalper.MIN_CONFIDENCE_FOR_ALERT,
                "require_ai_confirmation": spy_scalper.REQUIRE_AI_CONFIRMATION,
                "screenshot_active_recipes": active_recipes,
                "why_no_trades": why_no_trades,
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/screenshot-learning")
async def screenshot_learning():
    """
    Google Drive screenshot learning: stats, active recipes, recent learned trades.
    Data comes from wsb_snake_data/wsb_snake.db (screenshots, learned_trades, trade_recipes).
    """
    try:
        from wsb_snake.collectors.screenshot_system import screenshot_system
        from wsb_snake.db.database import get_connection

        screenshot_system._ensure_initialized()
        stats = screenshot_system.get_stats()
        recipes = screenshot_system.list_recipes()

        # Recent learned trades (last 20)
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, ticker, trade_type, entry_price, exit_price, profit_loss, profit_loss_pct,
                   detected_pattern, setup_description, trade_date, entry_time, created_at
            FROM learned_trades
            ORDER BY created_at DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()
        recent_trades = [{k: row[k] for k in row.keys()} for row in rows] if rows else []
        conn.close()

        return {
            "status": "ok",
            "folder_id": stats.get("folder_id"),
            "scan_interval_seconds": stats.get("scan_interval"),
            "collector": stats.get("collector", {}),
            "learner": stats.get("learner", {}),
            "active_recipes_count": len(recipes),
            "recipes": recipes,
            "recent_learned_trades": recent_trades,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Run on port 8081 (dashboard uses 8080)
    port = int(os.environ.get("PORT", 8081))
    uvicorn.run(app, host="0.0.0.0", port=port)
