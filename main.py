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
    """Get detailed status."""
    try:
        from wsb_snake.trading.alpaca_executor import alpaca_executor
        from wsb_snake.engines.spy_scalper import spy_scalper
        
        account = alpaca_executor.get_account()
        positions = alpaca_executor.get_positions()
        
        return {
            "status": "online",
            "snake_running": snake_running,
            "uptime_seconds": (datetime.now() - start_time).total_seconds() if start_time else 0,
            "account": {
                "buying_power": float(account.get("buying_power", 0)) if account else 0,
                "equity": float(account.get("equity", 0)) if account else 0
            },
            "positions": len(positions) if positions else 0,
            "config": {
                "max_daily_exposure": alpaca_executor.MAX_DAILY_EXPOSURE,
                "max_per_trade": alpaca_executor.MAX_PER_TRADE,
                "min_confidence": spy_scalper.MIN_CONFIDENCE_FOR_ALERT
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    # Run on port 5000 as required by Replit
    uvicorn.run(app, host="0.0.0.0", port=5000)
