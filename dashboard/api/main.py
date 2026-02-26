"""
FastAPI Dashboard Backend - Main Entry Point

This is the main entry point for the Intellibot Dashboard API.
Provides real-time trading data, signals, positions, and analytics
via REST endpoints and WebSocket connections.
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add project root to path for wsb_snake imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from wsb_snake package
from wsb_snake.config import (
    ZERO_DTE_UNIVERSE,
    MOMENTUM_UNIVERSE,
    LEAPS_UNIVERSE,
    DB_PATH,
    DATA_DIR,
)
from wsb_snake.db.database import (
    init_database,
    get_connection,
    get_recent_signals,
    get_daily_stats,
    get_daily_stats_for_report,
    get_recent_cpl_calls,
    get_governance_events,
    get_counterfactual_checkpoints,
)
from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# WebSocket Manager for Real-Time Updates
# =============================================================================

class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.

    Features:
    - Connection pooling with automatic cleanup
    - Channel-based subscriptions (signals, positions, stats)
    - Broadcast to all or specific channels
    - Heartbeat/ping support
    """

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "signals": set(),
            "positions": set(),
            "stats": set(),
            "all": set(),
        }
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: str = "all"):
        """Accept a new WebSocket connection and add to channel."""
        await websocket.accept()
        async with self._lock:
            if channel not in self.active_connections:
                self.active_connections[channel] = set()
            self.active_connections[channel].add(websocket)
            self.active_connections["all"].add(websocket)
        logger.info(f"WebSocket connected to channel: {channel}")

    async def disconnect(self, websocket: WebSocket, channel: str = "all"):
        """Remove a WebSocket connection from all channels."""
        async with self._lock:
            for ch in self.active_connections.values():
                ch.discard(websocket)
        logger.info(f"WebSocket disconnected from channel: {channel}")

    async def broadcast(self, message: Dict[str, Any], channel: str = "all"):
        """Broadcast a message to all connections in a channel."""
        if channel not in self.active_connections:
            return

        disconnected = set()
        for connection in self.active_connections[channel].copy():
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected sockets
        async with self._lock:
            for conn in disconnected:
                for ch in self.active_connections.values():
                    ch.discard(conn)

    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send personal message: {e}")

    @property
    def connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.active_connections["all"])


# Global WebSocket manager instance
ws_manager = ConnectionManager()


# =============================================================================
# Application Lifespan Events
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.

    Startup:
    - Initialize database connection
    - Verify wsb_snake data directory exists
    - Start background tasks (if any)

    Shutdown:
    - Close connections gracefully
    - Stop background tasks
    """
    # ===== STARTUP =====
    logger.info("Starting Intellibot Dashboard API...")

    # Initialize database
    try:
        init_database()
        logger.info(f"Database initialized at: {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    # Verify data directory exists
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created data directory: {DATA_DIR}")

    # Log universes loaded
    logger.info(f"0DTE Universe: {len(ZERO_DTE_UNIVERSE)} tickers")
    logger.info(f"Momentum Universe: {len(MOMENTUM_UNIVERSE)} tickers")
    logger.info(f"LEAPS Universe: {len(LEAPS_UNIVERSE)} tickers")

    logger.info("Dashboard API startup complete")

    yield

    # ===== SHUTDOWN =====
    logger.info("Shutting down Dashboard API...")

    # Close all WebSocket connections
    for channel in ws_manager.active_connections.values():
        for connection in channel.copy():
            try:
                await connection.close()
            except Exception:
                pass

    logger.info("Dashboard API shutdown complete")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Intellibot Dashboard API",
    description="Real-time trading dashboard for WSB Snake trading system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware - Allow all origins for dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check Endpoint
# =============================================================================

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns:
        - status: "healthy" or "unhealthy"
        - timestamp: Current server time
        - database: Database connection status
        - websocket_connections: Number of active WebSocket connections
    """
    db_status = "connected"
    try:
        conn = get_connection()
        conn.execute("SELECT 1")
        conn.close()
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status,
        "websocket_connections": ws_manager.connection_count,
        "version": "1.0.0",
    }


# =============================================================================
# API Routes - Import from routes module
# =============================================================================

# Import and include routers from the routes package
# Routers already have their prefix defined (e.g., /api/signals)
try:
    from dashboard.api.routes import (
        signals_router,
        positions_router,
        trades_router,
        pnl_router,
        market_router,
        risk_router,
        events_router,
        account_router,
    )

    # Include all routers - they have their own prefixes already defined
    app.include_router(signals_router)
    app.include_router(positions_router)
    app.include_router(trades_router)
    app.include_router(pnl_router)
    app.include_router(market_router)
    app.include_router(risk_router)
    app.include_router(events_router)
    app.include_router(account_router)

    logger.info("All routers loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import all routers: {e}")
    # Try individual imports for partial functionality
    try:
        from dashboard.api.routes.signals import router as signals_router
        app.include_router(signals_router)
        logger.info("Signals router loaded")
    except ImportError:
        pass

    try:
        from dashboard.api.routes.positions import router as positions_router
        app.include_router(positions_router)
        logger.info("Positions router loaded")
    except ImportError:
        pass

    try:
        from dashboard.api.routes.trades import router as trades_router
        app.include_router(trades_router)
        logger.info("Trades router loaded")
    except ImportError:
        pass

    try:
        from dashboard.api.routes.pnl import router as pnl_router
        app.include_router(pnl_router)
        logger.info("PnL router loaded")
    except ImportError:
        pass

    try:
        from dashboard.api.routes.market import router as market_router
        app.include_router(market_router)
        logger.info("Market router loaded")
    except ImportError:
        pass

    try:
        from dashboard.api.routes.risk import router as risk_router
        app.include_router(risk_router)
        logger.info("Risk router loaded")
    except ImportError:
        pass

    try:
        from dashboard.api.routes.events import router as events_router
        app.include_router(events_router)
        logger.info("Events router loaded")
    except ImportError:
        pass

    try:
        from dashboard.api.routes.account import router as account_router
        app.include_router(account_router)
        logger.info("Account router loaded")
    except ImportError:
        pass

# =============================================================================
# Additional Utility Endpoints
# =============================================================================

@app.get("/api/universes")
async def get_universes():
    """Get ticker universes for different strategies."""
    return {
        "zero_dte": ZERO_DTE_UNIVERSE,
        "momentum": MOMENTUM_UNIVERSE,
        "leaps": LEAPS_UNIVERSE,
    }


@app.get("/api/cpl/calls")
async def get_cpl_calls_endpoint(limit: int = 50, days_back: int = None):
    """Get recent CPL (Convexity Proof Layer) calls."""
    calls = get_recent_cpl_calls(limit=limit, days_back=days_back)
    return {"calls": calls, "count": len(calls)}


@app.get("/api/governance/events")
async def get_governance_endpoint(limit: int = 100, dedupe_key: str = None):
    """Get governance events from Apex layer."""
    events = get_governance_events(dedupe_key=dedupe_key, limit=limit)
    return {"events": events, "count": len(events)}


@app.get("/api/counterfactual")
async def get_counterfactual_endpoint(limit: int = 100, call_id: str = None):
    """Get counterfactual analysis checkpoints."""
    checkpoints = get_counterfactual_checkpoints(call_id=call_id, limit=limit)
    return {"checkpoints": checkpoints, "count": len(checkpoints)}


# =============================================================================
# VENOM WAR ROOM - Complete System Visibility
# =============================================================================

@app.get("/api/war-room")
async def get_war_room_state():
    """
    VENOM WAR ROOM - Every fang visible.

    Returns complete system state:
    - Live P&L per position
    - Win rate (daily, weekly, all-time)
    - GEX regime + flip distance
    - Every engine's status
    - Trade debate transcripts
    - Memory recalls
    - Specialist votes
    - Component health

    This is the MASTER endpoint for system visibility.
    """
    try:
        from wsb_snake.dashboard.war_room import get_war_room_state as fetch_state
        return fetch_state()
    except ImportError as e:
        logger.warning(f"War room import failed: {e}")
        # Fallback: build basic state manually
        return await _build_basic_war_room_state()
    except Exception as e:
        logger.error(f"War room error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}


async def _build_basic_war_room_state():
    """Fallback basic war room state when full module unavailable."""
    state = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "degraded",
        "note": "Full war room module not loaded",
    }

    # Try to get basic executor stats
    try:
        from wsb_snake.trading.alpaca_executor import alpaca_executor
        stats = alpaca_executor.get_session_stats()
        state["executor"] = stats
    except:
        pass

    # Try to get HYDRA state
    try:
        from wsb_snake.collectors.hydra_bridge import get_hydra_intel
        intel = get_hydra_intel()
        state["hydra"] = {
            "connected": intel.connected,
            "direction": intel.direction,
            "regime": intel.regime,
            "blowup_probability": intel.blowup_probability,
            "gex_regime": intel.gex_regime,
            "gex_flip_point": intel.gex_flip_point,
        }
    except:
        pass

    # Try to get risk governor state
    try:
        from wsb_snake.trading.risk_governor import get_risk_governor
        governor = get_risk_governor()
        state["governor"] = governor.get_weaponized_status()
    except:
        pass

    return state


@app.websocket("/ws/war-room")
async def war_room_websocket(websocket: WebSocket):
    """
    Real-time war room updates via WebSocket.

    Sends war room state updates every 5 seconds while connected.
    """
    import asyncio
    await websocket.accept()
    logger.info("War room WebSocket connected")

    try:
        while True:
            # Get current state
            try:
                from wsb_snake.dashboard.war_room import get_war_room_state as fetch_state
                state = fetch_state()
            except:
                state = await _build_basic_war_room_state()

            # Send to client
            await websocket.send_json({
                "type": "war_room_update",
                "data": state,
                "timestamp": datetime.utcnow().isoformat(),
            })

            # Wait 5 seconds
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        logger.info("War room WebSocket disconnected")
    except Exception as e:
        logger.error(f"War room WebSocket error: {e}")


# =============================================================================
# WebSocket Endpoints
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time updates.

    Clients can subscribe to channels:
    - "signals": Real-time trading signals
    - "positions": Position updates (open, close, P&L)
    - "stats": Live statistics updates
    - "all": All updates (default)

    Message format (client -> server):
    {
        "action": "subscribe" | "unsubscribe" | "ping",
        "channel": "signals" | "positions" | "stats" | "all"
    }

    Message format (server -> client):
    {
        "type": "signal" | "position" | "stats" | "pong",
        "data": {...},
        "timestamp": "ISO timestamp"
    }
    """
    channel = "all"
    await ws_manager.connect(websocket, channel)

    try:
        # Send initial connection confirmation
        await ws_manager.send_personal(websocket, {
            "type": "connected",
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat(),
        })

        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                action = message.get("action")

                if action == "ping":
                    await ws_manager.send_personal(websocket, {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                elif action == "subscribe":
                    new_channel = message.get("channel", "all")
                    async with ws_manager._lock:
                        if new_channel not in ws_manager.active_connections:
                            ws_manager.active_connections[new_channel] = set()
                        ws_manager.active_connections[new_channel].add(websocket)
                    await ws_manager.send_personal(websocket, {
                        "type": "subscribed",
                        "channel": new_channel,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

                elif action == "unsubscribe":
                    old_channel = message.get("channel", "all")
                    async with ws_manager._lock:
                        if old_channel in ws_manager.active_connections:
                            ws_manager.active_connections[old_channel].discard(websocket)
                    await ws_manager.send_personal(websocket, {
                        "type": "unsubscribed",
                        "channel": old_channel,
                        "timestamp": datetime.utcnow().isoformat(),
                    })

            except json.JSONDecodeError:
                await ws_manager.send_personal(websocket, {
                    "type": "error",
                    "message": "Invalid JSON",
                    "timestamp": datetime.utcnow().isoformat(),
                })

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, channel)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect(websocket, channel)


@app.websocket("/ws/{channel}")
async def websocket_channel_endpoint(websocket: WebSocket, channel: str):
    """
    Channel-specific WebSocket endpoint.

    Directly connects to a specific channel without subscription message.

    Channels:
    - /ws/signals - Trading signals only
    - /ws/positions - Position updates only
    - /ws/stats - Statistics updates only
    """
    if channel not in ["signals", "positions", "stats", "all"]:
        await websocket.close(code=4000, reason=f"Invalid channel: {channel}")
        return

    await ws_manager.connect(websocket, channel)

    try:
        await ws_manager.send_personal(websocket, {
            "type": "connected",
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat(),
        })

        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                if message.get("action") == "ping":
                    await ws_manager.send_personal(websocket, {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                    })
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, channel)
    except Exception as e:
        logger.error(f"WebSocket error on channel {channel}: {e}")
        await ws_manager.disconnect(websocket, channel)


# =============================================================================
# Static Files - Serve React Build
# =============================================================================

# Path to React build directory
STATIC_DIR = PROJECT_ROOT / "dashboard" / "build"

if STATIC_DIR.exists():
    # Mount static files for React app
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR / "static")), name="static")

    @app.get("/")
    async def serve_react_app():
        """Serve React app index.html."""
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/{full_path:path}")
    async def serve_react_routes(full_path: str):
        """
        Catch-all route for React Router.
        Serves index.html for all non-API routes.
        """
        # Don't catch API routes
        if full_path.startswith("api/") or full_path.startswith("ws"):
            return {"error": "Not found"}

        # Check if file exists in static
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))

        # Otherwise serve index.html for React Router
        return FileResponse(str(STATIC_DIR / "index.html"))
else:
    @app.get("/")
    async def root():
        """Root endpoint when React build not available."""
        return {
            "message": "Intellibot Dashboard API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "note": "React build not found. Run npm build in dashboard directory.",
        }


# =============================================================================
# Utility Functions for Broadcasting
# =============================================================================

async def broadcast_signal(signal_data: Dict[str, Any]):
    """Broadcast a new trading signal to WebSocket clients."""
    await ws_manager.broadcast({
        "type": "signal",
        "data": signal_data,
        "timestamp": datetime.utcnow().isoformat(),
    }, channel="signals")


async def broadcast_position_update(position_data: Dict[str, Any]):
    """Broadcast a position update to WebSocket clients."""
    await ws_manager.broadcast({
        "type": "position",
        "data": position_data,
        "timestamp": datetime.utcnow().isoformat(),
    }, channel="positions")


async def broadcast_stats_update(stats_data: Dict[str, Any]):
    """Broadcast statistics update to WebSocket clients."""
    await ws_manager.broadcast({
        "type": "stats",
        "data": stats_data,
        "timestamp": datetime.utcnow().isoformat(),
    }, channel="stats")


# Export for use by other modules
__all__ = [
    "app",
    "ws_manager",
    "broadcast_signal",
    "broadcast_position_update",
    "broadcast_stats_update",
]


# =============================================================================
# Development Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get port from environment or default to 8000
    port = int(os.environ.get("DASHBOARD_PORT", 8000))
    host = os.environ.get("DASHBOARD_HOST", "0.0.0.0")

    logger.info(f"Starting development server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
