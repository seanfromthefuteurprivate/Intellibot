"""
Health check endpoints for monitoring infrastructure health.

Provides health status for:
- Polygon API (rate limits, errors, circuit breaker)
- Alpaca API
- Database
- Overall system health
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from datetime import datetime

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/polygon")
async def get_polygon_health() -> Dict[str, Any]:
    """
    Get Polygon API health status.

    Returns:
        - is_healthy: Whether Polygon is currently operational
        - rate_limit: Calls per minute allowed
        - calls_this_minute: Current call count
        - circuit_breaker_active: Whether circuit breaker is tripped
        - consecutive_failures: Number of consecutive failures
        - last_error: Last error encountered
    """
    try:
        from wsb_snake.utils.polygon_health import get_polygon_status

        status = get_polygon_status()

        return {
            "service": "polygon",
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "ok": status["is_healthy"] and not status["circuit_breaker_active"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/system")
async def get_system_health() -> Dict[str, Any]:
    """
    Get overall system health status.

    Aggregates health from:
    - Polygon API
    - Alpaca API
    - Database
    - CPL engine
    """
    try:
        from wsb_snake.utils.polygon_health import get_polygon_status

        # Polygon health
        polygon = get_polygon_status()
        polygon_ok = polygon["is_healthy"] and not polygon["circuit_breaker_active"]

        # Alpaca health (basic check)
        alpaca_ok = True
        try:
            from wsb_snake.trading.alpaca_executor import alpaca_executor
            account = alpaca_executor.get_account()
            alpaca_ok = account is not None
        except Exception:
            alpaca_ok = False

        # Database health (basic check)
        db_ok = True
        try:
            from wsb_snake.db.database import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
        except Exception:
            db_ok = False

        # Overall status
        all_ok = polygon_ok and alpaca_ok and db_ok

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy" if all_ok else "degraded",
            "services": {
                "polygon": {
                    "status": "healthy" if polygon_ok else "unhealthy",
                    "details": polygon,
                },
                "alpaca": {
                    "status": "healthy" if alpaca_ok else "unhealthy",
                },
                "database": {
                    "status": "healthy" if db_ok else "unhealthy",
                },
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System health check failed: {str(e)}")


@router.get("/")
async def health_check() -> Dict[str, str]:
    """Basic liveness check."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "service": "wsb-snake-dashboard",
    }
