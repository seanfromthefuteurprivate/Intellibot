"""
FastAPI Router for Trade History

Endpoints:
- GET /api/trades - List recent trades with pagination
- GET /api/trades/today - Today's trades only
- GET /api/trades/stats - Aggregate stats (win_rate, total_pnl, avg_pnl)
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from wsb_snake.db.database import get_connection


router = APIRouter(prefix="/api/trades", tags=["trades"])


# ─────────────────────────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────────────────────────

class TradeRecord(BaseModel):
    """Individual trade record from trade_performance table."""
    id: int
    trade_date: str
    symbol: str
    engine: Optional[str] = None
    trade_type: Optional[str] = None
    entry_hour: Optional[int] = None
    session: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    r_multiple: Optional[float] = None
    exit_reason: Optional[str] = None
    holding_time_seconds: Optional[int] = None
    signal_id: Optional[int] = None
    event_tier: Optional[str] = None
    created_at: Optional[str] = None


class TradeListResponse(BaseModel):
    """Response for paginated trade list."""
    trades: List[TradeRecord]
    total: int
    limit: int
    offset: int


class TradeStatsResponse(BaseModel):
    """Aggregate trade statistics."""
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    avg_r_multiple: Optional[float] = None
    best_trade_pnl: Optional[float] = None
    worst_trade_pnl: Optional[float] = None


# ─────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────

def _row_to_trade(row) -> TradeRecord:
    """Convert a database row to a TradeRecord."""
    return TradeRecord(
        id=row["id"],
        trade_date=row["trade_date"],
        symbol=row["symbol"],
        engine=row["engine"],
        trade_type=row["trade_type"],
        entry_hour=row["entry_hour"],
        session=row["session"],
        pnl=row["pnl"],
        pnl_pct=row["pnl_pct"],
        r_multiple=row["r_multiple"],
        exit_reason=row["exit_reason"],
        holding_time_seconds=row["holding_time_seconds"],
        signal_id=row["signal_id"],
        event_tier=row["event_tier"],
        created_at=row["created_at"],
    )


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@router.get("", response_model=TradeListResponse)
@router.get("/", response_model=TradeListResponse, include_in_schema=False)
async def list_trades(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of trades to return"),
    offset: int = Query(default=0, ge=0, description="Number of trades to skip"),
) -> TradeListResponse:
    """
    List recent trades with pagination.

    Returns trades ordered by created_at descending (most recent first).
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Get total count
    cursor.execute("SELECT COUNT(*) as count FROM trade_performance")
    total = cursor.fetchone()["count"]

    # Get paginated trades
    cursor.execute("""
        SELECT * FROM trade_performance
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))

    rows = cursor.fetchall()
    conn.close()

    trades = [_row_to_trade(row) for row in rows]

    return TradeListResponse(
        trades=trades,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/today", response_model=TradeListResponse)
async def get_today_trades() -> TradeListResponse:
    """
    Get all trades from today.

    Uses the current UTC date to filter trades.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")

    conn = get_connection()
    cursor = conn.cursor()

    # Get count for today
    cursor.execute(
        "SELECT COUNT(*) as count FROM trade_performance WHERE trade_date = ?",
        (today,)
    )
    total = cursor.fetchone()["count"]

    # Get all trades for today
    cursor.execute("""
        SELECT * FROM trade_performance
        WHERE trade_date = ?
        ORDER BY created_at DESC
    """, (today,))

    rows = cursor.fetchall()
    conn.close()

    trades = [_row_to_trade(row) for row in rows]

    return TradeListResponse(
        trades=trades,
        total=total,
        limit=total,
        offset=0,
    )


@router.get("/stats", response_model=TradeStatsResponse)
async def get_trade_stats(
    date: Optional[str] = Query(default=None, description="Filter stats by date (YYYY-MM-DD). If not provided, returns all-time stats."),
) -> TradeStatsResponse:
    """
    Get aggregate trade statistics.

    Returns win rate, total P&L, average P&L, and other metrics.
    Optionally filter by a specific date.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if date:
        # Stats for specific date
        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(AVG(pnl), 0) as avg_pnl,
                AVG(r_multiple) as avg_r_multiple,
                MAX(pnl) as best_trade_pnl,
                MIN(pnl) as worst_trade_pnl
            FROM trade_performance
            WHERE trade_date = ?
        """, (date,))
    else:
        # All-time stats
        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(AVG(pnl), 0) as avg_pnl,
                AVG(r_multiple) as avg_r_multiple,
                MAX(pnl) as best_trade_pnl,
                MIN(pnl) as worst_trade_pnl
            FROM trade_performance
        """)

    row = cursor.fetchone()
    conn.close()

    total_trades = row["total_trades"] or 0
    wins = row["wins"] or 0
    losses = row["losses"] or 0

    # Calculate win rate (avoid division by zero)
    win_rate = (wins / total_trades) if total_trades > 0 else 0.0

    return TradeStatsResponse(
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_pnl=row["total_pnl"] or 0.0,
        avg_pnl=row["avg_pnl"] or 0.0,
        avg_r_multiple=row["avg_r_multiple"],
        best_trade_pnl=row["best_trade_pnl"],
        worst_trade_pnl=row["worst_trade_pnl"],
    )
