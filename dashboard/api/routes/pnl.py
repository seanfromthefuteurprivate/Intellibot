"""
P&L API Routes

FastAPI router for Profit & Loss data:
- Today's realized + unrealized P&L
- Daily P&L history (last 30 days)
- Total P&L summary with breakdown by engine
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Database connection
from wsb_snake.db.database import get_connection

# Alpaca executor for live position data
from wsb_snake.trading.alpaca_executor import alpaca_executor


router = APIRouter(prefix="/api/pnl", tags=["pnl"])


# ─────────────────────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────────────────────

class PositionPnL(BaseModel):
    """P&L for a single position."""
    symbol: str
    option_symbol: str
    qty: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    status: str


class TodayPnL(BaseModel):
    """Today's P&L summary."""
    date: str
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    trades_today: int
    wins: int
    losses: int
    win_rate: float
    open_positions: List[PositionPnL]


class DailyPnL(BaseModel):
    """P&L for a single day."""
    date: str
    pnl: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    avg_r_multiple: Optional[float] = None


class EnginePnL(BaseModel):
    """P&L breakdown by trading engine."""
    engine: str
    pnl: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    avg_r_multiple: Optional[float] = None


class PnLSummary(BaseModel):
    """Total P&L summary with breakdown."""
    total_pnl: float
    total_trades: int
    total_wins: int
    total_losses: int
    overall_win_rate: float
    avg_r_multiple: Optional[float] = None
    best_day: Optional[DailyPnL] = None
    worst_day: Optional[DailyPnL] = None
    by_engine: List[EnginePnL]
    by_symbol: Dict[str, float]


# ─────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────

def _get_unrealized_positions() -> List[PositionPnL]:
    """Get current open positions with unrealized P&L from Alpaca."""
    positions = []
    try:
        alpaca_positions = alpaca_executor.get_options_positions()
        for pos in alpaca_positions:
            qty = int(pos.get("qty", 0))
            entry_price = float(pos.get("avg_entry_price", 0))
            current_price = float(pos.get("current_price", 0))
            unrealized_pnl = float(pos.get("unrealized_pl", 0))

            if entry_price > 0:
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                unrealized_pnl_pct = 0.0

            positions.append(PositionPnL(
                symbol=pos.get("symbol", "")[:3],  # Extract underlying (e.g., SPY)
                option_symbol=pos.get("symbol", ""),
                qty=qty,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                status="OPEN"
            ))
    except Exception as e:
        # Return empty list if Alpaca unavailable
        pass

    return positions


def _get_realized_pnl_for_date(date_str: str) -> Dict:
    """Get realized P&L from database for a specific date."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COALESCE(SUM(pnl), 0) as total_pnl,
            COUNT(*) as trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
            AVG(r_multiple) as avg_r
        FROM trade_performance
        WHERE trade_date = ?
    """, (date_str,))

    row = cursor.fetchone()
    conn.close()

    trades = row["trades"] or 0
    wins = row["wins"] or 0

    return {
        "pnl": row["total_pnl"] or 0.0,
        "trades": trades,
        "wins": wins,
        "losses": row["losses"] or 0,
        "win_rate": (wins / trades * 100) if trades > 0 else 0.0,
        "avg_r_multiple": row["avg_r"]
    }


# ─────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────

@router.get("/today", response_model=TodayPnL)
async def get_today_pnl():
    """
    Get today's P&L including:
    - Realized P&L from closed trades
    - Unrealized P&L from open positions
    - Combined total P&L
    """
    today_str = datetime.utcnow().strftime("%Y-%m-%d")

    # Get realized P&L from database
    realized = _get_realized_pnl_for_date(today_str)

    # Get unrealized P&L from Alpaca positions
    open_positions = _get_unrealized_positions()
    unrealized_pnl = sum(pos.unrealized_pnl for pos in open_positions)

    # Also check executor's session stats for any in-memory trades
    session_stats = alpaca_executor.get_session_stats()

    # Use whichever has more trades (db or executor)
    if session_stats["total_trades"] > realized["trades"]:
        realized_pnl = session_stats["total_pnl"]
        trades_today = session_stats["total_trades"]
        wins = session_stats["winning_trades"]
        losses = trades_today - wins
    else:
        realized_pnl = realized["pnl"]
        trades_today = realized["trades"]
        wins = realized["wins"]
        losses = realized["losses"]

    win_rate = (wins / trades_today * 100) if trades_today > 0 else 0.0

    return TodayPnL(
        date=today_str,
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        total_pnl=realized_pnl + unrealized_pnl,
        trades_today=trades_today,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        open_positions=open_positions
    )


@router.get("/daily", response_model=List[DailyPnL])
async def get_daily_pnl(days: int = 30):
    """
    Get daily P&L for the last N days (default 30).
    Returns list of daily P&L records sorted by date descending.
    """
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 365")

    conn = get_connection()
    cursor = conn.cursor()

    # Calculate date range
    end_date = datetime.utcnow().strftime("%Y-%m-%d")
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")

    cursor.execute("""
        SELECT
            trade_date,
            COALESCE(SUM(pnl), 0) as total_pnl,
            COUNT(*) as trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
            AVG(r_multiple) as avg_r
        FROM trade_performance
        WHERE trade_date BETWEEN ? AND ?
        GROUP BY trade_date
        ORDER BY trade_date DESC
    """, (start_date, end_date))

    rows = cursor.fetchall()
    conn.close()

    daily_pnl = []
    for row in rows:
        trades = row["trades"] or 0
        wins = row["wins"] or 0

        daily_pnl.append(DailyPnL(
            date=row["trade_date"],
            pnl=row["total_pnl"] or 0.0,
            trades=trades,
            wins=wins,
            losses=row["losses"] or 0,
            win_rate=(wins / trades * 100) if trades > 0 else 0.0,
            avg_r_multiple=row["avg_r"]
        ))

    return daily_pnl


@router.get("/summary", response_model=PnLSummary)
async def get_pnl_summary():
    """
    Get total P&L summary with breakdown by:
    - Trading engine (scalper, momentum, macro, CPL)
    - Symbol
    - Best/worst days
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Overall totals
    cursor.execute("""
        SELECT
            COALESCE(SUM(pnl), 0) as total_pnl,
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
            AVG(r_multiple) as avg_r
        FROM trade_performance
    """)
    totals = cursor.fetchone()

    total_trades = totals["total_trades"] or 0
    total_wins = totals["wins"] or 0
    total_losses = totals["losses"] or 0

    # By engine
    cursor.execute("""
        SELECT
            COALESCE(engine, 'unknown') as engine,
            COALESCE(SUM(pnl), 0) as total_pnl,
            COUNT(*) as trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
            AVG(r_multiple) as avg_r
        FROM trade_performance
        GROUP BY engine
        ORDER BY total_pnl DESC
    """)
    engine_rows = cursor.fetchall()

    by_engine = []
    for row in engine_rows:
        trades = row["trades"] or 0
        wins = row["wins"] or 0
        by_engine.append(EnginePnL(
            engine=row["engine"] or "unknown",
            pnl=row["total_pnl"] or 0.0,
            trades=trades,
            wins=wins,
            losses=row["losses"] or 0,
            win_rate=(wins / trades * 100) if trades > 0 else 0.0,
            avg_r_multiple=row["avg_r"]
        ))

    # By symbol
    cursor.execute("""
        SELECT
            symbol,
            COALESCE(SUM(pnl), 0) as total_pnl
        FROM trade_performance
        GROUP BY symbol
        ORDER BY total_pnl DESC
    """)
    symbol_rows = cursor.fetchall()
    by_symbol = {row["symbol"]: row["total_pnl"] or 0.0 for row in symbol_rows if row["symbol"]}

    # Best day
    cursor.execute("""
        SELECT
            trade_date,
            COALESCE(SUM(pnl), 0) as total_pnl,
            COUNT(*) as trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
            AVG(r_multiple) as avg_r
        FROM trade_performance
        GROUP BY trade_date
        ORDER BY total_pnl DESC
        LIMIT 1
    """)
    best_row = cursor.fetchone()
    best_day = None
    if best_row and best_row["trade_date"]:
        trades = best_row["trades"] or 0
        wins = best_row["wins"] or 0
        best_day = DailyPnL(
            date=best_row["trade_date"],
            pnl=best_row["total_pnl"] or 0.0,
            trades=trades,
            wins=wins,
            losses=best_row["losses"] or 0,
            win_rate=(wins / trades * 100) if trades > 0 else 0.0,
            avg_r_multiple=best_row["avg_r"]
        )

    # Worst day
    cursor.execute("""
        SELECT
            trade_date,
            COALESCE(SUM(pnl), 0) as total_pnl,
            COUNT(*) as trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
            AVG(r_multiple) as avg_r
        FROM trade_performance
        GROUP BY trade_date
        ORDER BY total_pnl ASC
        LIMIT 1
    """)
    worst_row = cursor.fetchone()
    worst_day = None
    if worst_row and worst_row["trade_date"]:
        trades = worst_row["trades"] or 0
        wins = worst_row["wins"] or 0
        worst_day = DailyPnL(
            date=worst_row["trade_date"],
            pnl=worst_row["total_pnl"] or 0.0,
            trades=trades,
            wins=wins,
            losses=worst_row["losses"] or 0,
            win_rate=(wins / trades * 100) if trades > 0 else 0.0,
            avg_r_multiple=worst_row["avg_r"]
        )

    conn.close()

    return PnLSummary(
        total_pnl=totals["total_pnl"] or 0.0,
        total_trades=total_trades,
        total_wins=total_wins,
        total_losses=total_losses,
        overall_win_rate=(total_wins / total_trades * 100) if total_trades > 0 else 0.0,
        avg_r_multiple=totals["avg_r"],
        best_day=best_day,
        worst_day=worst_day,
        by_engine=by_engine,
        by_symbol=by_symbol
    )
