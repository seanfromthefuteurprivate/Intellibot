"""
FastAPI router for trading signals and alerts.
Provides endpoints to query signals from the database with pagination and filtering.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field


# Database path - check multiple possible locations
def get_db_path() -> Path:
    """Get the database path, checking multiple possible locations."""
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "wsb_snake_data" / "wsb_snake.db",
        Path(__file__).parent.parent.parent.parent / "wsb_snake.db",
    ]
    for path in possible_paths:
        if path.exists():
            return path
    # Default to first option even if doesn't exist (will raise error on connection)
    return possible_paths[0]


# Pydantic models for response schemas
class SignalBase(BaseModel):
    """Base signal model with common fields."""
    id: int
    ticker: str
    timestamp: str
    setup_type: str
    score: float
    tier: str
    price: Optional[float] = None
    change_pct: Optional[float] = None
    created_at: Optional[str] = None


class SignalSummary(SignalBase):
    """Signal summary for list views."""
    volume: Optional[int] = None
    social_velocity: Optional[float] = None
    sentiment_score: Optional[float] = None
    session_type: Optional[str] = None


class SignalDetail(SignalBase):
    """Full signal details with all features."""
    # Market features
    volume: Optional[int] = None
    vwap: Optional[float] = None
    range_pct: Optional[float] = None

    # Options features
    atm_iv: Optional[float] = None
    call_put_ratio: Optional[float] = None
    top_strike: Optional[float] = None
    options_pressure_score: Optional[float] = None

    # Social features
    social_velocity: Optional[float] = None
    sentiment_score: Optional[float] = None

    # Session context
    session_type: Optional[str] = None
    minutes_to_close: Optional[float] = None

    # Full feature and evidence data
    features: Optional[dict[str, Any]] = None
    evidence: Optional[dict[str, Any]] = None


class PaginatedSignals(BaseModel):
    """Paginated response for signals list."""
    signals: list[SignalSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


class ActiveSignalsResponse(BaseModel):
    """Response for active signals endpoint."""
    signals: list[SignalSummary]
    count: int
    as_of: str


# Create router
router = APIRouter(prefix="/api/signals", tags=["signals"])


def get_db_connection() -> sqlite3.Connection:
    """Create a database connection with row factory."""
    db_path = get_db_path()
    if not db_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Database not found at {db_path}"
        )
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Convert a sqlite3.Row to a dictionary."""
    return dict(zip(row.keys(), row))


@router.get("", response_model=PaginatedSignals)
@router.get("/", response_model=PaginatedSignals, include_in_schema=False)
async def get_signals(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of signals per page"),
    ticker: Optional[str] = Query(None, description="Filter by ticker symbol"),
    setup_type: Optional[str] = Query(None, description="Filter by setup type"),
    tier: Optional[str] = Query(None, description="Filter by tier (e.g., S, A, B)"),
    min_score: Optional[float] = Query(None, ge=0, le=100, description="Minimum score filter"),
    session_type: Optional[str] = Query(None, description="Filter by session type"),
) -> PaginatedSignals:
    """
    Get recent signals with pagination and optional filtering.

    Returns signals ordered by creation time (most recent first).
    Supports filtering by ticker, setup type, tier, minimum score, and session type.
    """
    conn = get_db_connection()
    try:
        # Build WHERE clause
        conditions = []
        params = []

        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker.upper())
        if setup_type:
            conditions.append("setup_type = ?")
            params.append(setup_type)
        if tier:
            conditions.append("tier = ?")
            params.append(tier.upper())
        if min_score is not None:
            conditions.append("score >= ?")
            params.append(min_score)
        if session_type:
            conditions.append("session_type = ?")
            params.append(session_type)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        # Get total count
        count_query = f"SELECT COUNT(*) FROM signals{where_clause}"
        cursor = conn.execute(count_query, params)
        total = cursor.fetchone()[0]

        # Calculate pagination
        total_pages = max(1, (total + page_size - 1) // page_size)
        offset = (page - 1) * page_size

        # Get signals
        query = f"""
            SELECT id, ticker, timestamp, setup_type, score, tier,
                   price, change_pct, volume, social_velocity,
                   sentiment_score, session_type, created_at
            FROM signals
            {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
        """
        params.extend([page_size, offset])

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        signals = [SignalSummary(**row_to_dict(row)) for row in rows]

        return PaginatedSignals(
            signals=signals,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
    finally:
        conn.close()


@router.get("/active", response_model=ActiveSignalsResponse)
async def get_active_signals(
    max_age_minutes: int = Query(
        30,
        ge=1,
        le=1440,
        description="Maximum age of signals in minutes to be considered active"
    ),
    min_score: Optional[float] = Query(
        None,
        ge=0,
        le=100,
        description="Minimum score filter for active signals"
    ),
    tier: Optional[str] = Query(None, description="Filter by tier (e.g., S, A, B)"),
) -> ActiveSignalsResponse:
    """
    Get currently active signals (not expired).

    Active signals are defined as those created within the specified max_age_minutes.
    Default is 30 minutes. Signals are ordered by score (highest first).
    """
    conn = get_db_connection()
    try:
        # Calculate cutoff time
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_age_minutes)
        cutoff_str = cutoff_time.isoformat()

        # Build query
        conditions = ["created_at >= ?"]
        params: list[Any] = [cutoff_str]

        if min_score is not None:
            conditions.append("score >= ?")
            params.append(min_score)
        if tier:
            conditions.append("tier = ?")
            params.append(tier.upper())

        where_clause = " WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT id, ticker, timestamp, setup_type, score, tier,
                   price, change_pct, volume, social_velocity,
                   sentiment_score, session_type, created_at
            FROM signals
            {where_clause}
            ORDER BY score DESC, created_at DESC
        """

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()

        signals = [SignalSummary(**row_to_dict(row)) for row in rows]

        return ActiveSignalsResponse(
            signals=signals,
            count=len(signals),
            as_of=datetime.utcnow().isoformat()
        )
    finally:
        conn.close()


@router.get("/{signal_id}", response_model=SignalDetail)
async def get_signal_details(signal_id: int) -> SignalDetail:
    """
    Get full details for a specific signal by ID.

    Returns all signal fields including parsed features and evidence JSON.
    """
    conn = get_db_connection()
    try:
        query = """
            SELECT id, ticker, timestamp, setup_type, score, tier,
                   price, volume, change_pct, vwap, range_pct,
                   atm_iv, call_put_ratio, top_strike, options_pressure_score,
                   social_velocity, sentiment_score,
                   session_type, minutes_to_close,
                   features_json, evidence_json, created_at
            FROM signals
            WHERE id = ?
        """

        cursor = conn.execute(query, [signal_id])
        row = cursor.fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Signal with id {signal_id} not found"
            )

        data = row_to_dict(row)

        # Parse JSON fields
        features = None
        evidence = None

        if data.get("features_json"):
            try:
                features = json.loads(data["features_json"])
            except json.JSONDecodeError:
                features = {"raw": data["features_json"]}

        if data.get("evidence_json"):
            try:
                evidence = json.loads(data["evidence_json"])
            except json.JSONDecodeError:
                evidence = {"raw": data["evidence_json"]}

        # Remove raw JSON fields and add parsed versions
        del data["features_json"]
        del data["evidence_json"]
        data["features"] = features
        data["evidence"] = evidence

        return SignalDetail(**data)
    finally:
        conn.close()
