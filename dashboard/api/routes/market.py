"""
Market Data API Routes

FastAPI router providing market data endpoints:
- VIX spot and term structure
- Market regime detection
- Price snapshots for tickers
"""

from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Path, Query
from pydantic import BaseModel, Field

# Import VIX structure collector
from wsb_snake.collectors.vix_structure import vix_structure, VIXStructureCollector

# Import regime detector
from wsb_snake.execution.regime_detector import (
    regime_detector,
    RegimeDetector,
    MarketRegime,
    RegimeState,
    detect_current_regime,
    get_signal_multipliers,
)

# Import polygon for price snapshots
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced, EnhancedPolygonAdapter


router = APIRouter(prefix="/api/market", tags=["market"])


# ============================================================================
# Response Models
# ============================================================================


class VIXSpotResponse(BaseModel):
    """VIX spot price response."""
    vix_spot: float = Field(..., description="Current VIX spot price")
    timestamp: str = Field(..., description="ISO timestamp of data")


class VIXTermStructureResponse(BaseModel):
    """VIX term structure response."""
    vix_spot: float = Field(..., description="VIX spot price")
    front_month: float = Field(..., description="Front month futures price")
    second_month: float = Field(..., description="Second month futures price")
    spot_to_front_pct: float = Field(..., description="Spot to front month percentage difference")
    front_to_second_pct: float = Field(..., description="Front to second month percentage difference")
    structure: str = Field(..., description="Term structure type (contango, backwardation, etc.)")
    fear_level: str = Field(..., description="Market fear level indicator")
    vix_regime: str = Field(..., description="VIX regime classification")
    signal: str = Field(..., description="Trading signal based on structure")
    options_bias: str = Field(..., description="Options strategy bias")
    is_backwardation: bool = Field(..., description="Whether VIX is in backwardation")
    is_contango: bool = Field(..., description="Whether VIX is in contango")
    futures: list = Field(..., description="List of futures prices by month")
    timestamp: str = Field(..., description="ISO timestamp of data")


class VIXResponse(BaseModel):
    """Combined VIX response with spot and term structure."""
    spot: VIXSpotResponse
    term_structure: VIXTermStructureResponse
    trading_signal: Dict[str, Any] = Field(..., description="Actionable trading signal")


class RegimeResponse(BaseModel):
    """Market regime response."""
    regime: str = Field(..., description="Current market regime classification")
    confidence: float = Field(..., description="Confidence in classification (0-1)")
    vix_level: float = Field(..., description="Current VIX level")
    vix_structure: str = Field(..., description="VIX term structure (contango/backwardation)")
    trend_strength: float = Field(..., description="Trend strength (-1 to +1)")
    mean_reversion_score: float = Field(..., description="Mean reversion likelihood (0-1)")
    regime_duration: int = Field(..., description="How long in current regime")
    data_points: int = Field(..., description="Number of data points used")
    detected_at: str = Field(..., description="ISO timestamp of detection")
    multipliers: Dict[str, float] = Field(..., description="Signal weight multipliers for current regime")


class SnapshotResponse(BaseModel):
    """Price snapshot response for a ticker."""
    symbol: str = Field(..., description="Ticker symbol")
    price: float = Field(..., description="Current price")
    today_open: float = Field(..., description="Today's opening price")
    today_high: float = Field(..., description="Today's high price")
    today_low: float = Field(..., description="Today's low price")
    today_close: float = Field(..., description="Today's closing/current price")
    today_volume: int = Field(..., description="Today's volume")
    today_vwap: float = Field(..., description="Today's volume-weighted average price")
    prev_close: float = Field(..., description="Previous day's close")
    prev_volume: int = Field(..., description="Previous day's volume")
    change_pct: float = Field(..., description="Percentage change from previous close")
    updated: int = Field(..., description="Last update timestamp (epoch)")
    timestamp: str = Field(..., description="ISO timestamp of request")


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/vix", response_model=VIXResponse, summary="Get VIX Data")
async def get_vix() -> VIXResponse:
    """
    Get current VIX spot price and term structure.

    Returns VIX spot, futures term structure, contango/backwardation status,
    fear levels, and actionable trading signals.
    """
    try:
        # Get VIX spot price
        vix_spot = vix_structure.get_vix_spot()

        # Get full term structure
        term_data = vix_structure.get_term_structure()

        # Get trading signal
        trading_signal = vix_structure.get_trading_signal()

        now = datetime.now().isoformat()

        return VIXResponse(
            spot=VIXSpotResponse(
                vix_spot=vix_spot,
                timestamp=now,
            ),
            term_structure=VIXTermStructureResponse(
                vix_spot=term_data.get("vix_spot", vix_spot),
                front_month=term_data.get("front_month", 0.0),
                second_month=term_data.get("second_month", 0.0),
                spot_to_front_pct=term_data.get("spot_to_front_pct", 0.0),
                front_to_second_pct=term_data.get("front_to_second_pct", 0.0),
                structure=term_data.get("structure", "unknown"),
                fear_level=term_data.get("fear_level", "unknown"),
                vix_regime=term_data.get("vix_regime", "unknown"),
                signal=term_data.get("signal", "neutral"),
                options_bias=term_data.get("options_bias", "balanced"),
                is_backwardation=term_data.get("is_backwardation", False),
                is_contango=term_data.get("is_contango", False),
                futures=term_data.get("futures", []),
                timestamp=term_data.get("timestamp", now),
            ),
            trading_signal=trading_signal,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch VIX data: {str(e)}")


@router.get("/regime", response_model=RegimeResponse, summary="Get Market Regime")
async def get_regime(
    fetch_fresh: bool = Query(
        default=False,
        description="Whether to fetch fresh data before detecting regime"
    )
) -> RegimeResponse:
    """
    Get current market regime from the HYDRA-inspired regime detector.

    Returns regime classification (trending_up, trending_down, mean_reverting,
    high_vol, crash, recovery, unknown) with confidence scores and signal
    weight multipliers for adaptive trading.

    Set fetch_fresh=true to fetch latest market data before detection.
    """
    try:
        # Optionally fetch fresh data
        if fetch_fresh:
            regime_detector.fetch_and_update()

        # Detect current regime
        state = regime_detector.detect_regime()

        # Get regime summary with additional metadata
        summary = regime_detector.get_regime_summary()

        # Get signal multipliers for current regime
        multipliers = regime_detector.get_regime_multipliers()

        return RegimeResponse(
            regime=state.regime.value,
            confidence=round(state.confidence, 4),
            vix_level=round(state.vix_level, 2),
            vix_structure=state.vix_structure,
            trend_strength=round(state.trend_strength, 4),
            mean_reversion_score=round(state.mean_reversion_score, 4),
            regime_duration=summary.get("regime_duration", 0),
            data_points=summary.get("data_points", 0),
            detected_at=state.detected_at.isoformat(),
            multipliers=multipliers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to detect regime: {str(e)}")


@router.get(
    "/snapshot/{ticker}",
    response_model=SnapshotResponse,
    summary="Get Price Snapshot"
)
async def get_snapshot(
    ticker: str = Path(
        ...,
        description="Stock ticker symbol (e.g., SPY, AAPL, TSLA)",
        min_length=1,
        max_length=10,
    )
) -> SnapshotResponse:
    """
    Get real-time price snapshot for a ticker.

    Returns current price, today's OHLCV data, previous day's close,
    percentage change, and VWAP.
    """
    # Normalize ticker to uppercase
    ticker = ticker.upper().strip()

    try:
        # Get snapshot from Polygon
        snapshot = polygon_enhanced.get_snapshot(ticker)

        if snapshot is None:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for ticker: {ticker}"
            )

        return SnapshotResponse(
            symbol=snapshot.get("symbol", ticker),
            price=snapshot.get("price", 0.0),
            today_open=snapshot.get("today_open", 0.0),
            today_high=snapshot.get("today_high", 0.0),
            today_low=snapshot.get("today_low", 0.0),
            today_close=snapshot.get("today_close", 0.0),
            today_volume=int(snapshot.get("today_volume", 0)),
            today_vwap=snapshot.get("today_vwap", 0.0),
            prev_close=snapshot.get("prev_close", 0.0),
            prev_volume=int(snapshot.get("prev_volume", 0)),
            change_pct=snapshot.get("change_pct", 0.0),
            updated=snapshot.get("updated", 0),
            timestamp=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch snapshot for {ticker}: {str(e)}"
        )
