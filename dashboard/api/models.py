"""
Pydantic Models for Dashboard API Responses

Provides typed response models for all API endpoints including:
- Positions and trades
- Signals and alerts
- PnL and performance metrics
- Market data
- Risk status
- Events and governance
- Account information
- Session statistics
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class PositionStatusEnum(str, Enum):
    """Position lifecycle status."""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class SignalTierEnum(str, Enum):
    """Signal quality tiers for routing."""
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    BLOCKED = "X"


class SignalActionEnum(str, Enum):
    """Recommended action for a signal."""
    WATCH = "WATCH"
    WAIT = "WAIT"
    ENTER = "ENTER"
    EXIT = "EXIT"
    HEDGE = "HEDGE"


class TimeHorizonEnum(str, Enum):
    """Trading time horizon."""
    ZERO_DTE = "0DTE"
    INTRADAY = "INTRADAY"
    SWING = "SWING"
    POSITION = "POSITION"


class OrderSideEnum(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class TradeTypeEnum(str, Enum):
    """Trade type for options."""
    CALLS = "CALLS"
    PUTS = "PUTS"


class EngineTypeEnum(str, Enum):
    """Trading engine type."""
    SCALPER = "scalper"
    MOMENTUM = "momentum"
    MACRO = "macro"
    CPL = "CPL"


# ============================================================================
# Nested Models
# ============================================================================

class MarketDataQuote(BaseModel):
    """Real-time quote data."""
    bid: float = Field(default=0.0, description="Current bid price")
    ask: float = Field(default=0.0, description="Current ask price")
    mid: float = Field(default=0.0, description="Mid-point price")
    spread: float = Field(default=0.0, description="Bid-ask spread")
    spread_pct: float = Field(default=0.0, description="Spread as percentage of mid")
    timestamp: Optional[datetime] = Field(default=None, description="Quote timestamp")


class RiskFlags(BaseModel):
    """Risk assessment flags for a signal or position."""
    low_liquidity: bool = Field(default=False, description="Volume below threshold")
    wide_spread: bool = Field(default=False, description="Spread exceeds threshold")
    pump_detected: bool = Field(default=False, description="Pump pattern detected")
    high_volatility: bool = Field(default=False, description="Volatility exceeds threshold")
    news_uncertainty: bool = Field(default=False, description="Pending news event")
    regime_unfavorable: bool = Field(default=False, description="Market regime unfavorable")
    blocked: bool = Field(default=False, description="Trade blocked by risk rules")
    block_reason: str = Field(default="", description="Reason for blocking")


class SocialMetrics(BaseModel):
    """Social/crowd metrics for a ticker."""
    mention_count: int = Field(default=0, description="Total mentions")
    velocity: float = Field(default=0.0, description="Mentions per minute")
    acceleration: float = Field(default=0.0, description="Velocity change rate")
    author_count: int = Field(default=0, description="Unique authors mentioning")
    upvote_ratio: float = Field(default=0.0, description="Average upvote ratio")
    comment_count: int = Field(default=0, description="Total comments")
    sentiment_score: float = Field(default=0.0, description="Sentiment -1 to +1")
    intent_tags: List[str] = Field(default_factory=list, description="Intent classifications")


class PriceLevels(BaseModel):
    """Key price levels for a ticker."""
    support: Optional[float] = Field(default=None, description="Support level")
    resistance: Optional[float] = Field(default=None, description="Resistance level")
    vwap: Optional[float] = Field(default=None, description="Volume-weighted average price")
    pivot: Optional[float] = Field(default=None, description="Pivot point")
    target_1: Optional[float] = Field(default=None, description="First target price")
    target_2: Optional[float] = Field(default=None, description="Second target price")
    stop_loss: Optional[float] = Field(default=None, description="Stop loss level")


# ============================================================================
# Main Response Models
# ============================================================================

class PositionResponse(BaseModel):
    """Response model for a trading position."""
    position_id: str = Field(..., description="Unique position identifier")
    symbol: str = Field(..., description="Underlying ticker symbol")
    option_symbol: str = Field(..., description="Full OCC option symbol")
    side: str = Field(..., description="Position side: 'long' or 'short'")
    trade_type: TradeTypeEnum = Field(..., description="CALLS or PUTS")
    qty: int = Field(..., description="Number of contracts")
    entry_price: float = Field(..., description="Entry price per contract")
    target_price: float = Field(..., description="Target exit price")
    stop_loss: float = Field(..., description="Stop loss price")
    status: PositionStatusEnum = Field(..., description="Current position status")
    entry_time: Optional[datetime] = Field(default=None, description="Entry timestamp")
    exit_price: Optional[float] = Field(default=None, description="Exit price if closed")
    exit_time: Optional[datetime] = Field(default=None, description="Exit timestamp if closed")
    pnl: float = Field(default=0.0, description="Realized P&L in dollars")
    pnl_pct: float = Field(default=0.0, description="Realized P&L percentage")
    alpaca_order_id: Optional[str] = Field(default=None, description="Alpaca order ID")
    exit_order_id: Optional[str] = Field(default=None, description="Exit order ID")
    exit_reason: Optional[str] = Field(default=None, description="Reason for exit")
    engine: EngineTypeEnum = Field(default=EngineTypeEnum.SCALPER, description="Trading engine")
    trimmed: bool = Field(default=False, description="Position partially trimmed")
    signal_id: Optional[int] = Field(default=None, description="Associated signal ID")
    current_price: Optional[float] = Field(default=None, description="Current market price")
    unrealized_pnl: Optional[float] = Field(default=None, description="Unrealized P&L")
    unrealized_pnl_pct: Optional[float] = Field(default=None, description="Unrealized P&L %")
    option_spec: Optional[str] = Field(default=None, description="Human-readable option spec")

    class Config:
        use_enum_values = True


class TradeResponse(BaseModel):
    """Response model for a completed trade."""
    id: int = Field(..., description="Trade ID")
    signal_id: Optional[int] = Field(default=None, description="Associated signal ID")
    ticker: str = Field(..., description="Ticker symbol")
    trade_type: str = Field(..., description="Trade type (CALLS/PUTS)")
    direction: str = Field(..., description="Trade direction (long/short)")
    engine: Optional[str] = Field(default=None, description="Trading engine used")
    entry_price: float = Field(..., description="Entry price")
    exit_price: Optional[float] = Field(default=None, description="Exit price")
    stop_price: Optional[float] = Field(default=None, description="Stop loss price")
    target_1_price: Optional[float] = Field(default=None, description="First target")
    target_2_price: Optional[float] = Field(default=None, description="Second target")
    position_size: int = Field(default=1, description="Number of contracts")
    status: str = Field(..., description="Trade status")
    fill_price: Optional[float] = Field(default=None, description="Actual fill price")
    fill_time: Optional[datetime] = Field(default=None, description="Fill timestamp")
    exit_time: Optional[datetime] = Field(default=None, description="Exit timestamp")
    exit_reason: Optional[str] = Field(default=None, description="Reason for exit")
    pnl: float = Field(default=0.0, description="Profit/loss in dollars")
    pnl_pct: float = Field(default=0.0, description="Profit/loss percentage")
    r_multiple: Optional[float] = Field(default=None, description="Risk-adjusted return")
    holding_time_seconds: Optional[int] = Field(default=None, description="Hold duration")
    session: Optional[str] = Field(default=None, description="Trading session")
    event_tier: Optional[str] = Field(default=None, description="Event tier (2X/4X/20X)")
    created_at: datetime = Field(..., description="Trade creation timestamp")


class SignalResponse(BaseModel):
    """Response model for a trading signal."""
    id: int = Field(..., description="Signal ID")
    ticker: str = Field(..., description="Ticker symbol")
    timestamp: datetime = Field(..., description="Signal timestamp")
    setup_type: str = Field(..., description="Setup/pattern type")
    score: float = Field(..., description="Signal score 0-100")
    tier: SignalTierEnum = Field(..., description="Signal quality tier")
    action: Optional[SignalActionEnum] = Field(default=None, description="Recommended action")
    horizon: Optional[TimeHorizonEnum] = Field(default=None, description="Time horizon")
    confidence: float = Field(default=0.0, description="Confidence 0-100")

    # Market data
    price: Optional[float] = Field(default=None, description="Current price")
    volume: Optional[int] = Field(default=None, description="Current volume")
    change_pct: Optional[float] = Field(default=None, description="Price change %")
    vwap: Optional[float] = Field(default=None, description="VWAP")
    range_pct: Optional[float] = Field(default=None, description="Day range %")

    # Options data
    atm_iv: Optional[float] = Field(default=None, description="ATM implied volatility")
    call_put_ratio: Optional[float] = Field(default=None, description="Call/put ratio")
    top_strike: Optional[float] = Field(default=None, description="Top traded strike")
    options_pressure_score: Optional[float] = Field(default=None, description="Options flow score")

    # Social data
    social_velocity: Optional[float] = Field(default=None, description="Social mention velocity")
    sentiment_score: Optional[float] = Field(default=None, description="Sentiment -1 to +1")

    # Context
    session_type: Optional[str] = Field(default=None, description="Session type")
    minutes_to_close: Optional[float] = Field(default=None, description="Minutes to market close")

    # Rich data
    risk: Optional[RiskFlags] = Field(default=None, description="Risk assessment")
    social: Optional[SocialMetrics] = Field(default=None, description="Social metrics")
    levels: Optional[PriceLevels] = Field(default=None, description="Key price levels")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    why: List[str] = Field(default_factory=list, description="Signal reasoning")
    summary: Optional[str] = Field(default=None, description="AI-generated summary")

    created_at: datetime = Field(..., description="Record creation timestamp")

    class Config:
        use_enum_values = True


class PnLSummary(BaseModel):
    """P&L summary for a time period."""
    period: str = Field(..., description="Period description (e.g., 'today', '2024-01-15')")
    total_trades: int = Field(default=0, description="Total number of trades")
    winning_trades: int = Field(default=0, description="Number of winning trades")
    losing_trades: int = Field(default=0, description="Number of losing trades")
    scratch_trades: int = Field(default=0, description="Number of scratch trades")
    win_rate: float = Field(default=0.0, description="Win rate as percentage")
    total_pnl: float = Field(default=0.0, description="Total P&L in dollars")
    avg_pnl: float = Field(default=0.0, description="Average P&L per trade")
    avg_win: float = Field(default=0.0, description="Average winning trade")
    avg_loss: float = Field(default=0.0, description="Average losing trade")
    largest_win: float = Field(default=0.0, description="Largest single win")
    largest_loss: float = Field(default=0.0, description="Largest single loss")
    total_r: float = Field(default=0.0, description="Total R-multiple")
    avg_r: float = Field(default=0.0, description="Average R-multiple")
    profit_factor: Optional[float] = Field(default=None, description="Gross profit / gross loss")
    sharpe_ratio: Optional[float] = Field(default=None, description="Risk-adjusted return")
    max_drawdown: Optional[float] = Field(default=None, description="Maximum drawdown")
    best_ticker: Optional[str] = Field(default=None, description="Best performing ticker")
    best_ticker_pnl: Optional[float] = Field(default=None, description="Best ticker P&L")
    worst_ticker: Optional[str] = Field(default=None, description="Worst performing ticker")
    worst_ticker_pnl: Optional[float] = Field(default=None, description="Worst ticker P&L")

    # Event tier breakdown
    tier_2x_count: int = Field(default=0, description="Number of 2X event trades")
    tier_4x_count: int = Field(default=0, description="Number of 4X event trades")
    tier_20x_count: int = Field(default=0, description="Number of 20X event trades")


class MarketData(BaseModel):
    """Market data for a ticker."""
    ticker: str = Field(..., description="Ticker symbol")
    price: float = Field(default=0.0, description="Current price")
    open: Optional[float] = Field(default=None, description="Open price")
    high: Optional[float] = Field(default=None, description="Day high")
    low: Optional[float] = Field(default=None, description="Day low")
    close: Optional[float] = Field(default=None, description="Previous close")
    volume: int = Field(default=0, description="Current volume")
    avg_volume: int = Field(default=0, description="Average daily volume")
    relative_volume: float = Field(default=0.0, description="Volume vs average")
    change: float = Field(default=0.0, description="Price change")
    change_pct: float = Field(default=0.0, description="Price change percentage")
    vwap: float = Field(default=0.0, description="Volume-weighted average price")
    spread: float = Field(default=0.0, description="Bid-ask spread")
    spread_pct: float = Field(default=0.0, description="Spread as percentage")
    volatility: float = Field(default=0.0, description="Current volatility")
    beta: Optional[float] = Field(default=None, description="Beta vs SPY")
    quote: Optional[MarketDataQuote] = Field(default=None, description="Real-time quote")

    # Options data
    iv: Optional[float] = Field(default=None, description="Implied volatility")
    iv_rank: Optional[float] = Field(default=None, description="IV rank 0-100")
    iv_percentile: Optional[float] = Field(default=None, description="IV percentile")
    put_call_ratio: Optional[float] = Field(default=None, description="Put/call ratio")

    # Technical levels
    support: Optional[float] = Field(default=None, description="Support level")
    resistance: Optional[float] = Field(default=None, description="Resistance level")
    pivot: Optional[float] = Field(default=None, description="Pivot point")

    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Data timestamp")


class RiskStatus(BaseModel):
    """Current risk status and limits."""
    # Daily limits
    daily_exposure_limit: float = Field(default=4000.0, description="Max daily exposure")
    daily_exposure_used: float = Field(default=0.0, description="Exposure used today")
    daily_exposure_remaining: float = Field(default=4000.0, description="Remaining exposure")
    daily_exposure_pct: float = Field(default=0.0, description="Exposure utilization %")

    # Position limits
    max_concurrent_positions: int = Field(default=3, description="Max concurrent positions")
    current_positions: int = Field(default=0, description="Current open positions")
    positions_available: int = Field(default=3, description="Available position slots")

    # Trade limits
    max_per_trade: float = Field(default=1000.0, description="Max per trade")
    daily_trade_count: int = Field(default=0, description="Trades today")

    # P&L-based risk
    daily_pnl: float = Field(default=0.0, description="Daily realized P&L")
    daily_pnl_limit: Optional[float] = Field(default=None, description="Daily loss limit")
    consecutive_losses: int = Field(default=0, description="Consecutive losing trades")
    max_consecutive_losses: int = Field(default=3, description="Max consecutive losses before pause")

    # Kill switches
    trading_paused: bool = Field(default=False, description="Trading currently paused")
    pause_reason: Optional[str] = Field(default=None, description="Reason for pause")
    kill_switch_active: bool = Field(default=False, description="Kill switch triggered")
    kill_switch_reason: Optional[str] = Field(default=None, description="Kill switch reason")

    # Market conditions
    vix_level: Optional[float] = Field(default=None, description="Current VIX level")
    vix_regime: Optional[str] = Field(default=None, description="VIX regime (low/normal/high)")
    volatility_factor: float = Field(default=1.0, description="Position size adjustment factor")

    # Flags
    risk_flags: Optional[RiskFlags] = Field(default=None, description="Active risk flags")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class EventResponse(BaseModel):
    """Response model for an event (economic, earnings, governance)."""
    id: int = Field(..., description="Event ID")
    event_type: str = Field(..., description="Type of event")
    ticker: Optional[str] = Field(default=None, description="Associated ticker")
    timestamp: datetime = Field(..., description="Event timestamp")

    # Event details
    title: Optional[str] = Field(default=None, description="Event title")
    description: Optional[str] = Field(default=None, description="Event description")
    tier: Optional[str] = Field(default=None, description="Event tier (2X/4X/20X)")
    impact: Optional[str] = Field(default=None, description="Expected impact level")

    # Governance-specific
    state_from: Optional[str] = Field(default=None, description="Previous state")
    state_to: Optional[str] = Field(default=None, description="New state")
    pnl_at_event: Optional[float] = Field(default=None, description="P&L at event time")
    reason: Optional[str] = Field(default=None, description="Event reason")
    dedupe_key: Optional[str] = Field(default=None, description="Deduplication key")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(..., description="Record creation timestamp")


class AccountInfo(BaseModel):
    """Account information and balances."""
    account_id: Optional[str] = Field(default=None, description="Account identifier")
    account_type: str = Field(default="paper", description="Account type (paper/live)")
    status: str = Field(default="active", description="Account status")

    # Balances
    equity: float = Field(default=0.0, description="Total equity")
    cash: float = Field(default=0.0, description="Cash balance")
    buying_power: float = Field(default=0.0, description="Available buying power")
    margin_used: float = Field(default=0.0, description="Margin in use")
    margin_available: float = Field(default=0.0, description="Available margin")

    # Day trading
    day_trade_count: int = Field(default=0, description="Day trades this week")
    pattern_day_trader: bool = Field(default=False, description="PDT flag")

    # Positions summary
    positions_count: int = Field(default=0, description="Number of open positions")
    positions_value: float = Field(default=0.0, description="Total positions value")

    # P&L
    unrealized_pnl: float = Field(default=0.0, description="Unrealized P&L")
    realized_pnl_today: float = Field(default=0.0, description="Realized P&L today")

    # Timestamps
    last_equity: Optional[float] = Field(default=None, description="Previous day equity")
    created_at: Optional[datetime] = Field(default=None, description="Account creation date")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class SessionStats(BaseModel):
    """Statistics for a trading session."""
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    session_date: str = Field(..., description="Session date (YYYY-MM-DD)")
    session_type: Optional[str] = Field(default=None, description="Session type")

    # Trade counts
    total_trades: int = Field(default=0, description="Total trades executed")
    winning_trades: int = Field(default=0, description="Number of winners")
    losing_trades: int = Field(default=0, description="Number of losers")
    scratch_trades: int = Field(default=0, description="Number of scratches")
    open_positions: int = Field(default=0, description="Currently open positions")

    # Performance
    win_rate: float = Field(default=0.0, description="Win rate percentage")
    total_pnl: float = Field(default=0.0, description="Total session P&L")
    daily_pnl: float = Field(default=0.0, description="Realized P&L today")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized P&L")
    avg_winner: float = Field(default=0.0, description="Average winning trade")
    avg_loser: float = Field(default=0.0, description="Average losing trade")

    # R-multiple stats
    total_r: float = Field(default=0.0, description="Total R-multiple")
    avg_r: float = Field(default=0.0, description="Average R-multiple")
    best_r: float = Field(default=0.0, description="Best R-multiple")
    worst_r: float = Field(default=0.0, description="Worst R-multiple")

    # Exposure
    max_exposure: float = Field(default=0.0, description="Maximum exposure reached")
    current_exposure: float = Field(default=0.0, description="Current exposure")
    exposure_limit: float = Field(default=4000.0, description="Exposure limit")

    # Signals
    signals_generated: int = Field(default=0, description="Signals generated")
    signals_traded: int = Field(default=0, description="Signals that led to trades")

    # By engine
    scalper_trades: int = Field(default=0, description="Scalper engine trades")
    scalper_pnl: float = Field(default=0.0, description="Scalper engine P&L")
    momentum_trades: int = Field(default=0, description="Momentum engine trades")
    momentum_pnl: float = Field(default=0.0, description="Momentum engine P&L")
    cpl_trades: int = Field(default=0, description="CPL engine trades")
    cpl_pnl: float = Field(default=0.0, description="CPL engine P&L")

    # Event breakdown
    tier_2x_trades: int = Field(default=0, description="2X event trades")
    tier_4x_trades: int = Field(default=0, description="4X event trades")
    tier_20x_trades: int = Field(default=0, description="20X event trades")

    # Best/worst
    best_trade_ticker: Optional[str] = Field(default=None, description="Best trade ticker")
    best_trade_pnl: float = Field(default=0.0, description="Best trade P&L")
    worst_trade_ticker: Optional[str] = Field(default=None, description="Worst trade ticker")
    worst_trade_pnl: float = Field(default=0.0, description="Worst trade P&L")

    # Timestamps
    session_start: Optional[datetime] = Field(default=None, description="Session start time")
    session_end: Optional[datetime] = Field(default=None, description="Session end time")
    last_trade_time: Optional[datetime] = Field(default=None, description="Last trade timestamp")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


# ============================================================================
# List Response Wrappers
# ============================================================================

class PositionListResponse(BaseModel):
    """List of positions with metadata."""
    positions: List[PositionResponse] = Field(default_factory=list)
    total: int = Field(default=0, description="Total number of positions")
    open_count: int = Field(default=0, description="Number of open positions")
    closed_count: int = Field(default=0, description="Number of closed positions")


class TradeListResponse(BaseModel):
    """List of trades with metadata."""
    trades: List[TradeResponse] = Field(default_factory=list)
    total: int = Field(default=0, description="Total number of trades")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=50, description="Items per page")
    has_more: bool = Field(default=False, description="More pages available")


class SignalListResponse(BaseModel):
    """List of signals with metadata."""
    signals: List[SignalResponse] = Field(default_factory=list)
    total: int = Field(default=0, description="Total number of signals")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=50, description="Items per page")
    has_more: bool = Field(default=False, description="More pages available")


class EventListResponse(BaseModel):
    """List of events with metadata."""
    events: List[EventResponse] = Field(default_factory=list)
    total: int = Field(default=0, description="Total number of events")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=50, description="Items per page")


# ============================================================================
# API Response Wrappers
# ============================================================================

class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = Field(default=True, description="Request success status")
    message: Optional[str] = Field(default=None, description="Response message")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy", description="Service status")
    version: Optional[str] = Field(default=None, description="API version")
    uptime_seconds: Optional[float] = Field(default=None, description="Uptime in seconds")
    database_connected: bool = Field(default=True, description="Database connection status")
    alpaca_connected: bool = Field(default=True, description="Alpaca connection status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
