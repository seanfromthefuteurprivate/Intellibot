/**
 * TypeScript type definitions for Intellibot Dashboard
 *
 * These types match the backend Pydantic models from the FastAPI routes.
 */

// =============================================================================
// Position Types
// =============================================================================

/**
 * Position status enum matching backend PositionStatus.
 */
export type PositionStatus = 'PENDING' | 'OPEN' | 'CLOSED' | 'CANCELLED' | 'EXPIRED';

/**
 * Trading position from AlpacaExecutor.
 * Matches: dashboard/api/routes/positions.py -> PositionResponse
 */
export interface Position {
  symbol: string;
  option_symbol: string;
  qty: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_pct: number;
  entry_time: string | null;
  status: PositionStatus | string;
  engine: string;
}

/**
 * List of positions response.
 * Matches: dashboard/api/routes/positions.py -> PositionsListResponse
 */
export interface PositionsListResponse {
  positions: Position[];
  count: number;
}

// =============================================================================
// Trade Types
// =============================================================================

/**
 * Individual trade record from trade_performance table.
 * Matches: dashboard/api/routes/trades.py -> TradeRecord
 */
export interface Trade {
  id: number;
  trade_date: string;
  symbol: string;
  engine: string | null;
  trade_type: string | null;
  entry_hour: number | null;
  session: string | null;
  pnl: number | null;
  pnl_pct: number | null;
  r_multiple: number | null;
  exit_reason: string | null;
  holding_time_seconds: number | null;
  signal_id: number | null;
  event_tier: string | null;
  created_at: string | null;
}

/**
 * Paginated trade list response.
 * Matches: dashboard/api/routes/trades.py -> TradeListResponse
 */
export interface TradeListResponse {
  trades: Trade[];
  total: number;
  limit: number;
  offset: number;
}

/**
 * Aggregate trade statistics.
 * Matches: dashboard/api/routes/trades.py -> TradeStatsResponse
 */
export interface TradeStats {
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  avg_r_multiple: number | null;
  best_trade_pnl: number | null;
  worst_trade_pnl: number | null;
}

// =============================================================================
// Signal Types
// =============================================================================

/**
 * Signal tier levels.
 */
export type SignalTier = 'S' | 'A' | 'B' | 'C' | 'D' | 'F';

/**
 * Base signal model with common fields.
 * Matches: dashboard/api/routes/signals.py -> SignalBase
 */
export interface SignalBase {
  id: number;
  ticker: string;
  timestamp: string;
  setup_type: string;
  score: number;
  tier: SignalTier | string;
  price: number | null;
  change_pct: number | null;
  created_at: string | null;
}

/**
 * Signal summary for list views.
 * Matches: dashboard/api/routes/signals.py -> SignalSummary
 */
export interface Signal extends SignalBase {
  volume: number | null;
  social_velocity: number | null;
  sentiment_score: number | null;
  session_type: string | null;
}

/**
 * Full signal details with all features.
 * Matches: dashboard/api/routes/signals.py -> SignalDetail
 */
export interface SignalDetail extends SignalBase {
  // Market features
  volume: number | null;
  vwap: number | null;
  range_pct: number | null;

  // Options features
  atm_iv: number | null;
  call_put_ratio: number | null;
  top_strike: number | null;
  options_pressure_score: number | null;

  // Social features
  social_velocity: number | null;
  sentiment_score: number | null;

  // Session context
  session_type: string | null;
  minutes_to_close: number | null;

  // Full feature and evidence data
  features: Record<string, unknown> | null;
  evidence: Record<string, unknown> | null;
}

/**
 * Paginated signals response.
 * Matches: dashboard/api/routes/signals.py -> PaginatedSignals
 */
export interface PaginatedSignals {
  signals: Signal[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

/**
 * Active signals response.
 * Matches: dashboard/api/routes/signals.py -> ActiveSignalsResponse
 */
export interface ActiveSignalsResponse {
  signals: Signal[];
  count: number;
  as_of: string;
}

// =============================================================================
// Event Types
// =============================================================================

/**
 * Event severity levels.
 * Matches: dashboard/api/websocket.py -> EventSeverity
 */
export type EventSeverity = 'info' | 'warning' | 'error' | 'critical';

/**
 * System event from WebSocket or telemetry.
 */
export interface Event {
  event_type: string;
  description: string;
  severity: EventSeverity;
  details: Record<string, unknown>;
  timestamp: string;
}

/**
 * Telemetry event types from governance system.
 */
export type TelemetryEventType =
  | 'STATE_TRANSITION'
  | 'RUNNER_LOCK_ENTERED'
  | 'RUNNER_LOCK_HEARTBEAT'
  | 'STRUCTURE_BREAK_DETECTED'
  | 'TP_CHECKPOINT'
  | 'TP_SUPPRESSED'
  | 'PREREGISTRATION_LOCKED'
  | 'POSITION_OPENED'
  | 'POSITION_CLOSED'
  | 'BUY'
  | 'SELL'
  | 'RELEASE';

/**
 * Governance event from telemetry bus.
 */
export interface GovernanceEvent {
  event_type: TelemetryEventType | string;
  note: string;
  dedupe_key: string | null;
  pnl_pct: number | null;
  entry_ref_price: number | null;
  current_ref_price: number | null;
  peak_ref_price: number | null;
  exit_ref_price: number | null;
  expansion_pct: number | null;
  current_state: string | null;
  exit_permission: string | null;
  reason: string | null;
  timestamp: string;
}

// =============================================================================
// P&L Types
// =============================================================================

/**
 * P&L for a single position.
 * Matches: dashboard/api/routes/pnl.py -> PositionPnL
 */
export interface PositionPnL {
  symbol: string;
  option_symbol: string;
  qty: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  status: string;
}

/**
 * Today's P&L summary.
 * Matches: dashboard/api/routes/pnl.py -> TodayPnL
 */
export interface TodayPnL {
  date: string;
  realized_pnl: number;
  unrealized_pnl: number;
  total_pnl: number;
  trades_today: number;
  wins: number;
  losses: number;
  win_rate: number;
  open_positions: PositionPnL[];
}

/**
 * P&L for a single day.
 * Matches: dashboard/api/routes/pnl.py -> DailyPnL
 */
export interface DailyPnL {
  date: string;
  pnl: number;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_r_multiple: number | null;
}

/**
 * P&L breakdown by trading engine.
 * Matches: dashboard/api/routes/pnl.py -> EnginePnL
 */
export interface EnginePnL {
  engine: string;
  pnl: number;
  trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_r_multiple: number | null;
}

/**
 * Total P&L summary with breakdown.
 * Matches: dashboard/api/routes/pnl.py -> PnLSummary
 */
export interface PnLSummary {
  total_pnl: number;
  total_trades: number;
  total_wins: number;
  total_losses: number;
  overall_win_rate: number;
  avg_r_multiple: number | null;
  best_day: DailyPnL | null;
  worst_day: DailyPnL | null;
  by_engine: EnginePnL[];
  by_symbol: Record<string, number>;
}

// =============================================================================
// Market Data Types
// =============================================================================

/**
 * Real-time market data for a symbol.
 */
export interface MarketData {
  symbol: string;
  price: number;
  bid: number;
  ask: number;
  spread: number;
  volume: number;
  change: number;
  change_pct: number;
  high: number;
  low: number;
  open: number;
  close: number;
  vwap: number | null;
  timestamp: string;
}

/**
 * Options quote data.
 */
export interface OptionQuote {
  symbol: string;
  underlying: string;
  strike: number;
  expiration: string;
  option_type: 'call' | 'put';
  bid: number;
  ask: number;
  mid: number;
  last: number;
  volume: number;
  open_interest: number;
  iv: number | null;
  delta: number | null;
  gamma: number | null;
  theta: number | null;
  vega: number | null;
  timestamp: string;
}

// =============================================================================
// Risk Types
// =============================================================================

/**
 * Kill switch and cooldown status.
 * Matches: dashboard/api/routes/risk.py -> KillSwitchStatus
 */
export interface KillSwitchStatus {
  kill_switch_active: boolean;
  in_cooldown: boolean;
  cooldown_reason: string;
  cooldown_until: string | null;
  consecutive_losses: number;
}

/**
 * Current risk limits configuration.
 * Matches: dashboard/api/routes/risk.py -> RiskLimits
 */
export interface RiskLimits {
  max_daily_loss: number;
  max_concurrent_positions_global: number;
  max_daily_exposure_global: number;
  max_positions_scalper: number;
  max_positions_momentum: number;
  max_positions_macro: number;
  max_positions_vol_sell: number;
  max_exposure_per_ticker: number;
  max_exposure_per_sector: number;
  max_premium_scalper: number;
  max_premium_momentum: number;
  max_premium_macro: number;
  max_premium_vol_sell: number;
  max_pct_buying_power_per_trade: number;
  consecutive_loss_threshold: number;
  cooldown_hours: number;
}

/**
 * Historical win rate statistics.
 * Matches: dashboard/api/routes/risk.py -> WinRateStats
 */
export interface WinRateStats {
  win_rate: number;
  win_count: number;
  loss_count: number;
  total_trades: number;
}

/**
 * Combined risk status for dashboard display.
 */
export interface RiskStatus {
  kill_switch: KillSwitchStatus;
  limits: RiskLimits;
  win_rate_stats: WinRateStats;
  current_exposure: number;
  exposure_pct: number;
  positions_open: number;
  daily_pnl: number;
  at_risk: boolean;
}

// =============================================================================
// Account Types
// =============================================================================

/**
 * Alpaca account information.
 * Matches: dashboard/api/routes/account.py -> AccountResponse
 */
export interface AccountInfo {
  buying_power: number;
  equity: number;
  cash: number;
  portfolio_value: number;
  currency: string;
  account_number: string;
  status: string;
  trading_blocked: boolean;
  transfers_blocked: boolean;
  pattern_day_trader: boolean;
  daytrade_count: number;
  last_equity: number;
  multiplier: string;
}

/**
 * Current trading session statistics.
 * Matches: dashboard/api/routes/account.py -> SessionStatsResponse
 */
export interface SessionStats {
  total_trades: number;
  winning_trades: number;
  win_rate: number;
  total_pnl: number;
  daily_pnl: number;
  open_positions: number;
  daily_exposure_used: number;
  max_daily_exposure: number;
  daily_trade_count: number;
  max_concurrent_positions: number;
}

// =============================================================================
// API Response Wrapper
// =============================================================================

/**
 * Generic API response wrapper for consistent error handling.
 */
export interface APIResponse<T> {
  data: T | null;
  success: boolean;
  error: string | null;
  message: string | null;
  status_code: number;
  timestamp: string;
}

/**
 * Paginated API response wrapper.
 */
export interface PaginatedAPIResponse<T> extends APIResponse<T[]> {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

/**
 * API error response.
 */
export interface APIError {
  detail: string;
  status_code: number;
  error_type: string | null;
  timestamp: string;
}

// =============================================================================
// WebSocket Message Types
// =============================================================================

/**
 * WebSocket message types.
 * Matches: dashboard/api/websocket.py -> MessageType
 */
export type WebSocketMessageType =
  | 'position_update'
  | 'trade_executed'
  | 'event'
  | 'pnl_update'
  | 'heartbeat'
  | 'error'
  | 'connected'
  | 'subscription';

/**
 * Base WebSocket message structure.
 * Matches: dashboard/api/websocket.py -> WebSocketMessage
 */
export interface WebSocketMessage<T = unknown> {
  type: WebSocketMessageType;
  data: T;
  timestamp: string;
}

/**
 * Position update WebSocket message data.
 */
export interface PositionUpdateData {
  positions: Position[];
}

/**
 * Trade executed WebSocket message data.
 */
export interface TradeExecutedData {
  trade: Trade;
}

/**
 * Event WebSocket message data.
 */
export interface EventData {
  event_type: string;
  description: string;
  severity: EventSeverity;
  details: Record<string, unknown>;
}

/**
 * P&L update WebSocket message data.
 */
export interface PnLUpdateData {
  total_pnl: number;
  daily_pnl: number;
  unrealized_pnl: number;
  realized_pnl: number;
  positions_pnl: Record<string, number>;
}

/**
 * Heartbeat WebSocket message data.
 */
export interface HeartbeatData {
  server_time: string;
  connected_clients: number;
  pong?: boolean;
}

/**
 * Connection confirmation WebSocket message data.
 */
export interface ConnectedData {
  client_id: string;
  subscriptions: string[];
  message: string;
}

/**
 * Subscription update WebSocket message data.
 */
export interface SubscriptionData {
  action: 'add' | 'remove';
  subscriptions: string[];
  current_subscriptions: string[];
}

/**
 * Error WebSocket message data.
 */
export interface ErrorData {
  error: string;
  code?: string;
  details?: Record<string, unknown>;
}

/**
 * Typed WebSocket message variants.
 */
export type TypedWebSocketMessage =
  | WebSocketMessage<PositionUpdateData>
  | WebSocketMessage<TradeExecutedData>
  | WebSocketMessage<EventData>
  | WebSocketMessage<PnLUpdateData>
  | WebSocketMessage<HeartbeatData>
  | WebSocketMessage<ConnectedData>
  | WebSocketMessage<SubscriptionData>
  | WebSocketMessage<ErrorData>;

/**
 * WebSocket client subscription request.
 */
export interface WebSocketSubscribeRequest {
  action: 'subscribe' | 'unsubscribe' | 'ping';
  types?: WebSocketMessageType[];
}

/**
 * WebSocket connection state.
 */
export type WebSocketConnectionState =
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'reconnecting'
  | 'error';

/**
 * WebSocket client info.
 */
export interface WebSocketClientInfo {
  client_id: string;
  connected_at: string;
  subscriptions: string[];
  last_heartbeat: string;
}

// =============================================================================
// Utility Types
// =============================================================================

/**
 * Nullable type helper.
 */
export type Nullable<T> = T | null;

/**
 * Optional type helper.
 */
export type Optional<T> = T | undefined;

/**
 * Deep partial type helper.
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Date string in ISO format.
 */
export type ISODateString = string;

/**
 * Timestamp in ISO format with timezone.
 */
export type ISOTimestamp = string;

/**
 * Currency amount (in dollars).
 */
export type CurrencyAmount = number;

/**
 * Percentage value (0-100 or 0-1 depending on context).
 */
export type Percentage = number;

/**
 * Ticker symbol.
 */
export type TickerSymbol = string;

/**
 * Option symbol in OCC format.
 */
export type OptionSymbol = string;

/**
 * Engine name type.
 */
export type EngineName =
  | 'scalper'
  | 'momentum'
  | 'macro'
  | 'vol_sell'
  | 'CPL'
  | 'institutional_scalper'
  | 'precious_metals'
  | string;

/**
 * Trading session type.
 */
export type SessionType =
  | 'premarket'
  | 'open'
  | 'morning'
  | 'midday'
  | 'afternoon'
  | 'power_hour'
  | 'close'
  | 'afterhours'
  | string;

/**
 * Option type (call or put).
 */
export type OptionType = 'call' | 'put';

/**
 * Trade side (buy or sell).
 */
export type TradeSide = 'buy' | 'sell';

/**
 * Order status.
 */
export type OrderStatus =
  | 'new'
  | 'partially_filled'
  | 'filled'
  | 'done_for_day'
  | 'canceled'
  | 'expired'
  | 'replaced'
  | 'pending_cancel'
  | 'pending_replace'
  | 'accepted'
  | 'pending_new'
  | 'accepted_for_bidding'
  | 'stopped'
  | 'rejected'
  | 'suspended'
  | 'calculated';
