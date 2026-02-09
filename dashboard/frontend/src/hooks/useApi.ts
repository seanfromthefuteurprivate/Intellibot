import { useState, useEffect, useCallback, useRef } from 'react';

// Base API URL - configure based on environment
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

// Types
export interface Position {
  symbol: string;
  qty: number;
  side: 'long' | 'short';
  entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  cost_basis: number;
  asset_id: string;
  exchange: string;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  qty: number;
  price: number;
  total: number;
  status: 'filled' | 'pending' | 'cancelled' | 'rejected';
  created_at: string;
  filled_at?: string;
  order_type: 'market' | 'limit' | 'stop' | 'stop_limit';
  strategy?: string;
}

export interface TradesOptions {
  page?: number;
  limit?: number;
  symbol?: string;
  status?: string;
  startDate?: string;
  endDate?: string;
}

export interface TradesResponse {
  trades: Trade[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
}

export interface PnLData {
  daily_pnl: number;
  daily_pnl_percent: number;
  weekly_pnl: number;
  weekly_pnl_percent: number;
  monthly_pnl: number;
  monthly_pnl_percent: number;
  total_pnl: number;
  total_pnl_percent: number;
  realized_pnl: number;
  unrealized_pnl: number;
  history: PnLHistoryItem[];
}

export interface PnLHistoryItem {
  date: string;
  pnl: number;
  cumulative_pnl: number;
  equity: number;
}

export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  high: number;
  low: number;
  open: number;
  prev_close: number;
  timestamp: string;
}

export interface MarketStatus {
  is_open: boolean;
  next_open: string;
  next_close: string;
  current_session: 'pre' | 'regular' | 'after' | 'closed';
}

export interface RiskStatus {
  overall_risk_score: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  daily_loss: number;
  daily_loss_limit: number;
  daily_loss_percent: number;
  max_drawdown: number;
  current_drawdown: number;
  position_concentration: number;
  margin_used: number;
  margin_available: number;
  buying_power: number;
  alerts: RiskAlert[];
}

export interface RiskAlert {
  id: string;
  type: 'warning' | 'critical';
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

export interface Account {
  id: string;
  account_number: string;
  status: 'active' | 'inactive' | 'restricted';
  currency: string;
  cash: number;
  portfolio_value: number;
  equity: number;
  last_equity: number;
  buying_power: number;
  daytrading_buying_power: number;
  pattern_day_trader: boolean;
  trading_blocked: boolean;
  transfers_blocked: boolean;
  account_blocked: boolean;
  created_at: string;
}

export interface SSEEvent {
  type: 'trade' | 'position' | 'order' | 'alert' | 'market' | 'system';
  data: unknown;
  timestamp: string;
}

// Generic API state
interface ApiState<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

// Generic fetch hook
function useApiCall<T>(
  endpoint: string,
  options?: {
    autoFetch?: boolean;
    refreshInterval?: number;
    params?: Record<string, string | number | undefined>;
  }
): ApiState<T> & { refetch: () => Promise<void> } {
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const { autoFetch = true, refreshInterval, params } = options || {};

  const fetchData = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));

    try {
      const url = new URL(`${API_BASE_URL}${endpoint}`);
      if (params) {
        Object.entries(params).forEach(([key, value]) => {
          if (value !== undefined) {
            url.searchParams.append(key, String(value));
          }
        });
      }

      const response = await fetch(url.toString(), {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      setState({ data, loading: false, error: null });
    } catch (error) {
      setState({
        data: null,
        loading: false,
        error: error instanceof Error ? error : new Error(String(error)),
      });
    }
  }, [endpoint, params]);

  useEffect(() => {
    if (autoFetch) {
      fetchData();
    }
  }, [autoFetch, fetchData]);

  useEffect(() => {
    if (refreshInterval && refreshInterval > 0) {
      const intervalId = setInterval(fetchData, refreshInterval);
      return () => clearInterval(intervalId);
    }
  }, [refreshInterval, fetchData]);

  return { ...state, refetch: fetchData };
}

// Hook: usePositions - Fetch and auto-refresh positions
export function usePositions(refreshInterval: number = 5000) {
  return useApiCall<Position[]>('/positions', {
    autoFetch: true,
    refreshInterval,
  });
}

// Hook: useTrades - Fetch trades with pagination
export function useTrades(options: TradesOptions = {}) {
  const { page = 1, limit = 20, symbol, status, startDate, endDate } = options;

  const params: Record<string, string | number | undefined> = {
    page,
    limit,
    symbol,
    status,
    start_date: startDate,
    end_date: endDate,
  };

  const result = useApiCall<TradesResponse>('/trades', {
    autoFetch: true,
    params,
  });

  return {
    ...result,
    trades: result.data?.trades || [],
    total: result.data?.total || 0,
    hasMore: result.data?.hasMore || false,
    currentPage: result.data?.page || page,
  };
}

// Hook: usePnL - Fetch P&L data
export function usePnL(refreshInterval: number = 30000) {
  return useApiCall<PnLData>('/pnl', {
    autoFetch: true,
    refreshInterval,
  });
}

// Hook: useMarket - Fetch market data
export function useMarket(symbols?: string[], refreshInterval: number = 10000) {
  const [state, setState] = useState<ApiState<{ quotes: MarketData[]; status: MarketStatus }>>({
    data: null,
    loading: false,
    error: null,
  });

  const fetchData = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));

    try {
      const [quotesResponse, statusResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/market/quotes${symbols ? `?symbols=${symbols.join(',')}` : ''}`, {
          credentials: 'include',
        }),
        fetch(`${API_BASE_URL}/market/status`, {
          credentials: 'include',
        }),
      ]);

      if (!quotesResponse.ok || !statusResponse.ok) {
        throw new Error('Failed to fetch market data');
      }

      const [quotes, status] = await Promise.all([
        quotesResponse.json(),
        statusResponse.json(),
      ]);

      setState({
        data: { quotes, status },
        loading: false,
        error: null,
      });
    } catch (error) {
      setState({
        data: null,
        loading: false,
        error: error instanceof Error ? error : new Error(String(error)),
      });
    }
  }, [symbols]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  useEffect(() => {
    if (refreshInterval > 0) {
      const intervalId = setInterval(fetchData, refreshInterval);
      return () => clearInterval(intervalId);
    }
  }, [refreshInterval, fetchData]);

  return {
    ...state,
    quotes: state.data?.quotes || [],
    marketStatus: state.data?.status || null,
    refetch: fetchData,
  };
}

// Hook: useRisk - Fetch risk status
export function useRisk(refreshInterval: number = 10000) {
  return useApiCall<RiskStatus>('/risk', {
    autoFetch: true,
    refreshInterval,
  });
}

// Hook: useEvents - SSE connection for real-time events
export function useEvents(eventTypes?: SSEEvent['type'][]) {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    try {
      const url = new URL(`${API_BASE_URL}/events/stream`);
      if (eventTypes && eventTypes.length > 0) {
        url.searchParams.append('types', eventTypes.join(','));
      }

      const eventSource = new EventSource(url.toString(), {
        withCredentials: true,
      });

      eventSource.onopen = () => {
        setConnected(true);
        setError(null);
        reconnectAttempts.current = 0;
      };

      eventSource.onmessage = (event) => {
        try {
          const parsedEvent: SSEEvent = JSON.parse(event.data);
          setEvents((prev) => [...prev.slice(-99), parsedEvent]);
        } catch (e) {
          console.error('Failed to parse SSE event:', e);
        }
      };

      eventSource.onerror = () => {
        setConnected(false);
        eventSource.close();

        if (reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
          reconnectAttempts.current += 1;

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        } else {
          setError(new Error('Failed to connect to event stream after multiple attempts'));
        }
      };

      eventSourceRef.current = eventSource;
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)));
    }
  }, [eventTypes]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setConnected(false);
  }, []);

  const clearEvents = useCallback(() => {
    setEvents([]);
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    events,
    latestEvent: events[events.length - 1] || null,
    connected,
    error,
    connect,
    disconnect,
    clearEvents,
  };
}

// Hook: useAccount - Fetch account info
export function useAccount(refreshInterval: number = 60000) {
  return useApiCall<Account>('/account', {
    autoFetch: true,
    refreshInterval,
  });
}

// Utility: Manual API call function for mutations
export async function apiCall<T>(
  endpoint: string,
  options: {
    method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
    body?: unknown;
    headers?: Record<string, string>;
  } = {}
): Promise<T> {
  const { method = 'GET', body, headers = {} } = options;

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    credentials: 'include',
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.message || `API error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

export default {
  usePositions,
  useTrades,
  usePnL,
  useMarket,
  useRisk,
  useEvents,
  useAccount,
  apiCall,
};
