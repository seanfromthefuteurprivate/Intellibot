import React, { useState, useEffect, useCallback } from 'react';

// Types
interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percent: number;
}

interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  pnl?: number;
  timestamp: string;
}

interface MarketData {
  vix: number;
  regime: 'low_volatility' | 'normal' | 'high_volatility' | 'extreme';
  spy_price: number;
  trend: 'bullish' | 'bearish' | 'neutral';
}

interface LiveEvent {
  id: string;
  type: 'trade' | 'signal' | 'alert' | 'system';
  message: string;
  timestamp: string;
  severity: 'info' | 'warning' | 'success' | 'error';
}

interface DashboardStats {
  todayPnl: number;
  todayPnlPercent: number;
  openPositions: number;
  winRate: number;
  totalTrades: number;
  winningTrades: number;
}

// Progress Ring Component
const ProgressRing: React.FC<{ progress: number; size?: number; strokeWidth?: number }> = ({
  progress,
  size = 120,
  strokeWidth = 8,
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  const getColor = (value: number): string => {
    if (value >= 60) return '#10b981';
    if (value >= 45) return '#f59e0b';
    return '#ef4444';
  };

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#374151"
          strokeWidth={strokeWidth}
          fill="none"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={getColor(progress)}
          strokeWidth={strokeWidth}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-500 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-2xl font-bold text-white">{progress.toFixed(1)}%</span>
      </div>
    </div>
  );
};

// Card Component
const Card: React.FC<{
  title: string;
  children: React.ReactNode;
  className?: string;
}> = ({ title, children, className = '' }) => (
  <div className={`bg-gray-800 rounded-xl p-6 border border-gray-700 ${className}`}>
    <h3 className="text-gray-400 text-sm font-medium uppercase tracking-wide mb-4">{title}</h3>
    {children}
  </div>
);

// P&L Card Component
const PnLCard: React.FC<{ pnl: number; pnlPercent: number }> = ({ pnl, pnlPercent }) => {
  const isPositive = pnl >= 0;
  const colorClass = isPositive ? 'text-green-400' : 'text-red-400';
  const bgGlow = isPositive ? 'shadow-green-500/20' : 'shadow-red-500/20';

  return (
    <Card title="Today's P&L" className={`shadow-lg ${bgGlow}`}>
      <div className="flex flex-col items-center justify-center py-4">
        <span className={`text-5xl font-bold ${colorClass} tracking-tight`}>
          {isPositive ? '+' : ''}${pnl.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </span>
        <span className={`text-xl mt-2 ${colorClass}`}>
          {isPositive ? '+' : ''}{pnlPercent.toFixed(2)}%
        </span>
      </div>
    </Card>
  );
};

// Open Positions Card Component
const OpenPositionsCard: React.FC<{ count: number; positions: Position[] }> = ({ count, positions }) => (
  <Card title="Open Positions">
    <div className="flex items-center justify-between">
      <span className="text-4xl font-bold text-white">{count}</span>
      <div className="text-right">
        {positions.slice(0, 3).map((pos, idx) => (
          <div key={idx} className="text-sm text-gray-400">
            {pos.symbol}: <span className={pos.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
              {pos.pnl >= 0 ? '+' : ''}{pos.pnl_percent.toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  </Card>
);

// Win Rate Card Component
const WinRateCard: React.FC<{ winRate: number; winningTrades: number; totalTrades: number }> = ({
  winRate,
  winningTrades,
  totalTrades,
}) => (
  <Card title="Win Rate">
    <div className="flex flex-col items-center">
      <ProgressRing progress={winRate} />
      <p className="text-gray-400 text-sm mt-3">
        {winningTrades} / {totalTrades} trades
      </p>
    </div>
  </Card>
);

// Market Regime Card Component
const MarketRegimeCard: React.FC<{ regime: MarketData['regime']; trend: MarketData['trend'] }> = ({
  regime,
  trend,
}) => {
  const regimeConfig: Record<MarketData['regime'], { label: string; color: string; bgColor: string }> = {
    low_volatility: { label: 'Low Volatility', color: 'text-blue-400', bgColor: 'bg-blue-500/20' },
    normal: { label: 'Normal', color: 'text-green-400', bgColor: 'bg-green-500/20' },
    high_volatility: { label: 'High Volatility', color: 'text-yellow-400', bgColor: 'bg-yellow-500/20' },
    extreme: { label: 'Extreme', color: 'text-red-400', bgColor: 'bg-red-500/20' },
  };

  const trendConfig: Record<MarketData['trend'], { icon: string; color: string }> = {
    bullish: { icon: '\u2191', color: 'text-green-400' },
    bearish: { icon: '\u2193', color: 'text-red-400' },
    neutral: { icon: '\u2194', color: 'text-gray-400' },
  };

  const config = regimeConfig[regime];
  const trendInfo = trendConfig[trend];

  return (
    <Card title="Market Regime">
      <div className="flex flex-col items-center">
        <div className={`px-4 py-2 rounded-full ${config.bgColor} ${config.color} font-semibold text-lg`}>
          {config.label}
        </div>
        <div className={`mt-3 flex items-center gap-2 ${trendInfo.color}`}>
          <span className="text-2xl">{trendInfo.icon}</span>
          <span className="capitalize">{trend}</span>
        </div>
      </div>
    </Card>
  );
};

// VIX Card Component
const VIXCard: React.FC<{ vix: number }> = ({ vix }) => {
  const getVIXColor = (value: number): string => {
    if (value < 15) return 'text-green-400';
    if (value < 20) return 'text-blue-400';
    if (value < 25) return 'text-yellow-400';
    if (value < 30) return 'text-orange-400';
    return 'text-red-400';
  };

  const getVIXLabel = (value: number): string => {
    if (value < 15) return 'Very Low';
    if (value < 20) return 'Low';
    if (value < 25) return 'Moderate';
    if (value < 30) return 'High';
    return 'Extreme';
  };

  return (
    <Card title="VIX Level">
      <div className="flex flex-col items-center">
        <span className={`text-4xl font-bold ${getVIXColor(vix)}`}>{vix.toFixed(2)}</span>
        <span className={`text-sm mt-2 ${getVIXColor(vix)}`}>{getVIXLabel(vix)}</span>
        <div className="w-full mt-4 h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${
              vix < 20 ? 'bg-green-500' : vix < 30 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${Math.min(vix / 50 * 100, 100)}%` }}
          />
        </div>
      </div>
    </Card>
  );
};

// Recent Trades Component
const RecentTradesCard: React.FC<{ trades: Trade[] }> = ({ trades }) => (
  <Card title="Recent Trades" className="col-span-2">
    <div className="space-y-3">
      {trades.length === 0 ? (
        <p className="text-gray-500 text-center py-4">No recent trades</p>
      ) : (
        trades.map((trade) => (
          <div
            key={trade.id}
            className="flex items-center justify-between p-3 bg-gray-900 rounded-lg border border-gray-700"
          >
            <div className="flex items-center gap-3">
              <span
                className={`px-2 py-1 rounded text-xs font-semibold uppercase ${
                  trade.side === 'buy' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                }`}
              >
                {trade.side}
              </span>
              <span className="text-white font-medium">{trade.symbol}</span>
              <span className="text-gray-400 text-sm">{trade.quantity} shares</span>
            </div>
            <div className="text-right">
              <div className="text-white">${trade.price.toFixed(2)}</div>
              {trade.pnl !== undefined && (
                <div className={trade.pnl >= 0 ? 'text-green-400 text-sm' : 'text-red-400 text-sm'}>
                  {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                </div>
              )}
            </div>
            <div className="text-gray-500 text-xs">
              {new Date(trade.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ))
      )}
    </div>
  </Card>
);

// Live Events Feed Component
const LiveEventsFeed: React.FC<{ events: LiveEvent[] }> = ({ events }) => {
  const severityConfig: Record<LiveEvent['severity'], { color: string; icon: string }> = {
    info: { color: 'text-blue-400', icon: '\u2139' },
    warning: { color: 'text-yellow-400', icon: '\u26a0' },
    success: { color: 'text-green-400', icon: '\u2713' },
    error: { color: 'text-red-400', icon: '\u2717' },
  };

  const typeConfig: Record<LiveEvent['type'], { bgColor: string }> = {
    trade: { bgColor: 'bg-purple-500/20' },
    signal: { bgColor: 'bg-blue-500/20' },
    alert: { bgColor: 'bg-orange-500/20' },
    system: { bgColor: 'bg-gray-500/20' },
  };

  return (
    <Card title="Live Events" className="col-span-2">
      <div className="space-y-2 max-h-80 overflow-y-auto">
        {events.length === 0 ? (
          <p className="text-gray-500 text-center py-4">No recent events</p>
        ) : (
          events.map((event) => {
            const severity = severityConfig[event.severity];
            const type = typeConfig[event.type];
            return (
              <div
                key={event.id}
                className={`flex items-start gap-3 p-3 rounded-lg ${type.bgColor} border border-gray-700`}
              >
                <span className={`text-lg ${severity.color}`}>{severity.icon}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-white text-sm">{event.message}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs text-gray-500 uppercase">{event.type}</span>
                    <span className="text-xs text-gray-600">|</span>
                    <span className="text-xs text-gray-500">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </Card>
  );
};

// Main Dashboard Component
const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats>({
    todayPnl: 0,
    todayPnlPercent: 0,
    openPositions: 0,
    winRate: 0,
    totalTrades: 0,
    winningTrades: 0,
  });
  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [marketData, setMarketData] = useState<MarketData>({
    vix: 18.5,
    regime: 'normal',
    spy_price: 450.0,
    trend: 'neutral',
  });
  const [events, setEvents] = useState<LiveEvent[]>([]);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Fetch dashboard data
  const fetchDashboardData = useCallback(async () => {
    try {
      // Fetch stats
      const statsRes = await fetch('/api/dashboard/stats');
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      }

      // Fetch positions
      const positionsRes = await fetch('/api/positions');
      if (positionsRes.ok) {
        const positionsData = await positionsRes.json();
        setPositions(positionsData);
      }

      // Fetch recent trades
      const tradesRes = await fetch('/api/trades/recent?limit=5');
      if (tradesRes.ok) {
        const tradesData = await tradesRes.json();
        setTrades(tradesData);
      }

      // Fetch market data
      const marketRes = await fetch('/api/market/data');
      if (marketRes.ok) {
        const marketDataResult = await marketData.json();
        setMarketData(marketDataResult);
      }

      setLastUpdate(new Date());
      setIsConnected(true);
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error);
      setIsConnected(false);
    }
  }, []);

  // WebSocket connection for live events
  useEffect(() => {
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/events`;
    let ws: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout;

    const connect = () => {
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'event') {
            setEvents((prev) => [data.payload, ...prev].slice(0, 50));
          } else if (data.type === 'stats_update') {
            setStats((prev) => ({ ...prev, ...data.payload }));
          } else if (data.type === 'market_update') {
            setMarketData((prev) => ({ ...prev, ...data.payload }));
          }
          setLastUpdate(new Date());
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        reconnectTimeout = setTimeout(connect, 5000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        ws?.close();
      };
    };

    connect();

    return () => {
      ws?.close();
      clearTimeout(reconnectTimeout);
    };
  }, []);

  // Initial data fetch and polling
  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 30000); // Poll every 30 seconds
    return () => clearInterval(interval);
  }, [fetchDashboardData]);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white">Trading Dashboard</h1>
            <p className="text-gray-400 mt-1">WSB Snake - Real-time Trading Monitor</p>
          </div>
          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2 ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
              <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'} animate-pulse`} />
              <span className="text-sm">{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            <span className="text-gray-500 text-sm">
              Last update: {lastUpdate.toLocaleTimeString()}
            </span>
          </div>
        </div>
      </header>

      {/* Dashboard Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Row 1: Key Metrics */}
        <PnLCard pnl={stats.todayPnl} pnlPercent={stats.todayPnlPercent} />
        <OpenPositionsCard count={stats.openPositions} positions={positions} />
        <WinRateCard
          winRate={stats.winRate}
          winningTrades={stats.winningTrades}
          totalTrades={stats.totalTrades}
        />
        <VIXCard vix={marketData.vix} />

        {/* Row 2: Market Regime spans 2 cols, Recent Trades spans 2 cols */}
        <div className="lg:col-span-2">
          <MarketRegimeCard regime={marketData.regime} trend={marketData.trend} />
        </div>
        <RecentTradesCard trades={trades} />

        {/* Row 3: Live Events Feed */}
        <div className="lg:col-span-4">
          <LiveEventsFeed events={events} />
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-8 text-center text-gray-600 text-sm">
        <p>WSB Snake Trading System | Built with React + TypeScript</p>
      </footer>
    </div>
  );
};

export default Dashboard;
