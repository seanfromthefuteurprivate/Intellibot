import React, { useState, useEffect, useCallback, createContext, useContext } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink, useLocation } from 'react-router-dom';

// Types
interface AccountData {
  balance: number;
  sessionPnL: number;
  sessionPnLPercent: number;
  buyingPower: number;
  dayTradesRemaining: number;
}

interface Position {
  symbol: string;
  qty: number;
  avgEntryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
}

interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  qty: number;
  price: number;
  timestamp: string;
  pnl?: number;
}

interface Signal {
  id: string;
  symbol: string;
  type: 'call' | 'put';
  conviction: number;
  source: string;
  timestamp: string;
}

interface RiskMetrics {
  totalExposure: number;
  maxDrawdown: number;
  winRate: number;
  sharpeRatio: number;
}

interface WebSocketMessage {
  type: 'account' | 'positions' | 'trade' | 'signal' | 'risk';
  data: AccountData | Position[] | Trade | Signal | RiskMetrics;
}

// WebSocket Context
interface WebSocketContextType {
  connected: boolean;
  accountData: AccountData | null;
  positions: Position[];
  trades: Trade[];
  signals: Signal[];
  riskMetrics: RiskMetrics | null;
}

const WebSocketContext = createContext<WebSocketContextType>({
  connected: false,
  accountData: null,
  positions: [],
  trades: [],
  signals: [],
  riskMetrics: null,
});

export const useWebSocket = () => useContext(WebSocketContext);

// WebSocket Provider Component
const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [connected, setConnected] = useState(false);
  const [accountData, setAccountData] = useState<AccountData | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);

  const connectWebSocket = useCallback(() => {
    const wsUrl = `ws://${window.location.hostname}:8080/ws`;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);

        switch (message.type) {
          case 'account':
            setAccountData(message.data as AccountData);
            break;
          case 'positions':
            setPositions(message.data as Position[]);
            break;
          case 'trade':
            setTrades((prev) => [message.data as Trade, ...prev].slice(0, 100));
            break;
          case 'signal':
            setSignals((prev) => [message.data as Signal, ...prev].slice(0, 50));
            break;
          case 'risk':
            setRiskMetrics(message.data as RiskMetrics);
            break;
        }
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected, reconnecting...');
      setConnected(false);
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      ws.close();
    };

    return ws;
  }, []);

  useEffect(() => {
    const ws = connectWebSocket();
    return () => {
      ws.close();
    };
  }, [connectWebSocket]);

  return (
    <WebSocketContext.Provider
      value={{ connected, accountData, positions, trades, signals, riskMetrics }}
    >
      {children}
    </WebSocketContext.Provider>
  );
};

// Styles
const styles = {
  app: {
    display: 'flex',
    minHeight: '100vh',
    backgroundColor: '#0a0a0a',
    color: '#ffffff',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  } as React.CSSProperties,
  sidebar: {
    width: '240px',
    backgroundColor: '#111111',
    borderRight: '1px solid #222222',
    display: 'flex',
    flexDirection: 'column',
    padding: '20px 0',
  } as React.CSSProperties,
  logo: {
    padding: '0 20px 30px',
    fontSize: '24px',
    fontWeight: 700,
    color: '#00ff88',
    borderBottom: '1px solid #222222',
    marginBottom: '20px',
  } as React.CSSProperties,
  navList: {
    listStyle: 'none',
    padding: 0,
    margin: 0,
  } as React.CSSProperties,
  navItem: {
    margin: '4px 12px',
  } as React.CSSProperties,
  navLink: {
    display: 'flex',
    alignItems: 'center',
    padding: '12px 16px',
    color: '#888888',
    textDecoration: 'none',
    borderRadius: '8px',
    transition: 'all 0.2s ease',
    fontSize: '14px',
    fontWeight: 500,
  } as React.CSSProperties,
  navLinkActive: {
    backgroundColor: '#1a1a1a',
    color: '#00ff88',
  } as React.CSSProperties,
  navIcon: {
    marginRight: '12px',
    fontSize: '18px',
  } as React.CSSProperties,
  mainContent: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
  } as React.CSSProperties,
  topBar: {
    height: '64px',
    backgroundColor: '#111111',
    borderBottom: '1px solid #222222',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 24px',
  } as React.CSSProperties,
  topBarLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '24px',
  } as React.CSSProperties,
  topBarRight: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
  } as React.CSSProperties,
  statusIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '13px',
    color: '#888888',
  } as React.CSSProperties,
  statusDot: (connected: boolean) => ({
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    backgroundColor: connected ? '#00ff88' : '#ff4444',
  }) as React.CSSProperties,
  metricCard: {
    display: 'flex',
    flexDirection: 'column',
    padding: '8px 16px',
    backgroundColor: '#1a1a1a',
    borderRadius: '8px',
  } as React.CSSProperties,
  metricLabel: {
    fontSize: '11px',
    color: '#666666',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  } as React.CSSProperties,
  metricValue: {
    fontSize: '18px',
    fontWeight: 600,
  } as React.CSSProperties,
  pageContent: {
    flex: 1,
    padding: '24px',
    overflowY: 'auto',
  } as React.CSSProperties,
  pageTitle: {
    fontSize: '28px',
    fontWeight: 700,
    marginBottom: '24px',
    color: '#ffffff',
  } as React.CSSProperties,
};

// Helper functions
const formatCurrency = (value: number): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  }).format(value);
};

const formatPercent = (value: number): string => {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
};

const getPnLColor = (value: number): string => {
  if (value > 0) return '#00ff88';
  if (value < 0) return '#ff4444';
  return '#888888';
};

// Navigation icons (simple SVG paths)
const icons = {
  dashboard: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z" />
    </svg>
  ),
  positions: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M4 6h16v2H4V6zm0 5h16v2H4v-2zm0 5h16v2H4v-2z" />
    </svg>
  ),
  trades: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z" />
    </svg>
  ),
  signals: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
    </svg>
  ),
  risk: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2L1 21h22L12 2zm0 3.99L19.53 19H4.47L12 5.99zM11 10v4h2v-4h-2zm0 6v2h2v-2h-2z" />
    </svg>
  ),
};

// Sidebar Component
const Sidebar: React.FC = () => {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: icons.dashboard },
    { path: '/positions', label: 'Positions', icon: icons.positions },
    { path: '/trades', label: 'Trades', icon: icons.trades },
    { path: '/signals', label: 'Signals', icon: icons.signals },
    { path: '/risk', label: 'Risk', icon: icons.risk },
  ];

  return (
    <aside style={styles.sidebar}>
      <div style={styles.logo}>IntelliBot</div>
      <nav>
        <ul style={styles.navList}>
          {navItems.map((item) => (
            <li key={item.path} style={styles.navItem}>
              <NavLink
                to={item.path}
                style={({ isActive }) => ({
                  ...styles.navLink,
                  ...(isActive ? styles.navLinkActive : {}),
                })}
              >
                <span style={styles.navIcon}>{item.icon}</span>
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </aside>
  );
};

// Top Bar Component
const TopBar: React.FC = () => {
  const { connected, accountData } = useWebSocket();

  return (
    <header style={styles.topBar}>
      <div style={styles.topBarLeft}>
        <div style={styles.statusIndicator}>
          <div style={styles.statusDot(connected)} />
          {connected ? 'Live' : 'Disconnected'}
        </div>
      </div>
      <div style={styles.topBarRight}>
        <div style={styles.metricCard as React.CSSProperties}>
          <span style={styles.metricLabel as React.CSSProperties}>Account Balance</span>
          <span style={styles.metricValue}>
            {accountData ? formatCurrency(accountData.balance) : '--'}
          </span>
        </div>
        <div style={styles.metricCard as React.CSSProperties}>
          <span style={styles.metricLabel as React.CSSProperties}>Session P&L</span>
          <span
            style={{
              ...styles.metricValue,
              color: accountData ? getPnLColor(accountData.sessionPnL) : '#888888',
            }}
          >
            {accountData
              ? `${formatCurrency(accountData.sessionPnL)} (${formatPercent(accountData.sessionPnLPercent)})`
              : '--'}
          </span>
        </div>
        <div style={styles.metricCard as React.CSSProperties}>
          <span style={styles.metricLabel as React.CSSProperties}>Buying Power</span>
          <span style={styles.metricValue}>
            {accountData ? formatCurrency(accountData.buyingPower) : '--'}
          </span>
        </div>
      </div>
    </header>
  );
};

// Page Components
const DashboardPage: React.FC = () => {
  const { accountData, positions, trades, signals, riskMetrics } = useWebSocket();

  return (
    <div style={styles.pageContent}>
      <h1 style={styles.pageTitle}>Dashboard</h1>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
        <div style={{ backgroundColor: '#111111', borderRadius: '12px', padding: '20px', border: '1px solid #222222' }}>
          <h3 style={{ color: '#888888', fontSize: '14px', marginBottom: '16px' }}>QUICK STATS</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#666666' }}>Open Positions</span>
              <span style={{ fontWeight: 600 }}>{positions.length}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#666666' }}>Today's Trades</span>
              <span style={{ fontWeight: 600 }}>{trades.length}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#666666' }}>Active Signals</span>
              <span style={{ fontWeight: 600 }}>{signals.length}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#666666' }}>Win Rate</span>
              <span style={{ fontWeight: 600, color: '#00ff88' }}>
                {riskMetrics ? `${(riskMetrics.winRate * 100).toFixed(1)}%` : '--'}
              </span>
            </div>
          </div>
        </div>
        <div style={{ backgroundColor: '#111111', borderRadius: '12px', padding: '20px', border: '1px solid #222222' }}>
          <h3 style={{ color: '#888888', fontSize: '14px', marginBottom: '16px' }}>RECENT ACTIVITY</h3>
          {trades.slice(0, 5).map((trade) => (
            <div
              key={trade.id}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                padding: '8px 0',
                borderBottom: '1px solid #222222',
              }}
            >
              <span>
                <span style={{ color: trade.side === 'buy' ? '#00ff88' : '#ff4444' }}>
                  {trade.side.toUpperCase()}
                </span>{' '}
                {trade.symbol}
              </span>
              <span style={{ color: '#888888' }}>{formatCurrency(trade.price * trade.qty)}</span>
            </div>
          ))}
          {trades.length === 0 && (
            <div style={{ color: '#666666', textAlign: 'center', padding: '20px' }}>
              No recent trades
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const PositionsPage: React.FC = () => {
  const { positions } = useWebSocket();

  return (
    <div style={styles.pageContent}>
      <h1 style={styles.pageTitle}>Positions</h1>
      <div style={{ backgroundColor: '#111111', borderRadius: '12px', border: '1px solid #222222', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #222222' }}>
              <th style={{ padding: '16px', textAlign: 'left', color: '#888888', fontSize: '12px' }}>SYMBOL</th>
              <th style={{ padding: '16px', textAlign: 'right', color: '#888888', fontSize: '12px' }}>QTY</th>
              <th style={{ padding: '16px', textAlign: 'right', color: '#888888', fontSize: '12px' }}>AVG ENTRY</th>
              <th style={{ padding: '16px', textAlign: 'right', color: '#888888', fontSize: '12px' }}>CURRENT</th>
              <th style={{ padding: '16px', textAlign: 'right', color: '#888888', fontSize: '12px' }}>P&L</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((position) => (
              <tr key={position.symbol} style={{ borderBottom: '1px solid #222222' }}>
                <td style={{ padding: '16px', fontWeight: 600 }}>{position.symbol}</td>
                <td style={{ padding: '16px', textAlign: 'right' }}>{position.qty}</td>
                <td style={{ padding: '16px', textAlign: 'right' }}>{formatCurrency(position.avgEntryPrice)}</td>
                <td style={{ padding: '16px', textAlign: 'right' }}>{formatCurrency(position.currentPrice)}</td>
                <td style={{ padding: '16px', textAlign: 'right', color: getPnLColor(position.unrealizedPnL) }}>
                  {formatCurrency(position.unrealizedPnL)} ({formatPercent(position.unrealizedPnLPercent)})
                </td>
              </tr>
            ))}
            {positions.length === 0 && (
              <tr>
                <td colSpan={5} style={{ padding: '40px', textAlign: 'center', color: '#666666' }}>
                  No open positions
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const TradesPage: React.FC = () => {
  const { trades } = useWebSocket();

  return (
    <div style={styles.pageContent}>
      <h1 style={styles.pageTitle}>Trades</h1>
      <div style={{ backgroundColor: '#111111', borderRadius: '12px', border: '1px solid #222222', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #222222' }}>
              <th style={{ padding: '16px', textAlign: 'left', color: '#888888', fontSize: '12px' }}>TIME</th>
              <th style={{ padding: '16px', textAlign: 'left', color: '#888888', fontSize: '12px' }}>SYMBOL</th>
              <th style={{ padding: '16px', textAlign: 'left', color: '#888888', fontSize: '12px' }}>SIDE</th>
              <th style={{ padding: '16px', textAlign: 'right', color: '#888888', fontSize: '12px' }}>QTY</th>
              <th style={{ padding: '16px', textAlign: 'right', color: '#888888', fontSize: '12px' }}>PRICE</th>
              <th style={{ padding: '16px', textAlign: 'right', color: '#888888', fontSize: '12px' }}>P&L</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((trade) => (
              <tr key={trade.id} style={{ borderBottom: '1px solid #222222' }}>
                <td style={{ padding: '16px', color: '#888888' }}>
                  {new Date(trade.timestamp).toLocaleTimeString()}
                </td>
                <td style={{ padding: '16px', fontWeight: 600 }}>{trade.symbol}</td>
                <td style={{ padding: '16px', color: trade.side === 'buy' ? '#00ff88' : '#ff4444' }}>
                  {trade.side.toUpperCase()}
                </td>
                <td style={{ padding: '16px', textAlign: 'right' }}>{trade.qty}</td>
                <td style={{ padding: '16px', textAlign: 'right' }}>{formatCurrency(trade.price)}</td>
                <td style={{ padding: '16px', textAlign: 'right', color: trade.pnl ? getPnLColor(trade.pnl) : '#888888' }}>
                  {trade.pnl ? formatCurrency(trade.pnl) : '--'}
                </td>
              </tr>
            ))}
            {trades.length === 0 && (
              <tr>
                <td colSpan={6} style={{ padding: '40px', textAlign: 'center', color: '#666666' }}>
                  No trades today
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const SignalsPage: React.FC = () => {
  const { signals } = useWebSocket();

  return (
    <div style={styles.pageContent}>
      <h1 style={styles.pageTitle}>Signals</h1>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '16px' }}>
        {signals.map((signal) => (
          <div
            key={signal.id}
            style={{
              backgroundColor: '#111111',
              borderRadius: '12px',
              padding: '20px',
              border: `1px solid ${signal.type === 'call' ? '#00ff8833' : '#ff444433'}`,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <span style={{ fontSize: '18px', fontWeight: 600 }}>{signal.symbol}</span>
              <span
                style={{
                  padding: '4px 12px',
                  borderRadius: '4px',
                  backgroundColor: signal.type === 'call' ? '#00ff8822' : '#ff444422',
                  color: signal.type === 'call' ? '#00ff88' : '#ff4444',
                  fontSize: '12px',
                  fontWeight: 600,
                }}
              >
                {signal.type.toUpperCase()}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#888888', fontSize: '14px' }}>
              <span>Conviction</span>
              <span style={{ color: '#ffffff' }}>{(signal.conviction * 100).toFixed(0)}%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#888888', fontSize: '14px', marginTop: '8px' }}>
              <span>Source</span>
              <span style={{ color: '#ffffff' }}>{signal.source}</span>
            </div>
            <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid #222222', color: '#666666', fontSize: '12px' }}>
              {new Date(signal.timestamp).toLocaleString()}
            </div>
          </div>
        ))}
        {signals.length === 0 && (
          <div style={{ gridColumn: '1 / -1', backgroundColor: '#111111', borderRadius: '12px', padding: '40px', textAlign: 'center', color: '#666666' }}>
            No active signals
          </div>
        )}
      </div>
    </div>
  );
};

const RiskPage: React.FC = () => {
  const { riskMetrics, positions } = useWebSocket();

  return (
    <div style={styles.pageContent}>
      <h1 style={styles.pageTitle}>Risk Management</h1>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '20px', marginBottom: '24px' }}>
        <div style={{ backgroundColor: '#111111', borderRadius: '12px', padding: '24px', border: '1px solid #222222' }}>
          <div style={{ color: '#888888', fontSize: '12px', marginBottom: '8px' }}>TOTAL EXPOSURE</div>
          <div style={{ fontSize: '28px', fontWeight: 700 }}>
            {riskMetrics ? formatCurrency(riskMetrics.totalExposure) : '--'}
          </div>
        </div>
        <div style={{ backgroundColor: '#111111', borderRadius: '12px', padding: '24px', border: '1px solid #222222' }}>
          <div style={{ color: '#888888', fontSize: '12px', marginBottom: '8px' }}>MAX DRAWDOWN</div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: '#ff4444' }}>
            {riskMetrics ? formatPercent(-riskMetrics.maxDrawdown) : '--'}
          </div>
        </div>
        <div style={{ backgroundColor: '#111111', borderRadius: '12px', padding: '24px', border: '1px solid #222222' }}>
          <div style={{ color: '#888888', fontSize: '12px', marginBottom: '8px' }}>WIN RATE</div>
          <div style={{ fontSize: '28px', fontWeight: 700, color: '#00ff88' }}>
            {riskMetrics ? `${(riskMetrics.winRate * 100).toFixed(1)}%` : '--'}
          </div>
        </div>
        <div style={{ backgroundColor: '#111111', borderRadius: '12px', padding: '24px', border: '1px solid #222222' }}>
          <div style={{ color: '#888888', fontSize: '12px', marginBottom: '8px' }}>SHARPE RATIO</div>
          <div style={{ fontSize: '28px', fontWeight: 700 }}>
            {riskMetrics ? riskMetrics.sharpeRatio.toFixed(2) : '--'}
          </div>
        </div>
      </div>
      <div style={{ backgroundColor: '#111111', borderRadius: '12px', padding: '24px', border: '1px solid #222222' }}>
        <h3 style={{ color: '#888888', fontSize: '14px', marginBottom: '20px' }}>POSITION RISK BREAKDOWN</h3>
        {positions.map((position) => {
          const riskPercent = Math.abs(position.unrealizedPnLPercent);
          return (
            <div key={position.symbol} style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontWeight: 600 }}>{position.symbol}</span>
                <span style={{ color: getPnLColor(position.unrealizedPnL) }}>
                  {formatPercent(position.unrealizedPnLPercent)}
                </span>
              </div>
              <div style={{ height: '4px', backgroundColor: '#222222', borderRadius: '2px', overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${Math.min(riskPercent * 10, 100)}%`,
                    height: '100%',
                    backgroundColor: position.unrealizedPnL >= 0 ? '#00ff88' : '#ff4444',
                    borderRadius: '2px',
                  }}
                />
              </div>
            </div>
          );
        })}
        {positions.length === 0 && (
          <div style={{ color: '#666666', textAlign: 'center', padding: '20px' }}>
            No positions to analyze
          </div>
        )}
      </div>
    </div>
  );
};

// Main App Component
const App: React.FC = () => {
  return (
    <WebSocketProvider>
      <Router>
        <div style={styles.app}>
          <Sidebar />
          <main style={styles.mainContent as React.CSSProperties}>
            <TopBar />
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/positions" element={<PositionsPage />} />
              <Route path="/trades" element={<TradesPage />} />
              <Route path="/signals" element={<SignalsPage />} />
              <Route path="/risk" element={<RiskPage />} />
            </Routes>
          </main>
        </div>
      </Router>
    </WebSocketProvider>
  );
};

export default App;
