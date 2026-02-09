import React, { useState, useEffect, useCallback } from 'react';

interface Position {
  id: string;
  symbol: string;
  option: string;
  qty: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  timeHeld: string;
  engine: string;
}

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

const formatCurrency = (value: number): string => {
  const absValue = Math.abs(value);
  const formatted = absValue.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return value < 0 ? `-${formatted}` : formatted;
};

const formatPercent = (value: number): string => {
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
};

const Positions: React.FC = () => {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchPositions = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/positions`);
      if (!response.ok) {
        throw new Error(`Failed to fetch positions: ${response.statusText}`);
      }
      const data = await response.json();
      setPositions(data.positions || []);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch positions');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPositions();

    const intervalId = setInterval(fetchPositions, 5000);

    return () => clearInterval(intervalId);
  }, [fetchPositions]);

  const getPnlColor = (value: number): string => {
    if (value > 0) return '#10b981'; // Green
    if (value < 0) return '#ef4444'; // Red
    return '#9ca3af'; // Gray for zero
  };

  const styles: { [key: string]: React.CSSProperties } = {
    container: {
      backgroundColor: '#0f0f0f',
      minHeight: '100vh',
      padding: '24px',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    },
    header: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '24px',
    },
    title: {
      color: '#ffffff',
      fontSize: '28px',
      fontWeight: 600,
      margin: 0,
    },
    lastUpdated: {
      color: '#6b7280',
      fontSize: '14px',
    },
    tableContainer: {
      backgroundColor: '#1a1a1a',
      borderRadius: '12px',
      overflow: 'hidden',
      border: '1px solid #2d2d2d',
    },
    table: {
      width: '100%',
      borderCollapse: 'collapse' as const,
    },
    th: {
      backgroundColor: '#252525',
      color: '#9ca3af',
      padding: '14px 16px',
      textAlign: 'left' as const,
      fontSize: '12px',
      fontWeight: 600,
      textTransform: 'uppercase' as const,
      letterSpacing: '0.5px',
      borderBottom: '1px solid #2d2d2d',
    },
    td: {
      padding: '14px 16px',
      borderBottom: '1px solid #2d2d2d',
      color: '#e5e7eb',
      fontSize: '14px',
    },
    symbolCell: {
      fontWeight: 600,
      color: '#ffffff',
    },
    optionCell: {
      color: '#a78bfa',
      fontFamily: 'monospace',
    },
    engineBadge: {
      display: 'inline-block',
      padding: '4px 10px',
      borderRadius: '6px',
      fontSize: '12px',
      fontWeight: 500,
      backgroundColor: '#3b82f6',
      color: '#ffffff',
    },
    emptyState: {
      display: 'flex',
      flexDirection: 'column' as const,
      alignItems: 'center',
      justifyContent: 'center',
      padding: '80px 24px',
      color: '#6b7280',
    },
    emptyIcon: {
      fontSize: '48px',
      marginBottom: '16px',
      opacity: 0.5,
    },
    emptyText: {
      fontSize: '18px',
      fontWeight: 500,
      marginBottom: '8px',
      color: '#9ca3af',
    },
    emptySubtext: {
      fontSize: '14px',
    },
    loadingContainer: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '80px 24px',
      color: '#9ca3af',
    },
    spinner: {
      width: '24px',
      height: '24px',
      border: '3px solid #2d2d2d',
      borderTopColor: '#3b82f6',
      borderRadius: '50%',
      marginRight: '12px',
      animation: 'spin 1s linear infinite',
    },
    errorContainer: {
      backgroundColor: '#7f1d1d',
      border: '1px solid #991b1b',
      borderRadius: '8px',
      padding: '16px',
      marginBottom: '24px',
      color: '#fecaca',
    },
    refreshIndicator: {
      display: 'inline-block',
      width: '8px',
      height: '8px',
      borderRadius: '50%',
      backgroundColor: '#10b981',
      marginRight: '8px',
      animation: 'pulse 2s infinite',
    },
  };

  const keyframes = `
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
  `;

  if (loading && positions.length === 0) {
    return (
      <div style={styles.container}>
        <style>{keyframes}</style>
        <div style={styles.header}>
          <h1 style={styles.title}>Open Positions</h1>
        </div>
        <div style={styles.tableContainer}>
          <div style={styles.loadingContainer}>
            <div style={styles.spinner}></div>
            <span>Loading positions...</span>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <style>{keyframes}</style>
      <div style={styles.header}>
        <h1 style={styles.title}>Open Positions</h1>
        <div style={styles.lastUpdated}>
          <span style={styles.refreshIndicator}></span>
          {lastUpdated && `Last updated: ${lastUpdated.toLocaleTimeString()}`}
          <span style={{ marginLeft: '8px', color: '#4b5563' }}>
            (auto-refresh every 5s)
          </span>
        </div>
      </div>

      {error && (
        <div style={styles.errorContainer}>
          <strong>Error:</strong> {error}
        </div>
      )}

      <div style={styles.tableContainer}>
        {positions.length === 0 ? (
          <div style={styles.emptyState}>
            <div style={styles.emptyIcon}>&#128200;</div>
            <div style={styles.emptyText}>No Open Positions</div>
            <div style={styles.emptySubtext}>
              When you open a trade, it will appear here
            </div>
          </div>
        ) : (
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Symbol</th>
                <th style={styles.th}>Option</th>
                <th style={{ ...styles.th, textAlign: 'right' }}>Qty</th>
                <th style={{ ...styles.th, textAlign: 'right' }}>Entry</th>
                <th style={{ ...styles.th, textAlign: 'right' }}>Current</th>
                <th style={{ ...styles.th, textAlign: 'right' }}>P&L</th>
                <th style={{ ...styles.th, textAlign: 'right' }}>P&L %</th>
                <th style={styles.th}>Time Held</th>
                <th style={styles.th}>Engine</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((position, index) => (
                <tr
                  key={position.id}
                  style={{
                    backgroundColor: index % 2 === 0 ? 'transparent' : '#151515',
                  }}
                >
                  <td style={{ ...styles.td, ...styles.symbolCell }}>
                    {position.symbol}
                  </td>
                  <td style={{ ...styles.td, ...styles.optionCell }}>
                    {position.option}
                  </td>
                  <td style={{ ...styles.td, textAlign: 'right' }}>
                    {position.qty}
                  </td>
                  <td style={{ ...styles.td, textAlign: 'right' }}>
                    {formatCurrency(position.entryPrice)}
                  </td>
                  <td style={{ ...styles.td, textAlign: 'right' }}>
                    {formatCurrency(position.currentPrice)}
                  </td>
                  <td
                    style={{
                      ...styles.td,
                      textAlign: 'right',
                      color: getPnlColor(position.pnl),
                      fontWeight: 600,
                    }}
                  >
                    {formatCurrency(position.pnl)}
                  </td>
                  <td
                    style={{
                      ...styles.td,
                      textAlign: 'right',
                      color: getPnlColor(position.pnlPercent),
                      fontWeight: 600,
                    }}
                  >
                    {formatPercent(position.pnlPercent)}
                  </td>
                  <td style={styles.td}>{position.timeHeld}</td>
                  <td style={styles.td}>
                    <span style={styles.engineBadge}>{position.engine}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};

export default Positions;
