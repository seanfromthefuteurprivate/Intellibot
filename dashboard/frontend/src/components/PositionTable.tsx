import React, { useState } from 'react';

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
  engine: string;
  openedAt: string | Date;
  stopLoss?: number;
  takeProfit?: number;
}

type SortField = 'symbol' | 'side' | 'quantity' | 'entryPrice' | 'currentPrice' | 'pnl' | 'pnlPercent' | 'engine' | 'timeHeld';
type SortDirection = 'asc' | 'desc';

interface PositionTableProps {
  positions: Position[];
  onSort?: (field: SortField, direction: SortDirection) => void;
}

const PositionTable: React.FC<PositionTableProps> = ({ positions, onSort }) => {
  const [sortField, setSortField] = useState<SortField>('symbol');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');

  const handleSort = (field: SortField) => {
    const newDirection = sortField === field && sortDirection === 'asc' ? 'desc' : 'asc';
    setSortField(field);
    setSortDirection(newDirection);
    onSort?.(field, newDirection);
  };

  const formatTimeHeld = (openedAt: string | Date): string => {
    const opened = new Date(openedAt);
    const now = new Date();
    const diffMs = now.getTime() - opened.getTime();

    const seconds = Math.floor(diffMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) {
      return `${days}d ${hours % 24}h`;
    }
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    }
    if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
  };

  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value: number): string => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  const getPnlColor = (pnl: number): string => {
    if (pnl > 0) return '#10b981'; // green
    if (pnl < 0) return '#ef4444'; // red
    return '#6b7280'; // gray
  };

  const getEngineColor = (engine: string): string => {
    const colors: Record<string, string> = {
      apex: '#8b5cf6',      // purple
      momentum: '#3b82f6',  // blue
      scalper: '#f59e0b',   // amber
      swing: '#10b981',     // green
      max: '#ef4444',       // red
      default: '#6b7280',   // gray
    };
    return colors[engine.toLowerCase()] || colors.default;
  };

  const SortIcon: React.FC<{ field: SortField }> = ({ field }) => {
    if (sortField !== field) {
      return (
        <span style={styles.sortIconInactive}>⇅</span>
      );
    }
    return (
      <span style={styles.sortIconActive}>
        {sortDirection === 'asc' ? '↑' : '↓'}
      </span>
    );
  };

  const sortedPositions = [...positions].sort((a, b) => {
    let aValue: string | number;
    let bValue: string | number;

    switch (sortField) {
      case 'timeHeld':
        aValue = new Date(a.openedAt).getTime();
        bValue = new Date(b.openedAt).getTime();
        break;
      case 'symbol':
      case 'side':
      case 'engine':
        aValue = a[sortField].toLowerCase();
        bValue = b[sortField].toLowerCase();
        break;
      default:
        aValue = a[sortField];
        bValue = b[sortField];
    }

    if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
    if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
    return 0;
  });

  return (
    <div style={styles.container}>
      <table style={styles.table}>
        <thead>
          <tr>
            <th style={styles.th} onClick={() => handleSort('symbol')}>
              Symbol <SortIcon field="symbol" />
            </th>
            <th style={styles.th} onClick={() => handleSort('side')}>
              Side <SortIcon field="side" />
            </th>
            <th style={styles.th} onClick={() => handleSort('quantity')}>
              Qty <SortIcon field="quantity" />
            </th>
            <th style={styles.th} onClick={() => handleSort('entryPrice')}>
              Entry <SortIcon field="entryPrice" />
            </th>
            <th style={styles.th} onClick={() => handleSort('currentPrice')}>
              Current <SortIcon field="currentPrice" />
            </th>
            <th style={styles.th} onClick={() => handleSort('pnl')}>
              P&L <SortIcon field="pnl" />
            </th>
            <th style={styles.th} onClick={() => handleSort('pnlPercent')}>
              P&L % <SortIcon field="pnlPercent" />
            </th>
            <th style={styles.th} onClick={() => handleSort('engine')}>
              Engine <SortIcon field="engine" />
            </th>
            <th style={styles.th} onClick={() => handleSort('timeHeld')}>
              Time Held <SortIcon field="timeHeld" />
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedPositions.length === 0 ? (
            <tr>
              <td colSpan={9} style={styles.emptyCell}>
                No open positions
              </td>
            </tr>
          ) : (
            sortedPositions.map((position) => (
              <tr key={position.id} style={styles.tr}>
                <td style={styles.td}>
                  <span style={styles.symbol}>{position.symbol}</span>
                </td>
                <td style={styles.td}>
                  <span
                    style={{
                      ...styles.sideBadge,
                      backgroundColor: position.side === 'long' ? '#064e3b' : '#7f1d1d',
                      color: position.side === 'long' ? '#34d399' : '#fca5a5',
                    }}
                  >
                    {position.side.toUpperCase()}
                  </span>
                </td>
                <td style={styles.td}>{position.quantity}</td>
                <td style={styles.td}>{formatCurrency(position.entryPrice)}</td>
                <td style={styles.td}>{formatCurrency(position.currentPrice)}</td>
                <td style={{ ...styles.td, color: getPnlColor(position.pnl) }}>
                  {formatCurrency(position.pnl)}
                </td>
                <td style={{ ...styles.td, color: getPnlColor(position.pnlPercent) }}>
                  {formatPercent(position.pnlPercent)}
                </td>
                <td style={styles.td}>
                  <span
                    style={{
                      ...styles.engineBadge,
                      backgroundColor: `${getEngineColor(position.engine)}20`,
                      color: getEngineColor(position.engine),
                      borderColor: getEngineColor(position.engine),
                    }}
                  >
                    {position.engine}
                  </span>
                </td>
                <td style={styles.td}>{formatTimeHeld(position.openedAt)}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    backgroundColor: '#1a1a2e',
    borderRadius: '8px',
    overflow: 'hidden',
    border: '1px solid #2d2d44',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '14px',
  },
  th: {
    backgroundColor: '#0f0f1a',
    color: '#a0a0b0',
    padding: '12px 16px',
    textAlign: 'left',
    fontWeight: 600,
    fontSize: '12px',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    cursor: 'pointer',
    userSelect: 'none',
    borderBottom: '1px solid #2d2d44',
    whiteSpace: 'nowrap',
  },
  tr: {
    borderBottom: '1px solid #2d2d44',
    transition: 'background-color 0.15s ease',
  },
  td: {
    padding: '12px 16px',
    color: '#e0e0e8',
    whiteSpace: 'nowrap',
  },
  emptyCell: {
    padding: '32px 16px',
    textAlign: 'center',
    color: '#6b7280',
    fontStyle: 'italic',
  },
  symbol: {
    fontWeight: 600,
    color: '#ffffff',
  },
  sideBadge: {
    padding: '4px 8px',
    borderRadius: '4px',
    fontSize: '11px',
    fontWeight: 600,
    letterSpacing: '0.05em',
  },
  engineBadge: {
    padding: '4px 10px',
    borderRadius: '12px',
    fontSize: '11px',
    fontWeight: 600,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    border: '1px solid',
  },
  sortIconActive: {
    marginLeft: '4px',
    color: '#8b5cf6',
  },
  sortIconInactive: {
    marginLeft: '4px',
    color: '#4b5563',
    opacity: 0.5,
  },
};

export default PositionTable;
