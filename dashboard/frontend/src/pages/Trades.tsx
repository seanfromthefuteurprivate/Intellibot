import React, { useState, useMemo, useCallback } from 'react';

// Types
interface Trade {
  id: string;
  date: string;
  time: string;
  symbol: string;
  type: 'BUY' | 'SELL' | 'LONG' | 'SHORT';
  entry: number;
  exit: number | null;
  pnl: number | null;
  pnlPercent: number | null;
  duration: string;
  engine: string;
  status: 'OPEN' | 'CLOSED';
}

interface SummaryStats {
  totalTrades: number;
  wins: number;
  losses: number;
  netPnL: number;
  winRate: number;
}

// Styles
const styles: Record<string, React.CSSProperties> = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#0d1117',
    color: '#c9d1d9',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
    padding: '24px',
  },
  header: {
    marginBottom: '24px',
  },
  title: {
    fontSize: '28px',
    fontWeight: 600,
    color: '#f0f6fc',
    margin: 0,
    marginBottom: '8px',
  },
  subtitle: {
    fontSize: '14px',
    color: '#8b949e',
    margin: 0,
  },
  summaryContainer: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '16px',
    marginBottom: '24px',
  },
  summaryCard: {
    backgroundColor: '#161b22',
    borderRadius: '8px',
    padding: '20px',
    border: '1px solid #30363d',
  },
  summaryLabel: {
    fontSize: '12px',
    color: '#8b949e',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
    marginBottom: '8px',
  },
  summaryValue: {
    fontSize: '24px',
    fontWeight: 600,
    color: '#f0f6fc',
  },
  summaryValuePositive: {
    fontSize: '24px',
    fontWeight: 600,
    color: '#3fb950',
  },
  summaryValueNegative: {
    fontSize: '24px',
    fontWeight: 600,
    color: '#f85149',
  },
  filtersContainer: {
    display: 'flex',
    gap: '16px',
    marginBottom: '24px',
    flexWrap: 'wrap' as const,
    alignItems: 'flex-end',
  },
  filterGroup: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '6px',
  },
  filterLabel: {
    fontSize: '12px',
    color: '#8b949e',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
  },
  input: {
    backgroundColor: '#0d1117',
    border: '1px solid #30363d',
    borderRadius: '6px',
    padding: '8px 12px',
    color: '#c9d1d9',
    fontSize: '14px',
    outline: 'none',
    minWidth: '150px',
  },
  select: {
    backgroundColor: '#0d1117',
    border: '1px solid #30363d',
    borderRadius: '6px',
    padding: '8px 12px',
    color: '#c9d1d9',
    fontSize: '14px',
    outline: 'none',
    cursor: 'pointer',
    minWidth: '150px',
  },
  clearButton: {
    backgroundColor: '#21262d',
    border: '1px solid #30363d',
    borderRadius: '6px',
    padding: '8px 16px',
    color: '#c9d1d9',
    fontSize: '14px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  tableContainer: {
    backgroundColor: '#161b22',
    borderRadius: '8px',
    border: '1px solid #30363d',
    overflow: 'hidden',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse' as const,
  },
  th: {
    textAlign: 'left' as const,
    padding: '12px 16px',
    fontSize: '12px',
    fontWeight: 600,
    color: '#8b949e',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
    borderBottom: '1px solid #30363d',
    backgroundColor: '#0d1117',
  },
  td: {
    padding: '12px 16px',
    fontSize: '14px',
    borderBottom: '1px solid #21262d',
    color: '#c9d1d9',
  },
  trHover: {
    transition: 'background-color 0.2s',
  },
  pnlPositive: {
    color: '#3fb950',
    fontWeight: 500,
  },
  pnlNegative: {
    color: '#f85149',
    fontWeight: 500,
  },
  pnlNeutral: {
    color: '#8b949e',
  },
  typeBadge: {
    display: 'inline-block',
    padding: '2px 8px',
    borderRadius: '12px',
    fontSize: '12px',
    fontWeight: 500,
  },
  typeBuy: {
    backgroundColor: 'rgba(63, 185, 80, 0.2)',
    color: '#3fb950',
  },
  typeSell: {
    backgroundColor: 'rgba(248, 81, 73, 0.2)',
    color: '#f85149',
  },
  engineBadge: {
    display: 'inline-block',
    padding: '2px 8px',
    borderRadius: '4px',
    fontSize: '11px',
    fontWeight: 500,
    backgroundColor: 'rgba(56, 139, 253, 0.2)',
    color: '#58a6ff',
  },
  paginationContainer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px',
    borderTop: '1px solid #30363d',
    backgroundColor: '#0d1117',
  },
  paginationInfo: {
    fontSize: '14px',
    color: '#8b949e',
  },
  paginationButtons: {
    display: 'flex',
    gap: '8px',
  },
  paginationButton: {
    backgroundColor: '#21262d',
    border: '1px solid #30363d',
    borderRadius: '6px',
    padding: '6px 12px',
    color: '#c9d1d9',
    fontSize: '14px',
    cursor: 'pointer',
    transition: 'background-color 0.2s',
  },
  paginationButtonDisabled: {
    backgroundColor: '#161b22',
    border: '1px solid #21262d',
    borderRadius: '6px',
    padding: '6px 12px',
    color: '#484f58',
    fontSize: '14px',
    cursor: 'not-allowed',
  },
  paginationButtonActive: {
    backgroundColor: '#388bfd',
    border: '1px solid #388bfd',
    borderRadius: '6px',
    padding: '6px 12px',
    color: '#ffffff',
    fontSize: '14px',
    cursor: 'pointer',
  },
  emptyState: {
    textAlign: 'center' as const,
    padding: '48px',
    color: '#8b949e',
  },
  statusOpen: {
    color: '#58a6ff',
  },
  statusClosed: {
    color: '#8b949e',
  },
};

// Mock data generator for demonstration
const generateMockTrades = (): Trade[] => {
  const symbols = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MSFT', 'GOOGL', 'META', 'AMZN', 'SPY', 'QQQ'];
  const engines = ['APEX', 'MAX_MODE', 'MOMENTUM', 'BREAKOUT', 'REVERSAL', 'SCALPER'];
  const types: ('BUY' | 'SELL' | 'LONG' | 'SHORT')[] = ['BUY', 'SELL', 'LONG', 'SHORT'];

  const trades: Trade[] = [];
  const now = new Date();

  for (let i = 0; i < 100; i++) {
    const date = new Date(now);
    date.setDate(date.getDate() - Math.floor(Math.random() * 30));

    const entry = 100 + Math.random() * 400;
    const isClosed = Math.random() > 0.1;
    const exit = isClosed ? entry * (0.9 + Math.random() * 0.2) : null;
    const pnl = exit ? (exit - entry) * (Math.random() * 100 + 10) : null;
    const pnlPercent = exit ? ((exit - entry) / entry) * 100 : null;

    const hours = Math.floor(Math.random() * 8);
    const minutes = Math.floor(Math.random() * 60);
    const duration = isClosed ? `${hours}h ${minutes}m` : '-';

    trades.push({
      id: `trade-${i + 1}`,
      date: date.toISOString().split('T')[0],
      time: `${String(9 + Math.floor(Math.random() * 7)).padStart(2, '0')}:${String(Math.floor(Math.random() * 60)).padStart(2, '0')}`,
      symbol: symbols[Math.floor(Math.random() * symbols.length)],
      type: types[Math.floor(Math.random() * types.length)],
      entry: entry,
      exit: exit,
      pnl: pnl,
      pnlPercent: pnlPercent,
      duration: duration,
      engine: engines[Math.floor(Math.random() * engines.length)],
      status: isClosed ? 'CLOSED' : 'OPEN',
    });
  }

  return trades.sort((a, b) => {
    const dateCompare = b.date.localeCompare(a.date);
    if (dateCompare !== 0) return dateCompare;
    return b.time.localeCompare(a.time);
  });
};

const ITEMS_PER_PAGE = 20;

const Trades: React.FC = () => {
  // State
  const [trades] = useState<Trade[]>(generateMockTrades);
  const [dateFrom, setDateFrom] = useState<string>('');
  const [dateTo, setDateTo] = useState<string>('');
  const [tickerFilter, setTickerFilter] = useState<string>('');
  const [engineFilter, setEngineFilter] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<string>('');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);

  // Get unique values for filter dropdowns
  const uniqueEngines = useMemo(() =>
    [...new Set(trades.map(t => t.engine))].sort(),
    [trades]
  );

  const uniqueSymbols = useMemo(() =>
    [...new Set(trades.map(t => t.symbol))].sort(),
    [trades]
  );

  // Filter trades
  const filteredTrades = useMemo(() => {
    return trades.filter(trade => {
      if (dateFrom && trade.date < dateFrom) return false;
      if (dateTo && trade.date > dateTo) return false;
      if (tickerFilter && trade.symbol !== tickerFilter) return false;
      if (engineFilter && trade.engine !== engineFilter) return false;
      if (typeFilter && trade.type !== typeFilter) return false;
      return true;
    });
  }, [trades, dateFrom, dateTo, tickerFilter, engineFilter, typeFilter]);

  // Calculate summary stats
  const summaryStats = useMemo((): SummaryStats => {
    const closedTrades = filteredTrades.filter(t => t.status === 'CLOSED' && t.pnl !== null);
    const wins = closedTrades.filter(t => (t.pnl ?? 0) > 0).length;
    const losses = closedTrades.filter(t => (t.pnl ?? 0) < 0).length;
    const netPnL = closedTrades.reduce((sum, t) => sum + (t.pnl ?? 0), 0);

    return {
      totalTrades: filteredTrades.length,
      wins,
      losses,
      netPnL,
      winRate: closedTrades.length > 0 ? (wins / closedTrades.length) * 100 : 0,
    };
  }, [filteredTrades]);

  // Pagination
  const totalPages = Math.ceil(filteredTrades.length / ITEMS_PER_PAGE);
  const paginatedTrades = useMemo(() => {
    const start = (currentPage - 1) * ITEMS_PER_PAGE;
    return filteredTrades.slice(start, start + ITEMS_PER_PAGE);
  }, [filteredTrades, currentPage]);

  // Reset to page 1 when filters change
  const handleFilterChange = useCallback((setter: React.Dispatch<React.SetStateAction<string>>) => {
    return (value: string) => {
      setter(value);
      setCurrentPage(1);
    };
  }, []);

  const clearFilters = useCallback(() => {
    setDateFrom('');
    setDateTo('');
    setTickerFilter('');
    setEngineFilter('');
    setTypeFilter('');
    setCurrentPage(1);
  }, []);

  // Format helpers
  const formatCurrency = (value: number | null): string => {
    if (value === null) return '-';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value: number | null): string => {
    if (value === null) return '-';
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  const formatPrice = (value: number): string => {
    return `$${value.toFixed(2)}`;
  };

  const getPnLStyle = (value: number | null): React.CSSProperties => {
    if (value === null) return styles.pnlNeutral;
    return value >= 0 ? styles.pnlPositive : styles.pnlNegative;
  };

  const getTypeBadgeStyle = (type: string): React.CSSProperties => {
    const isPositive = type === 'BUY' || type === 'LONG';
    return {
      ...styles.typeBadge,
      ...(isPositive ? styles.typeBuy : styles.typeSell),
    };
  };

  // Pagination controls
  const getPageNumbers = (): (number | string)[] => {
    const pages: (number | string)[] = [];
    const maxVisible = 5;

    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      if (currentPage <= 3) {
        for (let i = 1; i <= 4; i++) pages.push(i);
        pages.push('...');
        pages.push(totalPages);
      } else if (currentPage >= totalPages - 2) {
        pages.push(1);
        pages.push('...');
        for (let i = totalPages - 3; i <= totalPages; i++) pages.push(i);
      } else {
        pages.push(1);
        pages.push('...');
        for (let i = currentPage - 1; i <= currentPage + 1; i++) pages.push(i);
        pages.push('...');
        pages.push(totalPages);
      }
    }

    return pages;
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <h1 style={styles.title}>Trade History</h1>
        <p style={styles.subtitle}>View and analyze your trading activity</p>
      </div>

      {/* Summary Stats */}
      <div style={styles.summaryContainer}>
        <div style={styles.summaryCard}>
          <div style={styles.summaryLabel}>Total Trades</div>
          <div style={styles.summaryValue}>{summaryStats.totalTrades}</div>
        </div>
        <div style={styles.summaryCard}>
          <div style={styles.summaryLabel}>Wins</div>
          <div style={styles.summaryValuePositive}>{summaryStats.wins}</div>
        </div>
        <div style={styles.summaryCard}>
          <div style={styles.summaryLabel}>Losses</div>
          <div style={styles.summaryValueNegative}>{summaryStats.losses}</div>
        </div>
        <div style={styles.summaryCard}>
          <div style={styles.summaryLabel}>Net P&L</div>
          <div style={summaryStats.netPnL >= 0 ? styles.summaryValuePositive : styles.summaryValueNegative}>
            {formatCurrency(summaryStats.netPnL)}
          </div>
        </div>
        <div style={styles.summaryCard}>
          <div style={styles.summaryLabel}>Win Rate</div>
          <div style={styles.summaryValue}>{summaryStats.winRate.toFixed(1)}%</div>
        </div>
      </div>

      {/* Filters */}
      <div style={styles.filtersContainer}>
        <div style={styles.filterGroup}>
          <label style={styles.filterLabel}>Date From</label>
          <input
            type="date"
            style={styles.input}
            value={dateFrom}
            onChange={(e) => handleFilterChange(setDateFrom)(e.target.value)}
          />
        </div>
        <div style={styles.filterGroup}>
          <label style={styles.filterLabel}>Date To</label>
          <input
            type="date"
            style={styles.input}
            value={dateTo}
            onChange={(e) => handleFilterChange(setDateTo)(e.target.value)}
          />
        </div>
        <div style={styles.filterGroup}>
          <label style={styles.filterLabel}>Symbol</label>
          <select
            style={styles.select}
            value={tickerFilter}
            onChange={(e) => handleFilterChange(setTickerFilter)(e.target.value)}
          >
            <option value="">All Symbols</option>
            {uniqueSymbols.map(symbol => (
              <option key={symbol} value={symbol}>{symbol}</option>
            ))}
          </select>
        </div>
        <div style={styles.filterGroup}>
          <label style={styles.filterLabel}>Type</label>
          <select
            style={styles.select}
            value={typeFilter}
            onChange={(e) => handleFilterChange(setTypeFilter)(e.target.value)}
          >
            <option value="">All Types</option>
            <option value="BUY">BUY</option>
            <option value="SELL">SELL</option>
            <option value="LONG">LONG</option>
            <option value="SHORT">SHORT</option>
          </select>
        </div>
        <div style={styles.filterGroup}>
          <label style={styles.filterLabel}>Engine</label>
          <select
            style={styles.select}
            value={engineFilter}
            onChange={(e) => handleFilterChange(setEngineFilter)(e.target.value)}
          >
            <option value="">All Engines</option>
            {uniqueEngines.map(engine => (
              <option key={engine} value={engine}>{engine}</option>
            ))}
          </select>
        </div>
        <button
          style={styles.clearButton}
          onClick={clearFilters}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#30363d')}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#21262d')}
        >
          Clear Filters
        </button>
      </div>

      {/* Trade Table */}
      <div style={styles.tableContainer}>
        {paginatedTrades.length > 0 ? (
          <>
            <table style={styles.table}>
              <thead>
                <tr>
                  <th style={styles.th}>Date</th>
                  <th style={styles.th}>Time</th>
                  <th style={styles.th}>Symbol</th>
                  <th style={styles.th}>Type</th>
                  <th style={styles.th}>Entry</th>
                  <th style={styles.th}>Exit</th>
                  <th style={styles.th}>P&L</th>
                  <th style={styles.th}>P&L%</th>
                  <th style={styles.th}>Duration</th>
                  <th style={styles.th}>Engine</th>
                </tr>
              </thead>
              <tbody>
                {paginatedTrades.map((trade) => (
                  <tr
                    key={trade.id}
                    style={{
                      ...styles.trHover,
                      backgroundColor: hoveredRow === trade.id ? '#1c2128' : 'transparent',
                    }}
                    onMouseEnter={() => setHoveredRow(trade.id)}
                    onMouseLeave={() => setHoveredRow(null)}
                  >
                    <td style={styles.td}>{trade.date}</td>
                    <td style={styles.td}>{trade.time}</td>
                    <td style={{ ...styles.td, fontWeight: 600, color: '#f0f6fc' }}>{trade.symbol}</td>
                    <td style={styles.td}>
                      <span style={getTypeBadgeStyle(trade.type)}>{trade.type}</span>
                    </td>
                    <td style={styles.td}>{formatPrice(trade.entry)}</td>
                    <td style={styles.td}>
                      {trade.exit !== null ? formatPrice(trade.exit) : (
                        <span style={styles.statusOpen}>OPEN</span>
                      )}
                    </td>
                    <td style={{ ...styles.td, ...getPnLStyle(trade.pnl) }}>
                      {formatCurrency(trade.pnl)}
                    </td>
                    <td style={{ ...styles.td, ...getPnLStyle(trade.pnlPercent) }}>
                      {formatPercent(trade.pnlPercent)}
                    </td>
                    <td style={styles.td}>{trade.duration}</td>
                    <td style={styles.td}>
                      <span style={styles.engineBadge}>{trade.engine}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {/* Pagination */}
            <div style={styles.paginationContainer}>
              <div style={styles.paginationInfo}>
                Showing {((currentPage - 1) * ITEMS_PER_PAGE) + 1} to{' '}
                {Math.min(currentPage * ITEMS_PER_PAGE, filteredTrades.length)} of{' '}
                {filteredTrades.length} trades
              </div>
              <div style={styles.paginationButtons}>
                <button
                  style={currentPage === 1 ? styles.paginationButtonDisabled : styles.paginationButton}
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                >
                  Previous
                </button>
                {getPageNumbers().map((page, index) => (
                  typeof page === 'number' ? (
                    <button
                      key={index}
                      style={page === currentPage ? styles.paginationButtonActive : styles.paginationButton}
                      onClick={() => setCurrentPage(page)}
                    >
                      {page}
                    </button>
                  ) : (
                    <span key={index} style={{ padding: '6px 8px', color: '#8b949e' }}>
                      {page}
                    </span>
                  )
                ))}
                <button
                  style={currentPage === totalPages ? styles.paginationButtonDisabled : styles.paginationButton}
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                >
                  Next
                </button>
              </div>
            </div>
          </>
        ) : (
          <div style={styles.emptyState}>
            <p>No trades found matching your filters.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Trades;
