import React, { useState, useEffect, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

interface SignalSummary {
  id: number;
  ticker: string;
  timestamp: string;
  setup_type: string;
  score: number;
  tier: string;
  price: number | null;
  change_pct: number | null;
  volume: number | null;
  social_velocity: number | null;
  sentiment_score: number | null;
  session_type: string | null;
  created_at: string | null;
}

interface SignalDetail extends SignalSummary {
  vwap: number | null;
  range_pct: number | null;
  atm_iv: number | null;
  call_put_ratio: number | null;
  top_strike: number | null;
  options_pressure_score: number | null;
  minutes_to_close: number | null;
  features: Record<string, unknown> | null;
  evidence: Record<string, unknown> | null;
}

interface PaginatedSignals {
  signals: SignalSummary[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// ============================================================================
// Constants
// ============================================================================

const API_BASE = '/api/signals';

const TIER_COLORS: Record<string, { bg: string; border: string; text: string; badge: string }> = {
  S: {
    bg: 'bg-gradient-to-r from-yellow-900/30 to-amber-900/30',
    border: 'border-yellow-500',
    text: 'text-yellow-400',
    badge: 'bg-yellow-500 text-black',
  },
  A: {
    bg: 'bg-gradient-to-r from-green-900/30 to-emerald-900/30',
    border: 'border-green-500',
    text: 'text-green-400',
    badge: 'bg-green-500 text-black',
  },
  'A+': {
    bg: 'bg-gradient-to-r from-green-900/40 to-emerald-900/40',
    border: 'border-green-400',
    text: 'text-green-300',
    badge: 'bg-green-400 text-black',
  },
  B: {
    bg: 'bg-gradient-to-r from-yellow-900/20 to-orange-900/20',
    border: 'border-yellow-600',
    text: 'text-yellow-500',
    badge: 'bg-yellow-600 text-black',
  },
  C: {
    bg: 'bg-zinc-900/50',
    border: 'border-zinc-600',
    text: 'text-zinc-400',
    badge: 'bg-zinc-600 text-white',
  },
  X: {
    bg: 'bg-red-900/30',
    border: 'border-red-500',
    text: 'text-red-400',
    badge: 'bg-red-500 text-white',
  },
};

const TIER_OPTIONS = ['All', 'S', 'A+', 'A', 'B', 'C'];

// ============================================================================
// Utility Functions
// ============================================================================

function getTierStyle(tier: string) {
  return TIER_COLORS[tier] || TIER_COLORS.C;
}

function formatTimestamp(timestamp: string | null): string {
  if (!timestamp) return 'N/A';
  try {
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return timestamp;
  }
}

function formatRelativeTime(timestamp: string | null): string {
  if (!timestamp) return '';
  try {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  } catch {
    return '';
  }
}

function formatNumber(value: number | null, decimals: number = 2): string {
  if (value === null || value === undefined) return 'N/A';
  return value.toFixed(decimals);
}

function formatPercent(value: number | null): string {
  if (value === null || value === undefined) return 'N/A';
  const sign = value >= 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}%`;
}

function formatVolume(value: number | null): string {
  if (value === null || value === undefined) return 'N/A';
  if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
  return value.toString();
}

// ============================================================================
// Components
// ============================================================================

interface FilterBarProps {
  tickerFilter: string;
  setTickerFilter: (value: string) => void;
  tierFilter: string;
  setTierFilter: (value: string) => void;
  onRefresh: () => void;
  isLoading: boolean;
}

function FilterBar({
  tickerFilter,
  setTickerFilter,
  tierFilter,
  setTierFilter,
  onRefresh,
  isLoading,
}: FilterBarProps) {
  return (
    <div className="flex flex-wrap items-center gap-4 mb-6 p-4 bg-zinc-800/50 rounded-lg border border-zinc-700">
      {/* Ticker Filter */}
      <div className="flex items-center gap-2">
        <label htmlFor="ticker-filter" className="text-sm text-zinc-400">
          Ticker:
        </label>
        <input
          id="ticker-filter"
          type="text"
          value={tickerFilter}
          onChange={(e) => setTickerFilter(e.target.value.toUpperCase())}
          placeholder="e.g., SPY"
          className="px-3 py-1.5 bg-zinc-900 border border-zinc-600 rounded text-white text-sm w-24 focus:outline-none focus:border-blue-500"
        />
      </div>

      {/* Tier Filter */}
      <div className="flex items-center gap-2">
        <label htmlFor="tier-filter" className="text-sm text-zinc-400">
          Tier:
        </label>
        <select
          id="tier-filter"
          value={tierFilter}
          onChange={(e) => setTierFilter(e.target.value)}
          className="px-3 py-1.5 bg-zinc-900 border border-zinc-600 rounded text-white text-sm focus:outline-none focus:border-blue-500"
        >
          {TIER_OPTIONS.map((tier) => (
            <option key={tier} value={tier === 'All' ? '' : tier}>
              {tier}
            </option>
          ))}
        </select>
      </div>

      {/* Refresh Button */}
      <button
        onClick={onRefresh}
        disabled={isLoading}
        className="ml-auto px-4 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-600 disabled:cursor-not-allowed text-white text-sm rounded transition-colors flex items-center gap-2"
      >
        {isLoading ? (
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        ) : (
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
            />
          </svg>
        )}
        Refresh
      </button>
    </div>
  );
}

interface SignalCardProps {
  signal: SignalSummary;
  isExpanded: boolean;
  onToggle: () => void;
  details: SignalDetail | null;
  isLoadingDetails: boolean;
}

function SignalCard({
  signal,
  isExpanded,
  onToggle,
  details,
  isLoadingDetails,
}: SignalCardProps) {
  const tierStyle = getTierStyle(signal.tier);
  const changeColor =
    signal.change_pct === null
      ? 'text-zinc-400'
      : signal.change_pct >= 0
      ? 'text-green-400'
      : 'text-red-400';

  return (
    <div
      className={`rounded-lg border-l-4 ${tierStyle.border} ${tierStyle.bg} overflow-hidden transition-all duration-200`}
    >
      {/* Card Header - Always Visible */}
      <button
        onClick={onToggle}
        className="w-full p-4 flex items-center gap-4 text-left hover:bg-white/5 transition-colors"
      >
        {/* Tier Badge */}
        <div className={`px-2.5 py-1 rounded font-bold text-sm ${tierStyle.badge}`}>
          {signal.tier}
        </div>

        {/* Ticker & Setup */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-lg font-semibold text-white">{signal.ticker}</span>
            <span className="text-xs px-2 py-0.5 bg-zinc-700 rounded text-zinc-300">
              {signal.setup_type}
            </span>
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            {formatTimestamp(signal.created_at || signal.timestamp)}
            <span className="mx-1">-</span>
            <span className="text-zinc-400">
              {formatRelativeTime(signal.created_at || signal.timestamp)}
            </span>
          </div>
        </div>

        {/* Score */}
        <div className="text-right">
          <div className="text-2xl font-bold text-white">{signal.score.toFixed(0)}</div>
          <div className="text-xs text-zinc-500">Score</div>
        </div>

        {/* Price & Change */}
        <div className="text-right min-w-[80px]">
          <div className="text-white font-medium">
            ${formatNumber(signal.price)}
          </div>
          <div className={`text-sm ${changeColor}`}>
            {formatPercent(signal.change_pct)}
          </div>
        </div>

        {/* Expand Icon */}
        <svg
          className={`w-5 h-5 text-zinc-400 transition-transform ${
            isExpanded ? 'rotate-180' : ''
          }`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="px-4 pb-4 border-t border-zinc-700/50">
          {isLoadingDetails ? (
            <div className="py-8 text-center text-zinc-500">
              <svg className="animate-spin h-6 w-6 mx-auto mb-2" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Loading details...
            </div>
          ) : details ? (
            <div className="pt-4 space-y-4">
              {/* Quick Stats Row */}
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                <StatBox label="Volume" value={formatVolume(details.volume)} />
                <StatBox label="VWAP" value={`$${formatNumber(details.vwap)}`} />
                <StatBox label="ATM IV" value={details.atm_iv ? `${formatNumber(details.atm_iv)}%` : 'N/A'} />
                <StatBox
                  label="Call/Put"
                  value={formatNumber(details.call_put_ratio)}
                />
                <StatBox label="Sentiment" value={formatNumber(details.sentiment_score)} />
                <StatBox label="Velocity" value={formatNumber(details.social_velocity)} />
              </div>

              {/* Features Section */}
              {details.features && Object.keys(details.features).length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-zinc-300 mb-2">Features</h4>
                  <div className="bg-zinc-900/50 rounded p-3 overflow-x-auto">
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 text-sm">
                      {Object.entries(details.features).map(([key, value]) => (
                        <div key={key} className="flex justify-between gap-2">
                          <span className="text-zinc-500 truncate">{key}:</span>
                          <span className="text-zinc-300 font-mono">
                            {typeof value === 'number'
                              ? formatNumber(value, 3)
                              : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Evidence Section */}
              {details.evidence && Object.keys(details.evidence).length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-zinc-300 mb-2">Evidence</h4>
                  <div className="bg-zinc-900/50 rounded p-3 space-y-2">
                    {Object.entries(details.evidence).map(([key, value]) => (
                      <div key={key}>
                        <span className="text-xs text-zinc-500 uppercase">{key}</span>
                        <div className="text-sm text-zinc-300 mt-0.5">
                          {Array.isArray(value) ? (
                            <ul className="list-disc list-inside space-y-0.5">
                              {value.slice(0, 5).map((item, idx) => (
                                <li key={idx} className="truncate">
                                  {String(item)}
                                </li>
                              ))}
                              {value.length > 5 && (
                                <li className="text-zinc-500">
                                  +{value.length - 5} more...
                                </li>
                              )}
                            </ul>
                          ) : typeof value === 'object' ? (
                            <pre className="text-xs overflow-x-auto">
                              {JSON.stringify(value, null, 2)}
                            </pre>
                          ) : (
                            String(value)
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Session Info */}
              {(details.session_type || details.minutes_to_close !== null) && (
                <div className="flex items-center gap-4 text-sm text-zinc-400">
                  {details.session_type && (
                    <span className="px-2 py-0.5 bg-zinc-800 rounded">
                      Session: {details.session_type}
                    </span>
                  )}
                  {details.minutes_to_close !== null && (
                    <span>
                      {formatNumber(details.minutes_to_close, 0)} min to close
                    </span>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="py-4 text-center text-zinc-500">
              Failed to load details
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface StatBoxProps {
  label: string;
  value: string;
}

function StatBox({ label, value }: StatBoxProps) {
  return (
    <div className="bg-zinc-800/50 rounded p-2 text-center">
      <div className="text-xs text-zinc-500 mb-0.5">{label}</div>
      <div className="text-sm font-medium text-white">{value}</div>
    </div>
  );
}

interface PaginationProps {
  page: number;
  totalPages: number;
  total: number;
  pageSize: number;
  onPageChange: (page: number) => void;
}

function Pagination({ page, totalPages, total, pageSize, onPageChange }: PaginationProps) {
  const start = (page - 1) * pageSize + 1;
  const end = Math.min(page * pageSize, total);

  return (
    <div className="flex items-center justify-between mt-6 py-4 border-t border-zinc-700">
      <div className="text-sm text-zinc-400">
        Showing {start}-{end} of {total} signals
      </div>
      <div className="flex items-center gap-2">
        <button
          onClick={() => onPageChange(page - 1)}
          disabled={page <= 1}
          className="px-3 py-1.5 bg-zinc-700 hover:bg-zinc-600 disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed text-white text-sm rounded transition-colors"
        >
          Previous
        </button>
        <span className="text-sm text-zinc-400 px-2">
          Page {page} of {totalPages}
        </span>
        <button
          onClick={() => onPageChange(page + 1)}
          disabled={page >= totalPages}
          className="px-3 py-1.5 bg-zinc-700 hover:bg-zinc-600 disabled:bg-zinc-800 disabled:text-zinc-600 disabled:cursor-not-allowed text-white text-sm rounded transition-colors"
        >
          Next
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export default function Signals(): React.ReactElement {
  // State
  const [signals, setSignals] = useState<SignalSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [pageSize] = useState(20);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [tickerFilter, setTickerFilter] = useState('');
  const [tierFilter, setTierFilter] = useState('');

  // Expanded card state
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [signalDetails, setSignalDetails] = useState<Record<number, SignalDetail>>({});
  const [loadingDetails, setLoadingDetails] = useState<number | null>(null);

  // Fetch signals
  const fetchSignals = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        page: page.toString(),
        page_size: pageSize.toString(),
      });

      if (tickerFilter) params.append('ticker', tickerFilter);
      if (tierFilter) params.append('tier', tierFilter);

      const response = await fetch(`${API_BASE}?${params}`);

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const data: PaginatedSignals = await response.json();
      setSignals(data.signals);
      setTotal(data.total);
      setTotalPages(data.total_pages);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch signals');
      setSignals([]);
    } finally {
      setIsLoading(false);
    }
  }, [page, pageSize, tickerFilter, tierFilter]);

  // Fetch signal details
  const fetchDetails = useCallback(async (signalId: number) => {
    if (signalDetails[signalId]) return;

    setLoadingDetails(signalId);

    try {
      const response = await fetch(`${API_BASE}/${signalId}`);

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const data: SignalDetail = await response.json();
      setSignalDetails((prev) => ({ ...prev, [signalId]: data }));
    } catch (err) {
      console.error('Failed to fetch signal details:', err);
    } finally {
      setLoadingDetails(null);
    }
  }, [signalDetails]);

  // Handle expand/collapse
  const handleToggle = useCallback(
    (signalId: number) => {
      if (expandedId === signalId) {
        setExpandedId(null);
      } else {
        setExpandedId(signalId);
        fetchDetails(signalId);
      }
    },
    [expandedId, fetchDetails]
  );

  // Handle page change
  const handlePageChange = useCallback((newPage: number) => {
    setPage(newPage);
    setExpandedId(null);
  }, []);

  // Handle filter changes - reset to page 1
  useEffect(() => {
    setPage(1);
  }, [tickerFilter, tierFilter]);

  // Fetch signals on mount and when dependencies change
  useEffect(() => {
    fetchSignals();
  }, [fetchSignals]);

  return (
    <div className="min-h-screen bg-zinc-900 text-white p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-white">Signals</h1>
          <p className="text-zinc-400 mt-1">Recent trading signals and alerts</p>
        </div>

        {/* Filters */}
        <FilterBar
          tickerFilter={tickerFilter}
          setTickerFilter={setTickerFilter}
          tierFilter={tierFilter}
          setTierFilter={setTierFilter}
          onRefresh={fetchSignals}
          isLoading={isLoading}
        />

        {/* Error State */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/30 border border-red-500 rounded-lg text-red-400">
            <div className="font-medium">Error loading signals</div>
            <div className="text-sm mt-1">{error}</div>
          </div>
        )}

        {/* Loading State */}
        {isLoading && signals.length === 0 && (
          <div className="py-12 text-center">
            <svg className="animate-spin h-8 w-8 mx-auto mb-4 text-blue-500" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            <div className="text-zinc-400">Loading signals...</div>
          </div>
        )}

        {/* Empty State */}
        {!isLoading && signals.length === 0 && !error && (
          <div className="py-12 text-center">
            <svg
              className="mx-auto h-12 w-12 text-zinc-600 mb-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
              />
            </svg>
            <div className="text-zinc-400 text-lg">No signals found</div>
            <div className="text-zinc-500 text-sm mt-1">
              {tickerFilter || tierFilter
                ? 'Try adjusting your filters'
                : 'Signals will appear here when detected'}
            </div>
          </div>
        )}

        {/* Signal Cards */}
        {signals.length > 0 && (
          <div className="space-y-3">
            {signals.map((signal) => (
              <SignalCard
                key={signal.id}
                signal={signal}
                isExpanded={expandedId === signal.id}
                onToggle={() => handleToggle(signal.id)}
                details={signalDetails[signal.id] || null}
                isLoadingDetails={loadingDetails === signal.id}
              />
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <Pagination
            page={page}
            totalPages={totalPages}
            total={total}
            pageSize={pageSize}
            onPageChange={handlePageChange}
          />
        )}
      </div>
    </div>
  );
}
