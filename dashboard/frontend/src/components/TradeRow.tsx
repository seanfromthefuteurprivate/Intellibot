import React, { useState } from 'react';

export interface Trade {
  id: string;
  symbol: string;
  direction: 'long' | 'short';
  entryPrice: number;
  exitPrice: number | null;
  entryTime: Date | string;
  exitTime: Date | string | null;
  quantity: number;
  pnl: number | null;
  pnlPercent: number | null;
  status: 'open' | 'closed' | 'pending';
  strategy?: string;
  notes?: string;
  fees?: number;
  stopLoss?: number;
  takeProfit?: number;
}

interface TradeRowProps {
  trade: Trade;
  onSelect?: (trade: Trade) => void;
  showDetails?: boolean;
}

const formatDuration = (start: Date | string, end: Date | string | null): string => {
  const startDate = typeof start === 'string' ? new Date(start) : start;
  const endDate = end ? (typeof end === 'string' ? new Date(end) : end) : new Date();

  const diffMs = endDate.getTime() - startDate.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffDays > 0) {
    return `${diffDays}d ${diffHours % 24}h`;
  }
  if (diffHours > 0) {
    return `${diffHours}h ${diffMinutes % 60}m`;
  }
  if (diffMinutes > 0) {
    return `${diffMinutes}m ${diffSeconds % 60}s`;
  }
  return `${diffSeconds}s`;
};

const formatPrice = (price: number): string => {
  return price.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
};

const formatPnL = (pnl: number): string => {
  const prefix = pnl >= 0 ? '+' : '';
  return `${prefix}${formatPrice(pnl)}`;
};

const formatPercent = (percent: number): string => {
  const prefix = percent >= 0 ? '+' : '';
  return `${prefix}${percent.toFixed(2)}%`;
};

const formatDateTime = (date: Date | string): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

export const TradeRow: React.FC<TradeRowProps> = ({
  trade,
  onSelect,
  showDetails: initialShowDetails = false,
}) => {
  const [isExpanded, setIsExpanded] = useState(initialShowDetails);

  const isWin = trade.pnl !== null && trade.pnl > 0;
  const isLoss = trade.pnl !== null && trade.pnl < 0;
  const isOpen = trade.status === 'open';
  const isPending = trade.status === 'pending';

  const getRowBackgroundColor = (): string => {
    if (isPending) return 'bg-gray-50 dark:bg-gray-800';
    if (isOpen) return 'bg-blue-50 dark:bg-blue-900/20';
    if (isWin) return 'bg-green-50 dark:bg-green-900/20';
    if (isLoss) return 'bg-red-50 dark:bg-red-900/20';
    return 'bg-white dark:bg-gray-900';
  };

  const getPnLColor = (): string => {
    if (trade.pnl === null) return 'text-gray-500';
    if (isWin) return 'text-green-600 dark:text-green-400';
    if (isLoss) return 'text-red-600 dark:text-red-400';
    return 'text-gray-600 dark:text-gray-400';
  };

  const getStatusBadge = (): React.ReactNode => {
    const baseClasses = 'px-2 py-0.5 text-xs font-medium rounded-full';

    if (isPending) {
      return (
        <span className={`${baseClasses} bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300`}>
          Pending
        </span>
      );
    }
    if (isOpen) {
      return (
        <span className={`${baseClasses} bg-blue-200 text-blue-700 dark:bg-blue-800 dark:text-blue-300`}>
          Open
        </span>
      );
    }
    if (isWin) {
      return (
        <span className={`${baseClasses} bg-green-200 text-green-700 dark:bg-green-800 dark:text-green-300`}>
          Win
        </span>
      );
    }
    if (isLoss) {
      return (
        <span className={`${baseClasses} bg-red-200 text-red-700 dark:bg-red-800 dark:text-red-300`}>
          Loss
        </span>
      );
    }
    return (
      <span className={`${baseClasses} bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300`}>
        Break-even
      </span>
    );
  };

  const getDirectionBadge = (): React.ReactNode => {
    const isLong = trade.direction === 'long';
    const baseClasses = 'px-2 py-0.5 text-xs font-medium rounded';

    return (
      <span
        className={`${baseClasses} ${
          isLong
            ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300'
            : 'bg-rose-100 text-rose-700 dark:bg-rose-900 dark:text-rose-300'
        }`}
      >
        {isLong ? 'LONG' : 'SHORT'}
      </span>
    );
  };

  const handleRowClick = () => {
    setIsExpanded(!isExpanded);
    if (onSelect) {
      onSelect(trade);
    }
  };

  return (
    <>
      {/* Main Row */}
      <tr
        onClick={handleRowClick}
        className={`${getRowBackgroundColor()} hover:bg-opacity-80 cursor-pointer transition-colors border-b border-gray-200 dark:border-gray-700`}
      >
        {/* Symbol & Direction */}
        <td className="px-4 py-3 whitespace-nowrap">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-gray-900 dark:text-white">
              {trade.symbol}
            </span>
            {getDirectionBadge()}
          </div>
        </td>

        {/* Entry Price */}
        <td className="px-4 py-3 whitespace-nowrap text-gray-700 dark:text-gray-300">
          {formatPrice(trade.entryPrice)}
        </td>

        {/* Exit Price */}
        <td className="px-4 py-3 whitespace-nowrap text-gray-700 dark:text-gray-300">
          {trade.exitPrice !== null ? formatPrice(trade.exitPrice) : '-'}
        </td>

        {/* Quantity */}
        <td className="px-4 py-3 whitespace-nowrap text-gray-700 dark:text-gray-300">
          {trade.quantity}
        </td>

        {/* Duration */}
        <td className="px-4 py-3 whitespace-nowrap text-gray-600 dark:text-gray-400">
          {formatDuration(trade.entryTime, trade.exitTime)}
        </td>

        {/* P&L */}
        <td className="px-4 py-3 whitespace-nowrap">
          <div className="flex flex-col">
            <span className={`font-semibold ${getPnLColor()}`}>
              {trade.pnl !== null ? formatPnL(trade.pnl) : '-'}
            </span>
            {trade.pnlPercent !== null && (
              <span className={`text-xs ${getPnLColor()}`}>
                {formatPercent(trade.pnlPercent)}
              </span>
            )}
          </div>
        </td>

        {/* Status */}
        <td className="px-4 py-3 whitespace-nowrap">
          {getStatusBadge()}
        </td>

        {/* Expand Icon */}
        <td className="px-4 py-3 whitespace-nowrap text-gray-400">
          <svg
            className={`w-5 h-5 transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </td>
      </tr>

      {/* Expanded Details Row */}
      {isExpanded && (
        <tr className={`${getRowBackgroundColor()} border-b border-gray-200 dark:border-gray-700`}>
          <td colSpan={8} className="px-4 py-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              {/* Entry Time */}
              <div>
                <span className="block text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wide">
                  Entry Time
                </span>
                <span className="text-gray-900 dark:text-white">
                  {formatDateTime(trade.entryTime)}
                </span>
              </div>

              {/* Exit Time */}
              <div>
                <span className="block text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wide">
                  Exit Time
                </span>
                <span className="text-gray-900 dark:text-white">
                  {trade.exitTime ? formatDateTime(trade.exitTime) : 'Still open'}
                </span>
              </div>

              {/* Strategy */}
              {trade.strategy && (
                <div>
                  <span className="block text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wide">
                    Strategy
                  </span>
                  <span className="text-gray-900 dark:text-white">
                    {trade.strategy}
                  </span>
                </div>
              )}

              {/* Fees */}
              {trade.fees !== undefined && (
                <div>
                  <span className="block text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wide">
                    Fees
                  </span>
                  <span className="text-gray-900 dark:text-white">
                    {formatPrice(trade.fees)}
                  </span>
                </div>
              )}

              {/* Stop Loss */}
              {trade.stopLoss !== undefined && (
                <div>
                  <span className="block text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wide">
                    Stop Loss
                  </span>
                  <span className="text-red-600 dark:text-red-400">
                    {formatPrice(trade.stopLoss)}
                  </span>
                </div>
              )}

              {/* Take Profit */}
              {trade.takeProfit !== undefined && (
                <div>
                  <span className="block text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wide">
                    Take Profit
                  </span>
                  <span className="text-green-600 dark:text-green-400">
                    {formatPrice(trade.takeProfit)}
                  </span>
                </div>
              )}

              {/* Trade ID */}
              <div>
                <span className="block text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wide">
                  Trade ID
                </span>
                <span className="text-gray-900 dark:text-white font-mono text-xs">
                  {trade.id}
                </span>
              </div>

              {/* Net P&L (after fees) */}
              {trade.pnl !== null && trade.fees !== undefined && (
                <div>
                  <span className="block text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wide">
                    Net P&L
                  </span>
                  <span className={getPnLColor()}>
                    {formatPnL(trade.pnl - trade.fees)}
                  </span>
                </div>
              )}
            </div>

            {/* Notes Section */}
            {trade.notes && (
              <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                <span className="block text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wide mb-1">
                  Notes
                </span>
                <p className="text-gray-700 dark:text-gray-300 text-sm">
                  {trade.notes}
                </p>
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
};

export default TradeRow;
