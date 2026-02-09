import React, { useState, useEffect, useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
} from 'recharts';

// Types
interface RiskData {
  killSwitch: {
    active: boolean;
    activatedAt: string | null;
    reason: string | null;
  };
  cooldown: {
    active: boolean;
    endsAt: string | null;
    remainingSeconds: number;
  };
  dailyLoss: {
    limit: number;
    current: number;
    percentage: number;
  };
  positionLimits: {
    maxPositions: number;
    currentPositions: number;
    maxPositionSize: number;
    currentLargestPosition: number;
  };
  sectorExposure: SectorExposure[];
  consecutiveLosses: {
    current: number;
    max: number;
  };
  winRateHistory: WinRateDataPoint[];
}

interface SectorExposure {
  sector: string;
  exposure: number;
  limit: number;
  percentage: number;
}

interface WinRateDataPoint {
  date: string;
  winRate: number;
  trades: number;
}

// Mock data for demonstration
const mockRiskData: RiskData = {
  killSwitch: {
    active: false,
    activatedAt: null,
    reason: null,
  },
  cooldown: {
    active: true,
    endsAt: new Date(Date.now() + 300000).toISOString(),
    remainingSeconds: 300,
  },
  dailyLoss: {
    limit: 500,
    current: 127.50,
    percentage: 25.5,
  },
  positionLimits: {
    maxPositions: 10,
    currentPositions: 4,
    maxPositionSize: 5000,
    currentLargestPosition: 2350,
  },
  sectorExposure: [
    { sector: 'Technology', exposure: 3500, limit: 5000, percentage: 70 },
    { sector: 'Healthcare', exposure: 1200, limit: 4000, percentage: 30 },
    { sector: 'Finance', exposure: 800, limit: 3000, percentage: 26.7 },
    { sector: 'Energy', exposure: 500, limit: 2500, percentage: 20 },
    { sector: 'Consumer', exposure: 0, limit: 3500, percentage: 0 },
  ],
  consecutiveLosses: {
    current: 2,
    max: 5,
  },
  winRateHistory: [
    { date: '2024-01-01', winRate: 62, trades: 15 },
    { date: '2024-01-02', winRate: 58, trades: 12 },
    { date: '2024-01-03', winRate: 71, trades: 14 },
    { date: '2024-01-04', winRate: 55, trades: 18 },
    { date: '2024-01-05', winRate: 68, trades: 16 },
    { date: '2024-01-06', winRate: 64, trades: 11 },
    { date: '2024-01-07', winRate: 72, trades: 13 },
    { date: '2024-01-08', winRate: 59, trades: 17 },
    { date: '2024-01-09', winRate: 66, trades: 15 },
    { date: '2024-01-10', winRate: 61, trades: 14 },
  ],
};

// Utility Components
interface CardProps {
  children: React.ReactNode;
  className?: string;
}

const Card: React.FC<CardProps> = ({ children, className = '' }) => (
  <div className={`bg-white rounded-xl border border-gray-200 shadow-sm ${className}`}>
    {children}
  </div>
);

const CardHeader: React.FC<CardProps> = ({ children, className = '' }) => (
  <div className={`flex flex-col space-y-1.5 p-6 ${className}`}>{children}</div>
);

const CardTitle: React.FC<CardProps> = ({ children, className = '' }) => (
  <h3 className={`text-lg font-semibold leading-none tracking-tight ${className}`}>
    {children}
  </h3>
);

const CardContent: React.FC<CardProps> = ({ children, className = '' }) => (
  <div className={`p-6 pt-0 ${className}`}>{children}</div>
);

interface ProgressBarProps {
  value: number;
  max?: number;
  colorClass?: string;
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  colorClass = 'bg-blue-500',
  showLabel = false,
  size = 'md',
}) => {
  const percentage = Math.min((value / max) * 100, 100);
  const heightClass = size === 'sm' ? 'h-2' : size === 'lg' ? 'h-6' : 'h-4';

  return (
    <div className="w-full">
      <div className={`w-full bg-gray-200 rounded-full ${heightClass} overflow-hidden`}>
        <div
          className={`${colorClass} ${heightClass} rounded-full transition-all duration-500 ease-out`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      {showLabel && (
        <div className="flex justify-between mt-1 text-sm text-gray-600">
          <span>{value.toFixed(2)}</span>
          <span>{max.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
};

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'success' | 'danger' | 'warning' | 'info' | 'neutral';
}

const Badge: React.FC<BadgeProps> = ({ children, variant = 'neutral' }) => {
  const variantClasses = {
    success: 'bg-green-100 text-green-800 border-green-200',
    danger: 'bg-red-100 text-red-800 border-red-200',
    warning: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    info: 'bg-blue-100 text-blue-800 border-blue-200',
    neutral: 'bg-gray-100 text-gray-800 border-gray-200',
  };

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${variantClasses[variant]}`}
    >
      {children}
    </span>
  );
};

// Kill Switch Component
interface KillSwitchIndicatorProps {
  active: boolean;
  activatedAt: string | null;
  reason: string | null;
  onToggle?: () => void;
}

const KillSwitchIndicator: React.FC<KillSwitchIndicatorProps> = ({
  active,
  activatedAt,
  reason,
  onToggle,
}) => {
  return (
    <Card className="relative overflow-hidden">
      <div
        className={`absolute inset-0 opacity-10 ${active ? 'bg-red-500' : 'bg-green-500'}`}
      />
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Kill Switch</span>
          <Badge variant={active ? 'danger' : 'success'}>
            {active ? 'ACTIVATED' : 'INACTIVE'}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center py-6">
          <button
            onClick={onToggle}
            className={`relative w-32 h-32 rounded-full border-8 transition-all duration-300 transform hover:scale-105 focus:outline-none focus:ring-4 focus:ring-offset-2 ${
              active
                ? 'bg-red-500 border-red-600 focus:ring-red-300 shadow-lg shadow-red-500/50'
                : 'bg-green-500 border-green-600 focus:ring-green-300 shadow-lg shadow-green-500/50'
            }`}
            aria-label={active ? 'Deactivate kill switch' : 'Activate kill switch'}
          >
            <div className="absolute inset-0 flex items-center justify-center">
              <svg
                className="w-16 h-16 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                {active ? (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636"
                  />
                ) : (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                )}
              </svg>
            </div>
            <div
              className={`absolute -inset-1 rounded-full animate-ping opacity-25 ${
                active ? 'bg-red-400' : 'hidden'
              }`}
            />
          </button>
        </div>
        {active && activatedAt && (
          <div className="mt-4 p-3 bg-red-50 rounded-lg border border-red-200">
            <p className="text-sm text-red-700">
              <strong>Activated:</strong> {new Date(activatedAt).toLocaleString()}
            </p>
            {reason && (
              <p className="text-sm text-red-700 mt-1">
                <strong>Reason:</strong> {reason}
              </p>
            )}
          </div>
        )}
        <p className="mt-4 text-center text-sm text-gray-500">
          {active
            ? 'All trading is currently halted. Click to resume.'
            : 'Trading is active. Click to halt all trading.'}
        </p>
      </CardContent>
    </Card>
  );
};

// Cooldown Status Component
interface CooldownStatusProps {
  active: boolean;
  remainingSeconds: number;
}

const CooldownStatus: React.FC<CooldownStatusProps> = ({ active, remainingSeconds }) => {
  const [countdown, setCountdown] = useState(remainingSeconds);

  useEffect(() => {
    if (!active || countdown <= 0) return;

    const timer = setInterval(() => {
      setCountdown((prev) => Math.max(0, prev - 1));
    }, 1000);

    return () => clearInterval(timer);
  }, [active, countdown]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Cooldown Status</span>
          <Badge variant={active ? 'warning' : 'success'}>
            {active ? 'COOLING DOWN' : 'READY'}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center py-4">
          {active ? (
            <>
              <div className="relative w-24 h-24">
                <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="#e5e7eb"
                    strokeWidth="8"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="#f59e0b"
                    strokeWidth="8"
                    strokeLinecap="round"
                    strokeDasharray={`${(countdown / remainingSeconds) * 283} 283`}
                    className="transition-all duration-1000"
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl font-mono font-bold text-gray-800">
                    {formatTime(countdown)}
                  </span>
                </div>
              </div>
              <p className="mt-4 text-sm text-gray-500">
                Trading will resume after cooldown period
              </p>
            </>
          ) : (
            <>
              <div className="w-24 h-24 rounded-full bg-green-100 flex items-center justify-center">
                <svg
                  className="w-12 h-12 text-green-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 13l4 4L19 7"
                  />
                </svg>
              </div>
              <p className="mt-4 text-sm text-gray-500">System ready for trading</p>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

// Daily Loss Limit Component
interface DailyLossLimitProps {
  limit: number;
  current: number;
  percentage: number;
}

const DailyLossLimit: React.FC<DailyLossLimitProps> = ({ limit, current, percentage }) => {
  const getColorClass = (pct: number): string => {
    if (pct >= 80) return 'bg-red-500';
    if (pct >= 50) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getStatusVariant = (pct: number): 'success' | 'warning' | 'danger' => {
    if (pct >= 80) return 'danger';
    if (pct >= 50) return 'warning';
    return 'success';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Daily Loss Limit</span>
          <Badge variant={getStatusVariant(percentage)}>
            {percentage.toFixed(1)}% Used
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">Current Loss</span>
            <span className="font-semibold text-red-600">-${current.toFixed(2)}</span>
          </div>
          <ProgressBar value={percentage} colorClass={getColorClass(percentage)} size="lg" />
          <div className="flex justify-between text-sm">
            <span className="text-gray-500">Remaining</span>
            <span className="font-semibold text-green-600">
              ${(limit - current).toFixed(2)}
            </span>
          </div>
          <div className="pt-2 border-t border-gray-100">
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Daily Limit</span>
              <span className="text-sm font-medium">${limit.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Position Limits Component
interface PositionLimitsProps {
  maxPositions: number;
  currentPositions: number;
  maxPositionSize: number;
  currentLargestPosition: number;
}

const PositionLimits: React.FC<PositionLimitsProps> = ({
  maxPositions,
  currentPositions,
  maxPositionSize,
  currentLargestPosition,
}) => {
  const positionPercentage = (currentPositions / maxPositions) * 100;
  const sizePercentage = (currentLargestPosition / maxPositionSize) * 100;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Position Limits</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          <div>
            <div className="flex justify-between mb-2">
              <span className="text-sm text-gray-500">Open Positions</span>
              <span className="text-sm font-medium">
                {currentPositions} / {maxPositions}
              </span>
            </div>
            <ProgressBar
              value={positionPercentage}
              colorClass={positionPercentage >= 80 ? 'bg-yellow-500' : 'bg-blue-500'}
            />
          </div>
          <div>
            <div className="flex justify-between mb-2">
              <span className="text-sm text-gray-500">Largest Position Size</span>
              <span className="text-sm font-medium">
                ${currentLargestPosition.toLocaleString()} / ${maxPositionSize.toLocaleString()}
              </span>
            </div>
            <ProgressBar
              value={sizePercentage}
              colorClass={sizePercentage >= 80 ? 'bg-yellow-500' : 'bg-blue-500'}
            />
          </div>
          <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-100">
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <p className="text-2xl font-bold text-gray-800">{currentPositions}</p>
              <p className="text-xs text-gray-500">Active Positions</p>
            </div>
            <div className="text-center p-3 bg-gray-50 rounded-lg">
              <p className="text-2xl font-bold text-gray-800">
                {maxPositions - currentPositions}
              </p>
              <p className="text-xs text-gray-500">Available Slots</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Sector Exposure Component
interface SectorExposureProps {
  sectors: SectorExposure[];
}

const SectorExposureCard: React.FC<SectorExposureProps> = ({ sectors }) => {
  const getSectorColor = (sector: string): string => {
    const colors: Record<string, string> = {
      Technology: 'bg-blue-500',
      Healthcare: 'bg-green-500',
      Finance: 'bg-purple-500',
      Energy: 'bg-orange-500',
      Consumer: 'bg-pink-500',
      Industrial: 'bg-indigo-500',
      Materials: 'bg-teal-500',
      Utilities: 'bg-cyan-500',
    };
    return colors[sector] || 'bg-gray-500';
  };

  const getWarningLevel = (percentage: number): 'success' | 'warning' | 'danger' => {
    if (percentage >= 90) return 'danger';
    if (percentage >= 70) return 'warning';
    return 'success';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sector Exposure Limits</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {sectors.map((sector) => (
            <div key={sector.sector} className="space-y-2">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${getSectorColor(sector.sector)}`} />
                  <span className="text-sm font-medium">{sector.sector}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-sm text-gray-500">
                    ${sector.exposure.toLocaleString()} / ${sector.limit.toLocaleString()}
                  </span>
                  <Badge variant={getWarningLevel(sector.percentage)}>
                    {sector.percentage.toFixed(0)}%
                  </Badge>
                </div>
              </div>
              <ProgressBar
                value={sector.percentage}
                colorClass={
                  sector.percentage >= 90
                    ? 'bg-red-500'
                    : sector.percentage >= 70
                    ? 'bg-yellow-500'
                    : getSectorColor(sector.sector)
                }
                size="sm"
              />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

// Consecutive Losses Component
interface ConsecutiveLossesProps {
  current: number;
  max: number;
}

const ConsecutiveLosses: React.FC<ConsecutiveLossesProps> = ({ current, max }) => {
  const percentage = (current / max) * 100;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Consecutive Losses</span>
          <Badge variant={current >= max - 1 ? 'danger' : current >= max / 2 ? 'warning' : 'success'}>
            {current} / {max}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center py-4">
          <div className="flex gap-2">
            {Array.from({ length: max }).map((_, index) => (
              <div
                key={index}
                className={`w-8 h-8 rounded-full flex items-center justify-center border-2 transition-all ${
                  index < current
                    ? 'bg-red-500 border-red-600 text-white'
                    : 'bg-gray-100 border-gray-200 text-gray-400'
                }`}
              >
                {index < current ? (
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                      clipRule="evenodd"
                    />
                  </svg>
                ) : (
                  <span className="text-xs font-medium">{index + 1}</span>
                )}
              </div>
            ))}
          </div>
        </div>
        <div className="mt-4">
          <ProgressBar
            value={percentage}
            colorClass={
              percentage >= 80 ? 'bg-red-500' : percentage >= 50 ? 'bg-yellow-500' : 'bg-green-500'
            }
          />
        </div>
        <p className="mt-4 text-center text-sm text-gray-500">
          {current === 0
            ? 'No consecutive losses. Keep it up!'
            : current >= max
            ? 'Maximum consecutive losses reached. Trading paused.'
            : `${max - current} more consecutive losses until trading is paused.`}
        </p>
      </CardContent>
    </Card>
  );
};

// Win Rate Chart Component
interface WinRateChartProps {
  data: WinRateDataPoint[];
}

const WinRateChart: React.FC<WinRateChartProps> = ({ data }) => {
  const averageWinRate = useMemo(() => {
    if (data.length === 0) return 0;
    return data.reduce((acc, d) => acc + d.winRate, 0) / data.length;
  }, [data]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Win Rate History</span>
          <div className="flex items-center gap-2">
            <Badge variant={averageWinRate >= 55 ? 'success' : averageWinRate >= 45 ? 'warning' : 'danger'}>
              Avg: {averageWinRate.toFixed(1)}%
            </Badge>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="winRateGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 12, fill: '#6b7280' }}
                tickFormatter={(value) => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
              />
              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 12, fill: '#6b7280' }}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                }}
                formatter={(value: number, name: string) => [
                  name === 'winRate' ? `${value}%` : value,
                  name === 'winRate' ? 'Win Rate' : 'Trades',
                ]}
                labelFormatter={(label) =>
                  new Date(label).toLocaleDateString('en-US', {
                    weekday: 'short',
                    month: 'short',
                    day: 'numeric',
                  })
                }
              />
              <Legend
                formatter={(value) => (value === 'winRate' ? 'Win Rate' : 'Trades')}
              />
              <Area
                type="monotone"
                dataKey="winRate"
                stroke="#10b981"
                strokeWidth={2}
                fill="url(#winRateGradient)"
                dot={{ fill: '#10b981', strokeWidth: 2 }}
                activeDot={{ r: 6, fill: '#10b981' }}
              />
              <Line
                type="monotone"
                dataKey="trades"
                stroke="#6366f1"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={{ fill: '#6366f1', strokeWidth: 2, r: 3 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 grid grid-cols-3 gap-4 pt-4 border-t border-gray-100">
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">
              {Math.max(...data.map((d) => d.winRate))}%
            </p>
            <p className="text-xs text-gray-500">Best Day</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-800">
              {averageWinRate.toFixed(1)}%
            </p>
            <p className="text-xs text-gray-500">Average</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-red-600">
              {Math.min(...data.map((d) => d.winRate))}%
            </p>
            <p className="text-xs text-gray-500">Worst Day</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Main Risk Page Component
const Risk: React.FC = () => {
  const [riskData, setRiskData] = useState<RiskData>(mockRiskData);
  const [loading, setLoading] = useState(false);

  const handleKillSwitchToggle = () => {
    setRiskData((prev) => ({
      ...prev,
      killSwitch: {
        active: !prev.killSwitch.active,
        activatedAt: !prev.killSwitch.active ? new Date().toISOString() : null,
        reason: !prev.killSwitch.active ? 'Manual activation' : null,
      },
    }));
  };

  const refreshData = async () => {
    setLoading(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Risk Management</h1>
            <p className="mt-1 text-sm text-gray-500">
              Monitor and control trading risk parameters
            </p>
          </div>
          <button
            onClick={refreshData}
            disabled={loading}
            className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-lg shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
          >
            {loading ? (
              <svg
                className="animate-spin -ml-1 mr-2 h-4 w-4 text-gray-500"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
            ) : (
              <svg
                className="-ml-1 mr-2 h-4 w-4 text-gray-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
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

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {/* Kill Switch - Full width on small screens, span 1 on larger */}
          <div className="lg:col-span-1">
            <KillSwitchIndicator
              active={riskData.killSwitch.active}
              activatedAt={riskData.killSwitch.activatedAt}
              reason={riskData.killSwitch.reason}
              onToggle={handleKillSwitchToggle}
            />
          </div>

          {/* Cooldown Status */}
          <div className="lg:col-span-1">
            <CooldownStatus
              active={riskData.cooldown.active}
              remainingSeconds={riskData.cooldown.remainingSeconds}
            />
          </div>

          {/* Daily Loss Limit */}
          <div className="lg:col-span-1">
            <DailyLossLimit
              limit={riskData.dailyLoss.limit}
              current={riskData.dailyLoss.current}
              percentage={riskData.dailyLoss.percentage}
            />
          </div>

          {/* Position Limits */}
          <div className="lg:col-span-1">
            <PositionLimits
              maxPositions={riskData.positionLimits.maxPositions}
              currentPositions={riskData.positionLimits.currentPositions}
              maxPositionSize={riskData.positionLimits.maxPositionSize}
              currentLargestPosition={riskData.positionLimits.currentLargestPosition}
            />
          </div>

          {/* Consecutive Losses */}
          <div className="lg:col-span-1">
            <ConsecutiveLosses
              current={riskData.consecutiveLosses.current}
              max={riskData.consecutiveLosses.max}
            />
          </div>

          {/* Sector Exposure - Takes more space */}
          <div className="lg:col-span-1 xl:row-span-1">
            <SectorExposureCard sectors={riskData.sectorExposure} />
          </div>

          {/* Win Rate Chart - Full width */}
          <div className="lg:col-span-2 xl:col-span-3">
            <WinRateChart data={riskData.winRateHistory} />
          </div>
        </div>

        {/* Footer Stats */}
        <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <p className="text-sm text-gray-500">Total Risk Score</p>
                <p className="text-3xl font-bold text-gray-800">72</p>
                <Badge variant="warning">Moderate</Badge>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <p className="text-sm text-gray-500">Today's Trades</p>
                <p className="text-3xl font-bold text-gray-800">14</p>
                <Badge variant="info">Active</Badge>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <p className="text-sm text-gray-500">Total Exposure</p>
                <p className="text-3xl font-bold text-gray-800">$6,000</p>
                <Badge variant="success">Within Limits</Badge>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <div className="text-center">
                <p className="text-sm text-gray-500">Risk Events Today</p>
                <p className="text-3xl font-bold text-gray-800">2</p>
                <Badge variant="warning">Warnings</Badge>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Risk;
