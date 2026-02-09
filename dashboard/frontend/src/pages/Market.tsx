import React, { useState, useEffect, useMemo } from 'react';

// Types
interface MarketData {
  vix: {
    current: number;
    change: number;
    changePercent: number;
    termStructure: 'contango' | 'backwardation' | 'flat';
    vix1m: number;
    vix3m: number;
  };
  spy: {
    price: number;
    change: number;
    changePercent: number;
  };
  qqq: {
    price: number;
    change: number;
    changePercent: number;
  };
  regime: {
    name: string;
    description: string;
    color: string;
  };
}

interface SessionTiming {
  isMarketOpen: boolean;
  timeToOpen: string | null;
  timeToClose: string | null;
  sessionType: 'pre-market' | 'regular' | 'after-hours' | 'closed';
  nextEvent: string;
}

// VIX Gauge Component
const VixGauge: React.FC<{ value: number; change: number }> = ({ value, change }) => {
  // VIX ranges: 0-12 (very low), 12-20 (low), 20-30 (moderate), 30-40 (high), 40+ (extreme)
  const getVixColor = (vix: number): string => {
    if (vix < 12) return '#22c55e'; // Green - very low volatility
    if (vix < 20) return '#84cc16'; // Lime - low volatility
    if (vix < 30) return '#eab308'; // Yellow - moderate
    if (vix < 40) return '#f97316'; // Orange - high
    return '#ef4444'; // Red - extreme
  };

  const getVixLabel = (vix: number): string => {
    if (vix < 12) return 'Very Low';
    if (vix < 20) return 'Low';
    if (vix < 30) return 'Moderate';
    if (vix < 40) return 'High';
    return 'Extreme';
  };

  // Calculate rotation for gauge needle (-90 to 90 degrees for 0 to 80 VIX)
  const rotation = useMemo(() => {
    const clampedValue = Math.min(Math.max(value, 0), 80);
    return -90 + (clampedValue / 80) * 180;
  }, [value]);

  const color = getVixColor(value);
  const label = getVixLabel(value);

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 200 120" className="w-full max-w-xs">
        {/* Background arc */}
        <path
          d="M 20 100 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="#374151"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* Colored segments */}
        <path
          d="M 20 100 A 80 80 0 0 1 47 43"
          fill="none"
          stroke="#22c55e"
          strokeWidth="12"
          strokeLinecap="round"
        />
        <path
          d="M 47 43 A 80 80 0 0 1 100 20"
          fill="none"
          stroke="#84cc16"
          strokeWidth="12"
        />
        <path
          d="M 100 20 A 80 80 0 0 1 153 43"
          fill="none"
          stroke="#eab308"
          strokeWidth="12"
        />
        <path
          d="M 153 43 A 80 80 0 0 1 180 100"
          fill="none"
          stroke="#f97316"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {/* Needle */}
        <g transform={`rotate(${rotation}, 100, 100)`}>
          <line
            x1="100"
            y1="100"
            x2="100"
            y2="35"
            stroke={color}
            strokeWidth="3"
            strokeLinecap="round"
          />
          <circle cx="100" cy="100" r="8" fill={color} />
        </g>
        {/* Labels */}
        <text x="20" y="115" fill="#9ca3af" fontSize="10" textAnchor="middle">0</text>
        <text x="100" y="15" fill="#9ca3af" fontSize="10" textAnchor="middle">40</text>
        <text x="180" y="115" fill="#9ca3af" fontSize="10" textAnchor="middle">80</text>
      </svg>
      <div className="text-center mt-2">
        <div className="text-4xl font-bold" style={{ color }}>{value.toFixed(2)}</div>
        <div className="text-sm text-gray-400">{label}</div>
        <div className={`text-sm mt-1 ${change >= 0 ? 'text-red-400' : 'text-green-400'}`}>
          {change >= 0 ? '+' : ''}{change.toFixed(2)} ({change >= 0 ? '+' : ''}{((change / (value - change)) * 100).toFixed(2)}%)
        </div>
      </div>
    </div>
  );
};

// Term Structure Indicator Component
const TermStructureIndicator: React.FC<{
  structure: 'contango' | 'backwardation' | 'flat';
  vix1m: number;
  vix3m: number;
}> = ({ structure, vix1m, vix3m }) => {
  const getStructureInfo = () => {
    switch (structure) {
      case 'contango':
        return {
          color: '#22c55e',
          label: 'Contango',
          description: 'Normal market conditions - VIX futures higher than spot',
          icon: (
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          )
        };
      case 'backwardation':
        return {
          color: '#ef4444',
          label: 'Backwardation',
          description: 'Elevated fear - VIX futures lower than spot (hedging demand)',
          icon: (
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
            </svg>
          )
        };
      default:
        return {
          color: '#eab308',
          label: 'Flat',
          description: 'Neutral term structure',
          icon: (
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
            </svg>
          )
        };
    }
  };

  const info = getStructureInfo();
  const spread = ((vix3m - vix1m) / vix1m * 100).toFixed(2);

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-gray-400 text-sm">Term Structure</span>
        <div className="flex items-center gap-2" style={{ color: info.color }}>
          {info.icon}
          <span className="font-semibold">{info.label}</span>
        </div>
      </div>
      <p className="text-gray-400 text-sm mb-4">{info.description}</p>
      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="bg-gray-700 rounded p-2">
          <div className="text-xs text-gray-400">VIX 1M</div>
          <div className="text-lg font-semibold text-white">{vix1m.toFixed(2)}</div>
        </div>
        <div className="bg-gray-700 rounded p-2">
          <div className="text-xs text-gray-400">Spread</div>
          <div className={`text-lg font-semibold ${Number(spread) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {spread}%
          </div>
        </div>
        <div className="bg-gray-700 rounded p-2">
          <div className="text-xs text-gray-400">VIX 3M</div>
          <div className="text-lg font-semibold text-white">{vix3m.toFixed(2)}</div>
        </div>
      </div>
    </div>
  );
};

// Market Regime Card Component
const MarketRegimeCard: React.FC<{ regime: MarketData['regime'] }> = ({ regime }) => {
  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-gray-400 text-sm">Market Regime</span>
        <div
          className="px-3 py-1 rounded-full text-sm font-semibold"
          style={{ backgroundColor: regime.color + '20', color: regime.color }}
        >
          {regime.name}
        </div>
      </div>
      <p className="text-gray-300 text-sm">{regime.description}</p>
    </div>
  );
};

// Price Card Component
const PriceCard: React.FC<{
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
}> = ({ symbol, price, change, changePercent }) => {
  const isPositive = change >= 0;

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xl font-bold text-white">{symbol}</span>
        <div className={`flex items-center gap-1 ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d={isPositive ? "M5 15l7-7 7 7" : "M19 9l-7 7-7-7"}
            />
          </svg>
          <span className="text-sm font-medium">
            {isPositive ? '+' : ''}{changePercent.toFixed(2)}%
          </span>
        </div>
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-3xl font-bold text-white">${price.toFixed(2)}</span>
        <span className={`text-sm ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
          {isPositive ? '+' : ''}{change.toFixed(2)}
        </span>
      </div>
    </div>
  );
};

// Session Timing Component
const SessionTimingCard: React.FC<{ timing: SessionTiming }> = ({ timing }) => {
  const getSessionColor = (type: SessionTiming['sessionType']): string => {
    switch (type) {
      case 'regular':
        return '#22c55e';
      case 'pre-market':
        return '#3b82f6';
      case 'after-hours':
        return '#8b5cf6';
      default:
        return '#6b7280';
    }
  };

  const getSessionLabel = (type: SessionTiming['sessionType']): string => {
    switch (type) {
      case 'regular':
        return 'Regular Session';
      case 'pre-market':
        return 'Pre-Market';
      case 'after-hours':
        return 'After Hours';
      default:
        return 'Market Closed';
    }
  };

  const color = getSessionColor(timing.sessionType);
  const label = getSessionLabel(timing.sessionType);

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <span className="text-gray-400 text-sm">Session Status</span>
        <div className="flex items-center gap-2">
          <span
            className="w-3 h-3 rounded-full animate-pulse"
            style={{ backgroundColor: color }}
          />
          <span className="font-semibold" style={{ color }}>{label}</span>
        </div>
      </div>
      <div className="space-y-3">
        {timing.isMarketOpen && timing.timeToClose && (
          <div className="flex items-center justify-between">
            <span className="text-gray-400 text-sm">Time to Close</span>
            <span className="text-white font-mono text-lg">{timing.timeToClose}</span>
          </div>
        )}
        {!timing.isMarketOpen && timing.timeToOpen && (
          <div className="flex items-center justify-between">
            <span className="text-gray-400 text-sm">Time to Open</span>
            <span className="text-white font-mono text-lg">{timing.timeToOpen}</span>
          </div>
        )}
        <div className="flex items-center justify-between pt-2 border-t border-gray-700">
          <span className="text-gray-400 text-sm">Next Event</span>
          <span className="text-gray-300 text-sm">{timing.nextEvent}</span>
        </div>
      </div>
    </div>
  );
};

// Helper function to calculate session timing
const calculateSessionTiming = (): SessionTiming => {
  const now = new Date();
  const nyTime = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const hours = nyTime.getHours();
  const minutes = nyTime.getMinutes();
  const day = nyTime.getDay();
  const currentMinutes = hours * 60 + minutes;

  const preMarketOpen = 4 * 60; // 4:00 AM ET
  const marketOpen = 9 * 60 + 30; // 9:30 AM ET
  const marketClose = 16 * 60; // 4:00 PM ET
  const afterHoursClose = 20 * 60; // 8:00 PM ET

  const formatTime = (totalMinutes: number): string => {
    const hrs = Math.floor(totalMinutes / 60);
    const mins = totalMinutes % 60;
    const secs = 0;
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Weekend check
  if (day === 0 || day === 6) {
    const daysUntilMonday = day === 0 ? 1 : 2;
    return {
      isMarketOpen: false,
      timeToOpen: `${daysUntilMonday}d ${formatTime(marketOpen)}`,
      timeToClose: null,
      sessionType: 'closed',
      nextEvent: `Market opens Monday at 9:30 AM ET`
    };
  }

  // Pre-market
  if (currentMinutes >= preMarketOpen && currentMinutes < marketOpen) {
    const minutesToOpen = marketOpen - currentMinutes;
    return {
      isMarketOpen: false,
      timeToOpen: formatTime(minutesToOpen),
      timeToClose: null,
      sessionType: 'pre-market',
      nextEvent: 'Regular session opens at 9:30 AM ET'
    };
  }

  // Regular session
  if (currentMinutes >= marketOpen && currentMinutes < marketClose) {
    const minutesToClose = marketClose - currentMinutes;
    return {
      isMarketOpen: true,
      timeToOpen: null,
      timeToClose: formatTime(minutesToClose),
      sessionType: 'regular',
      nextEvent: 'Regular session closes at 4:00 PM ET'
    };
  }

  // After hours
  if (currentMinutes >= marketClose && currentMinutes < afterHoursClose) {
    return {
      isMarketOpen: false,
      timeToOpen: null,
      timeToClose: formatTime(afterHoursClose - currentMinutes),
      sessionType: 'after-hours',
      nextEvent: 'After hours closes at 8:00 PM ET'
    };
  }

  // Before pre-market or after after-hours
  let minutesToPreMarket: number;
  if (currentMinutes < preMarketOpen) {
    minutesToPreMarket = preMarketOpen - currentMinutes;
  } else {
    // After 8 PM, calculate time until next day's pre-market
    minutesToPreMarket = (24 * 60 - currentMinutes) + preMarketOpen;
  }

  return {
    isMarketOpen: false,
    timeToOpen: formatTime(minutesToPreMarket + (marketOpen - preMarketOpen)),
    timeToClose: null,
    sessionType: 'closed',
    nextEvent: 'Pre-market opens at 4:00 AM ET'
  };
};

// Main Market Page Component
const Market: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketData>({
    vix: {
      current: 18.45,
      change: -0.87,
      changePercent: -4.5,
      termStructure: 'contango',
      vix1m: 19.23,
      vix3m: 21.56
    },
    spy: {
      price: 502.34,
      change: 3.21,
      changePercent: 0.64
    },
    qqq: {
      price: 438.92,
      change: -1.45,
      changePercent: -0.33
    },
    regime: {
      name: 'Low Volatility Bull',
      description: 'Market characterized by steady upward momentum with VIX below 20. Favorable conditions for trend-following strategies and selling premium.',
      color: '#22c55e'
    }
  });

  const [sessionTiming, setSessionTiming] = useState<SessionTiming>(calculateSessionTiming());
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  // Update session timing every second
  useEffect(() => {
    const timer = setInterval(() => {
      setSessionTiming(calculateSessionTiming());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  // Simulate data refresh (replace with actual API call)
  useEffect(() => {
    const fetchData = async () => {
      // TODO: Replace with actual API call to fetch market data
      // Example:
      // const response = await fetch('/api/market-data');
      // const data = await response.json();
      // setMarketData(data);
      setLastUpdate(new Date());
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">Market Overview</h1>
            <p className="text-gray-400 text-sm mt-1">
              Last updated: {lastUpdate.toLocaleTimeString()}
            </p>
          </div>
          <button
            onClick={() => setLastUpdate(new Date())}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Refresh
          </button>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* VIX Section */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-gray-800 rounded-lg p-6">
              <h2 className="text-lg font-semibold mb-4 text-gray-300">VIX - Fear Index</h2>
              <VixGauge value={marketData.vix.current} change={marketData.vix.change} />
            </div>
            <TermStructureIndicator
              structure={marketData.vix.termStructure}
              vix1m={marketData.vix.vix1m}
              vix3m={marketData.vix.vix3m}
            />
          </div>

          {/* Middle Section */}
          <div className="lg:col-span-1 space-y-6">
            <PriceCard
              symbol="SPY"
              price={marketData.spy.price}
              change={marketData.spy.change}
              changePercent={marketData.spy.changePercent}
            />
            <PriceCard
              symbol="QQQ"
              price={marketData.qqq.price}
              change={marketData.qqq.change}
              changePercent={marketData.qqq.changePercent}
            />
            <MarketRegimeCard regime={marketData.regime} />
          </div>

          {/* Right Section */}
          <div className="lg:col-span-1 space-y-6">
            <SessionTimingCard timing={sessionTiming} />

            {/* Quick Stats */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-sm text-gray-400 mb-3">Quick Stats</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">SPY/QQQ Ratio</span>
                  <span className="text-white font-mono">
                    {(marketData.spy.price / marketData.qqq.price).toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">VIX/VIX3M Ratio</span>
                  <span className={`font-mono ${marketData.vix.current / marketData.vix.vix3m > 1 ? 'text-red-400' : 'text-green-400'}`}>
                    {(marketData.vix.current / marketData.vix.vix3m).toFixed(4)}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">Fear Level</span>
                  <div className="flex items-center gap-2">
                    <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all"
                        style={{
                          width: `${Math.min(100, (marketData.vix.current / 40) * 100)}%`,
                          backgroundColor: marketData.vix.current < 20 ? '#22c55e' : marketData.vix.current < 30 ? '#eab308' : '#ef4444'
                        }}
                      />
                    </div>
                    <span className="text-white font-mono text-sm">
                      {Math.min(100, Math.round((marketData.vix.current / 40) * 100))}%
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Regime History Mini */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-sm text-gray-400 mb-3">Regime Signals</h3>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-green-400"></span>
                  <span className="text-sm text-gray-300">VIX below 20 threshold</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-green-400"></span>
                  <span className="text-sm text-gray-300">Term structure in contango</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-green-400"></span>
                  <span className="text-sm text-gray-300">SPY positive on day</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-yellow-400"></span>
                  <span className="text-sm text-gray-300">QQQ slightly negative</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Market;
