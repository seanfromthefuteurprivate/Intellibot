import React, { useState, useEffect, useRef, useCallback } from 'react';

// Event Types
type EventType = 'trade' | 'signal' | 'regime_change' | 'risk_alert';
type EventSeverity = 'info' | 'success' | 'warning' | 'critical';

interface TradingEvent {
  id: string;
  timestamp: Date;
  type: EventType;
  severity: EventSeverity;
  title: string;
  description: string;
  metadata?: Record<string, unknown>;
}

// Icon Components
const TradeIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
  </svg>
);

const SignalIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
  </svg>
);

const RegimeIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <path d="M12 6v6l4 2" />
  </svg>
);

const RiskAlertIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

const PauseIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <rect x="6" y="4" width="4" height="16" />
    <rect x="14" y="4" width="4" height="16" />
  </svg>
);

const PlayIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <polygon points="5,3 19,12 5,21" />
  </svg>
);

// Helper Functions
const getEventIcon = (type: EventType): React.FC<{ className?: string }> => {
  const icons: Record<EventType, React.FC<{ className?: string }>> = {
    trade: TradeIcon,
    signal: SignalIcon,
    regime_change: RegimeIcon,
    risk_alert: RiskAlertIcon,
  };
  return icons[type];
};

const getEventTypeLabel = (type: EventType): string => {
  const labels: Record<EventType, string> = {
    trade: 'Trade',
    signal: 'Signal',
    regime_change: 'Regime Change',
    risk_alert: 'Risk Alert',
  };
  return labels[type];
};

const getSeverityColors = (severity: EventSeverity): { bg: string; border: string; text: string; icon: string } => {
  const colors: Record<EventSeverity, { bg: string; border: string; text: string; icon: string }> = {
    info: {
      bg: 'bg-slate-800/50',
      border: 'border-slate-600',
      text: 'text-slate-300',
      icon: 'text-slate-400',
    },
    success: {
      bg: 'bg-emerald-900/30',
      border: 'border-emerald-500/50',
      text: 'text-emerald-300',
      icon: 'text-emerald-400',
    },
    warning: {
      bg: 'bg-amber-900/30',
      border: 'border-amber-500/50',
      text: 'text-amber-300',
      icon: 'text-amber-400',
    },
    critical: {
      bg: 'bg-red-900/30',
      border: 'border-red-500/50',
      text: 'text-red-300',
      icon: 'text-red-400',
    },
  };
  return colors[severity];
};

const formatTimestamp = (date: Date): string => {
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });
};

const formatDateHeader = (date: Date): string => {
  return date.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
  });
};

// Event Card Component
interface EventCardProps {
  event: TradingEvent;
}

const EventCard: React.FC<EventCardProps> = ({ event }) => {
  const colors = getSeverityColors(event.severity);
  const Icon = getEventIcon(event.type);

  return (
    <div
      className={`
        ${colors.bg} ${colors.border}
        border rounded-lg p-4 mb-3
        transition-all duration-300 ease-out
        hover:scale-[1.01] hover:shadow-lg
        animate-slide-in
      `}
    >
      <div className="flex items-start gap-3">
        <div className={`${colors.icon} mt-0.5 flex-shrink-0`}>
          <Icon className="w-5 h-5" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2 mb-1">
            <div className="flex items-center gap-2">
              <span className={`text-sm font-semibold ${colors.text}`}>
                {event.title}
              </span>
              <span className={`
                text-xs px-2 py-0.5 rounded-full
                ${colors.bg} ${colors.border} border
                ${colors.text}
              `}>
                {getEventTypeLabel(event.type)}
              </span>
            </div>
            <span className="text-xs text-slate-500 flex-shrink-0">
              {formatTimestamp(event.timestamp)}
            </span>
          </div>
          <p className="text-sm text-slate-400 leading-relaxed">
            {event.description}
          </p>
          {event.metadata && Object.keys(event.metadata).length > 0 && (
            <div className="mt-2 flex flex-wrap gap-2">
              {Object.entries(event.metadata).map(([key, value]) => (
                <span
                  key={key}
                  className="text-xs bg-slate-700/50 text-slate-400 px-2 py-1 rounded"
                >
                  {key}: {String(value)}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Filter Button Component
interface FilterButtonProps {
  type: EventType | 'all';
  active: boolean;
  onClick: () => void;
  count?: number;
}

const FilterButton: React.FC<FilterButtonProps> = ({ type, active, onClick, count }) => {
  const Icon = type !== 'all' ? getEventIcon(type) : null;
  const label = type === 'all' ? 'All Events' : getEventTypeLabel(type);

  return (
    <button
      onClick={onClick}
      className={`
        flex items-center gap-2 px-3 py-2 rounded-lg
        text-sm font-medium transition-all duration-200
        ${active
          ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/25'
          : 'bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200'
        }
      `}
    >
      {Icon && <Icon className="w-4 h-4" />}
      <span>{label}</span>
      {count !== undefined && (
        <span className={`
          text-xs px-1.5 py-0.5 rounded-full
          ${active ? 'bg-blue-500 text-white' : 'bg-slate-700 text-slate-400'}
        `}>
          {count}
        </span>
      )}
    </button>
  );
};

// Connection Status Component
interface ConnectionStatusProps {
  status: 'connecting' | 'connected' | 'disconnected' | 'error';
}

const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ status }) => {
  const statusConfig = {
    connecting: { color: 'bg-yellow-500', text: 'Connecting...', pulse: true },
    connected: { color: 'bg-emerald-500', text: 'Live', pulse: true },
    disconnected: { color: 'bg-slate-500', text: 'Disconnected', pulse: false },
    error: { color: 'bg-red-500', text: 'Error', pulse: false },
  };

  const config = statusConfig[status];

  return (
    <div className="flex items-center gap-2 text-sm">
      <span className="relative flex h-2.5 w-2.5">
        {config.pulse && (
          <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${config.color} opacity-75`} />
        )}
        <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${config.color}`} />
      </span>
      <span className="text-slate-400">{config.text}</span>
    </div>
  );
};

// Main Events Page Component
const Events: React.FC = () => {
  const [events, setEvents] = useState<TradingEvent[]>([]);
  const [filter, setFilter] = useState<EventType | 'all'>('all');
  const [autoScroll, setAutoScroll] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>('connecting');
  const eventListRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Parse incoming SSE event data
  const parseEventData = useCallback((data: string): TradingEvent | null => {
    try {
      const parsed = JSON.parse(data);
      return {
        id: parsed.id || crypto.randomUUID(),
        timestamp: new Date(parsed.timestamp || Date.now()),
        type: parsed.type as EventType,
        severity: parsed.severity as EventSeverity,
        title: parsed.title || 'Event',
        description: parsed.description || '',
        metadata: parsed.metadata,
      };
    } catch (error) {
      console.error('Failed to parse event data:', error);
      return null;
    }
  }, []);

  // Connect to SSE endpoint
  const connectSSE = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    setConnectionStatus('connecting');

    // SSE endpoint - adjust URL based on your backend configuration
    const eventSource = new EventSource('/api/events/stream');
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setConnectionStatus('connected');
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };

    eventSource.onmessage = (event) => {
      const parsedEvent = parseEventData(event.data);
      if (parsedEvent) {
        setEvents((prev) => {
          // Keep last 500 events to prevent memory issues
          const newEvents = [parsedEvent, ...prev].slice(0, 500);
          return newEvents;
        });
      }
    };

    // Handle specific event types from SSE
    const eventTypes: EventType[] = ['trade', 'signal', 'regime_change', 'risk_alert'];
    eventTypes.forEach((type) => {
      eventSource.addEventListener(type, (event) => {
        const parsedEvent = parseEventData((event as MessageEvent).data);
        if (parsedEvent) {
          parsedEvent.type = type;
          setEvents((prev) => [parsedEvent, ...prev].slice(0, 500));
        }
      });
    });

    eventSource.onerror = () => {
      setConnectionStatus('error');
      eventSource.close();

      // Attempt to reconnect after 5 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        setConnectionStatus('disconnected');
        connectSSE();
      }, 5000);
    };
  }, [parseEventData]);

  // Initialize SSE connection
  useEffect(() => {
    connectSSE();

    // Generate demo events for development/testing
    const generateDemoEvent = (): TradingEvent => {
      const types: EventType[] = ['trade', 'signal', 'regime_change', 'risk_alert'];
      const severities: EventSeverity[] = ['info', 'success', 'warning', 'critical'];
      const type = types[Math.floor(Math.random() * types.length)];
      const severity = severities[Math.floor(Math.random() * severities.length)];

      const eventTemplates: Record<EventType, { title: string; description: string; metadata?: Record<string, unknown> }[]> = {
        trade: [
          { title: 'BUY Order Executed', description: 'Purchased 100 shares of AAPL at $178.25', metadata: { symbol: 'AAPL', quantity: 100, price: 178.25 } },
          { title: 'SELL Order Executed', description: 'Sold 50 shares of TSLA at $245.80', metadata: { symbol: 'TSLA', quantity: 50, price: 245.80 } },
          { title: 'Option Trade', description: 'Bought 5 NVDA 500C 2/16 contracts at $12.50', metadata: { symbol: 'NVDA', strike: 500, expiry: '2/16' } },
        ],
        signal: [
          { title: 'Bullish Signal Detected', description: 'RSI oversold bounce pattern on SPY 15-min chart', metadata: { indicator: 'RSI', timeframe: '15m' } },
          { title: 'MACD Crossover', description: 'Bullish MACD crossover on QQQ daily chart', metadata: { indicator: 'MACD', timeframe: '1D' } },
          { title: 'Volume Spike Alert', description: 'Unusual volume detected on AMD - 3x average', metadata: { symbol: 'AMD', multiplier: '3x' } },
        ],
        regime_change: [
          { title: 'Market Regime: Risk-On', description: 'Transitioning to aggressive positioning mode', metadata: { mode: 'aggressive' } },
          { title: 'Market Regime: Defensive', description: 'VIX spike detected, reducing exposure', metadata: { vix: 25.4 } },
          { title: 'Sector Rotation', description: 'Rotating from tech to energy sector', metadata: { from: 'XLK', to: 'XLE' } },
        ],
        risk_alert: [
          { title: 'Position Size Warning', description: 'NVDA position exceeds 15% of portfolio', metadata: { symbol: 'NVDA', percentage: '17.2%' } },
          { title: 'Drawdown Alert', description: 'Daily P&L approaching -2% threshold', metadata: { drawdown: '-1.8%' } },
          { title: 'Margin Warning', description: 'Margin utilization at 85%', metadata: { margin: '85%' } },
        ],
      };

      const template = eventTemplates[type][Math.floor(Math.random() * eventTemplates[type].length)];

      return {
        id: crypto.randomUUID(),
        timestamp: new Date(),
        type,
        severity,
        title: template.title,
        description: template.description,
        metadata: template.metadata,
      };
    };

    // For demo purposes, generate events periodically
    // Remove this in production when connected to real SSE endpoint
    const demoInterval = setInterval(() => {
      if (connectionStatus !== 'connected') {
        setConnectionStatus('connected');
      }
      const demoEvent = generateDemoEvent();
      setEvents((prev) => [demoEvent, ...prev].slice(0, 500));
    }, 3000);

    return () => {
      clearInterval(demoInterval);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connectSSE]);

  // Auto-scroll effect
  useEffect(() => {
    if (autoScroll && eventListRef.current) {
      eventListRef.current.scrollTop = 0;
    }
  }, [events, autoScroll]);

  // Filter events
  const filteredEvents = filter === 'all'
    ? events
    : events.filter((event) => event.type === filter);

  // Count events by type
  const eventCounts: Record<EventType, number> = {
    trade: events.filter((e) => e.type === 'trade').length,
    signal: events.filter((e) => e.type === 'signal').length,
    regime_change: events.filter((e) => e.type === 'regime_change').length,
    risk_alert: events.filter((e) => e.type === 'risk_alert').length,
  };

  // Group events by date
  const groupedEvents: { date: string; events: TradingEvent[] }[] = [];
  let currentDate = '';
  filteredEvents.forEach((event) => {
    const dateStr = formatDateHeader(event.timestamp);
    if (dateStr !== currentDate) {
      currentDate = dateStr;
      groupedEvents.push({ date: dateStr, events: [event] });
    } else {
      groupedEvents[groupedEvents.length - 1].events.push(event);
    }
  });

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      {/* CSS for animations */}
      <style>{`
        @keyframes slide-in {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-slide-in {
          animation: slide-in 0.3s ease-out;
        }
      `}</style>

      {/* Header */}
      <header className="sticky top-0 z-10 bg-slate-900/95 backdrop-blur-sm border-b border-slate-800">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-2xl font-bold text-white">Live Events</h1>
              <p className="text-sm text-slate-400 mt-1">
                Real-time trading activity and alerts
              </p>
            </div>
            <div className="flex items-center gap-4">
              <ConnectionStatus status={connectionStatus} />
              <button
                onClick={() => setAutoScroll(!autoScroll)}
                className={`
                  flex items-center gap-2 px-4 py-2 rounded-lg
                  text-sm font-medium transition-all duration-200
                  ${autoScroll
                    ? 'bg-blue-600 text-white'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                  }
                `}
                title={autoScroll ? 'Pause auto-scroll' : 'Resume auto-scroll'}
              >
                {autoScroll ? (
                  <>
                    <PauseIcon className="w-4 h-4" />
                    <span>Pause</span>
                  </>
                ) : (
                  <>
                    <PlayIcon className="w-4 h-4" />
                    <span>Resume</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Filters */}
          <div className="flex flex-wrap gap-2">
            <FilterButton
              type="all"
              active={filter === 'all'}
              onClick={() => setFilter('all')}
              count={events.length}
            />
            <FilterButton
              type="trade"
              active={filter === 'trade'}
              onClick={() => setFilter('trade')}
              count={eventCounts.trade}
            />
            <FilterButton
              type="signal"
              active={filter === 'signal'}
              onClick={() => setFilter('signal')}
              count={eventCounts.signal}
            />
            <FilterButton
              type="regime_change"
              active={filter === 'regime_change'}
              onClick={() => setFilter('regime_change')}
              count={eventCounts.regime_change}
            />
            <FilterButton
              type="risk_alert"
              active={filter === 'risk_alert'}
              onClick={() => setFilter('risk_alert')}
              count={eventCounts.risk_alert}
            />
          </div>
        </div>
      </header>

      {/* Event List */}
      <main className="max-w-6xl mx-auto px-4 py-6">
        <div
          ref={eventListRef}
          className="space-y-6 max-h-[calc(100vh-220px)] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent"
        >
          {groupedEvents.length === 0 ? (
            <div className="text-center py-12">
              <div className="text-slate-500 mb-2">
                <SignalIcon className="w-12 h-12 mx-auto opacity-50" />
              </div>
              <p className="text-slate-400 text-lg">No events yet</p>
              <p className="text-slate-500 text-sm mt-1">
                Events will appear here as they occur
              </p>
            </div>
          ) : (
            groupedEvents.map((group, groupIndex) => (
              <div key={`${group.date}-${groupIndex}`}>
                <div className="sticky top-0 z-5 py-2">
                  <span className="text-xs font-medium text-slate-500 bg-slate-900 px-2 py-1 rounded">
                    {group.date}
                  </span>
                </div>
                <div className="space-y-0">
                  {group.events.map((event) => (
                    <EventCard key={event.id} event={event} />
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      </main>

      {/* Footer Stats */}
      <footer className="fixed bottom-0 left-0 right-0 bg-slate-900/95 backdrop-blur-sm border-t border-slate-800 py-3">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex items-center justify-between text-sm text-slate-400">
            <div className="flex items-center gap-4">
              <span>
                Showing <span className="text-white font-medium">{filteredEvents.length}</span> of{' '}
                <span className="text-white font-medium">{events.length}</span> events
              </span>
            </div>
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                Success: {events.filter((e) => e.severity === 'success').length}
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-amber-500" />
                Warning: {events.filter((e) => e.severity === 'warning').length}
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-red-500" />
                Critical: {events.filter((e) => e.severity === 'critical').length}
              </span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Events;
