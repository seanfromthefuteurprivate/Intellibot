import React, { useState, useEffect, useRef, useCallback } from 'react';

// Event types supported by the feed
export type EventType =
  | 'trade_executed'
  | 'order_placed'
  | 'order_cancelled'
  | 'position_opened'
  | 'position_closed'
  | 'alert'
  | 'error'
  | 'info'
  | 'signal'
  | 'system';

// Event data structure
export interface FeedEvent {
  id: string;
  type: EventType;
  message: string;
  timestamp: Date | string;
  details?: Record<string, unknown>;
}

interface EventFeedProps {
  events: FeedEvent[];
  maxHeight?: string;
  onEventClick?: (event: FeedEvent) => void;
  className?: string;
}

// Color mapping for different event types
const eventTypeColors: Record<EventType, { bg: string; text: string; border: string }> = {
  trade_executed: { bg: 'bg-green-900/30', text: 'text-green-400', border: 'border-green-500/50' },
  order_placed: { bg: 'bg-blue-900/30', text: 'text-blue-400', border: 'border-blue-500/50' },
  order_cancelled: { bg: 'bg-yellow-900/30', text: 'text-yellow-400', border: 'border-yellow-500/50' },
  position_opened: { bg: 'bg-emerald-900/30', text: 'text-emerald-400', border: 'border-emerald-500/50' },
  position_closed: { bg: 'bg-purple-900/30', text: 'text-purple-400', border: 'border-purple-500/50' },
  alert: { bg: 'bg-orange-900/30', text: 'text-orange-400', border: 'border-orange-500/50' },
  error: { bg: 'bg-red-900/30', text: 'text-red-400', border: 'border-red-500/50' },
  info: { bg: 'bg-gray-800/50', text: 'text-gray-300', border: 'border-gray-600/50' },
  signal: { bg: 'bg-cyan-900/30', text: 'text-cyan-400', border: 'border-cyan-500/50' },
  system: { bg: 'bg-slate-800/50', text: 'text-slate-300', border: 'border-slate-600/50' },
};

// Icon components for different event types
const EventIcon: React.FC<{ type: EventType; className?: string }> = ({ type, className = '' }) => {
  const iconClass = `w-5 h-5 ${className}`;

  switch (type) {
    case 'trade_executed':
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
    case 'order_placed':
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
    case 'order_cancelled':
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
    case 'position_opened':
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      );
    case 'position_closed':
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
        </svg>
      );
    case 'alert':
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      );
    case 'error':
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
    case 'signal':
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      );
    case 'system':
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      );
    case 'info':
    default:
      return (
        <svg className={iconClass} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
  }
};

// Format timestamp for display
const formatTimestamp = (timestamp: Date | string): string => {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);

  // If within the last minute, show seconds ago
  if (diffSec < 60) {
    return diffSec <= 5 ? 'Just now' : `${diffSec}s ago`;
  }

  // If within the last hour, show minutes ago
  if (diffMin < 60) {
    return `${diffMin}m ago`;
  }

  // If within today, show time only
  if (diffHour < 24 && date.getDate() === now.getDate()) {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true
    });
  }

  // Otherwise show date and time
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true
  });
};

// Format event type for display
const formatEventType = (type: EventType): string => {
  return type
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

// Pause/Play button component
const PausePlayButton: React.FC<{ isPaused: boolean; onClick: () => void }> = ({ isPaused, onClick }) => (
  <button
    onClick={onClick}
    className={`
      flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium
      transition-all duration-200 ease-in-out
      ${isPaused
        ? 'bg-green-600 hover:bg-green-500 text-white'
        : 'bg-gray-700 hover:bg-gray-600 text-gray-200'
      }
    `}
    aria-label={isPaused ? 'Resume auto-scroll' : 'Pause auto-scroll'}
  >
    {isPaused ? (
      <>
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
          <path d="M8 5v14l11-7z" />
        </svg>
        <span>Resume</span>
      </>
    ) : (
      <>
        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
          <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
        </svg>
        <span>Pause</span>
      </>
    )}
  </button>
);

// Single event item component
const EventItem: React.FC<{
  event: FeedEvent;
  onClick?: (event: FeedEvent) => void;
}> = ({ event, onClick }) => {
  const colors = eventTypeColors[event.type] || eventTypeColors.info;

  return (
    <div
      className={`
        flex items-start gap-3 p-3 rounded-lg border cursor-pointer
        transition-all duration-200 ease-in-out
        ${colors.bg} ${colors.border}
        hover:scale-[1.01] hover:shadow-lg hover:shadow-black/20
      `}
      onClick={() => onClick?.(event)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          onClick?.(event);
        }
      }}
    >
      {/* Event Icon */}
      <div className={`flex-shrink-0 mt-0.5 ${colors.text}`}>
        <EventIcon type={event.type} />
      </div>

      {/* Event Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2 mb-1">
          <span className={`text-xs font-semibold uppercase tracking-wide ${colors.text}`}>
            {formatEventType(event.type)}
          </span>
          <span className="text-xs text-gray-500 flex-shrink-0">
            {formatTimestamp(event.timestamp)}
          </span>
        </div>
        <p className="text-sm text-gray-200 break-words">
          {event.message}
        </p>
        {event.details && Object.keys(event.details).length > 0 && (
          <div className="mt-2 text-xs text-gray-400 font-mono bg-black/20 rounded p-2">
            {Object.entries(event.details).map(([key, value]) => (
              <div key={key} className="truncate">
                <span className="text-gray-500">{key}:</span>{' '}
                <span className="text-gray-300">
                  {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

// Main EventFeed component
const EventFeed: React.FC<EventFeedProps> = ({
  events,
  maxHeight = '500px',
  onEventClick,
  className = '',
}) => {
  const [isPaused, setIsPaused] = useState(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const lastEventCountRef = useRef(events.length);

  // Scroll to bottom function
  const scrollToBottom = useCallback(() => {
    if (scrollContainerRef.current && !isPaused) {
      scrollContainerRef.current.scrollTo({
        top: scrollContainerRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [isPaused]);

  // Auto-scroll when new events arrive
  useEffect(() => {
    if (events.length > lastEventCountRef.current) {
      scrollToBottom();
    }
    lastEventCountRef.current = events.length;
  }, [events.length, scrollToBottom]);

  // Handle manual scroll - pause if user scrolls up
  const handleScroll = useCallback(() => {
    if (!scrollContainerRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;

    // Auto-resume if user scrolls back to bottom
    if (isAtBottom && isPaused) {
      setIsPaused(false);
    }
  }, [isPaused]);

  // Toggle pause state
  const togglePause = useCallback(() => {
    setIsPaused(prev => {
      if (prev) {
        // Resuming - scroll to bottom
        setTimeout(scrollToBottom, 50);
      }
      return !prev;
    });
  }, [scrollToBottom]);

  return (
    <div className={`flex flex-col bg-gray-900 rounded-xl border border-gray-800 overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gray-800/50 border-b border-gray-700">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-white">Live Events</h2>
          <span className="px-2 py-0.5 text-xs font-medium bg-gray-700 text-gray-300 rounded-full">
            {events.length} events
          </span>
          {isPaused && (
            <span className="px-2 py-0.5 text-xs font-medium bg-yellow-600/30 text-yellow-400 rounded-full animate-pulse">
              Paused
            </span>
          )}
        </div>
        <PausePlayButton isPaused={isPaused} onClick={togglePause} />
      </div>

      {/* Event List */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-3 space-y-2 scrollbar-thin scrollbar-track-gray-800 scrollbar-thumb-gray-600"
        style={{ maxHeight }}
      >
        {events.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-gray-500">
            <svg className="w-12 h-12 mb-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <p className="text-sm">No events yet</p>
            <p className="text-xs mt-1">Events will appear here as they occur</p>
          </div>
        ) : (
          events.map((event) => (
            <EventItem
              key={event.id}
              event={event}
              onClick={onEventClick}
            />
          ))
        )}
      </div>

      {/* Footer - Scroll to bottom button (shown when paused and not at bottom) */}
      {isPaused && events.length > 0 && (
        <div className="px-4 py-2 bg-gray-800/30 border-t border-gray-700/50">
          <button
            onClick={() => {
              setIsPaused(false);
              scrollToBottom();
            }}
            className="w-full py-2 text-sm text-gray-400 hover:text-white transition-colors flex items-center justify-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
            Jump to latest
          </button>
        </div>
      )}
    </div>
  );
};

export default EventFeed;
