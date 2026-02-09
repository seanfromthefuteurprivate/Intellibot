import React, { useEffect, useState } from 'react';

interface TopBarProps {
  accountBalance: number;
  todayPnL: number;
  isConnected: boolean;
}

type MarketStatus = 'open' | 'closed' | 'pre-market' | 'after-hours';

const TopBar: React.FC<TopBarProps> = ({ accountBalance, todayPnL, isConnected }) => {
  const [currentTime, setCurrentTime] = useState<string>('');
  const [marketStatus, setMarketStatus] = useState<MarketStatus>('closed');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const etTime = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));

      // Format time as HH:MM:SS AM/PM
      const hours = etTime.getHours();
      const minutes = etTime.getMinutes();
      const seconds = etTime.getSeconds();
      const ampm = hours >= 12 ? 'PM' : 'AM';
      const displayHours = hours % 12 || 12;
      const formattedTime = `${displayHours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')} ${ampm}`;

      setCurrentTime(formattedTime);

      // Determine market status
      const dayOfWeek = etTime.getDay();
      const currentHour = hours;
      const currentMinute = minutes;
      const timeInMinutes = currentHour * 60 + currentMinute;

      // Market hours: 9:30 AM - 4:00 PM ET
      const marketOpenTime = 9 * 60 + 30; // 9:30 AM
      const marketCloseTime = 16 * 60; // 4:00 PM
      const preMarketStart = 4 * 60; // 4:00 AM
      const afterHoursEnd = 20 * 60; // 8:00 PM

      // Check if it's a weekday
      if (dayOfWeek >= 1 && dayOfWeek <= 5) {
        if (timeInMinutes >= marketOpenTime && timeInMinutes < marketCloseTime) {
          setMarketStatus('open');
        } else if (timeInMinutes >= preMarketStart && timeInMinutes < marketOpenTime) {
          setMarketStatus('pre-market');
        } else if (timeInMinutes >= marketCloseTime && timeInMinutes < afterHoursEnd) {
          setMarketStatus('after-hours');
        } else {
          setMarketStatus('closed');
        }
      } else {
        setMarketStatus('closed');
      }
    };

    // Update immediately and then every second
    updateTime();
    const interval = setInterval(updateTime, 1000);

    return () => clearInterval(interval);
  }, []);

  const formatCurrency = (amount: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(amount);
  };

  const formatPnL = (amount: number): string => {
    const sign = amount >= 0 ? '+' : '';
    return `${sign}${formatCurrency(amount)}`;
  };

  const getPnLColor = (): string => {
    if (todayPnL > 0) return '#10b981'; // green
    if (todayPnL < 0) return '#ef4444'; // red
    return '#9ca3af'; // gray
  };

  const getMarketStatusConfig = (): { label: string; color: string; bgColor: string } => {
    switch (marketStatus) {
      case 'open':
        return { label: 'MARKET OPEN', color: '#10b981', bgColor: 'rgba(16, 185, 129, 0.15)' };
      case 'pre-market':
        return { label: 'PRE-MARKET', color: '#f59e0b', bgColor: 'rgba(245, 158, 11, 0.15)' };
      case 'after-hours':
        return { label: 'AFTER-HOURS', color: '#8b5cf6', bgColor: 'rgba(139, 92, 246, 0.15)' };
      case 'closed':
      default:
        return { label: 'MARKET CLOSED', color: '#6b7280', bgColor: 'rgba(107, 114, 128, 0.15)' };
    }
  };

  const marketConfig = getMarketStatusConfig();

  return (
    <div style={styles.container}>
      <div style={styles.leftSection}>
        <div style={styles.logoContainer}>
          <span style={styles.logo}>INTELLIBOT</span>
          <span style={styles.version}>v2.0</span>
        </div>
      </div>

      <div style={styles.centerSection}>
        {/* Account Balance */}
        <div style={styles.infoBlock}>
          <span style={styles.label}>Account Balance</span>
          <span style={styles.value}>{formatCurrency(accountBalance)}</span>
        </div>

        {/* Divider */}
        <div style={styles.divider} />

        {/* Today's P&L */}
        <div style={styles.infoBlock}>
          <span style={styles.label}>Today's P&L</span>
          <span style={{ ...styles.value, color: getPnLColor() }}>
            {formatPnL(todayPnL)}
          </span>
        </div>

        {/* Divider */}
        <div style={styles.divider} />

        {/* Market Status */}
        <div style={styles.infoBlock}>
          <span style={styles.label}>Market Status</span>
          <span
            style={{
              ...styles.statusBadge,
              color: marketConfig.color,
              backgroundColor: marketConfig.bgColor,
            }}
          >
            <span
              style={{
                ...styles.statusDot,
                backgroundColor: marketConfig.color,
                boxShadow: marketStatus === 'open' ? `0 0 8px ${marketConfig.color}` : 'none',
                animation: marketStatus === 'open' ? 'pulse 2s infinite' : 'none',
              }}
            />
            {marketConfig.label}
          </span>
        </div>
      </div>

      <div style={styles.rightSection}>
        {/* Current Time */}
        <div style={styles.timeBlock}>
          <span style={styles.time}>{currentTime}</span>
          <span style={styles.timezone}>ET</span>
        </div>

        {/* Divider */}
        <div style={styles.divider} />

        {/* Connection Status */}
        <div style={styles.connectionBlock}>
          <span
            style={{
              ...styles.connectionDot,
              backgroundColor: isConnected ? '#10b981' : '#ef4444',
              boxShadow: isConnected
                ? '0 0 8px rgba(16, 185, 129, 0.6)'
                : '0 0 8px rgba(239, 68, 68, 0.6)',
            }}
          />
          <span style={styles.connectionText}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Pulse animation keyframes (injected via style tag) */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% {
              opacity: 1;
            }
            50% {
              opacity: 0.5;
            }
          }
        `}
      </style>
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#0f0f1a',
    borderBottom: '1px solid #2d2d44',
    padding: '12px 24px',
    height: '64px',
    boxSizing: 'border-box',
  },
  leftSection: {
    display: 'flex',
    alignItems: 'center',
  },
  logoContainer: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '8px',
  },
  logo: {
    fontSize: '20px',
    fontWeight: 700,
    color: '#ffffff',
    letterSpacing: '1px',
    fontFamily: "'SF Mono', 'Fira Code', 'Consolas', monospace",
  },
  version: {
    fontSize: '11px',
    color: '#6b7280',
    fontWeight: 500,
  },
  centerSection: {
    display: 'flex',
    alignItems: 'center',
    gap: '24px',
  },
  infoBlock: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '4px',
  },
  label: {
    fontSize: '11px',
    fontWeight: 500,
    color: '#6b7280',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  value: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#ffffff',
    fontFamily: "'SF Mono', 'Fira Code', 'Consolas', monospace",
  },
  divider: {
    width: '1px',
    height: '32px',
    backgroundColor: '#2d2d44',
  },
  statusBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '4px 10px',
    borderRadius: '12px',
    fontSize: '11px',
    fontWeight: 600,
    letterSpacing: '0.5px',
  },
  statusDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
  rightSection: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
  },
  timeBlock: {
    display: 'flex',
    alignItems: 'baseline',
    gap: '6px',
  },
  time: {
    fontSize: '16px',
    fontWeight: 600,
    color: '#ffffff',
    fontFamily: "'SF Mono', 'Fira Code', 'Consolas', monospace",
  },
  timezone: {
    fontSize: '11px',
    fontWeight: 500,
    color: '#6b7280',
  },
  connectionBlock: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  connectionDot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
  },
  connectionText: {
    fontSize: '13px',
    fontWeight: 500,
    color: '#a0a0b0',
  },
};

export default TopBar;
