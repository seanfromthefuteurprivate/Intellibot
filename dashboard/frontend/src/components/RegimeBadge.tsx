import React, { useState } from 'react';

// Market regime types
export type MarketRegime =
  | 'TRENDING_UP'
  | 'TRENDING_DOWN'
  | 'MEAN_REVERTING'
  | 'HIGH_VOLATILITY'
  | 'LOW_VOLATILITY'
  | 'BREAKOUT'
  | 'CONSOLIDATING'
  | 'UNKNOWN';

// Regime configuration for styling and descriptions
interface RegimeConfig {
  label: string;
  color: string;
  backgroundColor: string;
  borderColor: string;
  icon: React.ReactNode;
  description: string;
}

// SVG Icons for each regime
const TrendUpIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
    <polyline points="17 6 23 6 23 12" />
  </svg>
);

const TrendDownIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polyline points="23 18 13.5 8.5 8.5 13.5 1 6" />
    <polyline points="17 18 23 18 23 12" />
  </svg>
);

const MeanRevertIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M2 12h4l3 -9l4 18l4 -9l3 0h4" />
  </svg>
);

const HighVolatilityIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
  </svg>
);

const LowVolatilityIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);

const BreakoutIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <circle cx="12" cy="12" r="10" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>
);

const ConsolidatingIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
    <line x1="9" y1="3" x2="9" y2="21" />
    <line x1="15" y1="3" x2="15" y2="21" />
  </svg>
);

const UnknownIcon: React.FC<{ size?: number }> = ({ size = 16 }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <circle cx="12" cy="12" r="10" />
    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2 -3 3 -3 3" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

// Regime configuration map
const REGIME_CONFIG: Record<MarketRegime, RegimeConfig> = {
  TRENDING_UP: {
    label: 'Trending Up',
    color: '#10b981',
    backgroundColor: 'rgba(16, 185, 129, 0.15)',
    borderColor: 'rgba(16, 185, 129, 0.4)',
    icon: <TrendUpIcon />,
    description:
      'Market is in a bullish trend. Prices are consistently making higher highs and higher lows. Momentum strategies tend to perform well.',
  },
  TRENDING_DOWN: {
    label: 'Trending Down',
    color: '#ef4444',
    backgroundColor: 'rgba(239, 68, 68, 0.15)',
    borderColor: 'rgba(239, 68, 68, 0.4)',
    icon: <TrendDownIcon />,
    description:
      'Market is in a bearish trend. Prices are consistently making lower highs and lower lows. Short positions or defensive strategies may be appropriate.',
  },
  MEAN_REVERTING: {
    label: 'Mean Reverting',
    color: '#8b5cf6',
    backgroundColor: 'rgba(139, 92, 246, 0.15)',
    borderColor: 'rgba(139, 92, 246, 0.4)',
    icon: <MeanRevertIcon />,
    description:
      'Market is oscillating around a mean value. Prices tend to revert to average after deviations. Range-bound strategies and contrarian plays work well.',
  },
  HIGH_VOLATILITY: {
    label: 'High Volatility',
    color: '#f59e0b',
    backgroundColor: 'rgba(245, 158, 11, 0.15)',
    borderColor: 'rgba(245, 158, 11, 0.4)',
    icon: <HighVolatilityIcon />,
    description:
      'Market is experiencing high volatility with large price swings. Increased risk and opportunity. Consider wider stops and smaller position sizes.',
  },
  LOW_VOLATILITY: {
    label: 'Low Volatility',
    color: '#6b7280',
    backgroundColor: 'rgba(107, 114, 128, 0.15)',
    borderColor: 'rgba(107, 114, 128, 0.4)',
    icon: <LowVolatilityIcon />,
    description:
      'Market is calm with low price movement. Breakouts may be imminent. Consider accumulating positions or waiting for volatility expansion.',
  },
  BREAKOUT: {
    label: 'Breakout',
    color: '#06b6d4',
    backgroundColor: 'rgba(6, 182, 212, 0.15)',
    borderColor: 'rgba(6, 182, 212, 0.4)',
    icon: <BreakoutIcon />,
    description:
      'Market is breaking out of a consolidation pattern. Strong directional move is likely. Momentum entry strategies are favored.',
  },
  CONSOLIDATING: {
    label: 'Consolidating',
    color: '#3b82f6',
    backgroundColor: 'rgba(59, 130, 246, 0.15)',
    borderColor: 'rgba(59, 130, 246, 0.4)',
    icon: <ConsolidatingIcon />,
    description:
      'Market is consolidating in a range. Accumulation or distribution phase. Wait for breakout confirmation before taking directional positions.',
  },
  UNKNOWN: {
    label: 'Unknown',
    color: '#9ca3af',
    backgroundColor: 'rgba(156, 163, 175, 0.15)',
    borderColor: 'rgba(156, 163, 175, 0.4)',
    icon: <UnknownIcon />,
    description:
      'Market regime cannot be determined. Insufficient data or mixed signals. Exercise caution and reduce position sizes.',
  },
};

// Component props
export interface RegimeBadgeProps {
  regime: MarketRegime;
  size?: 'small' | 'medium' | 'large';
  showLabel?: boolean;
  showIcon?: boolean;
  className?: string;
  style?: React.CSSProperties;
}

// Tooltip component
interface TooltipProps {
  content: string;
  children: React.ReactNode;
  visible: boolean;
}

const Tooltip: React.FC<TooltipProps> = ({ content, children, visible }) => {
  return (
    <div style={{ position: 'relative', display: 'inline-flex' }}>
      {children}
      {visible && (
        <div
          style={{
            position: 'absolute',
            bottom: 'calc(100% + 8px)',
            left: '50%',
            transform: 'translateX(-50%)',
            backgroundColor: '#1f2937',
            color: '#f9fafb',
            padding: '8px 12px',
            borderRadius: '6px',
            fontSize: '12px',
            lineHeight: '1.5',
            maxWidth: '280px',
            width: 'max-content',
            zIndex: 1000,
            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
            whiteSpace: 'normal',
          }}
        >
          {content}
          {/* Tooltip arrow */}
          <div
            style={{
              position: 'absolute',
              top: '100%',
              left: '50%',
              transform: 'translateX(-50%)',
              width: 0,
              height: 0,
              borderLeft: '6px solid transparent',
              borderRight: '6px solid transparent',
              borderTop: '6px solid #1f2937',
            }}
          />
        </div>
      )}
    </div>
  );
};

// Size configurations
const SIZE_CONFIG = {
  small: {
    padding: '4px 8px',
    fontSize: '11px',
    iconSize: 12,
    gap: '4px',
  },
  medium: {
    padding: '6px 12px',
    fontSize: '13px',
    iconSize: 16,
    gap: '6px',
  },
  large: {
    padding: '8px 16px',
    fontSize: '15px',
    iconSize: 20,
    gap: '8px',
  },
};

/**
 * RegimeBadge Component
 *
 * Displays the current market regime with color coding, icon, and tooltip.
 *
 * @example
 * ```tsx
 * <RegimeBadge regime="TRENDING_UP" />
 * <RegimeBadge regime="HIGH_VOLATILITY" size="large" />
 * <RegimeBadge regime="MEAN_REVERTING" showIcon={false} />
 * ```
 */
export const RegimeBadge: React.FC<RegimeBadgeProps> = ({
  regime,
  size = 'medium',
  showLabel = true,
  showIcon = true,
  className = '',
  style = {},
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const config = REGIME_CONFIG[regime] || REGIME_CONFIG.UNKNOWN;
  const sizeConfig = SIZE_CONFIG[size];

  // Clone icon with correct size
  const iconWithSize = showIcon
    ? React.cloneElement(config.icon as React.ReactElement<{ size?: number }>, {
        size: sizeConfig.iconSize,
      })
    : null;

  const badgeStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: sizeConfig.gap,
    padding: sizeConfig.padding,
    fontSize: sizeConfig.fontSize,
    fontWeight: 600,
    color: config.color,
    backgroundColor: config.backgroundColor,
    border: `1px solid ${config.borderColor}`,
    borderRadius: '6px',
    cursor: 'default',
    transition: 'all 0.2s ease',
    userSelect: 'none',
    ...(isHovered && {
      backgroundColor: config.borderColor,
      boxShadow: `0 0 8px ${config.borderColor}`,
    }),
    ...style,
  };

  return (
    <Tooltip content={config.description} visible={isHovered}>
      <div
        className={className}
        style={badgeStyle}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        role="status"
        aria-label={`Market regime: ${config.label}. ${config.description}`}
      >
        {iconWithSize}
        {showLabel && <span>{config.label}</span>}
      </div>
    </Tooltip>
  );
};

// Helper function to get regime config externally
export const getRegimeConfig = (regime: MarketRegime): RegimeConfig => {
  return REGIME_CONFIG[regime] || REGIME_CONFIG.UNKNOWN;
};

// Export all regime types for external use
export const MARKET_REGIMES: MarketRegime[] = [
  'TRENDING_UP',
  'TRENDING_DOWN',
  'MEAN_REVERTING',
  'HIGH_VOLATILITY',
  'LOW_VOLATILITY',
  'BREAKOUT',
  'CONSOLIDATING',
  'UNKNOWN',
];

export default RegimeBadge;
