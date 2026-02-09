import React, { ReactNode, CSSProperties } from 'react';

// Color variant types
export type ColorVariant = 'default' | 'success' | 'danger' | 'warning';

// Size types
export type CardSize = 'compact' | 'large';

// Change indicator direction
export type ChangeDirection = 'up' | 'down' | 'neutral';

export interface StatCardProps {
  /** The label/title for the stat */
  label: string;
  /** The main value to display */
  value: string | number;
  /** Optional change value (e.g., "+5.2%") */
  change?: string;
  /** Direction of change for styling */
  changeDirection?: ChangeDirection;
  /** Icon element to display */
  icon?: ReactNode;
  /** Color variant for the card */
  variant?: ColorVariant;
  /** Size of the card */
  size?: CardSize;
  /** Additional CSS class name */
  className?: string;
  /** Additional inline styles */
  style?: CSSProperties;
  /** Click handler */
  onClick?: () => void;
}

// Color palette for variants
const variantColors: Record<ColorVariant, { bg: string; border: string; text: string; icon: string }> = {
  default: {
    bg: '#ffffff',
    border: '#e5e7eb',
    text: '#374151',
    icon: '#6b7280',
  },
  success: {
    bg: '#f0fdf4',
    border: '#86efac',
    text: '#166534',
    icon: '#22c55e',
  },
  danger: {
    bg: '#fef2f2',
    border: '#fca5a5',
    text: '#991b1b',
    icon: '#ef4444',
  },
  warning: {
    bg: '#fffbeb',
    border: '#fcd34d',
    text: '#92400e',
    icon: '#f59e0b',
  },
};

// Size configurations
const sizeConfig: Record<CardSize, {
  padding: string;
  labelSize: string;
  valueSize: string;
  changeSize: string;
  iconSize: string;
  gap: string;
}> = {
  compact: {
    padding: '12px 16px',
    labelSize: '12px',
    valueSize: '20px',
    changeSize: '11px',
    iconSize: '32px',
    gap: '8px',
  },
  large: {
    padding: '20px 24px',
    labelSize: '14px',
    valueSize: '32px',
    changeSize: '13px',
    iconSize: '48px',
    gap: '12px',
  },
};

// Change direction colors
const changeColors: Record<ChangeDirection, string> = {
  up: '#22c55e',
  down: '#ef4444',
  neutral: '#6b7280',
};

// Change direction arrows
const changeArrows: Record<ChangeDirection, string> = {
  up: '\u2191',
  down: '\u2193',
  neutral: '\u2192',
};

/**
 * StatCard - A reusable statistics card component
 *
 * Displays a labeled value with optional change indicator and icon.
 * Supports multiple color variants and sizes.
 *
 * @example
 * ```tsx
 * <StatCard
 *   label="Total Revenue"
 *   value="$45,231"
 *   change="+12.5%"
 *   changeDirection="up"
 *   icon={<DollarIcon />}
 *   variant="success"
 *   size="large"
 * />
 * ```
 */
export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  change,
  changeDirection = 'neutral',
  icon,
  variant = 'default',
  size = 'large',
  className = '',
  style,
  onClick,
}) => {
  const colors = variantColors[variant];
  const sizes = sizeConfig[size];

  const cardStyle: CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: sizes.padding,
    backgroundColor: colors.bg,
    border: `1px solid ${colors.border}`,
    borderRadius: '12px',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
    cursor: onClick ? 'pointer' : 'default',
    transition: 'all 0.2s ease-in-out',
    ...style,
  };

  const contentStyle: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  };

  const labelStyle: CSSProperties = {
    fontSize: sizes.labelSize,
    fontWeight: 500,
    color: '#6b7280',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    margin: 0,
  };

  const valueContainerStyle: CSSProperties = {
    display: 'flex',
    alignItems: 'baseline',
    gap: sizes.gap,
  };

  const valueStyle: CSSProperties = {
    fontSize: sizes.valueSize,
    fontWeight: 700,
    color: colors.text,
    lineHeight: 1.2,
    margin: 0,
  };

  const changeStyle: CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '2px',
    fontSize: sizes.changeSize,
    fontWeight: 500,
    color: changeColors[changeDirection],
    padding: '2px 6px',
    backgroundColor: `${changeColors[changeDirection]}15`,
    borderRadius: '4px',
  };

  const iconContainerStyle: CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: sizes.iconSize,
    height: sizes.iconSize,
    borderRadius: '10px',
    backgroundColor: `${colors.icon}15`,
    color: colors.icon,
    flexShrink: 0,
  };

  const handleMouseEnter = (e: React.MouseEvent<HTMLDivElement>) => {
    if (onClick) {
      e.currentTarget.style.transform = 'translateY(-2px)';
      e.currentTarget.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
    }
  };

  const handleMouseLeave = (e: React.MouseEvent<HTMLDivElement>) => {
    if (onClick) {
      e.currentTarget.style.transform = 'translateY(0)';
      e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)';
    }
  };

  return (
    <div
      className={className}
      style={cardStyle}
      onClick={onClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => e.key === 'Enter' && onClick() : undefined}
    >
      <div style={contentStyle}>
        <p style={labelStyle}>{label}</p>
        <div style={valueContainerStyle}>
          <p style={valueStyle}>{value}</p>
          {change && (
            <span style={changeStyle}>
              <span>{changeArrows[changeDirection]}</span>
              <span>{change}</span>
            </span>
          )}
        </div>
      </div>
      {icon && <div style={iconContainerStyle}>{icon}</div>}
    </div>
  );
};

// Default export for convenience
export default StatCard;
