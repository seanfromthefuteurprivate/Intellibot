import React from 'react';

interface ProgressBarProps {
  value: number;
  max: number;
  label?: string;
  showPercentage?: boolean;
  height?: number;
  className?: string;
}

const getColorByPercentage = (percentage: number): string => {
  if (percentage <= 33) {
    return '#22c55e'; // green-500
  } else if (percentage <= 66) {
    return '#eab308'; // yellow-500
  } else {
    return '#ef4444'; // red-500
  }
};

const getBackgroundColorByPercentage = (percentage: number): string => {
  if (percentage <= 33) {
    return 'rgba(34, 197, 94, 0.2)'; // green with opacity
  } else if (percentage <= 66) {
    return 'rgba(234, 179, 8, 0.2)'; // yellow with opacity
  } else {
    return 'rgba(239, 68, 68, 0.2)'; // red with opacity
  }
};

const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max,
  label,
  showPercentage = true,
  height = 20,
  className = '',
}) => {
  const clampedValue = Math.min(Math.max(value, 0), max);
  const percentage = max > 0 ? (clampedValue / max) * 100 : 0;
  const roundedPercentage = Math.round(percentage);

  const fillColor = getColorByPercentage(percentage);
  const backgroundColor = getBackgroundColorByPercentage(percentage);

  const containerStyle: React.CSSProperties = {
    width: '100%',
    height: `${height}px`,
    backgroundColor: backgroundColor,
    borderRadius: `${height / 2}px`,
    overflow: 'hidden',
    position: 'relative',
  };

  const fillStyle: React.CSSProperties = {
    width: `${percentage}%`,
    height: '100%',
    backgroundColor: fillColor,
    borderRadius: `${height / 2}px`,
    transition: 'width 0.5s ease-in-out, background-color 0.3s ease',
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
  };

  const labelContainerStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '6px',
    fontSize: '14px',
    fontWeight: 500,
  };

  const percentageTextStyle: React.CSSProperties = {
    position: 'absolute',
    right: '8px',
    top: '50%',
    transform: 'translateY(-50%)',
    fontSize: '12px',
    fontWeight: 600,
    color: percentage > 50 ? '#ffffff' : fillColor,
    textShadow: percentage > 50 ? '0 1px 2px rgba(0,0,0,0.3)' : 'none',
  };

  const externalPercentageStyle: React.CSSProperties = {
    position: 'absolute',
    left: `calc(${percentage}% + 8px)`,
    top: '50%',
    transform: 'translateY(-50%)',
    fontSize: '12px',
    fontWeight: 600,
    color: fillColor,
  };

  return (
    <div className={className}>
      {label && (
        <div style={labelContainerStyle}>
          <span>{label}</span>
          {showPercentage && <span style={{ color: fillColor }}>{roundedPercentage}%</span>}
        </div>
      )}
      <div style={containerStyle}>
        <div style={fillStyle}>
          {/* Animated shine effect */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: '50%',
              background: 'linear-gradient(180deg, rgba(255,255,255,0.3) 0%, rgba(255,255,255,0) 100%)',
              borderRadius: `${height / 2}px ${height / 2}px 0 0`,
            }}
          />
        </div>
        {!label && showPercentage && (
          percentage > 30 ? (
            <span style={percentageTextStyle}>{roundedPercentage}%</span>
          ) : (
            <span style={externalPercentageStyle}>{roundedPercentage}%</span>
          )
        )}
      </div>
    </div>
  );
};

export default ProgressBar;
