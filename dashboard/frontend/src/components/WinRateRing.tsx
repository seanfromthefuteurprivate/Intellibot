import React from 'react';

interface WinRateRingProps {
  winRate: number;
  tradesCount: number;
  size?: number;
  strokeWidth?: number;
}

const WinRateRing: React.FC<WinRateRingProps> = ({
  winRate,
  tradesCount,
  size = 120,
  strokeWidth = 10,
}) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(Math.max(winRate, 0), 100);
  const strokeDashoffset = circumference - (progress / 100) * circumference;

  const getColor = (rate: number): string => {
    if (rate < 50) return '#ef4444'; // red
    if (rate <= 60) return '#eab308'; // yellow
    return '#22c55e'; // green
  };

  const color = getColor(winRate);

  return (
    <div
      style={{
        position: 'relative',
        width: size,
        height: size,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <svg
        width={size}
        height={size}
        style={{ transform: 'rotate(-90deg)' }}
      >
        {/* Background ring */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="#374151"
          strokeWidth={strokeWidth}
        />
        {/* Progress ring */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          style={{
            transition: 'stroke-dashoffset 0.5s ease-in-out, stroke 0.3s ease',
          }}
        />
      </svg>
      {/* Center content */}
      <div
        style={{
          position: 'absolute',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <span
          style={{
            fontSize: size * 0.22,
            fontWeight: 'bold',
            color: color,
            lineHeight: 1,
          }}
        >
          {winRate.toFixed(1)}%
        </span>
        <span
          style={{
            fontSize: size * 0.1,
            color: '#9ca3af',
            marginTop: 4,
          }}
        >
          {tradesCount} trades
        </span>
      </div>
    </div>
  );
};

export default WinRateRing;
