import React, { useEffect, useState, useRef } from 'react';

interface PnLCardProps {
  value: number;
  previousValue: number;
  label: string;
}

const PnLCard: React.FC<PnLCardProps> = ({ value, previousValue, label }) => {
  const [isAnimating, setIsAnimating] = useState(false);
  const [displayValue, setDisplayValue] = useState(value);
  const prevValueRef = useRef(value);

  const isPositive = value >= 0;
  const change = previousValue !== 0 ? ((value - previousValue) / Math.abs(previousValue)) * 100 : 0;
  const changeIsPositive = change >= 0;

  useEffect(() => {
    if (prevValueRef.current !== value) {
      setIsAnimating(true);

      // Animate the value change
      const startValue = prevValueRef.current;
      const endValue = value;
      const duration = 500;
      const startTime = Date.now();

      const animateValue = () => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function for smooth animation
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const currentValue = startValue + (endValue - startValue) * easeOutQuart;

        setDisplayValue(currentValue);

        if (progress < 1) {
          requestAnimationFrame(animateValue);
        } else {
          setDisplayValue(endValue);
          setTimeout(() => setIsAnimating(false), 200);
        }
      };

      requestAnimationFrame(animateValue);
      prevValueRef.current = value;
    }
  }, [value]);

  const formatCurrency = (amount: number): string => {
    const absAmount = Math.abs(amount);
    const sign = amount >= 0 ? '+' : '-';
    return `${sign}$${absAmount.toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  };

  const formatPercentage = (percent: number): string => {
    const sign = percent >= 0 ? '+' : '';
    return `${sign}${percent.toFixed(2)}%`;
  };

  return (
    <div style={styles.card}>
      <div style={styles.labelContainer}>
        <span style={styles.label}>{label}</span>
      </div>

      <div
        style={{
          ...styles.valueContainer,
          ...(isAnimating ? styles.animating : {}),
        }}
      >
        <span
          style={{
            ...styles.value,
            color: isPositive ? '#00E676' : '#FF5252',
            textShadow: isPositive
              ? '0 0 20px rgba(0, 230, 118, 0.5)'
              : '0 0 20px rgba(255, 82, 82, 0.5)',
          }}
        >
          {formatCurrency(displayValue)}
        </span>
      </div>

      <div style={styles.changeContainer}>
        <span
          style={{
            ...styles.changeIndicator,
            backgroundColor: changeIsPositive
              ? 'rgba(0, 230, 118, 0.15)'
              : 'rgba(255, 82, 82, 0.15)',
            color: changeIsPositive ? '#00E676' : '#FF5252',
          }}
        >
          <span style={styles.arrow}>
            {changeIsPositive ? '\u25B2' : '\u25BC'}
          </span>
          {formatPercentage(change)}
        </span>
        <span style={styles.vsText}>vs previous</span>
      </div>
    </div>
  );
};

const styles: { [key: string]: React.CSSProperties } = {
  card: {
    backgroundColor: '#1E1E2E',
    borderRadius: '16px',
    padding: '24px',
    border: '1px solid #2D2D3D',
    boxShadow: '0 4px 24px rgba(0, 0, 0, 0.4)',
    minWidth: '280px',
    transition: 'transform 0.2s ease, box-shadow 0.2s ease',
  },
  labelContainer: {
    marginBottom: '12px',
  },
  label: {
    fontSize: '14px',
    fontWeight: 500,
    color: '#9CA3AF',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.5px',
  },
  valueContainer: {
    marginBottom: '16px',
    transition: 'transform 0.15s ease',
  },
  animating: {
    transform: 'scale(1.02)',
  },
  value: {
    fontSize: '42px',
    fontWeight: 700,
    fontFamily: "'SF Mono', 'Fira Code', 'Consolas', monospace",
    letterSpacing: '-1px',
    transition: 'color 0.3s ease, text-shadow 0.3s ease',
  },
  changeContainer: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  changeIndicator: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    padding: '6px 12px',
    borderRadius: '20px',
    fontSize: '14px',
    fontWeight: 600,
    fontFamily: "'SF Mono', 'Fira Code', 'Consolas', monospace",
  },
  arrow: {
    fontSize: '10px',
  },
  vsText: {
    fontSize: '13px',
    color: '#6B7280',
  },
};

export default PnLCard;
