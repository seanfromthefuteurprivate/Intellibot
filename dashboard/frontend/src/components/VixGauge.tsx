import React from 'react';

interface VixGaugeProps {
  value: number;
  size?: number;
}

interface ZoneConfig {
  min: number;
  max: number;
  color: string;
  label: string;
}

const ZONES: ZoneConfig[] = [
  { min: 0, max: 15, color: '#22c55e', label: 'Calm' },
  { min: 15, max: 20, color: '#eab308', label: 'Elevated' },
  { min: 20, max: 30, color: '#f97316', label: 'High' },
  { min: 30, max: 50, color: '#ef4444', label: 'Extreme' },
];

const VixGauge: React.FC<VixGaugeProps> = ({ value, size = 200 }) => {
  const strokeWidth = size * 0.1;
  const radius = (size - strokeWidth) / 2;
  const center = size / 2;

  // Gauge spans from 135 degrees to 405 degrees (270 degree arc)
  const startAngle = 135;
  const endAngle = 405;
  const totalAngle = endAngle - startAngle;

  // Max VIX value for gauge (anything above will be capped visually)
  const maxVix = 50;
  const clampedValue = Math.min(Math.max(value, 0), maxVix);

  // Calculate the angle for the current value
  const valueAngle = startAngle + (clampedValue / maxVix) * totalAngle;

  // Convert angle to radians
  const toRadians = (angle: number) => (angle * Math.PI) / 180;

  // Get point on circle at given angle
  const getPoint = (angle: number, r: number) => ({
    x: center + r * Math.cos(toRadians(angle)),
    y: center + r * Math.sin(toRadians(angle)),
  });

  // Create arc path
  const createArc = (startDeg: number, endDeg: number, r: number) => {
    const start = getPoint(startDeg, r);
    const end = getPoint(endDeg, r);
    const largeArcFlag = endDeg - startDeg > 180 ? 1 : 0;

    return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`;
  };

  // Get zone segments for the arc
  const getZoneArcs = () => {
    return ZONES.map((zone) => {
      const zoneStartPercent = zone.min / maxVix;
      const zoneEndPercent = zone.max / maxVix;
      const zoneStartAngle = startAngle + zoneStartPercent * totalAngle;
      const zoneEndAngle = startAngle + zoneEndPercent * totalAngle;

      return {
        ...zone,
        startAngle: zoneStartAngle,
        endAngle: zoneEndAngle,
        path: createArc(zoneStartAngle, zoneEndAngle, radius),
      };
    });
  };

  // Get current zone based on value
  const getCurrentZone = (): ZoneConfig => {
    for (const zone of ZONES) {
      if (value >= zone.min && value < zone.max) {
        return zone;
      }
    }
    return ZONES[ZONES.length - 1];
  };

  const currentZone = getCurrentZone();
  const zoneArcs = getZoneArcs();

  // Needle endpoint
  const needleLength = radius - strokeWidth * 0.5;
  const needleEnd = getPoint(valueAngle, needleLength);
  const needleBase = getPoint(valueAngle, strokeWidth * 0.3);

  // Create needle polygon points
  const needleWidth = strokeWidth * 0.15;
  const perpAngle = valueAngle + 90;
  const baseLeft = getPoint(perpAngle, needleWidth);
  const baseRight = getPoint(perpAngle + 180, needleWidth);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      }}
    >
      <svg width={size} height={size * 0.85} viewBox={`0 0 ${size} ${size * 0.85}`}>
        {/* Zone arcs */}
        {zoneArcs.map((zone, index) => (
          <path
            key={index}
            d={zone.path}
            fill="none"
            stroke={zone.color}
            strokeWidth={strokeWidth}
            strokeLinecap="butt"
            opacity={0.3}
          />
        ))}

        {/* Active arc up to current value */}
        {clampedValue > 0 && (
          <path
            d={createArc(startAngle, valueAngle, radius)}
            fill="none"
            stroke={currentZone.color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
          />
        )}

        {/* Needle */}
        <line
          x1={center}
          y1={center}
          x2={needleEnd.x}
          y2={needleEnd.y}
          stroke={currentZone.color}
          strokeWidth={strokeWidth * 0.15}
          strokeLinecap="round"
        />

        {/* Needle center circle */}
        <circle
          cx={center}
          cy={center}
          r={strokeWidth * 0.4}
          fill={currentZone.color}
        />
        <circle
          cx={center}
          cy={center}
          r={strokeWidth * 0.25}
          fill="#1f2937"
        />

        {/* Value display */}
        <text
          x={center}
          y={center + radius * 0.35}
          textAnchor="middle"
          fontSize={size * 0.18}
          fontWeight="bold"
          fill={currentZone.color}
        >
          {value.toFixed(1)}
        </text>

        {/* Zone label */}
        <text
          x={center}
          y={center + radius * 0.55}
          textAnchor="middle"
          fontSize={size * 0.08}
          fontWeight="500"
          fill={currentZone.color}
          textTransform="uppercase"
        >
          {currentZone.label}
        </text>

        {/* VIX label */}
        <text
          x={center}
          y={center + radius * 0.7}
          textAnchor="middle"
          fontSize={size * 0.06}
          fill="#9ca3af"
        >
          VIX Index
        </text>

        {/* Zone tick marks and labels */}
        {[0, 15, 20, 30, 50].map((tickValue, index) => {
          const tickAngle = startAngle + (tickValue / maxVix) * totalAngle;
          const outerPoint = getPoint(tickAngle, radius + strokeWidth * 0.6);
          const innerPoint = getPoint(tickAngle, radius + strokeWidth * 0.3);
          const labelPoint = getPoint(tickAngle, radius + strokeWidth * 1.1);

          return (
            <g key={index}>
              <line
                x1={innerPoint.x}
                y1={innerPoint.y}
                x2={outerPoint.x}
                y2={outerPoint.y}
                stroke="#6b7280"
                strokeWidth={1.5}
              />
              <text
                x={labelPoint.x}
                y={labelPoint.y}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={size * 0.05}
                fill="#9ca3af"
              >
                {tickValue}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Zone legend */}
      <div
        style={{
          display: 'flex',
          gap: size * 0.08,
          marginTop: size * 0.05,
          flexWrap: 'wrap',
          justifyContent: 'center',
        }}
      >
        {ZONES.map((zone, index) => (
          <div
            key={index}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: size * 0.02,
              opacity: currentZone.label === zone.label ? 1 : 0.5,
            }}
          >
            <div
              style={{
                width: size * 0.04,
                height: size * 0.04,
                borderRadius: '50%',
                backgroundColor: zone.color,
              }}
            />
            <span
              style={{
                fontSize: size * 0.055,
                color: currentZone.label === zone.label ? zone.color : '#9ca3af',
                fontWeight: currentZone.label === zone.label ? 600 : 400,
              }}
            >
              {zone.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default VixGauge;
