"use client";

import * as React from "react";
import { useState } from "react";
import { ChevronDown, ChevronUp, Clock, TrendingUp } from "lucide-react";

// Signal tier types
type SignalTier = "S" | "A" | "B" | "C";

// Signal data interface
export interface SignalData {
  ticker: string;
  score: number;
  tier: SignalTier;
  setupType: string;
  timestamp: string | Date;
  details?: {
    entryPrice?: number;
    targetPrice?: number;
    stopLoss?: number;
    riskRewardRatio?: number;
    volume?: number;
    momentum?: string;
    catalysts?: string[];
    notes?: string;
  };
}

export interface SignalCardProps {
  signal: SignalData;
  className?: string;
}

// Tier configuration for colors and styling
const tierConfig: Record<
  SignalTier,
  { bgGradient: string; borderColor: string; label: string }
> = {
  S: {
    bgGradient: "linear-gradient(to right, #eab308, #f59e0b)",
    borderColor: "#facc15",
    label: "S-Tier",
  },
  A: {
    bgGradient: "linear-gradient(to right, #10b981, #22c55e)",
    borderColor: "#34d399",
    label: "A-Tier",
  },
  B: {
    bgGradient: "linear-gradient(to right, #3b82f6, #06b6d4)",
    borderColor: "#60a5fa",
    label: "B-Tier",
  },
  C: {
    bgGradient: "linear-gradient(to right, #64748b, #6b7280)",
    borderColor: "#94a3b8",
    label: "C-Tier",
  },
};

// Score bar color based on score value
function getScoreBarGradient(score: number): string {
  if (score >= 90) return "linear-gradient(to right, #facc15, #f59e0b)";
  if (score >= 75) return "linear-gradient(to right, #34d399, #22c55e)";
  if (score >= 60) return "linear-gradient(to right, #60a5fa, #06b6d4)";
  if (score >= 40) return "linear-gradient(to right, #fb923c, #f59e0b)";
  return "linear-gradient(to right, #f87171, #f43f5e)";
}

// Format timestamp for display
function formatTimestamp(timestamp: string | Date): string {
  const date = typeof timestamp === "string" ? new Date(timestamp) : timestamp;
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// Format currency values
function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

// Format large numbers with abbreviations
function formatVolume(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toString();
}

// Inline styles
const styles: Record<string, React.CSSProperties> = {
  card: {
    position: "relative",
    overflow: "hidden",
    transition: "all 0.2s ease",
    borderRadius: "8px",
    backgroundColor: "#ffffff",
    boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
    borderWidth: "1px",
    borderStyle: "solid",
    borderColor: "#e5e7eb",
  },
  cardHeader: {
    padding: "16px 16px 8px 16px",
  },
  headerRow: {
    display: "flex",
    alignItems: "flex-start",
    justifyContent: "space-between",
  },
  tickerContainer: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  ticker: {
    fontSize: "1.875rem",
    fontWeight: 700,
    letterSpacing: "-0.025em",
    color: "#111827",
    margin: 0,
  },
  trendIcon: {
    color: "#6b7280",
  },
  badge: {
    padding: "4px 12px",
    fontSize: "0.875rem",
    fontWeight: 700,
    color: "#ffffff",
    borderRadius: "9999px",
    display: "inline-block",
  },
  setupType: {
    fontSize: "0.875rem",
    fontWeight: 500,
    color: "#6b7280",
    marginTop: "4px",
  },
  cardContent: {
    padding: "0 16px 16px 16px",
    display: "flex",
    flexDirection: "column" as const,
    gap: "16px",
  },
  scoreSection: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "8px",
  },
  scoreHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  scoreLabel: {
    fontSize: "0.875rem",
    fontWeight: 500,
    color: "#6b7280",
  },
  scoreValue: {
    fontSize: "1.125rem",
    fontWeight: 700,
    color: "#111827",
  },
  scoreMax: {
    fontSize: "0.875rem",
    fontWeight: 400,
    color: "#6b7280",
  },
  scoreBarContainer: {
    position: "relative" as const,
    height: "12px",
    width: "100%",
    overflow: "hidden",
    borderRadius: "9999px",
    backgroundColor: "#f3f4f6",
  },
  scoreBar: {
    height: "100%",
    transition: "all 0.5s ease-out",
    borderRadius: "9999px",
  },
  timestamp: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    fontSize: "0.875rem",
    color: "#6b7280",
  },
  expandButton: {
    display: "flex",
    width: "100%",
    alignItems: "center",
    justifyContent: "space-between",
    borderRadius: "8px",
    border: "1px solid #e5e7eb",
    backgroundColor: "rgba(249, 250, 251, 0.5)",
    padding: "8px 16px",
    fontSize: "0.875rem",
    fontWeight: 500,
    color: "#6b7280",
    cursor: "pointer",
    transition: "all 0.15s ease",
  },
  detailsContainer: {
    marginTop: "12px",
    borderRadius: "8px",
    border: "1px solid #e5e7eb",
    backgroundColor: "rgba(249, 250, 251, 0.3)",
    padding: "16px",
    display: "flex",
    flexDirection: "column" as const,
    gap: "12px",
  },
  priceGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: "16px",
  },
  detailItem: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "4px",
  },
  detailLabel: {
    fontSize: "0.75rem",
    fontWeight: 500,
    color: "#6b7280",
    textTransform: "uppercase" as const,
    letterSpacing: "0.05em",
  },
  detailValue: {
    fontSize: "0.875rem",
    fontWeight: 600,
    color: "#111827",
  },
  detailValueGreen: {
    fontSize: "0.875rem",
    fontWeight: 600,
    color: "#10b981",
  },
  detailValueRed: {
    fontSize: "0.875rem",
    fontWeight: 600,
    color: "#ef4444",
  },
  metricsRow: {
    display: "flex",
    flexWrap: "wrap" as const,
    gap: "16px",
  },
  catalystsContainer: {
    display: "flex",
    flexDirection: "column" as const,
    gap: "8px",
  },
  catalystsList: {
    display: "flex",
    flexWrap: "wrap" as const,
    gap: "8px",
  },
  catalystBadge: {
    fontSize: "0.75rem",
    padding: "4px 8px",
    backgroundColor: "#f3f4f6",
    color: "#374151",
    borderRadius: "4px",
    display: "inline-block",
  },
  notesText: {
    fontSize: "0.875rem",
    color: "#111827",
    lineHeight: 1.625,
  },
};

function SignalCard({ signal, className }: SignalCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const tierStyle = tierConfig[signal.tier];
  const scoreBarGradient = getScoreBarGradient(signal.score);
  const hasDetails = signal.details && Object.keys(signal.details).length > 0;

  const cardStyle: React.CSSProperties = {
    ...styles.card,
    borderLeftWidth: "4px",
    borderLeftColor: tierStyle.borderColor,
    boxShadow: isHovered
      ? "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)"
      : styles.card.boxShadow,
  };

  return (
    <div
      className={className}
      style={cardStyle}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Card Header */}
      <div style={styles.cardHeader}>
        <div style={styles.headerRow}>
          {/* Ticker Symbol */}
          <div style={styles.tickerContainer}>
            <h3 style={styles.ticker}>{signal.ticker}</h3>
            <TrendingUp size={20} style={styles.trendIcon} />
          </div>

          {/* Tier Badge */}
          <span
            style={{
              ...styles.badge,
              background: tierStyle.bgGradient,
            }}
          >
            {tierStyle.label}
          </span>
        </div>

        {/* Setup Type */}
        <p style={styles.setupType}>{signal.setupType}</p>
      </div>

      {/* Card Content */}
      <div style={styles.cardContent}>
        {/* Score Section */}
        <div style={styles.scoreSection}>
          <div style={styles.scoreHeader}>
            <span style={styles.scoreLabel}>Signal Score</span>
            <span style={styles.scoreValue}>
              {signal.score}
              <span style={styles.scoreMax}>/100</span>
            </span>
          </div>

          {/* Visual Score Bar */}
          <div style={styles.scoreBarContainer}>
            <div
              style={{
                ...styles.scoreBar,
                width: `${Math.min(Math.max(signal.score, 0), 100)}%`,
                background: scoreBarGradient,
              }}
            />
          </div>
        </div>

        {/* Timestamp */}
        <div style={styles.timestamp}>
          <Clock size={16} />
          <span>{formatTimestamp(signal.timestamp)}</span>
        </div>

        {/* Expandable Details Section */}
        {hasDetails && (
          <div>
            <button
              style={styles.expandButton}
              onClick={() => setIsExpanded(!isExpanded)}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = "#f3f4f6";
                e.currentTarget.style.color = "#111827";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor =
                  "rgba(249, 250, 251, 0.5)";
                e.currentTarget.style.color = "#6b7280";
              }}
            >
              <span>View Details</span>
              {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>

            {isExpanded && (
              <div style={styles.detailsContainer}>
                {/* Price Targets */}
                {(signal.details?.entryPrice ||
                  signal.details?.targetPrice ||
                  signal.details?.stopLoss) && (
                  <div style={styles.priceGrid}>
                    {signal.details.entryPrice !== undefined && (
                      <div style={styles.detailItem}>
                        <p style={styles.detailLabel}>Entry</p>
                        <p style={styles.detailValue}>
                          {formatCurrency(signal.details.entryPrice)}
                        </p>
                      </div>
                    )}
                    {signal.details.targetPrice !== undefined && (
                      <div style={styles.detailItem}>
                        <p style={styles.detailLabel}>Target</p>
                        <p style={styles.detailValueGreen}>
                          {formatCurrency(signal.details.targetPrice)}
                        </p>
                      </div>
                    )}
                    {signal.details.stopLoss !== undefined && (
                      <div style={styles.detailItem}>
                        <p style={styles.detailLabel}>Stop Loss</p>
                        <p style={styles.detailValueRed}>
                          {formatCurrency(signal.details.stopLoss)}
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* Risk/Reward and Volume */}
                <div style={styles.metricsRow}>
                  {signal.details?.riskRewardRatio !== undefined && (
                    <div style={styles.detailItem}>
                      <p style={styles.detailLabel}>R/R Ratio</p>
                      <p style={styles.detailValue}>
                        1:{signal.details.riskRewardRatio.toFixed(1)}
                      </p>
                    </div>
                  )}
                  {signal.details?.volume !== undefined && (
                    <div style={styles.detailItem}>
                      <p style={styles.detailLabel}>Volume</p>
                      <p style={styles.detailValue}>
                        {formatVolume(signal.details.volume)}
                      </p>
                    </div>
                  )}
                  {signal.details?.momentum && (
                    <div style={styles.detailItem}>
                      <p style={styles.detailLabel}>Momentum</p>
                      <p style={styles.detailValue}>
                        {signal.details.momentum}
                      </p>
                    </div>
                  )}
                </div>

                {/* Catalysts */}
                {signal.details?.catalysts &&
                  signal.details.catalysts.length > 0 && (
                    <div style={styles.catalystsContainer}>
                      <p style={styles.detailLabel}>Catalysts</p>
                      <div style={styles.catalystsList}>
                        {signal.details.catalysts.map((catalyst, index) => (
                          <span key={index} style={styles.catalystBadge}>
                            {catalyst}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                {/* Notes */}
                {signal.details?.notes && (
                  <div style={styles.detailItem}>
                    <p style={styles.detailLabel}>Notes</p>
                    <p style={styles.notesText}>{signal.details.notes}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default SignalCard;
