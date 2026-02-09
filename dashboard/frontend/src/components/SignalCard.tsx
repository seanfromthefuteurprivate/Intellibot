"use client";

import * as React from "react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Card,
  CardContent,
  CardHeader,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
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
  { bgColor: string; textColor: string; borderColor: string; label: string }
> = {
  S: {
    bgColor: "bg-gradient-to-r from-yellow-500 to-amber-500",
    textColor: "text-white",
    borderColor: "border-yellow-400",
    label: "S-Tier",
  },
  A: {
    bgColor: "bg-gradient-to-r from-emerald-500 to-green-500",
    textColor: "text-white",
    borderColor: "border-emerald-400",
    label: "A-Tier",
  },
  B: {
    bgColor: "bg-gradient-to-r from-blue-500 to-cyan-500",
    textColor: "text-white",
    borderColor: "border-blue-400",
    label: "B-Tier",
  },
  C: {
    bgColor: "bg-gradient-to-r from-slate-500 to-gray-500",
    textColor: "text-white",
    borderColor: "border-slate-400",
    label: "C-Tier",
  },
};

// Score bar color based on score value
function getScoreBarColor(score: number): string {
  if (score >= 90) return "bg-gradient-to-r from-yellow-400 to-amber-500";
  if (score >= 75) return "bg-gradient-to-r from-emerald-400 to-green-500";
  if (score >= 60) return "bg-gradient-to-r from-blue-400 to-cyan-500";
  if (score >= 40) return "bg-gradient-to-r from-orange-400 to-amber-500";
  return "bg-gradient-to-r from-red-400 to-rose-500";
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

export function SignalCard({ signal, className }: SignalCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const tierStyle = tierConfig[signal.tier];
  const scoreBarColor = getScoreBarColor(signal.score);
  const hasDetails = signal.details && Object.keys(signal.details).length > 0;

  return (
    <Card
      className={cn(
        "relative overflow-hidden transition-all duration-200 hover:shadow-lg",
        `border-l-4 ${tierStyle.borderColor}`,
        className
      )}
    >
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          {/* Ticker Symbol */}
          <div className="flex items-center gap-3">
            <h3 className="text-3xl font-bold tracking-tight text-foreground">
              {signal.ticker}
            </h3>
            <TrendingUp className="h-5 w-5 text-muted-foreground" />
          </div>

          {/* Tier Badge */}
          <Badge
            className={cn(
              "px-3 py-1 text-sm font-bold",
              tierStyle.bgColor,
              tierStyle.textColor
            )}
          >
            {tierStyle.label}
          </Badge>
        </div>

        {/* Setup Type */}
        <p className="text-sm font-medium text-muted-foreground">
          {signal.setupType}
        </p>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Score Section */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-muted-foreground">
              Signal Score
            </span>
            <span className="text-lg font-bold text-foreground">
              {signal.score}
              <span className="text-sm font-normal text-muted-foreground">
                /100
              </span>
            </span>
          </div>

          {/* Visual Score Bar */}
          <div className="relative h-3 w-full overflow-hidden rounded-full bg-secondary">
            <div
              className={cn(
                "h-full transition-all duration-500 ease-out rounded-full",
                scoreBarColor
              )}
              style={{ width: `${Math.min(Math.max(signal.score, 0), 100)}%` }}
            />
          </div>
        </div>

        {/* Timestamp */}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Clock className="h-4 w-4" />
          <span>{formatTimestamp(signal.timestamp)}</span>
        </div>

        {/* Expandable Details Section */}
        {hasDetails && (
          <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
            <CollapsibleTrigger asChild>
              <button
                className={cn(
                  "flex w-full items-center justify-between rounded-lg border border-border bg-muted/50 px-4 py-2",
                  "text-sm font-medium text-muted-foreground",
                  "transition-colors hover:bg-muted hover:text-foreground",
                  "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                )}
              >
                <span>View Details</span>
                {isExpanded ? (
                  <ChevronUp className="h-4 w-4" />
                ) : (
                  <ChevronDown className="h-4 w-4" />
                )}
              </button>
            </CollapsibleTrigger>

            <CollapsibleContent className="mt-3">
              <div className="rounded-lg border border-border bg-muted/30 p-4 space-y-3">
                {/* Price Targets */}
                {(signal.details?.entryPrice ||
                  signal.details?.targetPrice ||
                  signal.details?.stopLoss) && (
                  <div className="grid grid-cols-3 gap-4">
                    {signal.details.entryPrice !== undefined && (
                      <div className="space-y-1">
                        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                          Entry
                        </p>
                        <p className="text-sm font-semibold text-foreground">
                          {formatCurrency(signal.details.entryPrice)}
                        </p>
                      </div>
                    )}
                    {signal.details.targetPrice !== undefined && (
                      <div className="space-y-1">
                        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                          Target
                        </p>
                        <p className="text-sm font-semibold text-emerald-500">
                          {formatCurrency(signal.details.targetPrice)}
                        </p>
                      </div>
                    )}
                    {signal.details.stopLoss !== undefined && (
                      <div className="space-y-1">
                        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                          Stop Loss
                        </p>
                        <p className="text-sm font-semibold text-red-500">
                          {formatCurrency(signal.details.stopLoss)}
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {/* Risk/Reward and Volume */}
                <div className="flex flex-wrap gap-4">
                  {signal.details?.riskRewardRatio !== undefined && (
                    <div className="space-y-1">
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                        R/R Ratio
                      </p>
                      <p className="text-sm font-semibold text-foreground">
                        1:{signal.details.riskRewardRatio.toFixed(1)}
                      </p>
                    </div>
                  )}
                  {signal.details?.volume !== undefined && (
                    <div className="space-y-1">
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                        Volume
                      </p>
                      <p className="text-sm font-semibold text-foreground">
                        {formatVolume(signal.details.volume)}
                      </p>
                    </div>
                  )}
                  {signal.details?.momentum && (
                    <div className="space-y-1">
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                        Momentum
                      </p>
                      <p className="text-sm font-semibold text-foreground">
                        {signal.details.momentum}
                      </p>
                    </div>
                  )}
                </div>

                {/* Catalysts */}
                {signal.details?.catalysts &&
                  signal.details.catalysts.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                        Catalysts
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {signal.details.catalysts.map((catalyst, index) => (
                          <Badge
                            key={index}
                            variant="secondary"
                            className="text-xs"
                          >
                            {catalyst}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                {/* Notes */}
                {signal.details?.notes && (
                  <div className="space-y-1">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                      Notes
                    </p>
                    <p className="text-sm text-foreground leading-relaxed">
                      {signal.details.notes}
                    </p>
                  </div>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}
      </CardContent>
    </Card>
  );
}

export default SignalCard;
