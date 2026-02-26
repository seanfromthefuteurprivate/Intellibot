"""
Trading Thesis Structure - Evidence-Based Trade Analysis

Based on TauricResearch/Trading-R1 patterns:
- Structured thesis composition with evidence linking
- Facts-grounded analysis where every claim links to data source
- Volatility-adjusted decision making

Key Features:
1. Each signal component traced to data source
2. Risk factors explicitly enumerated
3. Volatility adjustment for regime changes
4. Confidence breakdown by factor
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Optional, Any
import json

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class EvidenceSource(Enum):
    """Source types for evidence linking."""
    PRICE_ACTION = "price_action"
    VOLUME = "volume"
    OPTIONS_FLOW = "options_flow"
    TECHNICAL_INDICATOR = "technical"
    NEWS_SENTIMENT = "news"
    AI_MODEL = "ai_model"
    HYDRA_INTEL = "hydra"
    GEX_ANALYSIS = "gex"
    DARK_POOL = "dark_pool"
    PATTERN_MEMORY = "pattern_memory"
    TIME_LEARNING = "time_learning"
    DEBATE_CONSENSUS = "debate"
    SPECIALIST_SWARM = "swarm"


class ConfidenceLevel(Enum):
    """Confidence levels for claims."""
    HIGH = "high"       # 80%+ confidence in the claim
    MEDIUM = "medium"   # 60-80% confidence
    LOW = "low"         # 40-60% confidence
    SPECULATIVE = "speculative"  # <40% confidence


@dataclass
class EvidenceItem:
    """
    A single piece of evidence supporting or refuting the thesis.

    Based on Trading-R1's facts-grounded analysis approach.
    """
    claim: str                      # The claim being made
    source: EvidenceSource          # Where this evidence comes from
    source_detail: str              # Specific source (e.g., "RSI", "HYDRA GEX")
    data_value: Any                 # The actual data value
    confidence: ConfidenceLevel     # How confident we are in this evidence
    supports_direction: bool        # True if supports thesis, False if contradicts
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_data: Optional[Dict] = None  # Optional raw data for debugging

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "claim": self.claim,
            "source": self.source.value,
            "source_detail": self.source_detail,
            "data_value": str(self.data_value),
            "confidence": self.confidence.value,
            "supports_direction": self.supports_direction,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class RiskFactor:
    """
    A risk factor that could invalidate the thesis.
    """
    description: str                # What the risk is
    severity: str                   # "high", "medium", "low"
    probability: float              # 0-1 probability of occurring
    mitigation: Optional[str] = None  # How to mitigate
    source: Optional[EvidenceSource] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "description": self.description,
            "severity": self.severity,
            "probability": self.probability,
            "mitigation": self.mitigation,
            "source": self.source.value if self.source else None,
        }


@dataclass
class VolatilityAdjustment:
    """
    Volatility-based adjustment to the thesis.

    Based on Trading-R1's volatility-adjusted decision making.
    """
    vix_level: float
    regime: str                     # "low", "normal", "elevated", "crisis"
    position_size_multiplier: float  # 1.0 = normal, <1 = reduce, >1 = increase
    stop_distance_multiplier: float  # Wider stops in high vol
    target_distance_multiplier: float
    time_in_trade_adjustment: float  # Hold shorter in high vol
    reasoning: str


@dataclass
class TradingThesis:
    """
    Structured investment thesis for a trade.

    Based on TauricResearch/Trading-R1 pattern:
    - Evidence-based with clear source attribution
    - Explicit risk factors
    - Volatility adjustment
    - Confidence breakdown
    """
    # Core thesis
    ticker: str
    direction: str                  # "long" or "short"
    thesis_summary: str             # 1-2 sentence summary
    time_horizon: str               # "0dte", "swing", "position"

    # Evidence (pro and con)
    supporting_evidence: List[EvidenceItem] = field(default_factory=list)
    contradicting_evidence: List[EvidenceItem] = field(default_factory=list)

    # Risk assessment
    risk_factors: List[RiskFactor] = field(default_factory=list)
    max_acceptable_loss_pct: float = 2.0

    # Conviction breakdown
    pattern_conviction: float = 50.0      # From pattern recognition
    flow_conviction: float = 50.0         # From order flow
    technical_conviction: float = 50.0    # From technicals
    sentiment_conviction: float = 50.0    # From news/sentiment
    ai_conviction: float = 50.0           # From AI models
    final_conviction: float = 50.0        # Weighted aggregate

    # Volatility adjustment
    volatility_adjustment: Optional[VolatilityAdjustment] = None

    # Entry/exit parameters (after adjustment)
    entry_price: float = 0.0
    target_price: float = 0.0
    stop_price: float = 0.0
    position_size_pct: float = 1.0  # % of max position

    # Metadata
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    thesis_version: int = 1

    def calculate_final_conviction(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate weighted final conviction from components.

        Args:
            weights: Optional custom weights for each component.
                     Defaults to equal weighting.
        """
        if weights is None:
            weights = {
                "pattern": 0.25,
                "flow": 0.20,
                "technical": 0.20,
                "sentiment": 0.15,
                "ai": 0.20,
            }

        weighted_sum = (
            self.pattern_conviction * weights.get("pattern", 0.2) +
            self.flow_conviction * weights.get("flow", 0.2) +
            self.technical_conviction * weights.get("technical", 0.2) +
            self.sentiment_conviction * weights.get("sentiment", 0.2) +
            self.ai_conviction * weights.get("ai", 0.2)
        )

        # Adjust for contradicting evidence
        contradiction_penalty = len(self.contradicting_evidence) * 3
        weighted_sum = max(0, weighted_sum - contradiction_penalty)

        # Adjust for high-severity risks
        high_risk_count = sum(1 for r in self.risk_factors if r.severity == "high")
        risk_penalty = high_risk_count * 5
        weighted_sum = max(0, weighted_sum - risk_penalty)

        self.final_conviction = min(100, max(0, weighted_sum))
        return self.final_conviction

    def apply_volatility_adjustment(
        self,
        vix_level: float,
        base_stop_pct: float = 2.0,
        base_target_pct: float = 4.0,
    ) -> None:
        """
        Apply volatility-based adjustments to the thesis.

        Args:
            vix_level: Current VIX level
            base_stop_pct: Base stop loss percentage
            base_target_pct: Base target percentage
        """
        # Determine regime
        if vix_level < 15:
            regime = "low"
            size_mult = 1.2   # Can be more aggressive
            stop_mult = 0.8   # Tighter stops
            target_mult = 0.9
            time_adj = 1.2    # Can hold longer
        elif vix_level < 20:
            regime = "normal"
            size_mult = 1.0
            stop_mult = 1.0
            target_mult = 1.0
            time_adj = 1.0
        elif vix_level < 30:
            regime = "elevated"
            size_mult = 0.7   # Reduce size
            stop_mult = 1.3   # Wider stops
            target_mult = 1.2  # Bigger targets possible
            time_adj = 0.7    # Exit faster
        else:
            regime = "crisis"
            size_mult = 0.4   # Much smaller positions
            stop_mult = 2.0   # Very wide stops
            target_mult = 1.5  # Big moves possible
            time_adj = 0.5    # Quick exits

        self.volatility_adjustment = VolatilityAdjustment(
            vix_level=vix_level,
            regime=regime,
            position_size_multiplier=size_mult,
            stop_distance_multiplier=stop_mult,
            target_distance_multiplier=target_mult,
            time_in_trade_adjustment=time_adj,
            reasoning=f"VIX at {vix_level:.1f} indicates {regime} volatility regime",
        )

        # Apply to position parameters
        self.position_size_pct = size_mult
        if self.entry_price > 0:
            adjusted_stop_pct = base_stop_pct * stop_mult
            adjusted_target_pct = base_target_pct * target_mult

            if self.direction == "long":
                self.stop_price = self.entry_price * (1 - adjusted_stop_pct / 100)
                self.target_price = self.entry_price * (1 + adjusted_target_pct / 100)
            else:
                self.stop_price = self.entry_price * (1 + adjusted_stop_pct / 100)
                self.target_price = self.entry_price * (1 - adjusted_target_pct / 100)

    def add_evidence(
        self,
        claim: str,
        source: EvidenceSource,
        source_detail: str,
        data_value: Any,
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM,
        supports: bool = True,
    ) -> None:
        """
        Add evidence to the thesis.

        Args:
            claim: The claim this evidence supports
            source: Source type
            source_detail: Specific source name
            data_value: The data value
            confidence: Confidence level
            supports: True if supports thesis, False if contradicts
        """
        evidence = EvidenceItem(
            claim=claim,
            source=source,
            source_detail=source_detail,
            data_value=data_value,
            confidence=confidence,
            supports_direction=supports,
        )

        if supports:
            self.supporting_evidence.append(evidence)
        else:
            self.contradicting_evidence.append(evidence)

    def add_risk(
        self,
        description: str,
        severity: str = "medium",
        probability: float = 0.3,
        mitigation: Optional[str] = None,
        source: Optional[EvidenceSource] = None,
    ) -> None:
        """Add a risk factor."""
        risk = RiskFactor(
            description=description,
            severity=severity,
            probability=probability,
            mitigation=mitigation,
            source=source,
        )
        self.risk_factors.append(risk)

    def get_evidence_summary(self) -> str:
        """Get human-readable evidence summary."""
        lines = []

        if self.supporting_evidence:
            lines.append("SUPPORTING EVIDENCE:")
            for e in self.supporting_evidence[:5]:  # Top 5
                lines.append(f"  ✓ [{e.source_detail}] {e.claim} ({e.confidence.value})")

        if self.contradicting_evidence:
            lines.append("CONTRADICTING EVIDENCE:")
            for e in self.contradicting_evidence[:3]:  # Top 3
                lines.append(f"  ✗ [{e.source_detail}] {e.claim} ({e.confidence.value})")

        if self.risk_factors:
            lines.append("RISK FACTORS:")
            for r in self.risk_factors[:3]:
                lines.append(f"  ⚠ [{r.severity}] {r.description}")

        return "\n".join(lines)

    def get_conviction_breakdown(self) -> str:
        """Get conviction breakdown as string."""
        return (
            f"Conviction Breakdown:\n"
            f"  Pattern: {self.pattern_conviction:.0f}%\n"
            f"  Flow: {self.flow_conviction:.0f}%\n"
            f"  Technical: {self.technical_conviction:.0f}%\n"
            f"  Sentiment: {self.sentiment_conviction:.0f}%\n"
            f"  AI: {self.ai_conviction:.0f}%\n"
            f"  ─────────────\n"
            f"  Final: {self.final_conviction:.0f}%"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/serialization."""
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "thesis_summary": self.thesis_summary,
            "time_horizon": self.time_horizon,
            "supporting_evidence": [e.to_dict() for e in self.supporting_evidence],
            "contradicting_evidence": [e.to_dict() for e in self.contradicting_evidence],
            "risk_factors": [r.to_dict() for r in self.risk_factors],
            "conviction": {
                "pattern": self.pattern_conviction,
                "flow": self.flow_conviction,
                "technical": self.technical_conviction,
                "sentiment": self.sentiment_conviction,
                "ai": self.ai_conviction,
                "final": self.final_conviction,
            },
            "volatility_adjustment": {
                "vix": self.volatility_adjustment.vix_level,
                "regime": self.volatility_adjustment.regime,
                "size_mult": self.volatility_adjustment.position_size_multiplier,
            } if self.volatility_adjustment else None,
            "entry": self.entry_price,
            "target": self.target_price,
            "stop": self.stop_price,
            "position_size_pct": self.position_size_pct,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_context(
        cls,
        ticker: str,
        direction: str,
        context: Dict[str, Any],
    ) -> "TradingThesis":
        """
        Create a TradingThesis from context data.

        This factory method extracts relevant data from context
        and builds a structured thesis.
        """
        thesis = cls(
            ticker=ticker,
            direction=direction,
            thesis_summary=f"{direction.upper()} {ticker} based on multi-factor analysis",
            time_horizon=context.get("time_horizon", "0dte"),
        )

        # Extract pattern evidence
        pattern = context.get("pattern", "")
        pattern_conf = context.get("pattern_confidence", 50)
        if pattern:
            thesis.add_evidence(
                claim=f"Pattern detected: {pattern}",
                source=EvidenceSource.PRICE_ACTION,
                source_detail="Pattern Recognition",
                data_value=pattern,
                confidence=ConfidenceLevel.HIGH if pattern_conf > 70 else ConfidenceLevel.MEDIUM,
            )
            thesis.pattern_conviction = pattern_conf

        # Extract flow evidence
        flow_bias = context.get("flow_bias", "")
        if flow_bias and flow_bias not in ["neutral", "NEUTRAL"]:
            supports = (
                (direction == "long" and "bullish" in flow_bias.lower()) or
                (direction == "short" and "bearish" in flow_bias.lower())
            )
            thesis.add_evidence(
                claim=f"Flow bias: {flow_bias}",
                source=EvidenceSource.OPTIONS_FLOW,
                source_detail="Order Flow",
                data_value=flow_bias,
                confidence=ConfidenceLevel.MEDIUM,
                supports=supports,
            )
            thesis.flow_conviction = 70 if supports else 30

        # Extract technical evidence
        rsi = context.get("rsi")
        if rsi is not None:
            if rsi < 30:
                claim = f"RSI oversold ({rsi:.0f})"
                supports = direction == "long"
            elif rsi > 70:
                claim = f"RSI overbought ({rsi:.0f})"
                supports = direction == "short"
            else:
                claim = f"RSI neutral ({rsi:.0f})"
                supports = True

            thesis.add_evidence(
                claim=claim,
                source=EvidenceSource.TECHNICAL_INDICATOR,
                source_detail="RSI",
                data_value=rsi,
                confidence=ConfidenceLevel.MEDIUM,
                supports=supports,
            )

        # Extract HYDRA evidence
        hydra_direction = context.get("hydra_direction", "")
        if hydra_direction:
            supports = (
                (direction == "long" and hydra_direction == "BULLISH") or
                (direction == "short" and hydra_direction == "BEARISH")
            )
            thesis.add_evidence(
                claim=f"HYDRA direction: {hydra_direction}",
                source=EvidenceSource.HYDRA_INTEL,
                source_detail="HYDRA Predator",
                data_value=hydra_direction,
                confidence=ConfidenceLevel.HIGH,
                supports=supports,
            )

        # Extract GEX evidence
        gex_regime = context.get("gex_regime", "")
        if gex_regime:
            thesis.add_evidence(
                claim=f"GEX regime: {gex_regime}",
                source=EvidenceSource.GEX_ANALYSIS,
                source_detail="GEX Calculator",
                data_value=gex_regime,
                confidence=ConfidenceLevel.MEDIUM,
            )

        # Add standard risk factors
        vix = context.get("vix", context.get("vix_level", 20))
        if vix > 25:
            thesis.add_risk(
                description=f"Elevated VIX ({vix:.1f}) indicates high uncertainty",
                severity="high" if vix > 30 else "medium",
                probability=0.4,
                mitigation="Reduce position size",
                source=EvidenceSource.TECHNICAL_INDICATOR,
            )

        hours_to_expiry = context.get("hours_to_expiry", 6)
        if hours_to_expiry < 2:
            thesis.add_risk(
                description=f"Only {hours_to_expiry:.1f}h to expiry - theta acceleration",
                severity="high",
                probability=0.7,
                mitigation="Quick exit on first resistance",
                source=EvidenceSource.PRICE_ACTION,
            )

        # Apply volatility adjustment
        thesis.apply_volatility_adjustment(vix)

        # Calculate final conviction
        thesis.calculate_final_conviction()

        return thesis


# Factory functions
def create_thesis_from_setup(
    setup: Any,
    context: Dict[str, Any],
) -> TradingThesis:
    """
    Create a TradingThesis from a ScalpSetup or similar.

    Args:
        setup: Trade setup object
        context: Additional context

    Returns:
        Structured TradingThesis
    """
    ticker = context.get("ticker", getattr(setup, "ticker", "UNKNOWN"))
    direction = context.get("direction", getattr(setup, "direction", "long"))

    # Merge setup attributes into context
    merged_context = context.copy()
    merged_context["pattern"] = getattr(setup, "pattern", context.get("pattern", ""))
    merged_context["pattern_confidence"] = getattr(setup, "confidence", 50)
    merged_context["entry_price"] = getattr(setup, "entry_price", 0)

    thesis = TradingThesis.from_context(ticker, direction, merged_context)

    # Set entry price if available
    if thesis.entry_price == 0:
        thesis.entry_price = getattr(setup, "entry_price", context.get("price", 0))

    return thesis
