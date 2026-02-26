"""
GEX Calculator - Gamma Exposure Analytics for BERSERKER

Calculates dealer gamma exposure and zero gamma level (flip point)
for SPX/SPY options to identify high-probability directional setups.

Based on research from:
- perfiliev.com GEX calculation methodology
- Barchart.com GEX implementation
- MenthorQ gamma mechanics guide

Formula: GEX = Σ (gamma × 100 × spot² × 0.01 × OI × direction)
- Calls: +1 direction (dealers assumed long)
- Puts: -1 direction (dealers assumed short)
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
from scipy.stats import norm

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OptionData:
    """Single option contract data for GEX calculation."""

    strike: float
    expiry: date
    option_type: str  # "call" or "put"
    open_interest: int
    implied_volatility: float  # As decimal (0.25 = 25%)
    bid: float = 0.0
    ask: float = 0.0
    delta: Optional[float] = None
    gamma: Optional[float] = None


@dataclass
class GEXResult:
    """Result of GEX calculation."""

    total_gex: float  # Net GEX in billions
    call_gex: float  # Call contribution
    put_gex: float  # Put contribution (negative)
    gamma_flip: Optional[float]  # Price where GEX crosses zero
    flip_proximity: float  # Current distance from flip as percentage
    regime: str  # "positive_gamma" or "negative_gamma"
    support_levels: list = field(default_factory=list)  # High put gamma = support
    resistance_levels: list = field(default_factory=list)  # High call gamma = resistance
    calculated_at: datetime = field(default_factory=datetime.now)


@dataclass
class BerserkerSignal:
    """Signal for BERSERKER activation."""

    should_activate: bool
    direction: str  # "LONG" or "SHORT"
    flip_proximity: float
    gex_regime: str
    confidence: float
    reasoning: str


# ─────────────────────────────────────────────────────────────────────────────
# BLACK-SCHOLES GAMMA
# ─────────────────────────────────────────────────────────────────────────────


def black_scholes_gamma(
    spot: float,
    strike: float,
    time_to_expiry: float,  # In years
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes gamma for an option.

    Gamma is the same for calls and puts with same strike/expiry.

    Args:
        spot: Current underlying price
        strike: Option strike price
        time_to_expiry: Time to expiration in years
        risk_free_rate: Risk-free interest rate (decimal)
        volatility: Implied volatility (decimal)
        dividend_yield: Dividend yield (decimal)

    Returns:
        Gamma value (change in delta per $1 move in underlying)
    """
    if time_to_expiry <= 0:
        return 0.0

    if volatility <= 0:
        return 0.0

    try:
        sqrt_t = math.sqrt(time_to_expiry)
        d1 = (
            math.log(spot / strike)
            + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
        ) / (volatility * sqrt_t)

        # Gamma = phi(d1) / (S * sigma * sqrt(T))
        # where phi is the standard normal PDF
        gamma = norm.pdf(d1) / (spot * volatility * sqrt_t)

        return gamma

    except (ValueError, ZeroDivisionError) as e:
        log.warning(f"Gamma calculation error: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GEX CALCULATOR
# ─────────────────────────────────────────────────────────────────────────────


class GEXCalculator:
    """
    Gamma Exposure Calculator for options market analysis.

    Tracks dealer positioning to identify:
    1. Net gamma exposure (positive = dealers dampen moves, negative = amplify)
    2. Zero gamma level (flip point where dealer behavior changes)
    3. Support/resistance from high gamma strikes
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,  # Current ~5% Fed funds
        dividend_yield: float = 0.013,  # SPY ~1.3% yield
        contract_multiplier: int = 100,
    ):
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.contract_multiplier = contract_multiplier

        # Cache for recent calculations
        self._last_result: Optional[GEXResult] = None
        self._last_spot: Optional[float] = None

    def calculate_gex(
        self,
        spot_price: float,
        options_chain: list[OptionData],
        expirations_to_include: Optional[list[date]] = None,
    ) -> GEXResult:
        """
        Calculate total gamma exposure from options chain.

        Args:
            spot_price: Current underlying price
            options_chain: List of OptionData objects
            expirations_to_include: Filter to specific expiries (None = all)

        Returns:
            GEXResult with full analysis
        """
        today = date.today()
        call_gex = 0.0
        put_gex = 0.0
        strike_gex = {}  # Track GEX per strike for support/resistance

        for opt in options_chain:
            # Skip expired options
            if opt.expiry < today:
                continue

            # Filter by expiration if specified
            if expirations_to_include and opt.expiry not in expirations_to_include:
                continue

            # Calculate time to expiry in years
            dte = (opt.expiry - today).days
            time_to_expiry = dte / 365.0

            # Get gamma (calculate if not provided)
            if opt.gamma is not None:
                gamma = opt.gamma
            else:
                gamma = black_scholes_gamma(
                    spot=spot_price,
                    strike=opt.strike,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=self.risk_free_rate,
                    volatility=opt.implied_volatility,
                    dividend_yield=self.dividend_yield,
                )

            # GEX formula: gamma × 100 × spot² × 0.01 × OI
            # Result in dollars per 1% move
            contract_gex = (
                gamma
                * self.contract_multiplier
                * (spot_price**2)
                * 0.01
                * opt.open_interest
            )

            # Dealers assumed long calls, short puts
            if opt.option_type.lower() == "call":
                call_gex += contract_gex
                strike_gex.setdefault(opt.strike, {"call": 0, "put": 0})
                strike_gex[opt.strike]["call"] += contract_gex
            else:
                put_gex -= contract_gex  # Negative contribution
                strike_gex.setdefault(opt.strike, {"call": 0, "put": 0})
                strike_gex[opt.strike]["put"] += contract_gex

        total_gex = call_gex + put_gex

        # Convert to billions for readability
        total_gex_bn = total_gex / 1e9
        call_gex_bn = call_gex / 1e9
        put_gex_bn = put_gex / 1e9

        # Determine regime
        regime = "positive_gamma" if total_gex >= 0 else "negative_gamma"

        # Find gamma flip point
        gamma_flip = self._find_gamma_flip(spot_price, options_chain)

        # Calculate proximity to flip
        if gamma_flip:
            flip_proximity = abs(spot_price - gamma_flip) / spot_price
        else:
            flip_proximity = 1.0  # Far from flip

        # Find support/resistance levels (top 3 by absolute GEX)
        support_levels = []
        resistance_levels = []

        for strike, gex_data in sorted(strike_gex.items()):
            if gex_data["put"] > 0:  # High put gamma = support
                support_levels.append((strike, gex_data["put"]))
            if gex_data["call"] > 0:  # High call gamma = resistance
                resistance_levels.append((strike, gex_data["call"]))

        # Sort by GEX magnitude and take top 3
        support_levels = sorted(support_levels, key=lambda x: x[1], reverse=True)[:3]
        resistance_levels = sorted(
            resistance_levels, key=lambda x: x[1], reverse=True
        )[:3]

        result = GEXResult(
            total_gex=total_gex_bn,
            call_gex=call_gex_bn,
            put_gex=put_gex_bn,
            gamma_flip=gamma_flip,
            flip_proximity=flip_proximity,
            regime=regime,
            support_levels=[s[0] for s in support_levels],
            resistance_levels=[r[0] for r in resistance_levels],
        )

        # Cache result
        self._last_result = result
        self._last_spot = spot_price

        return result

    def _find_gamma_flip(
        self,
        current_spot: float,
        options_chain: list[OptionData],
        price_range_pct: float = 0.05,  # ±5% from current
        num_points: int = 50,
    ) -> Optional[float]:
        """
        Find the zero gamma level (flip point) where dealer GEX crosses zero.

        Args:
            current_spot: Current underlying price
            options_chain: Options chain data
            price_range_pct: Range to search as percentage of spot
            num_points: Number of price points to evaluate

        Returns:
            Gamma flip price level or None if not found in range
        """
        today = date.today()
        low_price = current_spot * (1 - price_range_pct)
        high_price = current_spot * (1 + price_range_pct)
        step = (high_price - low_price) / num_points

        gex_profile = []

        for i in range(num_points + 1):
            test_price = low_price + i * step
            total_gex = 0.0

            for opt in options_chain:
                if opt.expiry < today:
                    continue

                dte = (opt.expiry - today).days
                time_to_expiry = dte / 365.0

                gamma = black_scholes_gamma(
                    spot=test_price,
                    strike=opt.strike,
                    time_to_expiry=time_to_expiry,
                    risk_free_rate=self.risk_free_rate,
                    volatility=opt.implied_volatility,
                    dividend_yield=self.dividend_yield,
                )

                contract_gex = (
                    gamma
                    * self.contract_multiplier
                    * (test_price**2)
                    * 0.01
                    * opt.open_interest
                )

                if opt.option_type.lower() == "call":
                    total_gex += contract_gex
                else:
                    total_gex -= contract_gex

            gex_profile.append((test_price, total_gex))

        # Find zero crossing (sign change)
        for i in range(1, len(gex_profile)):
            p1, g1 = gex_profile[i - 1]
            p2, g2 = gex_profile[i]

            if g1 * g2 < 0:  # Sign change = zero crossing
                # Linear interpolation
                flip_point = p1 - g1 * (p2 - p1) / (g2 - g1)
                return flip_point

        return None

    def generate_berserker_signal(
        self,
        spot_price: float,
        options_chain: list[OptionData],
        hydra_direction: str,
        vix: float,
        flow_bias: Optional[str] = None,
    ) -> BerserkerSignal:
        """
        Generate BERSERKER activation signal based on GEX conditions.

        BERSERKER activates when:
        1. Price within 0.3% of gamma flip point
        2. HYDRA gives clear direction (BULLISH or BEARISH)
        3. VIX < 25 (not crisis mode)
        4. Optional: Flow bias confirms direction

        Args:
            spot_price: Current underlying price
            options_chain: Options chain data
            hydra_direction: HYDRA's directional bias
            vix: Current VIX level
            flow_bias: Optional flow direction confirmation

        Returns:
            BerserkerSignal with activation decision
        """
        # Calculate GEX
        gex_result = self.calculate_gex(spot_price, options_chain)

        # Check activation conditions
        conditions_met = []
        conditions_failed = []

        # Condition 1: Near gamma flip (within 0.3%)
        flip_threshold = 0.003
        near_flip = gex_result.flip_proximity < flip_threshold
        if near_flip:
            conditions_met.append(
                f"Near gamma flip ({gex_result.flip_proximity:.2%} < {flip_threshold:.1%})"
            )
        else:
            conditions_failed.append(
                f"Too far from flip ({gex_result.flip_proximity:.2%} > {flip_threshold:.1%})"
            )

        # Condition 2: Clear HYDRA direction
        clear_direction = hydra_direction in ["BULLISH", "BEARISH"]
        if clear_direction:
            conditions_met.append(f"HYDRA direction clear: {hydra_direction}")
        else:
            conditions_failed.append(f"HYDRA neutral: {hydra_direction}")

        # Condition 3: VIX manageable
        vix_ok = vix < 25
        if vix_ok:
            conditions_met.append(f"VIX acceptable: {vix:.1f}")
        else:
            conditions_failed.append(f"VIX too high: {vix:.1f}")

        # Condition 4 (optional): Flow confirms
        flow_confirms = True
        if flow_bias:
            flow_confirms = (
                (hydra_direction == "BULLISH" and flow_bias in ["bullish", "BULLISH"])
                or (
                    hydra_direction == "BEARISH" and flow_bias in ["bearish", "BEARISH"]
                )
                or flow_bias in ["neutral", "NEUTRAL"]
            )
            if flow_confirms:
                conditions_met.append(f"Flow confirms: {flow_bias}")
            else:
                conditions_failed.append(f"Flow contradicts: {flow_bias}")

        # Determine activation
        should_activate = near_flip and clear_direction and vix_ok and flow_confirms

        # Determine direction
        if hydra_direction == "BULLISH":
            direction = "LONG"
        elif hydra_direction == "BEARISH":
            direction = "SHORT"
        else:
            direction = "NONE"

        # Calculate confidence
        base_confidence = 0.0
        if should_activate:
            # Start at 60%, add bonuses
            base_confidence = 0.60

            # Closer to flip = higher confidence
            if gex_result.flip_proximity < 0.001:  # Within 0.1%
                base_confidence += 0.15
            elif gex_result.flip_proximity < 0.002:  # Within 0.2%
                base_confidence += 0.10

            # Lower VIX = higher confidence
            if vix < 15:
                base_confidence += 0.10
            elif vix < 20:
                base_confidence += 0.05

            # Flow confirmation bonus
            if flow_confirms and flow_bias:
                base_confidence += 0.05

        # Build reasoning
        reasoning_parts = []
        if conditions_met:
            reasoning_parts.append(f"✓ {', '.join(conditions_met)}")
        if conditions_failed:
            reasoning_parts.append(f"✗ {', '.join(conditions_failed)}")

        reasoning = " | ".join(reasoning_parts)

        return BerserkerSignal(
            should_activate=should_activate,
            direction=direction,
            flip_proximity=gex_result.flip_proximity,
            gex_regime=gex_result.regime,
            confidence=min(base_confidence, 0.95),
            reasoning=reasoning,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

_gex_calculator: Optional[GEXCalculator] = None


def get_gex_calculator() -> GEXCalculator:
    """Get singleton GEX calculator instance."""
    global _gex_calculator
    if _gex_calculator is None:
        _gex_calculator = GEXCalculator()
    return _gex_calculator


# ─────────────────────────────────────────────────────────────────────────────
# HYDRA INTEGRATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def parse_hydra_gex_data(hydra_response: dict) -> Optional[GEXResult]:
    """
    Parse GEX data from HYDRA /api/gex endpoint.

    Expected HYDRA response format:
    {
        "gex": {
            "total_gex_bn": -2.5,
            "call_gex_bn": 5.0,
            "put_gex_bn": -7.5,
            "gamma_flip": 5950.0,
            "regime": "negative_gamma"
        },
        "spot": 5920.0
    }
    """
    try:
        gex_data = hydra_response.get("gex", {})
        spot = hydra_response.get("spot", 0)

        if not gex_data or not spot:
            return None

        gamma_flip = gex_data.get("gamma_flip")
        if gamma_flip:
            flip_proximity = abs(spot - gamma_flip) / spot
        else:
            flip_proximity = 1.0

        return GEXResult(
            total_gex=gex_data.get("total_gex_bn", 0),
            call_gex=gex_data.get("call_gex_bn", 0),
            put_gex=gex_data.get("put_gex_bn", 0),
            gamma_flip=gamma_flip,
            flip_proximity=flip_proximity,
            regime=gex_data.get("regime", "unknown"),
            support_levels=gex_data.get("support_levels", []),
            resistance_levels=gex_data.get("resistance_levels", []),
        )

    except Exception as e:
        log.error(f"Error parsing HYDRA GEX data: {e}")
        return None


def check_berserker_conditions_from_hydra(
    hydra_predator_response: dict,
) -> BerserkerSignal:
    """
    Check BERSERKER activation using HYDRA /api/predator response.

    The predator endpoint aggregates GEX, flow, and direction data.
    """
    try:
        # Extract required fields
        gex_data = hydra_predator_response.get("gex", {})
        spot = hydra_predator_response.get("spot", 0)
        direction = hydra_predator_response.get("direction", "NEUTRAL")
        vix = hydra_predator_response.get("vix", 30)
        flow_bias = hydra_predator_response.get("flow", {}).get("bias", "neutral")

        # Get gamma flip and proximity
        gamma_flip = gex_data.get("gamma_flip")
        if gamma_flip and spot:
            flip_proximity = abs(spot - gamma_flip) / spot
        else:
            flip_proximity = 1.0

        # Check conditions
        near_flip = flip_proximity < 0.003  # Within 0.3%
        clear_direction = direction in ["BULLISH", "BEARISH"]
        vix_ok = vix < 25

        should_activate = near_flip and clear_direction and vix_ok

        # Determine trade direction
        trade_direction = "LONG" if direction == "BULLISH" else "SHORT"
        if direction == "NEUTRAL":
            trade_direction = "NONE"

        # Calculate confidence
        confidence = 0.0
        if should_activate:
            confidence = 0.60
            if flip_proximity < 0.001:
                confidence += 0.15
            if vix < 15:
                confidence += 0.10

        # Build reasoning
        reasons = []
        if near_flip:
            reasons.append(f"flip_proximity={flip_proximity:.3%}")
        if clear_direction:
            reasons.append(f"direction={direction}")
        if vix_ok:
            reasons.append(f"vix={vix:.1f}")

        return BerserkerSignal(
            should_activate=should_activate,
            direction=trade_direction,
            flip_proximity=flip_proximity,
            gex_regime=gex_data.get("regime", "unknown"),
            confidence=confidence,
            reasoning=" | ".join(reasons) if reasons else "Conditions not met",
        )

    except Exception as e:
        log.error(f"Error checking BERSERKER conditions: {e}")
        return BerserkerSignal(
            should_activate=False,
            direction="NONE",
            flip_proximity=1.0,
            gex_regime="error",
            confidence=0.0,
            reasoning=f"Error: {e}",
        )
