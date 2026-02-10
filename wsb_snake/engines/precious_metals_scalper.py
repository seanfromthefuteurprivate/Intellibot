"""
PRECIOUS METALS MOMENTUM SCALPER V2.0
=====================================
Technical Enhancement Module - January 29, 2026

Extends existing system with precious metals scalping capabilities.
Design Philosophy: ADDITIVE - plugs into existing architecture.

Strategies Implemented (from WSB analysis):
1. CATALYST_SCALP - Ultra-short-term trades on macro news
2. DIP_BUYER - Morning dip reversals (4-min SLV style trades)
3. MOMENTUM_SWING - Multi-day holds during strong trends
4. LEAPS_ACCUMULATOR - Long-term deep ITM accumulation

Target Tickers: GLD, SLV, IAU, PPLT, GOLD, SILVER
"""

import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pytz

from wsb_snake.utils.logger import get_logger
from wsb_snake.notifications.telegram_bot import send_alert
from wsb_snake.utils.rate_limit import get_limiter

log = get_logger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class MetalsStrategy(Enum):
    CATALYST_SCALP = "catalyst_scalp"
    DIP_BUYER = "dip_buyer"
    MOMENTUM_SWING = "momentum_swing"
    LEAPS_ACCUMULATOR = "leaps_accumulator"


class SignalDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class CatalystEvent:
    """Macro catalyst that drives precious metals moves."""
    source: str
    headline: str
    impact_score: float  # 1-10 scale
    direction: SignalDirection
    timestamp: datetime
    symbols_affected: List[str]
    keywords_matched: List[str]


@dataclass
class MomentumSignal:
    """Technical momentum signal for entry/exit."""
    symbol: str
    timeframe: str
    direction: SignalDirection
    score: float  # 0-100
    vwap_position: str  # ABOVE, BELOW, AT
    ema_alignment: bool
    rsi_value: float
    volume_ratio: float
    heikin_ashi_trend: str


@dataclass
class MetalsSetup:
    """Complete trading setup for precious metals."""
    strategy: MetalsStrategy
    symbol: str
    direction: SignalDirection
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    catalyst_score: float
    momentum_score: float
    option_recommendation: Optional[Dict] = None
    notes: str = ""


@dataclass
class GreekTargets:
    """Greek targets per strategy type."""
    delta_min: float
    delta_max: float
    gamma_preference: str  # HIGH, MODERATE, LOW
    theta_tolerance: str   # ACCEPT, MINIMIZE, VERY_LOW
    vega_preference: str   # HIGH, MODERATE, LOW


@dataclass
class PortfolioGreeks:
    """Aggregated portfolio Greeks."""
    net_delta: float = 0.0
    net_gamma: float = 0.0
    daily_theta: float = 0.0
    net_vega: float = 0.0


@dataclass
class HedgeRecommendation:
    """Hedge recommendation when Greek limits approached."""
    hedge_type: str  # DELTA_HEDGE, THETA_HEDGE, VEGA_HEDGE
    action: str      # SELL_CALLS, BUY_CALLS, ROLL_OUT, etc.
    size: float
    priority: str    # HIGH, MEDIUM, LOW
    reason: str


# ============================================================================
# CATALYST SCANNER MODULE
# ============================================================================

class CatalystScanner:
    """
    Monitors macro events that drive precious metals moves.
    Extends existing data feed infrastructure.
    """

    # Correlation feeds for precious metals
    CORRELATION_FEEDS = {
        'DXY': {  # Dollar Index (inverse correlation to gold)
            'threshold': 0.5,
            'direction': 'INVERSE',
            'weight': 0.25
        },
        'GC=F': {  # Gold Futures
            'threshold': 0.3,
            'direction': 'DIRECT',
            'weight': 0.30
        },
        'SI=F': {  # Silver Futures
            'threshold': 0.5,
            'direction': 'DIRECT',
            'weight': 0.20
        },
        '^TNX': {  # 10Y Treasury Yield (inverse)
            'threshold': 0.02,  # 2 basis points
            'direction': 'INVERSE',
            'weight': 0.15
        },
        '^VIX': {  # VIX (fear gauge)
            'threshold': 5.0,
            'direction': 'DIRECT',
            'weight': 0.10
        }
    }

    # Geopolitical news keywords
    NEWS_KEYWORDS = [
        'federal reserve', 'interest rate', 'inflation', 'cpi', 'ppi',
        'iran', 'russia', 'china', 'sanctions', 'tariff',
        'dollar', 'currency', 'gold reserve', 'bullion',
        'central bank', 'quantitative', 'stimulus', 'recession',
        'debt ceiling', 'treasury', 'fomc', 'powell', 'yellen'
    ]

    # High-impact event blackout periods
    BLACKOUT_EVENTS = ['FOMC', 'NFP', 'CPI', 'PPI', 'GDP']

    def __init__(self):
        self.recent_catalysts: List[CatalystEvent] = []
        self.limiter = get_limiter()
        log.info("Catalyst Scanner initialized for precious metals")

    def scan_for_catalysts(self, news_items: List[Dict]) -> List[CatalystEvent]:
        """Scan news for precious metals catalysts."""
        catalysts = []

        for item in news_items:
            headline = item.get('headline', '').lower()

            # Check for keyword matches
            matched_keywords = [kw for kw in self.NEWS_KEYWORDS if kw in headline]

            if matched_keywords:
                # Calculate impact score based on keywords and source
                impact = self._calculate_impact_score(item, matched_keywords)

                if impact >= 5:  # Minimum impact threshold
                    catalyst = CatalystEvent(
                        source=item.get('source', 'unknown'),
                        headline=item.get('headline', ''),
                        impact_score=impact,
                        direction=self._determine_direction(headline, matched_keywords),
                        timestamp=datetime.now(),
                        symbols_affected=self._get_affected_symbols(matched_keywords),
                        keywords_matched=matched_keywords
                    )
                    catalysts.append(catalyst)
                    self.recent_catalysts.append(catalyst)

        # Keep only last 50 catalysts
        self.recent_catalysts = self.recent_catalysts[-50:]

        return catalysts

    def _calculate_impact_score(self, item: Dict, keywords: List[str]) -> float:
        """Calculate catalyst impact score (1-10)."""
        score = len(keywords) * 2  # Base score from keyword count

        # Boost for high-impact sources
        source = item.get('source', '').lower()
        if any(s in source for s in ['reuters', 'bloomberg', 'wsj', 'ft']):
            score += 2

        # Boost for specific high-impact keywords
        high_impact = ['fomc', 'interest rate', 'inflation', 'fed']
        if any(hi in ' '.join(keywords) for hi in high_impact):
            score += 3

        return min(score, 10)

    def _determine_direction(self, headline: str, keywords: List[str]) -> SignalDirection:
        """Determine bullish/bearish impact on metals."""
        bullish_terms = ['dovish', 'cut', 'stimulus', 'uncertainty', 'crisis', 'war', 'inflation']
        bearish_terms = ['hawkish', 'hike', 'strong dollar', 'taper', 'rate increase']

        bullish_count = sum(1 for term in bullish_terms if term in headline)
        bearish_count = sum(1 for term in bearish_terms if term in headline)

        if bullish_count > bearish_count:
            return SignalDirection.BULLISH
        elif bearish_count > bullish_count:
            return SignalDirection.BEARISH
        return SignalDirection.NEUTRAL

    def _get_affected_symbols(self, keywords: List[str]) -> List[str]:
        """Determine which metals are most affected."""
        symbols = ['GLD', 'SLV']  # Default both

        if any('gold' in kw or 'bullion' in kw for kw in keywords):
            symbols = ['GLD', 'IAU', 'GOLD']
        if any('silver' in kw for kw in keywords):
            symbols = ['SLV', 'SILVER']
        if any('platinum' in kw or 'palladium' in kw for kw in keywords):
            symbols.append('PPLT')

        return symbols

    def calculate_aggregate_catalyst_score(self) -> float:
        """Calculate aggregate catalyst score from recent events."""
        if not self.recent_catalysts:
            return 0

        # Weight by recency (exponential decay)
        now = datetime.now()
        weighted_score = 0
        total_weight = 0

        for catalyst in self.recent_catalysts:
            age_hours = (now - catalyst.timestamp).total_seconds() / 3600
            recency_weight = max(0, 1 - (age_hours / 24))  # Decay over 24 hours
            weighted_score += catalyst.impact_score * recency_weight
            total_weight += recency_weight

        return (weighted_score / total_weight) * 10 if total_weight > 0 else 0


# ============================================================================
# MOMENTUM ANALYZER MODULE
# ============================================================================

class MomentumAnalyzer:
    """
    Technical analysis layer for precious metals entry/exit timing.
    Extends existing technical analysis infrastructure.
    """

    # Indicator configurations
    EMA_FAST = 9
    EMA_SLOW = 20
    EMA_TREND = 50
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    # Precious metals RSI adjustments (metals trend longer)
    RSI_TREND_ADJUST = {
        'uptrend': {'overbought': 80, 'oversold': 40},
        'downtrend': {'overbought': 60, 'oversold': 20}
    }

    def __init__(self):
        log.info("Momentum Analyzer initialized for precious metals")

    def analyze_momentum(self, symbol: str, bars: List[Dict]) -> MomentumSignal:
        """Generate comprehensive momentum signal."""
        if len(bars) < 50:
            return self._empty_signal(symbol)

        # Calculate indicators
        closes = [b.get('c', 0) for b in bars]
        highs = [b.get('h', 0) for b in bars]
        lows = [b.get('l', 0) for b in bars]
        volumes = [b.get('v', 0) for b in bars]

        current_price = closes[-1]

        # VWAP calculation
        vwap = self._calculate_vwap(bars[-50:])
        vwap_position = 'ABOVE' if current_price > vwap else 'BELOW'

        # EMA calculations
        ema9 = self._calculate_ema(closes, self.EMA_FAST)
        ema20 = self._calculate_ema(closes, self.EMA_SLOW)
        ema50 = self._calculate_ema(closes, self.EMA_TREND)
        ema_alignment = ema9 > ema20 > ema50  # Bullish alignment

        # RSI
        rsi = self._calculate_rsi(closes, self.RSI_PERIOD)

        # Volume analysis
        avg_volume = sum(volumes[-20:]) / 20 if volumes else 0
        current_volume = volumes[-1] if volumes else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        # Heikin Ashi trend
        ha_trend = self._analyze_heikin_ashi(bars[-10:])

        # Calculate overall momentum score
        score = self._calculate_momentum_score(
            vwap_position, ema_alignment, rsi, volume_ratio, ha_trend
        )

        # Determine direction
        if score >= 60 and vwap_position == 'ABOVE' and ema_alignment:
            direction = SignalDirection.BULLISH
        elif score <= 40 or (vwap_position == 'BELOW' and not ema_alignment):
            direction = SignalDirection.BEARISH
        else:
            direction = SignalDirection.NEUTRAL

        return MomentumSignal(
            symbol=symbol,
            timeframe='5m',
            direction=direction,
            score=score,
            vwap_position=vwap_position,
            ema_alignment=ema_alignment,
            rsi_value=rsi,
            volume_ratio=volume_ratio,
            heikin_ashi_trend=ha_trend
        )

    def _calculate_vwap(self, bars: List[Dict]) -> float:
        """Calculate Volume Weighted Average Price."""
        total_volume = 0
        total_vwap = 0

        for bar in bars:
            typical_price = (bar.get('h', 0) + bar.get('l', 0) + bar.get('c', 0)) / 3
            volume = bar.get('v', 0)
            total_vwap += typical_price * volume
            total_volume += volume

        return total_vwap / total_volume if total_volume > 0 else 0

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1] if prices else 0

        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA

        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))

        return ema

    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _analyze_heikin_ashi(self, bars: List[Dict]) -> str:
        """Analyze Heikin Ashi candles for trend clarity."""
        if len(bars) < 3:
            return 'neutral'

        bullish_count = 0
        for bar in bars[-3:]:
            ha_close = (bar.get('o', 0) + bar.get('h', 0) + bar.get('l', 0) + bar.get('c', 0)) / 4
            ha_open = (bar.get('o', 0) + bar.get('c', 0)) / 2
            if ha_close > ha_open:
                bullish_count += 1

        if bullish_count >= 2:
            return 'bullish'
        elif bullish_count <= 1:
            return 'bearish'
        return 'neutral'

    def _calculate_momentum_score(
        self, vwap_pos: str, ema_align: bool, rsi: float, vol_ratio: float, ha_trend: str
    ) -> float:
        """Calculate composite momentum score (0-100)."""
        score = 50  # Start neutral

        # VWAP position (+/- 15)
        if vwap_pos == 'ABOVE':
            score += 15
        else:
            score -= 15

        # EMA alignment (+/- 20)
        if ema_align:
            score += 20
        else:
            score -= 10

        # RSI contribution (+/- 15)
        if rsi > 50:
            score += min(15, (rsi - 50) / 2)
        else:
            score -= min(15, (50 - rsi) / 2)

        # Volume confirmation (+/- 10)
        if vol_ratio > 1.5:
            score += 10
        elif vol_ratio < 0.7:
            score -= 5

        # Heikin Ashi trend (+/- 10)
        if ha_trend == 'bullish':
            score += 10
        elif ha_trend == 'bearish':
            score -= 10

        return max(0, min(100, score))

    def _empty_signal(self, symbol: str) -> MomentumSignal:
        """Return empty signal when insufficient data."""
        return MomentumSignal(
            symbol=symbol,
            timeframe='5m',
            direction=SignalDirection.NEUTRAL,
            score=50,
            vwap_position='AT',
            ema_alignment=False,
            rsi_value=50,
            volume_ratio=1.0,
            heikin_ashi_trend='neutral'
        )


# ============================================================================
# OPTIONS SCREENER MODULE
# ============================================================================

class OptionsScreener:
    """
    Identifies optimal options contracts based on WSB patterns.
    Extends existing options infrastructure.
    """

    # Contract selection criteria by strategy
    SCREENING_CRITERIA = {
        'scalp': {  # 0-7 DTE
            'dte': {'min': 0, 'max': 7},
            'delta': {'min': 0.30, 'max': 0.50},
            'open_interest': {'min': 1000},
            'volume': {'min': 500},
            'bid_ask_spread': {'max_pct': 0.05},
            'iv_rank': {'max': 70}
        },
        'swing': {  # 7-45 DTE
            'dte': {'min': 7, 'max': 45},
            'delta': {'min': 0.40, 'max': 0.60},
            'open_interest': {'min': 500},
            'volume': {'min': 200},
            'bid_ask_spread': {'max_pct': 0.08},
            'iv_rank': {'max': 60}
        },
        'leaps': {  # 90+ DTE
            'dte': {'min': 90, 'max': 730},
            'delta': {'min': 0.60, 'max': 0.80},  # Deep ITM
            'open_interest': {'min': 100},
            'bid_ask_spread': {'max_pct': 0.10},
            'iv_rank': {'max': 50}
        }
    }

    def __init__(self):
        log.info("Options Screener initialized for precious metals")

    def recommend_contract(
        self, symbol: str, direction: SignalDirection, strategy_type: str, current_price: float
    ) -> Dict:
        """Recommend optimal options contract."""
        criteria = self.SCREENING_CRITERIA.get(strategy_type, self.SCREENING_CRITERIA['swing'])

        # Determine option type
        option_type = 'CALL' if direction == SignalDirection.BULLISH else 'PUT'

        # Calculate target strike based on delta preference
        mid_delta = (criteria['delta']['min'] + criteria['delta']['max']) / 2

        if option_type == 'CALL':
            # For calls, higher delta = lower strike (more ITM)
            strike_offset = (0.5 - mid_delta) * current_price * 0.1
        else:
            # For puts, higher delta = higher strike (more ITM)
            strike_offset = (mid_delta - 0.5) * current_price * 0.1

        target_strike = round(current_price + strike_offset, 0)

        # Calculate target expiration
        target_dte = (criteria['dte']['min'] + criteria['dte']['max']) // 2
        target_expiry = datetime.now() + timedelta(days=target_dte)

        return {
            'symbol': symbol,
            'option_type': option_type,
            'strike': target_strike,
            'dte_range': criteria['dte'],
            'target_expiry': target_expiry.strftime('%Y-%m-%d'),
            'delta_range': criteria['delta'],
            'min_open_interest': criteria['open_interest']['min'],
            'max_spread_pct': criteria['bid_ask_spread']['max_pct'],
            'strategy': strategy_type,
            'notes': f"{strategy_type.upper()} {option_type} near ${target_strike}"
        }


# ============================================================================
# GREEK OPTIMIZER MODULE
# ============================================================================

class GreekOptimizer:
    """
    Portfolio-level Greek management and hedging recommendations.
    Monitors aggregate exposure and suggests hedges when limits approached.
    """

    # Greek targets by strategy
    STRATEGY_GREEK_TARGETS = {
        MetalsStrategy.CATALYST_SCALP: GreekTargets(
            delta_min=0.35, delta_max=0.50,
            gamma_preference='HIGH',
            theta_tolerance='ACCEPT',
            vega_preference='MODERATE'
        ),
        MetalsStrategy.DIP_BUYER: GreekTargets(
            delta_min=0.40, delta_max=0.55,
            gamma_preference='HIGH',
            theta_tolerance='ACCEPT',
            vega_preference='LOW'
        ),
        MetalsStrategy.MOMENTUM_SWING: GreekTargets(
            delta_min=0.50, delta_max=0.65,
            gamma_preference='MODERATE',
            theta_tolerance='MINIMIZE',
            vega_preference='MODERATE'
        ),
        MetalsStrategy.LEAPS_ACCUMULATOR: GreekTargets(
            delta_min=0.70, delta_max=0.85,
            gamma_preference='LOW',
            theta_tolerance='VERY_LOW',
            vega_preference='HIGH'
        )
    }

    # Portfolio-level limits
    PORTFOLIO_LIMITS = {
        'net_delta': {'max': 500, 'rebalance_at': 400, 'warning': 350},
        'net_gamma': {'max': 100, 'warning': 75},
        'daily_theta': {'max': -500, 'warning': -350},
        'net_vega': {'max': 2000, 'warning': 1500}
    }

    def __init__(self):
        self.current_greeks = PortfolioGreeks()
        log.info("Greek Optimizer initialized with portfolio limits")

    def get_greek_targets(self, strategy: MetalsStrategy) -> GreekTargets:
        """Get Greek targets for a strategy."""
        return self.STRATEGY_GREEK_TARGETS.get(strategy)

    def update_portfolio_greeks(self, positions: List[Dict]) -> PortfolioGreeks:
        """
        Calculate aggregate portfolio Greeks from positions.

        Args:
            positions: List of position dicts with delta, gamma, theta, vega, quantity

        Returns:
            PortfolioGreeks with aggregated values
        """
        aggregate = PortfolioGreeks()

        for pos in positions:
            quantity = pos.get('quantity', 0)
            multiplier = 100  # Options multiplier

            aggregate.net_delta += pos.get('delta', 0) * quantity * multiplier
            aggregate.net_gamma += pos.get('gamma', 0) * quantity * multiplier
            aggregate.daily_theta += pos.get('theta', 0) * quantity * multiplier
            aggregate.net_vega += pos.get('vega', 0) * quantity * multiplier

        self.current_greeks = aggregate
        return aggregate

    def check_limits(self, greeks: PortfolioGreeks = None) -> Dict[str, Any]:
        """
        Check if portfolio Greeks are within limits.

        Returns:
            Dict with status, breaches, and warnings
        """
        if greeks is None:
            greeks = self.current_greeks

        result = {
            'status': 'OK',
            'breaches': [],
            'warnings': []
        }

        limits = self.PORTFOLIO_LIMITS

        # Check delta
        if abs(greeks.net_delta) > limits['net_delta']['max']:
            result['status'] = 'BREACH'
            result['breaches'].append({
                'greek': 'delta',
                'current': greeks.net_delta,
                'limit': limits['net_delta']['max']
            })
        elif abs(greeks.net_delta) > limits['net_delta']['warning']:
            result['warnings'].append({
                'greek': 'delta',
                'current': greeks.net_delta,
                'warning_level': limits['net_delta']['warning']
            })

        # Check gamma
        if abs(greeks.net_gamma) > limits['net_gamma']['max']:
            result['status'] = 'BREACH'
            result['breaches'].append({
                'greek': 'gamma',
                'current': greeks.net_gamma,
                'limit': limits['net_gamma']['max']
            })
        elif abs(greeks.net_gamma) > limits['net_gamma']['warning']:
            result['warnings'].append({
                'greek': 'gamma',
                'current': greeks.net_gamma,
                'warning_level': limits['net_gamma']['warning']
            })

        # Check theta (negative value, so comparison is different)
        if greeks.daily_theta < limits['daily_theta']['max']:
            result['status'] = 'BREACH'
            result['breaches'].append({
                'greek': 'theta',
                'current': greeks.daily_theta,
                'limit': limits['daily_theta']['max']
            })
        elif greeks.daily_theta < limits['daily_theta']['warning']:
            result['warnings'].append({
                'greek': 'theta',
                'current': greeks.daily_theta,
                'warning_level': limits['daily_theta']['warning']
            })

        # Check vega
        if abs(greeks.net_vega) > limits['net_vega']['max']:
            result['status'] = 'BREACH'
            result['breaches'].append({
                'greek': 'vega',
                'current': greeks.net_vega,
                'limit': limits['net_vega']['max']
            })
        elif abs(greeks.net_vega) > limits['net_vega']['warning']:
            result['warnings'].append({
                'greek': 'vega',
                'current': greeks.net_vega,
                'warning_level': limits['net_vega']['warning']
            })

        if result['breaches']:
            result['status'] = 'BREACH'
        elif result['warnings']:
            result['status'] = 'WARNING'

        return result

    def suggest_hedges(self, greeks: PortfolioGreeks = None) -> List[HedgeRecommendation]:
        """
        Suggest hedges when Greek limits are approached.

        Returns:
            List of HedgeRecommendation objects
        """
        if greeks is None:
            greeks = self.current_greeks

        hedges = []
        limits = self.PORTFOLIO_LIMITS

        # Delta hedge suggestion
        if abs(greeks.net_delta) > limits['net_delta']['warning']:
            hedge_size = abs(greeks.net_delta) * 0.3  # Hedge 30% of excess
            if greeks.net_delta > 0:
                action = 'SELL_CALLS'
                reason = f"Net delta {greeks.net_delta:.0f} exceeds warning ({limits['net_delta']['warning']})"
            else:
                action = 'BUY_CALLS'
                reason = f"Net delta {greeks.net_delta:.0f} below warning (-{limits['net_delta']['warning']})"

            hedges.append(HedgeRecommendation(
                hedge_type='DELTA_HEDGE',
                action=action,
                size=hedge_size,
                priority='HIGH' if abs(greeks.net_delta) > limits['net_delta']['rebalance_at'] else 'MEDIUM',
                reason=reason
            ))

        # Theta hedge suggestion
        if greeks.daily_theta < limits['daily_theta']['warning']:
            hedges.append(HedgeRecommendation(
                hedge_type='THETA_HEDGE',
                action='ROLL_OUT',
                size=0,  # Roll existing positions
                priority='HIGH',
                reason=f"Daily theta ${greeks.daily_theta:.0f} exceeds warning (${limits['daily_theta']['warning']})"
            ))

        # Vega hedge suggestion
        if abs(greeks.net_vega) > limits['net_vega']['warning']:
            hedges.append(HedgeRecommendation(
                hedge_type='VEGA_HEDGE',
                action='REDUCE_VEGA_EXPOSURE' if greeks.net_vega > 0 else 'ADD_LONG_VEGA',
                size=abs(greeks.net_vega) * 0.2,
                priority='MEDIUM',
                reason=f"Net vega {greeks.net_vega:.0f} exceeds warning ({limits['net_vega']['warning']})"
            ))

        return hedges

    def validate_new_position(self, position_greeks: Dict, strategy: MetalsStrategy) -> Dict[str, Any]:
        """
        Validate a new position against Greek targets and portfolio limits.

        Args:
            position_greeks: Dict with delta, gamma, theta, vega for new position
            strategy: The strategy type for Greek target validation

        Returns:
            Dict with approved status and any issues
        """
        targets = self.get_greek_targets(strategy)
        issues = []

        # Check delta within strategy targets
        delta = abs(position_greeks.get('delta', 0))
        if not (targets.delta_min <= delta <= targets.delta_max):
            issues.append(f"Delta {delta:.2f} outside target range [{targets.delta_min}-{targets.delta_max}]")

        # Check if adding position would breach portfolio limits
        projected = PortfolioGreeks(
            net_delta=self.current_greeks.net_delta + position_greeks.get('delta', 0) * position_greeks.get('quantity', 1) * 100,
            net_gamma=self.current_greeks.net_gamma + position_greeks.get('gamma', 0) * position_greeks.get('quantity', 1) * 100,
            daily_theta=self.current_greeks.daily_theta + position_greeks.get('theta', 0) * position_greeks.get('quantity', 1) * 100,
            net_vega=self.current_greeks.net_vega + position_greeks.get('vega', 0) * position_greeks.get('quantity', 1) * 100
        )

        limit_check = self.check_limits(projected)
        if limit_check['status'] == 'BREACH':
            for breach in limit_check['breaches']:
                issues.append(f"Would breach {breach['greek']} limit: {breach['current']:.0f} > {breach['limit']}")

        return {
            'approved': len(issues) == 0,
            'issues': issues,
            'projected_greeks': projected
        }


# ============================================================================
# POSITION SIZER MODULE
# ============================================================================

class PositionSizer:
    """
    Intelligent position sizing with multiple adjustment factors.
    Extends existing risk engine with volatility, conviction, heat, and timing adjustments.
    """

    # Base configuration
    CONFIG = {
        'base_risk_per_trade': 0.02,  # 2% base risk
        'max_risk_per_trade': 0.05,   # 5% max risk
        'min_contracts': 1,
        'max_contracts_scalp': 10,
        'max_contracts_swing': 20,
        'max_contracts_leaps': 5
    }

    # Strategy-specific sizing multipliers
    STRATEGY_MULTIPLIERS = {
        MetalsStrategy.CATALYST_SCALP: 0.8,    # Smaller due to speed
        MetalsStrategy.DIP_BUYER: 0.9,         # Moderate
        MetalsStrategy.MOMENTUM_SWING: 1.0,    # Standard
        MetalsStrategy.LEAPS_ACCUMULATOR: 1.2  # Larger for LEAPS
    }

    def __init__(self, account_value: float = 100000):
        self.account_value = account_value
        self.current_exposure = 0.0
        self.max_exposure = 0.40  # 40% max portfolio in metals
        log.info(f"Position Sizer initialized with account value ${account_value:,.0f}")

    def set_account_value(self, value: float):
        """Update account value for sizing calculations."""
        self.account_value = value

    def set_current_exposure(self, exposure: float):
        """Update current portfolio exposure."""
        self.current_exposure = exposure

    def calculate_size(
        self,
        signal: Dict,
        strategy: MetalsStrategy,
        contract_price: float,
        stop_loss_pct: float
    ) -> int:
        """
        Calculate optimal position size with all adjustments.

        Args:
            signal: Signal dict with symbol, catalystScore, momentumScore
            strategy: Strategy type
            contract_price: Option contract price
            stop_loss_pct: Stop loss percentage (e.g., 0.20 for 20%)

        Returns:
            Number of contracts to trade
        """
        # Base risk calculation
        base_risk = self.CONFIG['base_risk_per_trade']

        # Apply all multipliers
        adjustments = {
            'volatility': self._get_volatility_multiplier(signal.get('symbol', '')),
            'conviction': self._get_conviction_multiplier(signal),
            'heat': self._get_heat_multiplier(),
            'timing': self._get_timing_multiplier(),
            'strategy': self.STRATEGY_MULTIPLIERS.get(strategy, 1.0)
        }

        # Calculate adjusted risk
        adjusted_risk = base_risk
        for name, mult in adjustments.items():
            adjusted_risk *= mult

        # Cap at max risk
        adjusted_risk = min(adjusted_risk, self.CONFIG['max_risk_per_trade'])

        # Calculate dollar risk
        risk_dollars = self.account_value * adjusted_risk

        # Calculate max loss per contract
        max_loss_per_contract = contract_price * stop_loss_pct * 100  # 100 shares per contract

        if max_loss_per_contract <= 0:
            log.warning("Invalid max loss calculation, defaulting to 1 contract")
            return 1

        # Calculate contracts
        contracts = int(risk_dollars / max_loss_per_contract)

        # Apply strategy-specific limits
        max_contracts = self._get_max_contracts(strategy)
        contracts = max(self.CONFIG['min_contracts'], min(contracts, max_contracts))

        log.info(
            f"Position size: {contracts} contracts | "
            f"Risk: ${risk_dollars:.0f} | "
            f"Adjustments: vol={adjustments['volatility']:.2f}, "
            f"conv={adjustments['conviction']:.2f}, "
            f"heat={adjustments['heat']:.2f}, "
            f"time={adjustments['timing']:.2f}"
        )

        return contracts

    def _get_volatility_multiplier(self, symbol: str) -> float:
        """
        Adjust size based on current VIX level.
        Higher VIX = smaller position (more volatile market).
        """
        try:
            from wsb_snake.collectors.vix_structure import vix_structure
            vix_data = vix_structure.get_trading_signal()
            vix = vix_data.get("vix", 20.0)

            # VIX-based volatility scaling (same logic as alpaca_executor)
            if vix < 15:
                return 1.2  # Low vol - can size up slightly
            elif vix < 20:
                return 1.0  # Normal vol
            elif vix < 25:
                return 0.8  # Elevated vol - reduce size 20%
            elif vix < 35:
                return 0.6  # High vol - reduce size 40%
            else:
                return 0.5  # Crisis vol - reduce size 50%
        except Exception as e:
            log.debug(f"VIX fetch failed for volatility multiplier: {e}")
            return 1.0  # Default to neutral if VIX unavailable

    def _estimate_greeks(self, setup: Any, greek_targets: Any, contract_price: float) -> Dict:
        """
        Estimate option Greeks based on VIX and option characteristics.

        More accurate than hardcoded placeholders - scales with market conditions.
        In production, these would come from the options chain API.
        """
        try:
            from wsb_snake.collectors.vix_structure import vix_structure
            vix_data = vix_structure.get_trading_signal()
            vix = vix_data.get("vix", 20.0)
        except Exception:
            vix = 20.0  # Default VIX assumption

        # Delta from targets (this is already calculated correctly)
        delta = (greek_targets.delta_min + greek_targets.delta_max) / 2

        # Gamma scales with volatility - higher VIX = higher gamma (more convexity)
        # Base gamma ~0.03-0.08 for ATM options, scales with VIX
        base_gamma = 0.05
        gamma = base_gamma * (vix / 20.0)  # Scale by VIX ratio to normal

        # Theta (time decay) - always negative for long options
        # Higher VIX = higher premium = more theta decay
        # Base theta ~-0.05 to -0.15 per day for short-dated options
        base_theta = -0.08
        theta = base_theta * (vix / 20.0) * (1 + (contract_price / 5.0) * 0.1)

        # Vega (sensitivity to IV) - higher for ATM, lower for deep ITM/OTM
        # Base vega ~0.10-0.20 for ATM options
        base_vega = 0.12
        # ATM options have highest vega (delta ~0.5)
        atm_factor = 1.0 - abs(abs(delta) - 0.5) * 0.5
        vega = base_vega * atm_factor * (vix / 20.0)

        return {
            'delta': delta,
            'gamma': round(gamma, 4),
            'theta': round(theta, 4),
            'vega': round(vega, 4),
            'quantity': 1  # Will be updated after sizing
        }

    def _get_conviction_multiplier(self, signal: Dict) -> float:
        """
        Adjust size based on signal conviction (catalyst + momentum scores).
        Higher conviction = larger position.
        """
        catalyst_score = signal.get('catalyst_score', 50)
        momentum_score = signal.get('momentum_score', 50)

        # High conviction - both scores strong
        if catalyst_score > 85 and momentum_score > 80:
            return 1.4
        elif catalyst_score > 70 and momentum_score > 65:
            return 1.2
        # Low conviction - reduce size
        elif catalyst_score < 60 or momentum_score < 50:
            return 0.7
        return 1.0

    def _get_heat_multiplier(self) -> float:
        """
        Adjust size based on current portfolio heat (exposure level).
        Higher exposure = smaller new positions.
        """
        exposure_ratio = self.current_exposure / self.max_exposure

        if exposure_ratio > 0.8:
            return 0.5  # Near max exposure - reduce size 50%
        elif exposure_ratio > 0.6:
            return 0.75  # Elevated exposure - reduce size 25%
        return 1.0

    def _get_timing_multiplier(self) -> float:
        """
        Adjust size based on market timing.
        Volatile periods (open/close) = smaller positions.
        """
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

        # Minutes since market open
        if now_et < market_open:
            return 0.7  # Pre-market

        market_minutes = (now_et - market_open).total_seconds() / 60

        if market_minutes < 15:
            return 0.7  # First 15 minutes - volatile open
        elif market_minutes > 375:  # Last 15 minutes before 4pm
            return 0.8  # Volatile close
        elif 15 <= market_minutes <= 90:
            return 1.1  # Morning momentum window - slightly larger
        return 1.0

    def _get_max_contracts(self, strategy: MetalsStrategy) -> int:
        """Get maximum contracts allowed for a strategy."""
        if strategy == MetalsStrategy.CATALYST_SCALP:
            return self.CONFIG['max_contracts_scalp']
        elif strategy == MetalsStrategy.DIP_BUYER:
            return self.CONFIG['max_contracts_scalp']
        elif strategy == MetalsStrategy.MOMENTUM_SWING:
            return self.CONFIG['max_contracts_swing']
        elif strategy == MetalsStrategy.LEAPS_ACCUMULATOR:
            return self.CONFIG['max_contracts_leaps']
        return 10

    def get_sizing_summary(self, signal: Dict, strategy: MetalsStrategy) -> Dict:
        """Get a summary of sizing factors without calculating final size."""
        return {
            'base_risk': self.CONFIG['base_risk_per_trade'],
            'account_value': self.account_value,
            'adjustments': {
                'volatility': self._get_volatility_multiplier(signal.get('symbol', '')),
                'conviction': self._get_conviction_multiplier(signal),
                'heat': self._get_heat_multiplier(),
                'timing': self._get_timing_multiplier(),
                'strategy': self.STRATEGY_MULTIPLIERS.get(strategy, 1.0)
            },
            'max_contracts': self._get_max_contracts(strategy),
            'current_exposure': self.current_exposure,
            'max_exposure': self.max_exposure
        }


# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================

class CatalystScalpStrategy:
    """
    Ultra-short-term trades on macro news events.
    Target: 30% gain, 20% stop, 2 DTE max hold.
    """

    CONFIG = {
        'name': 'CATALYST_SCALP',
        'entry': {
            'catalyst_score_min': 70,
            'momentum_alignment': True,
            'volume_surge': 2.0,
            'vwap_position': 'ABOVE'
        },
        'exit': {
            'profit_target': 0.30,
            'stop_loss': 0.20,
            'time_stop_dte': 2,
            'trailing_stop': {'activate': 0.15, 'trail': 0.08}
        },
        'limits': {
            'max_concurrent': 2,
            'max_daily_trades': 5,
            'no_overnight': True
        }
    }

    def evaluate(self, catalyst_score: float, momentum: MomentumSignal) -> Optional[MetalsSetup]:
        """Evaluate if conditions met for catalyst scalp."""
        if catalyst_score < self.CONFIG['entry']['catalyst_score_min']:
            return None

        if momentum.score < 60:
            return None

        if momentum.volume_ratio < self.CONFIG['entry']['volume_surge']:
            return None

        if momentum.vwap_position != self.CONFIG['entry']['vwap_position']:
            return None

        # All conditions met
        return MetalsSetup(
            strategy=MetalsStrategy.CATALYST_SCALP,
            symbol=momentum.symbol,
            direction=momentum.direction,
            entry_price=0,  # Will be filled by executor
            target_price=0,
            stop_loss=0,
            confidence=min(95, catalyst_score + momentum.score / 2),
            catalyst_score=catalyst_score,
            momentum_score=momentum.score,
            notes=f"Catalyst scalp: {momentum.symbol} - Vol {momentum.volume_ratio:.1f}x"
        )


class DipBuyerStrategy:
    """
    Capitalizes on morning dips (like 4-minute SLV trade).
    Target: 25% gain, 15% stop, 30 min max hold.
    """

    CONFIG = {
        'name': 'DIP_BUYER',
        'dip_detection': {
            'min_dip_pct': 3.0,
            'max_dip_pct': 10.0,
            'time_window_minutes': 30,
            'volume_confirmation': 1.5
        },
        'reversal_signals': {
            'vwap_reclaim': True,
            'green_candle': True,
            'volume_shift': True
        },
        'exit': {
            'profit_target': 0.25,
            'stop_loss': 0.15,
            'time_stop_minutes': 30,
            'scale_out': [
                {'target': 0.15, 'size': 0.5},
                {'target': 0.25, 'size': 0.5}
            ]
        },
        'limits': {
            'max_concurrent': 1,
            'max_daily_trades': 3,
            'only_morning': True,  # 9:30-11:00 ET
            'symbols': ['SLV', 'GLD', 'IAU']
        }
    }

    def detect_dip(self, symbol: str, bars: List[Dict]) -> Optional[float]:
        """Detect if a qualifying dip occurred."""
        if len(bars) < 10:
            return None

        # Calculate dip from recent high
        recent_high = max(b.get('h', 0) for b in bars[-30:])
        current_low = bars[-1].get('l', 0)

        if recent_high <= 0:
            return None

        dip_pct = ((recent_high - current_low) / recent_high) * 100

        cfg = self.CONFIG['dip_detection']
        if cfg['min_dip_pct'] <= dip_pct <= cfg['max_dip_pct']:
            return dip_pct

        return None

    def evaluate(self, symbol: str, dip_pct: float, momentum: MomentumSignal) -> Optional[MetalsSetup]:
        """Evaluate if dip buy conditions met."""
        if symbol not in self.CONFIG['limits']['symbols']:
            return None

        # Check if within morning window
        et = pytz.timezone('US/Eastern')
        now_et = datetime.now(et)
        if self.CONFIG['limits']['only_morning']:
            if not (9 <= now_et.hour < 11 or (now_et.hour == 11 and now_et.minute == 0)):
                return None

        # Need reversal confirmation
        if momentum.vwap_position != 'ABOVE':  # Must reclaim VWAP
            return None

        if momentum.volume_ratio < self.CONFIG['dip_detection']['volume_confirmation']:
            return None

        return MetalsSetup(
            strategy=MetalsStrategy.DIP_BUYER,
            symbol=symbol,
            direction=SignalDirection.BULLISH,
            entry_price=0,
            target_price=0,
            stop_loss=0,
            confidence=70 + min(20, dip_pct * 3),
            catalyst_score=0,
            momentum_score=momentum.score,
            notes=f"Dip buy: {symbol} down {dip_pct:.1f}% - reversal detected"
        )


class MomentumSwingStrategy:
    """
    Multi-day holds during strong trends.
    Target: 100% gain, trailing stop, 45 DTE max.
    """

    CONFIG = {
        'name': 'MOMENTUM_SWING',
        'trend_criteria': {
            'ema_alignment': True,
            'adx_min': 25,
            'rsi_range': [40, 70],
            'volume_trend': 'INCREASING'
        },
        'entry': {
            'pullback_to_ema': 20,
            'rsi_pullback': [40, 50],
            'support_hold': True
        },
        'exit': {
            'profit_target': 1.00,
            'trailing_stop': {'activate': 0.30, 'trail': 0.15},
            'time_stop_dte': 45,
            'trend_break_exit': True
        },
        'limits': {
            'max_concurrent': 3,
            'max_daily_trades': 2,
            'overnight_allowed': True,
            'symbols': ['GLD', 'SLV', 'PPLT', 'IAU']
        }
    }

    def evaluate(self, symbol: str, momentum: MomentumSignal) -> Optional[MetalsSetup]:
        """Evaluate if swing conditions met."""
        if symbol not in self.CONFIG['limits']['symbols']:
            return None

        # Need strong trend
        if not momentum.ema_alignment:
            return None

        # RSI in trend range
        rsi_range = self.CONFIG['trend_criteria']['rsi_range']
        if not (rsi_range[0] <= momentum.rsi_value <= rsi_range[1]):
            return None

        # Need high momentum score
        if momentum.score < 70:
            return None

        return MetalsSetup(
            strategy=MetalsStrategy.MOMENTUM_SWING,
            symbol=symbol,
            direction=momentum.direction,
            entry_price=0,
            target_price=0,
            stop_loss=0,
            confidence=momentum.score,
            catalyst_score=0,
            momentum_score=momentum.score,
            notes=f"Swing: {symbol} - EMA aligned, RSI {momentum.rsi_value:.0f}"
        )


class LeapsAccumulatorStrategy:
    """
    Long-term deep ITM accumulation (like GLD $420C Jan '27).
    Target: Accumulate on dips, roll at 60 DTE.
    """

    CONFIG = {
        'name': 'LEAPS_ACCUMULATOR',
        'selection': {
            'min_dte': 180,
            'max_dte': 730,
            'delta_range': [0.70, 0.85],
            'prefer_quarterlys': True
        },
        'accumulation': {
            'method': 'DCA',
            'frequency': 'WEEKLY',
            'pullback_boost': True,
            'max_contracts': 20
        },
        'entry': {
            'iv_rank_max': 40,
            'technical_confirm': True,
            'macro_aligned': True
        },
        'management': {
            'roll_at_dte': 60,
            'roll_up_at_profit': 0.50,
            'partial_profit_take': {'at': 1.00, 'size': 0.25}
        },
        'limits': {
            'max_allocation_pct': 20,
            'symbols': ['GLD', 'SLV']
        }
    }

    def evaluate(self, symbol: str, momentum: MomentumSignal, iv_rank: float = 30) -> Optional[MetalsSetup]:
        """Evaluate if LEAPS accumulation conditions met."""
        if symbol not in self.CONFIG['limits']['symbols']:
            return None

        # Only accumulate when IV is cheap
        if iv_rank > self.CONFIG['entry']['iv_rank_max']:
            return None

        # Prefer bullish technical backdrop
        if momentum.score < 55:
            return None

        return MetalsSetup(
            strategy=MetalsStrategy.LEAPS_ACCUMULATOR,
            symbol=symbol,
            direction=SignalDirection.BULLISH,
            entry_price=0,
            target_price=0,
            stop_loss=0,
            confidence=60 + (40 - iv_rank),  # Higher confidence when IV is lower
            catalyst_score=0,
            momentum_score=momentum.score,
            notes=f"LEAPS: {symbol} - IV rank {iv_rank:.0f}, accumulate deep ITM calls"
        )


# ============================================================================
# MAIN PRECIOUS METALS SCALPER ENGINE
# ============================================================================

class PreciousMetalsScalper:
    """
    Main engine for precious metals momentum scalping.
    Integrates catalyst scanner, momentum analyzer, Greek optimizer,
    position sizer, and all strategies.
    """

    # Target tickers
    METALS_TICKERS = ['GLD', 'SLV', 'IAU', 'PPLT']

    # Minimum confidence to generate alert
    MIN_CONFIDENCE_FOR_ALERT = 75

    # Enhanced risk configuration from spec
    RISK_CONFIG = {
        'max_risk_per_trade': 0.02,
        'max_risk_per_day': 0.06,
        'metals_sector_max': 0.40,
        'single_symbol_max': 0.20,
        'max_concurrent_scalps': 3,
        'max_concurrent_swings': 5,
        'max_total_positions': 10,
        'daily_loss_limit': 0.05,
        'weekly_loss_limit': 0.10,
        'monthly_loss_limit': 0.15
    }

    def __init__(self, account_value: float = 100000):
        self.catalyst_scanner = CatalystScanner()
        self.momentum_analyzer = MomentumAnalyzer()
        self.options_screener = OptionsScreener()

        # New modules from spec
        self.greek_optimizer = GreekOptimizer()
        self.position_sizer = PositionSizer(account_value)

        # Strategy instances
        self.catalyst_scalp = CatalystScalpStrategy()
        self.dip_buyer = DipBuyerStrategy()
        self.momentum_swing = MomentumSwingStrategy()
        self.leaps_accumulator = LeapsAccumulatorStrategy()

        # Position tracking
        self.active_positions: List[Dict] = []
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.monthly_pnl = 0.0

        self.running = False
        self.last_scan = None

        log.info("Precious Metals Scalper V2.0 initialized")
        log.info(f"Monitoring: {', '.join(self.METALS_TICKERS)}")
        log.info(f"Account value: ${account_value:,.0f}")

    def scan_for_setups(self, market_data: Dict[str, List[Dict]], news_items: List[Dict] = None) -> List[MetalsSetup]:
        """
        Scan all metals for trading setups.

        Args:
            market_data: Dict of symbol -> list of bars
            news_items: Optional list of news items for catalyst scanning

        Returns:
            List of MetalsSetup objects for qualifying setups
        """
        setups = []

        # Scan for catalysts
        catalyst_score = 0
        if news_items:
            catalysts = self.catalyst_scanner.scan_for_catalysts(news_items)
            catalyst_score = self.catalyst_scanner.calculate_aggregate_catalyst_score()

        # Analyze each metal
        for symbol in self.METALS_TICKERS:
            bars = market_data.get(symbol, [])
            if not bars:
                continue

            # Get momentum signal
            momentum = self.momentum_analyzer.analyze_momentum(symbol, bars)
            current_price = bars[-1].get('c', 0) if bars else 0

            # Try each strategy

            # 1. Catalyst Scalp
            if catalyst_score > 50:
                setup = self.catalyst_scalp.evaluate(catalyst_score, momentum)
                if setup and setup.confidence >= self.MIN_CONFIDENCE_FOR_ALERT:
                    setup.option_recommendation = self.options_screener.recommend_contract(
                        symbol, setup.direction, 'scalp', current_price
                    )
                    setups.append(setup)

            # 2. Dip Buyer
            dip_pct = self.dip_buyer.detect_dip(symbol, bars)
            if dip_pct:
                setup = self.dip_buyer.evaluate(symbol, dip_pct, momentum)
                if setup and setup.confidence >= self.MIN_CONFIDENCE_FOR_ALERT:
                    setup.option_recommendation = self.options_screener.recommend_contract(
                        symbol, setup.direction, 'scalp', current_price
                    )
                    setups.append(setup)

            # 3. Momentum Swing
            setup = self.momentum_swing.evaluate(symbol, momentum)
            if setup and setup.confidence >= self.MIN_CONFIDENCE_FOR_ALERT:
                setup.option_recommendation = self.options_screener.recommend_contract(
                    symbol, setup.direction, 'swing', current_price
                )
                setups.append(setup)

            # 4. LEAPS Accumulator (check weekly)
            setup = self.leaps_accumulator.evaluate(symbol, momentum)
            if setup and setup.confidence >= self.MIN_CONFIDENCE_FOR_ALERT:
                setup.option_recommendation = self.options_screener.recommend_contract(
                    symbol, setup.direction, 'leaps', current_price
                )
                setups.append(setup)

        # Sort by confidence
        setups.sort(key=lambda s: s.confidence, reverse=True)

        self.last_scan = datetime.now()

        return setups

    def validate_and_size_setup(self, setup: MetalsSetup, contract_price: float) -> Dict[str, Any]:
        """
        Validate setup against Greek limits and calculate position size.

        Args:
            setup: The MetalsSetup to validate
            contract_price: Current contract price

        Returns:
            Dict with validation status, position size, and any warnings
        """
        result = {
            'approved': True,
            'position_size': 0,
            'warnings': [],
            'greek_check': None,
            'sizing_details': None
        }

        # Get Greek targets for the strategy
        greek_targets = self.greek_optimizer.get_greek_targets(setup.strategy)

        # Estimate position Greeks based on VIX and option characteristics
        # In production, these would come from the options chain API
        estimated_greeks = self._estimate_greeks(
            setup=setup,
            greek_targets=greek_targets,
            contract_price=contract_price
        )

        # Check current portfolio Greeks
        greek_check = self.greek_optimizer.check_limits()
        result['greek_check'] = greek_check

        if greek_check['status'] == 'BREACH':
            result['approved'] = False
            result['warnings'].append("Portfolio Greek limits breached - no new positions")
            return result

        if greek_check['status'] == 'WARNING':
            result['warnings'].extend([w['greek'] + ' approaching limit' for w in greek_check['warnings']])

        # Calculate position size
        signal = {
            'symbol': setup.symbol,
            'catalyst_score': setup.catalyst_score,
            'momentum_score': setup.momentum_score
        }

        # Get stop loss from strategy config
        stop_loss_pct = self._get_stop_loss_for_strategy(setup.strategy)

        position_size = self.position_sizer.calculate_size(
            signal=signal,
            strategy=setup.strategy,
            contract_price=contract_price,
            stop_loss_pct=stop_loss_pct
        )

        result['position_size'] = position_size
        result['sizing_details'] = self.position_sizer.get_sizing_summary(signal, setup.strategy)

        # Validate the sized position against Greek limits
        estimated_greeks['quantity'] = position_size
        greek_validation = self.greek_optimizer.validate_new_position(
            estimated_greeks, setup.strategy
        )

        if not greek_validation['approved']:
            result['approved'] = False
            result['warnings'].extend(greek_validation['issues'])

        return result

    def _get_stop_loss_for_strategy(self, strategy: MetalsStrategy) -> float:
        """Get stop loss percentage for a strategy."""
        stop_losses = {
            MetalsStrategy.CATALYST_SCALP: 0.20,
            MetalsStrategy.DIP_BUYER: 0.15,
            MetalsStrategy.MOMENTUM_SWING: 0.15,  # Uses trailing stop
            MetalsStrategy.LEAPS_ACCUMULATOR: 0.25
        }
        return stop_losses.get(strategy, 0.20)

    def update_positions(self, positions: List[Dict]):
        """
        Update active positions and recalculate portfolio Greeks.

        Args:
            positions: List of position dicts with greeks
        """
        self.active_positions = positions
        self.greek_optimizer.update_portfolio_greeks(positions)

        # Update position sizer with current exposure
        total_value = sum(p.get('market_value', 0) for p in positions)
        exposure_ratio = total_value / self.position_sizer.account_value if self.position_sizer.account_value > 0 else 0
        self.position_sizer.set_current_exposure(exposure_ratio)

    def get_hedge_recommendations(self) -> List[HedgeRecommendation]:
        """Get hedge recommendations based on current portfolio Greeks."""
        return self.greek_optimizer.suggest_hedges()

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status."""
        greek_check = self.greek_optimizer.check_limits()
        hedges = self.greek_optimizer.suggest_hedges()

        return {
            'greeks': {
                'net_delta': self.greek_optimizer.current_greeks.net_delta,
                'net_gamma': self.greek_optimizer.current_greeks.net_gamma,
                'daily_theta': self.greek_optimizer.current_greeks.daily_theta,
                'net_vega': self.greek_optimizer.current_greeks.net_vega
            },
            'greek_status': greek_check['status'],
            'greek_warnings': greek_check['warnings'],
            'greek_breaches': greek_check['breaches'],
            'hedge_recommendations': [
                {
                    'type': h.hedge_type,
                    'action': h.action,
                    'priority': h.priority,
                    'reason': h.reason
                } for h in hedges
            ],
            'active_positions': len(self.active_positions),
            'risk_config': self.RISK_CONFIG,
            'daily_pnl': self.daily_pnl,
            'last_scan': self.last_scan.isoformat() if self.last_scan else None
        }

    def check_risk_limits(self) -> Dict[str, Any]:
        """Check if any risk limits are breached."""
        result = {
            'can_trade': True,
            'breaches': [],
            'warnings': []
        }

        # Check daily loss limit
        if self.daily_pnl < -self.RISK_CONFIG['daily_loss_limit'] * self.position_sizer.account_value:
            result['can_trade'] = False
            result['breaches'].append('Daily loss limit reached')

        # Check position counts
        scalp_count = sum(1 for p in self.active_positions if p.get('strategy') in ['CATALYST_SCALP', 'DIP_BUYER'])
        swing_count = sum(1 for p in self.active_positions if p.get('strategy') in ['MOMENTUM_SWING', 'LEAPS_ACCUMULATOR'])

        if scalp_count >= self.RISK_CONFIG['max_concurrent_scalps']:
            result['warnings'].append(f'Max concurrent scalps ({self.RISK_CONFIG["max_concurrent_scalps"]}) reached')

        if swing_count >= self.RISK_CONFIG['max_concurrent_swings']:
            result['warnings'].append(f'Max concurrent swings ({self.RISK_CONFIG["max_concurrent_swings"]}) reached')

        if len(self.active_positions) >= self.RISK_CONFIG['max_total_positions']:
            result['can_trade'] = False
            result['breaches'].append('Max total positions reached')

        return result

    def format_alert(self, setup: MetalsSetup, validation: Dict = None) -> str:
        """Format setup as Telegram alert with Greek and sizing info."""
        direction_emoji = "🟢" if setup.direction == SignalDirection.BULLISH else "🔴"
        strategy_name = setup.strategy.value.upper().replace('_', ' ')

        msg = f"""{direction_emoji} **PRECIOUS METALS ALERT**

**{strategy_name}** - {setup.symbol}
Direction: {setup.direction.value.upper()}
Confidence: {setup.confidence:.0f}%

📊 **Analysis:**
- Catalyst Score: {setup.catalyst_score:.0f}
- Momentum Score: {setup.momentum_score:.0f}

"""

        if setup.option_recommendation:
            opt = setup.option_recommendation
            msg += f"""📋 **Recommended Contract:**
{opt['option_type']} ${opt['strike']:.0f}
Expiry: ~{opt['target_expiry']}
Delta: {opt['delta_range']['min']:.2f}-{opt['delta_range']['max']:.2f}
Min OI: {opt['min_open_interest']}

"""

        # Add sizing and Greek info if validation provided
        if validation:
            if validation.get('position_size'):
                msg += f"""📐 **Position Sizing:**
Contracts: {validation['position_size']}
"""
                if validation.get('sizing_details'):
                    adj = validation['sizing_details'].get('adjustments', {})
                    msg += f"""Adjustments: Vol={adj.get('volatility', 1):.2f}x, Conv={adj.get('conviction', 1):.2f}x, Heat={adj.get('heat', 1):.2f}x

"""

            if validation.get('greek_check'):
                gc = validation['greek_check']
                if gc['status'] == 'WARNING':
                    msg += f"""⚠️ **Greek Warnings:**
{', '.join([w['greek'] for w in gc.get('warnings', [])])}

"""

            if validation.get('warnings'):
                msg += f"""⚠️ **Warnings:**
{chr(10).join(['- ' + w for w in validation['warnings']])}

"""

        msg += f"💡 {setup.notes}"

        return msg

    def format_portfolio_status_alert(self) -> str:
        """Format portfolio status as Telegram alert."""
        status = self.get_portfolio_status()
        greeks = status['greeks']

        msg = f"""📊 **PORTFOLIO STATUS**

**Greeks:**
- Net Delta: {greeks['net_delta']:.0f}
- Net Gamma: {greeks['net_gamma']:.1f}
- Daily Theta: ${greeks['daily_theta']:.0f}
- Net Vega: {greeks['net_vega']:.0f}

**Status:** {status['greek_status']}
**Active Positions:** {status['active_positions']}
**Daily P&L:** ${status['daily_pnl']:.0f}
"""

        if status['hedge_recommendations']:
            msg += "\n**Hedge Recommendations:**\n"
            for h in status['hedge_recommendations']:
                msg += f"- [{h['priority']}] {h['type']}: {h['action']}\n"

        return msg


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Global instance (lazy initialization)
_precious_metals_scalper: Optional[PreciousMetalsScalper] = None


def get_precious_metals_scalper(account_value: float = None) -> PreciousMetalsScalper:
    """
    Get the global precious metals scalper instance.

    Args:
        account_value: Optional account value for initialization

    Returns:
        PreciousMetalsScalper instance
    """
    global _precious_metals_scalper
    if _precious_metals_scalper is None:
        _precious_metals_scalper = PreciousMetalsScalper(account_value or 100000)
    elif account_value:
        _precious_metals_scalper.position_sizer.set_account_value(account_value)
    return _precious_metals_scalper


def initialize_scalper(account_value: float) -> PreciousMetalsScalper:
    """
    Initialize the precious metals scalper with account value.

    Args:
        account_value: Account value for position sizing

    Returns:
        Initialized PreciousMetalsScalper instance
    """
    global _precious_metals_scalper
    _precious_metals_scalper = PreciousMetalsScalper(account_value)
    log.info(f"Precious Metals Scalper initialized with ${account_value:,.0f}")
    return _precious_metals_scalper


# Export classes for direct use
__all__ = [
    'PreciousMetalsScalper',
    'CatalystScanner',
    'MomentumAnalyzer',
    'OptionsScreener',
    'GreekOptimizer',
    'PositionSizer',
    'MetalsStrategy',
    'SignalDirection',
    'MetalsSetup',
    'GreekTargets',
    'PortfolioGreeks',
    'HedgeRecommendation',
    'get_precious_metals_scalper',
    'initialize_scalper'
]
