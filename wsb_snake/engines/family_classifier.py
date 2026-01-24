"""
Setup Family Classifier

Implements the 10 0DTE Setup Families conceptual taxonomy:
1. VWAP Reclaim + Gamma Snap
2. Strike Magnet Pin → Late Break
3. Afternoon Range Expansion (ARE)
4. Liquidity Sweep + Reversal
5. News-Assisted Gamma Ignition
6. Power-Hour Trend Continuation
7. False Break Trap → Real Move
8. Volatility Regime Shift
9. Crowd Ignition + Structure Confirmation
10. End-of-Day Mean Reversion Snap

Each family has:
- Unique probability curve over time
- Regime compatibility
- Death conditions
- Asymmetric vs consistent classification
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any
import math

from wsb_snake.utils.logger import log
from wsb_snake.utils.session_regime import get_eastern_time, get_session_info


class SetupFamily(Enum):
    """The 10 0DTE Setup Families."""
    VWAP_RECLAIM_GAMMA_SNAP = "vwap_reclaim"
    STRIKE_MAGNET_BREAK = "strike_magnet"
    AFTERNOON_RANGE_EXPANSION = "range_expansion"
    LIQUIDITY_SWEEP_REVERSAL = "liquidity_sweep"
    NEWS_GAMMA_IGNITION = "news_ignition"
    POWER_HOUR_CONTINUATION = "power_continuation"
    FALSE_BREAK_TRAP = "false_break"
    VOLATILITY_REGIME_SHIFT = "vol_regime"
    CROWD_IGNITION = "crowd_ignition"
    MEAN_REVERSION_SNAP = "mean_reversion"


class FamilyType(Enum):
    """Asymmetric vs Consistent family classification."""
    ASYMMETRIC = "asymmetric"
    CONSISTENT = "consistent"


class FamilyLifecycle(Enum):
    """Family lifecycle states."""
    DORMANT = "dormant"
    AWAKENING = "awakening"
    ALIVE = "alive"
    PEAKED = "peaked"
    DYING = "dying"
    DEAD = "dead"
    COOLDOWN = "cooldown"


@dataclass
class FamilyConfig:
    """Configuration for each setup family."""
    family: SetupFamily
    family_type: FamilyType
    
    min_viable_hour: int
    max_viable_hour: int
    peak_hour: int
    
    compatible_regimes: List[str]
    incompatible_regimes: List[str]
    
    base_probability: float
    peak_probability: float
    decay_rate: float
    
    cooldown_minutes: int = 30
    max_daily_signals: int = 3
    memory_lookback_days: int = 5
    
    required_conditions: List[str] = field(default_factory=list)


@dataclass
class FamilyState:
    """Runtime state for a family on a specific ticker."""
    family: SetupFamily
    ticker: str
    lifecycle: FamilyLifecycle = FamilyLifecycle.DORMANT
    
    viability_score: float = 0.0
    probability_score: float = 0.0
    
    conditions_met: Dict[str, bool] = field(default_factory=dict)
    signal_count_today: int = 0
    last_signal_time: Optional[datetime] = None
    
    recent_outcomes: List[bool] = field(default_factory=list)
    
    awakened_at: Optional[datetime] = None
    peaked_at: Optional[datetime] = None
    died_at: Optional[datetime] = None
    death_reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "family": self.family.value,
            "ticker": self.ticker,
            "lifecycle": self.lifecycle.value,
            "viability_score": round(self.viability_score, 3),
            "probability_score": round(self.probability_score, 3),
            "conditions_met": self.conditions_met,
            "signal_count_today": self.signal_count_today,
        }


FAMILY_CONFIGS: Dict[SetupFamily, FamilyConfig] = {
    SetupFamily.VWAP_RECLAIM_GAMMA_SNAP: FamilyConfig(
        family=SetupFamily.VWAP_RECLAIM_GAMMA_SNAP,
        family_type=FamilyType.CONSISTENT,
        min_viable_hour=14,
        max_viable_hour=16,
        peak_hour=15,
        compatible_regimes=["strong_bullish", "bullish", "bearish", "strong_bearish"],
        incompatible_regimes=["neutral"],
        base_probability=0.3,
        peak_probability=0.7,
        decay_rate=0.15,
        required_conditions=["vwap_reclaim", "volume_confirmation", "follow_through"],
    ),
    
    SetupFamily.STRIKE_MAGNET_BREAK: FamilyConfig(
        family=SetupFamily.STRIKE_MAGNET_BREAK,
        family_type=FamilyType.ASYMMETRIC,
        min_viable_hour=14,
        max_viable_hour=16,
        peak_hour=15,
        compatible_regimes=["neutral", "bullish", "bearish"],
        incompatible_regimes=[],
        base_probability=0.15,
        peak_probability=0.55,
        decay_rate=0.25,
        required_conditions=["pin_identified", "pin_mature", "break_confirmed"],
    ),
    
    SetupFamily.AFTERNOON_RANGE_EXPANSION: FamilyConfig(
        family=SetupFamily.AFTERNOON_RANGE_EXPANSION,
        family_type=FamilyType.CONSISTENT,
        min_viable_hour=13,
        max_viable_hour=16,
        peak_hour=14,
        compatible_regimes=["strong_bullish", "bullish", "bearish", "strong_bearish"],
        incompatible_regimes=["neutral"],
        base_probability=0.35,
        peak_probability=0.65,
        decay_rate=0.1,
        required_conditions=["morning_compressed", "range_expanding", "volume_surge"],
    ),
    
    SetupFamily.LIQUIDITY_SWEEP_REVERSAL: FamilyConfig(
        family=SetupFamily.LIQUIDITY_SWEEP_REVERSAL,
        family_type=FamilyType.ASYMMETRIC,
        min_viable_hour=13,
        max_viable_hour=16,
        peak_hour=15,
        compatible_regimes=["neutral", "bullish", "bearish"],
        incompatible_regimes=["strong_bullish", "strong_bearish"],
        base_probability=0.2,
        peak_probability=0.6,
        decay_rate=0.2,
        required_conditions=["key_level_proximity", "sweep_occurred", "snapback_volume"],
    ),
    
    SetupFamily.NEWS_GAMMA_IGNITION: FamilyConfig(
        family=SetupFamily.NEWS_GAMMA_IGNITION,
        family_type=FamilyType.ASYMMETRIC,
        min_viable_hour=9,
        max_viable_hour=16,
        peak_hour=15,
        compatible_regimes=["strong_bullish", "bullish", "bearish", "strong_bearish", "neutral"],
        incompatible_regimes=[],
        base_probability=0.25,
        peak_probability=0.7,
        decay_rate=0.3,
        required_conditions=["news_catalyst", "structure_primed", "immediate_acceleration"],
    ),
    
    SetupFamily.POWER_HOUR_CONTINUATION: FamilyConfig(
        family=SetupFamily.POWER_HOUR_CONTINUATION,
        family_type=FamilyType.CONSISTENT,
        min_viable_hour=15,
        max_viable_hour=16,
        peak_hour=15,
        compatible_regimes=["strong_bullish", "bullish", "bearish", "strong_bearish"],
        incompatible_regimes=["neutral"],
        base_probability=0.4,
        peak_probability=0.75,
        decay_rate=0.1,
        required_conditions=["trend_established", "consolidation_complete", "breakout_volume"],
    ),
    
    SetupFamily.FALSE_BREAK_TRAP: FamilyConfig(
        family=SetupFamily.FALSE_BREAK_TRAP,
        family_type=FamilyType.ASYMMETRIC,
        min_viable_hour=14,
        max_viable_hour=16,
        peak_hour=15,
        compatible_regimes=["neutral", "bullish", "bearish"],
        incompatible_regimes=[],
        base_probability=0.2,
        peak_probability=0.6,
        decay_rate=0.2,
        required_conditions=["prior_failed_break", "second_attempt", "stronger_volume"],
    ),
    
    SetupFamily.VOLATILITY_REGIME_SHIFT: FamilyConfig(
        family=SetupFamily.VOLATILITY_REGIME_SHIFT,
        family_type=FamilyType.ASYMMETRIC,
        min_viable_hour=13,
        max_viable_hour=16,
        peak_hour=15,
        compatible_regimes=["neutral"],
        incompatible_regimes=["strong_bullish", "strong_bearish"],
        base_probability=0.15,
        peak_probability=0.5,
        decay_rate=0.25,
        required_conditions=["low_intraday_vol", "sudden_expansion", "sustained_movement"],
    ),
    
    SetupFamily.CROWD_IGNITION: FamilyConfig(
        family=SetupFamily.CROWD_IGNITION,
        family_type=FamilyType.ASYMMETRIC,
        min_viable_hour=14,
        max_viable_hour=16,
        peak_hour=15,
        compatible_regimes=["bullish", "strong_bullish"],
        incompatible_regimes=["bearish", "strong_bearish"],
        base_probability=0.2,
        peak_probability=0.55,
        decay_rate=0.2,
        required_conditions=["social_attention", "technical_break", "sufficient_liquidity"],
    ),
    
    SetupFamily.MEAN_REVERSION_SNAP: FamilyConfig(
        family=SetupFamily.MEAN_REVERSION_SNAP,
        family_type=FamilyType.CONSISTENT,
        min_viable_hour=15,
        max_viable_hour=16,
        peak_hour=15,
        compatible_regimes=["strong_bullish", "strong_bearish"],
        incompatible_regimes=["neutral"],
        base_probability=0.3,
        peak_probability=0.65,
        decay_rate=0.15,
        required_conditions=["extended_move", "momentum_loss", "reversal_volume"],
    ),
}


class FamilyClassifier:
    """
    Classifies market conditions into setup families.
    
    Maintains a Family Viability Matrix ranking families by:
    - Regime compatibility
    - Time alignment
    - Volatility state
    - Liquidity
    - Recent outcomes
    """
    
    VIABILITY_THRESHOLD = 0.4
    
    def __init__(self):
        self._states: Dict[str, Dict[SetupFamily, FamilyState]] = {}
        self._daily_signals: Dict[SetupFamily, int] = {f: 0 for f in SetupFamily}
        self._family_outcomes: Dict[SetupFamily, List[Tuple[datetime, bool]]] = {
            f: [] for f in SetupFamily
        }
        self._last_reset_date: Optional[date] = None
    
    def _get_or_create_state(self, ticker: str, family: SetupFamily) -> FamilyState:
        """Get or create family state for ticker."""
        if ticker not in self._states:
            self._states[ticker] = {}
        if family not in self._states[ticker]:
            self._states[ticker][family] = FamilyState(family=family, ticker=ticker)
        return self._states[ticker][family]
    
    def _reset_daily_counts(self):
        """Reset daily signal counts at market open."""
        et_now = get_eastern_time()
        today = et_now.date()
        
        if self._last_reset_date != today:
            self._daily_signals = {f: 0 for f in SetupFamily}
            self._last_reset_date = today
            
            for ticker_states in self._states.values():
                for state in ticker_states.values():
                    state.signal_count_today = 0
    
    def _calculate_time_factor(self, config: FamilyConfig) -> float:
        """Calculate time-based probability factor."""
        et_now = get_eastern_time()
        hour = et_now.hour + et_now.minute / 60.0
        
        if hour < config.min_viable_hour or hour > config.max_viable_hour:
            return 0.0
        
        distance_to_peak = abs(hour - config.peak_hour)
        max_distance = max(
            config.peak_hour - config.min_viable_hour,
            config.max_viable_hour - config.peak_hour
        )
        
        if max_distance == 0:
            return 1.0
        
        time_factor = 1.0 - (distance_to_peak / max_distance) * config.decay_rate
        return max(0.0, min(1.0, time_factor))
    
    def _calculate_regime_factor(self, config: FamilyConfig, regime: str) -> float:
        """Calculate regime compatibility factor."""
        if regime in config.incompatible_regimes:
            return 0.0
        if regime in config.compatible_regimes:
            return 1.0
        return 0.5
    
    def _calculate_memory_factor(self, family: SetupFamily) -> float:
        """Calculate factor based on recent outcomes (memory veto)."""
        outcomes = self._family_outcomes.get(family, [])
        if not outcomes:
            return 1.0
        
        cutoff = get_eastern_time() - timedelta(days=FAMILY_CONFIGS[family].memory_lookback_days)
        recent = [o for t, o in outcomes if t > cutoff]
        
        if not recent:
            return 1.0
        
        win_rate = sum(recent) / len(recent)
        
        if len(recent) >= 3 and win_rate < 0.2:
            return 0.1
        
        return 0.5 + (win_rate * 0.5)
    
    def _calculate_saturation_factor(self, family: SetupFamily) -> float:
        """Calculate signal saturation (repetition exhaustion)."""
        config = FAMILY_CONFIGS[family]
        daily_count = self._daily_signals.get(family, 0)
        
        if daily_count >= config.max_daily_signals:
            return 0.0
        
        return 1.0 - (daily_count / config.max_daily_signals) * 0.5
    
    def calculate_viability(
        self,
        family: SetupFamily,
        regime: str,
        volatility_state: str = "normal",
        liquidity: float = 1.0,
    ) -> float:
        """
        Calculate viability score for a family.
        
        Returns score ∈ [0, 1].
        """
        config = FAMILY_CONFIGS[family]
        
        time_factor = self._calculate_time_factor(config)
        regime_factor = self._calculate_regime_factor(config, regime)
        memory_factor = self._calculate_memory_factor(family)
        saturation_factor = self._calculate_saturation_factor(family)
        
        vol_factor = 1.0
        if volatility_state == "low":
            if family == SetupFamily.VOLATILITY_REGIME_SHIFT:
                vol_factor = 1.2
            else:
                vol_factor = 0.8
        elif volatility_state == "high":
            if family == SetupFamily.VOLATILITY_REGIME_SHIFT:
                vol_factor = 0.5
            else:
                vol_factor = 1.1
        
        viability = (
            time_factor * 0.3 +
            regime_factor * 0.3 +
            memory_factor * 0.15 +
            saturation_factor * 0.15 +
            vol_factor * 0.05 +
            liquidity * 0.05
        )
        
        return max(0.0, min(1.0, viability))
    
    def detect_conditions(
        self,
        ticker: str,
        ignition_signal: Optional[Dict] = None,
        pressure_signal: Optional[Dict] = None,
        surge_signal: Optional[Dict] = None,
        momentum_data: Optional[Dict] = None,
        news_data: Optional[Dict] = None,
    ) -> Dict[SetupFamily, Dict[str, bool]]:
        """
        Detect which conditions are met for each family.
        
        Returns mapping of family -> {condition: bool}.
        """
        results = {}
        
        has_volume_surge = False
        has_momentum = False
        has_news = False
        has_vwap_reclaim = False
        has_range_expansion = False
        is_near_day_high = False
        is_near_day_low = False
        has_reversal = False
        trend_direction = "neutral"
        is_extended = False
        
        if ignition_signal:
            signals = ignition_signal.get("signals", [])
            has_volume_surge = "VOLUME_SURGE" in signals
            has_momentum = any(s in signals for s in ["RSI_OVERSOLD", "RSI_OVERBOUGHT", "MACD_BULLISH", "MACD_BEARISH"])
            is_near_day_high = "NEAR_DAY_HIGH" in signals
            is_near_day_low = "NEAR_DAY_LOW" in signals
            
            if any(s in signals for s in ["RSI_OVERBOUGHT", "NEAR_DAY_HIGH"]):
                trend_direction = "bullish"
            elif any(s in signals for s in ["RSI_OVERSOLD", "NEAR_DAY_LOW"]):
                trend_direction = "bearish"
        
        if pressure_signal:
            pressure_signals = pressure_signal.get("signals", [])
            has_reversal = "REVERSAL_SIGNAL" in pressure_signals
            is_extended = pressure_signal.get("is_extended", False)
        
        if surge_signal:
            surge_type = surge_signal.get("setup_type", "")
            has_vwap_reclaim = "VWAP_RECLAIM" in surge_type
            has_range_expansion = surge_signal.get("is_breakout", False)
        
        if news_data:
            has_news = len(news_data.get("headlines", [])) > 0
        
        results[SetupFamily.VWAP_RECLAIM_GAMMA_SNAP] = {
            "vwap_reclaim": has_vwap_reclaim,
            "volume_confirmation": has_volume_surge,
            "follow_through": has_momentum,
        }
        
        results[SetupFamily.STRIKE_MAGNET_BREAK] = {
            "pin_identified": not has_range_expansion and not has_volume_surge,
            "pin_mature": get_eastern_time().hour >= 14,
            "break_confirmed": has_range_expansion and has_volume_surge,
        }
        
        results[SetupFamily.AFTERNOON_RANGE_EXPANSION] = {
            "morning_compressed": get_eastern_time().hour >= 13,
            "range_expanding": has_range_expansion,
            "volume_surge": has_volume_surge,
        }
        
        results[SetupFamily.LIQUIDITY_SWEEP_REVERSAL] = {
            "key_level_proximity": is_near_day_high or is_near_day_low,
            "sweep_occurred": is_extended,
            "snapback_volume": has_reversal and has_volume_surge,
        }
        
        results[SetupFamily.NEWS_GAMMA_IGNITION] = {
            "news_catalyst": has_news,
            "structure_primed": has_momentum or has_range_expansion,
            "immediate_acceleration": has_volume_surge and has_momentum,
        }
        
        results[SetupFamily.POWER_HOUR_CONTINUATION] = {
            "trend_established": trend_direction != "neutral",
            "consolidation_complete": not has_range_expansion,
            "breakout_volume": has_volume_surge and has_range_expansion,
        }
        
        results[SetupFamily.FALSE_BREAK_TRAP] = {
            "prior_failed_break": False,
            "second_attempt": has_range_expansion,
            "stronger_volume": has_volume_surge,
        }
        
        results[SetupFamily.VOLATILITY_REGIME_SHIFT] = {
            "low_intraday_vol": not has_range_expansion,
            "sudden_expansion": has_range_expansion and has_volume_surge,
            "sustained_movement": has_momentum,
        }
        
        results[SetupFamily.CROWD_IGNITION] = {
            "social_attention": False,
            "technical_break": has_range_expansion,
            "sufficient_liquidity": True,
        }
        
        results[SetupFamily.MEAN_REVERSION_SNAP] = {
            "extended_move": is_extended,
            "momentum_loss": has_reversal,
            "reversal_volume": has_volume_surge and has_reversal,
        }
        
        return results
    
    def classify_setup(
        self,
        ticker: str,
        regime: str,
        ignition_signal: Optional[Dict] = None,
        pressure_signal: Optional[Dict] = None,
        surge_signal: Optional[Dict] = None,
        momentum_data: Optional[Dict] = None,
        news_data: Optional[Dict] = None,
    ) -> List[FamilyState]:
        """
        Classify current setup into families and return viable ones.
        
        Returns list of FamilyState sorted by viability (highest first).
        """
        self._reset_daily_counts()
        
        conditions = self.detect_conditions(
            ticker, ignition_signal, pressure_signal, surge_signal,
            momentum_data, news_data
        )
        
        viable_families = []
        
        for family, config in FAMILY_CONFIGS.items():
            viability = self.calculate_viability(family, regime)
            
            state = self._get_or_create_state(ticker, family)
            state.viability_score = viability
            state.conditions_met = conditions.get(family, {})
            
            conditions_met_count = sum(state.conditions_met.values())
            required_count = len(config.required_conditions)
            condition_ratio = conditions_met_count / required_count if required_count > 0 else 0
            
            time_factor = self._calculate_time_factor(config)
            base_prob = config.base_probability
            peak_prob = config.peak_probability
            state.probability_score = base_prob + (peak_prob - base_prob) * time_factor * condition_ratio
            
            if viability >= self.VIABILITY_THRESHOLD:
                if state.lifecycle == FamilyLifecycle.DORMANT:
                    state.lifecycle = FamilyLifecycle.AWAKENING
                    state.awakened_at = get_eastern_time()
                
                if condition_ratio >= 0.66:
                    state.lifecycle = FamilyLifecycle.ALIVE
                elif condition_ratio >= 1.0:
                    state.lifecycle = FamilyLifecycle.PEAKED
                    state.peaked_at = get_eastern_time()
                
                viable_families.append(state)
            else:
                self._check_death_conditions(state, viability, condition_ratio, regime)
        
        viable_families.sort(key=lambda s: (s.viability_score, s.probability_score), reverse=True)
        
        return viable_families
    
    def _check_death_conditions(
        self,
        state: FamilyState,
        viability: float,
        condition_ratio: float,
        regime: str,
    ):
        """Check and apply death conditions to a family state."""
        config = FAMILY_CONFIGS[state.family]
        
        if state.lifecycle in [FamilyLifecycle.DEAD, FamilyLifecycle.COOLDOWN]:
            return
        
        if regime in config.incompatible_regimes:
            state.lifecycle = FamilyLifecycle.DEAD
            state.death_reason = f"Regime poisoning: {regime}"
            state.died_at = get_eastern_time()
            return
        
        et_now = get_eastern_time()
        hour = et_now.hour
        if hour > config.max_viable_hour:
            state.lifecycle = FamilyLifecycle.DEAD
            state.death_reason = "Time window expired"
            state.died_at = get_eastern_time()
            return
        
        if state.signal_count_today >= config.max_daily_signals:
            state.lifecycle = FamilyLifecycle.COOLDOWN
            state.death_reason = "Repetition exhaustion"
            state.died_at = get_eastern_time()
            return
        
        if viability < 0.2:
            state.lifecycle = FamilyLifecycle.DYING
    
    def record_signal(self, ticker: str, family: SetupFamily):
        """Record that a signal was generated for this family."""
        state = self._get_or_create_state(ticker, family)
        state.signal_count_today += 1
        state.last_signal_time = get_eastern_time()
        self._daily_signals[family] = self._daily_signals.get(family, 0) + 1
    
    def record_outcome(self, family: SetupFamily, success: bool):
        """Record outcome for memory-based learning."""
        self._family_outcomes[family].append((get_eastern_time(), success))
        
        cutoff = get_eastern_time() - timedelta(days=30)
        self._family_outcomes[family] = [
            (t, o) for t, o in self._family_outcomes[family] if t > cutoff
        ]
    
    def get_family_leaderboard(self, regime: str) -> List[Dict]:
        """
        Get ranked list of all families by viability.
        
        Returns list of dicts with family info.
        """
        self._reset_daily_counts()
        
        rankings = []
        for family, config in FAMILY_CONFIGS.items():
            viability = self.calculate_viability(family, regime)
            
            rankings.append({
                "family": family.value,
                "family_type": config.family_type.value,
                "viability": round(viability, 3),
                "alive": viability >= self.VIABILITY_THRESHOLD,
                "peak_hour": config.peak_hour,
                "base_probability": config.base_probability,
                "peak_probability": config.peak_probability,
            })
        
        rankings.sort(key=lambda r: r["viability"], reverse=True)
        return rankings
    
    def get_ticker_families(self, ticker: str) -> List[Dict]:
        """Get all family states for a ticker."""
        if ticker not in self._states:
            return []
        
        return [state.to_dict() for state in self._states[ticker].values()]


family_classifier = FamilyClassifier()


def classify_setup(
    ticker: str,
    regime: str,
    ignition_signal: Optional[Dict] = None,
    pressure_signal: Optional[Dict] = None,
    surge_signal: Optional[Dict] = None,
    momentum_data: Optional[Dict] = None,
    news_data: Optional[Dict] = None,
) -> List[Dict]:
    """Classify setup into families."""
    states = family_classifier.classify_setup(
        ticker, regime, ignition_signal, pressure_signal, surge_signal,
        momentum_data, news_data
    )
    return [s.to_dict() for s in states]


def get_leaderboard(regime: str) -> List[Dict]:
    """Get family viability leaderboard."""
    return family_classifier.get_family_leaderboard(regime)


def record_family_signal(ticker: str, family_name: str):
    """Record signal for a family."""
    try:
        family = SetupFamily(family_name)
        family_classifier.record_signal(ticker, family)
    except ValueError:
        pass


def record_family_outcome(family_name: str, success: bool):
    """Record outcome for memory learning."""
    try:
        family = SetupFamily(family_name)
        family_classifier.record_outcome(family, success)
    except ValueError:
        pass
