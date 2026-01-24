"""
Inception Detector - Next-Gen Convex Instability Detection

Implements the "meta-sensory organs" that detect instability before standard indicators:

1. Event Horizon Detector - Variance of variance, correlation changes, instrument dispersion
2. Correlation Fracture Monitor - Watches for breaks in stable relationships
3. Liquidity Elasticity Sensor - Measures fragility (how easily price moves)
4. Temporal Anomaly Engine - Detects when time behavior is abnormal
5. Attention Surge Map - Tracks attention acceleration before narrative exists
6. Instability Index - Combines all sensors into inception detection

Mathematical Foundation:
- Effective volatility: σ'_eff = σ_eff × κ(regime) × (1 + ι(microstructure)) × (1 + γ*(options))
- Liquidity elasticity: ε = |ΔS| / |Q|
- Correlation fracture: C = Σ |ρ_ij(t) - ρ̄_ij|
- Instability index: I = g(h(t), ε, C, ∂σ', ∂N)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math
import statistics

from wsb_snake.utils.logger import log
from wsb_snake.utils.session_regime import get_eastern_time, get_session_info


class InstabilityState(Enum):
    """Instability regime states."""
    STABLE = "stable"
    WARMING = "warming"
    CRITICAL = "critical"
    INCEPTION = "inception"


@dataclass
class EventHorizonReading:
    """Event Horizon Detector output."""
    variance_of_variance: float = 0.0
    correlation_velocity: float = 0.0
    instrument_dispersion: float = 0.0
    sensitivity_score: float = 0.0
    phase_transition_risk: bool = False


@dataclass
class CorrelationFracture:
    """Correlation Fracture Monitor output."""
    spy_vix_fracture: bool = False
    qqq_spy_divergence: float = 0.0
    iv_price_disconnect: bool = False
    bidirectional_flow: bool = False
    fracture_score: float = 0.0
    anomalies: List[str] = field(default_factory=list)


@dataclass
class LiquidityElasticity:
    """Liquidity Elasticity Sensor output."""
    elasticity: float = 0.0
    absorption_capacity: float = 1.0
    air_pocket_detected: bool = False
    fragility_score: float = 0.0


@dataclass
class TemporalAnomaly:
    """Temporal Anomaly Engine output."""
    reaction_latency: float = 1.0
    event_compression: float = 0.0
    time_distortion_score: float = 0.0
    signals_per_minute: float = 0.0


@dataclass
class AttentionSurge:
    """Attention Surge Map output."""
    news_velocity: float = 0.0
    headline_duplication: float = 0.0
    keyword_emergence_speed: float = 0.0
    attention_acceleration: float = 0.0
    narrative_coherence: float = 1.0


@dataclass
class InceptionState:
    """Complete Inception Stack state."""
    timestamp: datetime
    ticker: str
    
    event_horizon: EventHorizonReading
    correlation_fracture: CorrelationFracture
    liquidity_elasticity: LiquidityElasticity
    temporal_anomaly: TemporalAnomaly
    attention_surge: AttentionSurge
    
    instability_index: float = 0.0
    instability_state: InstabilityState = InstabilityState.STABLE
    
    inception_detected: bool = False
    inception_confidence: float = 0.0
    
    signals: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "ticker": self.ticker,
            "instability_index": round(self.instability_index, 3),
            "instability_state": self.instability_state.value,
            "inception_detected": self.inception_detected,
            "inception_confidence": round(self.inception_confidence, 3),
            "event_horizon_score": round(self.event_horizon.sensitivity_score, 3),
            "correlation_fracture_score": round(self.correlation_fracture.fracture_score, 3),
            "liquidity_fragility": round(self.liquidity_elasticity.fragility_score, 3),
            "temporal_distortion": round(self.temporal_anomaly.time_distortion_score, 3),
            "attention_acceleration": round(self.attention_surge.attention_acceleration, 3),
            "signals": self.signals,
        }


class InceptionDetector:
    """
    The Inception Stack - Detects convex instability before the move has a name.
    
    Combines 6 sensors to identify the pre-surge state where small perturbations
    create outsized effects.
    """
    
    INSTABILITY_THRESHOLD = 0.65
    INCEPTION_THRESHOLD = 0.80
    
    SENSOR_WEIGHTS = {
        "event_horizon": 0.20,
        "correlation_fracture": 0.20,
        "liquidity_elasticity": 0.20,
        "temporal_anomaly": 0.15,
        "attention_surge": 0.15,
        "options_pressure": 0.10,
    }
    
    def __init__(self):
        self._price_history: Dict[str, List[Tuple[datetime, float, float]]] = {}
        self._correlation_baseline: Dict[str, float] = {
            "spy_qqq": 0.92,
            "spy_iwm": 0.85,
            "spy_vix": -0.75,
        }
        self._signal_history: Dict[str, List[datetime]] = {}
        self._news_history: List[Tuple[datetime, str]] = []
        
    def update_price_history(
        self,
        ticker: str,
        price: float,
        volume: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update price history for elasticity calculations."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        if ticker not in self._price_history:
            self._price_history[ticker] = []
            
        self._price_history[ticker].append((timestamp, price, volume))
        
        cutoff = timestamp - timedelta(hours=2)
        self._price_history[ticker] = [
            (t, p, v) for t, p, v in self._price_history[ticker]
            if t > cutoff
        ]
    
    def update_news_history(self, headlines: List[str], timestamp: Optional[datetime] = None) -> None:
        """Update news history for attention tracking."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        for headline in headlines:
            self._news_history.append((timestamp, headline))
        
        cutoff = timestamp - timedelta(hours=1)
        self._news_history = [(t, h) for t, h in self._news_history if t > cutoff]
    
    def record_signal(self, ticker: str, timestamp: Optional[datetime] = None) -> None:
        """Record signal occurrence for temporal analysis."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        if ticker not in self._signal_history:
            self._signal_history[ticker] = []
            
        self._signal_history[ticker].append(timestamp)
        
        cutoff = timestamp - timedelta(minutes=30)
        self._signal_history[ticker] = [t for t in self._signal_history[ticker] if t > cutoff]
    
    def detect_event_horizon(
        self,
        ticker: str,
        price_bars: List[Dict],
        options_data: Optional[Dict] = None,
        related_tickers: Optional[Dict[str, float]] = None,
    ) -> EventHorizonReading:
        """
        Event Horizon Detector - Detects pre-surge phase transitions.
        
        Looks for:
        - Variance of variance (volatility of volatility)
        - Rate of change of correlations
        - Dispersion between instruments
        """
        reading = EventHorizonReading()
        
        if not price_bars or len(price_bars) < 10:
            return reading
        
        returns = []
        for i in range(1, len(price_bars)):
            prev_close = price_bars[i-1].get("c", price_bars[i-1].get("close", 0))
            curr_close = price_bars[i].get("c", price_bars[i].get("close", 0))
            if prev_close > 0:
                ret = (curr_close - prev_close) / prev_close
                returns.append(ret)
        
        if len(returns) >= 5:
            window_size = min(5, len(returns) // 2)
            rolling_vars = []
            for i in range(len(returns) - window_size + 1):
                window = returns[i:i + window_size]
                if len(window) >= 2:
                    rolling_vars.append(statistics.variance(window))
            
            if len(rolling_vars) >= 2:
                reading.variance_of_variance = statistics.variance(rolling_vars)
        
        if related_tickers and len(related_tickers) >= 2:
            values = list(related_tickers.values())
            if all(v != 0 for v in values):
                mean_change = statistics.mean(values)
                dispersion = sum(abs(v - mean_change) for v in values) / len(values)
                reading.instrument_dispersion = dispersion
        
        if options_data:
            iv = options_data.get("iv_surface", {}).get("atm_iv", 0)
            if iv > 0.5:
                reading.sensitivity_score += 0.2
            
            gex = options_data.get("gex", {})
            if gex.get("gex_regime") == "negative":
                reading.sensitivity_score += 0.3
        
        reading.sensitivity_score += min(reading.variance_of_variance * 100, 0.3)
        reading.sensitivity_score += min(reading.instrument_dispersion * 10, 0.2)
        reading.sensitivity_score = min(reading.sensitivity_score, 1.0)
        
        reading.phase_transition_risk = reading.sensitivity_score > 0.6
        
        return reading
    
    def detect_correlation_fracture(
        self,
        ticker_prices: Dict[str, float],
        ticker_changes: Dict[str, float],
        options_data: Optional[Dict] = None,
    ) -> CorrelationFracture:
        """
        Correlation Fracture Monitor - Detects breaks in stable relationships.
        
        Watches for:
        - SPY up while VIX up (should be inverse)
        - QQQ diverging from SPY (normally high correlation)
        - IV expanding without price movement
        - Call and put volume both exploding (uncertainty)
        """
        fracture = CorrelationFracture()
        
        spy_change = ticker_changes.get("SPY", 0)
        vix_change = ticker_changes.get("VIX", 0)
        qqq_change = ticker_changes.get("QQQ", 0)
        
        if spy_change > 0.002 and vix_change > 0.01:
            fracture.spy_vix_fracture = True
            fracture.anomalies.append("SPY_VIX_POSITIVE_CORRELATION")
        elif spy_change < -0.002 and vix_change < -0.01:
            fracture.spy_vix_fracture = True
            fracture.anomalies.append("VIX_FALLING_WITH_SPY")
        
        fracture.qqq_spy_divergence = abs(qqq_change - spy_change)
        if fracture.qqq_spy_divergence > 0.005:
            fracture.anomalies.append("QQQ_SPY_DIVERGENCE")
        
        if options_data:
            iv = options_data.get("iv_surface", {}).get("atm_iv", 0)
            price_change = abs(ticker_changes.get(options_data.get("ticker", ""), 0))
            
            if iv > 0.4 and price_change < 0.002:
                fracture.iv_price_disconnect = True
                fracture.anomalies.append("IV_EXPANSION_NO_MOVE")
            
            metrics = options_data.get("metrics", {})
            call_vol = metrics.get("total_call_volume", 0)
            put_vol = metrics.get("total_put_volume", 0)
            
            if call_vol > 10000 and put_vol > 10000:
                ratio = max(call_vol, put_vol) / min(call_vol, put_vol) if min(call_vol, put_vol) > 0 else 10
                if ratio < 1.5:
                    fracture.bidirectional_flow = True
                    fracture.anomalies.append("BIDIRECTIONAL_VOLUME")
        
        fracture.fracture_score = 0.0
        if fracture.spy_vix_fracture:
            fracture.fracture_score += 0.35
        if fracture.qqq_spy_divergence > 0.005:
            fracture.fracture_score += min(fracture.qqq_spy_divergence * 20, 0.25)
        if fracture.iv_price_disconnect:
            fracture.fracture_score += 0.25
        if fracture.bidirectional_flow:
            fracture.fracture_score += 0.15
        
        fracture.fracture_score = min(fracture.fracture_score, 1.0)
        
        return fracture
    
    def detect_liquidity_elasticity(
        self,
        ticker: str,
        current_price: float,
        current_volume: float,
        options_data: Optional[Dict] = None,
    ) -> LiquidityElasticity:
        """
        Liquidity Elasticity Sensor - Measures fragility.
        
        Calculates:
        - ε = |ΔS| / |Q| (price change per unit volume)
        - Air pocket detection (gaps with low volume)
        - Absorption capacity
        """
        elasticity = LiquidityElasticity()
        
        history = self._price_history.get(ticker, [])
        
        if len(history) >= 2:
            recent = history[-10:] if len(history) >= 10 else history
            
            price_changes = []
            volume_changes = []
            
            for i in range(1, len(recent)):
                prev_t, prev_p, prev_v = recent[i-1]
                curr_t, curr_p, curr_v = recent[i]
                
                if prev_p > 0 and prev_v > 0:
                    price_change = abs(curr_p - prev_p) / prev_p
                    price_changes.append(price_change)
                    volume_changes.append(curr_v)
            
            if price_changes and volume_changes:
                avg_price_change = statistics.mean(price_changes)
                avg_volume = statistics.mean(volume_changes)
                
                if avg_volume > 0:
                    elasticity.elasticity = avg_price_change / (avg_volume / 1e6)
                
                for i, (pc, vol) in enumerate(zip(price_changes, volume_changes)):
                    if pc > avg_price_change * 2 and vol < avg_volume * 0.5:
                        elasticity.air_pocket_detected = True
                        break
        
        if options_data:
            gex = options_data.get("gex", {})
            total_gex = abs(gex.get("total_gex", 0))
            
            if total_gex > 0:
                elasticity.absorption_capacity = min(total_gex / 1e9, 1.0)
            
            if gex.get("gex_regime") == "negative":
                elasticity.elasticity *= 1.5
        
        elasticity.fragility_score = min(elasticity.elasticity * 10, 0.7)
        if elasticity.air_pocket_detected:
            elasticity.fragility_score += 0.3
        elasticity.fragility_score = min(elasticity.fragility_score, 1.0)
        
        return elasticity
    
    def detect_temporal_anomaly(
        self,
        ticker: str,
        recent_signals: int = 0,
        news_to_price_latency: float = 1.0,
    ) -> TemporalAnomaly:
        """
        Temporal Anomaly Engine - Detects abnormal time behavior.
        
        Watches for:
        - Faster than usual reactions
        - Event compression (multiple events in short window)
        - Instantaneous rather than delayed responses
        """
        anomaly = TemporalAnomaly()
        
        signal_times = self._signal_history.get(ticker, [])
        
        if signal_times:
            now = datetime.utcnow()
            recent = [t for t in signal_times if (now - t).total_seconds() < 300]
            anomaly.signals_per_minute = len(recent) / 5.0
            
            if len(recent) >= 3:
                intervals = []
                for i in range(1, len(recent)):
                    interval = (recent[i] - recent[i-1]).total_seconds()
                    intervals.append(interval)
                
                if intervals:
                    avg_interval = statistics.mean(intervals)
                    if avg_interval < 60:
                        anomaly.event_compression = 1.0 - (avg_interval / 60)
        
        anomaly.reaction_latency = news_to_price_latency
        if news_to_price_latency < 0.5:
            anomaly.time_distortion_score += 0.4
        
        anomaly.time_distortion_score += anomaly.event_compression * 0.3
        anomaly.time_distortion_score += min(anomaly.signals_per_minute * 0.1, 0.3)
        anomaly.time_distortion_score = min(anomaly.time_distortion_score, 1.0)
        
        return anomaly
    
    def detect_attention_surge(
        self,
        ticker: str,
        news_data: Optional[Dict] = None,
        social_data: Optional[Dict] = None,
    ) -> AttentionSurge:
        """
        Attention Surge Map - Tracks attention acceleration.
        
        Key insight: Big moves start when attention spikes before narrative coherence.
        "People are reacting faster than they can explain."
        """
        surge = AttentionSurge()
        
        if news_data:
            headlines = news_data.get("headlines", [])
            
            recent_news = [(t, h) for t, h in self._news_history 
                          if (datetime.utcnow() - t).total_seconds() < 300]
            
            surge.news_velocity = len(recent_news) / 5.0
            
            if headlines:
                unique_headlines = set(h.lower()[:50] for h in headlines)
                surge.headline_duplication = 1.0 - (len(unique_headlines) / len(headlines)) if headlines else 0
            
            keywords = news_data.get("keywords", [])
            if keywords:
                surge.keyword_emergence_speed = len(keywords) * 0.1
        
        if social_data:
            mention_rate = social_data.get("mention_rate", 0)
            baseline = social_data.get("baseline_rate", 1)
            
            if baseline > 0:
                surge.attention_acceleration = (mention_rate - baseline) / baseline
        
        if surge.news_velocity > 0 and surge.headline_duplication < 0.3:
            surge.narrative_coherence = 0.3
        
        overall = (
            surge.news_velocity * 0.3 +
            surge.headline_duplication * 0.2 +
            surge.attention_acceleration * 0.3 +
            (1 - surge.narrative_coherence) * 0.2
        )
        surge.attention_acceleration = min(overall, 1.0)
        
        return surge
    
    def calculate_instability_index(
        self,
        event_horizon: EventHorizonReading,
        correlation_fracture: CorrelationFracture,
        liquidity_elasticity: LiquidityElasticity,
        temporal_anomaly: TemporalAnomaly,
        attention_surge: AttentionSurge,
        options_pressure: float = 0.0,
    ) -> Tuple[float, InstabilityState]:
        """
        Calculate the Instability Index (I_t).
        
        I_t = g(h(t), ε_t, C_t, ∂σ', ∂N)
        
        Returns (index, state).
        """
        index = (
            self.SENSOR_WEIGHTS["event_horizon"] * event_horizon.sensitivity_score +
            self.SENSOR_WEIGHTS["correlation_fracture"] * correlation_fracture.fracture_score +
            self.SENSOR_WEIGHTS["liquidity_elasticity"] * liquidity_elasticity.fragility_score +
            self.SENSOR_WEIGHTS["temporal_anomaly"] * temporal_anomaly.time_distortion_score +
            self.SENSOR_WEIGHTS["attention_surge"] * attention_surge.attention_acceleration +
            self.SENSOR_WEIGHTS["options_pressure"] * options_pressure
        )
        
        if index >= self.INCEPTION_THRESHOLD:
            state = InstabilityState.INCEPTION
        elif index >= self.INSTABILITY_THRESHOLD:
            state = InstabilityState.CRITICAL
        elif index >= 0.4:
            state = InstabilityState.WARMING
        else:
            state = InstabilityState.STABLE
        
        return (index, state)
    
    def detect_inception(
        self,
        ticker: str,
        price_bars: List[Dict],
        current_price: float,
        current_volume: float,
        ticker_changes: Dict[str, float],
        options_data: Optional[Dict] = None,
        news_data: Optional[Dict] = None,
        related_tickers: Optional[Dict[str, float]] = None,
    ) -> InceptionState:
        """
        The Inception Stack - Complete instability detection.
        
        Returns InceptionState with all sensor readings and final verdict.
        """
        now = datetime.utcnow()
        
        self.update_price_history(ticker, current_price, current_volume, now)
        
        if news_data and "headlines" in news_data:
            self.update_news_history(news_data["headlines"], now)
        
        event_horizon = self.detect_event_horizon(
            ticker, price_bars, options_data, related_tickers
        )
        
        all_changes = {ticker: ticker_changes.get(ticker, 0)}
        for t, c in (related_tickers or {}).items():
            all_changes[t] = c
        
        correlation_fracture = self.detect_correlation_fracture(
            {ticker: current_price}, all_changes, options_data
        )
        
        liquidity_elasticity = self.detect_liquidity_elasticity(
            ticker, current_price, current_volume, options_data
        )
        
        temporal_anomaly = self.detect_temporal_anomaly(ticker)
        
        attention_surge = self.detect_attention_surge(ticker, news_data)
        
        options_pressure = 0.0
        if options_data:
            gex = options_data.get("gex", {})
            if gex.get("gex_regime") == "negative":
                options_pressure += 0.3
            
            max_pain = options_data.get("max_pain", {})
            if max_pain.get("distance_pct", 0) > 0.5:
                options_pressure += 0.2
            
            signals = options_data.get("signals", [])
            options_pressure += len(signals) * 0.1
            options_pressure = min(options_pressure, 1.0)
        
        instability_index, instability_state = self.calculate_instability_index(
            event_horizon, correlation_fracture, liquidity_elasticity,
            temporal_anomaly, attention_surge, options_pressure
        )
        
        signals = []
        
        if event_horizon.phase_transition_risk:
            signals.append("PHASE_TRANSITION_RISK")
        
        for anomaly in correlation_fracture.anomalies:
            signals.append(anomaly)
        
        if liquidity_elasticity.air_pocket_detected:
            signals.append("AIR_POCKET_DETECTED")
        
        if liquidity_elasticity.fragility_score > 0.5:
            signals.append("HIGH_FRAGILITY")
        
        if temporal_anomaly.event_compression > 0.5:
            signals.append("TIME_COMPRESSION")
        
        if attention_surge.narrative_coherence < 0.5:
            signals.append("ATTENTION_WITHOUT_NARRATIVE")
        
        inception_detected = (
            instability_state in [InstabilityState.CRITICAL, InstabilityState.INCEPTION] and
            len(signals) >= 2
        )
        
        confidence = instability_index * (1 + len(signals) * 0.1)
        confidence = min(confidence, 1.0)
        
        state = InceptionState(
            timestamp=now,
            ticker=ticker,
            event_horizon=event_horizon,
            correlation_fracture=correlation_fracture,
            liquidity_elasticity=liquidity_elasticity,
            temporal_anomaly=temporal_anomaly,
            attention_surge=attention_surge,
            instability_index=instability_index,
            instability_state=instability_state,
            inception_detected=inception_detected,
            inception_confidence=confidence,
            signals=signals,
        )
        
        if inception_detected:
            log.warning(f"🌀 INCEPTION DETECTED: {ticker} | Index={instability_index:.2f} | Signals={signals}")
        
        return state


inception_detector = InceptionDetector()


def detect_inception(
    ticker: str,
    price_bars: List[Dict],
    current_price: float,
    current_volume: float,
    ticker_changes: Dict[str, float],
    options_data: Optional[Dict] = None,
    news_data: Optional[Dict] = None,
    related_tickers: Optional[Dict[str, float]] = None,
) -> InceptionState:
    """Convenience function for inception detection."""
    return inception_detector.detect_inception(
        ticker, price_bars, current_price, current_volume,
        ticker_changes, options_data, news_data, related_tickers
    )
