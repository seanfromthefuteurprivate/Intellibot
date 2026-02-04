"""
Engine 4: Probability Generator

Fuses signals from all engines into a unified probability score.
Weights are adaptive based on historical performance (via Learning Engine).

Output:
- Combined probability score
- Confidence level
- Recommended action
- Risk parameters
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from wsb_snake.utils.logger import log
from wsb_snake.db.database import get_connection
from wsb_snake.learning.trade_learner import trade_learner


class ActionType(Enum):
    """Recommended trading actions."""
    STRONG_LONG = "STRONG_LONG"
    LONG = "LONG"
    WATCH = "WATCH"
    SHORT = "SHORT"
    STRONG_SHORT = "STRONG_SHORT"
    AVOID = "AVOID"


@dataclass
class ProbabilityOutput:
    """Combined probability output."""
    ticker: str
    combined_score: float  # 0-100
    probability_win: float  # 0-1.0
    confidence: str  # "low", "medium", "high"
    action: ActionType
    direction: str  # "long", "short", "neutral"
    
    # Component scores
    ignition_score: float
    pressure_score: float
    surge_score: float
    
    # Risk parameters
    max_position_pct: float  # Max position size as % of account
    risk_reward_ratio: float
    
    # Trade plan
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    
    # Timing
    optimal_entry_window: str
    time_sensitivity: str  # "low", "medium", "high", "critical"
    
    # Evidence
    bull_thesis: List[str]
    bear_thesis: List[str]
    
    generated_at: datetime
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "combined_score": self.combined_score,
            "probability_win": self.probability_win,
            "confidence": self.confidence,
            "action": self.action.value,
            "direction": self.direction,
            "ignition_score": self.ignition_score,
            "pressure_score": self.pressure_score,
            "surge_score": self.surge_score,
            "max_position_pct": self.max_position_pct,
            "risk_reward_ratio": self.risk_reward_ratio,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "optimal_entry_window": self.optimal_entry_window,
            "time_sensitivity": self.time_sensitivity,
            "bull_thesis": self.bull_thesis,
            "bear_thesis": self.bear_thesis,
            "generated_at": self.generated_at.isoformat(),
        }


class ProbabilityGenerator:
    """
    Engine 4: Fuses all signals into probability.
    """
    
    # Default weights (updated by Learning Engine)
    DEFAULT_WEIGHTS = {
        "ignition": 0.30,
        "pressure": 0.35,
        "surge": 0.35,
    }
    
    # Action thresholds
    STRONG_THRESHOLD = 80
    NORMAL_THRESHOLD = 60
    WATCH_THRESHOLD = 40
    
    def __init__(self):
        self.weights = self._load_weights()
    
    def _load_weights(self) -> Dict[str, float]:
        """Load current weights from database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT feature_name, weight FROM model_weights
                WHERE feature_name IN ('ignition', 'pressure', 'surge')
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            if rows:
                weights = {row["feature_name"]: row["weight"] for row in rows}
                # Normalize
                total = sum(weights.values())
                return {k: v/total for k, v in weights.items()}
            
        except Exception as e:
            log.debug(f"Using default weights: {e}")
        
        return self.DEFAULT_WEIGHTS.copy()
    
    def generate(
        self,
        ticker: str,
        ignition_signals: List[Dict],
        pressure_signals: List[Dict],
        surge_signals: List[Dict],
    ) -> Optional[ProbabilityOutput]:
        """
        Generate combined probability for a ticker.
        
        Args:
            ticker: Symbol to analyze
            ignition_signals: Signals from Engine 1
            pressure_signals: Signals from Engine 2
            surge_signals: Signals from Engine 3
            
        Returns:
            ProbabilityOutput with combined analysis
        """
        # Find signals for this ticker
        ignition = next((s for s in ignition_signals if s.get("ticker") == ticker), None)
        pressure = next((s for s in pressure_signals if s.get("ticker") == ticker), None)
        surge = next((s for s in surge_signals if s.get("ticker") == ticker), None)
        
        # Extract scores (0 if no signal)
        ignition_score = ignition.get("score", 0) if ignition else 0
        pressure_score = pressure.get("score", 0) if pressure else 0
        surge_score = surge.get("score", 0) if surge else 0
        
        # Skip if no signals at all
        if ignition_score == 0 and pressure_score == 0 and surge_score == 0:
            return None
        
        # Calculate weighted score
        combined_score = (
            ignition_score * self.weights.get("ignition", 0.33) +
            pressure_score * self.weights.get("pressure", 0.33) +
            surge_score * self.weights.get("surge", 0.33)
        )

        # Apply screenshot learning boost/penalty
        direction_guess = self._determine_direction(ignition, pressure, surge)
        trade_type = "CALLS" if direction_guess == "long" else "PUTS"
        current_hour = datetime.now().hour
        try:
            learning_boost, boost_reasons = trade_learner.get_confidence_adjustment(
                ticker=ticker,
                trade_type=trade_type,
                current_hour=current_hour,
                pattern=None  # Could extract from ignition/pressure signals
            )
            if learning_boost != 0:
                # Apply boost as percentage of score (e.g., +15% boost = score * 1.15)
                combined_score = combined_score * (1 + learning_boost)
                combined_score = min(100, max(0, combined_score))  # Clamp 0-100
                if boost_reasons:
                    log.info(f"Screenshot learning boost for {ticker}: {learning_boost:+.0%} - {boost_reasons[0]}")
        except Exception as e:
            log.debug(f"Screenshot learning check failed: {e}")

        # Convert to probability (sigmoid-like transformation)
        probability_win = self._score_to_probability(combined_score)
        
        # Determine confidence
        signal_count = sum([
            1 if ignition_score > 0 else 0,
            1 if pressure_score > 0 else 0,
            1 if surge_score > 0 else 0,
        ])
        
        confidence = self._calculate_confidence(signal_count, combined_score)
        
        # Determine direction
        direction = self._determine_direction(ignition, pressure, surge)
        
        # Determine action
        action = self._determine_action(combined_score, direction, confidence)
        
        # Build thesis
        bull_thesis, bear_thesis = self._build_thesis(ignition, pressure, surge)
        
        # Get trade levels
        entry, stop, t1, t2 = self._get_trade_levels(ignition, pressure, surge, direction)
        
        # Calculate position size and R:R
        risk_reward = self._calculate_risk_reward(entry, stop, t1)
        max_position = self._calculate_max_position(combined_score, confidence)
        
        # Timing
        entry_window, time_sensitivity = self._get_timing(surge)
        
        return ProbabilityOutput(
            ticker=ticker,
            combined_score=combined_score,
            probability_win=probability_win,
            confidence=confidence,
            action=action,
            direction=direction,
            ignition_score=ignition_score,
            pressure_score=pressure_score,
            surge_score=surge_score,
            max_position_pct=max_position,
            risk_reward_ratio=risk_reward,
            entry_price=entry,
            stop_loss=stop,
            target_1=t1,
            target_2=t2,
            optimal_entry_window=entry_window,
            time_sensitivity=time_sensitivity,
            bull_thesis=bull_thesis,
            bear_thesis=bear_thesis,
            generated_at=datetime.utcnow(),
        )
    
    def generate_all(
        self,
        ignition_signals: List[Dict],
        pressure_signals: List[Dict],
        surge_signals: List[Dict],
    ) -> List[ProbabilityOutput]:
        """Generate probabilities for all tickers with signals."""
        
        # Collect all unique tickers
        tickers = set()
        for s in ignition_signals + pressure_signals + surge_signals:
            tickers.add(s.get("ticker"))
        
        outputs = []
        for ticker in tickers:
            output = self.generate(ticker, ignition_signals, pressure_signals, surge_signals)
            if output and output.combined_score >= 30:  # Minimum threshold
                outputs.append(output)
        
        # Sort by combined score
        outputs.sort(key=lambda x: x.combined_score, reverse=True)
        
        return outputs
    
    def _score_to_probability(self, score: float) -> float:
        """Convert 0-100 score to 0-1 probability."""
        # Logistic transformation centered at 50
        import math
        x = (score - 50) / 15  # Scale factor
        return 1 / (1 + math.exp(-x))
    
    def _calculate_confidence(self, signal_count: int, score: float) -> str:
        """Calculate confidence level."""
        if signal_count >= 3 and score >= 70:
            return "high"
        elif signal_count >= 2 and score >= 50:
            return "medium"
        else:
            return "low"
    
    def _determine_direction(
        self,
        ignition: Optional[Dict],
        pressure: Optional[Dict],
        surge: Optional[Dict],
    ) -> str:
        """Determine trade direction from signals."""
        long_votes = 0
        short_votes = 0
        
        # Ignition: check velocity sign
        if ignition:
            velocity = ignition.get("velocity", 0)
            if velocity > 0:
                long_votes += 1
            elif velocity < 0:
                short_votes += 1
        
        # Pressure: check pressure type
        if pressure:
            ptype = pressure.get("pressure_type", "")
            if ptype in ["call_wall", "gamma_squeeze"]:
                long_votes += 1
            elif ptype in ["put_wall"]:
                short_votes += 1
        
        # Surge: direct direction
        if surge:
            if surge.get("direction") == "long":
                long_votes += 1
            elif surge.get("direction") == "short":
                short_votes += 1
        
        if long_votes > short_votes:
            return "long"
        elif short_votes > long_votes:
            return "short"
        else:
            return "neutral"
    
    def _determine_action(
        self,
        score: float,
        direction: str,
        confidence: str,
    ) -> ActionType:
        """Determine recommended action."""
        if direction == "neutral":
            if score >= self.WATCH_THRESHOLD:
                return ActionType.WATCH
            return ActionType.AVOID
        
        if direction == "long":
            if score >= self.STRONG_THRESHOLD and confidence == "high":
                return ActionType.STRONG_LONG
            elif score >= self.NORMAL_THRESHOLD:
                return ActionType.LONG
            elif score >= self.WATCH_THRESHOLD:
                return ActionType.WATCH
            else:
                return ActionType.AVOID
        else:  # short
            if score >= self.STRONG_THRESHOLD and confidence == "high":
                return ActionType.STRONG_SHORT
            elif score >= self.NORMAL_THRESHOLD:
                return ActionType.SHORT
            elif score >= self.WATCH_THRESHOLD:
                return ActionType.WATCH
            else:
                return ActionType.AVOID
    
    def _build_thesis(
        self,
        ignition: Optional[Dict],
        pressure: Optional[Dict],
        surge: Optional[Dict],
    ) -> tuple:
        """Build bull and bear thesis from evidence."""
        bull_points = []
        bear_points = []
        
        if ignition:
            for e in ignition.get("evidence", []):
                if any(kw in e.lower() for kw in ["up", "above", "breaking", "volume"]):
                    bull_points.append(f"🔥 {e}")
                elif any(kw in e.lower() for kw in ["down", "below"]):
                    bear_points.append(f"🔻 {e}")
        
        if pressure:
            ptype = pressure.get("pressure_type", "")
            if "call" in ptype or "gamma" in ptype:
                bull_points.append(f"📊 Options: {ptype}")
            elif "put" in ptype:
                bear_points.append(f"📊 Options: {ptype}")
        
        if surge:
            direction = surge.get("direction", "")
            for e in surge.get("evidence", []):
                if direction == "long":
                    bull_points.append(f"⚡ {e}")
                else:
                    bear_points.append(f"⚡ {e}")
        
        return bull_points[:5], bear_points[:5]
    
    def _get_trade_levels(
        self,
        ignition: Optional[Dict],
        pressure: Optional[Dict],
        surge: Optional[Dict],
        direction: str,
    ) -> tuple:
        """Extract trade levels from signals."""
        # Prefer surge levels if available
        if surge:
            return (
                surge.get("entry_zone", 0),
                surge.get("stop_loss", 0),
                surge.get("target_1", 0),
                surge.get("target_2", 0),
            )
        
        # Use pressure levels
        if pressure:
            spot = pressure.get("spot_price", 0)
            resistance = pressure.get("resistance_strike", spot * 1.02)
            support = pressure.get("support_strike", spot * 0.98)
            
            if direction == "long":
                return (spot, support, resistance, resistance * 1.01)
            else:
                return (spot, resistance, support, support * 0.99)
        
        # Use ignition price
        if ignition:
            price = ignition.get("price", 0)
            if direction == "long":
                return (price, price * 0.98, price * 1.02, price * 1.04)
            else:
                return (price, price * 1.02, price * 0.98, price * 0.96)
        
        return (0, 0, 0, 0)
    
    def _calculate_risk_reward(self, entry: float, stop: float, target: float) -> float:
        """Calculate risk/reward ratio."""
        if entry == 0 or stop == 0 or target == 0:
            return 0
        
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        return reward / risk if risk > 0 else 0
    
    def _calculate_max_position(self, score: float, confidence: str) -> float:
        """Calculate max position size as % of account."""
        base_position = 2.0  # 2% base position
        
        # Adjust based on score
        if score >= 80:
            base_position *= 1.5
        elif score >= 60:
            base_position *= 1.0
        else:
            base_position *= 0.5
        
        # Adjust based on confidence
        confidence_multiplier = {"high": 1.0, "medium": 0.75, "low": 0.5}
        base_position *= confidence_multiplier.get(confidence, 0.5)
        
        return min(base_position, 5.0)  # Cap at 5%
    
    def _get_timing(self, surge: Optional[Dict]) -> tuple:
        """Get timing recommendations."""
        if surge:
            urgency = surge.get("urgency", "medium")
            minutes = surge.get("minutes_to_close", 60)
            
            if minutes < 30:
                window = "Immediate (< 30 min to close)"
            elif minutes < 60:
                window = "Near-term (30-60 min)"
            else:
                window = "Flexible (> 60 min)"
            
            return window, urgency
        
        return "Flexible", "medium"


# Global instance
probability_generator = ProbabilityGenerator()


def generate_probabilities(
    ignition_signals: List[Dict],
    pressure_signals: List[Dict],
    surge_signals: List[Dict],
) -> List[Dict]:
    """Generate all probabilities and return as dicts."""
    outputs = probability_generator.generate_all(
        ignition_signals,
        pressure_signals,
        surge_signals,
    )
    return [o.to_dict() for o in outputs]
