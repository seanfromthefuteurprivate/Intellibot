"""
WSB Snake State Machine - The Predator Pattern

This implements the LURK → COILED → RATTLE → STRIKE → CONSTRICT → VENOM states
that make the engine behave like a predator, not a reactive alert system.

The state machine prevents premature alerts and ensures signals fire only
when multiple conditions align.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import threading

from wsb_snake.utils.logger import log
from wsb_snake.utils.session_regime import get_session_info, SessionType


class EngineState(Enum):
    """The 6 states of the Rattlesnake engine."""
    LURK = "LURK"           # Passive monitoring, building heat maps
    COILED = "COILED"       # Conditions building, sensitivity raised
    RATTLE = "RATTLE"       # Warning signals, publishing "watch" events
    STRIKE = "STRIKE"       # Attack mode, triggering alerts/paper trades
    CONSTRICT = "CONSTRICT" # Post-strike management
    VENOM = "VENOM"         # End-of-day postmortem


@dataclass
class TickerState:
    """State tracking for a single ticker."""
    ticker: str
    state: EngineState = EngineState.LURK
    state_entered_at: datetime = field(default_factory=datetime.now)
    
    # Coiled conditions
    compression_score: float = 0.0  # Low ATR into a level
    time_alignment: bool = False     # Approaching power hour
    concentration_score: float = 0.0 # Options near spot (when available)
    
    # Rattle conditions
    ignition_detected: bool = False
    headline_catalyst: bool = False
    volume_surge: bool = False
    
    # Strike conditions
    structure_break: bool = False
    direction_confirmed: str = "neutral"
    probability_score: float = 0.0
    
    # Constrict tracking
    entry_price: float = 0.0
    current_price: float = 0.0
    invalidation_level: float = 0.0
    target_level: float = 0.0
    time_in_trade: int = 0  # minutes
    
    # History
    state_history: List[tuple] = field(default_factory=list)
    
    def transition_to(self, new_state: EngineState, reason: str = ""):
        """Transition to a new state."""
        if new_state != self.state:
            self.state_history.append((
                self.state.value, 
                new_state.value, 
                datetime.now().isoformat(),
                reason
            ))
            old_state = self.state
            self.state = new_state
            self.state_entered_at = datetime.now()
            log.info(f"🐍 {self.ticker} STATE: {old_state.value} → {new_state.value} | {reason}")
    
    def time_in_state(self) -> int:
        """Minutes spent in current state."""
        return int((datetime.now() - self.state_entered_at).total_seconds() / 60)
    
    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "state": self.state.value,
            "state_entered_at": self.state_entered_at.isoformat(),
            "time_in_state_minutes": self.time_in_state(),
            "compression_score": self.compression_score,
            "ignition_detected": self.ignition_detected,
            "structure_break": self.structure_break,
            "direction_confirmed": self.direction_confirmed,
            "probability_score": self.probability_score,
        }


class RattlesnakeStateMachine:
    """
    The Rattlesnake State Machine.
    
    Controls when the engine is passive (LURK) vs aggressive (STRIKE).
    Prevents false signals by requiring state escalation.
    """
    
    # Thresholds for state transitions
    COILED_COMPRESSION_THRESHOLD = 0.6   # ATR compression ratio
    COILED_TIME_WINDOW_START = 13        # 1 PM ET - approaching power hour
    
    RATTLE_IGNITION_SCORE = 30           # Minimum ignition score
    RATTLE_VOLUME_RATIO = 2.0            # Volume surge threshold
    
    STRIKE_PROBABILITY_THRESHOLD = 0.55  # P(hit target) minimum
    STRIKE_COMBINED_SCORE = 50           # Minimum combined score
    
    CONSTRICT_TIME_STOP = 30             # Minutes before force exit
    
    def __init__(self):
        self._ticker_states: Dict[str, TickerState] = {}
        self._lock = threading.Lock()
        self._daily_strikes: List[Dict] = []
        self._daily_outcomes: List[Dict] = []
    
    def get_state(self, ticker: str) -> TickerState:
        """Get or create state for a ticker."""
        with self._lock:
            if ticker not in self._ticker_states:
                self._ticker_states[ticker] = TickerState(ticker=ticker)
            return self._ticker_states[ticker]
    
    def get_all_states(self) -> Dict[str, Dict]:
        """Get all ticker states as dicts."""
        with self._lock:
            return {t: s.to_dict() for t, s in self._ticker_states.items()}
    
    def reset_daily(self):
        """Reset all states for new trading day."""
        with self._lock:
            self._ticker_states.clear()
            self._daily_strikes.clear()
            self._daily_outcomes.clear()
            log.info("🐍 State machine reset for new day")
    
    def update_ticker(
        self,
        ticker: str,
        ignition_signal: Optional[Dict] = None,
        pressure_signal: Optional[Dict] = None,
        surge_signal: Optional[Dict] = None,
        probability_output: Optional[Dict] = None,
        current_price: float = 0.0,
    ) -> TickerState:
        """
        Update ticker state based on incoming signals.
        This is the core state machine logic.
        """
        state = self.get_state(ticker)
        session = get_session_info()
        
        # Update current price
        if current_price > 0:
            state.current_price = current_price
        
        # Extract scores
        ignition_score = ignition_signal.get("score", 0) if ignition_signal else 0
        pressure_score = pressure_signal.get("score", 0) if pressure_signal else 0
        surge_score = surge_signal.get("score", 0) if surge_signal else 0
        probability_score = probability_output.get("probability_win", 0) if probability_output else 0
        combined_score = probability_output.get("combined_score", 0) if probability_output else 0
        
        # Check for news catalyst
        has_news = False
        if ignition_signal and ignition_signal.get("evidence"):
            has_news = any("news" in str(e).lower() for e in ignition_signal.get("evidence", []))
        
        # Update conditions
        state.ignition_detected = ignition_score >= self.RATTLE_IGNITION_SCORE
        state.headline_catalyst = has_news
        state.probability_score = probability_score
        
        # Check volume surge
        if ignition_signal:
            evidence = ignition_signal.get("evidence", [])
            for e in evidence:
                if "volume" in str(e).lower() and "x" in str(e).lower():
                    state.volume_surge = True
                    break
        
        # Check structure break
        if pressure_signal:
            pressure_type = pressure_signal.get("pressure_type", "")
            if "breakout" in pressure_type.lower() or "momentum" in pressure_type.lower():
                state.structure_break = True
                state.direction_confirmed = pressure_signal.get("direction", "neutral")
        
        # Determine time alignment (approaching power hour)
        current_hour = datetime.now().hour
        state.time_alignment = current_hour >= self.COILED_TIME_WINDOW_START
        
        # STATE MACHINE LOGIC
        if state.state == EngineState.LURK:
            # Check transition to COILED
            should_coil = (
                state.time_alignment and
                session.get("session") in ["morning", "lunch", "power_hour_early"]
            )
            if should_coil:
                state.transition_to(EngineState.COILED, "Time window approaching + session favorable")
        
        elif state.state == EngineState.COILED:
            # Check transition to RATTLE
            rattle_conditions = sum([
                state.ignition_detected,
                state.headline_catalyst,
                state.volume_surge,
            ])
            
            if rattle_conditions >= 2:
                state.transition_to(EngineState.RATTLE, f"Multiple ignition signals ({rattle_conditions}/3)")
            elif state.time_in_state() > 60:
                # Timeout - go back to lurk if nothing happens
                state.transition_to(EngineState.LURK, "Coiled timeout - no ignition")
        
        elif state.state == EngineState.RATTLE:
            # Check transition to STRIKE
            strike_conditions = (
                state.structure_break and
                state.direction_confirmed != "neutral" and
                probability_score >= self.STRIKE_PROBABILITY_THRESHOLD and
                combined_score >= self.STRIKE_COMBINED_SCORE
            )
            
            if strike_conditions:
                state.transition_to(
                    EngineState.STRIKE, 
                    f"Structure break + direction {state.direction_confirmed} + P={probability_score:.2f}"
                )
                # Record strike
                self._daily_strikes.append({
                    "ticker": ticker,
                    "time": datetime.now().isoformat(),
                    "direction": state.direction_confirmed,
                    "probability": probability_score,
                    "price": current_price,
                })
            elif state.time_in_state() > 30:
                # Rattle timeout - step back
                state.transition_to(EngineState.COILED, "Rattle timeout - no confirmation")
        
        elif state.state == EngineState.STRIKE:
            # Immediately transition to CONSTRICT after strike
            if state.entry_price == 0:
                state.entry_price = current_price
            state.transition_to(EngineState.CONSTRICT, "Strike executed - managing position")
        
        elif state.state == EngineState.CONSTRICT:
            state.time_in_trade += 1
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # Time stop
            if state.time_in_trade >= self.CONSTRICT_TIME_STOP:
                should_exit = True
                exit_reason = f"Time stop ({self.CONSTRICT_TIME_STOP}min)"
            
            # Invalidation check (VWAP loss, structure break)
            if state.invalidation_level > 0:
                if state.direction_confirmed == "long" and current_price < state.invalidation_level:
                    should_exit = True
                    exit_reason = f"Invalidation level breached ({state.invalidation_level})"
                elif state.direction_confirmed == "short" and current_price > state.invalidation_level:
                    should_exit = True
                    exit_reason = f"Invalidation level breached ({state.invalidation_level})"
            
            # Target hit
            if state.target_level > 0:
                if state.direction_confirmed == "long" and current_price >= state.target_level:
                    should_exit = True
                    exit_reason = f"Target hit ({state.target_level})"
                elif state.direction_confirmed == "short" and current_price <= state.target_level:
                    should_exit = True
                    exit_reason = f"Target hit ({state.target_level})"
            
            if should_exit:
                # Record outcome
                pnl_pct = 0
                if state.entry_price > 0:
                    if state.direction_confirmed == "long":
                        pnl_pct = ((current_price - state.entry_price) / state.entry_price) * 100
                    else:
                        pnl_pct = ((state.entry_price - current_price) / state.entry_price) * 100
                
                self._daily_outcomes.append({
                    "ticker": ticker,
                    "exit_time": datetime.now().isoformat(),
                    "exit_reason": exit_reason,
                    "entry_price": state.entry_price,
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "direction": state.direction_confirmed,
                })
                
                # Reset state
                state.entry_price = 0
                state.invalidation_level = 0
                state.target_level = 0
                state.time_in_trade = 0
                state.structure_break = False
                state.transition_to(EngineState.LURK, exit_reason)
        
        # VENOM state is handled by end-of-day report
        
        return state
    
    def should_alert(self, ticker: str) -> bool:
        """Check if ticker is in STRIKE state (should alert)."""
        state = self.get_state(ticker)
        return state.state == EngineState.STRIKE
    
    def get_watch_tickers(self) -> List[str]:
        """Get tickers in RATTLE state (watch list)."""
        with self._lock:
            return [
                t for t, s in self._ticker_states.items() 
                if s.state == EngineState.RATTLE
            ]
    
    def get_active_tickers(self) -> List[str]:
        """Get tickers in STRIKE or CONSTRICT state."""
        with self._lock:
            return [
                t for t, s in self._ticker_states.items() 
                if s.state in [EngineState.STRIKE, EngineState.CONSTRICT]
            ]
    
    def generate_venom_report(self) -> Dict:
        """Generate end-of-day postmortem report."""
        total_strikes = len(self._daily_strikes)
        total_outcomes = len(self._daily_outcomes)
        
        wins = [o for o in self._daily_outcomes if o.get("pnl_pct", 0) > 0]
        losses = [o for o in self._daily_outcomes if o.get("pnl_pct", 0) <= 0]
        
        win_rate = len(wins) / total_outcomes if total_outcomes > 0 else 0
        avg_win = sum(w["pnl_pct"] for w in wins) / len(wins) if wins else 0
        avg_loss = sum(l["pnl_pct"] for l in losses) / len(losses) if losses else 0
        
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_strikes": total_strikes,
            "total_outcomes": total_outcomes,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "strikes": self._daily_strikes,
            "outcomes": self._daily_outcomes,
            "state_summary": {
                t: s.state.value for t, s in self._ticker_states.items()
            }
        }


# Global instance
state_machine = RattlesnakeStateMachine()


def update_state(
    ticker: str,
    ignition_signal: Optional[Dict] = None,
    pressure_signal: Optional[Dict] = None,
    surge_signal: Optional[Dict] = None,
    probability_output: Optional[Dict] = None,
    current_price: float = 0.0,
) -> Dict:
    """Update ticker state and return current state dict."""
    state = state_machine.update_ticker(
        ticker,
        ignition_signal,
        pressure_signal,
        surge_signal,
        probability_output,
        current_price,
    )
    return state.to_dict()


def should_strike(ticker: str) -> bool:
    """Check if we should fire an alert for this ticker."""
    return state_machine.should_alert(ticker)


def get_all_states() -> Dict:
    """Get all ticker states."""
    return state_machine.get_all_states()


def get_venom_report() -> Dict:
    """Get end-of-day report."""
    return state_machine.generate_venom_report()
