"""
Apex Runner Governance Layer — Exit Permission Only

Implements an IRREVERSIBLE state machine governing EXIT PERMISSION.
Does NOT modify entry detection, liquidity rules, diversity logic, or risk bounds.

IMMUTABLE AXIOMS:
A1) Profit magnitude NEVER justifies an exit.
A2) Only STRUCTURAL INVALIDATION justifies an exit.
A3) Take-profits are OBSERVATIONAL CHECKPOINTS, not exits.
A4) If continuation requires BELIEF, structure is invalid.
A5) Governance must be reversible only by structure, never by PnL.

VOLATILITY REGIMES:
1. SHOCK_FADE — TP ladder allowed, no runners
2. SHOCK_RANGE_EXPAND — 4x-6x zone, runners enabled
3. SUSTAINED_TREND — 10x+ zone, abandon fixed targets
4. CHAOS_WHIPSAW — Wait for re-stabilization
5. LATE_GAMMA_SQUEEZE — Time-aware runners
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)

# Global toggle — set False to disable all governance
GOVERNANCE_ENABLED = True


class GovernanceState(Enum):
    """Irreversible state machine for exit permission."""
    OBSERVE = 0       # Exit allowed; TPs visible (descriptive)
    CANDIDATE = 1     # Exit allowed; TPs non-binding; bias toward non-interference
    RUNNER_LOCK = 2   # Exit FORBIDDEN unless structure breaks
    RELEASE = 3       # Exit REQUIRED; overrides all other logic


class VolatilityRegime(Enum):
    """NFP event volatility regimes."""
    SHOCK_FADE = "SHOCK_FADE"                    # 1.5x-3x max
    SHOCK_RANGE_EXPAND = "SHOCK_RANGE_EXPAND"    # 4x-6x zone
    SUSTAINED_TREND = "SUSTAINED_TREND"          # 10x+ zone
    CHAOS_WHIPSAW = "CHAOS_WHIPSAW"              # Wait for re-stabilization
    LATE_GAMMA_SQUEEZE = "LATE_GAMMA_SQUEEZE"    # 5x-12x late day


@dataclass
class PositionState:
    """Tracks governance state for a single position."""
    dedupe_key: str
    state: GovernanceState = GovernanceState.OBSERVE
    regime: VolatilityRegime = VolatilityRegime.SHOCK_FADE
    entry_price: float = 0.0
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    peak_price: float = 0.0
    peak_pnl_pct: float = 0.0
    last_price: float = 0.0
    state_history: List[Tuple[GovernanceState, datetime, str]] = field(default_factory=list)
    tp_checkpoints_logged: List[float] = field(default_factory=list)
    structure_checks: List[Dict[str, Any]] = field(default_factory=list)
    runner_lock_entered_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None


@dataclass
class GovernanceDecision:
    """Result of governance evaluation."""
    exit_permitted: bool
    state: GovernanceState
    regime: VolatilityRegime
    reason: str
    tp_suppressed: bool = False
    structure_intact: bool = True
    telemetry_event: Optional[str] = None
    telemetry_note: Optional[str] = None


class ApexRunnerGovernance:
    """
    Exit permission governance layer.
    
    STATE 0 — OBSERVE: Exit allowed; TPs visible
    STATE 1 — CANDIDATE: Exit allowed; TPs non-binding
    STATE 2 — RUNNER_LOCK: Exit FORBIDDEN unless structure breaks
    STATE 3 — RELEASE: Exit REQUIRED
    """
    
    # Thresholds for state transitions
    CANDIDATE_THRESHOLD_PCT = 25.0      # Enter CANDIDATE at +25%
    RUNNER_LOCK_THRESHOLD_PCT = 50.0    # Enter RUNNER_LOCK at +50%
    PERSISTENCE_MINUTES = 5             # Time to confirm persistence
    
    # Structure invalidation thresholds
    DRAWDOWN_FROM_PEAK_PCT = 20.0       # Structure invalid if drops 20% from peak
    TIME_DECAY_DOMINANCE_MINUTES = 20   # Time decay overtakes after 20 min in range
    
    # Heartbeat rate limiting
    HEARTBEAT_INTERVAL_SECONDS = 60
    
    def __init__(self, enabled: bool = True, telemetry_bus=None, untruncated_tails: bool = False):
        self.enabled = enabled and GOVERNANCE_ENABLED
        self.positions: Dict[str, PositionState] = {}
        self.telemetry = telemetry_bus
        self.untruncated_tails = untruncated_tails
        logger.info(f"ApexRunnerGovernance initialized (enabled={self.enabled}, untruncated_tails={untruncated_tails})")
    
    def register_position(self, dedupe_key: str, entry_price: float, entry_time: Optional[datetime] = None) -> None:
        """Register a new position for governance tracking."""
        if not self.enabled:
            return
        
        pos = PositionState(
            dedupe_key=dedupe_key,
            entry_price=entry_price,
            entry_time=entry_time or datetime.now(timezone.utc),
            peak_price=entry_price,
            last_price=entry_price,
        )
        pos.state_history.append((GovernanceState.OBSERVE, pos.entry_time, "Position opened"))
        self.positions[dedupe_key] = pos

        self._emit_telemetry("STATE=OBSERVE", "Position registered; volatility displacement detected. Structure fragile.", dedupe_key=dedupe_key, pos=pos)
        logger.info(f"GOVERNANCE: Position registered {dedupe_key} @ \${entry_price:.2f}")
    
    def unregister_position(self, dedupe_key: str) -> None:
        """Remove position from governance tracking."""
        if dedupe_key in self.positions:
            del self.positions[dedupe_key]
    
    def evaluate_position(self, dedupe_key: str, current_price: float, current_iv: Optional[float] = None) -> GovernanceDecision:
        """Evaluate position and return governance decision."""
        if not self.enabled:
            return GovernanceDecision(exit_permitted=True, state=GovernanceState.OBSERVE, regime=VolatilityRegime.SHOCK_FADE, reason="Governance disabled")
        
        pos = self.positions.get(dedupe_key)
        if not pos:
            return GovernanceDecision(exit_permitted=True, state=GovernanceState.OBSERVE, regime=VolatilityRegime.SHOCK_FADE, reason="Position not tracked")
        
        # Update price tracking
        pos.last_price = current_price
        if current_price > pos.peak_price:
            pos.peak_price = current_price
            pos.peak_pnl_pct = self._calc_pnl_pct(pos.entry_price, current_price)
        
        pnl_pct = self._calc_pnl_pct(pos.entry_price, current_price)
        elapsed_minutes = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 60
        
        # Check for structural invalidation FIRST
        structure_intact, invalidation_reason = self._check_structure(pos, current_price)
        
        if not structure_intact and pos.state == GovernanceState.RUNNER_LOCK:
            self._transition_state(pos, GovernanceState.RELEASE, invalidation_reason)
            return GovernanceDecision(
                exit_permitted=True,
                state=GovernanceState.RELEASE,
                regime=pos.regime,
                reason=f"STRUCTURE_BREAK: {invalidation_reason}",
                structure_intact=False,
                telemetry_event="STRUCTURE_BREAK_DETECTED",
                telemetry_note=f"Continuation coherence collapsed; release executed | {invalidation_reason}",
            )
        
        # State machine progression
        if pos.state == GovernanceState.OBSERVE:
            decision = self._evaluate_observe(pos, pnl_pct, elapsed_minutes)
        elif pos.state == GovernanceState.CANDIDATE:
            decision = self._evaluate_candidate(pos, pnl_pct, elapsed_minutes, structure_intact)
        elif pos.state == GovernanceState.RUNNER_LOCK:
            decision = self._evaluate_runner_lock(pos, pnl_pct, structure_intact)
        else:
            decision = GovernanceDecision(exit_permitted=True, state=GovernanceState.RELEASE, regime=pos.regime, reason="Exit required; structure invalid")
        
        self._maybe_log_tp_checkpoint(pos, pnl_pct, decision)
        if pos.state == GovernanceState.RUNNER_LOCK:
            self._maybe_emit_heartbeat(pos)
        
        return decision
    
    def is_exit_permitted(self, dedupe_key: str, exit_reason: str, current_price: float, untruncated_tails: bool = False) -> Tuple[bool, str]:
        """Check if exit is permitted for a position. When untruncated_tails=True, SL/TIME only in OBSERVE/CANDIDATE (intelligent SL)."""
        if not self.enabled:
            return True, "Governance disabled"
        
        pos = self.positions.get(dedupe_key)
        if not pos:
            return True, "Position not tracked"
        
        decision = self.evaluate_position(dedupe_key, current_price)
        use_intelligent = untruncated_tails or self.untruncated_tails

        # SL: in untruncated mode permit only in OBSERVE/CANDIDATE (never cut a runner with SL)
        if exit_reason == "SL":
            if use_intelligent and pos.state == GovernanceState.RUNNER_LOCK:
                return False, "RUNNER_LOCK; SL not permitted — exit only on structure break"
            if use_intelligent and pos.state == GovernanceState.RELEASE:
                return True, "RELEASE; exit required"
            return True, "Stop loss permitted (structural protection)"
        
        # TIME: in untruncated mode do not permit in RUNNER_LOCK (runners run until structure breaks)
        if exit_reason == "TIME":
            if use_intelligent and pos.state == GovernanceState.RUNNER_LOCK:
                return False, "RUNNER_LOCK; TIME not permitted — exit only on structure break"
            return True, "Time expiry permitted"
        
        # TP exits governed by state machine
        if exit_reason == "TP":
            if pos.state == GovernanceState.RUNNER_LOCK:
                self._emit_telemetry("TP_SUPPRESSED", "Profit magnitude does not justify exit; structure intact", dedupe_key=dedupe_key, pos=pos, pnl_pct=self._calc_pnl_pct(pos.entry_price, current_price))
                return False, "RUNNER_LOCK active; TP suppressed (A1: profit never justifies exit)"
            elif pos.state == GovernanceState.RELEASE:
                return True, "RELEASE state; exit required"
            else:
                return True, f"Exit permitted in {pos.state.name} state"
        
        return decision.exit_permitted, decision.reason
    
    def force_release(self, dedupe_key: str, reason: str) -> None:
        """Force a position into RELEASE state."""
        pos = self.positions.get(dedupe_key)
        if pos and pos.state != GovernanceState.RELEASE:
            self._transition_state(pos, GovernanceState.RELEASE, reason)
    
    def get_state(self, dedupe_key: str) -> Optional[GovernanceState]:
        """Get current governance state for a position."""
        pos = self.positions.get(dedupe_key)
        return pos.state if pos else None
    
    def _evaluate_observe(self, pos: PositionState, pnl_pct: float, elapsed_minutes: float) -> GovernanceDecision:
        """Evaluate position in OBSERVE state."""
        if pnl_pct >= self.CANDIDATE_THRESHOLD_PCT and elapsed_minutes >= self.PERSISTENCE_MINUTES:
            self._transition_state(pos, GovernanceState.CANDIDATE, f"Expansion persists: +{pnl_pct:.1f}% after {elapsed_minutes:.0f} min")
            pos.regime = VolatilityRegime.SHOCK_RANGE_EXPAND
            return GovernanceDecision(
                exit_permitted=True,
                state=GovernanceState.CANDIDATE,
                regime=pos.regime,
                reason="Transition to CANDIDATE; persistence abnormal",
                telemetry_event="STATE_TRANSITION | OBSERVE->CANDIDATE",
                telemetry_note=f"Continuation persistence abnormal vs baseline; pnl={pnl_pct:.1f}%",
            )
        return GovernanceDecision(exit_permitted=True, state=GovernanceState.OBSERVE, regime=VolatilityRegime.SHOCK_FADE, reason="OBSERVE state; volatility displacement detected")
    
    def _evaluate_candidate(self, pos: PositionState, pnl_pct: float, elapsed_minutes: float, structure_intact: bool) -> GovernanceDecision:
        """Evaluate position in CANDIDATE state."""
        if pnl_pct >= self.RUNNER_LOCK_THRESHOLD_PCT and structure_intact:
            self._transition_state(pos, GovernanceState.RUNNER_LOCK, f"Coherence confirmed: +{pnl_pct:.1f}% with intact structure")
            pos.regime = VolatilityRegime.SUSTAINED_TREND
            pos.runner_lock_entered_at = datetime.now(timezone.utc)
            if self.telemetry and hasattr(self.telemetry, "emit_runner_lock_alert"):
                self.telemetry.emit_runner_lock_alert(
                    pos.dedupe_key,
                    pnl_pct,
                    entry_ref_price=pos.entry_price,
                    current_ref_price=pos.last_price,
                    peak_ref_price=pos.peak_price,
                )
            return GovernanceDecision(
                exit_permitted=False,
                state=GovernanceState.RUNNER_LOCK,
                regime=pos.regime,
                reason="RUNNER_LOCK entered; exit permission revoked",
                tp_suppressed=True,
                telemetry_event="RUNNER_LOCK_ENTERED",
                telemetry_note=f"Exit permission revoked; non-interference enforced; pnl={pnl_pct:.1f}%",
            )
        return GovernanceDecision(exit_permitted=True, state=GovernanceState.CANDIDATE, regime=pos.regime, reason="CANDIDATE state; TPs non-binding; bias toward hold")
    
    def _evaluate_runner_lock(self, pos: PositionState, pnl_pct: float, structure_intact: bool) -> GovernanceDecision:
        """Evaluate position in RUNNER_LOCK state."""
        if not structure_intact:
            self._transition_state(pos, GovernanceState.RELEASE, "Structure collapsed")
            return GovernanceDecision(
                exit_permitted=True,
                state=GovernanceState.RELEASE,
                regime=pos.regime,
                reason="RELEASE: structure invalid; exit required",
                structure_intact=False,
                telemetry_event="STRUCTURE_BREAK_DETECTED",
                telemetry_note="Continuation coherence collapsed; release executed",
            )
        return GovernanceDecision(exit_permitted=False, state=GovernanceState.RUNNER_LOCK, regime=pos.regime, reason="RUNNER_LOCK active; structure intact; exit forbidden", tp_suppressed=True, structure_intact=True)
    
    def _check_structure(self, pos: PositionState, current_price: float) -> Tuple[bool, Optional[str]]:
        """Check if structural coherence is intact."""
        pnl_pct = self._calc_pnl_pct(pos.entry_price, current_price)
        drawdown_from_peak = pos.peak_pnl_pct - pnl_pct if pos.peak_pnl_pct > 0 else 0
        
        if drawdown_from_peak >= self.DRAWDOWN_FROM_PEAK_PCT:
            return False, f"Drawdown {drawdown_from_peak:.1f}% from peak; dominance lost"
        
        if pos.peak_pnl_pct >= 25 and pnl_pct <= 5:
            return False, f"Price collapsed to entry; continuation failed"
        
        if pos.state == GovernanceState.RUNNER_LOCK and pos.runner_lock_entered_at:
            lock_duration = (datetime.now(timezone.utc) - pos.runner_lock_entered_at).total_seconds() / 60
            if lock_duration >= self.TIME_DECAY_DOMINANCE_MINUTES and current_price <= pos.peak_price * 0.95:
                return False, f"Time decay dominance; {lock_duration:.0f} min without new highs"
        
        if pos.state in (GovernanceState.CANDIDATE, GovernanceState.RUNNER_LOCK) and pnl_pct <= 0:
            return False, "PnL negative in advanced state; structure invalid"
        
        return True, None
    
    def _transition_state(self, pos: PositionState, new_state: GovernanceState, reason: str) -> None:
        """Transition to a new state."""
        old_state = pos.state
        pos.state = new_state
        pos.state_history.append((new_state, datetime.now(timezone.utc), reason))
        event = f"STATE_TRANSITION | {old_state.name}->{new_state.name}"
        self._emit_telemetry(event, reason, dedupe_key=pos.dedupe_key, pos=pos)
        logger.info(f"GOVERNANCE: {pos.dedupe_key} | {event} | {reason}")
    
    def _maybe_log_tp_checkpoint(self, pos: PositionState, pnl_pct: float, decision: GovernanceDecision) -> None:
        """Log TP checkpoints."""
        checkpoints = [25, 50, 75, 100, 150, 200, 300, 400, 500, 1000]
        for cp in checkpoints:
            if pnl_pct >= cp and cp not in pos.tp_checkpoints_logged:
                pos.tp_checkpoints_logged.append(cp)
                if pos.state == GovernanceState.RUNNER_LOCK:
                    self._emit_telemetry(f"TP_CHECKPOINT | +{cp}% (SUPPRESSED)", "Checkpoint observed; exit still forbidden; structure intact", dedupe_key=pos.dedupe_key, pos=pos)
                else:
                    self._emit_telemetry(f"TP_CHECKPOINT | +{cp}%", f"Checkpoint reached in {pos.state.name} state", dedupe_key=pos.dedupe_key, pos=pos)
    
    def _maybe_emit_heartbeat(self, pos: PositionState) -> None:
        """Emit rate-limited heartbeat during RUNNER_LOCK."""
        now = datetime.now(timezone.utc)
        if pos.last_heartbeat is None or (now - pos.last_heartbeat).total_seconds() >= self.HEARTBEAT_INTERVAL_SECONDS:
            pos.last_heartbeat = now
            pnl_pct = self._calc_pnl_pct(pos.entry_price, pos.last_price)
            self._emit_telemetry("RUNNER_LOCK_HEARTBEAT", f"Structure intact; exit still forbidden; pnl={pnl_pct:.1f}%", dedupe_key=pos.dedupe_key, pos=pos)
    
    def _audit_kwargs_from_pos(self, pos: PositionState) -> Dict[str, Any]:
        """Build audit snapshot kwargs from position for full Telegram format."""
        pnl = self._calc_pnl_pct(pos.entry_price, pos.last_price)
        exit_perm = "FORBIDDEN" if pos.state == GovernanceState.RUNNER_LOCK else (
            "REQUIRED" if pos.state == GovernanceState.RELEASE else "ALLOWED"
        )
        vol = "expanding" if pos.state == GovernanceState.RUNNER_LOCK else (
            "decaying" if pos.state == GovernanceState.RELEASE else "displacing"
        )
        dom = "lost" if pos.state == GovernanceState.RELEASE else "present"
        cont = "self-reinforcing" if pos.state == GovernanceState.RUNNER_LOCK else (
            "inert" if pos.state == GovernanceState.RELEASE else "fragile"
        )
        tp25 = "suppressed" if pos.state == GovernanceState.RUNNER_LOCK else ("observed" if pnl >= 25 else "not reached")
        tp50 = "suppressed" if pos.state == GovernanceState.RUNNER_LOCK else ("observed" if pnl >= 50 else "not reached")
        tp100 = "suppressed" if pos.state == GovernanceState.RUNNER_LOCK else ("observed" if pnl >= 100 else "not reached")
        entry_ts = pos.entry_time.strftime("%Y-%m-%d %H:%M:%S ET") if pos.entry_time else None
        return {
            "entry_ref_price": pos.entry_price,
            "current_ref_price": pos.last_price,
            "peak_ref_price": pos.peak_price,
            "expansion_pct": pnl,
            "current_state": pos.state.name,
            "exit_permission": exit_perm,
            "volatility_state": vol,
            "dominance": dom,
            "continuation_quality": cont,
            "tp25_status": tp25,
            "tp50_status": tp50,
            "tp100_status": tp100,
            "pnl_pct": pnl,
            "entry_ref_time": entry_ts,
        }

    def _emit_telemetry(
        self,
        event_type: str,
        note: str,
        dedupe_key: Optional[str] = None,
        pos: Optional[PositionState] = None,
        **kwargs,
    ) -> None:
        """Emit telemetry event to Telegram (with full audit context when pos provided)."""
        if self.telemetry:
            if pos is not None:
                kwargs = {**self._audit_kwargs_from_pos(pos), **kwargs}
            self.telemetry.emit(event_type, note, dedupe_key=dedupe_key, **kwargs)
        else:
            logger.info(f"CPL_EVENT | {event_type} | NOTE={note}")
    
    @staticmethod
    def _calc_pnl_pct(entry_price: float, current_price: float) -> float:
        """Calculate PnL percentage."""
        if entry_price <= 0:
            return 0.0
        return ((current_price - entry_price) / entry_price) * 100
