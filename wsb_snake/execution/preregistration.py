"""
PreRegistration and Invariants — Axiom locking for Apex Governance.

Locks the following axioms BEFORE a session starts to prevent post-hoc bias:
A1) Profit magnitude NEVER justifies an exit.
A2) Only STRUCTURAL INVALIDATION justifies an exit.
A3) Take-profits are OBSERVATIONAL CHECKPOINTS, not exits.
A4) If continuation requires BELIEF, structure is invalid.
A5) Governance must be reversible only by structure, never by PnL.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)

# Global toggle
PREREGISTRATION_ENABLED = True

# Immutable axioms (A1-A5)
AXIOMS = {
    "A1": "Profit magnitude NEVER justifies an exit",
    "A2": "Only STRUCTURAL INVALIDATION justifies an exit",
    "A3": "Take-profits are OBSERVATIONAL CHECKPOINTS, not exits",
    "A4": "If continuation requires BELIEF, structure is invalid",
    "A5": "Governance must be reversible only by structure, never by PnL",
}

# State definitions
STATE_DEFINITIONS = {
    "OBSERVE": "Exit allowed; TPs visible (descriptive only)",
    "CANDIDATE": "Exit allowed; TPs non-binding; bias toward non-interference",
    "RUNNER_LOCK": "Exit FORBIDDEN unless structure breaks",
    "RELEASE": "Exit REQUIRED; overrides all other logic",
}

# Prohibited behaviors
PROHIBITED_BEHAVIORS = [
    "Early exit based on profit magnitude alone",
    "TP execution during RUNNER_LOCK (unless structure breaks)",
    "Manual override of RUNNER_LOCK state",
    "Retrospective tuning of axioms after session start",
    "Emotion-based exit decisions",
]

# Telemetry schema
TELEMETRY_SCHEMA = {
    "CPL_EVENT | STATE=OBSERVE": "Initial state, exit permitted",
    "CPL_EVENT | STATE_TRANSITION | X→Y": "State change with reason",
    "CPL_EVENT | RUNNER_LOCK_ENTERED": "Exit permission revoked",
    "CPL_EVENT | RUNNER_LOCK_HEARTBEAT": "Structure still intact",
    "CPL_EVENT | STRUCTURE_BREAK_DETECTED": "Release triggered",
    "CPL_EVENT | TP_CHECKPOINT | pnl=X%": "Observational only",
    "CPL_EVENT | TP_SUPPRESSED": "Exit denied by RUNNER_LOCK",
}


@dataclass
class SessionLock:
    """A locked session with frozen axioms."""
    session_id: str
    locked_at: str
    axioms: Dict[str, str]
    state_definitions: Dict[str, str]
    prohibited_behaviors: List[str]
    telemetry_schema: Dict[str, str]
    hash_signature: str = ""
    
    def __post_init__(self):
        """Compute hash signature after initialization."""
        if not self.hash_signature:
            self.hash_signature = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of locked content."""
        content = json.dumps({
            "axioms": self.axioms,
            "state_definitions": self.state_definitions,
            "prohibited_behaviors": self.prohibited_behaviors,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class PreRegistration:
    """
    Pre-registration system for locking axioms before a session.
    
    Once locked, axioms cannot be modified until session ends.
    This prevents post-hoc bias and ensures audit integrity.
    """
    
    def __init__(self, db_enabled: bool = True, telemetry_bus=None):
        self.enabled = PREREGISTRATION_ENABLED
        self.db_enabled = db_enabled
        self.telemetry = telemetry_bus
        self.current_session: Optional[SessionLock] = None
        self._db = None
        logger.info(f"PreRegistration initialized (enabled={self.enabled})")
    
    def _get_db(self):
        """Lazy load database connection."""
        if self._db is None:
            try:
                from wsb_snake.db.database import get_connection
                self._db = get_connection()
            except ImportError:
                self._db = None
        return self._db
    
    def lock_session(self, session_id: Optional[str] = None) -> SessionLock:
        """
        Lock a new session with frozen axioms.
        
        Args:
            session_id: Optional session ID (auto-generated if not provided)
        
        Returns:
            SessionLock object with frozen axioms
        """
        if not self.enabled:
            logger.warning("PreRegistration disabled; session not locked")
            return None
        
        if self.current_session is not None:
            logger.warning(f"Session already locked: {self.current_session.session_id}")
            return self.current_session
        
        if session_id is None:
            session_id = f"CPL_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        locked_at = datetime.now(timezone.utc).isoformat()
        
        session = SessionLock(
            session_id=session_id,
            locked_at=locked_at,
            axioms=AXIOMS.copy(),
            state_definitions=STATE_DEFINITIONS.copy(),
            prohibited_behaviors=PROHIBITED_BEHAVIORS.copy(),
            telemetry_schema=TELEMETRY_SCHEMA.copy(),
        )
        
        self.current_session = session
        
        # Save to database if enabled
        if self.db_enabled:
            self._save_to_db(session)
        
        # Emit telemetry
        if self.telemetry:
            self.telemetry.emit_preregistration_locked(
                session_id=session_id,
                axioms=list(AXIOMS.values()),
            )
        
        logger.info(f"PREREGISTRATION | SESSION_LOCKED | {session_id} | hash={session.hash_signature}")
        return session
    
    def verify_invariants(self) -> bool:
        """
        Verify that all invariants (axioms) are still intact.
        
        Returns:
            True if all axioms match the locked version
        """
        if self.current_session is None:
            logger.warning("No session locked; cannot verify invariants")
            return False
        
        current_hash = self.current_session._compute_hash()
        if current_hash != self.current_session.hash_signature:
            logger.error(f"INVARIANT_VIOLATION | Hash mismatch: {current_hash} != {self.current_session.hash_signature}")
            return False
        
        logger.info(f"INVARIANTS_VERIFIED | hash={current_hash}")
        return True
    
    def get_locked_axioms(self, session_id: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Get the locked axioms for a session.
        
        Args:
            session_id: Session ID to query (uses current if not provided)
        
        Returns:
            Dict of axioms or None if not found
        """
        if session_id is None and self.current_session:
            return self.current_session.axioms
        
        if session_id is None:
            return None
        
        # Try to load from database
        if self.db_enabled:
            return self._load_from_db(session_id)
        
        return None
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get current session info."""
        if self.current_session is None:
            return None
        
        return {
            "session_id": self.current_session.session_id,
            "locked_at": self.current_session.locked_at,
            "hash_signature": self.current_session.hash_signature,
            "axiom_count": len(self.current_session.axioms),
            "state_count": len(self.current_session.state_definitions),
            "prohibited_count": len(self.current_session.prohibited_behaviors),
        }
    
    def unlock_session(self) -> bool:
        """
        Unlock the current session (end of trading session).
        
        Returns:
            True if session was unlocked
        """
        if self.current_session is None:
            return False
        
        session_id = self.current_session.session_id
        self.current_session = None
        logger.info(f"PREREGISTRATION | SESSION_UNLOCKED | {session_id}")
        return True
    
    def assert_axiom(self, axiom_key: str, context: str = "") -> bool:
        """
        Assert that an axiom is being followed.
        
        Args:
            axiom_key: A1, A2, A3, A4, or A5
            context: Additional context for logging
        
        Returns:
            True if axiom exists and is locked
        """
        if self.current_session is None:
            logger.warning(f"Cannot assert {axiom_key}; no session locked")
            return False
        
        if axiom_key not in self.current_session.axioms:
            logger.error(f"Unknown axiom: {axiom_key}")
            return False
        
        axiom_text = self.current_session.axioms[axiom_key]
        logger.info(f"AXIOM_ASSERTED | {axiom_key} | {axiom_text} | context={context}")
        return True
    
    def _save_to_db(self, session: SessionLock) -> None:
        """Save session lock to database."""
        db = self._get_db()
        if db is None:
            return
        
        try:
            cursor = db.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO preregistration_locks (
                    session_id, locked_at, axioms_json, state_definitions_json,
                    prohibited_behaviors_json, telemetry_schema_json, hash_signature
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.locked_at,
                json.dumps(session.axioms),
                json.dumps(session.state_definitions),
                json.dumps(session.prohibited_behaviors),
                json.dumps(session.telemetry_schema),
                session.hash_signature,
            ))
            db.commit()
        except Exception as e:
            logger.warning(f"DB save error: {e}")
    
    def _load_from_db(self, session_id: str) -> Optional[Dict[str, str]]:
        """Load session from database."""
        db = self._get_db()
        if db is None:
            return None
        
        try:
            cursor = db.cursor()
            cursor.execute(
                "SELECT axioms_json FROM preregistration_locks WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        except Exception as e:
            logger.warning(f"DB load error: {e}")
        
        return None


# Singleton instance
_preregistration: Optional[PreRegistration] = None


def get_preregistration() -> PreRegistration:
    """Get or create the singleton preregistration instance."""
    global _preregistration
    if _preregistration is None:
        _preregistration = PreRegistration()
    return _preregistration
