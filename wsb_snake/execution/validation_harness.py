"""
Blind and Red-Team Validation Harness — Testing for Apex Governance.

Blind Mode:
- Hide outcome magnitude during live run
- Log governance decisions without PnL visibility
- Reveal outcomes only after session ends

Red Team Mode:
- Inject noise to induce premature exits
- Attempt to maintain RUNNER_LOCK during invalid structure
- Log all attempts and outcomes
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)

# Global toggles
BLIND_MODE_ENABLED = False
RED_TEAM_MODE_ENABLED = False


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_type: str
    passed: bool
    details: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class BlindValidation:
    """
    Blind and red-team validation for governance testing.
    
    Blind Mode:
    - Masks PnL values during live run
    - Records decisions without outcome visibility
    - Reveals outcomes only after session ends
    
    Red Team Mode:
    - Injects noise into prices
    - Attempts to trigger invalid state transitions
    - Tests edge cases and boundary conditions
    """
    
    def __init__(
        self,
        blind_mode: bool = False,
        red_team_mode: bool = False,
        noise_factor: float = 0.05,
    ):
        self.blind_mode = blind_mode or BLIND_MODE_ENABLED
        self.red_team_mode = red_team_mode or RED_TEAM_MODE_ENABLED
        self.noise_factor = noise_factor
        
        # Blind mode state
        self.masked_decisions: List[Dict[str, Any]] = []
        self.revealed_outcomes: List[Dict[str, Any]] = []
        
        # Red team state
        self.injection_attempts: List[Dict[str, Any]] = []
        self.validation_results: List[ValidationResult] = []
        
        logger.info(f"BlindValidation initialized (blind={self.blind_mode}, red_team={self.red_team_mode})")
    
    # ─────────────────────────────────────────────────────────────────
    # Blind Mode Methods
    # ─────────────────────────────────────────────────────────────────
    
    def mask_pnl(self, pnl: float) -> str:
        """
        Mask PnL value in blind mode.
        
        Args:
            pnl: Actual PnL percentage
        
        Returns:
            "MASKED" in blind mode, or formatted PnL otherwise
        """
        if self.blind_mode:
            return "MASKED"
        return f"{pnl:+.1f}%"
    
    def mask_price(self, price: float) -> str:
        """
        Mask price value in blind mode.
        
        Args:
            price: Actual price
        
        Returns:
            "MASKED" in blind mode, or formatted price otherwise
        """
        if self.blind_mode:
            return "MASKED"
        return f"${price:.2f}"
    
    def record_blind_decision(
        self,
        dedupe_key: str,
        decision_type: str,
        governance_state: str,
        actual_pnl: float,
        exit_permitted: bool,
    ) -> None:
        """
        Record a governance decision without revealing PnL.
        
        Called during blind mode to track decisions for later analysis.
        """
        if not self.blind_mode:
            return
        
        self.masked_decisions.append({
            "dedupe_key": dedupe_key,
            "decision_type": decision_type,
            "governance_state": governance_state,
            "actual_pnl": actual_pnl,  # Stored but not revealed
            "exit_permitted": exit_permitted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        logger.info(f"BLIND_DECISION | {decision_type} | state={governance_state} | exit_permitted={exit_permitted} | pnl=MASKED")
    
    def reveal_outcomes(self) -> List[Dict[str, Any]]:
        """
        Reveal all masked outcomes at session end.
        
        Returns:
            List of decisions with actual PnL revealed
        """
        if not self.masked_decisions:
            return []
        
        self.revealed_outcomes = []
        for decision in self.masked_decisions:
            revealed = decision.copy()
            revealed["pnl_revealed"] = f"{decision['actual_pnl']:+.1f}%"
            self.revealed_outcomes.append(revealed)
            logger.info(f"BLIND_REVEAL | {decision['decision_type']} | pnl={decision['actual_pnl']:+.1f}%")
        
        return self.revealed_outcomes
    
    def generate_blind_report(self) -> Dict[str, Any]:
        """Generate report of blind mode decisions."""
        if not self.revealed_outcomes:
            self.reveal_outcomes()
        
        correct_decisions = sum(
            1 for d in self.revealed_outcomes
            if self._evaluate_decision_correctness(d)
        )
        
        return {
            "total_decisions": len(self.revealed_outcomes),
            "correct_decisions": correct_decisions,
            "accuracy": correct_decisions / len(self.revealed_outcomes) if self.revealed_outcomes else 0,
            "decisions": self.revealed_outcomes,
        }
    
    def _evaluate_decision_correctness(self, decision: Dict[str, Any]) -> bool:
        """Evaluate if a decision was correct based on outcome."""
        pnl = decision.get("actual_pnl", 0)
        exit_permitted = decision.get("exit_permitted", True)
        state = decision.get("governance_state", "OBSERVE")
        
        # In RUNNER_LOCK, exit should be forbidden unless PnL collapsed
        if state == "RUNNER_LOCK":
            if exit_permitted and pnl > 0:
                return False  # Should not have permitted exit
            if not exit_permitted and pnl < -20:
                return False  # Should have permitted exit (structure break)
        
        return True
    
    # ─────────────────────────────────────────────────────────────────
    # Red Team Mode Methods
    # ─────────────────────────────────────────────────────────────────
    
    def inject_noise(self, price: float) -> float:
        """
        Inject noise into price (red team mode).
        
        Args:
            price: Original price
        
        Returns:
            Price with random noise added (if red team mode)
        """
        if not self.red_team_mode:
            return price
        
        noise = random.uniform(-self.noise_factor, self.noise_factor)
        noisy_price = price * (1 + noise)
        
        self.injection_attempts.append({
            "original_price": price,
            "noise_factor": noise,
            "noisy_price": noisy_price,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        logger.debug(f"RED_TEAM_NOISE | original=${price:.2f} | noise={noise:+.2%} | result=${noisy_price:.2f}")
        return noisy_price
    
    def attempt_invalid_exit(
        self,
        governance_state: str,
        current_pnl: float,
    ) -> Tuple[bool, str]:
        """
        Attempt to induce an invalid exit (red team test).
        
        Returns:
            (should_exit, reason) - what the system should do
        """
        if not self.red_team_mode:
            return False, "red_team_disabled"
        
        # Try to induce exit during RUNNER_LOCK
        if governance_state == "RUNNER_LOCK":
            # Fabricate a reason to exit
            fake_reason = random.choice([
                "HIGH_PROFIT_TEMPTATION",
                "ARTIFICIAL_DRAWDOWN",
                "FAKE_STRUCTURE_BREAK",
            ])
            
            self.injection_attempts.append({
                "type": "invalid_exit_attempt",
                "governance_state": governance_state,
                "current_pnl": current_pnl,
                "fake_reason": fake_reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            logger.warning(f"RED_TEAM_ATTACK | Attempting invalid exit | reason={fake_reason}")
            
            # The system SHOULD reject this
            return False, fake_reason
        
        return False, "no_attack"
    
    def attempt_runner_lock_during_invalid_structure(
        self,
        current_pnl: float,
        peak_pnl: float,
    ) -> bool:
        """
        Attempt to maintain RUNNER_LOCK when structure is invalid.
        
        Returns:
            True if attack was attempted
        """
        if not self.red_team_mode:
            return False
        
        drawdown = peak_pnl - current_pnl
        
        # Structure is invalid if drawdown > 20%
        if drawdown > 20:
            self.injection_attempts.append({
                "type": "invalid_runner_lock_attempt",
                "current_pnl": current_pnl,
                "peak_pnl": peak_pnl,
                "drawdown": drawdown,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            logger.warning(f"RED_TEAM_ATTACK | Attempting invalid RUNNER_LOCK | drawdown={drawdown:.1f}%")
            return True
        
        return False
    
    def log_validation_result(
        self,
        test_type: str,
        passed: bool,
        details: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Log a validation test result.
        
        Args:
            test_type: Type of test (e.g., "runner_lock_integrity", "exit_permission")
            passed: Whether the test passed
            details: Description of the result
            metadata: Additional test data
        
        Returns:
            ValidationResult object
        """
        result = ValidationResult(
            test_type=test_type,
            passed=passed,
            details=details,
            metadata=metadata or {},
        )
        
        self.validation_results.append(result)
        
        status = "PASSED" if passed else "FAILED"
        logger.info(f"VALIDATION | {test_type} | {status} | {details}")
        
        return result
    
    def generate_red_team_report(self) -> Dict[str, Any]:
        """Generate report of red team testing."""
        passed_tests = sum(1 for r in self.validation_results if r.passed)
        failed_tests = sum(1 for r in self.validation_results if not r.passed)
        
        return {
            "total_injection_attempts": len(self.injection_attempts),
            "total_validation_tests": len(self.validation_results),
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / len(self.validation_results) if self.validation_results else 0,
            "injection_attempts": self.injection_attempts,
            "validation_results": [
                {
                    "test_type": r.test_type,
                    "passed": r.passed,
                    "details": r.details,
                    "timestamp": r.timestamp,
                }
                for r in self.validation_results
            ],
        }
    
    # ─────────────────────────────────────────────────────────────────
    # Test Suite Methods
    # ─────────────────────────────────────────────────────────────────
    
    def run_governance_tests(self, governance) -> Dict[str, Any]:
        """
        Run a suite of governance tests.
        
        Args:
            governance: ApexRunnerGovernance instance
        
        Returns:
            Test results summary
        """
        results = []
        
        # Test 1: TP suppression in RUNNER_LOCK
        results.append(self._test_tp_suppression(governance))
        
        # Test 2: SL always permitted
        results.append(self._test_sl_always_permitted(governance))
        
        # Test 3: Structure break triggers RELEASE
        results.append(self._test_structure_break_release(governance))
        
        # Test 4: State transition irreversibility
        results.append(self._test_state_irreversibility(governance))
        
        passed = sum(1 for r in results if r.passed)
        return {
            "total_tests": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "results": [{"test": r.test_type, "passed": r.passed, "details": r.details} for r in results],
        }
    
    def _test_tp_suppression(self, governance) -> ValidationResult:
        """Test that TP exits are suppressed in RUNNER_LOCK."""
        # Register test position
        governance.register_position("TEST_TP_SUPPRESS", entry_price=1.0)
        
        # Force into RUNNER_LOCK
        pos = governance.positions.get("TEST_TP_SUPPRESS")
        if pos:
            from wsb_snake.execution.apex_governance import GovernanceState
            pos.state = GovernanceState.RUNNER_LOCK
            
            # Try TP exit
            permitted, reason = governance.is_exit_permitted("TEST_TP_SUPPRESS", "TP", 1.50)
            
            governance.unregister_position("TEST_TP_SUPPRESS")
            
            return self.log_validation_result(
                "TP_SUPPRESSION",
                passed=not permitted,
                details=f"TP exit {'correctly suppressed' if not permitted else 'incorrectly permitted'}",
            )
        
        return self.log_validation_result("TP_SUPPRESSION", passed=False, details="Failed to create test position")
    
    def _test_sl_always_permitted(self, governance) -> ValidationResult:
        """Test that SL exits are always permitted."""
        governance.register_position("TEST_SL_ALWAYS", entry_price=1.0)
        
        pos = governance.positions.get("TEST_SL_ALWAYS")
        if pos:
            from wsb_snake.execution.apex_governance import GovernanceState
            pos.state = GovernanceState.RUNNER_LOCK
            
            permitted, reason = governance.is_exit_permitted("TEST_SL_ALWAYS", "SL", 0.80)
            
            governance.unregister_position("TEST_SL_ALWAYS")
            
            return self.log_validation_result(
                "SL_ALWAYS_PERMITTED",
                passed=permitted,
                details=f"SL exit {'correctly permitted' if permitted else 'incorrectly suppressed'}",
            )
        
        return self.log_validation_result("SL_ALWAYS_PERMITTED", passed=False, details="Failed to create test position")
    
    def _test_structure_break_release(self, governance) -> ValidationResult:
        """Test that structure break triggers RELEASE."""
        governance.register_position("TEST_STRUCTURE", entry_price=1.0)
        
        pos = governance.positions.get("TEST_STRUCTURE")
        if pos:
            from wsb_snake.execution.apex_governance import GovernanceState
            pos.state = GovernanceState.RUNNER_LOCK
            pos.peak_price = 2.0
            pos.peak_pnl_pct = 100.0
            
            # Simulate major drawdown
            decision = governance.evaluate_position("TEST_STRUCTURE", current_price=1.20)
            
            passed = decision.state == GovernanceState.RELEASE
            
            governance.unregister_position("TEST_STRUCTURE")
            
            return self.log_validation_result(
                "STRUCTURE_BREAK_RELEASE",
                passed=passed,
                details=f"Structure break {'correctly triggered RELEASE' if passed else 'failed to trigger RELEASE'}",
            )
        
        return self.log_validation_result("STRUCTURE_BREAK_RELEASE", passed=False, details="Failed to create test position")
    
    def _test_state_irreversibility(self, governance) -> ValidationResult:
        """Test that state transitions are irreversible (forward only)."""
        governance.register_position("TEST_IRREVERSIBLE", entry_price=1.0)
        
        pos = governance.positions.get("TEST_IRREVERSIBLE")
        if pos:
            from wsb_snake.execution.apex_governance import GovernanceState
            from datetime import timedelta
            
            # Transition to CANDIDATE
            pos.entry_time = datetime.now(timezone.utc) - timedelta(minutes=10)
            governance.evaluate_position("TEST_IRREVERSIBLE", current_price=1.30)
            
            after_candidate = pos.state
            
            # Try to go back to OBSERVE (should fail)
            governance.evaluate_position("TEST_IRREVERSIBLE", current_price=1.05)
            after_low = pos.state
            
            # State should not have reverted to OBSERVE
            passed = after_low != GovernanceState.OBSERVE and after_candidate.value >= GovernanceState.CANDIDATE.value
            
            governance.unregister_position("TEST_IRREVERSIBLE")
            
            return self.log_validation_result(
                "STATE_IRREVERSIBILITY",
                passed=passed,
                details=f"State transitions {'correctly irreversible' if passed else 'incorrectly reversible'}",
            )
        
        return self.log_validation_result("STATE_IRREVERSIBILITY", passed=False, details="Failed to create test position")


# Singleton instance
_validation_harness: Optional[BlindValidation] = None


def get_validation_harness(blind_mode: bool = False, red_team_mode: bool = False) -> BlindValidation:
    """Get or create the validation harness."""
    global _validation_harness
    if _validation_harness is None:
        _validation_harness = BlindValidation(blind_mode=blind_mode, red_team_mode=red_team_mode)
    return _validation_harness
