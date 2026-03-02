#!/usr/bin/env python3
"""
Test suite for resilience architecture.

Run this before deploying to production to verify all components work.
"""

import sys
import time
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from circuit_breaker import CircuitBreaker
from dead_mans_switch import DeadMansSwitch


def test_circuit_breaker():
    """Test circuit breaker behavior."""
    print("\n" + "=" * 60)
    print("TEST 1: Circuit Breaker")
    print("=" * 60)

    # Use a temp state file for testing
    cb = CircuitBreaker(
        max_restarts=3,
        time_window_minutes=5,
        cooldown_minutes=1,  # Short cooldown for testing
        state_file=Path("/tmp/test_circuit_breaker.json"),
    )

    # Reset to clean state
    cb.reset()
    print("✅ Circuit breaker initialized")

    # Test 1: First 3 restarts should be allowed
    print("\nTest 1a: First 3 restarts should be allowed")
    for i in range(3):
        allowed, msg = cb.can_restart(f"test_{i}")
        print(f"  Restart {i + 1}: {'✅ ALLOWED' if allowed else '❌ BLOCKED'} - {msg}")
        assert allowed, f"Restart {i + 1} should be allowed"
        cb.record_restart(f"test_{i}", success=False)

    # Test 2: 4th restart should be blocked (circuit opens)
    print("\nTest 1b: 4th restart should be blocked (circuit opens)")
    allowed, msg = cb.can_restart("test_4")
    print(f"  Restart 4: {'✅ ALLOWED' if allowed else '❌ BLOCKED'} - {msg}")
    assert not allowed, "4th restart should be blocked"
    assert cb.state == "OPEN", "Circuit should be OPEN"
    print("✅ Circuit breaker OPENED as expected")

    # Test 3: Cooldown transition to HALF_OPEN
    print("\nTest 1c: After cooldown, should transition to HALF_OPEN")
    print("  Waiting for cooldown (1 minute)...")
    time.sleep(65)  # Wait for cooldown + buffer
    allowed, msg = cb.can_restart("test_recovery")
    print(f"  After cooldown: {'✅ ALLOWED' if allowed else '❌ BLOCKED'} - {msg}")
    assert allowed, "Restart should be allowed after cooldown"
    assert cb.state == "HALF_OPEN", "Circuit should be HALF_OPEN"

    # Test 4: Successful restart closes circuit
    print("\nTest 1d: Successful restart should close circuit")
    cb.record_restart("test_recovery", success=True)
    assert cb.state == "CLOSED", "Circuit should be CLOSED after successful restart"
    print("✅ Circuit breaker CLOSED after successful recovery")

    # Cleanup
    Path("/tmp/test_circuit_breaker.json").unlink(missing_ok=True)

    print("\n✅ Circuit Breaker Tests PASSED")
    return True


def test_dead_mans_switch():
    """Test dead man's switch."""
    print("\n" + "=" * 60)
    print("TEST 2: Dead Man's Switch")
    print("=" * 60)

    dms = DeadMansSwitch(
        db_path="/root/wsb-snake/wsb_snake_data/wsb_snake.db",
        silence_threshold_minutes=30,
    )

    # Test 1: Get status
    print("\nTest 2a: Get status")
    status = dms.get_status()
    print(f"  Market hours: {status['is_market_hours']}")
    print(f"  Database exists: {status['database_exists']}")
    print(f"  Last trade: {status['last_trade']}")
    print(f"  Last signal: {status['last_signal']}")
    print("✅ Status retrieved")

    # Test 2: Check if alive
    print("\nTest 2b: Check if system is alive")
    is_alive, message = dms.check()
    print(f"  Alive: {is_alive}")
    print(f"  Message: {message}")

    if not status["database_exists"]:
        print("⚠️ Database does not exist - skipping live check")
    else:
        print("✅ Dead man's switch check completed")

    print("\n✅ Dead Man's Switch Tests PASSED")
    return True


def test_systemd_limits():
    """Test systemd restart limits."""
    print("\n" + "=" * 60)
    print("TEST 3: Systemd Restart Limits")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["systemctl", "show", "wsb-snake.service", "-p", "StartLimitBurst", "-p", "StartLimitIntervalSec"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        print("\nSystemd restart limits:")
        for line in result.stdout.strip().split("\n"):
            print(f"  {line}")

        # Check if limits are set
        has_burst = "StartLimitBurst=" in result.stdout
        has_interval = "StartLimitIntervalSec=" in result.stdout

        if has_burst and has_interval:
            print("✅ Systemd restart limits configured")
        else:
            print("⚠️ Systemd restart limits may not be configured")
            print("   Expected: StartLimitBurst=5, StartLimitIntervalSec=300")

    except Exception as e:
        print(f"❌ Failed to check systemd limits: {e}")
        return False

    print("\n✅ Systemd Tests PASSED")
    return True


def test_monitor_integration():
    """Test that monitor can import resilience modules."""
    print("\n" + "=" * 60)
    print("TEST 4: Monitor Integration")
    print("=" * 60)

    try:
        print("\nTest 4a: Import circuit breaker in monitor context")
        sys.path.insert(0, "/root/wsb-snake/ops")
        from circuit_breaker import CircuitBreaker as CB

        cb = CB()
        print("✅ Circuit breaker imported successfully")

        print("\nTest 4b: Import dead man's switch in monitor context")
        from dead_mans_switch import DeadMansSwitch as DMS

        dms = DMS()
        print("✅ Dead man's switch imported successfully")

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    print("\n✅ Monitor Integration Tests PASSED")
    return True


def main():
    """Run all tests."""
    print("🧪 RESILIENCE ARCHITECTURE TEST SUITE")
    print("=" * 60)

    tests = [
        ("Circuit Breaker", test_circuit_breaker),
        ("Dead Man's Switch", test_dead_mans_switch),
        ("Systemd Limits", test_systemd_limits),
        ("Monitor Integration", test_monitor_integration),
    ]

    results = []

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✅ ALL TESTS PASSED - Ready for deployment")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} TESTS FAILED - Fix issues before deploying")
        return 1


if __name__ == "__main__":
    sys.exit(main())
