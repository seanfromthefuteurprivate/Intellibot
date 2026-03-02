#!/usr/bin/env python3
"""
AGENT 2: AUTO-DEPLOYER — ops/deploy_agent.py
"I DEPLOY SAFELY. I VALIDATE FIRST. I ROLLBACK ON FAILURE."

Polls for new commits, validates Python syntax, deploys, monitors stability, auto-rollbacks.
"""

import os
import py_compile
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wsb_snake.notifications.telegram_bot import send_alert
from wsb_snake.utils.logger import get_logger

logger = get_logger("ops.deploy")

# Config
REPO_PATH = Path(__file__).parent.parent
SERVICE_NAME = "wsb-snake"
POLL_INTERVAL = 300  # 5 minutes
STABILITY_WINDOW = 120  # 2 minutes post-deploy check
MAX_CRASH_COUNT = 3  # Crashes within stability window to trigger rollback


def git_fetch() -> bool:
    """Fetch latest from remote."""
    try:
        result = subprocess.run(
            ["git", "fetch", "origin"],
            cwd=REPO_PATH,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Git fetch failed: {e}")
        return False


def get_current_commit() -> str:
    """Get current HEAD commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_PATH,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_remote_commit() -> str:
    """Get origin/main commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "origin/main"],
            cwd=REPO_PATH,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_changed_files(old_commit: str, new_commit: str) -> List[str]:
    """Get list of changed Python files between commits."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", old_commit, new_commit],
            cwd=REPO_PATH,
            capture_output=True,
            text=True,
            timeout=30,
        )
        files = result.stdout.strip().split("\n")
        return [f for f in files if f.endswith(".py") and f]
    except Exception as e:
        logger.error(f"Failed to get changed files: {e}")
        return []


def get_commit_message(commit: str) -> str:
    """Get commit message for a commit hash."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s", commit],
            cwd=REPO_PATH,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "Unknown"


def validate_python_syntax(files: List[str]) -> Tuple[bool, List[str]]:
    """Validate Python syntax for all changed files."""
    errors = []
    for filepath in files:
        full_path = REPO_PATH / filepath
        if not full_path.exists():
            continue  # File was deleted
        try:
            py_compile.compile(str(full_path), doraise=True)
        except py_compile.PyCompileError as e:
            errors.append(f"{filepath}: {e}")
    return len(errors) == 0, errors


def git_pull() -> bool:
    """Pull latest changes."""
    try:
        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=REPO_PATH,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Git pull failed: {e}")
        return False


def git_reset(commit: str) -> bool:
    """Reset to a specific commit (for rollback)."""
    try:
        result = subprocess.run(
            ["git", "reset", "--hard", commit],
            cwd=REPO_PATH,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Git reset failed: {e}")
        return False


def restart_service() -> bool:
    """Restart the main trading service."""
    try:
        result = subprocess.run(
            ["systemctl", "restart", SERVICE_NAME],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Service restart failed: {e}")
        return False


def check_service_status() -> Tuple[bool, str]:
    """Check if service is running and healthy."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", SERVICE_NAME],
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = result.stdout.strip()
        return status == "active", status
    except Exception as e:
        return False, str(e)


def get_recent_crashes(since_seconds: int = 120) -> int:
    """Count crashes in journal since timestamp."""
    try:
        result = subprocess.run(
            ["journalctl", "-u", SERVICE_NAME, "--no-pager", "-n", "100",
             "--since", f"{since_seconds}s ago"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        logs = result.stdout.lower()
        crash_indicators = ["traceback", "exception", "error", "crashed", "failed"]
        crash_count = sum(1 for indicator in crash_indicators if indicator in logs)
        return crash_count
    except Exception:
        return 0


def verify_stability(duration: int = 120) -> Tuple[bool, str]:
    """
    Monitor service for stability after deploy.
    Returns (stable, reason).
    """
    logger.info(f"Starting {duration}s stability check...")
    start_time = time.time()
    check_interval = 10
    crash_count = 0

    while time.time() - start_time < duration:
        # Check service status
        running, status = check_service_status()
        if not running:
            return False, f"Service not running: {status}"

        # Check for crashes in logs
        crashes = get_recent_crashes(since_seconds=check_interval + 5)
        if crashes > 0:
            crash_count += crashes
            if crash_count >= MAX_CRASH_COUNT:
                return False, f"Too many errors: {crash_count} crash indicators"

        time.sleep(check_interval)

    # Final status check
    running, status = check_service_status()
    if running:
        return True, "Service stable"
    return False, f"Service failed at end: {status}"


def deploy_new_version(old_commit: str, new_commit: str, changed_files: List[str]) -> bool:
    """
    Full deployment process:
    1. Pre-deploy validation
    2. Git pull
    3. Service restart
    4. Post-deploy verification
    5. Auto-rollback on failure
    """
    commit_msg = get_commit_message(new_commit)[:50]
    short_old = old_commit[:7]
    short_new = new_commit[:7]

    send_alert(
        f"🚀 DEPLOY STARTING\n"
        f"From: {short_old}\n"
        f"To: {short_new}\n"
        f"Msg: {commit_msg}\n"
        f"Changed: {len(changed_files)} Python files"
    )

    # STEP 1: Pre-deploy validation
    logger.info("STEP 1: Pre-deploy validation...")

    # First, temporarily fetch the new code to validate
    if not git_pull():
        send_alert(f"🔴 DEPLOY FAILED: Git pull failed")
        return False

    valid, errors = validate_python_syntax(changed_files)
    if not valid:
        error_summary = "\n".join(errors[:3])  # First 3 errors
        send_alert(f"🔴 DEPLOY ABORTED: Syntax errors\n{error_summary}")
        # Rollback the pull
        git_reset(old_commit)
        return False

    logger.info("STEP 1: Syntax validation passed")

    # STEP 2: Restart service
    logger.info("STEP 2: Restarting service...")
    if not restart_service():
        send_alert(f"🔴 DEPLOY FAILED: Service restart failed")
        git_reset(old_commit)
        restart_service()
        return False

    # Give service time to start
    time.sleep(5)

    # Check immediate start
    running, status = check_service_status()
    if not running:
        send_alert(f"🔴 DEPLOY FAILED: Service failed to start ({status})")
        logger.error("Service failed to start, initiating rollback...")
        git_reset(old_commit)
        restart_service()
        send_alert(f"↩️ ROLLBACK COMPLETE: Reverted to {short_old}")
        return False

    # STEP 3: Stability verification
    logger.info("STEP 3: Stability verification...")
    send_alert(f"⏳ DEPLOY: {STABILITY_WINDOW}s stability check started...")

    stable, reason = verify_stability(STABILITY_WINDOW)

    if not stable:
        send_alert(f"🔴 DEPLOY FAILED: {reason}\nInitiating rollback...")
        logger.error(f"Stability check failed: {reason}")
        git_reset(old_commit)
        restart_service()
        send_alert(f"↩️ ROLLBACK COMPLETE: Reverted to {short_old}")
        return False

    # STEP 4: Success
    send_alert(
        f"✅ DEPLOY SUCCESS\n"
        f"Commit: {short_new}\n"
        f"Msg: {commit_msg}\n"
        f"Stability: {STABILITY_WINDOW}s verified"
    )
    logger.info(f"Deploy complete: {short_old} → {short_new}")
    return True


def main():
    """Main deploy agent loop."""
    logger.info("🟢 DEPLOY AGENT STARTING")
    send_alert("🟢 DEPLOY AGENT ONLINE — watching for commits, auto-deploy enabled")

    last_known_commit = get_current_commit()
    logger.info(f"Current commit: {last_known_commit[:7]}")

    try:
        while True:
            # Fetch latest
            if not git_fetch():
                logger.warning("Git fetch failed, will retry...")
                time.sleep(POLL_INTERVAL)
                continue

            current_commit = get_current_commit()
            remote_commit = get_remote_commit()

            # Check for new commits
            if remote_commit and remote_commit != current_commit:
                changed_files = get_changed_files(current_commit, remote_commit)

                if changed_files:
                    logger.info(f"New commit detected: {remote_commit[:7]}")
                    logger.info(f"Changed files: {changed_files}")

                    success = deploy_new_version(
                        current_commit,
                        remote_commit,
                        changed_files
                    )

                    if success:
                        last_known_commit = remote_commit
                else:
                    # Non-Python changes, just pull silently
                    git_pull()
                    last_known_commit = get_current_commit()

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Deploy agent stopped by user")
    except Exception as e:
        logger.error(f"Deploy agent crashed: {e}")
        send_alert(f"🔴 DEPLOY AGENT CRASHED: {e}")
        raise


if __name__ == "__main__":
    main()
