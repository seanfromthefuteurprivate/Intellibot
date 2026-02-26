"""
PID Lock - Prevents duplicate process execution.

CRITICAL: Every run_*.py script should use this to prevent duplicates.
"""

import os
import sys
import fcntl
import atexit
from pathlib import Path
from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)

# Default lock directory
LOCK_DIR = Path(os.getenv("WSB_SNAKE_PATH", "/home/ubuntu/wsb-snake")) / "wsb_snake_data" / "locks"


class PIDLock:
    """
    File-based PID lock to prevent duplicate processes.

    Usage:
        lock = PIDLock("wsb-snake-main")
        if not lock.acquire():
            print("Another instance is running")
            sys.exit(1)
        # ... run your code ...
        # Lock is auto-released on exit
    """

    def __init__(self, name: str, lock_dir: Path = None):
        """
        Initialize PID lock.

        Args:
            name: Unique name for this process type (e.g., "wsb-snake-main", "cpl-scanner")
            lock_dir: Directory for lock files (default: wsb_snake_data/locks)
        """
        self.name = name
        self.lock_dir = lock_dir or LOCK_DIR
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.lock_dir / f"{name}.pid"
        self._fd = None
        self._locked = False

    def acquire(self, exit_on_fail: bool = False) -> bool:
        """
        Acquire the lock.

        Args:
            exit_on_fail: If True, exit the process if lock cannot be acquired

        Returns:
            True if lock acquired, False if another process holds it
        """
        try:
            self._fd = open(self.lock_file, 'w')
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write our PID
            self._fd.write(str(os.getpid()))
            self._fd.flush()
            self._locked = True

            # Auto-release on exit
            atexit.register(self.release)

            logger.info(f"PID lock acquired: {self.name} (PID: {os.getpid()})")
            return True

        except (IOError, OSError) as e:
            # Lock held by another process
            existing_pid = self._get_existing_pid()
            logger.warning(f"PID lock failed: {self.name} - another instance running (PID: {existing_pid})")

            if exit_on_fail:
                logger.error(f"Exiting: {self.name} already running")
                sys.exit(1)

            return False

    def release(self):
        """Release the lock."""
        if self._fd and self._locked:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
                self._fd.close()
                self.lock_file.unlink(missing_ok=True)
                logger.info(f"PID lock released: {self.name}")
            except Exception as e:
                logger.debug(f"Error releasing lock: {e}")
            finally:
                self._locked = False
                self._fd = None

    def _get_existing_pid(self) -> int:
        """Get PID of the process holding the lock."""
        try:
            if self.lock_file.exists():
                return int(self.lock_file.read_text().strip())
        except:
            pass
        return -1

    def is_locked(self) -> bool:
        """Check if lock is currently held (by any process)."""
        try:
            fd = open(self.lock_file, 'w')
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            fd.close()
            return False  # We could acquire it, so not locked
        except (IOError, OSError):
            return True  # Locked by another process

    def __enter__(self):
        """Context manager entry."""
        self.acquire(exit_on_fail=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


def acquire_lock(name: str, exit_on_fail: bool = True) -> PIDLock:
    """
    Convenience function to acquire a PID lock.

    Args:
        name: Lock name (e.g., "wsb-snake-main")
        exit_on_fail: Exit if lock cannot be acquired (default True)

    Returns:
        PIDLock instance (or exits if exit_on_fail=True and lock held)
    """
    lock = PIDLock(name)
    lock.acquire(exit_on_fail=exit_on_fail)
    return lock


def check_running(name: str) -> tuple:
    """
    Check if a process is already running.

    Args:
        name: Lock name to check

    Returns:
        (is_running: bool, pid: int or None)
    """
    lock = PIDLock(name)
    if lock.is_locked():
        return True, lock._get_existing_pid()
    return False, None
