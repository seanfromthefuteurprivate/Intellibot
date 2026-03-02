#!/usr/bin/env python3
"""
Dead Man's Switch - Monitors trading activity to detect silent failures.

Unlike service health checks, this validates BUSINESS LOGIC is working.
If no trades or signals happen for N minutes during market hours, alert.

This catches failures that service monitoring misses:
- Service is running but not trading
- API connections work but no signals generated
- Database writes work but logic is broken
"""

import os
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class DeadMansSwitch:
    """
    Monitors trading activity and alerts if system is silent.

    Checks:
    - Last trade execution time
    - Last signal generation time
    - Market hours awareness
    """

    def __init__(
        self,
        db_path: str = None,
        silence_threshold_minutes: int = 30,
    ):
        """
        Initialize dead man's switch.

        Args:
            db_path: Path to SQLite database
            silence_threshold_minutes: How long to wait before alerting
        """
        self.db_path = db_path or os.getenv(
            "WSB_SNAKE_DB_PATH",
            "/root/wsb-snake/wsb_snake_data/wsb_snake.db"
        )
        self.silence_threshold = timedelta(minutes=silence_threshold_minutes)
        self.last_alert: Optional[datetime] = None

    def is_market_hours(self) -> bool:
        """
        Check if currently in market hours (9:30 AM - 4:00 PM ET, weekdays).

        Returns:
            True if market is open
        """
        try:
            import zoneinfo

            et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))

            # Check if weekday (Monday=0, Sunday=6)
            if et.weekday() >= 5:  # Saturday=5, Sunday=6
                return False

            market_open = et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = et.replace(hour=16, minute=0, second=0, microsecond=0)

            return market_open <= et <= market_close

        except ImportError:
            logger.warning("zoneinfo not available, assuming market hours")
            return True
        except Exception as e:
            logger.error(f"Failed to check market hours: {e}")
            # Assume market hours on error to avoid missing real issues
            return True

    def get_last_trade_time(self) -> Optional[datetime]:
        """
        Get timestamp of last executed trade from database.

        Returns:
            Datetime of last filled trade, or None if no trades
        """
        if not os.path.exists(self.db_path):
            logger.warning(f"Database not found: {self.db_path}")
            return None

        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            cursor = conn.cursor()

            # Query for last filled trade
            cursor.execute(
                """
                SELECT MAX(timestamp)
                FROM trades
                WHERE status = 'filled'
            """
            )

            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                # Parse timestamp (assuming ISO format)
                return datetime.fromisoformat(result[0])

            return None

        except sqlite3.OperationalError as e:
            logger.error(f"Database query failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to query last trade: {e}")
            return None

    def get_last_signal_time(self) -> Optional[datetime]:
        """
        Get timestamp of last trading signal from database.

        Returns:
            Datetime of last non-neutral signal, or None if no signals
        """
        if not os.path.exists(self.db_path):
            logger.warning(f"Database not found: {self.db_path}")
            return None

        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            cursor = conn.cursor()

            # Query for last signal (any table that has signals)
            # This is flexible - adapt to your schema
            cursor.execute(
                """
                SELECT MAX(timestamp)
                FROM signals
                WHERE signal_type != 'NEUTRAL'
            """
            )

            result = cursor.fetchone()
            conn.close()

            if result and result[0]:
                return datetime.fromisoformat(result[0])

            return None

        except sqlite3.OperationalError as e:
            # Table might not exist
            logger.debug(f"Signals table query failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to query last signal: {e}")
            return None

    def check(self) -> tuple[bool, str]:
        """
        Check if system is alive based on trading activity.

        Returns:
            (is_alive: bool, message: str)
        """
        # Don't check outside market hours
        if not self.is_market_hours():
            return True, "Outside market hours"

        now = datetime.now()

        # Check last trade
        last_trade = self.get_last_trade_time()
        if last_trade:
            silence_duration = now - last_trade
            if silence_duration > self.silence_threshold:
                # Alert at most once per hour to avoid spam
                if self.last_alert is None or (now - self.last_alert) > timedelta(
                    hours=1
                ):
                    self.last_alert = now
                    message = (
                        f"No trades in {silence_duration.total_seconds() / 60:.1f}m "
                        f"(last: {last_trade.strftime('%H:%M:%S')})"
                    )
                    return False, message

        # Check last signal
        last_signal = self.get_last_signal_time()
        if last_signal:
            silence_duration = now - last_signal
            if silence_duration > self.silence_threshold:
                if self.last_alert is None or (now - self.last_alert) > timedelta(
                    hours=1
                ):
                    self.last_alert = now
                    message = (
                        f"No signals in {silence_duration.total_seconds() / 60:.1f}m "
                        f"(last: {last_signal.strftime('%H:%M:%S')})"
                    )
                    return False, message

        # If we have no trade or signal data at all during market hours, that's suspicious
        if not last_trade and not last_signal:
            return False, "No trade or signal data found in database"

        return True, "Trading activity normal"

    def get_status(self) -> dict:
        """Get current status for monitoring."""
        last_trade = self.get_last_trade_time()
        last_signal = self.get_last_signal_time()

        return {
            "is_market_hours": self.is_market_hours(),
            "last_trade": last_trade.isoformat() if last_trade else None,
            "last_signal": last_signal.isoformat() if last_signal else None,
            "silence_threshold_minutes": self.silence_threshold.total_seconds() / 60,
            "database_path": self.db_path,
            "database_exists": os.path.exists(self.db_path),
        }


def main():
    """CLI for testing."""
    import sys
    import json

    if len(sys.argv) > 1 and sys.argv[1] == "status":
        dms = DeadMansSwitch()
        status = dms.get_status()
        print(json.dumps(status, indent=2))
    else:
        dms = DeadMansSwitch()
        is_alive, message = dms.check()

        print(f"Dead Man's Switch Check:")
        print(f"  Alive: {is_alive}")
        print(f"  Message: {message}")

        if not is_alive:
            print("\n⚠️ System appears to be silent!")
            sys.exit(1)
        else:
            print("\n✅ System is active")


if __name__ == "__main__":
    main()
