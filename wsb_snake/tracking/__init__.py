"""
Signal tracking and monitoring module.

Provides comprehensive signal lifecycle tracking for reconciliation.

Exports:
- signal_tracker module functions for tracking signals in the database
- signal_monitor singleton for real-time monitoring
"""

from .signal_tracker import (
    init_signal_tracking_tables,
    track_new_signal,
    record_alert_sent,
    update_signal_status,
    mark_target_hit,
    mark_stop_hit,
    get_open_signals,
    get_signal_by_id,
    get_signal_alerts,
    get_signals_by_date,
    get_signal_stats,
)

from .signal_monitor import signal_monitor

__all__ = [
    "init_signal_tracking_tables",
    "track_new_signal",
    "record_alert_sent",
    "update_signal_status",
    "mark_target_hit",
    "mark_stop_hit",
    "get_open_signals",
    "get_signal_by_id",
    "get_signal_alerts",
    "get_signals_by_date",
    "get_signal_stats",
    "signal_monitor",
]
