"""
Execution layer for Jobs Day (NFP) — Convexity Proof Layer (CPL).
Emits atomic paper calls (BUY/SELL) for 0DTE options with deduplication and Telegram broadcast.
"""

from wsb_snake.execution.call_schema import JobsDayCall
from wsb_snake.execution.jobs_day_cpl import JobsDayCPL

__all__ = ["JobsDayCall", "JobsDayCPL"]
