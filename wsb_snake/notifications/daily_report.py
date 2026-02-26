"""
Daily 4:15 PM ET report — win rate, avg R, top/worst setups, regime, 2X/4X/20X event counts.
Sent to Telegram for full 10/10 target alignment.

ENHANCED by Agent 5 (COMMS OFFICER) to include:
- Total P&L, win/loss count, win rate
- Each trade detail with entry/exit prices
- Regime summary
- Signals generated vs executed count
- Portfolio value
"""

from datetime import datetime
from typing import Optional

from wsb_snake.db.database import get_daily_stats_for_report
from wsb_snake.notifications.telegram_bot import send_alert
from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


def _et_today() -> str:
    try:
        import pytz
        et = datetime.now(pytz.timezone("America/New_York"))
    except Exception:
        et = datetime.utcnow()
    return et.strftime("%Y-%m-%d")


def format_daily_report(stats: dict) -> str:
    """
    Format the end-of-day report for Telegram.
    Includes: win rate, avg R, total R, best/worst setups, 2X/4X/20X event counts, regime note.
    """
    date_str = stats.get("date", _et_today())
    trades = stats.get("trades", 0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    win_rate_pct = (stats.get("win_rate", 0) or 0) * 100
    avg_r = stats.get("avg_r", 0) or 0
    total_r = stats.get("total_r", 0) or 0
    best_ticker = stats.get("best_ticker") or "—"
    best_r = stats.get("best_r")
    worst_ticker = stats.get("worst_ticker") or "—"
    worst_r = stats.get("worst_r")
    t2 = stats.get("tier_2x", 0)
    t4 = stats.get("tier_4x", 0)
    t20 = stats.get("tier_20x", 0)

    lines = [
        "══════════════════════════════════════",
        "📊 *DAILY REPORT* — 4:15 PM ET",
        f"📅 *DATE* {date_str}",
        "══════════════════════════════════════",
        "",
        "*PERFORMANCE*",
        f"• Trades: {trades} | Wins: {wins} | Losses: {losses}",
        f"• Win rate: {win_rate_pct:.1f}%",
        f"• Avg R: {avg_r:+.2f}R | Total R: {total_r:+.2f}R",
        "",
        "*BEST / WORST*",
        f"• Best: {best_ticker}" + (f" ({best_r:+.2f}R)" if best_r is not None else ""),
        f"• Worst: {worst_ticker}" + (f" ({worst_r:+.2f}R)" if worst_r is not None else ""),
        "",
        "*EVENT TIERS (2X / 4X / 20X)*",
        f"• 2X events: {t2} | 4X events: {t4} | 20X events: {t20}",
        "",
        "_System looks for and capitalizes on 2x, 4x, 20x moves._",
        "══════════════════════════════════════",
    ]
    return "\n".join(lines)


def send_daily_report(date_str: Optional[str] = None) -> bool:
    """
    Build and send the daily report to Telegram.
    Call at 4:15 PM ET (e.g. from main scheduler or cron).

    NOTE: This is the BASIC report. For ENHANCED report with trade details,
    regime summary, and portfolio value, use:
        from wsb_snake.notifications.enhanced_comms import send_enhanced_eod_report
        send_enhanced_eod_report()
    """
    try:
        # Try enhanced report first
        try:
            from wsb_snake.notifications.enhanced_comms import send_enhanced_eod_report
            ok = send_enhanced_eod_report(date_str)
            if ok:
                logger.info("Enhanced daily report sent to Telegram")
                return ok
        except Exception as e:
            logger.warning(f"Enhanced report failed, falling back to basic: {e}")

        # Fallback to basic report
        stats = get_daily_stats_for_report(date_str)
        msg = format_daily_report(stats)
        ok = send_alert(msg)
        if ok:
            logger.info("Daily report sent to Telegram")
        return ok
    except Exception as e:
        logger.exception("Failed to send daily report: %s", e)
        return False
