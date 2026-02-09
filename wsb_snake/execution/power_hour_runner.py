"""
POWER HOUR RUNNER - Aggressive institutional-grade trading for market close

Uses ALL available systems:
- APEX Conviction Engine (multi-signal fusion)
- Technical Analysis (RSI, MACD, SMA, EMA)
- Candlestick Patterns (36 patterns)
- Order Flow (sweeps, blocks, institutional)
- Probability Generator
- Pattern Memory

Power Hour Mode (3:00-4:00 PM ET):
- 30-second scan intervals
- Aggressive position sizing
- Quick exits (+20%/-10%)
- Volume spike detection
- Only trades conviction > 70%
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from wsb_snake.utils.logger import get_logger
from wsb_snake.execution.apex_conviction_engine import apex_engine, ApexVerdict
from wsb_snake.collectors.polygon_options import polygon_options
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.notifications.telegram_bot import send_alert
from wsb_snake.trading.alpaca_executor import alpaca_executor
from wsb_snake.trading.risk_governor import TradingEngine

logger = get_logger(__name__)

# Power hour watchlist - high liquidity, high volatility
POWER_HOUR_WATCHLIST = [
    "SPY", "QQQ", "IWM",      # Index ETFs (most liquid)
    "NVDA", "TSLA", "AMD",    # High-vol tech
    "META", "AAPL", "AMZN",   # Mega caps
    "GLD", "SLV",             # Metals
]

# ========== MAX MODE SETTINGS - AGGRESSIVE LAST HOUR ==========
# Minimum conviction to trade (lowered for more opportunities)
MIN_CONVICTION = 68  # Institutional grade: Higher threshold for consistency

# Power hour timing (ET)
POWER_HOUR_START = 15  # 3 PM
POWER_HOUR_END = 16    # 4 PM

# Scan interval - FAST scanning for MAX MODE
SCAN_INTERVAL_SECONDS = 10  # MAX MODE: Scan every 10 seconds!

# Track executed trades to avoid duplicates
_executed_this_session: set = set()


def _now_et():
    """Get current time in ET."""
    try:
        import pytz
        return datetime.now(pytz.timezone("America/New_York"))
    except Exception:
        return datetime.utcnow()


def _is_power_hour() -> bool:
    """Check if we're in power hour."""
    now = _now_et()
    return POWER_HOUR_START <= now.hour < POWER_HOUR_END


def _is_market_open() -> bool:
    """Check if market is open."""
    now = _now_et()
    if now.weekday() >= 5:  # Weekend
        return False
    if now.hour < 9 or (now.hour == 9 and now.minute < 30):
        return False
    if now.hour >= 16:
        return False
    return True


def _get_spot_price(ticker: str) -> Optional[float]:
    """Get current spot price."""
    try:
        # PRIMARY: Use polygon_enhanced snapshot (more reliable)
        if polygon_enhanced:
            snap = polygon_enhanced.get_snapshot(ticker)
            if snap and snap.get("price"):
                return float(snap["price"])
        # FALLBACK: Try polygon_options quote
        quote = polygon_options.get_quote(ticker) if polygon_options else None
        if quote and quote.get("price"):
            return float(quote["price"])
    except Exception as e:
        logger.debug(f"Price fetch failed {ticker}: {e}")
    return None


def _get_atm_option(ticker: str, spot: float, side: str, expiry_date: str) -> Optional[Dict[str, Any]]:
    """Get ATM or near-ATM option for quick execution."""
    try:
        chain = polygon_options.get_chain_for_expiration(ticker, spot, expiry_date, strike_range=5)
        if not chain:
            return None

        contracts = chain.get("calls" if side == "CALL" else "puts", [])
        if not contracts:
            return None

        # Find ATM (closest to spot)
        atm = min(contracts, key=lambda c: abs(c.get("strike", 0) - spot))

        # Verify liquidity
        bid = atm.get("bid", 0)
        ask = atm.get("ask", 0)
        if bid <= 0 or ask <= 0:
            return None

        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid if mid > 0 else 1

        # Reject wide spreads
        if spread_pct > 0.15:
            logger.info(f"SPREAD_REJECT: {ticker} {side} spread {spread_pct:.1%}")
            return None

        return atm

    except Exception as e:
        logger.debug(f"Option fetch failed {ticker}: {e}")
        return None


def _format_verdict_alert(verdict: ApexVerdict, option: Dict[str, Any]) -> str:
    """Format Telegram alert for APEX verdict."""
    signals_summary = []
    for s in verdict.signals:
        if s.score > 68:
            signals_summary.append(f"  • {s.source}: {s.score:.0f} ({s.direction})")

    strike = option.get("strike", 0)
    ask = option.get("ask", 0)
    symbol = option.get("symbol", "")

    side = "CALL" if verdict.action == "BUY_CALLS" else "PUT"

    return f"""🎯 **APEX CONVICTION TRADE**

**{verdict.ticker}** {side} ${strike:.0f}
Conviction: **{verdict.conviction_score:.0f}%** ({verdict.direction})
Time Sensitivity: {verdict.time_sensitivity}

**Entry:** ${ask:.2f}
**Target:** +{verdict.target_pct*100:.0f}%
**Stop:** -{verdict.stop_pct*100:.0f}%
**Size Mult:** {verdict.position_size_multiplier:.1f}x

**Signal Breakdown:**
{chr(10).join(signals_summary[:5])}

**Reasons:**
{chr(10).join(['• ' + r for r in verdict.reasons[:3]])}

Option: `{symbol}`
"""


def _execute_apex_trade(verdict: ApexVerdict, option: Dict[str, Any], expiry_date: str) -> bool:
    """Execute trade on Alpaca based on APEX verdict."""
    global _executed_this_session

    ticker = verdict.ticker
    strike = option.get("strike", 0)
    ask = option.get("ask", 0)
    symbol = option.get("symbol", "")

    # Dedupe key
    dedupe_key = f"{ticker}|{strike}|{verdict.action}|{expiry_date}"
    if dedupe_key in _executed_this_session:
        logger.info(f"DEDUPE: Already traded {dedupe_key}")
        return False

    try:
        # Direction and option type
        direction = "long"  # We buy options
        opt_type = "call" if verdict.action == "BUY_CALLS" else "put"

        # Calculate targets
        target_price = ask * (1 + verdict.target_pct)
        stop_loss = ask * (1 - verdict.stop_pct)

        # Strip O: prefix if present
        clean_symbol = symbol.replace("O:", "") if symbol.startswith("O:") else symbol

        # Execute via Alpaca
        alpaca_pos = alpaca_executor.execute_scalp_entry(
            underlying=ticker,
            direction=direction,
            entry_price=ask,
            target_price=target_price,
            stop_loss=stop_loss,
            confidence=verdict.conviction_score,
            pattern=f"APEX_{verdict.direction}",
            engine=TradingEngine.SCALPER,
            strike_override=strike,
            option_symbol_override=clean_symbol,
            option_type_override=opt_type,
        )

        if alpaca_pos:
            _executed_this_session.add(dedupe_key)
            logger.info(f"APEX EXECUTED: {ticker} {verdict.action} @ ${ask:.2f} -> {alpaca_pos.option_symbol}")
            send_alert(f"✅ **APEX EXECUTED**\n{ticker} {opt_type.upper()} ${strike:.0f}\nConviction: {verdict.conviction_score:.0f}%\nOption: {alpaca_pos.option_symbol}")
            return True
        else:
            logger.warning(f"APEX SKIPPED: {ticker} (Alpaca rejected)")
            return False

    except Exception as e:
        logger.error(f"APEX EXECUTION ERROR: {e}")
        send_alert(f"❌ APEX ERROR: {ticker} - {str(e)[:100]}")
        return False


def scan_and_execute(watchlist: Optional[List[str]] = None, dry_run: bool = False) -> List[ApexVerdict]:
    """
    Scan watchlist with APEX conviction engine and execute high-conviction trades.
    """
    watchlist = watchlist or POWER_HOUR_WATCHLIST
    today = _now_et().strftime("%Y-%m-%d")
    executed_verdicts: List[ApexVerdict] = []

    for ticker in watchlist:
        try:
            # Get spot price
            spot = _get_spot_price(ticker)
            if not spot:
                logger.debug(f"No spot for {ticker}")
                continue

            # Run APEX analysis
            verdict = apex_engine.analyze(ticker, spot)

            # Log all analyses
            logger.info(f"APEX {ticker}: {verdict.conviction_score:.0f}% {verdict.direction} -> {verdict.action}")

            # Skip low conviction
            if verdict.conviction_score < MIN_CONVICTION:
                continue

            # Skip neutral
            if verdict.action == "NO_TRADE":
                continue

            # Get option for this trade
            side = "CALL" if verdict.action == "BUY_CALLS" else "PUT"
            option = _get_atm_option(ticker, spot, side, today)
            if not option:
                logger.info(f"No valid option for {ticker} {side}")
                continue

            # Send alert
            alert_msg = _format_verdict_alert(verdict, option)
            send_alert(alert_msg)
            logger.info(f"APEX ALERT: {ticker} {verdict.action} conviction={verdict.conviction_score:.0f}%")

            # Execute if not dry run
            if not dry_run:
                success = _execute_apex_trade(verdict, option, today)
                if success:
                    executed_verdicts.append(verdict)
            else:
                logger.info(f"DRY RUN: Would execute {ticker} {verdict.action}")
                executed_verdicts.append(verdict)

        except Exception as e:
            logger.error(f"APEX scan error {ticker}: {e}")
            continue

    return executed_verdicts


def run_power_hour(dry_run: bool = False, max_trades: int = 10):
    """
    Run aggressive power hour trading until market close.
    """
    global _executed_this_session
    _executed_this_session.clear()

    logger.info("=" * 60)
    logger.info("APEX POWER HOUR RUNNER - INSTITUTIONAL MODE")
    logger.info(f"Watchlist: {', '.join(POWER_HOUR_WATCHLIST)}")
    logger.info(f"Min Conviction: {MIN_CONVICTION}%")
    logger.info(f"Scan Interval: {SCAN_INTERVAL_SECONDS}s")
    logger.info(f"Max Trades: {max_trades}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info("=" * 60)

    send_alert(f"""🚀 **APEX POWER HOUR ACTIVATED**

Mode: {'DRY RUN' if dry_run else 'LIVE EXECUTION'}
Watchlist: {len(POWER_HOUR_WATCHLIST)} tickers
Min Conviction: {MIN_CONVICTION}%
Scan Interval: {SCAN_INTERVAL_SECONDS}s

Using ALL signals:
• Technical (RSI/MACD/SMA)
• Candlestick (36 patterns)
• Order Flow (sweeps/blocks)
• Probability Generator
• Pattern Memory

**Aggressive mode until 4:00 PM ET**
""")

    scan_count = 0
    total_executed = 0

    while _is_market_open():
        scan_count += 1
        now = _now_et()
        logger.info(f"[{now.strftime('%H:%M:%S')} ET] APEX Scan #{scan_count}")

        # Check if we've hit max trades
        if total_executed >= max_trades:
            logger.info(f"Max trades ({max_trades}) reached. Stopping.")
            break

        # Run scan
        verdicts = scan_and_execute(dry_run=dry_run)
        total_executed += len(verdicts)

        for v in verdicts:
            logger.info(f"  -> {v.ticker} {v.action} conviction={v.conviction_score:.0f}%")

        # Sleep until next scan
        logger.info(f"Next scan in {SCAN_INTERVAL_SECONDS}s...")
        time.sleep(SCAN_INTERVAL_SECONDS)

    # Summary
    logger.info("=" * 60)
    logger.info(f"APEX POWER HOUR COMPLETE")
    logger.info(f"Scans: {scan_count} | Executed: {total_executed}")
    logger.info("=" * 60)

    send_alert(f"""✅ **APEX POWER HOUR COMPLETE**

Scans: {scan_count}
Trades Executed: {total_executed}

Session ended at {_now_et().strftime('%H:%M:%S')} ET
""")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="APEX Power Hour Runner")
    parser.add_argument("--dry-run", action="store_true", help="Don't execute trades")
    parser.add_argument("--max-trades", type=int, default=10, help="Max trades per session")
    parser.add_argument("--single", action="store_true", help="Single scan (no loop)")
    args = parser.parse_args()

    if args.single:
        verdicts = scan_and_execute(dry_run=args.dry_run)
        print(f"Single scan complete: {len(verdicts)} trades")
    else:
        run_power_hour(dry_run=args.dry_run, max_trades=args.max_trades)
