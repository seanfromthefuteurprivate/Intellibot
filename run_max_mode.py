#!/usr/bin/env python3
"""
🔥 MAX MODE - AGGRESSIVE LAST HOUR PREDATOR TRADING 🔥

Settings:
- Conviction threshold: 55% (find more opportunities)
- Scan interval: 10 seconds (fast scanning)
- Target: +6% (quick profits, don't be greedy)
- Stop: -10% (wider to avoid noise exits)
- Trailing: +2% -> -5% stop, +3% -> breakeven, +5% -> +3% lock
- Max hold: 5 minutes (quick rotations)

Philosophy:
- Strike fast, book profit, hunt again
- Never let winners become losers
- Capture volatility, maximize rotations
- Predator mode until market close
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from wsb_snake.utils.logger import get_logger
from wsb_snake.execution.apex_conviction_engine import apex_engine
from wsb_snake.execution.regime_detector import regime_detector
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.collectors.polygon_options import polygon_options
from wsb_snake.notifications.telegram_bot import send_alert
from wsb_snake.trading.alpaca_executor import alpaca_executor

logger = get_logger(__name__)

# MAX MODE WATCHLIST - high volatility, high liquidity
MAX_MODE_WATCHLIST = [
    "SPY", "QQQ", "IWM",      # Index ETFs
    "NVDA", "TSLA", "AMD",    # High-vol tech
    "META", "AAPL", "AMZN",   # Mega caps
]

# JP MORGAN SCALP SETTINGS - Quality over quantity
MIN_CONVICTION = 68  # Institutional standard (was 55 - too low)
SCAN_INTERVAL = 15   # Slightly slower to reduce noise

def get_spot(ticker):
    """Get spot price using best available source."""
    try:
        snap = polygon_enhanced.get_snapshot(ticker)
        if snap and snap.get("price"):
            return float(snap["price"])
    except:
        pass
    return 0

def get_atm_option(ticker, spot, side, expiry_date):
    """Get ATM option for quick execution - uses ALPACA for real quotes."""
    try:
        # Get chain structure from Polygon (for strike discovery)
        chain = polygon_options.get_chain_for_expiration(ticker, spot, expiry_date, strike_range=5)
        if not chain:
            return None

        contracts = chain.get("calls" if side == "CALL" else "puts", [])
        if not contracts:
            return None

        # Find ATM (closest to spot)
        atm = min(contracts, key=lambda c: abs(c.get("strike", 0) - spot))
        strike = atm.get("strike", 0)
        symbol = atm.get("symbol", "").replace("O:", "")  # Strip Polygon prefix

        if not symbol:
            return None

        # GET REAL QUOTE FROM ALPACA (Polygon quotes are often $0)
        alpaca_quote = alpaca_executor.get_option_quote(symbol)
        if alpaca_quote:
            bid = float(alpaca_quote.get("bp", 0))
            ask = float(alpaca_quote.get("ap", 0))

            if bid > 0 and ask > 0:
                # Update the contract with Alpaca prices
                atm["bid"] = bid
                atm["ask"] = ask
                atm["symbol"] = symbol  # Use clean symbol

                mid = (bid + ask) / 2
                spread_pct = (ask - bid) / mid if mid > 0 else 1

                if spread_pct > 0.30:  # MAX MODE: Allow up to 30% spread
                    logger.info(f"SPREAD_WARNING: {ticker} {side} ${strike} spread {spread_pct:.1%}")

                logger.info(f"ALPACA QUOTE: {ticker} {side} ${strike} bid=${bid:.2f} ask=${ask:.2f}")
                return atm

        return None
    except Exception as e:
        logger.debug(f"Option fetch failed {ticker}: {e}")
        return None

def now_et():
    """Get current ET time."""
    try:
        import pytz
        return datetime.now(pytz.timezone("America/New_York"))
    except:
        return datetime.utcnow()

def is_market_open():
    """Check if market is open."""
    now = now_et()
    if now.weekday() >= 5:
        return False
    if now.hour < 9 or (now.hour == 9 and now.minute < 30):
        return False
    if now.hour >= 16:
        return False
    return True

def main():
    print("=" * 70)
    print("🔥🔥🔥 MAX MODE ACTIVATED - PREDATOR TRADING 🔥🔥🔥")
    print("=" * 70)
    print()
    print(f"Time: {now_et().strftime('%H:%M:%S')} ET")
    print(f"Watchlist: {', '.join(MAX_MODE_WATCHLIST)}")
    print(f"Min Conviction: {MIN_CONVICTION}%")
    print(f"Scan Interval: {SCAN_INTERVAL}s")
    print(f"Target: +6% | Stop: -10% | Max Hold: 5min")
    print()
    print("Strategy: Strike fast, book profit, hunt again!")
    print("=" * 70)
    print()

    send_alert(f"""🔥🔥🔥 **MAX MODE ACTIVATED** 🔥🔥🔥

**PREDATOR TRADING - LAST HOUR**

Watchlist: {len(MAX_MODE_WATCHLIST)} tickers
Min Conviction: {MIN_CONVICTION}%
Scan Interval: {SCAN_INTERVAL}s

**Quick Profit Settings:**
• Target: +10% (book it fast!)
• Stop: -8% (tight risk)
• Trail: +5% -> breakeven
• Max Hold: 8 minutes

**Philosophy:** Strike fast, book profit, hunt again!

🎯 Hunting for opportunities now...""")

    scan_count = 0
    trades_executed = 0

    # Sync any existing positions
    alpaca_executor.sync_existing_positions()
    alpaca_executor.start_monitoring()

    # Warm up regime detector with initial data
    logger.info("Warming up regime detector...")
    try:
        regime_state = regime_detector.fetch_and_update()
        logger.info(f"Regime: {regime_state.regime.value} (confidence={regime_state.confidence:.0%})")
    except Exception as e:
        logger.warning(f"Regime warmup failed: {e}")

    while is_market_open():
        scan_count += 1
        now = now_et()

        # Update regime every 5 minutes
        if int(time.time()) % 300 < SCAN_INTERVAL:
            try:
                regime_state = regime_detector.fetch_and_update()
                logger.info(f"Regime update: {regime_state.regime.value}")
            except:
                pass

        print(f"\n[{now.strftime('%H:%M:%S')}] 🔍 MAX MODE Scan #{scan_count}")

        for ticker in MAX_MODE_WATCHLIST:
            try:
                spot = get_spot(ticker)
                if not spot:
                    continue

                # Run APEX analysis
                verdict = apex_engine.analyze(ticker, spot)

                # Log all with emoji indicators
                if verdict.conviction_score >= MIN_CONVICTION:
                    emoji = "🎯" if verdict.action != "NO_TRADE" else "⚡"
                else:
                    emoji = "⏳"

                print(f"  {emoji} {ticker}: {verdict.conviction_score:.0f}% {verdict.direction} -> {verdict.action}")

                # Skip if below threshold or neutral
                if verdict.conviction_score < MIN_CONVICTION:
                    continue
                if verdict.action == "NO_TRADE":
                    continue

                # Found a trade opportunity!
                print(f"\n  🚀 HIGH CONVICTION SIGNAL: {ticker} {verdict.action}")

                # Get option
                today = now.strftime("%Y-%m-%d")
                side = "CALL" if verdict.action == "BUY_CALLS" else "PUT"
                option = get_atm_option(ticker, spot, side, today)

                if not option:
                    print(f"  ❌ No valid option for {ticker} {side}")
                    continue

                strike = option.get("strike", 0)
                ask = option.get("ask", 0)
                symbol = option.get("symbol", "")

                # Send alert
                send_alert(f"""🎯 **MAX MODE TRADE SIGNAL**

**{ticker}** {side} ${strike:.0f}
Conviction: **{verdict.conviction_score:.0f}%** ({verdict.direction})
Spot: ${spot:.2f}
Option Ask: ${ask:.2f}

**Executing NOW...**""")

                # Execute trade
                from wsb_snake.trading.risk_governor import TradingEngine

                clean_symbol = symbol.replace("O:", "") if symbol.startswith("O:") else symbol

                # CRITICAL FIX: Direction must match action (was hardcoded to "long")
                direction = "long" if verdict.action == "BUY_CALLS" else "short"

                # CRITICAL FIX: Target/stop must be appropriate for direction
                # For CALLS (long): we profit when option price goes UP
                # For PUTS (long put): we profit when option price goes UP (put premium rises when underlying falls)
                # Since we're always BUYING options (not shorting), target is always higher than entry
                target_price = ask * 1.06   # +6% target (achievable in 0DTE timeframe)
                stop_loss = ask * 0.90      # -10% stop (wider to avoid noise exits)

                alpaca_pos = alpaca_executor.execute_scalp_entry(
                    underlying=ticker,
                    direction=direction,
                    entry_price=ask,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    confidence=verdict.conviction_score,
                    pattern=f"MAX_MODE_{verdict.direction}",
                    engine=TradingEngine.SCALPER,
                    strike_override=strike,
                    option_symbol_override=clean_symbol,
                    option_type_override=side.lower(),
                )

                if alpaca_pos:
                    trades_executed += 1
                    print(f"  ✅ EXECUTED: {alpaca_pos.option_symbol}")
                    send_alert(f"✅ **MAX MODE EXECUTED**\n{ticker} {side} ${strike:.0f}\nTrades today: {trades_executed}")
                else:
                    print(f"  ⚠️ Trade skipped by executor")

            except Exception as e:
                logger.error(f"Error scanning {ticker}: {e}")
                continue

        # Brief sleep then scan again
        print(f"\n  ⏱️ Next scan in {SCAN_INTERVAL}s...")
        time.sleep(SCAN_INTERVAL)

    # Market closed
    print("\n" + "=" * 70)
    print("🏁 MAX MODE SESSION COMPLETE")
    print(f"Scans: {scan_count} | Trades: {trades_executed}")
    print("=" * 70)

    send_alert(f"""🏁 **MAX MODE SESSION COMPLETE**

Scans: {scan_count}
Trades Executed: {trades_executed}

Session ended at {now_et().strftime('%H:%M:%S')} ET""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 MAX MODE stopped by user")
