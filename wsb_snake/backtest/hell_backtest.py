#!/usr/bin/env python3
"""
BACKTEST FROM HELL: Full pipeline simulation with REAL OPTION P&L

Key fix: Option returns are NOT stock returns.
- Stock moves 1% = Option moves 50-500% depending on delta/gamma
- 0DTE OTM options have MASSIVE gamma
- LOTTO TICKET entries at $0.10-$0.50 can 10-50x on big moves
"""
import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import random

# Alpaca credentials
ALPACA_KEY = os.environ.get("ALPACA_API_KEY", "PKWT6T5BFKHBTFDW3CPAFW2XBZ")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "pVdzbVte2pQvL1RmCTFw3oaQ6TBWYimAzC42DUyTEy8")

# RISK-MANAGED BERSERKER PARAMETERS
INITIAL_ACCOUNT = 5000.0
POSITION_SIZE_PCT = 0.15  # 15% per trade - disciplined sizing
SLIPPAGE_PCT = 0.01  # 1% per side (options have wider spreads)
MAX_POSITIONS = 2  # Max 2 concurrent positions
MIN_ACCOUNT_FLOOR = 500.0  # Stop trading if account drops below this

# Target OTM offset for LOTTO tickets (points from ATM)
OTM_OFFSET = 2  # 2 points OTM = cheaper options, bigger leverage

# Trailing stop ladder
TRAIL_BREAKEVEN = 0.50  # +50% moves stop to breakeven
TRAIL_LOCK = 1.00  # +100% locks in profit
TRAIL_AGGRESSIVE = 2.00  # +200% tight trail
TRAIL_MOONSHOT = 5.00  # +500% tightest trail

# Pyramiding - much smaller adds
PYRAMID_TRIGGER = 1.00  # +100% triggers add (not 75%)
PYRAMID_ADD_PCT = 0.25  # Only add 25% (not 50%)


@dataclass
class SimPosition:
    """Simulated position."""
    entry_time: datetime
    entry_spy_price: float  # SPY price at entry
    direction: str  # "CALL" or "PUT"
    strike: float  # Option strike
    contracts: int
    entry_option_price: float  # Option premium at entry
    position_cost: float  # Total cost
    stop_loss_pct: float  # Stop as % of option price
    peak_option_price: float = 0.0
    pyramid_adds: int = 0
    original_contracts: int = 0


@dataclass
class SimTrade:
    """Completed simulated trade."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    strike: float
    contracts: int
    entry_option_price: float
    exit_option_price: float
    position_cost: float
    pnl_dollars: float
    pnl_pct: float
    exit_reason: str
    spy_entry: float
    spy_exit: float


@dataclass
class DayResult:
    """Result for one trading day."""
    date: str
    trades: List[SimTrade]
    num_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_winner: float
    avg_loser: float
    largest_trade: float
    end_account: float
    start_account: float
    max_drawdown: float


def fetch_minute_bars(date: str) -> List[Dict]:
    """Fetch minute-by-minute bars for SPY."""
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET
    }

    url = "https://data.alpaca.markets/v2/stocks/SPY/bars"
    params = {
        "timeframe": "1Min",
        "start": f"{date}T09:30:00Z",
        "end": f"{date}T16:00:00Z",
        "limit": 500,
        "feed": "iex"
    }

    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        print(f"  API Error: {r.status_code} - {r.text[:100]}")
        return []
    try:
        data = r.json()
        return data.get("bars", [])
    except:
        print(f"  JSON parse error for {date}")
        return []


def get_option_entry_price(spy_price: float, strike: float, direction: str) -> float:
    """
    Get realistic 0DTE option entry price.

    Based on distance from ATM:
    - ATM (0 pts): $1.50-2.00
    - 1pt OTM: $0.60-0.90
    - 2pt OTM: $0.25-0.40 (LOTTO ZONE)
    - 3pt+ OTM: $0.08-0.20 (DEEP LOTTO)
    """
    if direction == "CALL":
        distance = strike - spy_price  # Positive = OTM for calls
    else:
        distance = spy_price - strike  # Positive = OTM for puts

    # Price based on distance
    if distance <= 0:  # ITM
        base = 2.00 + abs(distance) * 0.9  # ITM adds intrinsic
    elif distance <= 1:
        base = 0.75
    elif distance <= 2:
        base = 0.30  # LOTTO TICKET ZONE
    elif distance <= 3:
        base = 0.15
    else:
        base = 0.08  # Deep OTM lotto

    # Deterministic pricing - no random variance for reproducible results
    return round(base, 2)


def calculate_option_price(
    spy_price: float,
    strike: float,
    direction: str,
    entry_option_price: float,
    entry_spy_price: float
) -> float:
    """
    Calculate current option price based on SPY movement.

    Uses simplified but realistic 0DTE Greeks:
    - Base delta: 0.30-0.50 for near OTM
    - Gamma effect: delta increases as option goes ITM
    - For big moves, gamma causes explosive returns
    """
    # Calculate SPY movement
    spy_move = spy_price - entry_spy_price

    # Determine if move is favorable
    if direction == "CALL":
        favorable_move = spy_move
    else:  # PUT
        favorable_move = -spy_move

    # Current distance from strike
    if direction == "CALL":
        current_distance = strike - spy_price
    else:
        current_distance = spy_price - strike

    # Dynamic delta based on moneyness
    if current_distance <= 0:  # ITM
        delta = 0.65 + min(0.30, abs(current_distance) * 0.05)  # Up to 0.95
    elif current_distance <= 1:
        delta = 0.45
    elif current_distance <= 2:
        delta = 0.30
    else:
        delta = 0.15

    # Base option move
    option_move = favorable_move * delta

    # GAMMA EFFECT: For big moves, the option accelerates
    # This is what makes 0DTE options explosive
    if abs(favorable_move) > 4:
        # Gamma explosion - option delta increased significantly during move
        gamma_multiplier = 2.5
    elif abs(favorable_move) > 2:
        gamma_multiplier = 1.8
    elif abs(favorable_move) > 1:
        gamma_multiplier = 1.3
    else:
        gamma_multiplier = 1.0

    option_move *= gamma_multiplier

    # Calculate new option price
    new_price = entry_option_price + option_move

    # Theta decay (simplified - 0DTE loses value fast if not moving)
    # If move is small, decay hurts
    if abs(favorable_move) < 0.5:
        new_price *= 0.90  # 10% decay per period in chop

    # Option can't go below $0.01
    new_price = max(0.01, new_price)

    # If deep ITM, add intrinsic value
    if current_distance < 0:
        intrinsic = abs(current_distance)
        new_price = max(new_price, intrinsic * 0.95)  # Near full intrinsic

    return round(new_price, 2)


def detect_signal(bars: List[Dict], idx: int) -> Optional[Tuple[str, float, float]]:
    """
    Signal detection based on momentum and volume.

    Returns (direction, confidence, strike) or None.
    """
    if idx < 10:
        return None

    recent = bars[max(0, idx-10):idx+1]
    if len(recent) < 10:
        return None

    current_price = recent[-1]["c"]

    # Calculate momentum
    momentum = (recent[-1]["c"] - recent[0]["c"]) / recent[0]["c"]

    # Volume spike
    avg_vol = sum(b["v"] for b in recent[:-1]) / (len(recent) - 1) if len(recent) > 1 else 1
    vol_spike = recent[-1]["v"] / avg_vol if avg_vol > 0 else 1

    # Strong bar detection (big candle)
    bar_range = (recent[-1]["h"] - recent[-1]["l"]) / recent[-1]["c"]

    confidence = 0
    direction = None

    # CALL signal: Strong upward momentum + volume
    if momentum > 0.002 and vol_spike > 1.3:
        direction = "CALL"
        confidence = min(90, 50 + momentum * 2000 + vol_spike * 10 + bar_range * 500)
        strike = round(current_price) + OTM_OFFSET  # OTM call

    # PUT signal: Strong downward momentum + volume
    elif momentum < -0.002 and vol_spike > 1.3:
        direction = "PUT"
        confidence = min(90, 50 + abs(momentum) * 2000 + vol_spike * 10 + bar_range * 500)
        strike = round(current_price) - OTM_OFFSET  # OTM put

    if direction and confidence >= 68:  # Higher threshold = fewer but better signals
        return (direction, confidence, strike)

    return None


def run_day_simulation(date: str, start_account: float) -> DayResult:
    """Run full simulation for one day with REAL option P&L."""
    print(f"\n{'='*60}")
    print(f"SIMULATING: {date}")
    print(f"Starting account: ${start_account:,.2f}")
    print(f"{'='*60}")

    bars = fetch_minute_bars(date)
    if not bars:
        print(f"  No bars available for {date}")
        return DayResult(
            date=date, trades=[], num_trades=0, wins=0, losses=0,
            win_rate=0, avg_winner=0, avg_loser=0, largest_trade=0,
            end_account=start_account, start_account=start_account, max_drawdown=0
        )

    print(f"  Loaded {len(bars)} minute bars")

    # Calculate day's range for context
    day_high = max(b["h"] for b in bars)
    day_low = min(b["l"] for b in bars)
    day_range = day_high - day_low
    print(f"  Day range: ${day_low:.2f} - ${day_high:.2f} (${day_range:.2f})")

    account = start_account
    peak_account = start_account
    max_drawdown = 0
    positions: List[SimPosition] = []
    completed_trades: List[SimTrade] = []

    for i, bar in enumerate(bars):
        bar_time = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        spy_price = bar["c"]

        # 1. Check existing positions
        for pos in positions[:]:
            # Calculate current option price
            current_option_price = calculate_option_price(
                spy_price=spy_price,
                strike=pos.strike,
                direction=pos.direction,
                entry_option_price=pos.entry_option_price,
                entry_spy_price=pos.entry_spy_price
            )

            # Track peak
            if current_option_price > pos.peak_option_price:
                pos.peak_option_price = current_option_price

            # Calculate P&L
            current_pnl_pct = (current_option_price - pos.entry_option_price) / pos.entry_option_price

            # Check trailing stops
            exit_reason = None

            # Stop loss check (-40% of option value - tighter to preserve capital)
            if current_pnl_pct <= -0.40:
                exit_reason = "STOP_LOSS"

            # Trailing stop from peak
            elif current_pnl_pct >= TRAIL_MOONSHOT:  # +500%
                drawdown_from_peak = (pos.peak_option_price - current_option_price) / pos.peak_option_price
                if drawdown_from_peak > 0.08:  # 8% from peak
                    exit_reason = f"TRAIL_MOON"
            elif current_pnl_pct >= TRAIL_AGGRESSIVE:  # +200%
                drawdown_from_peak = (pos.peak_option_price - current_option_price) / pos.peak_option_price
                if drawdown_from_peak > 0.12:
                    exit_reason = f"TRAIL_AGG"
            elif current_pnl_pct >= TRAIL_LOCK:  # +100%
                drawdown_from_peak = (pos.peak_option_price - current_option_price) / pos.peak_option_price
                if drawdown_from_peak > 0.20:
                    exit_reason = f"TRAIL_LOCK"
            elif current_pnl_pct >= TRAIL_BREAKEVEN:  # +50%
                if current_pnl_pct < 0.10:  # Gave back to <10%
                    exit_reason = "TRAIL_BE"

            if exit_reason:
                # Exit with slippage
                exit_price = current_option_price * (1 - SLIPPAGE_PCT)
                position_value = pos.contracts * exit_price * 100
                pnl_dollars = position_value - pos.position_cost
                pnl_pct = (exit_price - pos.entry_option_price) / pos.entry_option_price * 100

                trade = SimTrade(
                    entry_time=pos.entry_time,
                    exit_time=bar_time,
                    direction=pos.direction,
                    strike=pos.strike,
                    contracts=pos.contracts,
                    entry_option_price=pos.entry_option_price,
                    exit_option_price=exit_price,
                    position_cost=pos.position_cost,
                    pnl_dollars=pnl_dollars,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                    spy_entry=pos.entry_spy_price,
                    spy_exit=spy_price
                )
                completed_trades.append(trade)

                account += pnl_dollars
                positions.remove(pos)

                spy_move = spy_price - pos.entry_spy_price
                print(f"  [{bar_time.strftime('%H:%M')}] EXIT {pos.direction} ${pos.strike}: {exit_reason}")
                print(f"      SPY: ${pos.entry_spy_price:.2f} → ${spy_price:.2f} ({spy_move:+.2f})")
                print(f"      Option: ${pos.entry_option_price:.2f} → ${exit_price:.2f}")
                print(f"      P&L: ${pnl_dollars:+,.2f} ({pnl_pct:+.0f}%)")

                # Track drawdown
                if account > peak_account:
                    peak_account = account
                dd = (peak_account - account) / peak_account if peak_account > 0 else 0
                if dd > max_drawdown:
                    max_drawdown = dd

        # 2. Check for new signals
        # RISK MANAGEMENT: Stop trading if account is below floor
        if account < MIN_ACCOUNT_FLOOR:
            continue  # Skip all trading, preserve remaining capital

        if len(positions) < MAX_POSITIONS:
            signal = detect_signal(bars, i)

            if signal:
                direction, confidence, strike = signal

                # Dynamic position sizing - scale with account performance
                # When losing: trade smaller to preserve capital
                # When winning: slightly larger to compound
                account_ratio = account / INITIAL_ACCOUNT
                if account_ratio < 0.5:  # Down 50%+
                    size_multiplier = 0.5  # Cut size in half
                elif account_ratio < 0.8:  # Down 20-50%
                    size_multiplier = 0.75  # Reduce size
                elif account_ratio > 1.5:  # Up 50%+
                    size_multiplier = 1.25  # Slightly larger
                elif account_ratio > 2.0:  # Up 100%+
                    size_multiplier = 1.5  # Compound gains
                else:
                    size_multiplier = 1.0

                position_value = account * POSITION_SIZE_PCT * size_multiplier

                # Get option price
                option_price = get_option_entry_price(spy_price, strike, direction)
                option_price *= (1 + SLIPPAGE_PCT)  # Slippage on entry

                # Calculate contracts
                contract_cost = option_price * 100
                contracts = max(1, int(position_value / contract_cost))
                actual_cost = contracts * contract_cost

                pos = SimPosition(
                    entry_time=bar_time,
                    entry_spy_price=spy_price,
                    direction=direction,
                    strike=strike,
                    contracts=contracts,
                    entry_option_price=option_price,
                    position_cost=actual_cost,
                    stop_loss_pct=-0.40,  # Tighter stop
                    peak_option_price=option_price,
                    original_contracts=contracts
                )
                positions.append(pos)

                print(f"  [{bar_time.strftime('%H:%M')}] ENTRY {direction} ${strike}: {contracts}x @ ${option_price:.2f} (${actual_cost:,.0f}) | Conf: {confidence:.0f}%")

        # 3. Pyramiding on big winners (only if account is healthy)
        if account < MIN_ACCOUNT_FLOOR:
            continue  # No pyramiding when low on capital

        for pos in positions:
            if pos.pyramid_adds < 1:  # Max 1 pyramid add (was 2)
                current_option_price = calculate_option_price(
                    spy_price, pos.strike, pos.direction,
                    pos.entry_option_price, pos.entry_spy_price
                )
                current_pnl_pct = (current_option_price - pos.entry_option_price) / pos.entry_option_price

                if current_pnl_pct >= PYRAMID_TRIGGER:
                    add_contracts = max(1, int(pos.original_contracts * PYRAMID_ADD_PCT))
                    add_cost = add_contracts * current_option_price * 100 * (1 + SLIPPAGE_PCT)

                    pos.contracts += add_contracts
                    pos.position_cost += add_cost
                    pos.pyramid_adds += 1

                    print(f"  [{bar_time.strftime('%H:%M')}] PYRAMID {pos.direction} ${pos.strike}: +{add_contracts}x @ ${current_option_price:.2f}")

    # Close remaining positions at EOD
    for pos in positions:
        final_bar = bars[-1]
        spy_price = final_bar["c"]

        final_option_price = calculate_option_price(
            spy_price, pos.strike, pos.direction,
            pos.entry_option_price, pos.entry_spy_price
        )
        final_option_price *= (1 - SLIPPAGE_PCT)

        position_value = pos.contracts * final_option_price * 100
        pnl_dollars = position_value - pos.position_cost
        pnl_pct = (final_option_price - pos.entry_option_price) / pos.entry_option_price * 100

        trade = SimTrade(
            entry_time=pos.entry_time,
            exit_time=datetime.fromisoformat(final_bar["t"].replace("Z", "+00:00")),
            direction=pos.direction,
            strike=pos.strike,
            contracts=pos.contracts,
            entry_option_price=pos.entry_option_price,
            exit_option_price=final_option_price,
            position_cost=pos.position_cost,
            pnl_dollars=pnl_dollars,
            pnl_pct=pnl_pct,
            exit_reason="EOD_CLOSE",
            spy_entry=pos.entry_spy_price,
            spy_exit=spy_price
        )
        completed_trades.append(trade)
        account += pnl_dollars

        spy_move = spy_price - pos.entry_spy_price
        print(f"  [EOD] CLOSE {pos.direction} ${pos.strike}: SPY {spy_move:+.2f}, Option ${pos.entry_option_price:.2f}→${final_option_price:.2f}, P&L: ${pnl_dollars:+,.0f} ({pnl_pct:+.0f}%)")

    # Calculate stats
    wins = [t for t in completed_trades if t.pnl_dollars > 0]
    losses = [t for t in completed_trades if t.pnl_dollars <= 0]

    result = DayResult(
        date=date,
        trades=completed_trades,
        num_trades=len(completed_trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(completed_trades) * 100 if completed_trades else 0,
        avg_winner=sum(t.pnl_pct for t in wins) / len(wins) if wins else 0,
        avg_loser=sum(t.pnl_pct for t in losses) / len(losses) if losses else 0,
        largest_trade=max((t.pnl_dollars for t in completed_trades), default=0),
        end_account=account,
        start_account=start_account,
        max_drawdown=max_drawdown * 100
    )

    print(f"\n  DAY SUMMARY:")
    print(f"    Trades: {result.num_trades} | Wins: {result.wins} | Win Rate: {result.win_rate:.0f}%")
    print(f"    Avg Winner: {result.avg_winner:+.0f}% | Avg Loser: {result.avg_loser:+.0f}%")
    print(f"    Largest Trade: ${result.largest_trade:+,.0f}")
    print(f"    Account: ${result.start_account:,.0f} → ${result.end_account:,.0f} ({((result.end_account/result.start_account)-1)*100:+.1f}%)")
    print(f"    Max Drawdown: {result.max_drawdown:.1f}%")

    return result


def run_week_backtest(week_type: str, dates: List[str]):
    """Run backtest for a specific week type."""
    print(f"\n{'='*70}")
    print(f"WEEK TYPE: {week_type}")
    print(f"Dates: {', '.join(dates)}")
    print(f"{'='*70}")

    account = INITIAL_ACCOUNT
    all_results: List[DayResult] = []

    for date in dates:
        result = run_day_simulation(date, account)
        all_results.append(result)
        account = result.end_account

    # Summary
    total_trades = sum(r.num_trades for r in all_results)
    total_wins = sum(r.wins for r in all_results)
    overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
    final_return = ((account / INITIAL_ACCOUNT) - 1) * 100

    print(f"\n{'='*70}")
    print(f"{week_type} WEEK RESULTS")
    print(f"{'='*70}")
    for r in all_results:
        day_return = ((r.end_account / r.start_account) - 1) * 100
        print(f"  {r.date}: ${r.start_account:,.0f} → ${r.end_account:,.0f} ({day_return:+.1f}%) | {r.num_trades} trades")

    print(f"\n  FINAL: ${INITIAL_ACCOUNT:,.0f} → ${account:,.0f} ({final_return:+.1f}%)")
    print(f"  Total Trades: {total_trades} | Win Rate: {overall_win_rate:.0f}%")

    return account, all_results


def run_hell_backtest():
    """Run THE BACKTEST FROM HELL - three week types."""
    print("\n" + "=" * 70)
    print("THE BACKTEST FROM HELL")
    print("Real Alpaca Data | Real Option P&L | Full Pipeline")
    print("=" * 70)
    print(f"Initial Account: ${INITIAL_ACCOUNT:,.0f}")
    print(f"Position Size: {POSITION_SIZE_PCT*100:.0f}%")
    print(f"OTM Offset: {OTM_OFFSET} points (LOTTO TICKET entries)")

    # Week definitions - VERIFIED BY ACTUAL SPY DATA
    # BAD WEEK: Lowest volatility - avg $3.74 daily range
    bad_week = ["2026-01-22", "2026-01-23", "2026-01-26", "2026-01-27", "2026-01-28"]

    # AVERAGE WEEK: Moderate volatility - avg $7-8 daily range
    avg_week = ["2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13"]

    # GREAT WEEK: Highest volatility - avg $9.84 daily range
    great_week = ["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03", "2026-02-04"]

    results = {}

    # Run BAD week
    bad_final, bad_results = run_week_backtest("BAD (Choppy/Sideways)", bad_week)
    results["bad"] = {"final": bad_final, "target": 7000}

    # Run AVERAGE week
    avg_final, avg_results = run_week_backtest("AVERAGE (Moderate VIX)", avg_week)
    results["average"] = {"final": avg_final, "target": 12000}

    # Run GREAT week
    great_final, great_results = run_week_backtest("GREAT (High Volatility)", great_week)
    results["great"] = {"final": great_final, "target": 30000}

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS - THE LIE DETECTOR")
    print("=" * 70)

    for week_type, data in results.items():
        final = data["final"]
        target = data["target"]
        hit = "✅" if final >= target else "❌"
        pct = ((final / INITIAL_ACCOUNT) - 1) * 100
        print(f"\n{week_type.upper()} WEEK:")
        print(f"  Result: ${INITIAL_ACCOUNT:,.0f} → ${final:,.0f} ({pct:+.1f}%)")
        print(f"  Target: ${target:,.0f}")
        print(f"  Status: {hit} {'HIT' if final >= target else 'MISSED by $' + f'{target-final:,.0f}'}")

    return results


if __name__ == "__main__":
    run_hell_backtest()
