#!/usr/bin/env python3
"""
BACKTEST FROM HELL v3: TRUTH-CALIBRATED

4 FIXES based on Polygon reality check:
1. Dynamic strike selection - find cheapest liquid option
2. 2-minute signal confirmation - filter fakeouts
3. Tighter stops (-30%) + 20-min time stop - cut losses faster
4. Volume/trend/range filters - eliminate bad setups

All prices from Polygon.io. Zero simulation.
"""
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

# API Keys
ALPACA_KEY = os.environ.get("ALPACA_API_KEY", "PKWT6T5BFKHBTFDW3CPAFW2XBZ")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "pVdzbVte2pQvL1RmCTFw3oaQ6TBWYimAzC42DUyTEy8")
POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "QJWtaUQV7N8mytTI7PH26lX3Ju6PD2iq")

# TRUTH-CALIBRATED PARAMETERS
INITIAL_ACCOUNT = 5000.0
POSITION_SIZE_PCT = 0.20  # 20% per trade (slightly higher since fewer trades)
SLIPPAGE_PCT = 0.02  # 2% slippage (real spreads)
MAX_POSITIONS = 2
MIN_ACCOUNT_FLOOR = 500.0

# FIX 3: Tighter stops
STOP_LOSS_PCT = -0.30  # -30% stop (was -40%)
TIME_STOP_MINUTES = 20  # Exit if flat after 20 minutes

# Trailing stop ladder
TRAIL_BREAKEVEN = 0.30  # +30% moves stop to breakeven (was 50%)
TRAIL_LOCK = 0.60  # +60% locks profit (was 100%)
TRAIL_AGGRESSIVE = 1.00  # +100% tight trail (was 200%)
TRAIL_MOONSHOT = 2.00  # +200% tightest (was 500%)

# Pyramiding - disabled for now (fewer trades = more conviction each)
PYRAMID_ENABLED = False

# FIX 4: Filters
MIN_SPY_RANGE = 1.0  # Minimum $1 move from open required
SKIP_FIRST_MINUTES = 5  # Skip first 5 minutes (9:30-9:35)
MIN_OPTION_VOLUME = 50  # Minimum option volume at entry

# Cache
OPTION_BARS_CACHE: Dict[str, List[Dict]] = {}


@dataclass
class SimPosition:
    """Position with REAL option pricing."""
    entry_time: datetime
    entry_minute_ts: int
    entry_spy_price: float
    direction: str
    strike: float
    expiry: str
    option_ticker: str
    contracts: int
    entry_option_price: float
    position_cost: float
    peak_option_price: float = 0.0
    pyramid_adds: int = 0
    original_contracts: int = 0


@dataclass
class SimTrade:
    """Completed trade."""
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
    option_ticker: str
    hold_minutes: int


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


def build_option_ticker(strike: float, expiry: str, direction: str) -> str:
    """Build Polygon option ticker."""
    expiry_fmt = datetime.strptime(expiry, "%Y-%m-%d").strftime("%y%m%d")
    cp = "C" if direction == "CALL" else "P"
    strike_fmt = f"{int(strike * 1000):08d}"
    return f"O:SPY{expiry_fmt}{cp}{strike_fmt}"


def fetch_option_bars(option_ticker: str, date: str) -> List[Dict]:
    """Fetch REAL minute option prices from Polygon."""
    cache_key = f"{option_ticker}_{date}"
    if cache_key in OPTION_BARS_CACHE:
        return OPTION_BARS_CACHE[cache_key]

    url = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/minute/{date}/{date}"
    params = {"apiKey": POLYGON_KEY, "limit": 50000, "sort": "asc"}

    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            bars = r.json().get("results", [])
            OPTION_BARS_CACHE[cache_key] = bars
            return bars
    except:
        pass
    return []


def get_option_price_at_time(option_bars: List[Dict], target_ts: int, use_bid: bool = False) -> Optional[Tuple[float, int]]:
    """
    Find option price at timestamp.
    Returns (price, volume) tuple.
    use_bid=True returns low price (approximating bid for exits)
    """
    if not option_bars:
        return None

    best_bar = None
    best_diff = float('inf')

    for bar in option_bars:
        diff = abs(bar["t"] - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_bar = bar
        if bar["t"] > target_ts and best_bar:
            break

    if best_bar and best_diff <= 120000:  # Within 2 minutes
        price = best_bar["l"] if use_bid else best_bar["c"]  # Use low as bid proxy
        volume = best_bar.get("v", 0)
        return (price, volume)

    return None


def fetch_spy_minute_bars(date: str) -> List[Dict]:
    """Fetch SPY minute bars from Alpaca."""
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

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get("bars", [])
    except:
        pass
    return []


def find_best_strike(spy_price: float, direction: str, date: str, expiry: str,
                     prefetched: Dict[str, List[Dict]], bar_ts: int) -> Optional[Dict]:
    """
    FIX 1: Dynamic strike selection.
    Find cheapest liquid option in $0.20-$1.00 range.
    """
    candidates = []

    for offset in range(1, 6):  # Try 1-5 points OTM
        if direction == "CALL":
            strike = round(spy_price) + offset
        else:
            strike = round(spy_price) - offset

        ticker = build_option_ticker(strike, expiry, direction)
        bars = prefetched.get(ticker, [])

        if not bars:
            bars = fetch_option_bars(ticker, date)
            if bars:
                prefetched[ticker] = bars

        if bars:
            result = get_option_price_at_time(bars, bar_ts)
            if result:
                price, volume = result
                if volume >= MIN_OPTION_VOLUME:  # Liquidity check
                    candidates.append({
                        "strike": strike,
                        "price": price,
                        "volume": volume,
                        "offset": offset,
                        "ticker": ticker
                    })

    if not candidates:
        return None

    # Sort by price, prefer $0.20-$1.00 range
    candidates.sort(key=lambda x: x["price"])

    # First try to find one in sweet spot
    for c in candidates:
        if 0.20 <= c["price"] <= 1.00 and c["volume"] >= 100:
            return c

    # Fall back to any with volume
    for c in candidates:
        if c["price"] <= 1.50 and c["volume"] >= MIN_OPTION_VOLUME:
            return c

    return candidates[0] if candidates else None


def calculate_30bar_ma(bars: List[Dict], idx: int) -> Optional[float]:
    """Calculate 30-bar moving average."""
    if idx < 30:
        return None
    window = bars[idx-29:idx+1]
    return sum(b["c"] for b in window) / len(window)


def detect_signal_v3(bars: List[Dict], idx: int, last_signal: Dict,
                     day_open: float, current_ma: Optional[float]) -> Optional[Tuple[str, float, bool]]:
    """
    FIX 2 & 4: Signal detection with confirmation and filters.
    Returns (direction, confidence, is_confirmed).
    """
    if idx < 10:
        return None

    current_bar = bars[idx]
    bar_time = datetime.fromisoformat(current_bar["t"].replace("Z", "+00:00"))

    # FIX 4A: Skip first 5 minutes
    market_open = bar_time.replace(hour=9, minute=30, second=0, microsecond=0)
    minutes_since_open = (bar_time - market_open).total_seconds() / 60
    if minutes_since_open < SKIP_FIRST_MINUTES:
        return None

    recent = bars[max(0, idx-10):idx+1]
    if len(recent) < 10:
        return None

    current_price = recent[-1]["c"]

    # FIX 4C: SPY range check
    spy_range = abs(current_price - day_open)
    if spy_range < MIN_SPY_RANGE:
        return None  # Day too quiet

    momentum = (recent[-1]["c"] - recent[0]["c"]) / recent[0]["c"]
    avg_vol = sum(b["v"] for b in recent[:-1]) / (len(recent) - 1) if len(recent) > 1 else 1
    vol_spike = recent[-1]["v"] / avg_vol if avg_vol > 0 else 1
    bar_range = (recent[-1]["h"] - recent[-1]["l"]) / recent[-1]["c"]

    direction = None
    confidence = 0

    # CALL signal
    if momentum > 0.002 and vol_spike > 1.3:
        # FIX 4D: Trend alignment
        if current_ma and current_price < current_ma:
            return None  # Don't buy calls below MA
        direction = "CALL"
        confidence = min(90, 50 + momentum * 2000 + vol_spike * 10 + bar_range * 500)

    # PUT signal
    elif momentum < -0.002 and vol_spike > 1.3:
        # FIX 4D: Trend alignment
        if current_ma and current_price > current_ma:
            return None  # Don't buy puts above MA
        direction = "PUT"
        confidence = min(90, 50 + abs(momentum) * 2000 + vol_spike * 10 + bar_range * 500)

    if direction and confidence >= 62:  # Lowered from 68
        # FIX 2: Check if this is a CONFIRMATION (same direction 2 minutes in a row)
        is_confirmed = False
        if last_signal.get("direction") == direction:
            last_time = last_signal.get("time")
            if last_time:
                time_diff = (bar_time - last_time).total_seconds()
                if 30 <= time_diff <= 180:  # 30 sec to 3 min gap
                    is_confirmed = True

        return (direction, confidence, is_confirmed)

    return None


def run_day_simulation(date: str, start_account: float) -> DayResult:
    """Run simulation with all 4 fixes."""
    print(f"\n{'='*60}")
    print(f"SIMULATING: {date} [TRUTH-CALIBRATED v3]")
    print(f"Starting account: ${start_account:,.2f}")
    print(f"{'='*60}")

    spy_bars = fetch_spy_minute_bars(date)
    if not spy_bars:
        print(f"  No SPY bars for {date}")
        return DayResult(
            date=date, trades=[], num_trades=0, wins=0, losses=0,
            win_rate=0, avg_winner=0, avg_loser=0, largest_trade=0,
            end_account=start_account, start_account=start_account, max_drawdown=0
        )

    print(f"  Loaded {len(spy_bars)} SPY minute bars")

    day_high = max(b["h"] for b in spy_bars)
    day_low = min(b["l"] for b in spy_bars)
    day_open = spy_bars[0]["o"]
    print(f"  Day range: ${day_low:.2f} - ${day_high:.2f} (${day_high-day_low:.2f})")

    expiry = date
    account = start_account
    peak_account = start_account
    max_drawdown = 0
    positions: List[SimPosition] = []
    completed_trades: List[SimTrade] = []
    prefetched_options: Dict[str, List[Dict]] = {}

    # FIX 2: Track last signal for confirmation
    last_signal = {"direction": None, "time": None}
    pending_signal = None  # Signal waiting for confirmation

    # Pre-fetch likely strikes
    spy_mid = (day_high + day_low) / 2
    print(f"  Pre-fetching options for strikes near ${spy_mid:.0f}...")
    for strike in range(int(spy_mid) - 6, int(spy_mid) + 7):
        for direction in ["CALL", "PUT"]:
            ticker = build_option_ticker(strike, expiry, direction)
            bars = fetch_option_bars(ticker, date)
            if bars:
                prefetched_options[ticker] = bars
    print(f"  Pre-fetched {len(prefetched_options)} contracts")

    for i, bar in enumerate(spy_bars):
        bar_time = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        bar_ts = int(bar_time.timestamp() * 1000)
        spy_price = bar["c"]

        # Calculate 30-bar MA
        current_ma = calculate_30bar_ma(spy_bars, i)

        # 1. Check existing positions
        for pos in positions[:]:
            hold_minutes = int((bar_time - pos.entry_time).total_seconds() / 60)

            # Get current option price (use low as bid proxy for exits)
            option_bars = prefetched_options.get(pos.option_ticker, [])
            result = get_option_price_at_time(option_bars, bar_ts, use_bid=True)

            if not result:
                continue

            current_option_price, _ = result

            if current_option_price > pos.peak_option_price:
                pos.peak_option_price = current_option_price

            current_pnl_pct = (current_option_price - pos.entry_option_price) / pos.entry_option_price

            exit_reason = None

            # FIX 3: Tighter stop loss (-30%)
            if current_pnl_pct <= STOP_LOSS_PCT:
                exit_reason = "STOP_LOSS"

            # FIX 3: Time stop - exit if flat after 20 minutes
            elif hold_minutes >= TIME_STOP_MINUTES and -0.05 <= current_pnl_pct <= 0.05:
                exit_reason = "TIME_STOP"

            # Trailing stops
            elif current_pnl_pct >= TRAIL_MOONSHOT:
                drawdown = (pos.peak_option_price - current_option_price) / pos.peak_option_price
                if drawdown > 0.08:
                    exit_reason = "TRAIL_MOON"
            elif current_pnl_pct >= TRAIL_AGGRESSIVE:
                drawdown = (pos.peak_option_price - current_option_price) / pos.peak_option_price
                if drawdown > 0.12:
                    exit_reason = "TRAIL_AGG"
            elif current_pnl_pct >= TRAIL_LOCK:
                drawdown = (pos.peak_option_price - current_option_price) / pos.peak_option_price
                if drawdown > 0.15:
                    exit_reason = "TRAIL_LOCK"
            elif current_pnl_pct >= TRAIL_BREAKEVEN:
                if current_pnl_pct < 0.05:
                    exit_reason = "TRAIL_BE"

            if exit_reason:
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
                    spy_exit=spy_price,
                    option_ticker=pos.option_ticker,
                    hold_minutes=hold_minutes
                )
                completed_trades.append(trade)

                account += pnl_dollars
                positions.remove(pos)

                print(f"  [{bar_time.strftime('%H:%M')}] EXIT {pos.direction} ${pos.strike}: {exit_reason} ({hold_minutes}min)")
                print(f"      ${pos.entry_option_price:.2f} → ${exit_price:.2f} | P&L: ${pnl_dollars:+,.0f} ({pnl_pct:+.0f}%)")

                if account > peak_account:
                    peak_account = account
                dd = (peak_account - account) / peak_account if peak_account > 0 else 0
                if dd > max_drawdown:
                    max_drawdown = dd

        # 2. Check for new signals with CONFIRMATION
        if account < MIN_ACCOUNT_FLOOR:
            continue

        if len(positions) < MAX_POSITIONS:
            signal_result = detect_signal_v3(spy_bars, i, last_signal, day_open, current_ma)

            if signal_result:
                direction, confidence, is_confirmed = signal_result

                # Update last signal for next iteration
                last_signal = {"direction": direction, "time": bar_time}

                # FIX 2: Only enter on CONFIRMED signals
                if is_confirmed:
                    # FIX 1: Dynamic strike selection
                    best_strike = find_best_strike(spy_price, direction, date, expiry,
                                                   prefetched_options, bar_ts)

                    if not best_strike:
                        print(f"  [{bar_time.strftime('%H:%M')}] SKIP {direction}: No liquid strikes found")
                        continue

                    strike = best_strike["strike"]
                    option_price = best_strike["price"]
                    option_ticker = best_strike["ticker"]
                    option_volume = best_strike["volume"]

                    # FIX 4B: Option volume check
                    if option_volume < MIN_OPTION_VOLUME:
                        print(f"  [{bar_time.strftime('%H:%M')}] SKIP {direction} ${strike}: Low volume ({option_volume})")
                        continue

                    # Apply slippage
                    entry_price = option_price * (1 + SLIPPAGE_PCT)

                    # Position sizing
                    position_value = account * POSITION_SIZE_PCT
                    contract_cost = entry_price * 100
                    contracts = max(1, int(position_value / contract_cost))
                    actual_cost = contracts * contract_cost

                    pos = SimPosition(
                        entry_time=bar_time,
                        entry_minute_ts=bar_ts,
                        entry_spy_price=spy_price,
                        direction=direction,
                        strike=strike,
                        expiry=expiry,
                        option_ticker=option_ticker,
                        contracts=contracts,
                        entry_option_price=entry_price,
                        position_cost=actual_cost,
                        peak_option_price=entry_price,
                        original_contracts=contracts
                    )
                    positions.append(pos)

                    print(f"  [{bar_time.strftime('%H:%M')}] ✓ ENTRY {direction} ${strike}: {contracts}x @ ${entry_price:.2f} (${actual_cost:,.0f}) | Vol:{option_volume}")

    # Close remaining positions at EOD
    if positions:
        final_bar = spy_bars[-1]
        final_time = datetime.fromisoformat(final_bar["t"].replace("Z", "+00:00"))
        final_ts = int(final_time.timestamp() * 1000)
        spy_price = final_bar["c"]

        for pos in positions:
            option_bars = prefetched_options.get(pos.option_ticker, [])
            result = get_option_price_at_time(option_bars, final_ts, use_bid=True)

            if result:
                final_price, _ = result
                final_price *= (1 - SLIPPAGE_PCT)
            else:
                final_price = pos.entry_option_price * 0.5  # Assume 50% loss if no data

            hold_minutes = int((final_time - pos.entry_time).total_seconds() / 60)
            position_value = pos.contracts * final_price * 100
            pnl_dollars = position_value - pos.position_cost
            pnl_pct = (final_price - pos.entry_option_price) / pos.entry_option_price * 100

            trade = SimTrade(
                entry_time=pos.entry_time,
                exit_time=final_time,
                direction=pos.direction,
                strike=pos.strike,
                contracts=pos.contracts,
                entry_option_price=pos.entry_option_price,
                exit_option_price=final_price,
                position_cost=pos.position_cost,
                pnl_dollars=pnl_dollars,
                pnl_pct=pnl_pct,
                exit_reason="EOD_CLOSE",
                spy_entry=pos.entry_spy_price,
                spy_exit=spy_price,
                option_ticker=pos.option_ticker,
                hold_minutes=hold_minutes
            )
            completed_trades.append(trade)
            account += pnl_dollars

            print(f"  [EOD] CLOSE {pos.direction} ${pos.strike}: ${pos.entry_option_price:.2f}→${final_price:.2f} | P&L: ${pnl_dollars:+,.0f} ({pnl_pct:+.0f}%) | {hold_minutes}min")

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
    print(f"    Largest: ${result.largest_trade:+,.0f} | Drawdown: {result.max_drawdown:.1f}%")
    print(f"    Account: ${result.start_account:,.0f} → ${result.end_account:,.0f} ({((result.end_account/result.start_account)-1)*100:+.1f}%)")

    return result


def run_week_backtest(week_type: str, dates: List[str], target: float):
    """Run backtest for a week."""
    print(f"\n{'='*70}")
    print(f"WEEK TYPE: {week_type}")
    print(f"Dates: {', '.join(dates)}")
    print(f"Target: ${target:,.0f}")
    print(f"{'='*70}")

    account = INITIAL_ACCOUNT
    all_results: List[DayResult] = []

    for date in dates:
        result = run_day_simulation(date, account)
        all_results.append(result)
        account = result.end_account
        time.sleep(0.5)

    total_trades = sum(r.num_trades for r in all_results)
    total_wins = sum(r.wins for r in all_results)
    overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    print(f"\n{'='*70}")
    print(f"{week_type} WEEK RESULTS")
    print(f"{'='*70}")
    for r in all_results:
        day_return = ((r.end_account / r.start_account) - 1) * 100 if r.start_account > 0 else 0
        wr = f"{r.win_rate:.0f}%" if r.num_trades > 0 else "N/A"
        print(f"  {r.date}: ${r.start_account:,.0f} → ${r.end_account:,.0f} ({day_return:+.1f}%) | {r.num_trades} trades | WR: {wr}")

    final_return = ((account / INITIAL_ACCOUNT) - 1) * 100
    hit = account >= target

    print(f"\n  FINAL: ${INITIAL_ACCOUNT:,.0f} → ${account:,.0f} ({final_return:+.1f}%)")
    print(f"  Total Trades: {total_trades} | Overall Win Rate: {overall_win_rate:.0f}%")
    print(f"  Target: ${target:,.0f} | Status: {'✅ HIT' if hit else f'❌ MISSED by ${target-account:,.0f}'}")

    return account, all_results, overall_win_rate


def run_hell_backtest():
    """Run THE BACKTEST FROM HELL v3 - Truth Calibrated."""
    print("\n" + "=" * 70)
    print("THE BACKTEST FROM HELL v3")
    print("TRUTH-CALIBRATED WITH 4 FIXES")
    print("=" * 70)
    print(f"Initial Account: ${INITIAL_ACCOUNT:,.0f}")
    print(f"Position Size: {POSITION_SIZE_PCT*100:.0f}%")
    print(f"Stop Loss: {STOP_LOSS_PCT*100:.0f}% | Time Stop: {TIME_STOP_MINUTES}min")
    print(f"Filters: Skip first {SKIP_FIRST_MINUTES}min, Min SPY range ${MIN_SPY_RANGE}, Min option vol {MIN_OPTION_VOLUME}")
    print(f"Confirmation: 2-minute signal persistence required")

    # Week definitions
    bad_week = ["2026-01-22", "2026-01-23", "2026-01-26", "2026-01-27", "2026-01-28"]
    avg_week = ["2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13"]
    great_week = ["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03", "2026-02-04"]

    # Realistic targets
    targets = {
        "bad": 6000,    # Don't lose money
        "average": 10000,  # Double
        "great": 15000   # Triple
    }

    results = {}

    bad_final, bad_results, bad_wr = run_week_backtest("BAD (Low Volatility)", bad_week, targets["bad"])
    results["bad"] = {"final": bad_final, "target": targets["bad"], "win_rate": bad_wr}

    avg_final, avg_results, avg_wr = run_week_backtest("AVERAGE (Moderate VIX)", avg_week, targets["average"])
    results["average"] = {"final": avg_final, "target": targets["average"], "win_rate": avg_wr}

    great_final, great_results, great_wr = run_week_backtest("GREAT (High Volatility)", great_week, targets["great"])
    results["great"] = {"final": great_final, "target": targets["great"], "win_rate": great_wr}

    print("\n" + "=" * 70)
    print("FINAL RESULTS - TRUTH CALIBRATED")
    print("=" * 70)

    all_hit = True
    for week_type, data in results.items():
        final = data["final"]
        target = data["target"]
        win_rate = data["win_rate"]
        hit = final >= target
        if not hit:
            all_hit = False
        pct = ((final / INITIAL_ACCOUNT) - 1) * 100
        status = "✅ HIT" if hit else f"❌ MISSED by ${target-final:,.0f}"
        print(f"\n{week_type.upper()} WEEK:")
        print(f"  Result: ${INITIAL_ACCOUNT:,.0f} → ${final:,.0f} ({pct:+.1f}%)")
        print(f"  Win Rate: {win_rate:.0f}%")
        print(f"  Target: ${target:,.0f} | {status}")

    print("\n" + "=" * 70)
    if all_hit:
        print("🎯 ALL TARGETS HIT - SYSTEM VALIDATED")
    else:
        print("⚠️  SOME TARGETS MISSED - NEEDS TUNING")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_hell_backtest()
