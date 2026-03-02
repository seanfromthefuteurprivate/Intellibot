#!/usr/bin/env python3
"""
BACKTEST FROM HELL v7: V2 CORE + 3 FIXES
V7 = V2 base + MA alignment + 3 fixes (circuit breaker, $0.50 min, conviction sizing)
NO confirmation delay - enter immediately on momentum + volume + MA alignment
"""
import os, requests, time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

ALPACA_KEY = os.environ.get("ALPACA_API_KEY", "PKWT6T5BFKHBTFDW3CPAFW2XBZ")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "pVdzbVte2pQvL1RmCTFw3oaQ6TBWYimAzC42DUyTEy8")
POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "QJWtaUQV7N8mytTI7PH26lX3Ju6PD2iq")

# V2 parameters
INITIAL_ACCOUNT = 5000.0
SLIPPAGE_PCT = 0.02
MAX_POSITIONS = 3
MIN_ACCOUNT_FLOOR = 500.0
OTM_OFFSET = 2
TRAIL_BREAKEVEN, TRAIL_LOCK, TRAIL_AGGRESSIVE, TRAIL_MOONSHOT = 0.50, 1.00, 2.00, 5.00
PYRAMID_TRIGGER, PYRAMID_ADD_PCT = 1.00, 0.25

# FIX 2 & 3
MIN_ENTRY_PRICE = 0.50
DEFAULT_SIZE_PCT, HIGH_CONVICTION_SIZE_PCT = 0.15, 0.25
MAX_TOTAL_EXPOSURE = 0.40
HIGH_CONVICTION_MOMENTUM, HIGH_CONVICTION_VOLUME = 0.005, 2.0

OPTION_BARS_CACHE: Dict[str, List[Dict]] = {}

@dataclass
class SimPosition:
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
    conviction: str = "DEFAULT"

@dataclass
class SimTrade:
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
    conviction: str
    hold_minutes: int = 0

@dataclass
class DayResult:
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
    circuit_breaker_fired: bool = False
    signals_blocked: int = 0

def build_option_ticker(strike: float, expiry: str, direction: str) -> str:
    expiry_fmt = datetime.strptime(expiry, "%Y-%m-%d").strftime("%y%m%d")
    cp = "C" if direction == "CALL" else "P"
    return f"O:SPY{expiry_fmt}{cp}{int(strike * 1000):08d}"

def fetch_option_bars(option_ticker: str, date: str) -> List[Dict]:
    cache_key = f"{option_ticker}_{date}"
    if cache_key in OPTION_BARS_CACHE:
        return OPTION_BARS_CACHE[cache_key]
    url = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/minute/{date}/{date}"
    try:
        r = requests.get(url, params={"apiKey": POLYGON_KEY, "limit": 50000, "sort": "asc"}, timeout=10)
        if r.status_code == 200:
            bars = r.json().get("results", [])
            OPTION_BARS_CACHE[cache_key] = bars
            return bars
    except Exception:
        pass
    return []

def get_option_price_at_time(option_bars: List[Dict], target_ts: int, use_bid: bool = False) -> Optional[float]:
    if not option_bars: return None
    best_bar, best_diff = None, float('inf')
    for bar in option_bars:
        diff = abs(bar["t"] - target_ts)
        if diff < best_diff:
            best_diff, best_bar = diff, bar
        if bar["t"] > target_ts and best_bar: break
    if best_bar and best_diff <= 120000:
        return best_bar["l"] if use_bid else best_bar["c"]
    return None

def fetch_spy_minute_bars(date: str) -> List[Dict]:
    headers = {"APCA-API-KEY-ID": ALPACA_KEY, "APCA-API-SECRET-KEY": ALPACA_SECRET}
    url = "https://data.alpaca.markets/v2/stocks/SPY/bars"
    params = {"timeframe": "1Min", "start": f"{date}T09:30:00Z", "end": f"{date}T16:00:00Z", "limit": 500, "feed": "iex"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200: return r.json().get("bars", [])
    except Exception:
        pass
    return []

def calculate_ma(bars: List[Dict], idx: int, period: int) -> Optional[float]:
    if idx < period: return None
    return sum(b["c"] for b in bars[idx-period+1:idx+1]) / period

def detect_signal_v7(bars: List[Dict], idx: int) -> Optional[Tuple[str, float, float, str]]:
    """V2 signal: momentum + volume + MA alignment. NO confirmation delay."""
    if idx < 10: return None
    current_bar = bars[idx]
    bar_time = datetime.fromisoformat(current_bar["t"].replace("Z", "+00:00"))
    market_open = bar_time.replace(hour=9, minute=30, second=0, microsecond=0)
    if (bar_time - market_open).total_seconds() / 60 < 5: return None

    recent = bars[max(0, idx-10):idx+1]
    if len(recent) < 10: return None

    current_price = recent[-1]["c"]
    momentum = (recent[-1]["c"] - recent[0]["c"]) / recent[0]["c"]
    avg_vol = sum(b["v"] for b in recent[:-1]) / (len(recent) - 1) if len(recent) > 1 else 1
    vol_spike = recent[-1]["v"] / avg_vol if avg_vol > 0 else 1
    bar_range = (recent[-1]["h"] - recent[-1]["l"]) / recent[-1]["c"]

    ma9, ma20, ma30, ma50 = calculate_ma(bars, idx, 9), calculate_ma(bars, idx, 20), calculate_ma(bars, idx, 30), calculate_ma(bars, idx, 50)
    direction, confidence = None, 0

    # V2 signal: momentum > 0.2% + volume spike 1.3x + MA alignment = ENTER IMMEDIATELY
    if momentum > 0.002 and vol_spike > 1.3:
        if ma30 and current_price < ma30: return None  # MA alignment check
        direction, confidence = "CALL", min(90, 50 + momentum * 2000 + vol_spike * 10 + bar_range * 500)
        strike = round(current_price) + OTM_OFFSET
    elif momentum < -0.002 and vol_spike > 1.3:
        if ma30 and current_price > ma30: return None  # MA alignment check
        direction, confidence = "PUT", min(90, 50 + abs(momentum) * 2000 + vol_spike * 10 + bar_range * 500)
        strike = round(current_price) - OTM_OFFSET

    if not direction or confidence < 62: return None

    # Conviction check for sizing
    conviction = "DEFAULT"
    if ma9 and ma20 and ma50:
        ma_aligned = (current_price > ma9 > ma20 > ma50) if direction == "CALL" else (current_price < ma9 < ma20 < ma50)
        if abs(momentum) > HIGH_CONVICTION_MOMENTUM and vol_spike > HIGH_CONVICTION_VOLUME and ma_aligned:
            conviction = "HIGH"
    return (direction, confidence, strike, conviction)

def simulate_option_price(spy_price: float, strike: float, direction: str, entry_option_price: float, entry_spy_price: float) -> float:
    spy_move = spy_price - entry_spy_price
    favorable_move = spy_move if direction == "CALL" else -spy_move
    current_distance = (strike - spy_price) if direction == "CALL" else (spy_price - strike)

    if current_distance <= 0: delta = 0.65 + min(0.30, abs(current_distance) * 0.05)
    elif current_distance <= 1: delta = 0.45
    elif current_distance <= 2: delta = 0.30
    else: delta = 0.15

    option_move = favorable_move * delta
    if abs(favorable_move) > 4: option_move *= 2.5
    elif abs(favorable_move) > 2: option_move *= 1.8
    elif abs(favorable_move) > 1: option_move *= 1.3

    new_price = entry_option_price + option_move
    if abs(favorable_move) < 0.5: new_price *= 0.90
    new_price = max(0.01, new_price)
    if current_distance < 0: new_price = max(new_price, abs(current_distance) * 0.95)
    return round(new_price, 2)

def run_day_simulation(date: str, start_account: float) -> DayResult:
    print(f"\n{'='*60}\nSIMULATING: {date} [V7]\nStarting: ${start_account:,.2f}\n{'='*60}")

    spy_bars = fetch_spy_minute_bars(date)
    if not spy_bars:
        return DayResult(date=date, trades=[], num_trades=0, wins=0, losses=0, win_rate=0, avg_winner=0, avg_loser=0, largest_trade=0, end_account=start_account, start_account=start_account, max_drawdown=0)

    print(f"  {len(spy_bars)} bars | Range: ${min(b['l'] for b in spy_bars):.2f}-${max(b['h'] for b in spy_bars):.2f}")

    expiry, account, peak_account, max_drawdown = date, start_account, start_account, 0
    positions, completed_trades, prefetched = [], [], {}
    consecutive_losses, circuit_breaker_fired, signals_blocked = 0, False, 0

    spy_mid = (max(b["h"] for b in spy_bars) + min(b["l"] for b in spy_bars)) / 2
    for strike in range(int(spy_mid) - 5, int(spy_mid) + 6):
        for d in ["CALL", "PUT"]:
            ticker = build_option_ticker(strike, expiry, d)
            bars = fetch_option_bars(ticker, date)
            if bars: prefetched[ticker] = bars
    print(f"  Pre-fetched {len(prefetched)} contracts")

    for i, bar in enumerate(spy_bars):
        bar_time = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        bar_ts = int(bar_time.timestamp() * 1000)
        spy_price = bar["c"]

        # Check positions
        for pos in positions[:]:
            option_bars = prefetched.get(pos.option_ticker, [])
            real_price = get_option_price_at_time(option_bars, bar_ts, use_bid=True)
            current_price = real_price if real_price else simulate_option_price(spy_price, pos.strike, pos.direction, pos.entry_option_price, pos.entry_spy_price)

            if current_price > pos.peak_option_price: pos.peak_option_price = current_price
            pnl_pct = (current_price - pos.entry_option_price) / pos.entry_option_price

            exit_reason = None
            if pnl_pct <= -0.40: exit_reason = "STOP"
            elif pnl_pct >= TRAIL_MOONSHOT and (pos.peak_option_price - current_price) / pos.peak_option_price > 0.08: exit_reason = "TRAIL"
            elif pnl_pct >= TRAIL_AGGRESSIVE and (pos.peak_option_price - current_price) / pos.peak_option_price > 0.12: exit_reason = "TRAIL"
            elif pnl_pct >= TRAIL_LOCK and (pos.peak_option_price - current_price) / pos.peak_option_price > 0.20: exit_reason = "TRAIL"
            elif pnl_pct >= TRAIL_BREAKEVEN and pnl_pct < 0.10: exit_reason = "TRAIL_BE"

            if exit_reason:
                hold_min = int((bar_time - pos.entry_time).total_seconds() / 60)
                exit_price = current_price * (1 - SLIPPAGE_PCT)
                pnl_dollars = pos.contracts * exit_price * 100 - pos.position_cost
                trade = SimTrade(pos.entry_time, bar_time, pos.direction, pos.strike, pos.contracts, pos.entry_option_price, exit_price, pos.position_cost, pnl_dollars, pnl_pct*100, exit_reason, pos.entry_spy_price, spy_price, pos.option_ticker, pos.conviction, hold_min)
                completed_trades.append(trade)
                account += pnl_dollars
                positions.remove(pos)

                if pnl_dollars > 0: consecutive_losses = 0
                else:
                    consecutive_losses += 1
                    if consecutive_losses >= 2:
                        circuit_breaker_fired = True
                        print(f"  [{bar_time.strftime('%H:%M')}] CIRCUIT_BREAKER: 2 losses, trading halted")

                print(f"  [{bar_time.strftime('%H:%M')}] EXIT {pos.direction} ${pos.strike}: {exit_reason} P&L ${pnl_dollars:+,.0f} [{pos.conviction}]")
                if account > peak_account: peak_account = account
                dd = (peak_account - account) / peak_account if peak_account > 0 else 0
                if dd > max_drawdown: max_drawdown = dd

        if circuit_breaker_fired:
            # Count blocked signals (any valid signal after CB fires)
            signal = detect_signal_v7(spy_bars, i)
            if signal: signals_blocked += 1
            continue

        if account < MIN_ACCOUNT_FLOOR: continue

        current_exposure = sum(p.position_cost for p in positions) / account if account > 0 else 0

        # Check for signal - ENTER IMMEDIATELY on raw signal (no confirmation)
        signal = None
        if len(positions) < MAX_POSITIONS and current_exposure < MAX_TOTAL_EXPOSURE:
            signal = detect_signal_v7(spy_bars, i)

        if signal:
            direction, confidence, strike, conviction = signal
            option_ticker = build_option_ticker(strike, expiry, direction)

            option_bars = prefetched.get(option_ticker) or fetch_option_bars(option_ticker, date)
            if option_bars and option_ticker not in prefetched: prefetched[option_ticker] = option_bars

            real_entry = get_option_price_at_time(option_bars, bar_ts) if option_bars else None
            if real_entry: option_price = real_entry * (1 + SLIPPAGE_PCT)
            else:
                dist = abs(strike - spy_price)
                option_price = (2.00 if dist <= 0 else 0.75 if dist <= 1 else 0.30 if dist <= 2 else 0.15) * (1 + SLIPPAGE_PCT)

            if option_price < MIN_ENTRY_PRICE:
                print(f"  [{bar_time.strftime('%H:%M')}] BLOCKED: ${option_price:.2f} < $0.50")
                continue

            size_pct = HIGH_CONVICTION_SIZE_PCT if conviction == "HIGH" else DEFAULT_SIZE_PCT
            size_pct = min(size_pct, MAX_TOTAL_EXPOSURE - current_exposure)
            if size_pct < 0.05: continue

            ratio = account / INITIAL_ACCOUNT
            mult = 0.5 if ratio < 0.5 else 0.75 if ratio < 0.8 else 1.5 if ratio > 2.0 else 1.25 if ratio > 1.5 else 1.0

            contracts = max(1, int(account * size_pct * mult / (option_price * 100)))
            actual_cost = contracts * option_price * 100

            pos = SimPosition(bar_time, bar_ts, spy_price, direction, strike, expiry, option_ticker, contracts, option_price, actual_cost, option_price, 0, contracts, conviction)
            positions.append(pos)
            print(f"  [{bar_time.strftime('%H:%M')}] ENTRY {direction} ${strike}: {contracts}x @ ${option_price:.2f} [{conviction}]")

        # Pyramiding
        if circuit_breaker_fired or account < MIN_ACCOUNT_FLOOR: continue
        for pos in positions:
            if pos.pyramid_adds < 1:
                option_bars = prefetched.get(pos.option_ticker, [])
                real_price = get_option_price_at_time(option_bars, bar_ts)
                current_price = real_price if real_price else simulate_option_price(spy_price, pos.strike, pos.direction, pos.entry_option_price, pos.entry_spy_price)
                if (current_price - pos.entry_option_price) / pos.entry_option_price >= PYRAMID_TRIGGER:
                    add = max(1, int(pos.original_contracts * PYRAMID_ADD_PCT))
                    pos.contracts += add
                    pos.position_cost += add * current_price * 100 * (1 + SLIPPAGE_PCT)
                    pos.pyramid_adds += 1
                    print(f"  [{bar_time.strftime('%H:%M')}] PYRAMID {pos.direction} ${pos.strike}: +{add}x")

    # EOD close
    final_bar = spy_bars[-1]
    final_time = datetime.fromisoformat(final_bar["t"].replace("Z", "+00:00"))
    final_ts = int(final_time.timestamp() * 1000)

    for pos in positions:
        option_bars = prefetched.get(pos.option_ticker, [])
        real_price = get_option_price_at_time(option_bars, final_ts, use_bid=True)
        final_price = (real_price if real_price else simulate_option_price(final_bar["c"], pos.strike, pos.direction, pos.entry_option_price, pos.entry_spy_price)) * (1 - SLIPPAGE_PCT)
        hold_min = int((final_time - pos.entry_time).total_seconds() / 60)
        pnl_dollars = pos.contracts * final_price * 100 - pos.position_cost
        pnl_pct = (final_price - pos.entry_option_price) / pos.entry_option_price * 100
        trade = SimTrade(pos.entry_time, final_time, pos.direction, pos.strike, pos.contracts, pos.entry_option_price, final_price, pos.position_cost, pnl_dollars, pnl_pct, "EOD", pos.entry_spy_price, final_bar["c"], pos.option_ticker, pos.conviction, hold_min)
        completed_trades.append(trade)
        account += pnl_dollars
        print(f"  [EOD] CLOSE {pos.direction} ${pos.strike}: P&L ${pnl_dollars:+,.0f} [{pos.conviction}]")

    wins = [t for t in completed_trades if t.pnl_dollars > 0]
    losses = [t for t in completed_trades if t.pnl_dollars <= 0]
    result = DayResult(date, completed_trades, len(completed_trades), len(wins), len(losses),
        len(wins)/len(completed_trades)*100 if completed_trades else 0,
        sum(t.pnl_pct for t in wins)/len(wins) if wins else 0,
        sum(t.pnl_pct for t in losses)/len(losses) if losses else 0,
        max((t.pnl_dollars for t in completed_trades), default=0),
        account, start_account, max_drawdown*100, circuit_breaker_fired, signals_blocked)

    cb_msg = f" | CB: {signals_blocked} blocked" if circuit_breaker_fired else ""
    print(f"\n  SUMMARY: {result.num_trades} trades | {result.wins} wins | ${start_account:,.0f}->${account:,.0f} ({(account/start_account-1)*100:+.1f}%){cb_msg}")
    return result

def run_week_backtest(week_type: str, dates: List[str], target: float):
    print(f"\n{'='*70}\nWEEK: {week_type} | Target: ${target:,.0f}\n{'='*70}")

    account, all_results, all_trades = INITIAL_ACCOUNT, [], []
    for date in dates:
        result = run_day_simulation(date, account)
        all_results.append(result)
        all_trades.extend(result.trades)
        account = result.end_account
        time.sleep(0.5)

    total_trades = sum(r.num_trades for r in all_results)
    total_wins = sum(r.wins for r in all_results)
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    cb_days = sum(1 for r in all_results if r.circuit_breaker_fired)

    default_trades = [t for t in all_trades if t.conviction == "DEFAULT"]
    high_trades = [t for t in all_trades if t.conviction == "HIGH"]

    print(f"\n{'='*70}\n{week_type} RESULTS\n{'='*70}")
    for r in all_results:
        cb = " [CB]" if r.circuit_breaker_fired else ""
        print(f"  {r.date}: ${r.start_account:,.0f}->${r.end_account:,.0f} ({(r.end_account/r.start_account-1)*100:+.1f}%) | {r.num_trades}T | {r.win_rate:.0f}%WR{cb}")

    if default_trades:
        print(f"\n  DEFAULT(15%): {len(default_trades)}T | ${sum(t.pnl_dollars for t in default_trades):+,.0f}")
    if high_trades:
        print(f"  HIGH(25%): {len(high_trades)}T | ${sum(t.pnl_dollars for t in high_trades):+,.0f}")
    if cb_days:
        print(f"  CIRCUIT BREAKER: {cb_days} days, {sum(r.signals_blocked for r in all_results)} signals blocked")

    hit = account >= target
    print(f"\n  FINAL: ${INITIAL_ACCOUNT:,.0f}->${account:,.0f} ({(account/INITIAL_ACCOUNT-1)*100:+.1f}%) | {total_trades}T | {wr:.0f}%WR")
    print(f"  Target: ${target:,.0f} | {'HIT' if hit else f'MISSED ${target-account:,.0f}'}")
    return account, all_results, wr, all_trades

def run_hell_backtest():
    print(f"\n{'='*70}\nBACKTEST FROM HELL v7: V2+V3+3FIXES\n{'='*70}")
    print(f"Size: {DEFAULT_SIZE_PCT*100:.0f}%/{HIGH_CONVICTION_SIZE_PCT*100:.0f}% | Max: {MAX_TOTAL_EXPOSURE*100:.0f}% | Min: ${MIN_ENTRY_PRICE} | CB: 2 losses")

    # V2 ORIGINAL WEEKS - same as V2 for fair comparison
    weeks = {
        "BAD": (["2026-01-22", "2026-01-23", "2026-01-26", "2026-01-27", "2026-01-28"], 5100),
        "AVERAGE": (["2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13"], 8000),
        "GREAT": (["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03", "2026-02-04"], 10000),
    }

    results, all_trades = {}, []
    for name, (dates, target) in weeks.items():
        final, _, wr, trades = run_week_backtest(name, dates, target)
        results[name] = {"final": final, "target": target, "wr": wr}
        all_trades.extend(trades)

    print(f"\n{'='*70}\nFINAL RESULTS\n{'='*70}")
    all_hit = True
    for name, data in results.items():
        hit = data["final"] >= data["target"]
        if not hit: all_hit = False
        print(f"\n{name}: ${INITIAL_ACCOUNT:,.0f}->${data['final']:,.0f} ({(data['final']/INITIAL_ACCOUNT-1)*100:+.1f}%) | {data['wr']:.0f}%WR | {'HIT' if hit else 'MISS'}")

    if all_trades:
        wins = [t for t in all_trades if t.pnl_dollars > 0]
        losses = [t for t in all_trades if t.pnl_dollars <= 0]
        wr = len(wins) / len(all_trades)
        avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.pnl_pct for t in losses) / len(losses)) if losses else 0
        exp = (wr * avg_win) - ((1 - wr) * avg_loss)
        print(f"\n{'='*70}\nEXPECTANCY: {len(all_trades)}T | {wr*100:.0f}%WR | +{avg_win:.0f}%W / -{avg_loss:.0f}%L | {exp:+.0f}%/trade {'POSITIVE' if exp > 0 else 'NEGATIVE'}\n{'='*70}")

    print(f"\n{'ALL TARGETS HIT - V7 VALIDATED' if all_hit else 'SOME TARGETS MISSED'}")
    return results

if __name__ == "__main__":
    run_hell_backtest()
