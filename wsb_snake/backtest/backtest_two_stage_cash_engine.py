#!/usr/bin/env python3
"""
Backtest a filtered QQQ-first cash engine on real Polygon minute data.

Engine shape:
- Mid-morning signal at 10:00 ET, confirm at 10:15 ET
- Afternoon signal at 13:00 ET, confirm at 13:15 ET
- Optional close signal at 15:00 ET, confirm at 15:30 ET
- Moderate confirmed moves only; extreme moves are treated as exhaustion
- QQQ first, optional SPY fallback if explicitly requested
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Allow direct script execution from the repo root or by absolute path.
if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from wsb_snake.backtest.polygon_option_replay import (
    BACKTEST_DIR,
    ET,
    PolygonClient,
    find_bar_index,
    iter_weekdays,
    pick_option_contract,
)


@dataclass
class CashTrade:
    date: str
    window: str
    ticker: str
    direction: str
    signal_pct: float
    confirm_pct: float
    benchmark_signal_pct: Optional[float]
    benchmark_confirm_pct: Optional[float]
    option_symbol: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    qty: int
    pnl_dollars: float
    pnl_pct: float
    exit_reason: str
    hold_minutes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--polygon-key", default=os.getenv("POLYGON_API_KEY", ""))
    parser.add_argument("--tickers", default="QQQ")
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--morning-threshold", type=float, default=0.004)
    parser.add_argument("--afternoon-threshold", type=float, default=0.0045)
    parser.add_argument("--morning-signal-time", default="10:00")
    parser.add_argument("--morning-confirm-time", default="10:15")
    parser.add_argument("--afternoon-signal-time", default="13:00")
    parser.add_argument("--afternoon-confirm-time", default="13:15")
    parser.add_argument("--close-signal-time", default="")
    parser.add_argument("--close-confirm-time", default="")
    parser.add_argument("--close-threshold", type=float, default=0.0)
    parser.add_argument("--close-max-hold-minutes", type=int, default=None)
    parser.add_argument("--close-call-max-confirm-abs-pct", type=float, default=None)
    parser.add_argument("--close-put-max-confirm-abs-pct", type=float, default=None)
    parser.add_argument("--close-call-benchmark-ratio-min", type=float, default=None)
    parser.add_argument("--close-put-benchmark-ratio-min", type=float, default=None)
    parser.add_argument("--close-allowed-weekdays", default="")
    parser.add_argument("--position-size-usd", type=float, default=1000.0)
    parser.add_argument("--target-premium", type=float, default=1.25)
    parser.add_argument("--profit-target-pct", type=float, default=0.30)
    parser.add_argument("--stop-loss-pct", type=float, default=0.15)
    parser.add_argument("--max-hold-minutes", type=int, default=25)
    parser.add_argument("--slippage-pct", type=float, default=0.02)
    parser.add_argument("--call-max-confirm-abs-pct", type=float, default=0.008)
    parser.add_argument("--put-max-confirm-abs-pct", type=float, default=0.0072)
    parser.add_argument("--call-benchmark-ratio-min", type=float, default=0.5)
    parser.add_argument("--put-benchmark-ratio-min", type=float, default=0.0)
    parser.add_argument("--output", default="qqq_hybrid_cash_engine.json")
    return parser.parse_args()


def parse_clock(trade_date: str, clock: str) -> datetime:
    return datetime.strptime(f"{trade_date} {clock}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)


def parse_allowed_weekdays(value: str) -> Optional[set[str]]:
    if not value.strip():
        return None
    return {item.strip().title()[:3] for item in value.split(",") if item.strip()}


def bar_close_at_or_before(bars: List[dict], target_time: datetime) -> Optional[dict]:
    eligible = [row for row in bars if row["et"] <= target_time]
    return eligible[-1] if eligible else None


def percent_from_open(bars: List[dict], target_time: datetime) -> Optional[float]:
    if not bars:
        return None
    session_open = next((row for row in bars if row["et"].hour == 9 and row["et"].minute == 30), None)
    close_bar = bar_close_at_or_before(bars, target_time)
    if not session_open or not close_bar:
        return None
    open_price = float(session_open["o"])
    close_price = float(close_bar["c"])
    if open_price <= 0:
        return None
    return (close_price - open_price) / open_price


def stage_signal(
    client: PolygonClient,
    args: argparse.Namespace,
    tickers: List[str],
    trade_date: str,
    signal_time: datetime,
    confirm_time: datetime,
    threshold: float,
) -> Optional[Dict]:
    benchmark_signal_pct = None
    benchmark_confirm_pct = None
    if args.benchmark_ticker:
        benchmark_bars = client.get_underlying_bars(args.benchmark_ticker, trade_date)
        benchmark_signal_pct = percent_from_open(benchmark_bars, signal_time)
        benchmark_confirm_pct = percent_from_open(benchmark_bars, confirm_time)

    signals: List[Dict] = []
    for ticker in tickers:
        bars = client.get_underlying_bars(ticker, trade_date)
        signal_pct = percent_from_open(bars, signal_time)
        confirm_pct = percent_from_open(bars, confirm_time)
        if signal_pct is None or confirm_pct is None:
            continue

        direction = None
        if signal_pct >= threshold and confirm_pct >= signal_pct:
            direction = "CALL"
        elif signal_pct <= -threshold and confirm_pct <= signal_pct:
            direction = "PUT"

        if direction == "CALL" and args.call_max_confirm_abs_pct is not None:
            if abs(confirm_pct) > args.call_max_confirm_abs_pct:
                continue
        if direction == "PUT" and args.put_max_confirm_abs_pct is not None:
            if abs(confirm_pct) > args.put_max_confirm_abs_pct:
                continue

        if direction == "CALL" and args.call_benchmark_ratio_min > 0:
            if benchmark_confirm_pct is None or benchmark_confirm_pct <= 0:
                continue
            if abs(benchmark_confirm_pct) / abs(confirm_pct) < args.call_benchmark_ratio_min:
                continue
        if direction == "PUT" and args.put_benchmark_ratio_min > 0:
            if benchmark_confirm_pct is None or benchmark_confirm_pct >= 0:
                continue
            if abs(benchmark_confirm_pct) / abs(confirm_pct) < args.put_benchmark_ratio_min:
                continue

        if direction:
            signals.append(
                {
                    "ticker": ticker,
                    "direction": direction,
                    "signal_pct": signal_pct,
                    "confirm_pct": confirm_pct,
                    "benchmark_signal_pct": benchmark_signal_pct,
                    "benchmark_confirm_pct": benchmark_confirm_pct,
                    "bars": bars,
                }
            )

    if not signals:
        return None

    priority = {ticker: idx for idx, ticker in enumerate(tickers)}
    signals.sort(key=lambda row: (priority.get(row["ticker"], 99), -abs(row["confirm_pct"])))
    return signals[0]


def simulate_trade(
    client: PolygonClient,
    args: argparse.Namespace,
    trade_date: str,
    window: str,
    setup: Dict,
    entry_time: datetime,
) -> Optional[CashTrade]:
    bars = setup["bars"]
    entry_underlying_bar = next((row for row in bars if row["et"] >= entry_time), None)
    if not entry_underlying_bar:
        return None

    selected = pick_option_contract(
        client,
        setup["ticker"],
        trade_date,
        setup["direction"],
        float(entry_underlying_bar["o"]),
        args.target_premium,
        entry_underlying_bar["et"],
    )
    if not selected:
        return None

    option_symbol, option_entry_bar = selected
    option_bars = client.get_option_bars(option_symbol, trade_date)
    option_entry_idx = find_bar_index(option_bars, option_entry_bar["et"])
    if option_entry_idx is None:
        return None

    raw_entry = float(option_entry_bar["o"] or option_entry_bar["c"])
    entry_fill = raw_entry * (1 + args.slippage_pct)
    if entry_fill <= 0:
        return None

    qty = max(1, int(args.position_size_usd / (entry_fill * 100)))
    target_price = raw_entry * (1 + args.profit_target_pct)
    stop_price = raw_entry * (1 - args.stop_loss_pct)
    time_exit = entry_underlying_bar["et"] + timedelta(minutes=args.max_hold_minutes)

    exit_bar = option_entry_bar
    exit_reason = "TIME_EXIT"
    for idx in range(option_entry_idx, len(option_bars)):
        bar = option_bars[idx]
        if bar["et"] > time_exit:
            break
        if bar["l"] <= stop_price:
            exit_bar = bar
            exit_reason = "STOP_LOSS"
            break
        if bar["h"] >= target_price:
            exit_bar = bar
            exit_reason = "TARGET"
            break
        exit_bar = bar

    if exit_reason == "STOP_LOSS":
        raw_exit = min(stop_price, float(exit_bar["o"]))
    elif exit_reason == "TARGET":
        raw_exit = max(target_price, float(exit_bar["o"]))
    else:
        raw_exit = float(exit_bar["c"])

    exit_fill = raw_exit * (1 - args.slippage_pct)
    pnl_dollars = qty * (exit_fill - entry_fill) * 100
    pnl_pct = (exit_fill - entry_fill) / entry_fill if entry_fill else 0.0
    hold_minutes = max(0, int((exit_bar["et"] - option_entry_bar["et"]).total_seconds() // 60))

    return CashTrade(
        date=trade_date,
        window=window,
        ticker=setup["ticker"],
        direction=setup["direction"],
        signal_pct=setup["signal_pct"],
        confirm_pct=setup["confirm_pct"],
        benchmark_signal_pct=setup.get("benchmark_signal_pct"),
        benchmark_confirm_pct=setup.get("benchmark_confirm_pct"),
        option_symbol=option_symbol,
        entry_time=option_entry_bar["et"].isoformat(),
        exit_time=exit_bar["et"].isoformat(),
        entry_price=entry_fill,
        exit_price=exit_fill,
        qty=qty,
        pnl_dollars=pnl_dollars,
        pnl_pct=pnl_pct,
        exit_reason=exit_reason,
        hold_minutes=hold_minutes,
    )


def simulate_trade_with_overrides(
    client: PolygonClient,
    args: argparse.Namespace,
    trade_date: str,
    window: str,
    setup: Dict,
    entry_time: datetime,
    *,
    max_hold_minutes: Optional[int] = None,
) -> Optional[CashTrade]:
    original_max_hold = args.max_hold_minutes
    if max_hold_minutes is not None:
        args.max_hold_minutes = max_hold_minutes
    try:
        return simulate_trade(client, args, trade_date, window, setup, entry_time)
    finally:
        args.max_hold_minutes = original_max_hold


def summarize(trades: List[CashTrade]) -> Dict:
    winners = [trade for trade in trades if trade.pnl_dollars > 0]
    losers = [trade for trade in trades if trade.pnl_dollars <= 0]
    return {
        "trades": len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": len(winners) / len(trades) if trades else 0.0,
        "total_pnl_dollars": sum(trade.pnl_dollars for trade in trades),
        "avg_winner_pct": sum(trade.pnl_pct for trade in winners) / len(winners) if winners else 0.0,
        "avg_loser_pct": sum(trade.pnl_pct for trade in losers) / len(losers) if losers else 0.0,
    }


def run_backtest(args: argparse.Namespace) -> Dict:
    tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()]
    client = PolygonClient(args.polygon_key)
    trades: List[CashTrade] = []
    skipped_days: Dict[str, str] = {}
    close_allowed_weekdays = parse_allowed_weekdays(args.close_allowed_weekdays)

    for trade_date in iter_weekdays(args.start, args.end):
        day_trades = 0
        day_has_winner = False

        morning_signal = parse_clock(trade_date, args.morning_signal_time)
        morning_confirm = parse_clock(trade_date, args.morning_confirm_time)
        morning_setup = stage_signal(
            client,
            args,
            tickers,
            trade_date,
            morning_signal,
            morning_confirm,
            args.morning_threshold,
        )
        morning_trade = None
        if morning_setup:
            morning_trade = simulate_trade(
                client,
                args,
                trade_date,
                "morning",
                morning_setup,
                morning_confirm + timedelta(minutes=1),
            )
            if morning_trade:
                trades.append(morning_trade)
                day_trades += 1
                day_has_winner = morning_trade.pnl_dollars > 0

        allow_afternoon = morning_trade is None or morning_trade.pnl_dollars <= 0
        if allow_afternoon:
            afternoon_signal = parse_clock(trade_date, args.afternoon_signal_time)
            afternoon_confirm = parse_clock(trade_date, args.afternoon_confirm_time)
            afternoon_setup = stage_signal(
                client,
                args,
                tickers,
                trade_date,
                afternoon_signal,
                afternoon_confirm,
                args.afternoon_threshold,
            )
            if afternoon_setup and day_trades < 2:
                afternoon_trade = simulate_trade(
                    client,
                    args,
                    trade_date,
                    "afternoon",
                    afternoon_setup,
                    afternoon_confirm + timedelta(minutes=1),
                )
                if afternoon_trade:
                    trades.append(afternoon_trade)
                    day_trades += 1
                    if afternoon_trade.pnl_dollars > 0:
                        day_has_winner = True

        allow_close = (
            bool(args.close_signal_time and args.close_confirm_time and args.close_threshold > 0)
            and not day_has_winner
        )
        if allow_close:
            weekday = datetime.fromisoformat(trade_date).strftime("%a")
            if close_allowed_weekdays is None or weekday in close_allowed_weekdays:
                original_call_cap = args.call_max_confirm_abs_pct
                original_put_cap = args.put_max_confirm_abs_pct
                original_call_bench = args.call_benchmark_ratio_min
                original_put_bench = args.put_benchmark_ratio_min
                if args.close_call_max_confirm_abs_pct is not None:
                    args.call_max_confirm_abs_pct = args.close_call_max_confirm_abs_pct
                if args.close_put_max_confirm_abs_pct is not None:
                    args.put_max_confirm_abs_pct = args.close_put_max_confirm_abs_pct
                if args.close_call_benchmark_ratio_min is not None:
                    args.call_benchmark_ratio_min = args.close_call_benchmark_ratio_min
                if args.close_put_benchmark_ratio_min is not None:
                    args.put_benchmark_ratio_min = args.close_put_benchmark_ratio_min
                try:
                    close_signal = parse_clock(trade_date, args.close_signal_time)
                    close_confirm = parse_clock(trade_date, args.close_confirm_time)
                    close_setup = stage_signal(
                        client,
                        args,
                        tickers,
                        trade_date,
                        close_signal,
                        close_confirm,
                        args.close_threshold,
                    )
                    if close_setup and day_trades < 3:
                        close_trade = simulate_trade_with_overrides(
                            client,
                            args,
                            trade_date,
                            "close",
                            close_setup,
                            close_confirm + timedelta(minutes=1),
                            max_hold_minutes=args.close_max_hold_minutes,
                        )
                        if close_trade:
                            trades.append(close_trade)
                            day_trades += 1
                finally:
                    args.call_max_confirm_abs_pct = original_call_cap
                    args.put_max_confirm_abs_pct = original_put_cap
                    args.call_benchmark_ratio_min = original_call_bench
                    args.put_benchmark_ratio_min = original_put_bench

        if day_trades == 0:
            skipped_days[trade_date] = "no_confirmed_signal"

    return {
        "config": vars(args),
        "summary": summarize(trades),
        "trades": [asdict(trade) for trade in trades],
        "skipped_days": skipped_days,
    }


def main() -> None:
    args = parse_args()
    result = run_backtest(args)
    output_path = BACKTEST_DIR / args.output
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"saved {output_path}")


if __name__ == "__main__":
    main()
