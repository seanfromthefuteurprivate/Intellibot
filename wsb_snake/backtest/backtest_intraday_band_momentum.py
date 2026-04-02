#!/usr/bin/env python3
"""
Backtest a Concretum-style intraday band/VWAP momentum strategy on 0DTE options.

Research basis:
- Concretum's "Beat the Market" intraday momentum framework
- Dynamic bands around open / prior close
- VWAP confirmation
- Rebalancing only at fixed intervals

This adapts the underlying-share logic to a single 0DTE option position that is
opened when exposure turns on and closed when exposure turns off or flips.
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

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from wsb_snake.backtest.polygon_option_replay import (  # noqa: E402
    BACKTEST_DIR,
    ET,
    PolygonClient,
    iter_weekdays,
    pick_option_contract,
)


@dataclass
class BandTrade:
    date: str
    direction: str
    option_symbol: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    qty: int
    pnl_dollars: float
    pnl_pct: float
    exit_reason: str
    entry_underlying_price: float
    exit_underlying_price: float
    entry_move_pct: float
    exit_move_pct: float
    hold_minutes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--polygon-key", default=os.getenv("POLYGON_API_KEY", ""))
    parser.add_argument("--underlying", default="QQQ")
    parser.add_argument("--benchmark-ticker", default="")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--band-mult", type=float, default=1.0)
    parser.add_argument("--trade-freq", type=int, default=30)
    parser.add_argument("--first-check", default="10:00")
    parser.add_argument("--last-check", default="15:30")
    parser.add_argument("--target-premium", type=float, default=1.25)
    parser.add_argument("--position-size-usd", type=float, default=1000.0)
    parser.add_argument("--slippage-pct", type=float, default=0.02)
    parser.add_argument("--benchmark-ratio-min", type=float, default=0.0)
    parser.add_argument("--output", default="intraday_band_momentum.json")
    return parser.parse_args()


def parse_clock(trade_date: str, clock: str) -> datetime:
    return datetime.strptime(f"{trade_date} {clock}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)


def minute_index(bar_time: datetime) -> int:
    return (bar_time.hour * 60 + bar_time.minute) - (9 * 60 + 30)


def percent_from_open(bars: List[dict], target_time: datetime) -> Optional[float]:
    if not bars:
        return None
    session_open = next((row for row in bars if row["et"].hour == 9 and row["et"].minute == 30), None)
    close_bar = next((row for row in bars if row["et"] >= target_time), None)
    if not session_open or not close_bar:
        return None
    open_price = float(session_open["o"])
    close_price = float(close_bar["c"])
    if open_price <= 0:
        return None
    return (close_price - open_price) / open_price


def add_vwap(bars: List[dict]) -> List[dict]:
    cumulative_pv = 0.0
    cumulative_volume = 0.0
    enriched: List[dict] = []
    for bar in bars:
        hlc = (float(bar["h"]) + float(bar["l"]) + float(bar["c"])) / 3.0
        cumulative_pv += hlc * float(bar["v"])
        cumulative_volume += float(bar["v"])
        row = dict(bar)
        row["vwap"] = cumulative_pv / cumulative_volume if cumulative_volume > 0 else float(bar["c"])
        enriched.append(row)
    return enriched


def build_sigma_open(history: Dict[int, List[float]], lookback_days: int) -> Dict[int, Optional[float]]:
    sigma: Dict[int, Optional[float]] = {}
    for idx in range(390):
        values = history.get(idx, [])
        if len(values) >= lookback_days:
            sigma[idx] = sum(values[-lookback_days:]) / lookback_days
        else:
            sigma[idx] = None
    return sigma


def next_bar_at_or_after(rows: List[dict], target_time: datetime) -> Optional[dict]:
    return next((row for row in rows if row["et"] >= target_time), None)


def summarize(trades: List[BandTrade]) -> Dict:
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


def signal_for_bar(
    bar: dict,
    open_price: float,
    prior_close: float,
    sigma_value: Optional[float],
    band_mult: float,
) -> int:
    if sigma_value is None:
        return 0
    upper_band = max(open_price, prior_close) * (1 + band_mult * sigma_value)
    lower_band = min(open_price, prior_close) * (1 - band_mult * sigma_value)
    price = float(bar["c"])
    if price > upper_band and price > float(bar["vwap"]):
        return 1
    if price < lower_band and price < float(bar["vwap"]):
        return -1
    return 0


def run_backtest(args: argparse.Namespace) -> Dict:
    client = PolygonClient(args.polygon_key)
    all_days = list(iter_weekdays(args.start, args.end))
    underlying_by_day: Dict[str, List[dict]] = {}
    benchmark_by_day: Dict[str, List[dict]] = {}
    history_by_minute: Dict[int, List[float]] = {idx: [] for idx in range(390)}
    trades: List[BandTrade] = []
    skipped_days: Dict[str, str] = {}
    previous_day_close: Optional[float] = None

    for trade_date in all_days:
        bars_raw = client.get_underlying_bars(args.underlying, trade_date)
        bars = add_vwap(bars_raw)
        underlying_by_day[trade_date] = bars
        if args.benchmark_ticker:
            benchmark_by_day[trade_date] = client.get_underlying_bars(args.benchmark_ticker, trade_date)

    for trade_date in all_days:
        bars = underlying_by_day.get(trade_date, [])
        if not bars:
            skipped_days[trade_date] = "no_underlying_bars"
            continue

        if previous_day_close is None:
            day_open = float(bars[0]["o"])
            for bar in bars:
                idx = minute_index(bar["et"])
                if 0 <= idx < 390:
                    history_by_minute[idx].append(abs((float(bar["c"]) - day_open) / day_open))
            previous_day_close = float(bars[-1]["c"])
            skipped_days[trade_date] = "warmup"
            continue

        sigma_open = build_sigma_open(history_by_minute, args.lookback_days)
        if sigma_open.get(0) is None:
            day_open = float(bars[0]["o"])
            for bar in bars:
                idx = minute_index(bar["et"])
                if 0 <= idx < 390:
                    history_by_minute[idx].append(abs((float(bar["c"]) - day_open) / day_open))
            previous_day_close = float(bars[-1]["c"])
            skipped_days[trade_date] = "warmup"
            continue

        open_price = float(bars[0]["o"])
        benchmark_bars = benchmark_by_day.get(trade_date, [])
        current_position: Optional[Dict] = None
        trade_made = False

        first_check = parse_clock(trade_date, args.first_check)
        last_check = parse_clock(trade_date, args.last_check)
        current_check = first_check

        while current_check <= last_check:
            bar = next_bar_at_or_after(bars, current_check)
            if not bar:
                current_check += timedelta(minutes=args.trade_freq)
                continue

            idx = minute_index(bar["et"])
            signal = signal_for_bar(bar, open_price, previous_day_close, sigma_open.get(idx), args.band_mult)
            if signal != 0 and args.benchmark_ratio_min > 0 and benchmark_bars:
                benchmark_move = percent_from_open(benchmark_bars, bar["et"])
                underlying_move = percent_from_open(bars, bar["et"])
                if benchmark_move is None or underlying_move is None or underlying_move == 0:
                    signal = 0
                elif benchmark_move * underlying_move <= 0:
                    signal = 0
                elif abs(benchmark_move) / abs(underlying_move) < args.benchmark_ratio_min:
                    signal = 0

            next_check = current_check + timedelta(minutes=args.trade_freq)
            exit_probe_time = min(next_check + timedelta(minutes=1), bars[-1]["et"])

            if current_position and signal != current_position["signal"]:
                option_bars = client.get_option_bars(current_position["option_symbol"], trade_date)
                exit_bar = next_bar_at_or_after(option_bars, current_check + timedelta(minutes=1))
                if not exit_bar:
                    exit_bar = option_bars[-1] if option_bars else None
                underlying_exit = next_bar_at_or_after(bars, current_check + timedelta(minutes=1)) or bars[-1]
                if exit_bar:
                    raw_exit = float(exit_bar["c"])
                    exit_fill = raw_exit * (1 - args.slippage_pct)
                    pnl_dollars = current_position["qty"] * (exit_fill - current_position["entry_fill"]) * 100
                    pnl_pct = (exit_fill - current_position["entry_fill"]) / current_position["entry_fill"]
                    trades.append(
                        BandTrade(
                            date=trade_date,
                            direction=current_position["direction"],
                            option_symbol=current_position["option_symbol"],
                            signal_time=current_position["signal_time"].isoformat(),
                            entry_time=current_position["entry_bar"]["et"].isoformat(),
                            exit_time=exit_bar["et"].isoformat(),
                            entry_price=current_position["entry_fill"],
                            exit_price=exit_fill,
                            qty=current_position["qty"],
                            pnl_dollars=pnl_dollars,
                            pnl_pct=pnl_pct,
                            exit_reason="signal_flip" if signal != 0 else "signal_off",
                            entry_underlying_price=float(current_position["underlying_entry"]["o"]),
                            exit_underlying_price=float(underlying_exit["c"]),
                            entry_move_pct=current_position["entry_move_pct"],
                            exit_move_pct=percent_from_open(bars, underlying_exit["et"]) or 0.0,
                            hold_minutes=max(
                                0,
                                int((exit_bar["et"] - current_position["entry_bar"]["et"]).total_seconds() // 60),
                            ),
                        )
                    )
                    trade_made = True
                current_position = None

            if signal != 0 and current_position is None:
                direction = "CALL" if signal > 0 else "PUT"
                entry_underlying = next_bar_at_or_after(bars, current_check + timedelta(minutes=1))
                if not entry_underlying:
                    current_check += timedelta(minutes=args.trade_freq)
                    continue
                selected = pick_option_contract(
                    client,
                    args.underlying,
                    trade_date,
                    direction,
                    float(entry_underlying["o"]),
                    args.target_premium,
                    entry_underlying["et"],
                )
                if selected:
                    option_symbol, entry_bar = selected
                    raw_entry = float(entry_bar["o"] or entry_bar["c"])
                    entry_fill = raw_entry * (1 + args.slippage_pct)
                    qty = max(1, int(args.position_size_usd / (entry_fill * 100))) if entry_fill > 0 else 0
                    if qty > 0:
                        current_position = {
                            "signal": signal,
                            "direction": direction,
                            "signal_time": current_check,
                            "option_symbol": option_symbol,
                            "entry_bar": entry_bar,
                            "entry_fill": entry_fill,
                            "qty": qty,
                            "underlying_entry": entry_underlying,
                            "entry_move_pct": percent_from_open(bars, entry_underlying["et"]) or 0.0,
                        }

            current_check += timedelta(minutes=args.trade_freq)

        if current_position:
            option_bars = client.get_option_bars(current_position["option_symbol"], trade_date)
            exit_bar = option_bars[-1] if option_bars else None
            underlying_exit = bars[-1]
            if exit_bar:
                raw_exit = float(exit_bar["c"])
                exit_fill = raw_exit * (1 - args.slippage_pct)
                pnl_dollars = current_position["qty"] * (exit_fill - current_position["entry_fill"]) * 100
                pnl_pct = (exit_fill - current_position["entry_fill"]) / current_position["entry_fill"]
                trades.append(
                    BandTrade(
                        date=trade_date,
                        direction=current_position["direction"],
                        option_symbol=current_position["option_symbol"],
                        signal_time=current_position["signal_time"].isoformat(),
                        entry_time=current_position["entry_bar"]["et"].isoformat(),
                        exit_time=exit_bar["et"].isoformat(),
                        entry_price=current_position["entry_fill"],
                        exit_price=exit_fill,
                        qty=current_position["qty"],
                        pnl_dollars=pnl_dollars,
                        pnl_pct=pnl_pct,
                        exit_reason="eod",
                        entry_underlying_price=float(current_position["underlying_entry"]["o"]),
                        exit_underlying_price=float(underlying_exit["c"]),
                        entry_move_pct=current_position["entry_move_pct"],
                        exit_move_pct=percent_from_open(bars, underlying_exit["et"]) or 0.0,
                        hold_minutes=max(
                            0,
                            int((exit_bar["et"] - current_position["entry_bar"]["et"]).total_seconds() // 60),
                        ),
                    )
                )
                trade_made = True

        if not trade_made:
            skipped_days[trade_date] = "no_exposure_signal"

        for bar in bars:
            idx = minute_index(bar["et"])
            if 0 <= idx < 390:
                history_by_minute[idx].append(abs((float(bar["c"]) - open_price) / open_price))
        previous_day_close = float(bars[-1]["c"])

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
