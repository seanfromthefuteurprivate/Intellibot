#!/usr/bin/env python3
"""
Standalone paper runtime for the proven scheduled QQQ cash engine.

Strategy shape:
- QQQ only
- Morning window: 10:00 -> 10:15 ET
- Afternoon window: 13:00 -> 13:15 ET
- Close window: 15:00 -> 15:30 ET on Mon/Tue/Thu only
- Moderate confirmed moves only
- Fixed target/stop at +30% / -15%
- Custom time exits at 25 / 25 / 29 minutes

This runtime is intentionally separate from wsb_snake.main so it can be
paper-run cleanly without inheriting unrelated legacy orchestration.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

from wsb_snake.backtest.backtest_two_stage_cash_engine import (
    bar_close_at_or_before,
    parse_allowed_weekdays,
    parse_clock,
    stage_signal,
)
from wsb_snake.collectors.alpaca_intraday import ET, AlpacaIntradayClient
from wsb_snake.config import DATA_DIR
from wsb_snake.notifications.telegram_channels import send_alpaca_status
from wsb_snake.trading.alpaca_executor import AlpacaPosition, PositionStatus, alpaca_executor
from wsb_snake.trading.risk_governor import TradingEngine
from wsb_snake.utils.logger import get_logger

log = get_logger(__name__)

STRATEGY_NAME = "qqq_scheduled_cash"
STATE_PATH = Path(DATA_DIR) / f"{STRATEGY_NAME}_state.json"


@dataclass(frozen=True)
class WindowConfig:
    name: str
    signal_time: str
    confirm_time: str
    threshold: float
    max_hold_minutes: int
    call_max_confirm_abs_pct: float
    put_max_confirm_abs_pct: float
    call_benchmark_ratio_min: float = 0.5
    put_benchmark_ratio_min: float = 0.0
    allowed_weekdays: Optional[set[str]] = None
    execute: bool = True


@dataclass
class ManagedPositionState:
    window: str
    option_symbol: str
    position_id: str
    direction: str
    entered_at: str
    time_exit_at: str


@dataclass
class RuntimeState:
    trade_date: str
    window_status: Dict[str, str] = field(default_factory=dict)
    winning_windows: list[str] = field(default_factory=list)
    managed_positions: Dict[str, ManagedPositionState] = field(default_factory=dict)


WINDOWS = (
    WindowConfig(
        name="morning",
        signal_time="10:00",
        confirm_time="10:15",
        threshold=0.004,
        max_hold_minutes=25,
        call_max_confirm_abs_pct=0.008,
        put_max_confirm_abs_pct=0.0072,
    ),
    WindowConfig(
        name="afternoon",
        signal_time="13:00",
        confirm_time="13:15",
        threshold=0.0045,
        max_hold_minutes=25,
        call_max_confirm_abs_pct=0.008,
        put_max_confirm_abs_pct=0.0072,
    ),
    WindowConfig(
        name="close",
        signal_time="15:00",
        confirm_time="15:30",
        threshold=0.002,
        max_hold_minutes=29,
        call_max_confirm_abs_pct=0.012,
        put_max_confirm_abs_pct=0.012,
        allowed_weekdays=parse_allowed_weekdays("Mon,Tue,Thu"),
    ),
    WindowConfig(
        name="late_probe",
        signal_time="14:30",
        confirm_time="14:45",
        threshold=0.0025,
        max_hold_minutes=29,
        call_max_confirm_abs_pct=0.012,
        put_max_confirm_abs_pct=0.012,
        allowed_weekdays=parse_allowed_weekdays("Mon,Tue,Thu"),
        execute=False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="QQQ")
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--position-size-usd", type=float, default=1000.0)
    parser.add_argument("--max-daily-exposure-usd", type=float, default=2000.0)
    parser.add_argument("--profit-target-pct", type=float, default=0.30)
    parser.add_argument("--stop-loss-pct", type=float, default=0.15)
    parser.add_argument("--entry-grace-minutes", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--now-iso", default="")
    return parser.parse_args()


def now_et(now_iso: str = "") -> datetime:
    if now_iso:
        parsed = datetime.fromisoformat(now_iso)
        return parsed.astimezone(ET) if parsed.tzinfo else parsed.replace(tzinfo=ET)
    return datetime.now(ET)


def is_market_session(current: datetime) -> bool:
    if current.weekday() >= 5:
        return False
    market_open = current.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= current < market_close


def state_for_date(trade_date: str) -> RuntimeState:
    return RuntimeState(trade_date=trade_date)


def load_state(trade_date: str) -> RuntimeState:
    if not STATE_PATH.exists():
        return state_for_date(trade_date)

    try:
        raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if raw.get("trade_date") != trade_date:
            return state_for_date(trade_date)
        managed_positions = {
            key: ManagedPositionState(**value)
            for key, value in raw.get("managed_positions", {}).items()
        }
        return RuntimeState(
            trade_date=trade_date,
            window_status=dict(raw.get("window_status", {})),
            winning_windows=list(raw.get("winning_windows", [])),
            managed_positions=managed_positions,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to load %s: %s", STATE_PATH, exc)
        return state_for_date(trade_date)


def save_state(state: RuntimeState) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trade_date": state.trade_date,
        "window_status": state.window_status,
        "winning_windows": state.winning_windows,
        "managed_positions": {
            key: asdict(value) for key, value in state.managed_positions.items()
        },
    }
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_signal_args(
    benchmark_ticker: str,
    window: WindowConfig,
) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark_ticker=benchmark_ticker,
        call_max_confirm_abs_pct=window.call_max_confirm_abs_pct,
        put_max_confirm_abs_pct=window.put_max_confirm_abs_pct,
        call_benchmark_ratio_min=window.call_benchmark_ratio_min,
        put_benchmark_ratio_min=window.put_benchmark_ratio_min,
    )


def latest_spot_price(
    client: AlpacaIntradayClient,
    ticker: str,
    trade_date: str,
    current: datetime,
) -> Optional[float]:
    latest_trade = client.get_latest_trade_price(ticker)
    if latest_trade is not None:
        return latest_trade

    bars = client.get_underlying_bars(ticker, trade_date, current=current)
    bar = bar_close_at_or_before(bars, current)
    if not bar:
        return None
    return float(bar["c"])


def find_tracked_position(managed: ManagedPositionState) -> Optional[AlpacaPosition]:
    for position in alpaca_executor.positions.values():
        if position.position_id == managed.position_id or position.option_symbol == managed.option_symbol:
            return position
    return None


def option_mid_price(option_symbol: str) -> Optional[float]:
    quote = alpaca_executor.get_option_quote(option_symbol)
    if not quote:
        return None
    bid = float(quote.get("bp", 0) or 0)
    ask = float(quote.get("ap", 0) or 0)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    if ask > 0:
        return ask
    if bid > 0:
        return bid
    return None


def apply_runtime_risk_profile(args: argparse.Namespace) -> None:
    alpaca_executor.MAX_PER_TRADE = args.position_size_usd
    alpaca_executor.MAX_DAILY_EXPOSURE = args.max_daily_exposure_usd
    alpaca_executor.MAX_CONCURRENT_POSITIONS = 1
    alpaca_executor.SCALP_TARGET_PCT = 1.0 + args.profit_target_pct
    alpaca_executor.SCALP_STOP_PCT = 1.0 - args.stop_loss_pct
    alpaca_executor.SCALP_MAX_HOLD_MINUTES = max(window.max_hold_minutes for window in WINDOWS)


def restore_managed_positions(state: RuntimeState) -> None:
    if not state.managed_positions:
        return
    alpaca_executor.sync_existing_positions()
    if not alpaca_executor.running:
        alpaca_executor.start_monitoring()
    for managed in state.managed_positions.values():
        position = find_tracked_position(managed)
        if not position:
            continue
        position.engine = TradingEngine.MOMENTUM.value
        position.entry_time = datetime.fromisoformat(managed.entered_at)
        position.target_price = position.entry_price * alpaca_executor.SCALP_TARGET_PCT
        position.stop_loss = position.entry_price * alpaca_executor.SCALP_STOP_PCT


def settle_positions(state: RuntimeState, current: datetime) -> None:
    settled_keys: list[str] = []
    for key, managed in state.managed_positions.items():
        position = find_tracked_position(managed)
        if not position:
            settled_keys.append(key)
            continue

        if position.status == PositionStatus.CLOSED:
            if position.pnl > 0 and managed.window not in state.winning_windows:
                state.winning_windows.append(managed.window)
            state.window_status[managed.window] = f"closed:{position.exit_reason or 'unknown'}"
            settled_keys.append(key)
            continue

        if position.status != PositionStatus.OPEN:
            continue

        time_exit_at = datetime.fromisoformat(managed.time_exit_at)
        if current < time_exit_at:
            continue

        current_price = option_mid_price(position.option_symbol)
        if current_price is None:
            log.warning("No current quote for time exit %s", position.option_symbol)
            continue
        alpaca_executor.execute_exit(position, f"TIME EXIT ({managed.window})", current_price)
        if position.pnl > 0 and managed.window not in state.winning_windows:
            state.winning_windows.append(managed.window)
        state.window_status[managed.window] = "closed:time_exit"
        settled_keys.append(key)

    for key in settled_keys:
        state.managed_positions.pop(key, None)


def has_live_position(state: RuntimeState) -> bool:
    return bool(state.managed_positions)


def should_skip_for_prior_winner(state: RuntimeState, window: WindowConfig) -> bool:
    if not window.execute:
        return False
    if window.name == "morning":
        return False
    return bool(state.winning_windows)


def process_window(
    client: AlpacaIntradayClient,
    args: argparse.Namespace,
    state: RuntimeState,
    current: datetime,
    window: WindowConfig,
) -> None:
    if state.window_status.get(window.name):
        return

    weekday = current.strftime("%a")
    if window.allowed_weekdays and weekday not in window.allowed_weekdays:
        state.window_status[window.name] = f"skipped:weekday:{weekday}"
        return

    if should_skip_for_prior_winner(state, window):
        state.window_status[window.name] = "skipped:prior_winner"
        return

    if has_live_position(state):
        return

    trade_date = state.trade_date
    signal_time = parse_clock(trade_date, window.signal_time)
    confirm_time = parse_clock(trade_date, window.confirm_time)
    entry_ready_at = confirm_time + timedelta(minutes=1)
    if current < entry_ready_at:
        return
    if current > entry_ready_at + timedelta(minutes=args.entry_grace_minutes):
        state.window_status[window.name] = "missed:entry_grace"
        return

    signal_args = build_signal_args(args.benchmark_ticker, window)
    setup = stage_signal(
        client,
        signal_args,
        [args.ticker],
        trade_date,
        signal_time,
        confirm_time,
        window.threshold,
    )
    if not setup:
        state.window_status[window.name] = "no_signal"
        return

    if not window.execute:
        state.window_status[window.name] = (
            f"probe:{setup['direction']}:{setup['signal_pct']:.4f}:{setup['confirm_pct']:.4f}"
        )
        return

    spot_price = latest_spot_price(client, args.ticker, trade_date, current)
    if spot_price is None:
        state.window_status[window.name] = "blocked:no_spot"
        return

    if args.dry_run:
        state.window_status[window.name] = (
            f"dry_run:{setup['direction']}:{setup['signal_pct']:.4f}:{setup['confirm_pct']:.4f}"
        )
        send_alpaca_status(
            f"DRY RUN {STRATEGY_NAME} {window.name}: {args.ticker} {setup['direction']} "
            f"signal={setup['signal_pct']:.2%} confirm={setup['confirm_pct']:.2%} spot=${spot_price:.2f}"
        )
        return

    direction = "long" if setup["direction"] == "CALL" else "short"
    option_type = "call" if setup["direction"] == "CALL" else "put"
    confidence = 59.0  # Force fixed-size fallback sizing capped by MAX_PER_TRADE.
    target_price = spot_price * (1.01 if direction == "long" else 0.99)
    stop_loss = spot_price * (0.995 if direction == "long" else 1.005)
    pattern = f"QQQ_SCHED_{window.name.upper()}_{setup['direction']}"

    position = alpaca_executor.execute_scalp_entry(
        underlying=args.ticker,
        direction=direction,
        entry_price=spot_price,
        target_price=target_price,
        stop_loss=stop_loss,
        confidence=confidence,
        pattern=pattern,
        engine=TradingEngine.MOMENTUM,
        option_type_override=option_type,
    )
    if not position:
        state.window_status[window.name] = "blocked:executor"
        return

    managed = ManagedPositionState(
        window=window.name,
        option_symbol=position.option_symbol,
        position_id=position.position_id,
        direction=setup["direction"],
        entered_at=current.isoformat(),
        time_exit_at=(current + timedelta(minutes=window.max_hold_minutes)).isoformat(),
    )
    state.managed_positions[window.name] = managed
    state.window_status[window.name] = f"entered:{position.option_symbol}"
    send_alpaca_status(
        f"{STRATEGY_NAME} entered {window.name}: {args.ticker} {setup['direction']} "
        f"signal={setup['signal_pct']:.2%} confirm={setup['confirm_pct']:.2%} option={position.option_symbol}"
    )


def process_iteration(client: AlpacaIntradayClient, args: argparse.Namespace, current: datetime) -> RuntimeState:
    trade_date = current.date().isoformat()
    state = load_state(trade_date)
    settle_positions(state, current)
    for window in WINDOWS:
        process_window(client, args, state, current, window)
    save_state(state)
    return state


def run(args: argparse.Namespace) -> None:
    apply_runtime_risk_profile(args)
    current = now_et(args.now_iso)
    client = AlpacaIntradayClient()
    restore_managed_positions(load_state(current.date().isoformat()))

    if args.run_once:
        state = process_iteration(client, args, current)
        log.info("run_once complete: %s", state.window_status)
        return

    log.info("%s daemon starting", STRATEGY_NAME)
    while True:
        current = now_et()
        if is_market_session(current):
            state = process_iteration(client, args, current)
            log.debug("%s state: %s", STRATEGY_NAME, state.window_status)
        time.sleep(args.poll_seconds)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
