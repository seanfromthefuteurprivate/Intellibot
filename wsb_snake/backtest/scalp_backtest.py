"""
Lightweight backtester for scalper and momentum rules.

Replays rules on historical daily (or 1-min when available) bars to produce
win rate, total return, max drawdown, and Sharpe. Used to tune MIN_CONFIDENCE,
sector filter, and flow filter thresholds.

Inputs: Historical daily bars from Polygon (get_daily_bars).
Logic: Simplified momentum (volume surge + price change) and scalp proxy (daily move threshold).
Output: Metrics dict and optional sensitivity over MIN_CONFIDENCE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from wsb_snake.utils.logger import get_logger
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.config import MOMENTUM_UNIVERSE

log = get_logger(__name__)


@dataclass
class BacktestTrade:
    """Simulated trade."""
    ticker: str
    direction: str
    entry_bar_idx: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str  # "target", "stop", "time"


@dataclass
class BacktestResult:
    """Aggregate backtest metrics."""
    trades: List[BacktestTrade] = field(default_factory=list)
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_annual: float = 0.0
    n_trades: int = 0


def _avg_volume(bars: List[Dict], n: int = 20) -> float:
    if not bars or len(bars) < n:
        return 0.0
    vols = [b.get("v", b.get("volume", 0)) for b in bars[:n]]
    return sum(vols) / len(vols) if vols else 0.0


def _price_change_pct(bars: List[Dict], days: int) -> Optional[float]:
    if not bars or len(bars) <= days:
        return None
    c_now = bars[0].get("c", bars[0].get("close", 0))
    c_ago = bars[days].get("c", bars[days].get("close", 0))
    if not c_ago or c_ago <= 0:
        return None
    return (c_now / c_ago - 1) * 100


def run_momentum_backtest(
    ticker: str = "SPY",
    lookback_days: int = 60,
    volume_surge_mult: float = 1.4,
    price_up_5d_pct: float = 3.0,
    target_pct: float = 15.0,
    stop_pct: float = -8.0,
    hold_max_days: int = 5,
) -> BacktestResult:
    """
    Replay momentum-style rules on daily bars: volume surge + price up => long.
    Simulate exit at target_pct, stop_pct, or hold_max_days.
    """
    bars = polygon_enhanced.get_daily_bars(ticker, limit=lookback_days)
    if not bars or len(bars) < 25:
        return BacktestResult()
    trades: List[BacktestTrade] = []
    equity_curve: List[float] = [100.0]
    peak = 100.0
    max_dd = 0.0
    returns: List[float] = []
    i = 20
    while i < len(bars) - hold_max_days:
        vol_avg = _avg_volume(bars[i:], 20)
        vol_today = bars[i].get("v", bars[i].get("volume", 0))
        if vol_avg <= 0:
            i += 1
            continue
        surge = vol_today / vol_avg
        chg_5 = _price_change_pct(bars[i:], 5)
        if surge < volume_surge_mult or (chg_5 is None or chg_5 < price_up_5d_pct):
            i += 1
            continue
        entry_price = bars[i].get("c", bars[i].get("close", 0))
        if entry_price <= 0:
            i += 1
            continue
        exit_price = entry_price
        exit_reason = "time"
        for j in range(1, min(hold_max_days + 1, len(bars) - i)):
            future = bars[i + j]
            c = future.get("c", future.get("close", entry_price))
            pnl_pct = (c / entry_price - 1) * 100
            if pnl_pct >= target_pct:
                exit_price = c
                exit_reason = "target"
                break
            if pnl_pct <= stop_pct:
                exit_price = c
                exit_reason = "stop"
                break
            exit_price = c
            exit_reason = "time"
        pnl_pct = (exit_price / entry_price - 1) * 100
        trades.append(
            BacktestTrade(
                ticker=ticker,
                direction="long",
                entry_bar_idx=i,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
            )
        )
        ret = pnl_pct / 100.0
        returns.append(ret)
        new_equity = equity_curve[-1] * (1 + ret)
        equity_curve.append(new_equity)
        if new_equity > peak:
            peak = new_equity
        dd = (peak - new_equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
        i += hold_max_days + 1  # skip ahead after exit
    if not trades:
        return BacktestResult()
    total_return = (equity_curve[-1] / 100.0 - 1) * 100
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    win_rate = wins / len(trades) * 100
    avg_ret = sum(returns) / len(returns) if returns else 0
    std_ret = (sum((r - avg_ret) ** 2 for r in returns) / len(returns)) ** 0.5 if len(returns) > 1 else 0
    sharpe = (avg_ret / std_ret * (252 ** 0.5)) if std_ret > 0 else 0
    return BacktestResult(
        trades=trades,
        total_return_pct=total_return,
        win_rate=win_rate,
        max_drawdown_pct=max_dd,
        sharpe_annual=sharpe,
        n_trades=len(trades),
    )


def run_scalp_proxy_backtest(
    ticker: str = "SPY",
    lookback_days: int = 60,
    min_daily_move_pct: float = 0.5,
    target_pct: float = 1.0,
    stop_pct: float = -0.5,
) -> BacktestResult:
    """
    Simplified scalp proxy: "signal" when day opens with prior-day momentum;
    simulate same-day exit at target/stop (using daily bar as proxy for intraday).
    """
    bars = polygon_enhanced.get_daily_bars(ticker, limit=lookback_days)
    if not bars or len(bars) < 5:
        return BacktestResult()
    trades: List[BacktestTrade] = []
    equity_curve: List[float] = [100.0]
    peak = 100.0
    max_dd = 0.0
    returns: List[float] = []
    for i in range(2, len(bars) - 1):
        prev_close = bars[i + 1].get("c", 0)
        open_p = bars[i].get("o", bars[i].get("open", 0))
        close_p = bars[i].get("c", bars[i].get("close", 0))
        if prev_close <= 0 or open_p <= 0:
            continue
        prior_move = (open_p - prev_close) / prev_close * 100
        if abs(prior_move) < min_daily_move_pct:
            continue
        direction = "long" if prior_move > 0 else "short"
        entry_price = open_p
        move_pct = (close_p - open_p) / open_p * 100 if direction == "long" else (open_p - close_p) / open_p * 100
        if move_pct >= target_pct:
            pnl_pct = target_pct
            exit_reason = "target"
        elif move_pct <= -abs(stop_pct):
            pnl_pct = -abs(stop_pct)
            exit_reason = "stop"
        else:
            pnl_pct = move_pct
            exit_reason = "time"
        exit_price = entry_price * (1 + pnl_pct / 100) if direction == "long" else entry_price * (1 - pnl_pct / 100)
        trades.append(
            BacktestTrade(
                ticker=ticker,
                direction=direction,
                entry_bar_idx=i,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                exit_reason=exit_reason,
            )
        )
        ret = pnl_pct / 100.0
        returns.append(ret)
        new_equity = equity_curve[-1] * (1 + ret)
        equity_curve.append(new_equity)
        peak = max(peak, new_equity)
        dd = (peak - new_equity) / peak * 100 if peak > 0 else 0
    if not trades:
        return BacktestResult()
    total_return = (equity_curve[-1] / 100.0 - 1) * 100
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    win_rate = wins / len(trades) * 100
    avg_ret = sum(returns) / len(returns)
    std_ret = (sum((r - avg_ret) ** 2 for r in returns) / len(returns)) ** 0.5 if len(returns) > 1 else 0
    sharpe = (avg_ret / std_ret * (252 ** 0.5)) if std_ret > 0 else 0
    max_dd = max(
        (max(equity_curve[:j + 1]) - equity_curve[j]) / max(equity_curve[:j + 1]) * 100
        for j in range(len(equity_curve))
    ) if equity_curve else 0
    return BacktestResult(
        trades=trades,
        total_return_pct=total_return,
        win_rate=win_rate,
        max_drawdown_pct=max_dd,
        sharpe_annual=sharpe,
        n_trades=len(trades),
    )


def run_sensitivity(
    ticker: str = "SPY",
    min_confidence_levels: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Run momentum backtest over different thresholds; return metrics by threshold."""
    levels = min_confidence_levels or [60, 65, 70, 75]
    out: Dict[str, Any] = {}
    for thresh in levels:
        # Use price_up_5d_pct as proxy for "confidence" threshold (stricter = higher required move)
        price_thresh = 1.0 + (thresh - 65) * 0.2  # e.g. 65 -> 3%, 70 -> 4%
        res = run_momentum_backtest(
            ticker=ticker,
            price_up_5d_pct=max(1.0, price_thresh),
        )
        out[str(thresh)] = {
            "total_return_pct": res.total_return_pct,
            "win_rate": res.win_rate,
            "max_drawdown_pct": res.max_drawdown_pct,
            "sharpe_annual": res.sharpe_annual,
            "n_trades": res.n_trades,
        }
    return out


def get_backtest_report(ticker: str = "SPY") -> Dict[str, Any]:
    """Produce a small report for tuning: momentum + scalp proxy + sensitivity."""
    momentum = run_momentum_backtest(ticker=ticker)
    scalp = run_scalp_proxy_backtest(ticker=ticker)
    sensitivity = run_sensitivity(ticker=ticker)
    return {
        "ticker": ticker,
        "momentum": {
            "total_return_pct": momentum.total_return_pct,
            "win_rate": momentum.win_rate,
            "max_drawdown_pct": momentum.max_drawdown_pct,
            "sharpe_annual": momentum.sharpe_annual,
            "n_trades": momentum.n_trades,
        },
        "scalp_proxy": {
            "total_return_pct": scalp.total_return_pct,
            "win_rate": scalp.win_rate,
            "max_drawdown_pct": scalp.max_drawdown_pct,
            "sharpe_annual": scalp.sharpe_annual,
            "n_trades": scalp.n_trades,
        },
        "sensitivity_by_threshold": sensitivity,
    }
