"""
Jobs Report Tracker – Prep for NFP / Employment Situation (Feb 6, 2026).

Tracks high-impact tickers, builds options playbook for event day with:
- Watchlist of most volatile / rate-sensitive names
- Event-day options chain (0DTE or Friday expiry)
- Trend bias from existing WSB Snake data (momentum, technicals)
- Concrete buy/sell calls: ticker, direction, DTE, strike, entry, exit, stop

Optimized for small WeBull account (~$250): 1–2 positions, cheap OTM options.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.collectors.polygon_options import polygon_options
from wsb_snake.config import DATA_DIR
from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)

# --- Event config ---
# NFP rescheduled per BLS/Reuters: Jan 2026 Employment Situation = Wed Feb 11, 2026 @ 8:30am ET
JOBS_REPORT_EVENT_DATE = "2026-02-11"  # Wednesday
JOBS_REPORT_EVENT_TIME_ET = "08:30"    # NFP release
# Strike mode: full event-vol watchlist (index, vol, rates, dollar, metals, crypto beta, AI/mega, sectors)
JOBS_REPORT_WATCHLIST = [
    "SPY", "QQQ", "IWM", "DIA",         # core index
    "VXX", "UVXY",                      # VIX products (panic meter)
    "TLT", "IEF", "XLF",                # rates, financials
    "UUP", "GLD", "SLV", "GDX",         # dollar, metals, gold miners
    "MSTR", "COIN", "MARA", "RIOT",     # crypto beta
    "NVDA", "TSLA", "AAPL", "AMZN", "META", "GOOGL", "MSFT", "AMD",
    "ITB", "XHB", "XLY", "XLV",         # homebuilders, consumer, healthcare
]
# Primary plays for $250: SPY, QQQ (most liquid 0DTE, move on NFP)
PRIMARY_PLAYS = ["SPY", "QQQ"]
BUDGET_WEBBULL_USD = 250
MAX_POSITIONS = 2
MAX_COST_PER_CONTRACT_USD = 125  # So 2 contracts max ~$250


@dataclass
class TickerSnapshot:
    ticker: str
    price: float
    change_pct: float
    volume: int
    has_options: bool
    atm_strike: Optional[float] = None
    atm_call_ask: Optional[float] = None
    atm_put_ask: Optional[float] = None
    otm_call_ask: Optional[float] = None  # 1 strike OTM
    otm_put_ask: Optional[float] = None
    avg_iv: Optional[float] = None
    momentum_bias: str = "neutral"  # "bullish" | "bearish" | "neutral"
    momentum_score: float = 0.0


@dataclass
class OptionsCall:
    ticker: str
    direction: str  # "call" | "put"
    expiry: str
    dte: int
    strike: float
    entry_price_est: float
    exit_target_pct: float
    stop_loss_pct: float
    max_hold_minutes: int
    rationale: str
    contract_cost_est: float = 0.0
    suggested_contracts: int = 1


@dataclass
class JobsReportPlaybook:
    generated_at: str
    event_date: str
    event_time_et: str
    budget_usd: float
    watchlist: List[Dict[str, Any]] = field(default_factory=list)
    primary_tickers: List[str] = field(default_factory=list)
    recommended_trades: List[Dict[str, Any]] = field(default_factory=list)
    strategy_notes: str = ""


class JobsReportTracker:
    """
    Track jobs-report watchlist and build options playbook using existing
    Polygon data and WSB Snake momentum/trend logic.
    """

    def __init__(
        self,
        event_date: str = JOBS_REPORT_EVENT_DATE,
        watchlist: Optional[List[str]] = None,
        budget_usd: float = BUDGET_WEBBULL_USD,
    ):
        self.event_date = event_date
        self.watchlist = watchlist or JOBS_REPORT_WATCHLIST
        self.budget_usd = budget_usd
        self._snapshots: List[TickerSnapshot] = []

    def get_watchlist_quotes(self) -> List[TickerSnapshot]:
        """Fetch price and options data for each ticker in watchlist."""
        snapshots = []
        for ticker in self.watchlist:
            try:
                snap = self._snapshot_ticker(ticker)
                if snap:
                    snapshots.append(snap)
            except Exception as e:
                logger.warning(f"Jobs report tracker: skip {ticker}: {e}")
        self._snapshots = snapshots
        return snapshots

    def _snapshot_ticker(self, ticker: str) -> Optional[TickerSnapshot]:
        quote = polygon_options.get_quote(ticker)
        if not quote:
            # Fallback to polygon_enhanced snapshot
            snap = polygon_enhanced.get_snapshot(ticker) if polygon_enhanced else None
            if not snap:
                return None
            price = snap.get("price", 0) or 0
            # polygon_enhanced returns change_pct in percent points (e.g. 0.5 = 0.5%)
            raw_chg = snap.get("change_pct", 0) or 0
            change_pct = raw_chg / 100.0 if abs(raw_chg) >= 1 else raw_chg
            volume = snap.get("today_volume", 0) or snap.get("volume", 0) or 0
        else:
            price = quote.get("price", 0) or 0
            change_pct = quote.get("change_pct", 0) or 0
            volume = int(quote.get("volume", 0) or 0)
        if not price or price <= 0:
            return None

        # Options for event date (Friday)
        chain = polygon_options.get_chain_for_expiration(
            ticker, price, self.event_date, strike_range=8
        )
        calls = chain.get("calls") or []
        puts = chain.get("puts") or []
        has_options = bool(calls or puts)
        atm_strike = chain.get("atm_strike")
        metrics = chain.get("metrics") or {}
        avg_iv = metrics.get("avg_iv")

        atm_call_ask = atm_put_ask = otm_call_ask = otm_put_ask = None
        if calls:
            atm_calls = [c for c in calls if c.get("strike") and abs(c["strike"] - price) < 2]
            otm_calls = [c for c in calls if c.get("strike") and c["strike"] > price]
            if atm_calls:
                atm_call_ask = atm_calls[0].get("ask") or atm_calls[0].get("last_price")
            if otm_calls:
                otm_calls.sort(key=lambda x: x["strike"])
                otm_call_ask = otm_calls[0].get("ask") or otm_calls[0].get("last_price")
        if puts:
            atm_puts = [p for p in puts if p.get("strike") and abs(p["strike"] - price) < 2]
            otm_puts = [p for p in puts if p.get("strike") and p["strike"] < price]
            if atm_puts:
                atm_put_ask = atm_puts[0].get("ask") or atm_puts[0].get("last_price")
            if otm_puts:
                otm_puts.sort(key=lambda x: -x["strike"])
                otm_put_ask = otm_puts[0].get("ask") or otm_puts[0].get("last_price")

        momentum_bias, momentum_score = self._get_momentum_bias(ticker)
        return TickerSnapshot(
            ticker=ticker,
            price=price,
            change_pct=change_pct,
            volume=volume,
            has_options=has_options,
            atm_strike=atm_strike,
            atm_call_ask=atm_call_ask,
            atm_put_ask=atm_put_ask,
            otm_call_ask=otm_call_ask,
            otm_put_ask=otm_put_ask,
            avg_iv=avg_iv,
            momentum_bias=momentum_bias,
            momentum_score=momentum_score,
        )

    def _get_momentum_bias(self, ticker: str) -> tuple[str, float]:
        try:
            mom = polygon_enhanced.get_momentum_signals(ticker)
            if not mom.get("available") or not mom.get("signals"):
                return "neutral", 0.0
            signals = mom.get("signals") or []
            score = 0.0
            for sig in signals:
                if isinstance(sig, (list, tuple)) and len(sig) >= 2:
                    name, val = sig[0], sig[1]
                    if "UP" in str(name) or "GAP_UP" in str(name) or "HIGH" in str(name):
                        score += float(val) if isinstance(val, (int, float)) else 1
                    elif "DOWN" in str(name) or "GAP_DOWN" in str(name) or "LOW" in str(name):
                        score -= float(val) if isinstance(val, (int, float)) else 1
            if score > 0.3:
                return "bullish", score
            if score < -0.3:
                return "bearish", score
            return "neutral", score
        except Exception:
            return "neutral", 0.0

    def build_playbook(self) -> JobsReportPlaybook:
        """
        Build playbook: watchlist snapshots + recommended option trades
        for Friday event, optimized for small account.
        """
        if not self._snapshots:
            self.get_watchlist_quotes()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        watchlist_data = [self._snap_to_dict(s) for s in self._snapshots]
        recommended = self._recommend_trades()
        strategy = (
            "NFP 8:30 AM ET. Strong number + soft wages → risk-on (SPY/QQQ calls). "
            "Weak number or hot wages → risk-off (SPY/QQQ puts). "
            "Wait 2–5 min after print for initial spike, then enter 1 direction. "
            "Use 0DTE or same-day expiry; target +15–25%, stop -8%; max hold 15 min."
        )
        return JobsReportPlaybook(
            generated_at=now,
            event_date=self.event_date,
            event_time_et=JOBS_REPORT_EVENT_TIME_ET,
            budget_usd=self.budget_usd,
            watchlist=watchlist_data,
            primary_tickers=PRIMARY_PLAYS,
            recommended_trades=recommended,
            strategy_notes=strategy,
        )

    def _snap_to_dict(self, s: TickerSnapshot) -> Dict[str, Any]:
        return {
            "ticker": s.ticker,
            "price": round(s.price, 2),
            "change_pct": round(s.change_pct * 100, 2) if s.change_pct else 0,
            "volume": s.volume,
            "has_options": s.has_options,
            "atm_strike": s.atm_strike,
            "atm_call_ask": s.atm_call_ask,
            "atm_put_ask": s.atm_put_ask,
            "otm_call_ask": s.otm_call_ask,
            "otm_put_ask": s.otm_put_ask,
            "avg_iv_pct": round(s.avg_iv * 100, 1) if s.avg_iv else None,
            "momentum_bias": s.momentum_bias,
            "momentum_score": round(s.momentum_score, 2),
        }

    def _recommend_trades(self) -> List[Dict[str, Any]]:
        """Suggest 1–2 option plays for $250 account: SPY/QQQ, strike/entry/exit/stop."""
        out = []
        for ticker in PRIMARY_PLAYS:
            snap = next((s for s in self._snapshots if s.ticker == ticker), None)
            if not snap or not snap.has_options:
                continue
            strike = snap.atm_strike or round(snap.price, 0)
            # Prefer OTM to keep cost under budget
            call_ask = snap.otm_call_ask or snap.atm_call_ask
            put_ask = snap.otm_put_ask or snap.atm_put_ask
            cost_call = (call_ask or 0) * 100
            cost_put = (put_ask or 0) * 100
            if cost_call <= 0 and cost_put <= 0:
                continue
            contracts = 1
            if (call_ask or 0) > 0 and (call_ask or 0) * 100 <= MAX_COST_PER_CONTRACT_USD:
                entry_call = call_ask or 0
                out.append({
                    "ticker": ticker,
                    "direction": "call",
                    "expiry": self.event_date,
                    "dte": 0,
                    "strike": strike,
                    "entry_price_est": round(entry_call, 2),
                    "exit_target_pct": 25,
                    "stop_loss_pct": -8,
                    "max_hold_minutes": 15,
                    "contract_cost_est": round(entry_call * 100, 2),
                    "suggested_contracts": contracts,
                    "rationale": f"Momentum: {snap.momentum_bias}. NFP beat → risk-on → call.",
                })
            if (put_ask or 0) > 0 and (put_ask or 0) * 100 <= MAX_COST_PER_CONTRACT_USD:
                entry_put = put_ask or 0
                out.append({
                    "ticker": ticker,
                    "direction": "put",
                    "expiry": self.event_date,
                    "dte": 0,
                    "strike": strike,
                    "entry_price_est": round(entry_put, 2),
                    "exit_target_pct": 25,
                    "stop_loss_pct": -8,
                    "max_hold_minutes": 15,
                    "contract_cost_est": round(entry_put * 100, 2),
                    "suggested_contracts": contracts,
                    "rationale": f"Momentum: {snap.momentum_bias}. NFP miss → risk-off → put.",
                })
        return out[:4]  # Max 4 lines (2 tickers × call/put)

    def run(self, output_dir: Optional[Path] = None) -> JobsReportPlaybook:
        """Fetch data, build playbook, optionally write JSON + markdown."""
        playbook = self.build_playbook()
        out_dir = output_dir or Path(DATA_DIR)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "jobs_report_playbook.json"
        md_path = out_dir / "JOBS_REPORT_FEB6.md"
        with open(json_path, "w") as f:
            json.dump(asdict(playbook), f, indent=2)
        logger.info(f"Wrote {json_path}")
        self._write_markdown_playbook(playbook, md_path)
        logger.info(f"Wrote {md_path}")
        return playbook

    def _write_markdown_playbook(self, playbook: JobsReportPlaybook, path: Path) -> None:
        lines = [
            "# Jobs Report Options Playbook – Feb 6, 2026",
            "",
            f"**Generated:** {playbook.generated_at}  ",
            f"**Event:** NFP / Employment Situation @ {playbook.event_time_et} ET  ",
            f"**Budget (WeBull):** ${playbook.budget_usd}  ",
            "",
            "## Strategy (30-second version)",
            "",
            playbook.strategy_notes,
            "",
            "## Watchlist snapshot",
            "",
            "| Ticker | Price | Chg% | ATM Strike | Call (ATM) | Put (ATM) | OTM Call | OTM Put | IV% | Bias |",
            "|--------|-------|------|------------|------------|-----------|----------|--------|-----|------|",
        ]
        for w in playbook.watchlist:
            ticker = w.get("ticker", "")
            price = w.get("price", 0)
            chg = w.get("change_pct", 0)
            atm = w.get("atm_strike") or "-"
            call_a = w.get("atm_call_ask") or "-"
            put_a = w.get("atm_put_ask") or "-"
            otm_c = w.get("otm_call_ask") or "-"
            otm_p = w.get("otm_put_ask") or "-"
            iv = w.get("avg_iv_pct") if w.get("avg_iv_pct") is not None else "-"
            bias = w.get("momentum_bias", "")
            lines.append(f"| {ticker} | {price} | {chg}% | {atm} | {call_a} | {put_a} | {otm_c} | {otm_p} | {iv} | {bias} |")
        lines.extend([
            "",
            "## Recommended option plays ($250 account)",
            "",
            "| Ticker | Dir | Strike | Entry (est) | Target | Stop | Hold | Cost/contract |",
            "|--------|-----|--------|-------------|--------|------|------|---------------|",
        ])
        for t in playbook.recommended_trades:
            lines.append(
                f"| {t['ticker']} | {t['direction']} | {t['strike']} | {t['entry_price_est']} | "
                f"+{t['exit_target_pct']}% | {t['stop_loss_pct']}% | {t['max_hold_minutes']}m | "
                f"${t.get('contract_cost_est', 0)} |"
            )
        lines.extend(["", "### Rationale", ""])
        for t in playbook.recommended_trades:
            lines.append(f"- **{t['ticker']} {t['direction'].upper()}** @ {t['strike']}: {t['rationale']}")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
