"""
Call Object schema for Jobs Day (CPL) - atomic trade call representation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


def _et_now() -> str:
    try:
        import zoneinfo
        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except Exception:
        et = datetime.now(timezone.utc)
    return et.strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class JobsDayCall:
    """Atomic call object for CPL."""

    call_id: str
    timestamp_et: str
    engine: str = "CPL"
    underlying: str = ""
    action: str = "BUY"
    side: str = "CALL"
    expiry_date: str = ""
    dte: int = 0
    strike: float = 0.0
    option_symbol: Optional[str] = None
    option_descriptor: str = ""
    entry_trigger: Dict[str, Any] = field(default_factory=dict)
    stop_loss: Dict[str, Any] = field(default_factory=dict)
    take_profit: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    levels: Dict[str, float] = field(default_factory=dict)
    regime: str = "RISK_ON"
    vmss: float = 0.0
    vmss_components: Dict[str, float] = field(default_factory=dict)
    dedupe_key: str = ""
    # Event magnitude tier: 2X (+50–99%), 4X (+100–399%), 20X (+400%+ potential)
    event_tier: str = ""
    # Underlying spot at alert time (so PUT 77 vs SPOT 82 is clear)
    spot_at_alert: Optional[float] = None
    paper_capital_alloc: Dict[str, Any] = field(default_factory=dict)
    cooldowns: Dict[str, Any] = field(default_factory=dict)
    original_call_id: Optional[str] = None  # FIX 2: For SELL lineage tracking

    def generate_dedupe_key(self, window: str = "AM") -> str:
        strike_int = int(self.strike) if self.strike == int(self.strike) else int(round(self.strike * 1000) / 1000)
        key = f"{self.underlying.upper()}|{self.expiry_date}|{strike_int}|{self.side.upper()}|{self.engine}|{window}"
        self.dedupe_key = key
        return key

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_telegram_markdown(self, call_number: int) -> str:
        entry_price = self.entry_trigger.get("price", 0)
        stop_price = self.stop_loss.get("price", 0)
        stop_pct = self.stop_loss.get("pct", 0)
        lines = [
            f"*ACTION:* {self.action} {self.side}",
            f"*TICKER:* {self.underlying}",
            f"*CONTRACT:* `{self.option_symbol or self.option_descriptor}`",
            f"*STRIKE:* ${self.strike:.0f}",
            f"*EXPIRY:* {self.expiry_date} ({self.dte}DTE)",
            "",
            f"*ENTRY:* ${entry_price:.2f}",
            f"*STOP:* ${stop_price:.2f} ({stop_pct}%)",
        ]
        for i, tp in enumerate(self.take_profit[:3], 1):
            p = tp.get("price", 0)
            pct = tp.get("pct", 0)
            rule = tp.get("rule", f"TP{i}")
            lines.append(f"*TP{i}:* ${p:.2f} (+{pct}%) ({rule})")
        lines.extend(["", f"*REGIME:* {self.regime}", f"*CONFIDENCE:* {self.confidence:.0f}%", "*REASONS:*"])
        for r in self.reasons[:3]:
            lines.append(f"- {r}")
        vwap = self.levels.get("vwap", 0)
        key_level = self.levels.get("key_level", 0)
        invalid = self.levels.get("invalid_level", 0)
        lines.append(f"*LEVELS:* VWAP={vwap} | Key={key_level} | Invalid={invalid}")
        lines.append("")
        lines.append("EXECUTE WITHIN 5 MIN | DO NOT REPEAT")
        lines.append(f"DEDUPE: {self.dedupe_key}")
        return "\n".join(lines)

    @classmethod
    def create(
        cls,
        underlying: str,
        side: str,
        strike: float,
        expiry_date: str,
        dte: int,
        entry_price: float,
        stop_pct: float = -15,
        tp_pcts: Optional[List[float]] = None,
        option_symbol: Optional[str] = None,
        regime: str = "RISK_ON",
        confidence: float = 75.0,
        reasons: Optional[List[str]] = None,
        window: str = "AM",
        action: str = "BUY",  # FIX #5 & #8: Support SELL action
        original_call_id: Optional[str] = None,  # FIX 2: For SELL lineage
    ) -> "JobsDayCall":
        call_id = str(uuid.uuid4())
        timestamp_et = _et_now()
        tp_pcts = tp_pcts or [25, 50, 100]
        stop_price = round(entry_price * (1 + stop_pct / 100), 2)
        take_profit = [
            {"price": round(entry_price * (1 + pct / 100), 2), "pct": pct, "rule": f"TP{i}"}
            for i, pct in enumerate(tp_pcts, 1)
        ]
        if len(take_profit) >= 3:
            take_profit[2]["rule"] = "TP3 / Runner kill"
        opt_letter = "C" if side.upper() == "CALL" else "P"
        option_descriptor = f"{underlying} {dte}DTE {expiry_date} {int(strike)}{opt_letter}"
        entry_trigger = {"type": "LIMIT", "price": entry_price, "rule": f"ENTER when option_mid <= {entry_price:.2f}"}
        stop_loss = {"type": "HARD", "price": stop_price, "pct": stop_pct, "rule": f"EXIT if option_mid <= {stop_price:.2f}"}
        levels = {"vwap": 0.0, "key_level": 0.0, "invalid_level": 0.0}
        paper_capital_alloc = {"max_cost": 250, "contracts": 1, "max_slippage": 0.05}
        cooldowns = {"do_not_repeat_minutes": 45, "max_hold_minutes": 30}
        vmss_components = {"market_structure": 0, "options_liquidity": 0, "wsb_pressure": 0, "news_catalyst": 0}
        reasons = reasons or [f"CPL {regime} scenario", f"Strike {strike} {side}"]

        obj = cls(
            call_id=call_id,
            timestamp_et=timestamp_et,
            engine="CPL",
            underlying=underlying,
            action=action.upper(),  # BUY or SELL
            side=side.upper(),
            expiry_date=expiry_date,
            dte=dte,
            strike=strike,
            option_symbol=option_symbol,
            option_descriptor=option_descriptor,
            entry_trigger=entry_trigger,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reasons=reasons,
            levels=levels,
            regime=regime,
            vmss=0.0,
            vmss_components=vmss_components,
            dedupe_key="",
            paper_capital_alloc=paper_capital_alloc,
            cooldowns=cooldowns,
            original_call_id=original_call_id,
        )
        obj.generate_dedupe_key(window=window)
        return obj
