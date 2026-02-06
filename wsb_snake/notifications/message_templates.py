from datetime import datetime, timezone

from wsb_snake.signals.signal_types import Signal, SignalTier


def _et_now() -> str:
    """Current time ET for audit messages."""
    try:
        import zoneinfo
        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
        return et.strftime("%Y-%m-%d %H:%M:%S ET")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ─────────────────────────────────────────────────────────────────
# One-glance action lines — same format for ALL trackers (CPL, Zero Greed, etc.)
# First line of every trade alert so you can enter/exit in one look.
# ─────────────────────────────────────────────────────────────────

def action_line_enter(
    ticker: str,
    side: str,
    dte: int,
    strike: float,
    entry: float,
    stop: float,
    spot: float = None,
    contracts: int = 1,
) -> str:
    """One line: BUY NOW with everything needed to open the trade."""
    dte_s = "0DTE" if dte == 0 else f"{dte}DTE"
    spot_s = f" | Spot ${spot:.2f}" if spot is not None and spot > 0 else ""
    return "▶ *BUY NOW* | %s | %s | %s | Strike $%.2f | Entry $%.2f | Stop $%.2f%s" % (
        (ticker or "—").strip().upper(),
        (side or "—").upper(),
        dte_s,
        strike,
        entry,
        stop,
        spot_s,
    )


def action_line_exit(
    ticker: str,
    side: str,
    dte: int,
    strike: float,
    pnl_pct: float,
    exit_price: float,
    entry_price: float = None,
    contracts: int = 1,
) -> str:
    """One line: EXIT NOW with everything needed to close the trade."""
    dte_s = "0DTE" if dte == 0 else f"{dte}DTE"
    return "▶ *EXIT NOW* | %s | %s | %s | Strike $%.2f | P&L %+.1f%% | Sell @ $%.2f" % (
        (ticker or "—").strip().upper(),
        (side or "—").upper(),
        dte_s,
        strike,
        pnl_pct,
        exit_price,
    )


def copy_line_enter(ticker: str, side: str, strike: float, entry: float, stop: float, contracts: int = 1) -> str:
    """One line to copy into broker: BUY order."""
    return "COPY: BUY %d %s %s $%.2f @ limit $%.2f (stop $%.2f)" % (contracts, ticker or "—", (side or "—").upper(), strike, entry, stop)


def copy_line_exit(ticker: str, side: str, strike: float, contracts: int = 1) -> str:
    """One line to copy into broker: SELL order."""
    return "COPY: SELL %d %s %s $%.2f @ market" % (contracts, ticker or "—", (side or "—").upper(), strike)


def format_startup_message() -> str:
    """Format the 'Snake Online' startup message."""
    return """🐍 *WSB SNAKE ONLINE*

✅ Secrets loaded
✅ Connectors healthy
✅ Monitoring active

_Scanning for signals..._"""


def format_alert_message(signal: Signal) -> str:
    """
    Format an immediate alert message for Telegram.
    First line = one-glance so you never miss: ▶ ALERT | TICKER | Action | Price
    """
    tier_emoji = {
        SignalTier.A_PLUS: "🔥",
        SignalTier.A: "⚡",
        SignalTier.B: "👀",
    }
    emoji = tier_emoji.get(signal.tier, "📊")
    market = getattr(signal, "market", None)
    price = float(getattr(market, "price", 0) or 0) if market else 0.0
    action = getattr(signal, "action", None)
    action_val = action.value if hasattr(action, "value") else str(action) if action else "WATCH"
    # Line 1: one-glance — same idea as CPL/Zero Greed so all alerts are scannable
    lines = [
        "▶ *ALERT* | %s | %s | Price $%.2f | Score %.0f" % (signal.ticker, action_val, price, signal.score),
        "",
        f"{emoji} *WSB SNAKE ALERT — ${signal.ticker}*",
        f"Score: {signal.score:.0f}/100 | Tier: {signal.tier.value}",
        "",
    ]
    
    # Why (bullets)
    lines.append("*Why:*")
    if signal.why:
        for reason in signal.why[:3]:
            lines.append(f"• {reason}")
    else:
        lines.append(f"• Social velocity: {signal.social.velocity:.1f}/min")
        lines.append(f"• Price change: {signal.market.change_pct*100:.1f}%")
    lines.append("")
    
    # Market data
    lines.append("*Market:*")
    lines.append(f"Price: ${signal.market.price:.2f} ({signal.market.change_pct*100:+.1f}%)")
    if signal.market.volume:
        lines.append(f"Volume: {signal.market.volume:,}")
    lines.append("")
    
    # Levels (if available)
    if signal.levels:
        lines.append("*Levels:*")
        for level, value in signal.levels.items():
            lines.append(f"• {level}: ${value:.2f}")
        lines.append("")
    
    # Risk flags
    risk_flags = []
    if signal.risk.low_liquidity:
        risk_flags.append("⚠️ Low liquidity")
    if signal.risk.high_volatility:
        risk_flags.append("⚠️ High volatility")
    if signal.risk.pump_detected:
        risk_flags.append("🚨 Pump risk")
    
    if risk_flags:
        lines.append("*Risk:*")
        for flag in risk_flags[:2]:
            lines.append(flag)
        lines.append("")
    
    # Action
    lines.append(f"*Action:* {signal.action.value}")
    
    # AI Summary (if available)
    if signal.summary:
        lines.append("")
        lines.append(f"_{signal.summary[:200]}_")
    
    return "\n".join(lines)


def format_digest_message(signals: list) -> str:
    """
    Format a watchlist digest message.
    """
    if not signals:
        return "📋 *WSB SNAKE DIGEST*\n\nNo significant signals in this period."
    
    lines = [
        "📋 *WSB SNAKE DIGEST*",
        f"_Top {len(signals)} tickers to watch:_",
        "",
    ]
    
    for i, signal in enumerate(signals[:10], 1):
        change_str = f"{signal.market.change_pct*100:+.1f}%"
        risk_tag = ""
        if signal.risk.high_volatility:
            risk_tag = " ⚠️"
        if signal.risk.pump_detected:
            risk_tag = " 🚨"
            
        lines.append(f"{i}. *${signal.ticker}* — Score {signal.score:.0f} | {change_str}{risk_tag}")
        if signal.why:
            lines.append(f"   _{signal.why[0][:50]}_")
    
    lines.append("")
    lines.append("_Set alerts on key levels. DYOR._")
    
    return "\n".join(lines)


def format_error_message(error: str) -> str:
    """Format an error notification."""
    return f"⚠️ *WSB SNAKE ERROR*\n\n{error[:500]}"


def format_session_header(session_date: str, sequence_id: int, event_tiers: list = None) -> str:
    """
    Telegram session header for sequencing. Send once per session (or at CPL run start).
    event_tiers: e.g. ["20X", "4X", "2X"] to show focus for this run.
    """
    lines = [
        "══════════════════════════════════════",
        f"📅 *SESSION* {session_date}",
        f"📌 *SEQUENCE* #{sequence_id}",
    ]
    if event_tiers:
        lines.append(f"🎯 *FOCUS*: {' | '.join(event_tiers)} events")
    lines.extend([
        "══════════════════════════════════════",
        "",
    ])
    return "\n".join(lines)


def format_jobs_day_call(
    call,
    call_number: int,
    test_mode: bool = False,
    untruncated_tails: bool = False,
    trade_number: int = None,
    session_balance: float = None,
    contracts: int = 1,
    message_sequence: int = None,
) -> str:
    """
    Format a single CPL Jobs Day BUY for Telegram — MAXIMALLY DETAILED, AUDIT-PROOF.
    When untruncated_tails: add paper trader lines (EXECUTE, Trade #, balance, contracts, goal).
    """
    tag = " [TEST]" if test_mode else ""
    tier_tag = f" | {getattr(call, 'event_tier', '') or ''} EVENT" if getattr(call, "event_tier", None) else ""
    event_type = f"BUY | POSITION_OPENED | STATE=OBSERVE | CALL #{call_number}{tier_tag}{tag}"
    if message_sequence is not None:
        event_type = f"MSG #{message_sequence} | " + event_type
    entry_price = float(call.entry_trigger.get("price") or 0)
    stop_price = float(call.stop_loss.get("price") or 0)
    take_profit = call.take_profit or []
    max_hold = call.cooldowns.get("max_hold_minutes", 30)
    dedupe_short = (call.dedupe_key[:20] + "…") if len(call.dedupe_key or "") > 20 else (call.dedupe_key or "")
    now = _et_now()

    lines = []
    spot = getattr(call, "spot_at_alert", None)
    dte = getattr(call, "dte", 0)
    ticker = (call.underlying or "—").strip()
    # Line 1: one-glance ENTER — same format for all trackers so you never miss a trade
    lines.append(action_line_enter(ticker, call.side, dte, call.strike, entry_price, stop_price, spot=spot, contracts=contracts))
    lines.append(copy_line_enter(ticker, call.side, call.strike, entry_price, stop_price, contracts))
    lines.append("")
    if untruncated_tails:
        lines.extend([
            "*PAPER | SEQUENTIAL | $250 → 5 FIGURES*",
            "",
        ])
    lines.extend([
        "--------------------------------------------------",
        f"CPL_EVENT | {event_type}",
        "--------------------------------------------------",
        "",
        "[MODE]",
        "- System Mode: REAL TRADE",
        "- Execution: ENABLED",
        "- Purpose: Maximizing trading on high volatility days",
        "",
        "[INSTRUCTION]",
        "- Action: BUY",
        "- Instruction: ENTER NOW",
        f"- Contracts: {contracts}",
        f"- Underlying: {call.underlying or '—'}",
        f"- Side: {call.side or '—'}",
        f"- DTE: {getattr(call, 'dte', 0)} (0 = today expiry)",
        f"- Strike: ${call.strike:.2f} (exact)",
        f"- Entry Price: ${entry_price:.2f} (exact)",
        f"- Stop Loss: ${stop_price:.2f} (exact)",
        f"- Contract: `{call.option_symbol or call.option_descriptor}`",
        "",
    ])
    if spot is not None and spot > 0:
        lines.append(f"- Spot (underlying at alert): ${spot:.2f}")
    lines.extend([
        "[REFERENCE SNAPSHOTS]",
        f"- Entry Reference Price: ${entry_price:.2f}",
        f"- Entry Reference Time: {now}",
        f"- Current Reference Price: ${entry_price:.2f}",
        "- Peak Reference Price (if applicable): N/A",
        "- Exit Reference Price (ONLY on RELEASE): N/A",
        "- Expansion From Entry: 0.0%",
        "",
        "[STATE & PERMISSION]",
        "- Current State: OBSERVE",
        "- Exit Permission: ALLOWED",
        "- Reason: Structure fragile; exit allowed. Entry reference set.",
        "",
        "[STRUCTURAL READ]",
        "- Volatility State: displacing",
        "- Dominance: present",
        "- Continuation Quality: fragile",
        "",
        "[TP INTERPRETATION]",
        "- TP +25%: not reached",
        "- TP +50%: not reached",
        "- TP +100%: not reached",
        "- TP Role: STRUCTURAL CHECKPOINTS ONLY (never exits)",
        "",
        "[NAIVE SYSTEM COMPARISON]",
        "- A naive TP-based system would have exited here",
        "- This system explicitly refuses interference",
        "",
        "[WHAT CAN CHANGE STATE]",
        "- Only structural invalidation can force RELEASE",
        "- Profit magnitude is ignored entirely",
        "",
    ])
    for i, tp in enumerate(take_profit[:3], 1):
        p = tp.get("price", 0)
        pct = tp.get("pct", 0)
        rule = tp.get("rule", f"TP{i}")
        lines.append(f"- TP{i} (checkpoint): ${float(p):.2f} (+{pct}%) ({rule})")
    lines.extend([
        "",
        f"*Max Hold:* {max_hold} min | *Call Number:* {call_number} | *Regime:* {call.regime} | *Confidence:* {call.confidence:.0f}%",
        "*REASONS:*",
    ])
    for r in (call.reasons or [])[:3]:
        lines.append(f"• {r}")
    lines.extend([
        "",
        "EXECUTE WITHIN 5 MIN | DO NOT REPEAT",
    ])
    if untruncated_tails and trade_number is not None and session_balance is not None:
        lines.extend([
            f"*EXECUTE:* BUY {contracts}x {call.underlying} {call.side} {call.strike} {getattr(call, 'expiry_date', '')} @ limit ${entry_price:.2f} or better. Paper balance at risk: ${session_balance:.2f}",
            f"*Trade #{trade_number}* | Paper balance: ${session_balance:.2f} | Contracts: {contracts} | Goal: $250 → 5 figures",
        ])
    lines.extend([
        f"`{dedupe_short}`",
        f"TIME={now}",
        "--------------------------------------------------",
    ])
    return "\n".join(lines)


def format_jobs_day_sell(
    call,
    original_buy_number: int,
    exit_reason: str,
    exit_price: float,
    entry_price: float,
    pnl_pct: float,
    peak_ref_price: float = None,
    untruncated_tails: bool = False,
    trade_number: int = None,
    session_balance: float = None,
    contracts: int = 1,
    message_sequence: int = None,
    event_tier: str = None,
    r_multiple: float = None,
) -> str:
    """
    Format SELL for Telegram — MAXIMALLY DETAILED, AUDIT-PROOF.
    When untruncated_tails: add Trade #, PnL, new balance, NEXT: wait for BUY.
    """
    tier_tag = f" | {event_tier} EVENT" if event_tier else ""
    event_type = f"SELL | POSITION_CLOSED | {exit_reason}{tier_tag}"
    if message_sequence is not None:
        event_type = f"MSG #{message_sequence} | " + event_type
    now = _et_now()
    expansion_str = f"{pnl_pct:+.1f}%"
    peak_str = f"${peak_ref_price:.2f}" if peak_ref_price is not None else "N/A"
    dollar_pnl = (exit_price - entry_price) * 100 * contracts

    # Structural failure cause for RELEASE (dominance lost, coherence collapse, etc.)
    structural_cause = "N/A"
    if exit_reason == "STRUCTURE_BREAK":
        structural_cause = "Structural invalidation; dominance lost or coherence collapse"
    elif exit_reason == "SL":
        structural_cause = "Stop loss; structural protection"
    elif exit_reason == "TIME":
        structural_cause = "Time expiry; max hold reached"
    elif exit_reason == "TP":
        structural_cause = "Take profit; exit permitted by governance"

    tp25 = "observed" if pnl_pct >= 25 else "not reached"
    tp50 = "observed" if pnl_pct >= 50 else "not reached"
    tp100 = "observed" if pnl_pct >= 100 else "not reached"

    ticker = (call.underlying or "—").strip()
    dte = getattr(call, "dte", 0)
    # Line 1: one-glance EXIT — same format for all trackers so you never miss closing
    action_first = action_line_exit(ticker, call.side, dte, call.strike, pnl_pct, exit_price, entry_price, contracts)
    copy_first = copy_line_exit(ticker, call.side, call.strike, contracts)

    lines = [
        action_first,
        copy_first,
        "",
        "--------------------------------------------------",
        f"CPL_EVENT | {event_type}",
        "--------------------------------------------------",
        "",
        "[MODE]",
        "- System Mode: REAL TRADE",
        "- Execution: ENABLED",
        "- Purpose: Maximizing trading on high volatility days",
        "",
        "[INSTRUCTION]",
        "- Action: SELL",
        "- Instruction: EXIT NOW",
        f"- Contracts: {contracts}",
        f"- Underlying: {call.underlying or '—'}",
        f"- Side: {call.side or '—'}",
        f"- DTE: {getattr(call, 'dte', 0)}",
        f"- Strike: ${call.strike:.2f} (exact)",
        f"- Entry Price: ${entry_price:.2f} (exact)",
        f"- Exit Price: ${exit_price:.2f} (exact)",
        f"- Contract: `{call.option_symbol or call.option_descriptor}`",
        f"- Closing BUY #{original_buy_number}",
        "",
        "[REFERENCE SNAPSHOTS]",
        f"- Entry Reference Price: ${entry_price:.2f}",
        f"- Entry Reference Time: {now}",
        f"- Current Reference Price: ${exit_price:.2f}",
        f"- Peak Reference Price (if applicable): {peak_str}",
        f"- Exit Reference Price (ONLY on RELEASE): ${exit_price:.2f}",
        f"- Expansion From Entry: {expansion_str}",
        "",
        "[STATE & PERMISSION]",
        "- Current State: RELEASE",
        "- Exit Permission: REQUIRED",
        f"- Reason: Position closed; {exit_reason}; PnL {pnl_pct:+.1f}%",
        f"- Structural failure cause: {structural_cause}",
        "",
        "[STRUCTURAL READ]",
        "- Volatility State: decaying",
        "- Dominance: lost",
        "- Continuation Quality: inert",
        "",
        "[TP INTERPRETATION]",
        f"- TP +25%: {tp25}",
        f"- TP +50%: {tp50}",
        f"- TP +100%: {tp100}",
        "- TP Role: STRUCTURAL CHECKPOINTS ONLY (never exits)",
        "",
        "[NAIVE SYSTEM COMPARISON]",
        "- A naive TP-based system would have exited here",
        "- This system explicitly refuses interference",
        "",
        "[WHAT CAN CHANGE STATE]",
        "- Only structural invalidation can force RELEASE",
        "- Profit magnitude is ignored entirely",
        "",
        f"*REGIME:* {call.regime}",
        "",
    ]
    if call.reasons:
        lines.append("*REASONS:*")
        for r in call.reasons[:2]:
            lines.append(f"• {r}")
        lines.append("")
    lines.append("POSITION CLOSED | DO NOT RE-ENTER")
    if r_multiple is not None:
        lines.append(f"*R-MULTIPLE:* {r_multiple:+.2f}R")
    if untruncated_tails and trade_number is not None and session_balance is not None:
        pnl_sign = "+" if dollar_pnl >= 0 else ""
        lines.extend([
            "",
            f"*Trade #{trade_number} closed* | PnL: ${pnl_sign}{dollar_pnl:.2f} | New balance: ${session_balance:.2f}",
            "*NEXT:* Wait for the next BUY alert — that is your only open position until you SELL it.",
        ])
    if getattr(call, "original_call_id", None):
        lines.append(f"`LINEAGE: {call.original_call_id[:8]}...`")
    lines.append(f"TIME={now}")
    lines.append("--------------------------------------------------")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Apex Governance Layer Message Templates
# ─────────────────────────────────────────────────────────────────

def format_governance_event(
    event_type: str,
    note: str,
    dedupe_key: str = None,
    pnl_pct: float = None,
    state_from: str = None,
    state_to: str = None,
    timestamp_et: str = None,
) -> str:
    """
    Format a governance event for Telegram.
    
    Events:
    - STATE=OBSERVE/CANDIDATE/RUNNER_LOCK/RELEASE
    - STATE_TRANSITION | X→Y
    - RUNNER_LOCK_ENTERED
    - RUNNER_LOCK_HEARTBEAT
    - STRUCTURE_BREAK_DETECTED
    - TP_CHECKPOINT | +X%
    - TP_SUPPRESSED
    - PREREGISTRATION_LOCKED
    """
    from datetime import datetime, timezone
    
    # Get emoji based on event type
    emoji = "📊"  # Default
    if "RUNNER_LOCK_ENTERED" in event_type:
        emoji = "🔒"
    elif "RUNNER_LOCK_HEARTBEAT" in event_type:
        emoji = "💓"
    elif "STRUCTURE_BREAK" in event_type:
        emoji = "⚠️"
    elif "TP_SUPPRESSED" in event_type:
        emoji = "🚫"
    elif "TP_CHECKPOINT" in event_type:
        emoji = "📍"
    elif "STATE_TRANSITION" in event_type:
        emoji = "🔄"
    elif "OBSERVE" in event_type:
        emoji = "👁️"
    elif "CANDIDATE" in event_type:
        emoji = "🎯"
    elif "RELEASE" in event_type:
        emoji = "🚨"
    elif "PREREGISTRATION" in event_type:
        emoji = "🔐"
    
    # Format timestamp
    if timestamp_et is None:
        try:
            import zoneinfo
            et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
            timestamp_et = et.strftime("%Y-%m-%d %H:%M:%S ET")
        except Exception:
            timestamp_et = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    lines = [
        f"{emoji} *CPL_EVENT | {event_type}*",
        "",
        f"NOTE={note}",
        f"TIME={timestamp_et}",
    ]
    
    if pnl_pct is not None:
        lines.append(f"PNL={pnl_pct:+.1f}%")
    
    if state_from and state_to:
        lines.append(f"TRANSITION={state_from}→{state_to}")
    
    if dedupe_key:
        short_key = dedupe_key[:25] + "..." if len(dedupe_key) > 25 else dedupe_key
        lines.append(f"`{short_key}`")
    
    return "\n".join(lines)


def format_runner_lock_alert(
    dedupe_key: str,
    pnl_pct: float,
    ticker: str = None,
    timestamp_et: str = None,
) -> str:
    """
    Format critical RUNNER_LOCK entered alert.
    
    This is a high-priority alert when exit permission is revoked.
    """
    from datetime import datetime, timezone
    
    if timestamp_et is None:
        try:
            import zoneinfo
            et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
            timestamp_et = et.strftime("%Y-%m-%d %H:%M:%S ET")
        except Exception:
            timestamp_et = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    short_key = dedupe_key[:25] + "..." if len(dedupe_key) > 25 else dedupe_key
    ticker_line = f"*TICKER:* {ticker}\n" if ticker else ""
    first = f"▶ *HOLD — RUNNER LOCK* | {ticker or '—'} | PnL +{pnl_pct:.1f}% | DO NOT EXIT\n\n" if ticker else ""
    return f"""{first}🔒 *APEX RUNNER LOCK ENGAGED*

*EXIT PERMISSION REVOKED*

{ticker_line}Current PnL: +{pnl_pct:.1f}%

Axiom A1 Active:
_Profit magnitude NEVER justifies exit_

System will hold until:
• Structure breaks (drawdown from peak)
• Dominance lost (negotiation entered)
• Time decay overtakes momentum

DO NOT INTERFERE
`{short_key}`
TIME={timestamp_et}"""


def format_structure_break_alert(
    dedupe_key: str,
    reason: str,
    final_pnl: float,
    ticker: str = None,
    timestamp_et: str = None,
) -> str:
    """
    Format critical STRUCTURE BREAK alert — immediate exit required.
    
    This is the highest priority alert triggering mandatory exit.
    """
    from datetime import datetime, timezone
    
    if timestamp_et is None:
        try:
            import zoneinfo
            et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
            timestamp_et = et.strftime("%Y-%m-%d %H:%M:%S ET")
        except Exception:
            timestamp_et = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    short_key = dedupe_key[:25] + "..." if len(dedupe_key) > 25 else dedupe_key
    ticker_line = f"*TICKER:* {ticker}\n" if ticker else ""
    first = f"▶ *EXIT NOW* (STRUCTURE BREAK) | {ticker or '—'} | PnL {final_pnl:+.1f}%\n\n" if ticker else ""
    return f"""{first}⚠️ *STRUCTURE BREAK DETECTED*

*EXIT NOW REQUIRED*

{ticker_line}Reason: {reason}
Final PnL: {final_pnl:+.1f}%

Axiom A2 Triggered:
_Only structural invalidation justifies exit_

EXECUTE EXIT IMMEDIATELY
`{short_key}`
TIME={timestamp_et}"""


def format_tp_suppressed_alert(
    dedupe_key: str,
    pnl_pct: float,
    checkpoint: str = None,
    timestamp_et: str = None,
) -> str:
    """
    Format TP SUPPRESSED alert when exit is blocked by RUNNER_LOCK.
    """
    from datetime import datetime, timezone
    
    if timestamp_et is None:
        try:
            import zoneinfo
            et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
            timestamp_et = et.strftime("%Y-%m-%d %H:%M:%S ET")
        except Exception:
            timestamp_et = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    short_key = dedupe_key[:25] + "..." if len(dedupe_key) > 25 else dedupe_key
    checkpoint_line = f"Checkpoint: {checkpoint}\n" if checkpoint else ""
    
    return f"""🚫 *TP SUPPRESSED — RUNNER_LOCK ACTIVE*

{checkpoint_line}Current PnL: +{pnl_pct:.1f}%

Axiom A1:
_Profit magnitude does not justify exit_

Structure intact — exit still forbidden.
`{short_key}`
TIME={timestamp_et}"""
