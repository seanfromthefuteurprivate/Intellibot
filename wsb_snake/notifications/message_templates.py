from wsb_snake.signals.signal_types import Signal, SignalTier

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
    Structure: headline, why, levels, risk, plan
    """
    tier_emoji = {
        SignalTier.A_PLUS: "🔥",
        SignalTier.A: "⚡",
        SignalTier.B: "👀",
    }
    emoji = tier_emoji.get(signal.tier, "📊")
    
    # Headline
    lines = [
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
