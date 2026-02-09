"""
Broker-agnostic signal formatter for clean, actionable trading signals.
Creates beautifully formatted signals for Telegram notifications.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List
import uuid


class HoldGuidance(Enum):
    """Guidance for holding or exiting a position."""
    TAKE_PROFIT_NOW = "take_profit_now"
    HOLD_FOR_MORE = "hold_for_more"
    SCALE_OUT = "scale_out"
    TIGHT_STOP = "tight_stop"


@dataclass
class SignalTargets:
    """Price targets for a trading signal."""
    entry: float
    pl1: float  # +20%
    pl2: float  # +40%
    pl3: float  # +60%
    stop_loss: float  # -15%

    @classmethod
    def calculate(cls, entry_price: float) -> "SignalTargets":
        """
        Calculate profit targets and stop loss from entry price.

        PL1: +20% profit
        PL2: +40% profit
        PL3: +60% profit
        Stop Loss: -15% loss
        """
        return cls(
            entry=entry_price,
            pl1=round(entry_price * 1.20, 2),
            pl2=round(entry_price * 1.40, 2),
            pl3=round(entry_price * 1.60, 2),
            stop_loss=round(entry_price * 0.85, 2)
        )


@dataclass
class TradingSignal:
    """Complete trading signal with all relevant information."""
    signal_id: str
    ticker: str  # Option symbol (e.g., AAPL250214C00230000)
    underlying: str  # Underlying stock (e.g., AAPL)
    strike: float
    expiry: str  # Format: YYYY-MM-DD
    direction: str  # "CALL" or "PUT"
    contracts: int
    entry_price: float
    targets: SignalTargets
    confidence: float  # 0.0 to 1.0
    pattern: str  # Pattern that triggered the signal
    session: str  # "opening", "midday", "power_hour", "max_mode"
    max_hold_minutes: int
    volatility: str  # "low", "medium", "high", "extreme"
    hold_guidance: HoldGuidance
    hold_reasoning: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def __post_init__(self):
        """Set expiry time if not provided."""
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(minutes=self.max_hold_minutes)


def generate_signal_id() -> str:
    """Generate a unique signal ID."""
    timestamp = datetime.now().strftime("%H%M%S")
    unique_part = uuid.uuid4().hex[:6].upper()
    return f"SIG-{timestamp}-{unique_part}"


def _get_confidence_emoji(confidence: float) -> str:
    """Get emoji representation of confidence level."""
    if confidence >= 0.85:
        return "🔥"
    elif confidence >= 0.70:
        return "✅"
    elif confidence >= 0.55:
        return "⚡"
    else:
        return "⚠️"


def _get_volatility_warning(volatility: str) -> str:
    """Get volatility warning message."""
    warnings = {
        "low": "Stable conditions - normal position sizing",
        "medium": "Moderate volatility - watch closely",
        "high": "High volatility - consider smaller size",
        "extreme": "EXTREME volatility - max caution!"
    }
    return warnings.get(volatility, "Unknown volatility")


def _get_hold_guidance_text(guidance: HoldGuidance) -> tuple[str, str]:
    """Get hold guidance emoji and text."""
    guidance_map = {
        HoldGuidance.TAKE_PROFIT_NOW: ("💰", "TAKE PROFIT NOW - Don't wait!"),
        HoldGuidance.HOLD_FOR_MORE: ("📈", "HOLD FOR MORE - Momentum strong"),
        HoldGuidance.SCALE_OUT: ("⚖️", "SCALE OUT - Take partial profits"),
        HoldGuidance.TIGHT_STOP: ("🛡️", "TIGHT STOP - Protect gains")
    }
    return guidance_map.get(guidance, ("❓", "Monitor position"))


def _format_expiry_display(expiry: str) -> str:
    """Format expiry date for display."""
    try:
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
        today = datetime.now().date()
        days_to_expiry = (expiry_date.date() - today).days

        if days_to_expiry == 0:
            return f"{expiry} (0DTE! ⚡)"
        elif days_to_expiry == 1:
            return f"{expiry} (1DTE)"
        else:
            return f"{expiry} ({days_to_expiry}DTE)"
    except ValueError:
        return expiry


def format_buy_signal(signal: TradingSignal) -> str:
    """
    Create a beautifully formatted BUY signal for Telegram.

    Args:
        signal: The TradingSignal to format

    Returns:
        Formatted string ready for Telegram (markdown)
    """
    confidence_emoji = _get_confidence_emoji(signal.confidence)
    hold_emoji, hold_text = _get_hold_guidance_text(signal.hold_guidance)
    volatility_warning = _get_volatility_warning(signal.volatility)
    expiry_display = _format_expiry_display(signal.expiry)

    # Format the copy line for manual broker entry
    copy_line = f"{signal.underlying} ${signal.strike} {signal.direction} {signal.expiry}"

    # Calculate P/L amounts
    pl1_gain = signal.targets.pl1 - signal.entry_price
    pl2_gain = signal.targets.pl2 - signal.entry_price
    pl3_gain = signal.targets.pl3 - signal.entry_price
    stop_loss = signal.entry_price - signal.targets.stop_loss

    message = f"""
🚀 *NEW SIGNAL* {confidence_emoji}
━━━━━━━━━━━━━━━━━━━━━━

📋 *COPY FOR BROKER:*
`{copy_line}`

━━━━━━━━━━━━━━━━━━━━━━

*Details:*
• Underlying: *{signal.underlying}*
• Strike: *${signal.strike}*
• Direction: *{signal.direction}*
• Expiry: *{expiry_display}*
• Contracts: *{signal.contracts}*
• Entry: *${signal.entry_price:.2f}*

━━━━━━━━━━━━━━━━━━━━━━

🎯 *PROFIT TARGETS:*

• *PL1* (+20%): ${signal.targets.pl1:.2f}
  ↳ +${pl1_gain:.2f}/contract | _Scale out 1/3_

• *PL2* (+40%): ${signal.targets.pl2:.2f}
  ↳ +${pl2_gain:.2f}/contract | _Scale out 1/3_

• *PL3* (+60%): ${signal.targets.pl3:.2f}
  ↳ +${pl3_gain:.2f}/contract | _Close remaining_

🛑 *STOP LOSS* (-15%): ${signal.targets.stop_loss:.2f}
  ↳ -${stop_loss:.2f}/contract | _Exit all_

━━━━━━━━━━━━━━━━━━━━━━

⚡ *RISK GUIDANCE:*
• Max Hold: *{signal.max_hold_minutes} minutes*
• Volatility: *{signal.volatility.upper()}*
  ↳ _{volatility_warning}_

━━━━━━━━━━━━━━━━━━━━━━

{hold_emoji} *HOLD GUIDANCE:*
*{hold_text}*
_{signal.hold_reasoning}_

━━━━━━━━━━━━━━━━━━━━━━

📊 *Signal Info:*
• Pattern: {signal.pattern}
• Session: {signal.session.replace('_', ' ').title()}
• Confidence: {signal.confidence:.0%}
• ID: `{signal.signal_id}`

⏰ Expires: {signal.expires_at.strftime('%H:%M:%S') if signal.expires_at else 'N/A'}
"""

    return message.strip()


def format_exit_signal(
    signal: TradingSignal,
    current_price: float,
    reason: str,
    recommendation: str
) -> str:
    """
    Create a formatted EXIT alert.

    Args:
        signal: The original TradingSignal
        current_price: Current option price
        reason: Reason for exit alert
        recommendation: What action to take

    Returns:
        Formatted exit signal for Telegram
    """
    # Calculate P/L
    pl_amount = current_price - signal.entry_price
    pl_percent = ((current_price / signal.entry_price) - 1) * 100

    # Determine if profit or loss
    if pl_amount >= 0:
        pl_emoji = "💰" if pl_percent >= 20 else "📈"
        pl_color = "+"
    else:
        pl_emoji = "🔻"
        pl_color = ""

    message = f"""
🚨 *EXIT ALERT* {pl_emoji}
━━━━━━━━━━━━━━━━━━━━━━

*{signal.underlying}* ${signal.strike} {signal.direction}

━━━━━━━━━━━━━━━━━━━━━━

📊 *Current P/L:*
• Entry: ${signal.entry_price:.2f}
• Current: ${current_price:.2f}
• P/L: *{pl_color}${pl_amount:.2f}* ({pl_color}{pl_percent:.1f}%)

━━━━━━━━━━━━━━━━━━━━━━

❗ *Reason:*
_{reason}_

━━━━━━━━━━━━━━━━━━━━━━

✅ *Recommendation:*
*{recommendation}*

━━━━━━━━━━━━━━━━━━━━━━

Signal ID: `{signal.signal_id}`
"""

    return message.strip()


def format_time_warning(
    signal: TradingSignal,
    remaining_minutes: int,
    current_price: Optional[float] = None
) -> str:
    """
    Create a time-based warning message.

    Args:
        signal: The TradingSignal
        remaining_minutes: Minutes remaining until max hold
        current_price: Optional current price for P/L display

    Returns:
        Formatted time warning for Telegram
    """
    # Determine urgency level
    if remaining_minutes <= 5:
        urgency_emoji = "🚨"
        urgency_text = "CRITICAL"
    elif remaining_minutes <= 15:
        urgency_emoji = "⚠️"
        urgency_text = "WARNING"
    else:
        urgency_emoji = "⏰"
        urgency_text = "REMINDER"

    # Build P/L section if current price provided
    pl_section = ""
    if current_price is not None:
        pl_amount = current_price - signal.entry_price
        pl_percent = ((current_price / signal.entry_price) - 1) * 100
        pl_color = "+" if pl_amount >= 0 else ""
        pl_section = f"""
📊 *Current Status:*
• Price: ${current_price:.2f}
• P/L: *{pl_color}${pl_amount:.2f}* ({pl_color}{pl_percent:.1f}%)

"""

    message = f"""
{urgency_emoji} *TIME {urgency_text}* {urgency_emoji}
━━━━━━━━━━━━━━━━━━━━━━

*{signal.underlying}* ${signal.strike} {signal.direction}

⏱️ *{remaining_minutes} MINUTES REMAINING*

{pl_section}━━━━━━━━━━━━━━━━━━━━━━

💡 *Action Required:*
Review position and decide:
• Take profits if green
• Cut losses if red
• Set tighter stop if holding

━━━━━━━━━━━━━━━━━━━━━━

Signal ID: `{signal.signal_id}`
"""

    return message.strip()


def format_target_hit(
    signal: TradingSignal,
    target_level: str,
    current_price: float
) -> str:
    """
    Create a target hit notification.

    Args:
        signal: The TradingSignal
        target_level: Which target was hit (PL1, PL2, PL3)
        current_price: Current option price

    Returns:
        Formatted target hit message for Telegram
    """
    pl_amount = current_price - signal.entry_price
    pl_percent = ((current_price / signal.entry_price) - 1) * 100

    target_info = {
        "PL1": ("🎯", "+20%", "Scale out 1/3 of position"),
        "PL2": ("🎯🎯", "+40%", "Scale out another 1/3"),
        "PL3": ("🏆", "+60%", "Consider closing remaining")
    }

    emoji, gain, action = target_info.get(target_level, ("🎯", "Target", "Review position"))

    message = f"""
{emoji} *{target_level} TARGET HIT!* {emoji}
━━━━━━━━━━━━━━━━━━━━━━

*{signal.underlying}* ${signal.strike} {signal.direction}

✅ *{target_level} Reached ({gain})*
• Current: ${current_price:.2f}
• P/L: *+${pl_amount:.2f}* (+{pl_percent:.1f}%)

━━━━━━━━━━━━━━━━━━━━━━

💡 *Suggested Action:*
_{action}_

━━━━━━━━━━━━━━━━━━━━━━

Signal ID: `{signal.signal_id}`
"""

    return message.strip()


def format_stop_loss_hit(
    signal: TradingSignal,
    current_price: float
) -> str:
    """
    Create a stop loss hit notification.

    Args:
        signal: The TradingSignal
        current_price: Current option price

    Returns:
        Formatted stop loss message for Telegram
    """
    loss_amount = signal.entry_price - current_price
    loss_percent = ((signal.entry_price - current_price) / signal.entry_price) * 100

    message = f"""
🛑 *STOP LOSS HIT* 🛑
━━━━━━━━━━━━━━━━━━━━━━

*{signal.underlying}* ${signal.strike} {signal.direction}

❌ *Stop Loss Triggered*
• Entry: ${signal.entry_price:.2f}
• Current: ${current_price:.2f}
• Loss: *-${loss_amount:.2f}* (-{loss_percent:.1f}%)

━━━━━━━━━━━━━━━━━━━━━━

⚡ *Action:*
_Exit position immediately to limit losses_

━━━━━━━━━━━━━━━━━━━━━━

Signal ID: `{signal.signal_id}`
"""

    return message.strip()
