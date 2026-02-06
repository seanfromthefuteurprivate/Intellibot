"""
Telemetry Event Bus — Telegram-facing observability for Apex Governance.

Emits governance events to Telegram in a MAXIMALLY DETAILED, SELF-EXPLANATORY,
AUDIT-PROOF format. All events are observational and do not affect trading logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from wsb_snake.notifications.telegram_bot import send_alert
from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)

# Global toggle
TELEMETRY_ENABLED = True


def _et_now() -> str:
    """Get current time in ET format."""
    try:
        import zoneinfo
        et = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
        return et.strftime("%Y-%m-%d %H:%M:%S ET")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_price(v: Optional[float]) -> str:
    """Format price for audit; exact only."""
    if v is None:
        return "N/A"
    return f"${v:.2f}"


def _fmt_ts(dt: Optional[datetime]) -> str:
    """Format timestamp for audit."""
    if dt is None:
        return "N/A"
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S ET")
    except Exception:
        return str(dt)


class TelemetryEventBus:
    """
    Telegram-facing telemetry for governance events.
    
    Event types:
    - STATE=OBSERVE/CANDIDATE/RUNNER_LOCK/RELEASE
    - STATE_TRANSITION | X->Y
    - RUNNER_LOCK_ENTERED
    - RUNNER_LOCK_HEARTBEAT
    - STRUCTURE_BREAK_DETECTED
    - TP_CHECKPOINT | +X%
    - TP_SUPPRESSED
    - PREREGISTRATION_LOCKED
    """
    
    def __init__(self, telegram_enabled: bool = True):
        self.telegram_enabled = telegram_enabled and TELEMETRY_ENABLED
        self.event_count = 0
        self.session_start = datetime.now(timezone.utc)
        logger.info(f"TelemetryEventBus initialized (telegram={self.telegram_enabled})")
    
    def emit(
        self,
        event_type: str,
        note: str,
        dedupe_key: Optional[str] = None,
        pnl_pct: Optional[float] = None,
        **kwargs,
    ) -> bool:
        """
        Emit a governance event to Telegram.
        
        Returns True if sent successfully.
        """
        self.event_count += 1
        
        # Format the message
        msg = self.format_governance_event(event_type, note, dedupe_key, pnl_pct, **kwargs)
        
        # Log locally
        logger.info(f"TELEMETRY | {event_type} | {note}")
        
        # Send to Telegram
        if self.telegram_enabled:
            try:
                success = send_alert(msg)
                if success:
                    logger.debug(f"Telemetry sent to Telegram: {event_type}")
                else:
                    logger.warning(f"Failed to send telemetry: {event_type}")
                return success
            except Exception as e:
                logger.error(f"Telemetry send error: {e}")
                return False
        
        return True
    
    def _build_audit_message(
        self,
        event_type: str,
        note: str,
        dedupe_key: Optional[str] = None,
        pnl_pct: Optional[float] = None,
        entry_ref_price: Optional[float] = None,
        entry_ref_time: Optional[str] = None,
        current_ref_price: Optional[float] = None,
        peak_ref_price: Optional[float] = None,
        exit_ref_price: Optional[str] = None,
        expansion_pct: Optional[float] = None,
        current_state: Optional[str] = None,
        exit_permission: Optional[str] = None,
        reason: Optional[str] = None,
        volatility_state: Optional[str] = None,
        dominance: Optional[str] = None,
        continuation_quality: Optional[str] = None,
        tp25_status: str = "not reached",
        tp50_status: str = "not reached",
        tp100_status: str = "not reached",
        structural_failure_cause: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Build MAXIMALLY DETAILED, AUDIT-PROOF Telegram message."""
        emoji = self._get_emoji(event_type)
        now = _et_now()
        expansion_str = f"{expansion_pct:+.1f}%" if expansion_pct is not None else "N/A"
        entry_price_str = _fmt_price(entry_ref_price)
        current_price_str = _fmt_price(current_ref_price) if current_ref_price is not None else "N/A"
        peak_price_str = _fmt_price(peak_ref_price) if peak_ref_price is not None else "N/A"
        if exit_ref_price is None and isinstance(kwargs.get("exit_ref_price"), (int, float)):
            exit_ref_price = _fmt_price(float(kwargs["exit_ref_price"]))
        if exit_ref_price is None:
            exit_ref_price = "N/A"
        elif isinstance(exit_ref_price, (int, float)):
            exit_ref_price = _fmt_price(float(exit_ref_price))

        state = current_state or "N/A"
        perm = exit_permission or "N/A"
        reason_plain = reason or note
        vol_state = volatility_state or "N/A"
        dom = dominance or "N/A"
        cont_qual = continuation_quality or "N/A"
        failure_cause = structural_failure_cause or "N/A"

        lines = [
            "--------------------------------------------------",
            f"CPL_EVENT | {event_type}",
            "--------------------------------------------------",
            "",
            "[MODE]",
            "- System Mode: REAL TRADE",
            "- Execution: ENABLED",
            "- Purpose: Maximizing trading on high volatility days",
            "",
            "[REFERENCE SNAPSHOTS]",
            f"- Entry Reference Price: {entry_price_str}",
            f"- Entry Reference Time: {entry_ref_time or now}",
            f"- Current Reference Price: {current_price_str}",
            f"- Peak Reference Price (if applicable): {peak_price_str}",
            f"- Exit Reference Price (ONLY on RELEASE): {exit_ref_price}",
            f"- Expansion From Entry: {expansion_str}",
            "",
            "[STATE & PERMISSION]",
            f"- Current State: {state}",
            f"- Exit Permission: {perm}",
            f"- Reason: {reason_plain}",
            "",
            "[STRUCTURAL READ]",
            f"- Volatility State: {vol_state}",
            f"- Dominance: {dom}",
            f"- Continuation Quality: {cont_qual}",
            "",
            "[TP INTERPRETATION]",
            f"- TP +25%: {tp25_status}",
            f"- TP +50%: {tp50_status}",
            f"- TP +100%: {tp100_status}",
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
            "--------------------------------------------------",
        ]
        if dedupe_key:
            short_key = dedupe_key[:30] + "..." if len(dedupe_key) > 30 else dedupe_key
            lines.insert(-2, f"`{short_key}`")
            lines.insert(-2, "")
        if pnl_pct is not None:
            lines.insert(-2, f"PNL (observational): {pnl_pct:+.1f}%")
            lines.insert(-2, "")
        if state == "RELEASE" and structural_failure_cause:
            idx = next(i for i, s in enumerate(lines) if s.startswith("[STATE & PERMISSION]"))
            after_reason = next(i for i in range(idx, len(lines)) if lines[i].startswith("- Reason:"))
            lines.insert(after_reason + 1, f"- Structural failure cause: {structural_failure_cause}")
        lines.insert(-2, f"TIME={now}")
        return "\n".join(lines)

    def format_governance_event(
        self,
        event_type: str,
        note: str,
        dedupe_key: Optional[str] = None,
        pnl_pct: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Format a governance event for Telegram (full audit format)."""
        return self._build_audit_message(
            event_type,
            note,
            dedupe_key=dedupe_key,
            pnl_pct=pnl_pct,
            entry_ref_price=kwargs.get("entry_ref_price"),
            entry_ref_time=kwargs.get("entry_ref_time"),
            current_ref_price=kwargs.get("current_ref_price"),
            peak_ref_price=kwargs.get("peak_ref_price"),
            exit_ref_price=kwargs.get("exit_ref_price"),
            expansion_pct=kwargs.get("expansion_pct"),
            current_state=kwargs.get("current_state"),
            exit_permission=kwargs.get("exit_permission"),
            reason=kwargs.get("reason") or note,
            volatility_state=kwargs.get("volatility_state"),
            dominance=kwargs.get("dominance"),
            continuation_quality=kwargs.get("continuation_quality"),
            tp25_status=kwargs.get("tp25_status", "not reached"),
            tp50_status=kwargs.get("tp50_status", "not reached"),
            tp100_status=kwargs.get("tp100_status", "not reached"),
            structural_failure_cause=kwargs.get("structural_failure_cause"),
        )
    
    def emit_entry(self, dedupe_key: str, ticker: str, side: str, strike: float, entry_price: float) -> bool:
        """Emit position entry event (full audit format). Instruction: BUY / enter now."""
        msg = self._build_audit_message(
            "BUY | POSITION_OPENED | STATE=OBSERVE",
            "Volatility displacement detected; structure fragile. ENTER NOW.",
            dedupe_key=dedupe_key,
            entry_ref_price=entry_price,
            entry_ref_time=_et_now(),
            current_ref_price=entry_price,
            current_state="OBSERVE",
            exit_permission="ALLOWED",
            reason="Structure fragile; exit allowed. Entry reference set.",
            volatility_state="displacing",
            dominance="present",
            continuation_quality="fragile",
            tp25_status="not reached",
            tp50_status="not reached",
            tp100_status="not reached",
        )
        self.event_count += 1
        logger.info(f"TELEMETRY | BUY | POSITION_OPENED | {dedupe_key}")
        if self.telegram_enabled:
            try:
                return send_alert(msg)
            except Exception as e:
                logger.error(f"Telemetry send error: {e}")
                return False
        return True
    
    def emit_exit(
        self,
        dedupe_key: str,
        exit_reason: str,
        exit_price: float,
        entry_price: float,
        pnl_pct: float,
        final_state: str,
        peak_price: Optional[float] = None,
    ) -> bool:
        """Emit position exit event (full audit format). Instruction: SELL / exit now."""
        exit_perm = "REQUIRED" if final_state == "RELEASE" else "ALLOWED"
        msg = self._build_audit_message(
            "SELL | POSITION_CLOSED | " + exit_reason,
            "Exit executed. EXIT NOW. Final state=" + final_state,
            dedupe_key=dedupe_key,
            pnl_pct=pnl_pct,
            entry_ref_price=entry_price,
            current_ref_price=exit_price,
            peak_ref_price=peak_price,
            exit_ref_price=_fmt_price(exit_price),
            expansion_pct=pnl_pct,
            current_state=final_state,
            exit_permission=exit_perm,
            reason="Exit executed; " + exit_reason + "; PnL " + f"{pnl_pct:+.1f}%",
            volatility_state="decaying" if final_state == "RELEASE" else "N/A",
            dominance="lost" if final_state == "RELEASE" else "N/A",
            continuation_quality="inert" if final_state == "RELEASE" else "N/A",
            tp25_status="observed" if pnl_pct >= 25 else "not reached",
            tp50_status="observed" if pnl_pct >= 50 else "not reached",
            tp100_status="observed" if pnl_pct >= 100 else "not reached",
        )
        self.event_count += 1
        logger.info(f"TELEMETRY | SELL | POSITION_CLOSED | {dedupe_key}")
        if self.telegram_enabled:
            try:
                return send_alert(msg)
            except Exception as e:
                logger.error(f"Telemetry send error: {e}")
                return False
        return True
    
    def emit_runner_lock_alert(
        self,
        dedupe_key: str,
        pnl_pct: float,
        entry_ref_price: Optional[float] = None,
        current_ref_price: Optional[float] = None,
        peak_ref_price: Optional[float] = None,
    ) -> bool:
        """Emit critical RUNNER_LOCK entered alert (full audit). Exit FORBIDDEN; do not interfere."""
        msg = self._build_audit_message(
            "RUNNER_LOCK_ENTERED | EXIT FORBIDDEN",
            "Profit magnitude does not justify exit. Structure intact. DO NOT INTERFERE.",
            dedupe_key=dedupe_key,
            pnl_pct=pnl_pct,
            entry_ref_price=entry_ref_price,
            current_ref_price=current_ref_price,
            peak_ref_price=peak_ref_price,
            expansion_pct=pnl_pct,
            current_state="RUNNER_LOCK",
            exit_permission="FORBIDDEN",
            reason="Profit magnitude does not justify exit. Axiom A1 active.",
            volatility_state="expanding",
            dominance="present",
            continuation_quality="self-reinforcing",
            tp25_status="suppressed",
            tp50_status="suppressed",
            tp100_status="suppressed",
        )
        self.event_count += 1
        if self.telegram_enabled:
            try:
                return send_alert(msg)
            except Exception as e:
                logger.error(f"Telemetry send error: {e}")
                return False
        return True
    
    def emit_structure_break_alert(
        self,
        dedupe_key: str,
        reason: str,
        final_pnl: float,
        entry_ref_price: Optional[float] = None,
        exit_ref_price_val: Optional[float] = None,
        peak_ref_price: Optional[float] = None,
    ) -> bool:
        """Emit critical STRUCTURE BREAK alert (full audit). EXIT NOW REQUIRED."""
        exit_str = _fmt_price(exit_ref_price_val) if exit_ref_price_val is not None else "N/A"
        msg = self._build_audit_message(
            "RELEASE | STRUCTURE_BREAK_DETECTED | EXIT NOW REQUIRED",
            "Continuation coherence collapsed. EXECUTE EXIT IMMEDIATELY.",
            dedupe_key=dedupe_key,
            pnl_pct=final_pnl,
            entry_ref_price=entry_ref_price,
            current_ref_price=exit_ref_price_val,
            peak_ref_price=peak_ref_price,
            exit_ref_price=exit_str,
            expansion_pct=final_pnl,
            current_state="RELEASE",
            exit_permission="REQUIRED",
            reason="Structural invalidation; exit required. Axiom A2 triggered.",
            volatility_state="decaying",
            dominance="lost",
            continuation_quality="inert",
            structural_failure_cause=reason,
            tp25_status="observed",
            tp50_status="observed",
            tp100_status="observed",
        )
        self.event_count += 1
        if self.telegram_enabled:
            try:
                return send_alert(msg)
            except Exception as e:
                logger.error(f"Telemetry send error: {e}")
                return False
        return True
    
    def emit_preregistration_locked(self, session_id: str, axioms: list) -> bool:
        """Emit preregistration lock event."""
        msg_lines = [
            "🔐 *PREREGISTRATION LOCKED*",
            "",
            f"Session: {session_id}",
            "",
            "*Axioms Frozen:*",
        ]
        for i, axiom in enumerate(axioms, 1):
            msg_lines.append(f"A{i}) {axiom}")
        msg_lines.extend([
            "",
            "_No post-hoc tuning allowed_",
            f"TIME={_et_now()}",
        ])
        return send_alert("\n".join(msg_lines)) if self.telegram_enabled else True
    
    def _get_emoji(self, event_type: str) -> str:
        """Get emoji for event type."""
        if "RUNNER_LOCK_ENTERED" in event_type:
            return "🔒"
        elif "RUNNER_LOCK_HEARTBEAT" in event_type:
            return "💓"
        elif "STRUCTURE_BREAK" in event_type:
            return "⚠️"
        elif "TP_SUPPRESSED" in event_type:
            return "🚫"
        elif "TP_CHECKPOINT" in event_type:
            return "📍"
        elif "STATE_TRANSITION" in event_type:
            return "🔄"
        elif "OBSERVE" in event_type:
            return "👁️"
        elif "CANDIDATE" in event_type:
            return "🎯"
        elif "RELEASE" in event_type:
            return "🚨"
        elif "PREREGISTRATION" in event_type:
            return "🔐"
        else:
            return "📊"


# Singleton instance
_telemetry_bus: Optional[TelemetryEventBus] = None


def get_telemetry_bus() -> TelemetryEventBus:
    """Get or create the singleton telemetry bus."""
    global _telemetry_bus
    if _telemetry_bus is None:
        _telemetry_bus = TelemetryEventBus()
    return _telemetry_bus
