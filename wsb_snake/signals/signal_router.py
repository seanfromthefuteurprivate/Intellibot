from typing import List, Tuple
from wsb_snake.signals.signal_types import Signal, SignalTier, SignalAction
from wsb_snake.utils.logger import log

# Thresholds for signal routing
TIER_THRESHOLDS = {
    'A_PLUS': 85,   # Score >= 85 -> Immediate alert
    'A': 70,        # Score >= 70 -> Priority alert
    'B': 50,        # Score >= 50 -> Watchlist
    'C': 30,        # Score >= 30 -> Log only
    # Below 30 -> Ignore
}

def classify_signal(signal: Signal) -> Signal:
    """
    Classify a signal into A+/A/B/C tier based on score and risk.
    Mutates and returns the signal.
    """
    # Check for risk blocks first
    if signal.risk.blocked:
        signal.tier = SignalTier.BLOCKED
        signal.action = SignalAction.WAIT
        log.info(f"Signal {signal.ticker} BLOCKED: {signal.risk.block_reason}")
        return signal
    
    # Classify by score
    score = signal.score
    
    if score >= TIER_THRESHOLDS['A_PLUS']:
        signal.tier = SignalTier.A_PLUS
        signal.action = SignalAction.ENTER
    elif score >= TIER_THRESHOLDS['A']:
        signal.tier = SignalTier.A
        signal.action = SignalAction.WATCH
    elif score >= TIER_THRESHOLDS['B']:
        signal.tier = SignalTier.B
        signal.action = SignalAction.WATCH
    elif score >= TIER_THRESHOLDS['C']:
        signal.tier = SignalTier.C
        signal.action = SignalAction.WAIT
    else:
        signal.tier = SignalTier.C
        signal.action = SignalAction.WAIT
    
    return signal


def route_signals(signals: List[Signal]) -> Tuple[List[Signal], List[Signal], List[Signal]]:
    """
    Route signals into three buckets:
    - alerts: A+ and A tier (immediate Telegram alerts)
    - watchlist: B tier (digest)
    - logged: C tier and blocked (log only)
    
    Returns:
        Tuple of (alerts, watchlist, logged)
    """
    alerts = []
    watchlist = []
    logged = []
    
    for signal in signals:
        # Classify if not already done
        if signal.tier == SignalTier.C and signal.score > 0:
            classify_signal(signal)
        
        if signal.tier in (SignalTier.A_PLUS, SignalTier.A):
            alerts.append(signal)
        elif signal.tier == SignalTier.B:
            watchlist.append(signal)
        else:
            logged.append(signal)
    
    # Sort by score descending
    alerts.sort(key=lambda s: s.score, reverse=True)
    watchlist.sort(key=lambda s: s.score, reverse=True)
    
    log.info(f"Routed {len(signals)} signals: {len(alerts)} alerts, {len(watchlist)} watchlist, {len(logged)} logged")
    
    return alerts, watchlist, logged


def should_alert_immediately(signal: Signal) -> bool:
    """Check if a signal should trigger an immediate alert."""
    return signal.tier in (SignalTier.A_PLUS, SignalTier.A) and not signal.risk.blocked
