from wsb_snake.signals.signal_types import Signal, RiskFlags, MarketData, SocialMetrics
from wsb_snake.utils.logger import log

# Risk thresholds
THRESHOLDS = {
    'min_volume': 100000,           # Minimum daily volume
    'max_spread_pct': 0.5,          # Maximum spread as % of price
    'max_volatility': 0.10,         # Maximum daily volatility (10%)
    'min_author_diversity': 3,       # Minimum unique authors for mentions
    'max_mention_velocity': 50,      # Mentions/min above this = pump risk
}

def assess_risk(signal: Signal) -> Signal:
    """
    Assess risk factors and populate the risk flags.
    Mutates and returns the signal.
    """
    flags = RiskFlags()
    block_reasons = []
    
    market = signal.market
    social = signal.social
    
    # Liquidity check
    if market.volume < THRESHOLDS['min_volume']:
        flags.low_liquidity = True
        if market.volume < THRESHOLDS['min_volume'] / 2:
            block_reasons.append(f"Very low volume ({market.volume:,})")
    
    # Spread check
    if market.spread_pct > THRESHOLDS['max_spread_pct']:
        flags.wide_spread = True
        if market.spread_pct > THRESHOLDS['max_spread_pct'] * 2:
            block_reasons.append(f"Wide spread ({market.spread_pct:.2f}%)")
    
    # Volatility check
    if market.volatility > THRESHOLDS['max_volatility']:
        flags.high_volatility = True
        # Don't block on high volatility alone, it's expected in momentum plays
    
    # Pump detection (low author diversity + high velocity)
    if social.author_count < THRESHOLDS['min_author_diversity']:
        if social.velocity > THRESHOLDS['max_mention_velocity']:
            flags.pump_detected = True
            block_reasons.append(f"Pump suspected ({social.author_count} authors, {social.velocity:.1f}/min)")
    
    # Set blocked status if any hard blocks
    if block_reasons:
        flags.blocked = True
        flags.block_reason = "; ".join(block_reasons)
        log.warning(f"Risk block for {signal.ticker}: {flags.block_reason}")
    
    signal.risk = flags
    return signal


def calculate_risk_penalty(signal: Signal) -> float:
    """
    Calculate a risk penalty to subtract from the signal score.
    Returns a value between 0 and 30.
    """
    penalty = 0.0
    flags = signal.risk
    
    if flags.low_liquidity:
        penalty += 10
    if flags.wide_spread:
        penalty += 10
    if flags.high_volatility:
        penalty += 5
    if flags.pump_detected:
        penalty += 15
    if flags.news_uncertainty:
        penalty += 5
    if flags.regime_unfavorable:
        penalty += 5
    
    return min(penalty, 30)  # Cap at 30


def is_market_open() -> bool:
    """
    Check if US stock market is currently open.
    Simple heuristic based on time.
    """
    from datetime import datetime, timezone
    import pytz
    
    try:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        # Weekday check (Mon=0, Sun=6)
        if now.weekday() >= 5:
            return False
        
        # Market hours: 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    except:
        # If timezone check fails, assume open during typical hours
        return True
