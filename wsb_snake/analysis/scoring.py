from datetime import datetime
from typing import List, Dict, Any
from wsb_snake.signals.signal_types import Signal, MarketData, SocialMetrics, TimeHorizon
from wsb_snake.analysis.risk_model import assess_risk, calculate_risk_penalty
from wsb_snake.signals.signal_router import classify_signal
from wsb_snake.utils.logger import log

# Scoring weights
WEIGHTS = {
    'social_velocity': 15,       # Mentions per minute
    'social_acceleration': 10,   # Velocity change
    'author_diversity': 10,      # Unique authors (anti-pump)
    'sentiment': 10,             # Net bullish/bearish
    'price_momentum': 20,        # Price change %
    'volume_spike': 15,          # Volume vs average
    'market_alignment': 10,      # Alignment with SPY
    'intent_bonus': 10,          # 0DTE, earnings, squeeze tags
}


def score_ticker(
    ticker: str,
    market_data: Dict[str, Any],
    social_metrics: Dict[str, Any] = None,
) -> Signal:
    """
    Score a single ticker and return a complete Signal object.
    """
    signal = Signal(
        ticker=ticker,
        timestamp=datetime.utcnow(),
    )
    
    # Populate market data
    if market_data:
        signal.market = MarketData(
            price=market_data.get('price', 0),
            volume=market_data.get('volume', 0),
            change_pct=market_data.get('change', 0),
            spread_pct=market_data.get('spread', 0),
            avg_volume=market_data.get('avg_volume', 0),
            volatility=market_data.get('volatility', 0),
        )
    
    # Populate social metrics
    if social_metrics:
        signal.social = SocialMetrics(
            mention_count=social_metrics.get('count', 0),
            velocity=social_metrics.get('velocity', 0),
            acceleration=social_metrics.get('acceleration', 0),
            author_count=social_metrics.get('authors', 0),
            sentiment_score=social_metrics.get('sentiment', 0),
            intent_tags=social_metrics.get('intents', []),
        )
    
    # Calculate base score
    score = 0.0
    reasons = []
    
    # Social velocity score (0-15)
    if signal.social.velocity > 0:
        vel_score = min(signal.social.velocity * 3, WEIGHTS['social_velocity'])
        score += vel_score
        if vel_score > 10:
            reasons.append(f"High social velocity ({signal.social.velocity:.1f}/min)")
    
    # Price momentum score (0-20)
    change_pct = abs(signal.market.change_pct) * 100
    if change_pct > 0:
        mom_score = min(change_pct * 4, WEIGHTS['price_momentum'])
        score += mom_score
        direction = "up" if signal.market.change_pct > 0 else "down"
        if mom_score > 10:
            reasons.append(f"Strong {direction} move ({signal.market.change_pct*100:+.1f}%)")
    
    # Volume spike score (0-15)
    if signal.market.avg_volume > 0 and signal.market.volume > 0:
        vol_ratio = signal.market.volume / signal.market.avg_volume
        if vol_ratio > 1.5:
            vol_score = min((vol_ratio - 1) * 10, WEIGHTS['volume_spike'])
            score += vol_score
            if vol_score > 10:
                reasons.append(f"Volume spike ({vol_ratio:.1f}x avg)")
    
    # Sentiment score (0-10)
    if signal.social.sentiment_score != 0:
        sent_score = (signal.social.sentiment_score + 1) / 2 * WEIGHTS['sentiment']
        score += sent_score
        if signal.social.sentiment_score > 0.5:
            reasons.append("Strong bullish sentiment")
        elif signal.social.sentiment_score < -0.5:
            reasons.append("Strong bearish sentiment")
    
    # Intent bonus (0-10)
    high_value_intents = ['0DTE', 'EARNINGS', 'SQUEEZE', 'MOMENTUM']
    matching_intents = [i for i in signal.social.intent_tags if i in high_value_intents]
    if matching_intents:
        intent_score = len(matching_intents) * 3
        score += min(intent_score, WEIGHTS['intent_bonus'])
        reasons.append(f"Intent: {', '.join(matching_intents)}")
    
    # Base score for having any data
    if signal.market.price > 0:
        score += 10  # Base score
    
    # Assess risk and apply penalty
    assess_risk(signal)
    penalty = calculate_risk_penalty(signal)
    score = max(0, score - penalty)
    
    # Store final score and reasons
    signal.score = min(score, 100)
    signal.why = reasons[:5]
    
    # Classify into tier
    classify_signal(signal)
    
    # Set confidence based on data completeness
    data_points = sum([
        1 if signal.market.price > 0 else 0,
        1 if signal.market.volume > 0 else 0,
        1 if signal.social.mention_count > 0 else 0,
        1 if signal.social.velocity > 0 else 0,
    ])
    signal.confidence = data_points / 4
    
    log.debug(f"Scored {ticker}: {signal.score:.1f} ({signal.tier.value})")
    
    return signal


def score_tickers(
    tickers: List[str],
    market_data: Dict[str, Dict[str, Any]],
    social_data: Dict[str, Dict[str, Any]] = None,
) -> List[Signal]:
    """
    Score multiple tickers and return sorted signals.
    """
    if social_data is None:
        social_data = {}
    
    signals = []
    for ticker in tickers:
        mkt = market_data.get(ticker, {})
        soc = social_data.get(ticker, {})
        
        signal = score_ticker(ticker, mkt, soc)
        signals.append(signal)
    
    # Sort by score descending
    signals.sort(key=lambda s: s.score, reverse=True)
    
    log.info(f"Scored {len(signals)} tickers. Top: {[s.ticker for s in signals[:3]]}")
    
    return signals
