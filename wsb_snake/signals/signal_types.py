from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

class SignalTier(Enum):
    """Signal quality tiers for routing."""
    A_PLUS = "A+"    # Immediate alert
    A = "A"          # High priority alert
    B = "B"          # Watchlist / digest
    C = "C"          # Log only
    BLOCKED = "X"    # Risk blocked

class SignalAction(Enum):
    """Recommended action for a signal."""
    WATCH = "WATCH"
    WAIT = "WAIT"
    ENTER = "ENTER"
    EXIT = "EXIT"
    HEDGE = "HEDGE"

class TimeHorizon(Enum):
    """Trading time horizon."""
    ZERO_DTE = "0DTE"
    INTRADAY = "INTRADAY"
    SWING = "SWING"
    POSITION = "POSITION"

@dataclass
class MarketData:
    """Market data for a ticker."""
    price: float = 0.0
    volume: int = 0
    change_pct: float = 0.0
    spread_pct: float = 0.0
    avg_volume: int = 0
    volatility: float = 0.0
    vwap: float = 0.0
    
@dataclass
class SocialMetrics:
    """Social/crowd metrics for a ticker."""
    mention_count: int = 0
    velocity: float = 0.0          # mentions per minute
    acceleration: float = 0.0       # velocity change
    author_count: int = 0           # unique authors
    upvote_ratio: float = 0.0
    comment_count: int = 0
    sentiment_score: float = 0.0    # -1 to +1
    intent_tags: List[str] = field(default_factory=list)

@dataclass
class RiskFlags:
    """Risk assessment flags."""
    low_liquidity: bool = False
    wide_spread: bool = False
    pump_detected: bool = False
    high_volatility: bool = False
    news_uncertainty: bool = False
    regime_unfavorable: bool = False
    blocked: bool = False
    block_reason: str = ""

@dataclass
class Signal:
    """Complete signal with all components."""
    ticker: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Core scoring
    score: float = 0.0              # 0-100
    tier: SignalTier = SignalTier.C
    action: SignalAction = SignalAction.WATCH
    horizon: TimeHorizon = TimeHorizon.INTRADAY
    confidence: float = 0.0         # 0-1
    
    # Component data
    market: MarketData = field(default_factory=MarketData)
    social: SocialMetrics = field(default_factory=SocialMetrics)
    risk: RiskFlags = field(default_factory=RiskFlags)
    
    # AI summary
    summary: str = ""
    why: List[str] = field(default_factory=list)
    levels: Dict[str, float] = field(default_factory=dict)  # support, resistance, vwap
    
    # Evidence bundle
    evidence: List[str] = field(default_factory=list)  # links, excerpts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp.isoformat(),
            'score': self.score,
            'tier': self.tier.value,
            'action': self.action.value,
            'horizon': self.horizon.value,
            'confidence': self.confidence,
            'market': {
                'price': self.market.price,
                'volume': self.market.volume,
                'change_pct': self.market.change_pct,
            },
            'social': {
                'mention_count': self.social.mention_count,
                'velocity': self.social.velocity,
                'sentiment': self.social.sentiment_score,
            },
            'risk': {
                'blocked': self.risk.blocked,
                'block_reason': self.risk.block_reason,
            },
            'summary': self.summary,
            'why': self.why,
            'levels': self.levels,
            'evidence': self.evidence[:5],  # Limit evidence
        }
