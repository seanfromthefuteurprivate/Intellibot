from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any

class TimeWindowTracker:
    """
    Tracks ticker mentions and metrics over sliding time windows.
    Used for computing momentum/velocity signals.
    """
    
    def __init__(self, window_minutes: int = 15):
        self.window_minutes = window_minutes
        self.mentions: Dict[str, List[datetime]] = defaultdict(list)
        
    def add_mention(self, ticker: str, timestamp: datetime = None):
        """Record a mention of a ticker."""
        if timestamp is None:
            timestamp = datetime.utcnow()
        self.mentions[ticker].append(timestamp)
        
    def get_velocity(self, ticker: str) -> float:
        """
        Calculate mentions per minute for a ticker in the current window.
        """
        self._prune_old_mentions(ticker)
        count = len(self.mentions[ticker])
        return count / self.window_minutes if self.window_minutes > 0 else 0
    
    def get_acceleration(self, ticker: str) -> float:
        """
        Compare current window velocity to previous window.
        Returns positive if accelerating, negative if decelerating.
        """
        now = datetime.utcnow()
        cutoff_current = now - timedelta(minutes=self.window_minutes)
        cutoff_previous = cutoff_current - timedelta(minutes=self.window_minutes)
        
        current_count = sum(1 for t in self.mentions[ticker] if t >= cutoff_current)
        previous_count = sum(1 for t in self.mentions[ticker] 
                           if cutoff_previous <= t < cutoff_current)
        
        if previous_count == 0:
            return current_count  # All new = high acceleration
        return (current_count - previous_count) / previous_count
    
    def _prune_old_mentions(self, ticker: str):
        """Remove mentions older than 2x window size."""
        cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes * 2)
        self.mentions[ticker] = [t for t in self.mentions[ticker] if t >= cutoff]
    
    def get_top_tickers(self, limit: int = 10) -> List[tuple]:
        """Return top tickers by velocity."""
        velocities = []
        for ticker in list(self.mentions.keys()):
            vel = self.get_velocity(ticker)
            if vel > 0:
                velocities.append((ticker, vel))
        
        velocities.sort(key=lambda x: x[1], reverse=True)
        return velocities[:limit]

# Global tracker instance
tracker = TimeWindowTracker(window_minutes=15)
