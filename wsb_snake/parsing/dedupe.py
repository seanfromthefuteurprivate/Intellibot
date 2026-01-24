from typing import List, Dict, Any
from collections import defaultdict
from datetime import datetime, timedelta

class SignalDeduplicator:
    """
    Prevents duplicate signals for the same ticker within a cooldown period.
    """
    
    def __init__(self, cooldown_minutes: int = 30):
        self.cooldown_minutes = cooldown_minutes
        self.last_alert: Dict[str, datetime] = {}
        
    def should_alert(self, ticker: str) -> bool:
        """
        Check if we should send an alert for this ticker.
        Returns True if ticker hasn't been alerted recently.
        """
        now = datetime.utcnow()
        
        if ticker not in self.last_alert:
            return True
            
        elapsed = now - self.last_alert[ticker]
        return elapsed >= timedelta(minutes=self.cooldown_minutes)
    
    def mark_alerted(self, ticker: str):
        """Record that we sent an alert for this ticker."""
        self.last_alert[ticker] = datetime.utcnow()
    
    def filter_tickers(self, tickers: List[str]) -> List[str]:
        """
        Filter out tickers that were alerted recently.
        """
        return [t for t in tickers if self.should_alert(t)]
    
    def reset(self, ticker: str = None):
        """
        Reset cooldown for a specific ticker or all tickers.
        """
        if ticker:
            self.last_alert.pop(ticker, None)
        else:
            self.last_alert.clear()


def dedupe_mentions(mentions: List[Dict[str, Any]], key: str = 'ticker') -> List[Dict[str, Any]]:
    """
    Deduplicate a list of mention dicts, keeping the highest-scored one per ticker.
    """
    best = {}
    
    for mention in mentions:
        ticker = mention.get(key)
        if not ticker:
            continue
            
        if ticker not in best:
            best[ticker] = mention
        else:
            # Keep the one with higher score
            if mention.get('score', 0) > best[ticker].get('score', 0):
                best[ticker] = mention
    
    return list(best.values())


# Global deduplicator instance
deduplicator = SignalDeduplicator(cooldown_minutes=30)
