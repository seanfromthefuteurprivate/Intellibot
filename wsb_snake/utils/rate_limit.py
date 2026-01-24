import time
from collections import defaultdict
from typing import Dict

class RateLimiter:
    """
    Simple rate limiter for API calls.
    """
    
    def __init__(self):
        self.last_call: Dict[str, float] = defaultdict(float)
        self.cooldowns: Dict[str, float] = {
            'reddit': 2.0,      # 2 seconds between Reddit calls
            'alpaca': 0.5,      # 0.5 seconds between Alpaca calls
            'openai': 1.0,      # 1 second between OpenAI calls
            'telegram': 0.5,    # 0.5 seconds between Telegram calls
        }
    
    def wait_if_needed(self, service: str):
        """
        Block until the cooldown period has passed for the given service.
        """
        cooldown = self.cooldowns.get(service, 1.0)
        elapsed = time.time() - self.last_call[service]
        
        if elapsed < cooldown:
            time.sleep(cooldown - elapsed)
        
        self.last_call[service] = time.time()
    
    def can_call(self, service: str) -> bool:
        """
        Check if we can make a call without blocking.
        """
        cooldown = self.cooldowns.get(service, 1.0)
        elapsed = time.time() - self.last_call[service]
        return elapsed >= cooldown

# Global rate limiter instance
limiter = RateLimiter()
