"""
Congressional Trading Collector

Tracks politician stock trades from House and Senate.
100% FREE - No API key required!

Data sources:
- housestockwatcher.com/api (House of Representatives)
- senatestockwatcher.com/api (Senate)

Edge: Politicians often trade ahead of legislation, 
regulatory decisions, and non-public information.
"""

import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict

from wsb_snake.utils.logger import log


class CongressionalTradingCollector:
    """
    Collects congressional trading data from public APIs.
    
    Features:
    - House of Representatives trades
    - Senate trades
    - Filters for our universe tickers
    - Recent trade detection (last 30 days)
    """
    
    HOUSE_API = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
    SENATE_API = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/aggregate/all_transactions.json"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "WSBSnake/1.0 (Financial Research)",
            "Accept": "application/json",
        })
        
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 3600
        self.last_call = 0
        self.min_interval = 2.0
        
        log.info("Congressional Trading collector initialized (free)")
    
    def _rate_limit(self):
        """Respect rate limits"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    def get_house_trades(self, days_back: int = 30) -> List[Dict]:
        """
        Get recent House of Representatives trades.
        
        Args:
            days_back: How many days to look back
            
        Returns:
            List of recent trades
        """
        cache_key = f"house:{days_back}"
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["data"]
        
        self._rate_limit()
        
        try:
            response = self.session.get(self.HOUSE_API, timeout=30)
            
            if response.status_code == 200:
                all_trades = response.json()
                
                cutoff = datetime.now() - timedelta(days=days_back)
                recent_trades = []
                
                for trade in all_trades:
                    try:
                        trade_date = datetime.strptime(
                            trade.get("transaction_date", "1900-01-01"),
                            "%Y-%m-%d"
                        )
                        if trade_date >= cutoff:
                            recent_trades.append({
                                "chamber": "House",
                                "representative": trade.get("representative", "Unknown"),
                                "ticker": trade.get("ticker", ""),
                                "transaction_date": trade.get("transaction_date"),
                                "disclosure_date": trade.get("disclosure_date"),
                                "type": trade.get("type", ""),
                                "amount": trade.get("amount", ""),
                                "party": trade.get("party", ""),
                                "state": trade.get("state", ""),
                            })
                    except:
                        continue
                
                self.cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": recent_trades
                }
                
                return recent_trades
            else:
                log.debug(f"House API returned {response.status_code}")
                return []
                
        except Exception as e:
            log.debug(f"House trades error: {e}")
            return []
    
    def get_senate_trades(self, days_back: int = 30) -> List[Dict]:
        """
        Get recent Senate trades.
        
        Args:
            days_back: How many days to look back
            
        Returns:
            List of recent trades
        """
        cache_key = f"senate:{days_back}"
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached["timestamp"] < self.cache_ttl:
                return cached["data"]
        
        self._rate_limit()
        
        try:
            response = self.session.get(self.SENATE_API, timeout=30)
            
            if response.status_code == 200:
                all_trades = response.json()
                
                cutoff = datetime.now() - timedelta(days=days_back)
                recent_trades = []
                
                for trade in all_trades:
                    try:
                        trade_date = datetime.strptime(
                            trade.get("transaction_date", "1900-01-01"),
                            "%Y-%m-%d"
                        )
                        if trade_date >= cutoff:
                            recent_trades.append({
                                "chamber": "Senate",
                                "senator": trade.get("senator", "Unknown"),
                                "ticker": trade.get("ticker", ""),
                                "transaction_date": trade.get("transaction_date"),
                                "disclosure_date": trade.get("disclosure_date"),
                                "type": trade.get("type", ""),
                                "amount": trade.get("amount", ""),
                                "party": trade.get("party", ""),
                                "state": trade.get("state", ""),
                            })
                    except:
                        continue
                
                self.cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": recent_trades
                }
                
                return recent_trades
            else:
                log.debug(f"Senate API returned {response.status_code}")
                return []
                
        except Exception as e:
            log.debug(f"Senate trades error: {e}")
            return []
    
    def get_trades_for_ticker(self, symbol: str, days_back: int = 30) -> Dict[str, Any]:
        """
        Get all congressional trades for a specific ticker.
        
        Args:
            symbol: Stock ticker
            days_back: Days to look back
            
        Returns:
            Dict with trade summary and signal
        """
        house_trades = self.get_house_trades(days_back)
        senate_trades = self.get_senate_trades(days_back)
        
        symbol_upper = symbol.upper()
        
        house_matches = [t for t in house_trades if t.get("ticker", "").upper() == symbol_upper]
        senate_matches = [t for t in senate_trades if t.get("ticker", "").upper() == symbol_upper]
        
        all_matches = house_matches + senate_matches
        
        buys = [t for t in all_matches if "purchase" in t.get("type", "").lower()]
        sells = [t for t in all_matches if "sale" in t.get("type", "").lower()]
        
        if len(buys) > len(sells) * 2:
            signal = "strong_bullish"
            boost = 8
        elif len(buys) > len(sells):
            signal = "bullish"
            boost = 5
        elif len(sells) > len(buys) * 2:
            signal = "strong_bearish"
            boost = -8
        elif len(sells) > len(buys):
            signal = "bearish"
            boost = -5
        else:
            signal = "neutral"
            boost = 0
        
        return {
            "symbol": symbol,
            "total_trades": len(all_matches),
            "house_trades": len(house_matches),
            "senate_trades": len(senate_matches),
            "buys": len(buys),
            "sells": len(sells),
            "signal": signal,
            "score_boost": boost,
            "recent_trades": all_matches[:5],
            "days_analyzed": days_back,
        }
    
    def get_hot_tickers(self, days_back: int = 14) -> List[Dict]:
        """
        Get tickers with most congressional trading activity.
        
        Returns:
            List of tickers ranked by activity
        """
        house_trades = self.get_house_trades(days_back)
        senate_trades = self.get_senate_trades(days_back)
        
        all_trades = house_trades + senate_trades
        
        ticker_counts = defaultdict(lambda: {"buys": 0, "sells": 0, "total": 0})
        
        for trade in all_trades:
            ticker = trade.get("ticker", "").upper()
            if ticker and len(ticker) <= 5:
                ticker_counts[ticker]["total"] += 1
                if "purchase" in trade.get("type", "").lower():
                    ticker_counts[ticker]["buys"] += 1
                elif "sale" in trade.get("type", "").lower():
                    ticker_counts[ticker]["sells"] += 1
        
        hot_tickers = []
        for ticker, counts in ticker_counts.items():
            if counts["total"] >= 2:
                net_sentiment = counts["buys"] - counts["sells"]
                hot_tickers.append({
                    "ticker": ticker,
                    "total_trades": counts["total"],
                    "buys": counts["buys"],
                    "sells": counts["sells"],
                    "net_sentiment": net_sentiment,
                    "signal": "bullish" if net_sentiment > 0 else "bearish" if net_sentiment < 0 else "neutral",
                })
        
        hot_tickers.sort(key=lambda x: x["total_trades"], reverse=True)
        
        return hot_tickers[:20]


congressional_trading = CongressionalTradingCollector()
