"""
Finnhub Data Collector - News Sentiment, Social Sentiment, Insider Sentiment

Free tier: 60 calls/minute
Data:
- Company news with sentiment
- Social media sentiment (Twitter, Reddit, StockTwits aggregated)
- Insider sentiment (MSPR - Monthly Share Purchase Ratio)
"""

import os
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from wsb_snake.utils.logger import log

try:
    import finnhub
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False
    log.warning("finnhub-python not installed")


class FinnhubCollector:
    """
    Collects news sentiment, social sentiment, and insider data from Finnhub.
    Free tier: 60 calls/min
    """
    
    def __init__(self):
        self.api_key = os.environ.get("FINNHUB_API_KEY", "")
        self.client = None
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 120
        self.last_call = 0
        self.min_interval = 1.1
        
        if FINNHUB_AVAILABLE and self.api_key:
            self.client = finnhub.Client(api_key=self.api_key)
            log.info("Finnhub collector initialized")
        else:
            if not self.api_key:
                log.warning("FINNHUB_API_KEY not set - collector disabled")
    
    def _rate_limit(self):
        """Ensure we don't exceed 60 calls/min"""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if still valid"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return data
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[key] = (data, time.time())
    
    def get_news_sentiment(self, ticker: str) -> Dict:
        """
        Get news sentiment for a ticker.
        Returns sentiment scores from recent news articles.
        """
        if not self.client:
            return {"sentiment": 0, "buzz": 0, "articles": 0, "source": "finnhub_unavailable"}
        
        cache_key = f"news_sentiment_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            news = self.client.company_news(
                ticker,
                _from=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d")
            )
            
            if not news:
                result = {"sentiment": 0, "buzz": 0, "articles": 0, "headlines": [], "source": "finnhub"}
                self._set_cache(cache_key, result)
                return result
            
            total_sentiment = 0
            headlines = []
            
            for article in news[:10]:
                headline = article.get("headline", "")
                summary = article.get("summary", "")
                
                sentiment = self._analyze_headline_sentiment(headline + " " + summary)
                total_sentiment += sentiment
                
                headlines.append({
                    "title": headline[:100],
                    "sentiment": sentiment,
                    "source": article.get("source", "unknown"),
                    "datetime": article.get("datetime", 0)
                })
            
            avg_sentiment = total_sentiment / len(news[:10]) if news else 0
            
            result = {
                "sentiment": round(avg_sentiment, 2),
                "buzz": min(len(news), 20),
                "articles": len(news),
                "headlines": headlines[:5],
                "source": "finnhub"
            }
            
            self._set_cache(cache_key, result)
            log.debug(f"Finnhub news sentiment for {ticker}: {avg_sentiment:.2f} ({len(news)} articles)")
            return result
            
        except Exception as e:
            log.warning(f"Finnhub news sentiment error for {ticker}: {e}")
            return {"sentiment": 0, "buzz": 0, "articles": 0, "source": "error"}
    
    def _analyze_headline_sentiment(self, text: str) -> float:
        """
        Simple keyword-based sentiment analysis for headlines.
        Returns -1 to +1 score.
        """
        text = text.lower()
        
        bullish_words = [
            "surge", "soar", "jump", "rally", "gain", "rise", "bull", "buy",
            "upgrade", "beat", "exceed", "strong", "growth", "profit", "win",
            "breakthrough", "success", "record", "high", "boom", "skyrocket",
            "outperform", "positive", "optimistic", "bullish", "momentum"
        ]
        
        bearish_words = [
            "fall", "drop", "crash", "plunge", "decline", "lose", "bear", "sell",
            "downgrade", "miss", "below", "weak", "loss", "fail", "cut",
            "warning", "concern", "risk", "low", "bust", "tank", "tumble",
            "underperform", "negative", "pessimistic", "bearish", "selloff"
        ]
        
        bull_count = sum(1 for word in bullish_words if word in text)
        bear_count = sum(1 for word in bearish_words if word in text)
        
        if bull_count == 0 and bear_count == 0:
            return 0
        
        score = (bull_count - bear_count) / max(bull_count + bear_count, 1)
        return max(-1, min(1, score))
    
    def get_social_sentiment(self, ticker: str) -> Dict:
        """
        Get social media sentiment for a ticker.
        Aggregates Reddit, Twitter, StockTwits mentions.
        """
        if not self.client:
            return {"score": 0, "mentions": 0, "positive_ratio": 0.5, "source": "finnhub_unavailable"}
        
        cache_key = f"social_sentiment_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            
            social = self.client.stock_social_sentiment(ticker)
            
            if not social:
                result = {"score": 0, "mentions": 0, "positive_ratio": 0.5, "source": "finnhub"}
                self._set_cache(cache_key, result)
                return result
            
            reddit_data = social.get("reddit", [])
            twitter_data = social.get("twitter", [])
            
            total_mentions = 0
            total_positive = 0
            total_negative = 0
            
            for entry in reddit_data[-24:]:
                total_mentions += entry.get("mention", 0)
                total_positive += entry.get("positiveMention", 0)
                total_negative += entry.get("negativeMention", 0)
            
            for entry in twitter_data[-24:]:
                total_mentions += entry.get("mention", 0)
                total_positive += entry.get("positiveMention", 0)
                total_negative += entry.get("negativeMention", 0)
            
            if total_mentions > 0:
                positive_ratio = total_positive / total_mentions
                score = (total_positive - total_negative) / total_mentions
            else:
                positive_ratio = 0.5
                score = 0
            
            result = {
                "score": round(score, 3),
                "mentions": total_mentions,
                "positive_ratio": round(positive_ratio, 3),
                "positive": total_positive,
                "negative": total_negative,
                "source": "finnhub"
            }
            
            self._set_cache(cache_key, result)
            log.debug(f"Finnhub social sentiment for {ticker}: score={score:.3f}, mentions={total_mentions}")
            return result
            
        except Exception as e:
            log.warning(f"Finnhub social sentiment error for {ticker}: {e}")
            return {"score": 0, "mentions": 0, "positive_ratio": 0.5, "source": "error"}
    
    def get_insider_sentiment(self, ticker: str) -> Dict:
        """
        Get insider sentiment using MSPR (Monthly Share Purchase Ratio).
        MSPR: 100 = strong buying, -100 = strong selling
        """
        if not self.client:
            return {"mspr": 0, "change": 0, "buying": False, "source": "finnhub_unavailable"}
        
        cache_key = f"insider_sentiment_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            
            sentiment = self.client.stock_insider_sentiment(ticker, "2024-01-01", datetime.now().strftime("%Y-%m-%d"))
            
            if not sentiment or not sentiment.get("data"):
                result = {"mspr": 0, "change": 0, "buying": False, "source": "finnhub"}
                self._set_cache(cache_key, result)
                return result
            
            data = sentiment.get("data", [])
            recent = data[-3:] if len(data) >= 3 else data
            
            total_mspr = 0
            total_change = 0
            
            for entry in recent:
                total_mspr += entry.get("mspr", 0)
                total_change += entry.get("change", 0)
            
            avg_mspr = total_mspr / len(recent) if recent else 0
            
            result = {
                "mspr": round(avg_mspr, 2),
                "change": total_change,
                "buying": avg_mspr > 10,
                "selling": avg_mspr < -10,
                "source": "finnhub"
            }
            
            self._set_cache(cache_key, result)
            log.debug(f"Finnhub insider sentiment for {ticker}: MSPR={avg_mspr:.2f}")
            return result
            
        except Exception as e:
            log.warning(f"Finnhub insider sentiment error for {ticker}: {e}")
            return {"mspr": 0, "change": 0, "buying": False, "source": "error"}
    
    def get_all_sentiment(self, ticker: str) -> Dict:
        """
        Get combined sentiment data for a ticker.
        Returns news, social, and insider sentiment in one call.
        """
        news = self.get_news_sentiment(ticker)
        social = self.get_social_sentiment(ticker)
        insider = self.get_insider_sentiment(ticker)
        
        combined_score = 0
        weights = {"news": 0.4, "social": 0.35, "insider": 0.25}
        
        if news.get("articles", 0) > 0:
            combined_score += news.get("sentiment", 0) * weights["news"]
        
        if social.get("mentions", 0) > 0:
            combined_score += social.get("score", 0) * weights["social"]
        
        if insider.get("buying") or insider.get("selling"):
            insider_score = 0.5 if insider.get("buying") else -0.5 if insider.get("selling") else 0
            combined_score += insider_score * weights["insider"]
        
        direction = "bullish" if combined_score > 0.1 else "bearish" if combined_score < -0.1 else "neutral"
        
        return {
            "ticker": ticker,
            "combined_score": round(combined_score, 3),
            "direction": direction,
            "news": news,
            "social": social,
            "insider": insider,
            "timestamp": datetime.now().isoformat()
        }
    
    def calculate_signal_boost(self, ticker: str, signal_direction: str) -> float:
        """
        Calculate boost/penalty for a signal based on Finnhub sentiment.
        
        Returns:
            -0.2 to +0.2 adjustment to apply to signal score
        """
        if not self.client:
            return 0
        
        try:
            sentiment = self.get_all_sentiment(ticker)
            combined = sentiment.get("combined_score", 0)
            direction = sentiment.get("direction", "neutral")
            
            if signal_direction == "bullish":
                if direction == "bullish" and combined > 0.2:
                    return 0.15
                elif direction == "bullish":
                    return 0.08
                elif direction == "bearish" and combined < -0.2:
                    return -0.15
                elif direction == "bearish":
                    return -0.08
            elif signal_direction == "bearish":
                if direction == "bearish" and combined < -0.2:
                    return 0.15
                elif direction == "bearish":
                    return 0.08
                elif direction == "bullish" and combined > 0.2:
                    return -0.15
                elif direction == "bullish":
                    return -0.08
            
            return 0
            
        except Exception as e:
            log.warning(f"Finnhub boost calculation error: {e}")
            return 0
    
    # ========================
    # NEW RUTHLESS METHODS
    # ========================
    
    def get_earnings_calendar(self, from_date: str = None, to_date: str = None) -> List[Dict]:
        """
        Get upcoming earnings announcements.
        CRITICAL for avoiding or exploiting earnings volatility.
        """
        if not self.client:
            return []
        
        cache_key = f"earnings_calendar_{from_date}_{to_date}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            
            if not from_date:
                from_date = datetime.now().strftime("%Y-%m-%d")
            if not to_date:
                to_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            
            calendar = self.client.earnings_calendar(
                _from=from_date,
                to=to_date,
                symbol=None,
                international=False
            )
            
            earnings = []
            for entry in calendar.get("earningsCalendar", []):
                earnings.append({
                    "symbol": entry.get("symbol", ""),
                    "date": entry.get("date", ""),
                    "hour": entry.get("hour", ""),
                    "eps_estimate": entry.get("epsEstimate"),
                    "eps_actual": entry.get("epsActual"),
                    "revenue_estimate": entry.get("revenueEstimate"),
                    "revenue_actual": entry.get("revenueActual"),
                    "quarter": entry.get("quarter"),
                    "year": entry.get("year"),
                })
            
            self._set_cache(cache_key, earnings)
            return earnings
            
        except Exception as e:
            log.warning(f"Finnhub earnings calendar error: {e}")
            return []
    
    def get_recommendation_trends(self, ticker: str) -> Dict:
        """
        Get analyst recommendation trends.
        Strong buy/sell consensus = confirmation for our signals.
        """
        if not self.client:
            return {"available": False}
        
        cache_key = f"recommendations_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            
            recs = self.client.recommendation_trends(ticker)
            
            if not recs:
                return {"available": False, "ticker": ticker}
            
            latest = recs[0] if recs else {}
            
            strong_buy = latest.get("strongBuy", 0)
            buy = latest.get("buy", 0)
            hold = latest.get("hold", 0)
            sell = latest.get("sell", 0)
            strong_sell = latest.get("strongSell", 0)
            
            total = strong_buy + buy + hold + sell + strong_sell
            
            if total > 0:
                bullish_pct = (strong_buy + buy) / total
                bearish_pct = (sell + strong_sell) / total
            else:
                bullish_pct = 0.5
                bearish_pct = 0.5
            
            if bullish_pct > 0.7:
                consensus = "STRONG_BUY"
                score = 2
            elif bullish_pct > 0.5:
                consensus = "BUY"
                score = 1
            elif bearish_pct > 0.7:
                consensus = "STRONG_SELL"
                score = -2
            elif bearish_pct > 0.5:
                consensus = "SELL"
                score = -1
            else:
                consensus = "HOLD"
                score = 0
            
            result = {
                "available": True,
                "ticker": ticker,
                "period": latest.get("period", ""),
                "strong_buy": strong_buy,
                "buy": buy,
                "hold": hold,
                "sell": sell,
                "strong_sell": strong_sell,
                "total_analysts": total,
                "bullish_pct": bullish_pct,
                "bearish_pct": bearish_pct,
                "consensus": consensus,
                "score": score,
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            log.warning(f"Finnhub recommendation trends error for {ticker}: {e}")
            return {"available": False, "ticker": ticker}
    
    def get_price_target(self, ticker: str) -> Dict:
        """
        Get analyst price targets.
        Useful for setting realistic profit targets.
        """
        if not self.client:
            return {"available": False}
        
        cache_key = f"price_target_{ticker}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            
            target = self.client.price_target(ticker)
            
            if not target:
                return {"available": False, "ticker": ticker}
            
            result = {
                "available": True,
                "ticker": ticker,
                "target_high": target.get("targetHigh"),
                "target_low": target.get("targetLow"),
                "target_mean": target.get("targetMean"),
                "target_median": target.get("targetMedian"),
                "last_updated": target.get("lastUpdated", ""),
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            log.warning(f"Finnhub price target error for {ticker}: {e}")
            return {"available": False, "ticker": ticker}
    
    def get_economic_calendar(self) -> List[Dict]:
        """
        Get upcoming economic events (CPI, FOMC, Jobs, etc.).
        CRITICAL for avoiding or exploiting macro volatility.
        """
        if not self.client:
            return []
        
        cache_key = "economic_calendar"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            
            calendar = self.client.calendar_economic()
            
            events = []
            for entry in calendar.get("economicCalendar", [])[:50]:
                events.append({
                    "event": entry.get("event", ""),
                    "country": entry.get("country", ""),
                    "time": entry.get("time", ""),
                    "impact": entry.get("impact", ""),
                    "prev": entry.get("prev"),
                    "estimate": entry.get("estimate"),
                    "actual": entry.get("actual"),
                    "unit": entry.get("unit", ""),
                })
            
            self._set_cache(cache_key, events)
            return events
            
        except Exception as e:
            log.warning(f"Finnhub economic calendar error: {e}")
            return []
    
    def get_support_resistance(self, ticker: str, resolution: str = "D") -> Dict:
        """
        Get technical support/resistance levels.
        Uses Finnhub's pattern recognition.
        """
        if not self.client:
            return {"available": False}
        
        cache_key = f"support_resistance_{ticker}_{resolution}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        
        try:
            self._rate_limit()
            
            levels = self.client.support_resistance(ticker, resolution)
            
            result = {
                "available": True,
                "ticker": ticker,
                "levels": levels.get("levels", []),
            }
            
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            log.warning(f"Finnhub support/resistance error for {ticker}: {e}")
            return {"available": False, "ticker": ticker}
    
    def is_earnings_soon(self, ticker: str, days: int = 3) -> Dict:
        """
        Check if ticker has earnings within N days.
        Returns warning for 0DTE plays near earnings.
        """
        earnings = self.get_earnings_calendar()
        
        today = datetime.now().date()
        cutoff = today + timedelta(days=days)
        
        for entry in earnings:
            if entry.get("symbol", "").upper() == ticker.upper():
                earnings_date = datetime.strptime(entry["date"], "%Y-%m-%d").date()
                if today <= earnings_date <= cutoff:
                    return {
                        "has_earnings": True,
                        "date": entry["date"],
                        "hour": entry.get("hour", "unknown"),
                        "days_away": (earnings_date - today).days,
                        "warning": f"EARNINGS IN {(earnings_date - today).days} DAYS - HIGH VOLATILITY"
                    }
        
        return {"has_earnings": False}
    
    def get_ruthless_context(self, ticker: str) -> Dict:
        """
        Get ALL available Finnhub data for maximum ruthlessness.
        Combines all sentiment + recommendations + price targets.
        """
        return {
            "sentiment": self.get_all_sentiment(ticker),
            "recommendations": self.get_recommendation_trends(ticker),
            "price_target": self.get_price_target(ticker),
            "earnings_warning": self.is_earnings_soon(ticker),
            "support_resistance": self.get_support_resistance(ticker),
        }


finnhub_collector = FinnhubCollector()
