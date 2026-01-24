"""
Historical Training System

Downloads historical market data, analyzes past events,
and trains the Learning Memory with observed outcomes.

Key capabilities:
- Fetches 6 weeks of historical price/volatility data
- Identifies past earnings, macro events, and their outcomes
- Calculates accuracy of expected moves vs actual moves
- Updates Learning Memory weights based on patterns
- Builds comprehensive upcoming events calendar
"""

import os
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from wsb_snake.utils.logger import log
from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
from wsb_snake.collectors.fred_collector import fred_collector
from wsb_snake.collectors.earnings_calendar import earnings_calendar


@dataclass
class HistoricalEvent:
    """Record of a past market event and its outcome."""
    date: str
    event_type: str  # "earnings", "cpi", "fomc", "jobs", "gdp"
    symbol: str
    expected_move: float
    actual_move: float
    direction: str  # "up", "down", "flat"
    surprise_direction: str  # "beat", "miss", "inline"
    iv_before: float
    iv_after: float
    details: Dict[str, Any]


@dataclass
class UpcomingEvent:
    """Upcoming market event to watch."""
    date: str
    event_type: str
    symbol: Optional[str]
    expected_move: float
    historical_avg_move: float
    win_rate: float  # Historical accuracy for this event type
    details: Dict[str, Any]


class HistoricalTrainer:
    """
    Downloads historical data and trains the engine on past events.
    """
    
    POLYGON_URL = "https://api.polygon.io"
    FINNHUB_URL = "https://finnhub.io/api/v1"
    
    UNIVERSE = ["SPY", "QQQ", "IWM", "TSLA", "NVDA", "AAPL", "MSFT", "META", "AMD", "GOOGL", "AMZN"]
    
    MACRO_EVENTS = {
        "cpi": {"typical_move": 1.5, "volatility_impact": "high"},
        "fomc": {"typical_move": 1.0, "volatility_impact": "high"},
        "jobs": {"typical_move": 0.8, "volatility_impact": "medium"},
        "gdp": {"typical_move": 0.5, "volatility_impact": "low"},
        "ppi": {"typical_move": 0.6, "volatility_impact": "medium"},
        "retail_sales": {"typical_move": 0.4, "volatility_impact": "low"},
    }
    
    def __init__(self):
        self.polygon_key = os.environ.get("POLYGON_API_KEY", "")
        self.finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
        self.session = requests.Session()
        
        self.historical_events: List[HistoricalEvent] = []
        self.upcoming_events: List[UpcomingEvent] = []
        self.price_cache: Dict[str, List[Dict]] = {}
        self.training_stats: Dict[str, Any] = {}
        
        self.last_api_call = 0
        self.min_interval = 0.25  # 4 calls/sec for Polygon
        
        log.info("Historical Trainer initialized")
    
    def _rate_limit(self):
        """Rate limit API calls."""
        elapsed = time.time() - self.last_api_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_api_call = time.time()
    
    def fetch_historical_bars(
        self, 
        symbol: str, 
        weeks: int = 6,
        timespan: str = "day"
    ) -> List[Dict]:
        """
        Fetch historical price bars from Polygon.
        
        Args:
            symbol: Ticker symbol
            weeks: Number of weeks of history
            timespan: Bar timespan (day, hour, minute)
            
        Returns:
            List of OHLCV bars
        """
        cache_key = f"{symbol}:{weeks}:{timespan}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        if not self.polygon_key:
            log.warning("POLYGON_API_KEY not set - using simulated history")
            return self._simulate_history(symbol, weeks)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks)
        
        self._rate_limit()
        
        try:
            url = f"{self.POLYGON_URL}/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apiKey": self.polygon_key, "limit": 500}
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                bars = []
                for r in data.get("results", []):
                    bars.append({
                        "date": datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d"),
                        "open": r["o"],
                        "high": r["h"],
                        "low": r["l"],
                        "close": r["c"],
                        "volume": r["v"],
                        "vwap": r.get("vw", r["c"]),
                    })
                
                self.price_cache[cache_key] = bars
                log.info(f"Fetched {len(bars)} bars for {symbol}")
                return bars
            else:
                log.warning(f"Polygon error for {symbol}: {response.status_code}")
                return self._simulate_history(symbol, weeks)
                
        except Exception as e:
            log.warning(f"Historical fetch error for {symbol}: {e}")
            return self._simulate_history(symbol, weeks)
    
    def _simulate_history(self, symbol: str, weeks: int) -> List[Dict]:
        """Generate simulated historical data for testing."""
        import random
        
        base_prices = {
            "SPY": 580, "QQQ": 500, "IWM": 220, "TSLA": 400,
            "NVDA": 140, "AAPL": 230, "MSFT": 420, "META": 600,
            "AMD": 120, "GOOGL": 190, "AMZN": 220
        }
        
        base = base_prices.get(symbol, 100)
        bars = []
        price = base
        
        for i in range(weeks * 5):  # ~5 trading days per week
            date = (datetime.now() - timedelta(days=(weeks * 7) - i)).strftime("%Y-%m-%d")
            change = random.uniform(-0.02, 0.02)
            price = price * (1 + change)
            
            bars.append({
                "date": date,
                "open": price * 0.998,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": random.randint(10000000, 50000000),
                "vwap": price,
            })
        
        return bars
    
    def get_past_earnings(self, symbol: str, weeks: int = 6) -> List[Dict]:
        """
        Get past earnings events for a symbol.
        
        Returns list of past earnings with actual vs expected.
        """
        if not self.finnhub_key:
            return []
        
        self._rate_limit()
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks)
            
            url = f"{self.FINNHUB_URL}/stock/earnings"
            params = {
                "symbol": symbol,
                "token": self.finnhub_key,
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                earnings = response.json()
                
                past_earnings = []
                for e in earnings:
                    period = e.get("period", "")
                    if period:
                        try:
                            event_date = datetime.strptime(period, "%Y-%m-%d")
                            if start_date <= event_date <= end_date:
                                actual = e.get("actual")
                                estimate = e.get("estimate")
                                
                                if actual is not None and estimate is not None and estimate != 0:
                                    surprise = ((actual - estimate) / abs(estimate)) * 100
                                    past_earnings.append({
                                        "date": period,
                                        "symbol": symbol,
                                        "actual_eps": actual,
                                        "estimate_eps": estimate,
                                        "surprise_pct": surprise,
                                        "beat": actual > estimate,
                                    })
                        except ValueError:
                            continue
                
                return past_earnings
            
            return []
            
        except Exception as e:
            log.warning(f"Past earnings error for {symbol}: {e}")
            return []
    
    def get_past_macro_events(self, weeks: int = 6) -> List[Dict]:
        """
        Get past macro events from FRED.
        
        Returns list of CPI, jobs, FOMC dates.
        """
        events = []
        
        cpi_dates = fred_collector.get_cpi_release_dates()
        for date_str in cpi_dates:
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d")
                cutoff = datetime.now() - timedelta(weeks=weeks)
                if event_date >= cutoff and event_date <= datetime.now():
                    events.append({
                        "date": date_str,
                        "type": "cpi",
                        "name": "CPI Release",
                    })
            except ValueError:
                continue
        
        fomc_dates = [
            "2026-01-15", "2025-12-18", "2025-12-10", "2025-11-07",
            "2025-10-30", "2025-09-17", "2025-07-30", "2025-06-18",
        ]
        
        for date_str in fomc_dates:
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d")
                cutoff = datetime.now() - timedelta(weeks=weeks)
                if event_date >= cutoff and event_date <= datetime.now():
                    events.append({
                        "date": date_str,
                        "type": "fomc",
                        "name": "FOMC Meeting",
                    })
            except ValueError:
                continue
        
        jobs_dates = ["2026-01-10", "2025-12-06", "2025-11-01", "2025-10-04"]
        
        for date_str in jobs_dates:
            try:
                event_date = datetime.strptime(date_str, "%Y-%m-%d")
                cutoff = datetime.now() - timedelta(weeks=weeks)
                if event_date >= cutoff and event_date <= datetime.now():
                    events.append({
                        "date": date_str,
                        "type": "jobs",
                        "name": "Jobs Report (NFP)",
                    })
            except ValueError:
                continue
        
        return sorted(events, key=lambda x: x["date"], reverse=True)
    
    def calculate_event_outcome(
        self, 
        symbol: str, 
        event_date: str, 
        bars: List[Dict]
    ) -> Optional[Dict]:
        """
        Calculate the actual move on an event date.
        
        Returns move statistics for the event.
        """
        bars_by_date = {b["date"]: b for b in bars}
        
        try:
            event_dt = datetime.strptime(event_date, "%Y-%m-%d")
        except ValueError:
            return None
        
        event_bar = bars_by_date.get(event_date)
        
        prev_date = (event_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        for i in range(1, 5):
            if prev_date in bars_by_date:
                break
            prev_date = (event_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        
        prev_bar = bars_by_date.get(prev_date)
        
        if not prev_bar:
            return None
        
        if event_bar:
            open_price = event_bar["open"]
            close_price = event_bar["close"]
            high = event_bar["high"]
            low = event_bar["low"]
        else:
            next_date = (event_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            for i in range(1, 5):
                if next_date in bars_by_date:
                    break
                next_date = (event_dt + timedelta(days=i)).strftime("%Y-%m-%d")
            
            next_bar = bars_by_date.get(next_date)
            if not next_bar:
                return None
            
            open_price = prev_bar["close"]
            close_price = next_bar["close"]
            high = next_bar["high"]
            low = next_bar["low"]
        
        gap_move = ((open_price - prev_bar["close"]) / prev_bar["close"]) * 100
        day_move = ((close_price - prev_bar["close"]) / prev_bar["close"]) * 100
        range_move = ((high - low) / prev_bar["close"]) * 100
        
        return {
            "gap_move": gap_move,
            "day_move": day_move,
            "range_move": range_move,
            "direction": "up" if day_move > 0.1 else ("down" if day_move < -0.1 else "flat"),
            "prev_close": prev_bar["close"],
            "event_close": close_price,
        }
    
    def analyze_historical_events(self, weeks: int = 6) -> Dict[str, Any]:
        """
        Full historical analysis for all symbols and events.
        
        Returns training statistics.
        """
        log.info(f"Starting historical analysis for {weeks} weeks...")
        
        stats = {
            "symbols_analyzed": 0,
            "earnings_events": 0,
            "macro_events": 0,
            "total_events": 0,
            "earnings_accuracy": {},
            "macro_accuracy": {},
            "symbol_patterns": {},
        }
        
        for symbol in self.UNIVERSE:
            log.info(f"Analyzing {symbol}...")
            
            bars = self.fetch_historical_bars(symbol, weeks)
            if not bars:
                continue
            
            stats["symbols_analyzed"] += 1
            
            past_earnings = self.get_past_earnings(symbol, weeks)
            
            symbol_stats = {
                "earnings_events": [],
                "avg_earnings_move": 0,
                "beat_rate": 0,
                "post_beat_direction": {"up": 0, "down": 0, "flat": 0},
                "post_miss_direction": {"up": 0, "down": 0, "flat": 0},
            }
            
            for earning in past_earnings:
                outcome = self.calculate_event_outcome(symbol, earning["date"], bars)
                if outcome:
                    event = HistoricalEvent(
                        date=earning["date"],
                        event_type="earnings",
                        symbol=symbol,
                        expected_move=5.0,
                        actual_move=abs(outcome["day_move"]),
                        direction=outcome["direction"],
                        surprise_direction="beat" if earning["beat"] else "miss",
                        iv_before=0,
                        iv_after=0,
                        details={
                            "surprise_pct": earning["surprise_pct"],
                            "gap_move": outcome["gap_move"],
                            "range_move": outcome["range_move"],
                        }
                    )
                    self.historical_events.append(event)
                    symbol_stats["earnings_events"].append(event)
                    stats["earnings_events"] += 1
                    
                    if earning["beat"]:
                        symbol_stats["post_beat_direction"][outcome["direction"]] += 1
                    else:
                        symbol_stats["post_miss_direction"][outcome["direction"]] += 1
            
            if symbol_stats["earnings_events"]:
                moves = [e.actual_move for e in symbol_stats["earnings_events"]]
                symbol_stats["avg_earnings_move"] = sum(moves) / len(moves)
                
                beats = sum(1 for e in symbol_stats["earnings_events"] if e.surprise_direction == "beat")
                symbol_stats["beat_rate"] = beats / len(symbol_stats["earnings_events"])
            
            stats["symbol_patterns"][symbol] = symbol_stats
        
        macro_events = self.get_past_macro_events(weeks)
        
        for event in macro_events:
            spy_bars = self.price_cache.get(f"SPY:{weeks}:day", [])
            if spy_bars:
                outcome = self.calculate_event_outcome("SPY", event["date"], spy_bars)
                if outcome:
                    expected = self.MACRO_EVENTS.get(event["type"], {}).get("typical_move", 1.0)
                    
                    hist_event = HistoricalEvent(
                        date=event["date"],
                        event_type=event["type"],
                        symbol="SPY",
                        expected_move=expected,
                        actual_move=abs(outcome["day_move"]),
                        direction=outcome["direction"],
                        surprise_direction="inline",
                        iv_before=0,
                        iv_after=0,
                        details={
                            "event_name": event["name"],
                            "range_move": outcome["range_move"],
                        }
                    )
                    self.historical_events.append(hist_event)
                    stats["macro_events"] += 1
        
        stats["total_events"] = stats["earnings_events"] + stats["macro_events"]
        
        self.training_stats = stats
        log.info(f"Historical analysis complete: {stats['total_events']} events analyzed")
        
        return stats
    
    def get_upcoming_events(self, weeks: int = 4) -> List[UpcomingEvent]:
        """
        Build calendar of upcoming events for next N weeks.
        
        Returns list of events with historical context.
        """
        self.upcoming_events = []
        
        end_date = (datetime.now() + timedelta(weeks=weeks)).strftime("%Y-%m-%d")
        earnings = earnings_calendar.get_earnings_calendar(to_date=end_date)
        
        for e in earnings:
            if e.get("symbol") in self.UNIVERSE or e.get("symbol", "").upper() in self.UNIVERSE:
                symbol = e.get("symbol", "").upper()
                
                symbol_stats = self.training_stats.get("symbol_patterns", {}).get(symbol, {})
                avg_move = symbol_stats.get("avg_earnings_move", 5.0)
                beat_rate = symbol_stats.get("beat_rate", 0.5)
                
                event = UpcomingEvent(
                    date=e.get("date", ""),
                    event_type="earnings",
                    symbol=symbol,
                    expected_move=e.get("expected_move", avg_move),
                    historical_avg_move=avg_move,
                    win_rate=beat_rate,
                    details={
                        "estimate": e.get("estimate"),
                        "year_ago": e.get("year_ago_eps"),
                        "timing": e.get("timing", "unknown"),
                    }
                )
                self.upcoming_events.append(event)
        
        macro_calendar = fred_collector.get_economic_calendar()
        
        for event in macro_calendar:
            event_type = event.get("event_type", "other")
            if event_type in self.MACRO_EVENTS:
                macro_info = self.MACRO_EVENTS[event_type]
                
                up_event = UpcomingEvent(
                    date=event.get("date", ""),
                    event_type=event_type,
                    symbol=None,
                    expected_move=macro_info["typical_move"],
                    historical_avg_move=macro_info["typical_move"],
                    win_rate=0.6,
                    details={
                        "name": event.get("name", event_type.upper()),
                        "impact": macro_info["volatility_impact"],
                    }
                )
                self.upcoming_events.append(up_event)
        
        self.upcoming_events.sort(key=lambda x: x.date)
        
        log.info(f"Built upcoming events calendar: {len(self.upcoming_events)} events")
        return self.upcoming_events
    
    def train_learning_memory(self) -> Dict[str, Any]:
        """
        Use historical analysis to update Learning Memory weights.
        
        Returns training report.
        """
        from wsb_snake.engines.learning_memory import learning_memory
        
        report = {
            "events_processed": 0,
            "weight_updates": [],
            "accuracy_by_type": {},
        }
        
        earnings_events = [e for e in self.historical_events if e.event_type == "earnings"]
        
        if earnings_events:
            beats_up = sum(1 for e in earnings_events 
                         if e.surprise_direction == "beat" and e.direction == "up")
            beats_total = sum(1 for e in earnings_events if e.surprise_direction == "beat")
            
            if beats_total > 0:
                beat_accuracy = beats_up / beats_total
                report["accuracy_by_type"]["earnings_beat_calls"] = beat_accuracy
                
                if beat_accuracy > 0.6:
                    learning_memory.update_weight("earnings_beat", min(1.5, beat_accuracy * 1.5))
                    report["weight_updates"].append({
                        "feature": "earnings_beat",
                        "new_weight": beat_accuracy * 1.5,
                        "reason": f"Beat->Up accuracy: {beat_accuracy:.1%}"
                    })
            
            misses_down = sum(1 for e in earnings_events 
                             if e.surprise_direction == "miss" and e.direction == "down")
            misses_total = sum(1 for e in earnings_events if e.surprise_direction == "miss")
            
            if misses_total > 0:
                miss_accuracy = misses_down / misses_total
                report["accuracy_by_type"]["earnings_miss_puts"] = miss_accuracy
                
                if miss_accuracy > 0.6:
                    learning_memory.update_weight("earnings_miss", min(1.5, miss_accuracy * 1.5))
                    report["weight_updates"].append({
                        "feature": "earnings_miss",
                        "new_weight": miss_accuracy * 1.5,
                        "reason": f"Miss->Down accuracy: {miss_accuracy:.1%}"
                    })
        
        macro_events = [e for e in self.historical_events if e.event_type in self.MACRO_EVENTS]
        
        for event_type in self.MACRO_EVENTS:
            type_events = [e for e in macro_events if e.event_type == event_type]
            if type_events:
                expected = self.MACRO_EVENTS[event_type]["typical_move"]
                avg_actual = sum(e.actual_move for e in type_events) / len(type_events)
                
                move_ratio = avg_actual / expected if expected > 0 else 1.0
                report["accuracy_by_type"][f"{event_type}_move_ratio"] = move_ratio
                
                if move_ratio > 1.2:
                    weight = min(2.0, move_ratio)
                    learning_memory.update_weight(f"macro_{event_type}", weight)
                    report["weight_updates"].append({
                        "feature": f"macro_{event_type}",
                        "new_weight": weight,
                        "reason": f"Actual moves {move_ratio:.1f}x expected"
                    })
        
        report["events_processed"] = len(self.historical_events)
        
        log.info(f"Training complete: {report['events_processed']} events, {len(report['weight_updates'])} weight updates")
        
        return report
    
    def run_full_training(self, weeks: int = 6) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        1. Fetch historical data
        2. Analyze past events
        3. Update Learning Memory
        4. Build upcoming events calendar
        """
        log.info("="*50)
        log.info("HISTORICAL TRAINING STARTING")
        log.info("="*50)
        
        analysis_stats = self.analyze_historical_events(weeks)
        
        training_report = self.train_learning_memory()
        
        upcoming = self.get_upcoming_events(weeks=4)
        
        summary = {
            "training_date": datetime.now().isoformat(),
            "weeks_analyzed": weeks,
            "symbols_analyzed": analysis_stats["symbols_analyzed"],
            "total_events": analysis_stats["total_events"],
            "earnings_events": analysis_stats["earnings_events"],
            "macro_events": analysis_stats["macro_events"],
            "weight_updates": len(training_report["weight_updates"]),
            "upcoming_events": len(upcoming),
            "accuracy_metrics": training_report["accuracy_by_type"],
        }
        
        log.info("="*50)
        log.info("HISTORICAL TRAINING COMPLETE")
        log.info(f"  Analyzed: {summary['total_events']} events")
        log.info(f"  Updated: {summary['weight_updates']} weights")
        log.info(f"  Upcoming: {summary['upcoming_events']} events")
        log.info("="*50)
        
        return summary


historical_trainer = HistoricalTrainer()
