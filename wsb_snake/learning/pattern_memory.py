"""
Pattern Memory - Stores and recognizes successful 0DTE price patterns.

Learns from historical winning trades to recognize similar setups in real-time.
Stores:
- Price action patterns before profitable moves
- Volume patterns
- Technical indicator configurations
- Time-of-day patterns
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import sqlite3

from wsb_snake.db.database import get_connection
from wsb_snake.utils.logger import log


@dataclass
class PricePattern:
    """A stored price action pattern."""
    pattern_id: str
    symbol: str
    pattern_type: str  # "breakout", "reversal", "squeeze", "momentum"
    direction: str  # "bullish", "bearish"
    
    # Price action fingerprint
    bars_before: int  # How many bars to look back
    price_changes: List[float]  # Percentage changes per bar
    volume_ratios: List[float]  # Volume vs average
    
    # Technical indicators at trigger
    rsi_range: Tuple[float, float]
    vwap_position: str  # "above", "below", "crossing"
    
    # Outcome stats
    win_count: int = 0
    loss_count: int = 0
    avg_gain: float = 0.0
    avg_loss: float = 0.0
    
    # Time context
    best_hour: int = 10  # Hour of day (ET) with best results
    created_at: str = ""
    last_matched: str = ""


@dataclass 
class PatternMatch:
    """Result of pattern matching."""
    pattern_id: str
    pattern_type: str
    similarity_score: float  # 0-100
    historical_win_rate: float
    avg_move: float
    direction: str
    confidence: float


class PatternMemory:
    """
    Stores and matches price action patterns from successful trades.
    """
    
    MIN_SIMILARITY = 0.70  # Minimum similarity to consider a match
    MIN_SAMPLES = 3  # Minimum pattern occurrences to trust
    
    def __init__(self):
        self._init_tables()
        self.patterns: Dict[str, PricePattern] = {}
        self._load_patterns()
        log.info("Pattern Memory initialized")
    
    def _init_tables(self):
        """Create pattern storage tables."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_patterns (
                pattern_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                bars_before INTEGER DEFAULT 5,
                price_changes TEXT,
                volume_ratios TEXT,
                rsi_min REAL DEFAULT 0,
                rsi_max REAL DEFAULT 100,
                vwap_position TEXT DEFAULT 'neutral',
                win_count INTEGER DEFAULT 0,
                loss_count INTEGER DEFAULT 0,
                avg_gain REAL DEFAULT 0,
                avg_loss REAL DEFAULT 0,
                best_hour INTEGER DEFAULT 10,
                created_at TEXT,
                last_matched TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT,
                matched_at TEXT,
                symbol TEXT,
                similarity REAL,
                outcome TEXT,
                pnl_pct REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _load_patterns(self):
        """Load existing patterns from database."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM price_patterns")
        rows = cursor.fetchall()
        
        for row in rows:
            try:
                pattern = PricePattern(
                    pattern_id=row["pattern_id"],
                    symbol=row["symbol"],
                    pattern_type=row["pattern_type"],
                    direction=row["direction"],
                    bars_before=row["bars_before"],
                    price_changes=json.loads(row["price_changes"] or "[]"),
                    volume_ratios=json.loads(row["volume_ratios"] or "[]"),
                    rsi_range=(row["rsi_min"], row["rsi_max"]),
                    vwap_position=row["vwap_position"],
                    win_count=row["win_count"],
                    loss_count=row["loss_count"],
                    avg_gain=row["avg_gain"],
                    avg_loss=row["avg_loss"],
                    best_hour=row["best_hour"],
                    created_at=row["created_at"],
                    last_matched=row["last_matched"] or ""
                )
                self.patterns[pattern.pattern_id] = pattern
            except Exception as e:
                log.warning(f"Failed to load pattern: {e}")
        
        conn.close()
        log.info(f"Loaded {len(self.patterns)} price patterns")
    
    def store_pattern(
        self,
        symbol: str,
        bars: List[Dict],
        outcome: str,
        pnl_pct: float,
        rsi: float = 50,
        vwap_position: str = "neutral"
    ) -> str:
        """
        Store a new pattern from a trade outcome.
        
        Args:
            symbol: Ticker symbol
            bars: List of OHLCV bars before the trade
            outcome: "win" or "loss"
            pnl_pct: Percentage gain/loss
            rsi: RSI value at entry
            vwap_position: Position relative to VWAP
            
        Returns:
            Pattern ID
        """
        if len(bars) < 3:
            return ""
        
        # Calculate price changes
        price_changes = []
        for i in range(1, min(6, len(bars))):
            if bars[i-1].get("c", 0) > 0:
                change = (bars[i].get("c", 0) - bars[i-1].get("c", 0)) / bars[i-1].get("c", 1)
                price_changes.append(round(change * 100, 2))
        
        # Calculate volume ratios
        volumes = [b.get("v", 0) for b in bars[:6]]
        avg_vol = sum(volumes) / len(volumes) if volumes else 1
        volume_ratios = [round(v / avg_vol, 2) if avg_vol > 0 else 1.0 for v in volumes]
        
        # Classify pattern type
        pattern_type = self._classify_pattern(price_changes, volume_ratios)
        direction = "bullish" if sum(price_changes[-2:]) > 0 else "bearish"
        
        # Create fingerprint for pattern ID
        fingerprint = f"{symbol}_{pattern_type}_{direction}_{int(rsi/10)}"
        
        # Check if pattern exists
        if fingerprint in self.patterns:
            # Update existing pattern
            pattern = self.patterns[fingerprint]
            if outcome == "win":
                pattern.win_count += 1
                pattern.avg_gain = (pattern.avg_gain * (pattern.win_count - 1) + pnl_pct) / pattern.win_count
            else:
                pattern.loss_count += 1
                pattern.avg_loss = (pattern.avg_loss * (pattern.loss_count - 1) + abs(pnl_pct)) / pattern.loss_count
            
            pattern.last_matched = datetime.utcnow().isoformat()
            self._save_pattern(pattern)
            return fingerprint
        
        # Create new pattern
        now = datetime.utcnow()
        pattern = PricePattern(
            pattern_id=fingerprint,
            symbol=symbol,
            pattern_type=pattern_type,
            direction=direction,
            bars_before=len(price_changes),
            price_changes=price_changes,
            volume_ratios=volume_ratios,
            rsi_range=(max(0, rsi - 10), min(100, rsi + 10)),
            vwap_position=vwap_position,
            win_count=1 if outcome == "win" else 0,
            loss_count=0 if outcome == "win" else 1,
            avg_gain=pnl_pct if outcome == "win" else 0,
            avg_loss=abs(pnl_pct) if outcome == "loss" else 0,
            best_hour=now.hour,
            created_at=now.isoformat(),
            last_matched=now.isoformat()
        )
        
        self.patterns[fingerprint] = pattern
        self._save_pattern(pattern)
        
        log.info(f"Stored new pattern: {fingerprint}")
        return fingerprint
    
    def _classify_pattern(self, price_changes: List[float], volume_ratios: List[float]) -> str:
        """Classify the pattern type based on price action."""
        if not price_changes:
            return "unknown"
        
        # Check for breakout (sudden large move with volume)
        if len(price_changes) >= 2:
            last_move = abs(price_changes[-1])
            avg_move = sum(abs(c) for c in price_changes[:-1]) / max(1, len(price_changes) - 1)
            
            if last_move > avg_move * 2 and volume_ratios and volume_ratios[-1] > 1.5:
                return "breakout"
        
        # Check for squeeze (compression then expansion)
        if len(price_changes) >= 4:
            early_range = max(abs(c) for c in price_changes[:2])
            late_range = max(abs(c) for c in price_changes[-2:])
            if late_range > early_range * 1.5:
                return "squeeze"
        
        # Check for reversal (direction change)
        if len(price_changes) >= 3:
            early_dir = sum(price_changes[:2])
            late_dir = sum(price_changes[-2:])
            if (early_dir > 0 and late_dir < 0) or (early_dir < 0 and late_dir > 0):
                return "reversal"
        
        # Default to momentum
        return "momentum"
    
    def _save_pattern(self, pattern: PricePattern):
        """Save pattern to database."""
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO price_patterns
            (pattern_id, symbol, pattern_type, direction, bars_before,
             price_changes, volume_ratios, rsi_min, rsi_max, vwap_position,
             win_count, loss_count, avg_gain, avg_loss, best_hour,
             created_at, last_matched)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pattern.pattern_id,
            pattern.symbol,
            pattern.pattern_type,
            pattern.direction,
            pattern.bars_before,
            json.dumps(pattern.price_changes),
            json.dumps(pattern.volume_ratios),
            pattern.rsi_range[0],
            pattern.rsi_range[1],
            pattern.vwap_position,
            pattern.win_count,
            pattern.loss_count,
            pattern.avg_gain,
            pattern.avg_loss,
            pattern.best_hour,
            pattern.created_at,
            pattern.last_matched
        ))
        
        conn.commit()
        conn.close()
    
    def find_matching_patterns(
        self,
        symbol: str,
        bars: List[Dict],
        rsi: float = 50,
        vwap_position: str = "neutral"
    ) -> List[PatternMatch]:
        """
        Find patterns that match current market conditions.
        
        Args:
            symbol: Ticker symbol
            bars: Recent OHLCV bars
            rsi: Current RSI
            vwap_position: Current VWAP position
            
        Returns:
            List of matching patterns sorted by confidence
        """
        if len(bars) < 3:
            return []
        
        # Calculate current price changes
        current_changes = []
        for i in range(1, min(6, len(bars))):
            if bars[i-1].get("c", 0) > 0:
                change = (bars[i].get("c", 0) - bars[i-1].get("c", 0)) / bars[i-1].get("c", 1)
                current_changes.append(change * 100)
        
        # Calculate current volume ratios
        volumes = [b.get("v", 0) for b in bars[:6]]
        avg_vol = sum(volumes) / len(volumes) if volumes else 1
        current_vol_ratios = [v / avg_vol if avg_vol > 0 else 1.0 for v in volumes]
        
        matches = []
        
        for pattern in self.patterns.values():
            # Must match symbol
            if pattern.symbol != symbol:
                continue
            
            # Check RSI range
            if not (pattern.rsi_range[0] <= rsi <= pattern.rsi_range[1]):
                continue
            
            # Calculate similarity
            similarity = self._calculate_similarity(
                current_changes, 
                pattern.price_changes,
                current_vol_ratios,
                pattern.volume_ratios
            )
            
            if similarity < self.MIN_SIMILARITY:
                continue
            
            # Calculate win rate and confidence
            total = pattern.win_count + pattern.loss_count
            if total < self.MIN_SAMPLES:
                continue
            
            win_rate = pattern.win_count / total
            avg_move = pattern.avg_gain if win_rate > 0.5 else -pattern.avg_loss
            
            # Confidence based on sample size and win rate
            confidence = similarity * min(1.0, total / 10) * (0.5 + win_rate / 2)
            
            matches.append(PatternMatch(
                pattern_id=pattern.pattern_id,
                pattern_type=pattern.pattern_type,
                similarity_score=similarity * 100,
                historical_win_rate=win_rate,
                avg_move=avg_move,
                direction=pattern.direction,
                confidence=confidence * 100
            ))
        
        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        return matches[:5]  # Return top 5 matches
    
    def _calculate_similarity(
        self,
        current_prices: List[float],
        pattern_prices: List[float],
        current_vols: List[float],
        pattern_vols: List[float]
    ) -> float:
        """Calculate similarity between current action and stored pattern."""
        if not current_prices or not pattern_prices:
            return 0.0
        
        # Price similarity (direction and magnitude)
        min_len = min(len(current_prices), len(pattern_prices))
        price_diffs = []
        for i in range(min_len):
            # Same direction bonus
            same_dir = (current_prices[i] > 0) == (pattern_prices[i] > 0)
            if same_dir:
                # Compare magnitudes
                max_mag = max(abs(current_prices[i]), abs(pattern_prices[i]), 0.1)
                diff = 1 - abs(current_prices[i] - pattern_prices[i]) / max_mag
                price_diffs.append(max(0, diff))
            else:
                price_diffs.append(0)
        
        price_sim = sum(price_diffs) / len(price_diffs) if price_diffs else 0
        
        # Volume similarity
        vol_sim = 0.5  # Default
        if current_vols and pattern_vols:
            min_vol_len = min(len(current_vols), len(pattern_vols))
            vol_diffs = []
            for i in range(min_vol_len):
                max_vol = max(current_vols[i], pattern_vols[i], 0.1)
                diff = 1 - abs(current_vols[i] - pattern_vols[i]) / max_vol
                vol_diffs.append(max(0, diff))
            vol_sim = sum(vol_diffs) / len(vol_diffs) if vol_diffs else 0.5
        
        # Weighted average (price more important)
        return price_sim * 0.7 + vol_sim * 0.3
    
    def get_pattern_stats(self) -> Dict:
        """Get summary statistics of stored patterns."""
        total = len(self.patterns)
        if total == 0:
            return {"total_patterns": 0}
        
        wins = sum(p.win_count for p in self.patterns.values())
        losses = sum(p.loss_count for p in self.patterns.values())
        
        by_type = {}
        for p in self.patterns.values():
            if p.pattern_type not in by_type:
                by_type[p.pattern_type] = {"count": 0, "wins": 0}
            by_type[p.pattern_type]["count"] += 1
            by_type[p.pattern_type]["wins"] += p.win_count
        
        return {
            "total_patterns": total,
            "total_wins": wins,
            "total_losses": losses,
            "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0,
            "by_type": by_type
        }


# Global instance
pattern_memory = PatternMemory()
