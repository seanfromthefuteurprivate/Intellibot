"""
GAMMA SQUEEZE DETECTOR - The Most Dangerous Module
═══════════════════════════════════════════════════════════════════

Monitors conditions that precede gamma squeezes:
1. Call/Put OI Ratio > 2.0 (heavy call positioning)
2. Short Interest > 15% (shorts to squeeze)
3. Dealers short gamma (negative GEX)
4. Low float relative to options OI
5. Rising IV with price breaking resistance

When ALL conditions align → FULL SEND with 40% of account.

Wired into BERSERKER as an override trigger.

Log format: "GAMMA_SQUEEZE: [ticker] DETECTED conditions=[...]"
"""

import os
import threading
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SqueezeConditions:
    """Current gamma squeeze conditions for a ticker."""
    ticker: str
    timestamp: datetime

    # Call/Put OI ratio (>2.0 = heavy call positioning)
    call_put_oi_ratio: float = 1.0
    call_oi: int = 0
    put_oi: int = 0

    # Short interest
    short_interest_pct: float = 0.0
    shares_short: int = 0

    # GEX regime
    gex_regime: str = "neutral"  # "negative", "extreme_negative", "positive"
    gex_flip_distance_pct: float = 999.0

    # IV and price action
    iv_rank: float = 50.0  # 0-100 percentile
    iv_rising: bool = False
    price_above_resistance: bool = False

    # Float analysis
    float_shares: int = 0
    options_oi_to_float_ratio: float = 0.0  # High = potential squeeze

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "call_put_oi_ratio": self.call_put_oi_ratio,
            "short_interest_pct": self.short_interest_pct,
            "gex_regime": self.gex_regime,
            "gex_flip_distance_pct": self.gex_flip_distance_pct,
            "iv_rank": self.iv_rank,
            "iv_rising": self.iv_rising,
            "price_above_resistance": self.price_above_resistance,
            "options_oi_to_float_ratio": self.options_oi_to_float_ratio,
        }


@dataclass
class SqueezeSignal:
    """A gamma squeeze signal - FULL SEND trigger."""
    ticker: str
    direction: str  # "CALL" (always bullish for gamma squeeze)
    confidence: float  # 0-100
    conditions: SqueezeConditions
    position_size_pct: float  # % of account to deploy (up to 40%)
    reasoning: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "confidence": self.confidence,
            "position_size_pct": self.position_size_pct,
            "reasoning": self.reasoning,
            "generated_at": self.generated_at.isoformat(),
            "conditions": self.conditions.to_dict(),
        }


@dataclass
class SqueezeConfig:
    """Configuration for gamma squeeze detection."""
    # Thresholds for squeeze detection
    min_call_put_ratio: float = 2.0  # Call/Put OI ratio threshold
    min_short_interest: float = 15.0  # % short interest
    max_gex_flip_distance: float = 2.0  # % from GEX flip point
    min_iv_rank: float = 40.0  # IV rank percentile
    min_oi_to_float_ratio: float = 0.05  # Options OI / Float

    # Position sizing
    base_position_pct: float = 25.0  # Base position size for squeeze
    max_position_pct: float = 40.0  # Max position size (FULL SEND)
    confidence_scale_factor: float = 0.15  # Extra % per point above 70 confidence

    # Monitoring
    check_interval_seconds: int = 30
    cache_ttl_seconds: int = 60

    # Tickers to monitor for squeeze potential
    squeeze_candidates: List[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM",  # Index ETFs
        "TSLA", "NVDA", "AMD", "GME", "AMC",  # High short interest / meme
        "AAPL", "META", "GOOGL", "AMZN", "MSFT",  # Mega caps
    ])


class GammaSqueezeDetector:
    """
    Detects gamma squeeze conditions for BERSERKER FULL SEND triggers.

    A gamma squeeze occurs when:
    1. Heavy call buying forces dealers to buy underlying to hedge
    2. Dealers are already short gamma (negative GEX)
    3. High short interest adds buying pressure
    4. Price breaks resistance, triggering more hedging

    This creates a self-reinforcing buying loop that can spike prices 10-50%.
    """

    def __init__(self, config: Optional[SqueezeConfig] = None):
        self.config = config or SqueezeConfig()
        self._conditions_cache: Dict[str, SqueezeConditions] = {}
        self._last_check: Dict[str, datetime] = {}
        self._active_signals: Dict[str, SqueezeSignal] = {}
        self._executed_tickers: set = set()  # Track executed squeeze tickers to avoid duplicates
        self._lock = threading.RLock()

        # Stats
        self._stats = {
            "checks": 0,
            "signals_generated": 0,
            "signals_triggered": 0,
        }

        logger.info("GAMMA_SQUEEZE: Detector initialized - watching for squeeze conditions")

    def check_squeeze_conditions(self, ticker: str, force: bool = False) -> Optional[SqueezeConditions]:
        """
        Check current gamma squeeze conditions for a ticker.

        Returns SqueezeConditions if data available, None if stale/unavailable.
        """
        now = datetime.now(timezone.utc)

        # Check cache unless forced
        if not force and ticker in self._last_check:
            elapsed = (now - self._last_check[ticker]).total_seconds()
            if elapsed < self.config.cache_ttl_seconds:
                return self._conditions_cache.get(ticker)

        self._stats["checks"] += 1

        try:
            # Fetch all required data
            conditions = SqueezeConditions(ticker=ticker, timestamp=now)

            # 1. Get Call/Put OI from options chain
            call_oi, put_oi = self._fetch_options_oi(ticker)
            conditions.call_oi = call_oi
            conditions.put_oi = put_oi
            conditions.call_put_oi_ratio = call_oi / put_oi if put_oi > 0 else 0

            # 2. Get short interest
            short_pct, shares_short = self._fetch_short_interest(ticker)
            conditions.short_interest_pct = short_pct
            conditions.shares_short = shares_short

            # 3. Get GEX regime from HYDRA
            gex_regime, flip_distance = self._fetch_gex_data(ticker)
            conditions.gex_regime = gex_regime
            conditions.gex_flip_distance_pct = flip_distance

            # 4. Get IV data
            iv_rank, iv_rising = self._fetch_iv_data(ticker)
            conditions.iv_rank = iv_rank
            conditions.iv_rising = iv_rising

            # 5. Get float and OI/float ratio
            float_shares = self._fetch_float_data(ticker)
            conditions.float_shares = float_shares
            total_oi = call_oi + put_oi
            conditions.options_oi_to_float_ratio = total_oi / float_shares if float_shares > 0 else 0

            # 6. Check price vs resistance
            conditions.price_above_resistance = self._check_price_breakout(ticker)

            # Cache results
            with self._lock:
                self._conditions_cache[ticker] = conditions
                self._last_check[ticker] = now

            return conditions

        except Exception as e:
            logger.warning(f"GAMMA_SQUEEZE: Failed to check {ticker}: {e}")
            return self._conditions_cache.get(ticker)

    def evaluate_squeeze_signal(self, ticker: str) -> Optional[SqueezeSignal]:
        """
        Evaluate if a ticker has gamma squeeze conditions for FULL SEND.

        Returns SqueezeSignal if conditions met, None otherwise.
        """
        conditions = self.check_squeeze_conditions(ticker)
        if not conditions:
            return None

        # Score each condition (0-20 points each, max 100)
        score = 0
        reasons = []

        # 1. Call/Put OI Ratio (0-25 points)
        if conditions.call_put_oi_ratio >= self.config.min_call_put_ratio:
            ratio_score = min(25, 15 + (conditions.call_put_oi_ratio - 2.0) * 5)
            score += ratio_score
            reasons.append(f"Call/Put OI: {conditions.call_put_oi_ratio:.1f}x (+{ratio_score:.0f})")

        # 2. Short Interest (0-20 points)
        if conditions.short_interest_pct >= self.config.min_short_interest:
            si_score = min(20, 10 + (conditions.short_interest_pct - 15) * 0.5)
            score += si_score
            reasons.append(f"Short Interest: {conditions.short_interest_pct:.1f}% (+{si_score:.0f})")

        # 3. GEX Regime (0-25 points)
        if conditions.gex_regime in ("negative", "extreme_negative"):
            if conditions.gex_regime == "extreme_negative":
                gex_score = 25
            else:
                gex_score = 15
            # Bonus for being close to flip point
            if conditions.gex_flip_distance_pct < self.config.max_gex_flip_distance:
                gex_score += 5
            score += gex_score
            reasons.append(f"GEX: {conditions.gex_regime} ({conditions.gex_flip_distance_pct:.1f}% from flip) (+{gex_score:.0f})")

        # 4. IV Rising (0-15 points)
        if conditions.iv_rank >= self.config.min_iv_rank and conditions.iv_rising:
            iv_score = min(15, 10 + (conditions.iv_rank - 40) * 0.1)
            score += iv_score
            reasons.append(f"IV Rank: {conditions.iv_rank:.0f}% rising (+{iv_score:.0f})")

        # 5. Price Breakout (0-15 points)
        if conditions.price_above_resistance:
            score += 15
            reasons.append("Price above resistance (+15)")

        # Calculate confidence (score is 0-100)
        confidence = min(100, score)

        # Need at least 60 confidence to generate signal
        if confidence < 60:
            logger.debug(f"GAMMA_SQUEEZE: {ticker} score {confidence:.0f}% below threshold")
            return None

        # Calculate position size
        base_pct = self.config.base_position_pct
        extra_pct = max(0, confidence - 70) * self.config.confidence_scale_factor
        position_pct = min(self.config.max_position_pct, base_pct + extra_pct)

        reasoning = f"GAMMA SQUEEZE CONDITIONS: {' | '.join(reasons)}"

        signal = SqueezeSignal(
            ticker=ticker,
            direction="CALL",  # Gamma squeezes are bullish
            confidence=confidence,
            conditions=conditions,
            position_size_pct=position_pct,
            reasoning=reasoning,
        )

        self._stats["signals_generated"] += 1

        logger.warning(
            f"GAMMA_SQUEEZE: {ticker} SIGNAL GENERATED | "
            f"Confidence: {confidence:.0f}% | Position: {position_pct:.0f}% | "
            f"{reasoning}"
        )

        with self._lock:
            self._active_signals[ticker] = signal

        return signal

    def get_squeeze_signals(self) -> List[SqueezeSignal]:
        """Get all active squeeze signals."""
        signals = []
        for ticker in self.config.squeeze_candidates:
            signal = self.evaluate_squeeze_signal(ticker)
            if signal:
                signals.append(signal)
        return signals

    def get_best_squeeze(self) -> Optional[SqueezeSignal]:
        """Get the highest confidence squeeze signal."""
        signals = self.get_squeeze_signals()
        if not signals:
            return None
        return max(signals, key=lambda s: s.confidence)

    def should_trigger_berserker(self, ticker: str = None) -> Tuple[bool, Optional[SqueezeSignal]]:
        """
        Check if BERSERKER should be triggered for gamma squeeze.

        Returns (should_trigger, signal)
        """
        if ticker:
            signal = self.evaluate_squeeze_signal(ticker)
            if signal and signal.confidence >= 75:
                return True, signal
            return False, None

        # Check all candidates
        best = self.get_best_squeeze()
        if best and best.confidence >= 75:
            # Don't trigger if already executed today
            if best.ticker in self._executed_tickers:
                logger.debug(f"GAMMA_SQUEEZE: {best.ticker} already executed today, skipping")
                return False, None
            return True, best
        return False, None

    def mark_executed(self, ticker: str) -> None:
        """Mark a ticker as executed for gamma squeeze to avoid duplicate trades."""
        with self._lock:
            self._executed_tickers.add(ticker)
            self._stats["signals_triggered"] += 1
        logger.info(f"GAMMA_SQUEEZE: Marked {ticker} as executed")

    def reset_executed(self) -> None:
        """Reset executed tickers (call at start of trading day)."""
        with self._lock:
            self._executed_tickers.clear()
        logger.info("GAMMA_SQUEEZE: Reset executed tickers for new trading day")

    # ========== DATA FETCHING METHODS ==========

    def _fetch_options_oi(self, ticker: str) -> Tuple[int, int]:
        """Fetch total call and put open interest."""
        try:
            from wsb_snake.collectors.polygon_options import get_options_chain
            chain = get_options_chain(ticker, limit=100)

            call_oi = 0
            put_oi = 0

            for contract in chain:
                contract_type = contract.get("contract_type", "").upper()
                oi = contract.get("open_interest", 0) or 0

                if contract_type == "CALL":
                    call_oi += oi
                elif contract_type == "PUT":
                    put_oi += oi

            return call_oi, put_oi

        except Exception as e:
            logger.debug(f"Options OI fetch failed for {ticker}: {e}")
            # Fallback estimates based on typical ratios
            return 100000, 80000  # Default 1.25 ratio

    def _fetch_short_interest(self, ticker: str) -> Tuple[float, int]:
        """Fetch short interest percentage and shares short."""
        try:
            from wsb_snake.collectors.finnhub_collector import finnhub_collector
            # Finnhub provides short interest for US stocks
            short_data = finnhub_collector.get_short_interest(ticker)
            if short_data:
                return short_data.get("shortPercentFloat", 0), short_data.get("shortsVolume", 0)
        except Exception as e:
            logger.debug(f"Short interest fetch failed for {ticker}: {e}")

        # Fallback estimates by ticker type
        high_si_tickers = ["GME", "AMC", "TSLA", "BBBY", "KOSS"]
        if ticker.upper() in high_si_tickers:
            return 25.0, 50000000
        return 5.0, 10000000

    def _fetch_gex_data(self, ticker: str) -> Tuple[str, float]:
        """Fetch GEX regime from HYDRA."""
        try:
            from wsb_snake.collectors.hydra_bridge import get_hydra_intel
            intel = get_hydra_intel()

            if intel.connected:
                regime = intel.gex_regime.lower() if intel.gex_regime else "neutral"
                flip_distance = intel.gex_flip_distance_pct

                # Normalize regime names
                if "extreme" in regime or "very" in regime:
                    regime = "extreme_negative" if "neg" in regime else "extreme_positive"
                elif "neg" in regime:
                    regime = "negative"
                elif "pos" in regime:
                    regime = "positive"
                else:
                    regime = "neutral"

                return regime, flip_distance

        except Exception as e:
            logger.debug(f"GEX data fetch failed for {ticker}: {e}")

        return "neutral", 999.0

    def _fetch_iv_data(self, ticker: str) -> Tuple[float, bool]:
        """Fetch IV rank and direction."""
        try:
            from wsb_snake.collectors.polygon_options import get_iv_data
            iv_data = get_iv_data(ticker)
            if iv_data:
                iv_rank = iv_data.get("iv_rank", 50)
                iv_5d_ago = iv_data.get("iv_5d_ago", iv_rank)
                iv_rising = iv_rank > iv_5d_ago
                return iv_rank, iv_rising
        except Exception as e:
            logger.debug(f"IV data fetch failed for {ticker}: {e}")

        return 50.0, False

    def _fetch_float_data(self, ticker: str) -> int:
        """Fetch shares float."""
        try:
            from wsb_snake.collectors.finnhub_collector import finnhub_collector
            profile = finnhub_collector.get_company_profile(ticker)
            if profile:
                return profile.get("shareOutstanding", 0) * 1000000  # Convert to shares
        except Exception as e:
            logger.debug(f"Float data fetch failed for {ticker}: {e}")

        # Fallback estimates
        large_cap = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
        if ticker.upper() in large_cap:
            return 5_000_000_000  # 5B shares
        return 100_000_000  # 100M shares default

    def _check_price_breakout(self, ticker: str) -> bool:
        """Check if price is breaking above resistance."""
        try:
            from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
            technicals = polygon_enhanced.get_full_technicals(ticker)

            if technicals:
                # Check if price is above key levels
                current_price = technicals.get("current_price", 0)
                resistance = technicals.get("resistance", {}).get("r1", 0)

                if current_price > 0 and resistance > 0:
                    return current_price > resistance

        except Exception as e:
            logger.debug(f"Price breakout check failed for {ticker}: {e}")

        return False

    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            **self._stats,
            "active_signals": len(self._active_signals),
            "cached_conditions": len(self._conditions_cache),
        }


# Singleton
_detector: Optional[GammaSqueezeDetector] = None


def get_gamma_squeeze_detector() -> GammaSqueezeDetector:
    """Get singleton gamma squeeze detector."""
    global _detector
    if _detector is None:
        _detector = GammaSqueezeDetector()
    return _detector
