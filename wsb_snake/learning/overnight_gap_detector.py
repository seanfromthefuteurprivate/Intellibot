"""
OVERNIGHT GAP DETECTOR - Pre-Market Edge Detection
═══════════════════════════════════════════════════════════════════

Checks ES/NQ futures at 9:00 AM pre-market:
- Gap > 0.5% + negative GEX = queue aggressive trade for first 5 minutes
- Gap > 1% = BERSERKER auto-trigger at open

Log format: "OVERNIGHT_GAP: SPY gap={pct}% direction={dir} gex={regime} -> {action}"
"""

import os
import threading
import time
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import pytz

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GapAnalysis:
    """Analysis of overnight gap."""
    ticker: str
    timestamp: datetime

    # Gap data
    previous_close: float = 0.0
    premarket_price: float = 0.0
    gap_pct: float = 0.0
    gap_direction: str = "FLAT"  # "UP", "DOWN", "FLAT"

    # Futures data
    es_gap_pct: float = 0.0
    nq_gap_pct: float = 0.0

    # GEX context
    gex_regime: str = "neutral"
    gex_flip_distance_pct: float = 999.0

    # VIX context
    vix_level: float = 20.0
    vix_change_pct: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "gap_pct": self.gap_pct,
            "gap_direction": self.gap_direction,
            "es_gap_pct": self.es_gap_pct,
            "nq_gap_pct": self.nq_gap_pct,
            "gex_regime": self.gex_regime,
            "vix_level": self.vix_level,
        }


@dataclass
class GapSignal:
    """Signal from overnight gap analysis."""
    ticker: str
    direction: str  # "CALL" or "PUT"
    confidence: float
    trigger_type: str  # "OPEN_DRIVE" or "BERSERKER"
    gap_analysis: GapAnalysis
    reasoning: str
    execute_at: datetime  # When to execute (market open + buffer)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "direction": self.direction,
            "confidence": self.confidence,
            "trigger_type": self.trigger_type,
            "reasoning": self.reasoning,
            "execute_at": self.execute_at.isoformat(),
            "gap_analysis": self.gap_analysis.to_dict(),
        }


@dataclass
class GapConfig:
    """Configuration for gap detection."""
    # Gap thresholds
    min_gap_for_signal: float = 0.5  # 0.5% gap minimum
    berserker_gap_threshold: float = 1.0  # 1%+ gap = BERSERKER
    extreme_gap_threshold: float = 2.0  # 2%+ gap = high confidence

    # Timing
    check_time_et: str = "09:00"  # When to check pre-market (ET)
    execute_delay_seconds: int = 30  # Wait 30s after open for spread normalization
    open_drive_window_minutes: int = 5  # First 5 minutes for open drive trade

    # Position sizing
    open_drive_position_pct: float = 20.0  # 20% for regular gap
    berserker_position_pct: float = 35.0  # 35% for BERSERKER gap

    # Tickers to monitor
    monitor_tickers: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])


class OvernightGapDetector:
    """
    Detects overnight gaps for opening drive trades.

    Checks ES/NQ futures pre-market and correlates with GEX regime
    to identify high-probability opening trades.
    """

    def __init__(self, config: Optional[GapConfig] = None):
        self.config = config or GapConfig()
        self._et = pytz.timezone("America/New_York")
        self._gap_cache: Dict[str, GapAnalysis] = {}
        self._pending_signals: Dict[str, GapSignal] = {}
        self._executed_today: List[str] = []
        self._last_check_date: Optional[str] = None
        self._lock = threading.RLock()

        # Stats
        self._stats = {
            "checks": 0,
            "gaps_detected": 0,
            "signals_generated": 0,
            "berserker_triggers": 0,
        }

        logger.info("OVERNIGHT_GAP: Detector initialized - watching for opening gaps")

    def _get_et_now(self) -> datetime:
        """Get current time in Eastern."""
        return datetime.now(self._et)

    def _should_check(self) -> bool:
        """Check if we should run the gap analysis."""
        now = self._get_et_now()
        today = now.strftime("%Y-%m-%d")

        # Only check once per day
        if self._last_check_date == today:
            return False

        # Check between 9:00-9:30 AM ET
        if now.hour == 9 and 0 <= now.minute <= 30:
            return True

        return False

    def _reset_daily(self):
        """Reset daily state."""
        now = self._get_et_now()
        today = now.strftime("%Y-%m-%d")

        if self._last_check_date != today:
            with self._lock:
                self._executed_today = []
                self._pending_signals = {}
            logger.info("OVERNIGHT_GAP: Daily state reset")

    def analyze_gap(self, ticker: str, force: bool = False) -> Optional[GapAnalysis]:
        """
        Analyze overnight gap for a ticker.

        Returns GapAnalysis if gap is significant, None otherwise.
        """
        self._reset_daily()

        if not force and not self._should_check():
            return self._gap_cache.get(ticker)

        self._stats["checks"] += 1

        try:
            analysis = GapAnalysis(
                ticker=ticker,
                timestamp=datetime.now(timezone.utc)
            )

            # Get previous close
            prev_close = self._get_previous_close(ticker)
            analysis.previous_close = prev_close

            # Get pre-market price
            premarket = self._get_premarket_price(ticker)
            analysis.premarket_price = premarket

            # Calculate gap
            if prev_close > 0 and premarket > 0:
                analysis.gap_pct = ((premarket - prev_close) / prev_close) * 100

                if analysis.gap_pct > 0.1:
                    analysis.gap_direction = "UP"
                elif analysis.gap_pct < -0.1:
                    analysis.gap_direction = "DOWN"
                else:
                    analysis.gap_direction = "FLAT"

            # Get futures gaps (ES = SPY, NQ = QQQ)
            analysis.es_gap_pct = self._get_futures_gap("ES")
            analysis.nq_gap_pct = self._get_futures_gap("NQ")

            # Get GEX context
            gex_regime, flip_dist = self._get_gex_context()
            analysis.gex_regime = gex_regime
            analysis.gex_flip_distance_pct = flip_dist

            # Get VIX
            analysis.vix_level = self._get_vix_level()

            with self._lock:
                self._gap_cache[ticker] = analysis
                self._last_check_date = self._get_et_now().strftime("%Y-%m-%d")

            if abs(analysis.gap_pct) >= self.config.min_gap_for_signal:
                self._stats["gaps_detected"] += 1
                logger.info(
                    f"OVERNIGHT_GAP: {ticker} gap={analysis.gap_pct:+.2f}% "
                    f"direction={analysis.gap_direction} gex={analysis.gex_regime}"
                )

            return analysis

        except Exception as e:
            logger.warning(f"OVERNIGHT_GAP: Analysis failed for {ticker}: {e}")
            return None

    def generate_signal(self, ticker: str) -> Optional[GapSignal]:
        """
        Generate a gap trading signal if conditions are met.

        Returns GapSignal for opening drive or BERSERKER trigger.
        """
        analysis = self.analyze_gap(ticker)
        if not analysis:
            return None

        gap_pct = abs(analysis.gap_pct)

        # Check minimum gap threshold
        if gap_pct < self.config.min_gap_for_signal:
            return None

        # Determine direction
        if analysis.gap_direction == "UP":
            # Gap up + negative GEX = potential fade (but usually go with gap)
            if analysis.gex_regime in ("negative", "extreme_negative"):
                # Negative GEX + gap up = momentum continuation (dealers chase)
                direction = "CALL"
                confidence_boost = 10
            else:
                direction = "CALL"
                confidence_boost = 0
        elif analysis.gap_direction == "DOWN":
            # Gap down + negative GEX = momentum continuation (dealers sell)
            if analysis.gex_regime in ("negative", "extreme_negative"):
                direction = "PUT"
                confidence_boost = 10
            else:
                direction = "PUT"
                confidence_boost = 0
        else:
            return None  # No significant gap

        # Calculate confidence
        base_confidence = 60
        gap_confidence = min(30, gap_pct * 15)  # Up to +30 for gap size
        gex_confidence = confidence_boost
        confidence = base_confidence + gap_confidence + gex_confidence

        # Determine trigger type
        if gap_pct >= self.config.berserker_gap_threshold:
            trigger_type = "BERSERKER"
            position_pct = self.config.berserker_position_pct
        else:
            trigger_type = "OPEN_DRIVE"
            position_pct = self.config.open_drive_position_pct

        # Boost for extreme gaps
        if gap_pct >= self.config.extreme_gap_threshold:
            confidence = min(95, confidence + 10)
            position_pct = min(40, position_pct + 5)

        # Calculate execution time (market open + delay)
        now_et = self._get_et_now()
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        execute_at = market_open + timedelta(seconds=self.config.execute_delay_seconds)

        reasoning = (
            f"GAP: {analysis.gap_pct:+.2f}% {analysis.gap_direction} | "
            f"GEX: {analysis.gex_regime} ({analysis.gex_flip_distance_pct:.1f}% from flip) | "
            f"ES: {analysis.es_gap_pct:+.2f}% | NQ: {analysis.nq_gap_pct:+.2f}% | "
            f"VIX: {analysis.vix_level:.1f}"
        )

        signal = GapSignal(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            trigger_type=trigger_type,
            gap_analysis=analysis,
            reasoning=reasoning,
            execute_at=execute_at,
        )

        self._stats["signals_generated"] += 1
        if trigger_type == "BERSERKER":
            self._stats["berserker_triggers"] += 1

        logger.warning(
            f"OVERNIGHT_GAP: {ticker} SIGNAL | "
            f"Type: {trigger_type} | Direction: {direction} | "
            f"Confidence: {confidence:.0f}% | Position: {position_pct:.0f}% | "
            f"Execute at: {execute_at.strftime('%H:%M:%S')} ET"
        )

        with self._lock:
            self._pending_signals[ticker] = signal

        return signal

    def get_pending_signals(self) -> List[GapSignal]:
        """Get all pending gap signals for today."""
        with self._lock:
            return list(self._pending_signals.values())

    def should_trigger_berserker(self) -> Tuple[bool, Optional[GapSignal]]:
        """
        Check if any gap qualifies for BERSERKER auto-trigger.

        Returns (should_trigger, signal)
        """
        for ticker in self.config.monitor_tickers:
            signal = self.generate_signal(ticker)
            if signal and signal.trigger_type == "BERSERKER":
                return True, signal
        return False, None

    def mark_executed(self, ticker: str):
        """Mark a gap signal as executed."""
        with self._lock:
            self._executed_today.append(ticker)
            if ticker in self._pending_signals:
                del self._pending_signals[ticker]

    # ========== DATA FETCHING METHODS ==========

    def _get_previous_close(self, ticker: str) -> float:
        """Get previous day's close."""
        try:
            from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
            bars = polygon_enhanced.get_daily_bars(ticker, limit=2)
            if bars and len(bars) >= 2:
                return bars[-2].get("c", 0)
        except Exception as e:
            logger.debug(f"Previous close fetch failed for {ticker}: {e}")
        return 0

    def _get_premarket_price(self, ticker: str) -> float:
        """Get current pre-market price."""
        try:
            from wsb_snake.collectors.polygon_enhanced import polygon_enhanced
            quote = polygon_enhanced.get_quote(ticker)
            if quote:
                # Pre-market uses last trade price
                return quote.get("price", 0) or quote.get("last", 0)
        except Exception as e:
            logger.debug(f"Pre-market price fetch failed for {ticker}: {e}")
        return 0

    def _get_futures_gap(self, symbol: str) -> float:
        """Get futures gap percentage (ES or NQ)."""
        try:
            # Map to actual futures tickers
            futures_map = {
                "ES": "/ES",  # S&P 500 E-mini
                "NQ": "/NQ",  # Nasdaq E-mini
            }
            futures_ticker = futures_map.get(symbol, symbol)

            # Try to get futures data from available sources
            # Note: May need specific futures data provider
            from wsb_snake.collectors.polygon_enhanced import polygon_enhanced

            # Fallback: estimate from SPY/QQQ pre-market
            if symbol == "ES":
                spy_analysis = self._gap_cache.get("SPY")
                if spy_analysis:
                    return spy_analysis.gap_pct
            elif symbol == "NQ":
                qqq_analysis = self._gap_cache.get("QQQ")
                if qqq_analysis:
                    return qqq_analysis.gap_pct

        except Exception as e:
            logger.debug(f"Futures gap fetch failed for {symbol}: {e}")
        return 0

    def _get_gex_context(self) -> Tuple[str, float]:
        """Get GEX regime from HYDRA."""
        try:
            from wsb_snake.collectors.hydra_bridge import get_hydra_intel
            intel = get_hydra_intel()

            if intel.connected:
                regime = intel.gex_regime.lower() if intel.gex_regime else "neutral"
                flip_dist = intel.gex_flip_distance_pct

                # Normalize
                if "extreme" in regime or "very" in regime:
                    regime = "extreme_negative" if "neg" in regime else "extreme_positive"
                elif "neg" in regime:
                    regime = "negative"
                elif "pos" in regime:
                    regime = "positive"
                else:
                    regime = "neutral"

                return regime, flip_dist

        except Exception as e:
            logger.debug(f"GEX context fetch failed: {e}")
        return "neutral", 999.0

    def _get_vix_level(self) -> float:
        """Get current VIX level."""
        try:
            from wsb_snake.collectors.vix_structure import vix_structure
            signal = vix_structure.get_trading_signal()
            return signal.get("vix", 20.0)
        except Exception as e:
            logger.debug(f"VIX fetch failed: {e}")
        return 20.0

    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            **self._stats,
            "pending_signals": len(self._pending_signals),
            "executed_today": len(self._executed_today),
        }


# Singleton
_detector: Optional[OvernightGapDetector] = None


def get_overnight_gap_detector() -> OvernightGapDetector:
    """Get singleton overnight gap detector."""
    global _detector
    if _detector is None:
        _detector = OvernightGapDetector()
    return _detector
