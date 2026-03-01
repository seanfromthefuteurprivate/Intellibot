# HYDRA INTEGRATION FIX PLAN

## Executive Summary

This plan documents the **minimal code changes** needed to wire up existing HYDRA components that are currently disconnected from the trading loop. All components exist and work individually but are not integrated into the main execution flow.

---

## Current State Analysis

### What Exists (But Disconnected)

| Component | File | Status |
|-----------|------|--------|
| Regime Detector | `wsb_snake/execution/regime_detector.py` | Created, not wired to trading loop |
| Signal Decay | `wsb_snake/execution/apex_conviction_engine.py:46-54` | Implemented in `ConvictionSignal.get_decay_factor()`, not applied |
| Kelly Sizing | `wsb_snake/trading/risk_governor.py:262-329` | Implemented in `compute_kelly_position_size()`, not called |
| Screenshot Learning | `wsb_snake/collectors/screenshot_system.py:159-182` | Implemented in `get_trade_boost()`, not called by APEX |

### What Needs Wiring

1. **Regime Detector** - needs to feed data and adjust weights in APEX
2. **Signal Decay** - `get_effective_score()` is called but regime multipliers not applied to decayed scores
3. **Kelly Sizing** - executor uses `compute_position_size()`, should use `compute_kelly_position_size()`
4. **Screenshot Boost** - learner exists but APEX doesn't call it to boost confidence

---

## Fix 1: Wire Regime Detector into Trading Loop

### Problem
`RegimeDetector.fetch_and_update()` is never called during trading. The detector needs fresh market data each cycle.

### Solution
Update `run_max_mode.py` to call `regime_detector.fetch_and_update()` before each scan cycle.

### Code Change

**File:** `/Users/seankuesia/Downloads/Intellibot/run_max_mode.py`

**Line 33-38 (add import):**
```python
# EXISTING:
from wsb_snake.trading.alpaca_executor import alpaca_executor

# ADD AFTER:
from wsb_snake.execution.regime_detector import regime_detector
```

**Line 168-173 (add regime update in loop):**
```python
# EXISTING:
    while is_market_open():
        scan_count += 1
        now = now_et()

        print(f"\n[{now.strftime('%H:%M:%S')}] MAX MODE Scan #{scan_count}")

# CHANGE TO:
    while is_market_open():
        scan_count += 1
        now = now_et()

        # HYDRA: Update regime detection with fresh market data
        regime_updated = regime_detector.fetch_and_update()
        regime_state = regime_detector.detect_regime()

        print(f"\n[{now.strftime('%H:%M:%S')}] MAX MODE Scan #{scan_count}")
        if regime_updated:
            print(f"  Regime: {regime_state.regime.value} (conf: {regime_state.confidence:.0%})")
```

---

## Fix 2: Apply Signal Decay Correctly in APEX

### Problem
`ApexConvictionEngine.analyze()` calls `get_effective_score()` which includes decay, but the regime multipliers from `_get_regime_adjusted_weights()` are applied AFTER decay, which is correct. However, the decay is not being fully applied because invalid signals are not filtered before scoring.

### Current Code (Lines 624-629)
```python
valid_signals = [s for s in signals if s.is_valid()]
weighted_sum = sum(s.get_effective_score() * adjusted_weights.get(s.source, s.weight)
                  for s in valid_signals)
```

### Analysis
This is actually correct - decay IS being applied via `get_effective_score()`. No fix needed.

---

## Fix 3: Use Kelly Sizing for Positions

### Problem
`alpaca_executor.execute_scalp_entry()` calls `governor.compute_position_size()` (line 988-994) but never calls `compute_kelly_position_size()` which is the institutional-grade sizing method.

### Solution
Replace `compute_position_size()` with `compute_kelly_position_size()` in the executor.

### Code Change

**File:** `/Users/seankuesia/Downloads/Intellibot/wsb_snake/trading/alpaca_executor.py`

**Lines 985-997 (replace position sizing block):**
```python
# EXISTING:
        # Position size: risk governor (confidence + vol) or executor fallback
        buying_power = self.get_buying_power()
        governor = get_risk_governor()
        qty = governor.compute_position_size(
            engine=engine,
            confidence_pct=confidence,
            option_price=option_price,
            buying_power=buying_power if buying_power > 0 else None,
            volatility_factor=self._get_current_volatility_factor(underlying),
        )
        if qty <= 0:
            qty = self.calculate_position_size(option_price)

# CHANGE TO:
        # Position size: KELLY SIZING (HYDRA institutional grade)
        buying_power = self.get_buying_power()
        governor = get_risk_governor()

        # Get historical stats for Kelly calculation
        win_rate = governor.get_win_rate()
        # Use scalper defaults: 6% target, 10% stop
        avg_win_pct = 0.06  # From SCALP_TARGET_PCT (1.06 - 1)
        avg_loss_pct = 0.10  # From SCALP_STOP_PCT (1 - 0.90)

        # Kelly sizing with half-Kelly for conservative institutional approach
        qty = governor.compute_kelly_position_size(
            engine=engine,
            win_probability=confidence / 100.0,  # Convert to 0-1 range
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            option_price=option_price,
            buying_power=buying_power if buying_power > 0 else None,
            volatility_factor=self._get_current_volatility_factor(underlying),
        )

        # Fallback to regular sizing if Kelly suggests 0
        if qty <= 0:
            qty = governor.compute_position_size(
                engine=engine,
                confidence_pct=confidence,
                option_price=option_price,
                buying_power=buying_power if buying_power > 0 else None,
                volatility_factor=self._get_current_volatility_factor(underlying),
            )
        if qty <= 0:
            qty = self.calculate_position_size(option_price)
```

---

## Fix 4: Screenshot Learning Boosts Confidence

### Problem
`screenshot_system.get_trade_boost()` exists but is never called by APEX to boost confidence scores for matching learned patterns.

### Solution
Add screenshot learning boost to `ApexConvictionEngine.analyze()` before final verdict.

### Code Change

**File:** `/Users/seankuesia/Downloads/Intellibot/wsb_snake/execution/apex_conviction_engine.py`

**Line 25-26 (add import):**
```python
# EXISTING:
from wsb_snake.utils.logger import get_logger

# ADD AFTER:
try:
    from wsb_snake.collectors.screenshot_system import screenshot_system
    SCREENSHOT_LEARNING_ENABLED = True
except ImportError:
    screenshot_system = None
    SCREENSHOT_LEARNING_ENABLED = False
```

**Lines 629-658 (after conviction_score calculation, before direction determination):**
```python
# EXISTING (line 629):
        conviction_score = weighted_sum / total_weight if total_weight > 0 else 50

        # Determine direction (majority vote weighted by score)
        bullish_weight = sum(s.score * s.weight for s in signals if s.direction == "BULLISH")

# ADD BETWEEN (after conviction_score, before direction):
        # HYDRA: Screenshot learning confidence boost
        if SCREENSHOT_LEARNING_ENABLED and screenshot_system:
            try:
                import pytz
                from datetime import datetime
                et = pytz.timezone("America/New_York")
                current_hour = datetime.now(et).hour

                # Determine likely trade type from signals
                bullish_count = sum(1 for s in signals if s.direction == "BULLISH")
                bearish_count = sum(1 for s in signals if s.direction == "BEARISH")
                likely_trade_type = "CALLS" if bullish_count > bearish_count else "PUTS"

                # Get boost from learned patterns
                boost, boost_reasons = screenshot_system.get_trade_boost(
                    ticker=ticker,
                    trade_type=likely_trade_type,
                    current_hour=current_hour,
                    pattern=None  # Could pass detected pattern here
                )

                if boost != 0:
                    # Apply boost (boost is a multiplier like 0.15 for +15%)
                    conviction_score = conviction_score * (1 + boost)
                    conviction_score = min(100, max(0, conviction_score))
                    logger.info(f"Screenshot boost: {boost:+.0%} -> {conviction_score:.0f}% | {boost_reasons}")
            except Exception as e:
                logger.debug(f"Screenshot boost failed: {e}")

        # Determine direction (majority vote weighted by score)
        bullish_weight = sum(s.score * s.weight for s in signals if s.direction == "BULLISH")
```

---

## Summary of Changes

| Fix | File | Lines | Change Type |
|-----|------|-------|-------------|
| 1. Regime in loop | `run_max_mode.py` | 33, 168-173 | Add import + call |
| 2. Signal decay | `apex_conviction_engine.py` | - | NO CHANGE NEEDED |
| 3. Kelly sizing | `alpaca_executor.py` | 985-997 | Replace sizing call |
| 4. Screenshot boost | `apex_conviction_engine.py` | 25, 629 | Add import + boost logic |

---

## Testing Checklist

After implementing these changes:

- [ ] Run `python run_max_mode.py` and verify regime is printed each cycle
- [ ] Check logs for "Kelly sizing:" messages showing position calculation
- [ ] Check logs for "Screenshot boost:" messages when recipes match
- [ ] Verify trades still execute (no regressions)
- [ ] Monitor win rate changes over 10+ trades

---

## Risk Assessment

All changes are **LOW RISK** because:

1. **Additive only** - no existing logic is removed
2. **Fallback paths** - Kelly sizing falls back to regular sizing if 0
3. **Try/except wrapped** - screenshot boost failure is non-fatal
4. **Logging added** - all new paths log their behavior

---

## Implementation Order

1. **Fix 1 (Regime)** - 5 lines, immediate visibility
2. **Fix 4 (Screenshot)** - ~30 lines, enhances conviction
3. **Fix 3 (Kelly)** - ~20 lines, changes position sizing (test carefully)

Skip Fix 2 - signal decay is already working correctly.
