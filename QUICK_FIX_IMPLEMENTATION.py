#!/usr/bin/env python3
"""
QUICK FIX GUIDE - Apply to hell_backtest_v5.py
These are the exact code changes needed to align with 79 historical trades

Run: python3 QUICK_FIX_IMPLEMENTATION.py
This will show you exactly what to change
"""

# ============================================================================
# FIX #1: DELETE LOTTO_TICKET PATTERN (CRITICAL)
# ============================================================================
# Location: hell_backtest_v5.py, Lines 56-64
# Current code:
CURRENT_CODE_1 = '''
    "LOTTO_TICKET": {
        "otm_range": (3, 5),
        "price_range": (0.10, 0.50),
        "max_sizing": 0.12,
        "stop_loss": -0.75,
        "trail_trigger": 3.00,
        "trail_pct": 0.30,
        "max_hold_minutes": 90,
    },
'''

FIXED_CODE_1 = '''
    # DELETED: LOTTO_TICKET had 0% win rate on 12 historical trades
    # Always lost -27.95% average. Removes 12% of trades that lose.
'''

EXPLANATION_1 = """
WHY: Historical data shows:
  - 0 wins out of 12 cheap lotto trades
  - -27.95% to -69% average loss
  - Never profitable in any backtest range

ACTION: Simply delete the entire "LOTTO_TICKET" dict entry
"""

# ============================================================================
# FIX #2: ADD ENTRY PRICE FLOOR (CRITICAL)
# ============================================================================
# Location: hell_backtest_v5.py, Lines 481-552 (select_berserker_strike function)
# Change the price range checks:

CURRENT_CODE_2 = '''
def select_berserker_strike(spy_price: float, direction: str, pattern: str,
                            date: str, expiry: str, prefetched: Dict, bar_ts: int) -> Optional[Dict]:
    """Select strike for BERSERKER mode based on pattern."""
    config = PATTERN_CONFIG.get(pattern, PATTERN_CONFIG["DEFAULT"])
    otm_min, otm_max = config["otm_range"]
    price_min, price_max = config["price_range"]

    candidates = []

    for offset in range(otm_min, otm_max + 1):
        if direction == "CALL":
            strike = round(spy_price) + offset
        else:
            strike = round(spy_price) - offset

        ticker = build_option_ticker(strike, expiry, direction)
        bars = prefetched.get(ticker)

        if not bars:
            bars = fetch_option_bars(ticker, date)
            if bars:
                prefetched[ticker] = bars

        if bars:
            result = get_option_price_at_time(bars, bar_ts)
            if result:
                price, volume = result
                if volume >= 50 and price_min <= price <= price_max:  # <-- ADD FLOOR HERE
                    candidates.append({
                        "strike": strike,
                        "price": price,
                        "volume": volume,
                        "ticker": ticker,
                        "offset": offset
                    })
'''

FIXED_CODE_2 = '''
def select_berserker_strike(spy_price: float, direction: str, pattern: str,
                            date: str, expiry: str, prefetched: Dict, bar_ts: int) -> Optional[Dict]:
    """Select strike for BERSERKER mode based on pattern."""
    config = PATTERN_CONFIG.get(pattern, PATTERN_CONFIG["DEFAULT"])
    otm_min, otm_max = config["otm_range"]
    price_min, price_max = config["price_range"]

    # HISTORICAL DATA: Sub-$0.50 options have 0% win rate
    MIN_ENTRY_PRICE = 0.50  # Reject cheap lotto tickets
    MAX_ENTRY_PRICE = 5.00  # Don't go mega ITM either

    candidates = []

    for offset in range(otm_min, otm_max + 1):
        if direction == "CALL":
            strike = round(spy_price) + offset
        else:
            strike = round(spy_price) - offset

        ticker = build_option_ticker(strike, expiry, direction)
        bars = prefetched.get(ticker)

        if not bars:
            bars = fetch_option_bars(ticker, date)
            if bars:
                prefetched[ticker] = bars

        if bars:
            result = get_option_price_at_time(bars, bar_ts)
            if result:
                price, volume = result
                # ADD PRICE FLOOR CHECK
                if price < MIN_ENTRY_PRICE or price > MAX_ENTRY_PRICE:
                    continue  # Skip, outside acceptable price range
                if volume >= 50 and price_min <= price <= price_max:
                    candidates.append({
                        "strike": strike,
                        "price": price,
                        "volume": volume,
                        "ticker": ticker,
                        "offset": offset
                    })
'''

EXPLANATION_2 = """
WHY: Historical win rates by entry price:
  Under $0.10:    0% (0/2 trades)
  $0.10-$0.50:    0% (0/10 trades)  ← 12 LOSING TRADES
  $0.50-$1.00:   13% (1/8 trades)
  $1.00-$2.00:   25% (7/28 trades)  ← TARGET ZONE
  Over $2.00:    23% (7/31 trades)  ← GOOD ZONE

ACTION: Add 2 lines:
  MIN_ENTRY_PRICE = 0.50
  MAX_ENTRY_PRICE = 5.00

Then check: if price < MIN or price > MAX: continue
"""

# ============================================================================
# FIX #3: EXTEND HOLD TIMES (HIGH PRIORITY)
# ============================================================================
# Location: hell_backtest_v5.py, Lines 36-52 (SCALP_CONFIG)

CURRENT_CODE_3 = '''
SCALP_CONFIG = {
    "sizing": 0.10,
    "max_concurrent": 1,
    "max_daily_trades": 3,
    "profit_target": 0.25,
    "stop_loss": -0.15,
    "time_stop_minutes": 10,  # <-- CHANGE THIS
    "time_stop_min_gain": 0.10,
    "entry_momentum": 0.0015,
    "otm_offset": 0,
}
'''

FIXED_CODE_3 = '''
SCALP_CONFIG = {
    "sizing": 0.10,
    "max_concurrent": 1,
    "max_daily_trades": 3,
    "profit_target": 0.25,
    "stop_loss": -0.15,
    "time_stop_minutes": 30,  # CHANGE: 10 → 30 (winners hold 45 min avg)
    "time_stop_min_gain": 0.10,
    "entry_momentum": 0.0015,
    "otm_offset": 0,
}
'''

EXPLANATION_3 = """
WHY: Historical hold times:
  Winners avg hold: 45.4 minutes
  Losers avg hold:  16.0 minutes
  Winners hold 2.8x longer!

CURRENT: Exits winners at 10 min if not +10%
RESULT:  Winners never get to breathe

ACTION: Change line 42 from 10 to 30
"""

# ============================================================================
# FIX #4: EXTEND BERSERKER HOLDS (HIGH PRIORITY)
# ============================================================================
# Location: hell_backtest_v5.py, Lines 55-101 (PATTERN_CONFIG)

CURRENT_CODE_4 = '''
PATTERN_CONFIG = {
    "REVERSAL_PUT": {
        ...
        "max_hold_minutes": 45,  # <-- CHANGE THIS
    },
    "MOMENTUM_CALL": {
        ...
        "max_hold_minutes": 60,  # <-- CHANGE THIS
    },
    "PRECIOUS_METALS_MOMENTUM": {
        ...
        "max_hold_minutes": 50,  # <-- CHANGE THIS
    },
    "DEFAULT": {
        ...
        "max_hold_minutes": 60,  # <-- CHANGE THIS
    }
}
'''

FIXED_CODE_4 = '''
PATTERN_CONFIG = {
    "REVERSAL_PUT": {
        ...
        "max_hold_minutes": 70,  # CHANGE: 45 → 70
    },
    "MOMENTUM_CALL": {
        ...
        "max_hold_minutes": 75,  # CHANGE: 60 → 75
    },
    "PRECIOUS_METALS_MOMENTUM": {
        ...
        "max_hold_minutes": 75,  # CHANGE: 50 → 75
    },
    "DEFAULT": {
        ...
        "max_hold_minutes": 75,  # CHANGE: 60 → 75
    }
}
'''

EXPLANATION_4 = """
WHY: Winners hold 45-90 minutes before exiting profitably.
Current max of 45-60 cuts off winners too early.

ACTION: Change 4 lines to extend max_hold to 70-75 minutes
This matches historical 45 min avg + buffer for variance
"""

# ============================================================================
# FIX #5: MOVE SCALP WINDOWS (CRITICAL)
# ============================================================================
# Location: hell_backtest_v5.py, Lines 49-52

CURRENT_CODE_5 = '''
SCALP_WINDOWS = [
    (9, 35, 11, 0),    # 9:35 AM - 11:00 AM     <- NO HISTORICAL DATA
    (14, 0, 15, 30),   # 2:00 PM - 3:30 PM      <- 0% WIN RATE
]
'''

FIXED_CODE_5 = '''
SCALP_WINDOWS = [
    # Historical data shows:
    # 14:00-15:00: 0% win rate (0/11 trades)
    # 16:00-17:00: 33% win rate (1/3 trades)
    # 17:00-18:00: 29% win rate (4/14 trades)  <- BEST
    # 18:00-20:00: 19-23% win rate
    (16, 0, 20, 30),   # 4:00 PM - 8:30 PM (your best window)
]
'''

EXPLANATION_5 = """
WHY: Historical win rate by hour:
  14:00 (2 PM):  0% (0/6)
  15:00 (3 PM):  0% (0/5)
  16:00 (4 PM): 33% (1/3)
  17:00 (5 PM): 29% (4/14)  ← PEAK
  18:00 (6 PM): 19% (4/21)
  19:00 (7 PM): 18% (3/17)
  20:00 (8 PM): 23% (3/13)

PROBLEM: Current window (2-3 PM) is 0% win rate
SOLUTION: Move to 4-8:30 PM where data shows wins

ACTION: Replace line 49-52 with new window
"""

# ============================================================================
# FIX #6: ALSO UPDATE PRICE_RANGE IN CONFIGS (MEDIUM)
# ============================================================================
# Location: hell_backtest_v5.py, Lines 66-99

CURRENT_CODE_6 = '''
PATTERN_CONFIG = {
    "REVERSAL_PUT": {
        "otm_range": (1, 2),
        "price_range": (0.30, 1.50),  # <-- TOO LOW
        ...
    },
    "MOMENTUM_CALL": {
        "otm_range": (1, 3),
        "price_range": (0.30, 1.50),  # <-- TOO LOW
        ...
    },
    "PRECIOUS_METALS_MOMENTUM": {
        "otm_range": (1, 2),
        "price_range": (0.30, 1.50),  # <-- TOO LOW
        ...
    },
    "DEFAULT": {
        "otm_range": (2, 3),
        "price_range": (0.20, 1.00),  # <-- TOO LOW
        ...
    }
}
'''

FIXED_CODE_6 = '''
PATTERN_CONFIG = {
    "REVERSAL_PUT": {
        "otm_range": (1, 2),
        "price_range": (0.50, 2.00),  # CHANGE: 0.30 → 0.50 (price floor)
        ...
    },
    "MOMENTUM_CALL": {
        "otm_range": (1, 3),
        "price_range": (0.50, 2.00),  # CHANGE: 0.30 → 0.50
        ...
    },
    "PRECIOUS_METALS_MOMENTUM": {
        "otm_range": (1, 2),
        "price_range": (0.50, 2.00),  # CHANGE: 0.30 → 0.50
        ...
    },
    "DEFAULT": {
        "otm_range": (2, 3),
        "price_range": (0.50, 1.50),  # CHANGE: 0.20 → 0.50
        ...
    }
}
'''

EXPLANATION_6 = """
WHY: Enforce $0.50 minimum entry price across all patterns
Historical data is clear:
  $0.10-$0.50: 0% win rate (0/12)
  $0.50+:      22-25% win rate (14/35)

ACTION: Update 4 price_range tuples to start at 0.50
"""

# ============================================================================
# SUMMARY TABLE
# ============================================================================

SUMMARY = """
╔════════════════════════════════════════════════════════════════════════════╗
║ IMPLEMENTATION SUMMARY - Apply These 6 Fixes                              ║
╚════════════════════════════════════════════════════════════════════════════╝

FIX #1: DELETE LOTTO_TICKET PATTERN
  File: hell_backtest_v5.py
  Lines: 56-64
  Change: Delete entire "LOTTO_TICKET" block (9 lines)
  Impact: Removes 12% of guaranteed losing trades
  Effort: 30 seconds

FIX #2: ADD ENTRY PRICE FLOOR
  File: hell_backtest_v5.py
  Lines: 481-552 (select_berserker_strike function)
  Change: Add price validation (3 lines)
    MIN_ENTRY_PRICE = 0.50
    MAX_ENTRY_PRICE = 5.00
    if price < MIN or price > MAX: continue
  Impact: Rejects all sub-$0.50 options (0% win rate)
  Effort: 2 minutes

FIX #3: EXTEND SCALP TIME STOP
  File: hell_backtest_v5.py
  Line: 42
  Change: "time_stop_minutes": 10 → 30
  Impact: Winners get 30 min instead of 10 min to breathe
  Effort: 10 seconds

FIX #4: EXTEND BERSERKER HOLDS
  File: hell_backtest_v5.py
  Lines: 72, 81, 89, 99
  Change: max_hold_minutes: 45→70, 60→75, 50→75, 60→75
  Impact: Aligns with 45 min average winner hold time
  Effort: 1 minute

FIX #5: MOVE SCALP WINDOWS
  File: hell_backtest_v5.py
  Lines: 49-52
  Change: (9,35,11,0) + (14,0,15,30) → (16,0,20,30)
  Impact: Trade 4-8:30 PM (29% WR) instead of 2-3 PM (0% WR)
  Effort: 30 seconds

FIX #6: UPDATE PRICE RANGES
  File: hell_backtest_v5.py
  Lines: 66, 75, 84, 93
  Change: price_range: (0.30→0.50, 0.20→0.50)
  Impact: Enforces $0.50 minimum across all patterns
  Effort: 1 minute

TOTAL EFFORT: < 10 minutes
TOTAL IMPACT: Win rate 19% → 25-30% (estimated)

╔════════════════════════════════════════════════════════════════════════════╗
║ VERIFICATION CHECKLIST                                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

After applying fixes, verify:

□ LOTTO_TICKET no longer appears in output
□ No trades with entry price < $0.50
□ Scalp mode trades only 4-8:30 PM (check logs)
□ Hold times increased (check trade durations)
□ Win rate improves toward 25%+
□ No entries in 2-3 PM window
□ Total P&L improves (less negative or positive)

Test command:
python3 hell_backtest_v5.py 2>&1 | grep -E "LOTTO|entry|SCALP|Hold|WIN"

"""

print(SUMMARY)

print("\n" + "="*80)
print("DETAILED CHANGES BY FIX")
print("="*80)

print("\n" + "FIX #1: DELETE LOTTO_TICKET".center(80, "="))
print(EXPLANATION_1)
print("\nCURRENT:\n" + CURRENT_CODE_1)
print("\nFIXED:\n" + FIXED_CODE_1)

print("\n" + "FIX #2: ADD ENTRY PRICE FLOOR".center(80, "="))
print(EXPLANATION_2)
print("\nCURRENT:\n" + CURRENT_CODE_2)
print("\nFIXED:\n" + FIXED_CODE_2)

print("\n" + "FIX #3: EXTEND SCALP TIME_STOP".center(80, "="))
print(EXPLANATION_3)
print("\nCURRENT:\n" + CURRENT_CODE_3)
print("\nFIXED:\n" + FIXED_CODE_3)

print("\n" + "FIX #4: EXTEND BERSERKER HOLDS".center(80, "="))
print(EXPLANATION_4)
print("\nCURRENT:\n" + CURRENT_CODE_4)
print("\nFIXED:\n" + FIXED_CODE_4)

print("\n" + "FIX #5: MOVE SCALP WINDOWS".center(80, "="))
print(EXPLANATION_5)
print("\nCURRENT:\n" + CURRENT_CODE_5)
print("\nFIXED:\n" + FIXED_CODE_5)

print("\n" + "FIX #6: UPDATE PRICE RANGES".center(80, "="))
print(EXPLANATION_6)
print("\nCURRENT:\n" + CURRENT_CODE_6)
print("\nFIXED:\n" + FIXED_CODE_6)

print("\n" + "="*80)
print("QUESTIONS? Check these files:")
print("  1. TRADE_ANALYSIS_GAP_REPORT.md - Full analysis")
print("  2. BACKTEST_FIX_CHECKLIST.md - Detailed implementation guide")
print("  3. DATA_INSIGHTS_QUICK_REFERENCE.md - Quick lookup table")
print("="*80)
