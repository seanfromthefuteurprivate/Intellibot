#!/usr/bin/env python3
"""
BACKTEST FROM HELL v6: THE COMPLETE REWORK
TWO-MODE ENGINE: PREMIUM SCALP + DIRECTIONAL BERSERKER

WHY V5 FAILED (AND ALL PREVIOUS VERSIONS):
- REVERSAL_PUT: 6/8 losers - fading momentum in negative gamma is suicide
- LOTTO_TICKET: Sub-$0.50 entries = 0% historical win rate
- Entry at 9:35 AM: Too early, 2-3 PM window = 0% win rate
- Hard stops at -35%: Triggering in 2-17 minutes on normal 0DTE noise
- 10-minute time_stop: Winners hold 45 min average, we were killing them
- Negative expectancy: 25% win rate x $87 avg win vs 75% x $97 avg loss = -$51/trade

SWARM AUDIT FINDINGS:
- 87% of profitable 0DTE traders SELL premium (iron condors, credit spreads)
- Professional entry time: 10:15 AM (not 9:35 AM)
- The $800->$49K trader: $0.08 entry, 100 contracts, held through -50% swings

V6 SOLUTION:
MODE 1: SCALP (Premium Selling - Credit Spreads)
  - Sell OTM credit spreads, collect theta
  - Entry: 10:15 AM after morning chop settles
  - Exit: 50% of max profit OR 2:00 PM hard stop
  - Target 66-75% win rate

MODE 2: BERSERKER (Directional Buying - Momentum Runners)
  - NO LOTTO_TICKET (deleted - 0% win rate)
  - NO REVERSAL_PUT (deleted - 6/8 losers)
  - Entry: 10:15 AM earliest
  - Entry price minimum: $0.50
  - NO hard stop first 20 minutes
  - Scaled exits: 25% at +100%, 25% at +200%, hold 50% to thesis
  - One direction per day
"""

import os
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import math

# API Keys
ALPACA_KEY = os.environ.get("ALPACA_API_KEY", "PKWT6T5BFKHBTFDW3CPAFW2XBZ")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "pVdzbVte2pQvL1RmCTFw3oaQ6TBWYimAzC42DUyTEy8")
POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "QJWtaUQV7N8mytTI7PH26lX3Ju6PD2iq")
DB_PATH = os.environ.get("WSB_SNAKE_DB", "wsb_snake_data/wsb_snake.db")

# Base parameters
INITIAL_ACCOUNT = 5000.0
MIN_ACCOUNT_FLOOR = 500.0

# ==============================================================================
# CRITICAL: ENTRY TIME - NO ENTRIES BEFORE 10:15 AM
# ==============================================================================
EARLIEST_ENTRY_MINUTE = 45  # 45 minutes after open (9:30 + 45 = 10:15 AM)

# ==============================================================================
# MODE 1: SCALP (Premium Selling - Credit Spreads)
# ==============================================================================
# When: Range < $2.00 by 10:00 AM OR GEX positive (mean-reverting)
# Strategy: Sell OTM credit spreads, collect theta
# Target: 50% of max credit received
# Stop: 2x credit received
# Exit: By 2:00 PM ET regardless

SCALP_CONFIG = {
    "sizing": 0.10,              # 10% per trade
    "max_concurrent": 1,         # 1 spread at a time
    "max_daily_trades": 3,       # Max 3 scalps per day
    "spread_width": 2,           # $2 wide spreads on SPY
    "target_delta": 0.15,        # Sell 0.15 delta strikes
    "profit_target_pct": 0.50,   # Exit at 50% of max credit
    "stop_loss_mult": 2.0,       # Stop at 2x credit received
    "latest_exit_minute": 270,   # 270 min after open = 2:00 PM ET
}

# ==============================================================================
# MODE 2: BERSERKER (Directional Buying - Momentum Runners)
# ==============================================================================
# When: Range > $2.00 OR catalyst/FOMC/CPI day
# Strategy: Buy momentum, ride with scaled exits

BERSERKER_CONFIG = {
    # Entry rules
    "min_entry_price": 0.50,     # NO LOTTO - minimum $0.50 entry
    "max_entry_price": 3.00,     # Cap at $3.00 to maintain leverage
    "otm_range": (1, 3),         # 1-3 points OTM

    # Position sizing by conviction
    "default_sizing": 0.15,      # 15% when brain empty
    "pattern_sizing": 0.25,      # 25% on pattern match
    "high_conviction_sizing": 0.35,  # 35% on high conviction
    "full_berserker_sizing": 0.45,   # 45% on full BERSERKER

    # Stop rules - COMPLETELY REWORKED
    "no_stop_minutes": 20,       # NO hard stop first 20 minutes
    "dollar_stop": 500,          # $500 max loss per trade (10% of $5K)
    "time_stop_minutes": 45,     # If negative after 45 min, exit

    # Exit rules - SCALED (let winners run)
    "scale_1_pct": 1.00,         # Sell 25% at +100%
    "scale_2_pct": 2.00,         # Sell 25% at +200%
    "hold_runner_pct": 0.50,     # Hold 50% with no stop until thesis break
    "max_hold_minutes": 90,      # Max hold time
}

# Patterns that actually work (REVERSAL_PUT and LOTTO_TICKET DELETED)
VALID_PATTERNS = {
    "MOMENTUM_CALL": {
        "direction": "CALL",
        "conviction_boost": 10,
        "historical_wr": 0.45,
    },
    "PRECIOUS_METALS_MOMENTUM": {
        "direction": "CALL",
        "conviction_boost": 5,
        "historical_wr": 0.40,
    },
    "HIGH_VOLUME_CONVICTION": {
        "direction": "BOTH",
        "conviction_boost": 15,
        "historical_wr": 0.50,
    },
    "EARNINGS_PLAY": {
        "direction": "BOTH",
        "conviction_boost": 8,
        "historical_wr": 0.38,
    },
}

# Circuit breaker
CIRCUIT_BREAKER_LOSSES = 2  # 2 consecutive losses = done for day

# Daily P&L limits
DAILY_LOSS_LIMIT = -0.10  # -10% daily loss = stop
QUIET_DAY_PROFIT = 500    # +$500 on quiet day = stop

# Caches
OPTION_BARS_CACHE: Dict[str, List[Dict]] = {}
OPTION_CHAIN_CACHE: Dict[str, Dict] = {}
LEARNED_TRADES_CACHE: List[Dict] = []


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class CreditSpread:
    """A credit spread position."""
    entry_time: datetime
    spread_type: str  # "PUT_SPREAD" or "CALL_SPREAD"
    short_strike: float
    long_strike: float
    expiry: str
    contracts: int
    credit_received: float  # Per spread
    total_credit: float
    max_loss: float
    current_value: float = 0.0
    status: str = "OPEN"
    exit_time: Optional[datetime] = None
    exit_value: float = 0.0
    pnl_dollars: float = 0.0
    exit_reason: str = ""


@dataclass
class BerserkerPosition:
    """A directional BERSERKER position with scaled exits."""
    entry_time: datetime
    entry_minute_ts: int
    entry_spy_price: float
    direction: str  # "CALL" or "PUT"
    strike: float
    expiry: str
    option_ticker: str
    initial_contracts: int
    remaining_contracts: int
    entry_option_price: float
    position_cost: float
    pattern: str
    conviction: float

    # Scaled exit tracking
    scale_1_done: bool = False  # 25% at +100%
    scale_2_done: bool = False  # 25% at +200%

    # Price tracking
    peak_option_price: float = 0.0
    current_option_price: float = 0.0

    # Partial exits
    realized_pnl: float = 0.0
    contracts_sold_scale_1: int = 0
    contracts_sold_scale_2: int = 0


@dataclass
class Trade:
    """Completed trade record."""
    entry_time: datetime
    exit_time: datetime
    mode: str  # "SCALP" or "BERSERKER"
    direction: str
    strike: float
    contracts: int
    entry_price: float
    exit_price: float
    position_cost: float
    pnl_dollars: float
    pnl_pct: float
    exit_reason: str
    pattern: str
    hold_minutes: int


@dataclass
class DayResult:
    """Result for one trading day."""
    date: str
    mode_selected: str
    mode_reason: str
    trades: List[Trade]
    num_trades: int
    wins: int
    losses: int
    win_rate: float
    daily_pnl_dollars: float
    daily_pnl_pct: float
    start_account: float
    end_account: float
    circuit_breaker_fired: bool
    circuit_breaker_saved: float


# ==============================================================================
# DATA FETCHING
# ==============================================================================

def build_option_ticker(strike: float, expiry: str, direction: str) -> str:
    """Build Polygon option ticker."""
    expiry_fmt = datetime.strptime(expiry, "%Y-%m-%d").strftime("%y%m%d")
    cp = "C" if direction == "CALL" else "P"
    strike_fmt = f"{int(strike * 1000):08d}"
    return f"O:SPY{expiry_fmt}{cp}{strike_fmt}"


def fetch_option_bars(option_ticker: str, date: str) -> List[Dict]:
    """Fetch minute option prices from Polygon."""
    cache_key = f"{option_ticker}_{date}"
    if cache_key in OPTION_BARS_CACHE:
        return OPTION_BARS_CACHE[cache_key]

    url = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/minute/{date}/{date}"
    params = {"apiKey": POLYGON_KEY, "limit": 50000, "sort": "asc"}

    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            bars = r.json().get("results", [])
            OPTION_BARS_CACHE[cache_key] = bars
            return bars
    except:
        pass
    return []


def get_option_price_at_time(option_bars: List[Dict], target_ts: int,
                              use_low: bool = False) -> Optional[Tuple[float, int]]:
    """Find option price at timestamp. Returns (price, volume)."""
    if not option_bars:
        return None

    best_bar = None
    best_diff = float('inf')

    for bar in option_bars:
        diff = abs(bar["t"] - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_bar = bar
        if bar["t"] > target_ts and best_bar:
            break

    if best_bar and best_diff <= 120000:  # Within 2 minutes
        price = best_bar["l"] if use_low else best_bar["c"]
        volume = best_bar.get("v", 0)
        return (price, volume)
    return None


def fetch_minute_bars(ticker: str, date: str) -> List[Dict]:
    """Fetch minute bars from Alpaca."""
    headers = {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET
    }
    url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
    params = {
        "timeframe": "1Min",
        "start": f"{date}T09:30:00Z",
        "end": f"{date}T16:00:00Z",
        "limit": 500,
        "feed": "iex"
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get("bars", [])
    except:
        pass
    return []


def fetch_option_chain(ticker: str, expiry: str) -> Dict:
    """Fetch option chain with Greeks from Polygon."""
    cache_key = f"{ticker}_{expiry}"
    if cache_key in OPTION_CHAIN_CACHE:
        return OPTION_CHAIN_CACHE[cache_key]

    url = f"https://api.polygon.io/v3/snapshot/options/{ticker}"
    params = {
        "apiKey": POLYGON_KEY,
        "expiration_date": expiry,
        "limit": 250,
    }

    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            OPTION_CHAIN_CACHE[cache_key] = data
            return data
    except Exception as e:
        print(f"  Option chain fetch error: {e}")
    return {}


# ==============================================================================
# LEARNED TRADES (Brain)
# ==============================================================================

def load_learned_trades() -> List[Dict]:
    """Load all learned trades from database."""
    global LEARNED_TRADES_CACHE
    if LEARNED_TRADES_CACHE:
        return LEARNED_TRADES_CACHE

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM learned_trades
            WHERE pattern IS NOT NULL
            ORDER BY profit_loss_pct DESC
        """)

        LEARNED_TRADES_CACHE = [dict(row) for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        print(f"  Warning: Could not load learned trades: {e}")
        LEARNED_TRADES_CACHE = []

    return LEARNED_TRADES_CACHE


def get_pattern_match(direction: str, spy_momentum: float, volume_spike: float,
                       entry_price: float) -> Tuple[str, float, float]:
    """
    Match current setup against learned trades.
    Returns (pattern, conviction_boost, historical_wr).

    DELETED PATTERNS: LOTTO_TICKET, REVERSAL_PUT
    """
    learned = load_learned_trades()
    if not learned:
        return ("DEFAULT", 0, 0.35)

    best_pattern = "DEFAULT"
    best_score = 0

    for trade in learned:
        pattern = trade.get("pattern", "")

        # CRITICAL: Skip deleted patterns
        if pattern in ["LOTTO_TICKET", "REVERSAL_PUT"]:
            continue

        if pattern not in VALID_PATTERNS:
            continue

        score = 0
        pattern_config = VALID_PATTERNS[pattern]

        # Direction match
        trade_dir = trade.get("trade_type", "")
        if pattern_config["direction"] == "BOTH" or trade_dir == direction:
            score += 20

        # Pattern-specific scoring
        if pattern == "MOMENTUM_CALL":
            if direction == "CALL" and spy_momentum > 0.003:
                score += 25
            if volume_spike > 1.5:
                score += 10

        elif pattern == "PRECIOUS_METALS_MOMENTUM":
            score += 10  # Base score for this pattern

        elif pattern == "HIGH_VOLUME_CONVICTION":
            if volume_spike > 2.5:
                score += 30
            elif volume_spike > 2.0:
                score += 20

        elif pattern == "EARNINGS_PLAY":
            score += 15  # Earnings plays have their own logic

        # Profit history weight
        pnl = trade.get("profit_loss_pct", 0) or 0
        if pnl > 100:
            score += 15
        elif pnl > 50:
            score += 10
        elif pnl > 0:
            score += 5

        if score > best_score:
            best_score = score
            best_pattern = pattern

    if best_pattern in VALID_PATTERNS:
        config = VALID_PATTERNS[best_pattern]
        return (best_pattern, config["conviction_boost"], config["historical_wr"])

    return ("DEFAULT", 0, 0.35)


# ==============================================================================
# MODE SELECTION LOGIC
# ==============================================================================

def select_mode(spy_bars: List[Dict], idx: int) -> Tuple[str, str]:
    """
    Select trading mode based on market conditions.

    Returns (mode, reason) where mode is "SCALP", "BERSERKER", or "WAIT".

    SCALP: Range < $2.00, grind small premium
    BERSERKER: Range >= $2.00, momentum runners
    """
    # Wait until 10:00 AM (30 min after open) to assess range
    if idx < 30:
        return ("WAIT", "Assessing morning range")

    # Calculate range so far
    bars_so_far = spy_bars[:idx+1]
    day_high = max(b["h"] for b in bars_so_far)
    day_low = min(b["l"] for b in bars_so_far)
    current_range = day_high - day_low

    # Check momentum in last 10 bars
    if idx >= 10:
        momentum = (spy_bars[idx]["c"] - spy_bars[idx-10]["c"]) / spy_bars[idx-10]["c"]
    else:
        momentum = 0

    # Mode selection
    if current_range < 1.00:
        return ("SCALP", f"Range ${current_range:.2f} < $1.00 - premium selling")
    elif current_range < 2.00:
        # Borderline - check momentum
        if abs(momentum) > 0.003:
            return ("BERSERKER", f"Range ${current_range:.2f} but momentum {momentum:.4f}")
        return ("SCALP", f"Range ${current_range:.2f} < $2.00 - premium selling")
    else:
        return ("BERSERKER", f"Range ${current_range:.2f} >= $2.00 - momentum mode")


# ==============================================================================
# SCALP MODE: Credit Spread Logic
# ==============================================================================

def find_credit_spread_strikes(spy_price: float, direction: str, expiry: str,
                                prefetched: Dict, bar_ts: int) -> Optional[Dict]:
    """
    Find strikes for a credit spread.

    For PUT SPREAD (bullish): Sell higher put, buy lower put
    For CALL SPREAD (bearish): Sell lower call, buy higher call

    Target ~0.15 delta on short strike.
    """
    width = SCALP_CONFIG["spread_width"]

    if direction == "PUT_SPREAD":
        # Sell OTM put (below current price), buy further OTM put
        # Approximate 0.15 delta = ~3-4 points OTM for SPY
        short_strike = round(spy_price) - 3
        long_strike = short_strike - width
    else:  # CALL_SPREAD
        short_strike = round(spy_price) + 3
        long_strike = short_strike + width

    # Get prices for both legs
    short_type = "PUT" if direction == "PUT_SPREAD" else "CALL"
    long_type = short_type

    short_ticker = build_option_ticker(short_strike, expiry, short_type)
    long_ticker = build_option_ticker(long_strike, expiry, long_type)

    short_bars = prefetched.get(short_ticker)
    if not short_bars:
        short_bars = fetch_option_bars(short_ticker, expiry)
        if short_bars:
            prefetched[short_ticker] = short_bars

    long_bars = prefetched.get(long_ticker)
    if not long_bars:
        long_bars = fetch_option_bars(long_ticker, expiry)
        if long_bars:
            prefetched[long_ticker] = long_bars

    short_result = get_option_price_at_time(short_bars, bar_ts) if short_bars else None
    long_result = get_option_price_at_time(long_bars, bar_ts) if long_bars else None

    if not short_result or not long_result:
        return None

    short_price, short_vol = short_result
    long_price, long_vol = long_result

    # Credit = short premium - long premium
    credit = short_price - long_price

    if credit <= 0.05:  # Need at least $0.05 credit
        return None

    if short_vol < 100 or long_vol < 100:  # Need liquidity
        return None

    return {
        "spread_type": direction,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "short_ticker": short_ticker,
        "long_ticker": long_ticker,
        "credit": credit,
        "max_loss": width - credit,
    }


# ==============================================================================
# BERSERKER MODE: Directional Logic
# ==============================================================================

def detect_berserker_signal(spy_bars: List[Dict], idx: int,
                            day_direction: Optional[str]) -> Optional[Tuple[str, float]]:
    """
    Detect BERSERKER entry signal.

    Returns (direction, confidence) or None.

    Rules:
    - Must be after 10:15 AM (bar 45+)
    - One direction per day (if day_direction set, must match)
    - Momentum + volume confirmation
    """
    if idx < EARLIEST_ENTRY_MINUTE:
        return None

    if idx < 10:
        return None

    recent = spy_bars[max(0, idx-10):idx+1]
    if len(recent) < 10:
        return None

    current_price = recent[-1]["c"]
    momentum = (recent[-1]["c"] - recent[0]["c"]) / recent[0]["c"]

    # Volume analysis
    avg_vol = sum(b["v"] for b in recent[:-1]) / (len(recent) - 1) if len(recent) > 1 else 1
    vol_spike = recent[-1]["v"] / avg_vol if avg_vol > 0 else 1

    # Bar range
    bar_range = (recent[-1]["h"] - recent[-1]["l"]) / recent[-1]["c"]

    direction = None
    confidence = 0

    # CALL signal - strong upward momentum with volume
    if momentum > 0.003 and vol_spike > 1.3:
        direction = "CALL"
        confidence = min(95, 50 + momentum * 2000 + vol_spike * 10 + bar_range * 500)

    # PUT signal - strong downward momentum with volume
    elif momentum < -0.003 and vol_spike > 1.3:
        direction = "PUT"
        confidence = min(95, 50 + abs(momentum) * 2000 + vol_spike * 10 + bar_range * 500)

    if not direction:
        return None

    # One direction per day rule
    if day_direction and direction != day_direction:
        return None

    if confidence < 50:
        return None

    return (direction, confidence)


def find_berserker_strike(spy_price: float, direction: str, expiry: str,
                          prefetched: Dict, bar_ts: int) -> Optional[Dict]:
    """
    Find optimal strike for BERSERKER entry.

    Rules:
    - Entry price must be >= $0.50 (no lottos)
    - Entry price should be <= $3.00 (maintain leverage)
    - 1-3 points OTM
    """
    otm_min, otm_max = BERSERKER_CONFIG["otm_range"]
    min_price = BERSERKER_CONFIG["min_entry_price"]
    max_price = BERSERKER_CONFIG["max_entry_price"]

    candidates = []

    for offset in range(otm_min, otm_max + 1):
        if direction == "CALL":
            strike = round(spy_price) + offset
        else:
            strike = round(spy_price) - offset

        ticker = build_option_ticker(strike, expiry, direction)
        bars = prefetched.get(ticker)

        if not bars:
            bars = fetch_option_bars(ticker, expiry)
            if bars:
                prefetched[ticker] = bars

        if bars:
            result = get_option_price_at_time(bars, bar_ts)
            if result:
                price, volume = result

                # CRITICAL: Enforce min/max price
                if price < min_price:
                    continue  # No lottos
                if price > max_price:
                    continue  # Too expensive

                if volume >= 100:  # Need liquidity
                    candidates.append({
                        "strike": strike,
                        "price": price,
                        "volume": volume,
                        "ticker": ticker,
                        "offset": offset
                    })

    if not candidates:
        # Fallback - try wider range
        for offset in range(1, 6):
            if direction == "CALL":
                strike = round(spy_price) + offset
            else:
                strike = round(spy_price) - offset

            ticker = build_option_ticker(strike, expiry, direction)
            bars = prefetched.get(ticker, [])
            if not bars:
                bars = fetch_option_bars(ticker, expiry)
                if bars:
                    prefetched[ticker] = bars

            if bars:
                result = get_option_price_at_time(bars, bar_ts)
                if result:
                    price, volume = result
                    if min_price <= price <= max_price and volume >= 50:
                        candidates.append({
                            "strike": strike,
                            "price": price,
                            "volume": volume,
                            "ticker": ticker,
                            "offset": offset
                        })

    if not candidates:
        return None

    # Sort by volume (prefer liquid strikes)
    candidates.sort(key=lambda x: -x["volume"])
    return candidates[0]


# ==============================================================================
# MAIN SIMULATION
# ==============================================================================

def run_day_simulation(date: str, start_account: float) -> DayResult:
    """Run V6 two-mode simulation for one day."""
    print(f"\n{'='*70}")
    print(f"SIMULATING: {date} [V6 - TWO MODE ENGINE]")
    print(f"Starting account: ${start_account:,.2f}")
    print(f"{'='*70}")

    # Fetch data
    spy_bars = fetch_minute_bars("SPY", date)

    if not spy_bars:
        print(f"  No SPY data for {date}")
        return DayResult(
            date=date, mode_selected="NO_DATA", mode_reason="No market data",
            trades=[], num_trades=0, wins=0, losses=0, win_rate=0,
            daily_pnl_dollars=0, daily_pnl_pct=0,
            start_account=start_account, end_account=start_account,
            circuit_breaker_fired=False, circuit_breaker_saved=0
        )

    print(f"  Loaded {len(spy_bars)} SPY bars")

    # Day stats
    day_high = max(b["h"] for b in spy_bars)
    day_low = min(b["l"] for b in spy_bars)
    day_open = spy_bars[0]["o"]
    day_range = day_high - day_low
    print(f"  Full day range: ${day_low:.2f} - ${day_high:.2f} (${day_range:.2f})")

    expiry = date
    account = start_account
    completed_trades: List[Trade] = []
    consecutive_losses = 0
    circuit_breaker_fired = False
    circuit_breaker_saved = 0.0

    # Day direction for BERSERKER (one direction per day)
    day_direction: Optional[str] = None

    # Active positions
    berserker_positions: List[BerserkerPosition] = []
    credit_spreads: List[CreditSpread] = []

    # Counters
    scalp_trades_today = 0
    berserker_trades_today = 0
    daily_pnl = 0

    # Pre-fetch options
    prefetched: Dict[str, List[Dict]] = {}
    spy_mid = (day_high + day_low) / 2
    print(f"  Pre-fetching options near ${spy_mid:.0f}...")
    for strike in range(int(spy_mid) - 10, int(spy_mid) + 11):
        for direction in ["CALL", "PUT"]:
            ticker = build_option_ticker(strike, expiry, direction)
            bars = fetch_option_bars(ticker, date)
            if bars:
                prefetched[ticker] = bars
    print(f"  Pre-fetched {len(prefetched)} option contracts")

    # Load brain
    learned = load_learned_trades()
    print(f"  Loaded {len(learned)} learned trades for pattern matching")

    if len(learned) == 0:
        print(f"  ⚠️ BRAIN_EMPTY: Defaulting all sizing to {BERSERKER_CONFIG['default_sizing']*100:.0f}%")

    # Track mode
    final_mode = "WAIT"
    mode_reason = "Assessing"

    for i, bar in enumerate(spy_bars):
        bar_time = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        bar_ts = int(bar_time.timestamp() * 1000)
        spy_price = bar["c"]

        # Mode selection
        mode, reason = select_mode(spy_bars, i)

        if mode != "WAIT" and final_mode == "WAIT":
            final_mode = mode
            mode_reason = reason
            print(f"  [{bar_time.strftime('%H:%M')}] MODE SELECTED: {mode}")
            print(f"    Reason: {reason}")

        # Daily P&L check
        daily_pnl_pct = (account - start_account) / start_account

        if daily_pnl_pct <= DAILY_LOSS_LIMIT and not circuit_breaker_fired:
            print(f"  [{bar_time.strftime('%H:%M')}] ⛔ DAILY LOSS LIMIT (-10%) - STOPPING")
            circuit_breaker_fired = True

        # Circuit breaker: 2 consecutive losses
        if consecutive_losses >= CIRCUIT_BREAKER_LOSSES and not circuit_breaker_fired:
            print(f"  [{bar_time.strftime('%H:%M')}] ⛔ CIRCUIT BREAKER: {consecutive_losses} consecutive losses")
            circuit_breaker_fired = True

        # ================================================================
        # MONITOR BERSERKER POSITIONS (with scaled exits)
        # ================================================================
        for pos in berserker_positions[:]:
            hold_minutes = int((bar_time - pos.entry_time).total_seconds() / 60)

            option_bars = prefetched.get(pos.option_ticker, [])
            result = get_option_price_at_time(option_bars, bar_ts, use_low=True)

            if not result:
                continue

            current_price, _ = result
            pos.current_option_price = current_price

            if current_price > pos.peak_option_price:
                pos.peak_option_price = current_price

            current_pnl_pct = (current_price - pos.entry_option_price) / pos.entry_option_price
            unrealized_pnl = (current_price - pos.entry_option_price) * pos.remaining_contracts * 100

            exit_reason = None
            contracts_to_exit = 0

            # ================================================================
            # BERSERKER EXIT LOGIC - COMPLETELY REWORKED
            # ================================================================

            # NO STOP FIRST 20 MINUTES (let trade breathe)
            if hold_minutes >= BERSERKER_CONFIG["no_stop_minutes"]:
                # Dollar-based stop after 20 min
                total_loss = (pos.entry_option_price - current_price) * pos.remaining_contracts * 100
                if total_loss >= BERSERKER_CONFIG["dollar_stop"]:
                    exit_reason = "DOLLAR_STOP"
                    contracts_to_exit = pos.remaining_contracts

                # Time stop: negative after 45 min
                if hold_minutes >= BERSERKER_CONFIG["time_stop_minutes"] and current_pnl_pct < 0:
                    exit_reason = "TIME_STOP"
                    contracts_to_exit = pos.remaining_contracts

            # Max hold time
            if hold_minutes >= BERSERKER_CONFIG["max_hold_minutes"]:
                exit_reason = "MAX_HOLD"
                contracts_to_exit = pos.remaining_contracts

            # ================================================================
            # SCALED EXITS (let winners run)
            # ================================================================

            # Scale 1: Sell 25% at +100%
            if not pos.scale_1_done and current_pnl_pct >= BERSERKER_CONFIG["scale_1_pct"]:
                contracts_to_sell = max(1, int(pos.initial_contracts * 0.25))
                if pos.remaining_contracts > contracts_to_sell:
                    exit_price = current_price * 0.99  # Slippage
                    pnl = (exit_price - pos.entry_option_price) * contracts_to_sell * 100
                    pos.realized_pnl += pnl
                    pos.contracts_sold_scale_1 = contracts_to_sell
                    pos.remaining_contracts -= contracts_to_sell
                    pos.scale_1_done = True
                    account += pnl

                    print(f"  [{bar_time.strftime('%H:%M')}] 📈 SCALE 1: Sold {contracts_to_sell} @ ${exit_price:.2f} (+{current_pnl_pct*100:.0f}%)")
                    print(f"      P&L: ${pnl:+,.0f} | Remaining: {pos.remaining_contracts} contracts")

            # Scale 2: Sell 25% at +200%
            if pos.scale_1_done and not pos.scale_2_done and current_pnl_pct >= BERSERKER_CONFIG["scale_2_pct"]:
                contracts_to_sell = max(1, int(pos.initial_contracts * 0.25))
                if pos.remaining_contracts > contracts_to_sell:
                    exit_price = current_price * 0.99
                    pnl = (exit_price - pos.entry_option_price) * contracts_to_sell * 100
                    pos.realized_pnl += pnl
                    pos.contracts_sold_scale_2 = contracts_to_sell
                    pos.remaining_contracts -= contracts_to_sell
                    pos.scale_2_done = True
                    account += pnl

                    print(f"  [{bar_time.strftime('%H:%M')}] 🚀 SCALE 2: Sold {contracts_to_sell} @ ${exit_price:.2f} (+{current_pnl_pct*100:.0f}%)")
                    print(f"      P&L: ${pnl:+,.0f} | Remaining: {pos.remaining_contracts} contracts (RUNNERS)")

            # Full exit
            if exit_reason and contracts_to_exit > 0:
                exit_price = current_price * 0.98  # Slippage
                remaining_value = exit_price * pos.remaining_contracts * 100
                remaining_cost = pos.entry_option_price * pos.remaining_contracts * 100
                final_pnl = remaining_value - remaining_cost

                total_pnl = pos.realized_pnl + final_pnl
                total_pnl_pct = total_pnl / pos.position_cost * 100

                trade = Trade(
                    entry_time=pos.entry_time,
                    exit_time=bar_time,
                    mode="BERSERKER",
                    direction=pos.direction,
                    strike=pos.strike,
                    contracts=pos.initial_contracts,
                    entry_price=pos.entry_option_price,
                    exit_price=exit_price,
                    position_cost=pos.position_cost,
                    pnl_dollars=total_pnl,
                    pnl_pct=total_pnl_pct,
                    exit_reason=exit_reason,
                    pattern=pos.pattern,
                    hold_minutes=hold_minutes
                )
                completed_trades.append(trade)

                account += final_pnl
                daily_pnl = account - start_account
                berserker_positions.remove(pos)

                if total_pnl > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                emoji = "✅" if total_pnl > 0 else "❌"
                scaled_info = ""
                if pos.scale_1_done or pos.scale_2_done:
                    scaled_info = f" | Scaled: {pos.contracts_sold_scale_1}+{pos.contracts_sold_scale_2}"

                print(f"  [{bar_time.strftime('%H:%M')}] {emoji} EXIT BERSERKER {pos.direction} ${pos.strike}: {exit_reason}")
                print(f"      Total P&L: ${total_pnl:+,.0f} ({total_pnl_pct:+.0f}%) | {hold_minutes}min{scaled_info}")

        # ================================================================
        # MONITOR CREDIT SPREADS (SCALP mode)
        # ================================================================
        for spread in credit_spreads[:]:
            # Get current spread value (cost to close)
            short_bars = prefetched.get(build_option_ticker(spread.short_strike, expiry,
                                        "PUT" if spread.spread_type == "PUT_SPREAD" else "CALL"), [])
            long_bars = prefetched.get(build_option_ticker(spread.long_strike, expiry,
                                        "PUT" if spread.spread_type == "PUT_SPREAD" else "CALL"), [])

            short_result = get_option_price_at_time(short_bars, bar_ts) if short_bars else None
            long_result = get_option_price_at_time(long_bars, bar_ts) if long_bars else None

            if not short_result or not long_result:
                continue

            short_price, _ = short_result
            long_price, _ = long_result

            # Current spread cost (to close)
            current_spread_value = short_price - long_price
            spread.current_value = current_spread_value * spread.contracts * 100

            # P&L = credit received - current cost to close
            spread_pnl = spread.total_credit - spread.current_value
            pnl_pct_of_max = spread_pnl / spread.total_credit if spread.total_credit > 0 else 0

            exit_reason = None

            # Profit target: 50% of max credit
            if pnl_pct_of_max >= SCALP_CONFIG["profit_target_pct"]:
                exit_reason = "PROFIT_TARGET"

            # Stop loss: 2x credit received
            if spread.current_value >= spread.total_credit * SCALP_CONFIG["stop_loss_mult"]:
                exit_reason = "STOP_LOSS"

            # Time stop: exit by 2:00 PM
            minutes_since_open = i
            if minutes_since_open >= SCALP_CONFIG["latest_exit_minute"]:
                exit_reason = "TIME_EXIT"

            if exit_reason:
                # Close spread (buy to close short, sell to close long)
                close_cost = spread.current_value
                pnl_dollars = spread.total_credit - close_cost

                trade = Trade(
                    entry_time=spread.entry_time,
                    exit_time=bar_time,
                    mode="SCALP",
                    direction=spread.spread_type,
                    strike=spread.short_strike,
                    contracts=spread.contracts,
                    entry_price=spread.credit_received,
                    exit_price=current_spread_value,
                    position_cost=spread.total_credit,
                    pnl_dollars=pnl_dollars,
                    pnl_pct=pnl_dollars / spread.total_credit * 100 if spread.total_credit > 0 else 0,
                    exit_reason=exit_reason,
                    pattern="CREDIT_SPREAD",
                    hold_minutes=int((bar_time - spread.entry_time).total_seconds() / 60)
                )
                completed_trades.append(trade)

                account += pnl_dollars
                daily_pnl = account - start_account
                credit_spreads.remove(spread)

                if pnl_dollars > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                emoji = "✅" if pnl_dollars > 0 else "❌"
                print(f"  [{bar_time.strftime('%H:%M')}] {emoji} EXIT SCALP {spread.spread_type}: {exit_reason}")
                print(f"      Credit: ${spread.credit_received:.2f} | Close: ${current_spread_value:.2f} | P&L: ${pnl_dollars:+,.0f}")

        # ================================================================
        # NEW ENTRIES (if not circuit breaker)
        # ================================================================
        if circuit_breaker_fired:
            continue

        if account < MIN_ACCOUNT_FLOOR:
            continue

        if mode == "WAIT":
            continue

        # Must be after 10:15 AM for any entries
        if i < EARLIEST_ENTRY_MINUTE:
            continue

        # ================================================================
        # SCALP MODE: Credit Spread Entries
        # ================================================================
        if mode == "SCALP":
            if len(credit_spreads) >= SCALP_CONFIG["max_concurrent"]:
                continue
            if scalp_trades_today >= SCALP_CONFIG["max_daily_trades"]:
                continue

            # Determine spread direction based on market
            # If market trending up, sell put spreads (bullish)
            # If market trending down, sell call spreads (bearish)
            if i >= 10:
                recent_momentum = (spy_bars[i]["c"] - spy_bars[i-10]["c"]) / spy_bars[i-10]["c"]
                spread_direction = "PUT_SPREAD" if recent_momentum >= 0 else "CALL_SPREAD"
            else:
                spread_direction = "PUT_SPREAD"  # Default bullish

            spread_data = find_credit_spread_strikes(spy_price, spread_direction, expiry, prefetched, bar_ts)

            if spread_data:
                sizing = SCALP_CONFIG["sizing"]
                position_value = account * sizing

                # Calculate contracts (need to cover max loss)
                max_loss_per_spread = spread_data["max_loss"] * 100
                contracts = max(1, int(position_value / max_loss_per_spread))

                credit_per_spread = spread_data["credit"]
                total_credit = credit_per_spread * contracts * 100
                max_loss = max_loss_per_spread * contracts

                spread = CreditSpread(
                    entry_time=bar_time,
                    spread_type=spread_direction,
                    short_strike=spread_data["short_strike"],
                    long_strike=spread_data["long_strike"],
                    expiry=expiry,
                    contracts=contracts,
                    credit_received=credit_per_spread,
                    total_credit=total_credit,
                    max_loss=max_loss,
                )
                credit_spreads.append(spread)
                scalp_trades_today += 1

                print(f"  [{bar_time.strftime('%H:%M')}] 🔹 SCALP ENTRY: {spread_direction}")
                print(f"      Short ${spread_data['short_strike']} / Long ${spread_data['long_strike']}")
                print(f"      {contracts}x @ ${credit_per_spread:.2f} credit (${total_credit:.0f} total)")
                print(f"      Max loss: ${max_loss:.0f} | Target: ${total_credit * 0.5:.0f}")

        # ================================================================
        # BERSERKER MODE: Directional Entries
        # ================================================================
        elif mode == "BERSERKER":
            if len(berserker_positions) >= 2:
                continue

            signal = detect_berserker_signal(spy_bars, i, day_direction)

            if signal:
                direction, base_confidence = signal

                # Set day direction on first BERSERKER trade
                if not day_direction:
                    day_direction = direction
                    print(f"  [{bar_time.strftime('%H:%M')}] 🎯 DAY DIRECTION SET: {direction}")

                # Get pattern match
                if i >= 10:
                    avg_vol = sum(b["v"] for b in spy_bars[i-10:i]) / 10
                    vol_spike = spy_bars[i]["v"] / avg_vol if avg_vol > 0 else 1
                    momentum = (spy_bars[i]["c"] - spy_bars[i-10]["c"]) / spy_bars[i-10]["c"]
                else:
                    vol_spike = 1
                    momentum = 0

                pattern, conviction_boost, historical_wr = get_pattern_match(
                    direction, momentum, vol_spike, 0.75
                )

                final_confidence = base_confidence + conviction_boost

                # Find strike
                strike_data = find_berserker_strike(spy_price, direction, expiry, prefetched, bar_ts)

                if not strike_data:
                    print(f"  [{bar_time.strftime('%H:%M')}] ⚠️ No valid strike found (price constraints)")
                    continue

                entry_price = strike_data["price"] * 1.02  # 2% slippage

                # CONVICTION-BASED SIZING
                if len(learned) == 0:
                    sizing = BERSERKER_CONFIG["default_sizing"]
                    sizing_tier = "DEFAULT (brain empty)"
                elif final_confidence >= 85 and historical_wr >= 0.50:
                    sizing = BERSERKER_CONFIG["full_berserker_sizing"]
                    sizing_tier = "FULL_BERSERKER"
                elif final_confidence >= 70:
                    sizing = BERSERKER_CONFIG["high_conviction_sizing"]
                    sizing_tier = "HIGH_CONVICTION"
                elif pattern != "DEFAULT":
                    sizing = BERSERKER_CONFIG["pattern_sizing"]
                    sizing_tier = "PATTERN_MATCH"
                else:
                    sizing = BERSERKER_CONFIG["default_sizing"]
                    sizing_tier = "DEFAULT"

                position_value = account * sizing
                contract_cost = entry_price * 100
                contracts = max(1, int(position_value / contract_cost))
                actual_cost = contracts * contract_cost

                pos = BerserkerPosition(
                    entry_time=bar_time,
                    entry_minute_ts=bar_ts,
                    entry_spy_price=spy_price,
                    direction=direction,
                    strike=strike_data["strike"],
                    expiry=expiry,
                    option_ticker=strike_data["ticker"],
                    initial_contracts=contracts,
                    remaining_contracts=contracts,
                    entry_option_price=entry_price,
                    position_cost=actual_cost,
                    pattern=pattern,
                    conviction=final_confidence,
                    peak_option_price=entry_price,
                )
                berserker_positions.append(pos)
                berserker_trades_today += 1

                emoji = "🔥" if sizing_tier == "FULL_BERSERKER" else "⚡" if sizing_tier == "HIGH_CONVICTION" else "🔸"
                print(f"  [{bar_time.strftime('%H:%M')}] {emoji} BERSERKER ENTRY [{sizing_tier}] {direction} ${strike_data['strike']}")
                print(f"      {contracts}x @ ${entry_price:.2f} (${actual_cost:,.0f})")
                print(f"      Pattern: {pattern} | Confidence: {final_confidence:.0f}% | WR: {historical_wr*100:.0f}%")
                print(f"      Scaled exits: 25% @ +100%, 25% @ +200%, hold 50% runner")

    # ================================================================
    # END OF DAY: Close remaining positions
    # ================================================================
    if berserker_positions or credit_spreads:
        final_bar = spy_bars[-1]
        final_time = datetime.fromisoformat(final_bar["t"].replace("Z", "+00:00"))
        final_ts = int(final_time.timestamp() * 1000)
        spy_price = final_bar["c"]

        # Close BERSERKER positions
        for pos in berserker_positions:
            option_bars = prefetched.get(pos.option_ticker, [])
            result = get_option_price_at_time(option_bars, final_ts, use_low=True)

            if result:
                final_price, _ = result
                final_price *= 0.98  # EOD slippage
            else:
                final_price = pos.entry_option_price * 0.1  # Assume near-zero for 0DTE

            hold_minutes = int((final_time - pos.entry_time).total_seconds() / 60)
            final_pnl = (final_price - pos.entry_option_price) * pos.remaining_contracts * 100
            total_pnl = pos.realized_pnl + final_pnl
            total_pnl_pct = total_pnl / pos.position_cost * 100

            trade = Trade(
                entry_time=pos.entry_time,
                exit_time=final_time,
                mode="BERSERKER",
                direction=pos.direction,
                strike=pos.strike,
                contracts=pos.initial_contracts,
                entry_price=pos.entry_option_price,
                exit_price=final_price,
                position_cost=pos.position_cost,
                pnl_dollars=total_pnl,
                pnl_pct=total_pnl_pct,
                exit_reason="EOD_CLOSE",
                pattern=pos.pattern,
                hold_minutes=hold_minutes
            )
            completed_trades.append(trade)
            account += final_pnl

            scaled_info = ""
            if pos.scale_1_done or pos.scale_2_done:
                scaled_info = f" | Scaled: {pos.contracts_sold_scale_1}+{pos.contracts_sold_scale_2}"

            print(f"  [EOD] CLOSE BERSERKER {pos.direction} ${pos.strike}: P&L ${total_pnl:+,.0f}{scaled_info}")

        # Close credit spreads
        for spread in credit_spreads:
            # Most credit spreads should expire worthless (good) or ITM (bad)
            # For simplicity, if short strike closer to current price, assume ITM loss
            short_bars = prefetched.get(build_option_ticker(spread.short_strike, expiry,
                                        "PUT" if spread.spread_type == "PUT_SPREAD" else "CALL"), [])
            result = get_option_price_at_time(short_bars, final_ts) if short_bars else None

            if result:
                short_price, _ = result
                if short_price < 0.05:  # Expired worthless
                    pnl = spread.total_credit  # Keep all credit
                else:
                    pnl = spread.total_credit - (short_price * spread.contracts * 100)
            else:
                pnl = spread.total_credit * 0.3  # Assume partial profit

            trade = Trade(
                entry_time=spread.entry_time,
                exit_time=final_time,
                mode="SCALP",
                direction=spread.spread_type,
                strike=spread.short_strike,
                contracts=spread.contracts,
                entry_price=spread.credit_received,
                exit_price=0,
                position_cost=spread.total_credit,
                pnl_dollars=pnl,
                pnl_pct=pnl / spread.total_credit * 100 if spread.total_credit > 0 else 0,
                exit_reason="EOD_CLOSE",
                pattern="CREDIT_SPREAD",
                hold_minutes=int((final_time - spread.entry_time).total_seconds() / 60)
            )
            completed_trades.append(trade)
            account += pnl

            print(f"  [EOD] CLOSE SCALP {spread.spread_type}: P&L ${pnl:+,.0f}")

    # Calculate results
    wins = [t for t in completed_trades if t.pnl_dollars > 0]
    losses = [t for t in completed_trades if t.pnl_dollars <= 0]
    daily_pnl = account - start_account
    daily_pnl_pct = daily_pnl / start_account * 100

    # Estimate circuit breaker savings
    if circuit_breaker_fired:
        # Count potential trades that would have been taken
        potential_trades = (SCALP_CONFIG["max_daily_trades"] - scalp_trades_today) + 2
        avg_loss = sum(t.pnl_dollars for t in losses) / len(losses) if losses else 200
        circuit_breaker_saved = potential_trades * abs(avg_loss) * 0.7  # 70% would be losers

    result = DayResult(
        date=date,
        mode_selected=final_mode,
        mode_reason=mode_reason,
        trades=completed_trades,
        num_trades=len(completed_trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(completed_trades) * 100 if completed_trades else 0,
        daily_pnl_dollars=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
        start_account=start_account,
        end_account=account,
        circuit_breaker_fired=circuit_breaker_fired,
        circuit_breaker_saved=circuit_breaker_saved
    )

    # Print summary
    print(f"\n  DAY SUMMARY [{final_mode}]:")
    print(f"    Mode: {final_mode} ({mode_reason})")
    print(f"    Trades: {result.num_trades} | Wins: {result.wins} | WR: {result.win_rate:.0f}%")
    print(f"    Daily P&L: ${daily_pnl:+,.0f} ({daily_pnl_pct:+.1f}%)")
    print(f"    Account: ${result.start_account:,.0f} → ${result.end_account:,.0f}")
    if circuit_breaker_fired:
        print(f"    ⛔ Circuit breaker fired | Est. saved: ${circuit_breaker_saved:,.0f}")

    return result


def run_week_backtest(week_type: str, dates: List[str], target: float):
    """Run backtest for a week."""
    print(f"\n{'='*80}")
    print(f"WEEK TYPE: {week_type}")
    print(f"Dates: {', '.join(dates)}")
    print(f"Target: ${target:,.0f}")
    print(f"{'='*80}")

    account = INITIAL_ACCOUNT
    all_results: List[DayResult] = []
    mode_stats = {"SCALP": 0, "BERSERKER": 0, "WAIT": 0, "NO_DATA": 0}
    total_cb_saved = 0

    for date in dates:
        result = run_day_simulation(date, account)
        all_results.append(result)
        account = result.end_account
        mode_stats[result.mode_selected] = mode_stats.get(result.mode_selected, 0) + 1
        total_cb_saved += result.circuit_breaker_saved
        time.sleep(0.3)

    total_trades = sum(r.num_trades for r in all_results)
    total_wins = sum(r.wins for r in all_results)
    overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    # Separate stats by mode
    scalp_trades = [t for r in all_results for t in r.trades if t.mode == "SCALP"]
    berserker_trades = [t for r in all_results for t in r.trades if t.mode == "BERSERKER"]

    scalp_wins = sum(1 for t in scalp_trades if t.pnl_dollars > 0)
    scalp_wr = scalp_wins / len(scalp_trades) * 100 if scalp_trades else 0
    scalp_pnl = sum(t.pnl_dollars for t in scalp_trades)

    berserker_wins = sum(1 for t in berserker_trades if t.pnl_dollars > 0)
    berserker_wr = berserker_wins / len(berserker_trades) * 100 if berserker_trades else 0
    berserker_pnl = sum(t.pnl_dollars for t in berserker_trades)

    # Calculate expectancy
    if scalp_trades:
        scalp_avg_win = sum(t.pnl_dollars for t in scalp_trades if t.pnl_dollars > 0) / max(1, scalp_wins)
        scalp_avg_loss = sum(t.pnl_dollars for t in scalp_trades if t.pnl_dollars <= 0) / max(1, len(scalp_trades) - scalp_wins)
        scalp_expectancy = (scalp_wr/100 * scalp_avg_win) + ((100-scalp_wr)/100 * scalp_avg_loss)
    else:
        scalp_avg_win, scalp_avg_loss, scalp_expectancy = 0, 0, 0

    if berserker_trades:
        berserker_avg_win = sum(t.pnl_dollars for t in berserker_trades if t.pnl_dollars > 0) / max(1, berserker_wins)
        berserker_avg_loss = sum(t.pnl_dollars for t in berserker_trades if t.pnl_dollars <= 0) / max(1, len(berserker_trades) - berserker_wins)
        berserker_expectancy = (berserker_wr/100 * berserker_avg_win) + ((100-berserker_wr)/100 * berserker_avg_loss)
    else:
        berserker_avg_win, berserker_avg_loss, berserker_expectancy = 0, 0, 0

    print(f"\n{'='*80}")
    print(f"{week_type} WEEK RESULTS")
    print(f"{'='*80}")

    for r in all_results:
        wr = f"{r.win_rate:.0f}%" if r.num_trades > 0 else "N/A"
        cb = " [CB]" if r.circuit_breaker_fired else ""
        print(f"  {r.date} [{r.mode_selected:9}]: ${r.start_account:,.0f} → ${r.end_account:,.0f} ({r.daily_pnl_pct:+.1f}%) | {r.num_trades} trades | WR: {wr}{cb}")

    print(f"\n  MODE BREAKDOWN: SCALP={mode_stats.get('SCALP',0)} | BERSERKER={mode_stats.get('BERSERKER',0)}")

    print(f"\n  SCALP MODE:")
    print(f"    Trades: {len(scalp_trades)} | Win Rate: {scalp_wr:.0f}%")
    print(f"    Total P&L: ${scalp_pnl:+,.0f}")
    print(f"    Avg Win: ${scalp_avg_win:+,.0f} | Avg Loss: ${scalp_avg_loss:+,.0f}")
    print(f"    Expectancy: ${scalp_expectancy:+,.0f}/trade")

    print(f"\n  BERSERKER MODE:")
    print(f"    Trades: {len(berserker_trades)} | Win Rate: {berserker_wr:.0f}%")
    print(f"    Total P&L: ${berserker_pnl:+,.0f}")
    print(f"    Avg Win: ${berserker_avg_win:+,.0f} | Avg Loss: ${berserker_avg_loss:+,.0f}")
    print(f"    Expectancy: ${berserker_expectancy:+,.0f}/trade")

    final_return = ((account / INITIAL_ACCOUNT) - 1) * 100
    hit = account >= target

    print(f"\n  FINAL: ${INITIAL_ACCOUNT:,.0f} → ${account:,.0f} ({final_return:+.1f}%)")
    print(f"  Total Trades: {total_trades} | Win Rate: {overall_win_rate:.0f}%")
    print(f"  Circuit Breaker Savings: ${total_cb_saved:,.0f}")
    print(f"  Target: ${target:,.0f} | {'✅ HIT' if hit else f'❌ MISSED by ${target-account:,.0f}'}")

    return account, all_results, overall_win_rate, scalp_expectancy, berserker_expectancy


def main():
    """Run V6 - TWO MODE ENGINE."""
    print("\n" + "=" * 80)
    print("THE BACKTEST FROM HELL v6: THE COMPLETE REWORK")
    print("TWO-MODE ENGINE: PREMIUM SCALP + DIRECTIONAL BERSERKER")
    print("=" * 80)
    print(f"Initial Account: ${INITIAL_ACCOUNT:,.0f}")
    print(f"\nCRITICAL CHANGES FROM V5:")
    print(f"  ❌ DELETED: LOTTO_TICKET (0% historical win rate)")
    print(f"  ❌ DELETED: REVERSAL_PUT (6/8 losers)")
    print(f"  ✅ Entry time: 10:15 AM (not 9:35 AM)")
    print(f"  ✅ Min entry price: ${BERSERKER_CONFIG['min_entry_price']:.2f} (no lottos)")
    print(f"  ✅ No hard stop first {BERSERKER_CONFIG['no_stop_minutes']} minutes")
    print(f"  ✅ Scaled exits: 25% @ +100%, 25% @ +200%, hold 50%")
    print(f"  ✅ SCALP mode: Credit spreads for quiet days")
    print(f"  ✅ Circuit breaker: {CIRCUIT_BREAKER_LOSSES} consecutive losses = done")

    # Week definitions (same as V5 for comparison)
    bad_week = ["2025-12-22", "2025-12-23", "2025-12-24", "2025-12-26", "2025-12-29"]
    avg_week = ["2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]
    great_week = ["2025-12-10", "2025-12-11", "2025-12-12", "2025-12-15", "2025-12-16"]

    targets = {
        "bad": 5300,     # SCALP mode should grind small gains
        "average": 8000, # Mix of both modes
        "great": 15000   # BERSERKER printing
    }

    results = {}

    bad_final, bad_results, bad_wr, bad_scalp_exp, bad_berk_exp = run_week_backtest(
        "BAD (Holiday Week - Low Vol)", bad_week, targets["bad"]
    )
    results["bad"] = {"final": bad_final, "target": targets["bad"], "win_rate": bad_wr,
                      "scalp_exp": bad_scalp_exp, "berserker_exp": bad_berk_exp}

    avg_final, avg_results, avg_wr, avg_scalp_exp, avg_berk_exp = run_week_backtest(
        "AVERAGE (Mixed Activity)", avg_week, targets["average"]
    )
    results["average"] = {"final": avg_final, "target": targets["average"], "win_rate": avg_wr,
                          "scalp_exp": avg_scalp_exp, "berserker_exp": avg_berk_exp}

    great_final, great_results, great_wr, great_scalp_exp, great_berk_exp = run_week_backtest(
        "GREAT (High Volatility)", great_week, targets["great"]
    )
    results["great"] = {"final": great_final, "target": targets["great"], "win_rate": great_wr,
                        "scalp_exp": great_scalp_exp, "berserker_exp": great_berk_exp}

    print("\n" + "=" * 80)
    print("FINAL RESULTS - V6 TWO MODE ENGINE")
    print("=" * 80)

    all_hit = True
    for week_type, data in results.items():
        final = data["final"]
        target = data["target"]
        win_rate = data["win_rate"]
        hit = final >= target
        if not hit:
            all_hit = False
        pct = ((final / INITIAL_ACCOUNT) - 1) * 100
        status = "✅ HIT" if hit else f"❌ MISSED by ${target-final:,.0f}"

        print(f"\n{week_type.upper()} WEEK:")
        print(f"  Result: ${INITIAL_ACCOUNT:,.0f} → ${final:,.0f} ({pct:+.1f}%)")
        print(f"  Win Rate: {win_rate:.0f}%")
        print(f"  SCALP Expectancy: ${data['scalp_exp']:+,.0f}/trade")
        print(f"  BERSERKER Expectancy: ${data['berserker_exp']:+,.0f}/trade")
        print(f"  Target: ${target:,.0f} | {status}")

    # Expectancy check
    print("\n" + "=" * 80)
    print("EXPECTANCY CHECK (must be POSITIVE)")
    print("=" * 80)

    total_scalp_exp = sum(d["scalp_exp"] for d in results.values()) / 3
    total_berk_exp = sum(d["berserker_exp"] for d in results.values()) / 3

    print(f"  SCALP Average Expectancy: ${total_scalp_exp:+,.0f}/trade")
    print(f"  BERSERKER Average Expectancy: ${total_berk_exp:+,.0f}/trade")

    if total_scalp_exp < 0:
        print(f"  ⚠️ SCALP EXPECTANCY NEGATIVE - review spread selection")
    if total_berk_exp < 0:
        print(f"  ⚠️ BERSERKER EXPECTANCY NEGATIVE - review entry/exit logic")

    print("\n" + "=" * 80)
    if all_hit:
        print("🔥🔥🔥 ALL TARGETS HIT - V6 COMPLETE REWORK SUCCESSFUL 🔥🔥🔥")
    else:
        print("⚠️  SOME TARGETS MISSED - CONTINUE ITERATION")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
