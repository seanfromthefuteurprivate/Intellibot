#!/usr/bin/env python3
"""
BACKTEST FROM HELL v5: FINAL BOSS MODE
THREE TRADING MODES - ADAPTIVE TO MARKET CONDITIONS

MODE 1: DEAD DAY - Range < $1.00 by bar 30 → NO TRADES
MODE 2: SCALP MODE - Range $1.00-$2.00 → ATM options, tight stops, quick profits
MODE 3: BERSERKER MODE - Range > $2.00 → Full brain with pattern matching

DYNAMIC MODE SWITCHING every 30 minutes based on current price action.
"""
import os
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

# API Keys
ALPACA_KEY = os.environ.get("ALPACA_API_KEY", "PKWT6T5BFKHBTFDW3CPAFW2XBZ")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "pVdzbVte2pQvL1RmCTFw3oaQ6TBWYimAzC42DUyTEy8")
POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "QJWtaUQV7N8mytTI7PH26lX3Ju6PD2iq")
DB_PATH = os.environ.get("WSB_SNAKE_DB", "wsb_snake_data/wsb_snake.db")

# Base parameters
INITIAL_ACCOUNT = 5000.0
MIN_ACCOUNT_FLOOR = 500.0

# MODE THRESHOLDS
DEAD_THRESHOLD = 1.00      # Range < $1 = DEAD day
SCALP_THRESHOLD = 2.00     # Range $1-$2 = SCALP mode
# Range > $2 = BERSERKER mode

# SCALP MODE PARAMETERS
SCALP_CONFIG = {
    "sizing": 0.10,            # 10% per trade
    "max_concurrent": 1,       # 1 position at a time
    "max_daily_trades": 3,     # Max 3 trades per day
    "profit_target": 0.25,     # +25% = take profit
    "stop_loss": -0.15,        # -15% = cut fast
    "time_stop_minutes": 10,   # Exit if not +10% in 10 min
    "time_stop_min_gain": 0.10,# Need +10% to stay past time stop
    "entry_momentum": 0.0015,  # 0.15% momentum in 5 bars
    "otm_offset": 0,           # ATM options only
}

# SCALP TIME WINDOWS (market hours)
SCALP_WINDOWS = [
    (9, 35, 11, 0),    # 9:35 AM - 11:00 AM
    (14, 0, 15, 30),   # 2:00 PM - 3:30 PM
]

# BERSERKER MODE PARAMETERS (from v4 with corrected sizing)
PATTERN_CONFIG = {
    "LOTTO_TICKET": {
        "otm_range": (3, 5),
        "price_range": (0.10, 0.50),
        "max_sizing": 0.12,        # 12% - small bets, many shots
        "stop_loss": -0.75,        # Wide stop - binary outcomes
        "trail_trigger": 3.00,
        "trail_pct": 0.30,
        "max_hold_minutes": 90,
    },
    "REVERSAL_PUT": {
        "otm_range": (1, 2),
        "price_range": (0.30, 1.50),
        "max_sizing": 0.35,        # 35% - BERSERKER trades
        "stop_loss": -0.35,
        "trail_trigger": 1.00,
        "trail_pct": 0.15,
        "max_hold_minutes": 45,
    },
    "MOMENTUM_CALL": {
        "otm_range": (1, 3),
        "price_range": (0.30, 1.50),
        "max_sizing": 0.30,        # 30%
        "stop_loss": -0.40,
        "trail_trigger": 0.75,
        "trail_pct": 0.20,
        "max_hold_minutes": 60,
    },
    "PRECIOUS_METALS_MOMENTUM": {
        "otm_range": (1, 2),
        "price_range": (0.30, 1.50),
        "max_sizing": 0.25,
        "stop_loss": -0.40,
        "trail_trigger": 0.80,
        "trail_pct": 0.18,
        "max_hold_minutes": 50,
    },
    "DEFAULT": {
        "otm_range": (2, 3),
        "price_range": (0.20, 1.00),
        "max_sizing": 0.15,        # 15% conservative
        "stop_loss": -0.40,
        "trail_trigger": 1.00,
        "trail_pct": 0.20,
        "max_hold_minutes": 60,
    }
}

# DAILY P&L PROTECTION
QUIET_DAY_STOP_PROFIT = 500      # Stop trading on quiet day at +$500
VOLATILE_DAY_REDUCE_PCT = 0.50   # After +50% daily, reduce sizing to 10%
DAILY_LOSS_LIMIT = -0.10         # -10% daily loss = stop trading

# Conviction tiers for BERSERKER mode
CONVICTION_TIERS = {
    "BERSERKER": {"min": 85, "sizing_mult": 1.0},
    "LARGE": {"min": 70, "sizing_mult": 0.80},
    "MEDIUM": {"min": 50, "sizing_mult": 0.60},
    "SMALL": {"min": 30, "sizing_mult": 0.40},
    "MINIMUM": {"min": 0, "sizing_mult": 0.25},
}

# Caches
OPTION_BARS_CACHE: Dict[str, List[Dict]] = {}
LEARNED_TRADES_CACHE: List[Dict] = []


@dataclass
class Position:
    entry_time: datetime
    entry_minute_ts: int
    entry_spy_price: float
    direction: str
    strike: float
    expiry: str
    option_ticker: str
    contracts: int
    entry_option_price: float
    position_cost: float
    mode: str  # "SCALP" or "BERSERKER"
    pattern: str
    conviction: float
    stop_loss_pct: float
    profit_target_pct: float
    trail_trigger: float
    trail_pct: float
    max_hold_minutes: int
    peak_option_price: float = 0.0


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    strike: float
    contracts: int
    entry_option_price: float
    exit_option_price: float
    position_cost: float
    pnl_dollars: float
    pnl_pct: float
    exit_reason: str
    spy_entry: float
    spy_exit: float
    mode: str
    pattern: str
    hold_minutes: int


@dataclass
class DayResult:
    date: str
    mode_history: List[str]
    primary_mode: str
    trades: List[Trade]
    num_trades: int
    wins: int
    losses: int
    win_rate: float
    end_account: float
    start_account: float
    daily_pnl_dollars: float
    daily_pnl_pct: float


# ============================================================================
# MODE DETECTION
# ============================================================================

def get_current_mode(spy_bars: List[Dict], current_idx: int) -> str:
    """Determine trading mode based on current day's price action."""
    if current_idx < 30:
        return "WAIT"  # First 30 minutes = observe only

    bars_so_far = spy_bars[:current_idx+1]
    day_high = max(b["h"] for b in bars_so_far)
    day_low = min(b["l"] for b in bars_so_far)
    current_range = day_high - day_low

    if current_range < DEAD_THRESHOLD:
        return "DEAD"
    elif current_range < SCALP_THRESHOLD:
        return "SCALP"
    else:
        return "BERSERKER"


def is_in_scalp_window(bar_time: datetime) -> bool:
    """Check if current time is in a valid scalp window."""
    hour = bar_time.hour
    minute = bar_time.minute

    for start_h, start_m, end_h, end_m in SCALP_WINDOWS:
        start_mins = start_h * 60 + start_m
        end_mins = end_h * 60 + end_m
        current_mins = hour * 60 + minute

        if start_mins <= current_mins <= end_mins:
            return True
    return False


# ============================================================================
# DATA FETCHING
# ============================================================================

def build_option_ticker(strike: float, expiry: str, direction: str) -> str:
    expiry_fmt = datetime.strptime(expiry, "%Y-%m-%d").strftime("%y%m%d")
    cp = "C" if direction == "CALL" else "P"
    strike_fmt = f"{int(strike * 1000):08d}"
    return f"O:SPY{expiry_fmt}{cp}{strike_fmt}"


def fetch_option_bars(option_ticker: str, date: str) -> List[Dict]:
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


def get_option_price_at_time(option_bars: List[Dict], target_ts: int, use_low: bool = False) -> Optional[Tuple[float, int]]:
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

    if best_bar and best_diff <= 120000:
        price = best_bar["l"] if use_low else best_bar["c"]
        volume = best_bar.get("v", 0)
        return (price, volume)
    return None


def fetch_minute_bars(ticker: str, date: str) -> List[Dict]:
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


# ============================================================================
# LEARNED TRADES (for BERSERKER mode)
# ============================================================================

def load_learned_trades():
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
        LEARNED_TRADES_CACHE = []

    return LEARNED_TRADES_CACHE


def calculate_semantic_match(direction: str, time_bucket: str, spy_range_pct: float,
                            volume_spike: float, entry_price: float) -> Tuple[float, str, Dict]:
    """Match against learned trades for pattern detection."""
    learned = load_learned_trades()
    if not learned:
        return (0.0, "DEFAULT", {})

    best_score = 0
    best_pattern = "DEFAULT"
    best_trade = {}

    for trade in learned:
        score = 0
        trade_dir = "CALL" if trade.get("trade_type") == "CALL" else "PUT"
        if trade_dir == direction:
            score += 20

        pattern = trade.get("pattern", "")

        if pattern == "LOTTO_TICKET":
            if entry_price < 0.50:
                score += 25
            if time_bucket in ["opening_drive", "power_hour"]:
                score += 15
        elif pattern == "REVERSAL_PUT":
            if direction == "PUT" and spy_range_pct < -0.3:
                score += 30
            if time_bucket == "power_hour":
                score += 10
        elif pattern == "MOMENTUM_CALL":
            if direction == "CALL" and spy_range_pct > 0.3:
                score += 25
        elif pattern == "PRECIOUS_METALS_MOMENTUM":
            score += 10

        trade_pnl = trade.get("profit_loss_pct", 0) or 0
        if trade_pnl > 100:
            score += 15
        elif trade_pnl > 50:
            score += 10

        if score > best_score:
            best_score = score
            best_pattern = pattern if pattern else "DEFAULT"
            best_trade = trade

    return (best_score, best_pattern, best_trade)


def get_time_bucket(dt: datetime) -> str:
    hour = dt.hour
    minute = dt.minute
    total_minutes = hour * 60 + minute

    if total_minutes < 600:
        return "opening_drive"
    elif total_minutes < 720:
        return "mid_morning"
    elif total_minutes < 840:
        return "midday"
    elif total_minutes < 900:
        return "afternoon"
    else:
        return "power_hour"


# ============================================================================
# SCALP MODE LOGIC
# ============================================================================

def detect_scalp_signal(bars: List[Dict], idx: int) -> Optional[Tuple[str, float]]:
    """Detect scalp entry signal - need momentum in ATM-friendly conditions."""
    if idx < 5:
        return None

    recent = bars[max(0, idx-5):idx+1]
    if len(recent) < 5:
        return None

    current_price = recent[-1]["c"]
    momentum = (recent[-1]["c"] - recent[0]["c"]) / recent[0]["c"]

    # Check volume
    avg_vol = sum(b["v"] for b in recent[:-1]) / (len(recent) - 1) if len(recent) > 1 else 1
    vol_spike = recent[-1]["v"] / avg_vol if avg_vol > 0 else 1

    # Need clear momentum for scalp
    if momentum > SCALP_CONFIG["entry_momentum"] and vol_spike > 1.2:
        return ("CALL", momentum * 1000)
    elif momentum < -SCALP_CONFIG["entry_momentum"] and vol_spike > 1.2:
        return ("PUT", abs(momentum) * 1000)

    return None


def get_atm_option(spy_price: float, direction: str, expiry: str, date: str,
                   prefetched: Dict, bar_ts: int) -> Optional[Dict]:
    """Get ATM option for scalp mode."""
    strike = round(spy_price)  # ATM = nearest dollar
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
            if volume >= 100:  # ATM should have huge volume
                return {
                    "strike": strike,
                    "price": price,
                    "volume": volume,
                    "ticker": ticker
                }
    return None


# ============================================================================
# BERSERKER MODE LOGIC (from v4)
# ============================================================================

def detect_berserker_signal(bars: List[Dict], idx: int, day_open: float) -> Optional[Tuple[str, float, float]]:
    """Signal detection for BERSERKER mode."""
    if idx < 10:
        return None

    recent = bars[max(0, idx-10):idx+1]
    if len(recent) < 10:
        return None

    current_price = recent[-1]["c"]
    momentum = (recent[-1]["c"] - recent[0]["c"]) / recent[0]["c"]

    avg_vol = sum(b["v"] for b in recent[:-1]) / (len(recent) - 1) if len(recent) > 1 else 1
    vol_spike = recent[-1]["v"] / avg_vol if avg_vol > 0 else 1
    bar_range = (recent[-1]["h"] - recent[-1]["l"]) / recent[-1]["c"]

    direction = None
    confidence = 0

    if momentum > 0.002 and vol_spike > 1.3:
        direction = "CALL"
        confidence = min(90, 50 + momentum * 2000 + vol_spike * 10 + bar_range * 500)
        strike = round(current_price) + 2

    elif momentum < -0.002 and vol_spike > 1.3:
        direction = "PUT"
        confidence = min(90, 50 + abs(momentum) * 2000 + vol_spike * 10 + bar_range * 500)
        strike = round(current_price) - 2

    if direction and confidence >= 50:
        return (direction, confidence, strike)

    return None


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
                if volume >= 50 and price_min <= price <= price_max:
                    candidates.append({
                        "strike": strike,
                        "price": price,
                        "volume": volume,
                        "ticker": ticker,
                        "offset": offset
                    })

    if not candidates:
        # Fallback
        for offset in range(1, 6):
            if direction == "CALL":
                strike = round(spy_price) + offset
            else:
                strike = round(spy_price) - offset

            ticker = build_option_ticker(strike, expiry, direction)
            bars = prefetched.get(ticker, [])
            if not bars:
                bars = fetch_option_bars(ticker, date)
                if bars:
                    prefetched[ticker] = bars

            if bars:
                result = get_option_price_at_time(bars, bar_ts)
                if result:
                    price, volume = result
                    if volume >= 30:
                        candidates.append({
                            "strike": strike,
                            "price": price,
                            "volume": volume,
                            "ticker": ticker,
                            "offset": offset
                        })

    if not candidates:
        return None

    if pattern == "LOTTO_TICKET":
        candidates.sort(key=lambda x: x["price"])
    else:
        candidates.sort(key=lambda x: -x["volume"])

    return candidates[0]


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_day_simulation(date: str, start_account: float) -> DayResult:
    """Run V5 multi-mode simulation for one day."""
    print(f"\n{'='*70}")
    print(f"SIMULATING: {date} [V5 - FINAL BOSS MODE]")
    print(f"Starting account: ${start_account:,.2f}")
    print(f"{'='*70}")

    # Fetch SPY and QQQ bars
    spy_bars = fetch_minute_bars("SPY", date)
    qqq_bars = fetch_minute_bars("QQQ", date)

    if not spy_bars:
        print(f"  No SPY data for {date}")
        return DayResult(
            date=date, mode_history=[], primary_mode="NO_DATA", trades=[],
            num_trades=0, wins=0, losses=0, win_rate=0,
            end_account=start_account, start_account=start_account,
            daily_pnl_dollars=0, daily_pnl_pct=0
        )

    print(f"  Loaded {len(spy_bars)} SPY bars")

    day_high = max(b["h"] for b in spy_bars)
    day_low = min(b["l"] for b in spy_bars)
    day_open = spy_bars[0]["o"]
    day_range = day_high - day_low
    print(f"  Full day range: ${day_low:.2f} - ${day_high:.2f} (${day_range:.2f})")

    expiry = date
    account = start_account
    positions: List[Position] = []
    completed_trades: List[Trade] = []
    mode_history: List[str] = []
    consecutive_losses = 0
    scalp_trades_today = 0
    daily_pnl = 0
    daily_stopped = False
    protect_mode = False  # After +50% daily gain

    # Pre-fetch options
    prefetched: Dict[str, List[Dict]] = {}
    spy_mid = (day_high + day_low) / 2
    print(f"  Pre-fetching options near ${spy_mid:.0f}...")
    for strike in range(int(spy_mid) - 8, int(spy_mid) + 9):
        for direction in ["CALL", "PUT"]:
            ticker = build_option_ticker(strike, expiry, direction)
            bars = fetch_option_bars(ticker, date)
            if bars:
                prefetched[ticker] = bars
    print(f"  Pre-fetched {len(prefetched)} option contracts")

    # Load learned trades for BERSERKER mode
    learned = load_learned_trades()
    print(f"  Loaded {len(learned)} learned trades for pattern matching")

    for i, bar in enumerate(spy_bars):
        bar_time = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        bar_ts = int(bar_time.timestamp() * 1000)
        spy_price = bar["c"]

        # Get current mode
        mode = get_current_mode(spy_bars, i)

        # Track mode changes
        if not mode_history or mode_history[-1] != mode:
            mode_history.append(mode)
            if mode != "WAIT":
                print(f"  [{bar_time.strftime('%H:%M')}] MODE: {mode}")

        # Check daily stop conditions
        daily_pnl_pct = (account - start_account) / start_account

        if daily_pnl_pct <= DAILY_LOSS_LIMIT and not daily_stopped:
            print(f"  [{bar_time.strftime('%H:%M')}] ⛔ DAILY LOSS LIMIT (-10%) - STOPPING")
            daily_stopped = True

        if mode == "SCALP" and daily_pnl >= QUIET_DAY_STOP_PROFIT and not daily_stopped:
            print(f"  [{bar_time.strftime('%H:%M')}] 💰 QUIET DAY PROFIT TARGET (+${QUIET_DAY_STOP_PROFIT}) - STOPPING")
            daily_stopped = True

        if daily_pnl_pct >= 0.50 and not protect_mode:
            print(f"  [{bar_time.strftime('%H:%M')}] 🛡️ PROTECT MODE: +50% daily gain, reducing sizing")
            protect_mode = True

        # Monitor existing positions
        for pos in positions[:]:
            hold_minutes = int((bar_time - pos.entry_time).total_seconds() / 60)
            option_bars = prefetched.get(pos.option_ticker, [])
            result = get_option_price_at_time(option_bars, bar_ts, use_low=True)

            if not result:
                continue

            current_price, _ = result

            if current_price > pos.peak_option_price:
                pos.peak_option_price = current_price

            current_pnl_pct = (current_price - pos.entry_option_price) / pos.entry_option_price
            exit_reason = None

            if pos.mode == "SCALP":
                # SCALP exits
                if current_pnl_pct >= pos.profit_target_pct:
                    exit_reason = "PROFIT_TARGET"
                elif current_pnl_pct <= pos.stop_loss_pct:
                    exit_reason = "STOP_LOSS"
                elif hold_minutes >= SCALP_CONFIG["time_stop_minutes"] and current_pnl_pct < SCALP_CONFIG["time_stop_min_gain"]:
                    exit_reason = "TIME_STOP"
            else:
                # BERSERKER exits
                if current_pnl_pct <= pos.stop_loss_pct:
                    exit_reason = "STOP_LOSS"
                elif hold_minutes >= pos.max_hold_minutes:
                    exit_reason = "TIME_LIMIT"
                elif current_pnl_pct >= pos.trail_trigger:
                    drawdown = (pos.peak_option_price - current_price) / pos.peak_option_price
                    if drawdown > pos.trail_pct:
                        exit_reason = "TRAIL_STOP"

            if exit_reason:
                exit_price = current_price * 0.99  # 1% slippage
                position_value = pos.contracts * exit_price * 100
                pnl_dollars = position_value - pos.position_cost
                pnl_pct = (exit_price - pos.entry_option_price) / pos.entry_option_price * 100

                trade = Trade(
                    entry_time=pos.entry_time,
                    exit_time=bar_time,
                    direction=pos.direction,
                    strike=pos.strike,
                    contracts=pos.contracts,
                    entry_option_price=pos.entry_option_price,
                    exit_option_price=exit_price,
                    position_cost=pos.position_cost,
                    pnl_dollars=pnl_dollars,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                    spy_entry=pos.entry_spy_price,
                    spy_exit=spy_price,
                    mode=pos.mode,
                    pattern=pos.pattern,
                    hold_minutes=hold_minutes
                )
                completed_trades.append(trade)

                account += pnl_dollars
                daily_pnl = account - start_account
                positions.remove(pos)

                if pnl_dollars > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                emoji = "✅" if pnl_dollars > 0 else "❌"
                print(f"  [{bar_time.strftime('%H:%M')}] {emoji} EXIT [{pos.mode}] {pos.direction} ${pos.strike}: {exit_reason}")
                print(f"      ${pos.entry_option_price:.2f} → ${exit_price:.2f} | P&L: ${pnl_dollars:+,.0f} ({pnl_pct:+.0f}%) | {hold_minutes}min")

        # Check for new entries
        if daily_stopped:
            continue

        if account < MIN_ACCOUNT_FLOOR:
            continue

        if mode == "DEAD" or mode == "WAIT":
            continue

        # CIRCUIT BREAKER: 2 consecutive losses = done for day
        if consecutive_losses >= 2:
            if not daily_stopped:
                print(f"  [{bar_time.strftime('%H:%M')}] ⛔ CIRCUIT BREAKER: 2 consecutive losses")
                daily_stopped = True
            continue

        # =====================
        # SCALP MODE ENTRIES
        # =====================
        if mode == "SCALP":
            if len(positions) >= SCALP_CONFIG["max_concurrent"]:
                continue
            if scalp_trades_today >= SCALP_CONFIG["max_daily_trades"]:
                continue
            if not is_in_scalp_window(bar_time):
                continue

            signal = detect_scalp_signal(spy_bars, i)
            if signal:
                direction, strength = signal

                # Get ATM option
                option_data = get_atm_option(spy_price, direction, expiry, date, prefetched, bar_ts)
                if not option_data:
                    continue

                entry_price = option_data["price"] * 1.01  # 1% slippage

                # Sizing
                sizing = SCALP_CONFIG["sizing"]
                if protect_mode:
                    sizing = 0.10  # Reduce to 10% in protect mode

                position_value = account * sizing
                contract_cost = entry_price * 100
                contracts = max(1, int(position_value / contract_cost))
                actual_cost = contracts * contract_cost

                pos = Position(
                    entry_time=bar_time,
                    entry_minute_ts=bar_ts,
                    entry_spy_price=spy_price,
                    direction=direction,
                    strike=option_data["strike"],
                    expiry=expiry,
                    option_ticker=option_data["ticker"],
                    contracts=contracts,
                    entry_option_price=entry_price,
                    position_cost=actual_cost,
                    mode="SCALP",
                    pattern="ATM_SCALP",
                    conviction=strength,
                    stop_loss_pct=SCALP_CONFIG["stop_loss"],
                    profit_target_pct=SCALP_CONFIG["profit_target"],
                    trail_trigger=0,
                    trail_pct=0,
                    max_hold_minutes=SCALP_CONFIG["time_stop_minutes"],
                    peak_option_price=entry_price
                )
                positions.append(pos)
                scalp_trades_today += 1

                print(f"  [{bar_time.strftime('%H:%M')}] 🔹 SCALP ENTRY {direction} ${option_data['strike']} ATM")
                print(f"      {contracts}x @ ${entry_price:.2f} (${actual_cost:,.0f}) | Target: +25% | Stop: -15%")

        # =====================
        # BERSERKER MODE ENTRIES
        # =====================
        elif mode == "BERSERKER":
            if len(positions) >= 2:
                continue

            signal = detect_berserker_signal(spy_bars, i, day_open)
            if signal:
                direction, confidence, suggested_strike = signal

                # Get time bucket and calculate semantic match
                time_bucket = get_time_bucket(bar_time)
                spy_range_pct = ((spy_price - day_open) / day_open) * 100

                if i >= 10:
                    avg_vol = sum(b["v"] for b in spy_bars[i-10:i]) / 10
                    vol_spike = spy_bars[i]["v"] / avg_vol if avg_vol > 0 else 1
                else:
                    vol_spike = 1

                semantic_score, pattern, _ = calculate_semantic_match(
                    direction, time_bucket, spy_range_pct, vol_spike, 0.50
                )

                # Boost confidence with semantic match
                final_confidence = min(95, confidence + semantic_score * 0.3)

                if final_confidence < 30:
                    continue

                # Get pattern config
                config = PATTERN_CONFIG.get(pattern, PATTERN_CONFIG["DEFAULT"])

                # Select strike
                strike_data = select_berserker_strike(
                    spy_price, direction, pattern, date, expiry, prefetched, bar_ts
                )

                if not strike_data:
                    continue

                entry_price = strike_data["price"] * 1.02  # 2% slippage

                # Sizing based on conviction
                tier_name = "MINIMUM"
                for name, tier in CONVICTION_TIERS.items():
                    if final_confidence >= tier["min"]:
                        tier_name = name
                        break

                base_sizing = config["max_sizing"]
                tier_mult = CONVICTION_TIERS[tier_name]["sizing_mult"]
                final_sizing = base_sizing * tier_mult

                if protect_mode:
                    final_sizing = 0.10  # Reduce in protect mode

                position_value = account * final_sizing
                contract_cost = entry_price * 100
                contracts = max(1, int(position_value / contract_cost))
                actual_cost = contracts * contract_cost

                pos = Position(
                    entry_time=bar_time,
                    entry_minute_ts=bar_ts,
                    entry_spy_price=spy_price,
                    direction=direction,
                    strike=strike_data["strike"],
                    expiry=expiry,
                    option_ticker=strike_data["ticker"],
                    contracts=contracts,
                    entry_option_price=entry_price,
                    position_cost=actual_cost,
                    mode="BERSERKER",
                    pattern=pattern,
                    conviction=final_confidence,
                    stop_loss_pct=config["stop_loss"],
                    profit_target_pct=3.0,  # No hard profit target, use trail
                    trail_trigger=config["trail_trigger"],
                    trail_pct=config["trail_pct"],
                    max_hold_minutes=config["max_hold_minutes"],
                    peak_option_price=entry_price
                )
                positions.append(pos)

                emoji = "🔥" if tier_name == "BERSERKER" else "⚡" if tier_name == "LARGE" else "🔸"
                print(f"  [{bar_time.strftime('%H:%M')}] {emoji} BERSERKER ENTRY [{tier_name}] {direction} ${strike_data['strike']}")
                print(f"      {contracts}x @ ${entry_price:.2f} (${actual_cost:,.0f}) | Conv:{final_confidence:.0f} | {pattern}")

    # Close remaining positions at EOD
    if positions:
        final_bar = spy_bars[-1]
        final_time = datetime.fromisoformat(final_bar["t"].replace("Z", "+00:00"))
        final_ts = int(final_time.timestamp() * 1000)
        spy_price = final_bar["c"]

        for pos in positions:
            option_bars = prefetched.get(pos.option_ticker, [])
            result = get_option_price_at_time(option_bars, final_ts, use_low=True)

            if result:
                final_price, _ = result
                final_price *= 0.98
            else:
                final_price = pos.entry_option_price * 0.5

            hold_minutes = int((final_time - pos.entry_time).total_seconds() / 60)
            position_value = pos.contracts * final_price * 100
            pnl_dollars = position_value - pos.position_cost
            pnl_pct = (final_price - pos.entry_option_price) / pos.entry_option_price * 100

            trade = Trade(
                entry_time=pos.entry_time,
                exit_time=final_time,
                direction=pos.direction,
                strike=pos.strike,
                contracts=pos.contracts,
                entry_option_price=pos.entry_option_price,
                exit_option_price=final_price,
                position_cost=pos.position_cost,
                pnl_dollars=pnl_dollars,
                pnl_pct=pnl_pct,
                exit_reason="EOD_CLOSE",
                spy_entry=pos.entry_spy_price,
                spy_exit=spy_price,
                mode=pos.mode,
                pattern=pos.pattern,
                hold_minutes=hold_minutes
            )
            completed_trades.append(trade)
            account += pnl_dollars

            print(f"  [EOD] CLOSE [{pos.mode}] {pos.direction} ${pos.strike}: P&L ${pnl_dollars:+,.0f} ({pnl_pct:+.0f}%)")

    # Determine primary mode
    mode_counts = {}
    for m in mode_history:
        if m not in ["WAIT"]:
            mode_counts[m] = mode_counts.get(m, 0) + 1
    primary_mode = max(mode_counts, key=mode_counts.get) if mode_counts else "DEAD"

    wins = [t for t in completed_trades if t.pnl_dollars > 0]
    losses = [t for t in completed_trades if t.pnl_dollars <= 0]
    daily_pnl = account - start_account
    daily_pnl_pct = daily_pnl / start_account * 100

    result = DayResult(
        date=date,
        mode_history=mode_history,
        primary_mode=primary_mode,
        trades=completed_trades,
        num_trades=len(completed_trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(completed_trades) * 100 if completed_trades else 0,
        end_account=account,
        start_account=start_account,
        daily_pnl_dollars=daily_pnl,
        daily_pnl_pct=daily_pnl_pct
    )

    print(f"\n  DAY SUMMARY [{primary_mode}]:")
    print(f"    Mode progression: {' → '.join(mode_history)}")
    print(f"    Trades: {result.num_trades} | Wins: {result.wins} | WR: {result.win_rate:.0f}%")
    print(f"    Daily P&L: ${daily_pnl:+,.0f} ({daily_pnl_pct:+.1f}%)")
    print(f"    Account: ${result.start_account:,.0f} → ${result.end_account:,.0f}")

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
    mode_stats = {"DEAD": 0, "SCALP": 0, "BERSERKER": 0}

    for date in dates:
        result = run_day_simulation(date, account)
        all_results.append(result)
        account = result.end_account
        mode_stats[result.primary_mode] = mode_stats.get(result.primary_mode, 0) + 1
        time.sleep(0.3)

    total_trades = sum(r.num_trades for r in all_results)
    total_wins = sum(r.wins for r in all_results)
    overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    print(f"\n{'='*80}")
    print(f"{week_type} WEEK RESULTS")
    print(f"{'='*80}")
    for r in all_results:
        day_return = r.daily_pnl_pct
        wr = f"{r.win_rate:.0f}%" if r.num_trades > 0 else "N/A"
        print(f"  {r.date} [{r.primary_mode:9}]: ${r.start_account:,.0f} → ${r.end_account:,.0f} ({day_return:+.1f}%) | {r.num_trades} trades | WR: {wr}")

    final_return = ((account / INITIAL_ACCOUNT) - 1) * 100
    hit = account >= target

    print(f"\n  MODE BREAKDOWN: DEAD={mode_stats.get('DEAD',0)} | SCALP={mode_stats.get('SCALP',0)} | BERSERKER={mode_stats.get('BERSERKER',0)}")
    print(f"\n  FINAL: ${INITIAL_ACCOUNT:,.0f} → ${account:,.0f} ({final_return:+.1f}%)")
    print(f"  Total Trades: {total_trades} | Win Rate: {overall_win_rate:.0f}%")
    print(f"  Target: ${target:,.0f} | {'✅ HIT' if hit else f'❌ MISSED by ${target-account:,.0f}'}")

    return account, all_results, overall_win_rate


def main():
    """Run V5 - FINAL BOSS MODE."""
    print("\n" + "=" * 80)
    print("THE BACKTEST FROM HELL v5: FINAL BOSS MODE")
    print("THREE MODES: DEAD | SCALP | BERSERKER")
    print("=" * 80)
    print(f"Initial Account: ${INITIAL_ACCOUNT:,.0f}")
    print(f"Mode Thresholds: DEAD < ${DEAD_THRESHOLD:.2f} < SCALP < ${SCALP_THRESHOLD:.2f} < BERSERKER")
    print(f"Daily Loss Limit: {DAILY_LOSS_LIMIT*100:.0f}%")

    # FRESH WEEKS - never tested before
    bad_week = ["2025-12-22", "2025-12-23", "2025-12-24", "2025-12-26", "2025-12-29"]
    avg_week = ["2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]
    great_week = ["2025-12-10", "2025-12-11", "2025-12-12", "2025-12-15", "2025-12-16"]

    targets = {
        "bad": 5500,      # Even DEAD days should preserve capital
        "average": 8000,  # Mix of SCALP and BERSERKER
        "great": 15000    # BERSERKER mode printing
    }

    results = {}

    bad_final, bad_results, bad_wr = run_week_backtest(
        "BAD (Holiday Week - Low Vol)", bad_week, targets["bad"]
    )
    results["bad"] = {"final": bad_final, "target": targets["bad"], "win_rate": bad_wr}

    avg_final, avg_results, avg_wr = run_week_backtest(
        "AVERAGE (Mixed Activity)", avg_week, targets["average"]
    )
    results["average"] = {"final": avg_final, "target": targets["average"], "win_rate": avg_wr}

    great_final, great_results, great_wr = run_week_backtest(
        "GREAT (High Volatility)", great_week, targets["great"]
    )
    results["great"] = {"final": great_final, "target": targets["great"], "win_rate": great_wr}

    print("\n" + "=" * 80)
    print("FINAL RESULTS - V5 FINAL BOSS MODE")
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
        print(f"  Target: ${target:,.0f} | {status}")

    print("\n" + "=" * 80)
    if all_hit:
        print("🔥🔥🔥 ALL TARGETS HIT - V5 FINAL BOSS DEFEATED 🔥🔥🔥")
    else:
        print("⚠️  SOME TARGETS MISSED - CONTINUE ITERATION")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
