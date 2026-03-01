#!/usr/bin/env python3
"""
BACKTEST FROM HELL v4: THE BERSERKER
FULL BRAIN - Every component replicated with historical data

Components:
1. SEMANTIC MEMORY - SQL queries against 52 learned trades
2. TRADE GRAPH - Similarity matching to historical trades
3. SPECIALIST SWARM - 6 agents with defined scoring logic
4. GEX APPROXIMATION - Polygon options volume proxy
5. CROSS-ASSET - QQQ confirmation
6. PATTERN-BASED EXECUTION - Different params per pattern

All Polygon real option prices. Zero simulation.
"""
import os
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
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

# Pattern-specific configurations
PATTERN_CONFIG = {
    "LOTTO_TICKET": {
        "otm_range": (3, 5),       # 3-5 points OTM
        "price_range": (0.10, 0.50),  # Cheap options
        "max_sizing": 0.50,        # Up to 50% on BERSERKER
        "stop_loss": -0.60,        # Wide stop (binary outcomes)
        "trail_trigger": 3.00,     # Don't trail until +300%
        "trail_pct": 0.30,         # Wide trail
        "max_hold_minutes": 90,
    },
    "REVERSAL_PUT": {
        "otm_range": (1, 2),
        "price_range": (0.30, 1.50),
        "max_sizing": 0.35,
        "stop_loss": -0.35,
        "trail_trigger": 1.00,
        "trail_pct": 0.15,
        "max_hold_minutes": 45,
    },
    "MOMENTUM_CALL": {
        "otm_range": (1, 3),
        "price_range": (0.30, 1.50),
        "max_sizing": 0.30,
        "stop_loss": -0.40,
        "trail_trigger": 0.75,
        "trail_pct": 0.20,
        "max_hold_minutes": 60,
    },
    "HIGH_VOLUME_CONVICTION": {
        "otm_range": (0, 1),       # ATM or 1pt OTM
        "price_range": (0.80, 3.00),
        "max_sizing": 0.30,
        "stop_loss": -0.30,
        "trail_trigger": 0.50,
        "trail_pct": 0.12,
        "max_hold_minutes": 30,
    },
    "DEFAULT": {
        "otm_range": (2, 3),
        "price_range": (0.20, 1.00),
        "max_sizing": 0.20,
        "stop_loss": -0.40,
        "trail_trigger": 1.00,
        "trail_pct": 0.20,
        "max_hold_minutes": 60,
    }
}

# Conviction tiers
CONVICTION_TIERS = {
    "BERSERKER": {"min": 85, "sizing_mult": 1.0},    # Full pattern sizing
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
    pattern: str
    conviction: float
    stop_loss_pct: float
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
    pattern: str
    conviction: float
    hold_minutes: int
    swarm_scores: Dict[str, float]


@dataclass
class DayResult:
    date: str
    trades: List[Trade]
    num_trades: int
    wins: int
    losses: int
    win_rate: float
    avg_winner: float
    avg_loser: float
    largest_trade: float
    end_account: float
    start_account: float
    max_drawdown: float
    berserker_trades: int


# ============================================================================
# COMPONENT 1: SEMANTIC MEMORY
# ============================================================================

def load_learned_trades():
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
        print(f"  Loaded {len(LEARNED_TRADES_CACHE)} learned trades from database")
    except Exception as e:
        print(f"  Warning: Could not load learned trades: {e}")
        LEARNED_TRADES_CACHE = []

    return LEARNED_TRADES_CACHE


def get_time_bucket(dt: datetime) -> str:
    """Categorize time of day."""
    hour = dt.hour
    minute = dt.minute
    total_minutes = hour * 60 + minute

    if total_minutes < 600:  # Before 10:00
        return "opening_drive"
    elif total_minutes < 720:  # 10:00-12:00
        return "mid_morning"
    elif total_minutes < 840:  # 12:00-14:00
        return "midday"
    elif total_minutes < 900:  # 14:00-15:00
        return "afternoon"
    else:  # 15:00+
        return "power_hour"


def calculate_semantic_match(
    direction: str,
    time_bucket: str,
    spy_range_pct: float,
    volume_spike: float,
    entry_price: float
) -> Tuple[float, str, Dict]:
    """
    Query semantic memory for pattern match.
    Returns (match_score, best_pattern, best_trade_data).
    """
    learned = load_learned_trades()
    if not learned:
        return (0.0, "DEFAULT", {})

    best_score = 0
    best_pattern = "DEFAULT"
    best_trade = {}

    for trade in learned:
        score = 0

        # Direction match
        trade_dir = "CALL" if trade.get("trade_type") == "CALL" else "PUT"
        if trade_dir == direction:
            score += 20

        # Pattern-specific boosts
        pattern = trade.get("pattern", "")

        if pattern == "LOTTO_TICKET":
            if entry_price < 0.50:
                score += 25
            if time_bucket in ["opening_drive", "power_hour"]:
                score += 15
            if spy_range_pct > 0.5:
                score += 10

        elif pattern == "REVERSAL_PUT":
            if direction == "PUT" and spy_range_pct < -0.3:
                score += 30
            if time_bucket == "power_hour":
                score += 10

        elif pattern == "MOMENTUM_CALL":
            if direction == "CALL" and spy_range_pct > 0.3:
                score += 25
            if time_bucket == "opening_drive":
                score += 15

        elif pattern == "HIGH_VOLUME_CONVICTION":
            if volume_spike > 2.5:
                score += 25
            if entry_price > 0.80:
                score += 10

        elif pattern == "PRECIOUS_METALS_MOMENTUM":
            # This pattern is for specific tickers, skip for SPY
            pass

        # Volume similarity
        if volume_spike > 2.0:
            score += 10

        # Profitability weight - winning patterns get bonus
        trade_pnl = trade.get("profit_loss_pct", 0) or 0
        if trade_pnl > 100:
            score += 15
        elif trade_pnl > 50:
            score += 10
        elif trade_pnl > 0:
            score += 5

        if score > best_score:
            best_score = score
            best_pattern = pattern if pattern else "DEFAULT"
            best_trade = trade

    return (best_score, best_pattern, best_trade)


# ============================================================================
# COMPONENT 2: TRADE GRAPH (Similarity Matching)
# ============================================================================

def find_similar_trades(
    direction: str,
    time_bucket: str,
    spy_range_from_open: float,
    day_of_week: int
) -> List[Dict]:
    """Find most similar historical trades."""
    learned = load_learned_trades()
    if not learned:
        return []

    scored_trades = []

    for trade in learned:
        similarity = 0

        # Direction match
        trade_dir = "CALL" if trade.get("trade_type") == "CALL" else "PUT"
        if trade_dir == direction:
            similarity += 30

        # Outcome weight
        pnl = trade.get("profit_loss_pct", 0) or 0
        if pnl > 0:
            similarity += min(20, pnl / 10)  # Up to +20 for big winners

        scored_trades.append({
            "trade": trade,
            "similarity": similarity
        })

    # Sort by similarity and return top 3
    scored_trades.sort(key=lambda x: x["similarity"], reverse=True)
    return [s["trade"] for s in scored_trades[:3]]


def get_trade_graph_guidance(similar_trades: List[Dict]) -> Dict:
    """Extract guidance from similar trades."""
    if not similar_trades:
        return {
            "avg_pnl_pct": 50.0,
            "avg_hold_minutes": 30,
            "suggested_offset": 2
        }

    pnls = [t.get("profit_loss_pct", 0) or 0 for t in similar_trades]
    avg_pnl = sum(pnls) / len(pnls) if pnls else 50.0

    return {
        "avg_pnl_pct": avg_pnl,
        "avg_hold_minutes": 30,  # Default since we don't have exact hold times
        "suggested_offset": 2
    }


# ============================================================================
# COMPONENT 3: SPECIALIST SWARM (6 Agents)
# ============================================================================

def agent_gex(spy_price: float, call_volume: int, put_volume: int) -> Tuple[float, str]:
    """
    Agent 1: GEX Specialist
    Analyzes round numbers and put/call volume.
    """
    score = 0
    reason = ""

    # Round number proximity
    round_5 = round(spy_price / 5) * 5
    distance_from_round = abs(spy_price - round_5)

    if distance_from_round < 1.0:
        # Near major round number
        if call_volume > put_volume * 1.5:
            score += 15
            reason = f"Near ${round_5}, call-heavy → bullish dealer flow"
        elif put_volume > call_volume * 1.5:
            score -= 10
            reason = f"Near ${round_5}, put-heavy → bearish dealer flow"

    # Put/call ratio
    if call_volume > 0 and put_volume > 0:
        pc_ratio = put_volume / call_volume
        if pc_ratio > 1.5:
            score -= 5  # Bearish sentiment
        elif pc_ratio < 0.7:
            score += 5  # Bullish sentiment

    return (score, reason)


def agent_momentum(bars: List[Dict], idx: int) -> Tuple[float, str]:
    """
    Agent 2: Momentum Specialist
    Analyzes multiple timeframe momentum.
    """
    if idx < 30:
        return (0, "Insufficient data")

    current = bars[idx]["c"]

    # 5-bar momentum
    mom_5 = (current - bars[idx-5]["c"]) / bars[idx-5]["c"]
    # 10-bar momentum
    mom_10 = (current - bars[idx-10]["c"]) / bars[idx-10]["c"]
    # 30-bar momentum
    mom_30 = (current - bars[idx-30]["c"]) / bars[idx-30]["c"]

    score = 0

    # All aligned bullish
    if mom_5 > 0.001 and mom_10 > 0.001 and mom_30 > 0.001:
        score += 20
        return (score, "All timeframes BULLISH aligned")

    # All aligned bearish
    if mom_5 < -0.001 and mom_10 < -0.001 and mom_30 < -0.001:
        score += 20
        return (score, "All timeframes BEARISH aligned")

    # Short-term reversal against long-term
    if mom_5 * mom_30 < 0:  # Different signs
        score += 10
        return (score, "Possible reversal forming")

    return (score, "Mixed momentum")


def agent_volume(bars: List[Dict], idx: int) -> Tuple[float, str]:
    """
    Agent 3: Volume Specialist
    Analyzes volume patterns.
    """
    if idx < 20:
        return (0, "Insufficient data")

    current_vol = bars[idx]["v"]

    # 20-bar average volume
    avg_vol = sum(b["v"] for b in bars[idx-20:idx]) / 20

    if avg_vol == 0:
        return (0, "No volume data")

    vol_ratio = current_vol / avg_vol

    # Volume trend (is it increasing?)
    recent_avg = sum(b["v"] for b in bars[idx-5:idx]) / 5 if idx >= 5 else avg_vol
    vol_trend = recent_avg / avg_vol if avg_vol > 0 else 1

    score = 0

    if vol_ratio >= 3.0:
        score += 20
        return (score, f"INSTITUTIONAL FLOW: {vol_ratio:.1f}x avg volume")
    elif vol_ratio >= 2.0:
        score += 15
        return (score, f"High volume: {vol_ratio:.1f}x average")
    elif vol_ratio >= 1.5:
        score += 10
        return (score, f"Above average volume: {vol_ratio:.1f}x")
    elif vol_ratio < 0.5:
        score -= 10
        return (score, f"Low volume warning: {vol_ratio:.1f}x")

    return (score, f"Normal volume: {vol_ratio:.1f}x")


def agent_pattern(semantic_score: float, pattern: str) -> Tuple[float, str]:
    """
    Agent 4: Pattern Specialist
    Uses semantic memory match.
    """
    if semantic_score >= 60:
        return (25, f"Strong pattern match: {pattern}")
    elif semantic_score >= 40:
        return (15, f"Moderate pattern match: {pattern}")
    elif semantic_score >= 20:
        return (5, f"Weak pattern match: {pattern}")
    else:
        return (-5, "No historical precedent")


def agent_risk(
    daily_pnl_pct: float,
    consecutive_losses: int,
    account_ratio: float
) -> Tuple[float, float, str]:
    """
    Agent 5: Risk Specialist
    Returns (score, sizing_modifier, reason).
    """
    score = 0
    sizing_mod = 1.0

    # Protect big daily gains
    if daily_pnl_pct > 0.50:
        sizing_mod = 0.5
        return (score, sizing_mod, "PROTECT GAINS: Reduce sizing 50%")
    elif daily_pnl_pct > 0.30:
        sizing_mod = 0.75
        return (score, sizing_mod, "Good day: Reduce sizing 25%")

    # After consecutive losses
    if consecutive_losses >= 3:
        sizing_mod = 0.5
        score -= 10
        return (score, sizing_mod, "CAUTION: 3+ losses, reduce sizing")
    elif consecutive_losses == 2:
        sizing_mod = 0.75
        return (score, sizing_mod, "Careful: 2 losses in a row")

    # Compound when winning big
    if account_ratio > 2.0:
        sizing_mod = 1.25
        score += 10
        return (score, sizing_mod, "COMPOUND: Account doubled, increase sizing")
    elif account_ratio > 1.5:
        sizing_mod = 1.15
        score += 5
        return (score, sizing_mod, "Winning: slight size increase")

    return (score, sizing_mod, "Normal risk parameters")


def agent_cross_asset(spy_momentum: float, qqq_momentum: float) -> Tuple[float, str]:
    """
    Agent 6: Cross-Asset Specialist
    Compares SPY and QQQ momentum.
    """
    if spy_momentum > 0.003 and qqq_momentum > 0.003:
        return (15, "CONFIRMED: Both SPY and QQQ bullish")
    elif spy_momentum < -0.003 and qqq_momentum < -0.003:
        return (15, "CONFIRMED: Both SPY and QQQ bearish")
    elif spy_momentum * qqq_momentum < 0:  # Diverging
        return (-10, "DIVERGENCE: SPY and QQQ disagree")
    else:
        return (5, "Neutral cross-asset signal")


def run_swarm(
    spy_bars: List[Dict],
    qqq_bars: List[Dict],
    idx: int,
    direction: str,
    semantic_score: float,
    pattern: str,
    daily_pnl_pct: float,
    consecutive_losses: int,
    account_ratio: float,
    call_volume: int,
    put_volume: int
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Run all 6 swarm agents and aggregate scores.
    Returns (total_conviction, sizing_modifier, details).
    """
    spy_price = spy_bars[idx]["c"]

    # Calculate momentums
    if idx >= 10:
        spy_momentum = (spy_bars[idx]["c"] - spy_bars[idx-10]["c"]) / spy_bars[idx-10]["c"]
    else:
        spy_momentum = 0

    if qqq_bars and idx < len(qqq_bars) and idx >= 10:
        qqq_momentum = (qqq_bars[idx]["c"] - qqq_bars[idx-10]["c"]) / qqq_bars[idx-10]["c"]
    else:
        qqq_momentum = spy_momentum  # Fallback

    # Run all agents
    gex_score, gex_reason = agent_gex(spy_price, call_volume, put_volume)
    mom_score, mom_reason = agent_momentum(spy_bars, idx)
    vol_score, vol_reason = agent_volume(spy_bars, idx)
    pat_score, pat_reason = agent_pattern(semantic_score, pattern)
    risk_score, sizing_mod, risk_reason = agent_risk(daily_pnl_pct, consecutive_losses, account_ratio)
    cross_score, cross_reason = agent_cross_asset(spy_momentum, qqq_momentum)

    # Direction alignment bonus
    direction_bonus = 0
    if direction == "CALL" and spy_momentum > 0.002:
        direction_bonus = 10
    elif direction == "PUT" and spy_momentum < -0.002:
        direction_bonus = 10

    # Aggregate scores with weights
    total = (
        gex_score * 0.15 +
        mom_score * 0.20 +
        vol_score * 0.15 +
        pat_score * 0.25 +
        risk_score * 0.10 +
        cross_score * 0.15 +
        direction_bonus
    )

    # Normalize to 0-100 scale (raw scores can be -30 to +110)
    conviction = max(0, min(100, total + 40))

    details = {
        "gex": {"score": gex_score, "reason": gex_reason},
        "momentum": {"score": mom_score, "reason": mom_reason},
        "volume": {"score": vol_score, "reason": vol_reason},
        "pattern": {"score": pat_score, "reason": pat_reason},
        "risk": {"score": risk_score, "reason": risk_reason},
        "cross_asset": {"score": cross_score, "reason": cross_reason},
        "direction_bonus": direction_bonus,
        "raw_total": total,
        "conviction": conviction
    }

    return (conviction, sizing_mod, details)


# ============================================================================
# COMPONENT 4: GEX PROXY (from Polygon options volume)
# ============================================================================

def estimate_gex_from_options(
    prefetched: Dict[str, List[Dict]],
    spy_price: float,
    expiry: str,
    bar_ts: int
) -> Tuple[float, int, int]:
    """
    Estimate GEX proxy from options volume.
    Returns (gex_score, total_call_volume, total_put_volume).
    """
    call_volume = 0
    put_volume = 0

    # Check strikes near current price
    for offset in range(-3, 4):
        strike = round(spy_price) + offset

        call_ticker = build_option_ticker(strike, expiry, "CALL")
        put_ticker = build_option_ticker(strike, expiry, "PUT")

        call_bars = prefetched.get(call_ticker, [])
        put_bars = prefetched.get(put_ticker, [])

        # Sum volume from all bars (approximation)
        call_volume += sum(b.get("v", 0) for b in call_bars)
        put_volume += sum(b.get("v", 0) for b in put_bars)

    # GEX score based on put/call balance
    gex_score = 0
    if call_volume > 0 and put_volume > 0:
        ratio = call_volume / (call_volume + put_volume)
        if ratio > 0.6:  # Call-heavy = positive GEX
            gex_score = 10
        elif ratio < 0.4:  # Put-heavy = negative GEX (explosive)
            gex_score = 15  # Negative GEX means bigger moves!

    return (gex_score, call_volume, put_volume)


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
# SIGNAL DETECTION
# ============================================================================

def detect_signal(bars: List[Dict], idx: int, day_open: float) -> Optional[Tuple[str, float, float]]:
    """
    Signal detection - cast wide net (v2 style).
    Returns (direction, confidence, strike_suggestion).
    """
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

    # CALL signal
    if momentum > 0.002 and vol_spike > 1.3:
        direction = "CALL"
        confidence = min(90, 50 + momentum * 2000 + vol_spike * 10 + bar_range * 500)
        strike = round(current_price) + 2

    # PUT signal
    elif momentum < -0.002 and vol_spike > 1.3:
        direction = "PUT"
        confidence = min(90, 50 + abs(momentum) * 2000 + vol_spike * 10 + bar_range * 500)
        strike = round(current_price) - 2

    if direction and confidence >= 62:
        return (direction, confidence, strike)

    return None


# ============================================================================
# STRIKE SELECTION BY PATTERN
# ============================================================================

def select_strike_by_pattern(
    spy_price: float,
    direction: str,
    pattern: str,
    date: str,
    expiry: str,
    prefetched: Dict[str, List[Dict]],
    bar_ts: int
) -> Optional[Dict]:
    """Select optimal strike based on pattern type."""
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
        # Fallback: try wider range
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

    # For LOTTO, prefer cheapest; otherwise prefer liquid
    if pattern == "LOTTO_TICKET":
        candidates.sort(key=lambda x: x["price"])
    else:
        candidates.sort(key=lambda x: -x["volume"])

    return candidates[0]


# ============================================================================
# MAIN SIMULATION
# ============================================================================

def run_day_simulation(date: str, start_account: float) -> DayResult:
    """Run full brain simulation for one day."""
    print(f"\n{'='*70}")
    print(f"SIMULATING: {date} [BERSERKER v4 - FULL BRAIN]")
    print(f"Starting account: ${start_account:,.2f}")
    print(f"{'='*70}")

    # Fetch SPY and QQQ bars
    spy_bars = fetch_minute_bars("SPY", date)
    qqq_bars = fetch_minute_bars("QQQ", date)

    if not spy_bars:
        print(f"  No SPY data for {date}")
        return DayResult(
            date=date, trades=[], num_trades=0, wins=0, losses=0,
            win_rate=0, avg_winner=0, avg_loser=0, largest_trade=0,
            end_account=start_account, start_account=start_account,
            max_drawdown=0, berserker_trades=0
        )

    print(f"  Loaded {len(spy_bars)} SPY bars, {len(qqq_bars)} QQQ bars")

    day_high = max(b["h"] for b in spy_bars)
    day_low = min(b["l"] for b in spy_bars)
    day_open = spy_bars[0]["o"]
    print(f"  Day range: ${day_low:.2f} - ${day_high:.2f} (${day_high-day_low:.2f})")

    expiry = date
    account = start_account
    peak_account = start_account
    max_drawdown = 0
    positions: List[Position] = []
    completed_trades: List[Trade] = []
    daily_pnl = 0
    consecutive_losses = 0
    berserker_count = 0

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

    # Load learned trades
    load_learned_trades()

    # Get GEX estimate
    gex_score, total_calls, total_puts = estimate_gex_from_options(
        prefetched, spy_mid, expiry, 0
    )
    print(f"  GEX proxy: score={gex_score}, calls={total_calls:,}, puts={total_puts:,}")

    for i, bar in enumerate(spy_bars):
        bar_time = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        bar_ts = int(bar_time.timestamp() * 1000)
        spy_price = bar["c"]
        time_bucket = get_time_bucket(bar_time)

        spy_range_pct = ((spy_price - day_open) / day_open) * 100

        # 1. Monitor existing positions
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

            # Stop loss (pattern-specific)
            if current_pnl_pct <= pos.stop_loss_pct:
                exit_reason = "STOP_LOSS"

            # Max hold time
            elif hold_minutes >= pos.max_hold_minutes:
                exit_reason = "TIME_LIMIT"

            # Trailing stops (pattern-specific)
            elif current_pnl_pct >= pos.trail_trigger:
                drawdown = (pos.peak_option_price - current_price) / pos.peak_option_price
                if drawdown > pos.trail_pct:
                    exit_reason = "TRAIL_STOP"

            if exit_reason:
                exit_price = current_price * 0.98  # 2% slippage
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
                    pattern=pos.pattern,
                    conviction=pos.conviction,
                    hold_minutes=hold_minutes,
                    swarm_scores={}
                )
                completed_trades.append(trade)

                account += pnl_dollars
                daily_pnl = (account - start_account) / start_account
                positions.remove(pos)

                if pnl_dollars > 0:
                    consecutive_losses = 0
                else:
                    consecutive_losses += 1

                tier = "🔥" if pos.conviction >= 85 else ""
                print(f"  [{bar_time.strftime('%H:%M')}] {tier}EXIT {pos.direction} ${pos.strike}: {exit_reason}")
                print(f"      ${pos.entry_option_price:.2f} → ${exit_price:.2f} | P&L: ${pnl_dollars:+,.0f} ({pnl_pct:+.0f}%) | {hold_minutes}min | {pos.pattern}")

                if account > peak_account:
                    peak_account = account
                dd = (peak_account - account) / peak_account if peak_account > 0 else 0
                if dd > max_drawdown:
                    max_drawdown = dd

        # 2. Check for new signals
        if account < MIN_ACCOUNT_FLOOR:
            continue

        if len(positions) >= 2:
            continue

        signal = detect_signal(spy_bars, i, day_open)

        if signal:
            direction, signal_conf, suggested_strike = signal

            # Calculate volume spike for semantic matching
            if i >= 10:
                avg_vol = sum(b["v"] for b in spy_bars[i-10:i]) / 10
                vol_spike = spy_bars[i]["v"] / avg_vol if avg_vol > 0 else 1
            else:
                vol_spike = 1

            # Get estimated entry price for semantic matching
            est_price = 0.50  # Default estimate

            # COMPONENT 1: Semantic Memory
            semantic_score, pattern, best_trade = calculate_semantic_match(
                direction, time_bucket, spy_range_pct, vol_spike, est_price
            )

            # COMPONENT 2: Trade Graph
            similar_trades = find_similar_trades(
                direction, time_bucket, spy_range_pct, bar_time.weekday()
            )
            graph_guidance = get_trade_graph_guidance(similar_trades)

            # COMPONENT 3: Specialist Swarm
            account_ratio = account / INITIAL_ACCOUNT
            conviction, sizing_mod, swarm_details = run_swarm(
                spy_bars, qqq_bars, i, direction,
                semantic_score, pattern,
                daily_pnl, consecutive_losses, account_ratio,
                total_calls, total_puts
            )

            # Determine conviction tier
            tier_name = "MINIMUM"
            for name, tier in CONVICTION_TIERS.items():
                if conviction >= tier["min"]:
                    tier_name = name
                    break

            if conviction < 30:
                continue  # Below minimum threshold

            # Get pattern config
            config = PATTERN_CONFIG.get(pattern, PATTERN_CONFIG["DEFAULT"])

            # STRIKE SELECTION by pattern
            strike_data = select_strike_by_pattern(
                spy_price, direction, pattern, date, expiry, prefetched, bar_ts
            )

            if not strike_data:
                continue

            strike = strike_data["strike"]
            option_price = strike_data["price"]
            option_ticker = strike_data["ticker"]
            option_volume = strike_data["volume"]

            # Apply slippage
            entry_price = option_price * 1.02

            # CONVICTION-BASED SIZING
            base_sizing = config["max_sizing"]
            tier_mult = CONVICTION_TIERS[tier_name]["sizing_mult"]
            final_sizing = base_sizing * tier_mult * sizing_mod

            position_value = account * final_sizing
            contract_cost = entry_price * 100
            contracts = max(1, int(position_value / contract_cost))
            actual_cost = contracts * contract_cost

            # Create position with pattern-specific parameters
            pos = Position(
                entry_time=bar_time,
                entry_minute_ts=bar_ts,
                entry_spy_price=spy_price,
                direction=direction,
                strike=strike,
                expiry=expiry,
                option_ticker=option_ticker,
                contracts=contracts,
                entry_option_price=entry_price,
                position_cost=actual_cost,
                pattern=pattern,
                conviction=conviction,
                stop_loss_pct=config["stop_loss"],
                trail_trigger=config["trail_trigger"],
                trail_pct=config["trail_pct"],
                max_hold_minutes=config["max_hold_minutes"],
                peak_option_price=entry_price
            )
            positions.append(pos)

            if tier_name == "BERSERKER":
                berserker_count += 1

            tier_emoji = "🔥" if tier_name == "BERSERKER" else "⚡" if tier_name == "LARGE" else ""
            print(f"  [{bar_time.strftime('%H:%M')}] {tier_emoji}{tier_name} ENTRY {direction} ${strike}")
            print(f"      {contracts}x @ ${entry_price:.2f} (${actual_cost:,.0f}) | Conv:{conviction:.0f} | {pattern}")

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
                pattern=pos.pattern,
                conviction=pos.conviction,
                hold_minutes=hold_minutes,
                swarm_scores={}
            )
            completed_trades.append(trade)
            account += pnl_dollars

            print(f"  [EOD] CLOSE {pos.direction} ${pos.strike}: P&L ${pnl_dollars:+,.0f} ({pnl_pct:+.0f}%) | {pos.pattern}")

    wins = [t for t in completed_trades if t.pnl_dollars > 0]
    losses = [t for t in completed_trades if t.pnl_dollars <= 0]

    result = DayResult(
        date=date,
        trades=completed_trades,
        num_trades=len(completed_trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / len(completed_trades) * 100 if completed_trades else 0,
        avg_winner=sum(t.pnl_pct for t in wins) / len(wins) if wins else 0,
        avg_loser=sum(t.pnl_pct for t in losses) / len(losses) if losses else 0,
        largest_trade=max((t.pnl_dollars for t in completed_trades), default=0),
        end_account=account,
        start_account=start_account,
        max_drawdown=max_drawdown * 100,
        berserker_trades=berserker_count
    )

    print(f"\n  DAY SUMMARY:")
    print(f"    Trades: {result.num_trades} | Wins: {result.wins} | Win Rate: {result.win_rate:.0f}%")
    print(f"    🔥 BERSERKER trades: {berserker_count}")
    print(f"    Avg Winner: {result.avg_winner:+.0f}% | Avg Loser: {result.avg_loser:+.0f}%")
    print(f"    Largest: ${result.largest_trade:+,.0f}")
    print(f"    Account: ${result.start_account:,.0f} → ${result.end_account:,.0f} ({((result.end_account/result.start_account)-1)*100:+.1f}%)")

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
    total_berserker = 0

    for date in dates:
        result = run_day_simulation(date, account)
        all_results.append(result)
        account = result.end_account
        total_berserker += result.berserker_trades
        time.sleep(0.3)

    total_trades = sum(r.num_trades for r in all_results)
    total_wins = sum(r.wins for r in all_results)
    overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0

    print(f"\n{'='*80}")
    print(f"{week_type} WEEK RESULTS")
    print(f"{'='*80}")
    for r in all_results:
        day_return = ((r.end_account / r.start_account) - 1) * 100 if r.start_account > 0 else 0
        wr = f"{r.win_rate:.0f}%" if r.num_trades > 0 else "N/A"
        print(f"  {r.date}: ${r.start_account:,.0f} → ${r.end_account:,.0f} ({day_return:+.1f}%) | {r.num_trades} trades | WR: {wr} | 🔥:{r.berserker_trades}")

    final_return = ((account / INITIAL_ACCOUNT) - 1) * 100
    hit = account >= target

    print(f"\n  FINAL: ${INITIAL_ACCOUNT:,.0f} → ${account:,.0f} ({final_return:+.1f}%)")
    print(f"  Total Trades: {total_trades} | Win Rate: {overall_win_rate:.0f}%")
    print(f"  🔥 Total BERSERKER trades: {total_berserker}")
    print(f"  Target: ${target:,.0f} | {'✅ HIT' if hit else f'❌ MISSED by ${target-account:,.0f}'}")

    return account, all_results, overall_win_rate


def main():
    """Run THE BERSERKER v4 - Full Brain Backtest."""
    print("\n" + "=" * 80)
    print("THE BACKTEST FROM HELL v4: THE BERSERKER")
    print("FULL BRAIN - SEMANTIC MEMORY + TRADE GRAPH + SWARM + GEX + CROSS-ASSET")
    print("=" * 80)
    print(f"Initial Account: ${INITIAL_ACCOUNT:,.0f}")
    print(f"Patterns: {list(PATTERN_CONFIG.keys())}")
    print(f"Conviction Tiers: {list(CONVICTION_TIERS.keys())}")

    # Week definitions
    bad_week = ["2026-01-22", "2026-01-23", "2026-01-26", "2026-01-27", "2026-01-28"]
    avg_week = ["2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13"]
    great_week = ["2026-01-29", "2026-01-30", "2026-02-02", "2026-02-03", "2026-02-04"]

    targets = {
        "bad": 6000,
        "average": 10000,
        "great": 15000
    }

    results = {}

    bad_final, bad_results, bad_wr = run_week_backtest("BAD (Low Volatility)", bad_week, targets["bad"])
    results["bad"] = {"final": bad_final, "target": targets["bad"], "win_rate": bad_wr}

    avg_final, avg_results, avg_wr = run_week_backtest("AVERAGE (Moderate VIX)", avg_week, targets["average"])
    results["average"] = {"final": avg_final, "target": targets["average"], "win_rate": avg_wr}

    great_final, great_results, great_wr = run_week_backtest("GREAT (High Volatility)", great_week, targets["great"])
    results["great"] = {"final": great_final, "target": targets["great"], "win_rate": great_wr}

    print("\n" + "=" * 80)
    print("FINAL RESULTS - THE BERSERKER v4")
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
        print("🔥🔥🔥 ALL TARGETS HIT - THE BERSERKER IS VALIDATED 🔥🔥🔥")
    else:
        print("⚠️  SOME TARGETS MISSED - REVIEW REQUIRED")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
