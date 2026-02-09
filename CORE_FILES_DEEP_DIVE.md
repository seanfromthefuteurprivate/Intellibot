# WSB SNAKE - CORE FILES HYPER-DETAILED DOCUMENTATION

**THIS DOCUMENT IS FOR ANOTHER CLAUDE AGENT. ABSORB EVERY DETAIL.**

This documentation covers the 5 core files that control 90% of the system's behavior. Each section is exhaustive, line-by-line, with no detail spared.

---

# TABLE OF CONTENTS

1. [run_max_mode.py](#1-run_max_modepy---the-entry-point)
2. [wsb_snake/trading/alpaca_executor.py](#2-alpaca_executorpy---the-trade-engine)
3. [wsb_snake/trading/risk_governor.py](#3-risk_governorpy---the-gatekeeper)
4. [wsb_snake/execution/apex_conviction_engine.py](#4-apex_conviction_enginepy---the-brain)
5. [wsb_snake/collectors/polygon_enhanced.py](#5-polygon_enhancedpy---the-data-lifeline)

---

# 1. run_max_mode.py - THE ENTRY POINT

## File Location
`/Users/seankuesia/Downloads/Intellibot/run_max_mode.py`

## PURPOSE

**This script is an aggressive last-hour-of-trading-day options scalping predator that continuously scans a watchlist of high-volatility tickers, runs them through a multi-signal conviction engine, and automatically executes 0DTE options trades on Alpaca when conviction exceeds 68%, with built-in position monitoring, trailing stops, and Telegram alerts.**

---

## IMPORTS & DEPENDENCIES

### Standard Library Imports

| Import | What It Provides | Why It's Needed |
|--------|------------------|-----------------|
| `os` | Environment variable access, filesystem path operations | Used to insert the script's directory into `sys.path` for local imports |
| `sys` | System-specific parameters and functions | `sys.path.insert()` to allow imports from the project root |
| `time` | Time-related functions | `time.sleep(SCAN_INTERVAL)` to pause between scan cycles |
| `datetime` from `datetime` | Date/time manipulation | Used in `now_et()` to get current Eastern Time for market hours check |

### Path Manipulation (Line 25)
```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```
**Purpose**: Ensures Python can import local `wsb_snake` modules regardless of where the script is executed from.

### Optional Import: `dotenv` (Lines 27-31)
```python
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass
```
**Purpose**: Loads environment variables from a `.env` file if present. Fails silently if `dotenv` is not installed.

### Project-Specific Imports

| Import | Source Module | What It Provides |
|--------|---------------|------------------|
| `get_logger` | `wsb_snake.utils.logger` | Standardized logging facility |
| `apex_engine` | `wsb_snake.execution.apex_conviction_engine` | Multi-signal fusion engine |
| `polygon_enhanced` | `wsb_snake.collectors.polygon_enhanced` | Real-time stock snapshots |
| `polygon_options` | `wsb_snake.collectors.polygon_options` | Options chain data |
| `send_alert` | `wsb_snake.notifications.telegram_bot` | Telegram notifications |
| `alpaca_executor` | `wsb_snake.trading.alpaca_executor` | Trade execution singleton |

---

## CONSTANTS

### `MAX_MODE_WATCHLIST` (Lines 43-48)
```python
MAX_MODE_WATCHLIST = [
    "SPY", "QQQ", "IWM",      # Index ETFs
    "NVDA", "TSLA", "AMD",    # High-vol tech
    "META", "AAPL", "AMZN",   # Mega caps
]
```
**Value**: 9 tickers
**Why These**: Index ETFs with highest options liquidity + high-beta tech stocks with massive 0DTE volume

### `MIN_CONVICTION` (Line 50)
```python
MIN_CONVICTION = 68  # Institutional standard (was 55 - too low)
```
**Value**: 68 (percent)
**Why 68%**: Raised from 55 because 55 generated too much noise/false signals. 68% is "institutional standard".

### `SCAN_INTERVAL` (Line 51)
```python
SCAN_INTERVAL = 15   # Slightly slower to reduce noise
```
**Value**: 15 seconds

---

## FUNCTIONS

### `get_spot(ticker)` - Lines 53-61

**Purpose**: Get spot price using best available source.

```python
def get_spot(ticker):
    try:
        snap = polygon_enhanced.get_snapshot(ticker)
        if snap and snap.get("price"):
            return float(snap["price"])
    except:
        pass
    return 0
```

**Returns**: `float` (spot price) or `0` (failure)

### `get_atm_option(ticker, spot, side, expiry_date)` - Lines 63-107

**Purpose**: Get ATM option for quick execution - uses ALPACA for real quotes.

**Critical Logic**:
1. Get chain structure from Polygon (for strike discovery)
2. Find ATM (closest to spot)
3. **GET REAL QUOTE FROM ALPACA** (Polygon quotes are often $0)
4. Validate bid/ask > 0
5. Warn if spread > 30%

**Returns**: `dict` with `strike`, `bid`, `ask`, `symbol` or `None`

### `now_et()` - Lines 109-115

**Purpose**: Get current Eastern Time.

**Fallback**: If pytz not installed, returns UTC (DANGER: market hours will be wrong!)

### `is_market_open()` - Lines 117-126

**Purpose**: Check if market is open.

**Checks**:
- Weekend (Sat=5, Sun=6) → False
- Before 9:30 AM ET → False
- After 4:00 PM ET → False

**NOT Handled**: Market holidays, early close days

### `main()` - Lines 128-278

**THE CORE EXECUTION LOOP**

**Initialization**:
1. Print startup banner
2. Send startup alert to Telegram
3. Initialize counters (scan_count=0, trades_executed=0)
4. **CRITICAL**: `alpaca_executor.sync_existing_positions()` - crash recovery
5. **CRITICAL**: `alpaca_executor.start_monitoring()` - start background thread

**Main Loop** (while market open):
```
FOR EACH TICKER in watchlist:
    ├── Get spot price
    ├── Skip if no price
    ├── Run APEX conviction analysis
    ├── Log result with emoji
    │
    ├── IF conviction < 68%: SKIP
    ├── IF action == "NO_TRADE": SKIP
    │
    ├── HIGH CONVICTION TRADE:
    │   ├── Get ATM option contract
    │   ├── Send trade signal alert
    │   ├── Determine direction (long/short)
    │   ├── Calculate target (+6%) and stop (-10%)
    │   ├── Execute via alpaca_executor
    │   └── Send execution alert
    │
    └── Handle exceptions (log, continue)

Sleep SCAN_INTERVAL seconds
```

---

## BUGS THAT WERE FIXED

### The Direction Bug (Lines 228-229)

**Was**: `direction = "long"` hardcoded
**Now**: `direction = "long" if verdict.action == "BUY_CALLS" else "short"`

**Impact**: PUT signals were executing as CALLS. 100% inverted P&L.

### Target/Stop Calculation (Lines 235-236)
```python
target_price = ask * 1.06   # +6% target (achievable in 0DTE timeframe)
stop_loss = ask * 0.90      # -10% stop (wider to avoid noise exits)
```

---

## DANGER ZONES

1. **Silent Failures in `get_spot()`** - All exceptions silently swallowed
2. **pytz Fallback** - If pytz not installed, market hours calculated in UTC
3. **No Holiday Check** - Script will run on market holidays
4. **30% Spread Warning Doesn't Block** - Wide spreads still trade
5. **Keyboard Interrupt Doesn't Clean Up** - No graceful shutdown

---

# 2. alpaca_executor.py - THE TRADE ENGINE

## File Location
`/Users/seankuesia/Downloads/Intellibot/wsb_snake/trading/alpaca_executor.py`

**Total Lines:** 1392

## PURPOSE

**This file is the EXECUTION ENGINE that takes trading signals and converts them into real Alpaca paper/live trades for SPY 0DTE options, managing the full lifecycle from entry to exit with trailing stops, position sizing, risk limits, and automatic crash recovery.**

---

## CLASS STRUCTURE

### Enums (3)

| Enum | Values | Purpose |
|------|--------|---------|
| `OrderSide` | `BUY = "buy"`, `SELL = "sell"` | Order direction |
| `OrderType` | `MARKET = "market"`, `LIMIT = "limit"` | Order execution type |
| `PositionStatus` | `PENDING`, `OPEN`, `CLOSED`, `CANCELLED` | Position lifecycle state |

### Dataclass: AlpacaPosition

| Field | Type | Purpose |
|-------|------|---------|
| `position_id` | `str` | Unique ID like `"SPY_143052"` |
| `symbol` | `str` | Underlying ticker |
| `option_symbol` | `str` | OCC format symbol |
| `side` | `str` | `"long"` or `"short"` |
| `trade_type` | `str` | `"CALLS"` or `"PUTS"` |
| `qty` | `int` | Number of contracts |
| `entry_price` | `float` | Entry price per contract |
| `target_price` | `float` | Target exit price |
| `stop_loss` | `float` | Stop loss price |
| `status` | `PositionStatus` | Current lifecycle state |
| `entry_time` | `Optional[datetime]` | When position was filled |
| `exit_price` | `Optional[float]` | Actual exit price |
| `pnl` | `float` | Dollar P&L |

---

## CRITICAL CONSTANTS

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_DAILY_EXPOSURE` | **$4,000** | Max daily exposure |
| `MAX_PER_TRADE` | **$1,000** | Max per single trade |
| `MAX_CONCURRENT_POSITIONS` | **3** | Max positions at once |
| `_SCALP_TARGET_PCT_DEFAULT` | **1.06** | +6% target |
| `_SCALP_STOP_PCT_DEFAULT` | **0.90** | -10% initial stop |
| `_SCALP_MAX_HOLD_MINUTES_DEFAULT` | **5** | 5 minute max hold |

---

## OCC OPTION SYMBOL FORMAT

**THE 8-DIGIT PADDING IS CRITICAL:**

```
SPY260208C00600000
│   │     │ │
│   │     │ └── Strike × 1000, padded to 8 digits (600.00 = 00600000)
│   │     └──── C = Call, P = Put
│   └────────── YYMMDD (Feb 8, 2026 = 260208)
└────────────── Underlying ticker
```

**Format Code:**
```python
strike_str = f"{int(strike * 1000):08d}"  # 8-digit zero-padded
```

**THE BUG THAT WAS FIXED:** Previously used 7-digit padding (`{:07d}`) which caused 404 errors on position close.

---

## KEY METHODS

### `format_option_symbol()` (Lines 426-439)

**THE CRITICAL OCC FORMAT METHOD**

```python
def format_option_symbol(self, underlying, expiry, strike, option_type):
    date_str = expiry.strftime("%y%m%d")
    strike_str = f"{int(strike * 1000):08d}"  # 8 DIGITS!
    return f"{underlying}{date_str}{option_type}{strike_str}"
```

### `execute_scalp_entry()` (Lines 719-1089)

**THE MAIN TRADE EXECUTION - Step by Step:**

1. **Daily Reset Check**
2. **Risk Governor Check** - `can_trade()`
3. **Concurrent Position Check** - Max 3
4. **Option Type Resolution** - CALLS or PUTS
5. **Validation** - stop < entry < target
6. **Expiry Calculation** - Same day for 0DTE
7. **Strike & Symbol Resolution**
8. **Quote Validation** - bid > 0
9. **Position Sizing** - via risk governor
10. **Daily Exposure Check**
11. **Order Placement**
12. **Position Creation**
13. **Auto-Start Monitoring** - CRITICAL FIX

### `_check_exits()` (Lines 1305-1374)

**TRAILING STOP PROGRESSION:**

| Profit Level | Stop Moves To | Effect |
|--------------|---------------|--------|
| +5% | +3% | Lock in 3% profit minimum |
| +3% | 0% (breakeven) | No loss possible |
| +2% | -5% | Reduce risk from -10% to -5% |

**Exit Conditions:**
- `current_price >= target_price` → "TARGET HIT"
- `current_price <= stop_loss` → "STOP LOSS"
- `elapsed >= 5 minutes` → "TIME DECAY"

### `sync_existing_positions()` (Lines 280-367)

**CRASH RECOVERY** - Picks up orphaned positions from Alpaca on startup.

---

## API ENDPOINTS USED

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v2/account` | GET | Get account info |
| `/v2/positions` | GET | Get all positions |
| `/v2/positions/{symbol}` | DELETE | Close position |
| `/v2/orders` | POST | Place order |
| `/v2/orders/{order_id}` | GET | Get order status |
| `/v1beta1/options/quotes/latest` | GET | Get option quote |

---

## BUGS FIXED

1. **Symbol Format Mismatch** - Changed from 7-digit to 8-digit padding
2. **$50k Position Bug** - MAX_PER_TRADE was 50000, now 1000
3. **Positions Never Closed** - Auto-start monitoring when trades execute

---

# 3. risk_governor.py - THE GATEKEEPER

## File Location
`/Users/seankuesia/Downloads/Intellibot/wsb_snake/trading/risk_governor.py`

## PURPOSE

**The RiskGovernor is the CENTRAL GATEKEEPER that ALL trading engines MUST pass through before ANY trade execution, enforcing daily loss limits, position caps, ticker/sector exposure limits, and dynamic position sizing based on confidence and volatility.**

---

## TradingEngine ENUM

```python
class TradingEngine(Enum):
    SCALPER = "scalper"    # 0DTE / intraday SPY/QQQ/ETF
    MOMENTUM = "momentum"  # Small-cap breakout
    MACRO = "macro"        # Commodity/LEAPS
    VOL_SELL = "vol_sell"  # Credit spreads
```

---

## GovernorConfig - EVERY FIELD

| Field | Default Value | Description |
|-------|---------------|-------------|
| `max_daily_loss` | **-$200** | KILL SWITCH TRIGGER |
| `kill_switch_manual` | `False` | Manual override to halt |
| `max_concurrent_positions_global` | **3** | Max open positions |
| `max_daily_exposure_global` | **$4,000** | Max daily exposure |
| `max_positions_scalper` | **2** | Scalper engine limit |
| `max_positions_momentum` | **1** | Momentum engine limit |
| `max_positions_macro` | **1** | Macro engine limit |
| `max_positions_vol_sell` | **1** | Vol sell engine limit |
| `max_exposure_per_ticker` | **$1,000** | Max per ticker |
| `max_exposure_per_sector` | **$2,000** | Max per sector |
| `max_premium_scalper` | **$1,000** | Base premium for scalper |
| `max_premium_momentum` | **$800** | Base premium for momentum |
| `max_premium_macro` | **$1,500** | Base premium for macro |
| `max_pct_buying_power_per_trade` | **5%** | Max % of buying power |

---

## can_trade() - THE CRITICAL GATE

**Check Sequence:**

```
CHECK 1: KILL SWITCH
└── if kill_switch_manual → BLOCK

CHECK 2: DAILY P&L LOSS LIMIT
└── if daily_pnl <= -$200 → BLOCK

CHECK 3: GLOBAL POSITION COUNT
└── if positions >= 3 → BLOCK

CHECK 4: DAILY EXPOSURE CAP
└── if exposure >= $4,000 → BLOCK

CHECK 5: PER-TICKER EXPOSURE
└── if ticker_exposure >= $1,000 → BLOCK

CHECK 6: PER-SECTOR EXPOSURE
└── if sector_exposure >= $2,000 → BLOCK

SUCCESS: return True, "ok"
```

---

## compute_position_size()

**Calculates number of contracts:**

```python
base_cap = get_max_premium_for_engine(engine)  # $1000 for SCALPER
confidence_scale = max(0.5, min(1.0, confidence_pct / 100.0))
vol_scale = 1.0 / max(0.5, min(2.0, volatility_factor))
max_premium = base_cap * confidence_scale * vol_scale

contract_cost = option_price * 100
num_contracts = int(max_premium / contract_cost)
```

---

## SECTOR_MAP

```python
SECTOR_MAP = {
    "SPY": "index", "QQQ": "index", "IWM": "index",
    "SLV": "commodity", "GLD": "commodity",
    "TSLA": "tech", "NVDA": "tech", "AAPL": "tech",
    "RKLB": "space", "ASTS": "space",
    # ... etc
}
DEFAULT_SECTOR = "other"
```

---

## SINGLETON PATTERN

```python
_governor: Optional[RiskGovernor] = None

def get_risk_governor() -> RiskGovernor:
    global _governor
    if _governor is None:
        _governor = RiskGovernor()
    return _governor
```

---

# 4. apex_conviction_engine.py - THE BRAIN

## File Location
`/Users/seankuesia/Downloads/Intellibot/wsb_snake/execution/apex_conviction_engine.py`

## PURPOSE

**The APEX Conviction Engine is an institutional-grade multi-signal fusion system that aggregates conviction scores from 6 independent analysis engines using weighted averaging to produce a final trading verdict (BUY_CALLS, BUY_PUTS, or NO_TRADE) only when combined conviction exceeds 68%.**

---

## CONSTANTS

| Constant | Value | Purpose |
|----------|-------|---------|
| `STRONG_CONVICTION` | **75** | Threshold for STRONG direction |
| `TRADE_THRESHOLD` | **68** | MINIMUM conviction to trade |
| `AVOID_THRESHOLD` | **50** | No directional edge |

---

## SIGNAL WEIGHTS - The 6 Engines

```python
WEIGHTS = {
    "technical": 0.20,      # 20% - RSI, MACD, SMA, EMA
    "candlestick": 0.15,    # 15% - 36 patterns + confluence
    "order_flow": 0.20,     # 20% - Sweeps, blocks, institutional
    "probability": 0.20,    # 20% - Multi-engine fusion
    "pattern_memory": 0.15, # 15% - Historical match
    "ai_verdict": 0.10,     # 10% - GPT-4/Gemini (pending)
}
```

---

## DATA CLASSES

### ConvictionSignal
```python
@dataclass
class ConvictionSignal:
    source: str       # Engine name
    score: float      # 0-100
    direction: str    # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float # 0-100
    reason: str       # Explanation
    weight: float     # 0-1
```

### ApexVerdict
```python
@dataclass
class ApexVerdict:
    ticker: str
    conviction_score: float    # 0-100
    direction: str            # "STRONG_LONG", "LONG", "NEUTRAL", etc.
    action: str               # "BUY_CALLS", "BUY_PUTS", "NO_TRADE"
    signals: List[ConvictionSignal]
    target_pct: float
    stop_pct: float
    position_size_multiplier: float
```

---

## SIGNAL COLLECTION

### Technical Signal
- RSI < 30 (oversold) → +15 BULLISH
- RSI > 70 (overbought) → +15 BEARISH
- MACD histogram > 0 → +15 BULLISH
- Price > SMA AND EMA → +10 BULLISH

### Candlestick Signal
- Top 3 patterns: `strength * reliability * 10`
- Confluence > 70 → +15
- "STRONG_BUY" signal → +10 BULLISH

### Order Flow Signal
- "STRONG_BUY" → +25 BULLISH
- Institutional % > 30% → +15
- Sweep % > 20% with BUY → +10 BULLISH

### Probability Signal
- Direct `combined_score` (0-100)
- `win_probability` for confidence

### Pattern Memory Signal
- Similarity > 0.8 → +25
- Win rate > 0.7 → +20
- Win rate < 0.4 → **-15 PENALTY**

### AI Verdict Signal
- Currently returns NEUTRAL (pending integration)

---

## FUSION ALGORITHM

```python
conviction_score = Σ(signal.score × signal.weight) / Σ(signal.weight)
```

### Direction Determination
```python
bullish_weight = sum(s.score * s.weight for s if s.direction == "BULLISH")
bearish_weight = sum(s.score * s.weight for s if s.direction == "BEARISH")

if bullish_weight > bearish_weight * 1.1:
    direction = "LONG" or "STRONG_LONG"
elif bearish_weight > bullish_weight * 1.1:
    direction = "SHORT" or "STRONG_SHORT"
else:
    direction = "NEUTRAL"
```

---

## VERDICT GENERATION

| Condition | Action |
|-----------|--------|
| conviction >= 68 AND direction = LONG/STRONG_LONG | **BUY_CALLS** |
| conviction >= 68 AND direction = SHORT/STRONG_SHORT | **BUY_PUTS** |
| conviction < 68 OR direction = NEUTRAL | **NO_TRADE** |

---

## POWER HOUR ADJUSTMENTS (3-4 PM ET)

| Setting | Power Hour | Standard |
|---------|------------|----------|
| Target % | **20%** | 25% |
| Stop % | **10%** | 12% |
| Size Multiplier | **× 1.2** | 1.0 |

---

# 5. polygon_enhanced.py - THE DATA LIFELINE

## File Location
`/Users/seankuesia/Downloads/Intellibot/wsb_snake/collectors/polygon_enhanced.py`

## PURPOSE

**This is the PRIMARY MARKET DATA LIFELINE - a rate-limit-conscious Polygon.io adapter that collects real-time stock snapshots, OHLCV bars, technical indicators, NBBO quotes, trade flow data, and market regime analysis - all optimized for 0DTE SPY scalping with 120-second caching to survive the 5-requests-per-minute basic plan limit.**

---

## RATE LIMITS - THE BRUTAL REALITY

```python
REQUESTS_PER_MINUTE = 5  # Polygon basic plan limit
```

**CRITICAL**: Exceed this and you get HTTP 429 errors.

---

## CACHING SYSTEM

### Request Cache (120 seconds TTL)
```python
self._cache: Dict[str, Tuple[datetime, Any]] = {}
self._cache_ttl = 120  # 2 minutes
```

### Scan Cache (60 seconds TTL)
```python
self._scan_cache: Dict[str, Any] = {}
self._scan_cache_ttl = 60
```

**Flow**:
1. Check cache → return if < 120 seconds old
2. Check rate limit → wait or return stale cache
3. Make API request
4. Cache result

---

## STOCK DATA METHODS

### `get_snapshot(ticker)`
**Endpoint**: `/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}`

**Returns**:
```python
{
    "symbol": str,
    "price": float,
    "today_open": float,
    "today_high": float,
    "today_low": float,
    "today_volume": int,
    "today_vwap": float,
    "change_pct": float,
}
```

### `get_intraday_bars(ticker, timespan, multiplier, limit)`
**Endpoint**: `/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}`

**Returns**: List of OHLCV bars

### `get_daily_bars(ticker, limit)`
**Endpoint**: `/v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}`

---

## ORDER FLOW METHODS

### `get_classified_trades(ticker)`
**Returns**:
```python
{
    "sweep_count": int,
    "sweep_pct": float,
    "sweep_direction": str,  # "BUY", "SELL", "NONE"
    "block_count": int,
    "institutional_pct": float,
    "is_institutional_active": bool,  # True if > 20%
}
```

### Trade Classification
- `SWEEP`: Intermarket sweep (aggressive institutional)
- `BLOCK`: Size >= 10,000 shares
- `LARGE`: Size >= 1,000 shares
- `ODD_LOT`: Size < 100 shares (retail)

### `analyze_order_flow(ticker)`
**Flow Signals**:
- `STRONG_BUY`: Price up + bid/ask imbalance > 0.1
- `BUY`: Price up
- `STRONG_SELL`: Price down + bid/ask imbalance < -0.1
- `SELL`: Price down

---

## TECHNICAL INDICATOR METHODS

### `get_rsi(ticker, window, timespan)`
**Endpoint**: `/v1/indicators/rsi/{ticker}`

### `get_sma(ticker, window, timespan)`
**Endpoint**: `/v1/indicators/sma/{ticker}`

### `get_ema(ticker, window, timespan)`
**Endpoint**: `/v1/indicators/ema/{ticker}`

### `get_macd(ticker, timespan, short, long, signal)`
**Endpoint**: `/v1/indicators/macd/{ticker}`
**Default**: 12/26/9

### `get_full_technicals(ticker)`
Combines all indicators with signal generation:
- RSI_OVERBOUGHT (> 70) → -1
- RSI_OVERSOLD (< 30) → +1
- MACD_BULLISH (histogram > 0) → +1
- ABOVE_SMA20 → +1

---

## MARKET REGIME METHODS

### `get_market_regime()`
**Regime Classification**:
| Score | Regime |
|-------|--------|
| > 15 | strong_bullish |
| > 5 | bullish |
| > -5 | neutral |
| > -15 | bearish |
| <= -15 | strong_bearish |

---

## RATE LIMIT DEFENSE SYSTEM

**3-Layer Defense:**

1. **Request Tracking**: Prunes requests older than 60 seconds
2. **Pre-Request Check**: If >= 5 requests in window, activate defense
3. **Defense Actions**:
   - Return stale cache if available
   - Wait up to 30 seconds for window to clear
   - Proceed once clear

**Post-429 Handling**:
```python
if resp.status_code == 429:
    return self._cache[cache_key][1]  # Return ANY cached data
```

---

## INTEGRATION POINTS

| File | Methods Used |
|------|--------------|
| `apex_conviction_engine.py` | Full instance |
| `run_max_mode.py` | `get_snapshot` |
| `spy_scalper.py` | `get_classified_trades`, `get_intraday_bars` |
| `probability_engine.py` | `get_momentum_signals`, `get_market_regime`, `get_full_technicals` |

---

## CRITICAL NOTES

1. **ALWAYS assume data may be 120 seconds stale**
2. **The 5 req/min limit is BRUTAL** - plan API calls carefully
3. **Check `available` field** - many methods return `{"available": False}`
4. **Timestamps are milliseconds** - not seconds
5. **Options data is REFERENCE ONLY** - no real-time pricing on basic plan

---

# APPENDIX: QUICK REFERENCE

## Critical Numbers

| Setting | Value |
|---------|-------|
| MAX_PER_TRADE | $1,000 |
| MAX_DAILY_EXPOSURE | $4,000 |
| MAX_CONCURRENT_POSITIONS | 3 |
| KILL_SWITCH | -$200 daily loss |
| TRADE_THRESHOLD | 68% conviction |
| TARGET | +6% |
| STOP | -10% |
| MAX_HOLD | 5 minutes |

## OCC Symbol Format
```
SPY260208C00600000
    ^^^^^^ ^^^^^^^^
    YYMMDD 8-DIGIT STRIKE
```

## Trailing Stop Progression
| Profit | Stop Moves To |
|--------|---------------|
| +2% | -5% |
| +3% | Breakeven |
| +5% | +3% |

## Signal Weights
| Engine | Weight |
|--------|--------|
| Technical | 20% |
| Candlestick | 15% |
| Order Flow | 20% |
| Probability | 20% |
| Pattern Memory | 15% |
| AI Verdict | 10% |

---

*Last Updated: Feb 8, 2026*
*For: Agent Handover Documentation*
