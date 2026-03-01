# Small-Cap Options Revival Plan: Flow-First Strategy

## Executive Summary

Small-cap options (LUNR, SLS, RKLB, etc.) failed because we traded them like ETFs. The cutting-edge approach is **"Flow-First"** - only enter when institutional sweeps are detected, with strict liquidity filters.

---

## Why Small-Cap Options Failed

| Factor | ETF (SPY) | Small-Cap (LUNR) |
|--------|-----------|------------------|
| Bid-Ask Spread | 0.5-2% | 20-50% |
| Open Interest | 100,000+ | 100-1,000 |
| Daily Volume | Millions | Hundreds |
| Market Makers | 10+ competing | 1-2 (if any) |
| Gamma Data | SpotGamma coverage | None |
| Institutional Flow | Heavy, trackable | Sparse, critical |

**Key Insight:** On small caps, you're trading AGAINST market makers with massive edge. The ONLY way to win is to detect when institutions are forcing direction.

---

## The Flow-First Strategy

### Core Principle
> "Never trade small-cap options unless institutional money just entered."

### Detection Method: Unusual Whales API

**Endpoint:** `/api/stock/{ticker}/options-flow`

**Signals to Track:**
1. **Sweeps** - Aggressive orders that cross exchanges (urgency signal)
2. **Block trades** - Large single prints (institutional size)
3. **Opening transactions** - New positions (not closing)
4. **Premium > $100K** - Serious money, not retail

### Entry Rules

```python
class FlowFirstEntry:
    """Only enter small-cap options with institutional confirmation"""

    MIN_SWEEP_PREMIUM = 100_000      # $100K minimum
    MIN_OPEN_INTEREST = 500          # Must have liquidity
    MAX_BID_ASK_SPREAD_PCT = 15      # No wider than 15%
    FOLLOW_DELAY_SECONDS = 30        # Don't front-run, follow

    def should_enter(self, ticker: str, flow: UnusualWhalesFlow) -> bool:
        # Must have sweep in last 5 minutes
        recent_sweeps = flow.get_sweeps(ticker, minutes=5)
        if not recent_sweeps:
            return False

        # Sweep must be large enough
        total_premium = sum(s.premium for s in recent_sweeps)
        if total_premium < self.MIN_SWEEP_PREMIUM:
            return False

        # Check liquidity
        chain = get_options_chain(ticker)
        target_strike = recent_sweeps[0].strike
        if chain.get_oi(target_strike) < self.MIN_OPEN_INTEREST:
            return False

        # Check spread
        spread_pct = chain.get_spread_pct(target_strike)
        if spread_pct > self.MAX_BID_ASK_SPREAD_PCT:
            return False

        return True
```

---

## API Integration: Unusual Whales

### Subscription Required
- **Tier:** API Access or Super Live Buffet ($48/month)
- **Endpoints Needed:**
  - `/api/stock/{ticker}/options-flow` - Real-time flow
  - `/api/flow/alerts` - Institutional alerts
  - `/api/darkpool/recent` - Dark pool prints

### Data Points to Capture

```python
@dataclass
class SmallCapFlowSignal:
    ticker: str
    timestamp: datetime

    # Sweep data
    sweep_side: str          # "call" or "put"
    sweep_premium: float     # Total $ spent
    sweep_type: str          # "sweep", "block", "split"
    is_opening: bool         # New position vs closing

    # Contract details
    strike: float
    expiry: date
    spot_price: float

    # Liquidity check
    open_interest: int
    bid_ask_spread_pct: float

    # Institutional markers
    is_above_ask: bool       # Paid up = urgency
    exchange_count: int      # Multi-exchange = sweep

    def is_actionable(self) -> bool:
        """True if this flow signal is worth following"""
        return (
            self.sweep_premium >= 100_000 and
            self.is_opening and
            self.is_above_ask and
            self.open_interest >= 500 and
            self.bid_ask_spread_pct <= 15
        )
```

---

## Momentum Engine Upgrade

### Current State (DISABLED)
```python
MOMENTUM_USE_OPTIONS = False  # We disabled this
```

### New State: Flow-Conditional Options

```python
# In config.py
MOMENTUM_USE_OPTIONS = True  # Re-enable
MOMENTUM_REQUIRE_FLOW = True  # NEW: Only with flow confirmation

# In momentum_engine.py
def execute_momentum_entry(candidate: MomentumCandidate) -> bool:
    """Execute momentum entry - ONLY with institutional flow confirmation"""

    from wsb_snake.config import MOMENTUM_REQUIRE_FLOW

    if MOMENTUM_REQUIRE_FLOW:
        from wsb_snake.collectors.unusual_whales import unusual_whales

        # Check for recent institutional flow
        flow = unusual_whales.get_recent_flow(
            ticker=candidate.ticker,
            minutes=10
        )

        if not flow.has_institutional_sweep():
            log.info(f"Momentum skip {candidate.ticker}: No institutional flow detected")
            return False

        # Verify flow direction matches our bias
        if candidate.direction == "long" and flow.net_direction != "bullish":
            log.info(f"Momentum skip {candidate.ticker}: Flow is bearish, we're bullish")
            return False

    # Proceed with entry...
```

---

## Liquidity Filters (Critical)

### Before ANY Small-Cap Option Trade

```python
class LiquidityGate:
    """Block trades on illiquid options"""

    # Strict minimums for small caps
    MIN_OPEN_INTEREST = 500
    MIN_DAILY_VOLUME = 100
    MAX_SPREAD_PCT = 15.0
    MIN_UNDERLYING_VOLUME = 1_000_000  # Stock must be liquid too

    def passes(self, option: OptionContract) -> Tuple[bool, str]:
        if option.open_interest < self.MIN_OPEN_INTEREST:
            return False, f"OI {option.open_interest} < {self.MIN_OPEN_INTEREST}"

        if option.volume < self.MIN_DAILY_VOLUME:
            return False, f"Volume {option.volume} < {self.MIN_DAILY_VOLUME}"

        spread_pct = (option.ask - option.bid) / option.mid * 100
        if spread_pct > self.MAX_SPREAD_PCT:
            return False, f"Spread {spread_pct:.1f}% > {self.MAX_SPREAD_PCT}%"

        return True, "Liquidity OK"
```

---

## Dark Pool Integration

### Why It Matters for Small Caps

Dark pool prints on small caps are RARE but SIGNIFICANT. A $1M dark pool print on LUNR means someone is building a position.

### Unusual Whales Dark Pool Endpoint

```python
def check_dark_pool_accumulation(ticker: str) -> bool:
    """Check if institutional accumulation via dark pools"""

    prints = unusual_whales.get_dark_pool(ticker, days=5)

    # Calculate net dark pool flow
    total_bought = sum(p.value for p in prints if p.side == "buy")
    total_sold = sum(p.value for p in prints if p.side == "sell")

    net_flow = total_bought - total_sold

    # Significant accumulation = bullish
    if net_flow > 5_000_000:  # $5M+ net buying
        return True

    return False
```

---

## Expiry Selection: 0DTE or Nothing

### The Problem with Weeklies

LUNR/SLS trades used **weekly expiry** (5+ days). On small caps:
- Theta decay = -5% to -15% per day
- No gamma acceleration until expiry day
- Wide spreads compound losses

### The Solution: 0DTE Only for Small Caps

```python
# In momentum_engine.py - CHANGE THIS
def get_expiry_for_small_cap(ticker: str) -> Optional[date]:
    """Only trade 0DTE for small caps - or don't trade at all"""

    today = date.today()

    # Check if 0DTE options exist for this ticker
    chain = get_options_chain(ticker, expiry=today)

    if not chain.has_contracts():
        # No 0DTE available - DO NOT TRADE WEEKLIES
        log.warning(f"{ticker} has no 0DTE options - skipping")
        return None

    return today  # Only 0DTE
```

---

## Complete Flow-First Architecture

```
                    Unusual Whales API
                           │
                           ▼
              ┌────────────────────────┐
              │   Sweep Detector       │
              │   - Premium > $100K    │
              │   - Opening position   │
              │   - Above ask (urgent) │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Liquidity Gate       │
              │   - OI > 500           │
              │   - Spread < 15%       │
              │   - Volume > 100       │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Expiry Filter        │
              │   - 0DTE ONLY          │
              │   - No weeklies        │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Direction Match      │
              │   - Sweep direction    │
              │   - Dark pool bias     │
              │   - Momentum confirm   │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   EXECUTE TRADE        │
              │   - Follow the flow    │
              │   - Tight stops        │
              │   - Quick exits        │
              └────────────────────────┘
```

---

## Risk Management for Small Caps

### Position Sizing

```python
# Smaller positions on small caps due to liquidity risk
SMALL_CAP_MAX_POSITION = 500  # vs $1000 for ETFs
SMALL_CAP_MAX_CONTRACTS = 5   # Hard limit
```

### Exit Rules

```python
# Faster exits due to spread drag
SMALL_CAP_TARGET_PCT = 0.20   # +20% (vs +6% for ETFs)
SMALL_CAP_STOP_PCT = -0.15    # -15% (vs -10% for ETFs)
SMALL_CAP_MAX_HOLD_MIN = 15   # 15 minutes max (vs 5 for scalps)
```

### Flow Reversal Exit

```python
def should_emergency_exit(ticker: str, position_direction: str) -> bool:
    """Exit if institutional flow reverses against us"""

    flow = unusual_whales.get_recent_flow(ticker, minutes=5)

    if position_direction == "long":
        # Exit if big put sweeps come in
        if flow.has_sweep(side="put", min_premium=50_000):
            return True

    return False
```

---

## Implementation Roadmap

### Week 1: Unusual Whales Integration
1. Subscribe to Unusual Whales ($48/month)
2. Create `wsb_snake/collectors/unusual_whales.py`
3. Implement sweep detection logic
4. Test on paper with alerts only (no execution)

### Week 2: Liquidity Gates
1. Add liquidity filters to `alpaca_executor.py`
2. Implement spread checking before entry
3. Add OI/volume minimums
4. Test rejection rates

### Week 3: Re-enable Momentum with Flow
1. Set `MOMENTUM_USE_OPTIONS = True`
2. Set `MOMENTUM_REQUIRE_FLOW = True`
3. Implement 0DTE-only expiry selection
4. Paper trade for validation

### Week 4: Go Live
1. Start with $250 max per small-cap trade
2. Monitor fill quality and slippage
3. Adjust parameters based on results
4. Scale up if profitable

---

## Expected Performance

### Without Flow-First (Current - DISABLED)
- Win Rate: ~30%
- Avg Loss: -25% (spread + theta)
- Result: **Consistent losses**

### With Flow-First (Proposed)
- Trade Frequency: -80% (only with flow)
- Win Rate: ~55-60% (following smart money)
- Avg Win: +20%
- Avg Loss: -15%
- Result: **Marginally profitable if executed well**

### Honest Assessment
> Small-cap options will NEVER be as profitable as ETF scalping. The edge from flow-following is thin and requires perfect execution. Consider keeping them disabled unless you have specific alpha on a ticker.

---

## Alternative: Equity-Only Momentum

If the complexity isn't worth it, keep:
```python
MOMENTUM_USE_OPTIONS = False
```

And trade small-cap EQUITY (shares) instead:
- No theta decay
- No spread drag
- Can hold overnight
- Still benefits from momentum signals

---

## Summary

| Question | Answer |
|----------|--------|
| Can APIs help small-cap options? | **Yes, but marginally** |
| Which API? | **Unusual Whales** ($48/month) |
| What's the strategy? | **Flow-First** - Only trade with institutional sweeps |
| Is it worth it? | **Questionable** - Thin edge, high complexity |
| Best alternative? | **Equity-only momentum** - No theta, no spreads |

---

*Plan Created: 2026-02-09 | WSB Snake HYDRA Small-Cap Options Strategy*
