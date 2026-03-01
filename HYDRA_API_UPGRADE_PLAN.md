# HYDRA API Upgrade Plan - Institutional-Grade Data Stack

## Executive Summary

After comprehensive research by 6 specialized agents, this plan outlines the highest-impact API integrations for the WSB Snake 0DTE scalping system. Priority is given to **real-time, actionable signals** with sub-minute alpha decay.

---

## Current State vs Target State

| Category | Current | Target | Priority |
|----------|---------|--------|----------|
| Options Flow | Barchart scraping (15min delay, 10% coverage) | Unusual Whales API (real-time, 100% coverage) | **P0** |
| Gamma Exposure | None | SpotGamma GEX (Call/Put walls, Vol Trigger) | **P0** |
| Macro Events | Partial (Finnhub) | FRED + BLS calendar integration | **P1** |
| Execution | Alpaca Paper (100ms) | IB or Tradier for live (50ms) | **P1** |
| Crypto Correlation | None | CoinGlass liquidations + Binance | **P2** |
| AI Sentiment | Partial | Enhanced with regime-specific weighting | **P2** |

---

## PHASE 0: Critical Integrations (Immediate - This Week)

### 0.1 SpotGamma GEX Integration ($89/month)

**Why Critical:** Put/Call walls have 83-89% hold rates. Gamma flip level determines volatility regime.

**Data Points to Capture:**
- Call Wall (resistance)
- Put Wall (support)
- Gamma Flip Level (bullish/bearish threshold)
- Vol Trigger (low/high vol regime)
- 1-Day Expected Range

**Integration Location:** `wsb_snake/execution/regime_detector.py`

```python
class GEXData:
    call_wall: float      # Maximum call gamma strike
    put_wall: float       # Maximum put gamma strike
    gamma_flip: float     # Zero gamma level
    vol_trigger: float    # Volatility regime threshold
    net_gex: float        # Aggregate dealer gamma

def get_trading_bias(price: float, gex: GEXData) -> str:
    """Determine bias based on price vs GEX levels"""
    if price > gex.vol_trigger:
        # Positive gamma - mean reversion expected
        return "SELL_WALLS"  # Fade moves to walls
    else:
        # Negative gamma - trending expected
        return "TRADE_BREAKS"  # Trade breakouts
```

**Trading Rules:**
| Price Location | GEX Regime | Strategy |
|---------------|------------|----------|
| Near Call Wall | Positive GEX | Short calls / bearish |
| Near Put Wall | Positive GEX | Long calls / bullish |
| Below Vol Trigger | Negative GEX | Trade momentum, wider stops |
| Above Vol Trigger | Positive GEX | Mean reversion, tighter stops |

---

### 0.2 Unusual Whales API ($48/month or $528/year)

**Why Critical:** Real-time options flow replaces 15-min delayed scraping. Captures institutional sweeps, dark pool prints, and 0DTE-specific flow.

**Replace:** `wsb_snake/collectors/options_flow_scraper.py`

**Key Endpoints:**
- `/api/stock/{ticker}/options-flow` - Real-time sweeps
- `/api/darkpool/recent` - Dark pool prints
- `/api/options/zero-dte` - 0DTE specific flow
- `/api/flow/alerts` - Institutional alerts

**Integration Points:**
1. `spy_scalper.py` - Add sweep confirmation before entry
2. `apex_conviction_engine.py` - Weight order_flow signal higher with real data
3. `risk_governor.py` - Block trades against heavy institutional flow

```python
# New signal in APEX
class UnusualWhalesSignal:
    sweep_direction: str  # "bullish" or "bearish"
    sweep_volume: int
    dark_pool_prints: List[Dict]
    institutional_alert: bool

    def get_flow_bias(self) -> float:
        """Return -1 to +1 flow bias"""
        if self.sweep_volume > 1_000_000:
            return 0.8 if self.sweep_direction == "bullish" else -0.8
        return 0.0
```

---

## PHASE 1: High-Value Additions (Week 2-3)

### 1.1 FRED API Integration (FREE)

**Purpose:** Yield curve monitoring, financial stress indicators

**Key Series:**
- `DGS10` - 10-Year Treasury Yield
- `T10Y2Y` - 2s10s Yield Curve Spread
- `VIXCLS` - VIX Close
- `BAMLH0A0HYM2` - High Yield Spread

**Integration Location:** `wsb_snake/collectors/fred_collector.py` (already exists, enhance)

**New Feature:** Pre-market regime check
```python
def get_macro_regime() -> str:
    spread_2s10s = fred.get_series('T10Y2Y').iloc[-1]
    if spread_2s10s < 0:
        return "INVERTED_CURVE"  # Risk-off
    elif spread_2s10s < 0.5:
        return "FLAT_CURVE"  # Cautious
    else:
        return "NORMAL_CURVE"  # Risk-on
```

### 1.2 BLS Economic Calendar Integration (FREE)

**Purpose:** Flag high-volatility days (CPI, NFP)

**Critical Releases:**
| Release | Day | Time | Volatility Impact |
|---------|-----|------|-------------------|
| CPI | 12th-13th monthly | 8:30 AM ET | EXTREME |
| Employment (NFP) | 1st Friday | 8:30 AM ET | EXTREME |
| Weekly Claims | Every Thursday | 8:30 AM ET | MODERATE |

**Integration:**
```python
def is_high_vol_event_day() -> Tuple[bool, str]:
    """Check if today has major economic release"""
    # Parse BLS calendar
    if is_cpi_day():
        return True, "CPI"
    if is_nfp_day():
        return True, "NFP"
    return False, None

# In spy_scalper.py
if is_high_vol_event_day()[0]:
    # Increase position size (higher vol = higher opportunity)
    # Widen stops (more noise)
    # Focus on 8:30-9:30 AM window
```

### 1.3 Execution Upgrade Path

**Current:** Alpaca Paper Trading (~100-200ms latency, simulated fills)

**Recommended Upgrade:** Interactive Brokers
- 50-70ms latency
- Smart Order Router for best fills
- Proven 0DTE execution quality
- Sub-second fills in volatility

**Alternative:** Tradier Pro ($10/month)
- $0/contract for ETF options
- Good for high-frequency testing

**Implementation:** Add `wsb_snake/trading/ib_executor.py` as alternative to `alpaca_executor.py`

---

## PHASE 2: Supplementary Data (Week 4+)

### 2.1 CoinGlass Liquidation Monitoring ($79/month)

**Purpose:** Early warning for equity volatility spikes

**Key Signals:**
- Liquidation cascade > $500M/hour = equity vol incoming
- Extreme funding rates = sentiment extreme
- BTC overnight moves > 3% = gap risk

**Integration:**
```python
class CryptoVolatilityWarning:
    liquidation_1h: float
    funding_rate: float
    btc_change_24h: float

    def should_increase_vol_sizing(self) -> bool:
        return self.liquidation_1h > 500_000_000
```

### 2.2 Binance WebSocket (FREE)

**Purpose:** Real-time BTC/ETH for correlation monitoring

**Use Cases:**
- Overnight gap prediction
- Intraday correlation tracking
- Risk-off early warning

---

## PHASE 3: NOT RECOMMENDED

These were researched but provide insufficient value for 0DTE:

| API | Reason to Skip |
|-----|----------------|
| OPRA Direct | $1,700+/mo, Polygon already sources from OPRA |
| CBOE DataShop | Marginal improvement over Polygon Greeks |
| Orbital Insight | Satellite imagery = days/weeks alpha decay |
| Placer.ai | Foot traffic = days/weeks alpha decay |
| CoinMetrics | On-chain metrics too slow for intraday |
| CFTC COT | Weekly updates, useless for 0DTE timing |

---

## Implementation Priority Matrix

| Priority | API | Cost | Alpha Potential | Effort |
|----------|-----|------|-----------------|--------|
| **P0** | SpotGamma GEX | $89/mo | **HIGH** - 83-89% wall hold rates | Medium |
| **P0** | Unusual Whales | $48/mo | **HIGH** - Real-time institutional flow | Low |
| **P1** | FRED Macro | FREE | **Medium** - Regime context | Low |
| **P1** | BLS Calendar | FREE | **Medium** - Event day awareness | Low |
| **P1** | IB Execution | $0 (commissions only) | **Medium** - Better fills | Medium |
| **P2** | CoinGlass | $79/mo | **Low-Medium** - Vol warning | Low |
| **P2** | Binance | FREE | **Low** - Correlation baseline | Low |

---

## Cost Summary

### Minimum Viable Upgrade (Immediate Impact)
| Item | Monthly Cost |
|------|-------------|
| SpotGamma Standard | $89 |
| Unusual Whales Super | $48 |
| **Total** | **$137/month** |

### Full Stack Upgrade
| Item | Monthly Cost |
|------|-------------|
| SpotGamma Pro | $129 |
| Unusual Whales API | $48 |
| CoinGlass Startup | $79 |
| IB Commissions | ~$50 (volume dependent) |
| **Total** | **~$306/month** |

---

## Files to Create/Modify

### NEW FILES:
1. `wsb_snake/collectors/spotgamma_collector.py` - GEX data integration
2. `wsb_snake/collectors/unusual_whales_collector.py` - Options flow API
3. `wsb_snake/collectors/bls_calendar.py` - Economic event calendar
4. `wsb_snake/collectors/coinglass_collector.py` - Liquidation monitoring
5. `wsb_snake/trading/ib_executor.py` - Interactive Brokers executor

### MODIFY:
1. `wsb_snake/execution/regime_detector.py` - Add GEX regime signals
2. `wsb_snake/execution/apex_conviction_engine.py` - Weight real flow data
3. `wsb_snake/engines/spy_scalper.py` - Use GEX walls for entries/exits
4. `wsb_snake/trading/risk_governor.py` - Event day position sizing

---

## Success Metrics

After implementation, track:
1. **Win rate improvement** - Target: +5-10% from GEX walls
2. **Flow signal accuracy** - Target: 70%+ directional accuracy
3. **Event day performance** - Target: 2x returns on CPI/NFP days
4. **Slippage reduction** - Target: -30% with IB vs Alpaca paper

---

## Execution Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | SpotGamma + Unusual Whales | New collectors, APEX integration |
| 2 | FRED + BLS | Macro regime, event calendar |
| 3 | IB Executor | Live execution capability |
| 4 | CoinGlass + Binance | Crypto correlation signals |
| 5+ | Optimization | Tune weights, backtest improvements |

---

## Risk Mitigation

1. **API Failures:** Maintain Polygon as fallback for all new data sources
2. **Cost Control:** Start with minimum tiers, upgrade based on ROI
3. **Complexity:** Add one integration at a time, validate before proceeding
4. **Overfitting:** Backtest all new signals on historical data first

---

*Plan Generated: 2026-02-09 | WSB Snake HYDRA v3.0*
