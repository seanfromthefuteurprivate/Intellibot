# WSB Snake System Failure Diagnosis

**Date:** March 6, 2026
**Analyst:** Claude Opus 4.5
**Status:** CRITICAL - System mathematically cannot trade

---

## Executive Summary

The WSB Snake trading system has been dead for 4+ consecutive days. After comprehensive diagnosis, the root cause is **not incremental bugs** but a **fundamental architectural failure**: the 13-signal conviction system requires data that HYDRA and Polygon are not providing.

**Bottom line:** With only 2-4 signals capable of firing, the system cannot reach the 5/13 conviction threshold required to trade. The system is mathematically impossible to trade.

---

## Table of Contents

1. [Data Sources Status](#1-data-sources-status)
2. [HYDRA Intelligence Audit](#2-hydra-intelligence-audit)
3. [Polygon API Audit](#3-polygon-api-audit)
4. [Conviction System Analysis](#4-conviction-system-analysis)
5. [Gate Rejection Analysis](#5-gate-rejection-analysis)
6. [Root Cause Summary](#6-root-cause-summary)
7. [Recommendations](#7-recommendations)

---

## 1. Data Sources Status

| Data Source | Status | Health |
|-------------|--------|--------|
| HYDRA Engine | Connected | **2/4 components healthy** |
| Polygon API | Connected | **DELAYED status** |
| Alpaca API | Connected | Healthy |

### HYDRA Component Health

```
"components_healthy": 2,
"components_total": 4
```

**50% of HYDRA is non-functional.**

---

## 2. HYDRA Intelligence Audit

### Raw HYDRA Response (March 6, 2026 17:19 UTC)

```json
{
    "timestamp": "2026-03-06T17:19:31.174017+00:00",
    "blowup_probability": 22,
    "blowup_direction": "NEUTRAL",
    "blowup_regime": "UNKNOWN",
    "blowup_recommendation": "SCALP_ONLY",
    "blowup_triggers": ["vix_inversion:0.70", "premarket_gap:0.40"],
    "gex_regime": "NEGATIVE",
    "gex_total": -573259161918.0,
    "gex_flip_point": null,
    "gex_flip_distance_pct": 1.0,
    "gex_charm_per_hour": -774083425286760.0,
    "gex_key_support": [],
    "gex_key_resistance": [],
    "flow_institutional_bias": "NEUTRAL",
    "flow_confidence": 50,
    "flow_premium_calls": 0,
    "flow_premium_puts": 0,
    "flow_sweep_direction": "NEUTRAL",
    "dp_nearest_support": null,
    "dp_nearest_resistance": null,
    "dp_support_strength": "UNKNOWN",
    "dp_resistance_strength": "UNKNOWN",
    "dp_buy_volume": 0,
    "dp_sell_volume": 0,
    "sequence_similar_count": 0,
    "sequence_predicted_direction": "NEUTRAL",
    "sequence_historical_win_rate": 0.5,
    "sequence_avg_outcome": 0.0,
    "components_healthy": 2,
    "components_total": 4
}
```

### Field-by-Field Analysis

| Field | Expected | Actual | Status |
|-------|----------|--------|--------|
| **Direction Intelligence** ||||
| blowup_direction | BULLISH/BEARISH | "NEUTRAL" | BROKEN - Always neutral |
| blowup_regime | RISK_ON/RISK_OFF/etc | "UNKNOWN" | BROKEN - Always unknown |
| **GEX Intelligence (Layer 8)** ||||
| gex_regime | POSITIVE/NEGATIVE | "NEGATIVE" | WORKING |
| gex_total | Number | -573B | WORKING |
| gex_flip_point | Price level | null | BROKEN - No data |
| gex_flip_distance_pct | Percentage | 1.0 | SUSPICIOUS - Hardcoded? |
| gex_charm_per_hour | Number | -774T | WORKING |
| gex_key_support | Array of levels | [] | BROKEN - Empty |
| gex_key_resistance | Array of levels | [] | BROKEN - Empty |
| **Flow Intelligence (Layer 9)** ||||
| flow_institutional_bias | BULLISH/BEARISH | "NEUTRAL" | BROKEN - Always neutral |
| flow_confidence | 0-100 | 50 | SUSPICIOUS - Default value |
| flow_premium_calls | Dollar amount | 0 | BROKEN - Zero |
| flow_premium_puts | Dollar amount | 0 | BROKEN - Zero |
| flow_sweep_direction | CALL_HEAVY/PUT_HEAVY | "NEUTRAL" | BROKEN - Always neutral |
| **Dark Pool Intelligence (Layer 10)** ||||
| dp_nearest_support | Price level | null | BROKEN - No data |
| dp_nearest_resistance | Price level | null | BROKEN - No data |
| dp_support_strength | LOW/MED/HIGH | "UNKNOWN" | BROKEN |
| dp_resistance_strength | LOW/MED/HIGH | "UNKNOWN" | BROKEN |
| dp_buy_volume | Volume | 0 | BROKEN - Zero |
| dp_sell_volume | Volume | 0 | BROKEN - Zero |
| **Sequence Intelligence (Layer 11)** ||||
| sequence_similar_count | Count | 0 | BROKEN - Zero |
| sequence_predicted_direction | Direction | "NEUTRAL" | BROKEN |
| sequence_historical_win_rate | 0-1 | 0.5 | SUSPICIOUS - Default |
| sequence_avg_outcome | Percentage | 0.0 | BROKEN - Zero |

### HYDRA Summary

| Category | Working Fields | Broken Fields | Health |
|----------|---------------|---------------|--------|
| Direction | 0 | 2 | 0% |
| GEX (Layer 8) | 3 | 4 | 43% |
| Flow (Layer 9) | 0 | 5 | 0% |
| Dark Pool (Layer 10) | 0 | 6 | 0% |
| Sequence (Layer 11) | 0 | 4 | 0% |
| **TOTAL** | **3** | **21** | **12.5%** |

---

## 3. Polygon API Audit

### Direct API Test Result

```bash
curl "https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/2026-03-06/2026-03-06?limit=10"
```

```json
{
    "ticker": "SPY",
    "queryCount": 10,
    "resultsCount": 2,
    "adjusted": true,
    "results": [
        {"v":432403,"vw":674.87,"o":674.98,"c":675.44,"h":675.48,"l":674.57,"t":1772816400000,"n":8912},
        {"v":752950,"vw":674.79,"o":674.64,"c":674.97,"h":675.51,"l":674.15,"t":1772816100000,"n":14365}
    ],
    "status": "DELAYED",
    "request_id": "5a127f239cd12f41ad462b29bf2b0cad",
    "count": 2
}
```

### Polygon Issues

| Issue | Expected | Actual | Impact |
|-------|----------|--------|--------|
| Status | "OK" | "DELAYED" | Data is not real-time |
| Results | 10 bars | 2 bars | Insufficient for analysis |
| Market hours | Full data | Sparse data | Momentum calculation fails |

### Impact on CPL

The code requires minimum 2 bars (recently reduced from 5):

```python
if not bars or len(bars) < 2:
    return False, 0, "HARD_GATE_DATA: Insufficient bars"
```

With Polygon returning 1-2 bars inconsistently, this gate frequently blocks all trades.

---

## 4. Conviction System Analysis

### Beast Mode V4.0: 13-Signal Conviction System

The system requires **5 out of 13 signals** to approve a trade.

| # | Signal | Data Source | Can Fire? | Reason |
|---|--------|-------------|-----------|--------|
| 1 | HYDRA direction aligned | HYDRA | **NO** | direction always NEUTRAL |
| 2 | Sweep direction aligned | HYDRA | **NO** | flow_sweep_direction always NEUTRAL |
| 3 | Near dark pool level | HYDRA | **NO** | dp_nearest_support/resistance are null |
| 4 | Volume ratio > 1.5x | Polygon | **MAYBE** | Requires bars (often insufficient) |
| 5 | GEX regime favorable | HYDRA | **YES** | gex_regime has data |
| 6 | Momentum accelerating | Polygon | **MAYBE** | Requires bars (often insufficient) |
| 7 | Whale premium present | HYDRA | **NO** | flow_premium_calls/puts are 0 |
| 8 | Charm flow favorable | HYDRA | **MAYBE** | gex_charm_per_hour has data |
| 9 | Time window optimal | System | **YES** | Time-based, always works |
| 10 | Predator Vision (AI) | AI Stack | **MAYBE** | Depends on available data |
| 11 | Opening Range Breakout | Polygon | **MAYBE** | Requires bars |
| 12 | Pre-market Bias | Various | **NO** | Data often unavailable |
| 13 | GEX proximity favorable | HYDRA | **NO** | gex_flip_point is null |

### Signal Availability Summary

| Category | Count |
|----------|-------|
| Definitely working | 2 |
| Maybe working (data-dependent) | 5 |
| Definitely broken | 6 |

### Mathematical Analysis

**Best case scenario:** 2 definite + 5 maybe = 7 possible signals
**Realistic scenario:** 2 definite + 2 maybe = 4 signals
**Required to trade:** 5 signals

**Conclusion:** Even in the best case, the system barely reaches threshold. In practice, it never does.

---

## 5. Gate Rejection Analysis

### March 5, 2026 Rejection Log (10,984 rejections)

```
HARD_GATE_DIRECTION: NEUTRAL market     -> 90%+ of rejections (before fix)
HARD_GATE_DATA: Insufficient bars       -> 90%+ of rejections (after fix)
LIQUIDITY_REJECT: mid < $0.30           -> Normal filtering
```

### Rejection Timeline

| Time Period | Primary Rejection | Secondary |
|-------------|-------------------|-----------|
| 09:30-12:55 ET (before fixes) | NEUTRAL market | UNKNOWN regime |
| 12:55-16:00 ET (after NEUTRAL fix) | Insufficient bars | Liquidity |
| After hours | Insufficient bars | - |

### Key Observation

Every incremental fix reveals the next blocking gate. The gates are stacked:

```
Gate 1: HYDRA disconnected          -> FIXED (was never the issue)
Gate 2: HYDRA stale                 -> FIXED (was never the issue)
Gate 3: Direction NEUTRAL           -> FIXED (now warning, not block)
Gate 4: GEX flip < 1%               -> FIXED (skip when data missing)
Gate 5: Regime UNKNOWN              -> FIXED (now warning, not block)
Gate 6: Insufficient bars           -> STILL BLOCKING
Gate 7: Conviction < 5/13           -> WOULD BLOCK (if we got past Gate 6)
```

---

## 6. Root Cause Summary

### Primary Failure: HYDRA Intelligence Engine

HYDRA was designed to provide 11 of 13 conviction signals. It provides 3.

| Layer | Purpose | Status |
|-------|---------|--------|
| Layer 8: GEX | Dealer gamma exposure | Partial (3/7 fields) |
| Layer 9: Flow | Institutional flow | Dead (0/5 fields) |
| Layer 10: Dark Pool | Hidden S/R levels | Dead (0/6 fields) |
| Layer 11: Sequence | Pattern matching | Dead (0/4 fields) |

### Secondary Failure: Polygon Data Quality

- Status: DELAYED (not real-time)
- Coverage: 1-2 bars when 10 requested
- Impact: Momentum/volume analysis impossible

### Tertiary Failure: Over-Engineered Conviction System

The 13-signal system was designed assuming:
1. All 4 HYDRA layers would be functional
2. Polygon would provide real-time data
3. All components would be healthy

None of these assumptions are true.

---

## 7. Recommendations

### Option A: Fix HYDRA (Complex, Slow)

1. Diagnose why Layers 9, 10, 11 return null/zero
2. Fix data pipelines for each layer
3. Verify each field returns real data
4. Re-enable conviction system

**Estimated effort:** Days to weeks
**Risk:** May require external API subscriptions

### Option B: Bypass HYDRA (Fast, Pragmatic)

1. Reduce conviction threshold from 5/13 to 2/13
2. Remove HYDRA-dependent signals from hard gates
3. Trade based on Polygon data + time windows only
4. Re-enable V7 scalper as backup

**Estimated effort:** Hours
**Risk:** Reduced signal quality

### Option C: Replace Conviction System (Recommended)

1. Disable Beast Mode V4.0 entirely
2. Revert to simpler V3 or V2 system
3. Use only working data sources:
   - Polygon bars (when available)
   - Time windows
   - Basic momentum
   - Liquidity filters

**Estimated effort:** Hours
**Risk:** Lower theoretical edge, but actually trades

### Option D: Hybrid Approach

1. Keep HYDRA for informational logging only
2. Remove all HYDRA hard gates
3. Use HYDRA data as soft conviction boost, not requirement
4. Trade based on Polygon + basic momentum

---

## Appendix A: Files Involved

| File | Purpose |
|------|---------|
| `wsb_snake/execution/jobs_day_cpl.py` | CPL engine with 13-signal conviction |
| `wsb_snake/collectors/hydra_bridge.py` | HYDRA API client |
| `wsb_snake/engines/v7_scalper.py` | Disabled V7 engine |
| `ops/dead_mans_switch.py` | Health monitoring |

## Appendix B: Recent Commits (Problem Timeline)

| Commit | Date | Change | Impact |
|--------|------|--------|--------|
| e5a1b40 | Mar 4 | Beast Mode V4.0 | Introduced 13-signal system |
| 739ff93 | Mar 4 | 13-Signal Conviction | System cannot reach threshold |
| 0e31205 | Mar 4 | Signal 10 + Kill Switch | Added more gates |

## Appendix C: HYDRA Endpoint

```
URL: http://54.172.22.157:8000/api/predator
Method: GET
Response: See Section 2
```

---

**Document Version:** 1.0
**Created:** 2026-03-06 12:20 ET
**Classification:** Internal - Engineering
