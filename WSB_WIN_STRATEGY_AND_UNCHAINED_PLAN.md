# WSB Win Analysis & Wilder Plan of Action

Study of winning trades from WSB screenshots: logic, execution patterns, and a concrete plan to go harder with the Snake.

---

## 1. WSB Win Patterns (from your screenshots)

### A. **LEAPS / Macro (long-term directional)**

| Trade | Logic | Execution |
|-------|--------|-----------|
| **SLV $40 Call 1/21/28** | Bullish macro on silver; multi-year hold. | Buy LEAPS call, hold through volatility, sell to close when thesis pays (+337%, $1.3k→$5.7k). |
| **SPY Dec 2027 $590 / $690 Calls** | Bullish on broad market long term. | Buy 1–2 year LEAPS, ride trend; +60%, +51% open P&L. |
| **META $620 Call May 2026** | Conviction in mega-cap growth. | LEAPS call, high delta (~0.67), hold for multi-month move; +34.57%. |

**Pattern:** Macro/thesis-driven, long-dated calls (1–3 years). Size once, hold; exit on target or thesis break. Not scalping.

---

### B. **Small-cap / momentum (weeks to months)**

| Trade | Logic | Execution |
|-------|--------|-----------|
| **ASTS, RKLB, MU, INTC, CLSK** (portfolio +$376K/mo) | Concentrated bets on space/semi/theme. | Equities + LEAPS; high conviction, large size; swing when thesis plays. |
| **USAR 155 calls @ $0.86 → ~$6.87** | Sector momentum; “sector slighted down hard” → trim. | Buy cheap calls on momentum name; sell majority on sector weakness (“perfect trims”), let remainder run. 700%+ type move. |
| **LUNR, NBIS, ONDS, RKLB, POET, SLS** (LEAPS 2026/27) | Thematic (space/tech); long-dated options. | Buy 1–2 year calls on high-beta names; hold through chop; +54%, +238%, +97%, +42%, +33% (SLS -11% shows risk). |
| **ENPH $40 Call 3/20/26** | Medium-term bullish on clean energy. | 87 contracts, ~2.5 month hold; +64% total despite bad days. |

**Pattern:** Thematic + momentum. Enter on narrative/catalyst; size up on conviction; trim on sector weakness or target, let runners run.

---

### C. **Index / ETF short-term (daily or weekly)**

| Trade | Logic | Execution |
|-------|--------|-----------|
| **QQQ 21 Jan 26 615 C** | Strong day in QQQ; weekly/near-term call. | Large size; +189% in one day, $49.5K P/L day. |
| **SPY shares + LEAPS** | Core exposure (70 shares) + leveraged LEAPS. | Mix of equity and options; LEAPS for leverage on same thesis. |

**Pattern:** Index/ETF direction + volatility; weekly or 0DTE-style size when setup is there. Quick in/out or same-day hold.

---

### D. **YOLO / single-name conviction**

| Trade | Logic | Execution |
|-------|--------|-----------|
| **PYPL €16K calls** | High conviction on PYPL move; currently -1.24%. | Large single-name bet; accept drawdown; target big payoff. |

**Pattern:** One big bet per idea; size reflects conviction; risk is one name.

---

## 2. Logic and execution (summary)

- **Time horizon:**  
  - **Scalp:** 0DTE / same day (what we do now).  
  - **Momentum:** Days to weeks (USAR-style trim-and-hold, ENPH-style 2–3 month).  
  - **Macro/LEAPS:** Months to years (SLV, SPY, META LEAPS).

- **Entry:**  
  - Scalp: VWAP/pattern/volume (already in Snake).  
  - Momentum: narrative + catalyst + technical (sector/theme + breakout).  
  - Macro: macro thesis + valuation/trend (e.g. silver, indexes, mega-cap).

- **Exit:**  
  - Scalp: target/stop/time (already in Snake).  
  - Momentum: “sector slighted down” → trim majority; target hit → trim; rest runs.  
  - Macro: target % or thesis invalidation; hold through drawdowns.

- **Sizing:**  
  - High conviction = larger size (ASTS $139K, USAR 155 contracts).  
  - Diversify across themes (space, semi, commodity) not just tickers.

- **Risk:**  
  - Single names can blow up (SLS -11%, PYPL currently red).  
  - Sector rotation can hit many positions at once (“sector slighted down”).

---

## 3. Wilder plan of action (concrete)

### 3.1 Keep and sharpen current engine (SCALPER)

- **Already:** 0DTE SPY/QQQ/ETF, VWAP + candlestick + AI, risk governor, position sizing.
- **Go harder:**  
  - Add **QQQ** and **IWM** explicitly to same 0DTE loop (not just SPY).  
  - **Sector-based kill:** If index/sector “slighted down hard” (e.g. VIX spike, sector ETF breakdown), reduce or pause new scalps.  
  - **Trim-on-strength:** Take partial profit at e.g. +15% on 0DTE, let rest run to target/stop.

### 3.2 Add MOMENTUM engine (small-cap / thematic)

- **Goal:** Replicate USAR/LUNR/ONDS style: thematic momentum, trim on sector weakness, let runners run.
- **Universe:** ASTS, RKLB, LUNR, PL, ONDS, POET, SLS, NBIS, ENPH, USAR, THH (already in config).
- **Signals:**  
  - Narrative/catalyst (news, Reddit/WSB buzz, earnings).  
  - Technical: breakout, volume surge, multi-day momentum.  
  - Optional: GPT-4o on weekly chart for “thesis still intact?”
- **Execution:**  
  - Prefer **weekly or 1–2 month options** (not 0DTE) for momentum.  
  - **Trim rules:** e.g. trim 50–70% at +50% or when sector indicator weakens; trail rest.  
  - **Engine tag:** `TradingEngine.MOMENTUM` → governor uses momentum limits (e.g. max 2 positions, per-ticker cap).
- **Risk:** Max per name, max sector exposure (risk governor already has sector caps).

### 3.3 Add MACRO / LEAPS engine (commodity + index LEAPS)

- **Goal:** Replicate SLV/SPY/META style: long-dated calls on macro thesis.
- **Universe:** SLV, GLD, GDX, SPY, QQQ, IWM, META, AAPL, etc. (liquid, macro-tradeable).
- **Signals:**  
  - Macro: inflation/real rates, commodity supply/demand, index trend (e.g. 200-day).  
  - Optional: GPT-4o on monthly chart + “macro thesis” prompt.  
  - Entry on pullback in trend (e.g. dip to 50-day) to buy LEAPS.
- **Execution:**  
  - **LEAPS only** (e.g. 1–2 year expiry).  
  - Size smaller per trade (capital tied longer).  
  - Exit: target (e.g. +50–100%) or thesis break (e.g. trend break, macro regime change).  
  - **Engine tag:** `TradingEngine.MACRO` → governor uses macro limits (e.g. max 2 LEAPS, higher per-trade premium cap).

### 3.4 Cross-cutting “go harder” rules

- **Sector awareness:**  
  - One “sector strength” score (e.g. XLF for financials, XLE for energy, ARKK for growth).  
  - If sector “slighted down hard” (e.g. -2% and volume spike), momentum engine trims or skips new entries in that sector.
- **Conviction sizing:**  
  - Use confidence + thesis strength to scale size (already started with governor’s `compute_position_size`).  
  - For momentum/macro, allow slightly larger max premium per trade for “highest conviction” signals only.
- **Trim-and-hold:**  
  - For non-0DTE: at +X% (e.g. 50%) close half; trail remainder (e.g. breakeven or +20% trailing).
- **Reddit/WSB as filter:**  
  - Use WSB buzz as one input for momentum universe (e.g. ticker heat, not as sole entry).  
  - Avoid buying “after the pump”; prefer early momentum or pullback.

---

## 4. Implementation order (recommended)

1. **Document and config**  
   - This doc.  
   - Add `MOMENTUM_UNIVERSE` and `LEAPS_UNIVERSE` (and optionally `LEAPS_EXPIRY_MONTHS_MIN`) in `config.py`.

2. **Momentum engine (stub + wiring)**  
   - New module e.g. `wsb_snake/engines/momentum_engine.py`:  
     - Scan momentum universe; score by trend + volume + optional catalyst; output candidate + confidence.  
     - On “strike”: call `alpaca_executor.execute_scalp_entry(..., engine=TradingEngine.MOMENTUM)` (or a dedicated `execute_momentum_entry` that uses weekly/next-month options).  
   - Integrate with risk governor (already has `MOMENTUM` limits).

3. **Macro / LEAPS engine (stub + wiring)**  
   - New module e.g. `wsb_snake/engines/leaps_engine.py`:  
     - Scan LEAPS universe; macro + trend filter; optional AI “thesis” check.  
     - On “strike”: open LEAPS only (e.g. 12–24 months out), tag `TradingEngine.MACRO`.  
   - Integrate with risk governor (already has `MACRO` limits).

4. **Sector filter**  
   - Small module or inside engines: fetch sector ETF or index; “sector slighted down” = no new momentum entries in that sector, consider trim.

5. **Trim-and-hold and partial profit**  
   - In executor or learning: for non-0DTE positions, at configurable % gain close part; trail the rest (e.g. zero_greed_exit style for momentum/LEAPS).

---

## 5. How this fits the current Snake

- **Risk governor:** Already has `TradingEngine.SCALPER | MOMENTUM | MACRO`, per-engine and per-sector limits, kill switch, daily PnL. No change needed for “go harder” except possibly tuning limits.
- **Config:** `ZERO_DTE_UNIVERSE` stays; add momentum and LEAPS universes and any new globals (e.g. sector ETF symbols).
- **Executor:** Already has `engine=` and governor; add optional `execute_momentum_entry` / `execute_leaps_entry` (or reuse `execute_scalp_entry` with different expiry/engine) so momentum and LEAPS use correct expiries and tags.
- **Spy_scalper:** Keeps 0DTE only; can add QQQ/IWM in same loop and sector-awareness as above.

---

## 6. One-line summary

**WSB wins:** Scalp (0DTE), momentum (thematic trim-and-hold), macro (LEAPS hold).  
**Wilder plan:** Keep scalp engine, add momentum engine (thematic + trim on sector weakness), add LEAPS engine (macro thesis + long-dated), add sector filter and trim-and-hold logic, and use the existing risk governor so we can go harder without blowing up.
