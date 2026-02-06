# Why Zero Trades? Is Screenshot Learning Doing Anything?

## 1. Is screenshot learning actually affecting trading?

**Only when there are “active recipes.”**

- The system **does use** screenshot learning: `trade_learner.get_confidence_adjustment()` is called from the **spy scalper** and the **probability generator**. When a current setup (ticker, call/put, time window, pattern) matches a **recipe**, it **adds up to +15% confidence** so more setups can clear the trade threshold.
- **Recipes** come only from **learned_trades** in the DB: trades extracted from screenshots that are **winners with >10% gain**. A recipe is created/updated per pattern (e.g. `SPY_CALLS_breakout_power_hour`). The **Trade Learner only loads recipes that have `source_trade_count >= 2`** (at least two winning trades for that pattern).
- So:
  - **0 screenshots processed** → 0 learned_trades → **0 recipes** → screenshot learning adds **nothing** (boost is always 0).
  - **Only 1 winning trade** for a pattern → 1 recipe row but **not loaded** (need ≥2) → still **no boost**.
  - **2+ winning trades** for the same pattern → recipe is active → matching live setups get a **confidence boost**.

**Bottom line:** Screenshot learning “does something” only when you have at least **two winning (>10% gain) trades** per pattern in the DB. Until then, the system runs normally but screenshot data does not change any trade decision.

---

## 2. Why zero trades executed on Alpaca today?

Trades are placed in two ways:

1. **Spy scalper** (runs every 30s during market hours): when a setup passes **all** gates, it calls `alpaca_executor.execute_scalp_entry()`.
2. **Orchestrator** (scheduled pipeline): when a probability score is high enough and other checks pass, it can also execute.

For the **spy scalper** (main execution path), **all** of the following must be true or the system does **not** place a trade:

| Gate | What it means |
|------|----------------|
| **Market open** | `is_market_open()` is True (9:30–16:00 ET). Outside that, the scalper doesn’t run scans. |
| **Confidence ≥ 85%** | After all boosts (pattern, time, **screenshot recipes**), `final_confidence >= MIN_CONFIDENCE_FOR_ALERT` (85%). |
| **AI confirmation** | `REQUIRE_AI_CONFIRMATION = True`: Predator Stack (GPT-4o/DeepSeek) must return STRIKE_CALLS or STRIKE_PUTS. |
| **Order flow agrees** | Sweep direction and sweep % (≥8%) must align with the setup direction. |
| **Sector not slighted** | SPY not weak (no “sector slighted down” skip). |
| **No earnings soon** | No earnings within 2 days for the ticker (IV crush risk). |
| **Regime match** | No long in trend_down, no short in trend_up. |
| **Cooldown** | 20-minute cooldown per ticker after a trade. |
| **SNIPER mode** | Only the **single best** setup per 30s cycle is sent to AI; if that one fails any gate, no trade. |

So: **the system is “working” (scanning, AI, alerts pipeline) but “doing nothing” on Alpaca** when **no setup clears every gate**. That’s by design: the bar is high on purpose (fewer, higher-quality trades).

---

## 3. How to see what’s going on

**API (on the app/droplet):**

- **`GET /status`**  
  Now includes a **`diagnostics`** block:
  - `market_open`
  - `session`
  - `min_confidence_for_alert` (85)
  - `require_ai_confirmation`
  - **`screenshot_active_recipes`** – if this is **0**, screenshot learning is not changing any confidence.
  - **`why_no_trades`** – short human-readable summary (e.g. “Market closed” or “No setup cleared all gates…”).

- **`GET /screenshot-learning`**  
  Shows how many screenshots were processed, how many learned_trades and **active recipes** there are. If `active_recipes_count` is 0, screenshot learning is not affecting trading yet.

**Logs (on the droplet):**

- When a setup almost trades but is rejected, the scalper logs the reason, e.g.:
  - `❌ {ticker} {pattern} @ {conf}% below alert threshold`
  - `⏸️ Flow disagrees – skipping scalp`
  - `⏸️ Sector slighted down – skipping scalp`
  - `⏸️ AI not confirmed`
- So: `journalctl -u wsb-snake -f` (or your log sink) shows **why** no trade was taken.

---

## 4. Summary

| Question | Answer |
|----------|--------|
| Is screenshot learning doing anything? | **Only if** there are **active recipes** (≥2 winning trades per pattern). Check **`/status` → `diagnostics.screenshot_active_recipes`** or **`/screenshot-learning` → `active_recipes_count`**. If 0, it’s not affecting trades. |
| Why zero trades today? | **No setup passed every gate**: 85%+ confidence, AI confirm, flow agree, sector/earnings/regime/cooldown. **`/status` → `diagnostics.why_no_trades`** gives a one-line summary. |
| System “working” but “doing nothing”? | Yes: scanning and AI run, but by design trades only when a setup is strong enough and all filters pass. Zero trades on a given day is normal when that never happens. |

To get screenshot learning to matter: add enough **winning** trade screenshots to the Drive folder so the system can build recipes with **≥2** wins per pattern. To get more trades (if you want): lower `MIN_CONFIDENCE_FOR_ALERT` or relax other gates in code (trade-off: more trades, lower quality).
