# Jobs Report (NFP) – Options Playbook

**NFP rescheduled (BLS/Reuters):** Jan 2026 Employment Situation = **Wed Feb 11, 2026 @ 8:30 AM ET** (not Fri Feb 6).  
**Budget (WeBull):** $250 | **Goal:** Maximize profit on event day with options.  
**Strike mode:** Full event-vol watchlist (index, VIX, rates, dollar, metals, crypto beta, AI/mega). **Tomorrow (and every vol day):** same tickers, any movement → BUY/SELL calls on Telegram; run from 9:30–16:00 ET to maximize capital (0DTE = today’s expiry each day).  
**Telegram test:** `python3 run_snake_cpl.py --broadcast --test-mode` — sends 5 BUY alerts to verify open/close flow.  
**Live tomorrow:** `python3 run_snake_cpl.py --broadcast --loop 60 --untruncated-tails` — start by 9:25 ET.

**Phase 3 cron (9:30 AM every weekday):**
```cron
30 9 * * 1-5 /Users/seankuesia/Downloads/Intellibot/scripts/run_phase3_go_live.sh
```
- **Install:** `crontab -e` → paste the line above (use your actual project path). Save and exit.
- **Confirm:** `crontab -l` — you should see the phase3 line. Cron uses system time (set system to ET for 9:30 ET).

---

## 1. Which stocks are most impacted?

| Ticker | Why it moves on NFP |
|--------|----------------------|
| **SPY** | S&P 500 proxy; broad risk-on/risk-off. Most liquid 0DTE. |
| **QQQ** | Nasdaq; rate-sensitive growth. Big move on surprise. |
| **IWM** | Small caps; high beta to jobs and rates. |
| **TLT** | Long Treasury ETF; rates proxy. Strong NFP → yields up → TLT down. |
| **XLF** | Financials; rate-sensitive. |

**Primary plays for $250:** SPY and QQQ (best liquidity and 0DTE availability).

---

## 2. How to capitalize (strategy)

- **Strong NFP + soft wages** → risk-on → **SPY/QQQ calls** (consider 1–2 OTM call contracts).
- **Weak NFP or hot wages** → risk-off → **SPY/QQQ puts** (consider 1–2 OTM put contracts).
- **Timing:** Wait **2–5 minutes after 8:30** for the first spike, then enter **one** direction (don’t chase both).
- **Expiry:** Use **0DTE** (same-day expiry) or Friday weekly.
- **Target:** +15–25% on the option. **Stop:** -8%. **Max hold:** ~15 minutes (theta kills 0DTE).

---

## 3. System: track now, get buy/sell calls

1. **Run the tracker** (up to daily until Friday):
   ```bash
   python run_jobs_report_tracker.py
   ```
2. **Outputs:**
   - `wsb_snake_data/jobs_report_playbook.json` – watchlist snapshot + suggested plays.
   - `wsb_snake_data/JOBS_REPORT_FEB6.md` – same in markdown (refreshed each run).
3. **Playbook includes:**
   - Ticker, price, change%, ATM strike, call/put ask (ATM and OTM), IV, momentum bias.
   - **Recommended trades:** ticker, direction (call/put), strike, entry (est), target %, stop %, max hold.
4. **WSB Snake / Intellibot:** The tracker uses the same Polygon data and momentum logic as the main snake. For a **single AI “bias” call** (e.g. “favor calls vs puts into the number”), you can run the main snake’s predator stack on SPY/QQQ the morning of Feb 6 and combine with the playbook.

---

## 4. $250 WeBull constraints

- **Max positions:** 1–2 (e.g. one SPY call *or* one QQQ put, or one of each if under budget).
- **Max cost per contract:** ~$125 so two contracts stay under $250.
- Prefer **OTM** options to keep premium low; avoid ATM if too expensive.
- One directional bet **after** the number is simpler and usually better than straddles with $250.

---

## 5. Quick reference

| Item | Value |
|------|--------|
| Event date | 2026-02-06 (Friday) |
| Event time | 8:30 AM ET |
| Watchlist | SPY, QQQ, IWM, TLT, XLF |
| Primary plays | SPY, QQQ |
| Target | +15–25% |
| Stop | -8% |
| Max hold | 15 min |
| Tracker script | `python run_jobs_report_tracker.py` |
| Playbook path | `wsb_snake_data/JOBS_REPORT_FEB6.md` |

*Refresh the playbook by re-running the tracker; it will overwrite the generated files with latest prices and option quotes.*

**If the watchlist is empty:** ensure `POLYGON_API_KEY` is set in `.env` (or environment) and run `python run_jobs_report_tracker.py` again.
