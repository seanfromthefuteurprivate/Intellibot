# Events & Strike Mode

## NFP date (BLS/Reuters)

**Jan 2026 Employment Situation (NFP / unemployment / wages)** is rescheduled due to shutdown:

- **Wed Feb 11, 2026 @ 8:30am ET** — real “jobs-data 0DTE volatility day”
- Fri Feb 6 is **not** NFP day; it can still be volatile (risk-off, earnings, crypto, AI).

System is configured for **Feb 11** as NFP event date. CPL loop runs until **Wed Feb 11, 5 PM ET**.

---

## Strike mode: capture all events

**Strike mode** = full event-vol watchlist + Telegram BUY/SELL alerts so the system captures all macro/WSB events.

### Watchlist (index, vol, rates, dollar, metals, crypto beta, AI/mega, sectors)

| Category | Tickers |
|----------|--------|
| Index | SPY, QQQ, IWM, DIA |
| Vol (panic meter) | VXX, UVXY |
| Rates | TLT, IEF, XLF |
| Dollar / metals | UUP, GLD, SLV, GDX |
| Crypto beta (WSB focus) | MSTR, COIN, MARA, RIOT |
| AI / mega-cap | NVDA, TSLA, AAPL, AMZN, META, GOOGL, MSFT, AMD |
| Sectors | ITB, XHB, XLY, XLV |

### What to watch

- **Feb 6 (no NFP):** Is the driver equities, rates, or crypto? QQQ vs TLT; MSTR vs BTC; VXX/UVXY.
- **Feb 11 (NFP):** First 1–10 min after 8:30am — 2Y yield, QQQ vs IWM, UUP, VIX products.

---

## Telegram alerts: open/close testing

To verify Telegram BUY/SELL flow (open/close, etc.):

```bash
cd /path/to/Intellibot
python3 run_snake_cpl.py --broadcast --test-mode
```

- Generates calls (dry_run internally), then **sends 5 execution-complete BUY messages** to Telegram.
- No DB writes; no positions opened. Use to confirm alerts deliver.

Single dry-run (no Telegram):

```bash
python3 run_snake_cpl.py --dry-run --max-calls 3 --untruncated-tails
```

Live run (Telegram + DB, loop until event day 5 PM ET):

```bash
python3 run_snake_cpl.py --broadcast --loop 60 --untruncated-tails
```

---

## Cron (phase2 / phase3)

- **Phase2:** dry-run, 3 calls — e.g. pre-open check.
- **Phase3:** broadcast, loop 60s, untruncated-tails — full send from market open to close on event day.

Start phase3 by **9:25 ET** on Wed Feb 11 so CPL is live at 9:30.
