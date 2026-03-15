"""
FRESH BLOOD — Configuration

Simple. Verified. No bullshit.
"""

import os

# Alpaca Credentials (Paper Trading Account)
# Note: Using hardcoded values to avoid env var conflicts
# Change to your live credentials before production
ALPACA_KEY = "PKWT6T5BFKHBTFDW3CPAFW2XBZ"
ALPACA_SECRET = "pVdzbVte2pQvL1RmCTFw3oaQ6TBWYimAzC42DUyTEy8"
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Use paper trading by default
USE_PAPER = True
ALPACA_BASE_URL = ALPACA_PAPER_URL if USE_PAPER else ALPACA_LIVE_URL

# Tickers to scan for gaps
TICKERS = [
    "COIN", "MARA", "RIOT",  # Crypto proxies - volatile
    "HOOD", "SMCI",          # High beta tech
    "AMD", "NVDA",           # Semiconductors
    "TSLA", "META",          # Mega cap volatile
    "GOOGL", "AMZN"          # Mega cap
]

# === STRATEGY RULES (VERIFIED WITH REAL DATA) ===
#
# What we learned from 8 verified trades:
# - Gap UPs 5-10% CONTINUE up (6 losers) → SKIP
# - Gap DOWNs 5%+ FADE (HOOD -6% → +73%) → TRADE
# - Gap UPs 10%+ FADE (MARA +14.6% → +142%) → TRADE
#
# V2 Rules applied to history: 2/2 wins, +215% total
# V1 Rules (any 5%+ gap): 2/8 wins, -25% total

GAP_DOWN_MIN = -0.05      # -5% gap down = buy CALLS
GAP_UP_MIN = 0.10         # +10% gap up = buy PUTS
# Gap UPs between 5-10% = SKIP (danger zone)

# Position sizing
POSITION_SIZE_USD = 2500  # $2,500 per trade
MAX_POSITIONS = 2         # Max 2 concurrent positions
MAX_DAILY_LOSS = -500     # Stop trading if down $500 in a day

# Risk management
STOP_LOSS_PCT = 0.40      # -40% hard stop
PROFIT_TARGET_PCT = 1.00  # +100% take profit (optional)

# Timing
ENTRY_WINDOW_START = "09:40"  # Don't enter before 9:40 AM ET
ENTRY_WINDOW_END = "10:00"    # Don't enter after 10:00 AM ET
EXIT_TIME = "15:50"           # Exit all positions by 3:50 PM ET

# Logging
LOG_FILE = "fresh_blood.log"
TRADES_FILE = "fresh_blood_trades.json"
