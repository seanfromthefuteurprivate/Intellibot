import os
from dotenv import load_dotenv

load_dotenv()

# Reddit
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "python:wsb-snake:v1.0")

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Alpaca (Paper Trading + News)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Polygon.io (Options Chain Data)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_BASE_URL = "https://api.polygon.io"

# Benzinga (News)
BENZINGA_API_KEY = os.getenv("BENZINGA_API_KEY")
BENZINGA_BASE_URL = "https://api.benzinga.com/api/v2"

# 0DTE Universe - Tickers to monitor for options activity
ZERO_DTE_UNIVERSE = [
    "SPY", "QQQ", "IWM",  # Index ETFs with daily 0DTE
    "SLV", "GLD", "GDX", "GDXJ",  # Metals ETFs - high volatility plays
    "USO", "XLE",  # Energy ETFs
    "TSLA", "NVDA", "AAPL", "META", "AMD", "AMZN", "GOOGL", "MSFT",  # Mega caps
    "THH",  # Reddit watchlist - potential breakout
    # WSB core holdings (space/AI/autonomous)
    "RKLB", "ASTS", "NBIS", "PL", "LUNR", "ONDS", "SLS",
    # WSB YOLO plays
    "POET", "ENPH", "USAR", "PYPL"
]

# ETFs with daily 0DTE options (not just Friday weeklies)
DAILY_0DTE_TICKERS = ["SPY", "QQQ", "IWM", "SLV", "GLD", "USO", "XLE"]

# WSB Wilder Plan: Momentum (small-cap / thematic) – weeks to months
# Trim on sector weakness; let runners run (see WSB_WIN_STRATEGY_AND_UNCHAINED_PLAN.md)
MOMENTUM_UNIVERSE = [
    "RKLB", "ASTS", "LUNR", "PL", "ONDS", "POET", "SLS", "NBIS",
    "ENPH", "USAR", "THH", "CLSK", "MU", "INTC",
]

# WSB Wilder Plan: LEAPS / Macro – 1–3 year thesis (commodity + index + mega-cap)
LEAPS_UNIVERSE = [
    "SLV", "GLD", "GDX", "GDXJ", "USO", "XLE",  # Commodities / energy
    "SPY", "QQQ", "IWM",  # Index ETFs
    "META", "AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "PYPL",  # Mega caps
]
LEAPS_EXPIRY_MONTHS_MIN = 12  # Only consider options with >= 12 months to expiry

# Session time windows (Eastern Time)
SESSION_WINDOWS = {
    "premarket": (4, 0, 9, 30),      # 4:00 AM - 9:30 AM ET
    "open": (9, 30, 10, 30),          # 9:30 AM - 10:30 AM ET (first hour)
    "morning": (10, 30, 12, 0),       # 10:30 AM - 12:00 PM ET
    "lunch": (12, 0, 13, 0),          # 12:00 PM - 1:00 PM ET (chop zone)
    "power_hour_early": (13, 0, 15, 0),  # 1:00 PM - 3:00 PM ET
    "power_hour": (15, 0, 16, 0),     # 3:00 PM - 4:00 PM ET (final hour)
    "afterhours": (16, 0, 20, 0),     # 4:00 PM - 8:00 PM ET
}

# Database
DB_PATH = "wsb_snake.db"
