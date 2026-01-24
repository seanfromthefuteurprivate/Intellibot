import re
from typing import List, Set
from wsb_snake.utils.logger import log

# Common stock tickers that are frequently mentioned on WSB
KNOWN_TICKERS: Set[str] = {
    # Mega caps
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'BRK', 'JPM',
    # Popular WSB tickers
    'GME', 'AMC', 'BBBY', 'BB', 'NOK', 'PLTR', 'SOFI', 'WISH', 'CLOV', 'SPCE',
    'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'F', 'GM', 'RIVN',
    # ETFs
    'SPY', 'QQQ', 'IWM', 'DIA', 'VXX', 'UVXY', 'SQQQ', 'TQQQ', 'SPXU', 'SPXL',
    'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE',
    # Meme / momentum
    'HOOD', 'COIN', 'RBLX', 'SNOW', 'CRWD', 'NET', 'DDOG', 'ZS', 'OKTA',
    'SQ', 'PYPL', 'SHOP', 'ROKU', 'SPOT', 'SNAP', 'PINS', 'TWTR', 'UBER', 'LYFT',
    # Semiconductors
    'AMD', 'INTC', 'MU', 'QCOM', 'AVGO', 'TXN', 'MRVL', 'ON', 'AMAT', 'LRCX', 'KLAC',
    # Pharma / biotech
    'MRNA', 'PFE', 'JNJ', 'ABBV', 'LLY', 'BMY', 'GILD', 'REGN', 'VRTX', 'BIIB',
    # Energy
    'XOM', 'CVX', 'OXY', 'SLB', 'HAL', 'DVN', 'EOG', 'PXD', 'FANG', 'MPC',
    # Banks / Finance
    'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'SCHW', 'BLK',
    # Retail
    'WMT', 'TGT', 'COST', 'HD', 'LOW', 'TJX', 'ROST', 'DG', 'DLTR',
    # Other popular
    'DIS', 'NFLX', 'CMCSA', 'T', 'VZ', 'TMUS', 'CRM', 'ORCL', 'IBM', 'CSCO',
    'BA', 'CAT', 'DE', 'MMM', 'HON', 'GE', 'RTX', 'LMT', 'NOC',
}

# Words that look like tickers but aren't
FALSE_POSITIVES: Set[str] = {
    'THE', 'AND', 'FOR', 'THIS', 'THAT', 'WITH', 'FROM', 'HAVE', 'BEEN', 'WERE',
    'WILL', 'WHAT', 'WHEN', 'WHERE', 'WHO', 'WHY', 'HOW', 'ALL', 'ANY', 'CAN',
    'HAS', 'HAD', 'WAS', 'ARE', 'BUT', 'NOT', 'YOU', 'YOUR', 'OUR', 'HIS', 'HER',
    'ITS', 'OUT', 'INTO', 'JUST', 'LIKE', 'MAKE', 'MADE', 'MUCH', 'MORE', 'MOST',
    # WSB slang
    'WSB', 'YOLO', 'DD', 'POV', 'CEO', 'IPO', 'WTF', 'IRS', 'USA', 'GDP', 'SEC',
    'FED', 'ATH', 'ATL', 'EOD', 'EOW', 'EOM', 'IMO', 'IMHO', 'TBH', 'FYI', 'LOL',
    'OMG', 'RIP', 'FOMO', 'HODL', 'BTFD', 'GTFO', 'STFU', 'LMAO', 'ROFL',
    'OTM', 'ITM', 'ATM', 'IV', 'DTE', 'FD', 'FDS', 'OI', 'RSI', 'MACD', 'EMA',
    'SMA', 'VWAP', 'PE', 'EPS', 'YOY', 'QOQ', 'MOM', 'DAD', 'BRO', 'SIS',
    'NYC', 'LA', 'SF', 'DC', 'UK', 'EU', 'US', 'CA', 'TX', 'FL', 'NY',
    # Common words
    'NEW', 'OLD', 'BIG', 'TOP', 'LOW', 'HIGH', 'UP', 'DOWN', 'NOW', 'THEN',
    'BEST', 'NEXT', 'LAST', 'WEEK', 'DAY', 'YEAR', 'TIME', 'WAY', 'CALL', 'PUT',
    'BUY', 'SELL', 'LONG', 'SHORT', 'HOLD', 'MOON', 'GAIN', 'LOSS', 'RED', 'GREEN',
}


def extract_tickers(text: str, use_known_only: bool = False) -> List[str]:
    """
    Extract stock tickers from text.
    
    Args:
        text: Raw text to parse
        use_known_only: If True, only return tickers in KNOWN_TICKERS set
        
    Returns:
        List of unique tickers found (uppercase)
    """
    if not text:
        return []
    
    # Pattern: $TICKER or standalone uppercase 2-5 letter words
    # Matches: $AAPL, AAPL, $tsla, etc.
    pattern = r'\$?([A-Za-z]{2,5})\b'
    
    matches = re.findall(pattern, text)
    
    # Normalize to uppercase
    candidates = [m.upper() for m in matches]
    
    # Filter
    tickers = []
    seen = set()
    
    for ticker in candidates:
        if ticker in seen:
            continue
        if ticker in FALSE_POSITIVES:
            continue
        if use_known_only and ticker not in KNOWN_TICKERS:
            continue
        
        # If not using known_only, still require it to look like a ticker
        if not use_known_only:
            # Must be 2-5 uppercase letters
            if not (2 <= len(ticker) <= 5 and ticker.isalpha() and ticker.isupper()):
                continue
            # Must be in known tickers OR the original text had $ prefix
            if ticker not in KNOWN_TICKERS:
                # Check if it was explicitly marked with $
                if f'${ticker}' not in text.upper() and f'$ {ticker}' not in text.upper():
                    continue
        
        tickers.append(ticker)
        seen.add(ticker)
    
    return tickers


def extract_tickers_with_count(text: str) -> dict:
    """
    Extract tickers and count their occurrences.
    
    Returns:
        Dict of {ticker: count}
    """
    if not text:
        return {}
    
    pattern = r'\$?([A-Za-z]{2,5})\b'
    matches = re.findall(pattern, text)
    
    counts = {}
    for m in matches:
        ticker = m.upper()
        if ticker in FALSE_POSITIVES:
            continue
        if ticker not in KNOWN_TICKERS:
            # Only count if it had $ prefix
            if f'${ticker}' not in text.upper():
                continue
        counts[ticker] = counts.get(ticker, 0) + 1
    
    return counts
