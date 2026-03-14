#!/usr/bin/env python3
"""
WSB JUPITER DIRECT - GAP SCANNER
================================
Scans all tickers for 5%+ overnight gaps at market open.
Returns ranked list of tradeable gap fade opportunities.
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class GapScanner:
    """Scans for tradeable gap fade setups."""

    # Universe of volatile stocks for gap scanning
    UNIVERSE = [
        'TSLA', 'NVDA', 'AMD', 'COIN', 'MARA', 'RIOT', 'PLTR', 'SOFI',
        'GME', 'AMC', 'HOOD', 'SMCI', 'ARM', 'SNAP', 'SQ', 'SHOP',
        'RBLX', 'DKNG', 'META', 'GOOGL', 'AAPL', 'MSFT', 'AMZN', 'NFLX',
        'BABA', 'NIO', 'LCID', 'RIVN', 'UBER', 'LYFT', 'ABNB', 'CRWD',
        'ZM', 'DOCU', 'ROKU', 'SQ', 'PYPL', 'AFRM', 'UPST', 'OPEN'
    ]

    MIN_GAP_PCT = 5.0  # Minimum gap size to trade
    MIN_PRICE = 5.0    # Minimum stock price (avoid penny stocks)
    MIN_VOLUME = 500000  # Minimum average volume

    def __init__(self, alpaca_key: str, alpaca_secret: str):
        self.api_key = alpaca_key
        self.api_secret = alpaca_secret
        self.headers = {
            'APCA-API-KEY-ID': alpaca_key,
            'APCA-API-SECRET-KEY': alpaca_secret
        }
        self.base_url = 'https://data.alpaca.markets/v2'

    def get_previous_close(self, ticker: str, date: str) -> Optional[float]:
        """Get previous trading day's close price."""
        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
            prev = dt - timedelta(days=1)
            # Skip weekends
            while prev.weekday() >= 5:
                prev -= timedelta(days=1)
            prev_str = prev.strftime("%Y-%m-%d")

            url = f"{self.base_url}/stocks/{ticker}/bars"
            params = {
                'start': f'{prev_str}T00:00:00-04:00',
                'end': f'{date}T09:30:00-04:00',
                'timeframe': '1Day',
                'limit': 5
            }
            r = requests.get(url, headers=self.headers, params=params, timeout=5)
            if r.status_code == 200:
                bars = r.json().get('bars', [])
                if bars:
                    return bars[-1]['c']
            return None
        except Exception:
            return None

    def get_opening_price(self, ticker: str, date: str) -> Optional[Dict]:
        """Get opening price and first 30 min data."""
        try:
            url = f"{self.base_url}/stocks/{ticker}/bars"
            params = {
                'start': f'{date}T09:30:00-04:00',
                'end': f'{date}T10:00:00-04:00',
                'timeframe': '1Min',
                'limit': 30
            }
            r = requests.get(url, headers=self.headers, params=params, timeout=5)
            if r.status_code == 200:
                bars = r.json().get('bars', [])
                if bars:
                    return {
                        'open': bars[0]['o'],
                        'high': max(b['h'] for b in bars),
                        'low': min(b['l'] for b in bars),
                        'close': bars[-1]['c'],
                        'volume': sum(b['v'] for b in bars),
                        'bars': bars
                    }
            return None
        except Exception:
            return None

    def calculate_gap(self, prev_close: float, open_price: float) -> float:
        """Calculate gap percentage."""
        return ((open_price - prev_close) / prev_close) * 100

    def analyze_volume_profile(self, bars: List[Dict]) -> Dict:
        """Analyze volume decay pattern (key for gap fade)."""
        if len(bars) < 6:
            return {'decay': 0, 'pattern': 'UNKNOWN'}

        # Compare first 5min volume to last 5min volume
        first_vol = sum(b['v'] for b in bars[:5])
        last_vol = sum(b['v'] for b in bars[-5:])

        if first_vol > 0:
            decay = ((first_vol - last_vol) / first_vol) * 100
        else:
            decay = 0

        # Declining volume = Gap and Crap (fadeable)
        # Sustained volume = Gap and Go (avoid)
        if decay > 50:
            pattern = 'GAP_AND_CRAP'
        elif decay > 20:
            pattern = 'MODERATE_DECAY'
        elif decay < -20:
            pattern = 'GAP_AND_GO'
        else:
            pattern = 'NEUTRAL'

        return {'decay': decay, 'pattern': pattern}

    def scan(self, date: str = None) -> List[Dict]:
        """
        Scan all tickers for gap fade opportunities.

        Args:
            date: Date to scan (YYYY-MM-DD format). Defaults to today.

        Returns:
            List of gap opportunities, ranked by gap size.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        gaps = []

        for ticker in self.UNIVERSE:
            try:
                # Get previous close
                prev_close = self.get_previous_close(ticker, date)
                if not prev_close or prev_close < self.MIN_PRICE:
                    continue

                # Get opening data
                opening = self.get_opening_price(ticker, date)
                if not opening:
                    continue

                # Calculate gap
                gap_pct = self.calculate_gap(prev_close, opening['open'])

                # Filter by gap size
                if abs(gap_pct) < self.MIN_GAP_PCT:
                    continue

                # Analyze volume
                vol_analysis = self.analyze_volume_profile(opening['bars'])

                # Determine trade direction (FADE the gap)
                if gap_pct > 0:
                    direction = 'PUT'  # Gap up = buy puts to fade
                    target = prev_close  # Target is gap fill
                else:
                    direction = 'CALL'  # Gap down = buy calls to fade
                    target = prev_close

                # Skip Gap and Go patterns (they continue, don't fade)
                if vol_analysis['pattern'] == 'GAP_AND_GO':
                    continue

                gaps.append({
                    'ticker': ticker,
                    'date': date,
                    'gap_pct': gap_pct,
                    'direction': direction,
                    'prev_close': prev_close,
                    'open': opening['open'],
                    'current': opening['close'],
                    'target': target,
                    'volume_decay': vol_analysis['decay'],
                    'pattern': vol_analysis['pattern'],
                    'first_30_high': opening['high'],
                    'first_30_low': opening['low'],
                    'first_30_volume': opening['volume']
                })

            except Exception as e:
                continue

        # Rank by gap size (biggest first = most extended = best fade)
        gaps.sort(key=lambda x: abs(x['gap_pct']), reverse=True)

        return gaps

    def get_best_trade(self, date: str = None) -> Optional[Dict]:
        """Get the single best gap fade trade for the day."""
        gaps = self.scan(date)
        if gaps:
            return gaps[0]
        return None


if __name__ == '__main__':
    import os

    # Test with environment variables
    key = os.environ.get('ALPACA_API_KEY')
    secret = os.environ.get('ALPACA_SECRET_KEY')

    if key and secret:
        scanner = GapScanner(key, secret)
        gaps = scanner.scan()

        print("="*60)
        print("GAP SCANNER RESULTS")
        print("="*60)

        for g in gaps[:10]:
            print(f"\n{g['ticker']}: Gap {g['gap_pct']:+.1f}% -> {g['direction']}")
            print(f"  Pattern: {g['pattern']}")
            print(f"  Volume Decay: {g['volume_decay']:.1f}%")
            print(f"  Target: ${g['target']:.2f}")
    else:
        print("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
