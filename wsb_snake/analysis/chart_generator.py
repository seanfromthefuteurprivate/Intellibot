"""
Chart Generator - Creates candlestick charts with indicators for AI analysis
"""
import io
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import mplfinance as mpf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class ChartGenerator:
    """Generates candlestick charts with technical indicators for AI vision analysis."""
    
    def __init__(self):
        self.style = mpf.make_mpf_style(
            base_mpf_style='charles',
            marketcolors=mpf.make_marketcolors(
                up='#26a69a',
                down='#ef5350',
                edge='inherit',
                wick='inherit',
                volume='in'
            ),
            figcolor='#1e1e1e',
            facecolor='#1e1e1e',
            edgecolor='#333333',
            gridcolor='#333333',
            gridstyle='-',
            gridaxis='both',
            y_on_right=True,
            rc={
                'font.size': 8,
                'axes.labelcolor': 'white',
                'axes.titlecolor': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white'
            }
        )
    
    def generate_chart(
        self,
        ticker: str,
        ohlcv_data: List[Dict[str, Any]],
        indicators: Optional[Dict[str, Any]] = None,
        timeframe: str = "5min"
    ) -> Optional[str]:
        """
        Generate a candlestick chart and return as base64 encoded image.
        
        Args:
            ticker: Stock ticker symbol
            ohlcv_data: List of OHLCV bars from Polygon
            indicators: Optional dict with RSI, MACD, SMA values
            timeframe: Chart timeframe label
            
        Returns:
            Base64 encoded PNG image string, or None on error
        """
        try:
            if not ohlcv_data or len(ohlcv_data) < 10:
                logger.warning(f"Insufficient data for {ticker} chart")
                return None
            
            df = pd.DataFrame(ohlcv_data)
            
            if 't' in df.columns:
                df['Date'] = pd.to_datetime(df['t'], unit='ms')
            elif 'timestamp' in df.columns:
                df['Date'] = pd.to_datetime(df['timestamp'])
            else:
                logger.warning(f"No timestamp column found for {ticker}")
                return None
            
            df.set_index('Date', inplace=True)
            
            column_map = {
                'o': 'Open', 'open': 'Open',
                'h': 'High', 'high': 'High',
                'l': 'Low', 'low': 'Low',
                'c': 'Close', 'close': 'Close',
                'v': 'Volume', 'volume': 'Volume'
            }
            df.rename(columns=column_map, inplace=True)
            
            required = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required:
                if col not in df.columns:
                    logger.warning(f"Missing {col} column for {ticker}")
                    return None
            
            df = df[required].copy()
            df = df.tail(60)
            
            addplots = []
            
            if indicators:
                if 'sma_20' in indicators and indicators['sma_20']:
                    sma_series = pd.Series(indicators['sma_20'], index=df.index[-len(indicators['sma_20']):])
                    addplots.append(mpf.make_addplot(sma_series, color='yellow', width=1))
                
                if 'sma_50' in indicators and indicators['sma_50']:
                    sma_series = pd.Series(indicators['sma_50'], index=df.index[-len(indicators['sma_50']):])
                    addplots.append(mpf.make_addplot(sma_series, color='blue', width=1))
                
                if 'vwap' in indicators and indicators['vwap']:
                    vwap_series = pd.Series(indicators['vwap'], index=df.index[-len(indicators['vwap']):])
                    addplots.append(mpf.make_addplot(vwap_series, color='purple', width=1.5, linestyle='--'))
            
            buf = io.BytesIO()
            
            plot_kwargs = {
                'type': 'candle',
                'style': self.style,
                'title': f'{ticker} {timeframe} Chart',
                'ylabel': 'Price',
                'volume': True,
                'figsize': (10, 6),
                'returnfig': True,
                'tight_layout': True
            }
            
            if addplots:
                plot_kwargs['addplot'] = addplots
            
            fig, axes = mpf.plot(df, **plot_kwargs)
            
            current_price = df['Close'].iloc[-1]
            high_of_day = df['High'].max()
            low_of_day = df['Low'].min()
            
            axes[0].annotate(
                f'Price: ${current_price:.2f}\nHigh: ${high_of_day:.2f}\nLow: ${low_of_day:.2f}',
                xy=(0.02, 0.98),
                xycoords='axes fraction',
                fontsize=8,
                color='white',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8)
            )
            
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1e1e1e')
            plt.close(fig)
            
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            logger.info(f"Generated chart for {ticker} ({len(df)} bars)")
            return image_base64
            
        except Exception as e:
            logger.error(f"Error generating chart for {ticker}: {e}")
            return None
    
    def generate_multi_timeframe(
        self,
        ticker: str,
        data_5min: List[Dict],
        data_15min: List[Dict],
        data_1hr: List[Dict]
    ) -> Dict[str, Optional[str]]:
        """Generate charts for multiple timeframes."""
        return {
            '5min': self.generate_chart(ticker, data_5min, timeframe="5min"),
            '15min': self.generate_chart(ticker, data_15min, timeframe="15min"),
            '1hr': self.generate_chart(ticker, data_1hr, timeframe="1hr")
        }
