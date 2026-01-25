"""
Scalp Chart Generator - Specialized charts for 0DTE surgical precision
Generates VWAP bands, volume profile, delta bars for apex predator analysis
"""
import io
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


class ScalpChartGenerator:
    """
    Generates surgical precision charts for 0DTE scalping.
    
    Chart Types:
    1. VWAP Bands Chart - Shows price with VWAP ±1σ, ±2σ bands
    2. Volume Profile Chart - Horizontal histogram showing volume at price levels
    3. Delta Bars Chart - Net buying vs selling pressure per candle
    4. Combined Predator Chart - All-in-one view for AI analysis
    """
    
    def __init__(self):
        self.colors = {
            'bg': '#0d1117',
            'grid': '#21262d',
            'text': '#c9d1d9',
            'up': '#26a69a',
            'down': '#ef5350',
            'vwap': '#f0b90b',
            'band1': '#3fb950',
            'band2': '#f85149',
            'volume_up': '#26a69a80',
            'volume_down': '#ef535080',
            'delta_positive': '#26a69a',
            'delta_negative': '#ef5350',
            'profile': '#58a6ff'
        }
    
    def _prepare_dataframe(self, ohlcv_data: List[Dict]) -> Optional[pd.DataFrame]:
        """Convert OHLCV data to DataFrame with proper formatting."""
        try:
            if not ohlcv_data or len(ohlcv_data) < 10:
                return None
            
            df = pd.DataFrame(ohlcv_data)
            
            if 't' in df.columns:
                df['Date'] = pd.to_datetime(df['t'], unit='ms')
            elif 'timestamp' in df.columns:
                df['Date'] = pd.to_datetime(df['timestamp'])
            else:
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
                    return None
            
            result = df[required].copy()
            
            # Drop NaN rows and validate data
            result = result.dropna()
            if len(result) < 10:
                return None
            
            # Ensure numeric types
            for col in required:
                result[col] = pd.to_numeric(result[col], errors='coerce')
            
            result = result.dropna()
            if len(result) < 10:
                return None
                
            return result
            
        except Exception as e:
            logger.error(f"Error preparing dataframe: {e}")
            return None
    
    def _calculate_vwap_bands(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate VWAP and standard deviation bands.
        
        Returns:
            vwap, upper_band1, lower_band1, upper_band2, lower_band2
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        cumulative_tp_vol = (typical_price * df['Volume']).cumsum()
        cumulative_vol = df['Volume'].cumsum()
        
        vwap = cumulative_tp_vol / cumulative_vol
        
        squared_diff = ((typical_price - vwap) ** 2 * df['Volume']).cumsum()
        variance = squared_diff / cumulative_vol
        std_dev = np.sqrt(variance)
        
        upper_band1 = vwap + std_dev
        lower_band1 = vwap - std_dev
        upper_band2 = vwap + (2 * std_dev)
        lower_band2 = vwap - (2 * std_dev)
        
        return vwap, upper_band1, lower_band1, upper_band2, lower_band2
    
    def _calculate_delta(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate delta (buying vs selling pressure) per bar.
        Approximated from price action when tick data unavailable.
        
        Positive delta = more buying pressure
        Negative delta = more selling pressure
        """
        body = df['Close'] - df['Open']
        range_size = df['High'] - df['Low']
        
        body_ratio = body / range_size.replace(0, np.nan)
        body_ratio = body_ratio.fillna(0)
        
        delta = body_ratio * df['Volume']
        
        return delta
    
    def _calculate_volume_profile(self, df: pd.DataFrame, num_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate volume profile (volume at price levels).
        
        Returns:
            price_levels, volumes_at_levels
        """
        price_min = df['Low'].min()
        price_max = df['High'].max()
        
        price_bins = np.linspace(price_min, price_max, num_bins + 1)
        volumes = np.zeros(num_bins)
        
        for _, row in df.iterrows():
            bar_low = row['Low']
            bar_high = row['High']
            bar_vol = row['Volume']
            
            for i in range(num_bins):
                bin_low = price_bins[i]
                bin_high = price_bins[i + 1]
                
                if bar_high >= bin_low and bar_low <= bin_high:
                    overlap = min(bar_high, bin_high) - max(bar_low, bin_low)
                    bar_range = bar_high - bar_low
                    if bar_range > 0:
                        volumes[i] += bar_vol * (overlap / bar_range)
        
        price_levels = (price_bins[:-1] + price_bins[1:]) / 2
        
        return price_levels, volumes
    
    def generate_vwap_bands_chart(
        self,
        ticker: str,
        ohlcv_data: List[Dict],
        timeframe: str = "1min"
    ) -> Optional[str]:
        """
        Generate VWAP bands chart for scalping analysis.
        Shows VWAP with ±1σ and ±2σ bands - key for bounce/rejection detection.
        """
        df = self._prepare_dataframe(ohlcv_data)
        if df is None:
            return None
        
        try:
            df = df.tail(60)
            vwap, ub1, lb1, ub2, lb2 = self._calculate_vwap_bands(df)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                           gridspec_kw={'height_ratios': [3, 1]},
                                           facecolor=self.colors['bg'])
            ax1.set_facecolor(self.colors['bg'])
            ax2.set_facecolor(self.colors['bg'])
            
            x = range(len(df))
            
            ax1.fill_between(x, ub2, lb2, alpha=0.1, color=self.colors['band2'], label='±2σ Band')
            ax1.fill_between(x, ub1, lb1, alpha=0.2, color=self.colors['band1'], label='±1σ Band')
            
            ax1.plot(x, vwap, color=self.colors['vwap'], linewidth=2, label='VWAP', linestyle='--')
            
            for i, (idx, row) in enumerate(df.iterrows()):
                color = self.colors['up'] if row['Close'] >= row['Open'] else self.colors['down']
                ax1.plot([i, i], [row['Low'], row['High']], color=color, linewidth=1)
                ax1.add_patch(Rectangle(
                    (i - 0.3, min(row['Open'], row['Close'])),
                    0.6,
                    abs(row['Close'] - row['Open']),
                    facecolor=color,
                    edgecolor=color
                ))
            
            current_price = df['Close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            
            distance_from_vwap = ((current_price - current_vwap) / current_vwap) * 100
            position = "ABOVE" if current_price > current_vwap else "BELOW"
            
            ax1.axhline(y=current_price, color='white', linestyle=':', alpha=0.5)
            
            info_text = f"Price: ${current_price:.2f} | VWAP: ${current_vwap:.2f}\n"
            info_text += f"{position} VWAP by {abs(distance_from_vwap):.2f}%"
            
            ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                    fontsize=10, color=self.colors['text'], verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=self.colors['grid'], alpha=0.8))
            
            ax1.set_title(f'{ticker} VWAP Bands ({timeframe})', color=self.colors['text'], fontsize=12)
            ax1.legend(loc='upper right', facecolor=self.colors['grid'], labelcolor=self.colors['text'])
            ax1.grid(True, alpha=0.3, color=self.colors['grid'])
            ax1.tick_params(colors=self.colors['text'])
            ax1.set_ylabel('Price', color=self.colors['text'])
            
            colors = [self.colors['up'] if df['Close'].iloc[i] >= df['Open'].iloc[i] 
                     else self.colors['down'] for i in range(len(df))]
            ax2.bar(x, df['Volume'], color=colors, alpha=0.7)
            ax2.set_ylabel('Volume', color=self.colors['text'])
            ax2.grid(True, alpha=0.3, color=self.colors['grid'])
            ax2.tick_params(colors=self.colors['text'])
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=self.colors['bg'])
            plt.close(fig)
            buf.seek(0)
            
            logger.info(f"Generated VWAP bands chart for {ticker}")
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error generating VWAP bands chart: {e}")
            return None
    
    def generate_delta_chart(
        self,
        ticker: str,
        ohlcv_data: List[Dict],
        timeframe: str = "1min"
    ) -> Optional[str]:
        """
        Generate delta bars chart showing buying vs selling pressure.
        Critical for detecting momentum shifts and traps.
        """
        df = self._prepare_dataframe(ohlcv_data)
        if df is None:
            return None
        
        try:
            df = df.tail(60)
            delta = self._calculate_delta(df)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                           gridspec_kw={'height_ratios': [2, 1]},
                                           facecolor=self.colors['bg'])
            ax1.set_facecolor(self.colors['bg'])
            ax2.set_facecolor(self.colors['bg'])
            
            x = range(len(df))
            
            for i, (idx, row) in enumerate(df.iterrows()):
                color = self.colors['up'] if row['Close'] >= row['Open'] else self.colors['down']
                ax1.plot([i, i], [row['Low'], row['High']], color=color, linewidth=1)
                ax1.add_patch(Rectangle(
                    (i - 0.3, min(row['Open'], row['Close'])),
                    0.6,
                    abs(row['Close'] - row['Open']) or 0.01,
                    facecolor=color,
                    edgecolor=color
                ))
            
            ax1.set_title(f'{ticker} Price Action ({timeframe})', color=self.colors['text'], fontsize=12)
            ax1.grid(True, alpha=0.3, color=self.colors['grid'])
            ax1.tick_params(colors=self.colors['text'])
            ax1.set_ylabel('Price', color=self.colors['text'])
            
            delta_colors = [self.colors['delta_positive'] if d >= 0 else self.colors['delta_negative'] 
                           for d in delta]
            ax2.bar(x, delta, color=delta_colors, alpha=0.8)
            ax2.axhline(y=0, color='white', linewidth=0.5, alpha=0.5)
            
            cumulative_delta = delta.cumsum()
            ax2_twin = ax2.twinx()
            ax2_twin.plot(x, cumulative_delta, color='#f0b90b', linewidth=1.5, label='Cumulative Delta')
            ax2_twin.tick_params(colors='#f0b90b')
            
            net_delta = delta.iloc[-10:].sum()
            delta_trend = "BUYERS" if net_delta > 0 else "SELLERS"
            ax2.text(0.02, 0.95, f"Net Delta (10 bars): {delta_trend} in control",
                    transform=ax2.transAxes, fontsize=9, color=self.colors['text'],
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=self.colors['grid'], alpha=0.8))
            
            ax2.set_title('Delta (Buying vs Selling Pressure)', color=self.colors['text'], fontsize=10)
            ax2.set_ylabel('Delta', color=self.colors['text'])
            ax2.grid(True, alpha=0.3, color=self.colors['grid'])
            ax2.tick_params(colors=self.colors['text'])
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=self.colors['bg'])
            plt.close(fig)
            buf.seek(0)
            
            logger.info(f"Generated delta chart for {ticker}")
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error generating delta chart: {e}")
            return None
    
    def generate_volume_profile_chart(
        self,
        ticker: str,
        ohlcv_data: List[Dict],
        timeframe: str = "1min"
    ) -> Optional[str]:
        """
        Generate volume profile chart showing volume at each price level.
        Shows where trapped buyers/sellers create squeeze fuel.
        """
        df = self._prepare_dataframe(ohlcv_data)
        if df is None:
            return None
        
        try:
            df = df.tail(60)
            price_levels, volumes = self._calculate_volume_profile(df, num_bins=25)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                           gridspec_kw={'width_ratios': [3, 1]},
                                           facecolor=self.colors['bg'])
            ax1.set_facecolor(self.colors['bg'])
            ax2.set_facecolor(self.colors['bg'])
            
            x = range(len(df))
            for i, (idx, row) in enumerate(df.iterrows()):
                color = self.colors['up'] if row['Close'] >= row['Open'] else self.colors['down']
                ax1.plot([i, i], [row['Low'], row['High']], color=color, linewidth=1)
                ax1.add_patch(Rectangle(
                    (i - 0.3, min(row['Open'], row['Close'])),
                    0.6,
                    abs(row['Close'] - row['Open']) or 0.01,
                    facecolor=color,
                    edgecolor=color
                ))
            
            ax1.set_title(f'{ticker} Price Action ({timeframe})', color=self.colors['text'], fontsize=12)
            ax1.grid(True, alpha=0.3, color=self.colors['grid'])
            ax1.tick_params(colors=self.colors['text'])
            ax1.set_ylabel('Price', color=self.colors['text'])
            
            max_vol = volumes.max()
            normalized_vol = volumes / max_vol if max_vol > 0 else volumes
            
            poc_idx = np.argmax(volumes)
            poc_price = price_levels[poc_idx]
            
            for i, (price, vol) in enumerate(zip(price_levels, normalized_vol)):
                color = '#f0b90b' if i == poc_idx else self.colors['profile']
                ax2.barh(price, vol, height=(price_levels[1] - price_levels[0]) * 0.9,
                        color=color, alpha=0.7)
            
            ax1.axhline(y=poc_price, color='#f0b90b', linestyle='--', alpha=0.7, label=f'POC: ${poc_price:.2f}')
            ax1.legend(loc='upper left', facecolor=self.colors['grid'], labelcolor=self.colors['text'])
            
            high_vol_threshold = np.percentile(volumes, 80)
            high_vol_zones = price_levels[volumes >= high_vol_threshold]
            
            zone_text = f"POC: ${poc_price:.2f}\nHigh Volume Zones:\n"
            for zone in high_vol_zones[:3]:
                zone_text += f"  ${zone:.2f}\n"
            
            ax2.text(0.98, 0.98, zone_text, transform=ax2.transAxes,
                    fontsize=9, color=self.colors['text'], verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor=self.colors['grid'], alpha=0.8))
            
            ax2.set_title('Volume Profile', color=self.colors['text'], fontsize=10)
            ax2.set_xlabel('Relative Volume', color=self.colors['text'])
            ax2.tick_params(colors=self.colors['text'])
            ax2.grid(True, alpha=0.3, color=self.colors['grid'])
            
            ax1.set_ylim(price_levels.min() - 0.5, price_levels.max() + 0.5)
            ax2.set_ylim(price_levels.min() - 0.5, price_levels.max() + 0.5)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=self.colors['bg'])
            plt.close(fig)
            buf.seek(0)
            
            logger.info(f"Generated volume profile chart for {ticker}")
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error generating volume profile chart: {e}")
            return None
    
    def generate_predator_chart(
        self,
        ticker: str,
        ohlcv_data: List[Dict],
        timeframe: str = "1min"
    ) -> Optional[str]:
        """
        Generate combined predator chart - all-in-one surgical view.
        Includes: Price with VWAP bands, Volume, Delta, and mini volume profile.
        """
        df = self._prepare_dataframe(ohlcv_data)
        if df is None:
            return None
        
        try:
            df = df.tail(60)
            
            vwap, ub1, lb1, ub2, lb2 = self._calculate_vwap_bands(df)
            delta = self._calculate_delta(df)
            price_levels, volumes = self._calculate_volume_profile(df, num_bins=20)
            
            fig = plt.figure(figsize=(16, 10), facecolor=self.colors['bg'])
            gs = GridSpec(3, 4, figure=fig, height_ratios=[3, 1, 1], width_ratios=[3, 0.5, 0.02, 0.48])
            
            ax_price = fig.add_subplot(gs[0, 0])
            ax_profile = fig.add_subplot(gs[0, 1], sharey=ax_price)
            ax_volume = fig.add_subplot(gs[1, 0:2])
            ax_delta = fig.add_subplot(gs[2, 0:2])
            
            for ax in [ax_price, ax_profile, ax_volume, ax_delta]:
                ax.set_facecolor(self.colors['bg'])
            
            x = range(len(df))
            
            ax_price.fill_between(x, ub2, lb2, alpha=0.1, color=self.colors['band2'])
            ax_price.fill_between(x, ub1, lb1, alpha=0.2, color=self.colors['band1'])
            ax_price.plot(x, vwap, color=self.colors['vwap'], linewidth=2, linestyle='--')
            
            for i, (idx, row) in enumerate(df.iterrows()):
                color = self.colors['up'] if row['Close'] >= row['Open'] else self.colors['down']
                ax_price.plot([i, i], [row['Low'], row['High']], color=color, linewidth=1)
                ax_price.add_patch(Rectangle(
                    (i - 0.3, min(row['Open'], row['Close'])),
                    0.6,
                    abs(row['Close'] - row['Open']) or 0.01,
                    facecolor=color,
                    edgecolor=color
                ))
            
            current_price = df['Close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            distance = ((current_price - current_vwap) / current_vwap) * 100
            position = "ABOVE" if current_price > current_vwap else "BELOW"
            
            net_delta = delta.iloc[-10:].sum()
            delta_control = "BUYERS" if net_delta > 0 else "SELLERS"
            
            info = f"${current_price:.2f} | {position} VWAP {abs(distance):.1f}% | {delta_control}"
            ax_price.set_title(f'{ticker} PREDATOR VIEW ({timeframe}) - {info}', 
                              color=self.colors['text'], fontsize=11, fontweight='bold')
            ax_price.grid(True, alpha=0.3, color=self.colors['grid'])
            ax_price.tick_params(colors=self.colors['text'])
            
            max_vol = volumes.max()
            normalized = volumes / max_vol if max_vol > 0 else volumes
            poc_idx = np.argmax(volumes)
            
            for i, (price, vol) in enumerate(zip(price_levels, normalized)):
                color = '#f0b90b' if i == poc_idx else self.colors['profile']
                ax_profile.barh(price, vol, height=(price_levels[1] - price_levels[0]) * 0.9,
                               color=color, alpha=0.7)
            
            ax_profile.tick_params(colors=self.colors['text'], labelleft=False)
            ax_profile.set_xlim(0, 1.2)
            
            vol_colors = [self.colors['up'] if df['Close'].iloc[i] >= df['Open'].iloc[i] 
                         else self.colors['down'] for i in range(len(df))]
            ax_volume.bar(x, df['Volume'], color=vol_colors, alpha=0.7)
            ax_volume.set_ylabel('Vol', color=self.colors['text'], fontsize=9)
            ax_volume.grid(True, alpha=0.3, color=self.colors['grid'])
            ax_volume.tick_params(colors=self.colors['text'])
            
            delta_colors = [self.colors['delta_positive'] if d >= 0 else self.colors['delta_negative'] 
                           for d in delta]
            ax_delta.bar(x, delta, color=delta_colors, alpha=0.8)
            ax_delta.axhline(y=0, color='white', linewidth=0.5, alpha=0.5)
            ax_delta.set_ylabel('Delta', color=self.colors['text'], fontsize=9)
            ax_delta.grid(True, alpha=0.3, color=self.colors['grid'])
            ax_delta.tick_params(colors=self.colors['text'])
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor=self.colors['bg'])
            plt.close(fig)
            buf.seek(0)
            
            logger.info(f"Generated predator chart for {ticker}")
            return base64.b64encode(buf.read()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error generating predator chart: {e}")
            return None
    
    def generate_all_charts(
        self,
        ticker: str,
        ohlcv_data: List[Dict],
        timeframe: str = "1min"
    ) -> Dict[str, Optional[str]]:
        """Generate all chart types for comprehensive analysis."""
        return {
            'vwap_bands': self.generate_vwap_bands_chart(ticker, ohlcv_data, timeframe),
            'delta': self.generate_delta_chart(ticker, ohlcv_data, timeframe),
            'volume_profile': self.generate_volume_profile_chart(ticker, ohlcv_data, timeframe),
            'predator': self.generate_predator_chart(ticker, ohlcv_data, timeframe)
        }


scalp_chart_generator = ScalpChartGenerator()
