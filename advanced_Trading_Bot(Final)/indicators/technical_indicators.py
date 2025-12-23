"""
Advanced Technical Indicators with Whale Tracking
Includes A/D Line, Ichimoku, ATR, Bollinger Bands, and volume anomaly detection
"""

import numpy as np
import pandas as pd
from typing import Tuple


class AdvancedIndicators:
    """
    Advanced technical indicators for smart money detection
    """
    
    @staticmethod
    def accumulation_distribution_line(
        df: pd.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume'
    ) -> pd.Series:
        """
        Accumulation/Distribution Line - detects divergences between price and volume
        
        The A/D line is a cumulative indicator that uses volume and price to assess
        whether a stock is being accumulated or distributed.
        
        Formula:
        1. Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
        2. Money Flow Volume = Money Flow Multiplier × Volume
        3. A/D Line = Previous A/D + Money Flow Volume
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with A/D line values
        """
        high = df[high_col]
        low = df[low_col]
        close = df[close_col]
        volume = df[volume_col]
        
        # Money Flow Multiplier
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
        
        # Money Flow Volume
        mfv = mfm * volume
        
        # Accumulation/Distribution Line (cumulative)
        ad_line = mfv.cumsum()
        
        return ad_line
    
    @staticmethod
    def accumulation_distribution_oscillator(
        df: pd.DataFrame,
        fast_period: int = 3,
        slow_period: int = 10
    ) -> pd.Series:
        """
        A/D Oscillator - shows momentum of accumulation/distribution
        
        Args:
            df: DataFrame with A/D line
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            
        Returns:
            Series with A/D oscillator values
        """
        ad_line = AdvancedIndicators.accumulation_distribution_line(df)
        
        # Calculate EMAs
        fast_ema = ad_line.ewm(span=fast_period, adjust=False).mean()
        slow_ema = ad_line.ewm(span=slow_period, adjust=False).mean()
        
        # Oscillator
        ad_oscillator = fast_ema - slow_ema
        
        return ad_oscillator
    
    @staticmethod
    def on_balance_volume(
        df: pd.DataFrame,
        close_col: str = 'close',
        volume_col: str = 'volume'
    ) -> pd.Series:
        """
        On-Balance Volume (OBV) - measures buying/selling pressure
        
        Formula:
        - If Close > Previous Close: OBV = Previous OBV + Volume
        - If Close < Previous Close: OBV = Previous OBV - Volume
        - If Close = Previous Close: OBV = Previous OBV
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with OBV values
        """
        close = df[close_col]
        volume = df[volume_col]
        
        obv = np.zeros(len(df))
        obv[0] = volume.iloc[0]
        
        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i-1]:
                obv[i] = obv[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv[i] = obv[i-1] - volume.iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=df.index)
    
    @staticmethod
    def volume_price_trend(
        df: pd.DataFrame,
        close_col: str = 'close',
        volume_col: str = 'volume'
    ) -> pd.Series:
        """
        Volume Price Trend (VPT) - similar to OBV but considers magnitude of price change
        
        Formula:
        VPT = Previous VPT + Volume × (Close - Previous Close) / Previous Close
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with VPT values
        """
        close = df[close_col]
        volume = df[volume_col]
        
        # Calculate percentage change in close
        close_pct_change = close.pct_change()
        
        # VPT
        vpt = (volume * close_pct_change).cumsum()
        
        return vpt
    
    @staticmethod
    def ichimoku_cloud(
        df: pd.DataFrame,
        conversion_period: int = 9,
        base_period: int = 26,
        span_b_period: int = 52,
        displacement: int = 26
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Ichimoku Cloud - comprehensive trend following system
        
        Components:
        1. Tenkan-sen (Conversion Line) = (9-period high + 9-period low) / 2
        2. Kijun-sen (Base Line) = (26-period high + 26-period low) / 2
        3. Senkou Span A (Leading Span A) = (Conversion Line + Base Line) / 2, shifted 26 periods ahead
        4. Senkou Span B (Leading Span B) = (52-period high + 52-period low) / 2, shifted 26 periods ahead
        5. Chikou Span (Lagging Span) = Close price shifted 26 periods back
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span)
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Tenkan-sen (Conversion Line)
        conversion_high = high.rolling(window=conversion_period).max()
        conversion_low = low.rolling(window=conversion_period).min()
        tenkan_sen = (conversion_high + conversion_low) / 2
        
        # Kijun-sen (Base Line)
        base_high = high.rolling(window=base_period).max()
        base_low = low.rolling(window=base_period).min()
        kijun_sen = (base_high + base_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        
        # Senkou Span B (Leading Span B)
        span_b_high = high.rolling(window=span_b_period).max()
        span_b_low = low.rolling(window=span_b_period).min()
        senkou_span_b = ((span_b_high + span_b_low) / 2).shift(displacement)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-displacement)
        
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    
    @staticmethod
    def average_true_range(
        df: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Average True Range (ATR) - measures volatility
        
        Used for dynamic stop-loss positioning
        
        Formula:
        1. True Range = max[(High - Low), |High - Previous Close|, |Low - Previous Close|]
        2. ATR = Moving Average of True Range
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # True Range
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR (using Wilder's smoothing)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        num_std: float = 2.0,
        price_col: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands - volatility bands around moving average
        
        Formula:
        1. Middle Band = 20-period SMA
        2. Upper Band = Middle Band + (2 × 20-period Standard Deviation)
        3. Lower Band = Middle Band - (2 × 20-period Standard Deviation)
        
        Args:
            df: DataFrame with price data
            period: Moving average period
            num_std: Number of standard deviations
            price_col: Price column to use
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        price = df[price_col]
        
        # Middle band (SMA)
        middle_band = price.rolling(window=period).mean()
        
        # Standard deviation
        std = price.rolling(window=period).std()
        
        # Upper and lower bands
        upper_band = middle_band + (num_std * std)
        lower_band = middle_band - (num_std * std)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def bollinger_bandwidth(
        df: pd.DataFrame,
        period: int = 20,
        num_std: float = 2.0
    ) -> pd.Series:
        """
        Bollinger Band Width - measures volatility squeeze
        
        Low bandwidth indicates low volatility (potential breakout coming)
        High bandwidth indicates high volatility
        
        Formula:
        Bandwidth = (Upper Band - Lower Band) / Middle Band
        
        Args:
            df: DataFrame with price data
            period: Moving average period
            num_std: Number of standard deviations
            
        Returns:
            Series with bandwidth values
        """
        upper, middle, lower = AdvancedIndicators.bollinger_bands(df, period, num_std)
        
        bandwidth = (upper - lower) / middle
        
        return bandwidth
    
    @staticmethod
    def relative_strength_index(
        df: pd.DataFrame,
        period: int = 14,
        price_col: str = 'close'
    ) -> pd.Series:
        """
        RSI - Relative Strength Index
        
        Measures momentum, overbought/oversold conditions
        
        Formula:
        1. Calculate price changes
        2. Separate gains and losses
        3. Calculate average gain and average loss
        4. RS = Average Gain / Average Loss
        5. RSI = 100 - (100 / (1 + RS))
        
        Args:
            df: DataFrame with price data
            period: RSI period
            price_col: Price column to use
            
        Returns:
            Series with RSI values (0-100)
        """
        price = df[price_col]
        
        # Calculate price changes
        delta = price.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses (using Wilder's smoothing)
        avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def moving_average_convergence_divergence(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        price_col: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD - Moving Average Convergence Divergence
        
        Trend-following momentum indicator
        
        Formula:
        1. MACD Line = 12-period EMA - 26-period EMA
        2. Signal Line = 9-period EMA of MACD Line
        3. Histogram = MACD Line - Signal Line
        
        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            price_col: Price column to use
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        price = df[price_col]
        
        # Calculate EMAs
        fast_ema = price.ewm(span=fast_period, adjust=False).mean()
        slow_ema = price.ewm(span=slow_period, adjust=False).mean()
        
        # MACD line
        macd_line = fast_ema - slow_ema
        
        # Signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def detect_volume_anomalies(
        df: pd.DataFrame,
        window: int = 20,
        threshold: float = 2.5
    ) -> pd.Series:
        """
        Detect volume anomalies (potential whale activity)
        
        Returns 1 for anomalies, 0 for normal volume
        
        Args:
            df: DataFrame with volume data
            window: Rolling window for baseline calculation
            threshold: Number of standard deviations for anomaly
            
        Returns:
            Series with anomaly flags (0 or 1)
        """
        volume = df['volume']
        
        # Calculate rolling mean and std
        volume_ma = volume.rolling(window=window).mean()
        volume_std = volume.rolling(window=window).std()
        
        # Z-score
        z_score = (volume - volume_ma) / (volume_std + 1e-10)
        
        # Flag anomalies
        anomalies = (abs(z_score) > threshold).astype(int)
        
        return anomalies
    
    @staticmethod
    def price_volume_divergence(
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """
        Detect divergence between price and volume trends
        
        Positive divergence: Price down, volume up (accumulation)
        Negative divergence: Price up, volume down (distribution)
        
        Args:
            df: DataFrame with OHLCV data
            window: Window for trend calculation
            
        Returns:
            Series with divergence score (-1 to 1)
        """
        close = df['close']
        volume = df['volume']
        
        # Calculate trends (normalized)
        price_trend = (close - close.shift(window)) / (close.shift(window) + 1e-10)
        volume_trend = (volume - volume.shift(window)) / (volume.shift(window) + 1e-10)
        
        # Divergence score
        # Positive: price down, volume up
        # Negative: price up, volume down
        divergence = volume_trend - price_trend
        
        return divergence


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all advanced indicators to a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all indicators added
    """
    # Whale tracking indicators
    df['ad_line'] = AdvancedIndicators.accumulation_distribution_line(df)
    df['ad_oscillator'] = AdvancedIndicators.accumulation_distribution_oscillator(df)
    df['obv'] = AdvancedIndicators.on_balance_volume(df)
    df['vpt'] = AdvancedIndicators.volume_price_trend(df)
    
    # Ichimoku Cloud
    tenkan, kijun, span_a, span_b, chikou = AdvancedIndicators.ichimoku_cloud(df)
    df['ichimoku_tenkan'] = tenkan
    df['ichimoku_kijun'] = kijun
    df['ichimoku_span_a'] = span_a
    df['ichimoku_span_b'] = span_b
    df['ichimoku_chikou'] = chikou
    
    # Volatility indicators
    df['atr'] = AdvancedIndicators.average_true_range(df)
    upper_bb, middle_bb, lower_bb = AdvancedIndicators.bollinger_bands(df)
    df['bb_upper'] = upper_bb
    df['bb_middle'] = middle_bb
    df['bb_lower'] = lower_bb
    df['bb_width'] = AdvancedIndicators.bollinger_bandwidth(df)
    
    # Momentum indicators
    df['rsi'] = AdvancedIndicators.relative_strength_index(df)
    macd, signal, hist = AdvancedIndicators.moving_average_convergence_divergence(df)
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist
    
    # Volume anomalies
    df['volume_anomaly'] = AdvancedIndicators.detect_volume_anomalies(df)
    df['pv_divergence'] = AdvancedIndicators.price_volume_divergence(df)
    
    return df


def test_indicators():
    """Test the indicators"""
    print("=" * 60)
    print("Testing Advanced Indicators")
    print("=" * 60)
    
    # Create sample data
    import yfinance as yf
    
    df = yf.Ticker("BTC-USD").history(period="1mo", interval="1h")
    df.columns = [col.lower() for col in df.columns]
    
    print(f"📊 Original data shape: {df.shape}")
    
    # Add all indicators
    df = add_all_indicators(df)
    
    print(f"📊 Data shape after indicators: {df.shape}")
    print(f"\n📋 Added indicators:")
    indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    for col in indicator_cols:
        print(f"  - {col}")
    
    print(f"\n📊 Sample with indicators:")
    print(df[['close', 'rsi', 'macd', 'ad_line', 'volume_anomaly']].tail())
    
    print(f"\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_indicators()
