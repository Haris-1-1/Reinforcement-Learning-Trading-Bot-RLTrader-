"""
Technical Indicators Calculator
Calculates various technical analysis indicators for trading.
"""

import pandas as pd
import numpy as np
from typing import List


class TechnicalIndicators:
    """Calculate technical indicators for price data."""

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe.

        Args:
            df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)

        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()

        # Ensure columns are 1D (flatten if multi-level)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Convert to numeric and ensure 1D series
        close = pd.to_numeric(df['Close'], errors='coerce')
        high = pd.to_numeric(df['High'], errors='coerce')
        low = pd.to_numeric(df['Low'], errors='coerce')
        volume = pd.to_numeric(df['Volume'], errors='coerce')

        # ============ MOVING AVERAGES ============
        df['MA5'] = close.rolling(window=5).mean()
        df['MA20'] = close.rolling(window=20).mean()
        df['MA50'] = close.rolling(window=50).mean()

        # ============ RSI (Relative Strength Index) ============
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        # ============ MACD ============
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # ============ BOLLINGER BANDS ============
        df['BB_middle'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'].replace(0, np.nan)

        # ============ ATR (Average True Range) ============
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()

        # ============ VOLUME INDICATORS ============
        df['Volume_MA'] = volume.rolling(window=20).mean()
        df['Volume_ratio'] = volume / df['Volume_MA'].replace(0, np.nan)

        # ============ RETURNS ============
        df['Returns'] = close.pct_change()
        df['Log_returns'] = np.log(close / close.shift(1))

        # Drop rows with NaN (from indicator calculations)
        df = df.dropna()
        
        print(f"Added technical indicators. Remaining rows after dropna: {len(df)}")

        return df

    @staticmethod
    def get_feature_columns() -> List[str]:
        """
        Get list of all feature column names.

        Returns:
            List of feature column names
        """
        return [
            # OHLCV
            'Open', 'High', 'Low', 'Close', 'Volume',
            # Moving Averages
            'MA5', 'MA20', 'MA50',
            # RSI
            'RSI',
            # MACD
            'MACD', 'MACD_signal', 'MACD_hist',
            # Bollinger Bands
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
            # ATR
            'ATR',
            # Volume
            'Volume_MA', 'Volume_ratio',
            # Returns
            'Returns', 'Log_returns'
        ]

    @staticmethod
    def normalize_features(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
        """
        Normalize features using z-score normalization.

        Args:
            df: DataFrame with features
            feature_cols: List of columns to normalize (if None, normalize all)

        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        
        if feature_cols is None:
            feature_cols = TechnicalIndicators.get_feature_columns()
        
        for col in feature_cols:
            if col in df.columns:
                col_data = df[col]
                mean = col_data.mean()
                std = col_data.std()
                if pd.notna(std) and std > 0:
                    df[col] = (col_data - mean) / std
                else:
                    df[col] = 0.0
                    
        return df
