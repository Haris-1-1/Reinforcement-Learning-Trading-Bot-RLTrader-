"""
Technical Indicators Calculator
Calculates various technical analysis indicators for trading.
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


class TechnicalIndicators:
    """Calculate technical indicators for price data."""

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()

        # Ensure all columns are 1D Series (squeeze out extra dimensions)
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()

        # Moving Averages
        df['MA5'] = SMAIndicator(close=close, window=5).sma_indicator()
        df['MA20'] = SMAIndicator(close=close, window=20).sma_indicator()
        df['MA50'] = SMAIndicator(close=close, window=50).sma_indicator()

        # RSI (Relative Strength Index)
        df['RSI'] = RSIIndicator(close=close, window=14).rsi()

        # MACD (Moving Average Convergence Divergence)
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()

        # Bollinger Bands
        bbands = BollingerBands(close=close, window=20, window_dev=2)
        df['BB_upper'] = bbands.bollinger_hband()
        df['BB_middle'] = bbands.bollinger_mavg()
        df['BB_lower'] = bbands.bollinger_lband()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

        # ATR (Average True Range) - Volatility
        df['ATR'] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

        # Volume indicators
        df['Volume_MA'] = volume.rolling(window=20).mean()
        df['Volume_ratio'] = volume / df['Volume_MA']

        # Price changes
        df['Returns'] = close.pct_change()
        df['Log_returns'] = np.log(close / close.shift(1))

        # Drop NaN values (from indicator calculations)
        df = df.dropna()

        return df

    @staticmethod
    def get_feature_columns() -> list:
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
    def normalize_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Normalize features using z-score normalization.

        Args:
            df: DataFrame with features
            feature_cols: List of columns to normalize

        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        for col in feature_cols:
            if col in df.columns:
                col_data = df[col].squeeze()  # Ensure 1D Series
                mean = col_data.mean()
                std = col_data.std()
                if pd.notna(std) and std > 0:
                    df[col] = (col_data - mean) / std
        return df
