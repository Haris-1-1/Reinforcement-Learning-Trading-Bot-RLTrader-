"""
Data Loader for cryptocurrency price data.
Fetches and processes OHLCV data with technical indicators.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Tuple, Optional
from .indicators import TechnicalIndicators


class DataLoader:
    """Load and process cryptocurrency market data."""

    def __init__(self, symbol: str, start_date: str, end_date: str,
                 interval: str = '1d', test_split: float = 0.2):
        """
        Initialize DataLoader.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD', 'ETH-USD')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', '15m', etc.)
            test_split: Fraction of data to use for testing (0.0-1.0)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.test_split = test_split

        self.data = None
        self.train_data = None
        self.test_data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load data from yfinance.

        Returns:
            DataFrame with OHLCV data
        """
        print(f"Loading {self.symbol} data from {self.start_date} to {self.end_date}...")

        data = yf.download(
            self.symbol,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            progress=False
        )

        if data.empty:
            raise ValueError(f"No data found for {self.symbol}")

        # Reset index to have Date as a column
        data = data.reset_index()

        print(f"Loaded {len(data)} rows of data")

        return data

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicators added
        """
        print("Calculating technical indicators...")
        df = TechnicalIndicators.add_all_indicators(df)
        print(f"Added technical indicators. Remaining rows: {len(df)}")
        return df

    def split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.

        Args:
            df: Full dataset

        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * (1 - self.test_split))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"Train set: {len(train_df)} rows")
        print(f"Test set: {len(test_df)} rows")

        return train_df, test_df

    def prepare_data(self, normalize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load, process, and split data.

        Args:
            normalize: Whether to normalize features

        Returns:
            Tuple of (train_data, test_data)
        """
        # Load raw data
        self.data = self.load_data()

        # Add technical indicators
        self.data = self.add_technical_indicators(self.data)

        # Split into train/test
        self.train_data, self.test_data = self.split_train_test(self.data)

        # Optional normalization
        if normalize:
            feature_cols = TechnicalIndicators.get_feature_columns()
            self.train_data = TechnicalIndicators.normalize_features(
                self.train_data, feature_cols
            )
            self.test_data = TechnicalIndicators.normalize_features(
                self.test_data, feature_cols
            )
            print("Features normalized")

        return self.train_data, self.test_data

    def get_price_array(self, dataset: str = 'train') -> np.ndarray:
        """
        Get close prices as numpy array.

        Args:
            dataset: 'train' or 'test'

        Returns:
            Numpy array of close prices
        """
        if dataset == 'train':
            if self.train_data is None:
                raise ValueError("Train data not loaded. Call prepare_data() first.")
            return self.train_data['Close'].values
        else:
            if self.test_data is None:
                raise ValueError("Test data not loaded. Call prepare_data() first.")
            return self.test_data['Close'].values

    def get_feature_matrix(self, dataset: str = 'train') -> np.ndarray:
        """
        Get feature matrix for RL agent.

        Args:
            dataset: 'train' or 'test'

        Returns:
            Numpy array of features (n_samples, n_features)
        """
        feature_cols = TechnicalIndicators.get_feature_columns()

        if dataset == 'train':
            if self.train_data is None:
                raise ValueError("Train data not loaded. Call prepare_data() first.")
            return self.train_data[feature_cols].values
        else:
            if self.test_data is None:
                raise ValueError("Test data not loaded. Call prepare_data() first.")
            return self.test_data[feature_cols].values

    @staticmethod
    def get_supported_symbols() -> list:
        """
        Get list of commonly supported cryptocurrency symbols.

        Returns:
            List of symbol strings
        """
        return [
            'BTC-USD',
            'ETH-USD',
            'BNB-USD',
            'ADA-USD',
            'SOL-USD',
            'DOT-USD',
            'DOGE-USD',
            'MATIC-USD'
        ]
