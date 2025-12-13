"""
Data Loader for cryptocurrency price data.
Fetches and processes OHLCV data with technical indicators.

WICHTIG: Speichert ORIGINALE Preise separat für Trading!
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Optional
from utils.indicators import TechnicalIndicators


class DataLoader:
    """Load and process cryptocurrency market data."""

    def __init__(
        self,
        symbol: str = 'BTC-USD',
        start_date: str = '2020-01-01',
        end_date: str = '2024-01-01',
        interval: str = '1d',
        test_split: float = 0.2
    ):
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
        
        # NEU: Speichere originale Preise separat
        self.original_prices_train = None
        self.original_prices_test = None

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

        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Reset index to have Date as a column
        data = data.reset_index()

        print(f"Loaded {len(data)} rows of data")
        print(f"Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
        print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")

        return data

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe."""
        print("Calculating technical indicators...")
        df = TechnicalIndicators.add_all_indicators(df)
        return df

    def split_train_test(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        split_idx = int(len(df) * (1 - self.test_split))
        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

        print(f"Train set: {len(train_df)} rows ({100*(1-self.test_split):.0f}%)")
        print(f"Test set: {len(test_df)} rows ({100*self.test_split:.0f}%)")

        return train_df, test_df

    def prepare_data(self, normalize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load, process, and split data.
        
        WICHTIG: Speichert originale Preise BEVOR normalisiert wird!

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

        # ════════════════════════════════════════════════════════════
        # NEU: SPEICHERE ORIGINALE PREISE BEVOR NORMALISIERUNG!
        # ════════════════════════════════════════════════════════════
        self.original_prices_train = self.train_data['Close'].values.copy()
        self.original_prices_test = self.test_data['Close'].values.copy()
        
        print(f"\n💰 Original Train Prices: ${self.original_prices_train.min():.2f} - ${self.original_prices_train.max():.2f}")
        print(f"💰 Original Test Prices: ${self.original_prices_test.min():.2f} - ${self.original_prices_test.max():.2f}")

        # Optional normalization (nur für Features, nicht für Trading!)
        if normalize:
            print("\nNormalizing features for neural network...")
            feature_cols = TechnicalIndicators.get_feature_columns()
            self.train_data = TechnicalIndicators.normalize_features(self.train_data, feature_cols)
            self.test_data = TechnicalIndicators.normalize_features(self.test_data, feature_cols)
            print("Features normalized ✓")

        return self.train_data, self.test_data
    
    def get_original_prices(self, dataset: str = 'train') -> np.ndarray:
        """
        Get ORIGINAL (nicht-normalisierte) Preise für Trading.
        
        Args:
            dataset: 'train' or 'test'
            
        Returns:
            Array mit echten USD Preisen
        """
        if dataset == 'train':
            if self.original_prices_train is None:
                raise ValueError("Data not loaded. Call prepare_data() first.")
            return self.original_prices_train
        else:
            if self.original_prices_test is None:
                raise ValueError("Data not loaded. Call prepare_data() first.")
            return self.original_prices_test

    @staticmethod
    def get_supported_symbols() -> list:
        """Get list of commonly supported cryptocurrency symbols."""
        return [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 
            'SOL-USD', 'DOT-USD', 'DOGE-USD', 'XRP-USD'
        ]
