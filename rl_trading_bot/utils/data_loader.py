"""
Data Loader for Q-Learning Trading Bot
Fixed version with MultiIndex handling and retry logic
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple


class DataLoader:
    """
    Load and preprocess data for Q-Learning
    """
    
    def __init__(self, 
                 symbol: str,
                 start_date: str,
                 end_date: str,
                 interval: str = '1d',
                 test_split: float = 0.2):  # Changed to test_split for compatibility
        """
        Initialize Data Loader
        
        Args:
            symbol: Ticker symbol (e.g., 'BTC-USD')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', etc.)
            test_split: Fraction for testing (0.2 = 20% test, 80% train)
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.test_split = test_split
        self.train_test_split = 1.0 - test_split  # Convert to train split internally
        
        self.data = None
        self.train_data = None
        self.test_data = None
        self.original_prices_train = None
        self.original_prices_test = None
        
    def load_data(self) -> pd.DataFrame:
        """Download data from yfinance with retry logic"""
        print(f"Loading {self.symbol} data from {self.start_date} to {self.end_date}...")
        
        max_retries = 3
        timeout = 30
        
        for attempt in range(max_retries):
            try:
                # Download data
                data = yf.download(
                    self.symbol,
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.interval,
                    progress=False,
                    auto_adjust=True,
                    timeout=timeout
                )
                
                if data.empty:
                    if attempt < max_retries - 1:
                        print(f"Empty data, retrying ({attempt+1}/{max_retries})...")
                        continue
                    raise ValueError(f"No data downloaded for {self.symbol}")
                
                # Success!
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Download failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                    print(f"Retrying in 2 seconds...")
                    import time
                    time.sleep(2)
                else:
                    raise ValueError(f"No data downloaded for {self.symbol}")
        
        print(f"Loaded {len(data)} rows of data")
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            print("✓ Flattened MultiIndex columns")
        
        # Reset index to make date a column
        data = data.reset_index()
        
        # Rename 'Date' to 'Datetime' for consistency
        if 'Date' in data.columns:
            data = data.rename(columns={'Date': 'Datetime'})
        
        print(f"Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
        print(f"Columns: {list(data.columns)}")
        
        return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Volume change
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Price change
        df['Price_Change'] = df['Close'].pct_change()
        
        # Drop NaN
        df = df.dropna()
        
        print(f"Technical indicators added. Final shape: {df.shape}")
        
        return df
    
    def prepare_data(self, normalize: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data
        
        Args:
            normalize: Whether to normalize data
            
        Returns:
            Tuple of (train_data, test_data)
        """
        # Load data
        self.data = self.load_data()
        
        # Add technical indicators
        self.data = self.add_technical_indicators(self.data)
        
        # Split train/test
        split_idx = int(len(self.data) * self.train_test_split)
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()
        
        # Store original prices BEFORE any normalization
        self.original_prices_train = self.train_data['Close'].values.copy()
        self.original_prices_test = self.test_data['Close'].values.copy()
        
        print(f"\nData Split:")
        print(f"  Training:   {len(self.train_data)} rows ({self.train_test_split*100:.0f}%)")
        print(f"  Test:       {len(self.test_data)} rows ({(1-self.train_test_split)*100:.0f}%)")
        
        # Optional normalization (Q-Learning usually doesn't need it)
        if normalize:
            self.train_data, self.test_data = self._normalize_data(
                self.train_data, 
                self.test_data
            )
            print("✓ Data normalized (min-max scaling)")
        
        return self.train_data, self.test_data
    
    def _normalize_data(self, 
                       train_data: pd.DataFrame, 
                       test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize data using min-max scaling based on training data only
        
        Args:
            train_data: Training dataset
            test_data: Test dataset
            
        Returns:
            Tuple of (normalized_train, normalized_test)
        """
        train_normalized = train_data.copy()
        test_normalized = test_data.copy()
        
        # Columns to normalize (exclude Datetime)
        cols_to_normalize = [col for col in train_data.columns 
                           if col not in ['Datetime']]
        
        # Calculate min and max from TRAINING data only
        for col in cols_to_normalize:
            min_val = train_data[col].min()
            max_val = train_data[col].max()
            
            # Avoid division by zero
            if max_val - min_val == 0:
                train_normalized[col] = 0
                test_normalized[col] = 0
            else:
                # Normalize to [0, 1] range
                train_normalized[col] = (train_data[col] - min_val) / (max_val - min_val)
                test_normalized[col] = (test_data[col] - min_val) / (max_val - min_val)
        
        return train_normalized, test_normalized
    
    def get_original_prices(self, split: str = 'train') -> np.ndarray:
        """
        Get original prices (before normalization) for backward compatibility
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            Array of original prices
        """
        if split.lower() == 'train':
            return self.original_prices_train
        elif split.lower() == 'test':
            return self.original_prices_test
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'test'")


if __name__ == "__main__":
    print("="*70)
    print("Data Loader for Q-Learning - Ready to use!")
    print("="*70)
    print("\nUsage:")
    print("  from utils.data_loader import DataLoader")
    print("  loader = DataLoader('BTC-USD', '2023-01-01', '2025-12-15')")
    print("  train_data, test_data = loader.prepare_data()")