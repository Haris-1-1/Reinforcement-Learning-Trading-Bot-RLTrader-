import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple
class DataLoader:
    def __init__(self,
                 symbol: str,
                 start_date: str,
                 end_date: str,
                 interval: str = '1d',
                 test_split: float = 0.2):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.test_split = test_split
        self.train_test_split = 1.0 - test_split
        self.data = None
        self.train_data = None
        self.test_data = None
        self.original_prices_train = None
        self.original_prices_test = None
    def load_data(self) -> pd.DataFrame:
        print(f"Loading {self.symbol} data from {self.start_date} to {self.end_date}...")
        max_retries = 3
        timeout = 30
        for attempt in range(max_retries):
            try:
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
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            print("✓ Flattened MultiIndex columns")
        data = data.reset_index()
        if 'Date' in data.columns:
            data = data.rename(columns={'Date': 'Datetime'})
        print(f"Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
        print(f"Columns: {list(data.columns)}")
        return data
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Price_Change'] = df['Close'].pct_change()
        df = df.dropna()
        print(f"Technical indicators added. Final shape: {df.shape}")
        return df
    def prepare_data(self, normalize: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.data = self.load_data()
        self.data = self.add_technical_indicators(self.data)
        split_idx = int(len(self.data) * self.train_test_split)
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()
        self.original_prices_train = self.train_data['Close'].values.copy()
        self.original_prices_test = self.test_data['Close'].values.copy()
        print(f"\nData Split:")
        print(f"  Training:   {len(self.train_data)} rows ({self.train_test_split*100:.0f}%)")
        print(f"  Test:       {len(self.test_data)} rows ({(1-self.train_test_split)*100:.0f}%)")
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
        train_normalized = train_data.copy()
        test_normalized = test_data.copy()
        cols_to_normalize = [col for col in train_data.columns
                           if col not in ['Datetime']]
        for col in cols_to_normalize:
            min_val = train_data[col].min()
            max_val = train_data[col].max()
            if max_val - min_val == 0:
                train_normalized[col] = 0
                test_normalized[col] = 0
            else:
                train_normalized[col] = (train_data[col] - min_val) / (max_val - min_val)
                test_normalized[col] = (test_data[col] - min_val) / (max_val - min_val)
        return train_normalized, test_normalized
    def get_original_prices(self, split: str = 'train') -> np.ndarray:
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