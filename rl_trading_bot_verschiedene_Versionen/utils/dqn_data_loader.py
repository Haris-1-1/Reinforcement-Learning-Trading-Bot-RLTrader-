import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, Optional

class DQNDataLoader:

    def __init__(self,
                 ticker: str = 'BTC-USD',
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 interval: str = '1d',
                 train_test_split: float = 0.8):
        self.ticker = ticker
        self.interval = interval
        self.train_test_split = train_test_split

        if end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            self.end_date = end_date

        if start_date is None:
            start = datetime.now() - timedelta(days=730)
            self.start_date = start.strftime('%Y-%m-%d')
        else:
            self.start_date = start_date

        self.data = None
        self.train_data = None
        self.test_data = None
        self.norm_params = {}

    def load_data(self) -> pd.DataFrame:
        print(f"[DQN DataLoader] Loading {self.ticker} data from {self.start_date} to {self.end_date}...")

        try:
            data = yf.download(
                self.ticker,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False,
                auto_adjust=True
            )

            if data.empty:
                raise ValueError(f"No data downloaded for {self.ticker}")

            data = data.reset_index()

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
                print(f"[DQN DataLoader] ✓ Flattened MultiIndex columns")

            print(f"[DQN DataLoader] Columns: {list(data.columns)}")

            if 'Date' in data.columns:
                data.rename(columns={'Date': 'Datetime'}, inplace=True)
            elif 'Datetime' not in data.columns:
                data['Datetime'] = data.index

            data['Datetime'] = pd.to_datetime(data['Datetime'])

            print(f"[DQN DataLoader] Loaded {len(data)} rows")
            print(f"[DQN DataLoader] Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
            print(f"[DQN DataLoader] Columns: {list(data.columns)}")

            return data

        except Exception as e:
            print(f"[DQN DataLoader] Error loading data: {str(e)}")
            raise

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        print("[DQN DataLoader] Adding technical indicators...")

        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']

        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']

        df['Price_Change_1'] = df['Close'].pct_change(periods=1)
        df['Price_Change_5'] = df['Close'].pct_change(periods=5)
        df['Price_Change_10'] = df['Close'].pct_change(periods=10)

        df['Volatility_10'] = df['Close'].rolling(window=10).std()
        df['Volatility_30'] = df['Close'].rolling(window=30).std()

        df['High_Low_Range'] = df['High'] - df['Low']
        df['High_Low_Ratio'] = df['High'] / df['Low']

        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)

        print(f"[DQN DataLoader] Indicators added. Dropped {rows_before - rows_after} rows with NaN")
        print(f"[DQN DataLoader] Final shape: {df.shape}")

        return df

    def prepare_data(self, normalize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.data = self.load_data()

        self.data = self.add_technical_indicators(self.data)

        split_idx = int(len(self.data) * self.train_test_split)
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()

        self.original_prices_train = self.train_data['Close'].values.copy()
        self.original_prices_test = self.test_data['Close'].values.copy()

        print(f"\n[DQN DataLoader] Data Split:")
        print(f"  Training:   {len(self.train_data)} rows ({self.train_test_split*100:.0f}%)")
        print(f"  Test:       {len(self.test_data)} rows ({(1-self.train_test_split)*100:.0f}%)")

        if normalize:
            self.train_data, self.test_data = self._normalize_data(
                self.train_data,
                self.test_data
            )
            print(f"[DQN DataLoader] ✓ Data normalized (min-max scaling)")

        return self.train_data, self.test_data

    def _normalize_data(self,
                       train_data: pd.DataFrame,
                       test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_normalized = train_data.copy()
        test_normalized = test_data.copy()

        cols_to_normalize = [col for col in train_data.columns
                           if col not in ['Datetime']]

        self.norm_params = {}

        for col in cols_to_normalize:
            min_val = train_data[col].min()
            max_val = train_data[col].max()
            self.norm_params[col] = {'min': min_val, 'max': max_val}

            if max_val - min_val == 0:
                train_normalized[col] = 0
                test_normalized[col] = 0
            else:
                train_normalized[col] = (train_data[col] - min_val) / (max_val - min_val)
                test_normalized[col] = (test_data[col] - min_val) / (max_val - min_val)

        return train_normalized, test_normalized

    def get_latest_data(self, lookback_days: int = 100) -> pd.DataFrame:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        print(f"[DQN DataLoader] Fetching latest {lookback_days} days of data...")

        data = yf.download(
            self.ticker,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval=self.interval,
            progress=False,
            auto_adjust=True
        )

        data = data.reset_index()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if 'Date' in data.columns:
            data.rename(columns={'Date': 'Datetime'}, inplace=True)
        elif 'Datetime' not in data.columns:
            data['Datetime'] = data.index

        data['Datetime'] = pd.to_datetime(data['Datetime'])

        data = self.add_technical_indicators(data)

        print(f"[DQN DataLoader] Latest data: {len(data)} rows")

        return data

    def denormalize_price(self, normalized_price: float) -> float:
        if 'Close' not in self.norm_params:
            return normalized_price

        min_val = self.norm_params['Close']['min']
        max_val = self.norm_params['Close']['max']

        return normalized_price * (max_val - min_val) + min_val


def create_dqn_data_loader(
    ticker: str = 'BTC-USD',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = '1d',
    train_test_split: float = 0.8
) -> DQNDataLoader:
    return DQNDataLoader(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        train_test_split=train_test_split
    )


if __name__ == "__main__":
    print("="*70)
    print("Testing DQN Data Loader")
    print("="*70)

    loader = DQNDataLoader(
        ticker='BTC-USD',
        start_date='2023-01-01',
        interval='1d'
    )

    train_data, test_data = loader.prepare_data(normalize=True)

    print("\n" + "="*70)
    print("Training Data Sample:")
    print(train_data.head())

    print("\n" + "="*70)
    print("Test Data Sample:")
    print(test_data.head())

    print("\n" + "="*70)
    print("✓ DQN Data Loader test completed successfully!")
