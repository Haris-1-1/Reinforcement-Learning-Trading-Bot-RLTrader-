"""
Advanced Data Loader with Multi-Timeframe Support and Sequence Logic
Supports whale tracking, volume anomaly detection, and cyclical time encoding
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AdvancedDataLoader:
    """
    Advanced data loader with multi-timeframe analysis and whale tracking
    """
    
    def __init__(
        self,
        ticker: str = "BTC-USD",
        interval: str = "15m",
        lookback_periods: int = 60,
        sequence_length: int = 30,
        multi_timeframe: bool = True
    ):
        """
        Args:
            ticker: Trading pair symbol
            interval: Primary timeframe (15m, 1h, etc.)
            lookback_periods: How many periods to fetch initially
            sequence_length: Length of sequences for DRQN (e.g., 30 candles)
            multi_timeframe: Whether to include higher timeframe features
        """
        self.ticker = ticker
        self.interval = interval
        self.lookback_periods = lookback_periods
        self.sequence_length = sequence_length
        self.multi_timeframe = multi_timeframe
        
        # Timeframe mappings for multi-timeframe analysis
        self.timeframe_hierarchy = {
            "15m": ["1h", "4h"],
            "1h": ["4h", "1d"],
            "5m": ["15m", "1h"]
        }
        
        self.data = None
        self.sequences = None
        
    def fetch_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            # Default: fetch last 2 years
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        print(f"📊 Fetching {self.ticker} data from {start_date} to {end_date}...")
        
        try:
            ticker_obj = yf.Ticker(self.ticker)
            df = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval=self.interval
            )
            
            if df.empty:
                raise ValueError(f"No data retrieved for {self.ticker}")
                
            # Clean column names
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Got: {df.columns}")
            
            # Add timestamp features
            df = self._add_temporal_features(df)
            
            print(f"✅ Successfully fetched {len(df)} candles")
            
            self.data = df
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            raise
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical time encoding features (sin/cos for hour, day of week, etc.)
        This helps the neural network understand market cycles
        """
        # Hour of day (0-23)
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (0-6)
        df['day_of_week'] = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Day of month (1-31)
        df['day_of_month'] = df.index.day
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        return df
    
    def fetch_multi_timeframe_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> dict:
        """
        Fetch data for multiple timeframes for context
        
        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        if not self.multi_timeframe:
            return {self.interval: self.fetch_data(start_date, end_date)}
        
        timeframes_to_fetch = [self.interval]
        if self.interval in self.timeframe_hierarchy:
            timeframes_to_fetch.extend(self.timeframe_hierarchy[self.interval])
        
        multi_tf_data = {}
        
        for tf in timeframes_to_fetch:
            print(f"🔄 Fetching {tf} timeframe...")
            loader = AdvancedDataLoader(
                ticker=self.ticker,
                interval=tf,
                lookback_periods=self.lookback_periods,
                sequence_length=self.sequence_length,
                multi_timeframe=False  # Avoid recursive fetching
            )
            multi_tf_data[tf] = loader.fetch_data(start_date, end_date)
        
        return multi_tf_data
    
    def create_sequences(
        self,
        data: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for DRQN training
        
        Args:
            data: DataFrame with features
            feature_columns: List of column names to use as features
            
        Returns:
            Tuple of (sequences, targets) with shape:
            - sequences: (num_sequences, sequence_length, num_features)
            - targets: (num_sequences,) - next candle's close price for supervised component
        """
        # Extract feature values
        feature_values = data[feature_columns].values
        
        # Handle NaN values
        feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        num_samples = len(feature_values) - self.sequence_length
        num_features = len(feature_columns)
        
        sequences = np.zeros((num_samples, self.sequence_length, num_features))
        targets = np.zeros(num_samples)
        
        for i in range(num_samples):
            sequences[i] = feature_values[i:i + self.sequence_length]
            # Target is the next close price (for supervised trend prediction)
            targets[i] = data['close'].iloc[i + self.sequence_length]
        
        self.sequences = sequences
        
        print(f"✅ Created {num_samples} sequences of length {self.sequence_length}")
        print(f"   Shape: {sequences.shape}")
        
        return sequences, targets
    
    def get_current_context(
        self,
        index: int,
        data: pd.DataFrame,
        feature_columns: List[str]
    ) -> np.ndarray:
        """
        Get the sequence context for a specific index (used during trading simulation)
        
        Args:
            index: Current index in the dataset
            data: DataFrame with features
            feature_columns: List of column names to use
            
        Returns:
            Array of shape (sequence_length, num_features)
        """
        if index < self.sequence_length:
            # Pad with zeros if we don't have enough history
            padding_length = self.sequence_length - index
            available_data = data[feature_columns].iloc[:index].values
            padding = np.zeros((padding_length, len(feature_columns)))
            return np.vstack([padding, available_data])
        
        return data[feature_columns].iloc[index - self.sequence_length:index].values
    
    def calculate_volume_profile(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """
        Calculate volume profile to detect accumulation/distribution zones
        
        Args:
            df: DataFrame with OHLCV data
            window: Rolling window for volume analysis
            
        Returns:
            DataFrame with volume profile features
        """
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=window).mean()
        
        # Volume ratio (current volume vs moving average)
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
        
        # Detect volume spikes (potential whale activity)
        df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
        
        # Volume-weighted average price (VWAP)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        
        return df
    
    def normalize_data(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        method: str = 'minmax'
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Normalize features for neural network input
        
        Args:
            df: DataFrame with features
            feature_columns: Columns to normalize
            method: 'minmax' or 'zscore'
            
        Returns:
            Tuple of (normalized_df, scaler_params)
        """
        scaler_params = {}
        
        for col in feature_columns:
            if col not in df.columns:
                continue
                
            values = df[col].values
            
            if method == 'minmax':
                min_val = np.nanmin(values)
                max_val = np.nanmax(values)
                df[col] = (values - min_val) / (max_val - min_val + 1e-10)
                scaler_params[col] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
                
            elif method == 'zscore':
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                df[col] = (values - mean_val) / (std_val + 1e-10)
                scaler_params[col] = {'mean': mean_val, 'std': std_val, 'method': 'zscore'}
        
        return df, scaler_params


def test_data_loader():
    """Test the advanced data loader"""
    print("=" * 60)
    print("Testing Advanced Data Loader")
    print("=" * 60)
    
    # Initialize loader
    loader = AdvancedDataLoader(
        ticker="BTC-USD",
        interval="15m",
        sequence_length=30,
        multi_timeframe=True
    )
    
    # Fetch data
    df = loader.fetch_data(
        start_date="2024-01-01",
        end_date="2024-12-01"
    )
    
    print(f"\n📋 Data shape: {df.shape}")
    print(f"📋 Columns: {df.columns.tolist()}")
    print(f"\n📊 Sample data:")
    print(df.head())
    
    # Test volume profile
    df = loader.calculate_volume_profile(df)
    print(f"\n📊 Volume profile columns added:")
    print([col for col in df.columns if 'volume' in col or 'vwap' in col])
    
    # Test sequence creation
    feature_cols = ['close', 'volume', 'hour_sin', 'hour_cos']
    sequences, targets = loader.create_sequences(df, feature_cols)
    
    print(f"\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_data_loader()
