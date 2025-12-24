"""
ENHANCED DATA LOADER - Improved Version
========================================
Features: RandomForest Supervised Model, Cyclical Time Features
"""

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler  # Better than StandardScaler for outliers
from .indicators import TechnicalIndicators
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Manages data loading, feature engineering, and supervised 'intuition' model
    
    Improvements:
    - RandomForest instead of LogisticRegression
    - RobustScaler instead of StandardScaler
    - More features for supervised model
    - Better error handling
    """
    
    def __init__(
        self,
        symbol: str = 'BTC-USD',
        start_date: str = '2020-01-01',
        end_date: str = '2024-01-01',
        interval: str = '1h',
        test_split: float = 0.15
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.test_split = test_split
        
        # Storage for original prices (needed for environment)
        self.original_prices_train = None
        self.original_prices_test = None
        
        # Supervised model (IMPROVED: RandomForest instead of LogisticRegression)
        self.supervised_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        # Scaler (IMPROVED: RobustScaler handles outliers better)
        self.feature_scaler = RobustScaler()
    
    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds cyclical time features (sin/cos encoding)
        """
        # Ensure Date column exists
        if 'Date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Month (1-12)
            df['month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
            
            # Day of Week (0-6)
            df['day_sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 7)
            
            # Hour (for intraday data)
            if 'h' in self.interval or 'm' in self.interval:
                df['hour_sin'] = np.sin(2 * np.pi * df['Date'].dt.hour / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['Date'].dt.hour / 24)
            else:
                df['hour_sin'] = 0.0
                df['hour_cos'] = 0.0
        else:
            # Fallback: all zeros
            df['month_sin'] = df['month_cos'] = 0.0
            df['day_sin'] = df['day_cos'] = 0.0
            df['hour_sin'] = df['hour_cos'] = 0.0
        
        return df
    
    def _train_supervised_intuition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trains supervised model to predict next-candle direction
        
        IMPROVED:
        - RandomForest instead of LogisticRegression
        - More features (not just 3)
        - Feature importance analysis
        """
        df = df.copy()
        
        # Target: 1 if next close > current close, else 0
        df['Target_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # IMPROVED: Use more features for supervised model
        intuition_features = [
            'RSI', 'Returns', 'Vol_Ratio',
            'MACD_Hist', 'BB_Pos', 'ATR_Pct',
            'Stoch_K', 'ADX', 'MFI'
        ]
        
        # Check which features are available
        available_features = [f for f in intuition_features if f in df.columns]
        
        if len(available_features) < 3:
            print(f"Warning: Only {len(available_features)} features available for supervised model!")
            df['Trend_Prob'] = 0.5
            return df
        
        # Train only on first 70% (prevent data leakage)
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx].copy()
        
        # Remove NaN
        train_df = train_df.dropna(subset=available_features + ['Target_Up'])
        
        if len(train_df) < 100:
            print("Warning: Not enough data for supervised model!")
            df['Trend_Prob'] = 0.5
            return df
        
        X_train = train_df[available_features]
        y_train = train_df['Target_Up']
        
        # Train model
        print("\nTraining Supervised Intuition Model...")
        self.supervised_model.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = self.supervised_model.score(X_train, y_train)
        print(f"  Training Accuracy: {train_accuracy:.3f}")
        
        # Feature importance
        importances = self.supervised_model.feature_importances_
        print("  Feature Importance:")
        for feat, imp in sorted(zip(available_features, importances), 
                               key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {feat}: {imp:.3f}")
        
        # Predict on entire dataset
        X_full = df[available_features].fillna(0)
        
        try:
            probs = self.supervised_model.predict_proba(X_full)[:, 1]
            df['Trend_Prob'] = probs
        except Exception as e:
            print(f"Warning: Prediction failed ({e}), using default 0.5")
            df['Trend_Prob'] = 0.5
        
        return df
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main function: Load data → Add indicators → Add time features → Train supervised → Scale
        
        Returns:
            train_df, test_df (both ready for DQN)
        """
        print(f"\n{'='*60}")
        print(f"DATA LOADER: {self.symbol}")
        print(f"{'='*60}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Interval: {self.interval}")
        
        # --- 1. DOWNLOAD DATA ---
        print("\n[1/6] Downloading data from yfinance...")
        try:
            raw_df = yf.download(
                self.symbol,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False
            )
        except Exception as e:
            print(f"ERROR: Failed to download data: {e}")
            return None, None
        
        # MultiIndex fix
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        
        raw_df = raw_df.reset_index()
        
        if len(raw_df) < 100:
            print(f"ERROR: Not enough data ({len(raw_df)} rows)")
            return None, None
        
        print(f"  Downloaded {len(raw_df)} rows")
        
        # --- 2. TECHNICAL INDICATORS ---
        print("\n[2/6] Adding technical indicators...")
        df = TechnicalIndicators.add_all_indicators(raw_df)
        print(f"  Added {len(TechnicalIndicators.get_feature_columns())} indicators")
        
        # --- 3. CYCLICAL TIME FEATURES ---
        print("\n[3/6] Adding cyclical time features...")
        df = self._add_cyclical_features(df)
        cyclical_cols = [c for c in df.columns if '_sin' in c or '_cos' in c]
        print(f"  Added {len(cyclical_cols)} time features")
        
        # --- 4. SUPERVISED INTUITION ---
        print("\n[4/6] Training supervised intuition model...")
        df = self._train_supervised_intuition(df)
        
        # --- 5. FEATURE SELECTION ---
        print("\n[5/6] Preparing features...")
        
        # Get technical features
        feature_cols = TechnicalIndicators.get_feature_columns()
        
        # Add cyclical features
        feature_cols.extend(cyclical_cols)
        
        # Verify all features exist
        missing = [f for f in feature_cols if f not in df.columns]
        if missing:
            print(f"  Warning: Missing features: {missing}")
            feature_cols = [f for f in feature_cols if f in df.columns]
        
        print(f"  Total features: {len(feature_cols)}")
        
        # Handle NaN
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # --- 6. TRAIN/TEST SPLIT & SCALING ---
        print("\n[6/6] Train/Test split & scaling...")
        
        split_idx = int(len(df) * (1 - self.test_split))
        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
        
        # Store original prices
        self.original_prices_train = train_df['Close'].values
        self.original_prices_test = test_df['Close'].values
        
        # Scale features (FIT on train, TRANSFORM on both)
        train_features_scaled = self.feature_scaler.fit_transform(train_df[feature_cols])
        test_features_scaled = self.feature_scaler.transform(test_df[feature_cols])
        
        # Replace in DataFrames
        train_df[feature_cols] = train_features_scaled
        test_df[feature_cols] = test_features_scaled
        
        print(f"  Train samples: {len(train_df)}")
        print(f"  Test samples: {len(test_df)}")
        
        print(f"\n{'='*60}")
        print("DATA PREPARATION COMPLETE")
        print(f"{'='*60}\n")
        
        return train_df, test_df


if __name__ == "__main__":
    # Quick test
    loader = DataLoader(symbol='BTC-USD', start_date='2023-01-01', end_date='2023-12-31')
    train_df, test_df = loader.prepare_data()
    
    if train_df is not None:
        print(f"\nTest successful!")
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
