import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .indicators import TechnicalIndicators
class DataLoader:
    def __init__(self, symbol='BTC-USD', start_date='2020-01-01', end_date='2025-12-15',
                 interval='1d', test_split=0.15):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.test_split = test_split
        self.original_prices_train = None
        self.original_prices_test = None
        self.supervised_model = LogisticRegression(max_iter=1000)
        self.feature_scaler = StandardScaler()
    def _add_cyclical_features(self, df):
        if 'Date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 7)
            if 'h' in self.interval or 'm' in self.interval:
                df['hour_sin'] = np.sin(2 * np.pi * df['Date'].dt.hour / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['Date'].dt.hour / 24)
        return df
    def _train_supervised_intuition(self, df: pd.DataFrame):
        df['Target_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        intuition_features = ['RSI', 'Returns', 'Vol_Ratio']
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx].copy().dropna()
        X = train_df[intuition_features].fillna(0)
        y = train_df['Target_Up']
        if len(X) > 0:
            self.supervised_model.fit(X, y)
            all_X = df[intuition_features].fillna(0)
            df['Trend_Prob'] = self.supervised_model.predict_proba(all_X)[:, 1]
        else:
            df['Trend_Prob'] = 0.5
        return df
    def prepare_data(self):
        print(f"Lade Daten für {self.symbol}...")
        try:
            raw_df = yf.download(self.symbol, start=self.start_date, end=self.end_date,
                                 interval=self.interval, progress=False)
        except Exception as e:
            print(f"Fehler beim Download: {e}")
            return None, None
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        raw_df = raw_df.reset_index()
        if len(raw_df) < 50:
            print("Zu wenige Daten geladen.")
            return None, None
        df = TechnicalIndicators.add_all_indicators(raw_df)
        df = self._add_cyclical_features(df)
        df = self._train_supervised_intuition(df)
        feature_cols = TechnicalIndicators.get_feature_columns()
        cyclic_cols = [c for c in df.columns if '_sin' in c or '_cos' in c]
        feature_cols.extend(cyclic_cols)
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            print(f"Warnung: Folgende Features fehlen: {missing_cols}")
        df[feature_cols] = df[feature_cols].fillna(0)
        split_idx = int(len(df) * (1 - self.test_split))
        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
        self.original_prices_train = train_df['Close'].values
        self.original_prices_test = test_df['Close'].values
        train_features_scaled = self.feature_scaler.fit_transform(train_df[feature_cols])
        test_features_scaled = self.feature_scaler.transform(test_df[feature_cols])
        train_df[feature_cols] = train_features_scaled
        test_df[feature_cols] = test_features_scaled
        print("-" * 30)
        print(f"Datenvorbereitung abgeschlossen.")
        print(f"Features gesamt: {len(feature_cols)}")
        print(f"Feature Liste: {feature_cols}")
        print(f"Train Samples: {len(train_df)} | Test Samples: {len(test_df)}")
        print("-" * 30)
        return train_df, test_df