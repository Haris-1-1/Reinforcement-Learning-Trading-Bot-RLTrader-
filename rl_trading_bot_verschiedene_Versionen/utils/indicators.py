import pandas as pd
import numpy as np
from typing import List


class TechnicalIndicators:

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = pd.to_numeric(df['Close'], errors='coerce')
        high = pd.to_numeric(df['High'], errors='coerce')
        low = pd.to_numeric(df['Low'], errors='coerce')
        volume = pd.to_numeric(df['Volume'], errors='coerce')

        df['MA5'] = close.rolling(window=5).mean()
        df['MA20'] = close.rolling(window=20).mean()
        df['MA50'] = close.rolling(window=50).mean()

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        df['BB_middle'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle'].replace(0, np.nan)

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()

        df['Volume_MA'] = volume.rolling(window=20).mean()
        df['Volume_ratio'] = volume / df['Volume_MA'].replace(0, np.nan)

        df['Returns'] = close.pct_change()
        df['Log_returns'] = np.log(close / close.shift(1))

        df = df.dropna()

        print(f"Added technical indicators. Remaining rows after dropna: {len(df)}")

        return df

    @staticmethod
    def get_feature_columns() -> List[str]:
        return [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA20', 'MA50',
            'RSI',
            'MACD', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
            'ATR',
            'Volume_MA', 'Volume_ratio',
            'Returns', 'Log_returns'
        ]

    @staticmethod
    def normalize_features(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
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
