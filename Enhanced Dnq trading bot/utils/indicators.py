import pandas as pd
import numpy as np

class TechnicalIndicators:
    """
    Die vollständige Sammlung: Trend, Momentum, Volatilität, Volumen & Ichimoku.
    """
    @staticmethod
    def add_all_indicators(df: pd.DataFrame):
        df = df.copy()
        
        # --- 1. TREND (MAs & MACD) ---
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # --- 2. ICHIMOKU CLOUD ---
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        high_9 = df['High'].rolling(window=9).max()
        low_9 = df['Low'].rolling(window=9).min()
        df['Ichi_Tenkan'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        high_26 = df['High'].rolling(window=26).max()
        low_26 = df['Low'].rolling(window=26).min()
        df['Ichi_Kijun'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A): (Conversion + Base) / 2
        # Wir verschieben NICHT in die Zukunft, damit der Bot den *aktuellen* Cloud-Wert hat
        df['Ichi_Senkou_A'] = ((df['Ichi_Tenkan'] + df['Ichi_Kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
        high_52 = df['High'].rolling(window=52).max()
        low_52 = df['Low'].rolling(window=52).min()
        df['Ichi_Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
        
        # --- 3. MOMENTUM (RSI & ADX) ---
        df['Returns'] = df['Close'].pct_change()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ADX (Trendstärke) - Vereinfachte Berechnung
        plus_dm = df['High'].diff()
        minus_dm = df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        
        tr = pd.concat([
            df['High'] - df['Low'], 
            (df['High'] - df['Close'].shift()).abs(), 
            (df['Low'] - df['Close'].shift()).abs()
        ], axis=1).max(axis=1)
        
        atr14 = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr14)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        df['ADX'] = dx.rolling(window=14).mean()
        
        # --- 4. VOLUMEN (OBV) ---
        # On-Balance Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

        # --- 5. VOLATILITÄT (BB & ATR) ---
        df['BB_Mid'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Mid'] + (std * 2)
        df['BB_Lower'] = df['BB_Mid'] - (std * 2)
        
        df['ATR'] = atr14 # Wurde oben für ADX schon berechnet
        
        # NaN Werte auffüllen (wichtig wegen der Shifts beim Ichimoku)
        return df.fillna(method='bfill').fillna(0)

    @staticmethod
    def get_feature_columns():
        return [
            # Trend
            'MA5', 'MA20', 'MA50', 'MACD', 'MACD_Signal', 'ADX',
            # Ichimoku
            'Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_Senkou_A', 'Ichi_Senkou_B',
            # Momentum
            'RSI', 'Returns', 
            # Volumen
            'Vol_Ratio', 'OBV',
            # Volatilität
            'BB_Mid', 'BB_Upper', 'BB_Lower', 'ATR', 
            # KI-Intuition
            'Trend_Prob'
        ]