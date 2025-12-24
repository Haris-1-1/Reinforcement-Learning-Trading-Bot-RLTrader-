import pandas as pd
import numpy as np
from typing import List

class TechnicalIndicators:
    """
    Erweiterte Indikatoren für den Enhanced DQN.
    Fokus: Whale-Tracking, Marktstruktur und Zeit-Zyklen.
    """

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Sicherstellen, dass Spaltennamen sauber sind (yfinance MultiIndex Fix)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float)

        # --- 1. SMART MONEY / WHALE TRACKING ---
        # A-D Line (Accumulation/Distribution): Erkennt Divergenzen zwischen Preis und Volumen
        ad_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, 1)
        df['AD_Line'] = (ad_multiplier * volume).cumsum()
        
        # Volumen-Anomalien: Aktuelles Volumen im Verhältnis zum 50-Tage-Schnitt
        df['Vol_Ratio'] = volume / volume.rolling(window=50).mean().replace(0, 1)

        # --- 2. TREND & STRUKTUR (Ichimoku & Bollinger) ---
        # Ichimoku Cloud - Tenkan-sen (9-period high + low)/2
        df['Tenkan_sen'] = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Ichimoku Cloud - Kijun-sen (26-period high + low)/2
        df['Kijun_sen'] = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Bollinger Bänder: Misst Überkauft/Überverkauft Zustände
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        df['BB_Upper'] = sma20 + (2 * std20)
        df['BB_Lower'] = sma20 - (2 * std20)
        # BB_Pos: Relative Position im Band (0 = Unten, 1 = Oben)
        df['BB_Pos'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']).replace(0, 0.001)

        # --- 3. RISIKOMANAGEMENT (ATR) ---
        # ATR (Average True Range): Basis für dynamische Volatilitätsmessung
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()

        # --- 4. CYCLICAL TIME ENCODING ---
        # Wandelt Zeit in Sinus/Cosinus um, damit 23 Uhr nah an 0 Uhr liegt
        if 'Date' in df.columns or isinstance(df.index, pd.DatetimeIndex):
            # Sicherstellen, dass es ein Datetime-Objekt ist
            dates = pd.to_datetime(df['Date']) if 'Date' in df.columns else df.index
            
            # .dt Accessor verwenden, um auf Zeitkomponenten einer Series zuzugreifen
            df['Hour_Sin'] = np.sin(2 * np.pi * dates.dt.hour / 24)
            df['Hour_Cos'] = np.cos(2 * np.pi * dates.dt.hour / 24)
            
            # Wochentag (0-6)
            df['Day_Sin'] = np.sin(2 * np.pi * dates.dt.dayofweek / 7)
            df['Day_Cos'] = np.cos(2 * np.pi * dates.dt.dayofweek / 7)
        else:
            # Fallback falls keine Zeitdaten vorhanden
            df['Hour_Sin'] = df['Hour_Cos'] = df['Day_Sin'] = df['Day_Cos'] = 0.0
        # --- 5. KLASSISCHE OSZILLATOREN ---
        # RSI (Relative Strength Index)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Returns (Wichtig für das Momentum-Verständnis des Agenten)
        df['Returns'] = close.pct_change()

        # Cleanup: Entferne Zeilen mit NaN (entstehen durch Rolling Windows)
        df = df.dropna()
        return df

    @staticmethod
    def get_feature_columns() -> List[str]:
        """Gibt die Liste der Spalten zurück, die der Agent als Input sieht."""
        return [
            'RSI', 'Vol_Ratio', 'AD_Line', 'Returns',
            'BB_Pos', 'ATR', 'Tenkan_sen', 'Kijun_sen',
            'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos',
            'Trend_Prob' # Wird im DataLoader durch das Supervised Modell ergänzt
        ]