import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .indicators import TechnicalIndicators

class DataLoader:
    """
    Verwaltet das Laden von Daten, Feature-Engineering und 
    das Supervised-Modell für die 'Trend-Vorahnung'.
    """
    def __init__(self, symbol='BTC-USD', start_date='2020-01-01', end_date='2025-12-15', 
                 interval='1d', test_split=0.15):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.test_split = test_split
        
        # Speicher für echte Preise (wichtig für Env-Berechnungen)
        self.original_prices_train = None
        self.original_prices_test = None
        
        # Das Supervised Modell für die "Vorahnung"
        self.supervised_model = LogisticRegression(max_iter=1000)
        self.feature_scaler = StandardScaler()

    def _train_supervised_intuition(self, df: pd.DataFrame):
        """
        Trainiert ein logistisches Regressionsmodell, um die 
        Wahrscheinlichkeit eines Kursanstiegs vorherzusagen.
        """
        # Target: 1 wenn Preis der nächsten Kerze höher ist, sonst 0
        df['Target_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Wir nutzen einfache Features für die Intuition
        intuition_features = ['RSI', 'Returns', 'Vol_Ratio']
        
        # Training nur auf dem ersten Teil der Daten (verhindert Data Leakage)
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx].copy().dropna()
        
        X = train_df[intuition_features]
        y = train_df['Target_Up']
        
        if len(X) > 0:
            self.supervised_model.fit(X, y)
            # Vorhersage-Wahrscheinlichkeit für den gesamten Datensatz
            probs = self.supervised_model.predict_proba(df[intuition_features].fillna(0))[:, 1]
            df['Trend_Prob'] = probs
        else:
            df['Trend_Prob'] = 0.5 # Fallback
            
        return df

    def prepare_data(self):
        """
        Lädt Daten von Yahoo Finance und bereitet sie für den Enhanced DQN vor.
        """
        print(f"Lade Daten für {self.symbol}...")
        raw_df = yf.download(self.symbol, start=self.start_date, end=self.end_date, 
                             interval=self.interval, progress=False)
        
        # MultiIndex Korrektur
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        raw_df = raw_df.reset_index()

        # 1. Technische Indikatoren hinzufügen
        df = TechnicalIndicators.add_all_indicators(raw_df)
        
        # 2. Supervised Intuition Feature generieren
        df = self._train_supervised_intuition(df)
        
        # 3. Train/Test Split
        split_idx = int(len(df) * (1 - self.test_split))
        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
        
        # 4. Echte Preise für das Environment extrahieren
        self.original_prices_train = train_df['Close'].values
        self.original_prices_test = test_df['Close'].values
        
        # 5. Normalisierung der Features für den DQN (Z-Score)
        feature_cols = TechnicalIndicators.get_feature_columns()
        
        # Fit nur auf Training, Transform auf beides (Wichtig!)
        train_features_scaled = self.feature_scaler.fit_transform(train_df[feature_cols])
        test_features_scaled = self.feature_scaler.transform(test_df[feature_cols])
        
        # Ersetze Werte im DF mit skalierten Werten
        train_df[feature_cols] = train_features_scaled
        test_df[feature_cols] = test_features_scaled
        
        print(f"Datenvorbereitung abgeschlossen. Train: {len(train_df)}, Test: {len(test_df)}")
        return train_df, test_df