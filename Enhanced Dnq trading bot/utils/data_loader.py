import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .indicators import TechnicalIndicators

class DataLoader:
    """
    Verwaltet das Laden von Daten, Feature-Engineering (Technisch + Zyklisch) 
    und das Supervised-Modell.
    """
    def __init__(self, symbol='BTC-USD', start_date='2020-01-01', end_date='2025-12-15', 
                 interval='1d', test_split=0.15):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.test_split = test_split
        
        # Speicher für echte Preise (wichtig für Env-Berechnungen/Backtest)
        self.original_prices_train = None
        self.original_prices_test = None
        
        # Modelle und Scaler
        self.supervised_model = LogisticRegression(max_iter=1000)
        self.feature_scaler = StandardScaler()

    def _add_cyclical_features(self, df):
        """
        Wandelt Datumsinformationen in zyklische Sinus/Cosinus Features um.
        """
        # Sicherstellen, dass 'Date' im DataFrame verfügbar ist
        if 'Date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            
        if 'Date' in df.columns:
            # Datetime-Konvertierung zur Sicherheit
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Monat (Jahreszyklus)
            df['month_sin'] = np.sin(2 * np.pi * df['Date'].dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['Date'].dt.month / 12)
            
            # Wochentag (Wochenzyklus)
            df['day_sin'] = np.sin(2 * np.pi * df['Date'].dt.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['Date'].dt.dayofweek / 7)
            
            # Bei Intraday-Daten (Stunden/Minuten) auch die Uhrzeit zyklisch machen
            if 'h' in self.interval or 'm' in self.interval:
                df['hour_sin'] = np.sin(2 * np.pi * df['Date'].dt.hour / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['Date'].dt.hour / 24)
                
        return df

    def _train_supervised_intuition(self, df: pd.DataFrame):
        """
        Trainiert die "Intuition" (Logistic Regression) basierend auf Momentum-Indikatoren.
        """
        # Target: 1 wenn Preis der nächsten Kerze höher ist, sonst 0
        df['Target_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Features für die Intuition (müssen in indicators.py berechnet worden sein)
        intuition_features = ['RSI', 'Returns', 'Vol_Ratio']
        
        # Training nur auf dem ersten Teil der Daten (verhindert Data Leakage)
        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx].copy().dropna()
        
        # Sicherstellen, dass keine NaNs das Training crashen
        X = train_df[intuition_features].fillna(0)
        y = train_df['Target_Up']
        
        if len(X) > 0:
            self.supervised_model.fit(X, y)
            # Vorhersage für den gesamten DataFrame
            all_X = df[intuition_features].fillna(0)
            df['Trend_Prob'] = self.supervised_model.predict_proba(all_X)[:, 1]
        else:
            # Fallback, falls nicht genügend Daten
            df['Trend_Prob'] = 0.5 
            
        return df

    def prepare_data(self):
        """
        Hauptfunktion: Lädt Raw Data -> Fügt Indikatoren hinzu -> Split -> Skalierung.
        """
        print(f"Lade Daten für {self.symbol}...")
        try:
            raw_df = yf.download(self.symbol, start=self.start_date, end=self.end_date, 
                                 interval=self.interval, progress=False)
        except Exception as e:
            print(f"Fehler beim Download: {e}")
            return None, None
        
        # MultiIndex-Korrektur (Problem bei neueren yfinance Versionen)
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        raw_df = raw_df.reset_index()

        if len(raw_df) < 50:
            print("Zu wenige Daten geladen.")
            return None, None

        # --- SCHRITT 1: Technische Indikatoren (Ichimoku, MACD, etc.) ---
        # Ruft deine neue "Alles drin"-Klasse auf
        df = TechnicalIndicators.add_all_indicators(raw_df)
        
        # --- SCHRITT 2: Zyklische Zeit-Features ---
        df = self._add_cyclical_features(df)
        
        # --- SCHRITT 3: Supervised Intuition (Trend_Prob) ---
        df = self._train_supervised_intuition(df)
        
        # --- SCHRITT 4: Feature-Auswahl ---
        # Holt die Liste der technischen Features (ca. 20 Stk.)
        feature_cols = TechnicalIndicators.get_feature_columns()
        
        # Fügt automatisch alle zyklischen Features hinzu (_sin, _cos)
        cyclic_cols = [c for c in df.columns if '_sin' in c or '_cos' in c]
        feature_cols.extend(cyclic_cols)
        
        # Überprüfung
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            print(f"Warnung: Folgende Features fehlen: {missing_cols}")
        
        # NaN-Bereinigung (wichtig für neuronale Netze)
        df[feature_cols] = df[feature_cols].fillna(0)

        # --- SCHRITT 5: Train/Test Split ---
        split_idx = int(len(df) * (1 - self.test_split))
        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
        
        # Preise speichern für Backtesting/Env
        self.original_prices_train = train_df['Close'].values
        self.original_prices_test = test_df['Close'].values
        
        # --- SCHRITT 6: Skalierung (StandardScaler) ---
        # Fit NUR auf Trainingsdaten um Data Leakage zu verhindern
        train_features_scaled = self.feature_scaler.fit_transform(train_df[feature_cols])
        # Transform auf Testdaten mit den Parametern vom Training
        test_features_scaled = self.feature_scaler.transform(test_df[feature_cols])
        
        # Überschreiben der Daten im DF mit den skalierten Werten
        train_df[feature_cols] = train_features_scaled
        test_df[feature_cols] = test_features_scaled
        
        print("-" * 30)
        print(f"Datenvorbereitung abgeschlossen.")
        print(f"Features gesamt: {len(feature_cols)}")
        print(f"Feature Liste: {feature_cols}")
        print(f"Train Samples: {len(train_df)} | Test Samples: {len(test_df)}")
        print("-" * 30)
        
        return train_df, test_df