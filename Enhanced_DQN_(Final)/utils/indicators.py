"""
ENHANCED TECHNICAL INDICATORS - COMPLETE VERSION
=================================================
42 Features: Smart Money, Advanced Momentum, Time, Structure
"""

import pandas as pd
import numpy as np
from typing import List

class TechnicalIndicators:
    """
    Vollständige Indikator-Suite für Enhanced DQN Trading Bot
    Features: Trend, Momentum, Volume, Volatility, Smart Money, Time
    """
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt ALLE technischen Indikatoren hinzu (42 Features)
        """
        df = df.copy()
        
        # Sicherstellen dass OHLCV vorhanden sind
        close = df['Close'].astype(float)
        high = df['High'].astype(float)
        low = df['Low'].astype(float)
        volume = df['Volume'].astype(float)
        
        # ========================================
        # 1. TREND INDICATORS
        # ========================================
        
        # Moving Averages
        df['MA5'] = close.rolling(window=5).mean()
        df['MA20'] = close.rolling(window=20).mean()
        df['MA50'] = close.rolling(window=50).mean()
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']  # NEW!
        
        # MA Crossovers (Binary Signals)
        df['MA_Cross_5_20'] = (df['MA5'] > df['MA20']).astype(int)
        df['MA_Cross_20_50'] = (df['MA20'] > df['MA50']).astype(int)
        
        # ADX (Trend Strength)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr14 = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr14)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr14)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        df['ADX'] = dx.rolling(window=14).mean()
        
        # ========================================
        # 2. ICHIMOKU CLOUD
        # ========================================
        
        # Tenkan-sen (Conversion Line)
        high_9 = high.rolling(window=9).max()
        low_9 = low.rolling(window=9).min()
        df['Ichi_Tenkan'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = high.rolling(window=26).max()
        low_26 = low.rolling(window=26).min()
        df['Ichi_Kijun'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        df['Ichi_Senkou_A'] = ((df['Ichi_Tenkan'] + df['Ichi_Kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        high_52 = high.rolling(window=52).max()
        low_52 = low.rolling(window=52).min()
        df['Ichi_Senkou_B'] = ((high_52 + low_52) / 2).shift(26)
        
        # Cloud Strength (Distance between spans)
        df['Cloud_Strength'] = (df['Ichi_Senkou_A'] - df['Ichi_Senkou_B']) / close
        
        # ========================================
        # 3. MOMENTUM INDICATORS
        # ========================================
        
        # Returns (Multiple Timeframes)
        df['Returns'] = close.pct_change()
        df['Returns_5'] = close.pct_change(periods=5)
        df['Returns_20'] = close.pct_change(periods=20)
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Stochastic Oscillator
        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        df['Stoch_K'] = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, 1)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # ========================================
        # 4. SMART MONEY / WHALE TRACKING
        # ========================================
        
        # Accumulation/Distribution Line
        clv = ((close - low) - (high - close)) / (high - low).replace(0, 1)
        df['AD_Line'] = (clv * volume).cumsum()
        
        # Volume Ratio (vs 50-day average)
        vol_mean_50 = volume.rolling(window=50).mean()
        df['Vol_Ratio'] = volume / vol_mean_50.replace(0, 1)
        
        # Volume Spike (Z-Score)
        vol_std = volume.rolling(window=50).std()
        df['Vol_Spike'] = (volume - vol_mean_50) / vol_std.replace(0, 1)
        
        # Money Flow Index (MFI) - Volume-weighted RSI
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        
        mfi_ratio = positive_mf / negative_mf.replace(0, np.nan)
        df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        
        # On-Balance Volume
        df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        # ========================================
        # 5. VOLATILITY INDICATORS
        # ========================================
        
        # Bollinger Bands
        bb_mid = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df['BB_Upper'] = bb_mid + (bb_std * 2)
        df['BB_Lower'] = bb_mid - (bb_std * 2)
        df['BB_Mid'] = bb_mid
        
        # BB Position (Where is price in the band?)
        df['BB_Pos'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']).replace(0, 0.001)
        df['BB_Pos'] = df['BB_Pos'].clip(0, 1)  # Keep between 0 and 1
        
        # BB Width (Volatility measure)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / bb_mid
        
        # ATR (Average True Range)
        df['ATR'] = atr14
        df['ATR_Pct'] = df['ATR'] / close  # Normalized
        
        # Historical Volatility
        df['Hist_Vol'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        # ========================================
        # 6. SUPPORT/RESISTANCE
        # ========================================
        
        # Simple S/R based on rolling min/max
        df['Support_20'] = low.rolling(window=20).min()
        df['Resistance_20'] = high.rolling(window=20).max()
        df['SR_Position'] = (close - df['Support_20']) / \
                            (df['Resistance_20'] - df['Support_20']).replace(0, 1)
        df['SR_Position'] = df['SR_Position'].clip(0, 1)
        
        # Price Distance from MAs
        df['Dist_MA20'] = (close - df['MA20']) / df['MA20']
        df['Dist_MA50'] = (close - df['MA50']) / df['MA50']
        
        # ========================================
        # 7. PRICE ACTION
        # ========================================
        
        # Price Range as % of close
        df['Price_Range'] = (high - low) / close
        
        # Gap (Open vs Previous Close)
        if 'Open' in df.columns:
            df['Gap'] = (df['Open'] - close.shift(1)) / close.shift(1)
        else:
            df['Gap'] = 0.0
        
        # ========================================
        # CLEANUP
        # ========================================
        
        # Replace inf/-inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Backfill then forward fill remaining NaNs
        df = df.bfill().ffill()
        
        # Final fill any remaining NaNs with 0
        df = df.fillna(0)
        
        return df
    
    @staticmethod
    def get_feature_columns() -> List[str]:
        """
        Returns list of ALL 42 technical features
        (Time features added by DataLoader)
        """
        return [
            # Trend (9)
            'MA5', 'MA20', 'MA50',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'MA_Cross_5_20', 'MA_Cross_20_50',
            'ADX',
            
            # Ichimoku (5)
            'Ichi_Tenkan', 'Ichi_Kijun',
            'Ichi_Senkou_A', 'Ichi_Senkou_B',
            'Cloud_Strength',
            
            # Momentum (6)
            'Returns', 'Returns_5', 'Returns_20',
            'RSI',
            'Stoch_K', 'Stoch_D',
            
            # Smart Money / Volume (6)
            'AD_Line', 'Vol_Ratio', 'Vol_Spike',
            'MFI', 'OBV',
            
            # Volatility (6)
            'BB_Upper', 'BB_Lower', 'BB_Mid',
            'BB_Pos', 'BB_Width',
            'ATR_Pct', 'Hist_Vol',
            
            # Structure (6)
            'Support_20', 'Resistance_20', 'SR_Position',
            'Dist_MA20', 'Dist_MA50',
            
            # Price Action (2)
            'Price_Range', 'Gap',
            
            # Supervised Signal (1) - Added by DataLoader
            'Trend_Prob'
        ]
    
    @staticmethod
    def get_feature_groups() -> dict:
        """Returns features grouped by category for analysis"""
        return {
            'trend': ['MA5', 'MA20', 'MA50', 'MACD', 'MACD_Signal', 'MACD_Hist', 
                      'MA_Cross_5_20', 'MA_Cross_20_50', 'ADX'],
            'ichimoku': ['Ichi_Tenkan', 'Ichi_Kijun', 'Ichi_Senkou_A', 
                         'Ichi_Senkou_B', 'Cloud_Strength'],
            'momentum': ['Returns', 'Returns_5', 'Returns_20', 'RSI', 'Stoch_K', 'Stoch_D'],
            'smart_money': ['AD_Line', 'Vol_Ratio', 'Vol_Spike', 'MFI', 'OBV'],
            'volatility': ['BB_Upper', 'BB_Lower', 'BB_Mid', 'BB_Pos', 'BB_Width', 
                          'ATR_Pct', 'Hist_Vol'],
            'structure': ['Support_20', 'Resistance_20', 'SR_Position', 
                          'Dist_MA20', 'Dist_MA50'],
            'price_action': ['Price_Range', 'Gap'],
            'supervised': ['Trend_Prob']
        }


if __name__ == "__main__":
    # Quick test
    print("Technical Indicators Module")
    print(f"Total Features: {len(TechnicalIndicators.get_feature_columns())}")
    print("\nFeature Groups:")
    for group, features in TechnicalIndicators.get_feature_groups().items():
        print(f"  {group}: {len(features)} features")
