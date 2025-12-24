import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List

class AdvancedTradingEnv(gym.Env):
    """
    Enhanced Trading Environment für DQN mit Windowing und Action Masking.
    """
    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        original_prices: np.ndarray,
        initial_cash: float = 10000.0,
        trading_fee_maker: float = 0.001,
        trading_fee_taker: float = 0.002,
        slippage: float = 0.001,
        trade_frequency_penalty: float = 0.00005,
        window_size: int = 10,  # NEU: Das Gedächtnis des Agenten
        feature_columns: Optional[List[str]] = None
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.prices = original_prices.astype(float)
        self.initial_cash = float(initial_cash)
        
        # Einstellungen
        self.trading_fee_maker = trading_fee_maker
        self.trading_fee_taker = trading_fee_taker
        self.slippage = slippage
        self.trade_frequency_penalty = trade_frequency_penalty
        self.window_size = window_size

        # Validation
        if len(self.prices) != len(self.df):
            raise ValueError("Länge von df und original_prices muss übereinstimmen!")

        # 1. Feature Definition (Smart Money Features müssen im df sein)
        # Wir erwarten, dass df bereits A-D Line, Vol_Ratio, Time Encoding etc. enthält
        if feature_columns is None:
            # Fallback: Alle numerischen Spalten außer Datum/Target
            self.feature_columns = [col for col in df.columns if col not in ['Date', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        else:
            self.feature_columns = feature_columns

        self.n_market_features = len(self.feature_columns)
        self.n_portfolio_features = 3  # Cash-Ratio, Position, Unrealized PnL

        print(f"Environment initialisiert:")
        print(f" - Window Size: {self.window_size}")
        print(f" - Features pro Step: {self.n_market_features}")
        print(f" - Total Input Size (DQN): {self.window_size * self.n_market_features + self.n_portfolio_features}")

        # 2. Observation Space (Das Fenster + Portfolio Status)
        # Shape: (Window_Size * Market_Features) + Portfolio_Features
        total_obs_size = (self.window_size * self.n_market_features) + self.n_portfolio_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )

        # 3. Action Space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Interne Variablen
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        # Portfolio State
        self.cash = 0.0
        self.coins = 0.0
        self.position = 0.0 # 0.0 = Flat, 1.0 = Invested
        self.entry_price = 0.0 # Für PnL Berechnung
        
        # Tracking
        self.portfolio_value = 0.0
        self.last_portfolio_value = 0.0
        self.trades = []
        self.trade_count = 0
        self.last_trade_step = -100

    def _get_price(self, step: int) -> float:
        return float(self.prices[step])

    def get_action_mask(self) -> np.ndarray:
        """
        Gibt eine Maske zurück, welche Aktionen gerade erlaubt sind.
        [Hold, Buy, Sell] -> 1 = Erlaubt, 0 = Verboten
        """
        mask = np.array([1, 1, 1], dtype=np.int8)
        
        # Wenn wir voll investiert sind, können wir nicht kaufen
        if self.position >= 0.99:
            mask[1] = 0 # Buy verboten
            
        # Wenn wir gar nichts haben, können wir nicht verkaufen
        if self.position <= 0.01:
            mask[2] = 0 # Sell verboten
            
        return mask

    def _get_observation(self) -> np.ndarray:
        """
        Erstellt das 'Windowed' Observation Array.
        """
        # 1. Market Data Window holen
        # Wenn wir am Anfang sind (step < window_size), müssen wir padden (auffüllen)
        if self.current_step < self.window_size:
            # Padding mit Nullen oder der ersten Zeile
            padding_needed = self.window_size - (self.current_step + 1)
            # Daten vom Start bis jetzt
            real_data = self.df.iloc[0 : self.current_step + 1][self.feature_columns].values
            # Padding erstellen (wir wiederholen einfach die erste Zeile, das ist neutraler als Nullen)
            padding = np.tile(real_data[0], (padding_needed, 1))
            window_data = np.vstack([padding, real_data])
        else:
            # Normaler Fall: Hole die letzten 'window_size' Zeilen
            start_idx = self.current_step - self.window_size + 1
            end_idx = self.current_step + 1
            window_data = self.df.iloc[start_idx : end_idx][self.feature_columns].values

        # 2. Flatten (Flachdrücken für MLP)
        flat_window = window_data.flatten().astype(np.float32)

        # 3. Portfolio Features berechnen
        current_price = self._get_price(self.current_step)
        
        # Unrealized PnL (nur relevant wenn investiert)
        if self.position > 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = 0.0

        portfolio_features = np.array([
            self.cash / self.initial_cash,  # Cash Ratio (Normalized)
            self.position,                  # Position (0 or 1)
            unrealized_pnl                  # Wie gut läuft der aktuelle Trade?
        ], dtype=np.float32)

        # 4. Zusammenfügen
        return np.concatenate([flat_window, portfolio_features])

    def _get_info(self) -> Dict:
        """Zusätzliche Infos, wichtig für Debugging und Masking."""
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'action_mask': self.get_action_mask(), # WICHTIG für den Agenten
            'trade_count': self.trade_count
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start etwas später, damit wir echte Historie für das Window haben, 
        # aber wir lösen das oben durch Padding, also Start bei 0 ist ok.
        self.current_step = 0
        
        self.cash = self.initial_cash
        self.coins = 0.0
        self.position = 0.0
        self.entry_price = 0.0
        
        self.portfolio_value = self.initial_cash
        self.last_portfolio_value = self.initial_cash
        
        self.trades = []
        self.trade_count = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        current_price = self._get_price(self.current_step)
        self.last_portfolio_value = self.portfolio_value
        
        # --- INVALID ACTION CHECK (Physikalisches Gesetz) ---
        # Wenn der Agent etwas Unmögliches versucht, zwingen wir ihn zu HOLD (0)
        # und geben ihm optional eine kleine Strafe.
        mask = self.get_action_mask()
        if mask[action] == 0:
            # Ungültige Aktion!
            # Wir ändern die Aktion zu HOLD, damit das Env nicht crasht
            action = 0 
            # Optional: Kleiner negativer Reward als "Erziehung", 
            # aber Masking im Agenten ist besser.
        
        # --- EXECUTION ---
        reward_penalty = 0.0
        
        # BUY
        if action == 1: 
            # Berechne Kosten
            exec_price = current_price * (1 + self.slippage)
            fee = self.cash * self.trading_fee_taker
            cost = self.cash - fee
            
            # Ausführen
            self.coins = cost / exec_price
            self.cash = 0.0
            self.position = 1.0
            self.entry_price = exec_price
            
            self.trade_count += 1
            self.last_trade_step = self.current_step
            
            # Logging
            self.trades.append({
                'step': self.current_step, 'action': 'BUY', 'price': exec_price, 
                'value': self.portfolio_value
            })

        # SELL
        elif action == 2:
            # Berechne Erlös
            exec_price = current_price * (1 - self.slippage)
            gross_value = self.coins * exec_price
            fee = gross_value * self.trading_fee_taker
            
            # Ausführen
            self.cash = gross_value - fee
            self.coins = 0.0
            self.position = 0.0
            self.entry_price = 0.0
            
            self.trade_count += 1
            self.last_trade_step = self.current_step
            
            # Logging
            self.trades.append({
                'step': self.current_step, 'action': 'SELL', 'price': exec_price, 
                'value': self.portfolio_value
            })
            
        # --- REWARD BERECHNUNG ---
        # 1. Neuer Portfolio Wert
        new_price = self._get_price(self.current_step) # Wir nehmen Close des gleichen Steps als Bewertung
        # (In Backtesting oft next Open, aber Close ist ok für RL Training)
        
        self.portfolio_value = self.cash + (self.coins * new_price)
        
        # 2. Percentage Returns (Log Returns sind numerisch stabiler für RL)
        if self.last_portfolio_value > 0:
            step_reward = np.log(self.portfolio_value / self.last_portfolio_value)
        else:
            step_reward = 0.0
            
        # 3. Frequency Penalty (gegen Overtrading)
        if action != 0:
            # Strafe jeden Trade leicht, damit er nur tradet wenn der Gewinn > Kosten ist
            step_reward -= self.trade_frequency_penalty

        # --- NEXT STEP ---
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def get_trades_df(self):
        return pd.DataFrame(self.trades)