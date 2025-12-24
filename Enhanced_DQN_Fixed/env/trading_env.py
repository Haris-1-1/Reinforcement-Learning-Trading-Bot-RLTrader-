"""
TRADING ENVIRONMENT - Window-Based with Action Masking
=======================================================
Gymnasium-compatible environment for DQN trading
"""

import numpy as np
import pandas as pd

class TradingEnvironment:
    """
    Window-based Trading Environment for Enhanced DQN
    
    Features:
    - Window-based observations (看 zurück in die Historie)
    - Action Masking (verhindert invalide Trades)
    - Portfolio tracking
    - Trading fees & slippage
    - Log returns for reward
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        original_prices: np.ndarray,
        window_size: int = 24,
        initial_cash: float = 10000.0,
        fee: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Args:
            df: DataFrame mit Features (bereits skaliert!)
            original_prices: Echte Preise für Portfolio-Berechnung
            window_size: Anzahl Zeitschritte für Observation Window
            initial_cash: Startkapital
            fee: Trading fee (0.001 = 0.1%)
            slippage: Slippage (0.0005 = 0.05%)
        """
        self.df = df.reset_index(drop=True)
        self.prices = original_prices.astype(float)
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.fee = fee
        self.slippage = slippage
        
        # CRITICAL: Remove Date column if it exists (causes Timestamp error)
        if 'Date' in self.df.columns:
            self.df = self.df.drop('Date', axis=1)
        
        # Validation
        if len(self.df) != len(self.prices):
            raise ValueError(f"DataFrame length ({len(self.df)}) != prices length ({len(self.prices)})")
        
        # Feature columns (alle außer Open/High/Low/Close/Volume)
        excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        potential_features = [col for col in self.df.columns if col not in excluded_cols]
        
        # Only keep numeric columns
        self.feature_cols = []
        for col in potential_features:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.feature_cols.append(col)
            else:
                print(f"  Warning: Skipping non-numeric column '{col}'")
        
        self.n_features = len(self.feature_cols)
        
        print(f"Environment initialized:")
        print(f"  Window Size: {self.window_size}")
        print(f"  Features per step: {self.n_features}")
        print(f"  Total observation size: {self.window_size * self.n_features + 3}")  # +3 for portfolio features
        print(f"  Steps available: {len(self.df)}")
        print(f"  Initial Cash: ${self.initial_cash:.2f}")
        
        # State variables (initialized in reset())
        self.current_step = 0
        self.cash = 0.0
        self.coins = 0.0
        self.position = 0.0  # 0.0 = flat, 1.0 = fully invested
        self.entry_price = 0.0
        self.portfolio_value = 0.0
        self.last_portfolio_value = 0.0
        
        # Tracking
        self.trades = []
        self.trade_count = 0
    
    def _get_price(self, step: int) -> float:
        """Get real price at given step"""
        return float(self.prices[step])
    
    def get_action_mask(self) -> np.ndarray:
        """
        Returns action mask [Hold, Buy, Sell]
        1 = allowed, 0 = forbidden
        """
        mask = np.ones(3, dtype=np.int8)
        
        # Can't buy if already fully invested
        if self.position >= 0.99:
            mask[1] = 0
        
        # Can't sell if no position
        if self.position <= 0.01:
            mask[2] = 0
        
        return mask
    
    def _get_observation(self) -> np.ndarray:
        """
        Creates windowed observation
        
        Returns:
            1D array of size: window_size * n_features + 3
        """
        # 1. Get market data window
        if self.current_step < self.window_size:
            # Pad with first row if needed
            padding_needed = self.window_size - (self.current_step + 1)
            real_data = self.df.iloc[0:self.current_step + 1][self.feature_cols].values
            padding = np.tile(real_data[0], (padding_needed, 1))
            window_data = np.vstack([padding, real_data])
        else:
            start_idx = self.current_step - self.window_size + 1
            end_idx = self.current_step + 1
            window_data = self.df.iloc[start_idx:end_idx][self.feature_cols].values
        
        # 2. Flatten window - convert to float32 with error handling
        try:
            flat_window = window_data.astype(np.float32).flatten()
        except (ValueError, TypeError) as e:
            # Fallback: force conversion column by column
            print(f"Warning: Data conversion issue at step {self.current_step}: {e}")
            flat_list = []
            for row in window_data:
                for val in row:
                    try:
                        flat_list.append(float(val))
                    except:
                        flat_list.append(0.0)
            flat_window = np.array(flat_list, dtype=np.float32)
        
        # 3. Portfolio features
        current_price = self._get_price(self.current_step)
        
        # Unrealized PnL
        if self.position > 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = 0.0
        
        portfolio_features = np.array([
            self.cash / self.initial_cash,  # Cash ratio (normalized)
            self.position,                   # Position (0 or 1)
            unrealized_pnl                   # Current trade performance
        ], dtype=np.float32)
        
        # 4. Concatenate
        observation = np.concatenate([flat_window, portfolio_features])
        
        return observation
    
    def _get_info(self) -> dict[str, any]:
        """Returns info dict with current state"""
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'cash': self.cash,
            'coins': self.coins,
            'action_mask': self.get_action_mask(),
            'trade_count': self.trade_count,
            'profit': self.portfolio_value - self.initial_cash
        }
    
    def reset(self) -> tuple[np.ndarray, dict]:
        """
        Reset environment to initial state
        
        Returns:
            observation, info
        """
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
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        current_price = self._get_price(self.current_step)
        self.last_portfolio_value = self.portfolio_value
        
        # --- ACTION EXECUTION ---
        
        # Check if action is valid
        mask = self.get_action_mask()
        if mask[action] == 0:
            # Invalid action! Force to Hold
            action = 0
        
        # BUY
        if action == 1 and self.cash > 0:
            exec_price = current_price * (1 + self.slippage)
            fee_amount = self.cash * self.fee
            cost = self.cash - fee_amount
            
            self.coins = cost / exec_price
            self.cash = 0.0
            self.position = 1.0
            self.entry_price = exec_price
            self.trade_count += 1
            
            self.trades.append({
                'step': self.current_step,
                'action': 'BUY',
                'price': exec_price,
                'value': self.portfolio_value
            })
        
        # SELL
        elif action == 2 and self.coins > 0:
            exec_price = current_price * (1 - self.slippage)
            gross_value = self.coins * exec_price
            fee_amount = gross_value * self.fee
            
            self.cash = gross_value - fee_amount
            self.coins = 0.0
            self.position = 0.0
            self.entry_price = 0.0
            self.trade_count += 1
            
            self.trades.append({
                'step': self.current_step,
                'action': 'SELL',
                'price': exec_price,
                'value': self.portfolio_value
            })
        
        # --- REWARD CALCULATION ---
        
        # Update portfolio value
        new_price = self._get_price(self.current_step)
        self.portfolio_value = self.cash + (self.coins * new_price)
        
        # Log returns (numerically stable)
        if self.last_portfolio_value > 0:
            reward = np.log(self.portfolio_value / self.last_portfolio_value)
        else:
            reward = 0.0
        
        # Small penalty for trading (encourages meaningful trades only)
        if action != 0:
            reward -= 0.0001
        
        # --- NEXT STEP ---
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def get_trades_df(self) -> pd.DataFrame:
        """Returns DataFrame of all trades"""
        return pd.DataFrame(self.trades)
    
    def get_portfolio_history(self) -> dict:
        """Returns portfolio history for plotting"""
        return {
            'final_value': self.portfolio_value,
            'total_return': (self.portfolio_value - self.initial_cash) / self.initial_cash,
            'total_trades': self.trade_count,
            'trades': self.trades
        }


if __name__ == "__main__":
    # Quick test
    print("Trading Environment Module")
    print("Features:")
    print("  - Window-based observations")
    print("  - Action masking")
    print("  - Portfolio tracking")
    print("  - Trading fees & slippage")