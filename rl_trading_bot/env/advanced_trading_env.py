"""
Advanced Trading Environment for Reinforcement Learning.
FIX: Verwendet ECHTE Preise für Trading, normalisierte Features für State.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List


class AdvancedTradingEnv(gym.Env):
    """
    Advanced trading environment with realistic constraints.
    
    WICHTIG: 
    - State (Observation) = Normalisierte Features (für besseres Lernen)
    - Trading Preise = ECHTE USD Preise (für realistisches P&L)
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        original_prices: np.ndarray,  # NEU: Echte Preise!
        initial_cash: float = 10000.0,
        trading_fee_maker: float = 0.001,
        trading_fee_taker: float = 0.002,
        slippage: float = 0.001,
        trade_frequency_penalty: float = 0.00005,  # REDUZIERT von 0.0001
        max_position_size: float = 1.0,
        feature_columns: Optional[List[str]] = None
    ):
        """
        Initialize Trading Environment.

        Args:
            df: DataFrame mit normalisierten Features
            original_prices: Array mit ECHTEN USD Preisen (gleiche Länge wie df!)
            initial_cash: Startkapital
            trading_fee_maker: Maker Fee (0.1%)
            trading_fee_taker: Taker Fee (0.2%)
            slippage: Slippage (0.1%)
            trade_frequency_penalty: Strafe für zu häufiges Trading
            max_position_size: Max Position (1.0 = 100%)
            feature_columns: Liste der Feature-Spalten
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_cash = float(initial_cash)

        # ════════════════════════════════════════════════════════════
        # NEU: ECHTE PREISE für Trading!
        # ════════════════════════════════════════════════════════════
        self.prices = original_prices.astype(float)
        
        # Validierung
        if len(self.prices) != len(self.df):
            raise ValueError(f"Prices length ({len(self.prices)}) must match df length ({len(self.df)})")
        
        print(f"✅ Using REAL prices for trading: ${self.prices.min():.2f} - ${self.prices.max():.2f}")
        print(f"   Price change over period: {((self.prices[-1]/self.prices[0])-1)*100:.1f}%")

        # Trading constraints
        self.trading_fee_maker = trading_fee_maker
        self.trading_fee_taker = trading_fee_taker
        self.slippage = slippage
        self.trade_frequency_penalty = trade_frequency_penalty
        self.max_position_size = max_position_size

        # Feature columns
        default_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA20', 'MA50', 'RSI',
            'MACD', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
            'ATR', 'Volume_MA', 'Volume_ratio',
            'Returns', 'Log_returns'
        ]
        
        if feature_columns is None:
            self.feature_columns = [col for col in default_features if col in self.df.columns]
        else:
            self.feature_columns = [col for col in feature_columns if col in self.df.columns]
        
        print(f"   Using {len(self.feature_columns)} features for state")

        n_features = len(self.feature_columns) + 3  # +3 for portfolio state

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell

        # Episode tracking
        self.current_step = 0
        self.max_steps = len(self.df) - 1

        # Portfolio state
        self.cash = 0.0
        self.coins = 0.0
        self.position = 0.0
        self.portfolio_value = 0.0
        self.last_portfolio_value = 0.0

        # Trading history
        self.trades: List[Dict] = []
        self.trade_count = 0
        self.last_trade_step = -100  # Erlaube frühe Trades

        # Performance tracking
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

    def _get_price(self, step: int) -> float:
        """Get REAL trading price at step."""
        return float(self.prices[step])

    def _get_observation(self) -> np.ndarray:
        """Get normalized state for agent."""
        row = self.df.iloc[self.current_step]
        market_features = []
        for col in self.feature_columns:
            val = row[col]
            if hasattr(val, 'item'):
                val = val.item()
            market_features.append(float(val))
        
        market_features = np.array(market_features, dtype=np.float32)

        # Portfolio features
        portfolio_features = np.array([
            self.cash / self.initial_cash,
            self.position,
            self.portfolio_value / self.initial_cash
        ], dtype=np.float32)

        obs = np.concatenate([market_features, portfolio_features])
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get current info."""
        return {
            'step': self.current_step,
            'cash': self.cash,
            'coins': self.coins,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'current_price': self._get_price(self.current_step),
            'trade_count': self.trade_count,
            'total_fees_paid': self.total_fees_paid,
            'total_slippage_cost': self.total_slippage_cost
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_cash
        self.coins = 0.0
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        self.last_portfolio_value = self.initial_cash

        self.trades = []
        self.trade_count = 0
        self.last_trade_step = -100
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step."""
        current_price = self._get_price(self.current_step)
        self.last_portfolio_value = self.portfolio_value

        # Execute trade
        self._execute_trade(action, current_price)

        # Move forward
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Calculate new portfolio value
        new_price = self._get_price(min(self.current_step, self.max_steps - 1))
        self.portfolio_value = self.cash + self.coins * new_price

        # Calculate reward
        if self.last_portfolio_value > 0:
            reward = (self.portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        else:
            reward = 0.0

        # Trade frequency penalty (nur wenn zu schnell hintereinander)
        if action != 0 and self.trade_count > 1:
            steps_since_last = self.current_step - self.last_trade_step
            if 0 < steps_since_last < 3:  # Nur wenn < 3 Tage
                reward -= self.trade_frequency_penalty

        reward = float(np.clip(reward, -1.0, 1.0))
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _execute_trade(self, action: int, current_price: float):
        """Execute trade with fees and slippage."""
        
        # BUY
        if action == 1 and self.position < 0.5 and self.cash > 0:
            exec_price = current_price * (1 + self.slippage)
            fee_rate = self.trading_fee_taker
            available = self.cash / (1 + fee_rate)
            
            self.coins = available / exec_price
            fee_paid = self.cash - available
            
            self.cash = 0.0
            self.position = 1.0
            
            self.total_fees_paid += fee_paid
            self.total_slippage_cost += (exec_price - current_price) * self.coins
            
            self.trades.append({
                'step': self.current_step,
                'action': 'BUY',
                'price': exec_price,
                'amount': self.coins,
                'fee': fee_paid,
                'portfolio_value': self.portfolio_value
            })
            self.trade_count += 1
            self.last_trade_step = self.current_step
            
        # SELL
        elif action == 2 and self.position > 0.5 and self.coins > 0:
            exec_price = current_price * (1 - self.slippage)
            fee_rate = self.trading_fee_taker
            
            gross = self.coins * exec_price
            fee_paid = gross * fee_rate
            
            self.cash = gross - fee_paid
            
            self.total_fees_paid += fee_paid
            self.total_slippage_cost += (current_price - exec_price) * self.coins
            
            self.trades.append({
                'step': self.current_step,
                'action': 'SELL',
                'price': exec_price,
                'amount': self.coins,
                'fee': fee_paid,
                'portfolio_value': self.portfolio_value
            })
            
            self.coins = 0.0
            self.position = 0.0
            self.trade_count += 1
            self.last_trade_step = self.current_step

    def render(self):
        """Render current state."""
        price = self._get_price(self.current_step)
        pos_str = "HODLING 🟢" if self.position > 0.5 else "CASH 🔴"
        print(f"Step {self.current_step:4d} | ${price:,.0f} | {pos_str} | Portfolio: ${self.portfolio_value:,.2f}")

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
