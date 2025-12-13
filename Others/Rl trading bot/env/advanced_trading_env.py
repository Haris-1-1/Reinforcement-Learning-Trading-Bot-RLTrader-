"""
Advanced Trading Environment for Reinforcement Learning.
Features extended state space, technical indicators, and realistic trading constraints.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List


class AdvancedTradingEnv(gym.Env):
    """
    Advanced trading environment with realistic constraints and extended state space.

    State space includes:
    - Price features (OHLCV - normalized)
    - Technical indicators (MA, RSI, MACD, Bollinger Bands, ATR - normalized)
    - Portfolio state (cash ratio, position, portfolio value ratio)

    Action space:
    - 0: Hold
    - 1: Buy (go from cash to crypto)
    - 2: Sell (go from crypto to cash)

    Constraints:
    - Trading fees (maker/taker)
    - Slippage
    - Trade frequency penalty
    - Position size limits
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_cash: float = 10000.0,
        trading_fee_maker: float = 0.001,
        trading_fee_taker: float = 0.002,
        slippage: float = 0.001,
        trade_frequency_penalty: float = 0.0001,
        max_position_size: float = 1.0,
        min_position_size: float = 0.0,
        enable_execution_delay: bool = False,
        execution_delay_steps: int = 0,
        feature_columns: Optional[List[str]] = None,
        price_column: str = 'Close'
    ):
        """
        Initialize Advanced Trading Environment.

        Args:
            df: DataFrame with OHLCV data and technical indicators
            initial_cash: Starting cash amount
            trading_fee_maker: Maker fee (limit orders)
            trading_fee_taker: Taker fee (market orders)
            slippage: Price slippage percentage
            trade_frequency_penalty: Penalty for excessive trading
            max_position_size: Maximum position size as fraction of capital
            min_position_size: Minimum position size
            enable_execution_delay: Whether to simulate execution delay
            execution_delay_steps: Number of steps to delay execution
            feature_columns: List of feature column names for state
            price_column: Column to use for trading prices
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_cash = float(initial_cash)

        # Trading constraints
        self.trading_fee_maker = trading_fee_maker
        self.trading_fee_taker = trading_fee_taker
        self.slippage = slippage
        self.trade_frequency_penalty = trade_frequency_penalty
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.enable_execution_delay = enable_execution_delay
        self.execution_delay_steps = execution_delay_steps

        # Determine which columns to use as features
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
        
        # Extract ORIGINAL prices for trading (before any normalization)
        # We need to reconstruct original prices if they're normalized
        raw_prices = self.df[price_column].values.astype(float)
        
        # Check if prices look normalized (z-score normalized data has mean ~0, std ~1)
        price_mean = np.mean(raw_prices)
        price_std = np.std(raw_prices)
        
        if abs(price_mean) < 10 and price_std < 10:
            # Prices appear normalized - we need to generate synthetic prices for trading
            # Use a realistic BTC-like price range
            print("Note: Using synthetic price series for trading (features appear normalized)")
            base_price = 30000.0  # Base price similar to BTC
            # Generate price series with similar pattern to normalized data
            self.prices = base_price * (1 + 0.02 * raw_prices)  # Scale normalized to ~2% moves
        else:
            # Prices are in original form
            self.prices = raw_prices
        
        print(f"Price range: ${self.prices.min():.2f} - ${self.prices.max():.2f}")
        print(f"Using {len(self.feature_columns)} features for state")

        n_features = len(self.feature_columns) + 3  # +3 for cash_ratio, position, portfolio_ratio

        # State space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Episode tracking
        self.current_step = 0
        self.max_steps = len(self.df) - 1

        # Portfolio state
        self.cash = 0.0
        self.coins = 0.0
        self.position = 0.0  # 0 = no position, 1 = full position
        self.portfolio_value = 0.0
        self.last_portfolio_value = 0.0

        # Trading history
        self.trades: List[Dict] = []
        self.trade_count = 0
        self.last_trade_step = -10

        # Performance tracking
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

    def _get_price(self, step: int) -> float:
        """Get the trading price at a given step."""
        return float(self.prices[step])

    def _get_observation(self) -> np.ndarray:
        """Get current observation (state)."""
        # Market features from dataframe
        row = self.df.iloc[self.current_step]
        market_features = []
        for col in self.feature_columns:
            val = row[col]
            if hasattr(val, 'item'):
                val = val.item()
            market_features.append(float(val))
        
        market_features = np.array(market_features, dtype=np.float32)

        # Portfolio features (normalized)
        portfolio_features = np.array([
            self.cash / self.initial_cash,  # Cash ratio
            self.position,  # Position (0 or 1)
            self.portfolio_value / self.initial_cash  # Portfolio value ratio
        ], dtype=np.float32)

        # Combine
        obs = np.concatenate([market_features, portfolio_features])
        
        # Clean up any NaN or Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
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
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_cash
        self.coins = 0.0
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        self.last_portfolio_value = self.initial_cash

        self.trades = []
        self.trade_count = 0
        self.last_trade_step = -10
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        current_price = self._get_price(self.current_step)
        
        # Store old portfolio value
        self.last_portfolio_value = self.portfolio_value

        # Execute trade
        self._execute_trade(action, current_price)

        # Move to next step
        self.current_step += 1

        # Check if done
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Calculate new portfolio value
        if not terminated:
            new_price = self._get_price(self.current_step)
        else:
            new_price = self._get_price(self.current_step - 1)
        
        self.portfolio_value = self.cash + self.coins * new_price

        # Calculate reward (percentage change in portfolio value)
        if self.last_portfolio_value > 0:
            reward = (self.portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        else:
            reward = 0.0

        # Penalty for trading too frequently
        if action != 0 and self.trade_count > 0:
            steps_since_last = self.current_step - self.last_trade_step
            if 0 < steps_since_last < 5:
                reward -= self.trade_frequency_penalty

        # Clip reward
        reward = float(np.clip(reward, -1.0, 1.0))

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _execute_trade(self, action: int, current_price: float):
        """Execute a trade with fees and slippage."""
        
        # BUY: Only if we have cash and no position
        if action == 1 and self.position < 0.5 and self.cash > 0:
            # Apply slippage (we pay more when buying)
            exec_price = current_price * (1 + self.slippage)
            
            # Calculate fees
            fee_rate = self.trading_fee_taker
            available = self.cash / (1 + fee_rate)
            
            # Buy coins
            self.coins = available / exec_price
            fee_paid = self.cash - available
            
            self.cash = 0.0
            self.position = 1.0
            
            # Track costs
            self.total_fees_paid += fee_paid
            self.total_slippage_cost += (exec_price - current_price) * self.coins
            
            # Record trade
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
            
        # SELL: Only if we have coins
        elif action == 2 and self.position > 0.5 and self.coins > 0:
            # Apply slippage (we get less when selling)
            exec_price = current_price * (1 - self.slippage)
            
            # Calculate proceeds and fees
            gross = self.coins * exec_price
            fee_rate = self.trading_fee_taker
            fee_paid = gross * fee_rate
            
            self.cash = gross - fee_paid
            
            # Track costs
            self.total_fees_paid += fee_paid
            self.total_slippage_cost += (current_price - exec_price) * self.coins
            
            # Record trade
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
        pos_str = "IN POSITION" if self.position > 0.5 else "CASH"
        print(f"Step {self.current_step:4d} | Price: ${price:,.2f} | "
              f"{pos_str:11s} | Portfolio: ${self.portfolio_value:,.2f} | "
              f"Trades: {self.trade_count}")

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
