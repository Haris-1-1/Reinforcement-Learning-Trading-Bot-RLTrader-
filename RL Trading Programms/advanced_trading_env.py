"""
Advanced Trading Environment for Reinforcement Learning.
Features extended state space, technical indicators, and realistic trading constraints.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any


class AdvancedTradingEnv(gym.Env):
    """
    Advanced trading environment with realistic constraints and extended state space.

    State space includes:
    - Price features (OHLCV)
    - Technical indicators (MA, RSI, MACD, Bollinger Bands, ATR)
    - Portfolio state (cash, position, value)

    Action space:
    - 0: Hold
    - 1: Buy
    - 2: Sell

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
        feature_columns: Optional[list] = None
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
            feature_columns: List of feature column names
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash

        # Trading constraints
        self.trading_fee_maker = trading_fee_maker
        self.trading_fee_taker = trading_fee_taker
        self.slippage = slippage
        self.trade_frequency_penalty = trade_frequency_penalty
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.enable_execution_delay = enable_execution_delay
        self.execution_delay_steps = execution_delay_steps

        # Feature columns (OHLCV + indicators)
        if feature_columns is None:
            self.feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA5', 'MA20', 'MA50', 'RSI',
                'MACD', 'MACD_signal', 'MACD_hist',
                'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
                'ATR', 'Volume_MA', 'Volume_ratio',
                'Returns', 'Log_returns'
            ]
        else:
            self.feature_columns = feature_columns

        n_features = len(self.feature_columns) + 3  # +3 for cash, position, portfolio_value

        # State space: market features + portfolio state
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
        self.trades = []
        self.trade_count = 0
        self.last_trade_step = 0

        # Performance tracking
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state).

        Returns:
            State vector with market features and portfolio state
        """
        # Market features
        market_features = self.df.loc[self.current_step, self.feature_columns].values.astype(np.float32)

        # Portfolio features (normalized) - ensure all are scalars
        # Convert to float to handle both scalar and Series values
        cash_val = float(self.cash.item() if hasattr(self.cash, 'item') else self.cash)
        pos_val = float(self.position.item() if hasattr(self.position, 'item') else self.position)
        pv_val = float(self.portfolio_value.item() if hasattr(self.portfolio_value, 'item') else self.portfolio_value)

        portfolio_features = np.array([
            cash_val / self.initial_cash,  # Normalized cash
            pos_val,  # Position (0 or 1)
            pv_val / self.initial_cash  # Normalized portfolio value
        ], dtype=np.float32)

        # Combine all features
        obs = np.concatenate([market_features, portfolio_features])

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional info about current state.

        Returns:
            Dictionary with portfolio metrics
        """
        current_price = self.df.loc[self.current_step, 'Close']

        # Ensure all values are scalars
        cash_val = float(self.cash.item() if hasattr(self.cash, 'item') else self.cash)
        coins_val = float(self.coins.item() if hasattr(self.coins, 'item') else self.coins)
        pos_val = float(self.position.item() if hasattr(self.position, 'item') else self.position)
        pv_val = float(self.portfolio_value.item() if hasattr(self.portfolio_value, 'item') else self.portfolio_value)
        price_val = float(current_price.item() if hasattr(current_price, 'item') else current_price)
        fees_val = float(self.total_fees_paid.item() if hasattr(self.total_fees_paid, 'item') else self.total_fees_paid)
        slip_val = float(self.total_slippage_cost.item() if hasattr(self.total_slippage_cost, 'item') else self.total_slippage_cost)

        return {
            'step': self.current_step,
            'cash': cash_val,
            'coins': coins_val,
            'position': pos_val,
            'portfolio_value': pv_val,
            'current_price': price_val,
            'trade_count': self.trade_count,
            'total_fees_paid': fees_val,
            'total_slippage_cost': slip_val
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_cash
        self.coins = 0.0
        self.position = 0.0
        self.portfolio_value = self.initial_cash
        self.last_portfolio_value = self.initial_cash

        self.trades = []
        self.trade_count = 0
        self.last_trade_step = 0
        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def _execute_trade(self, action: int, current_price: float) -> float:
        """
        Execute a trade with fees and slippage.

        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            current_price: Current market price

        Returns:
            Transaction cost (fees + slippage)
        """
        transaction_cost = 0.0

        # Apply slippage (price impact)
        if action == 1:  # Buy
            execution_price = current_price * (1 + self.slippage)
        elif action == 2:  # Sell
            execution_price = current_price * (1 - self.slippage)
        else:
            return 0.0

        # Execute trade
        if action == 1 and self.position == 0:  # Buy
            # Use taker fee (market order)
            fee = self.trading_fee_taker

            # Calculate number of coins to buy
            total_cost = self.cash / (1 + fee)
            self.coins = total_cost / execution_price
            fee_paid = self.cash - total_cost
            self.cash = 0.0
            self.position = 1.0

            transaction_cost = fee_paid + (execution_price - current_price) * self.coins
            self.total_fees_paid += fee_paid
            self.total_slippage_cost += (execution_price - current_price) * self.coins

            # Record trade
            self.trades.append({
                'step': self.current_step,
                'action': 'BUY',
                'price': execution_price,
                'amount': self.coins,
                'fee': fee_paid,
                'portfolio_value': self.portfolio_value
            })
            self.trade_count += 1
            self.last_trade_step = self.current_step

        elif action == 2 and self.position == 1:  # Sell
            # Use taker fee (market order)
            fee = self.trading_fee_taker

            # Sell all coins
            gross_proceeds = self.coins * execution_price
            fee_paid = gross_proceeds * fee
            self.cash = gross_proceeds - fee_paid
            self.coins = 0.0
            self.position = 0.0

            transaction_cost = fee_paid + (current_price - execution_price) * self.coins
            self.total_fees_paid += fee_paid
            self.total_slippage_cost += abs(current_price - execution_price) * self.coins

            # Record trade
            self.trades.append({
                'step': self.current_step,
                'action': 'SELL',
                'price': execution_price,
                'amount': self.coins,
                'fee': fee_paid,
                'portfolio_value': self.portfolio_value
            })
            self.trade_count += 1
            self.last_trade_step = self.current_step

        return transaction_cost

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get current price (ensure scalar)
        current_price = self.df.loc[self.current_step, 'Close']
        current_price = float(current_price.item() if hasattr(current_price, 'item') else current_price)

        # Store old portfolio value
        self.last_portfolio_value = self.portfolio_value

        # Execute trade
        transaction_cost = self._execute_trade(action, current_price)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False

        if not terminated:
            # Get new price (ensure scalar)
            new_price = self.df.loc[self.current_step, 'Close']
            new_price = float(new_price.item() if hasattr(new_price, 'item') else new_price)

            # Update portfolio value (ensure scalar)
            cash_val = float(self.cash.item() if hasattr(self.cash, 'item') else self.cash)
            coins_val = float(self.coins.item() if hasattr(self.coins, 'item') else self.coins)
            self.portfolio_value = cash_val + coins_val * new_price

            # Calculate reward (realized profit) - ensure scalar
            last_pv = float(self.last_portfolio_value.item() if hasattr(self.last_portfolio_value, 'item') else self.last_portfolio_value)
            curr_pv = float(self.portfolio_value)
            reward = float((curr_pv - last_pv) / last_pv)

            # Apply trade frequency penalty
            if action != 0:  # If not holding
                steps_since_last_trade = self.current_step - self.last_trade_step
                if steps_since_last_trade < 5:  # Penalize trading too frequently
                    reward -= self.trade_frequency_penalty

        else:
            # Final step
            final_price = self.df.loc[self.current_step - 1, 'Close']
            final_price = float(final_price.item() if hasattr(final_price, 'item') else final_price)
            cash_val = float(self.cash.item() if hasattr(self.cash, 'item') else self.cash)
            coins_val = float(self.coins.item() if hasattr(self.coins, 'item') else self.coins)
            self.portfolio_value = cash_val + coins_val * final_price

            last_pv = float(self.last_portfolio_value.item() if hasattr(self.last_portfolio_value, 'item') else self.last_portfolio_value)
            curr_pv = float(self.portfolio_value)
            reward = float((curr_pv - last_pv) / last_pv)

        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment (print current state)."""
        current_price = self.df.loc[self.current_step, 'Close']
        print(f"Step: {self.current_step}, "
              f"Price: {current_price:.2f}, "
              f"Position: {self.position}, "
              f"Portfolio Value: {self.portfolio_value:.2f}, "
              f"Trades: {self.trade_count}")

    def get_trades_df(self) -> pd.DataFrame:
        """
        Get trades as DataFrame.

        Returns:
            DataFrame with trade history
        """
        return pd.DataFrame(self.trades)
