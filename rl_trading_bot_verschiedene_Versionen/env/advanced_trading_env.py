import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List


class AdvancedTradingEnv(gym.Env):
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
        max_position_size: float = 1.0,
        feature_columns: Optional[List[str]] = None
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_cash = float(initial_cash)

        self.prices = original_prices.astype(float)

        if len(self.prices) != len(self.df):
            raise ValueError(f"Prices length ({len(self.prices)}) must match df length ({len(self.df)})")

        print(f" Using real prices for trading: ${self.prices.min():.2f} - ${self.prices.max():.2f}")
        print(f"   Price change over period: {((self.prices[-1]/self.prices[0])-1)*100:.1f}%")

        self.trading_fee_maker = trading_fee_maker
        self.trading_fee_taker = trading_fee_taker
        self.slippage = slippage
        self.trade_frequency_penalty = trade_frequency_penalty
        self.max_position_size = max_position_size

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

        n_features = len(self.feature_columns) + 3

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.current_step = 0
        self.max_steps = len(self.df) - 1

        self.cash = 0.0
        self.coins = 0.0
        self.position = 0.0
        self.portfolio_value = 0.0
        self.last_portfolio_value = 0.0

        self.trades: List[Dict] = []
        self.trade_count = 0
        self.last_trade_step = -100

        self.total_fees_paid = 0.0
        self.total_slippage_cost = 0.0

    def _get_price(self, step: int) -> float:
        return float(self.prices[step])

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        market_features = []
        for col in self.feature_columns:
            val = row[col]
            if hasattr(val, 'item'):
                val = val.item()
            market_features.append(float(val))

        market_features = np.array(market_features, dtype=np.float32)

        portfolio_features = np.array([
            self.cash / self.initial_cash,
            self.position,
            self.portfolio_value / self.initial_cash
        ], dtype=np.float32)

        obs = np.concatenate([market_features, portfolio_features])
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs

    def _get_info(self) -> Dict[str, Any]:
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
        current_price = self._get_price(self.current_step)
        self.last_portfolio_value = self.portfolio_value

        self._execute_trade(action, current_price)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        new_price = self._get_price(min(self.current_step, self.max_steps - 1))
        self.portfolio_value = self.cash + self.coins * new_price

        if self.last_portfolio_value > 0:
            reward = (self.portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        else:
            reward = 0.0

        if action != 0 and self.trade_count > 1:
            steps_since_last = self.current_step - self.last_trade_step
            if 0 < steps_since_last < 3:
                reward -= self.trade_frequency_penalty

        reward = float(np.clip(reward, -1.0, 1.0))

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _execute_trade(self, action: int, current_price: float):

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
        price = self._get_price(self.current_step)
        pos_str = "holding " if self.position > 0.5 else "not holding"
        print(f"Step {self.current_step:4d} | ${price:,.0f} | {pos_str} | Portfolio: ${self.portfolio_value:,.2f}")

    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
