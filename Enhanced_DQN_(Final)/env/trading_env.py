import numpy as np
import pandas as pd
class TradingEnvironment:
    def __init__(
        self,
        df: pd.DataFrame,
        original_prices: np.ndarray,
        window_size: int = 24,
        initial_cash: float = 10000.0,
        fee: float = 0.001,
        slippage: float = 0.0005
    ):
        self.df = df.reset_index(drop=True)
        self.prices = original_prices.astype(float)
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.fee = fee
        self.slippage = slippage
        if 'Date' in self.df.columns:
            self.df = self.df.drop('Date', axis=1)
        if len(self.df) != len(self.prices):
            raise ValueError(f"DataFrame length ({len(self.df)}) != prices length ({len(self.prices)})")
        excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        potential_features = [col for col in self.df.columns if col not in excluded_cols]
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
        print(f"  Total observation size: {self.window_size * self.n_features + 3}")
        print(f"  Steps available: {len(self.df)}")
        print(f"  Initial Cash: ${self.initial_cash:.2f}")
        self.current_step = 0
        self.cash = 0.0
        self.coins = 0.0
        self.position = 0.0
        self.entry_price = 0.0
        self.portfolio_value = 0.0
        self.last_portfolio_value = 0.0
        self.trades = []
        self.trade_count = 0
    def _get_price(self, step: int) -> float:
        return float(self.prices[step])
    def get_action_mask(self) -> np.ndarray:
        mask = np.ones(3, dtype=np.int8)
        if self.position >= 0.99:
            mask[1] = 0
        if self.position <= 0.01:
            mask[2] = 0
        return mask
    def _get_observation(self) -> np.ndarray:
        if self.current_step < self.window_size:
            padding_needed = self.window_size - (self.current_step + 1)
            real_data = self.df.iloc[0:self.current_step + 1][self.feature_cols].values
            padding = np.tile(real_data[0], (padding_needed, 1))
            window_data = np.vstack([padding, real_data])
        else:
            start_idx = self.current_step - self.window_size + 1
            end_idx = self.current_step + 1
            window_data = self.df.iloc[start_idx:end_idx][self.feature_cols].values
        try:
            flat_window = window_data.astype(np.float32).flatten()
        except (ValueError, TypeError) as e:
            print(f"Warning: Data conversion issue at step {self.current_step}: {e}")
            flat_list = []
            for row in window_data:
                for val in row:
                    try:
                        flat_list.append(float(val))
                    except:
                        flat_list.append(0.0)
            flat_window = np.array(flat_list, dtype=np.float32)
        current_price = self._get_price(self.current_step)
        if self.position > 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = 0.0
        portfolio_features = np.array([
            self.cash / self.initial_cash,
            self.position,
            unrealized_pnl
        ], dtype=np.float32)
        observation = np.concatenate([flat_window, portfolio_features])
        return observation
    def _get_info(self) -> dict[str, any]:
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
        current_price = self._get_price(self.current_step)
        self.last_portfolio_value = self.portfolio_value
        mask = self.get_action_mask()
        if mask[action] == 0:
            action = 0
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
        new_price = self._get_price(self.current_step)
        self.portfolio_value = self.cash + (self.coins * new_price)
        if self.last_portfolio_value > 0:
            reward = np.log(self.portfolio_value / self.last_portfolio_value)
        else:
            reward = 0.0
        if action != 0:
            reward -= 0.0001
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, terminated, truncated, info
    def get_trades_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)
    def get_portfolio_history(self) -> dict:
        return {
            'final_value': self.portfolio_value,
            'total_return': (self.portfolio_value - self.initial_cash) / self.initial_cash,
            'total_trades': self.trade_count,
            'trades': self.trades
        }
if __name__ == "__main__":
    print("Trading Environment Module")
    print("Features:")
    print("  - Window-based observations")
    print("  - Action masking")
    print("  - Portfolio tracking")
    print("  - Trading fees & slippage")