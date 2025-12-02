import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MiniTradingEnv(gym.Env):
    def __init__(self, prices: np.ndarray, initial_cash: float = 1000.0):
        super().__init__()
        self.prices = prices          # z.B. Array von Bitcoin-Preisen
        self.initial_cash = initial_cash

        # State: [return_t, position]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0.0]),
            high=np.array([np.inf, 1.0]),
            dtype=np.float32
        )

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        self.reset()

    def _get_obs(self):
        return np.array([float(self.return_t), float(self.position)], dtype=np.float32)

    def _get_info(self):
        return {"portfolio_value": self.portfolio_value}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 1                      # Start bei Index 1, damit t-1 existiert
        self.cash = self.initial_cash
        self.position = 0.0             # 0 = keine Coins, 1 = voll investiert
        self.coins = 0.0
        self.portfolio_value = self.cash
        self.return_t = 0.0             # Kein Return beim ersten Schritt
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        price_prev = self.prices[self.t - 1]
        price = self.prices[self.t]

        # Preis-Return
        self.return_t = (price - price_prev) / price_prev

        # Aktion ausführen
        if action == 1 and self.position == 0:  # Buy
            self.coins = self.cash / price
            self.cash = 0.0
            self.position = 1.0
        elif action == 2 and self.position == 1:  # Sell
            self.cash = self.coins * price
            self.coins = 0.0
            self.position = 0.0

        # Portfolio-Wert berechnen
        old_value = self.portfolio_value
        self.portfolio_value = self.cash + self.coins * price

        # Reward = prozentuale Änderung (Rendite dieses Schritts)
        reward = (self.portfolio_value - old_value) / old_value if old_value != 0 else 0.0

        # Zeitschritt weiter
        self.t += 1
        terminated = self.t >= len(self.prices) - 1
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

