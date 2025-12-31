import numpy as np
import pandas as pd
import yfinance as yf
from mini_trading_env import MiniTradingEnv

data = yf.download("BTC-USD", start="2020-01-01", end="2024-01-01", interval="1d")
prices = data["Close"].dropna().values.astype(float)

env = MiniTradingEnv(prices)
n_actions = env.action_space.n

n_return_bins = 20
return_min, return_max = -0.05, 0.05
q_table = np.zeros((n_return_bins, 2, n_actions))

def discretize_state(obs):
    ret, pos = obs
    ret_clipped = np.clip(ret, return_min, return_max)
    bin_idx = int((ret_clipped - return_min) / (return_max - return_min) * (n_return_bins - 1))
    pos_idx = int(pos)
    return bin_idx, pos_idx

alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
n_episodes = 500

for episode in range(n_episodes):
    obs, info = env.reset()
    state_idx = discretize_state(obs)
    done = False
    total_reward = 0.0

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_idx])

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_state_idx = discretize_state(next_obs)

        old_value = q_table[state_idx + (action,)]
        next_max = np.max(q_table[next_state_idx])

        q_table[state_idx + (action,)] = old_value + alpha * (
            reward + gamma * next_max - old_value
        )

        state_idx = next_state_idx
        done = terminated or truncated
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    final_value = float(info["portfolio_value"])
    episode_profit = float(final_value - env.initial_cash)
    episode_roi = float(episode_profit / env.initial_cash)

    print(
        f"Episode {episode}, Profit: {episode_profit:.2f} $, "
        f"ROI: {episode_roi:.2%}"
)

np.save("q_table.npy", q_table)

buy_hold_profit = (prices[-1] - prices[0]) * (env.initial_cash / prices[0])
buy_hold_roi = buy_hold_profit / env.initial_cash

buy_hold_profit = float(buy_hold_profit)
buy_hold_roi = float(buy_hold_roi)

print(
    f"Buy&Hold Profit: {buy_hold_profit:.2f} $, "
    f"ROI: {buy_hold_roi:.2%}"
)

