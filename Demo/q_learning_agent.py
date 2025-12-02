# q_learning_agent.py
import numpy as np
import pandas as pd
import yfinance as yf
from mini_trading_env import MiniTradingEnv

# BTC-Daten mit yfinance laden
data = yf.download("BTC-USD", start="2020-01-01", end="2024-01-01", interval="1d")
prices = data["Close"].dropna().values.astype(float)

# 2) Env
env = MiniTradingEnv(prices)
n_actions = env.action_space.n

# 3) Q-Tabelle
n_return_bins = 20
return_min, return_max = -0.05, 0.05
q_table = np.zeros((n_return_bins, 2, n_actions))

def discretize_state(obs):
    ret, pos = obs
    ret_clipped = np.clip(ret, return_min, return_max)
    bin_idx = int((ret_clipped - return_min) / (return_max - return_min) * (n_return_bins - 1))
    pos_idx = int(pos)
    return bin_idx, pos_idx

# 4) Hyperparameter
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
n_episodes = 500

# 5) Training-Loop
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
    print(f"Episode {episode}, total reward: {total_reward}")

# Optional: Q-Table speichern
np.save("q_table.npy", q_table)

