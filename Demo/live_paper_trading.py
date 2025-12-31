import time
import os
import numpy as np
import yfinance as yf
from mini_trading_env import MiniTradingEnv

BASE_DIR = os.path.dirname(__file__)
q_path = os.path.join(BASE_DIR, "q_table.npy")

q_table = np.load(q_path)

n_return_bins = 20
return_min, return_max = -0.05, 0.05

def discretize_state(obs):
    ret, pos = obs
    ret_clipped = np.clip(ret, return_min, return_max)
    bin_idx = int((ret_clipped - return_min) / (return_max - return_min) * (n_return_bins - 1))
    pos_idx = int(pos)
    return bin_idx, pos_idx

hist = yf.download("BTC-USD", period="7d", interval="1m")
prices = hist["Close"].dropna().values.astype(float)

env = MiniTradingEnv(prices)
obs, info = env.reset()

print(f"Start Portfolio: {info['portfolio_value']} $")

while True:
    state_idx = discretize_state(obs)

    action = int(np.argmax(q_table[state_idx]))

    live = yf.download("BTC-USD", period="2d", interval="1m")
    if live.empty:
        print("Keine Live-Daten bekommen, überspringe diesen Tick.")
        time.sleep(60)
        continue

    new_price = float(live["Close"].iloc[-1])

    env.prices = np.append(env.prices, new_price)

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"Preis: {new_price:.2f} $, "
        f"Aktion: {action}, "
        f"Portfolio: {info['portfolio_value']:.2f} $"
    )

    time.sleep(60)
