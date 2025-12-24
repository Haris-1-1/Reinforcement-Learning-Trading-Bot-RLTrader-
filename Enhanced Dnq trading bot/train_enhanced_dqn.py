import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# --- PFAD-FIX START ---
# Ermittelt das Verzeichnis, in dem dieses Skript liegt
current_dir = os.path.dirname(os.path.abspath(__file__))
# Fügt dieses Verzeichnis zum Python-Pfad hinzu, damit 'env', 'utils' etc. gefunden werden
if current_dir not in sys.path:
    sys.path.append(current_dir)
# --- PFAD-FIX ENDE ---

# Jetzt funktionieren die Imports korrekt
from env.advanced_trading_env import AdvancedTradingEnv
from utils.data_loader import DataLoader
from agents.dqn_agent import DQNAgent
from utils.indicators import TechnicalIndicators

def run_benchmark_strategies(prices, initial_cash):
    """Berechnet die Benchmarks: Buy & Hold, Random und MA Crossover."""
    # 1. Buy & Hold
    bh_return = (prices[-1] / prices[0]) - 1
    bh_final = initial_cash * (1 + bh_return)

    # 2. Random Trading (Durchschnitt aus 5 Durchläufen)
    random_returns = []
    for _ in range(5):
        cash = initial_cash
        coins = 0
        for i in range(len(prices)-1):
            action = np.random.choice([0, 1, 2]) # Hold, Buy, Sell
            if action == 1 and cash > 0: # Buy
                coins = cash / prices[i]
                cash = 0
            elif action == 2 and coins > 0: # Sell
                cash = coins * prices[i]
                coins = 0
        final = cash + (coins * prices[-1])
        random_returns.append((final - initial_cash) / initial_cash)
    random_avg_return = np.mean(random_returns)

    # 3. Moving Average Crossover (SMA 20 / SMA 50)
    df_ma = pd.DataFrame({'Close': prices})
    df_ma['SMA20'] = df_ma['Close'].rolling(window=20).mean()
    df_ma['SMA50'] = df_ma['Close'].rolling(window=50).mean()
    
    cash, coins = initial_cash, 0
    for i in range(50, len(prices)-1):
        if df_ma['SMA20'].iloc[i] > df_ma['SMA50'].iloc[i] and cash > 0: # Buy
            coins = cash / prices[i]
            cash = 0
        elif df_ma['SMA20'].iloc[i] < df_ma['SMA50'].iloc[i] and coins > 0: # Sell
            cash = coins * prices[i]
            coins = 0
    ma_final = cash + (coins * prices[-1])
    ma_return = (ma_final - initial_cash) / initial_cash

    return bh_return, random_avg_return, ma_return

def train_enhanced_bot():
    # 1. KONFIGURATION (Basierend auf Exp 11)
    config = {
        "data": {
            "symbol": "BTC-USD",
            "start_date": "2020-01-01",
            "end_date": "2025-12-15",
            "interval": "1d",
            "window_size": 10,
            "test_split": 0.15
        },
        "environment": {
            "initial_cash": 10000.0,
            "trading_fee_maker": 0.0,
            "trading_fee_taker": 0.002,
            "slippage": 0.001,
            "trade_frequency_penalty": 0.002
        },
        "dqn": {
            "learning_rate": 0.0001,
            "hidden_sizes": [128, 128, 64],
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay_steps": 50000,
            "replay_buffer_size": 10000,
            "batch_size": 64,
            "target_update_freq": 1000
        },
        "training": {
            "total_timesteps": 100000,
            "log_interval": 10000
        }
    }

    # 2. DATEN LADEN
    loader = DataLoader(
        symbol=config['data']['symbol'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        interval=config['data']['interval'],
        test_split=config['data']['test_split']
    )
    train_df, test_df = loader.prepare_data()

    # 3. ENVIRONMENT & AGENT INITIALISIEREN
    env = AdvancedTradingEnv(
        df=train_df,
        original_prices=loader.original_prices_train,
        initial_cash=config['environment']['initial_cash'],
        trading_fee_taker=config['environment']['trading_fee_taker'],
        slippage=config['environment']['slippage'],
        trade_frequency_penalty=config['environment']['trade_frequency_penalty'],
        window_size=config['data']['window_size'],
        feature_columns=TechnicalIndicators.get_feature_columns()
    )

    agent = DQNAgent(env, config)

    # 4. TRAINING LOOP
    print("\nStarting Enhanced DQN Training...")
    timestep = 0
    while timestep < config['training']['total_timesteps']:
        obs, info = env.reset()
        done = False
        while not done:
            # Predict mit Action Masking
            action = agent.predict(obs, deterministic=False, action_mask=info['action_mask'])
            
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Store transition mit Action Mask für den nächsten Schritt
            agent.store_transition(obs, action, reward, next_obs, done, next_info['action_mask'])
            
            # Training Step
            loss = agent.train_step()
            
            # Update Target Network
            if agent.steps_done % config['dqn']['target_update_freq'] == 0:
                agent.update_target_network()
                
            obs = next_obs
            info = next_info
            timestep += 1
            agent.steps_done += 1

            if timestep % config['training']['log_interval'] == 0:
                print(f"Step {timestep}/{config['training']['total_timesteps']} | "
                      f"Epsilon: {agent.epsilon:.2f} | Portfolio: ${info['portfolio_value']:.2f}")

    # 5. EVALUATION & BENCHMARKS
    print("\n--- FINAL EVALUATION (TEST DATA) ---")
    test_env = AdvancedTradingEnv(
        df=test_df,
        original_prices=loader.original_prices_test,
        initial_cash=config['environment']['initial_cash'],
        window_size=config['data']['window_size'],
        feature_columns=TechnicalIndicators.get_feature_columns()
    )
    
    obs, info = test_env.reset()
    done = False
    while not done:
        action = agent.predict(obs, deterministic=True, action_mask=info['action_mask'])
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

    agent_return = (info['portfolio_value'] - config['environment']['initial_cash']) / config['environment']['initial_cash']
    
    # Benchmarks berechnen
    bh_ret, rand_ret, ma_ret = run_benchmark_strategies(loader.original_prices_test, config['environment']['initial_cash'])

    print(f"\nRESULTS ON TEST DATA:")
    print(f"{'Strategy':<25} | {'Return':<10}")
    print("-" * 40)
    print(f"{'ENHANCED DQN (Bot)':<25} | {agent_return*100:>+8.2f}%")
    print(f"{'Buy & Hold':<25} | {bh_ret*100:>+8.2f}%")
    print(f"{'Random Trading':<25} | {rand_ret*100:>+8.2f}%")
    print(f"{'MA Crossover':<25} | {ma_ret*100:>+8.2f}%")
    
    # Speichern
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    agent.update_target_network() # Finaler Sync
    # Hier müsste eine Speicherfunktion für das PyTorch Modell folgen (save_state_dict)

if __name__ == "__main__":
    train_enhanced_bot()