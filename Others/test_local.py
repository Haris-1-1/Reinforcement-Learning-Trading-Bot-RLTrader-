"""
Local Test with Sample Data - VERSION 2
Tests the fixed code with generated data.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.indicators import TechnicalIndicators
from env.advanced_trading_env import AdvancedTradingEnv
from agents.q_learning_agent import QLearningAgent


def generate_btc_data(n_days: int = 800) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate realistic BTC price data.
    
    Returns:
        Tuple of (DataFrame with indicators, original_prices array)
    """
    np.random.seed(42)
    
    # Realistic BTC parameters
    start_price = 10000.0
    volatility = 0.035  # 3.5% daily
    drift = 0.002  # Slight upward trend
    
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Geometric Brownian Motion with momentum
    returns = np.random.normal(drift, volatility, n_days)
    
    # Add some momentum/trending behavior
    for i in range(1, len(returns)):
        returns[i] += returns[i-1] * 0.1  # 10% autocorrelation
    
    prices = start_price * np.cumprod(1 + returns)
    
    # Generate OHLCV
    data = {
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': prices * (1 + np.random.uniform(0.01, 0.04, n_days)),
        'Low': prices * (1 - np.random.uniform(0.01, 0.04, n_days)),
        'Close': prices,
        'Volume': np.random.uniform(1e9, 5e9, n_days)
    }
    
    df = pd.DataFrame(data)
    df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
    df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
    
    print(f"Generated {n_days} days of data")
    print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    print(f"Total price change: {((prices[-1]/prices[0])-1)*100:.1f}%")
    
    return df, prices


def test_with_sample_data():
    """Test the full pipeline."""
    
    print("\n" + "="*70)
    print("🧪 RL TRADING BOT - LOCAL TEST v2")
    print("="*70 + "\n")

    # ════════════════════════════════════════════════════════════════
    # GENERATE DATA
    # ════════════════════════════════════════════════════════════════
    print("📊 GENERATING SAMPLE DATA")
    print("-"*40)
    
    raw_df, original_prices = generate_btc_data(n_days=800)
    
    # Add indicators
    df = TechnicalIndicators.add_all_indicators(raw_df)
    
    # Speichere originale Preise nach dropna
    # Die Indikatoren droppen die ersten ~50 Zeilen
    start_idx = len(original_prices) - len(df)
    original_prices = original_prices[start_idx:]
    
    print(f"After indicators: {len(df)} rows")
    print(f"Prices aligned: {len(original_prices)} prices")
    
    # Normalize features (aber behalte originale Preise!)
    df_normalized = TechnicalIndicators.normalize_features(df.copy())
    
    # Train/Test split
    split_idx = int(len(df) * 0.8)
    train_df = df_normalized.iloc[:split_idx].reset_index(drop=True)
    test_df = df_normalized.iloc[split_idx:].reset_index(drop=True)
    train_prices = original_prices[:split_idx]
    test_prices = original_prices[split_idx:]
    
    print(f"\nTrain: {len(train_df)} rows, prices ${train_prices.min():.0f}-${train_prices.max():.0f}")
    print(f"Test: {len(test_df)} rows, prices ${test_prices.min():.0f}-${test_prices.max():.0f}")

    # ════════════════════════════════════════════════════════════════
    # CREATE ENVIRONMENT
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🏗️ CREATING ENVIRONMENT")
    print("-"*40)
    
    config = {
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.00005
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99,
            'n_bins': 15
        }
    }
    
    env = AdvancedTradingEnv(
        df=train_df,
        original_prices=train_prices,
        **config['environment']
    )

    # ════════════════════════════════════════════════════════════════
    # TEST RANDOM ACTIONS
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🎲 TESTING RANDOM ACTIONS")
    print("-"*40)
    
    obs, info = env.reset()
    print(f"Initial portfolio: ${info['portfolio_value']:.2f}")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        action_name = ['Hold', 'Buy', 'Sell'][action]
        print(f"  Step {i+1}: {action_name:4s} | Reward: {reward:+.4f} | Portfolio: ${info['portfolio_value']:,.2f}")
        if terminated:
            break
    
    print("\n✅ Environment works!")

    # ════════════════════════════════════════════════════════════════
    # TRAIN AGENT
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🎯 TRAINING Q-LEARNING AGENT")
    print("-"*40)
    
    # Fresh environment
    env = AdvancedTradingEnv(
        df=train_df,
        original_prices=train_prices,
        **config['environment']
    )
    
    agent = QLearningAgent(env, config)
    
    metrics = agent.train(
        total_timesteps=50000,  # Kürzer für schnellen Test
        log_interval=10000
    )

    # ════════════════════════════════════════════════════════════════
    # EVALUATE
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("📊 EVALUATION")
    print("-"*40)
    
    # Train evaluation
    print("\n🏋️ Train Performance:")
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    train_return = (info['portfolio_value'] - 10000) / 10000
    print(f"   Portfolio: ${info['portfolio_value']:,.2f}")
    print(f"   Return: {train_return*100:+.2f}%")
    print(f"   Trades: {info['trade_count']}")
    
    # Test evaluation
    print("\n🧪 Test Performance:")
    test_env = AdvancedTradingEnv(
        df=test_df,
        original_prices=test_prices,
        **config['environment']
    )
    
    obs, info = test_env.reset()
    done = False
    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
    
    test_return = (info['portfolio_value'] - 10000) / 10000
    print(f"   Portfolio: ${info['portfolio_value']:,.2f}")
    print(f"   Return: {test_return*100:+.2f}%")
    print(f"   Trades: {info['trade_count']}")

    # Buy & Hold
    train_bh = (train_prices[-1] / train_prices[0] - 1) * 100
    test_bh = (test_prices[-1] / test_prices[0] - 1) * 100
    
    print(f"\n📈 Buy & Hold:")
    print(f"   Train: {train_bh:+.2f}%")
    print(f"   Test: {test_bh:+.2f}%")

    # ════════════════════════════════════════════════════════════════
    # SAVE
    # ════════════════════════════════════════════════════════════════
    os.makedirs('results', exist_ok=True)
    agent.save('results/test_model_v2.pkl')

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print(f"""
Final Epsilon: {metrics['final_epsilon']:.4f}
Episodes: {metrics['total_episodes']}

Train Return: {train_return*100:+.2f}%  (Buy&Hold: {train_bh:+.2f}%)
Test Return:  {test_return*100:+.2f}%  (Buy&Hold: {test_bh:+.2f}%)
""")

    return agent


if __name__ == '__main__':
    agent = test_with_sample_data()
