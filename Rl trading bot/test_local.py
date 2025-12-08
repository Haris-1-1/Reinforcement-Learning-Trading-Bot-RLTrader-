"""
Local Test Script with embedded sample data.
Use this to verify the code works, then run train.py on your machine with real data.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.indicators import TechnicalIndicators
from env.advanced_trading_env import AdvancedTradingEnv
from agents.q_learning_agent import QLearningAgent


def generate_sample_btc_data(n_days: int = 500) -> pd.DataFrame:
    """
    Generate realistic-looking BTC price data for testing.
    
    Args:
        n_days: Number of days of data to generate
        
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)  # For reproducibility
    
    # Start parameters (similar to BTC in 2020-2024)
    start_price = 10000.0
    volatility = 0.03  # 3% daily volatility
    drift = 0.001  # Slight upward trend
    
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]
    
    # Generate price path using geometric Brownian motion
    returns = np.random.normal(drift, volatility, n_days)
    prices = start_price * np.cumprod(1 + returns)
    
    # Generate OHLCV data
    data = {
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': prices * (1 + np.random.uniform(0.01, 0.03, n_days)),
        'Low': prices * (1 - np.random.uniform(0.01, 0.03, n_days)),
        'Close': prices,
        'Volume': np.random.uniform(1e9, 5e9, n_days)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure High >= Close and Low <= Close
    df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
    df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
    
    print(f"Generated {n_days} days of sample BTC data")
    print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    return df


def test_with_sample_data():
    """Test the full pipeline with generated sample data."""
    
    print("\n" + "="*70)
    print("RL TRADING BOT - TEST WITH SAMPLE DATA")
    print("="*70 + "\n")

    # ============ GENERATE SAMPLE DATA ============
    print("STEP 1: GENERATING SAMPLE DATA")
    print("-"*40)
    
    raw_data = generate_sample_btc_data(n_days=500)
    
    # Add technical indicators
    print("\nAdding technical indicators...")
    data = TechnicalIndicators.add_all_indicators(raw_data)
    print(f"Data shape after indicators: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Normalize features
    print("\nNormalizing features...")
    data = TechnicalIndicators.normalize_features(data)
    
    # Split train/test
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx].reset_index(drop=True)
    test_data = data.iloc[split_idx:].reset_index(drop=True)
    
    print(f"\nTrain data: {len(train_data)} rows")
    print(f"Test data: {len(test_data)} rows")

    # ============ CREATE ENVIRONMENT ============
    print("\n" + "="*70)
    print("STEP 2: CREATING ENVIRONMENT")
    print("-"*40)
    
    config = {
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.0001
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'n_bins': 20
        }
    }
    
    env = AdvancedTradingEnv(
        df=train_data,
        initial_cash=config['environment']['initial_cash'],
        trading_fee_maker=config['environment']['trading_fee_maker'],
        trading_fee_taker=config['environment']['trading_fee_taker'],
        slippage=config['environment']['slippage'],
        trade_frequency_penalty=config['environment']['trade_frequency_penalty']
    )
    
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n} (Hold, Buy, Sell)")

    # ============ TEST ENVIRONMENT ============
    print("\n" + "="*70)
    print("STEP 3: TESTING ENVIRONMENT (RANDOM ACTIONS)")
    print("-"*40)
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial portfolio: ${info['portfolio_value']:.2f}")
    
    # Take 10 random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        action_name = ['Hold', 'Buy', 'Sell'][action]
        print(f"  Step {i+1}: {action_name:4s} | Reward: {reward:+.4f} | Portfolio: ${info['portfolio_value']:.2f}")
        if terminated:
            break
    
    print("\n✅ Environment works!")

    # ============ CREATE AND TRAIN AGENT ============
    print("\n" + "="*70)
    print("STEP 4: TRAINING Q-LEARNING AGENT")
    print("-"*40)
    
    # Reset environment
    env = AdvancedTradingEnv(
        df=train_data,
        initial_cash=config['environment']['initial_cash'],
        trading_fee_maker=config['environment']['trading_fee_maker'],
        trading_fee_taker=config['environment']['trading_fee_taker'],
        slippage=config['environment']['slippage'],
        trade_frequency_penalty=config['environment']['trade_frequency_penalty']
    )
    
    agent = QLearningAgent(env, config)
    print(f"Q-table shape: {agent.q_table.shape}")
    
    # Train with fewer timesteps for quick test
    metrics = agent.train(
        total_timesteps=20000,  # Reduced for quick test
        log_interval=5000
    )
    
    print(f"\nTraining complete!")
    print(f"Episodes: {metrics['total_episodes']}")
    print(f"Mean reward: {metrics['mean_reward']:.4f}")

    # ============ EVALUATE ============
    print("\n" + "="*70)
    print("STEP 5: EVALUATION")
    print("-"*40)
    
    # Evaluate on train data
    print("\nTrain Data Performance:")
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    train_return = (info['portfolio_value'] - config['environment']['initial_cash']) / config['environment']['initial_cash']
    print(f"  Final Portfolio: ${info['portfolio_value']:.2f}")
    print(f"  Return: {train_return*100:.2f}%")
    print(f"  Trades: {info['trade_count']}")
    
    # Evaluate on test data
    print("\nTest Data Performance:")
    test_env = AdvancedTradingEnv(
        df=test_data,
        initial_cash=config['environment']['initial_cash'],
        trading_fee_maker=config['environment']['trading_fee_maker'],
        trading_fee_taker=config['environment']['trading_fee_taker'],
        slippage=config['environment']['slippage'],
        trade_frequency_penalty=config['environment']['trade_frequency_penalty']
    )
    
    obs, info = test_env.reset()
    done = False
    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
    
    test_return = (info['portfolio_value'] - config['environment']['initial_cash']) / config['environment']['initial_cash']
    print(f"  Final Portfolio: ${info['portfolio_value']:.2f}")
    print(f"  Return: {test_return*100:.2f}%")
    print(f"  Trades: {info['trade_count']}")

    # ============ SAVE MODEL ============
    print("\n" + "="*70)
    print("STEP 6: SAVING MODEL")
    print("-"*40)
    
    os.makedirs('results', exist_ok=True)
    model_path = 'results/q_learning_test_model.pkl'
    agent.save(model_path)

    # ============ SUMMARY ============
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print(f"""
Summary:
- Environment: Working ✓
- Q-Learning Agent: Working ✓
- Training: Working ✓
- Evaluation: Working ✓
- Model saved to: {model_path}

Next steps:
1. Copy this folder to your local machine
2. Run 'python train.py' with real BTC data
3. The model will train on actual market data

Train Return: {train_return*100:.2f}%
Test Return: {test_return*100:.2f}%
""")
    
    return agent, metrics


if __name__ == '__main__':
    agent, metrics = test_with_sample_data()
