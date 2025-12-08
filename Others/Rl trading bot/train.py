"""
Training Script for RL Trading Bot
Usage: python train.py
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from env.advanced_trading_env import AdvancedTradingEnv
from agents.q_learning_agent import QLearningAgent


def train_agent():
    """Train the Q-Learning agent on BTC data."""
    
    print("\n" + "="*70)
    print("RL TRADING BOT - Q-LEARNING TRAINING")
    print("="*70 + "\n")

    # ============ CONFIGURATION ============
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2020-01-01',
            'end_date': '2024-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,    # 0.1%
            'trading_fee_taker': 0.002,    # 0.2%
            'slippage': 0.001,              # 0.1%
            'trade_frequency_penalty': 0.0001
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'n_bins': 20
        },
        'training': {
            'total_timesteps': 50000,
            'log_interval': 5000
        }
    }

    # ============ LOAD DATA ============
    print("="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    data_loader = DataLoader(
        symbol=config['data']['symbol'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        interval=config['data']['interval'],
        test_split=config['data']['test_split']
    )
    
    train_data, test_data = data_loader.prepare_data(normalize=True)
    
    print(f"\nTrain data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # ============ CREATE ENVIRONMENT ============
    print("\n" + "="*70)
    print("STEP 2: CREATING ENVIRONMENT")
    print("="*70)
    
    env = AdvancedTradingEnv(
        df=train_data,
        initial_cash=config['environment']['initial_cash'],
        trading_fee_maker=config['environment']['trading_fee_maker'],
        trading_fee_taker=config['environment']['trading_fee_taker'],
        slippage=config['environment']['slippage'],
        trade_frequency_penalty=config['environment']['trade_frequency_penalty']
    )
    
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n} actions (Hold, Buy, Sell)")

    # ============ CREATE AGENT ============
    print("\n" + "="*70)
    print("STEP 3: CREATING Q-LEARNING AGENT")
    print("="*70)
    
    agent = QLearningAgent(env, config)
    print(f"Q-table shape: {agent.q_table.shape}")

    # ============ TRAIN ============
    print("\n" + "="*70)
    print("STEP 4: TRAINING")
    print("="*70)
    
    metrics = agent.train(
        total_timesteps=config['training']['total_timesteps'],
        log_interval=config['training']['log_interval']
    )

    # ============ SAVE MODEL ============
    print("\n" + "="*70)
    print("STEP 5: SAVING MODEL")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'results/q_learning_model_{timestamp}.pkl'
    agent.save(model_path)
    
    # Save config
    config_path = f'results/config_{timestamp}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")

    # ============ EVALUATE ON TRAIN DATA ============
    print("\n" + "="*70)
    print("STEP 6: EVALUATION ON TRAINING DATA")
    print("="*70)
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    train_final_value = info['portfolio_value']
    train_return = (train_final_value - config['environment']['initial_cash']) / config['environment']['initial_cash']
    
    print(f"Initial Cash: ${config['environment']['initial_cash']:,.2f}")
    print(f"Final Portfolio Value: ${train_final_value:,.2f}")
    print(f"Total Return: {train_return*100:.2f}%")
    print(f"Total Trades: {info['trade_count']}")
    print(f"Fees Paid: ${info['total_fees_paid']:.2f}")

    # ============ EVALUATE ON TEST DATA ============
    print("\n" + "="*70)
    print("STEP 7: EVALUATION ON TEST DATA (OUT-OF-SAMPLE)")
    print("="*70)
    
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
    
    test_final_value = info['portfolio_value']
    test_return = (test_final_value - config['environment']['initial_cash']) / config['environment']['initial_cash']
    
    print(f"Initial Cash: ${config['environment']['initial_cash']:,.2f}")
    print(f"Final Portfolio Value: ${test_final_value:,.2f}")
    print(f"Total Return: {test_return*100:.2f}%")
    print(f"Total Trades: {info['trade_count']}")
    print(f"Fees Paid: ${info['total_fees_paid']:.2f}")

    # ============ BUY & HOLD COMPARISON ============
    print("\n" + "="*70)
    print("STEP 8: BUY & HOLD COMPARISON")
    print("="*70)
    
    # Train period Buy & Hold
    train_start_price = train_data['Close'].iloc[0]
    train_end_price = train_data['Close'].iloc[-1]
    # For normalized data, we need original prices - use a simpler calculation
    train_bh_return = (train_end_price - train_start_price) / abs(train_start_price) if train_start_price != 0 else 0
    
    # Test period Buy & Hold
    test_start_price = test_data['Close'].iloc[0]
    test_end_price = test_data['Close'].iloc[-1]
    test_bh_return = (test_end_price - test_start_price) / abs(test_start_price) if test_start_price != 0 else 0
    
    print(f"\n{'Strategy':<20} | {'Train Return':>15} | {'Test Return':>15}")
    print("-" * 55)
    print(f"{'Q-Learning Agent':<20} | {train_return*100:>14.2f}% | {test_return*100:>14.2f}%")
    print(f"{'Buy & Hold':<20} | {train_bh_return*100:>14.2f}% | {test_bh_return*100:>14.2f}%")
    print("-" * 55)
    
    # Outperformance
    train_outperf = train_return - train_bh_return
    test_outperf = test_return - test_bh_return
    print(f"{'Outperformance':<20} | {train_outperf*100:>14.2f}% | {test_outperf*100:>14.2f}%")

    # ============ SUMMARY ============
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {model_path}")
    print(f"Config saved to: {config_path}")
    print(f"\nTo load and use this model later:")
    print(f"  agent.load('{model_path}')")
    
    return agent, metrics


def test_environment():
    """Quick test to ensure environment works."""
    print("\n" + "="*70)
    print("QUICK ENVIRONMENT TEST")
    print("="*70 + "\n")
    
    # Load a small amount of data
    data_loader = DataLoader(
        symbol='BTC-USD',
        start_date='2023-01-01',
        end_date='2023-06-01',
        interval='1d',
        test_split=0.2
    )
    
    train_data, _ = data_loader.prepare_data(normalize=True)
    
    # Create environment
    env = AdvancedTradingEnv(
        df=train_data,
        initial_cash=10000.0
    )
    
    # Run a few random steps
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial portfolio value: ${info['portfolio_value']:.2f}")
    
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        action_name = ['Hold', 'Buy', 'Sell'][action]
        print(f"Step {i+1}: Action={action_name}, Reward={reward:.4f}, Portfolio=${info['portfolio_value']:.2f}")
        
        if terminated:
            break
    
    print("\n✅ Environment test passed!")
    return True


if __name__ == '__main__':
    # First test the environment
    test_environment()
    
    # Then train the agent
    agent, metrics = train_agent()
