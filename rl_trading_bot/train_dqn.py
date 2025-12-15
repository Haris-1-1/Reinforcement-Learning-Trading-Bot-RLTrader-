"""
Training Script for DQN Trading Bot

Uses Deep Q-Network instead of Q-Learning.
All configs are adjustable at the top of the file.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dqn_data_loader import DQNDataLoader
from env.advanced_trading_env import AdvancedTradingEnv
from agents.dqn_agent import DQNAgent


def train_dqn_agent():
    """Train DQN agent."""
    
    print("\n" + "="*70)
    print("🚀 RL TRADING BOT - DQN TRAINING")
    print("="*70)
    print("Using Deep Q-Network with Experience Replay")
    print("="*70 + "\n")

    # ════════════════════════════════════════════════════════════════
    # CONFIGURATION - ALLES HIER ÄNDERBAR!
    # ════════════════════════════════════════════════════════════════
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2024-01-01',
            'end_date': '2025-12-15',
            'interval': '1h',  
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005
        },
        'dqn': {
            # Neural Network
            'learning_rate': 0.0001,      # Änderbar: 0.001 (schneller) bis 0.00001 (langsamer)
            'hidden_sizes': [128, 128, 64],  # Änderbar: [64, 64] (klein) bis [256, 256, 128] (groß)
            
            # Reinforcement Learning
            'gamma': 0.99,                # Discount factor (0.95-0.99)
            'epsilon_start': 1.0,         # Exploration start (immer 1.0)
            'epsilon_end': 0.01,          # Exploration end (0.01-0.1)
            'epsilon_decay_steps': 50000, # Änderbar: Schneller = weniger Steps
            
            # Experience Replay
            'replay_buffer_size': 10000,  # Änderbar: Mehr = mehr Memory needed
            'batch_size': 64,             # Änderbar: 32 (klein) bis 128 (groß)
            
            # Target Network
            'target_update_freq': 1000    # Änderbar: Wie oft Target Network updated wird
        },
        'training': {
            'total_timesteps': 50000,   # Änderbar: DQN braucht mehr als Q-Learning!
            'log_interval': 10000
        }
    }

    # ════════════════════════════════════════════════════════════════
    # STEP 1: LOAD DATA
    # ════════════════════════════════════════════════════════════════
    print("="*70)
    print("📊 STEP 1: LOADING DATA")
    print("="*70)
    
    data_loader = DQNDataLoader(
        ticker=config['data']['symbol'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        interval=config['data']['interval'],
        train_test_split=(1.0 - config['data']['test_split'])  # Convert test_split to train_split
    )
    
    train_data, test_data = data_loader.prepare_data(normalize=True)
    
    # Get original prices for trading (stored BEFORE normalization in data loader)
    original_prices_train = data_loader.original_prices_train
    original_prices_test = data_loader.original_prices_test
    
    print(f"\n📈 Train data: {len(train_data)} days")
    print(f"📈 Test data: {len(test_data)} days")
    print(f"💰 Price range (train): ${original_prices_train.min():.2f} - ${original_prices_train.max():.2f}")

    # ════════════════════════════════════════════════════════════════
    # STEP 2: CREATE ENVIRONMENT
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🏗️ STEP 2: CREATING ENVIRONMENT")
    print("="*70)
    
    env = AdvancedTradingEnv(
        df=train_data,
        original_prices=original_prices_train,
        initial_cash=config['environment']['initial_cash'],
        trading_fee_maker=config['environment']['trading_fee_maker'],
        trading_fee_taker=config['environment']['trading_fee_taker'],
        slippage=config['environment']['slippage'],
        trade_frequency_penalty=config['environment']['trade_frequency_penalty']
    )

    # ════════════════════════════════════════════════════════════════
    # STEP 3: CREATE DQN AGENT
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🤖 STEP 3: CREATING DQN AGENT")
    print("="*70)
    
    agent = DQNAgent(env, config)

    # ════════════════════════════════════════════════════════════════
    # STEP 4: TRAIN
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🎯 STEP 4: TRAINING")
    print("="*70)
    
    metrics = agent.train(
        total_timesteps=config['training']['total_timesteps'],
        log_interval=config['training']['log_interval']
    )

    # ════════════════════════════════════════════════════════════════
    # STEP 5: SAVE MODEL
    # ════════════════════════════════════════════════════════════════
    print("="*70)
    print("💾 STEP 5: SAVING MODEL")
    print("="*70)
    
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'results/dqn_{timestamp}.pth'
    agent.save(model_path)
    
    config_path = f'results/config_dqn_{timestamp}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")

    # ════════════════════════════════════════════════════════════════
    # STEP 6: EVALUATE ON TRAIN DATA
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("📊 STEP 6: EVALUATION ON TRAINING DATA")
    print("="*70)
    
    obs, info = env.reset()
    done = False
    
    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    train_final = info['portfolio_value']
    train_return = (train_final - config['environment']['initial_cash']) / config['environment']['initial_cash']
    
    print(f"Initial: ${config['environment']['initial_cash']:,.2f}")
    print(f"Final:   ${train_final:,.2f}")
    print(f"Return:  {train_return*100:+.2f}%")
    print(f"Trades:  {info['trade_count']}")
    print(f"Fees:    ${info['total_fees_paid']:.2f}")

    # ════════════════════════════════════════════════════════════════
    # STEP 7: EVALUATE ON TEST DATA
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🧪 STEP 7: EVALUATION ON TEST DATA")
    print("="*70)
    
    test_env = AdvancedTradingEnv(
        df=test_data,
        original_prices=original_prices_test,
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
    
    test_final = info['portfolio_value']
    test_return = (test_final - config['environment']['initial_cash']) / config['environment']['initial_cash']
    
    print(f"Initial: ${config['environment']['initial_cash']:,.2f}")
    print(f"Final:   ${test_final:,.2f}")
    print(f"Return:  {test_return*100:+.2f}%")
    print(f"Trades:  {info['trade_count']}")
    print(f"Fees:    ${info['total_fees_paid']:.2f}")

    # ════════════════════════════════════════════════════════════════
    # STEP 8: BUY & HOLD COMPARISON
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("📈 STEP 8: BUY & HOLD COMPARISON")
    print("="*70)
    
    # Train Buy & Hold
    train_bh_return = (original_prices_train[-1] / original_prices_train[0] - 1)
    train_bh_final = config['environment']['initial_cash'] * (1 + train_bh_return)
    
    # Test Buy & Hold
    test_bh_return = (original_prices_test[-1] / original_prices_test[0] - 1)
    test_bh_final = config['environment']['initial_cash'] * (1 + test_bh_return)
    
    print(f"\n{'Strategy':<20} | {'Train':>12} | {'Test':>12}")
    print("-" * 50)
    print(f"{'DQN Agent':<20} | {train_return*100:>+11.2f}% | {test_return*100:>+11.2f}%")
    print(f"{'Buy & Hold':<20} | {train_bh_return*100:>+11.2f}% | {test_bh_return*100:>+11.2f}%")
    print("-" * 50)
    print(f"{'Outperformance':<20} | {(train_return-train_bh_return)*100:>+11.2f}% | {(test_return-test_bh_return)*100:>+11.2f}%")
    
    print(f"\n💰 Final Values:")
    print(f"   Agent Train: ${train_final:,.2f}  |  Buy&Hold: ${train_bh_final:,.2f}")
    print(f"   Agent Test:  ${test_final:,.2f}  |  Buy&Hold: ${test_bh_final:,.2f}")

    # ════════════════════════════════════════════════════════════════
    # STEP 9: TRAINING METRICS
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("📉 STEP 9: TRAINING METRICS")
    print("="*70)
    
    print(f"Total Episodes: {metrics['total_episodes']}")
    print(f"Mean Episode Reward: {metrics['mean_reward']:.4f}")
    print(f"Final Epsilon: {metrics['final_epsilon']:.4f}")
    if metrics.get('losses'):
        print(f"Mean Loss (last 100): {np.mean(metrics['losses'][-100:]):.4f}")

    # ════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Config: {config_path}")
    
    # Bewertung
    if test_return > test_bh_return:
        print("\n🏆 AGENT BEATS BUY & HOLD! 🏆")
    elif test_return > 0:
        print("\n✅ Agent made profit, but didn't beat Buy & Hold")
    else:
        print("\n⚠️ Agent lost money - try:")
        print("   - Increase total_timesteps (DQN needs more training)")
        print("   - Adjust learning_rate (try 0.0001 or 0.00005)")
        print("   - Increase hidden_sizes ([256, 256, 128])")
        print("   - Check if epsilon_decay_steps is appropriate")
    
    return agent, metrics


if __name__ == '__main__':
    agent, metrics = train_dqn_agent()