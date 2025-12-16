"""
Training Script for RL Trading Bot - VERSION 2

FIXES:
1. Verwendet ECHTE Preise für Trading (nicht normalisiert)
2. Schnellerer Epsilon Decay
3. Mehr Training Timesteps
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from env.advanced_trading_env import AdvancedTradingEnv
from agents.q_learning_agent import QLearningAgent

#python rl_trading_bot\train_qlearning.py
def train_agent():
    """Train Q-Learning agent with REAL prices."""
    
    print("\n" + "="*70)
    print("🚀 RL TRADING BOT - Q-LEARNING TRAINING v2")
    print("="*70)
    print("FIXES: Real prices, faster epsilon decay, more training")
    print("="*70 + "\n")

    # ════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ════════════════════════════════════════════════════════════════
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2025-10-20',
            'end_date': '2025-12-15',
            'interval': '15min',  
            'test_split': 0.15
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.000,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.002  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.95,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 1000000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }

    # ════════════════════════════════════════════════════════════════
    # STEP 1: LOAD DATA
    # ════════════════════════════════════════════════════════════════
    print("="*70)
    print("📊 STEP 1: LOADING DATA")
    print("="*70)
    
    data_loader = DataLoader(
        symbol=config['data']['symbol'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        interval=config['data']['interval'],
        test_split=config['data']['test_split']
    )
    
    train_data, test_data = data_loader.prepare_data(normalize=True)
    
    # ════════════════════════════════════════════════════════════════
    # NEU: Hole ORIGINALE Preise!
    # ════════════════════════════════════════════════════════════════
    original_prices_train = data_loader.get_original_prices('train')
    original_prices_test = data_loader.get_original_prices('test')
    
    print(f"\n📈 Train data: {len(train_data)} days")
    print(f"📈 Test data: {len(test_data)} days")

    # ════════════════════════════════════════════════════════════════
    # STEP 2: CREATE ENVIRONMENT WITH REAL PRICES
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🏗️ STEP 2: CREATING ENVIRONMENT")
    print("="*70)
    
    env = AdvancedTradingEnv(
        df=train_data,
        original_prices=original_prices_train,  # ECHTE PREISE!
        initial_cash=config['environment']['initial_cash'],
        trading_fee_maker=config['environment']['trading_fee_maker'],
        trading_fee_taker=config['environment']['trading_fee_taker'],
        slippage=config['environment']['slippage'],
        trade_frequency_penalty=config['environment']['trade_frequency_penalty']
    )

    # ════════════════════════════════════════════════════════════════
    # STEP 3: CREATE AGENT
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🤖 STEP 3: CREATING Q-LEARNING AGENT")
    print("="*70)
    
    agent = QLearningAgent(env, config)

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
    model_path = f'results/q_learning_v2_{timestamp}.pkl'
    agent.save(model_path)
    
    config_path = f'results/config_v2_{timestamp}.json'
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
        original_prices=original_prices_test,  # ECHTE PREISE!
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
    print(f"{'Q-Learning Agent':<20} | {train_return*100:>+11.2f}% | {test_return*100:>+11.2f}%")
    print(f"{'Buy & Hold':<20} | {train_bh_return*100:>+11.2f}% | {test_bh_return*100:>+11.2f}%")
    print("-" * 50)
    print(f"{'Outperformance':<20} | {(train_return-train_bh_return)*100:>+11.2f}% | {(test_return-test_bh_return)*100:>+11.2f}%")
    
    print(f"\n💰 Final Values:")
    print(f"   Agent Train: ${train_final:,.2f}  |  Buy&Hold: ${train_bh_final:,.2f}")
    print(f"   Agent Test:  ${test_final:,.2f}  |  Buy&Hold: ${test_bh_final:,.2f}")

    # ════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Final Epsilon: {metrics['final_epsilon']:.4f}")
    print(f"Total Episodes: {metrics['total_episodes']}")
    
    # Bewertung
    if test_return > test_bh_return:
        print("\n🏆 AGENT BEATS BUY & HOLD! 🏆")
    elif test_return > 0:
        print("\n✅ Agent made profit, but didn't beat Buy & Hold")
    else:
        print("\n⚠️ Agent lost money - needs more training or tuning")
    
    return agent, metrics


if __name__ == '__main__':
    agent, metrics = train_agent()
