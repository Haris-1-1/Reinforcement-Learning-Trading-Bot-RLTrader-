"""
Compare all 3 trained agents on test data to see which makes the most profit.
"""

import os
import sys
import numpy as np
from utils.data_loader import DataLoader
from env.advanced_trading_env import AdvancedTradingEnv
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent


def compare_agents(results_dir):
    """
    Load and compare all trained agents on test data.

    Args:
        results_dir: Directory containing trained models
    """

    print("\n" + "=" * 60)
    print("COMPARING ALL 3 AGENTS ON TEST DATA")
    print("=" * 60 + "\n")

    # Configuration
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
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.0001
        }
    }

    # Load test data
    print("Loading test data...")
    data_loader = DataLoader(
        symbol=config['data']['symbol'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        interval=config['data']['interval'],
        test_split=config['data']['test_split']
    )
    _, test_data = data_loader.prepare_data()
    print(f"Test data: {len(test_data)} rows\n")

    # Results storage
    results = {}

    # === TEST AGENT 1: Q-LEARNING ===
    print("-" * 60)
    print("AGENT 1/3: Q-LEARNING")
    print("-" * 60)

    env1 = AdvancedTradingEnv(test_data, **config['environment'])
    agent1 = QLearningAgent(env1, config)
    agent1.load(os.path.join(results_dir, 'q_learning_model.pkl'))

    obs, info = env1.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent1.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env1.step(action)
        episode_reward += reward
        done = terminated or truncated

    results['Q-Learning'] = {
        'final_value': info['portfolio_value'],
        'profit': info['portfolio_value'] - config['environment']['initial_cash'],
        'return_pct': ((info['portfolio_value'] - config['environment']['initial_cash']) / config['environment']['initial_cash']) * 100,
        'trades': info['trade_count'],
        'fees_paid': info['total_fees_paid']
    }

    print(f"Final Portfolio Value: ${results['Q-Learning']['final_value']:.2f}")
    print(f"Profit: ${results['Q-Learning']['profit']:.2f}")
    print(f"Return: {results['Q-Learning']['return_pct']:.2f}%")
    print(f"Total Trades: {results['Q-Learning']['trades']}\n")

    # === TEST AGENT 2: DQN ===
    print("-" * 60)
    print("AGENT 2/3: DQN")
    print("-" * 60)

    env2 = AdvancedTradingEnv(test_data, **config['environment'])
    agent2 = DQNAgent(env2, config)
    agent2.load(os.path.join(results_dir, 'dqn_model.zip'))

    obs, info = env2.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent2.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env2.step(action)
        episode_reward += reward
        done = terminated or truncated

    results['DQN'] = {
        'final_value': info['portfolio_value'],
        'profit': info['portfolio_value'] - config['environment']['initial_cash'],
        'return_pct': ((info['portfolio_value'] - config['environment']['initial_cash']) / config['environment']['initial_cash']) * 100,
        'trades': info['trade_count'],
        'fees_paid': info['total_fees_paid']
    }

    print(f"Final Portfolio Value: ${results['DQN']['final_value']:.2f}")
    print(f"Profit: ${results['DQN']['profit']:.2f}")
    print(f"Return: {results['DQN']['return_pct']:.2f}%")
    print(f"Total Trades: {results['DQN']['trades']}\n")

    # === TEST AGENT 3: PPO ===
    print("-" * 60)
    print("AGENT 3/3: PPO")
    print("-" * 60)

    env3 = AdvancedTradingEnv(test_data, **config['environment'])
    agent3 = PPOAgent(env3, config)
    agent3.load(os.path.join(results_dir, 'ppo_model.zip'))

    obs, info = env3.reset()
    done = False
    episode_reward = 0

    while not done:
        action = agent3.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env3.step(action)
        episode_reward += reward
        done = terminated or truncated

    results['PPO'] = {
        'final_value': info['portfolio_value'],
        'profit': info['portfolio_value'] - config['environment']['initial_cash'],
        'return_pct': ((info['portfolio_value'] - config['environment']['initial_cash']) / config['environment']['initial_cash']) * 100,
        'trades': info['trade_count'],
        'fees_paid': info['total_fees_paid']
    }

    print(f"Final Portfolio Value: ${results['PPO']['final_value']:.2f}")
    print(f"Profit: ${results['PPO']['profit']:.2f}")
    print(f"Return: {results['PPO']['return_pct']:.2f}%")
    print(f"Total Trades: {results['PPO']['trades']}\n")

    # === BUY & HOLD BASELINE ===
    print("-" * 60)
    print("BASELINE: BUY & HOLD")
    print("-" * 60)

    buy_hold_return = ((test_data['Close'].iloc[-1] - test_data['Close'].iloc[0]) / test_data['Close'].iloc[0]) * 100
    buy_hold_value = config['environment']['initial_cash'] * (1 + buy_hold_return / 100)

    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Buy & Hold Value: ${buy_hold_value:.2f}\n")

    # === FINAL COMPARISON ===
    print("\n" + "=" * 60)
    print("FINAL COMPARISON - PAPER TRADING RESULTS")
    print("=" * 60 + "\n")

    print(f"{'Agent':<15} | {'Profit':>10} | {'Return':>8} | {'Trades':>6} | {'Fees':>10}")
    print("-" * 60)

    for agent_name, data in results.items():
        print(f"{agent_name:<15} | ${data['profit']:>9.2f} | {data['return_pct']:>7.2f}% | {data['trades']:>6} | ${data['fees_paid']:>9.2f}")

    print(f"{'Buy & Hold':<15} | ${buy_hold_value - config['environment']['initial_cash']:>9.2f} | {buy_hold_return:>7.2f}% | {'1':>6} | ${'0.00':>9}")
    print("-" * 60 + "\n")

    # Find best agent
    best_agent = max(results.items(), key=lambda x: x[1]['profit'])
    print(f"🏆 WINNER: {best_agent[0]}")
    print(f"   Profit: ${best_agent[1]['profit']:.2f}")
    print(f"   Return: {best_agent[1]['return_pct']:.2f}%")

    if best_agent[1]['return_pct'] > buy_hold_return:
        print(f"   ✅ Outperformed Buy & Hold by {best_agent[1]['return_pct'] - buy_hold_return:.2f}%!")
    else:
        print(f"   ❌ Underperformed Buy & Hold by {buy_hold_return - best_agent[1]['return_pct']:.2f}%")

    print(f"\nUse '{best_agent[0]}' for live paper trading!\n")

    return best_agent[0]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compare_agents.py <results_directory>")
        print("Example: python compare_agents.py results/all_agents_20241206_231234")
        sys.exit(1)

    results_dir = sys.argv[1]

    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found!")
        sys.exit(1)

    compare_agents(results_dir)
