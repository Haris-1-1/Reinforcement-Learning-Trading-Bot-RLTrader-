"""
Simple script to train all 3 agents (Q-Learning, DQN, PPO).
"""

import os
from datetime import datetime
from utils.data_loader import DataLoader
from env.advanced_trading_env import AdvancedTradingEnv
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent


def train_all_agents():
    """Train all 3 agents and save models."""

    print("\n" + "=" * 60)
    print("TRAINING ALL 3 AGENTS")
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
        },
        'training': {
            'total_timesteps': 100000
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'n_bins': 10
        },
        'dqn': {
            'learning_rate': 0.0001,
            'buffer_size': 100000,
            'batch_size': 64,
            'gamma': 0.99,
            'policy_kwargs': {'net_arch': [128, 128]}
        },
        'ppo': {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'policy_kwargs': {'net_arch': [128, 128]}
        }
    }

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/all_agents_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    print(f"Results will be saved to: {results_dir}\n")

    # Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    data_loader = DataLoader(
        symbol=config['data']['symbol'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        interval=config['data']['interval'],
        test_split=config['data']['test_split']
    )
    train_data, test_data = data_loader.prepare_data()
    print(f"Train data: {len(train_data)} rows")
    print(f"Test data: {len(test_data)} rows\n")

    # Store results
    results = {}

    # === AGENT 1: Q-LEARNING ===
    print("\n" + "=" * 60)
    print("AGENT 1/3: Q-LEARNING")
    print("=" * 60 + "\n")

    env1 = AdvancedTradingEnv(train_data, **config['environment'])
    agent1 = QLearningAgent(env1, config)

    print("Training Q-Learning agent...")
    metrics1 = agent1.train(total_timesteps=config['training']['total_timesteps'])

    model_path1 = os.path.join(results_dir, 'q_learning_model.pkl')
    agent1.save(model_path1)

    results['Q-Learning'] = {
        'model_path': model_path1,
        'mean_reward': metrics1['mean_reward'],
        'total_episodes': metrics1['total_episodes']
    }

    print(f"✅ Q-Learning trained! Mean reward: {metrics1['mean_reward']:.4f}\n")

    # === AGENT 2: DQN ===
    print("\n" + "=" * 60)
    print("AGENT 2/3: DQN (Deep Q-Network)")
    print("=" * 60 + "\n")

    env2 = AdvancedTradingEnv(train_data, **config['environment'])
    agent2 = DQNAgent(env2, config)

    print("Training DQN agent...")
    metrics2 = agent2.train(total_timesteps=config['training']['total_timesteps'])

    model_path2 = os.path.join(results_dir, 'dqn_model')
    agent2.save(model_path2)

    results['DQN'] = {
        'model_path': model_path2,
        'mean_reward': metrics2['mean_reward'],
        'total_episodes': metrics2['total_episodes']
    }

    print(f"✅ DQN trained! Mean reward: {metrics2['mean_reward']:.4f}\n")

    # === AGENT 3: PPO ===
    print("\n" + "=" * 60)
    print("AGENT 3/3: PPO (Proximal Policy Optimization)")
    print("=" * 60 + "\n")

    env3 = AdvancedTradingEnv(train_data, **config['environment'])
    agent3 = PPOAgent(env3, config)

    print("Training PPO agent...")
    metrics3 = agent3.train(total_timesteps=config['training']['total_timesteps'])

    model_path3 = os.path.join(results_dir, 'ppo_model')
    agent3.save(model_path3)

    results['PPO'] = {
        'model_path': model_path3,
        'mean_reward': metrics3['mean_reward'],
        'total_episodes': metrics3['total_episodes']
    }

    print(f"✅ PPO trained! Mean reward: {metrics3['mean_reward']:.4f}\n")

    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60 + "\n")

    print(f"All models saved to: {results_dir}\n")
    print("Training Results:")
    print("-" * 60)

    for agent_name, data in results.items():
        print(f"{agent_name:15} | Mean Reward: {data['mean_reward']:8.4f} | Episodes: {data['total_episodes']:4}")

    print("-" * 60 + "\n")

    # Find best agent
    best_agent = max(results.items(), key=lambda x: x[1]['mean_reward'])
    print(f"🏆 Best Agent: {best_agent[0]} (Mean Reward: {best_agent[1]['mean_reward']:.4f})")

    print("\nNext steps:")
    print("1. Run 'python compare_agents.py' to compare on test data")
    print("2. Run 'python live_paper_trading.py' for live paper trading\n")

    return results_dir


if __name__ == '__main__':
    train_all_agents()
