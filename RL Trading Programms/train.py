"""
Training script for RL trading agents.
Usage: python train.py --config configs/default_config.yaml
"""

import argparse
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.config_loader import ConfigLoader
from utils.visualization import Visualizer
from env.advanced_trading_env import AdvancedTradingEnv
from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent


def main():
    parser = argparse.ArgumentParser(description='Train RL Trading Agent')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = ConfigLoader.load_config(args.config)

    # Create results directory
    exp_name = config.get('experiment', {}).get('name', 'default')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(
        config.get('training', {}).get('save_path', 'results'),
        f"{exp_name}_{timestamp}"
    )
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'plots'), exist_ok=True)

    print(f"Results will be saved to: {results_dir}")

    # Save config to results directory
    ConfigLoader.save_config(config, os.path.join(results_dir, 'config.yaml'))

    # Load data
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)

    data_config = config['data']
    loader = DataLoader(
        symbol=data_config['symbol'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        interval=data_config['interval'],
        test_split=data_config.get('test_split', 0.2)
    )

    train_data, test_data = loader.prepare_data(normalize=True)

    print(f"Training on {data_config['symbol']} from {data_config['start_date']} to {data_config['end_date']}")

    # Create environment
    print("\n" + "="*50)
    print("CREATING ENVIRONMENT")
    print("="*50)

    env_config = config['environment']
    env = AdvancedTradingEnv(
        df=train_data,
        initial_cash=env_config['initial_cash'],
        trading_fee_maker=env_config['trading_fee_maker'],
        trading_fee_taker=env_config['trading_fee_taker'],
        slippage=env_config['slippage'],
        trade_frequency_penalty=env_config['trade_frequency_penalty'],
        max_position_size=env_config['max_position_size'],
        enable_execution_delay=env_config.get('enable_execution_delay', False),
        execution_delay_steps=env_config.get('execution_delay_steps', 0)
    )

    print(f"Environment created with {len(train_data)} training steps")

    # Create agent
    print("\n" + "="*50)
    print("CREATING AGENT")
    print("="*50)

    agent_type = config['agent']['type']
    print(f"Agent type: {agent_type}")

    if agent_type == 'dqn':
        agent = DQNAgent(env, config['agent'])
    elif agent_type == 'q_learning':
        agent = QLearningAgent(env, config['agent'])
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Train agent
    print("\n" + "="*50)
    print("TRAINING")
    print("="*50)

    training_config = config['training']
    metrics = agent.train(
        total_timesteps=training_config['total_timesteps'],
        log_interval=training_config.get('log_interval', 1000)
    )

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Total episodes: {metrics['total_episodes']}")
    print(f"Mean episode reward: {metrics['mean_reward']:.4f}")

    # Save model
    model_path = os.path.join(results_dir, 'model')
    if agent_type == 'dqn':
        model_path += '.zip'
    else:
        model_path += '.pkl'

    agent.save(model_path)

    # Plot training rewards
    print("\n" + "="*50)
    print("GENERATING PLOTS")
    print("="*50)

    visualizer = Visualizer()
    visualizer.plot_training_rewards(
        metrics['episode_rewards'],
        window=10,
        save_path=os.path.join(results_dir, 'plots', 'training_rewards.png')
    )

    # Evaluate on training data
    print("\n" + "="*50)
    print("EVALUATING ON TRAINING DATA")
    print("="*50)

    obs, info = env.reset()
    done = False
    portfolio_values = [info['portfolio_value']]

    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append(info['portfolio_value'])

    # Get trades
    trades_df = env.get_trades_df()

    # Generate performance report
    visualizer.generate_performance_report(
        initial_value=env_config['initial_cash'],
        final_value=info['portfolio_value'],
        trades_df=trades_df,
        df=train_data
    )

    # Save metrics
    import json
    metrics_save = {
        'agent_type': agent_type,
        'symbol': data_config['symbol'],
        'initial_value': env_config['initial_cash'],
        'final_value': float(info['portfolio_value']),
        'total_return': float((info['portfolio_value'] - env_config['initial_cash']) / env_config['initial_cash']),
        'total_trades': int(info['trade_count']),
        'total_fees_paid': float(info['total_fees_paid']),
        'mean_episode_reward': float(metrics['mean_reward']),
        'total_episodes': int(metrics['total_episodes'])
    }

    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_save, f, indent=4)

    # Save trades
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(results_dir, 'trades.csv'), index=False)

    print(f"\nAll results saved to: {results_dir}")
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
