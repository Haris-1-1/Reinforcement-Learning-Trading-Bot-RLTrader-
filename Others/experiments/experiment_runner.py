"""
Experiment Runner for comparing different configurations.
"""

import os
import sys
import json
from typing import List, Dict
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import DataLoader
from utils.config_loader import ConfigLoader
from utils.visualization import Visualizer
from env.advanced_trading_env import AdvancedTradingEnv
from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent


class ExperimentRunner:
    """Run and compare multiple experiments."""

    def __init__(self, base_config_path: str, results_base_dir: str = 'results/experiments'):
        """
        Initialize Experiment Runner.

        Args:
            base_config_path: Path to base configuration file
            results_base_dir: Base directory for experiment results
        """
        self.base_config = ConfigLoader.load_config(base_config_path)
        self.results_base_dir = results_base_dir
        self.experiments = []
        self.results = []

    def add_experiment(self, name: str, config_overrides: Dict):
        """
        Add an experiment with configuration overrides.

        Args:
            name: Experiment name
            config_overrides: Dictionary with configuration overrides
        """
        experiment_config = ConfigLoader.merge_configs(self.base_config, config_overrides)
        experiment_config['experiment']['name'] = name

        self.experiments.append({
            'name': name,
            'config': experiment_config
        })

        print(f"Added experiment: {name}")

    def run_experiment(self, experiment: Dict) -> Dict:
        """
        Run a single experiment.

        Args:
            experiment: Experiment dictionary

        Returns:
            Results dictionary
        """
        name = experiment['name']
        config = experiment['config']

        print("\n" + "="*70)
        print(f"RUNNING EXPERIMENT: {name}")
        print("="*70)

        # Create results directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(self.results_base_dir, f"{name}_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)

        # Save config
        ConfigLoader.save_config(config, os.path.join(exp_dir, 'config.yaml'))

        # Load data
        data_config = config['data']
        loader = DataLoader(
            symbol=data_config['symbol'],
            start_date=data_config['start_date'],
            end_date=data_config['end_date'],
            interval=data_config['interval'],
            test_split=data_config.get('test_split', 0.2)
        )

        train_data, test_data = loader.prepare_data(normalize=True)

        # Create environment
        env_config = config['environment']
        env = AdvancedTradingEnv(
            df=train_data,
            **{k: v for k, v in env_config.items() if k != 'initial_cash'},
            initial_cash=env_config['initial_cash']
        )

        # Create agent
        agent_type = config['agent']['type']

        if agent_type == 'dqn':
            agent = DQNAgent(env, config['agent'])
        elif agent_type == 'q_learning':
            agent = QLearningAgent(env, config['agent'])
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        # Train
        training_config = config['training']
        metrics = agent.train(
            total_timesteps=training_config['total_timesteps'],
            log_interval=training_config.get('log_interval', 1000)
        )

        # Save model
        model_path = os.path.join(exp_dir, f'model.{"zip" if agent_type == "dqn" else "pkl"}')
        agent.save(model_path)

        # Evaluate on test data
        test_env = AdvancedTradingEnv(
            df=test_data,
            **{k: v for k, v in env_config.items() if k != 'initial_cash'},
            initial_cash=env_config['initial_cash']
        )

        obs, info = test_env.reset()
        done = False
        test_portfolio_values = [info['portfolio_value']]

        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            test_portfolio_values.append(info['portfolio_value'])

        # Calculate metrics
        final_value = info['portfolio_value']
        total_return = (final_value - env_config['initial_cash']) / env_config['initial_cash']

        # Buy & Hold baseline
        buy_hold_return = (test_data['Close'].iloc[-1] - test_data['Close'].iloc[0]) / test_data['Close'].iloc[0]

        result = {
            'name': name,
            'agent_type': agent_type,
            'symbol': data_config['symbol'],
            'initial_value': env_config['initial_cash'],
            'final_value': float(final_value),
            'total_return': float(total_return),
            'buy_hold_return': float(buy_hold_return),
            'outperformance': float(total_return - buy_hold_return),
            'total_trades': int(info['trade_count']),
            'total_fees_paid': float(info['total_fees_paid']),
            'mean_episode_reward': float(metrics['mean_reward']),
            'result_dir': exp_dir
        }

        # Save metrics
        with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
            json.dump(result, f, indent=4)

        print(f"\nExperiment '{name}' complete!")
        print(f"Final Value: ${final_value:.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Outperformance vs Buy&Hold: {result['outperformance']:.2%}")

        return result

    def run_all_experiments(self) -> List[Dict]:
        """
        Run all added experiments.

        Returns:
            List of result dictionaries
        """
        print(f"\nRunning {len(self.experiments)} experiments...")

        for experiment in self.experiments:
            result = self.run_experiment(experiment)
            self.results.append(result)

        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETE")
        print("="*70)

        return self.results

    def generate_comparison_report(self, save_path: Optional[str] = None):
        """
        Generate comparison report of all experiments.

        Args:
            save_path: Optional path to save report
        """
        if not self.results:
            print("No results to compare. Run experiments first.")
            return

        # Create comparison DataFrame
        df = pd.DataFrame(self.results)

        print("\n" + "="*70)
        print("EXPERIMENT COMPARISON REPORT")
        print("="*70)
        print(df[['name', 'agent_type', 'symbol', 'total_return', 'outperformance', 'total_trades']].to_string(index=False))
        print("="*70)

        # Plot comparison
        visualizer = Visualizer()

        # Compare final values
        final_values = {r['name']: r['final_value'] for r in self.results}
        visualizer.plot_comparison(
            final_values,
            save_path=os.path.join(self.results_base_dir, 'comparison_final_values.png') if save_path is None else save_path
        )

        # Save comparison CSV
        csv_path = os.path.join(self.results_base_dir, 'comparison_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nComparison results saved to: {csv_path}")

        return df


# Example usage
if __name__ == '__main__':
    # Initialize runner with base config
    runner = ExperimentRunner('configs/default_config.yaml')

    # Experiment 1: DQN with constraints
    runner.add_experiment('DQN_with_constraints', {
        'agent': {'type': 'dqn'},
        'environment': {'enable_constraints': True}
    })

    # Experiment 2: DQN without constraints
    runner.add_experiment('DQN_no_constraints', {
        'agent': {'type': 'dqn'},
        'environment': {
            'trading_fee_maker': 0.0,
            'trading_fee_taker': 0.0,
            'slippage': 0.0,
            'trade_frequency_penalty': 0.0
        }
    })

    # Experiment 3: Q-Learning with constraints
    runner.add_experiment('QLearning_with_constraints', {
        'agent': {'type': 'q_learning'},
        'environment': {'enable_constraints': True}
    })

    # Experiment 4: Different crypto (ETH)
    runner.add_experiment('DQN_ETH', {
        'agent': {'type': 'dqn'},
        'data': {'symbol': 'ETH-USD'}
    })

    # Run all experiments
    results = runner.run_all_experiments()

    # Generate comparison report
    runner.generate_comparison_report()
