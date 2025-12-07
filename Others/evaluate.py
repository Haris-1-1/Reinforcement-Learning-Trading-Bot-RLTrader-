"""
Evaluation script for RL trading agents.
Usage: python evaluate.py --model results/exp_001/model.zip --config results/exp_001/config.yaml
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.config_loader import ConfigLoader
from utils.visualization import Visualizer
from env.advanced_trading_env import AdvancedTradingEnv
from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent


def main():
    parser = argparse.ArgumentParser(description='Evaluate RL Trading Agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='test',
                       choices=['train', 'test'],
                       help='Dataset to evaluate on')
    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = ConfigLoader.load_config(args.config)

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

    # Select dataset
    eval_data = test_data if args.dataset == 'test' else train_data
    print(f"Evaluating on {args.dataset} data: {len(eval_data)} steps")

    # Create environment
    print("\n" + "="*50)
    print("CREATING ENVIRONMENT")
    print("="*50)

    env_config = config['environment']
    env = AdvancedTradingEnv(
        df=eval_data,
        initial_cash=env_config['initial_cash'],
        trading_fee_maker=env_config['trading_fee_maker'],
        trading_fee_taker=env_config['trading_fee_taker'],
        slippage=env_config['slippage'],
        trade_frequency_penalty=env_config['trade_frequency_penalty'],
        max_position_size=env_config['max_position_size'],
        enable_execution_delay=env_config.get('enable_execution_delay', False),
        execution_delay_steps=env_config.get('execution_delay_steps', 0)
    )

    # Load agent
    print("\n" + "="*50)
    print("LOADING AGENT")
    print("="*50)

    agent_type = config['agent']['type']
    print(f"Agent type: {agent_type}")

    if agent_type == 'dqn':
        agent = DQNAgent(env, config['agent'])
        agent.load(args.model)
    elif agent_type == 'q_learning':
        agent = QLearningAgent(env, config['agent'])
        agent.load(args.model)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Run evaluation
    print("\n" + "="*50)
    print("RUNNING EVALUATION")
    print("="*50)

    obs, info = env.reset()
    done = False
    portfolio_values = [info['portfolio_value']]
    step_count = 0

    while not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append(info['portfolio_value'])
        step_count += 1

        if step_count % 100 == 0:
            print(f"Step {step_count}/{len(eval_data)}, "
                  f"Portfolio: ${info['portfolio_value']:.2f}")

    # Get trades
    trades_df = env.get_trades_df()

    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)

    # Calculate Buy & Hold baseline
    initial_price = eval_data['Close'].iloc[0]
    final_price = eval_data['Close'].iloc[-1]
    buy_hold_return = (final_price - initial_price) / initial_price
    buy_hold_final_value = env_config['initial_cash'] * (1 + buy_hold_return)

    # Create Buy & Hold portfolio values
    buy_hold_values = env_config['initial_cash'] * (eval_data['Close'] / initial_price).values

    print(f"\nBuy & Hold Final Value: ${buy_hold_final_value:.2f}")
    print(f"Buy & Hold Return: {buy_hold_return:.2%}")

    # Generate performance report
    visualizer = Visualizer()
    metrics = visualizer.generate_performance_report(
        initial_value=env_config['initial_cash'],
        final_value=info['portfolio_value'],
        trades_df=trades_df,
        df=eval_data
    )

    # Create results directory
    results_dir = os.path.dirname(args.model)
    eval_dir = os.path.join(results_dir, f'evaluation_{args.dataset}')
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_dir, 'plots'), exist_ok=True)

    # Plot portfolio value
    portfolio_df = pd.DataFrame({
        'portfolio_value': portfolio_values[:-1]  # Remove last duplicate
    })

    visualizer.plot_portfolio_value(
        portfolio_df,
        buy_hold_value=buy_hold_values,
        save_path=os.path.join(eval_dir, 'plots', 'portfolio_value.png')
    )

    # Plot trades on price chart
    if not trades_df.empty:
        visualizer.plot_trades(
            eval_data,
            trades_df,
            save_path=os.path.join(eval_dir, 'plots', 'trades.png')
        )

    # Plot comparison
    comparison_results = {
        'RL Agent': info['portfolio_value'],
        'Buy & Hold': buy_hold_final_value
    }

    visualizer.plot_comparison(
        comparison_results,
        save_path=os.path.join(eval_dir, 'plots', 'comparison.png')
    )

    # Save detailed metrics
    import json
    detailed_metrics = {
        'dataset': args.dataset,
        'agent_type': agent_type,
        'symbol': data_config['symbol'],
        'initial_value': env_config['initial_cash'],
        'final_value': float(info['portfolio_value']),
        'total_return': float((info['portfolio_value'] - env_config['initial_cash']) / env_config['initial_cash']),
        'buy_hold_final_value': float(buy_hold_final_value),
        'buy_hold_return': float(buy_hold_return),
        'outperformance': float((info['portfolio_value'] - buy_hold_final_value) / buy_hold_final_value),
        'total_trades': int(info['trade_count']),
        'total_fees_paid': float(info['total_fees_paid']),
        'total_slippage_cost': float(info['total_slippage_cost']),
        'final_cash': float(info['cash']),
        'final_position': float(info['position'])
    }

    with open(os.path.join(eval_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(detailed_metrics, f, indent=4)

    # Save trades
    if not trades_df.empty:
        trades_df.to_csv(os.path.join(eval_dir, 'trades.csv'), index=False)

    # Save portfolio values
    portfolio_df.to_csv(os.path.join(eval_dir, 'portfolio_values.csv'), index=False)

    print(f"\nEvaluation results saved to: {eval_dir}")

    # Optional: Interactive plot
    try:
        print("\nGenerating interactive chart...")
        visualizer.plot_interactive_chart(eval_data, trades_df)
    except Exception as e:
        print(f"Could not generate interactive chart: {e}")

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
