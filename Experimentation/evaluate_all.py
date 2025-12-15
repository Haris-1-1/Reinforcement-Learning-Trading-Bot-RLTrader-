"""
Comprehensive Agent Evaluation & Comparison
Loads all trained agents and runs fair comparison
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from env.advanced_trading_env import AdvancedTradingEnv
from utils.dqn_data_loader import DQNDataLoader
from compare_agents import AgentComparison


def load_latest_model(agent_type: str, results_dir: str = 'results') -> str:
    """
    Find latest trained model for given agent type
    
    Args:
        agent_type: 'q_learning', 'dqn', or 'ppo'
        results_dir: Directory containing saved models
        
    Returns:
        Path to latest model file
    """
    if agent_type == 'q_learning':
        pattern = f"{results_dir}/q_learning*.pkl"
    elif agent_type == 'dqn':
        pattern = f"{results_dir}/dqn*.pth"
    elif agent_type == 'ppo':
        pattern = f"{results_dir}/ppo*.pth"
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Get most recent file
    latest = max(files, key=os.path.getctime)
    return latest


def evaluate_all_agents():
    """
    Main evaluation function - loads and compares all trained agents
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE AGENT EVALUATION & COMPARISON")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # ════════════════════════════════════════════════════════════════
    # STEP 1: LOAD TEST DATA (SAME FOR ALL AGENTS)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("📊 STEP 1: LOADING TEST DATA")
    print("="*70)
    
    # Use same config as training
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-11-11',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005
        }
    }
    
    # Load data
    data_loader = DQNDataLoader(
        ticker=config['data']['symbol'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        interval=config['data']['interval'],
        train_test_split=(1.0 - config['data']['test_split'])
    )
    
    train_data, test_data = data_loader.prepare_data(normalize=True)
    original_prices_test = data_loader.original_prices_test
    
    print(f"✓ Test data loaded: {len(test_data)} days")
    print(f"✓ Price range: ${original_prices_test.min():.2f} - ${original_prices_test.max():.2f}")
    
    # ════════════════════════════════════════════════════════════════
    # STEP 2: INITIALIZE COMPARISON TOOL
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🔧 STEP 2: INITIALIZING COMPARISON TOOL")
    print("="*70)
    
    comparison = AgentComparison(
        test_data=test_data,
        original_prices=original_prices_test,
        initial_cash=config['environment']['initial_cash']
    )
    
    print("✓ Comparison tool ready")
    
    # ════════════════════════════════════════════════════════════════
    # STEP 3: CALCULATE BUY & HOLD BASELINE
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("📈 STEP 3: CALCULATING BASELINE")
    print("="*70)
    
    comparison.calculate_baseline()
    
    # ════════════════════════════════════════════════════════════════
    # STEP 4: LOAD & EVALUATE Q-LEARNING AGENT
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🤖 STEP 4: Q-LEARNING AGENT")
    print("="*70)
    
    q_model_path = load_latest_model('q_learning')
    
    if q_model_path:
        print(f"Found model: {q_model_path}")
        
        try:
            # Create environment
            env_q = AdvancedTradingEnv(
                df=test_data,
                original_prices=original_prices_test,
                initial_cash=config['environment']['initial_cash']
            )
            
            # Load Q-Learning agent
            q_agent = QLearningAgent(env_q, config={'learning_rate': 0.1, 'gamma': 0.95})
            q_agent.load(q_model_path)
            
            # Add to comparison
            comparison.add_agent(q_agent, env_q, 'Q-Learning')
            
            print("✓ Q-Learning evaluation complete")
        
        except Exception as e:
            print(f"❌ Error loading Q-Learning agent: {str(e)}")
    else:
        print("⚠️ No Q-Learning model found - skipping")
    
    # ════════════════════════════════════════════════════════════════
    # STEP 5: LOAD & EVALUATE DQN AGENT
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🤖 STEP 5: DQN AGENT")
    print("="*70)
    
    dqn_model_path = load_latest_model('dqn')
    
    if dqn_model_path:
        print(f"Found model: {dqn_model_path}")
        
        try:
            # Create environment
            env_dqn = AdvancedTradingEnv(
                df=test_data,
                original_prices=original_prices_test,
                initial_cash=config['environment']['initial_cash']
            )
            
            # Load DQN agent
            dqn_agent = DQNAgent(
                env=env_dqn,
                learning_rate=0.0001,
                gamma=0.99,
                epsilon_start=0.01,  # Set to minimum for evaluation
                epsilon_end=0.01,
                epsilon_decay_steps=1,
                replay_buffer_size=10000,
                batch_size=64,
                target_update_freq=1000,
                hidden_sizes=[128, 128, 64]
            )
            dqn_agent.load(dqn_model_path)
            
            # Add to comparison
            comparison.add_agent(dqn_agent, env_dqn, 'DQN')
            
            print("✓ DQN evaluation complete")
        
        except Exception as e:
            print(f"❌ Error loading DQN agent: {str(e)}")
    else:
        print("⚠️ No DQN model found - skipping")
    
    # ════════════════════════════════════════════════════════════════
    # STEP 6: LOAD & EVALUATE PPO AGENT
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("🤖 STEP 6: PPO AGENT")
    print("="*70)
    
    ppo_model_path = load_latest_model('ppo')
    
    if ppo_model_path:
        print(f"Found model: {ppo_model_path}")
        
        try:
            # Create environment
            env_ppo = AdvancedTradingEnv(
                df=test_data,
                original_prices=original_prices_test,
                initial_cash=config['environment']['initial_cash']
            )
            
            # Load PPO agent
            ppo_agent = PPOAgent(
                env=env_ppo,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                max_grad_norm=0.5,
                n_epochs=10,
                batch_size=64,
                hidden_sizes=[256, 256]
            )
            ppo_agent.load(ppo_model_path)
            
            # Add to comparison
            comparison.add_agent(ppo_agent, env_ppo, 'PPO')
            
            print("✓ PPO evaluation complete")
        
        except Exception as e:
            print(f"❌ Error loading PPO agent: {str(e)}")
    else:
        print("⚠️ No PPO model found - skipping")
    
    # ════════════════════════════════════════════════════════════════
    # STEP 7: GENERATE COMPARISON REPORT
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("📊 STEP 7: GENERATING COMPARISON REPORT")
    print("="*70)
    
    # Print report to console
    report = comparison.generate_report()
    print("\n" + report)
    
    # ════════════════════════════════════════════════════════════════
    # STEP 8: SAVE ALL RESULTS
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("💾 STEP 8: SAVING RESULTS")
    print("="*70)
    
    comparison.save_all_results('results/comparison')
    
    # ════════════════════════════════════════════════════════════════
    # STEP 9: FINAL RECOMMENDATIONS
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("💡 STEP 9: RECOMMENDATIONS")
    print("="*70)
    
    if len(comparison.results) > 0:
        df = comparison.rank_agents('Total Return (%)')
        winner = df.iloc[0]
        
        print(f"\n🏆 RECOMMENDED AGENT: {winner['Agent']}")
        print(f"   Reason: Highest total return ({winner['Total Return (%)']:+.2f}%)")
        
        # Check if significantly better than baseline
        if comparison.baseline_results:
            baseline_return = comparison.baseline_results['total_return_pct']
            if winner['Total Return (%)'] > baseline_return + 5:
                print(f"   ✅ Significantly outperforms Buy & Hold (+{winner['Total Return (%)'] - baseline_return:.2f}%)")
            elif winner['Total Return (%)'] > baseline_return:
                print(f"   ✓ Beats Buy & Hold (+{winner['Total Return (%)'] - baseline_return:.2f}%)")
            else:
                print(f"   ⚠️ Underperforms Buy & Hold ({winner['Total Return (%)'] - baseline_return:.2f}%)")
                print("   Consider more training or different hyperparameters")
        
        # Risk assessment
        if winner['Sharpe Ratio'] > 1:
            print(f"   ✅ Good risk-adjusted returns (Sharpe: {winner['Sharpe Ratio']:.3f})")
        else:
            print(f"   ⚠️ Low risk-adjusted returns (Sharpe: {winner['Sharpe Ratio']:.3f})")
        
        if winner['Max Drawdown (%)'] < 20:
            print(f"   ✅ Acceptable risk (Max DD: {winner['Max Drawdown (%)']:.2f}%)")
        else:
            print(f"   ⚠️ High risk (Max DD: {winner['Max Drawdown (%)']:.2f}%)")
    
    print("\n" + "="*70)
    print("✓ EVALUATION COMPLETE!")
    print("="*70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    try:
        evaluate_all_agents()
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
