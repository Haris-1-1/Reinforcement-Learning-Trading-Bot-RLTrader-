import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.ppo_agent import PPOAgent
from env.advanced_trading_env import AdvancedTradingEnv
from utils.dqn_data_loader import DQNDataLoader
def train_ppo_agent():
    print("\n" + "="*70)
    print("RL TRADING BOT - PPO TRAINING")
    print("="*70)
    print("Using Proximal Policy Optimization (PPO)")
    print("="*70)
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2025-10-20',
            'end_date': '2025-12-10',
            'interval': '15min',
            'test_split': 0.15
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.00,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.002
        },
        'ppo': {
            'learning_rate': 3e-4,
            'hidden_sizes': [256, 256],
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'n_epochs': 10,
            'batch_size': 64,
            'update_interval': 2048
        },
        'training': {
            'total_timesteps': 1000000,
            'log_interval': 10000
        }
    }
    print("\n" + "="*70)
    print(" STEP 1: LOADING DATA")
    print("="*70)
    data_loader = DQNDataLoader(
        ticker=config['data']['symbol'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        interval=config['data']['interval'],
        train_test_split=(1.0 - config['data']['test_split'])
    )
    train_data, test_data = data_loader.prepare_data(normalize=True)
    original_prices_train = data_loader.original_prices_train
    original_prices_test = data_loader.original_prices_test
    print(f"\n Train data: {len(train_data)} days")
    print(f" Test data: {len(test_data)} days")
    print(f" Price range (train): ${original_prices_train.min():.2f} - ${original_prices_train.max():.2f}")
    print("\n" + "="*70)
    print(" STEP 2: CREATING ENVIRONMENT")
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
    print(f"✓ Environment created with {len(train_data)} timesteps")
    print("\n" + "="*70)
    print(" STEP 3: CREATING PPO AGENT")
    print("="*70)
    agent = PPOAgent(
        env=env,
        learning_rate=config['ppo']['learning_rate'],
        gamma=config['ppo']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        clip_epsilon=config['ppo']['clip_epsilon'],
        value_coef=config['ppo']['value_coef'],
        entropy_coef=config['ppo']['entropy_coef'],
        max_grad_norm=config['ppo']['max_grad_norm'],
        n_epochs=config['ppo']['n_epochs'],
        batch_size=config['ppo']['batch_size'],
        hidden_sizes=config['ppo']['hidden_sizes']
    )
    print("\n" + "="*70)
    print(" STEP 4: TRAINING")
    print("="*70)
    metrics = agent.train(
        total_timesteps=config['training']['total_timesteps'],
        log_interval=config['training']['log_interval'],
        update_interval=config['ppo']['update_interval']
    )
    print("\n" + "="*70)
    print(" STEP 5: SAVING MODEL")
    print("="*70)
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'results/ppo_{timestamp}.pth'
    config_path = f'results/config_ppo_{timestamp}.json'
    agent.save(model_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Config saved to {config_path}")
    print("\n" + "="*70)
    print(" STEP 6: EVALUATING ON TRAINING DATA")
    print("="*70)
    env_eval_train = AdvancedTradingEnv(
        df=train_data,
        original_prices=original_prices_train,
        initial_cash=config['environment']['initial_cash']
    )
    obs, _ = env_eval_train.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env_eval_train.step(action)
        done = terminated or truncated
        total_reward += reward
    final_portfolio = env_eval_train.portfolio_value
    initial_cash = config['environment']['initial_cash']
    returns_pct = ((final_portfolio - initial_cash) / initial_cash) * 100
    print(f"Initial Portfolio: ${initial_cash:,.2f}")
    print(f"Final Portfolio:   ${final_portfolio:,.2f}")
    print(f"Returns:           {returns_pct:+.2f}%")
    print(f"Total Reward:      {total_reward:.2f}")
    print(f"Trades:            {info.get('trade_count', 0)}")
    print(f"Fees:              ${info.get('total_fees_paid', 0):.2f}")
    buy_hold_return = ((original_prices_train[-1] - original_prices_train[0]) / original_prices_train[0]) * 100
    print(f"\n Buy & Hold:     {buy_hold_return:+.2f}%")
    print(f" PPO Agent:      {returns_pct:+.2f}%")
    print(f" Difference:     {returns_pct - buy_hold_return:+.2f}%")
    print("\n" + "="*70)
    print(" STEP 7: EVALUATING ON TEST DATA")
    print("="*70)
    env_eval_test = AdvancedTradingEnv(
        df=test_data,
        original_prices=original_prices_test,
        initial_cash=config['environment']['initial_cash']
    )
    obs, _ = env_eval_test.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env_eval_test.step(action)
        done = terminated or truncated
        total_reward += reward
    final_portfolio_test = env_eval_test.portfolio_value
    returns_pct_test = ((final_portfolio_test - initial_cash) / initial_cash) * 100
    print(f"Initial Portfolio: ${initial_cash:,.2f}")
    print(f"Final Portfolio:   ${final_portfolio_test:,.2f}")
    print(f"Returns:           {returns_pct_test:+.2f}%")
    print(f"Total Reward:      {total_reward:.2f}")
    print(f"Trades:            {info.get('trade_count', 0)}")
    print(f"Fees:              ${info.get('total_fees_paid', 0):.2f}")
    buy_hold_return_test = ((original_prices_test[-1] - original_prices_test[0]) / original_prices_test[0]) * 100
    print(f"\n Buy & Hold:     {buy_hold_return_test:+.2f}%")
    print(f" PPO Agent:      {returns_pct_test:+.2f}%")
    print(f" Difference:     {returns_pct_test - buy_hold_return_test:+.2f}%")
    print("\n" + "="*70)
    print(" STEP 8: BUY & HOLD BASELINE COMPARISON")
    print("="*70)
    print("\n TRAINING DATA:")
    print(f"  Buy & Hold:  {buy_hold_return:+.2f}%")
    print(f"  PPO Agent:   {returns_pct:+.2f}%")
    print(f"  Difference:  {returns_pct - buy_hold_return:+.2f}%")
    print("\n TEST DATA:")
    print(f"  Buy & Hold:  {buy_hold_return_test:+.2f}%")
    print(f"  PPO Agent:   {returns_pct_test:+.2f}%")
    print(f"  Difference:  {returns_pct_test - buy_hold_return_test:+.2f}%")
    print("\n" + "="*70)
    print(" STEP 9: FINAL METRICS & RECOMMENDATIONS")
    print("="*70)
    avg_episode_reward = np.mean(metrics['episode_rewards'][-100:]) if metrics['episode_rewards'] else 0
    print(f"\n Average Episode Reward (last 100): {avg_episode_reward:.2f}")
    print(f" Total Episodes: {len(metrics['episode_rewards'])}")
    print("\n PERFORMANCE ASSESSMENT:")
    if returns_pct_test > buy_hold_return_test + 5:
        print("   EXCELLENT! Agent significantly outperforms Buy & Hold on test data")
        print("   Ready for live trading consideration")
    elif returns_pct_test > buy_hold_return_test:
        print("  ✓ GOOD! Agent beats Buy & Hold on test data")
        print("   Consider further training or hyperparameter tuning")
    elif returns_pct_test > buy_hold_return_test - 5:
        print("   OKAY. Agent performs similar to Buy & Hold")
        print("   More training or different parameters recommended")
    else:
        print("   POOR. Agent underperforms Buy & Hold significantly")
        print("   Needs more training, different hyperparameters, or more data")
    print("\n RECOMMENDATIONS:")
    if len(train_data) < 500:
        print("  • Consider using more training data (currently <500 days)")
    if returns_pct_test < 0:
        print("  • Try increasing update_interval for more stable learning")
        print("  • Try reducing clip_epsilon for more conservative updates")
    if avg_episode_reward < 0:
        print("  • Consider adjusting reward function")
        print("  • Try different learning_rate (current: 3e-4)")
    print("\n" + "="*70)
    print(" TRAINING COMPLETED!")
    print("="*70)
    print(f"Model saved: {model_path}")
    print(f"Config saved: {config_path}")
    print("="*70)
    return agent, metrics
if __name__ == "__main__":
    try:
        agent, metrics = train_ppo_agent()
    except KeyboardInterrupt:
        print("\n\n Training interrupted by user")
    except Exception as e:
        print(f"\n\n Error during training: {str(e)}")
        raise