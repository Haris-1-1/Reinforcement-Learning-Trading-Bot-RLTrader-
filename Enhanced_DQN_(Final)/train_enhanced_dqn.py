import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from utils.data_loader import DataLoader
from agents.dqn_agent import Agent
from env.trading_env import TradingEnvironment
from utils.visualizer import TrainingVisualizer
CONFIG = {
    "data": {
        "symbol": "BTC-USD",
        "start_date": "2024-01-01",
        "end_date": "2025-11-01",
        "interval": "1h",
        "window_size": 24,
        "test_split": 0.15
    },
    "environment": {
        "initial_cash": 10000.0,
        "fee": 0.001,
        "slippage": 0.0005
    },
    "agent": {
        "batch_size": 64,
        "episodes": 50,
        "target_update_freq": 2000,
        "learning_rate": 0.00005,
        "epsilon_start": 1.0,
        "epsilon_min": 0.15,
        "epsilon_decay": 0.99999
    },
    "paths": {
        "models": "models/",
        "logs": "logs/"
    }
}
def ensure_directories():
    for path in CONFIG["paths"].values():
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
def run_benchmarks(prices: np.ndarray, initial_cash: float) -> dict:
    print("\n" + "="*60)
    print("CALCULATING BENCHMARKS")
    print("="*60)
    results = {}
    bh_return = (prices[-1] / prices[0]) - 1
    results['buy_hold'] = {
        'return': bh_return,
        'final_value': initial_cash * (1 + bh_return)
    }
    print(f"Buy & Hold: {bh_return*100:+.2f}%")
    random_returns = []
    fee = CONFIG['environment']['fee']
    for run in range(10):
        cash = initial_cash
        coins = 0
        for i in range(len(prices)-1):
            action = np.random.choice([0, 1, 2])
            price = prices[i]
            if action == 1 and cash > 0:
                fee_amount = cash * fee
                coins = (cash - fee_amount) / price
                cash = 0
            elif action == 2 and coins > 0:
                cash = coins * price
                fee_amount = cash * fee
                cash -= fee_amount
                coins = 0
        final_value = cash + (coins * prices[-1])
        random_returns.append((final_value - initial_cash) / initial_cash)
    results['random'] = {
        'return': np.mean(random_returns),
        'std': np.std(random_returns),
        'final_value': initial_cash * (1 + np.mean(random_returns))
    }
    print(f"Random Trading: {np.mean(random_returns)*100:+.2f}% (±{np.std(random_returns)*100:.2f}%)")
    df_ma = pd.DataFrame({'Close': prices})
    df_ma['SMA20'] = df_ma['Close'].rolling(window=20).mean()
    df_ma['SMA50'] = df_ma['Close'].rolling(window=50).mean()
    cash, coins = initial_cash, 0
    for i in range(50, len(prices)-1):
        price = prices[i]
        sma20 = df_ma['SMA20'].iloc[i]
        sma50 = df_ma['SMA50'].iloc[i]
        if sma20 > sma50 and cash > 0:
            fee_amount = cash * fee
            coins = (cash - fee_amount) / price
            cash = 0
        elif sma20 < sma50 and coins > 0:
            cash = coins * price
            fee_amount = cash * fee
            cash -= fee_amount
            coins = 0
    ma_final = cash + (coins * prices[-1])
    ma_return = (ma_final - initial_cash) / initial_cash
    results['ma_crossover'] = {
        'return': ma_return,
        'final_value': ma_final
    }
    print(f"MA Crossover (20/50): {ma_return*100:+.2f}%")
    print("="*60 + "\n")
    return results
def save_checkpoint(agent: Agent, episode: int, portfolio_value: float, is_best: bool = False):
    filename = f"checkpoint_ep{episode}.pth" if not is_best else "best_model.pth"
    path = os.path.join(CONFIG["paths"]["models"], filename)
    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.policy_net.state_dict(),
        'target_state_dict': agent.target_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'portfolio_value': portfolio_value,
        'config': CONFIG
    }
    torch.save(checkpoint, path)
    if is_best:
        print(f"   NEW BEST MODEL SAVED: ${portfolio_value:.2f}")
def train():
    ensure_directories()
    print("\n" + "
    print("  ENHANCED DQN TRADING BOT - TRAINING")
    print("
    loader = DataLoader(
        symbol=CONFIG['data']['symbol'],
        start_date=CONFIG['data']['start_date'],
        end_date=CONFIG['data']['end_date'],
        interval=CONFIG['data']['interval'],
        test_split=CONFIG['data']['test_split']
    )
    train_df, test_df = loader.prepare_data()
    if train_df is None:
        print("ERROR: Data loading failed!")
        return
    window_size = CONFIG['data']['window_size']
    print("\nInitializing environment and agent...")
    env = TradingEnvironment(
        df=train_df,
        original_prices=loader.original_prices_train,
        window_size=window_size,
        initial_cash=CONFIG['environment']['initial_cash'],
        fee=CONFIG['environment']['fee'],
        slippage=CONFIG['environment']['slippage']
    )
    actual_feature_count = env.n_features
    print(f"\nAgent Configuration:")
    print(f"  Actual features in environment: {actual_feature_count}")
    print(f"  Window size: {window_size}")
    print(f"  Total input dimension: {window_size * actual_feature_count + 3}")
    agent = Agent(
        state_size=actual_feature_count,
        action_size=3,
        window_size=window_size
    )
    print("\n" + "="*70)
    print(f"TRAINING START - {CONFIG['agent']['episodes']} Episodes")
    print("="*70 + "\n")
    best_portfolio = 0
    total_steps = 0
    episode_rewards = []
    episode_portfolios = []
    episode_trades = []
    epsilon_history = []
    loss_history = []
    for episode in range(1, CONFIG['agent']['episodes'] + 1):
        state, info = env.reset()
        state = state.reshape(1, -1)
        done = False
        episode_reward = 0
        pbar = tqdm(total=len(train_df), desc=f"Episode {episode}/{CONFIG['agent']['episodes']}", unit="step")
        while not done:
            action = agent.act(state, action_mask=info['action_mask'])
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            next_state = next_state.reshape(1, -1)
            agent.remember(state, action, reward, next_state, done, next_info['action_mask'])
            loss = agent.replay()
            if loss is not None:
                loss_history.append(loss)
            if total_steps % 100 == 0:
                epsilon_history.append(agent.epsilon)
            if total_steps % CONFIG['agent']['target_update_freq'] == 0:
                agent.update_target_network()
            state = next_state
            info = next_info
            episode_reward += reward
            total_steps += 1
            pbar.set_postfix({
                'ε': f"{agent.epsilon:.3f}",
                'Portfolio': f"${info['portfolio_value']:.0f}",
                'Trades': info['trade_count']
            })
            pbar.update(1)
        pbar.close()
        final_value = env.portfolio_value
        episode_rewards.append(episode_reward)
        episode_portfolios.append(final_value)
        episode_trades.append(env.trade_count)
        print(f"Episode {episode}: Portfolio=${final_value:.2f} | Reward={episode_reward:.4f} | Trades={env.trade_count}")
        if final_value > best_portfolio:
            best_portfolio = final_value
            save_checkpoint(agent, episode, final_value, is_best=True)
        if episode % 5 == 0:
            save_checkpoint(agent, episode, final_value, is_best=False)
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nCreating training visualizations...")
    visualizer = TrainingVisualizer(save_dir="plots")
    visualizer.plot_training_progress(
        episode_rewards=episode_rewards,
        episode_portfolios=episode_portfolios,
        episode_trades=episode_trades,
        epsilon_history=epsilon_history,
        loss_history=loss_history,
        initial_cash=CONFIG['environment']['initial_cash']
    )
    print("Training plots created!\n")
    print("\n" + "="*70)
    print("EVALUATION ON TEST DATA")
    print("="*70 + "\n")
    best_model_path = os.path.join(CONFIG["paths"]["models"], "best_model.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best model")
    test_env = TradingEnvironment(
        df=test_df,
        original_prices=loader.original_prices_test,
        window_size=window_size,
        initial_cash=CONFIG['environment']['initial_cash'],
        fee=CONFIG['environment']['fee'],
        slippage=CONFIG['environment']['slippage']
    )
    agent.is_eval = True
    state, info = test_env.reset()
    state = state.reshape(1, -1)
    done = False
    print("Running bot on test data...")
    with tqdm(total=len(test_df), desc="Test", unit="step") as pbar:
        while not done:
            action = agent.act(state, action_mask=info['action_mask'])
            next_state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            state = next_state.reshape(1, -1)
            pbar.update(1)
    bot_final = info['portfolio_value']
    bot_return = (bot_final - CONFIG['environment']['initial_cash']) / CONFIG['environment']['initial_cash']
    benchmarks = run_benchmarks(loader.original_prices_test, CONFIG['environment']['initial_cash'])
    print("\n" + "
    print("  FINAL RESULTS")
    print("
    print(f"{'STRATEGY':<30} {'RETURN':>12} {'FINAL VALUE':>15}")
    print("-" * 70)
    print(f"{'Enhanced DQN (Bot)':<30} {bot_return*100:>+11.2f}% ${bot_final:>14,.2f}")
    print(f"{'Buy & Hold':<30} {benchmarks['buy_hold']['return']*100:>+11.2f}% ${benchmarks['buy_hold']['final_value']:>14,.2f}")
    print(f"{'MA Crossover (20/50)':<30} {benchmarks['ma_crossover']['return']*100:>+11.2f}% ${benchmarks['ma_crossover']['final_value']:>14,.2f}")
    print(f"{'Random Trading (avg)':<30} {benchmarks['random']['return']*100:>+11.2f}% ${benchmarks['random']['final_value']:>14,.2f}")
    print("-" * 70)
    print("\nCreating evaluation visualizations...")
    visualizer.plot_benchmark_comparison(
        results={
            'bot_return': bot_return,
            'buy_hold': benchmarks['buy_hold']['return'],
            'ma_crossover': benchmarks['ma_crossover']['return'],
            'random': benchmarks['random']['return']
        },
        initial_cash=CONFIG['environment']['initial_cash']
    )
    trades_df = test_env.get_trades_df()
    if len(trades_df) > 0:
        visualizer.plot_trade_analysis(
            trades_df=trades_df,
            prices=loader.original_prices_test
        )
    print("All plots saved to 'plots/' directory!")
    results = {
        'symbol': CONFIG['data']['symbol'],
        'test_period': f"{CONFIG['data']['start_date']} to {CONFIG['data']['end_date']}",
        'bot_return': float(bot_return),
        'bot_final_value': float(bot_final),
        'benchmarks': {k: {'return': float(v['return']), 'final_value': float(v['final_value'])}
                      for k, v in benchmarks.items()},
        'config': CONFIG
    }
    results_path = os.path.join(CONFIG["paths"]["logs"], f"results_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print("\n" + "
if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()