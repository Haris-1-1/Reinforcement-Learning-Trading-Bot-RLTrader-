import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import time
from datetime import datetime
from tqdm import tqdm
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
from utils.data_loader import DataLoader
from utils.agent import Agent
from utils.environment import TradingEnvironment
from utils.indicators import TechnicalIndicators
CONFIG = {
    "data": {
        "symbol": "BTC-USD",
        "start_date": "2020-01-01",
        "end_date": "2024-01-01",
        "interval": "1h",
        "window_size": 24,
        "test_split": 0.15
    },
    "environment": {
        "initial_cash": 10000.0,
        "fee": 0.001,
    },
    "agent": {
        "batch_size": 64,
        "episodes": 50,
        "target_update_freq": 1000,
        "learning_rate": 0.0001,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.99995
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
            print(f"Verzeichnis erstellt: {path}")
def run_benchmark_strategies(prices, initial_cash):
    print("\nBerechne Benchmarks...")
    bh_return = (prices[-1] / prices[0]) - 1
    bh_final = initial_cash * (1 + bh_return)
    random_returns = []
    for _ in range(10):
        cash = initial_cash
        coins = 0
        fee = CONFIG["environment"]["fee"]
        for i in range(len(prices)-1):
            action = np.random.choice([0, 1, 2])
            current_price = prices[i]
            if action == 1 and cash > 0:
                coins = (cash * (1 - fee)) / current_price
                cash = 0
            elif action == 2 and coins > 0:
                cash = (coins * current_price) * (1 - fee)
                coins = 0
        final = cash + (coins * prices[-1])
        random_returns.append((final - initial_cash) / initial_cash)
    random_avg_return = np.mean(random_returns)
    df_ma = pd.DataFrame({'Close': prices})
    df_ma['SMA20'] = df_ma['Close'].rolling(window=20).mean()
    df_ma['SMA50'] = df_ma['Close'].rolling(window=50).mean()
    cash, coins = initial_cash, 0
    fee = CONFIG["environment"]["fee"]
    for i in range(50, len(prices)-1):
        price = prices[i]
        sma20 = df_ma['SMA20'].iloc[i]
        sma50 = df_ma['SMA50'].iloc[i]
        if sma20 > sma50 and cash > 0:
            coins = (cash * (1 - fee)) / price
            cash = 0
        elif sma20 < sma50 and coins > 0:
            cash = (coins * price) * (1 - fee)
            coins = 0
    ma_final = cash + (coins * prices[-1])
    ma_return = (ma_final - initial_cash) / initial_cash
    return bh_return, random_avg_return, ma_return
def save_checkpoint(agent, episode, portfolio_value, is_best=False):
    filename = f"checkpoint_ep{episode}.pth"
    if is_best:
        filename = "best_model.pth"
    path = os.path.join(CONFIG["paths"]["models"], filename)
    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
        'portfolio_value': portfolio_value,
        'config': CONFIG
    }
    torch.save(checkpoint, path)
    if is_best:
        print(f" -> NEUES BESTES MODELL GESPEICHERT: ${portfolio_value:.2f}")
def train_enhanced_bot():
    ensure_directories()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starte Data Loader...")
    loader = DataLoader(
        symbol=CONFIG['data']['symbol'],
        start_date=CONFIG['data']['start_date'],
        end_date=CONFIG['data']['end_date'],
        interval=CONFIG['data']['interval'],
        test_split=CONFIG['data']['test_split']
    )
    train_df, test_df = loader.prepare_data()
    if train_df is None:
        print("KRITISCHER FEHLER: Keine Daten geladen.")
        return
    feature_count = len(train_df.columns)
    window_size = CONFIG['data']['window_size']
    input_dim = window_size * feature_count
    print(f"Feature Engineering abgeschlossen.")
    print(f" -> Features pro Step: {feature_count}")
    print(f" -> NN Input Dimension: {input_dim}")
    env = TradingEnvironment(
        df=train_df,
        original_prices=loader.original_prices_train,
        window_size=window_size,
        initial_cash=CONFIG['environment']['initial_cash'],
        fee=CONFIG['environment']['fee']
    )
    agent = Agent(
        state_size=feature_count,
        action_size=3,
        window_size=window_size
    )
    agent.batch_size = CONFIG['agent']['batch_size']
    agent.epsilon = CONFIG['agent']['epsilon_start']
    agent.epsilon_min = CONFIG['agent']['epsilon_min']
    agent.epsilon_decay = CONFIG['agent']['epsilon_decay']
    agent.learning_rate = CONFIG['agent']['learning_rate']
    print("\n" + "="*60)
    print(f" STARTING DUELING DOUBLE DQN TRAINING")
    print(f" Episoden: {CONFIG['agent']['episodes']} | Device: {agent.device}")
    print("="*60)
    best_portfolio = 0
    total_steps_global = 0
    for e in range(1, CONFIG['agent']['episodes'] + 1):
        state = env.reset()
        state = np.reshape(state, [1, input_dim])
        done = False
        episode_profit = 0
        pbar = tqdm(total=len(train_df), desc=f"Ep {e}/{CONFIG['agent']['episodes']}", unit="step")
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, input_dim])
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            if total_steps_global % CONFIG['agent']['target_update_freq'] == 0:
                agent.update_target_network()
            state = next_state
            total_steps_global += 1
            episode_profit = info['profit']
            pbar.set_postfix({
                "Epsilon": f"{agent.epsilon:.2f}",
                "Portfolio": f"${info['portfolio_value']:.0f}"
            })
            pbar.update(1)
        pbar.close()
        final_value = env.portfolio_value
        if final_value > best_portfolio:
            best_portfolio = final_value
            save_checkpoint(agent, e, final_value, is_best=True)
        if e % 5 == 0:
            save_checkpoint(agent, e, final_value, is_best=False)
    print("\nTraining abgeschlossen.")
    print("\n" + "="*60)
    print(" FINAL EVALUATION (TEST DATA)")
    print("="*60)
    test_env = TradingEnvironment(
        df=test_df,
        original_prices=loader.original_prices_test,
        window_size=window_size,
        initial_cash=CONFIG['environment']['initial_cash'],
        fee=CONFIG['environment']['fee']
    )
    state = test_env.reset()
    state = np.reshape(state, [1, input_dim])
    done = False
    agent.is_eval = True
    while not done:
        action = agent.act(state)
        next_state, _, done, info = test_env.step(action)
        state = np.reshape(next_state, [1, input_dim])
    bot_final = info['portfolio_value']
    bot_return = (bot_final - CONFIG['environment']['initial_cash']) / CONFIG['environment']['initial_cash']
    bh_ret, rand_ret, ma_ret = run_benchmark_strategies(
        loader.original_prices_test,
        CONFIG['environment']['initial_cash']
    )
    print("\n" + "
    print(f" RESULTS SUMMARY ({CONFIG['data']['symbol']})")
    print("
    print(f"{'STRATEGY':<25} | {'RETURN':<10} | {'FINAL BALANCE'}")
    print("-" * 50)
    print(f"{'DUELING DDQN (Bot)':<25} | {bot_return*100:>+8.2f}% | ${bot_final:.2f}")
    print(f"{'Buy & Hold':<25} | {bh_ret*100:>+8.2f}% | ${CONFIG['environment']['initial_cash']*(1+bh_ret):.2f}")
    print(f"{'MA Crossover (20/50)':<25} | {ma_ret*100:>+8.2f}% | ${CONFIG['environment']['initial_cash']*(1+ma_ret):.2f}")
    print(f"{'Random Trading':<25} | {rand_ret*100:>+8.2f}% | ${CONFIG['environment']['initial_cash']*(1+rand_ret):.2f}")
    print("
    save_checkpoint(agent, CONFIG['agent']['episodes'], bot_final, is_best=False)
if __name__ == "__main__":
    try:
        train_enhanced_bot()
    except KeyboardInterrupt:
        print("\nTraining durch Benutzer abgebrochen.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()