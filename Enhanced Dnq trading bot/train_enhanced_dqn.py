import os
import sys
import numpy as np
import pandas as pd
import torch
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
    "symbol": "BTC-USD",
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "interval": "1h",
    "window_size": 24,
    "episodes": 50,
    "batch_size": 64,
    "target_update_freq": 1000,
    "initial_cash": 10000.0,
    "model_save_path": "models/",
}
def ensure_directories():
    if not os.path.exists(CONFIG["model_save_path"]):
        os.makedirs(CONFIG["model_save_path"])
def run_benchmarks(prices, initial_cash):
    bh_return = (prices[-1] / prices[0]) - 1
    random_profits = []
    for _ in range(10):
        cash = initial_cash
        coins = 0
        for i in range(len(prices)-1):
            action = np.random.choice([0, 1, 2])
            if action == 1 and cash > 0:
                coins = cash / prices[i] * 0.999
                cash = 0
            elif action == 2 and coins > 0:
                cash = coins * prices[i] * 0.999
                coins = 0
        final_val = cash + (coins * prices[-1])
        random_profits.append((final_val - initial_cash) / initial_cash)
    avg_random_return = np.mean(random_profits)
    return bh_return, avg_random_return
def train():
    ensure_directories()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Lade Daten für {CONFIG['symbol']}...")
    loader = DataLoader(
        symbol=CONFIG['symbol'],
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        interval=CONFIG['interval']
    )
    train_df, test_df = loader.prepare_data()
    if train_df is None:
        print("Fehler: Keine Daten geladen. Abbruch.")
        return
    feature_count = len(train_df.columns)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Feature-Engineering fertig.")
    print(f" -> Der Agent sieht {feature_count} Indikatoren pro Zeitschritt.")
    print(f" -> Input Size für NN: {CONFIG['window_size']} x {feature_count} = {CONFIG['window_size'] * feature_count}")
    env = TradingEnvironment(train_df, loader.original_prices_train, window_size=CONFIG['window_size'])
    agent = Agent(
        state_size=feature_count,
        action_size=3,
        window_size=CONFIG['window_size']
    )
    total_steps = 0
    best_portfolio = 0
    print("\n" + "="*50)
    print(f" START TRAINING ({CONFIG['episodes']} Episoden)")
    print("="*50)
    for e in range(1, CONFIG['episodes'] + 1):
        state = env.reset()
        state = np.reshape(state, [1, CONFIG['window_size'] * feature_count])
        done = False
        episode_profit = 0
        pbar = tqdm(total=len(train_df), desc=f"Ep {e}/{CONFIG['episodes']}", unit="step")
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, CONFIG['window_size'] * feature_count])
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            if total_steps % CONFIG['target_update_freq'] == 0:
                agent.update_target_network()
            state = next_state
            episode_profit = info.get('profit', 0)
            portfolio_value = info.get('portfolio_value', CONFIG['initial_cash'])
            total_steps += 1
            pbar.set_postfix({
                "Epsilon": f"{agent.epsilon:.2f}",
                "Portfolio": f"${portfolio_value:.0f}"
            })
            pbar.update(1)
        pbar.close()
        current_final_portfolio = env.portfolio_value
        if current_final_portfolio > best_portfolio:
            best_portfolio = current_final_portfolio
            torch.save(agent.policy_net.state_dict(), f"{CONFIG['model_save_path']}best_model.pth")
            tqdm.write(f" -> Neues Best-Modell gespeichert! Portfolio: ${best_portfolio:.2f}")
        if e % 5 == 0:
            torch.save(agent.policy_net.state_dict(), f"{CONFIG['model_save_path']}checkpoint_ep{e}.pth")
    print("\n" + "="*50)
    print(" TRAINING ABGESCHLOSSEN. Starte Evaluation...")
    print("="*50)
    agent.is_eval = True
    test_env = TradingEnvironment(test_df, loader.original_prices_test, window_size=CONFIG['window_size'])
    state = test_env.reset()
    state = np.reshape(state, [1, CONFIG['window_size'] * feature_count])
    done = False
    while not done:
        action = agent.act(state)
        next_state, _, done, info = test_env.step(action)
        state = np.reshape(next_state, [1, CONFIG['window_size'] * feature_count])
    final_portfolio = info['portfolio_value']
    bot_return = (final_portfolio - CONFIG['initial_cash']) / CONFIG['initial_cash']
    bh_ret, rand_ret = run_benchmarks(loader.original_prices_test, CONFIG['initial_cash'])
    print("\n" + "
    print(f" ERGEBNISSE AUF TESTDATEN ({CONFIG['symbol']})")
    print("
    print(f"{'Strategie':<25} | {'Return':<10} | {'Endkapital ($)'}")
    print("-" * 60)
    print(f"{'DUELING DDQN (Dein Bot)':<25} | {bot_return*100:>+8.2f}% | ${final_portfolio:.2f}")
    print(f"{'Buy & Hold':<25} | {bh_ret*100:>+8.2f}% | ${CONFIG['initial_cash']*(1+bh_ret):.2f}")
    print(f"{'Random Trading (Avg)':<25} | {rand_ret*100:>+8.2f}% | ${CONFIG['initial_cash']*(1+rand_ret):.2f}")
    print("
if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining durch Benutzer abgebrochen.")
    except Exception as e:
        print(f"\nKritischer Fehler: {e}")
        import traceback
        traceback.print_exc()