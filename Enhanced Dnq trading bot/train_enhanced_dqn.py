import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm  # Fortschrittsbalken

# --- PFAD-FIX START ---
# Wir stellen sicher, dass Python unsere Module im 'utils' Ordner findet
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- IMPORTS UNSERER MODULE ---
from utils.data_loader import DataLoader
from utils.agent import Agent  # Der Dueling Double DQN Agent
from utils.environment import TradingEnvironment # Die Markt-Simulation
from utils.indicators import TechnicalIndicators # Für Benchmark-Berechnungen

# --- KONFIGURATION & HYPERPARAMETER ---
CONFIG = {
    "symbol": "BTC-USD",
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "interval": "1h",             # Stunden-Daten für mehr Details
    "window_size": 24,            # Der Bot sieht 24 Stunden zurück
    "episodes": 50,               # Anzahl der Trainings-Durchläufe
    "batch_size": 64,             # Wie viele Erinnerungen pro Lernschritt
    "target_update_freq": 1000,   # Alle 1000 Schritte das "Sicherheits-Netz" updaten
    "initial_cash": 10000.0,      # Startkapital
    "model_save_path": "models/", # Wo speichern wir den Bot?
}

def ensure_directories():
    """Erstellt notwendige Ordner, falls sie fehlen."""
    if not os.path.exists(CONFIG["model_save_path"]):
        os.makedirs(CONFIG["model_save_path"])

def run_benchmarks(prices, initial_cash):
    """
    Vergleichsstrategien, um zu sehen, ob der Bot wirklich gut ist.
    """
    # 1. Buy & Hold (Einfach am Anfang kaufen und halten)
    bh_return = (prices[-1] / prices[0]) - 1
    
    # 2. Random Trading (Durchschnitt aus 10 zufälligen Versuchen)
    random_profits = []
    for _ in range(10):
        cash = initial_cash
        coins = 0
        for i in range(len(prices)-1):
            action = np.random.choice([0, 1, 2]) # 0=Hold, 1=Buy, 2=Sell
            if action == 1 and cash > 0:
                coins = cash / prices[i] * 0.999 # 0.1% Gebühr simuliert
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
    
    # --- 1. DATEN LADEN & VORBEREITEN ---
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Lade Daten für {CONFIG['symbol']}...")
    loader = DataLoader(
        symbol=CONFIG['symbol'],
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        interval=CONFIG['interval']
    )
    
    # Hier passiert die Magie: Ichimoku, Whale-Signale, Zyklus-Features werden gebaut
    train_df, test_df = loader.prepare_data()
    
    if train_df is None:
        print("Fehler: Keine Daten geladen. Abbruch.")
        return

    # Ermitteln, wie viele Features wir jetzt haben (sollten ca. 24-26 sein)
    feature_count = len(train_df.columns)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Feature-Engineering fertig.")
    print(f" -> Der Agent sieht {feature_count} Indikatoren pro Zeitschritt.")
    print(f" -> Input Size für NN: {CONFIG['window_size']} x {feature_count} = {CONFIG['window_size'] * feature_count}")

    # --- 2. ENVIRONMENT & AGENT SETUP ---
    env = TradingEnvironment(train_df, loader.original_prices_train, window_size=CONFIG['window_size'])
    
    # Initialisiere unseren Dueling Double DQN Agenten
    agent = Agent(
        state_size=feature_count,
        action_size=3, # Buy, Sell, Hold
        window_size=CONFIG['window_size']
    )

    # Globale Stats
    total_steps = 0
    best_portfolio = 0

    print("\n" + "="*50)
    print(f" START TRAINING ({CONFIG['episodes']} Episoden)")
    print("="*50)

    # --- 3. TRAINING LOOP ---
    for e in range(1, CONFIG['episodes'] + 1):
        # Reset Env
        state = env.reset()
        # Form anpassen für NN: [1, Input_Dim]
        state = np.reshape(state, [1, CONFIG['window_size'] * feature_count])
        
        done = False
        episode_profit = 0
        
        # tqdm Progressbar für schöne Anzeige
        pbar = tqdm(total=len(train_df), desc=f"Ep {e}/{CONFIG['episodes']}", unit="step")
        
        while not done:
            # A. Handeln
            action = agent.act(state)
            
            # B. Ausführen
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, CONFIG['window_size'] * feature_count])
            
            # C. Merken (Experience Replay)
            agent.remember(state, action, reward, next_state, done)
            
            # D. Lernen (Training auf Batch)
            agent.replay()
            
            # E. Target Network Update (Double DQN Feature)
            # Das verhindert, dass der Bot "zu optimistisch" wird
            if total_steps % CONFIG['target_update_freq'] == 0:
                agent.update_target_network()
            
            state = next_state
            episode_profit = info.get('profit', 0)
            portfolio_value = info.get('portfolio_value', CONFIG['initial_cash'])
            total_steps += 1
            
            # Anzeige aktualisieren
            pbar.set_postfix({
                "Epsilon": f"{agent.epsilon:.2f}", 
                "Portfolio": f"${portfolio_value:.0f}"
            })
            pbar.update(1)

        pbar.close()
        
        # --- CHECKPOINTING ---
        # Speichern, wenn wir ein neues Allzeithoch im Portfolio erreicht haben
        current_final_portfolio = env.portfolio_value
        if current_final_portfolio > best_portfolio:
            best_portfolio = current_final_portfolio
            torch.save(agent.policy_net.state_dict(), f"{CONFIG['model_save_path']}best_model.pth")
            tqdm.write(f" -> Neues Best-Modell gespeichert! Portfolio: ${best_portfolio:.2f}")

        # Regelmäßiges Backup alle 5 Episoden
        if e % 5 == 0:
            torch.save(agent.policy_net.state_dict(), f"{CONFIG['model_save_path']}checkpoint_ep{e}.pth")

    print("\n" + "="*50)
    print(" TRAINING ABGESCHLOSSEN. Starte Evaluation...")
    print("="*50)

    # --- 4. EVALUATION (TEST DATA) ---
    # Agent in den "Ernst-Modus" schalten (Kein Zufall mehr)
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

    # --- 5. BENCHMARKS BERECHNEN ---
    bh_ret, rand_ret = run_benchmarks(loader.original_prices_test, CONFIG['initial_cash'])

    # --- 6. ERGEBNIS-BERICHT ---
    print("\n" + "#"*40)
    print(f" ERGEBNISSE AUF TESTDATEN ({CONFIG['symbol']})")
    print("#"*40)
    print(f"{'Strategie':<25} | {'Return':<10} | {'Endkapital ($)'}")
    print("-" * 60)
    print(f"{'DUELING DDQN (Dein Bot)':<25} | {bot_return*100:>+8.2f}% | ${final_portfolio:.2f}")
    print(f"{'Buy & Hold':<25} | {bh_ret*100:>+8.2f}% | ${CONFIG['initial_cash']*(1+bh_ret):.2f}")
    print(f"{'Random Trading (Avg)':<25} | {rand_ret*100:>+8.2f}% | ${CONFIG['initial_cash']*(1+rand_ret):.2f}")
    print("#"*40)

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining durch Benutzer abgebrochen.")
    except Exception as e:
        print(f"\nKritischer Fehler: {e}")
        import traceback
        traceback.print_exc()