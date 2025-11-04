# Reinforcement-Learning-Trading-Bot-RLTrader-
Entwicklung eines autonomen Trading-Bots, der mit Reinforcement Learning (RL) lernt, profitable Handelsentscheidungen auf Basis historischer und aktueller Marktdaten zu treffen.

---

## Vorgehen / Projektplan

---

### **Phase 1 – Datenbeschaffung & Aufbereitung**
Echte Marktdaten (z. B. Bitcoin, Ethereum) sammeln, bereinigen und als Trainingsgrundlage speichern.
### **Phase 2 – Trading-Umgebung (OpenAI Gym)**
Erstellen einer Simulationsumgebung, in der der RL-Agent handeln kann.
### **Phase 3 – RL-Agent (z. B. DQN oder PPO)**
Der Agent soll durch Training lernen, profitabel zu handeln.
### **Phase 4 – Evaluation & Visualisierung**
Die Performance des Bots messen und mit Baseline-Strategien vergleichen.
### **Phase 5 – Dokumentation & Präsentation**
Abschlussbericht, Code-Dokumentation und Präsentationsslides.



**Phase 1 – Datenbeschaffung & Aufbereitung**
**Aufgaben:**
- Auswahl der Coins (Start: `BTC-USD`, optional später `ETH-USD`, `BNB-USD`)
- Laden historischer Daten über `yfinance`
- Bereinigung (fehlende Werte, unnötige Spalten)
- Speichern als CSV in `data/processed/`

**Output:**  
`data/processed/BTC-USD.csv`

---

### **Phase 2 – Trading-Umgebung (OpenAI Gym)**
Erstellen einer Simulationsumgebung, in der der RL-Agent handeln kann.

**Aufgaben:**
- Implementierung von `TradingEnv` in `src/env/trading_env.py`
- Definition der Aktionen (Buy / Hold / Sell)
- Definition der Reward-Funktion (Gewinn, Verlust, Transaktionskosten)
- Test der Umgebung mit Zufallsaktionen (`env.step()`)

**Output:**  
Eine funktionierende Gym-Umgebung mit Reward-Logik.

---

### **Phase 3 – RL-Agent (z. B. DQN oder PPO)**
**Ziel:**  
Der Agent soll durch Training lernen, profitabel zu handeln.

**Aufgaben:**
- Implementierung eines RL-Modells in `src/agents/dqn_agent.py`
- Nutzung von `stable-baselines3`
- Verbindung der Umgebung mit dem Agenten
- Training über mehrere Episoden

**Output:**  
Gespeichertes Modell in `models/dqn_btc_model.zip`

---

### **Phase 4 – Evaluation & Visualisierung**
**Ziel:**  
Die Performance des Bots messen und mit Baseline-Strategien vergleichen.

**Aufgaben:**
- Rewards, Balance, Trades visualisieren (`matplotlib`)
- Vergleich mit Buy & Hold
- Berechnung von Metriken (Total Return, Sharpe Ratio)

**Output:**  
Plots und Metriken in `results/`

---

### **Phase 5 – Dokumentation & Präsentation**
**Ziel:**  
Abschlussbericht, Code-Dokumentation und Präsentationsslides.

**Aufgaben:**
- README und Projektdokumentation ergänzen
- Code kommentieren und auf GitHub sauber strukturieren
- Präsentation vorbereiten

**Output:**  
Dokumentation im Ordner `docs/` und finale Slides.

---

## Struktur im Repository
