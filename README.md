# Reinforcement-Learning-Trading-Bot-RLTrader-
Entwicklung eines autonomen Trading-Bots, der mit Reinforcement Learning (RL) lernt, profitable Handelsentscheidungen auf Basis historischer und aktueller Marktdaten zu treffen.

---

## Vorgehen / Projektplan

---

### **Phase 1 – Datenbeschaffung & Aufbereitung**
Echte Marktdaten (z. B. Bitcoin, Ethereum) sammeln, bereinigen und als Trainingsgrundlage speichern.
**Output:** `data/processed/BTC-USD.csv
### **Phase 2 – Trading-Umgebung (OpenAI Gym)**
Erstellen einer Simulationsumgebung, in der der RL-Agent handeln kann.
**Output:** Eine funktionierende Gym-Umgebung mit Reward-Logik.
### **Phase 3 – RL-Agent (z. B. DQN oder PPO)**
Der Agent soll durch Training lernen, profitabel zu handeln.
**Output:** Gespeichertes Modell in `models/dqn_btc_model.zip`
### **Phase 4 – Evaluation & Visualisierung**
Die Performance des Bots messen und mit Baseline-Strategien vergleichen.
**Output:** Plots und Metriken in `results/`
### **Phase 5 – Dokumentation & Präsentation**
Abschlussbericht, Code-Dokumentation und Präsentationsslides.
**Output:** Dokumentation im Ordner `docs/` und finale Slides.

---

### **Phase 1 – Datenbeschaffung & Aufbereitung**
#### Kryptowährung Auswahl (Start: `BTC-USD`, optional später `ETH-USD`, `BNB-USD`)
#### Datenquelle (Laden historischer Daten über `yfinance`)
#### Pre-Datenaufarbeitung
#### Datenaufarbeitung (fehlende Werte, unnötige Spalten)
#### Post-Datenaufarbeitung

---

### **Phase 2 – Trading-Umgebung (OpenAI Gym)**

**Aufgaben:**
- Implementierung von `TradingEnv` in `src/env/trading_env.py`
- Definition der Aktionen (Buy / Hold / Sell)
- Definition der Reward-Funktion (Gewinn, Verlust, Transaktionskosten)
- Test der Umgebung mit Zufallsaktionen (`env.step()`)


---

### **Phase 3 – RL-Agent (z. B. DQN oder PPO)**

**Aufgaben:**
- Implementierung eines RL-Modells in `src/agents/dqn_agent.py`
- Nutzung von `stable-baselines3`
- Verbindung der Umgebung mit dem Agenten
- Training über mehrere Episoden


---

### **Phase 4 – Evaluation & Visualisierung**

**Aufgaben:**
- Rewards, Balance, Trades visualisieren (`matplotlib`)
- Vergleich mit Buy & Hold
- Berechnung von Metriken (Total Return, Sharpe Ratio)


---

### **Phase 5 – Dokumentation & Präsentation**


**Aufgaben:**
- README und Projektdokumentation ergänzen
- Code kommentieren und auf GitHub sauber strukturieren
- Präsentation vorbereiten


---

## Struktur im Repository
