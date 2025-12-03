# Reinforcement-Learning-Trading-Bot-RLTrader-
Entwicklung eines autonomen Trading-Bots, der mit Reinforcement Learning (RL) lernt, profitable Handelsentscheidungen auf Basis historischer und aktueller Marktdaten zu treffen.

---

## Vorgehen / Projektplan

---

### **Phase 1 – Datenbeschaffung & Aufbereitung**
#### Kryptowährung Auswahl (Start: `BTC-USD`, optional später `ETH-USD`, `BNB-USD`)
Wir haben uns für den bekannten Bitcoin entschieden weil
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
