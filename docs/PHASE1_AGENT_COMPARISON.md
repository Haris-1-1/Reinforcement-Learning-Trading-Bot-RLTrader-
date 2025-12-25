# Phase 1: Agenten-Vergleich

> Systematische Evaluierung von drei Reinforcement Learning Strategien unter realistischen Marktbedingungen

---

## Überblick

In Phase 1 wurden drei fundamentale RL-Ansätze gegeneinander getestet, um den geeignetsten Kandidaten für die weitere Optimierung zu identifizieren. Alle Agenten trainierten unter identischen Bedingungen in einer professionellen Simulationsumgebung mit realistischen Marktfriktionen.

---

## Die Simulationsumgebung: Advanced Trading Environment

Die Umgebung bildet reale Marktmechanismen ab und ist das Fundament für valides Training.

### Kernkomponenten

**1. Zustandsraum (State Space)**

Der Agent erhält zwei parallele Datenströme:

**Normalisierte Features** (für das neuronale Netz):
- OHLCV-Preisdaten
- Technische Indikatoren (MA5, MA20, MA50, RSI, MACD, Bollinger Bands, ATR)
- Alle Werte werden per Z-Score-Normalisierung skaliert

**Originale Preise** (für die Abrechnung):
- Unveränderte Schlusskurse für präzise Gewinn/Verlust-Berechnung
- Exakte Gebührenberechnung in USD

Diese Trennung ist kritisch: Der Agent lernt auf statistisch optimierten Daten, während die finanzielle Abrechnung unverfälscht bleibt.

**2. Marktfriktionen** (realitätsnahe Constraints)

| Constraint | Implementierung | Auswirkung |
|------------|----------------|------------|
| **Trading Fees** | Maker: 0.1%, Taker: 0.2% | Direkter Kostenabzug bei jedem Trade |
| **Slippage** | 0.1% Preisimpakt | Reduziert den tatsächlichen Ausführungspreis |
| **Frequency Penalty** | -0.00005 pro Trade | Bestraft exzessives Handeln |
| **Execution Delay** | Konfigurierbare Latenz | Simuliert asynchrone Ausführung |

Diese Constraints zwingen den Agenten, nur Trades mit hohem Erwartungswert einzugehen.

---

## Die drei Kandidaten

### 1. Q-Learning (Tabellarisch)

**Konzept:**
Q-Learning speichert Erfolgswahrscheinlichkeiten in einer Tabelle. Für jede Marktsituation (z.B. RSI=65, MACD=positiv) wird der erwartete Gewinn für jede Aktion (Kaufen/Verkaufen/Halten) gespeichert.

**Das Problem: Diskretisierung**

Finanzmärkte sind kontinuierlich (RSI kann 62.53 sein), aber Q-Learning benötigt diskrete Zustände. Lösung: "Binning" - die Werte werden in Intervalle eingeteilt (z.B. 15 Stufen).

**Warum es scheiterte:**

Je mehr Indikatoren hinzugefügt werden, desto explosiver wächst die Tabelle:
- 5 Indikatoren mit je 15 Stufen = 15^5 = 759,375 Zustände
- 10 Indikatoren = 15^10 = 576 Milliarden Zustände

Der "Fluch der Dimensionalität" macht effizientes Lernen unmöglich.

**Code-Snippet:** [rl_trading_bot_verschiedene_Versionen/agents/q_learning_agent.py](../rl_trading_bot_verschiedene_Versionen/agents/q_learning_agent.py)

**Experiment 1 Ergebnis:**
- Training: +256% (massives Overfitting)
- Test: -25.74% (komplettes Versagen bei Generalisierung)

---

### 2. DQN (Deep Q-Network)

**Konzept:**
Anstatt einer Tabelle nutzt DQN ein neuronales Netz, das kontinuierliche Werte verarbeiten kann. Das Netz "lernt" eine Funktion, die für jeden Marktzustand die beste Aktion vorhersagt.

**Die Architektur:**

```
Input (normalisierte Features)
    ↓
Dense Layer (128 Neuronen) + ReLU + Dropout
    ↓
Dense Layer (128 Neuronen) + ReLU + Dropout
    ↓
Dense Layer (64 Neuronen) + ReLU
    ↓
Output (3 Q-Werte: Hold, Buy, Sell)
```

**Zwei kritische Mechanismen:**

**Experience Replay:**
Der Agent speichert vergangene Erfahrungen (state, action, reward, next_state) in einem Puffer. Beim Training wird zufällig aus diesem Puffer gesampelt, um schädliche zeitliche Korrelationen aufzubrechen.

Warum wichtig? Ohne Replay würde der Agent nur von aufeinanderfolgenden Zeitschritten lernen (Montag → Dienstag → Mittwoch). Das führt zu Overfitting auf bestimmte Marktphasen. Durch zufälliges Sampling sieht das Netz viele verschiedene Situationen gemischt - wie flashcards beim Lernen.

**Target Network:**
Zwei identische Netzwerke werden parallel genutzt:
- **Policy Network**: Macht die aktuellen Entscheidungen
- **Target Network**: Bewertet diese Entscheidungen (wird langsamer aktualisiert)

Warum wichtig? Stell dir vor, du lernst für eine Prüfung, aber die Musterlösung ändert sich jede Sekunde. Du würdest nie konvergieren. Das Target Network ist wie eine "stabile Musterlösung", die nur alle 1000 Schritte aktualisiert wird.

**Code-Snippet:** [rl_trading_bot_verschiedene_Versionen/agents/dqn_agent.py](../rl_trading_bot_verschiedene_Versionen/agents/dqn_agent.py)

**Experiment 1 Ergebnis:**
- Training: Stabil
- Test: -23.90% (besser als Q-Learning, aber noch kein Gewinn)

**Experiment 11 (optimiert):**
- Test: +25.06% Outperformance gegenüber Buy & Hold

---

### 3. PPO (Proximal Policy Optimization)

**Konzept:**
PPO verfolgt einen fundamental anderen Ansatz. Anstatt den Wert von Aktionen zu schätzen (wie DQN), lernt PPO direkt eine Strategie (Policy).

**Der Unterschied:**
- **DQN**: "Kaufen hat Q-Wert 0.8, Halten hat 0.3" → wähle Kaufen
- **PPO**: "70% Wahrscheinlichkeit für Kaufen, 20% für Halten, 10% für Verkaufen" → sample aus Verteilung

**Die "Clipped Objective Function":**

PPO hat einen eingebauten Sicherheitsmechanismus. Stell dir vor, du hast zufällig einen großen Gewinn gemacht. Ein normaler Agent würde denken "Das war genial!" und seine Strategie radikal ändern. PPO sagt: "Moment, das könnte Glück gewesen sein. Ich ändere mich nur in kleinen Schritten."

Mathematisch wird verhindert, dass die neue Policy zu stark von der alten abweicht (Clipping). Das macht PPO besonders robust gegen das stochastische Rauschen von Finanzmärkten.

**Code-Snippet:** [rl_trading_bot_verschiedene_Versionen/agents/ppo_agent.py](../rl_trading_bot_verschiedene_Versionen/agents/ppo_agent.py)

**Experiment 1 Ergebnis:**
- Test: +7.16% (einziger profitabler Agent in Experiment 1)
- Outperformance gegenüber Buy & Hold: +32.72%
- Bewies robustes Risikomanagement in Bärenmarkt

---

## Experimentreihe (Experimente 1-12)

### Experiment 1: Initiale Evaluation

**Setup:**
- Zeitraum: 01.01.2024 - 15.12.2025 (1h-Intervall)
- Trainingsschritte: 500,000
- Test-Split: 20%

**Ergebnisse:**

| Agent | Training | Test | Interpretation |
|-------|----------|------|----------------|
| **PPO** | Stabil | **+7.16%** | Testsieger: Robuste Strategie |
| **Q-Learning** | +256% | -25.74% | Massives Overfitting |
| **DQN** | Stabil | -23.90% | Potential, aber unoptimiert |
| **Buy & Hold** | - | -25.56% | Markt-Baseline (Bärenmarkt) |

**Erkenntnis:** PPO zeigte kurzfristige Stabilität, DQN das höchste Optimierungspotenzial.

### Weitere Erkenntnisse (Exp. 2-12)

**Experiment 2: Kostenanalyse**
- Gebühren entfernt → exzessives Overtrading → massive Nettoverluste
- Fazit: Transaktionskosten-Modellierung ist essentiell

**Experiment 10-11: DQN-Optimierung**
- Reduktion der Trade-Strafen
- Anpassung des Train-Test-Splits (85/15)
- Ergebnis: **DQN erzielte +25.06% Outperformance**

**Das "Passivitäts-Problem":**
Mehrere Experimente (5, 6, 11) zeigten: Bei zu hohen Strafen oder unbekannten Marktphasen verweigerten Agenten komplett das Handeln ("Angst-Phänomen"). Der Agent lernt: "Wenn ich nichts tue, kann ich nichts verlieren."

---

## Fazit Phase 1

### Gewinner: DQN

Obwohl PPO in Experiment 1 der einzige profitable Agent war, zeigte DQN nach Optimierung (Exp. 11) das höchste Potenzial für signifikante Renditen.

### Schlüsselerkenntnisse

1. **Standard-RL-Modelle sind hochsensibel** auf Hyperparameter und Marktrauschen
2. **Feature-Wichtung ist kritisch** - mehr Indikatoren ≠ besser
3. **Gebühren sind der "Boss-Fight"** - ohne intelligente Trade-Selektion wird Kapital vernichtet
4. **Generalisierung > Training Performance** - hohe Trainingsgewinne bedeuten oft Overfitting

### Nächste Schritte → Phase 2

DQN wurde als Basis-Agent für die Optimierungsphase ausgewählt. Die Herausforderung:
- Overfitting bekämpfen
- Zeitliche Zusammenhänge verstehen (Trend-Erkennung)
- Institutionelle Aktivitäten erkennen ("Smart Money")

---

## Code-Referenzen

- **Q-Learning Agent**: [agents/q_learning_agent.py](../rl_trading_bot_verschiedene_Versionen/agents/q_learning_agent.py)
- **DQN Agent**: [agents/dqn_agent.py](../rl_trading_bot_verschiedene_Versionen/agents/dqn_agent.py)
- **PPO Agent**: [agents/ppo_agent.py](../rl_trading_bot_verschiedene_Versionen/agents/ppo_agent.py)
- **Trading Environment**: [env/advanced_trading_env.py](../rl_trading_bot_verschiedene_Versionen/env/advanced_trading_env.py)
- **Training Scripts**:
  - [train_qlearning.py](../rl_trading_bot_verschiedene_Versionen/train_qlearning.py)
  - [train_dqn.py](../rl_trading_bot_verschiedene_Versionen/train_dqn.py)
  - [train_ppo.py](../rl_trading_bot_verschiedene_Versionen/train_ppo.py)

---

[← Zurück zum Hauptmenü](../README.md) | [Weiter zu Phase 2 →](PHASE2_ENHANCED_DQN.md)
