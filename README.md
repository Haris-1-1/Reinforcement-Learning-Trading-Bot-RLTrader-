# RL Trading Bot - Reinforcement Learning Trading Agent

> Entwicklung eines autonomen Trading-Bots, der mit Deep Reinforcement Learning profitable Handelsentscheidungen auf Basis historischer Marktdaten trifft.

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Projektübersicht

Dieses Projekt dokumentiert die experimentelle Entwicklung eines Trading-Agenten, der mittels **Deep Reinforcement Learning (DQN)** profitable Entscheidungen unter Berücksichtigung realer Marktfriktionen (Gebühren, Slippage) trifft.

### Finale Ergebnisse (Out-of-Sample Test)

| Strategie | Return | Performance |
|-----------|--------|-------------|
| **Enhanced DQN** | **+26.23%** | Gewinn in fallendem Markt |
| Buy & Hold | -7.17% | Marktverlust |
| MA Crossover | -5.25% | Technische Strategie |
| Random Trading | -56.29% | Baseline (Gebührenvernichtung) |

**Outperformance: +33% Alpha gegenüber dem Markt**

---

## Projektphasen

Das Projekt durchlief drei strategische Phasen:

### Phase 1: Agenten-Vergleich (3 RL-Strategien)
Systematischer Vergleich von drei Reinforcement Learning Ansätzen unter realistischen Marktbedingungen.

**[Zur detaillierten Dokumentation →](docs/PHASE1_AGENT_COMPARISON.md)**

**Getestete Algorithmen:**
- Q-Learning (Tabellarisch)
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)

**Ergebnis:** DQN zeigte das höchste Potenzial für signifikante Renditen.

---

### Phase 2: Enhanced "Smart Money" DQN
Strategische Neuausrichtung mit fortgeschrittenem Feature Engineering und institutionellem Tracking.

**[Zur detaillierten Dokumentation →](docs/PHASE2_ENHANCED_DQN.md)**

**Key Features:**
- Windowed Deep Q-Learning (zeitlicher Kontext)
- Smart Money Indikatoren (Ichimoku, OBV, A/D Line)
- Cyclical Time Encoding (sin/cos Zyklen)
- Hybrid-Komponente (Supervised "Intuition")

---

### Phase 3: Finales System - Dueling Double DQN
Höchste Evolutionsstufe mit modernster RL-Architektur und 42 technischen Indikatoren.

**[Zur detaillierten Dokumentation →](docs/PHASE3_FINAL_SYSTEM.md)**

**Architektur-Highlights:**
- Dueling Network (Value + Advantage Streams)
- Double DQN (Policy + Target Networks)
- Action Masking (verhindert ungültige Aktionen)
- 42 Features für "Whale Tracking"

---

## Quick Start

### Installation

```bash
# Repository klonen
git clone https://github.com/Haris-1-1/Reinforcement-Learning-Trading-Bot-RLTrader-.git
cd Reinforcement-Learning-Trading-Bot-RLTrader-

# Dependencies installieren
pip install -r requirements.txt
```

### Training starten

```bash
# Finales System trainieren
cd Enhanced_DQN_(Final)
python train_enhanced_dqn.py
```

### Live Paper Trading

```bash
cd Enhanced_DQN_(Final)/Simulation
python start_live_trading.py
```

---

## Projektstruktur

```
Reinforcement-Learning-Trading-Bot-RLTrader-/
│
├── Demo/                                    # Phase 0: Initialer Proof-of-Concept
│
├── rl_trading_bot_verschiedene_Versionen/  # Phase 1: Agenten-Vergleich
│   ├── agents/                             # Q-Learning, DQN, PPO
│   ├── env/                                # Advanced Trading Environment
│   └── train_*.py                          # Training Scripts
│
├── Enhanced Dnq trading bot/               # Phase 2: Enhanced DQN (Zwischenstufe)
│
├── Enhanced_DQN_(Final)/                   # Phase 3: Finales System
│   ├── agents/                             # Dueling Double DQN
│   ├── env/                                # Gymnasium Environment
│   ├── utils/                              # Data Loader, Indicators, Visualizer
│   ├── Simulation/                         # Live Paper Trading
│   ├── models/                             # Gespeicherte Modelle
│   └── plots/                              # Visualisierungen
│
└── docs/                                   # Detaillierte Dokumentationen
    ├── PHASE1_AGENT_COMPARISON.md
    ├── PHASE2_ENHANCED_DQN.md
    └── PHASE3_FINAL_SYSTEM.md
```

---

## Technische Highlights

### 1. Smart Money Tracking
Der Bot erkennt institutionelle Aktivitäten durch:
- **On-Balance Volume (OBV)**: Kapitalzuflüsse vor Preisbewegungen
- **Money Flow Index (MFI)**: Smart Money Footprint
- **A/D Line**: Verdeckte Akkumulation
- **Volume Spikes**: Wendepunkte durch institutionelle Käufe

### 2. Cyclical Time Encoding
Zeitliche Muster werden durch **sin/cos-Transformation** auf einen Kreis projiziert:
- Markt-Öffnungszeiten
- Wochenend-Liquidität
- Zyklische Wiederholungen

### 3. Dueling Architecture
Das neuronale Netz trennt:
- **Value Stream**: "Wie gefährlich ist der Markt?"
- **Advantage Stream**: "Wie viel besser ist Kaufen vs. Halten?"

Dies ermöglicht schnelleres Lernen von Risikovermeidung in Bärenmärkten.

---

## Wichtige Erkenntnisse

1. **Informationsvorsprung entscheidend**: Die Erweiterung auf 42 Features (Smart Money Indikatoren) war kritisch für den Erfolg.

2. **Architektur matters**: Dueling DQN + Action Masking führte zu signifikant schnellerer und stabilerer Konvergenz.

3. **Gebühren als Lern-Signal**: Die Trade Frequency Penalty lehrte dem Agenten "Geduld" - er führt nur Trades mit hohem Erwartungswert aus.

4. **Robustheit in Bärenmärkten**: Der Agent erzielte +26% Gewinn während der Markt -7% verlor - Beweis für echtes "Alpha".

---

## Benchmarks & Validierung

Alle Tests erfolgten **Out-of-Sample** (Januar 2024 - November 2025) mit:
- 0.1% Trading Fees
- 0.05% Slippage
- Frequency Penalty

Vergleich gegen:
- Buy & Hold (passive Referenz)
- Random Agent (Gebühren-Baseline)
- MA Crossover 20/50 (technische Strategie)

---

## Zukünftige Erweiterungen

- **Multi-Asset Training**: Korrelationen zwischen Assets nutzen
- **Sentiment-Analyse**: NLP-basierte News-Integration
- **Transformer-Architekturen**: Attention-Mechanismen für längere zeitliche Abhängigkeiten

---

## Autoren

**Haris Salii & Fenlin Chirakal**

Reinforcement Learning Projekt - 2025

---

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert.

---

## Ressourcen
- [Phase 1: Agenten-Vergleich](docs/PHASE1_AGENT_COMPARISON.md)
- [Phase 2: Enhanced DQN](docs/PHASE2_ENHANCED_DQN.md)
- [Phase 3: Finales System](docs/PHASE3_FINAL_SYSTEM.md)
