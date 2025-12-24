# 🚀 Advanced Trading Bot - DRQN with Whale Tracking

Ein hochmoderner Trading Bot basierend auf **Deep Recurrent Q-Networks (DRQN)** mit fortgeschrittenen Features wie Whale Tracking, Invalid Action Masking und Multi-Timeframe Analyse.

## 📋 Übersicht

Dieser Trading Bot implementiert die im Strategiedokument beschriebene Architektur:

- **DRQN (Deep Recurrent Q-Network)** mit GRU-Schichten für Sequenz-Verarbeitung
- **Whale Tracking** durch A/D Line, OBV, und Volume-Anomalie-Erkennung
- **Invalid Action Masking** zur Effizienzsteigerung
- **Cyclical Time Encoding** für Marktzyklen-Verständnis
- **Dynamic Stop-Loss** basierend auf ATR
- **Realistische Kosten**: Maker/Taker Fees, Slippage
- **Supervised Trend Predictor** als "Intuition" für den RL-Agenten

## 🏗️ Projekt-Struktur

```
advanced_trading_bot/
├── data/
│   └── data_loader.py           # Erweiterte Daten-Pipeline
├── indicators/
│   └── technical_indicators.py  # Whale Tracking Indikatoren
├── models/
│   └── (für zukünftige Modelle)
├── agents/
│   └── drqn_agent.py           # DRQN Agent Implementation
├── environments/
│   └── trading_env.py          # Trading Environment
├── backtesting/
│   └── baseline_strategies.py  # Baseline Strategien
├── utils/
│   └── (Hilfsfunktionen)
├── visualization/
│   └── (Visualisierung)
├── train.py                     # Haupt-Trainings-Pipeline
├── requirements.txt             # Python Dependencies
└── README.md                    # Diese Datei
```

## 🔧 Installation

### 1. Repository klonen (oder Dateien kopieren)

```bash
cd advanced_trading_bot
```

### 2. Python Environment erstellen

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows
```

### 3. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 4. PyTorch installieren (falls nicht automatisch installiert)

```bash
# CPU Version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# GPU Version (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Quick Start

### Vollständiges Training

```bash
python train.py
```

Dies führt die komplette Pipeline aus:
1. Daten-Download (BTC-USD, 2020-2024)
2. Feature Engineering (40+ Indikatoren)
3. DRQN Training (50 Episodes)
4. Evaluation auf Test-Daten
5. Vergleich mit Baselines (Buy&Hold, Random, MA Crossover)
6. Visualisierung der Ergebnisse

### Einzelne Komponenten testen

```bash
# Daten-Loader testen
python data/data_loader.py

# Indikatoren testen
python indicators/technical_indicators.py

# Agent testen
python agents/drqn_agent.py

# Environment testen
python environments/trading_env.py

# Baselines testen
python backtesting/baseline_strategies.py
```

## 📊 Features im Detail

### 1. **DRQN Architektur**

- **GRU-Schichten**: Verarbeitet Sequenzen von 30 Kerzen
- **Attention Mechanism**: Fokussiert auf wichtige Zeitpunkte
- **Dueling DQN**: Separate Value- und Advantage-Streams
- **Layer Normalization**: Stabilisiert Training

### 2. **Whale Tracking Indikatoren**

- **A/D Line (Accumulation/Distribution)**: Erkennt versteckten Kauf-/Verkaufsdruck
- **OBV (On-Balance Volume)**: Misst Volumen-Momentum
- **VPT (Volume Price Trend)**: Berücksichtigt Preis-Änderungs-Magnitude
- **Volume Anomalien**: Z-Score basierte Erkennung ungewöhnlicher Volumina
- **Price-Volume Divergence**: Erkennt Divergenzen zwischen Preis und Volumen

### 3. **Technische Indikatoren**

- **Ichimoku Cloud**: Umfassendes Trend-System
- **ATR (Average True Range)**: Volatilitäts-Messung
- **Bollinger Bands**: Volatilitäts-Bänder
- **RSI (Relative Strength Index)**: Momentum-Indikator
- **MACD**: Trend-Following Momentum

### 4. **Environment Features**

- **Realistische Kosten**:
  - Maker Fee: 0.1%
  - Taker Fee: 0.2%
  - Slippage: 0.1%
- **Dynamic Stop-Loss**: Basierend auf ATR (2x Multiplikator)
- **Drawdown Penalties**: Bestraft große Verluste
- **Multi-Timeframe Support**: Kontext von höheren Timeframes

### 5. **Invalid Action Masking**

Verhindert physikalisch unmögliche Aktionen:
- Kein Kauf ohne Geld
- Kein Verkauf ohne Position
- Eliminiert Millionen sinnloser Trainingsschritte

### 6. **Cyclical Time Encoding**

Zeitliche Features als Sin/Cos-Wellen:
- Stunde des Tages (0-23)
- Tag der Woche (0-6)
- Tag des Monats (1-31)

## 📈 Training-Prozess

### Standard-Konfiguration

```python
trainer = TradingBotTrainer(
    ticker="BTC-USD",
    interval="1h",
    train_start="2020-01-01",
    train_end="2024-06-01",
    test_start="2024-06-01",
    test_end="2024-12-01",
    sequence_length=30,
    initial_balance=10000.0
)
```

### Training-Parameter

- **Episodes**: 50-100 (kann erhöht werden für bessere Ergebnisse)
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Epsilon Decay**: 0.995
- **Gamma (Discount Factor)**: 0.99
- **Replay Buffer Size**: 10,000

### Training-Ablauf

1. **Episode Start**: Environment und Agent zurücksetzen
2. **Aktion wählen**: Epsilon-greedy mit Invalid Action Masking
3. **Schritt ausführen**: Trade im Environment
4. **Erfahrung speichern**: In Replay Buffer
5. **Training**: Mini-Batch aus Replay Buffer
6. **Target Update**: Alle 5-10 Episodes

## 🏆 Evaluation & Benchmarks

### Baseline-Strategien

1. **Buy & Hold**: Kaufen am Start, Halten bis Ende
2. **Random Trading**: Zufällige Kauf/Verkauf-Entscheidungen
3. **MA Crossover**: Golden/Death Cross (MA50/MA200)
4. **RSI Momentum**: Kauf bei Oversold, Verkauf bei Overbought

### Metriken

- **Profit %**: Gesamtgewinn in Prozent
- **Sharpe Ratio**: Risiko-adjustierte Rendite
- **Max Drawdown**: Maximaler Verlust vom Peak
- **Win Rate**: Prozentsatz profitabler Trades
- **Total Trades**: Anzahl ausgeführter Trades

## 📊 Ergebnisse visualisieren

Nach dem Training werden automatisch erstellt:

1. **training_progress.png**: 
   - Episode Rewards
   - Net Worth Entwicklung
   - Profit % über Zeit
   - Training Loss

2. **comparison_results.csv**: 
   - Vergleich aller Strategien
   - Alle Metriken in Tabellen-Form

## ⚙️ Anpassung & Erweiterung

### Neue Indikatoren hinzufügen

```python
# In indicators/technical_indicators.py
@staticmethod
def my_custom_indicator(df: pd.DataFrame) -> pd.Series:
    # Ihre Implementierung
    return indicator_values

# In add_all_indicators():
df['my_indicator'] = AdvancedIndicators.my_custom_indicator(df)
```

### Training-Parameter anpassen

```python
# In train.py
trainer.train(
    num_episodes=200,      # Mehr Episodes
    update_target_every=10,
    save_every=20,
    render_every=50
)
```

### Andere Assets trainieren

```python
trainer = TradingBotTrainer(
    ticker="ETH-USD",      # Ethereum
    interval="15m",        # 15-Minuten Kerzen
    # ...
)
```

## 🔬 Experimente

### Empfohlene Tests

1. **Verschiedene Timeframes**: 15m vs 1h vs 4h
2. **Verschiedene Assets**: BTC, ETH, Aktien
3. **Längere Training**: 200-500 Episodes
4. **Hyperparameter Tuning**:
   - Hidden Size (64, 128, 256)
   - Sequence Length (20, 30, 60)
   - Learning Rate (0.0001, 0.001)

### A/B Testing

```python
# Variante A: Standard DRQN
agent_a = DRQNAgent(hidden_size=128, ...)

# Variante B: Größeres Netzwerk
agent_b = DRQNAgent(hidden_size=256, ...)

# Training und Vergleich
```

## 📝 Logging & Monitoring

### TensorBoard Integration (Optional)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/drqn_experiment')

# In Training-Loop:
writer.add_scalar('Reward/episode', episode_reward, episode)
writer.add_scalar('NetWorth/episode', net_worth, episode)
```

### Checkpoint Management

Agents werden automatisch gespeichert:
- Alle 10 Episodes: `agent_episode_N.pt`
- Final: `agent_final.pt`

Laden eines gespeicherten Agents:

```python
agent = DRQNAgent(...)
agent.load('./results/agent_final.pt')
```

## 🐛 Troubleshooting

### Häufige Probleme

**1. CUDA out of memory**
```python
# Reduziere Batch Size
agent = DRQNAgent(batch_size=16)  # statt 32
```

**2. NaN in Training Loss**
```python
# Reduziere Learning Rate
agent = DRQNAgent(learning_rate=0.00001)
```

**3. Agent macht keine Trades**
```python
# Erhöhe Epsilon oder Training-Duration
agent = DRQNAgent(epsilon_start=1.0, epsilon_decay=0.99)
```

**4. Zu langsames Training**
```python
# Nutze größeres Interval (weniger Daten)
trainer = TradingBotTrainer(interval="4h")  # statt "15m"
```

## 📚 Weiterführende Literatur

- [DQN Paper](https://arxiv.org/abs/1312.5602) - Mnih et al. 2013
- [DRQN Paper](https://arxiv.org/abs/1507.06527) - Hausknecht & Stone 2015
- [Dueling DQN](https://arxiv.org/abs/1511.06581) - Wang et al. 2016
- [Technical Analysis](https://www.investopedia.com/terms/t/technicalanalysis.asp)

## 🤝 Contributing

Verbesserungsvorschläge willkommen!

1. Fork the project
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📄 Lizenz

MIT License - siehe LICENSE file

## 👨‍💻 Autor

Entwickelt basierend auf dem Strategiedokument zur Neuausrichtung des Trading Bot Projekts.

## 🙏 Danksagungen

- **yfinance** für Datenzugriff
- **PyTorch** für Deep Learning Framework
- **OpenAI Gym** für Environment Interface

---

**Status**: 🟢 Production Ready

**Letzte Aktualisierung**: Dezember 2024

**Version**: 1.0.0
