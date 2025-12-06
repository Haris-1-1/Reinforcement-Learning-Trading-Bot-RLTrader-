# Advanced RL Trading Bot

An experiment-friendly reinforcement learning trading bot with extended state space, technical indicators, and realistic trading constraints.

## Features

- **Advanced Trading Environment**: Extended state space with OHLCV data and 15+ technical indicators
- **Multiple RL Agents**: DQN (Deep Q-Network), Q-Learning, and optional PPO
- **Realistic Constraints**: Trading fees, slippage, position limits, and trade frequency penalties
- **Experiment Framework**: Easy comparison of different agents, time periods, and cryptocurrencies
- **Comprehensive Visualization**: Portfolio performance, trade analysis, and interactive charts
- **Configuration-Driven**: YAML-based configuration for easy experimentation

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r ../requirements.txt
```

### 2. Train an Agent

```bash
# Train with default configuration (DQN on BTC)
python train.py --config configs/default_config.yaml
```

### 3. Evaluate the Agent

```bash
# Evaluate on test data
python evaluate.py --model results/exp_001/model.zip --config results/exp_001/config.yaml --dataset test
```

## Project Structure

```
RL Trading Programms/
├── agents/                     # RL agents
│   ├── base_agent.py          # Abstract base class
│   ├── dqn_agent.py           # Deep Q-Network
│   └── q_learning_agent.py    # Tabular Q-Learning
├── env/                        # Trading environment
│   └── advanced_trading_env.py # Gymnasium environment with constraints
├── utils/                      # Utilities
│   ├── data_loader.py         # Data fetching and processing
│   ├── indicators.py          # Technical indicators
│   ├── visualization.py       # Plotting and analysis
│   └── config_loader.py       # Configuration management
├── experiments/                # Experiment framework
│   └── experiment_runner.py   # Run multiple experiments
├── configs/                    # Configuration files
│   └── default_config.yaml    # Default hyperparameters
├── results/                    # Training results
├── train.py                    # Training script
├── evaluate.py                 # Evaluation script
└── README.md                   # This file
```

## Configuration

All experiments are controlled via YAML config files. See `configs/default_config.yaml` for a complete example.

### Key Configuration Sections

**Data Configuration**:
```yaml
data:
  symbol: "BTC-USD"            # Crypto symbol
  start_date: "2020-01-01"     # Training start
  end_date: "2024-01-01"       # Training end
  interval: "1d"               # Data interval
  test_split: 0.2              # Test set fraction
```

**Environment Constraints**:
```yaml
environment:
  initial_cash: 10000.0
  trading_fee_maker: 0.001     # 0.1% maker fee
  trading_fee_taker: 0.002     # 0.2% taker fee
  slippage: 0.001              # 0.1% slippage
  trade_frequency_penalty: 0.0001
  max_position_size: 1.0       # Max 100% invested
```

**Agent Selection**:
```yaml
agent:
  type: "dqn"                  # Options: "dqn", "q_learning", "ppo"
```

**Training Parameters**:
```yaml
training:
  total_timesteps: 100000
  save_freq: 10000
  log_interval: 1000
```

## Running Experiments

### Single Experiment

```bash
# Train a DQN agent on BTC
python train.py --config configs/default_config.yaml

# Evaluate on test data
python evaluate.py --model results/exp_001/model.zip --config results/exp_001/config.yaml
```

### Multiple Experiments (Comparison)

```python
from experiments.experiment_runner import ExperimentRunner

# Initialize runner
runner = ExperimentRunner('configs/default_config.yaml')

# Add experiments
runner.add_experiment('DQN_BTC', {'agent': {'type': 'dqn'}})
runner.add_experiment('QLearning_BTC', {'agent': {'type': 'q_learning'}})
runner.add_experiment('DQN_ETH', {'data': {'symbol': 'ETH-USD'}})

# Run all experiments
results = runner.run_all_experiments()

# Generate comparison report
runner.generate_comparison_report()
```

## Technical Indicators

The environment uses the following technical indicators:

- **Moving Averages**: MA5, MA20, MA50
- **Momentum**: RSI (Relative Strength Index)
- **Trend**: MACD (Moving Average Convergence Divergence)
- **Volatility**: Bollinger Bands, ATR (Average True Range)
- **Volume**: Volume MA, Volume Ratio
- **Returns**: Simple returns, Log returns

## Agent Comparison

| Agent | Pros | Cons | Best For |
|-------|------|------|----------|
| **DQN** | Handles high-dimensional states, strong performance | Requires more data, longer training | Complex markets, long training |
| **Q-Learning** | Fast, interpretable, works with limited data | Limited to discretized states | Quick experiments, baseline |
| **PPO** (optional) | Stable, good for continuous actions | More complex setup | Advanced experiments |

## Trading Constraints

### Fees
- **Maker Fee**: 0.1% (default) - limit orders
- **Taker Fee**: 0.2% (default) - market orders

### Slippage
- Simulates price impact: 0.1% (default)
- Buy price increased, sell price decreased

### Trade Frequency Penalty
- Discourages overtrading
- Applied when trades happen too frequently (< 5 steps apart)

### Position Size Constraints
- Maximum position: 100% of capital (configurable)
- Binary position: 0 (no position) or 1 (fully invested)

## Experiment Ideas

1. **With vs. Without Constraints**
   - Compare agent performance with and without fees/slippage
   - Understand impact of realistic trading costs

2. **Different Training Periods**
   - 5 years vs. 1 year vs. 3 months vs. 1 week
   - Evaluate how training duration affects performance

3. **Different Cryptocurrencies**
   - BTC vs. ETH vs. BNB vs. others
   - Find which assets work best for RL trading

4. **Agent Comparison**
   - DQN vs. Q-Learning vs. PPO
   - Determine best agent for your use case

5. **Hyperparameter Tuning**
   - Learning rate, gamma, network architecture
   - Optimize agent performance

## Results and Metrics

After training, you'll get:

### Metrics
- **Total Return**: Percentage gain/loss
- **Final Portfolio Value**: Dollar value at end
- **Total Trades**: Number of buy/sell transactions
- **Win Rate**: Percentage of profitable trades
- **Fees Paid**: Total trading fees
- **Outperformance**: Return vs. Buy & Hold

### Visualizations
- **Portfolio Value**: Agent vs. Buy & Hold over time
- **Trades**: Buy/sell markers on price chart
- **Training Rewards**: Episode rewards with moving average
- **Interactive Charts**: Plotly charts with indicators

## Tips for Best Results

1. **Start with DQN**: Best balance of performance and ease of use
2. **Use Longer Training Periods**: More data = better learning
3. **Enable Constraints**: More realistic results
4. **Compare with Buy & Hold**: Always include baseline
5. **Experiment with Hyperparameters**: Fine-tune for your data
6. **Monitor Training**: Check episode rewards for convergence
7. **Evaluate on Test Data**: Avoid overfitting to training period

## Troubleshooting

### Agent not learning
- Increase training timesteps
- Adjust learning rate
- Check reward function
- Verify data quality

### Poor performance on test data
- Model overfitting - reduce complexity
- Increase test set size
- Use different time periods

### Training too slow
- Use smaller network architecture
- Reduce buffer size (DQN)
- Use Q-Learning for quick experiments

## Next Steps

1. **Experiment with different configurations**
2. **Try different cryptocurrencies and time periods**
3. **Implement custom reward functions**
4. **Add more technical indicators**
5. **Build the optional web interface**

## Advanced Features (TODO)

- [ ] PPO Agent implementation
- [ ] Multi-asset portfolio support
- [ ] Custom reward functions
- [ ] Live trading integration
- [ ] Streamlit web dashboard
- [ ] Hyperparameter optimization (Optuna)
- [ ] Ensemble agents

## Contributing

Feel free to experiment and extend this framework:
- Add new agents in `agents/`
- Create custom environments in `env/`
- Add indicators in `utils/indicators.py`
- Build new visualizations in `utils/visualization.py`

## License

Educational project for reinforcement learning and trading research.

**Disclaimer**: This is for educational purposes only. Not financial advice. Do not use for live trading without extensive testing and risk management.
