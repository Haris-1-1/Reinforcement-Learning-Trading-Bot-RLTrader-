# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

I want to expand my RL Paper Trading Bot and make it experiment-friendly. Here's what I need:

GOAL:
Create an advanced RL Paper Trading Bot in the "RL Trading Programms" folder that builds on the Demo version but is significantly more complex and experiment-friendly.

REQUIREMENTS:

1. ENVIRONMENT (env/advanced_trading_env.py):
   - Extend the State to include:
     * Price-based features: Open, High, Low, Close, Volume
     * Technical indicators: MA5, MA20, MA50, RSI, MACD, Bollinger Bands, ATR (Average True Range)
   - Add Constraints:
     * Trading Fees (Maker/Taker)
     * Slippage
     * Trade Frequency Penalty (to discourage excessive trading)
     * Position Size Constraints
     * Optional: Execution Delay
   - Keep the reward structure: realized profit (not unrealized)

2. AGENTS (agents/):
   - Create at least 2-3 different agents for comparison:
     * Improved Q-Learning Agent
     * DQN Agent (Deep Q-Network)
     * Optional: PPO Agent
   - Each agent should be easily interchangeable

3. EXPERIMENT FRAMEWORK:
   - Create an experiment system that enables the following comparisons:
     * With vs. without constraints
     * Different training periods (5 years, 1 year, 3 months, 1 week)
     * Different cryptos (BTC, ETH, etc.)
     * Different agents
   - Save all results in a structured way in results/

4. DATA HANDLING (utils/):
   - Create data_loader.py for feature calculation
   - Automatically calculate all technical indicators
   - Support multiple data sources (yfinance)

5. TRAINING & EVALUATION:
   - Create train.py for training with configuration
   - Create evaluate.py for backtesting
   - Visualization: Portfolio value, trades, comparison with Buy&Hold

6. CONFIGURATION:
   - Use config files (YAML or JSON) for easy experimentation
   - All hyperparameters should be configurable

IMPORTANT:
- Code should be modular and well-documented
- Each experiment should be reproducible
- Use the existing Demo as reference, but build from scratch
- Create a README.md with instructions for experimenting

OPTIONAL ADDON - INTERACTIVE WEBSITE:
Additionally, if time allows, create a separate web interface where I can:
- Start/stop training runs with custom configurations
- Configure experiments through an intuitive UI (select agent, crypto, timeframe, constraints)
- View real-time training metrics and plots (rewards, portfolio value, loss curves)
- Compare different experiment results side-by-side
- Visualize trades on interactive price charts (with indicators)
- See portfolio performance over time with detailed statistics
- Use Streamlit (recommended) or Dash/Flask
- Launch with one command (e.g., `streamlit run app.py`)

Start with the core structure and base components first. The website is a nice-to-have addon that can be built afterwards if there's time.
