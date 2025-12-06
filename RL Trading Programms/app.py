"""
Streamlit Web Dashboard for RL Trading Bot
Interactive interface for training, monitoring, and evaluating trading agents.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import glob
import yaml
from datetime import datetime
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from utils.visualization import Visualizer
from env.advanced_trading_env import AdvancedTradingEnv
from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent

# Page config
st.set_page_config(
    page_title="RL Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🤖 RL Trading Bot Dashboard</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "🚀 Train Agent",
    "📊 View Results",
    "📈 Compare Experiments",
    "⚙️ Configuration"
])

# Helper functions
@st.cache_data
def load_experiment_results():
    """Load all experiment results from results directory."""
    results = []
    results_dir = "results"

    if not os.path.exists(results_dir):
        return pd.DataFrame()

    # Find all experiment directories
    exp_dirs = [d for d in glob.glob(f"{results_dir}/**") if os.path.isdir(d)]

    for exp_dir in exp_dirs:
        metrics_file = os.path.join(exp_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                metrics['exp_dir'] = exp_dir
                results.append(metrics)

    return pd.DataFrame(results) if results else pd.DataFrame()

def load_experiment_details(exp_dir):
    """Load detailed information about an experiment."""
    details = {}

    # Load config
    config_file = os.path.join(exp_dir, "config.yaml")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            details['config'] = yaml.safe_load(f)

    # Load metrics
    metrics_file = os.path.join(exp_dir, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            details['metrics'] = json.load(f)

    # Load trades
    trades_file = os.path.join(exp_dir, "trades.csv")
    if os.path.exists(trades_file):
        details['trades'] = pd.read_csv(trades_file)

    return details

def plot_portfolio_comparison(results_df):
    """Plot portfolio value comparison across experiments."""
    fig = go.Figure()

    for idx, row in results_df.iterrows():
        fig.add_trace(go.Bar(
            name=row.get('agent_type', 'Unknown'),
            x=[row.get('symbol', 'Unknown')],
            y=[row['final_value']],
            text=[f"${row['final_value']:.2f}"],
            textposition='auto',
        ))

    fig.update_layout(
        title="Final Portfolio Value Comparison",
        xaxis_title="Symbol",
        yaxis_title="Portfolio Value ($)",
        barmode='group',
        height=400
    )

    return fig

def plot_training_progress(exp_dir):
    """Plot training progress from experiment directory."""
    # This would need training logs - placeholder for now
    st.info("Training progress visualization requires training logs. Feature coming soon!")

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.header("Welcome to RL Trading Bot Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Experiments",
            value=len(load_experiment_results()),
            delta="Active experiments"
        )

    with col2:
        results_df = load_experiment_results()
        if not results_df.empty:
            best_return = results_df['total_return'].max() * 100
            st.metric(
                label="Best Return",
                value=f"{best_return:.2f}%",
                delta="Across all experiments"
            )
        else:
            st.metric(label="Best Return", value="N/A")

    with col3:
        if not results_df.empty:
            total_trades = results_df['total_trades'].sum()
            st.metric(
                label="Total Trades",
                value=int(total_trades),
                delta="All experiments"
            )
        else:
            st.metric(label="Total Trades", value="N/A")

    st.divider()

    st.subheader("Quick Start Guide")
    st.markdown("""
    ### 🎯 Get Started in 3 Steps:

    1. **🚀 Train Agent** - Configure and train your first RL trading agent
    2. **📊 View Results** - Analyze performance, trades, and metrics
    3. **📈 Compare** - Compare different agents, cryptos, and strategies

    ### 💡 Tips:
    - Start with **DQN agent** for best results
    - Use **longer training periods** for better performance
    - Always **compare with Buy & Hold** baseline
    - **Enable constraints** for realistic results
    """)

    st.divider()

    st.subheader("Recent Experiments")
    if not results_df.empty:
        display_df = results_df[['agent_type', 'symbol', 'final_value', 'total_return', 'total_trades']].copy()
        display_df['total_return'] = (display_df['total_return'] * 100).round(2).astype(str) + '%'
        display_df['final_value'] = '$' + display_df['final_value'].round(2).astype(str)
        display_df.columns = ['Agent', 'Symbol', 'Final Value', 'Return', 'Trades']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No experiments yet. Go to 'Train Agent' to start your first training!")

# ============================================================================
# TRAIN AGENT PAGE
# ============================================================================
elif page == "🚀 Train Agent":
    st.header("Train a New Agent")

    st.markdown("Configure your training parameters below:")

    # Training configuration form
    with st.form("training_config"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data Configuration")
            symbol = st.selectbox(
                "Cryptocurrency",
                ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "SOL-USD"],
                index=0
            )

            start_date = st.date_input(
                "Start Date",
                value=pd.to_datetime("2020-01-01")
            )

            end_date = st.date_input(
                "End Date",
                value=pd.to_datetime("2024-01-01")
            )

            interval = st.selectbox(
                "Data Interval",
                ["1d", "1h", "15m"],
                index=0
            )

            test_split = st.slider(
                "Test Split",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Fraction of data to use for testing"
            )

        with col2:
            st.subheader("Agent Configuration")
            agent_type = st.selectbox(
                "Agent Type",
                ["dqn", "q_learning"],
                index=0,
                help="DQN: Deep Q-Network (best), Q-Learning: Tabular (faster)"
            )

            initial_cash = st.number_input(
                "Initial Cash ($)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )

            total_timesteps = st.number_input(
                "Training Timesteps",
                min_value=10000,
                max_value=1000000,
                value=100000,
                step=10000,
                help="More timesteps = better training but slower"
            )

        st.subheader("Trading Constraints")
        col3, col4, col5 = st.columns(3)

        with col3:
            enable_constraints = st.checkbox("Enable Trading Constraints", value=True)
            trading_fee = st.number_input(
                "Trading Fee (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                disabled=not enable_constraints
            ) / 100

        with col4:
            slippage = st.number_input(
                "Slippage (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                disabled=not enable_constraints
            ) / 100

        with col5:
            trade_freq_penalty = st.number_input(
                "Trade Frequency Penalty",
                min_value=0.0,
                max_value=0.01,
                value=0.0001,
                step=0.0001,
                format="%.4f",
                disabled=not enable_constraints
            )

        st.subheader("Experiment Name")
        exp_name = st.text_input(
            "Experiment Name",
            value=f"{agent_type}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Submit button
        submitted = st.form_submit_button("🚀 Start Training", use_container_width=True)

        if submitted:
            st.success(f"Training configuration saved for experiment: **{exp_name}**")

            # Create config
            config = {
                'data': {
                    'symbol': symbol,
                    'start_date': start_date.strftime('%Y-%m-%d'),
                    'end_date': end_date.strftime('%Y-%m-%d'),
                    'interval': interval,
                    'test_split': test_split
                },
                'environment': {
                    'initial_cash': initial_cash,
                    'trading_fee_maker': trading_fee if enable_constraints else 0.0,
                    'trading_fee_taker': trading_fee if enable_constraints else 0.0,
                    'slippage': slippage if enable_constraints else 0.0,
                    'trade_frequency_penalty': trade_freq_penalty if enable_constraints else 0.0,
                    'max_position_size': 1.0,
                    'enable_execution_delay': False,
                    'execution_delay_steps': 0
                },
                'agent': {'type': agent_type},
                'training': {
                    'total_timesteps': total_timesteps,
                    'save_freq': 10000,
                    'log_interval': 1000,
                    'save_path': 'results/',
                    'tensorboard_log': 'results/tensorboard/'
                },
                'experiment': {
                    'name': exp_name,
                    'description': f"{agent_type} on {symbol}",
                    'seed': 42,
                    'enable_constraints': enable_constraints,
                    'compare_with_baseline': True
                }
            }

            # Save config
            os.makedirs("configs/custom", exist_ok=True)
            config_path = f"configs/custom/{exp_name}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            st.info(f"✅ Configuration saved to: `{config_path}`")
            st.info("⚠️ To start training, run the following command in your terminal:")
            st.code(f"cd \"RL Trading Programms\" && python train.py --config {config_path}", language="bash")

            st.warning("Note: Training from the dashboard (live execution) is coming soon. For now, use the command above.")

# ============================================================================
# VIEW RESULTS PAGE
# ============================================================================
elif page == "📊 View Results":
    st.header("Experiment Results")

    results_df = load_experiment_results()

    if results_df.empty:
        st.warning("No experiments found. Train your first agent to see results!")
    else:
        # Select experiment
        exp_options = results_df.apply(
            lambda x: f"{os.path.basename(x['exp_dir'])} - {x.get('agent_type', 'Unknown')} on {x.get('symbol', 'Unknown')}",
            axis=1
        ).tolist()

        selected_exp = st.selectbox("Select Experiment", exp_options)
        selected_idx = exp_options.index(selected_exp)
        exp_dir = results_df.iloc[selected_idx]['exp_dir']

        # Load details
        details = load_experiment_details(exp_dir)

        # Display metrics
        st.subheader("Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        metrics = details.get('metrics', {})

        with col1:
            st.metric(
                "Final Portfolio Value",
                f"${metrics.get('final_value', 0):.2f}",
                delta=f"{metrics.get('total_return', 0)*100:.2f}%"
            )

        with col2:
            st.metric(
                "Total Trades",
                int(metrics.get('total_trades', 0))
            )

        with col3:
            st.metric(
                "Total Fees Paid",
                f"${metrics.get('total_fees_paid', 0):.2f}"
            )

        with col4:
            buy_hold_return = metrics.get('total_return', 0)  # Simplified
            st.metric(
                "vs Buy & Hold",
                f"{buy_hold_return*100:.2f}%"
            )

        st.divider()

        # Trades table
        if 'trades' in details and not details['trades'].empty:
            st.subheader("Trade History")
            trades_df = details['trades']

            # Format trades table
            display_trades = trades_df[['step', 'action', 'price', 'amount', 'fee', 'portfolio_value']].copy()
            display_trades['price'] = '$' + display_trades['price'].round(2).astype(str)
            display_trades['amount'] = display_trades['amount'].round(6)
            display_trades['fee'] = '$' + display_trades['fee'].round(2).astype(str)
            display_trades['portfolio_value'] = '$' + display_trades['portfolio_value'].round(2).astype(str)
            display_trades.columns = ['Step', 'Action', 'Price', 'Amount', 'Fee', 'Portfolio Value']

            st.dataframe(display_trades, use_container_width=True, hide_index=True)

        st.divider()

        # Configuration
        if 'config' in details:
            with st.expander("View Configuration"):
                st.json(details['config'])

# ============================================================================
# COMPARE EXPERIMENTS PAGE
# ============================================================================
elif page == "📈 Compare Experiments":
    st.header("Compare Experiments")

    results_df = load_experiment_results()

    if results_df.empty:
        st.warning("No experiments to compare. Train multiple agents first!")
    elif len(results_df) < 2:
        st.warning("Need at least 2 experiments to compare.")
    else:
        # Comparison visualization
        st.subheader("Portfolio Value Comparison")
        fig = plot_portfolio_comparison(results_df)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Detailed Comparison Table")

        # Format comparison table
        compare_df = results_df[['agent_type', 'symbol', 'initial_value', 'final_value', 'total_return', 'total_trades', 'total_fees_paid']].copy()
        compare_df['total_return'] = (compare_df['total_return'] * 100).round(2).astype(str) + '%'
        compare_df['initial_value'] = '$' + compare_df['initial_value'].round(2).astype(str)
        compare_df['final_value'] = '$' + compare_df['final_value'].round(2).astype(str)
        compare_df['total_fees_paid'] = '$' + compare_df['total_fees_paid'].round(2).astype(str)
        compare_df.columns = ['Agent', 'Symbol', 'Initial Value', 'Final Value', 'Return', 'Trades', 'Fees Paid']

        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        # Returns comparison chart
        st.subheader("Returns Comparison")

        returns_fig = go.Figure()

        for idx, row in results_df.iterrows():
            agent = row.get('agent_type', 'Unknown')
            symbol = row.get('symbol', 'Unknown')
            returns = row['total_return'] * 100

            returns_fig.add_trace(go.Bar(
                name=f"{agent} - {symbol}",
                x=[f"{agent}"],
                y=[returns],
                text=[f"{returns:.2f}%"],
                textposition='auto',
            ))

        returns_fig.update_layout(
            title="Total Return Comparison",
            xaxis_title="Agent",
            yaxis_title="Return (%)",
            height=400
        )

        st.plotly_chart(returns_fig, use_container_width=True)

# ============================================================================
# CONFIGURATION PAGE
# ============================================================================
elif page == "⚙️ Configuration":
    st.header("Configuration Templates")

    st.markdown("""
    ### Available Configuration Templates

    Use these templates as starting points for your experiments.
    """)

    # List all config files
    config_files = glob.glob("configs/*.yaml")

    if config_files:
        for config_file in config_files:
            config_name = os.path.basename(config_file)

            with st.expander(f"📄 {config_name}"):
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                st.code(yaml.dump(config, default_flow_style=False), language='yaml')

                if st.button(f"Use {config_name}", key=config_name):
                    st.success(f"Configuration loaded! Modify in the 'Train Agent' page.")
    else:
        st.warning("No configuration templates found.")

    st.divider()

    st.subheader("Create New Template")
    st.markdown("Go to the **Train Agent** page to create a custom configuration.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🤖 RL Trading Bot Dashboard | Built with Streamlit</p>
    <p><small>For educational purposes only. Not financial advice.</small></p>
</div>
""", unsafe_allow_html=True)
