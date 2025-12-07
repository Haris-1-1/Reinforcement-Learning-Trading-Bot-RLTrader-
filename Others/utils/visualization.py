"""
Visualization utilities for trading bot performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Visualizer:
    """Visualization tools for trading performance analysis."""

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        sns.set_palette("husl")

    @staticmethod
    def plot_portfolio_value(df: pd.DataFrame, buy_hold_value: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None):
        """
        Plot portfolio value over time.

        Args:
            df: DataFrame with trading history
            buy_hold_value: Buy & Hold portfolio values for comparison
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df.index, df['portfolio_value'], label='RL Agent', linewidth=2)

        if buy_hold_value is not None:
            ax.plot(df.index, buy_hold_value, label='Buy & Hold',
                   linestyle='--', linewidth=2, alpha=0.7)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Portfolio Value ($)')
        ax.set_title('Portfolio Value Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Portfolio value plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_trades(df: pd.DataFrame, trades_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot price chart with trade markers.

        Args:
            df: DataFrame with price data
            trades_df: DataFrame with trade history
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot price
        ax.plot(df.index, df['Close'], label='Price', linewidth=1.5, color='black', alpha=0.7)

        # Mark buy trades
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        if not buy_trades.empty:
            ax.scatter(buy_trades['step'], buy_trades['price'],
                      color='green', marker='^', s=100, label='Buy', zorder=5)

        # Mark sell trades
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        if not sell_trades.empty:
            ax.scatter(sell_trades['step'], sell_trades['price'],
                      color='red', marker='v', s=100, label='Sell', zorder=5)

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Price ($)')
        ax.set_title('Price Chart with Trade Markers')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trades plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_training_rewards(episode_rewards: List[float], window: int = 10,
                             save_path: Optional[str] = None):
        """
        Plot training episode rewards with moving average.

        Args:
            episode_rewards: List of episode rewards
            window: Window size for moving average
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        episodes = range(len(episode_rewards))
        ax.plot(episodes, episode_rewards, alpha=0.3, label='Episode Reward')

        # Moving average
        if len(episode_rewards) >= window:
            ma = pd.Series(episode_rewards).rolling(window=window).mean()
            ax.plot(episodes, ma, linewidth=2, label=f'MA({window})')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Rewards Over Episodes')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training rewards plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_comparison(results: dict, save_path: Optional[str] = None):
        """
        Plot comparison of different experiments.

        Args:
            results: Dictionary mapping experiment names to final portfolio values
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(results.keys())
        values = list(results.values())

        bars = ax.bar(names, values, color=sns.color_palette("husl", len(names)))

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height:.2f}',
                   ha='center', va='bottom')

        ax.set_ylabel('Final Portfolio Value ($)')
        ax.set_title('Experiment Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")

        plt.show()

    @staticmethod
    def plot_interactive_chart(df: pd.DataFrame, trades_df: Optional[pd.DataFrame] = None):
        """
        Create interactive plotly chart with price and indicators.

        Args:
            df: DataFrame with OHLCV and indicators
            trades_df: Optional DataFrame with trade history
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Price & Indicators', 'RSI', 'Volume')
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Moving averages
        for ma in ['MA5', 'MA20', 'MA50']:
            if ma in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(width=1)),
                    row=1, col=1
                )

        # Bollinger Bands
        if 'BB_upper' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                          line=dict(dash='dash', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                          line=dict(dash='dash', width=1)),
                row=1, col=1
            )

        # Add trade markers
        if trades_df is not None and not trades_df.empty:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']

            if not buy_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_trades['step'],
                        y=buy_trades['price'],
                        mode='markers',
                        name='Buy',
                        marker=dict(symbol='triangle-up', size=12, color='green')
                    ),
                    row=1, col=1
                )

            if not sell_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_trades['step'],
                        y=sell_trades['price'],
                        mode='markers',
                        name='Sell',
                        marker=dict(symbol='triangle-down', size=12, color='red')
                    ),
                    row=1, col=1
                )

        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # Volume
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightblue'),
            row=3, col=1
        )

        # Update layout
        fig.update_layout(
            title='Trading Analysis Dashboard',
            xaxis_rangeslider_visible=False,
            height=900,
            showlegend=True
        )

        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)

        fig.show()

    @staticmethod
    def generate_performance_report(initial_value: float, final_value: float,
                                   trades_df: pd.DataFrame, df: pd.DataFrame) -> dict:
        """
        Generate performance metrics report.

        Args:
            initial_value: Initial portfolio value
            final_value: Final portfolio value
            trades_df: DataFrame with trade history
            df: DataFrame with price data

        Returns:
            Dictionary with performance metrics
        """
        total_return = (final_value - initial_value) / initial_value
        total_trades = len(trades_df)

        # Calculate win rate
        if total_trades > 0:
            # Simple win rate based on whether trade increased portfolio value
            wins = sum(trades_df['portfolio_value'].diff() > 0)
            win_rate = wins / max(1, total_trades - 1)
        else:
            win_rate = 0.0

        # Buy & Hold comparison
        buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]

        metrics = {
            'Initial Value': f'${initial_value:.2f}',
            'Final Value': f'${final_value:.2f}',
            'Total Return': f'{total_return:.2%}',
            'Total Trades': total_trades,
            'Win Rate': f'{win_rate:.2%}',
            'Buy & Hold Return': f'{buy_hold_return:.2%}',
            'Outperformance': f'{(total_return - buy_hold_return):.2%}'
        }

        print("\n" + "="*50)
        print("PERFORMANCE REPORT")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key:.<30} {value}")
        print("="*50 + "\n")

        return metrics
