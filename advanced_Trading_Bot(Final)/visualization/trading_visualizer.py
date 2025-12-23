"""
Visualization module for trading bot analysis
Creates detailed plots for whale detection, trade analysis, and performance metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
import os

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


class TradingVisualizer:
    """
    Visualization tools for trading bot analysis
    """
    
    def __init__(self, save_dir: str = "./visualizations"):
        """
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_price_with_trades(
        self,
        data: pd.DataFrame,
        trades: List[Dict],
        title: str = "Price Chart with Trades",
        filename: str = "price_with_trades.png"
    ):
        """
        Plot price chart with buy/sell markers
        
        Args:
            data: DataFrame with OHLCV data
            trades: List of trade dictionaries
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot price
        ax.plot(data.index, data['close'], label='Close Price', linewidth=1.5, color='blue', alpha=0.7)
        
        # Plot trades
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        if buy_trades:
            buy_indices = [t['step'] for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            ax.scatter(
                [data.index[i] for i in buy_indices],
                buy_prices,
                marker='^',
                s=200,
                color='green',
                label='Buy',
                zorder=5,
                edgecolors='black',
                linewidths=1.5
            )
        
        if sell_trades:
            sell_indices = [t['step'] for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            ax.scatter(
                [data.index[i] for i in sell_indices],
                sell_prices,
                marker='v',
                s=200,
                color='red',
                label='Sell',
                zorder=5,
                edgecolors='black',
                linewidths=1.5
            )
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        plt.close()
    
    def plot_whale_detection(
        self,
        data: pd.DataFrame,
        title: str = "Whale Activity Detection",
        filename: str = "whale_detection.png"
    ):
        """
        Plot whale detection indicators
        
        Args:
            data: DataFrame with whale tracking indicators
            title: Plot title
            filename: Output filename
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot 1: Price with volume spikes
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        ax1.plot(data.index, data['close'], label='Close Price', color='blue', linewidth=1.5)
        ax1_twin.bar(
            data.index,
            data['volume'],
            alpha=0.3,
            color='gray',
            label='Volume'
        )
        
        # Highlight volume spikes
        if 'volume_spike' in data.columns:
            volume_spikes = data[data['volume_spike'] == 1]
            if not volume_spikes.empty:
                ax1_twin.bar(
                    volume_spikes.index,
                    volume_spikes['volume'],
                    alpha=0.7,
                    color='red',
                    label='Volume Spike'
                )
        
        ax1.set_ylabel('Price ($)', fontweight='bold')
        ax1_twin.set_ylabel('Volume', fontweight='bold')
        ax1.set_title('Price with Volume Spikes (Potential Whale Activity)')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: A/D Line
        if 'ad_line' in data.columns:
            ax2 = axes[1]
            ax2.plot(data.index, data['ad_line'], label='A/D Line', color='purple', linewidth=1.5)
            ax2.fill_between(data.index, data['ad_line'], alpha=0.3, color='purple')
            ax2.set_ylabel('A/D Line', fontweight='bold')
            ax2.set_title('Accumulation/Distribution Line')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: OBV
        if 'obv' in data.columns:
            ax3 = axes[2]
            ax3.plot(data.index, data['obv'], label='OBV', color='orange', linewidth=1.5)
            ax3.fill_between(data.index, data['obv'], alpha=0.3, color='orange')
            ax3.set_ylabel('OBV', fontweight='bold')
            ax3.set_title('On-Balance Volume')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Price-Volume Divergence
        if 'pv_divergence' in data.columns:
            ax4 = axes[3]
            ax4.plot(data.index, data['pv_divergence'], label='P/V Divergence', color='teal', linewidth=1.5)
            ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
            ax4.fill_between(
                data.index,
                data['pv_divergence'],
                0,
                where=(data['pv_divergence'] > 0),
                alpha=0.3,
                color='green',
                label='Accumulation'
            )
            ax4.fill_between(
                data.index,
                data['pv_divergence'],
                0,
                where=(data['pv_divergence'] < 0),
                alpha=0.3,
                color='red',
                label='Distribution'
            )
            ax4.set_ylabel('Divergence', fontweight='bold')
            ax4.set_xlabel('Date', fontweight='bold')
            ax4.set_title('Price-Volume Divergence (Positive = Accumulation, Negative = Distribution)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        plt.close()
    
    def plot_equity_curve_comparison(
        self,
        results: Dict[str, List[float]],
        title: str = "Equity Curve Comparison",
        filename: str = "equity_comparison.png"
    ):
        """
        Plot equity curves for multiple strategies
        
        Args:
            results: Dictionary with strategy name as key and net worth history as value
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (strategy, net_worth_history) in enumerate(results.items()):
            ax.plot(
                net_worth_history,
                label=strategy,
                linewidth=2,
                color=colors[i % len(colors)],
                alpha=0.8
            )
        
        ax.set_xlabel('Time Step', fontsize=12, fontweight='bold')
        ax.set_ylabel('Net Worth ($)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        plt.close()
    
    def plot_strategy_comparison(
        self,
        comparison_df: pd.DataFrame,
        title: str = "Strategy Comparison",
        filename: str = "strategy_comparison.png"
    ):
        """
        Plot comparison metrics for different strategies
        
        Args:
            comparison_df: DataFrame with strategy comparison results
            title: Plot title
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Define metrics to plot
        metrics = [
            ('profit_pct', 'Profit (%)'),
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('max_drawdown', 'Max Drawdown'),
            ('win_rate', 'Win Rate'),
            ('total_trades', 'Total Trades'),
            ('final_net_worth', 'Final Net Worth ($)')
        ]
        
        # Color palette
        colors = sns.color_palette("husl", len(comparison_df))
        
        for idx, (metric, label) in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            if metric in comparison_df.columns:
                values = comparison_df[metric].values
                strategies = comparison_df['strategy'].values
                
                bars = ax.barh(strategies, values, color=colors, edgecolor='black', linewidth=1.5)
                
                # Add value labels
                for i, (bar, val) in enumerate(zip(bars, values)):
                    if metric == 'final_net_worth':
                        ax.text(
                            val, bar.get_y() + bar.get_height()/2,
                            f'${val:,.0f}',
                            va='center',
                            ha='left',
                            fontweight='bold',
                            fontsize=9
                        )
                    elif metric in ['profit_pct', 'max_drawdown', 'win_rate']:
                        ax.text(
                            val, bar.get_y() + bar.get_height()/2,
                            f'{val:.2f}%' if metric != 'sharpe_ratio' else f'{val:.2f}',
                            va='center',
                            ha='left',
                            fontweight='bold',
                            fontsize=9
                        )
                    else:
                        ax.text(
                            val, bar.get_y() + bar.get_height()/2,
                            f'{val:.2f}',
                            va='center',
                            ha='left',
                            fontweight='bold',
                            fontsize=9
                        )
                
                ax.set_xlabel(label, fontweight='bold')
                ax.set_title(label, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        plt.close()
    
    def plot_drawdown_analysis(
        self,
        net_worth_history: List[float],
        title: str = "Drawdown Analysis",
        filename: str = "drawdown_analysis.png"
    ):
        """
        Plot drawdown analysis
        
        Args:
            net_worth_history: History of net worth values
            title: Plot title
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        net_worth = np.array(net_worth_history)
        
        # Calculate running maximum (peak)
        peak = np.maximum.accumulate(net_worth)
        
        # Calculate drawdown
        drawdown = (peak - net_worth) / peak * 100
        
        # Plot 1: Net Worth with Peak
        ax1 = axes[0]
        ax1.plot(net_worth, label='Net Worth', linewidth=2, color='blue')
        ax1.plot(peak, label='Peak', linewidth=2, color='green', linestyle='--')
        ax1.fill_between(range(len(net_worth)), net_worth, peak, alpha=0.3, color='red')
        ax1.set_ylabel('Net Worth ($)', fontweight='bold')
        ax1.set_title('Net Worth vs Peak')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        ax2 = axes[1]
        ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.5, color='red')
        ax2.plot(drawdown, linewidth=2, color='darkred')
        ax2.axhline(y=np.max(drawdown), color='black', linestyle='--', linewidth=1,
                   label=f'Max Drawdown: {np.max(drawdown):.2f}%')
        ax2.set_xlabel('Time Step', fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontweight='bold')
        ax2.set_title('Drawdown Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        plt.close()
    
    def plot_trade_analysis(
        self,
        trades: List[Dict],
        title: str = "Trade Analysis",
        filename: str = "trade_analysis.png"
    ):
        """
        Plot detailed trade analysis
        
        Args:
            trades: List of trade dictionaries
            title: Plot title
            filename: Output filename
        """
        if not trades:
            print("⚠️  No trades to analyze")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Extract sell trades (which have profit information)
        sell_trades = [t for t in trades if t['action'] == 'SELL' and 'profit' in t]
        
        if not sell_trades:
            print("⚠️  No completed trades to analyze")
            return
        
        profits = [t['profit'] for t in sell_trades]
        
        # Plot 1: Profit Distribution
        ax1 = axes[0, 0]
        ax1.hist(profits, bins=20, edgecolor='black', color='steelblue', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break Even')
        ax1.axvline(x=np.mean(profits), color='green', linestyle='--', linewidth=2,
                   label=f'Mean: ${np.mean(profits):.2f}')
        ax1.set_xlabel('Profit ($)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.set_title('Trade Profit Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Profit
        ax2 = axes[0, 1]
        cumulative_profit = np.cumsum(profits)
        ax2.plot(cumulative_profit, linewidth=2, color='green')
        ax2.fill_between(range(len(cumulative_profit)), cumulative_profit, 0,
                        alpha=0.3, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Trade Number', fontweight='bold')
        ax2.set_ylabel('Cumulative Profit ($)', fontweight='bold')
        ax2.set_title('Cumulative Profit Over Trades')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Win/Loss Ratio
        ax3 = axes[1, 0]
        winning_trades = sum(1 for p in profits if p > 0)
        losing_trades = sum(1 for p in profits if p < 0)
        breakeven_trades = sum(1 for p in profits if p == 0)
        
        categories = ['Winning', 'Losing', 'Breakeven']
        values = [winning_trades, losing_trades, breakeven_trades]
        colors_pie = ['green', 'red', 'gray']
        
        ax3.pie(values, labels=categories, autopct='%1.1f%%', startangle=90,
               colors=colors_pie, textprops={'fontweight': 'bold'})
        ax3.set_title('Win/Loss Distribution')
        
        # Plot 4: Profit by Trade
        ax4 = axes[1, 1]
        trade_numbers = range(1, len(profits) + 1)
        colors_bar = ['green' if p > 0 else 'red' for p in profits]
        ax4.bar(trade_numbers, profits, color=colors_bar, edgecolor='black', linewidth=1)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xlabel('Trade Number', fontweight='bold')
        ax4.set_ylabel('Profit ($)', fontweight='bold')
        ax4.set_title('Profit/Loss by Trade')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
        plt.close()


def test_visualizer():
    """Test the visualizer"""
    print("="*60)
    print("Testing Trading Visualizer")
    print("="*60)
    
    # Create sample data
    import yfinance as yf
    from indicators.technical_indicators import add_all_indicators
    
    df = yf.Ticker("BTC-USD").history(period="3mo", interval="1d")
    df.columns = [col.lower() for col in df.columns]
    df = add_all_indicators(df)
    
    # Create visualizer
    viz = TradingVisualizer(save_dir="./test_visualizations")
    
    # Test 1: Whale detection
    print("\n1. Creating whale detection plot...")
    viz.plot_whale_detection(df)
    
    # Test 2: Sample trades
    sample_trades = [
        {'step': 10, 'action': 'BUY', 'price': df['close'].iloc[10]},
        {'step': 20, 'action': 'SELL', 'price': df['close'].iloc[20], 'profit': 100},
        {'step': 30, 'action': 'BUY', 'price': df['close'].iloc[30]},
        {'step': 40, 'action': 'SELL', 'price': df['close'].iloc[40], 'profit': -50},
    ]
    
    print("2. Creating price chart with trades...")
    viz.plot_price_with_trades(df, sample_trades)
    
    print("\n✅ Test completed! Check ./test_visualizations/ for plots")


if __name__ == "__main__":
    test_visualizer()
