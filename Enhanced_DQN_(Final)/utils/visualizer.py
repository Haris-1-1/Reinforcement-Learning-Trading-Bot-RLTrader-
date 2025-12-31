import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
class TrainingVisualizer:
    def __init__(self, save_dir: str = "plots"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.style.use('seaborn-v0_8-darkgrid')
    def plot_training_progress(
        self,
        episode_rewards: list[float],
        episode_portfolios: list[float],
        episode_trades: list[int],
        epsilon_history: list[float],
        loss_history: list[float],
        initial_cash: float = 10000.0
    ):
        fig = plt.figure(figsize=(16, 12))
        episodes = range(1, len(episode_rewards) + 1)
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(episodes, episode_portfolios, 'b-', linewidth=2, label='Portfolio Value')
        ax1.axhline(y=initial_cash, color='r', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.fill_between(episodes, initial_cash, episode_portfolios,
                         where=np.array(episode_portfolios) >= initial_cash,
                         color='green', alpha=0.3, label='Profit')
        ax1.fill_between(episodes, initial_cash, episode_portfolios,
                         where=np.array(episode_portfolios) < initial_cash,
                         color='red', alpha=0.3, label='Loss')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.set_title('Portfolio Value Over Training', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        best_idx = np.argmax(episode_portfolios)
        worst_idx = np.argmin(episode_portfolios)
        ax1.plot(best_idx + 1, episode_portfolios[best_idx], 'g*', markersize=15)
        ax1.annotate(f'Best: ${episode_portfolios[best_idx]:.0f}',
                    xy=(best_idx + 1, episode_portfolios[best_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, color='green', fontweight='bold')
        ax2 = plt.subplot(3, 2, 2)
        returns = [(p - initial_cash) / initial_cash * 100 for p in episode_portfolios]
        ax2.plot(episodes, returns, 'g-', linewidth=2)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.fill_between(episodes, 0, returns, where=np.array(returns) >= 0,
                        color='green', alpha=0.3)
        ax2.fill_between(episodes, 0, returns, where=np.array(returns) < 0,
                        color='red', alpha=0.3)
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.set_title('Cumulative Return Over Training', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        final_return = returns[-1]
        ax2.text(0.95, 0.95, f'Final Return: {final_return:+.2f}%',
                transform=ax2.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(episodes, episode_rewards, 'purple', linewidth=2, alpha=0.7)
        if len(episode_rewards) > 10:
            ma = pd.Series(episode_rewards).rolling(window=min(10, len(episode_rewards))).mean()
            ax3.plot(episodes, ma, 'r-', linewidth=3, label='10-Episode MA')
            ax3.legend()
        ax3.set_xlabel('Episode', fontsize=12)
        ax3.set_ylabel('Reward', fontsize=12)
        ax3.set_title('Episode Rewards (Log Returns)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax4 = plt.subplot(3, 2, 4)
        ax4.bar(episodes, episode_trades, color='orange', alpha=0.7)
        avg_trades = np.mean(episode_trades)
        ax4.axhline(y=avg_trades, color='r', linestyle='--', linewidth=2,
                   label=f'Average: {avg_trades:.1f}')
        ax4.set_xlabel('Episode', fontsize=12)
        ax4.set_ylabel('Number of Trades', fontsize=12)
        ax4.set_title('Trading Frequency', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(epsilon_history, 'c-', linewidth=2)
        ax5.set_xlabel('Training Step', fontsize=12)
        ax5.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
        ax5.set_title('Epsilon Decay (Exploration  Exploitation)', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim([0, 1.05])
        if len(epsilon_history) > 0:
            ax5.text(0.5, 0.95, f'Start: {epsilon_history[0]:.3f}\nEnd: {epsilon_history[-1]:.3f}',
                    transform=ax5.transAxes, fontsize=11,
                    verticalalignment='top', horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax6 = plt.subplot(3, 2, 6)
        if len(loss_history) > 0:
            if len(loss_history) > 1000:
                sample_indices = np.linspace(0, len(loss_history)-1, 1000, dtype=int)
                sampled_loss = [loss_history[i] for i in sample_indices]
                sampled_steps = sample_indices
            else:
                sampled_loss = loss_history
                sampled_steps = range(len(loss_history))
            ax6.plot(sampled_steps, sampled_loss, 'r-', linewidth=1, alpha=0.5)
            if len(sampled_loss) > 50:
                ma_loss = pd.Series(sampled_loss).rolling(window=50).mean()
                ax6.plot(sampled_steps, ma_loss, 'b-', linewidth=2, label='50-Step MA')
                ax6.legend()
            ax6.set_xlabel('Training Step', fontsize=12)
            ax6.set_ylabel('Loss', fontsize=12)
            ax6.set_title('Training Loss', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No Loss Data', transform=ax6.transAxes,
                    fontsize=14, ha='center', va='center')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'training_progress.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved: {save_path}")
        return fig
    def plot_benchmark_comparison(
        self,
        results: dict[str, float],
        initial_cash: float = 10000.0
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        strategies = ['Enhanced DQN\n(Bot)', 'Buy & Hold', 'MA Crossover\n(20/50)', 'Random\nTrading']
        returns = [
            results.get('bot_return', 0) * 100,
            results.get('buy_hold', 0) * 100,
            results.get('ma_crossover', 0) * 100,
            results.get('random', 0) * 100
        ]
        colors = ['green' if r > 0 else 'red' for r in returns]
        colors[0] = 'darkgreen'
        bars = ax1.bar(strategies, returns, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax1.set_ylabel('Return (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Strategy Returns Comparison', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ret:+.2f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=12, fontweight='bold')
        final_values = [initial_cash * (1 + r/100) for r in returns]
        bars2 = ax2.bar(strategies, final_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax2.axhline(y=initial_cash, color='red', linestyle='--', linewidth=2, label='Initial Capital')
        ax2.set_ylabel('Final Portfolio Value ($)', fontsize=14, fontweight='bold')
        ax2.set_title('Final Portfolio Comparison', fontsize=16, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars2, final_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${val:,.0f}',
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'benchmark_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Benchmark plot saved: {save_path}")
        return fig
    def plot_trade_analysis(
        self,
        trades_df: pd.DataFrame,
        prices: np.ndarray
    ):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        ax1.plot(prices, 'k-', linewidth=1, alpha=0.5, label='Price')
        if len(trades_df) > 0:
            buys = trades_df[trades_df['action'] == 'BUY']
            sells = trades_df[trades_df['action'] == 'SELL']
            ax1.scatter(buys['step'], buys['price'], color='green', marker='^',
                       s=100, label='Buy', zorder=5)
            ax1.scatter(sells['step'], sells['price'], color='red', marker='v',
                       s=100, label='Sell', zorder=5)
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.set_title('Trade Execution on Price Chart', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if len(trades_df) > 0 and 'value' in trades_df.columns:
            ax2.plot(trades_df['step'], trades_df['value'], 'b-', linewidth=2, marker='o')
            ax2.set_xlabel('Time Step', fontsize=12)
            ax2.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax2.set_title('Portfolio Value at Each Trade', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            if len(trades_df) > 1:
                first_val = trades_df['value'].iloc[0]
                last_val = trades_df['value'].iloc[-1]
                ax2.text(0.02, 0.98, f'Start: ${first_val:,.0f}',
                        transform=ax2.transAxes, fontsize=11,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                ax2.text(0.98, 0.98, f'End: ${last_val:,.0f}',
                        transform=ax2.transAxes, fontsize=11,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'trade_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trade analysis plot saved: {save_path}")
        return fig
if __name__ == "__main__":
    print("Training Visualizer Module")
    print("Creates comprehensive training plots:")
    print("  - Training Progress (6 panels)")
    print("  - Benchmark Comparison")
    print("  - Trade Analysis")