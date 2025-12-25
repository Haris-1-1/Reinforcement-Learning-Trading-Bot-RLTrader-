"""
LIVE TRADING DASHBOARD - Enhanced Real-time Visualization
==========================================================
Modern, clean dashboard with comprehensive trading metrics
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

class LiveTradingDashboard:
    """
    Enhanced Real-time Dashboard for Live Trading
    
    Features:
    - Large, clear price chart with trade markers
    - Portfolio value tracking
    - Profit/Loss visualization
    - Win rate and trade statistics
    - Current position indicator
    - Performance metrics
    """
    
    def __init__(self, log_file: str = "logs/live_trading_log.json"):
        self.log_file = log_file
        self.last_update = None
        
        # Set style for better looks
        plt.style.use('dark_background')
        
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.patch.set_facecolor('#0a0a0a')
        
        # Main title
        self.fig.suptitle('🤖 LIVE PAPER TRADING DASHBOARD', 
                         fontsize=20, fontweight='bold', color='#00ff88')
        
        # Create grid layout (3 rows, 3 columns)
        gs = self.fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                                   left=0.05, right=0.98, top=0.94, bottom=0.05)
        
        # Large price chart (top, full width)
        self.ax_price = self.fig.add_subplot(gs[0, :])
        
        # Portfolio value (middle left, spans 2 cols)
        self.ax_portfolio = self.fig.add_subplot(gs[1, :2])
        
        # Current stats box (middle right)
        self.ax_stats = self.fig.add_subplot(gs[1, 2])
        self.ax_stats.axis('off')
        
        # P/L chart (bottom left, spans 2 cols)
        self.ax_pl = self.fig.add_subplot(gs[2, :2])
        
        # Trade metrics (bottom right)
        self.ax_metrics = self.fig.add_subplot(gs[2, 2])
        self.ax_metrics.axis('off')
    
    def load_data(self):
        """Load latest data from log file"""
        if not os.path.exists(self.log_file):
            return None
        
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            self.last_update = datetime.now()
            return data
        except:
            return None
    
    def _format_currency(self, value):
        """Format value as currency"""
        return f"${value:,.2f}"
    
    def _format_percent(self, value):
        """Format value as percentage"""
        return f"{value:+.2f}%"
    
    def update(self, frame):
        """Update all dashboard components"""
        data = self.load_data()
        
        if data is None:
            # Show waiting message
            for ax in [self.ax_price, self.ax_portfolio, self.ax_pl]:
                ax.clear()
                ax.text(0.5, 0.5, '⏳ Waiting for trading data...\n\nStart live_paper_trading.py first!',
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, color='yellow')
                ax.set_xticks([])
                ax.set_yticks([])
            return
        
        portfolio_history = data.get('portfolio_history', [])
        trades = data.get('trades', [])
        config = data.get('config', {})
        
        if not portfolio_history:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        initial_cash = config.get('initial_cash', 10000)
        
        # ========================================
        # 1. PRICE CHART WITH TRADES (TOP)
        # ========================================
        self.ax_price.clear()
        
        # Plot price
        self.ax_price.plot(df['timestamp'], df['price'], 
                          color='#00d4ff', linewidth=2.5, label='Price', zorder=1)
        
        # Add trades
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            buys = trades_df[trades_df['action'] == 'BUY']
            sells = trades_df[trades_df['action'] == 'SELL']
            
            if not buys.empty:
                self.ax_price.scatter(buys['timestamp'], buys['price'], 
                                     color='#00ff00', marker='^', s=300, 
                                     label='BUY', zorder=5, edgecolors='white', linewidths=2)
            
            if not sells.empty:
                self.ax_price.scatter(sells['timestamp'], sells['price'], 
                                     color='#ff0000', marker='v', s=300, 
                                     label='SELL', zorder=5, edgecolors='white', linewidths=2)
        
        # Styling
        self.ax_price.set_title('💰 PRICE CHART', fontsize=14, fontweight='bold', 
                               color='#00ff88', pad=10)
        self.ax_price.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
        self.ax_price.legend(loc='upper left', fontsize=10, framealpha=0.9)
        self.ax_price.grid(True, alpha=0.2, linestyle='--')
        self.ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Current price annotation
        current_price = df['price'].iloc[-1]
        self.ax_price.axhline(y=current_price, color='#00d4ff', 
                             linestyle='--', alpha=0.5, linewidth=1)
        self.ax_price.text(0.99, 0.95, f'Current: ${current_price:,.2f}',
                          transform=self.ax_price.transAxes,
                          fontsize=12, fontweight='bold', color='#00d4ff',
                          ha='right', va='top',
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # ========================================
        # 2. PORTFOLIO VALUE (MIDDLE LEFT)
        # ========================================
        self.ax_portfolio.clear()
        
        # Plot portfolio value
        self.ax_portfolio.plot(df['timestamp'], df['portfolio_value'],
                              color='#ffaa00', linewidth=3, label='Portfolio Value')
        
        # Initial capital line
        self.ax_portfolio.axhline(y=initial_cash, color='white', 
                                 linestyle='--', linewidth=1.5, 
                                 label='Initial Capital', alpha=0.6)
        
        # Fill areas
        self.ax_portfolio.fill_between(df['timestamp'], initial_cash, df['portfolio_value'],
                                       where=df['portfolio_value'] >= initial_cash,
                                       color='#00ff00', alpha=0.2, label='Profit')
        self.ax_portfolio.fill_between(df['timestamp'], initial_cash, df['portfolio_value'],
                                       where=df['portfolio_value'] < initial_cash,
                                       color='#ff0000', alpha=0.2, label='Loss')
        
        # Styling
        self.ax_portfolio.set_title('📊 PORTFOLIO VALUE', fontsize=14, 
                                   fontweight='bold', color='#00ff88', pad=10)
        self.ax_portfolio.set_ylabel('Value ($)', fontsize=11, fontweight='bold')
        self.ax_portfolio.legend(loc='upper left', fontsize=9, framealpha=0.9)
        self.ax_portfolio.grid(True, alpha=0.2, linestyle='--')
        self.ax_portfolio.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Current value annotation
        current_value = df['portfolio_value'].iloc[-1]
        self.ax_portfolio.text(0.99, 0.95, f'${current_value:,.2f}',
                              transform=self.ax_portfolio.transAxes,
                              fontsize=13, fontweight='bold', color='#ffaa00',
                              ha='right', va='top',
                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # ========================================
        # 3. CURRENT STATUS BOX (MIDDLE RIGHT)
        # ========================================
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        latest = df.iloc[-1]
        
        # Position indicator
        position_text = '🟢 LONG' if latest['position'] > 0 else '⚪ FLAT'
        position_color = '#00ff00' if latest['position'] > 0 else '#666666'
        
        # P/L color
        pl_color = '#00ff00' if latest['profit'] >= 0 else '#ff0000'
        
        stats_text = f"""
╔═══════════════════════════╗
║    CURRENT STATUS         ║
╠═══════════════════════════╣
║                           ║
║  Position: {position_text}         ║
║                           ║
║  Portfolio Value          ║
║  {self._format_currency(latest['portfolio_value'])}              ║
║                           ║
║  Total P/L                ║
║  {self._format_currency(latest['profit'])} ({self._format_percent(latest['profit_pct'])})     ║
║                           ║
║  Cash: {self._format_currency(latest['cash'])}         ║
║  Coins: {latest['coins']:.6f}        ║
║                           ║
║  Total Trades: {len(trades)}          ║
║                           ║
║  Last Update              ║
║  {latest['timestamp'].strftime('%H:%M:%S')}                  ║
║                           ║
╚═══════════════════════════╝
        """
        
        self.ax_stats.text(0.5, 0.5, stats_text,
                          transform=self.ax_stats.transAxes,
                          fontsize=10, fontfamily='monospace',
                          ha='center', va='center', color='white',
                          bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                                   edgecolor=position_color, linewidth=3, alpha=0.9))
        
        # ========================================
        # 4. PROFIT/LOSS CHART (BOTTOM LEFT)
        # ========================================
        self.ax_pl.clear()
        
        # Plot P/L
        self.ax_pl.plot(df['timestamp'], df['profit'], 
                       color='#ff00ff', linewidth=3)
        
        # Zero line
        self.ax_pl.axhline(y=0, color='white', linestyle='-', 
                          linewidth=1, alpha=0.5)
        
        # Fill areas
        self.ax_pl.fill_between(df['timestamp'], 0, df['profit'],
                               where=df['profit'] >= 0,
                               color='#00ff00', alpha=0.3)
        self.ax_pl.fill_between(df['timestamp'], 0, df['profit'],
                               where=df['profit'] < 0,
                               color='#ff0000', alpha=0.3)
        
        # Styling
        self.ax_pl.set_title('📈 CUMULATIVE PROFIT/LOSS', fontsize=14, 
                            fontweight='bold', color='#00ff88', pad=10)
        self.ax_pl.set_ylabel('P/L ($)', fontsize=11, fontweight='bold')
        self.ax_pl.set_xlabel('Time', fontsize=11, fontweight='bold')
        self.ax_pl.grid(True, alpha=0.2, linestyle='--')
        self.ax_pl.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        # Current P/L annotation
        current_pl = df['profit'].iloc[-1]
        pl_text_color = '#00ff00' if current_pl >= 0 else '#ff0000'
        self.ax_pl.text(0.99, 0.95, f'{self._format_currency(current_pl)}',
                       transform=self.ax_pl.transAxes,
                       fontsize=13, fontweight='bold', color=pl_text_color,
                       ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # ========================================
        # 5. TRADE METRICS (BOTTOM RIGHT)
        # ========================================
        self.ax_metrics.clear()
        self.ax_metrics.axis('off')
        
        # Calculate metrics
        total_trades = len(trades)
        
        if total_trades > 0:
            # Win rate calculation (simplified - based on profitable sells)
            profitable_trades = 0
            for trade in trades:
                if trade['action'] == 'SELL' and 'profit' in trade:
                    if trade['profit'] > 0:
                        profitable_trades += 1
            
            win_rate = (profitable_trades / max(1, total_trades//2)) * 100 if total_trades >= 2 else 0
            
            # Average profit per trade
            avg_profit = latest['profit'] / max(1, total_trades)
            
            # Best/worst trade
            profits = [t.get('profit', 0) for t in trades if 'profit' in t]
            best_trade = max(profits) if profits else 0
            worst_trade = min(profits) if profits else 0
        else:
            win_rate = 0
            avg_profit = 0
            best_trade = 0
            worst_trade = 0
        
        # Return %
        return_pct = latest['profit_pct']
        
        metrics_text = f"""
╔═══════════════════════════╗
║   PERFORMANCE METRICS     ║
╠═══════════════════════════╣
║                           ║
║  Total Return             ║
║  {self._format_percent(return_pct)}                 ║
║                           ║
║  Total Trades: {total_trades}          ║
║                           ║
║  Win Rate: {win_rate:.1f}%          ║
║                           ║
║  Avg P/L per Trade        ║
║  {self._format_currency(avg_profit)}              ║
║                           ║
║  Best Trade               ║
║  {self._format_currency(best_trade)}              ║
║                           ║
║  Worst Trade              ║
║  {self._format_currency(worst_trade)}              ║
║                           ║
║  Trading Since            ║
║  {df['timestamp'].iloc[0].strftime('%H:%M:%S')}                  ║
║                           ║
╚═══════════════════════════╝
        """
        
        # Color based on performance
        border_color = '#00ff00' if return_pct >= 0 else '#ff0000'
        
        self.ax_metrics.text(0.5, 0.5, metrics_text,
                            transform=self.ax_metrics.transAxes,
                            fontsize=10, fontfamily='monospace',
                            ha='center', va='center', color='white',
                            bbox=dict(boxstyle='round', facecolor='#1a1a1a', 
                                     edgecolor=border_color, linewidth=3, alpha=0.9))
    
    def start(self, interval: int = 5000):
        """
        Start live dashboard
        
        Args:
            interval: Update interval in milliseconds (default 5000 = 5 seconds)
        """
        print("\n" + "="*70)
        print("LIVE TRADING DASHBOARD")
        print("="*70)
        print(f"Monitoring: {self.log_file}")
        print(f"Update interval: {interval/1000:.1f} seconds")
        print("\nClose the window to stop monitoring.")
        print("="*70 + "\n")
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig, 
            self.update, 
            interval=interval,
            cache_frame_data=False
        )
        
        plt.show()


def main():
    """Main function"""
    dashboard = LiveTradingDashboard()
    dashboard.start(interval=5000)  # Update every 5 seconds


if __name__ == "__main__":
    main()