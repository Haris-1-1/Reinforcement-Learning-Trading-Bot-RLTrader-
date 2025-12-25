"""
LIVE TRADING DASHBOARD - Real-time visualization
=================================================
Shows live trading performance and portfolio evolution
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import pandas as pd
import numpy as np

class LiveTradingDashboard:
    """
    Real-time dashboard for live trading
    
    Shows:
    - Price chart with trades
    - Portfolio value over time
    - P/L evolution
    - Current position
    """
    
    def __init__(self, log_file: str = "logs/live_trading_log.json"):
        self.log_file = log_file
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Live Paper Trading Dashboard', fontsize=16, fontweight='bold')
        
        # Subplots
        self.ax1 = plt.subplot(2, 2, 1)  # Price + Trades
        self.ax2 = plt.subplot(2, 2, 2)  # Portfolio Value
        self.ax3 = plt.subplot(2, 2, 3)  # P/L
        self.ax4 = plt.subplot(2, 2, 4)  # Stats
        
        plt.tight_layout()
    
    def load_data(self):
        """Load latest data from log file"""
        if not os.path.exists(self.log_file):
            return None
        
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            return data
        except:
            return None
    
    def update(self, frame):
        """Update dashboard (called by animation)"""
        data = self.load_data()
        
        if data is None:
            return
        
        portfolio_history = data.get('portfolio_history', [])
        trades = data.get('trades', [])
        
        if not portfolio_history:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(portfolio_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Clear all axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # --- PLOT 1: Price Chart with Trades ---
        self.ax1.plot(df['timestamp'], df['price'], 'k-', linewidth=2, label='Price')
        
        # Add trades
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            buys = trades_df[trades_df['action'] == 'BUY']
            sells = trades_df[trades_df['action'] == 'SELL']
            
            if not buys.empty:
                self.ax1.scatter(buys['timestamp'], buys['price'], 
                               color='green', marker='^', s=200, 
                               label='Buy', zorder=5, edgecolors='black', linewidths=2)
            
            if not sells.empty:
                self.ax1.scatter(sells['timestamp'], sells['price'], 
                               color='red', marker='v', s=200, 
                               label='Sell', zorder=5, edgecolors='black', linewidths=2)
        
        self.ax1.set_xlabel('Time', fontsize=10)
        self.ax1.set_ylabel('Price ($)', fontsize=10)
        self.ax1.set_title('Price Chart with Trades', fontsize=12, fontweight='bold')
        self.ax1.legend(loc='best')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.tick_params(axis='x', rotation=45)
        
        # --- PLOT 2: Portfolio Value ---
        self.ax2.plot(df['timestamp'], df['portfolio_value'], 
                     'b-', linewidth=3, label='Portfolio Value')
        
        initial_cash = data.get('config', {}).get('initial_cash', 10000)
        self.ax2.axhline(y=initial_cash, color='r', linestyle='--', 
                        linewidth=2, label='Initial Capital', alpha=0.7)
        
        # Fill area
        self.ax2.fill_between(df['timestamp'], initial_cash, df['portfolio_value'],
                             where=df['portfolio_value'] >= initial_cash,
                             color='green', alpha=0.3, label='Profit')
        self.ax2.fill_between(df['timestamp'], initial_cash, df['portfolio_value'],
                             where=df['portfolio_value'] < initial_cash,
                             color='red', alpha=0.3, label='Loss')
        
        self.ax2.set_xlabel('Time', fontsize=10)
        self.ax2.set_ylabel('Value ($)', fontsize=10)
        self.ax2.set_title('Portfolio Value Evolution', fontsize=12, fontweight='bold')
        self.ax2.legend(loc='best')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.tick_params(axis='x', rotation=45)
        
        # --- PLOT 3: P/L Over Time ---
        self.ax3.plot(df['timestamp'], df['profit'], 'purple', linewidth=3)
        self.ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        self.ax3.fill_between(df['timestamp'], 0, df['profit'],
                             where=df['profit'] >= 0,
                             color='green', alpha=0.3)
        self.ax3.fill_between(df['timestamp'], 0, df['profit'],
                             where=df['profit'] < 0,
                             color='red', alpha=0.3)
        
        self.ax3.set_xlabel('Time', fontsize=10)
        self.ax3.set_ylabel('Profit/Loss ($)', fontsize=10)
        self.ax3.set_title('Cumulative P/L', fontsize=12, fontweight='bold')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.tick_params(axis='x', rotation=45)
        
        # --- PLOT 4: Current Stats ---
        self.ax4.axis('off')
        
        latest = df.iloc[-1]
        
        stats_text = f"""
        ╔══════════════════════════════════════╗
        ║      LIVE TRADING STATISTICS         ║
        ╠══════════════════════════════════════╣
        ║                                      ║
        ║  Current Price                       ║
        ║     ${latest['price']:,.2f}                           ║
        ║                                      ║
        ║  Portfolio Value                     ║
        ║     ${latest['portfolio_value']:,.2f}                           ║
        ║                                      ║
        ║  Total P/L                           ║
        ║     ${latest['profit']:+,.2f} ({latest['profit_pct']:+.2f}%)               ║
        ║                                      ║
        ║  Cash                                ║
        ║     ${latest['cash']:,.2f}                           ║
        ║                                      ║
        ║  Coins                               ║
        ║     {latest['coins']:.6f}                        ║
        ║                                      ║
        ║  Position                            ║
        ║     {latest['position']:.1%}                              ║
        ║                                      ║
        ║  Total Trades                        ║
        ║     {len(trades)}                                  ║
        ║                                      ║
        ║  Last Update                         ║
        ║     {latest['timestamp'].strftime('%H:%M:%S')}                           ║
        ║                                      ║
        ╚══════════════════════════════════════╝
        """
        
        # Color code P/L
        color = 'green' if latest['profit'] >= 0 else 'red'
        
        self.ax4.text(0.5, 0.5, stats_text, 
                     transform=self.ax4.transAxes,
                     fontsize=11,
                     verticalalignment='center',
                     horizontalalignment='center',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Add status indicator
        status_color = 'green' if latest['position'] > 0 else 'gray'
        status_text = 'LONG' if latest['position'] > 0 else 'FLAT'
        
        self.ax4.text(0.5, 0.95, status_text,
                     transform=self.ax4.transAxes,
                     fontsize=14, fontweight='bold',
                     verticalalignment='top',
                     horizontalalignment='center',
                     color=status_color)
        
        plt.tight_layout()
    
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
