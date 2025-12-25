"""
LIVE PAPER TRADING - Real-time trading with trained DQN model
==============================================================
Uses best_model.pth to trade on live data
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from datetime import datetime, timedelta
from collections import deque

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from agents.dqn_agent import Agent
from utils.indicators import TechnicalIndicators
from sklearn.preprocessing import RobustScaler

# ========================================
# CONFIGURATION
# ========================================

LIVE_CONFIG = {
    "symbol": "BTC-USD",
    "interval": "1m",  # 1-minute bars for live trading
    "window_size": 24,
    "initial_cash": 10000.0,
    "fee": 0.001,
    "slippage": 0.0005,
    "update_interval": 60,  # Check every 60 seconds
    "model_path": "../models/best_model.pth",
    "log_file": "../logs/live_trading_log.json"
}

# ========================================
# LIVE TRADER CLASS
# ========================================

class LivePaperTrader:
    """
    Live Paper Trading Bot
    
    Features:
    - Loads trained DQN model
    - Fetches live data every minute
    - Calculates technical indicators
    - Makes trading decisions
    - Tracks portfolio in real-time
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.symbol = config['symbol']
        self.interval = config['interval']
        self.window_size = config['window_size']
        
        # Portfolio state
        self.cash = config['initial_cash']
        self.coins = 0.0
        self.position = 0.0
        self.entry_price = 0.0
        
        # Historical data storage
        self.price_history = deque(maxlen=200)  # Keep last 200 bars
        self.feature_history = deque(maxlen=200)
        
        # Scaler (will be fitted on initial data)
        self.scaler = RobustScaler()
        self.scaler_fitted = False
        
        # Trades log
        self.trades = []
        self.portfolio_history = []
        
        # Load model
        self._load_model()
        
        print("\n" + "="*70)
        print("LIVE PAPER TRADING BOT INITIALIZED")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Interval: {self.interval}")
        print(f"Initial Cash: ${self.cash:,.2f}")
        print(f"Model: {config['model_path']}")
        print("="*70 + "\n")
    
    def _load_model(self):
        """Load trained DQN model"""
        model_path = self.config['model_path']
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # CRITICAL: Extract exact configuration from checkpoint
        # This ensures we match the training setup exactly
        if 'config' in checkpoint:
            training_config = checkpoint['config']
            trained_window_size = training_config.get('data', {}).get('window_size', 24)
            print(f"   Detected training window size: {trained_window_size}")
            
            # Override our window size to match training
            self.window_size = trained_window_size
            self.config['window_size'] = trained_window_size
        
        # Calculate input dimension from checkpoint weights
        # This is the most reliable way to get the exact feature count
        first_layer_weight = checkpoint['model_state_dict']['feature_layer.0.weight']
        input_dim = first_layer_weight.shape[1]  # e.g., 1155
        
        # Back-calculate state_size from input_dim
        # input_dim = (state_size * window_size) + 3 portfolio features
        # So: state_size = (input_dim - 3) / window_size
        state_size = (input_dim - 3) // self.window_size
        
        print(f"   Detected input dimension: {input_dim}")
        print(f"   Calculated state_size: {state_size}")
        print(f"   Window size: {self.window_size}")
        
        # Initialize agent with EXACT same parameters
        self.agent = Agent(
            state_size=state_size,
            action_size=3,
            window_size=self.window_size,
            is_eval=True  # Disable exploration
        )
        
        # Load weights - should work now!
        self.agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.agent.epsilon = 0.0  # No exploration in live trading!
        
        print(f"✅ Model loaded successfully")
        print(f"   Training Episode: {checkpoint.get('episode', 'Unknown')}")
        print(f"   Portfolio at save: ${checkpoint.get('portfolio_value', 0):,.2f}\n")
        
        # Store for later use
        self.state_size = state_size
    
    def _fetch_latest_data(self) -> pd.DataFrame:
        """
        Fetch latest price data from Yahoo Finance
        
        Returns:
            DataFrame with OHLCV data
        """
        # Get data for last 200 bars to ensure enough for indicators
        end_time = datetime.now()
        
        # Calculate start time based on interval
        if self.interval == '1m':
            start_time = end_time - timedelta(minutes=250)
        elif self.interval == '5m':
            start_time = end_time - timedelta(minutes=250*5)
        elif self.interval == '15m':
            start_time = end_time - timedelta(minutes=250*15)
        elif self.interval == '1h':
            start_time = end_time - timedelta(hours=250)
        else:
            start_time = end_time - timedelta(days=30)
        
        try:
            df = yf.download(
                self.symbol,
                start=start_time,
                end=end_time,
                interval=self.interval,
                progress=False
            )
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            
            if len(df) < 100:
                print(f"⚠️  Warning: Only {len(df)} bars fetched")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all 42 technical indicators"""
        df = TechnicalIndicators.add_all_indicators(df)
        
        # Add cyclical time features
        if 'Date' in df.columns or 'Datetime' in df.columns:
            date_col = 'Date' if 'Date' in df.columns else 'Datetime'
            df[date_col] = pd.to_datetime(df[date_col])
            
            df['month_sin'] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df[date_col].dt.month / 12)
            df['day_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofweek / 7)
            df['hour_sin'] = np.sin(2 * np.pi * df[date_col].dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df[date_col].dt.hour / 24)
        else:
            df['month_sin'] = df['month_cos'] = 0.0
            df['day_sin'] = df['day_cos'] = 0.0
            df['hour_sin'] = df['hour_cos'] = 0.0
        
        # Placeholder for supervised signal (we don't retrain in live trading)
        df['Trend_Prob'] = 0.5
        
        return df
    
    def _get_feature_columns(self) -> list:
        """Get all feature column names - dynamically based on what exists in data"""
        base_features = TechnicalIndicators.get_feature_columns()
        time_features = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos']
        
        all_features = base_features + time_features
        
        # If we know the exact state_size from model, we can verify
        if hasattr(self, 'state_size'):
            expected_features = self.state_size
            if len(all_features) != expected_features:
                print(f"⚠️  Warning: Feature count mismatch!")
                print(f"   Expected: {expected_features} features")
                print(f"   Generated: {len(all_features)} features")
                print(f"   This might cause issues - using what we have...")
        
        return all_features
    
    def _prepare_observation(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare observation for agent
        
        Args:
            df: DataFrame with all features
        
        Returns:
            1D array ready for agent with EXACT dimensions expected by model
        """
        feature_cols = self._get_feature_columns()
        
        # Filter to only existing columns
        existing_cols = [col for col in feature_cols if col in df.columns]
        
        # CRITICAL: Ensure we have the right number of features
        if hasattr(self, 'state_size'):
            if len(existing_cols) < self.state_size:
                print(f"⚠️  Warning: Missing features!")
                print(f"   Need {self.state_size}, have {len(existing_cols)}")
                print(f"   Padding with zeros...")
                
                # Pad missing features with zeros
                missing_count = self.state_size - len(existing_cols)
                for i in range(missing_count):
                    col_name = f'padding_{i}'
                    df[col_name] = 0.0
                    existing_cols.append(col_name)
            
            elif len(existing_cols) > self.state_size:
                print(f"⚠️  Warning: Too many features!")
                print(f"   Need {self.state_size}, have {len(existing_cols)}")
                print(f"   Truncating to first {self.state_size}...")
                existing_cols = existing_cols[:self.state_size]
        
        # Get latest window_size bars
        if len(df) < self.window_size:
            # Pad if needed
            padding_needed = self.window_size - len(df)
            real_data = df[existing_cols].values
            padding = np.tile(real_data[0], (padding_needed, 1))
            window_data = np.vstack([padding, real_data])
        else:
            window_data = df.iloc[-self.window_size:][existing_cols].values
        
        # Scale features
        if not self.scaler_fitted:
            # Fit scaler on first batch
            all_data = df[existing_cols].values
            self.scaler.fit(all_data)
            self.scaler_fitted = True
            print(f"✅ Feature scaler fitted ({len(existing_cols)} features)\n")
        
        # Transform window
        window_scaled = self.scaler.transform(window_data)
        
        # Flatten
        flat_window = window_scaled.flatten()
        
        # Add portfolio features
        current_price = df['Close'].iloc[-1]
        
        if self.position > 0 and self.entry_price > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
        else:
            unrealized_pnl = 0.0
        
        portfolio_features = np.array([
            self.cash / self.config['initial_cash'],
            self.position,
            unrealized_pnl
        ], dtype=np.float32)
        
        # Concatenate
        observation = np.concatenate([flat_window, portfolio_features])
        
        # FINAL CHECK: Verify dimension
        expected_dim = (self.state_size * self.window_size) + 3
        if len(observation) != expected_dim:
            print(f"❌ ERROR: Observation dimension mismatch!")
            print(f"   Expected: {expected_dim}")
            print(f"   Got: {len(observation)}")
            raise ValueError(f"Observation dimension mismatch: {len(observation)} != {expected_dim}")
        
        return observation
    
    def _get_action_mask(self) -> np.ndarray:
        """Get valid actions mask"""
        mask = np.ones(3, dtype=np.int8)
        
        if self.position >= 0.99:
            mask[1] = 0  # Can't buy
        
        if self.position <= 0.01:
            mask[2] = 0  # Can't sell
        
        return mask
    
    def _execute_trade(self, action: int, current_price: float):
        """
        Execute trade action
        
        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            current_price: Current market price
        """
        action_names = ['HOLD', 'BUY', 'SELL']
        
        # BUY
        if action == 1 and self.cash > 0:
            exec_price = current_price * (1 + self.config['slippage'])
            fee_amount = self.cash * self.config['fee']
            cost = self.cash - fee_amount
            
            self.coins = cost / exec_price
            self.cash = 0.0
            self.position = 1.0
            self.entry_price = exec_price
            
            trade = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'action': 'BUY',
                'price': exec_price,
                'coins': self.coins,
                'portfolio_value': self.get_portfolio_value(current_price)
            }
            self.trades.append(trade)
            
            print(f"🟢 BUY  @ ${exec_price:,.2f} | Coins: {self.coins:.6f}")
        
        # SELL
        elif action == 2 and self.coins > 0:
            exec_price = current_price * (1 - self.config['slippage'])
            gross_value = self.coins * exec_price
            fee_amount = gross_value * self.config['fee']
            
            self.cash = gross_value - fee_amount
            profit = self.cash - self.config['initial_cash']
            
            self.coins = 0.0
            self.position = 0.0
            self.entry_price = 0.0
            
            trade = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'action': 'SELL',
                'price': exec_price,
                'cash': self.cash,
                'profit': profit,
                'portfolio_value': self.get_portfolio_value(current_price)
            }
            self.trades.append(trade)
            
            print(f"🔴 SELL @ ${exec_price:,.2f} | Cash: ${self.cash:,.2f} | P/L: ${profit:+,.2f}")
        
        # HOLD
        else:
            print(f"⚪ HOLD @ ${current_price:,.2f}")
    
    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        return self.cash + (self.coins * current_price)
    
    def _log_status(self, current_price: float):
        """Log current status"""
        portfolio_value = self.get_portfolio_value(current_price)
        profit = portfolio_value - self.config['initial_cash']
        profit_pct = (profit / self.config['initial_cash']) * 100
        
        status = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'price': current_price,
            'portfolio_value': portfolio_value,
            'profit': profit,
            'profit_pct': profit_pct,
            'position': self.position,
            'cash': self.cash,
            'coins': self.coins
        }
        
        self.portfolio_history.append(status)
        
        print(f"\n📊 Portfolio: ${portfolio_value:,.2f} | P/L: ${profit:+,.2f} ({profit_pct:+.2f}%)")
        print(f"   Position: {self.position:.1%} | Cash: ${self.cash:,.2f} | Coins: {self.coins:.6f}")
    
    def _save_logs(self):
        """Save trading logs to file"""
        logs = {
            'config': self.config,
            'trades': self.trades,
            'portfolio_history': self.portfolio_history,
            'final_stats': {
                'total_trades': len(self.trades),
                'final_portfolio': self.portfolio_history[-1] if self.portfolio_history else None
            }
        }
        
        log_file = self.config['log_file']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        print(f"\n💾 Logs saved to: {log_file}")
    
    def run(self, duration_minutes: int = 60):
        """
        Run live paper trading
        
        Args:
            duration_minutes: How long to run (in minutes)
        """
        print(f"\n🚀 Starting Live Paper Trading")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Update interval: {self.config['update_interval']} seconds\n")
        print("="*70)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        iteration = 0
        
        try:
            while datetime.now() < end_time:
                iteration += 1
                current_time = datetime.now()
                
                print(f"\n[{current_time.strftime('%H:%M:%S')}] Iteration {iteration}")
                print("-" * 70)
                
                # 1. Fetch latest data
                print("📥 Fetching latest data...")
                df = self._fetch_latest_data()
                
                if df is None or len(df) < 50:
                    print("⚠️  Insufficient data, waiting...")
                    time.sleep(self.config['update_interval'])
                    continue
                
                # 2. Add indicators
                df = self._add_technical_indicators(df)
                
                # 3. Prepare observation
                observation = self._prepare_observation(df)
                
                # 4. Get current price
                current_price = df['Close'].iloc[-1]
                print(f"💰 Current Price: ${current_price:,.2f}")
                
                # 5. Get action from agent
                action_mask = self._get_action_mask()
                action = self.agent.act(observation, action_mask=action_mask)
                
                # 6. Execute trade
                self._execute_trade(action, current_price)
                
                # 7. Log status
                self._log_status(current_price)
                
                # 8. Save logs
                self._save_logs()
                
                # 9. Wait for next update
                remaining_time = self.config['update_interval']
                print(f"\n⏳ Next update in {remaining_time}s...")
                time.sleep(remaining_time)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Trading interrupted by user")
        
        finally:
            # Final report
            print("\n" + "="*70)
            print("LIVE TRADING SESSION COMPLETE")
            print("="*70)
            
            if self.portfolio_history:
                final = self.portfolio_history[-1]
                print(f"\nFinal Portfolio: ${final['portfolio_value']:,.2f}")
                print(f"Total Profit/Loss: ${final['profit']:+,.2f} ({final['profit_pct']:+.2f}%)")
                print(f"Total Trades: {len(self.trades)}")
                
                if self.trades:
                    print(f"\nTrade Summary:")
                    for i, trade in enumerate(self.trades, 1):
                        print(f"  {i}. {trade['timestamp']} - {trade['action']} @ ${trade['price']:,.2f}")
            
            self._save_logs()
            print("\n" + "="*70 + "\n")


# ========================================
# MAIN
# ========================================

def main():
    """Main function"""
    print("\n" + "#"*70)
    print("  LIVE PAPER TRADING BOT")
    print("#"*70 + "\n")
    
    # Check if model exists
    if not os.path.exists(LIVE_CONFIG['model_path']):
        print(f"❌ ERROR: Model not found at {LIVE_CONFIG['model_path']}")
        print("Please train the model first using train_enhanced_dqn.py")
        return
    
    # Initialize trader
    trader = LivePaperTrader(LIVE_CONFIG)
    
    # Ask user for duration
    print("\n" + "="*70)
    print("How long would you like to run live trading?")
    print("="*70)
    print("Options:")
    print("  1. Quick Test (5 minutes)")
    print("  2. Short Session (30 minutes)")
    print("  3. Normal Session (60 minutes)")
    print("  4. Long Session (120 minutes)")
    print("  5. Custom duration")
    
    choice = input("\nEnter choice (1-5) [default: 1]: ").strip() or "1"
    
    duration_map = {
        '1': 5,
        '2': 30,
        '3': 60,
        '4': 120
    }
    
    if choice in duration_map:
        duration = duration_map[choice]
    elif choice == '5':
        duration = int(input("Enter duration in minutes: "))
    else:
        duration = 5
        print("Invalid choice, using 5 minutes")
    
    # Run trading
    trader.run(duration_minutes=duration)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()