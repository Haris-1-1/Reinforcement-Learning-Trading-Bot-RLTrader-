"""
Advanced Trading Environment with realistic constraints
Includes maker/taker fees, slippage, drawdown penalties, and multi-timeframe features
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import gymnasium as gym
from gymnasium import spaces
from torch import seed


class AdvancedTradingEnv(gym.Env):
    """
    Advanced trading environment for reinforcement learning
    
    Features:
    - Realistic transaction costs (maker/taker fees)
    - Slippage simulation
    - Drawdown penalties
    - Multi-timeframe features
    - Position sizing
    - Dynamic stop-loss based on ATR
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        maker_fee: float = 0.001,  # 0.1%
        taker_fee: float = 0.002,  # 0.2%
        slippage: float = 0.001,  # 0.1%
        max_drawdown_penalty: float = 0.5,
        sequence_length: int = 30,
        feature_columns: list = None,
        reward_scaling: float = 1.0,
        use_atr_stop: bool = True,
        atr_multiplier: float = 2.0
    ):
        """
        Args:
            data: DataFrame with OHLCV and indicator data
            initial_balance: Starting capital
            maker_fee: Maker fee (limit orders)
            taker_fee: Taker fee (market orders)
            slippage: Slippage factor
            max_drawdown_penalty: Penalty multiplier for exceeding max drawdown
            sequence_length: Length of observation sequences
            feature_columns: List of feature column names
            reward_scaling: Factor to scale rewards
            use_atr_stop: Whether to use ATR-based stop loss
            atr_multiplier: Multiplier for ATR stop loss
        """
        super(AdvancedTradingEnv, self).__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.max_drawdown_penalty = max_drawdown_penalty
        self.sequence_length = sequence_length
        self.reward_scaling = reward_scaling
        self.use_atr_stop = use_atr_stop
        self.atr_multiplier = atr_multiplier
        
        # Feature columns
        if feature_columns is None:
            # Use all numeric columns except OHLCV
            self.feature_columns = [
                col for col in data.columns 
                if col not in ['open', 'high', 'low', 'close', 'volume']
                and pd.api.types.is_numeric_dtype(data[col])
            ]
        else:
            self.feature_columns = feature_columns
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: sequences of features
        # FIXED: Use scalar bounds, not arrays
        num_features = len(self.feature_columns)
        obs_shape = (sequence_length, num_features)

        self.observation_space = spaces.Box(
            low=np.ones(obs_shape, dtype=np.float32) * -1e8,
            high=np.ones(obs_shape, dtype=np.float32) * 1e8,
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.position = 0  # 0 = no position, 1 = long
        self.entry_price = 0
        self.stop_loss_price = 0
        
        # Performance tracking
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        self.episode_profit = 0
        self.trades = []
        self.net_worth_history = []
        self.drawdown_history = []
        
        # Reset environment
        self.reset()
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state
    
        Returns:
        Initial observation and info dict
        """
        super().reset(seed=seed)
    
    # Start from a random point that allows for sequence_length history
        max_start = len(self.data) - 1000
    
    # FIX: Ensure max_start is greater than sequence_length
        if max_start <= self.sequence_length:
            max_start = len(self.data) - 1  # Use almost all data
    
        if max_start <= self.sequence_length:
        # Data is too short, just start after sequence_length
            self.current_step = self.sequence_length
        else:
            self.current_step = np.random.randint(
            self.sequence_length,
            max_start
        )
           # Reset account
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.position = 0
        self.entry_price = 0
        self.stop_loss_price = 0
    
    # Reset performance tracking
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.episode_profit = 0
        self.trades = []
        self.net_worth_history = [self.initial_balance]
        self.drawdown_history = [0]

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation (sequence of features)
        
        Returns:
            Array of shape (sequence_length, num_features)
        """
        start_idx = max(0, self.current_step - self.sequence_length)
        end_idx = self.current_step
        
        # Get feature sequence
        feature_data = self.data[self.feature_columns].iloc[start_idx:end_idx]
        
        # Pad if necessary
        if len(feature_data) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(feature_data), len(self.feature_columns)))
            feature_data = np.vstack([padding, feature_data.values])
        else:
            feature_data = feature_data.values
        
        # Replace NaN and inf values
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_data.astype(np.float32)
    
    def _calculate_stop_loss(self, entry_price: float) -> float:
        """
        Calculate stop loss price based on ATR
        
        Args:
            entry_price: Entry price for the position
            
        Returns:
            Stop loss price
        """
        if not self.use_atr_stop:
            return 0
        
        # Get current ATR
        if 'atr' in self.data.columns:
            atr = self.data['atr'].iloc[self.current_step]
        else:
            # Fallback: use 2% of entry price
            atr = entry_price * 0.02
        
        # Stop loss is entry price minus (ATR * multiplier)
        stop_loss = entry_price - (atr * self.atr_multiplier)
        
        return max(stop_loss, entry_price * 0.90)  # At least 10% stop loss
    
    def _check_stop_loss(self) -> bool:
        """
        Check if stop loss has been hit
        
        Returns:
            True if stop loss hit, False otherwise
        """
        if self.position == 0 or self.stop_loss_price == 0:
            return False
        
        current_price = self._get_current_price()
        
        # Check if price dropped below stop loss
        if current_price <= self.stop_loss_price:
            return True
        
        return False
    
    def _get_current_price(self) -> float:
        """Get current price (close price of current candle)"""
        return self.data['close'].iloc[self.current_step]
    
    def _execute_trade(self, action: int) -> Tuple[float, bool]:
        """
        Execute a trade action
        
        Args:
            action: 0 = hold, 1 = buy, 2 = sell
            
        Returns:
            Tuple of (cost, trade_executed)
        """
        current_price = self._get_current_price()
        cost = 0
        trade_executed = False
        
        # Apply slippage (price moves against us)
        if action == 1:  # Buy
            execution_price = current_price * (1 + self.slippage)
        elif action == 2:  # Sell
            execution_price = current_price * (1 - self.slippage)
        else:
            execution_price = current_price
        
        # IN: advanced_Trading_Bot(Final)/environments/trading_env.py

        if action == 1:  # Buy
            # ... (Der Teil mit DEBUG BUY BLOCKED print ist okay, lass ihn oder lösch ihn) ...
            
            # --- HIER BEGINNT DIE KORREKTUR ---
            
            # 1. Berechne, wie viel Geld wir für den Kauf haben (Abzüglich Gebühren!)
            # Formel: Available = Balance / (1 + fee_rate)
            available_for_purchase = self.balance / (1 + self.taker_fee)
            
            # 2. Berechne Anzahl der Shares
            shares_to_buy = available_for_purchase / execution_price
            
            # 3. Berechne Gebühr und totale Kosten zur Kontrolle
            fee = shares_to_buy * execution_price * self.taker_fee
            cost = (shares_to_buy * execution_price) + fee
            
            # Sicherheitspuffer für Fließkomma-Ungenauigkeiten (Rundungsfehler)
            if cost <= self.balance + 0.01: 
                self.shares_held = shares_to_buy
                self.balance -= cost
                # Falls Balance minimal negativ wird durch Rundung, setze auf 0
                if self.balance < 0: self.balance = 0
                
                self.total_shares_bought += shares_to_buy
                self.position = 1
                self.entry_price = execution_price
                
                # Set stop loss
                self.stop_loss_price = self._calculate_stop_loss(execution_price)
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'price': execution_price,
                    'shares': shares_to_buy,
                    'cost': cost,
                    'balance': self.balance
                })
                
                trade_executed = True
                print(f"✅ BUY EXECUTED! Price: {execution_price:.2f}, Shares: {shares_to_buy:.6f}")
            else:
                print(f"❌ BUY FAILED: Cost {cost:.2f} > Balance {self.balance:.2f}")
        
        elif action == 2:  # Sell
            if self.position == 1 and self.shares_held > 0:
                # Sell all shares
                proceeds = self.shares_held * execution_price
                
                # Apply maker fee (limit order)
                fee = proceeds * self.maker_fee
                proceeds -= fee
                
                # Calculate profit/loss
                cost_basis = self.shares_held * self.entry_price
                profit = proceeds - cost_basis
                
                self.balance += proceeds
                self.total_shares_sold += self.shares_held
                self.shares_held = 0
                self.position = 0
                self.entry_price = 0
                self.stop_loss_price = 0
                self.episode_profit += profit
                
                # Record trade
                self.trades.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': execution_price,
                    'shares': self.shares_held,
                    'proceeds': proceeds,
                    'profit': profit,
                    'balance': self.balance
                })
                
                trade_executed = True
        
        return cost, trade_executed
    
    def _calculate_reward(self, action: int, trade_executed: bool) -> float:
        """
        Calculate reward for the current step
        
        Reward components:
        1. Change in net worth (profit/loss)
        2. Drawdown penalty (if max drawdown exceeded)
        3. Trade execution penalty (transaction costs)
        4. Holding position reward (small positive reward for active trading)
        
        Args:
            action: Action taken
            trade_executed: Whether a trade was executed
            
        Returns:
            Reward value
        """
        # Calculate current net worth
        current_price = self._get_current_price()
        current_net_worth = self.balance + (self.shares_held * current_price)
        
        # Reward component 1: Change in net worth
        net_worth_change = current_net_worth - self.net_worth
        reward = net_worth_change / self.initial_balance  # Normalize by initial balance
        
        # Update net worth
        self.net_worth = current_net_worth
        self.net_worth_history.append(current_net_worth)
        
        # Update max net worth
        if current_net_worth > self.max_net_worth:
            self.max_net_worth = current_net_worth
        
        # Reward component 2: Drawdown penalty
        drawdown = (self.max_net_worth - current_net_worth) / self.max_net_worth
        self.drawdown_history.append(drawdown)
        
        if drawdown > 0.2:  # More than 20% drawdown
            reward -= self.max_drawdown_penalty * drawdown
        
        # Reward component 3: Trade execution penalty (discourages excessive trading)
        if trade_executed:
            reward -= 0  # Small penalty for transaction costs
        
        # Reward component 4: Holding position reward
        if self.position == 1:
            # Small positive reward for being in the market
            reward += 0.001
        
        # Scale reward
        reward *= self.reward_scaling
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step in the environment
        
        Args:
            action: Action to take (0=hold, 1=buy, 2=sell)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Check for stop loss hit
        if self._check_stop_loss():
            # Force sell
            action = 2
        
        # Execute trade
        cost, trade_executed = self._execute_trade(action)
        
        # Calculate reward
        reward = self._calculate_reward(action, trade_executed)
        
        # Move to next step 
        self.current_step += 1 
        
        # Check if episode is done
        terminated = False
        truncated = False
        
        if self.current_step >= len(self.data) - 1:
            truncated = True
        
        # If we're broke, episode is done
        if self.net_worth <= 0:
            terminated = True
            reward = -10  # Large penalty for going broke
        
        # Get next observation
        observation = self._get_observation()
        
        # Info dictionary
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'position': self.position,
            'action': action,
            'trade_executed': trade_executed,
            'current_price': self._get_current_price(),
            'episode_profit': self.episode_profit,
            'total_trades': len(self.trades),
            'max_drawdown': max(self.drawdown_history) if self.drawdown_history else 0
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment
        
        Args:
            mode: Rendering mode
        """
        current_price = self._get_current_price()
        profit = self.net_worth - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        
        print(f"\n{'='*60}")
        print(f"Step: {self.current_step}/{len(self.data)}")
        print(f"{'='*60}")
        print(f"💰 Balance: ${self.balance:,.2f}")
        print(f"📊 Shares Held: {self.shares_held:.4f}")
        print(f"💵 Current Price: ${current_price:,.2f}")
        print(f"💼 Net Worth: ${self.net_worth:,.2f}")
        print(f"📈 Profit: ${profit:,.2f} ({profit_pct:+.2f}%)")
        print(f"🎯 Position: {'LONG' if self.position == 1 else 'NONE'}")
        if self.position == 1:
            print(f"🛡️  Stop Loss: ${self.stop_loss_price:,.2f}")
        print(f"📊 Total Trades: {len(self.trades)}")
        print(f"📉 Max Drawdown: {max(self.drawdown_history)*100:.2f}%")
        print(f"{'='*60}\n")
    
    def get_metrics(self) -> Dict:
        """
        Get performance metrics for the episode
        
        Returns:
            Dictionary of metrics
        """
        profit = self.net_worth - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        
        # Calculate Sharpe ratio
        if len(self.net_worth_history) > 1:
            returns = np.diff(self.net_worth_history) / np.array(self.net_worth_history[:-1])
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        metrics = {
            'final_net_worth': self.net_worth,
            'profit': profit,
            'profit_pct': profit_pct,
            'total_trades': len(self.trades),
            'total_shares_bought': self.total_shares_bought,
            'total_shares_sold': self.total_shares_sold,
            'max_drawdown': max(self.drawdown_history) if self.drawdown_history else 0,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': self._calculate_win_rate()
        }
        
        return metrics
    
    def _calculate_win_rate(self) -> float:
        """
        Calculate win rate (percentage of profitable trades)
        
        Returns:
            Win rate (0-1)
        """
        if not self.trades:
            return 0
        
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        
        if not sell_trades:
            return 0
        
        winning_trades = sum(1 for t in sell_trades if t.get('profit', 0) > 0)
        win_rate = winning_trades / len(sell_trades)
        
        return win_rate


def test_environment():
    """Test the trading environment"""
    print("=" * 60)
    print("Testing Advanced Trading Environment")
    print("=" * 60)
    
    # Create sample data
    import yfinance as yf
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from indicators.technical_indicators import add_all_indicators
    
    print("📥 Fetching data...")
    df = yf.Ticker("BTC-USD").history(period="3mo", interval="1h")
    df.columns = [col.lower() for col in df.columns]
    
    print("📊 Adding indicators...")
    df = add_all_indicators(df)
    
    print("🏗️  Creating environment...")
    env = AdvancedTradingEnv(
        data=df,
        initial_balance=10000,
        sequence_length=30
    )
    
    print(f"\n✅ Environment created:")
    print(f"   - Data shape: {df.shape}")
    print(f"   - Feature columns: {len(env.feature_columns)}")
    print(f"   - Observation shape: {env.observation_space.shape}")
    print(f"   - Action space: {env.action_space}")
    
    # Run a few random steps
    print(f"\n🎮 Testing random actions...")
    obs, _ = env.reset()
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        print(f"\nStep {i+1}:")
        print(f"  Action: {['HOLD', 'BUY', 'SELL'][action]}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Net Worth: ${info['net_worth']:,.2f}")
        print(f"  Position: {info['position']}")
        
        if done:
            break
    
    # Get metrics
    metrics = env.get_metrics()
    print(f"\n📊 Final Metrics:")
    for key, value in metrics.items():
        print(f"   - {key}: {value}")
    
    print(f"\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_environment()