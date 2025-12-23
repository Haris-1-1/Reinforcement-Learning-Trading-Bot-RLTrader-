"""
Baseline trading strategies for benchmarking
Includes: Buy & Hold, Random Trading, Moving Average Crossover
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class BaselineStrategy:
    """Base class for baseline strategies"""
    
    def __init__(self, initial_balance: float = 10000.0):
        """
        Args:
            initial_balance: Starting capital
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.shares_held = 0
        self.position = 0  # 0 = no position, 1 = long
        self.trades = []
        self.net_worth_history = []
        
    def reset(self):
        """Reset strategy to initial state"""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.position = 0
        self.trades = []
        self.net_worth_history = [self.initial_balance]
    
    def get_action(self, data: pd.DataFrame, index: int) -> int:
        """
        Get action for current step (to be implemented by subclasses)
        
        Args:
            data: DataFrame with price data
            index: Current index
            
        Returns:
            Action (0=hold, 1=buy, 2=sell)
        """
        raise NotImplementedError
    
    def execute_trade(self, action: int, price: float, fee: float = 0.001):
        """
        Execute a trade
        
        Args:
            action: Action to take (0=hold, 1=buy, 2=sell)
            price: Current price
            fee: Trading fee
        """
        if action == 1 and self.position == 0:  # Buy
            shares_to_buy = self.balance / price
            cost = shares_to_buy * price * (1 + fee)
            
            if cost <= self.balance:
                self.shares_held = shares_to_buy
                self.balance -= cost
                self.position = 1
                
                self.trades.append({
                    'action': 'BUY',
                    'price': price,
                    'shares': shares_to_buy,
                    'cost': cost
                })
        
        elif action == 2 and self.position == 1:  # Sell
            proceeds = self.shares_held * price * (1 - fee)
            
            self.balance += proceeds
            profit = proceeds - (self.shares_held * self.trades[-1]['price'])
            
            self.trades.append({
                'action': 'SELL',
                'price': price,
                'shares': self.shares_held,
                'proceeds': proceeds,
                'profit': profit
            })
            
            self.shares_held = 0
            self.position = 0
    
    def update_net_worth(self, current_price: float):
        """Update net worth history"""
        net_worth = self.balance + (self.shares_held * current_price)
        self.net_worth_history.append(net_worth)
    
    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with performance metrics
        """
        self.reset()
        
        for i in range(len(data)):
            current_price = data['close'].iloc[i]
            
            # Get action from strategy
            action = self.get_action(data, i)
            
            # Execute trade
            self.execute_trade(action, current_price)
            
            # Update net worth
            self.update_net_worth(current_price)
        
        # Close any open position at the end
        if self.position == 1:
            final_price = data['close'].iloc[-1]
            self.execute_trade(2, final_price)
            self.update_net_worth(final_price)
        
        return self.get_metrics()
    
    def get_metrics(self) -> Dict:
        """Calculate performance metrics"""
        final_net_worth = self.net_worth_history[-1]
        profit = final_net_worth - self.initial_balance
        profit_pct = (profit / self.initial_balance) * 100
        
        # Calculate returns
        returns = np.diff(self.net_worth_history) / np.array(self.net_worth_history[:-1])
        
        # Sharpe ratio
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(self.net_worth_history)
        drawdown = (peak - self.net_worth_history) / peak
        max_drawdown = np.max(drawdown)
        
        # Win rate
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        if sell_trades:
            winning_trades = sum(1 for t in sell_trades if t.get('profit', 0) > 0)
            win_rate = winning_trades / len(sell_trades)
        else:
            win_rate = 0
        
        return {
            'strategy': self.__class__.__name__,
            'final_net_worth': final_net_worth,
            'profit': profit,
            'profit_pct': profit_pct,
            'total_trades': len([t for t in self.trades if t['action'] in ['BUY', 'SELL']]),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }


class BuyAndHoldStrategy(BaselineStrategy):
    """
    Buy and Hold strategy
    
    Buy at the start, hold until the end
    The benchmark strategy that shows market performance
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        super().__init__(initial_balance)
        self.bought = False
    
    def reset(self):
        super().reset()
        self.bought = False
    
    def get_action(self, data: pd.DataFrame, index: int) -> int:
        """
        Buy at the start, hold forever
        
        Args:
            data: DataFrame with price data
            index: Current index
            
        Returns:
            Action (0=hold, 1=buy)
        """
        if not self.bought and self.position == 0:
            self.bought = True
            return 1  # Buy
        return 0  # Hold


class RandomTradingStrategy(BaselineStrategy):
    """
    Random trading strategy
    
    Makes random buy/sell decisions
    Used to prove that the RL agent actually learned something
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        trade_probability: float = 0.05,
        seed: int = 42
    ):
        """
        Args:
            initial_balance: Starting capital
            trade_probability: Probability of making a trade at each step
            seed: Random seed for reproducibility
        """
        super().__init__(initial_balance)
        self.trade_probability = trade_probability
        np.random.seed(seed)
    
    def get_action(self, data: pd.DataFrame, index: int) -> int:
        """
        Make random trading decisions
        
        Args:
            data: DataFrame with price data
            index: Current index
            
        Returns:
            Random action
        """
        # Random chance to trade
        if np.random.random() > self.trade_probability:
            return 0  # Hold
        
        # Random buy or sell
        if self.position == 0:
            return 1  # Buy
        else:
            return 2  # Sell


class MovingAverageCrossoverStrategy(BaselineStrategy):
    """
    Moving Average Crossover strategy
    
    Classic technical analysis strategy:
    - Buy when fast MA crosses above slow MA (Golden Cross)
    - Sell when fast MA crosses below slow MA (Death Cross)
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        fast_period: int = 50,
        slow_period: int = 200
    ):
        """
        Args:
            initial_balance: Starting capital
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
        """
        super().__init__(initial_balance)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.mas_calculated = False
        self.fast_ma = None
        self.slow_ma = None
    
    def calculate_mas(self, data: pd.DataFrame):
        """Calculate moving averages"""
        if not self.mas_calculated:
            self.fast_ma = data['close'].rolling(window=self.fast_period).mean()
            self.slow_ma = data['close'].rolling(window=self.slow_period).mean()
            self.mas_calculated = True
    
    def get_action(self, data: pd.DataFrame, index: int) -> int:
        """
        Get action based on MA crossover
        
        Args:
            data: DataFrame with price data
            index: Current index
            
        Returns:
            Action based on MA crossover
        """
        # Calculate MAs if not done yet
        if not self.mas_calculated:
            self.calculate_mas(data)
        
        # Need enough data for slow MA
        if index < self.slow_period:
            return 0  # Hold
        
        # Get current and previous MA values
        fast_current = self.fast_ma.iloc[index]
        slow_current = self.slow_ma.iloc[index]
        fast_previous = self.fast_ma.iloc[index - 1]
        slow_previous = self.slow_ma.iloc[index - 1]
        
        # Check for crossover
        # Golden Cross: fast MA crosses above slow MA
        if fast_previous <= slow_previous and fast_current > slow_current:
            if self.position == 0:
                return 1  # Buy
        
        # Death Cross: fast MA crosses below slow MA
        elif fast_previous >= slow_previous and fast_current < slow_current:
            if self.position == 1:
                return 2  # Sell
        
        return 0  # Hold


class RSIMomentumStrategy(BaselineStrategy):
    """
    RSI-based momentum strategy
    
    - Buy when RSI crosses above 30 (oversold)
    - Sell when RSI crosses above 70 (overbought)
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        rsi_period: int = 14,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70
    ):
        """
        Args:
            initial_balance: Starting capital
            rsi_period: Period for RSI calculation
            oversold_threshold: RSI level considered oversold
            overbought_threshold: RSI level considered overbought
        """
        super().__init__(initial_balance)
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.rsi_calculated = False
        self.rsi = None
    
    def calculate_rsi(self, data: pd.DataFrame):
        """Calculate RSI"""
        if not self.rsi_calculated:
            if 'rsi' in data.columns:
                self.rsi = data['rsi']
            else:
                # Calculate RSI if not in data
                delta = data['close'].diff()
                gains = delta.where(delta > 0, 0)
                losses = -delta.where(delta < 0, 0)
                
                avg_gains = gains.ewm(alpha=1/self.rsi_period, adjust=False).mean()
                avg_losses = losses.ewm(alpha=1/self.rsi_period, adjust=False).mean()
                
                rs = avg_gains / (avg_losses + 1e-10)
                self.rsi = 100 - (100 / (1 + rs))
            
            self.rsi_calculated = True
    
    def get_action(self, data: pd.DataFrame, index: int) -> int:
        """
        Get action based on RSI
        
        Args:
            data: DataFrame with price data
            index: Current index
            
        Returns:
            Action based on RSI levels
        """
        # Calculate RSI if not done yet
        if not self.rsi_calculated:
            self.calculate_rsi(data)
        
        # Need enough data for RSI
        if index < self.rsi_period:
            return 0  # Hold
        
        current_rsi = self.rsi.iloc[index]
        previous_rsi = self.rsi.iloc[index - 1]
        
        # Buy signal: RSI crosses above oversold threshold
        if previous_rsi <= self.oversold_threshold and current_rsi > self.oversold_threshold:
            if self.position == 0:
                return 1  # Buy
        
        # Sell signal: RSI crosses above overbought threshold
        elif previous_rsi <= self.overbought_threshold and current_rsi > self.overbought_threshold:
            if self.position == 1:
                return 2  # Sell
        
        return 0  # Hold


def compare_strategies(
    data: pd.DataFrame,
    strategies: List[BaselineStrategy]
) -> pd.DataFrame:
    """
    Compare multiple strategies on the same data
    
    Args:
        data: DataFrame with OHLCV data
        strategies: List of strategy instances
        
    Returns:
        DataFrame with comparison metrics
    """
    results = []
    
    for strategy in strategies:
        print(f"📊 Running {strategy.__class__.__name__}...")
        metrics = strategy.run_backtest(data)
        results.append(metrics)
    
    # Create comparison DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by profit percentage
    df_results = df_results.sort_values('profit_pct', ascending=False)
    
    return df_results


def test_baselines():
    """Test baseline strategies"""
    print("=" * 60)
    print("Testing Baseline Strategies")
    print("=" * 60)
    
    # Fetch data
    import yfinance as yf
    from indicators.technical_indicators import add_all_indicators
    
    print("📥 Fetching BTC data...")
    df = yf.Ticker("BTC-USD").history(period="6mo", interval="1d")
    df.columns = [col.lower() for col in df.columns]
    
    print("📊 Adding indicators...")
    df = add_all_indicators(df)
    
    # Initialize strategies
    strategies = [
        BuyAndHoldStrategy(initial_balance=10000),
        RandomTradingStrategy(initial_balance=10000, trade_probability=0.05),
        MovingAverageCrossoverStrategy(initial_balance=10000, fast_period=50, slow_period=200),
        RSIMomentumStrategy(initial_balance=10000)
    ]
    
    # Compare strategies
    print("\n🏁 Running strategy comparison...")
    results = compare_strategies(df, strategies)
    
    print("\n" + "=" * 60)
    print("📊 STRATEGY COMPARISON RESULTS")
    print("=" * 60)
    print(results.to_string(index=False))
    
    print("\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_baselines()
