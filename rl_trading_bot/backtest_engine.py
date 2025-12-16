"""
Professional Backtesting Engine
Evaluates trained RL agents with standardized metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import pickle


class BacktestMetrics:
    """Calculate professional trading metrics"""
    
    @staticmethod
    def total_return(initial_value: float, final_value: float) -> float:
        """Calculate total return percentage"""
        return ((final_value - initial_value) / initial_value) * 100
    
    @staticmethod
    def annualized_return(total_return_pct: float, days: int) -> float:
        """Calculate annualized return"""
        if days == 0:
            return 0
        years = days / 365.25
        return ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio
        Measures risk-adjusted return
        
        Higher is better (>1 is good, >2 is excellent)
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Annualize
        sharpe = (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)
        return sharpe
    
    @staticmethod
    def max_drawdown(portfolio_values: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown
        
        Returns:
            (max_drawdown_pct, start_idx, end_idx)
        """
        if len(portfolio_values) == 0:
            return 0.0, 0, 0
        
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax * 100
        
        max_dd = np.min(drawdown)
        end_idx = np.argmin(drawdown)
        start_idx = np.argmax(portfolio_values[:end_idx + 1]) if end_idx > 0 else 0
        
        return abs(max_dd), start_idx, end_idx
    
    @staticmethod
    def win_rate(trades: List[Dict]) -> float:
        """Calculate percentage of winning trades"""
        if len(trades) == 0:
            return 0.0
        
        winning_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
        return (winning_trades / len(trades)) * 100
    
    @staticmethod
    def profit_factor(trades: List[Dict]) -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        
        >1 means profitable, >2 is excellent
        """
        if len(trades) == 0:
            return 0.0
        
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def volatility(returns: np.ndarray, annualize: bool = True) -> float:
        """Calculate volatility (standard deviation of returns)"""
        if len(returns) == 0:
            return 0.0
        
        vol = np.std(returns)
        
        if annualize:
            vol *= np.sqrt(252)  # Annualize for daily returns
        
        return vol * 100  # Convert to percentage
    
    @staticmethod
    def calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
        """
        Calculate Calmar Ratio (Return / Max Drawdown)
        
        Higher is better (>1 is good, >3 is excellent)
        """
        if max_drawdown == 0:
            return 0.0
        
        return annualized_return / max_drawdown
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino Ratio (like Sharpe but only penalizes downside volatility)
        
        Higher is better
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return - risk_free_rate / 252) / downside_std * np.sqrt(252)
        return sortino


class BacktestEngine:
    """
    Professional backtesting engine for RL trading agents
    """
    
    def __init__(self, initial_cash: float = 10000.0):
        """
        Initialize backtesting engine
        
        Args:
            initial_cash: Starting capital
        """
        self.initial_cash = initial_cash
        self.metrics_calculator = BacktestMetrics()
    
    def run_backtest(self, 
                     agent, 
                     env, 
                     data: pd.DataFrame,
                     agent_name: str = "Agent") -> Dict:
        """
        Run backtest on an agent
        
        Args:
            agent: Trained RL agent
            env: Trading environment
            data: Test data
            agent_name: Name for reporting
            
        Returns:
            Dictionary with all metrics and results
        """
        print(f"\n{'='*70}")
        print(f"🔍 BACKTESTING: {agent_name}")
        print(f"{'='*70}")
        print(f"Data points: {len(data)}")
        print(f"Initial capital: ${self.initial_cash:,.2f}")
        
        # Track results
        portfolio_values = []
        trades = []
        actions_taken = []
        rewards_received = []
        positions = []
        
        # Reset environment
        obs, _ = env.reset()
        done = False
        step = 0
        
        current_position = 0  # 0 = no position, 1 = long
        entry_price = 0
        
        # Run through entire test period
        while not done:
            # Agent makes decision
            action = agent.predict(obs)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track metrics
            portfolio_values.append(env.portfolio_value)
            actions_taken.append(action)
            rewards_received.append(reward)
            
            # Track trades
            # Get current price - handle different environment implementations
            if hasattr(env, 'original_prices'):
                current_price = env.original_prices[env.current_step]
            elif hasattr(env, 'prices'):
                current_price = env.prices[env.current_step]
            else:
                # Fallback: try to get from dataframe
                try:
                    current_price = env.df.iloc[env.current_step]['Close']
                except:
                    current_price = 0  # Fallback value
            
            if action == 1 and current_position == 0:  # Buy
                entry_price = current_price
                current_position = 1
                trades.append({
                    'type': 'BUY',
                    'step': step,
                    'price': current_price,
                    'pnl': 0
                })
            elif action == 2 and current_position == 1:  # Sell
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                pnl_absolute = (current_price - entry_price) * (env.coins if hasattr(env, 'coins') else 1)
                current_position = 0
                trades.append({
                    'type': 'SELL',
                    'step': step,
                    'price': current_price,
                    'pnl': pnl_absolute,
                    'pnl_pct': pnl_pct
                })
            
            positions.append(current_position)
            
            obs = next_obs
            step += 1
        
        # Calculate metrics
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        final_value = portfolio_values[-1]
        total_return = self.metrics_calculator.total_return(self.initial_cash, final_value)
        annualized_return = self.metrics_calculator.annualized_return(total_return, len(data))
        sharpe = self.metrics_calculator.sharpe_ratio(returns)
        max_dd, dd_start, dd_end = self.metrics_calculator.max_drawdown(portfolio_values)
        volatility = self.metrics_calculator.volatility(returns)
        sortino = self.metrics_calculator.sortino_ratio(returns)
        calmar = self.metrics_calculator.calmar_ratio(annualized_return, max_dd)
        
        # Trade statistics
        win_rate = self.metrics_calculator.win_rate(trades)
        profit_factor = self.metrics_calculator.profit_factor(trades)
        
        # Action statistics
        action_counts = {
            'Hold': actions_taken.count(0),
            'Buy': actions_taken.count(1),
            'Sell': actions_taken.count(2)
        }
        
        # Compile results
        results = {
            'agent_name': agent_name,
            'initial_capital': self.initial_cash,
            'final_capital': final_value,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown_pct': max_dd,
            'volatility_pct': volatility,
            'calmar_ratio': calmar,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len([t for t in trades if t['type'] == 'SELL']),
            'total_steps': len(portfolio_values),
            'action_counts': action_counts,
            'portfolio_values': portfolio_values.tolist(),
            'trades': trades,
            'returns': returns.tolist(),
            'actions': actions_taken,
            'rewards': rewards_received,
            'positions': positions
        }
        
        # Print summary
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """Print formatted backtest results"""
        print(f"\n📊 BACKTEST RESULTS: {results['agent_name']}")
        print("="*70)
        
        print("\n💰 RETURNS:")
        print(f"  Initial Capital:     ${results['initial_capital']:,.2f}")
        print(f"  Final Capital:       ${results['final_capital']:,.2f}")
        print(f"  Total Return:        {results['total_return_pct']:+.2f}%")
        print(f"  Annualized Return:   {results['annualized_return_pct']:+.2f}%")
        
        print("\n📈 RISK METRICS:")
        print(f"  Sharpe Ratio:        {results['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio:       {results['sortino_ratio']:.3f}")
        print(f"  Max Drawdown:        {results['max_drawdown_pct']:.2f}%")
        print(f"  Volatility:          {results['volatility_pct']:.2f}%")
        print(f"  Calmar Ratio:        {results['calmar_ratio']:.3f}")
        
        print("\n🎯 TRADING STATISTICS:")
        print(f"  Total Trades:        {results['total_trades']}")
        print(f"  Win Rate:            {results['win_rate_pct']:.2f}%")
        print(f"  Profit Factor:       {results['profit_factor']:.3f}")
        
        print("\n🎮 ACTIONS:")
        for action, count in results['action_counts'].items():
            pct = (count / results['total_steps']) * 100
            print(f"  {action:6s}: {count:4d} ({pct:5.1f}%)")
        
        print("\n" + "="*70)
    
    def compare_to_baseline(self, 
                           results: Dict, 
                           baseline_return: float,
                           baseline_name: str = "Buy & Hold") -> Dict:
        """
        Compare agent results to baseline (e.g., Buy & Hold)
        
        Args:
            results: Agent backtest results
            baseline_return: Baseline return percentage
            baseline_name: Name of baseline strategy
            
        Returns:
            Comparison metrics
        """
        agent_return = results['total_return_pct']
        outperformance = agent_return - baseline_return
        
        comparison = {
            'agent_name': results['agent_name'],
            'agent_return': agent_return,
            'baseline_name': baseline_name,
            'baseline_return': baseline_return,
            'outperformance': outperformance,
            'outperformance_pct': (outperformance / abs(baseline_return)) * 100 if baseline_return != 0 else 0,
            'beats_baseline': agent_return > baseline_return
        }
        
        return comparison
    
    def save_results(self, results: Dict, filepath: str):
        """Save backtest results to JSON file"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        results_serializable = convert_to_native(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=4)
        
        print(f"✓ Results saved to {filepath}")


def calculate_buy_and_hold(initial_cash: float, 
                           prices: np.ndarray) -> Dict:
    """
    Calculate Buy & Hold baseline performance
    
    Args:
        initial_cash: Starting capital
        prices: Price series
        
    Returns:
        Dictionary with Buy & Hold metrics
    """
    if len(prices) == 0:
        return {}
    
    # Buy at first price, hold, sell at last price
    initial_price = prices[0]
    final_price = prices[-1]
    
    coins_bought = initial_cash / initial_price
    final_value = coins_bought * final_price
    
    total_return = ((final_value - initial_cash) / initial_cash) * 100
    
    # Calculate portfolio values over time
    portfolio_values = (prices / initial_price) * initial_cash
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    metrics = BacktestMetrics()
    sharpe = metrics.sharpe_ratio(returns)
    max_dd, _, _ = metrics.max_drawdown(portfolio_values)
    volatility = metrics.volatility(returns)
    annualized_return = metrics.annualized_return(total_return, len(prices))
    
    return {
        'strategy': 'Buy & Hold',
        'initial_capital': initial_cash,
        'final_capital': final_value,
        'total_return_pct': total_return,
        'annualized_return_pct': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd,
        'volatility_pct': volatility,
        'portfolio_values': portfolio_values.tolist(),
        'returns': returns.tolist()
    }


if __name__ == "__main__":
    print("="*70)
    print("Backtesting Engine - Ready to use!")
    print("="*70)
    print("\nUsage:")
    print("  from backtest_engine import BacktestEngine, calculate_buy_and_hold")
    print("  engine = BacktestEngine(initial_cash=10000)")
    print("  results = engine.run_backtest(agent, env, test_data)")