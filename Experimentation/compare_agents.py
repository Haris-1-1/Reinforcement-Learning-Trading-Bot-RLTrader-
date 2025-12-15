"""
Agent Comparison Tool
Compare multiple RL agents with standardized evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import json
import os
from datetime import datetime
from scipy import stats

from backtest_engine import BacktestEngine, calculate_buy_and_hold


class AgentComparison:
    """
    Compare multiple trained agents on same test data
    """
    
    def __init__(self, test_data: pd.DataFrame, 
                 original_prices: np.ndarray,
                 initial_cash: float = 10000.0):
        """
        Initialize comparison tool
        
        Args:
            test_data: Test dataset (same for all agents)
            original_prices: Original price series
            initial_cash: Starting capital (same for all agents)
        """
        self.test_data = test_data
        self.original_prices = original_prices
        self.initial_cash = initial_cash
        self.engine = BacktestEngine(initial_cash=initial_cash)
        
        self.results = {}
        self.baseline_results = None
    
    def add_agent(self, agent, env, agent_name: str):
        """
        Add an agent to comparison
        
        Args:
            agent: Trained RL agent
            env: Trading environment (configured with test data)
            agent_name: Name for this agent (e.g., 'Q-Learning', 'DQN', 'PPO')
        """
        print(f"\n{'='*70}")
        print(f"Adding agent: {agent_name}")
        print(f"{'='*70}")
        
        results = self.engine.run_backtest(
            agent=agent,
            env=env,
            data=self.test_data,
            agent_name=agent_name
        )
        
        self.results[agent_name] = results
    
    def calculate_baseline(self):
        """Calculate Buy & Hold baseline"""
        print(f"\n{'='*70}")
        print("Calculating Buy & Hold Baseline")
        print(f"{'='*70}")
        
        self.baseline_results = calculate_buy_and_hold(
            initial_cash=self.initial_cash,
            prices=self.original_prices
        )
        
        print("\n💰 Buy & Hold Results:")
        print(f"  Total Return: {self.baseline_results['total_return_pct']:+.2f}%")
        print(f"  Sharpe Ratio: {self.baseline_results['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {self.baseline_results['max_drawdown_pct']:.2f}%")
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table with all metrics
        
        Returns:
            DataFrame with agent comparison
        """
        if len(self.results) == 0:
            print("⚠️ No agents to compare!")
            return pd.DataFrame()
        
        # Collect all metrics
        comparison_data = []
        
        # Add baseline first
        if self.baseline_results:
            comparison_data.append({
                'Agent': 'Buy & Hold',
                'Total Return (%)': self.baseline_results['total_return_pct'],
                'Ann. Return (%)': self.baseline_results['annualized_return_pct'],
                'Sharpe Ratio': self.baseline_results['sharpe_ratio'],
                'Sortino Ratio': 0,  # Not calculated for baseline
                'Max Drawdown (%)': self.baseline_results['max_drawdown_pct'],
                'Volatility (%)': self.baseline_results['volatility_pct'],
                'Calmar Ratio': 0,
                'Win Rate (%)': 0,
                'Profit Factor': 0,
                'Total Trades': 0,
                'Final Capital ($)': self.baseline_results['final_capital']
            })
        
        # Add all agents
        for name, results in self.results.items():
            comparison_data.append({
                'Agent': name,
                'Total Return (%)': results['total_return_pct'],
                'Ann. Return (%)': results['annualized_return_pct'],
                'Sharpe Ratio': results['sharpe_ratio'],
                'Sortino Ratio': results['sortino_ratio'],
                'Max Drawdown (%)': results['max_drawdown_pct'],
                'Volatility (%)': results['volatility_pct'],
                'Calmar Ratio': results['calmar_ratio'],
                'Win Rate (%)': results['win_rate_pct'],
                'Profit Factor': results['profit_factor'],
                'Total Trades': results['total_trades'],
                'Final Capital ($)': results['final_capital']
            })
        
        df = pd.DataFrame(comparison_data)
        
        return df
    
    def rank_agents(self, metric: str = 'Total Return (%)') -> pd.DataFrame:
        """
        Rank agents by specified metric
        
        Args:
            metric: Metric to rank by
            
        Returns:
            Sorted DataFrame
        """
        df = self.generate_comparison_table()
        
        if metric not in df.columns:
            print(f"⚠️ Metric '{metric}' not found!")
            return df
        
        # Higher is better for most metrics
        ascending = False
        if 'Drawdown' in metric or 'Volatility' in metric:
            ascending = True  # Lower is better
        
        df_sorted = df.sort_values(by=metric, ascending=ascending)
        df_sorted['Rank'] = range(1, len(df_sorted) + 1)
        
        # Reorder columns to put Rank first
        cols = ['Rank'] + [col for col in df_sorted.columns if col != 'Rank']
        df_sorted = df_sorted[cols]
        
        return df_sorted
    
    def calculate_statistical_significance(self, agent1: str, agent2: str) -> Dict:
        """
        Test if difference between two agents is statistically significant
        Uses t-test on daily returns
        
        Args:
            agent1: Name of first agent
            agent2: Name of second agent
            
        Returns:
            Dictionary with test results
        """
        if agent1 not in self.results or agent2 not in self.results:
            return {'error': 'Agent not found'}
        
        returns1 = np.array(self.results[agent1]['returns'])
        returns2 = np.array(self.results[agent2]['returns'])
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(returns1, returns2)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(returns1) - np.mean(returns2)
        pooled_std = np.sqrt((np.std(returns1)**2 + np.std(returns2)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
        
        result = {
            'agent1': agent1,
            'agent2': agent2,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05,
            'effect_size': 'small' if abs(cohens_d) < 0.5 else ('medium' if abs(cohens_d) < 0.8 else 'large')
        }
        
        return result
    
    def generate_report(self) -> str:
        """
        Generate comprehensive comparison report
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*70)
        report.append("AGENT COMPARISON REPORT")
        report.append("="*70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Test Period: {len(self.test_data)} days")
        report.append(f"Initial Capital: ${self.initial_cash:,.2f}")
        report.append("")
        
        # Overall ranking
        report.append("="*70)
        report.append("OVERALL RANKING (by Total Return)")
        report.append("="*70)
        
        df_ranked = self.rank_agents('Total Return (%)')
        report.append(df_ranked.to_string(index=False))
        report.append("")
        
        # Risk-adjusted ranking
        report.append("="*70)
        report.append("RISK-ADJUSTED RANKING (by Sharpe Ratio)")
        report.append("="*70)
        
        df_sharpe = self.rank_agents('Sharpe Ratio')
        report.append(df_sharpe[['Rank', 'Agent', 'Sharpe Ratio', 'Total Return (%)']].to_string(index=False))
        report.append("")
        
        # Best by different metrics
        report.append("="*70)
        report.append("BEST PERFORMERS BY CATEGORY")
        report.append("="*70)
        
        df_all = self.generate_comparison_table()
        
        metrics_to_check = [
            ('Total Return (%)', False),
            ('Sharpe Ratio', False),
            ('Sortino Ratio', False),
            ('Calmar Ratio', False),
            ('Win Rate (%)', False),
            ('Profit Factor', False),
            ('Max Drawdown (%)', True),  # Lower is better
            ('Volatility (%)', True)     # Lower is better
        ]
        
        for metric, lower_better in metrics_to_check:
            if metric in df_all.columns:
                if lower_better:
                    best_idx = df_all[metric].idxmin()
                else:
                    best_idx = df_all[metric].idxmax()
                
                best_agent = df_all.loc[best_idx, 'Agent']
                best_value = df_all.loc[best_idx, metric]
                
                report.append(f"  {metric:25s}: {best_agent:15s} ({best_value:.2f})")
        
        report.append("")
        
        # Statistical significance
        if len(self.results) >= 2:
            report.append("="*70)
            report.append("STATISTICAL SIGNIFICANCE TESTS")
            report.append("="*70)
            
            agent_names = list(self.results.keys())
            for i in range(len(agent_names)):
                for j in range(i + 1, len(agent_names)):
                    sig_test = self.calculate_statistical_significance(
                        agent_names[i], 
                        agent_names[j]
                    )
                    
                    report.append(f"\n{agent_names[i]} vs {agent_names[j]}:")
                    report.append(f"  P-value: {sig_test['p_value']:.4f}")
                    report.append(f"  Significant: {'Yes' if sig_test['significant'] else 'No'}")
                    report.append(f"  Effect Size: {sig_test['effect_size']} (Cohen's d = {sig_test['cohens_d']:.3f})")
        
        report.append("")
        
        # Winner declaration
        report.append("="*70)
        report.append("🏆 WINNER")
        report.append("="*70)
        
        df_final = self.rank_agents('Total Return (%)')
        winner = df_final.iloc[0]
        
        report.append(f"\n🥇 Champion: {winner['Agent']}")
        report.append(f"   Total Return: {winner['Total Return (%)']:+.2f}%")
        report.append(f"   Sharpe Ratio: {winner['Sharpe Ratio']:.3f}")
        report.append(f"   Max Drawdown: {winner['Max Drawdown (%)']:.2f}%")
        
        if self.baseline_results:
            baseline_return = self.baseline_results['total_return_pct']
            outperformance = winner['Total Return (%)'] - baseline_return
            report.append(f"\n   vs Buy & Hold: {outperformance:+.2f}% ({winner['Total Return (%)']:+.2f}% vs {baseline_return:+.2f}%)")
        
        report.append("")
        report.append("="*70)
        
        return "\n".join(report)
    
    def save_report(self, filepath: str):
        """Save comparison report to file"""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved to {filepath}")
    
    def save_all_results(self, output_dir: str = 'results/comparison'):
        """
        Save all comparison results
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save comparison table
        df = self.generate_comparison_table()
        csv_path = f"{output_dir}/comparison_table_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Comparison table saved to {csv_path}")
        
        # Save report
        report_path = f"{output_dir}/comparison_report_{timestamp}.txt"
        self.save_report(report_path)
        
        # Save detailed results for each agent
        for name, results in self.results.items():
            agent_file = f"{output_dir}/{name.lower().replace(' ', '_')}_{timestamp}.json"
            with open(agent_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"✓ {name} results saved to {agent_file}")
        
        # Save baseline
        if self.baseline_results:
            baseline_file = f"{output_dir}/buy_and_hold_{timestamp}.json"
            with open(baseline_file, 'w') as f:
                json.dump(self.baseline_results, f, indent=4)
            print(f"✓ Baseline results saved to {baseline_file}")
        
        print(f"\n✓ All results saved to {output_dir}/")


if __name__ == "__main__":
    print("="*70)
    print("Agent Comparison Tool - Ready to use!")
    print("="*70)
    print("\nUsage:")
    print("  from compare_agents import AgentComparison")
    print("  comparison = AgentComparison(test_data, original_prices)")
    print("  comparison.add_agent(q_agent, env, 'Q-Learning')")
    print("  comparison.add_agent(dqn_agent, env, 'DQN')")
    print("  comparison.add_agent(ppo_agent, env, 'PPO')")
    print("  comparison.calculate_baseline()")
    print("  comparison.save_all_results()")
