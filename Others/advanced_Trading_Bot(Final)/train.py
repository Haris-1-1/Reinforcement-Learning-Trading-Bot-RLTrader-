"""
Main training pipeline for DRQN Trading Agent
Includes training, evaluation, and comparison with baselines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import AdvancedDataLoader
from indicators.technical_indicators import add_all_indicators
from agents.drqn_agent import DRQNAgent
from environments.trading_env import AdvancedTradingEnv
from backtesting.baseline_strategies import (
    BuyAndHoldStrategy,
    RandomTradingStrategy,
    MovingAverageCrossoverStrategy,
    compare_strategies
)


class TradingBotTrainer:
    """
    Main trainer for the advanced trading bot
    """
    
    def __init__(
        self,
        ticker: str = "BTC-USD",
        interval: str = "1h",
        train_start: str = "2020-01-01",
        train_end: str = "2024-06-01",
        test_start: str = "2024-06-01",
        test_end: str = "2024-12-01",
        sequence_length: int = 30,
        initial_balance: float = 10000.0,
        save_dir: str = "./results"
    ):
        """
        Args:
            ticker: Trading pair symbol
            interval: Timeframe (15m, 1h, etc.)
            train_start: Training data start date
            train_end: Training data end date
            test_start: Test data start date
            test_end: Test data end date
            sequence_length: Length of input sequences
            initial_balance: Starting capital
            save_dir: Directory to save results
        """
        self.ticker = ticker
        self.interval = interval
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.sequence_length = sequence_length
        self.initial_balance = initial_balance
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = AdvancedDataLoader(
            ticker=ticker,
            interval=interval,
            sequence_length=sequence_length
        )
        
        # Data storage
        self.train_data = None
        self.test_data = None
        self.train_env = None
        self.test_env = None
        self.agent = None
        
        # Training metrics
        self.episode_rewards = []
        self.episode_net_worths = []
        self.episode_profits = []
        
    def prepare_data(self):
        """
        Fetch and prepare training and test data
        """
        print("\n" + "="*60)
        print("📥 PREPARING DATA")
        print("="*60)
        
        # Fetch training data
        print(f"\n🔄 Fetching training data ({self.train_start} to {self.train_end})...")
        train_df = self.data_loader.fetch_data(
            start_date=self.train_start,
            end_date=self.train_end
        )
        
        # Add indicators
        print("📊 Adding technical indicators...")
        train_df = add_all_indicators(train_df)
        
        # Add volume profile
        train_df = self.data_loader.calculate_volume_profile(train_df)
        
        # Remove NaN values
        train_df = train_df.dropna()
        
        self.train_data = train_df
        
        print(f"✅ Training data prepared: {len(train_df)} candles")
        print(f"   Features: {len(train_df.columns)} columns")
        
        # Fetch test data
        print(f"\n🔄 Fetching test data ({self.test_start} to {self.test_end})...")
        test_df = self.data_loader.fetch_data(
            start_date=self.test_start,
            end_date=self.test_end
        )
        
        # Add indicators
        test_df = add_all_indicators(test_df)
        test_df = self.data_loader.calculate_volume_profile(test_df)
        test_df = test_df.dropna()
        
        self.test_data = test_df
        
        print(f"✅ Test data prepared: {len(test_df)} candles")
        
        return train_df, test_df
    
    def create_environments(self):
        """
        Create training and test environments
        """
        print("\n" + "="*60)
        print("🏗️  CREATING ENVIRONMENTS")
        print("="*60)
        
        # Select feature columns (exclude raw OHLCV, keep indicators)
        feature_columns = [
            col for col in self.train_data.columns
            if col not in ['open', 'high', 'low', 'volume']
            and pd.api.types.is_numeric_dtype(self.train_data[col])
        ]
        
        print(f"\n📊 Using {len(feature_columns)} features:")
        for i, col in enumerate(feature_columns[:10], 1):
            print(f"   {i}. {col}")
        if len(feature_columns) > 10:
            print(f"   ... and {len(feature_columns) - 10} more")
        
        # Create training environment
        self.train_env = AdvancedTradingEnv(
            data=self.train_data,
            initial_balance=self.initial_balance,
            sequence_length=self.sequence_length,
            feature_columns=feature_columns,
            maker_fee=0.005,
            taker_fee=0.001,
            slippage=0.00
        )
        
        # Create test environment
        self.test_env = AdvancedTradingEnv(
            data=self.test_data,
            initial_balance=self.initial_balance,
            sequence_length=self.sequence_length,
            feature_columns=feature_columns,
            maker_fee=0.005,
            taker_fee=0.001,
            slippage=0.00
        )
        
        print(f"\n✅ Environments created")
        print(f"   - Observation shape: {self.train_env.observation_space.shape}")
        print(f"   - Action space: {self.train_env.action_space}")
        
        return feature_columns
    
    def initialize_agent(self, state_size: int):
        """
        Initialize DRQN agent
        
        Args:
            state_size: Dimension of state features
        """
        print("\n" + "="*60)
        print("🤖 INITIALIZING DRQN AGENT")
        print("="*60)
        
        self.agent = DRQNAgent(
            state_size=state_size,
            action_size=3,
            sequence_length=self.sequence_length,
            hidden_size=128,
            learning_rate=0.0001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=32
        )
        
        print(f"\n✅ Agent initialized")
        print(f"   - State size: {state_size}")
        print(f"   - Hidden size: 128")
        print(f"   - Sequence length: {self.sequence_length}")
        
        return self.agent
    
    def train(
        self,
        num_episodes: int = 100,
        update_target_every: int = 10,
        save_every: int = 10,
        render_every: int = 20
    ):
        """
        Train the DRQN agent
        
        Args:
            num_episodes: Number of training episodes
            update_target_every: Update target network every N episodes
            save_every: Save agent every N episodes
            render_every: Render environment every N episodes
        """
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        print(f"Training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Reset environment and agent
            state, _ = self.train_env.reset()
            self.agent.reset_hidden_state()
            
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                # Get valid actions
                valid_actions = self.agent.get_valid_actions(
                    position=self.train_env.position,
                    cash=self.train_env.balance,
                    holdings=self.train_env.shares_held,
                    current_price=self.train_env._get_current_price()
                )
                
                # Select action
                action = self.agent.select_action(state, valid_actions, training=True)
                
                # Take step
                next_state, reward, terminated, truncated, info = self.train_env.step(action)
                done = terminated or truncated
                
                # Get next valid actions
                next_valid_actions = self.agent.get_valid_actions(
                    position=self.train_env.position,
                    cash=self.train_env.balance,
                    holdings=self.train_env.shares_held,
                    current_price=self.train_env._get_current_price()
                )
                
                # Store transition
                self.agent.memory.push(
                    state, action, reward, next_state, done, next_valid_actions
                )
                
                # Train agent
                self.agent.train_step()
                
                # Update state
                state = next_state
                episode_reward += reward
                step += 1
                
                # Render occasionally
                if episode % render_every == 0 and step % 100 == 0:
                    self.train_env.render()
            
            # Decay epsilon ONCE per episode (not per step!)
            if self.agent.epsilon > self.agent.epsilon_end:
                self.agent.epsilon *= self.agent.epsilon_decay
            
            # Get episode metrics
            metrics = self.train_env.get_metrics()
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_net_worths.append(metrics['final_net_worth'])
            self.episode_profits.append(metrics['profit_pct'])
            
            # Update target network
            if episode % update_target_every == 0:
                self.agent.update_target_network()
            
            # Save agent
            if episode % save_every == 0 and episode > 0:
                save_path = os.path.join(self.save_dir, f"agent_episode_{episode}.pt")
                self.agent.save(save_path)
            
            # Print progress
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'='*60}")
            print(f"📊 Reward: {episode_reward:.2f}")
            print(f"💰 Final Net Worth: ${metrics['final_net_worth']:,.2f}")
            print(f"📈 Profit: {metrics['profit_pct']:.2f}%")
            print(f"🎯 Total Trades: {metrics['total_trades']}")
            print(f"🎲 Epsilon: {self.agent.epsilon:.4f}")
            print(f"📉 Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
            print(f"✅ Win Rate: {metrics['win_rate']*100:.2f}%")
            
            # Show average of last 10 episodes
            if episode >= 10:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_profit = np.mean(self.episode_profits[-10:])
                print(f"\n📊 Last 10 Episodes Average:")
                print(f"   - Reward: {avg_reward:.2f}")
                print(f"   - Profit: {avg_profit:.2f}%")
        
        # Save final agent
        final_save_path = os.path.join(self.save_dir, "agent_final.pt")
        self.agent.save(final_save_path)
        
        print("\n✅ Training completed!")
    
    def evaluate(self, render: bool = True) -> Dict:
        """
        Evaluate agent on test data
        
        Args:
            render: Whether to render environment
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*60)
        print("📊 EVALUATING ON TEST DATA")
        print("="*60)
        
        state, _ = self.test_env.reset()
        self.agent.reset_hidden_state()
        
        done = False
        step = 0
        
        while not done:
            # Get valid actions
            valid_actions = self.agent.get_valid_actions(
                position=self.test_env.position,
                cash=self.test_env.balance,
                holdings=self.test_env.shares_held,
                current_price=self.test_env._get_current_price()
            )
            
            # Select action (no exploration)
            action = self.agent.select_action(state, valid_actions, training=False)
            
            # Take step
            state, reward, terminated, truncated, info = self.test_env.step(action)
            done = terminated or truncated
            step += 1
            
            # Render occasionally
            if render and step % 100 == 0:
                self.test_env.render()
        
        # Get final metrics
        metrics = self.test_env.get_metrics()
        
        print("\n" + "="*60)
        print("🏁 TEST RESULTS")
        print("="*60)
        print(f"💰 Final Net Worth: ${metrics['final_net_worth']:,.2f}")
        print(f"📈 Profit: ${metrics['profit']:,.2f} ({metrics['profit_pct']:.2f}%)")
        print(f"🎯 Total Trades: {metrics['total_trades']}")
        print(f"📉 Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"✅ Win Rate: {metrics['win_rate']*100:.2f}%")
        
        return metrics
    
    def compare_with_baselines(self) -> pd.DataFrame:
        """
        Compare DRQN agent with baseline strategies
        
        Returns:
            DataFrame with comparison results
        """
        print("\n" + "="*60)
        print("🏆 COMPARING WITH BASELINES")
        print("="*60)
        
        # Run DRQN evaluation
        drqn_metrics = self.evaluate(render=False)
        
        # Initialize baseline strategies
        baselines = [
            BuyAndHoldStrategy(initial_balance=self.initial_balance),
            RandomTradingStrategy(initial_balance=self.initial_balance),
            MovingAverageCrossoverStrategy(initial_balance=self.initial_balance)
        ]
        
        # Run baseline strategies
        baseline_results = compare_strategies(self.test_data, baselines)
        
        # Add DRQN results
        drqn_result = pd.DataFrame([{
            'strategy': 'DRQN Agent',
            'final_net_worth': drqn_metrics['final_net_worth'],
            'profit': drqn_metrics['profit'],
            'profit_pct': drqn_metrics['profit_pct'],
            'total_trades': drqn_metrics['total_trades'],
            'sharpe_ratio': drqn_metrics['sharpe_ratio'],
            'max_drawdown': drqn_metrics['max_drawdown'],
            'win_rate': drqn_metrics['win_rate']
        }])
        
        # Combine results
        comparison = pd.concat([drqn_result, baseline_results], ignore_index=True)
        comparison = comparison.sort_values('profit_pct', ascending=False)
        
        # Save results
        comparison.to_csv(os.path.join(self.save_dir, 'comparison_results.csv'), index=False)
        
        print("\n" + "="*60)
        print("📊 STRATEGY COMPARISON")
        print("="*60)
        print(comparison.to_string(index=False))
        
        return comparison
    
    def plot_training_progress(self):
        """
        Plot training progress metrics
        """
        print("\n📈 Generating training plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DRQN Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode Rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        axes[0, 0].plot(pd.Series(self.episode_rewards).rolling(10).mean(), 
                       linewidth=2, label='10-Episode MA')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Net Worth
        axes[0, 1].plot(self.episode_net_worths, alpha=0.6, label='Net Worth')
        axes[0, 1].axhline(y=self.initial_balance, color='r', 
                          linestyle='--', label='Initial Balance')
        axes[0, 1].plot(pd.Series(self.episode_net_worths).rolling(10).mean(),
                       linewidth=2, label='10-Episode MA')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Net Worth ($)')
        axes[0, 1].set_title('Episode Net Worth')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Profit Percentage
        axes[1, 0].plot(self.episode_profits, alpha=0.6, label='Profit %')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', label='Break Even')
        axes[1, 0].plot(pd.Series(self.episode_profits).rolling(10).mean(),
                       linewidth=2, label='10-Episode MA')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Profit (%)')
        axes[1, 0].set_title('Episode Profit Percentage')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Loss
        if len(self.agent.losses) > 0:
            axes[1, 1].plot(self.agent.losses, alpha=0.6)
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training plot saved to {plot_path}")
        
        plt.close()


def main():
    """
    Main function to run the complete training pipeline
    """
    print("\n" + "="*60)
    print("🤖 ADVANCED TRADING BOT - DRQN TRAINING PIPELINE")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize trainer
    trainer = TradingBotTrainer(
        ticker="BTC-USD",
        interval="1h",  # Using 1h for faster training (15m has too many candles)
        train_start="2024-01-01",
        train_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-12-01",
        sequence_length=30,
        initial_balance=10000.0,
        save_dir="./trading_bot_results"
    )
    
    # Step 1: Prepare data
    train_data, test_data = trainer.prepare_data()
    
    # Step 2: Create environments
    feature_columns = trainer.create_environments()
    
    # Step 3: Initialize agent
    trainer.initialize_agent(state_size=len(feature_columns))
    
    # Step 4: Train agent
    trainer.train(
        num_episodes=50,  # Start with fewer episodes for testing
        update_target_every=5,
        save_every=10,
        render_every=10
    )
    
    # Step 5: Plot training progress
    trainer.plot_training_progress()
    
    # Step 6: Evaluate on test data
    test_metrics = trainer.evaluate(render=False)
    
    # Step 7: Compare with baselines
    comparison = trainer.compare_with_baselines()
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📁 Results saved to: {trainer.save_dir}")


if __name__ == "__main__":
    main()