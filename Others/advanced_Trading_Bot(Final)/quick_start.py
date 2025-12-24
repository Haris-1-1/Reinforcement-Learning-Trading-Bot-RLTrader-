"""
Quick Start Script - Test individual components quickly
Run this before full training to ensure everything works
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_all_components():
    """Test all components of the trading bot"""
    
    print("\n" + "="*70)
    print("🚀 ADVANCED TRADING BOT - COMPONENT TESTS")
    print("="*70)
    
    success_count = 0
    total_tests = 5
    
    # Test 1: Data Loader
    print("\n" + "="*70)
    print("TEST 1/5: Data Loader")
    print("="*70)
    try:
        from data.data_loader import AdvancedDataLoader
        
        loader = AdvancedDataLoader(
            ticker="BTC-USD",
            interval="1h",
            sequence_length=30
        )
        
        df = loader.fetch_data(start_date="2024-11-01", end_date="2024-12-01")
        
        print(f"✅ Data Loader: PASSED")
        print(f"   - Fetched {len(df)} candles")
        print(f"   - Columns: {df.shape[1]}")
        success_count += 1
    except Exception as e:
        print(f"❌ Data Loader: FAILED")
        print(f"   Error: {e}")
    
    # Test 2: Technical Indicators
    print("\n" + "="*70)
    print("TEST 2/5: Technical Indicators")
    print("="*70)
    try:
        from indicators.technical_indicators import add_all_indicators
        
        df_with_indicators = add_all_indicators(df.copy())
        
        indicator_count = len(df_with_indicators.columns) - len(df.columns)
        
        print(f"✅ Technical Indicators: PASSED")
        print(f"   - Added {indicator_count} indicators")
        print(f"   - Sample indicators: {list(df_with_indicators.columns[-5:])}")
        success_count += 1
    except Exception as e:
        print(f"❌ Technical Indicators: FAILED")
        print(f"   Error: {e}")
    
    # Test 3: DRQN Agent
    print("\n" + "="*70)
    print("TEST 3/5: DRQN Agent")
    print("="*70)
    try:
        from agents.drqn_agent import DRQNAgent
        import numpy as np
        
        agent = DRQNAgent(
            state_size=20,
            action_size=3,
            sequence_length=30,
            hidden_size=64  # Smaller for faster testing
        )
        
        # Test action selection
        dummy_sequence = np.random.randn(30, 20)
        valid_actions = np.array([1, 1, 0])  # Can hold or buy
        action = agent.select_action(dummy_sequence, valid_actions, training=False)
        
        print(f"✅ DRQN Agent: PASSED")
        print(f"   - Device: {agent.device}")
        print(f"   - Hidden size: {agent.policy_net.hidden_size}")
        print(f"   - Test action: {action}")
        success_count += 1
    except Exception as e:
        print(f"❌ DRQN Agent: FAILED")
        print(f"   Error: {e}")
    
    # Test 4: Trading Environment
    print("\n" + "="*70)
    print("TEST 4/5: Trading Environment")
    print("="*70)
    try:
        from environments.trading_env import AdvancedTradingEnv
    
        print("Creating environment...")
        env = AdvancedTradingEnv(
        data=df_with_indicators,
        initial_balance=10000,
        sequence_length=30
        )
    
        print("Environment created, testing reset...")
        obs, _ = env.reset()
    
        print("Reset successful, testing step...")
        action = 0  # Hold
        obs, reward, terminated, truncated, info = env.step(action)
    
        print(f"✅ Trading Environment: PASSED")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - Feature count: {len(env.feature_columns)}")
        print(f"   - Initial net worth: ${info['net_worth']:,.2f}")
        success_count += 1
    except Exception as e:
        print(f"❌ Trading Environment: FAILED")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()  # DIESE ZEILE HINZUFÜGEN!
    # Test 5: Baseline Strategies
        print("\n" + "="*70)
        print("TEST 5/5: Baseline Strategies")
        print("="*70)
    try:
        from backtesting.baseline_strategies import (
            BuyAndHoldStrategy,
            MovingAverageCrossoverStrategy
        )
        
        # Test Buy & Hold
        bh_strategy = BuyAndHoldStrategy(initial_balance=10000)
        bh_metrics = bh_strategy.run_backtest(df_with_indicators)
        
        print(f"✅ Baseline Strategies: PASSED")
        print(f"   - Buy & Hold profit: {bh_metrics['profit_pct']:.2f}%")
        print(f"   - Buy & Hold trades: {bh_metrics['total_trades']}")
        success_count += 1
    except Exception as e:
        print(f"❌ Baseline Strategies: FAILED")
        print(f"   Error: {e}")
    
    # Final Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"Tests Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("✅ ALL TESTS PASSED! Ready for full training.")
        print("\nNext steps:")
        print("1. Run: python train.py")
        print("2. Monitor training progress")
        print("3. Check results in ./trading_bot_results/")
        return True
    else:
        print(f"❌ {total_tests - success_count} test(s) failed. Please fix errors before training.")
        return False


def quick_demo():
    """Run a quick demo with minimal training"""
    
    print("\n" + "="*70)
    print("🎮 QUICK DEMO - 5 Training Episodes")
    print("="*70)
    
    try:
        from train import TradingBotTrainer
        
        trainer = TradingBotTrainer(
            ticker="BTC-USD",
            interval="1h",
            train_start="2024-10-01",
            train_end="2024-11-15",
            test_start="2024-11-15",
            test_end="2024-12-01",
            sequence_length=20,  # Shorter for faster demo
            initial_balance=10000.0,
            save_dir="./demo_results"
        )
        
        # Prepare data
        trainer.prepare_data()
        
        # Create environments
        feature_columns = trainer.create_environments()
        
        # Initialize agent with smaller network
        trainer.initialize_agent(state_size=len(feature_columns))
        trainer.agent.policy_net.hidden_size = 64  # Smaller for demo
        
        # Quick training - just 5 episodes
        trainer.train(
            num_episodes=5,
            update_target_every=2,
            save_every=5,
            render_every=1
        )
        
        # Evaluate
        trainer.evaluate(render=False)
        
        # Compare with baselines
        trainer.compare_with_baselines()
        
        print("\n✅ Quick demo completed!")
        print(f"📁 Results saved to: {trainer.save_dir}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Start for Advanced Trading Bot')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['test', 'demo', 'both'],
        default='both',
        help='Mode: test components, run demo, or both'
    )
    
    args = parser.parse_args()
    
    if args.mode in ['test', 'both']:
        # Run component tests
        tests_passed = test_all_components()
        
        if not tests_passed and args.mode == 'both':
            print("\n⚠️  Tests failed. Skipping demo.")
            return
    
    if args.mode in ['demo', 'both']:
        # Run quick demo
        print("\n" + "="*70)
        input("Press Enter to start quick demo (5 episodes)...")
        quick_demo()
    
    print("\n" + "="*70)
    print("🎉 Quick Start Completed!")
    print("="*70)
    print("\nFor full training, run:")
    print("  python train.py")


if __name__ == "__main__":
    main()
