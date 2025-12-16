"""
Debug Script - See what agents are seeing during evaluation
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np

# Now import from project
try:
    from agents.ppo_agent import PPOAgent
    from env.advanced_trading_env import AdvancedTradingEnv
    from utils.dqn_data_loader import DQNDataLoader
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current dir: {current_dir}")
    print(f"Parent dir: {parent_dir}")
    print("Make sure this script is in rl_trading_bot/ folder!")
    sys.exit(1)

# Load test data
data_loader = DQNDataLoader(
    ticker='BTC-USD',
    start_date='2023-01-01',
    end_date='2025-11-11',
    interval='1d',
    train_test_split=0.8
)

train_data, test_data = data_loader.prepare_data(normalize=True)
original_prices_test = data_loader.original_prices_test

print("="*70)
print("🔍 DEBUGGING AGENT BEHAVIOR")
print("="*70)

# Create environment
env = AdvancedTradingEnv(
    df=test_data,
    original_prices=original_prices_test,
    initial_cash=10000.0
)

# Load PPO agent
ppo_agent = PPOAgent(
    env=env,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    n_epochs=10,
    batch_size=64,
    hidden_sizes=[256, 256]
)

# Try to load
try:
    ppo_agent.load('results/ppo_20251215_104457.pth')
    print("✓ PPO Agent loaded")
except Exception as e:
    print(f"❌ Could not load agent: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("Testing first 20 steps:")
print("="*70)

obs, _ = env.reset()
done = False
step = 0

action_counts = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell

print(f"\n{'Step':<6} {'Price':<12} {'State Sample':<40} {'Action':<10}")
print("-"*70)

while not done and step < 20:
    # Get action
    action = ppo_agent.predict(obs)
    action_counts[action] += 1
    
    # Get current price
    current_price = original_prices_test[env.current_step]
    
    # Show state sample (first 5 features)
    state_sample = f"[{obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}, ...]"
    
    # Action name
    action_name = ['HOLD', 'BUY', 'SELL'][action]
    
    print(f"{step:<6} ${current_price:<11.2f} {state_sample:<40} {action_name:<10}")
    
    # Take step
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    obs = next_obs
    step += 1

print("\n" + "="*70)
print("📊 ACTION DISTRIBUTION (first 20 steps):")
print("="*70)
print(f"  HOLD: {action_counts[0]} ({action_counts[0]/20*100:.1f}%)")
print(f"  BUY:  {action_counts[1]} ({action_counts[1]/20*100:.1f}%)")
print(f"  SELL: {action_counts[2]} ({action_counts[2]/20*100:.1f}%)")

print("\n" + "="*70)
print("🔍 STATE STATISTICS:")
print("="*70)

# Reset and get first observation
obs, _ = env.reset()
print(f"State shape: {obs.shape}")
print(f"State range: [{obs.min():.3f}, {obs.max():.3f}]")
print(f"State mean: {obs.mean():.3f}")
print(f"State std: {obs.std():.3f}")

print("\n" + "="*70)
print("💡 DIAGNOSIS:")
print("="*70)

if action_counts[0] == 20:
    print("❌ Agent ONLY does HOLD!")
    print("   Possible causes:")
    print("   1. State distribution mismatch (training vs. test)")
    print("   2. Agent is too conservative")
    print("   3. Normalization issue")
elif action_counts[0] > 15:
    print("⚠️ Agent mostly does HOLD")
    print("   Agent might be too conservative")
else:
    print("✓ Agent is trading!")
    print(f"   Trade rate: {(20-action_counts[0])/20*100:.1f}%")

print("\n" + "="*70)