import numpy as np
import gymnasium as gym
from gymnasium import spaces

print("Testing Box creation...")

# Test 1: Scalar bounds
try:
    box1 = spaces.Box(low=-1e8, high=1e8, shape=(30, 31), dtype=np.float32)
    print("✅ Test 1 passed: scalar bounds")
except Exception as e:
    print(f"❌ Test 1 failed: {e}")

# Test 2: Array bounds
try:
    box2 = spaces.Box(
        low=np.ones((30, 31), dtype=np.float32) * -1e8,
        high=np.ones((30, 31), dtype=np.float32) * 1e8,
        dtype=np.float32
    )
    print("✅ Test 2 passed: array bounds")
except Exception as e:
    print(f"❌ Test 2 failed: {e}")

# Test 3: Simple bounds
try:
    box3 = spaces.Box(low=-1.0, high=1.0, shape=(30, 31), dtype=np.float32)
    print("✅ Test 3 passed: simple bounds")
except Exception as e:
    print(f"❌ Test 3 failed: {e}")

print("\nDone!")