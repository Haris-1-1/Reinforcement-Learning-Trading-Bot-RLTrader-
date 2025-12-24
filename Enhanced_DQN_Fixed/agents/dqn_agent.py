"""
DUELING DOUBLE DQN AGENT - With Action Masking
===============================================
Features: Dueling Architecture, Double DQN, Action Masking, Experience Replay
"""

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

# ========================================
# 1. DUELING DQN NETWORK
# ========================================

class DuelingDQNNetwork(nn.Module):
    """
    Dueling Architecture: Separates State Value and Action Advantages
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(DuelingDQNNetwork, self).__init__()
        
        # Feature Extraction (Shared)
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Overfitting protection
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value Stream: V(s) - How good is this state?
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream: A(s,a) - How much better is each action?
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


# ========================================
# 2. AGENT
# ========================================

class Agent:
    """
    Dueling Double DQN Agent with Action Masking
    
    Features:
    - Dueling Architecture
    - Double DQN (separate target network)
    - Experience Replay
    - Epsilon-Greedy Exploration
    - Action Masking (prevents invalid actions)
    - Gradient Clipping
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        window_size: int,
        is_eval: bool = False
    ):
        """
        Args:
            state_size: Number of features per timestep
            action_size: Number of actions (3: Hold, Buy, Sell)
            window_size: Number of timesteps in window
            is_eval: If True, disables exploration
        """
        self.state_size = state_size
        self.action_size = action_size
        self.window_size = window_size
        
        # Input dimension = window_size * features + portfolio features (3)
        self.input_dim = (state_size * window_size) + 3
        
        # Experience Replay Buffer
        self.memory = deque(maxlen=100000)
        
        # Evaluation mode
        self.is_eval = is_eval
        
        # Hyperparameters
        self.gamma = 0.99              # Discount factor
        self.epsilon = 1.0             # Exploration rate (start)
        self.epsilon_min = 0.05        # Minimum exploration
        self.epsilon_decay = 0.99995   # Very slow decay for complex learning
        self.learning_rate = 0.0001
        self.batch_size = 64
        
        # Hardware acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent using device: {self.device}")
        
        # --- DOUBLE DQN SETUP ---
        # Policy Network: Makes decisions
        self.policy_net = DuelingDQNNetwork(self.input_dim, action_size).to(self.device)
        
        # Target Network: Evaluates decisions (stable)
        self.target_net = DuelingDQNNetwork(self.input_dim, action_size).to(self.device)
        
        # Sync networks at start
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def update_target_network(self):
        """Copy weights from policy net to target net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def act(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        """
        Select action using epsilon-greedy policy with action masking
        
        Args:
            state: Current state observation
            action_mask: Binary mask [1,1,1] = all allowed, [1,0,1] = buy forbidden
        
        Returns:
            action: 0=Hold, 1=Buy, 2=Sell
        """
        # Random exploration
        if not self.is_eval and np.random.rand() <= self.epsilon:
            if action_mask is not None:
                # Only choose from valid actions
                valid_actions = np.where(action_mask == 1)[0]
                return np.random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        # Exploitation: Use policy network
        # Ensure state is 1D before adding batch dimension
        if len(state.shape) > 1:
            state = state.flatten()
        state = torch.FloatTensor(state).view(1, -1).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            
            # Apply action mask
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                # Set invalid actions to very negative value
                q_values = q_values + (mask_tensor - 1) * 1e9
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done, next_action_mask: Optional[np.ndarray] = None):
        """
        Store experience in replay buffer
        
        Args:
            next_action_mask: Action mask for next state (needed for Double DQN)
        """
        # DON'T flatten - state and next_state are already 1D arrays from environment
        # Flattening can cause dimension mismatches
        
        # Store with action mask for next state
        if next_action_mask is None:
            next_action_mask = np.ones(self.action_size)
        
        self.memory.append((state, action, reward, next_state, done, next_action_mask))
    
    def replay(self):
        """
        Train on batch from experience replay
        Uses Double DQN logic with action masking
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample mini-batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch tensors - ensure 2D shape [batch_size, features]
        states_list = [i[0] for i in minibatch]
        next_states_list = [i[3] for i in minibatch]
        
        # Convert to numpy arrays first, then to tensors
        states = torch.FloatTensor(np.array(states_list)).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states_list)).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)
        next_masks = torch.FloatTensor(np.array([i[5] for i in minibatch])).to(self.device)
        
        # CRITICAL: Squeeze any extra dimensions to get [batch_size, features]
        while len(states.shape) > 2:
            states = states.squeeze(1)
        while len(next_states.shape) > 2:
            next_states = next_states.squeeze(1)
        
        # Ensure we have 2D tensors
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        if len(next_states.shape) == 1:
            next_states = next_states.unsqueeze(0)
        
        # --- DOUBLE DQN WITH ACTION MASKING ---
        
        with torch.no_grad():
            # 1. Policy net selects best action for next state
            next_q_policy = self.policy_net(next_states)
            
            # Apply action mask
            next_q_policy = next_q_policy + (next_masks - 1) * 1e9
            
            # Get best actions
            next_actions = next_q_policy.argmax(dim=1)
            
            # 2. Target net evaluates THOSE actions
            next_q_target = self.target_net(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Bellman equation
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()


if __name__ == "__main__":
    # Quick test
    print("Dueling Double DQN Agent with Action Masking")
    print("Features:")
    print("  - Dueling Architecture (Value + Advantage streams)")
    print("  - Double DQN (Policy + Target networks)")
    print("  - Action Masking (prevents invalid actions)")
    print("  - Experience Replay (100k buffer)")
    print("  - Gradient Clipping")
    print("  - Dropout (0.2)")