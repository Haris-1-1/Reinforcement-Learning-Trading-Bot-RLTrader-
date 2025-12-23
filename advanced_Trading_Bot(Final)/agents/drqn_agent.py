
"""
Deep Recurrent Q-Network (DRQN) Agent with GRU layers
Includes invalid action masking and supervised trend prediction component
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Optional, List


class GRUQNetwork(nn.Module):
    """
    GRU-based Q-Network for processing sequential data
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_gru_layers: int = 2,
        num_actions: int = 3,
        dropout: float = 0.2
    ):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of GRU hidden state
            num_gru_layers: Number of GRU layers
            num_actions: Number of possible actions (hold, buy, sell)
            dropout: Dropout probability
        """
        super(GRUQNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers
        self.num_actions = num_actions
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_size)
        
        # GRU layers for sequential processing
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0
        )
        
        # Attention mechanism to focus on important time steps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fully connected layers for Q-value estimation
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_actions)
        
        # Separate value and advantage streams (Dueling DQN architecture)
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, num_actions)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Hidden state from previous step (optional)
            
        Returns:
            Tuple of (q_values, new_hidden_state)
        """
        batch_size = x.size(0)
        
        # Normalize input
        x = self.input_norm(x)
        
        # GRU processing
        if hidden is None:
            gru_out, hidden = self.gru(x)
        else:
            gru_out, hidden = self.gru(x, hidden)
        
        # Apply attention to focus on important timesteps
        attended_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Take the last timestep output
        last_hidden = attended_out[:, -1, :]
        
        # Dueling DQN: separate value and advantage streams
        value = self.value_stream(last_hidden)
        advantages = self.advantage_stream(last_hidden)
        
        # Combine value and advantages
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values, hidden
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Initialize hidden state"""
        return torch.zeros(
            self.num_gru_layers,
            batch_size,
            self.hidden_size
        )


class SupervisedTrendPredictor(nn.Module):
    """
    Supervised learning component to predict short-term trend
    This provides "intuition" as an additional input feature to the RL agent
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64
    ):
        """
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden layers
        """
        super(SupervisedTrendPredictor, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 3)  # 3 classes: down, neutral, up
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict trend probabilities
        
        Args:
            x: Input features (last candle data)
            
        Returns:
            Probabilities for [down, neutral, up]
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(x, dim=-1)
        
        return probs


class ReplayBuffer:
    """
    Experience replay buffer for DRQN
    Stores sequences instead of single transitions
    """
    
    def __init__(self, capacity: int, sequence_length: int):
        """
        Args:
            capacity: Maximum number of sequences to store
            sequence_length: Length of each sequence
        """
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length
        
    def push(
        self,
        state_sequence: np.ndarray,
        action: int,
        reward: float,
        next_state_sequence: np.ndarray,
        done: bool,
        valid_actions: np.ndarray
    ):
        """
        Add a sequence to the buffer
        
        Args:
            state_sequence: Sequence of states (sequence_length, feature_size)
            action: Action taken
            reward: Reward received
            next_state_sequence: Next sequence of states
            done: Whether episode ended
            valid_actions: Mask of valid actions
        """
        self.buffer.append(
            (state_sequence, action, reward, next_state_sequence, done, valid_actions)
        )
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of sequences
        
        Args:
            batch_size: Number of sequences to sample
            
        Returns:
            Tuple of batched sequences
        """
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones, valid_actions = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(valid_actions)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DRQNAgent:
    """
    Deep Recurrent Q-Network Agent with advanced features:
    - GRU-based architecture for sequential processing
    - Invalid action masking
    - Supervised trend prediction component
    - Dueling DQN architecture
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int = 3,
        sequence_length: int = 30,
        hidden_size: int = 128,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        device: str = None
    ):
        """
        Args:
            state_size: Dimension of state features
            action_size: Number of possible actions (0=hold, 1=buy, 2=sell)
            sequence_length: Length of input sequences
            hidden_size: Size of GRU hidden state
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            device: Device to use ('cuda' or 'cpu')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = GRUQNetwork(
            input_size=state_size,
            hidden_size=hidden_size,
            num_actions=action_size
        ).to(self.device)
        
        self.target_net = GRUQNetwork(
            input_size=state_size,
            hidden_size=hidden_size,
            num_actions=action_size
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Supervised trend predictor
        self.trend_predictor = SupervisedTrendPredictor(
            input_size=state_size
        ).to(self.device)
        
        # Optimizers
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.trend_optimizer = optim.Adam(self.trend_predictor.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size, sequence_length)
        
        # Hidden state
        self.hidden_state = None
        
        # Training statistics
        self.losses = []
        self.q_values = []
        
    def get_valid_actions(
        self,
        position: int,
        cash: float,
        holdings: float,
        current_price: float
    ) -> np.ndarray:
        """
        Get mask of valid actions based on current state
        
        Invalid Action Masking prevents the agent from taking physically impossible actions
        
        Args:
            position: Current position (0=none, 1=long)
            cash: Available cash
            holdings: Number of shares held
            current_price: Current asset price
            
        Returns:
            Binary mask where 1 = valid action, 0 = invalid action
        """
        valid_actions = np.zeros(self.action_size)
        
        # Action 0: Hold (always valid)
        valid_actions[0] = 1
        
        # Action 1: Buy (valid only if we have cash and no position)
        if cash > current_price * 1.1 and position == 0:  # 10% Buffer
            valid_actions[1] = 1
        
        # Action 2: Sell (valid only if we have holdings)
        if holdings > 0 and position == 1:
            valid_actions[2] = 1
        
        return valid_actions
    
    def predict_trend(self, current_state: np.ndarray) -> np.ndarray:
        """
        Predict short-term trend using supervised component
        
        Args:
            current_state: Current state features
            
        Returns:
            Probabilities for [down, neutral, up]
        """
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            trend_probs = self.trend_predictor(state_tensor)
        
        return trend_probs.cpu().numpy()[0]
    
    def select_action(
        self,
        state_sequence: np.ndarray,
        valid_actions: np.ndarray,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy with invalid action masking
        
        Args:
            state_sequence: Sequence of states (sequence_length, state_size)
            valid_actions: Mask of valid actions
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Random action from valid actions only
            valid_action_indices = np.where(valid_actions == 1)[0]
            return np.random.choice(valid_action_indices)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values, self.hidden_state = self.policy_net(state_tensor, self.hidden_state)
            
            # Apply invalid action masking
            # Set Q-values of invalid actions to -inf so they're never selected
            q_values_masked = q_values.clone()
            invalid_actions = (valid_actions == 0)
            q_values_masked[0, invalid_actions] = float('-inf')
            
            action = q_values_masked.argmax().item()
            
            # Store Q-values for analysis
            self.q_values.append(q_values.cpu().numpy()[0])
        
        return action
    
    def train_step(self):
        """
        Perform one training step
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, valid_actions = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        valid_actions = torch.FloatTensor(valid_actions).to(self.device)
        
        # Current Q-values
        current_q_values, _ = self.policy_net(states, None)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states, None)
            
            # Apply invalid action masking to next states
            next_q_values_masked = next_q_values.clone()
            next_q_values_masked[valid_actions == 0] = float('-inf')
            
            next_q_values = next_q_values_masked.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Store loss
        self.losses.append(loss.item())
    
    def update_target_network(self):
        """
        Update target network with policy network weights
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def reset_hidden_state(self):
        """
        Reset hidden state (call at start of new episode)
        """
        self.hidden_state = None
    
    def save(self, filepath: str):
        """
        Save agent to file
        
        Args:
            filepath: Path to save file
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'trend_predictor': self.trend_predictor.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses
        }, filepath)
        
        print(f"💾 Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load agent from file
        
        Args:
            filepath: Path to load file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.trend_predictor.load_state_dict(checkpoint['trend_predictor'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint['losses']
        
        print(f"📂 Agent loaded from {filepath}")


def test_drqn_agent():
    """Test the DRQN agent"""
    print("=" * 60)
    print("Testing DRQN Agent")
    print("=" * 60)
    
    # Initialize agent
    agent = DRQNAgent(
        state_size=20,
        action_size=3,
        sequence_length=30,
        hidden_size=128
    )
    
    print(f"\n🤖 Agent initialized:")
    print(f"   - State size: {agent.state_size}")
    print(f"   - Action size: {agent.action_size}")
    print(f"   - Sequence length: {agent.sequence_length}")
    print(f"   - Device: {agent.device}")
    
    # Test action masking
    valid_actions = agent.get_valid_actions(
        position=0,
        cash=10000,
        holdings=0,
        current_price=50000
    )
    print(f"\n🎭 Valid actions (no position, has cash): {valid_actions}")
    
    valid_actions = agent.get_valid_actions(
        position=1,
        cash=0,
        holdings=1,
        current_price=50000
    )
    print(f"🎭 Valid actions (has position, no cash): {valid_actions}")
    
    # Test action selection
    dummy_sequence = np.random.randn(30, 20)
    action = agent.select_action(dummy_sequence, valid_actions, training=False)
    print(f"\n🎯 Selected action: {action}")
    
    print(f"\n✅ Test completed successfully!")


if __name__ == "__main__":
    test_drqn_agent()