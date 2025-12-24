import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Optional

class DQNNetwork(nn.Module):
    """
    Standard Deep Q-Network (Feed Forward)
    Flattens the sequence input into a single vector.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_actions: int = 3,
        dropout: float = 0.2
    ):
        super(DQNNetwork, self).__init__()
        
        # Input Layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size) # Batch Norm hilft beim Lernen
        self.dropout1 = nn.Dropout(dropout)
        
        # Hidden Layer 1
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        
        # Hidden Layer 2
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        
        # Output Streams (Dueling Architecture: Value & Advantage)
        self.value_stream = nn.Linear(hidden_size // 2, 1)
        self.advantage_stream = nn.Linear(hidden_size // 2, num_actions)
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape ist: (batch_size, sequence_length, features)
        # Wir müssen es flach machen: (batch_size, sequence_length * features)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.relu(self.bn3(self.fc3(x)))
        
        # Dueling Logic
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done, valid_actions):
        self.buffer.append((state, action, reward, next_state, done, valid_actions))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, valid_actions = zip(*batch)
        return (
            np.array(states), np.array(actions), np.array(rewards),
            np.array(next_states), np.array(dones), np.array(valid_actions)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        state_size: int, # Anzahl Features
        sequence_length: int, # Länge der Historie (z.B. 30)
        action_size: int = 3,
        hidden_size: int = 256,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,
        batch_size: int = 64,
        device: str = None
    ):
        self.state_size = state_size
        self.sequence_length = sequence_length
        # Der Input für das Netzwerk ist Features * SequenceLength (alles flach)
        self.flat_input_size = state_size * sequence_length
        
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"🖥️ DQN Agent initialized on: {self.device}")
        
        # Networks
        self.policy_net = DQNNetwork(self.flat_input_size, hidden_size, action_size).to(self.device)
        self.target_net = DQNNetwork(self.flat_input_size, hidden_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.losses = []

    def get_valid_actions(self, position, cash, holdings, current_price):
        """Action Masking Logik (übernommen und korrigiert)"""
        valid_actions = np.zeros(self.action_size)
        valid_actions[0] = 1 # Hold
        
        # Buy: Wenn Cash > $10 und keine Position
        if cash > 10.0 and position == 0:
            valid_actions[1] = 1
            
        # Sell: Wenn Position da ist
        if position == 1 and holdings > 0:
            valid_actions[2] = 1
            
        return valid_actions

    def select_action(self, state, valid_actions, training=True):
        if training and random.random() < self.epsilon:
            valid_indices = np.where(valid_actions == 1)[0]
            return np.random.choice(valid_indices)
        
        # State vorbereiten
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            
            # Masking
            q_values_masked = q_values.clone()
            invalid_actions = (torch.FloatTensor(valid_actions).to(self.device) == 0)
            q_values_masked[0, invalid_actions] = float('-inf')
            
            action = q_values_masked.argmax().item()
            
        self.policy_net.train()
        return action

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones, valid_actions = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        valid_actions_tensor = torch.FloatTensor(valid_actions).to(self.device)
        
        # Current Q
        curr_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q (Double DQN Logic + Masking)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            # Wichtig: Auch beim Target Masking beachten!
            next_q_values[valid_actions_tensor == 0] = float('-inf')
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
        loss = nn.MSELoss()(curr_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses
        }, filepath)
        print(f"💾 DQN Agent saved to {filepath}")
        
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        if 'losses' in checkpoint: self.losses = checkpoint['losses']
        print(f"📂 DQN Agent loaded from {filepath}")