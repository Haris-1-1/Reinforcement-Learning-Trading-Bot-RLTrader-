import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Dict, List

class DQNetwork(nn.Module):
    """
    Härtung gegen Overfitting durch Dropout und kompakte Architektur.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super(DQNetwork, self).__init__()
        layers = []
        prev_size = state_dim
        
        for i, h in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            # Dropout hilft gegen das Auswendiglernen von Rauschen
            layers.append(nn.Dropout(p=0.2)) 
            prev_size = h
            
        layers.append(nn.Linear(prev_size, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DQNAgent:
    def __init__(self, env, config: Dict):
        self.env = env
        self.config = config
        
        # Hyperparameter aus deiner Exp 11 Config
        dqn_cfg = config.get('dqn', {})
        self.lr = dqn_cfg.get('learning_rate', 0.0001)
        self.gamma = dqn_cfg.get('gamma', 0.99)
        self.batch_size = dqn_cfg.get('batch_size', 64)
        self.epsilon = dqn_cfg.get('epsilon_start', 1.0)
        self.eps_end = dqn_cfg.get('epsilon_end', 0.01)
        self.eps_decay = dqn_cfg.get('epsilon_decay_steps', 50000)
        self.target_update_freq = dqn_cfg.get('target_update_freq', 1000)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dimensionen (Input = Window_Size * Features + Portfolio)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Netzwerke
        self.policy_net = DQNetwork(self.state_dim, self.action_dim, dqn_cfg.get('hidden_sizes', [128, 128])).to(self.device)
        self.target_net = DQNetwork(self.state_dim, self.action_dim, dqn_cfg.get('hidden_sizes', [128, 128])).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer: AdamW mit Weight Decay gegen Overfitting
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, weight_decay=1e-4)
        self.memory = deque(maxlen=dqn_cfg.get('replay_buffer_size', 10000))
        self.steps_done = 0

    def predict(self, state: np.ndarray, deterministic: bool = False, action_mask: np.ndarray = None):
        """
        Wählt eine Aktion unter Berücksichtigung des Action Maskings.
        """
        # Epsilon-Greedy Exploration
        if not deterministic and random.random() < self.epsilon:
            if action_mask is not None:
                # Nur aus erlaubten Aktionen wählen
                allowed_indices = np.where(action_mask == 1)[0]
                return random.choice(allowed_indices)
            return self.env.action_space.sample()

        # Exploitation
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            
            # Action Masking Anwendung
            if action_mask is not None:
                mask_t = torch.FloatTensor(action_mask).to(self.device)
                # Setze verbotene Aktionen auf einen extrem niedrigen Wert
                q_values = q_values + (mask_t - 1) * 1e9
                
            return q_values.argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample Batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, masks = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        masks = torch.FloatTensor(np.array(masks)).to(self.device)

        # Q(s, a)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q mit Action Masking auf dem Target Network
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            # Maskiere ungültige Aktionen im nächsten Schritt
            next_q_values = next_q_values + (masks - 1) * 1e9
            max_next_q = next_q_values.max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping zur Stabilisierung
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Epsilon Decay
        if self.epsilon > self.eps_end:
            self.epsilon -= (1.0 - self.eps_end) / self.eps_decay
            
        return loss.item()

    def store_transition(self, s, a, r, s_next, done, mask):
        self.memory.append((s, a, r, s_next, done, mask))

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())