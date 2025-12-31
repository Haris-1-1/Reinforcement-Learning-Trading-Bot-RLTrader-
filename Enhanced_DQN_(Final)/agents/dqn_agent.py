import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DuelingDQNNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals
class Agent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        window_size: int,
        is_eval: bool = False
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.window_size = window_size
        self.input_dim = (state_size * window_size) + 3
        self.memory = deque(maxlen=100000)
        self.is_eval = is_eval
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent using device: {self.device}")
        self.policy_net = DuelingDQNNetwork(self.input_dim, action_size).to(self.device)
        self.target_net = DuelingDQNNetwork(self.input_dim, action_size).to(self.device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    def act(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> int:
        if not self.is_eval and np.random.rand() <= self.epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask == 1)[0]
                return np.random.choice(valid_actions)
            return random.randrange(self.action_size)
        if len(state.shape) > 1:
            state = state.flatten()
        state = torch.FloatTensor(state).view(1, -1).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                q_values = q_values + (mask_tensor - 1) * 1e9
        return q_values.argmax().item()
    def remember(self, state, action, reward, next_state, done, next_action_mask: Optional[np.ndarray] = None):
        if next_action_mask is None:
            next_action_mask = np.ones(self.action_size)
        self.memory.append((state, action, reward, next_state, done, next_action_mask))
    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        minibatch = random.sample(self.memory, self.batch_size)
        states_list = [i[0] for i in minibatch]
        next_states_list = [i[3] for i in minibatch]
        states = torch.FloatTensor(np.array(states_list)).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states_list)).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)
        next_masks = torch.FloatTensor(np.array([i[5] for i in minibatch])).to(self.device)
        while len(states.shape) > 2:
            states = states.squeeze(1)
        while len(next_states.shape) > 2:
            next_states = next_states.squeeze(1)
        if len(states.shape) == 1:
            states = states.unsqueeze(0)
        if len(next_states.shape) == 1:
            next_states = next_states.unsqueeze(0)
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)
            next_q_policy = next_q_policy + (next_masks - 1) * 1e9
            next_actions = next_q_policy.argmax(dim=1)
            next_q_target = self.target_net(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            decay_amount = (1.0 - self.epsilon_min) / 1000000
            self.epsilon = max(self.epsilon_min, self.epsilon - decay_amount)
        return loss.item()
if __name__ == "__main__":
    print("Dueling Double DQN Agent with Action Masking")
    print("Features:")
    print("  - Dueling Architecture (Value + Advantage streams)")
    print("  - Double DQN (Policy + Target networks)")
    print("  - Action Masking (prevents invalid actions)")
    print("  - Experience Replay (100k buffer)")
    print("  - Gradient Clipping")
    print("  - Dropout (0.2)")