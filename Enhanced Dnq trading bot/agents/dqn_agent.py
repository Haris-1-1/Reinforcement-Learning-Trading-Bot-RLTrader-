import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
class DuelingDQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
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
        qvals = values + (advantages - advantages.mean())
        return qvals
class Agent:
    def __init__(self, state_size, action_size, window_size, is_eval=False):
        self.state_size = state_size
        self.action_size = action_size
        self.window_size = window_size
        self.input_dim = state_size * window_size
        self.memory = deque(maxlen=100000)
        self.inventory = []
        self.is_eval = is_eval
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQNNetwork(self.input_dim, action_size).to(self.device)
        self.target_net = DuelingDQNNetwork(self.input_dim, action_size).to(self.device)
        self.update_target_network()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).view(1, -1).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return np.argmax(q_values.cpu().data.numpy())
    def remember(self, state, action, reward, next_state, done):
        state = state.flatten()
        next_state = next_state.flatten()
        self.memory.append((state, action, reward, next_state, done))
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay