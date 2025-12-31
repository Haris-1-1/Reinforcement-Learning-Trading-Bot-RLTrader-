import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, Tuple
from agents.base_agent import BaseAgent
class DQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list):
        super(DQNetwork, self).__init__()
        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.4))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, action_dim))
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def forward(self, state):
        return self.network(state)
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    def __len__(self):
        return len(self.buffer)
class DQNAgent(BaseAgent):
    def __init__(self, env, config: Dict):
        super().__init__(env, config)
        dqn_config = config.get('dqn', {})
        self.learning_rate = dqn_config.get('learning_rate', 0.0001)
        self.gamma = dqn_config.get('gamma', 0.99)
        self.epsilon_start = dqn_config.get('epsilon_start', 1.0)
        self.epsilon_end = dqn_config.get('epsilon_end', 0.01)
        self.epsilon_decay_steps = dqn_config.get('epsilon_decay_steps', 50000)
        self.replay_buffer_size = dqn_config.get('replay_buffer_size', 10000)
        self.batch_size = dqn_config.get('batch_size', 64)
        self.target_update_freq = dqn_config.get('target_update_freq', 1000)
        self.hidden_sizes = dqn_config.get('hidden_sizes', [128, 128, 64])
        self.epsilon = self.epsilon_start
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.policy_net = DQNetwork(
            self.state_dim,
            self.action_dim,
            self.hidden_sizes
        ).to(self.device)
        self.target_net = DQNetwork(
            self.state_dim,
            self.action_dim,
            self.hidden_sizes
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_step = 0
        print(f"\n{'='*60}")
        print(f"DQN AGENT INITIALIZED")
        print(f"{'='*60}")
        print(f"State dim: {self.state_dim}")
        print(f"Action dim: {self.action_dim}")
        print(f"Hidden layers: {self.hidden_sizes}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Replay buffer: {self.replay_buffer_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"{'='*60}\n")
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        if not deterministic and random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            action = q_values.argmax(dim=1).item()
        return action
    def _update_epsilon(self, step: int):
        decay = min(step / self.epsilon_decay_steps, 1.0)
        self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * decay
    def _train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return loss.item()
    def train(self, total_timesteps: int, **kwargs) -> Dict:
        log_interval = kwargs.get('log_interval', 10000)
        timestep = 0
        episode = 0
        print(f"\n{'='*60}")
        print(f"DQN TRAINING")
        print(f"{'='*60}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Epsilon: {self.epsilon_start:.2f}  {self.epsilon_end:.2f}")
        print(f"Decay steps: {self.epsilon_decay_steps:,}")
        print(f"{'='*60}\n")
        losses = []
        while timestep < total_timesteps:
            obs, info = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_losses = []
            while not done and timestep < total_timesteps:
                action = self.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                self.replay_buffer.push(obs, action, reward, next_obs, done)
                loss = self._train_step()
                if loss is not None:
                    episode_losses.append(loss)
                if self.training_step % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                self._update_epsilon(timestep)
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                timestep += 1
                self.training_step += 1
                if timestep % log_interval == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                    avg_loss = np.mean(losses[-100:]) if losses else 0
                    print(f"Step {timestep:>7,}/{total_timesteps:,} | "
                          f"Ep {episode:>3} | "
                          f"ε={self.epsilon:.3f} | "
                          f"Avg Reward: {avg_reward:+.4f} | "
                          f"Avg Loss: {avg_loss:.4f} | "
                          f"Buffer: {len(self.replay_buffer):>5,}")
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            if episode_losses:
                losses.extend(episode_losses)
            episode += 1
            if episode % 20 == 0:
                pv = info.get('portfolio_value', 0)
                trades = info.get('trade_count', 0)
                avg_loss = np.mean(episode_losses) if episode_losses else 0
                print(f"  └─ Episode {episode}: Reward={episode_reward:+.4f} | "
                      f"Portfolio=${pv:,.0f} | Trades={trades} | "
                      f"Loss={avg_loss:.4f} | ε={self.epsilon:.3f}")
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Episodes: {episode}")
        print(f"Final Epsilon: {self.epsilon:.4f}")
        print(f"Mean Reward: {np.mean(self.episode_rewards):.4f}")
        print(f"Best Episode Reward: {max(self.episode_rewards):.4f}")
        print(f"{'='*60}\n")
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': losses,
            'mean_reward': float(np.mean(self.episode_rewards)),
            'total_episodes': len(self.episode_rewards),
            'final_epsilon': self.epsilon
        }
    def save(self, path: str):
        save_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'config': self.config,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_sizes': self.hidden_sizes
        }
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    def load(self, path: str):
        save_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(save_dict['policy_net_state_dict'])
        self.target_net.load_state_dict(save_dict['target_net_state_dict'])
        self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        self.epsilon = save_dict.get('epsilon', self.epsilon_end)
        self.training_step = save_dict.get('training_step', 0)
        print(f"Model loaded from {path}")