import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import pickle
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
from agents.base_agent import BaseAgent
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = [256, 256]):
        super(ActorCriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_size
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    def forward(self, state):
        x = self.shared(state)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value
    def get_action_probs(self, state):
        x = self.shared(state)
        return self.actor(x)
    def get_value(self, state):
        x = self.shared(state)
        return self.critic(x)
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    def get(self):
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probs),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.dones)
        )
    def __len__(self):
        return len(self.states)
class PPOAgent(BaseAgent):
    def __init__(self,
                 env,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 hidden_sizes: List[int] = [256, 256],
                 device: str = None):
        config = {
            'learning_rate': learning_rate,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'clip_epsilon': clip_epsilon,
            'value_coef': value_coef,
            'entropy_coef': entropy_coef,
            'max_grad_norm': max_grad_norm,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'hidden_sizes': hidden_sizes
        }
        super().__init__(env, config)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        self.state_dim = self._get_state_dim()
        self.action_dim = env.action_space.n
        self.policy = ActorCriticNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_sizes=hidden_sizes
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.buffer = RolloutBuffer()
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_values = []
        self._print_agent_info()
    def _get_state_dim(self):
        obs, _ = self.env.reset()
        return len(obs)
    def _print_agent_info(self):
        print("="*60)
        print("PPO AGENT INITIALIZED")
        print("="*60)
        print(f"State dim: {self.state_dim}")
        print(f"Action dim: {self.action_dim}")
        print(f"Hidden layers: {self.policy.shared}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"Gamma (discount): {self.gamma}")
        print(f"GAE Lambda: {self.gae_lambda}")
        print(f"Clip epsilon: {self.clip_epsilon}")
        print(f"Epochs per batch: {self.n_epochs}")
        print(f"Batch size: {self.batch_size}")
        print("="*60)
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.policy(state_tensor)
        if deterministic:
            action = torch.argmax(action_probs, dim=1).item()
            log_prob = torch.log(action_probs[0, action])
        else:
            dist = Categorical(action_probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action))
        return action, log_prob.item(), value.item()
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = np.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
        returns = advantages + values
        return advantages, returns
    def update(self):
        states, actions, old_log_probs, rewards, values, dones = self.buffer.get()
        next_state = states[-1]
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.policy(next_state_tensor)
            next_value = next_value.item()
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        for epoch in range(self.n_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                action_probs, values = self.policy(batch_states)
                dist = Categorical(action_probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(value_loss.item())
                self.entropy_values.append(entropy.item())
        self.buffer.clear()
    def train(self, total_timesteps: int, log_interval: int = 10000, update_interval: int = 2048):
        print("\n" + "="*60)
        print("PPO TRAINING")
        print("="*60)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Update interval: {update_interval:,} steps")
        print(f"Epochs per update: {self.n_epochs}")
        print("="*60 + "\n")
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        for step in range(total_timesteps):
            action, log_prob, value = self.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.buffer.store(obs, action, log_prob, reward, value, done)
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            if len(self.buffer) >= update_interval or done:
                self.update()
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
            if (step + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                avg_length = np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0
                avg_actor_loss = np.mean(self.actor_losses[-100:]) if self.actor_losses else 0
                avg_critic_loss = np.mean(self.critic_losses[-100:]) if self.critic_losses else 0
                avg_entropy = np.mean(self.entropy_values[-100:]) if self.entropy_values else 0
                print(f"Step {step+1:,}/{total_timesteps:,}")
                print(f"  Avg Reward (100 ep): {avg_reward:.2f}")
                print(f"  Avg Length (100 ep): {avg_length:.1f}")
                print(f"  Actor Loss: {avg_actor_loss:.4f}")
                print(f"  Critic Loss: {avg_critic_loss:.4f}")
                print(f"  Entropy: {avg_entropy:.4f}")
                print()
        print("="*60)
        print("✓ Training completed!")
        print("="*60)
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'entropy_values': self.entropy_values
        }
    def predict(self, state: np.ndarray) -> int:
        action, _, _ = self.select_action(state, deterministic=True)
        return action
    def save(self, filepath: str):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'entropy_values': self.entropy_values
        }, filepath)
        print(f"✓ PPO Agent saved to {filepath}")
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.actor_losses = checkpoint.get('actor_losses', [])
        self.critic_losses = checkpoint.get('critic_losses', [])
        self.entropy_values = checkpoint.get('entropy_values', [])
        print(f"✓ PPO Agent loaded from {filepath}")