"""
Improved Q-Learning Agent with feature discretization.
"""

import numpy as np
import pickle
from typing import Dict, Tuple
from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning agent with state discretization.
    Uses binning to discretize the continuous state space.
    """

    def __init__(self, env, config: Dict):
        """
        Initialize Q-Learning Agent.

        Args:
            env: Trading environment
            config: Configuration dictionary
        """
        super().__init__(env, config)

        q_config = config.get('q_learning', {})

        self.learning_rate = q_config.get('learning_rate', 0.1)
        self.gamma = q_config.get('gamma', 0.99)
        self.epsilon = q_config.get('epsilon_start', 1.0)
        self.epsilon_min = q_config.get('epsilon_end', 0.01)
        self.epsilon_decay = q_config.get('epsilon_decay', 0.995)
        self.n_bins = q_config.get('n_bins', 20)

        # Initialize Q-table
        # We'll use a simplified state space: [price_trend, rsi_level, position]
        n_actions = env.action_space.n
        self.q_table = np.zeros((self.n_bins, self.n_bins, 2, n_actions))

        # For state discretization
        self.price_trend_min = -0.05
        self.price_trend_max = 0.05
        self.rsi_min = 0
        self.rsi_max = 100

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []

    def _discretize_state(self, observation: np.ndarray) -> Tuple[int, int, int]:
        """
        Discretize continuous state into bins.

        Args:
            observation: Continuous state vector

        Returns:
            Tuple of (price_trend_bin, rsi_bin, position_bin)
        """
        # Extract relevant features from observation
        # Assuming observation has Returns and RSI
        # observation structure: [...features..., cash, position, portfolio_value]

        # Extract features (adjust indices based on your feature columns)
        # For simplicity, we use the last few features
        returns = observation[-24]  # Returns feature
        rsi = observation[-21]  # RSI feature (approximate index)
        position = observation[-2]  # Position

        # Discretize price trend (returns)
        price_trend_clipped = np.clip(returns, self.price_trend_min, self.price_trend_max)
        price_trend_bin = int(
            (price_trend_clipped - self.price_trend_min) /
            (self.price_trend_max - self.price_trend_min) *
            (self.n_bins - 1)
        )
        price_trend_bin = np.clip(price_trend_bin, 0, self.n_bins - 1)

        # Discretize RSI
        rsi_normalized = (rsi + 3) / 6  # Assuming RSI was normalized to ~[-3, 3]
        rsi_normalized = np.clip(rsi_normalized, 0, 1)
        rsi_bin = int(rsi_normalized * (self.n_bins - 1))
        rsi_bin = np.clip(rsi_bin, 0, self.n_bins - 1)

        # Position (already discrete: 0 or 1)
        position_bin = int(position)

        return price_trend_bin, rsi_bin, position_bin

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action using epsilon-greedy policy.

        Args:
            observation: Current state
            deterministic: If True, always exploit; if False, use epsilon-greedy

        Returns:
            Action to take
        """
        if not deterministic and np.random.rand() < self.epsilon:
            # Explore
            return self.env.action_space.sample()
        else:
            # Exploit
            state_idx = self._discretize_state(observation)
            action = np.argmax(self.q_table[state_idx])
            return int(action)

    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """
        Train the Q-Learning agent.

        Args:
            total_timesteps: Number of timesteps to train
            **kwargs: Additional arguments

        Returns:
            Dictionary with training metrics
        """
        log_interval = kwargs.get('log_interval', 1000)

        timestep = 0
        episode = 0

        while timestep < total_timesteps:
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done and timestep < total_timesteps:
                # Select action
                action = self.predict(obs, deterministic=False)

                # Take step
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Get state indices
                state_idx = self._discretize_state(obs)
                next_state_idx = self._discretize_state(next_obs)

                # Q-learning update
                old_value = self.q_table[state_idx + (action,)]
                next_max = np.max(self.q_table[next_state_idx])

                # Update Q-value
                new_value = old_value + self.learning_rate * (
                    reward + self.gamma * next_max - old_value
                )
                self.q_table[state_idx + (action,)] = new_value

                obs = next_obs
                episode_reward += reward
                episode_length += 1
                timestep += 1

                if timestep % log_interval == 0:
                    print(f"Timestep: {timestep}/{total_timesteps}, "
                          f"Epsilon: {self.epsilon:.3f}, "
                          f"Episode: {episode}")

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            episode += 1

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'mean_reward': np.mean(self.episode_rewards),
            'total_episodes': len(self.episode_rewards)
        }

    def save(self, path: str):
        """
        Save Q-table and agent parameters.

        Args:
            path: Path to save model
        """
        save_dict = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Q-Learning model saved to {path}")

    def load(self, path: str):
        """
        Load Q-table and agent parameters.

        Args:
            path: Path to load model from
        """
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.q_table = save_dict['q_table']
        self.epsilon = save_dict['epsilon']
        print(f"Q-Learning model loaded from {path}")
