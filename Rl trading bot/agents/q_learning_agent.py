"""
Improved Q-Learning Agent with feature discretization.
"""

import numpy as np
import pickle
from typing import Dict, Tuple
from agents.base_agent import BaseAgent


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

        # Get feature column names from environment
        self.feature_columns = env.feature_columns
        
        # Find indices for key features
        self._find_feature_indices()

        # Initialize Q-table: [price_trend_bin, rsi_bin, position_bin, action]
        n_actions = env.action_space.n
        self.q_table = np.zeros((self.n_bins, self.n_bins, 2, n_actions))

        # For state discretization
        self.price_trend_min = -0.05
        self.price_trend_max = 0.05

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []

    def _find_feature_indices(self):
        """Find indices of key features in observation vector."""
        # Feature columns + 3 portfolio features (cash, position, portfolio_value)
        n_features = len(self.feature_columns)
        
        # Returns index (in feature columns)
        if 'Returns' in self.feature_columns:
            self.returns_idx = self.feature_columns.index('Returns')
        else:
            self.returns_idx = self.feature_columns.index('Close')  # Fallback
            
        # RSI index
        if 'RSI' in self.feature_columns:
            self.rsi_idx = self.feature_columns.index('RSI')
        else:
            self.rsi_idx = 0  # Fallback
            
        # Position is always at index -2 (second to last in observation)
        self.position_idx = -2
        
        print(f"Feature indices - Returns: {self.returns_idx}, RSI: {self.rsi_idx}")

    def _discretize_state(self, observation: np.ndarray) -> Tuple[int, int, int]:
        """
        Discretize continuous state into bins.

        Args:
            observation: Continuous state vector

        Returns:
            Tuple of (price_trend_bin, rsi_bin, position_bin)
        """
        # Extract features using correct indices
        returns = float(observation[self.returns_idx])
        rsi = float(observation[self.rsi_idx])
        position = float(observation[self.position_idx])

        # Discretize price trend (returns)
        price_trend_clipped = np.clip(returns, self.price_trend_min, self.price_trend_max)
        price_trend_bin = int(
            (price_trend_clipped - self.price_trend_min) /
            (self.price_trend_max - self.price_trend_min) *
            (self.n_bins - 1)
        )
        price_trend_bin = int(np.clip(price_trend_bin, 0, self.n_bins - 1))

        # Discretize RSI (normalized RSI ranges roughly from -3 to +3 after z-score)
        # Map to 0-1 range then to bins
        rsi_normalized = (rsi + 3) / 6  # Assuming z-score normalized
        rsi_normalized = np.clip(rsi_normalized, 0, 1)
        rsi_bin = int(rsi_normalized * (self.n_bins - 1))
        rsi_bin = int(np.clip(rsi_bin, 0, self.n_bins - 1))

        # Position (already discrete: 0 or 1)
        position_bin = int(round(position))
        position_bin = int(np.clip(position_bin, 0, 1))

        return (price_trend_bin, rsi_bin, position_bin)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action using epsilon-greedy policy.

        Args:
            observation: Current state
            deterministic: If True, always exploit; if False, use epsilon-greedy

        Returns:
            Action to take (0=Hold, 1=Buy, 2=Sell)
        """
        if not deterministic and np.random.rand() < self.epsilon:
            # Explore: random action
            return self.env.action_space.sample()
        else:
            # Exploit: best action from Q-table
            state_idx = self._discretize_state(observation)
            action = int(np.argmax(self.q_table[state_idx]))
            return action

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

        print(f"\n{'='*60}")
        print(f"Starting Q-Learning Training")
        print(f"Total timesteps: {total_timesteps}")
        print(f"{'='*60}\n")

        while timestep < total_timesteps:
            obs, info = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done and timestep < total_timesteps:
                # Select action (epsilon-greedy)
                action = self.predict(obs, deterministic=False)

                # Take step in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Get state indices for Q-table
                state_idx = self._discretize_state(obs)
                next_state_idx = self._discretize_state(next_obs)

                # Q-learning update
                old_value = self.q_table[state_idx + (action,)]
                next_max = np.max(self.q_table[next_state_idx])

                # Bellman equation update
                new_value = old_value + self.learning_rate * (
                    reward + self.gamma * next_max - old_value
                )
                self.q_table[state_idx + (action,)] = new_value

                obs = next_obs
                episode_reward += reward
                episode_length += 1
                timestep += 1

                # Log progress
                if timestep % log_interval == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                    print(f"Timestep: {timestep:,}/{total_timesteps:,} | "
                          f"Episode: {episode} | "
                          f"Epsilon: {self.epsilon:.3f} | "
                          f"Avg Reward (last 10): {avg_reward:.4f}")

            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            episode += 1

            # Log episode completion
            if episode % 10 == 0:
                portfolio_value = info.get('portfolio_value', 0)
                print(f"Episode {episode} done | "
                      f"Reward: {episode_reward:.4f} | "
                      f"Portfolio: ${portfolio_value:.2f} | "
                      f"Trades: {info.get('trade_count', 0)}")

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Total Episodes: {episode}")
        print(f"Mean Reward: {np.mean(self.episode_rewards):.4f}")
        print(f"{'='*60}\n")

        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'mean_reward': float(np.mean(self.episode_rewards)),
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
            'config': self.config,
            'feature_columns': self.feature_columns,
            'returns_idx': self.returns_idx,
            'rsi_idx': self.rsi_idx
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
        self.epsilon = save_dict.get('epsilon', 0.01)
        self.feature_columns = save_dict.get('feature_columns', self.feature_columns)
        self.returns_idx = save_dict.get('returns_idx', self.returns_idx)
        self.rsi_idx = save_dict.get('rsi_idx', self.rsi_idx)
        print(f"Q-Learning model loaded from {path}")
