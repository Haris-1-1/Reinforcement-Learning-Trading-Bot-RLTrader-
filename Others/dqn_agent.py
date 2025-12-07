"""
DQN (Deep Q-Network) Agent using Stable-Baselines3.
"""

import numpy as np
from typing import Dict
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from .base_agent import BaseAgent


class TrainingCallback(BaseCallback):
    """Callback for tracking training progress."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1

        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True


class DQNAgent(BaseAgent):
    """DQN Agent for trading."""

    def __init__(self, env, config: Dict):
        """
        Initialize DQN Agent.

        Args:
            env: Trading environment
            config: Configuration dictionary with DQN hyperparameters
        """
        super().__init__(env, config)

        dqn_config = config.get('dqn', {})

        # Create DQN model
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=dqn_config.get('learning_rate', 0.0001),
            buffer_size=dqn_config.get('buffer_size', 100000),
            learning_starts=dqn_config.get('learning_starts', 1000),
            batch_size=dqn_config.get('batch_size', 64),
            tau=dqn_config.get('tau', 0.005),
            gamma=dqn_config.get('gamma', 0.99),
            train_freq=dqn_config.get('train_freq', 4),
            gradient_steps=dqn_config.get('gradient_steps', 1),
            target_update_interval=dqn_config.get('target_update_interval', 1000),
            exploration_fraction=dqn_config.get('exploration_fraction', 0.1),
            exploration_initial_eps=dqn_config.get('exploration_initial_eps', 1.0),
            exploration_final_eps=dqn_config.get('exploration_final_eps', 0.05),
            policy_kwargs=dqn_config.get('policy_kwargs', {'net_arch': [128, 128]}),
            tensorboard_log=config.get('training', {}).get('tensorboard_log'),
            verbose=1
        )

    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """
        Train the DQN agent.

        Args:
            total_timesteps: Number of timesteps to train
            **kwargs: Additional arguments (log_interval, etc.)

        Returns:
            Dictionary with training metrics
        """
        callback = TrainingCallback()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=kwargs.get('log_interval', 1000),
            progress_bar=True
        )

        return {
            'episode_rewards': callback.episode_rewards,
            'episode_lengths': callback.episode_lengths,
            'mean_reward': np.mean(callback.episode_rewards) if callback.episode_rewards else 0.0,
            'total_episodes': len(callback.episode_rewards)
        }

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action given observation.

        Args:
            observation: Current state
            deterministic: Whether to use deterministic policy

        Returns:
            Action to take
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return int(action)

    def save(self, path: str):
        """
        Save DQN model.

        Args:
            path: Path to save model
        """
        self.model.save(path)
        print(f"DQN model saved to {path}")

    def load(self, path: str):
        """
        Load DQN model.

        Args:
            path: Path to load model from
        """
        self.model = DQN.load(path, env=self.env)
        print(f"DQN model loaded from {path}")
