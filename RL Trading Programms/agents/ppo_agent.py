"""
PPO (Proximal Policy Optimization) Agent using Stable-Baselines3.
"""

import numpy as np
from typing import Dict
from stable_baselines3 import PPO
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


class PPOAgent(BaseAgent):
    """PPO Agent for trading."""

    def __init__(self, env, config: Dict):
        """
        Initialize PPO Agent.

        Args:
            env: Trading environment
            config: Configuration dictionary with PPO hyperparameters
        """
        super().__init__(env, config)

        ppo_config = config.get('ppo', {})

        # Create PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=ppo_config.get('learning_rate', 0.0003),
            n_steps=ppo_config.get('n_steps', 2048),
            batch_size=ppo_config.get('batch_size', 64),
            n_epochs=ppo_config.get('n_epochs', 10),
            gamma=ppo_config.get('gamma', 0.99),
            gae_lambda=ppo_config.get('gae_lambda', 0.95),
            clip_range=ppo_config.get('clip_range', 0.2),
            ent_coef=ppo_config.get('ent_coef', 0.0),
            vf_coef=ppo_config.get('vf_coef', 0.5),
            max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
            policy_kwargs=ppo_config.get('policy_kwargs', {'net_arch': [128, 128]}),
            tensorboard_log=config.get('training', {}).get('tensorboard_log'),
            verbose=1
        )

    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """
        Train the PPO agent.

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
            log_interval=kwargs.get('log_interval', 10),
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
        Save PPO model.

        Args:
            path: Path to save model
        """
        self.model.save(path)
        print(f"PPO model saved to {path}")

    def load(self, path: str):
        """
        Load PPO model.

        Args:
            path: Path to load model from
        """
        self.model = PPO.load(path, env=self.env)
        print(f"PPO model loaded from {path}")
