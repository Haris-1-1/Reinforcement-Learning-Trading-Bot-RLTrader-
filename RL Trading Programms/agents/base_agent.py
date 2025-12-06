"""
Base Agent class defining the interface for all RL agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all trading agents."""

    def __init__(self, env: Any, config: Dict):
        """
        Initialize base agent.

        Args:
            env: Trading environment
            config: Agent configuration dictionary
        """
        self.env = env
        self.config = config
        self.model = None

    @abstractmethod
    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """
        Train the agent.

        Args:
            total_timesteps: Number of timesteps to train
            **kwargs: Additional training arguments

        Returns:
            Dictionary with training metrics
        """
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """
        Predict action given observation.

        Args:
            observation: Current state
            deterministic: Whether to use deterministic policy

        Returns:
            Action to take
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load model from disk.

        Args:
            path: Path to load model from
        """
        pass

    def get_name(self) -> str:
        """
        Get agent name.

        Returns:
            Agent name string
        """
        return self.__class__.__name__
