from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseAgent(ABC):

    def __init__(self, env: Any, config: Dict):
        self.env = env
        self.config = config
        self.model = None

    @abstractmethod
    def train(self, total_timesteps: int, **kwargs) -> Dict:
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    def get_name(self) -> str:
        return self.__class__.__name__
