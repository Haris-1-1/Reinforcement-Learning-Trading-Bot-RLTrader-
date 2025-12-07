"""
Configuration loader for YAML config files.
"""

import yaml
from typing import Dict, Any
import os


class ConfigLoader:
    """Load and parse YAML configuration files."""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print(f"Loaded configuration from {config_path}")
        return config

    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary
            save_path: Path to save YAML file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Saved configuration to {save_path}")

    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """
        Merge two configurations (override takes precedence).

        Args:
            base_config: Base configuration
            override_config: Override configuration

        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged
