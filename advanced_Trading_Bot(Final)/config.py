"""
Configuration file for Advanced Trading Bot
Centralized parameter management for easy experimentation
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    """Data fetching and preprocessing configuration"""
    ticker: str = "BTC-USD"
    interval: str = "1h"  # 15m, 1h, 4h, 1d
    train_start: str = "2020-01-01"
    train_end: str = "2024-06-01"
    test_start: str = "2024-06-01"
    test_end: str = "2024-12-01"
    sequence_length: int = 30
    multi_timeframe: bool = True


@dataclass
class EnvironmentConfig:
    """Trading environment configuration"""
    initial_balance: float = 10000.0
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.002  # 0.2%
    slippage: float = 0.001  # 0.1%
    max_drawdown_penalty: float = 0.5
    reward_scaling: float = 1.0
    use_atr_stop: bool = True
    atr_multiplier: float = 2.0


@dataclass
class AgentConfig:
    """DRQN agent configuration"""
    hidden_size: int = 128
    num_gru_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.0001
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    batch_size: int = 32
    use_attention: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_episodes: int = 100
    update_target_every: int = 10
    save_every: int = 10
    render_every: int = 20
    max_steps_per_episode: Optional[int] = None
    early_stopping_patience: int = 20
    early_stopping_threshold: float = -1000.0


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    create_plots: bool = True
    plot_training_progress: bool = True
    plot_whale_detection: bool = True
    plot_trade_analysis: bool = True
    plot_comparison: bool = True
    dpi: int = 300


@dataclass
class Config:
    """Main configuration container"""
    data: DataConfig = DataConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    agent: AgentConfig = AgentConfig()
    training: TrainingConfig = TrainingConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    
    # Output directories
    save_dir: str = "./results"
    model_dir: str = "./models"
    log_dir: str = "./logs"
    
    # Miscellaneous
    random_seed: int = 42
    device: str = "auto"  # "auto", "cuda", or "cpu"
    verbose: int = 1  # 0=silent, 1=normal, 2=debug


# Preset configurations for different use cases

def get_quick_test_config() -> Config:
    """Configuration for quick testing (fast, small model)"""
    config = Config()
    
    # Minimal data
    config.data.train_start = "2024-10-01"
    config.data.train_end = "2024-11-15"
    config.data.test_start = "2024-11-15"
    config.data.test_end = "2024-12-01"
    config.data.sequence_length = 20
    
    # Smaller model
    config.agent.hidden_size = 64
    config.agent.batch_size = 16
    config.agent.buffer_size = 1000
    
    # Short training
    config.training.num_episodes = 5
    config.training.update_target_every = 2
    config.training.save_every = 5
    
    return config


def get_production_config() -> Config:
    """Configuration for production training (full performance)"""
    config = Config()
    
    # Full historical data
    config.data.train_start = "2020-01-01"
    config.data.train_end = "2024-06-01"
    config.data.test_start = "2024-06-01"
    config.data.test_end = "2024-12-01"
    config.data.sequence_length = 60  # Longer sequences
    
    # Larger model
    config.agent.hidden_size = 256
    config.agent.num_gru_layers = 3
    config.agent.batch_size = 64
    config.agent.buffer_size = 50000
    
    # Extensive training
    config.training.num_episodes = 500
    config.training.update_target_every = 10
    config.training.save_every = 25
    
    return config


def get_ethereum_config() -> Config:
    """Configuration for Ethereum trading"""
    config = Config()
    config.data.ticker = "ETH-USD"
    return config


def get_high_frequency_config() -> Config:
    """Configuration for high-frequency trading (15m intervals)"""
    config = Config()
    
    # High frequency data
    config.data.interval = "15m"
    config.data.sequence_length = 40
    
    # Smaller model for faster decisions
    config.agent.hidden_size = 96
    config.agent.learning_rate = 0.0005  # Higher learning rate
    
    return config


def get_conservative_config() -> Config:
    """Configuration for conservative trading (lower risk)"""
    config = Config()
    
    # Conservative parameters
    config.environment.use_atr_stop = True
    config.environment.atr_multiplier = 1.5  # Tighter stop loss
    config.environment.max_drawdown_penalty = 1.0  # Higher drawdown penalty
    
    config.agent.epsilon_decay = 0.99  # Slower exploration decay
    
    return config


def get_aggressive_config() -> Config:
    """Configuration for aggressive trading (higher risk/reward)"""
    config = Config()
    
    # Aggressive parameters
    config.environment.atr_multiplier = 3.0  # Wider stop loss
    config.environment.max_drawdown_penalty = 0.2  # Lower drawdown penalty
    
    config.agent.epsilon_decay = 0.99  # Faster exploration decay
    config.agent.learning_rate = 0.001  # Higher learning rate
    
    return config


# Example usage and configuration display
def print_config(config: Config, name: str = "Configuration"):
    """
    Pretty print configuration
    
    Args:
        config: Configuration object
        name: Name of the configuration
    """
    print("\n" + "="*70)
    print(f"📋 {name}")
    print("="*70)
    
    print("\n🔹 DATA CONFIG:")
    for key, value in config.data.__dict__.items():
        print(f"  {key:.<30} {value}")
    
    print("\n🔹 ENVIRONMENT CONFIG:")
    for key, value in config.environment.__dict__.items():
        print(f"  {key:.<30} {value}")
    
    print("\n🔹 AGENT CONFIG:")
    for key, value in config.agent.__dict__.items():
        print(f"  {key:.<30} {value}")
    
    print("\n🔹 TRAINING CONFIG:")
    for key, value in config.training.__dict__.items():
        print(f"  {key:.<30} {value}")
    
    print("\n🔹 OUTPUT:")
    print(f"  save_dir:.................... {config.save_dir}")
    print(f"  model_dir:................... {config.model_dir}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Show all preset configurations
    
    print("\n" + "="*70)
    print("🎯 AVAILABLE CONFIGURATIONS")
    print("="*70)
    
    configs = {
        "Default": Config(),
        "Quick Test": get_quick_test_config(),
        "Production": get_production_config(),
        "Ethereum": get_ethereum_config(),
        "High Frequency": get_high_frequency_config(),
        "Conservative": get_conservative_config(),
        "Aggressive": get_aggressive_config()
    }
    
    for name, config in configs.items():
        print_config(config, name)
        print("\n" + "-"*70 + "\n")
    
    print("\n" + "="*70)
    print("💡 Usage Example:")
    print("="*70)
    print("""
from config import get_production_config

# Load preset configuration
config = get_production_config()

# Customize specific parameters
config.data.ticker = "ETH-USD"
config.agent.hidden_size = 256
config.training.num_episodes = 200

# Use in training
trainer = TradingBotTrainer(config=config)
    """)
