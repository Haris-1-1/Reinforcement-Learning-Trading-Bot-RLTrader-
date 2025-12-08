"""
Improved Q-Learning Agent with FASTER Epsilon Decay.

FIXES:
1. Epsilon decay viel schneller (0.99 statt 0.995)
2. Mehr Bins für bessere Diskretisierung
3. Bessere Feature-Extraktion
"""

import numpy as np
import pickle
from typing import Dict, Tuple
from agents.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent with state discretization.
    
    Verbesserungen:
    - Schnellerer Epsilon Decay (erreicht 0.01 in ~460 Episodes statt ~920)
    - Mehr State-Bins für feinere Unterscheidung
    - Zusätzliche Features: MACD für Trend-Erkennung
    """

    def __init__(self, env, config: Dict):
        """Initialize Q-Learning Agent."""
        super().__init__(env, config)

        q_config = config.get('q_learning', {})

        self.learning_rate = q_config.get('learning_rate', 0.1)
        self.gamma = q_config.get('gamma', 0.99)
        
        # ════════════════════════════════════════════════════════════
        # FIX: SCHNELLERER EPSILON DECAY!
        # ════════════════════════════════════════════════════════════
        self.epsilon = q_config.get('epsilon_start', 1.0)
        self.epsilon_min = q_config.get('epsilon_end', 0.01)
        self.epsilon_decay = q_config.get('epsilon_decay', 0.99)  # War 0.995!
        
        # Mehr Bins = feinere Unterscheidung
        self.n_bins = q_config.get('n_bins', 15)

        # Feature indices
        self.feature_columns = env.feature_columns
        self._find_feature_indices()

        # Q-table: [returns_bin, rsi_bin, macd_bin, position, action]
        # NEU: 4D State statt 3D für bessere Entscheidungen
        n_actions = env.action_space.n
        self.q_table = np.zeros((self.n_bins, self.n_bins, self.n_bins, 2, n_actions))
        
        print(f"Q-table shape: {self.q_table.shape}")
        print(f"Epsilon decay: {self.epsilon_decay} (reaches 0.01 after ~{int(np.log(0.01)/np.log(self.epsilon_decay))} episodes)")

        # Discretization bounds
        self.returns_min, self.returns_max = -0.10, 0.10  # ±10% daily return

        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []

    def _find_feature_indices(self):
        """Find indices of key features."""
        # Returns
        if 'Returns' in self.feature_columns:
            self.returns_idx = self.feature_columns.index('Returns')
        else:
            self.returns_idx = 0
            
        # RSI
        if 'RSI' in self.feature_columns:
            self.rsi_idx = self.feature_columns.index('RSI')
        else:
            self.rsi_idx = 0
            
        # MACD (NEU!)
        if 'MACD_hist' in self.feature_columns:
            self.macd_idx = self.feature_columns.index('MACD_hist')
        elif 'MACD' in self.feature_columns:
            self.macd_idx = self.feature_columns.index('MACD')
        else:
            self.macd_idx = 0
            
        self.position_idx = -2  # Position ist immer vorletzte

    def _discretize_state(self, observation: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Discretize state into bins.
        
        Returns:
            Tuple of (returns_bin, rsi_bin, macd_bin, position_bin)
        """
        returns = float(observation[self.returns_idx])
        rsi = float(observation[self.rsi_idx])
        macd = float(observation[self.macd_idx])
        position = float(observation[self.position_idx])

        # Returns bin
        returns_clipped = np.clip(returns, self.returns_min, self.returns_max)
        returns_bin = int((returns_clipped - self.returns_min) / 
                         (self.returns_max - self.returns_min) * (self.n_bins - 1))
        returns_bin = int(np.clip(returns_bin, 0, self.n_bins - 1))

        # RSI bin (normalized RSI is roughly -3 to +3)
        rsi_norm = (rsi + 3) / 6
        rsi_norm = np.clip(rsi_norm, 0, 1)
        rsi_bin = int(rsi_norm * (self.n_bins - 1))
        rsi_bin = int(np.clip(rsi_bin, 0, self.n_bins - 1))
        
        # MACD bin (normalized MACD is roughly -3 to +3)
        macd_norm = (macd + 3) / 6
        macd_norm = np.clip(macd_norm, 0, 1)
        macd_bin = int(macd_norm * (self.n_bins - 1))
        macd_bin = int(np.clip(macd_bin, 0, self.n_bins - 1))

        # Position bin
        position_bin = int(round(position))
        position_bin = int(np.clip(position_bin, 0, 1))

        return (returns_bin, rsi_bin, macd_bin, position_bin)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Predict action."""
        if not deterministic and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_idx = self._discretize_state(observation)
            action = int(np.argmax(self.q_table[state_idx]))
            return action

    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """Train the agent."""
        log_interval = kwargs.get('log_interval', 5000)

        timestep = 0
        episode = 0

        print(f"\n{'='*60}")
        print(f"Q-LEARNING TRAINING")
        print(f"{'='*60}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Epsilon: {self.epsilon:.2f} → {self.epsilon_min:.2f}")
        print(f"Decay rate: {self.epsilon_decay}")
        print(f"{'='*60}\n")

        while timestep < total_timesteps:
            obs, info = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done and timestep < total_timesteps:
                action = self.predict(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Q-Learning update
                state_idx = self._discretize_state(obs)
                next_state_idx = self._discretize_state(next_obs)

                old_value = self.q_table[state_idx + (action,)]
                next_max = np.max(self.q_table[next_state_idx])

                new_value = old_value + self.learning_rate * (
                    reward + self.gamma * next_max - old_value
                )
                self.q_table[state_idx + (action,)] = new_value

                obs = next_obs
                episode_reward += reward
                episode_length += 1
                timestep += 1

                if timestep % log_interval == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
                    print(f"Step {timestep:>7,}/{total_timesteps:,} | "
                          f"Ep {episode:>3} | "
                          f"ε={self.epsilon:.3f} | "
                          f"Avg Reward: {avg_reward:+.4f}")

            # ════════════════════════════════════════════════════════════
            # EPSILON DECAY nach jeder Episode
            # ════════════════════════════════════════════════════════════
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            episode += 1

            # Detaillierter Log alle 20 Episodes
            if episode % 20 == 0:
                pv = info.get('portfolio_value', 0)
                trades = info.get('trade_count', 0)
                print(f"  └─ Episode {episode}: Reward={episode_reward:+.4f} | "
                      f"Portfolio=${pv:,.0f} | Trades={trades} | ε={self.epsilon:.3f}")

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
            'mean_reward': float(np.mean(self.episode_rewards)),
            'total_episodes': len(self.episode_rewards),
            'final_epsilon': self.epsilon
        }

    def save(self, path: str):
        """Save model."""
        save_dict = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'config': self.config,
            'feature_columns': self.feature_columns,
            'returns_idx': self.returns_idx,
            'rsi_idx': self.rsi_idx,
            'macd_idx': self.macd_idx
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)

        self.q_table = save_dict['q_table']
        self.epsilon = save_dict.get('epsilon', 0.01)
        print(f"Model loaded from {path}")
