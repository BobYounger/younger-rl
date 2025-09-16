from dataclasses import dataclass
from typing import Optional

@dataclass
class SACContinuousConfig:
    """SAC Continuous algorithm hyperparameters configuration."""

    # Environment & Global
    env_id: str = "LunarLanderContinuous-v3"
    seed: int = 0
    device: str = "cuda"

    # Network architecture
    net_hidden_dim: int = 256
    net_activation: str = "relu"

    # Algorithm parameters
    algo_lr_actor: float = 3e-4
    algo_lr_critic: float = 3e-4
    algo_lr_alpha: float = 3e-4
    algo_gamma: float = 0.99
    algo_tau: float = 0.005
    algo_target_entropy: Optional[float] = None  # Will be set to -action_dim

    # Training parameters
    train_total_steps: int = 300_000
    train_start_steps: int = 5_000      # Random exploration steps
    train_batch_size: int = 256
    train_update_after: int = 1_000     # Start training after this many steps
    train_update_every: int = 1
    train_eval_every: int = 5_000
    train_eval_episodes: int = 5

    # ReplayBuffer parameters
    buffer_capacity: int = 1_000_000

    # Visualization
    save_gif: bool = True
    gif_max_steps: int = 200            # Max steps per episode in GIF
    gif_episodes: int = 3               # Number of episodes to record

    @classmethod
    def pendulum(cls):
        """Preset for Pendulum-v1"""
        return cls(
            env_id="Pendulum-v1",
            train_total_steps=50_000,
            train_start_steps=10_000,
            train_eval_every=5_000
        )

    @classmethod
    def lunarlander(cls):
        """Preset for LunarLanderContinuous-v3"""
        return cls(
            env_id="LunarLanderContinuous-v3",
            train_total_steps=300_000,
            train_start_steps=5_000,
            train_eval_every=5_000
        )

# Backward compatibility
@dataclass
class SACConfig(SACContinuousConfig):
    """Alias for backward compatibility"""
    pass