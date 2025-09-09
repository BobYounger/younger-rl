from dataclasses import dataclass
from typing import Optional

@dataclass
class SACConfig:
    """SAC algorithm hyperparameters configuration."""
    
    # Environment & Global
    env_id: str = "CartPole-v1"
    seed: int = 0
    device: str = "cpu"
    
    # Network architecture (Actor, QCritic, VCritic)
    net_hidden_sizes: tuple = (128, 128)
    net_activation: str = "relu"
    
    # Algorithm training parameters (SACDiscreteVAlgorithm)
    algo_lr_actor: float = 3e-4
    algo_lr_critic: float = 3e-4
    algo_lr_temperature: float = 3e-4
    algo_gamma: float = 0.99
    algo_tau: float = 0.005
    algo_alpha: float = 0.1
    algo_auto_alpha: bool = False
    algo_target_entropy: Optional[float] = None
    max_grad_norm: float = 10.0
    
    # ReplayBuffer parameters
    buffer_capacity: int = 1_000_000
    buffer_batch_size: int = 512
    
    # RolloutCollector parameters
    collector_warmup_steps: int = 5_000
    collector_eval_episodes: int = 10
    
    # Training loop parameters
    train_total_steps: int = 1_000_000
    train_update_every: int = 1
    train_soft_update_every: int = 1
    train_eval_every: int = 5_000
    
    # Logging and saving
    log_every: int = 1_000
    save_every: int = 50_000
    save_dir: str = "./checkpoints"