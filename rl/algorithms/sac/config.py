from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SACConfig:
    env_id: str = "Pendulum-v1"
    seed: int = 0
    device: str = "cpu"

    actor_hidden_dims: tuple[int, ...] = (256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "relu"

    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4

    total_steps: int = 250_000
    replay_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0

    learning_starts: int = 500
    train_frequency: int = 1
    gradient_steps: int = 1

    init_temperature: float = 0.2
    target_entropy: float | None = None

    normalize_observations: bool = False
    eval_interval: int = 5_000
    eval_episodes: int = 5
    save_best_checkpoint: bool = True

    log_interval: int = 1_000
    run_dir: str = "runs/sac"
    run_name: str | None = None
    plot_metrics: bool = True
    save_gif: bool = True
    gif_episodes: int = 1
    gif_max_steps: int = 300
