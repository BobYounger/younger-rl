from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PPOContinuousConfig:
    env_id: str = "Pendulum-v1"
    seed: int = 0
    device: str = "cpu"

    hidden_dims: tuple[int, ...] = (256, 256)
    activation: str = "tanh"

    learning_rate: float = 3e-4
    rollout_steps: int = 512
    total_steps: int = 100_000
    update_epochs: int = 10
    minibatch_size: int = 128

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    reward_scale: float = 0.1
    normalize_observations: bool = True
    eval_interval: int = 5
    eval_episodes: int = 5
    save_best_checkpoint: bool = True

    log_interval: int = 5
    run_dir: str = "runs/ppo"
    run_name: str | None = None
    plot_metrics: bool = True
    save_gif: bool = True
    gif_episodes: int = 1
    gif_max_steps: int = 300

    @property
    def num_updates(self) -> int:
        return max(1, self.total_steps // self.rollout_steps)
