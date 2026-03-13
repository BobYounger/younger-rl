from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import numpy as np
from gymnasium.spaces import Box

from rl.buffers import ReplayBuffer
from rl.common import MetricLogger, RunningMeanStd, save_checkpoint, set_seed, setup_logging
from rl.envs import make_env

from .agent import SACAgent
from .config import SACConfig


def train_sac(config: SACConfig) -> Dict[str, float]:
    set_seed(config.seed)
    env = make_env(config.env_id, seed=config.seed)

    if not isinstance(env.action_space, Box):
        env.close()
        raise ValueError(f"{config.env_id} is not a continuous control environment.")

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    obs_shape = env.observation_space.shape
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    obs_normalizer = RunningMeanStd(obs_shape)
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        config=config,
    )
    replay_buffer = ReplayBuffer(
        obs_shape=obs_dim,
        capacity=config.replay_size,
        action_shape=action_dim,
        action_dtype=np.float32,
        device=config.device,
    )

    run_name = config.run_name or f"sac-{config.env_id.lower()}-seed{config.seed}"
    runtime_logger = setup_logging(config.run_dir, run_name=run_name, logger_name=f"sac.{run_name}", console=True)
    metrics_logger = MetricLogger(config.run_dir, run_name=run_name)
    config_path = metrics_logger.save_config(asdict(config))
    run_dir = metrics_logger.log_dir

    obs, _ = env.reset(seed=config.seed)
    obs_normalizer.update(obs)
    episode_return = 0.0
    episode_length = 0
    completed_returns: list[float] = []
    metrics_history: list[Dict[str, float]] = []
    eval_returns: list[float] = []
    best_eval_return = float("-inf")
    best_checkpoint_path: Path | None = None
    last_update_metrics = {
        "actor_loss": float("nan"),
        "critic_loss": float("nan"),
        "alpha_loss": float("nan"),
        "alpha": float("nan"),
        "entropy": float("nan"),
        "q_mean": float("nan"),
        "target_q_mean": float("nan"),
    }

    for step in range(1, config.total_steps + 1):
        normalized_obs = normalize_observation(obs, obs_normalizer, config.normalize_observations)
        if step < config.learning_starts:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.act(normalized_obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
        normalized_next_obs = normalize_observation(next_obs, obs_normalizer, config.normalize_observations)
        obs_normalizer.update(next_obs)
        done = terminated or truncated

        replay_buffer.add(
            obs=np.asarray(normalized_obs, dtype=np.float32),
            action=np.asarray(action, dtype=np.float32),
            reward=float(reward) * config.reward_scale,
            next_obs=np.asarray(normalized_next_obs, dtype=np.float32),
            done=float(terminated),
        )

        episode_return += float(reward)
        episode_length += 1

        if step >= config.learning_starts and step % config.train_frequency == 0:
            update_metrics = []
            for _ in range(config.gradient_steps):
                batch = replay_buffer.sample(config.batch_size)
                update_metrics.append(agent.update(batch))
            last_update_metrics = _aggregate_metrics(update_metrics)

        if done:
            completed_returns.append(episode_return)
            runtime_logger.info(
                "episode done | return=%.3f | length=%d | total_steps=%d",
                episode_return,
                episode_length,
                step,
            )
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
        else:
            obs = next_obs

        eval_return = float("nan")
        if step % config.eval_interval == 0:
            eval_return = evaluate_policy(
                agent=agent,
                env_id=config.env_id,
                seed=config.seed + step,
                episodes=config.eval_episodes,
                obs_normalizer=obs_normalizer,
                normalize_observations=config.normalize_observations,
            )
            eval_returns.append(eval_return)
            if config.save_best_checkpoint and eval_return > best_eval_return:
                best_eval_return = eval_return
                best_checkpoint_path = save_checkpoint(
                    run_dir / "best_checkpoint.pt",
                    model=agent.actor,
                    optimizer=agent.actor_optimizer,
                    step=step,
                    extra={
                        "config_path": str(config_path),
                        "agent_state": agent.state_dict(),
                        "obs_normalizer": obs_normalizer.state_dict(),
                        "best_eval_return": best_eval_return,
                    },
                )

        if step % config.log_interval == 0 or step == config.total_steps:
            mean_return = float(np.mean(completed_returns[-10:])) if completed_returns else float("nan")
            metrics = {
                "total_steps": step,
                "episode_return_mean_10": mean_return,
                "eval_return": eval_return,
                **last_update_metrics,
            }
            metrics_history.append(metrics)
            metrics_logger.log_metrics(metrics, step=step, stdout=False)
            runtime_logger.info(
                "steps=%d | return10=%.3f | actor=%.4f | critic=%.4f | alpha=%.4f | q=%.4f",
                step,
                mean_return,
                last_update_metrics["actor_loss"],
                last_update_metrics["critic_loss"],
                last_update_metrics["alpha"],
                last_update_metrics["q_mean"],
            )
            if step % config.eval_interval == 0:
                runtime_logger.info("eval | episodes=%d | return=%.3f", config.eval_episodes, eval_return)

    env.close()
    checkpoint_path = save_checkpoint(
        run_dir / "checkpoint.pt",
        model=agent.actor,
        optimizer=agent.actor_optimizer,
        step=config.total_steps,
        extra={
            "config_path": str(config_path),
            "agent_state": agent.state_dict(),
            "obs_normalizer": obs_normalizer.state_dict(),
        },
    )
    if config.plot_metrics and metrics_history:
        plot_training_curves(metrics_history, run_dir / "training_curves.png")
    if config.save_gif:
        record_policy_gif(
            agent=agent,
            env_id=config.env_id,
            output_path=run_dir / "policy.gif",
            seed=config.seed,
            episodes=config.gif_episodes,
            max_steps=config.gif_max_steps,
            obs_normalizer=obs_normalizer,
            normalize_observations=config.normalize_observations,
        )

    summary = {
        "total_steps": float(config.total_steps),
        "episode_return_mean_10": float(np.mean(completed_returns[-10:])) if completed_returns else float("nan"),
        "eval_return": float(np.mean(eval_returns[-3:])) if eval_returns else float("nan"),
        "best_eval_return": float(best_eval_return) if eval_returns else float("nan"),
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else "",
        "run_dir": str(run_dir),
        **last_update_metrics,
    }
    return summary


def evaluate_policy(
    agent: SACAgent,
    env_id: str,
    seed: int,
    episodes: int = 5,
    obs_normalizer: RunningMeanStd | None = None,
    normalize_observations: bool = True,
) -> float:
    env = make_env(env_id, seed=seed)
    returns: list[float] = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0
            while not done:
                normalized_obs = normalize_observation(obs, obs_normalizer, normalize_observations)
                action = agent.act(normalized_obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
                episode_return += float(reward)
                done = terminated or truncated
            returns.append(episode_return)
    finally:
        env.close()
    return float(np.mean(returns))


def normalize_observation(
    obs: np.ndarray,
    obs_normalizer: RunningMeanStd | None,
    enabled: bool,
) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if not enabled or obs_normalizer is None:
        return obs
    return obs_normalizer.normalize(obs)


def plot_training_curves(metrics_history: list[Dict[str, float]], output_path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps = [row["total_steps"] for row in metrics_history]
    returns = [row["episode_return_mean_10"] for row in metrics_history]
    actor_losses = [row["actor_loss"] for row in metrics_history]
    critic_losses = [row["critic_loss"] for row in metrics_history]
    alphas = [row["alpha"] for row in metrics_history]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    plots = [
        ("Return(10)", returns),
        ("Actor Loss", actor_losses),
        ("Critic Loss", critic_losses),
        ("Alpha", alphas),
    ]
    for ax, (title, values) in zip(axes, plots):
        ax.plot(steps, values, linewidth=2.0)
        ax.set_title(title)
        ax.set_xlabel("Steps")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def record_policy_gif(
    agent: SACAgent,
    env_id: str,
    output_path: str | Path,
    seed: int = 0,
    episodes: int = 1,
    max_steps: int = 300,
    obs_normalizer: RunningMeanStd | None = None,
    normalize_observations: bool = True,
) -> Path:
    import imageio.v2 as imageio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(env_id, seed=seed, render_mode="rgb_array")
    frames: list[np.ndarray] = []

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            for _ in range(max_steps):
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                normalized_obs = normalize_observation(obs, obs_normalizer, normalize_observations)
                action = agent.act(normalized_obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action.astype(np.float32))
                if terminated or truncated:
                    break
    finally:
        env.close()

    if not frames:
        raise RuntimeError("No frames captured while recording policy GIF.")

    imageio.mimsave(output_path, frames, fps=30, loop=0)
    return output_path


def _aggregate_metrics(metrics: list[Dict[str, float]]) -> Dict[str, float]:
    keys = metrics[0].keys()
    return {key: float(np.mean([row[key] for row in metrics])) for key in keys}
