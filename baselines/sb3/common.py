from __future__ import annotations

import os
import sys
import json
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_run_dir(base_dir: str | Path, run_name: str) -> Path:
    run_dir = Path(base_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class MetricsCallback(BaseCallback):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is not None:
                self.episode_returns.append(float(episode["r"]))
                self.episode_lengths.append(int(episode["l"]))
        return True


def make_monitored_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def evaluate_policy_model(model, env_id: str, seed: int, episodes: int) -> float:
    env = gym.make(env_id)
    returns: list[float] = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                done = terminated or truncated
            returns.append(episode_return)
    finally:
        env.close()
    return float(np.mean(returns))


def record_gif(model, env_id: str, seed: int, output_path: Path, max_steps: int = 300) -> None:
    import imageio.v2 as imageio

    env = gym.make(env_id, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(max_steps):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()

    if frames:
        imageio.mimsave(output_path, frames, fps=30, loop=0)


def save_summary(run_dir: Path, summary: dict[str, object]) -> None:
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
