from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from baselines.sb3.common import make_run_dir


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SB3 PPO baseline for Pendulum-v1")
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="ppo-pendulum")
    parser.add_argument("--run-dir", default="runs/sb3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def evaluate_policy(model: PPO, env_id: str, seed: int, episodes: int) -> float:
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


def record_gif(model: PPO, env_id: str, seed: int, output_path: Path, max_steps: int = 300) -> None:
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


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.run_dir, args.run_name)

    train_env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, args.seed + 10_000)

    metrics_callback = MetricsCallback()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    model.learn(total_timesteps=args.total_steps, callback=[metrics_callback, eval_callback])
    model.save(run_dir / "final_model")

    final_eval_return = evaluate_policy(model, args.env, args.seed + 20_000, args.eval_episodes)
    summary = {
        "env_id": args.env,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "final_eval_return": final_eval_return,
        "num_recorded_episodes": len(metrics_callback.episode_returns),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_gif:
        record_gif(model, args.env, args.seed + 30_000, run_dir / "policy.gif")

    print(summary)


if __name__ == "__main__":
    main()
