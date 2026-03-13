from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

from baselines.sb3.common import (
    MetricsCallback,
    evaluate_policy_model,
    make_monitored_env,
    make_run_dir,
    record_gif,
    save_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SB3 DDPG baseline for Pendulum-v1")
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="ddpg-pendulum")
    parser.add_argument("--run-dir", default="runs/sb3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir(args.run_dir, args.run_name)

    train_env = make_monitored_env(args.env, args.seed)
    eval_env = make_monitored_env(args.env, args.seed + 10_000)
    action_dim = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))

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

    model = DDPG(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )

    model.learn(total_timesteps=args.total_steps, callback=[metrics_callback, eval_callback])
    model.save(run_dir / "final_model")

    final_eval_return = evaluate_policy_model(model, args.env, args.seed + 20_000, args.eval_episodes)
    summary = {
        "algo": "ddpg",
        "env_id": args.env,
        "total_steps": args.total_steps,
        "seed": args.seed,
        "final_eval_return": final_eval_return,
        "num_recorded_episodes": len(metrics_callback.episode_returns),
    }
    save_summary(run_dir, summary)

    if not args.no_gif:
        record_gif(model, args.env, args.seed + 30_000, run_dir / "policy.gif")

    print(summary)


if __name__ == "__main__":
    main()
