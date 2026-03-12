from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.algorithms.ppo import PPOContinuousConfig, train_ppo_continuous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reinforcement learning agents.")
    parser.add_argument("--algo", default="ppo", choices=["ppo"])
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.algo != "ppo":
        raise ValueError(f"Unsupported algorithm: {args.algo}")

    config = PPOContinuousConfig(
        env_id=args.env,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        device=args.device,
        seed=args.seed,
        run_name=args.run_name,
        plot_metrics=not args.no_plot,
        save_gif=not args.no_gif,
    )
    summary = train_ppo_continuous(config)
    print(summary)


if __name__ == "__main__":
    main()
