from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.algorithms.ddpg import DDPGConfig, train_ddpg
from rl.algorithms.ppo import PPOContinuousConfig, train_ppo_continuous
from rl.algorithms.sac import SACConfig, train_sac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train reinforcement learning agents.")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "ddpg", "sac"])
    parser.add_argument("--env", default="Pendulum-v1")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    kwargs = {}
    if args.total_steps is not None:
        kwargs["total_steps"] = args.total_steps

    if args.algo == "ppo":
        if args.rollout_steps is not None:
            kwargs["rollout_steps"] = args.rollout_steps
        config = PPOContinuousConfig(
            env_id=args.env,
            device=args.device,
            seed=args.seed,
            run_name=args.run_name,
            plot_metrics=not args.no_plot,
            save_gif=not args.no_gif,
            **kwargs,
        )
        summary = train_ppo_continuous(config)
    elif args.algo == "ddpg":
        config = DDPGConfig(
            env_id=args.env,
            device=args.device,
            seed=args.seed,
            run_name=args.run_name,
            plot_metrics=not args.no_plot,
            save_gif=not args.no_gif,
            **kwargs,
        )
        summary = train_ddpg(config)
    elif args.algo == "sac":
        config = SACConfig(
            env_id=args.env,
            device=args.device,
            seed=args.seed,
            run_name=args.run_name,
            plot_metrics=not args.no_plot,
            save_gif=not args.no_gif,
            **kwargs,
        )
        summary = train_sac(config)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algo}")
    print(summary)


if __name__ == "__main__":
    main()
