from __future__ import annotations

from typing import Optional

import gymnasium as gym


def make_env(env_id: str, seed: Optional[int] = None, render_mode: str | None = None):
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=int(seed))
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(int(seed))
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(int(seed))
    return env
