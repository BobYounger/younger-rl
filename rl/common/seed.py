from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic and torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def seed_env(env, seed: Optional[int] = None):
    """Seed a Gymnasium-style environment and its spaces."""
    if seed is None:
        return env.reset()

    obs, info = env.reset(seed=int(seed))

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(int(seed))
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(int(seed))

    return obs, info
