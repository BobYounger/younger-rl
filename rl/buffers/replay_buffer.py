from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size replay buffer for off-policy algorithms."""

    def __init__(
        self,
        obs_shape: int | tuple[int, ...],
        capacity: int,
        action_shape: int | tuple[int, ...] = 1,
        device: str | torch.device = "cpu",
        action_dtype: np.dtype = np.int64,
    ) -> None:
        obs_shape = _as_shape(obs_shape)
        action_shape = _as_shape(action_shape)

        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_dtype = np.dtype(action_dtype)

        self.observations = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, *action_shape), dtype=self.action_dtype)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    def reset(self) -> None:
        self._ptr = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action,
        reward: float,
        next_obs: np.ndarray,
        done: bool | float,
    ) -> None:
        idx = self._ptr

        self.observations[idx] = np.asarray(obs, dtype=np.float32)
        self.next_observations[idx] = np.asarray(next_obs, dtype=np.float32)
        self.actions[idx] = np.asarray(action, dtype=self.action_dtype).reshape(self.action_shape)
        self.rewards[idx] = float(reward)
        self.dones[idx] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def extend(self, batch: Mapping[str, np.ndarray]) -> None:
        obs_batch = batch["obs"]
        action_batch = batch["actions"]
        reward_batch = batch["rewards"]
        next_obs_batch = batch["next_obs"]
        done_batch = batch["dones"]

        for obs, action, reward, next_obs, done in zip(
            obs_batch, action_batch, reward_batch, next_obs_batch, done_batch
        ):
            self.add(obs, action, float(reward), next_obs, bool(done))

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        indices = np.random.randint(0, self._size, size=int(batch_size))
        return self._get_batch(indices)

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.sample(batch_size)

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "action_shape": self.action_shape,
            "observations": self.observations.copy(),
            "next_observations": self.next_observations.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "dones": self.dones.copy(),
            "ptr": self._ptr,
            "size": self._size,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        self._ptr = int(state_dict["ptr"])
        self._size = int(state_dict["size"])
        self.observations[...] = np.asarray(state_dict["observations"], dtype=np.float32)
        self.next_observations[...] = np.asarray(state_dict["next_observations"], dtype=np.float32)
        self.actions[...] = np.asarray(state_dict["actions"], dtype=self.actions.dtype)
        self.rewards[...] = np.asarray(state_dict["rewards"], dtype=np.float32)
        self.dones[...] = np.asarray(state_dict["dones"], dtype=np.float32)

    def _get_batch(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        actions = self.actions[indices]
        if self.action_shape == (1,) and np.issubdtype(self.action_dtype, np.integer):
            actions = actions.reshape(-1)

        return {
            "obs": torch.as_tensor(self.observations[indices], dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(actions, device=self.device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(
                self.next_observations[indices], dtype=torch.float32, device=self.device
            ),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device),
        }


def _as_shape(shape: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(int(dim) for dim in shape)
