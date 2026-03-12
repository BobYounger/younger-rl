from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class RolloutBuffer:
    """Trajectory buffer for on-policy algorithms such as PPO/A2C."""

    def __init__(
        self,
        obs_shape: int | tuple[int, ...],
        capacity: int,
        action_shape: int | tuple[int, ...] = 1,
        device: str | torch.device = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        action_dtype: np.dtype = np.int64,
    ) -> None:
        obs_shape = _as_shape(obs_shape)
        action_shape = _as_shape(action_shape)

        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_dtype = np.dtype(action_dtype)

        self.observations = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, *action_shape), dtype=self.action_dtype)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.values = np.zeros((self.capacity,), dtype=np.float32)
        self.next_values = np.zeros((self.capacity,), dtype=np.float32)
        self.log_probs = np.zeros((self.capacity,), dtype=np.float32)
        self.advantages = np.zeros((self.capacity,), dtype=np.float32)
        self.returns = np.zeros((self.capacity,), dtype=np.float32)

        self._ptr = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._ptr

    @property
    def size(self) -> int:
        return len(self)

    def reset(self) -> None:
        self._ptr = 0
        self._full = False

    def add(
        self,
        obs: np.ndarray,
        action,
        reward: float,
        done: bool | float,
        value: float,
        next_value: float,
        log_prob: float,
    ) -> None:
        if self._ptr >= self.capacity:
            raise ValueError("Rollout buffer is full. Call reset() after finishing an update.")

        idx = self._ptr
        self.observations[idx] = np.asarray(obs, dtype=np.float32)
        self.actions[idx] = np.asarray(action, dtype=self.action_dtype).reshape(self.action_shape)
        self.rewards[idx] = float(reward)
        self.dones[idx] = float(done)
        self.values[idx] = float(value)
        self.next_values[idx] = float(next_value)
        self.log_probs[idx] = float(log_prob)
        self._ptr += 1
        self._full = self._ptr == self.capacity

    def finish_path(self) -> None:
        size = len(self)
        if size == 0:
            return

        advantages = np.zeros((size,), dtype=np.float32)
        gae = 0.0

        for t in reversed(range(size)):
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * self.next_values[t] * non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages[t] = gae

        self.advantages[:size] = advantages
        self.returns[:size] = advantages + self.values[:size]

    def get(self, normalize_advantages: bool = True) -> Dict[str, torch.Tensor]:
        size = len(self)
        if size == 0:
            raise ValueError("Cannot read from an empty rollout buffer.")

        advantages = self.advantages[:size].copy()
        if normalize_advantages:
            advantages = _normalize(advantages)

        actions = self.actions[:size]
        if self.action_shape == (1,) and np.issubdtype(self.action_dtype, np.integer):
            actions = actions.reshape(-1)

        return {
            "obs": torch.as_tensor(self.observations[:size], dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(actions, device=self.device),
            "rewards": torch.as_tensor(self.rewards[:size], dtype=torch.float32, device=self.device),
            "dones": torch.as_tensor(self.dones[:size], dtype=torch.float32, device=self.device),
            "values": torch.as_tensor(self.values[:size], dtype=torch.float32, device=self.device),
            "next_values": torch.as_tensor(self.next_values[:size], dtype=torch.float32, device=self.device),
            "log_probs": torch.as_tensor(self.log_probs[:size], dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor(self.returns[:size], dtype=torch.float32, device=self.device),
        }


def _as_shape(shape: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (int(shape),)
    return tuple(int(dim) for dim in shape)


def _normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = values.mean()
    std = values.std()
    return (values - mean) / (std + eps)
