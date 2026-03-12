from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Track running mean and variance for streaming normalization."""

    def __init__(self, shape: int | tuple[int, ...], epsilon: float = 1e-4) -> None:
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = tuple(int(dim) for dim in shape)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.shape):
            x = x.reshape(1, *self.shape)

        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        normalized = (x - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)
        return np.clip(normalized, -clip, clip)

    def state_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.mean = np.asarray(state_dict["mean"], dtype=np.float64)
        self.var = np.asarray(state_dict["var"], dtype=np.float64)
        self.count = float(state_dict["count"])

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-8)
        self.count = total_count
