from __future__ import annotations
from typing import Dict, Optional, List
import numpy as np
import torch
import gymnasium as gym

class ReplayBuffer:
    def __init__(self, state_dim: int, capacity: int, device: str = "cpu") -> None:
        """Replay buffer with ring storage.
        Args:
            state_dim: dimension of state vector
            capacity: max number of transitions to keep (buffer capacity)
            device: torch device for sampled tensors
        """
        self.state_dim = int(state_dim)
        self.capacity = int(capacity)
        self.device = torch.device(device)

        # Ring storage
        self._state       = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_state  = np.zeros((capacity, state_dim), dtype=np.float32)
        self._action      = np.zeros((capacity,), dtype=np.int64)     # discrete index
        self._reward      = np.zeros((capacity,), dtype=np.float32)
        self._done        = np.zeros((capacity,), dtype=np.float32)   # 0.0 or 1.0

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._ptr = 0
        self._size = 0

    def store(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: float) -> None:
        """Store a single transition.
        Args:
            state:      shape (state_dim,)
            action:     discrete action index (int)
            reward:     scalar reward (float)
            next_state: shape (state_dim,)
            done:       0.0 (not terminal) or 1.0 (terminal/truncated)
        """
        idx = self._ptr
        self._state[idx]      = np.asarray(state, dtype=np.float32)
        self._action[idx]     = int(action)
        self._reward[idx]     = float(reward)
        self._next_state[idx] = np.asarray(next_state, dtype=np.float32)
        self._done[idx]       = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # alias
    push = store

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Uniformly sample a mini-batch and return torch tensors on self.device.
        Shapes: state/next_state -> (B, state_dim); action/reward/done -> (B,)
        """
        assert self._size > 0, "ReplayBuffer is empty."
        idxs = np.random.randint(0, self._size, size=batch_size)

        state      = torch.as_tensor(self._state[idxs],      dtype=torch.float32, device=self.device)
        action     = torch.as_tensor(self._action[idxs],     dtype=torch.long,    device=self.device)
        reward     = torch.as_tensor(self._reward[idxs],     dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(self._next_state[idxs], dtype=torch.float32, device=self.device)
        done       = torch.as_tensor(self._done[idxs],       dtype=torch.float32, device=self.device)

        return dict(state=state, action=action, reward=reward, next_state=next_state, done=done)

class RolloutCollector:
    def __init__(
        self,
        env_id: str,
        algorithm,                 # 需实现 select_action(state, eval_mode=False)
        buffer,                    # ReplayBuffer 实例
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.env_id = env_id
        self.algorithm = algorithm
        self.buffer = buffer
        self.device = torch.device(device)

        self.env = gym.make(env_id)
        self._random_state = np.random.RandomState(seed)
        self._env_seed = seed

        # 推断空间维度（仅做自检/日志用途）
        obs, _ = self.env.reset(seed=seed)
        self.state_dim = int(np.array(obs, dtype=np.float32).reshape(-1).shape[0])
        assert hasattr(self.env.action_space, "n"), "需要离散动作空间 env.action_space.n"
        self.act_dim = int(self.env.action_space.n)

        self._state = obs.astype(np.float32)
        self._ep_ret = 0.0
        self._ep_len = 0

        # 统计
        self.total_steps = 0
        self.total_episodes = 0

    def reset_env(self, seed: Optional[int] = None) -> np.ndarray:
        """重置环境并返回初始状态。"""
        if seed is None:
            obs, _ = self.env.reset()
        else:
            obs, _ = self.env.reset(seed=seed)
        self._state = obs.astype(np.float32)
        self._ep_ret = 0.0
        self._ep_len = 0
        return self._state

    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)  # (1, S)
        return s

    def _random_action(self) -> int:
        return int(self._random_state.randint(0, self.act_dim))

    def select_action(self, state: np.ndarray, mode: str = "stochastic") -> int:
        """选择动作。
        mode: "random" 随机；"stochastic" 策略采样；"greedy" 策略贪心（eval_mode=True）
        返回离散动作索引 (int)
        """
        if mode == "random":
            return self._random_action()
        eval_mode = (mode == "greedy")
        with torch.no_grad():
            s = self._state_to_tensor(state)
            a = self.algorithm.select_action(s, eval_mode=eval_mode)
            # 兼容 (B,) 或标量张量
            if isinstance(a, torch.Tensor):
                a = int(a.squeeze().item())
            else:
                a = int(a)
        return a

    def collect(self, n_steps: int, mode: str = "stochastic") -> Dict[str, float]:
        """与环境交互 n_steps 步，并把数据写入经验池。
        返回一个简要日志：完成的回合数、最近若干回合回报均值等。
        """
        done_eps_returns: List[float] = []
        done_eps_lengths: List[int] = []

        for _ in range(n_steps):
            action = self.select_action(self._state, mode=mode)

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done_env = float(terminated or truncated)
            done_train = float(terminated)

            self.buffer.store(
                state=self._state,
                action=action,
                reward=float(reward),
                next_state=next_state.astype(np.float32),
                done=done_train,
            )

            self.total_steps += 1
            self._ep_ret += float(reward)
            self._ep_len += 1

            if done_env:
                done_eps_returns.append(self._ep_ret)
                done_eps_lengths.append(self._ep_len)
                self.total_episodes += 1
                self.reset_env()  # 无种子重置，继续收集
            else:
                self._state = next_state.astype(np.float32)

        # 汇总日志
        mean_ret = float(np.mean(done_eps_returns)) if done_eps_returns else 0.0
        mean_len = float(np.mean(done_eps_lengths)) if done_eps_lengths else 0.0
        return {
            "steps": float(n_steps),
            "episodes_finished": float(len(done_eps_returns)),
            "episode_return_mean": mean_ret,
            "episode_length_mean": mean_len,
            "total_steps": float(self.total_steps),
            "total_episodes": float(self.total_episodes),
        }

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
            

if __name__ == "__main__":
    # quick shape test
    pool = ReplayBuffer(state_dim=4, capacity=5)
    for i in range(7):
        s = np.ones(4, dtype=np.float32) * i
        a = i % 3
        r = float(i)
        s2 = s + 0.5
        d = float(i % 2 == 0)
        pool.store(s, a, r, s2, d)
    batch = pool.sample_batch(3)
    for k, v in batch.items():
        print(k, v.shape, v.dtype)
    print("len=", len(pool))
