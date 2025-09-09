from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

EPS = 1e-8

class Actor(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden=(256, 256)) -> None:
        super().__init__()
        self.act_dim = int(act_dim)
        layers = []
        in_f = state_dim
        for h in hidden:
            layers.append(nn.Linear(in_f, h))
            layers.append(nn.ReLU())
            in_f = h
        layers.append(nn.Linear(in_f, act_dim))  # 输出动作概率logits
        self.net = nn.Sequential(*layers)

        # 统一初始化（Xavier + bias=0）
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回动作概率分布的logits。
        state: (B, state_dim)
        returns: logits (B, act_dim)
        """
        return self.net(state)
    
    def get_probs(self, state: torch.Tensor) -> torch.Tensor:
        """返回动作概率分布。
        state: (B, state_dim) 
        returns: probs (B, act_dim)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样（或确定性）动作及其对数概率。
        返回:
            action: 动作索引, shape (B,)
            log_prob: 对数概率, shape (B,)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            # 选择概率最大的动作
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)).squeeze(-1) + EPS)
        else:
            # 按概率分布采样
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob

    @torch.no_grad()
    def act(self, state: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        """给评估/环境交互用的便捷方法。
        返回离散动作索引。
        """
        action, _ = self.sample(state, deterministic=eval_mode)
        return action

class QCritic(nn.Module):
    """State → Action-Value vector: Q(s, ·)

    Args:
        state_dim (int): 状态维度
        act_dim (int): 离散动作数目
        hidden (tuple[int,int]): 隐藏层宽度，默认 (256, 256)
    """
    def __init__(self, state_dim: int, act_dim: int, hidden=(256, 256)) -> None:
        super().__init__()
        layers = []
        in_f = state_dim
        for h in hidden:
            layers.append(nn.Linear(in_f, h))
            layers.append(nn.ReLU())
            in_f = h
        layers.append(nn.Linear(in_f, act_dim))  # 输出每个动作的 Q 值
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回所有动作的 Q 值向量 Q(s, ·)，形状 (B, act_dim)。"""
        return self.net(state)

    def q_of_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """返回指定动作的 Q(s,a)。
        Args:
            state: (B, state_dim)
            action: (B,) 或 (B,1) 的整型索引
        Returns:
            q_sa: (B,) 的标量向量
        """
        q_all = self.forward(state)  # (B, act_dim)
        if action.dim() == 2:
            action = action.squeeze(-1)
        q_sa = q_all.gather(-1, action.long().unsqueeze(-1)).squeeze(-1)
        return q_sa

    @torch.no_grad()
    def greedy_action(self, state: torch.Tensor) -> torch.Tensor:
        """基于当前 Q 估计的贪婪动作 argmax_a Q(s,a)，返回形状 (B,)。"""
        q_all = self.forward(state)
        return torch.argmax(q_all, dim=-1)

class VCritic(nn.Module):
    """State → Value scalar: V(s)

    Args:
        state_dim (int): 状态维度
        hidden (tuple[int,int]): 隐藏层宽度，默认 (256, 256)
    """
    def __init__(self, state_dim: int, hidden=(256, 256)) -> None:
        super().__init__()
        layers = []
        in_f = state_dim
        for h in hidden:
            layers.append(nn.Linear(in_f, h))
            layers.append(nn.ReLU())
            in_f = h
        layers.append(nn.Linear(in_f, 1))  # 输出标量 V(s)
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """返回 V(s)，形状 (B,)。"""
        v = self.net(state)            # (B, 1)
        return v.squeeze(-1)         # → (B,)

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> torch.Tensor:
        """评估模式便捷接口：直接复用 forward，保持单一逻辑。"""
        return self.forward(state)     # (B,)

