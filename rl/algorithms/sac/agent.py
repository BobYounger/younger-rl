from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from rl.networks import build_mlp

from .config import SACConfig


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
TANH_EPS = 1e-6


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dims: tuple[int, ...],
        activation: str,
    ) -> None:
        super().__init__()
        feature_dim = hidden_dims[-1]
        self.backbone = build_mlp(obs_dim, hidden_dims, feature_dim, activation=activation)
        self.mu_head = nn.Linear(feature_dim, action_dim)
        self.log_std_head = nn.Linear(feature_dim, action_dim)

        action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_low", action_low_tensor)
        self.register_buffer("action_high", action_high_tensor)
        self.register_buffer("action_scale", (action_high_tensor - action_low_tensor) / 2.0)
        self.register_buffer("action_bias", (action_high_tensor + action_low_tensor) / 2.0)
        self.apply(_init_linear_weights)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(obs)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self(obs)
        if deterministic:
            raw_action = mu
            dist = None
        else:
            dist = Normal(mu, log_std.exp())
            raw_action = dist.rsample()

        squashed = torch.tanh(raw_action)
        action = squashed * self.action_scale + self.action_bias

        if deterministic:
            log_prob = torch.zeros(obs.shape[0], device=obs.device, dtype=torch.float32)
        else:
            assert dist is not None
            log_prob = dist.log_prob(raw_action).sum(dim=-1)
            correction = torch.log(self.action_scale * (1.0 - squashed.pow(2)) + TANH_EPS).sum(dim=-1)
            log_prob = log_prob - correction

        return action, log_prob


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
    ) -> None:
        super().__init__()
        self.q_network = build_mlp(obs_dim + action_dim, hidden_dims, 1, activation=activation)
        self.apply(_init_linear_weights)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q_network(torch.cat([obs, action], dim=-1)).squeeze(-1)


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: SACConfig,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.action_low = action_low.astype(np.float32)
        self.action_high = action_high.astype(np.float32)
        self.target_entropy = config.target_entropy if config.target_entropy is not None else -float(action_dim)

        self.actor = SquashedGaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
        ).to(self.device)
        self.critic1 = Critic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
        ).to(self.device)
        self.critic2 = Critic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
        ).to(self.device)
        self.critic1_target = Critic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
        ).to(self.device)
        self.critic2_target = Critic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
        ).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.log_alpha = torch.tensor(
            np.log(config.init_temperature),
            device=self.device,
            dtype=torch.float32,
            requires_grad=True,
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=config.critic_learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=config.critic_learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_learning_rate)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor.sample(obs_tensor, deterministic=deterministic)
        action_np = action.squeeze(0).cpu().numpy()
        return np.clip(action_np, self.action_low, self.action_high)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs, deterministic=False)
            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_probs
            targets = rewards + (1.0 - dones) * self.config.gamma * target_q

        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        critic1_loss = torch.nn.functional.mse_loss(current_q1, targets)
        critic2_loss = torch.nn.functional.mse_loss(current_q2, targets)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), max_norm=0.5)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        nn.utils.clip_grad_norm_(self.critic2.parameters(), max_norm=0.5)
        self.critic2_optimizer.step()

        sampled_actions, log_probs = self.actor.sample(obs, deterministic=False)
        q1_pi = self.critic1(obs, sampled_actions)
        q2_pi = self.critic2(obs, sampled_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_probs - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float((0.5 * (critic1_loss + critic2_loss)).item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "entropy": float((-log_probs).mean().item()),
            "q_mean": float(q_pi.mean().item()),
            "target_q_mean": float(targets.mean().item()),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.critic1.load_state_dict(state_dict["critic1"])
        self.critic2.load_state_dict(state_dict["critic2"])
        self.critic1_target.load_state_dict(state_dict["critic1_target"])
        self.critic2_target.load_state_dict(state_dict["critic2_target"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(state_dict["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(state_dict["critic2_optimizer"])
        self.log_alpha.data.copy_(torch.as_tensor(state_dict["log_alpha"], device=self.device))
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.config.tau
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def _init_linear_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0.0)
