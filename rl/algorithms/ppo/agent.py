from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from rl.networks import build_mlp

from .config import PPOContinuousConfig


class ActorCriticContinuous(nn.Module):
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
        self.actor_body = build_mlp(obs_dim, hidden_dims, hidden_dims[-1], activation=activation)
        self.critic = build_mlp(obs_dim, hidden_dims, 1, activation=activation)
        self.mu_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)
        self.apply(_init_linear_weights)
        nn.init.uniform_(self.log_std, -0.5, -0.5)

    def policy_dist(self, obs: torch.Tensor) -> Normal:
        features = self.actor_body(obs)
        mu = self.mu_head(features)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class PPOContinuousAgent:
    _tanh_eps = 1e-6

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: PPOContinuousConfig,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = ActorCriticContinuous(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.model.policy_dist(obs_tensor)
        raw_action = dist.mean if deterministic else dist.rsample()
        action = self._squash_action(raw_action)
        log_prob = self._log_prob_from_raw(dist, raw_action)
        value = self.model.value(obs_tensor)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def predict_value(self, obs: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.model.value(obs_tensor).item())

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["log_probs"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)
        old_values = batch["values"].to(self.device)

        batch_size = obs.shape[0]
        minibatch_size = min(self.config.minibatch_size, batch_size)

        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        approx_kl_total = 0.0
        updates = 0
        early_stopped = False

        for _ in range(self.config.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_values = old_values[mb_idx]

                dist = self.model.policy_dist(mb_obs)
                raw_actions = self._unsquash_action(mb_actions)
                new_log_probs = self._log_prob_from_raw(dist, raw_actions)
                entropy = dist.entropy().sum(dim=-1).mean()
                values = self.model.value(mb_obs)

                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_coef,
                    1.0 + self.config.clip_coef,
                )

                policy_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                value_pred_clipped = mb_old_values + (values - mb_old_values).clamp(
                    -self.config.value_clip_coef,
                    self.config.value_clip_coef,
                )
                value_loss_unclipped = (values - mb_returns).pow(2)
                value_loss_clipped = (value_pred_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().abs()

                policy_loss_total += float(policy_loss.item())
                value_loss_total += float(value_loss.item())
                entropy_total += float(entropy.item())
                approx_kl_total += float(approx_kl.item())
                updates += 1

                if approx_kl.item() > self.config.target_kl:
                    early_stopped = True
                    break
            if early_stopped:
                break

        return {
            "policy_loss": policy_loss_total / max(updates, 1),
            "value_loss": value_loss_total / max(updates, 1),
            "entropy": entropy_total / max(updates, 1),
            "approx_kl": approx_kl_total / max(updates, 1),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def _squash_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_action) * self.model.action_scale + self.model.action_bias

    def _unsquash_action(self, action: torch.Tensor) -> torch.Tensor:
        normalized = (action - self.model.action_bias) / self.model.action_scale
        normalized = torch.clamp(normalized, -1.0 + self._tanh_eps, 1.0 - self._tanh_eps)
        return 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))

    def _log_prob_from_raw(self, dist: Normal, raw_action: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        correction = torch.log(self.model.action_scale * (1.0 - squashed.pow(2)) + self._tanh_eps).sum(dim=-1)
        return log_prob - correction


def _init_linear_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        gain = nn.init.calculate_gain("tanh")
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)
