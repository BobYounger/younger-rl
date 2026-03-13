from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from rl.networks import build_mlp

from .config import DDPGConfig


class DeterministicActor(nn.Module):
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
        self.network = build_mlp(obs_dim, hidden_dims, action_dim, activation=activation)
        action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)
        action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_low", action_low_tensor)
        self.register_buffer("action_high", action_high_tensor)
        self.register_buffer("action_scale", (action_high_tensor - action_low_tensor) / 2.0)
        self.register_buffer("action_bias", (action_high_tensor + action_low_tensor) / 2.0)
        self.apply(_init_linear_weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized_action = torch.tanh(self.network(obs))
        return normalized_action * self.action_scale + self.action_bias


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


class DDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: DDPGConfig,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.action_low = action_low.astype(np.float32)
        self.action_high = action_high.astype(np.float32)

        self.actor = DeterministicActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
        ).to(self.device)
        self.actor_target = DeterministicActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dims=config.actor_hidden_dims,
            activation=config.activation,
        ).to(self.device)
        self.critic = Critic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
        ).to(self.device)
        self.critic_target = Critic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.critic_hidden_dims,
            activation=config.activation,
        ).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_learning_rate)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_tensor).squeeze(0).cpu().numpy()
        if not deterministic:
            noise = np.random.normal(0.0, self.config.exploration_noise_std, size=action.shape).astype(np.float32)
            noise = np.clip(
                noise,
                -self.config.exploration_noise_clip,
                self.config.exploration_noise_clip,
            )
            action = action + noise
        return np.clip(action, self.action_low, self.action_high)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_actions)
            targets = rewards + (1.0 - dones) * self.config.gamma * target_q

        current_q = self.critic(obs, actions)
        critic_loss = torch.nn.functional.mse_loss(current_q, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        actor_actions = self.actor(obs)
        actor_loss = -self.critic(obs, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(current_q.mean().item()),
            "target_q_mean": float(targets.mean().item()),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_target.load_state_dict(state_dict["actor_target"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.config.tau
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def _init_linear_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0.0)
