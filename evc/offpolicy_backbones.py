from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .merged_core import Actor, Critic, to_numpy_1d


class GaussianActor(nn.Module):
    """Squashed Gaussian actor whose forward pass is deterministic."""

    def __init__(
        self,
        obs_dim: int = 11,
        action_dim: int = 1,
        hidden_dim: int = 256,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, self.action_dim)
        self.fc_log_std = nn.Linear(self.hidden_dim, self.action_dim)
        self.register_buffer("action_scale", torch.ones(self.action_dim, dtype=torch.float32))
        self.register_buffer("action_bias", torch.zeros(self.action_dim, dtype=torch.float32))

    def _distribution(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        hidden = F.relu(self.fc1(obs.float()))
        hidden = F.relu(self.fc2(hidden))
        mean = self.fc_mu(hidden)
        log_std = self.fc_log_std(hidden).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._distribution(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mean_action(obs)

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._distribution(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()
        squashed = torch.tanh(raw_action)
        action = squashed * self.action_scale + self.action_bias
        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(self.action_scale * (1.0 - squashed.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def log_prob_entropy(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Log probability of a bounded action and base-Gaussian entropy."""
        mean, log_std = self._distribution(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        normalized = ((action - self.action_bias) / self.action_scale).clamp(-0.999999, 0.999999)
        raw_action = torch.atanh(normalized)
        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(self.action_scale * (1.0 - normalized.pow(2)) + 1e-6)
        return log_prob.sum(dim=-1), normal.entropy().sum(dim=-1)


class MinTwinCritic(nn.Module):
    """Attack-facing scalar critic for algorithms trained with twin Q networks."""

    def __init__(self, critic1: nn.Module, critic2: nn.Module) -> None:
        super().__init__()
        self.critic1 = critic1
        self.critic2 = critic2

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.minimum(self.critic1(obs, action), self.critic2(obs, action))


class TD3Agent:
    algorithm = "td3"

    def __init__(
        self,
        device: torch.device,
        *,
        obs_dim: int = 11,
        action_dim: int = 1,
        hidden_dim: int = 256,
        gamma: float = 0.9,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        exploration_noise: float = 0.1,
    ) -> None:
        self.device = device
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.actor = Actor(self.obs_dim, self.action_dim, self.hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.critic1 = Critic(self.obs_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic2 = Critic(self.obs_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1).to(device)
        self.critic2_target = copy.deepcopy(self.critic2).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=float(critic_lr))
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=float(critic_lr))
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_delay = max(1, int(policy_delay))
        self.exploration_noise = float(exploration_noise)
        self.update_count = 0

    @torch.no_grad()
    def act_tensor(self, obs: torch.Tensor, *, explore: bool) -> torch.Tensor:
        action = self.actor(obs.to(self.device))
        if explore and self.exploration_noise > 0.0:
            action = action + torch.randn_like(action) * self.exploration_noise
        return action.clamp(-1.0, 1.0)

    @torch.no_grad()
    def act(self, obs, *, explore: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(to_numpy_1d(obs), dtype=torch.float32, device=self.device)
        return self.act_tensor(obs_t, explore=explore).reshape(-1).cpu().numpy().astype(np.float32)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.update_count += 1
        with torch.no_grad():
            noise = (torch.randn_like(batch["actions"]) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(batch["next_observations"]) + noise).clamp(-1.0, 1.0)
            target_q = torch.minimum(
                self.critic1_target(batch["next_observations"], next_action),
                self.critic2_target(batch["next_observations"], next_action),
            ).reshape(-1)
            target = batch["rewards"] + (1.0 - batch["dones"]) * self.gamma * target_q

        q1 = self.critic1(batch["observations"], batch["actions"]).reshape(-1)
        q2 = self.critic2(batch["observations"], batch["actions"]).reshape(-1)
        critic1_loss = F.mse_loss(q1, target)
        critic2_loss = F.mse_loss(q2, target)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        actor_loss_value = float("nan")
        if self.update_count % self.policy_delay == 0:
            actor_action = self.actor(batch["observations"])
            actor_loss = -self.critic1(batch["observations"], actor_action).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            actor_loss_value = float(actor_loss.detach().cpu())

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        return {
            "actor_loss": actor_loss_value,
            "critic1_loss": float(critic1_loss.detach().cpu()),
            "critic2_loss": float(critic2_loss.detach().cpu()),
            "mean_q": float(torch.minimum(q1, q2).detach().mean().cpu()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.data.copy_(
                self.tau * source_parameter.data + (1.0 - self.tau) * target_parameter.data
            )

    def attack_critic(self) -> MinTwinCritic:
        return MinTwinCritic(self.critic1, self.critic2)


class SACAgent:
    algorithm = "sac"

    def __init__(
        self,
        device: torch.device,
        *,
        obs_dim: int = 11,
        action_dim: int = 1,
        hidden_dim: int = 256,
        gamma: float = 0.9,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        initial_alpha: float = 0.2,
        target_entropy: float | None = None,
    ) -> None:
        self.device = device
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.actor = GaussianActor(self.obs_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic1 = Critic(self.obs_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic2 = Critic(self.obs_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1).to(device)
        self.critic2_target = copy.deepcopy(self.critic2).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=float(critic_lr))
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=float(critic_lr))
        self.log_alpha = torch.tensor(
            math.log(float(initial_alpha)), dtype=torch.float32, device=device, requires_grad=True
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=float(alpha_lr))
        self.target_entropy = (
            -float(self.action_dim) if target_entropy is None else float(target_entropy)
        )
        self.gamma = float(gamma)
        self.tau = float(tau)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act_tensor(self, obs: torch.Tensor, *, explore: bool) -> torch.Tensor:
        obs = obs.to(self.device)
        if explore:
            action, _ = self.actor.sample_action(obs)
            return action
        return self.actor(obs)

    @torch.no_grad()
    def act(self, obs, *, explore: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(to_numpy_1d(obs), dtype=torch.float32, device=self.device)
        return self.act_tensor(obs_t, explore=explore).reshape(-1).cpu().numpy().astype(np.float32)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample_action(batch["next_observations"])
            next_q = torch.minimum(
                self.critic1_target(batch["next_observations"], next_action),
                self.critic2_target(batch["next_observations"], next_action),
            ).reshape(-1)
            target = batch["rewards"] + (1.0 - batch["dones"]) * self.gamma * (
                next_q - self.alpha.detach() * next_log_prob
            )

        q1 = self.critic1(batch["observations"], batch["actions"]).reshape(-1)
        q2 = self.critic2(batch["observations"], batch["actions"]).reshape(-1)
        critic1_loss = F.mse_loss(q1, target)
        critic2_loss = F.mse_loss(q2, target)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        sampled_action, log_prob = self.actor.sample_action(batch["observations"])
        sampled_q = torch.minimum(
            self.critic1(batch["observations"], sampled_action),
            self.critic2(batch["observations"], sampled_action),
        ).reshape(-1)
        actor_loss = (self.alpha.detach() * log_prob - sampled_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic1_loss": float(critic1_loss.detach().cpu()),
            "critic2_loss": float(critic2_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "mean_q": float(torch.minimum(q1, q2).detach().mean().cpu()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            target_parameter.data.copy_(
                self.tau * source_parameter.data + (1.0 - self.tau) * target_parameter.data
            )

    def attack_critic(self) -> MinTwinCritic:
        return MinTwinCritic(self.critic1, self.critic2)


def create_agent(algorithm: str, device: torch.device, **kwargs) -> TD3Agent | SACAgent:
    name = str(algorithm).strip().lower()
    if name == "td3":
        return TD3Agent(device, **kwargs)
    if name == "sac":
        return SACAgent(device, **kwargs)
    raise ValueError(f"Unsupported off-policy backbone: {algorithm!r}")


def initialize_from_ddpg_bundle(
    agent: TD3Agent | SACAgent,
    path: str | Path,
) -> None:
    payload = torch.load(Path(path), map_location=agent.device, weights_only=False)
    actor_state = payload.get("actor_state_dict")
    critic_state = payload.get("critic_state_dict")
    if actor_state is None or critic_state is None:
        raise ValueError(f"DDPG bundle lacks actor or critic state: {path}")
    if isinstance(agent, TD3Agent):
        agent.actor.load_state_dict(actor_state)
        agent.actor_target.load_state_dict(actor_state)
    else:
        compatible_actor_state = {
            key: value
            for key, value in actor_state.items()
            if key in agent.actor.state_dict() and agent.actor.state_dict()[key].shape == value.shape
        }
        missing, unexpected = agent.actor.load_state_dict(compatible_actor_state, strict=False)
        allowed_missing = {"fc_log_std.weight", "fc_log_std.bias"}
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(
                f"Unexpected SAC actor initialization mismatch: missing={missing}, unexpected={unexpected}"
            )
        nn.init.constant_(agent.actor.fc_log_std.weight, 0.0)
        nn.init.constant_(agent.actor.fc_log_std.bias, -2.0)
    agent.critic1.load_state_dict(critic_state)
    agent.critic2.load_state_dict(critic_state)
    agent.critic1_target.load_state_dict(critic_state)
    agent.critic2_target.load_state_dict(critic_state)


def save_backbone_bundle(
    agent: TD3Agent | SACAgent,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_type": "offpolicy_backbone_bundle",
        "schema_version": 1,
        "algorithm": agent.algorithm,
        "actor_config": {
            "obs_dim": agent.obs_dim,
            "action_dim": agent.action_dim,
            "hidden_dim": agent.hidden_dim,
        },
        "actor_state_dict": agent.actor.state_dict(),
        "critic1_state_dict": agent.critic1.state_dict(),
        "critic2_state_dict": agent.critic2.state_dict(),
        "metadata": dict(metadata or {}),
    }
    if isinstance(agent, SACAgent):
        payload["log_alpha"] = float(agent.log_alpha.detach().cpu())
        payload["target_entropy"] = float(agent.target_entropy)
    torch.save(payload, path)
    return path


def load_backbone_bundle(
    path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, MinTwinCritic, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("model_type") != "offpolicy_backbone_bundle":
        raise ValueError(f"Not an off-policy backbone bundle: {path}")
    algorithm = str(payload["algorithm"]).lower()
    config = dict(payload.get("actor_config") or {})
    obs_dim = int(config.get("obs_dim", 11))
    action_dim = int(config.get("action_dim", 1))
    hidden_dim = int(config.get("hidden_dim", 256))
    if algorithm == "td3":
        actor: nn.Module = Actor(obs_dim, action_dim, hidden_dim)
    elif algorithm == "sac":
        actor = GaussianActor(obs_dim, action_dim, hidden_dim)
    else:
        raise ValueError(f"Unsupported algorithm in bundle: {algorithm!r}")
    critic1 = Critic(obs_dim, action_dim, hidden_dim)
    critic2 = Critic(obs_dim, action_dim, hidden_dim)
    actor.load_state_dict(payload["actor_state_dict"])
    critic1.load_state_dict(payload["critic1_state_dict"])
    critic2.load_state_dict(payload["critic2_state_dict"])
    actor = actor.to(device).eval()
    critic1 = critic1.to(device).eval()
    critic2 = critic2.to(device).eval()
    attack_critic = MinTwinCritic(critic1, critic2).to(device).eval()
    for module in (actor, attack_critic):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return actor, attack_critic, payload


def load_evaluation_backbone(
    algorithm: str,
    path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    name = str(algorithm).strip().lower()
    if name == "ppo":
        from .ppo_backbone import load_ppo_bundle

        return load_ppo_bundle(path, device)
    if name in {"td3", "sac"}:
        actor, critic, payload = load_backbone_bundle(path, device)
        payload_algorithm = str(payload.get("algorithm", "")).lower()
        if payload_algorithm != name:
            raise ValueError(
                f"Requested {name!r}, but bundle contains {payload_algorithm!r}: {path}"
            )
        return actor, critic, payload
    if name != "ddpg":
        raise ValueError(f"Unsupported evaluation backbone: {algorithm!r}")

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    actor_state = payload.get("actor_state_dict")
    critic_state = payload.get("critic_state_dict")
    if actor_state is None or critic_state is None:
        raise ValueError(f"DDPG bundle lacks actor or critic state: {path}")
    actor = Actor().to(device)
    critic = Critic().to(device)
    actor.load_state_dict(actor_state)
    critic.load_state_dict(critic_state)
    actor.eval()
    critic.eval()
    for module in (actor, critic):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    normalized_payload = dict(payload)
    normalized_payload["algorithm"] = "ddpg"
    return actor, critic, normalized_payload


__all__ = [
    "GaussianActor",
    "MinTwinCritic",
    "TD3Agent",
    "SACAgent",
    "create_agent",
    "initialize_from_ddpg_bundle",
    "save_backbone_bundle",
    "load_backbone_bundle",
    "load_evaluation_backbone",
]
