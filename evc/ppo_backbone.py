from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .offpolicy_backbones import GaussianActor


def physics_safety_action(
    obs: torch.Tensor,
    *,
    target_soc: float = 0.93,
    running_floor: float = 0.16,
    step_soc_delta: float = 0.07 * 0.25 / 0.04992,
) -> torch.Tensor:
    """Differentiable feasibility prior derived from the charging dynamics."""
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    soc = obs[:, 0]
    remaining_slots = (obs[:, 1] * 12.0).clamp_min(1.0)
    terminal_need = (float(target_soc) - soc) / (remaining_slots * float(step_soc_delta))
    running_need = (float(running_floor) - soc) / float(step_soc_delta)
    return torch.maximum(terminal_need, running_need).clamp(-1.0, 1.0).unsqueeze(-1)


class StateValueNetwork(nn.Module):
    """PPO state-value network V(s)."""

    def __init__(self, obs_dim: int = 11, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = nn.Linear(self.obs_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        hidden = F.relu(self.fc1(obs.float()))
        hidden = F.relu(self.fc2(hidden))
        return self.out(hidden)


class ValueAsQAdapter(nn.Module):
    """Expose PPO V(s) through the critic(obs, action) DTSR contract.

    The action argument is intentionally ignored.  Gradients with respect to
    observations remain available to value-oriented attacks and DTSR routing.
    """

    def __init__(self, value_network: StateValueNetwork) -> None:
        super().__init__()
        self.value_network = value_network
        self.uses_state_value = True

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del action
        return self.value_network(obs)


class PPOAgent:
    algorithm = "ppo"

    def __init__(
        self,
        device: torch.device,
        *,
        obs_dim: int = 11,
        action_dim: int = 1,
        hidden_dim: int = 256,
        gamma: float = 0.9,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        actor_lr: float = 3e-4,
        value_lr: float = 1e-3,
        entropy_coef: float = 0.002,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 512,
        target_kl: float = 0.03,
        prior_behavior_coef: float = 0.05,
        initial_action_mean: float = 0.45,
        initial_log_std: float = -1.5,
    ) -> None:
        self.device = device
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.actor = GaussianActor(self.obs_dim, self.action_dim, self.hidden_dim).to(device)
        self.value = StateValueNetwork(self.obs_dim, self.hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=float(value_lr))
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)
        self.target_kl = float(target_kl)
        self.prior_behavior_coef = float(prior_behavior_coef)

        # A neutral charging bias avoids an unsafe zero-action cold start while
        # leaving all state-dependent weights independently initialized.
        bounded_mean = float(np.clip(initial_action_mean, -0.95, 0.95))
        with torch.no_grad():
            self.actor.fc_mu.weight.mul_(0.01)
            self.actor.fc_mu.bias.fill_(math.atanh(bounded_mean))
            self.actor.fc_log_std.weight.zero_()
            self.actor.fc_log_std.bias.fill_(float(initial_log_std))

    @torch.no_grad()
    def act_tensor(self, obs: torch.Tensor, *, explore: bool) -> torch.Tensor:
        obs = obs.to(self.device)
        if explore:
            action, _ = self.actor.sample_action(obs)
            return action.clamp(-1.0, 1.0)
        return self.actor(obs).clamp(-1.0, 1.0)

    def update(self, rollout: dict[str, np.ndarray | torch.Tensor]) -> dict[str, float]:
        tensors: dict[str, torch.Tensor] = {}
        for key, value in rollout.items():
            tensors[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        observations = tensors["observations"]
        next_observations = tensors["next_observations"]
        actions = tensors["actions"]
        rewards = tensors["rewards"].reshape(-1)
        dones = tensors["dones"].reshape(-1)

        with torch.no_grad():
            old_log_prob, _ = self.actor.log_prob_entropy(observations, actions)
            old_values = self.value(observations).reshape(-1)
            next_values = self.value(next_observations).reshape(-1)
            deltas = rewards + (1.0 - dones) * self.gamma * next_values - old_values
            if "trajectory_ids" in rollout:
                trajectory_ids = np.asarray(rollout["trajectory_ids"]).reshape(-1)
                delta_np = deltas.detach().cpu().numpy()
                done_np = dones.detach().cpu().numpy()
                advantage_np = np.zeros_like(delta_np, dtype=np.float32)
                for trajectory_id in np.unique(trajectory_ids):
                    trajectory_indices = np.flatnonzero(trajectory_ids == trajectory_id)
                    accumulator = 0.0
                    for index in trajectory_indices[::-1]:
                        accumulator = float(delta_np[index]) + (
                            self.gamma
                            * self.gae_lambda
                            * (1.0 - float(done_np[index]))
                            * accumulator
                        )
                        advantage_np[index] = accumulator
                raw_advantages = torch.as_tensor(advantage_np, dtype=torch.float32, device=self.device)
            else:
                raw_advantages = deltas
            returns = raw_advantages + old_values
            advantages = raw_advantages
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)

        sample_count = int(observations.shape[0])
        indices = np.arange(sample_count)
        actor_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        clip_fractions: list[float] = []
        kls: list[float] = []
        epochs_completed = 0
        stop_early = False
        for epoch in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, sample_count, self.minibatch_size):
                batch_indices = torch.as_tensor(
                    indices[start : start + self.minibatch_size],
                    dtype=torch.long,
                    device=self.device,
                )
                new_log_prob, entropy = self.actor.log_prob_entropy(
                    observations[batch_indices], actions[batch_indices]
                )
                log_ratio = new_log_prob - old_log_prob[batch_indices]
                ratio = log_ratio.exp()
                unclipped = ratio * advantages[batch_indices]
                clipped = ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages[batch_indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                prior_action = physics_safety_action(observations[batch_indices])
                prior_loss = F.mse_loss(self.actor(observations[batch_indices]), prior_action)
                actor_loss = (
                    policy_loss
                    - self.entropy_coef * entropy.mean()
                    + self.prior_behavior_coef * prior_loss
                )
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_prediction = self.value(observations[batch_indices]).reshape(-1)
                value_loss = self.value_coef * F.mse_loss(value_prediction, returns[batch_indices])
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                approximate_kl = float(((ratio - 1.0) - log_ratio).mean().detach().cpu())
                actor_losses.append(float(actor_loss.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy.mean().detach().cpu()))
                clip_fractions.append(float((torch.abs(ratio - 1.0) > self.clip_ratio).float().mean().detach().cpu()))
                kls.append(approximate_kl)
                if approximate_kl > self.target_kl:
                    stop_early = True
                    break
            epochs_completed = epoch + 1
            if stop_early:
                break

        with torch.no_grad():
            final_values = self.value(observations).reshape(-1)
            return_variance = torch.var(returns, unbiased=False)
            explained_variance = 1.0 - torch.var(returns - final_values, unbiased=False) / return_variance.clamp_min(1e-8)
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "explained_variance": float(explained_variance.cpu()),
            "update_epochs_completed": float(epochs_completed),
            "rollout_transitions": float(sample_count),
        }

    def attack_critic(self) -> ValueAsQAdapter:
        return ValueAsQAdapter(self.value)


def save_ppo_bundle(
    agent: PPOAgent,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "ppo_backbone_bundle",
            "schema_version": 1,
            "algorithm": "ppo",
            "actor_config": {
                "obs_dim": agent.obs_dim,
                "action_dim": agent.action_dim,
                "hidden_dim": agent.hidden_dim,
            },
            "actor_state_dict": agent.actor.state_dict(),
            "value_state_dict": agent.value.state_dict(),
            "critic_adapter": "state_value_as_q; action_argument_ignored",
            "metadata": dict(metadata or {}),
        },
        path,
    )
    return path


def load_ppo_bundle(
    path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, ValueAsQAdapter, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("model_type") != "ppo_backbone_bundle" or str(payload.get("algorithm", "")).lower() != "ppo":
        raise ValueError(f"Not a PPO backbone bundle: {path}")
    config = dict(payload.get("actor_config") or {})
    obs_dim = int(config.get("obs_dim", 11))
    action_dim = int(config.get("action_dim", 1))
    hidden_dim = int(config.get("hidden_dim", 256))
    actor = GaussianActor(obs_dim, action_dim, hidden_dim).to(device)
    value = StateValueNetwork(obs_dim, hidden_dim).to(device)
    actor.load_state_dict(payload["actor_state_dict"])
    value.load_state_dict(payload["value_state_dict"])
    actor.eval()
    value.eval()
    critic = ValueAsQAdapter(value).to(device).eval()
    for module in (actor, critic):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return actor, critic, payload


__all__ = [
    "StateValueNetwork",
    "ValueAsQAdapter",
    "PPOAgent",
    "save_ppo_bundle",
    "load_ppo_bundle",
    "physics_safety_action",
]
