from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .formal_experimental_long_horizon import (
    build_formal_experimental_long_horizon_attacker,
)
from .merged_attacks import PGDStateAttacker
from .offpolicy_backbones import MinTwinCritic


class TwinCriticAggregate(nn.Module):
    def __init__(self, critic1: nn.Module, critic2: nn.Module, mode: str) -> None:
        super().__init__()
        self.critic1 = critic1
        self.critic2 = critic2
        self.mode = str(mode).lower()
        if self.mode not in {"min", "mean", "q1", "q2"}:
            raise ValueError(f"Unknown twin critic aggregation: {mode!r}")

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        if self.mode == "min":
            return torch.minimum(q1, q2)
        if self.mode == "mean":
            return 0.5 * (q1 + q2)
        return q1 if self.mode == "q1" else q2


def critic_for_mode(critic: nn.Module, mode: str) -> nn.Module:
    token = str(mode or "single").lower()
    if isinstance(critic, MinTwinCritic):
        if token == "single":
            token = "min"
        return TwinCriticAggregate(critic.critic1, critic.critic2, token)
    if token not in {"single", "min", "mean", "q1", "q2"}:
        raise ValueError(f"Unknown critic mode: {mode!r}")
    return critic


class MultiRestartPointwiseAttacker:
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        *,
        algorithm: str,
        device: torch.device,
        epsilon: float,
        alpha: float,
        iters: int,
        restarts: int,
        seed: int,
        obs_low: np.ndarray,
        obs_high: np.ndarray,
        q_mode: str = "single",
    ) -> None:
        self.actor = actor
        self.critic = critic_for_mode(critic, q_mode)
        self.algorithm = str(algorithm)
        self.device = device
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.iters = int(iters)
        self.restarts = int(restarts)
        self.seed = int(seed)
        if self.algorithm not in {"opposite_pgd", "q_function"}:
            raise ValueError(f"Unsupported pointwise native attack: {algorithm!r}")
        if self.iters <= 0 or self.restarts <= 0 or self.alpha <= 0.0:
            raise ValueError("Native pointwise attack parameters must be positive.")
        self.attackers = [
            PGDStateAttacker(
                actor,
                device=device,
                algorithm=self.algorithm,
                epsilon=self.epsilon,
                alpha=self.alpha,
                iters=self.iters,
                seed=self.seed + restart * 1_000_003,
                obs_low=obs_low,
                obs_high=obs_high,
                critic=self.critic if self.algorithm == "q_function" else None,
                attack_state_scope="all",
            )
            for restart in range(self.restarts)
        ]

    def reset(self) -> None:
        for attacker in self.attackers:
            attacker.reset()

    def _score(self, clean: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            clean_action = self.actor(clean).reshape(clean.shape[0], -1)
            candidate_action = self.actor(candidate).reshape(candidate.shape[0], -1)
            if self.algorithm == "opposite_pgd":
                return (candidate_action - clean_action).square().sum(dim=1)
            critic_obs = candidate if bool(getattr(self.critic, "uses_state_value", False)) else clean
            q_value = self.critic(critic_obs, candidate_action).reshape(-1)
            return -q_value

    def attack(
        self,
        obs_batch: np.ndarray,
        target_actions: np.ndarray | None = None,
    ) -> np.ndarray:
        del target_actions
        clean_np = np.asarray(obs_batch, dtype=np.float32)
        if clean_np.ndim == 1:
            clean_np = clean_np.reshape(1, -1)
        clean = torch.as_tensor(clean_np, dtype=torch.float32, device=self.device)
        best = None
        best_score = None
        for attacker in self.attackers:
            candidate_np = attacker.attack(clean_np)
            candidate = torch.as_tensor(
                candidate_np, dtype=torch.float32, device=self.device
            )
            score = self._score(clean, candidate)
            if best is None:
                best = candidate.clone()
                best_score = score.clone()
                continue
            choose = score > best_score
            best[choose] = candidate[choose]
            best_score[choose] = score[choose]
        return best.detach().cpu().numpy().astype(np.float32)


def build_native_attacker(
    attack_key: str,
    profile: dict[str, Any],
    *,
    actor: nn.Module,
    critic: nn.Module,
    device: torch.device,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    seed: int,
):
    key = str(attack_key)
    kind = str(profile.get("kind", "pointwise"))
    if kind == "pointwise":
        return MultiRestartPointwiseAttacker(
            actor,
            critic,
            algorithm=key,
            device=device,
            epsilon=float(profile["epsilon"]),
            alpha=float(profile["alpha"]),
            iters=int(profile["iters"]),
            restarts=int(profile["restarts"]),
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            q_mode=str(profile.get("q_mode", "single")),
        )
    if kind == "long_horizon":
        return build_formal_experimental_long_horizon_attacker(
            key,
            actor=actor,
            critic=critic,
            device=device,
            obs_low=obs_low,
            obs_high=obs_high,
            seed=seed,
            attack_state_scope="local",
            attack_overrides=dict(profile["attack_overrides"]),
        )
    raise ValueError(f"Unknown native attack profile kind: {kind!r}")


__all__ = [
    "TwinCriticAggregate",
    "MultiRestartPointwiseAttacker",
    "critic_for_mode",
    "build_native_attacker",
]
