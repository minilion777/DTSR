from __future__ import annotations

import numpy as np
import torch
from torch import nn

from evc.native_backbone_attacks import (
    MultiRestartPointwiseAttacker,
    TwinCriticAggregate,
)


class LinearActor(nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(obs[:, :1] - 0.5 * obs[:, 1:2])


class OffsetCritic(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = float(offset)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return obs[:, :1] + action + self.offset


class StateValueCritic(nn.Module):
    uses_state_value = True

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        del action
        return obs[:, :1] + 0.5 * obs[:, 1:2]


def test_twin_critic_aggregate_modes() -> None:
    obs = torch.tensor([[0.2], [0.8]])
    action = torch.tensor([[0.1], [-0.3]])
    critic1 = OffsetCritic(1.0)
    critic2 = OffsetCritic(-2.0)
    q1 = critic1(obs, action)
    q2 = critic2(obs, action)

    assert torch.allclose(TwinCriticAggregate(critic1, critic2, "min")(obs, action), q2)
    assert torch.allclose(TwinCriticAggregate(critic1, critic2, "mean")(obs, action), 0.5 * (q1 + q2))
    assert torch.allclose(TwinCriticAggregate(critic1, critic2, "q1")(obs, action), q1)
    assert torch.allclose(TwinCriticAggregate(critic1, critic2, "q2")(obs, action), q2)


def test_multirestart_pointwise_attack_respects_shape_and_budget() -> None:
    epsilon = 0.10
    attacker = MultiRestartPointwiseAttacker(
        LinearActor(),
        OffsetCritic(0.0),
        algorithm="opposite_pgd",
        device=torch.device("cpu"),
        epsilon=epsilon,
        alpha=0.02,
        iters=5,
        restarts=3,
        seed=42,
        obs_low=np.zeros(11, dtype=np.float32),
        obs_high=np.ones(11, dtype=np.float32),
    )
    clean = np.full((4, 11), 0.5, dtype=np.float32)
    adversarial = attacker.attack(clean)

    assert adversarial.shape == clean.shape
    assert adversarial.dtype == np.float32
    assert np.isfinite(adversarial).all()
    assert float(np.max(np.abs(adversarial - clean))) <= epsilon + 1e-6
    assert bool(np.all(adversarial >= 0.0))
    assert bool(np.all(adversarial <= 1.0))


def test_q_attack_supports_ppo_state_value_adapter_gradient() -> None:
    epsilon = 0.10
    attacker = MultiRestartPointwiseAttacker(
        LinearActor(),
        StateValueCritic(),
        algorithm="q_function",
        device=torch.device("cpu"),
        epsilon=epsilon,
        alpha=0.02,
        iters=5,
        restarts=1,
        seed=7,
        obs_low=np.zeros(11, dtype=np.float32),
        obs_high=np.ones(11, dtype=np.float32),
    )
    clean = np.full((4, 11), 0.5, dtype=np.float32)
    adversarial = attacker.attack(clean)

    assert adversarial.shape == clean.shape
    assert np.isfinite(adversarial).all()
    assert float(np.max(np.abs(adversarial - clean))) <= epsilon + 1e-6
    assert float(np.max(np.abs(adversarial - clean))) > 0.0
