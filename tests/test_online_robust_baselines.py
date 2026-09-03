from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from evc.merged_core import Actor, Critic, TRAIN_PROFILE
from evc.online_atla_ppo_lstm_sa import (
    AtlaPpoLstmSaAgent,
    _attack_recurrent_observation,
    _perturb_from_raw,
    distill_atla_lstm_policy,
)
from evc.robust_bounds import observation_bounds_across_scenarios, scalar_action_stability_loss
from evc.sa_ddpg import SADDPG1DCrownAgent


def _random_batch(actor: Actor, batch_size: int = 8) -> dict[str, torch.Tensor]:
    observations = torch.rand(batch_size, 11)
    next_observations = torch.rand(batch_size, 11)
    with torch.no_grad():
        actions = actor(observations)
    return {
        'observations': observations,
        'next_observations': next_observations,
        'actions': actions,
        'rewards': -torch.rand(batch_size),
        'dones': torch.zeros(batch_size),
    }


def test_sa_ddpg_uses_clean_bellman_update_and_crown_regularizer() -> None:
    torch.manual_seed(17)
    actor = Actor(obs_dim=11, action_dim=1, hidden_dim=256)
    agent = SADDPG1DCrownAgent(
        actor,
        torch.device('cpu'),
        obs_low=np.zeros(11, dtype=np.float32),
        obs_high=np.ones(11, dtype=np.float32),
        epsilon=0.1,
        state_scope='all',
        actor_reg_weight=0.3,
    )
    agent.set_bound_epsilon(0.02)
    stats = agent.update(_random_batch(actor))

    assert stats['sa_paper_faithful_update'] == 1.0
    assert stats['sa_bound_epsilon'] == 0.02
    assert stats['update_adv_frac'] == 0.0
    assert stats['target_adv_frac'] == 0.0
    assert stats['sa_reachable_action_width_mean'] > 0.0
    assert stats['actor_reg_loss'] >= 0.0
    assert all(np.isfinite(float(value)) for value in stats.values())


def test_scalar_stability_loss_is_linear_maximum_deviation() -> None:
    clean = torch.tensor([[0.2], [0.5]])
    lower = torch.tensor([[-0.1], [0.4]])
    upper = torch.tensor([[0.4], [0.9]])
    loss, per_sample = scalar_action_stability_loss(clean, lower, upper)

    torch.testing.assert_close(per_sample, torch.tensor([0.3, 0.4]))
    torch.testing.assert_close(loss, torch.tensor(0.35))


def test_atla_raw_delta_uses_smooth_bounded_parameterization() -> None:
    clean = torch.full((1, 3), 0.5)
    raw = torch.tensor([[0.0, 1.0, 100.0]])
    perturbed = _perturb_from_raw(
        clean,
        raw,
        epsilon=0.1,
        mask=torch.ones(3),
        obs_low=torch.zeros(1, 3),
        obs_high=torch.ones(1, 3),
    )

    assert torch.all(perturbed - clean <= 0.1 + 1e-7)
    assert torch.all(perturbed - clean >= -0.1 - 1e-7)
    torch.testing.assert_close(
        perturbed[0, 1],
        torch.tensor(0.5 + 0.1 * np.tanh(1.0), dtype=torch.float32),
    )


def test_atla_lstm_distillation_preserves_vehicle_sequences() -> None:
    torch.manual_seed(31)
    rng = np.random.default_rng(31)
    device = torch.device('cpu')
    teacher = Actor().to(device).eval()
    agent = AtlaPpoLstmSaAgent(device=device)
    clean = rng.random((12, 11), dtype=np.float32)
    episodes = np.array([0] * 6 + [1] * 6, dtype=np.int64)
    vehicles = np.array([0, 0, 0, 1, 1, 1] * 2, dtype=np.int64)
    times = np.array([0, 1, 2, 0, 1, 2] * 2, dtype=np.int64)
    before = {name: value.detach().clone() for name, value in agent.policy.state_dict().items()}

    stats = distill_atla_lstm_policy(
        agent,
        teacher,
        clean,
        episodes,
        vehicles,
        times,
        epochs=1,
        batch_size=2,
        max_sessions=3,
    )

    assert stats['distill_sessions'] == 3.0
    assert np.isfinite(stats['distill_loss'])
    assert any(not torch.equal(before[name], value) for name, value in agent.policy.state_dict().items())


def test_recurrent_external_attack_respects_linf_budget() -> None:
    torch.manual_seed(37)
    device = torch.device('cpu')
    agent = AtlaPpoLstmSaAgent(device=device)
    clean = np.full(11, 0.5, dtype=np.float32)
    mask = torch.ones(11)
    generator = torch.Generator(device=device)
    generator.manual_seed(37)

    adversarial = _attack_recurrent_observation(
        agent,
        clean,
        None,
        algorithm='opposite_pgd',
        critic=Critic(),
        epsilon=0.1,
        alpha=0.02,
        iters=2,
        mask=mask,
        obs_low_t=torch.zeros(1, 11),
        obs_high_t=torch.ones(1, 11),
        generator=generator,
    )

    assert float(np.max(np.abs(adversarial - clean))) <= 0.100001


def test_global_observation_bounds_cover_every_training_scenario(monkeypatch) -> None:
    bounds = {
        'first.csv': (np.array([0.0, -1.0]), np.array([1.0, 2.0])),
        'second.csv': (np.array([-2.0, 0.0]), np.array([3.0, 1.0])),
    }

    class FakeEnv:
        def __init__(self, *, signals_path, reward_profile):
            self.signals_path = signals_path

        def observation_bounds(self, *, max_duration_of_stay):
            return bounds[self.signals_path]

    monkeypatch.setattr('evc.robust_bounds.ChargingEnv', FakeEnv)
    scenarios = [SimpleNamespace(signals_path='first.csv'), SimpleNamespace(signals_path='second.csv')]
    low, high = observation_bounds_across_scenarios(
        scenarios,
        reward_profile=TRAIN_PROFILE,
        max_duration_of_stay=24.0,
    )

    np.testing.assert_allclose(low, [-2.0, -1.0])
    np.testing.assert_allclose(high, [3.0, 2.0])
