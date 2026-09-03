from __future__ import annotations

import numpy as np
import torch

from evc.merged_core import Actor
from evc.sa_ddpg import StateObservationAdversary
from evc.robust_bounds import actor_split_crown_action_bounds
from evc.wocar import (
    WocaR1DIntervalAgent,
    actor_crown_action_bounds,
    actor_ibp_action_bounds,
    interval_q_extrema_1d,
    projected_q_extrema_1d,
)


def test_actor_ibp_interval_contains_sampled_scalar_actions() -> None:
    torch.manual_seed(7)
    actor = Actor(obs_dim=11, action_dim=1, hidden_dim=32)
    clean = 0.2 + 0.6 * torch.rand(5, 11)
    obs_low = torch.zeros(11)
    obs_high = torch.ones(11)

    clean_actions = actor(clean)
    for bound_fn in (actor_ibp_action_bounds, actor_crown_action_bounds):
        point_lower, point_upper = bound_fn(
            actor,
            clean,
            epsilon=0.0,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_indices=tuple(range(11)),
        )
        torch.testing.assert_close(point_lower, clean_actions, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(point_upper, clean_actions, atol=1e-6, rtol=1e-6)

        lower, upper = bound_fn(
            actor,
            clean,
            epsilon=0.1,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_indices=tuple(range(11)),
        )
        for _ in range(64):
            perturbed = torch.clamp(clean + torch.empty_like(clean).uniform_(-0.1, 0.1), 0.0, 1.0)
            sampled_actions = actor(perturbed)
            assert torch.all(sampled_actions >= lower - 1e-6)
            assert torch.all(sampled_actions <= upper + 1e-6)


def test_split_crown_remains_certified_and_no_looser_than_unsplit() -> None:
    torch.manual_seed(9)
    actor = Actor(obs_dim=11, action_dim=1, hidden_dim=32)
    clean = 0.2 + 0.6 * torch.rand(4, 11)
    kwargs = {
        'epsilon': 0.1,
        'obs_low': torch.zeros(11),
        'obs_high': torch.ones(11),
        'attack_indices': tuple(range(11)),
    }
    plain_lower, plain_upper = actor_crown_action_bounds(actor, clean, **kwargs)
    split_lower, split_upper = actor_split_crown_action_bounds(
        actor, clean, split_dimensions=2, **kwargs
    )
    assert torch.all(split_lower >= plain_lower - 1e-6)
    assert torch.all(split_upper <= plain_upper + 1e-6)
    for _ in range(64):
        perturbed = torch.clamp(clean + torch.empty_like(clean).uniform_(-0.1, 0.1), 0.0, 1.0)
        sampled_actions = actor(perturbed)
        assert torch.all(sampled_actions >= split_lower - 1e-6)
        assert torch.all(sampled_actions <= split_upper + 1e-6)


def test_projected_action_search_never_weakens_grid_extrema() -> None:
    torch.manual_seed(10)
    from evc.merged_core import Critic

    critic = Critic(obs_dim=11, action_dim=1, hidden_dim=32)
    observations = torch.rand(5, 11)
    lower = -torch.ones(5, 1)
    upper = torch.ones(5, 1)
    grid_min, grid_max, _, _ = interval_q_extrema_1d(
        critic, observations, lower, upper, grid_size=9
    )
    projected_min, projected_max, _, _ = projected_q_extrema_1d(
        critic, observations, lower, upper, grid_size=9, steps=5
    )
    assert torch.all(projected_min <= grid_min + 1e-6)
    assert torch.all(projected_max >= grid_max - 1e-6)


def test_wocar_interval_update_uses_independent_worst_critic() -> None:
    torch.manual_seed(11)
    np.random.seed(11)
    device = torch.device('cpu')
    actor = Actor(obs_dim=11, action_dim=1, hidden_dim=256)
    agent = WocaR1DIntervalAgent(
        actor,
        device,
        obs_low=np.zeros(11, dtype=np.float32),
        obs_high=np.ones(11, dtype=np.float32),
        epsilon=0.1,
        state_scope='all',
        worst_action_grid_size=7,
        state_importance_grid_size=7,
    )
    agent.set_bound_epsilon(0.1)
    assert agent.worst_critic is not None
    assert agent.worst_critic is not agent.critic

    observations = torch.rand(8, 11)
    next_observations = torch.rand(8, 11)
    with torch.no_grad():
        actions = actor(observations)
    batch = {
        'observations': observations,
        'next_observations': next_observations,
        'actions': actions,
        'rewards': -torch.rand(8),
        'dones': torch.zeros(8),
    }
    before = {key: value.detach().clone() for key, value in agent.worst_critic.state_dict().items()}
    stats = agent.update(batch)

    assert all(np.isfinite(float(value)) for value in stats.values() if isinstance(value, (float, int)))
    assert stats['separate_worst_critic'] == 1.0
    assert stats['worst_target_le_clean_frac'] == 1.0
    assert stats['wocar_bound_epsilon'] == 0.1
    assert stats['reachable_action_width_mean'] >= 0.0
    assert any(not torch.equal(before[key], value) for key, value in agent.worst_critic.state_dict().items())


def test_wocar_interval_attack_guidance_is_auxiliary_and_scheduled() -> None:
    torch.manual_seed(19)
    np.random.seed(19)
    device = torch.device('cpu')
    actor = Actor(obs_dim=11, action_dim=1, hidden_dim=256)
    adversary = StateObservationAdversary(
        device=device,
        epsilon=0.1,
        steps=1,
        obs_low=np.zeros(11, dtype=np.float32),
        obs_high=np.ones(11, dtype=np.float32),
        attack_state_scope='all',
    )
    agent = WocaR1DIntervalAgent(
        actor,
        device,
        obs_low=np.zeros(11, dtype=np.float32),
        obs_high=np.ones(11, dtype=np.float32),
        epsilon=0.1,
        state_scope='all',
        worst_action_grid_size=5,
        state_importance_grid_size=5,
        candidate_adversary=adversary,
        candidate_worst_weight=0.15,
        candidate_q_weight=0.1,
        candidate_reg_weight=0.03,
        candidate_interval_margin=0.05,
    )
    agent.set_bound_epsilon(0.05)
    agent.set_policy_robustness(
        worst_policy_weight=0.25,
        state_reg_weight=0.05,
        target_lambda=0.4,
    )
    observations = 0.1 + 0.8 * torch.rand(6, 11)
    next_observations = 0.1 + 0.8 * torch.rand(6, 11)
    batch = {
        'observations': observations,
        'next_observations': next_observations,
        'actions': actor(observations).detach(),
        'rewards': -torch.rand(6),
        'dones': torch.zeros(6),
    }

    stats = agent.update(batch, attack_families=('opposite_pgd', 'q_function'))

    assert stats['separate_worst_critic'] == 1.0
    assert stats['candidate_attack_count'] == 2.0
    assert stats['candidate_worst_weight'] == 0.075
    assert stats['candidate_q_weight'] == 0.05
    assert stats['candidate_reg_weight'] == 0.015
    assert stats['target_lambda'] == 0.4
    assert stats['actor_q_term_active'] == 1.0
    assert 0.0 < stats['update_adv_linf'] <= 0.05 + 1e-6
    assert stats['target_candidate_count'] == 7.0
    assert stats['candidate_reg_action_mse_mean'] > 0.0
    assert stats['candidate_target_tightening_mean'] >= 0.0
    assert stats['preserve_reachable_superset'] == 1.0
    assert stats['candidate_interval_tightening_mean'] == 0.0
    assert stats['candidate_target_tightening_mean'] == 0.0
