from __future__ import annotations

import numpy as np
import torch

from evc.merged_core import Actor
from evc.sa_ddpg import SADDPG1DCrownAgent, StateObservationAdversary


def test_sa_crown_short_focus_uses_damage_weighted_attack_candidates() -> None:
    torch.manual_seed(23)
    np.random.seed(23)
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
    agent = SADDPG1DCrownAgent(
        actor,
        device,
        obs_low=np.zeros(11, dtype=np.float32),
        obs_high=np.ones(11, dtype=np.float32),
        epsilon=0.1,
        state_scope='all',
        actor_reg_weight=0.05,
        directional_adversary=adversary,
        directional_attack_families=('opposite_pgd', 'q_function', 'sgld_maxdiff'),
        directional_reg_weight=0.25,
        directional_top_fraction=0.5,
    )
    agent.set_bound_epsilon(0.1)
    observations = 0.1 + 0.8 * torch.rand(8, 11)
    batch = {
        'observations': observations,
        'next_observations': 0.1 + 0.8 * torch.rand(8, 11),
        'actions': actor(observations).detach(),
        'rewards': -torch.rand(8),
        'dones': torch.zeros(8),
        'is_new_arrivals': torch.zeros(8),
    }

    before_rs = {
        key: value.detach().clone() for key, value in agent.robust_sarsa_critic.state_dict().items()
    }
    stats = agent.update(batch, attack_family=('opposite_pgd', 'q_function', 'sgld_maxdiff'))

    assert all(np.isfinite(float(value)) for value in stats.values())
    assert stats['sa_directional_candidate_count'] == 3.0
    assert stats['sa_directional_selected_fraction'] == 0.5
    assert stats['sa_directional_damage_mean'] >= 0.0
    assert stats['sa_directional_reg_loss'] >= 0.0
    assert stats['sa_paper_faithful_update'] == 0.0
    assert stats['sa_bound_epsilon'] == 0.1
    assert stats['robust_sarsa_loss'] >= 0.0
    assert any(
        not torch.equal(before_rs[key], value)
        for key, value in agent.robust_sarsa_critic.state_dict().items()
    )
