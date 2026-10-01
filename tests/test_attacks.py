from __future__ import annotations

import unittest

import numpy as np
import torch

from evc.attacks import (
    ALL_PAPER_ATTACK_NAMES,
    AttackContext,
    SHORT_HORIZON_ATTACK_NAMES,
    attack_batch_by_context,
    build_short_horizon_attacker,
)


class _Actor(torch.nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(obs[:, :1] - 0.5)


class _Critic(torch.nn.Module):
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return action + 0.1 * obs[:, :1]


class AttackSuiteTests(unittest.TestCase):
    def test_paper_attack_inventory(self):
        self.assertEqual(len(SHORT_HORIZON_ATTACK_NAMES), 6)
        self.assertEqual(len(ALL_PAPER_ATTACK_NAMES), 8)

    def test_short_horizon_factories(self):
        actor = _Actor()
        critic = _Critic()
        low = np.zeros(11, dtype=np.float32)
        high = np.ones(11, dtype=np.float32)
        expected = {
            "opposite_pgd": "opposite_pgd",
            "opposite_fgsm": "opposite_fgsm",
            "q_function": "q_function",
            "electhacker_C": "electhacker",
            "electhacker_F": "electhacker",
            "electhacker_O": "electhacker",
        }
        for name, algorithm in expected.items():
            attacker = build_short_horizon_attacker(
                name,
                actor=actor,
                critic=critic,
                device=torch.device("cpu"),
                obs_low=low,
                obs_high=high,
            )
            self.assertEqual(attacker.algorithm, algorithm)

    def test_electhacker_uses_scenario_target(self):
        attacker = build_short_horizon_attacker(
            "electhacker_C",
            actor=_Actor(),
            device=torch.device("cpu"),
            obs_low=np.zeros(11, dtype=np.float32),
            obs_high=np.ones(11, dtype=np.float32),
            seed=42,
        )
        obs = np.full(11, 0.5, dtype=np.float32)
        context = AttackContext("C", 0, 100.0, 0, False)
        attacked, flags = attack_batch_by_context(attacker, [obs], [context])
        self.assertTrue(flags[0])
        self.assertGreater(float(np.max(np.abs(attacked[0] - obs))), 0.0)


if __name__ == "__main__":
    unittest.main()
