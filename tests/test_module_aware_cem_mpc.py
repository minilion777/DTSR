from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from evc.long_horizon_attacks import build_long_horizon_attacker, canonical_long_horizon_attack_name
from evc.module_aware_attacks import CEMMPCConfig


class DummyActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(11, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.tanh(self.linear(x))


class ModuleAwareCEMMPCTest(unittest.TestCase):
    def test_config_aliases(self) -> None:
        cfg = CEMMPCConfig.from_overrides({"objective": "economic_shift", "knowledge": "k4", "horizon": 4, "samples": 8})
        self.assertEqual(cfg.objective, "economic")
        self.assertEqual(cfg.knowledge_level, "K4")
        self.assertEqual(cfg.horizon, 4)
        self.assertEqual(cfg.samples, 8)

    def test_canonical_name(self) -> None:
        self.assertEqual(canonical_long_horizon_attack_name("adaptive_knowledge_ladder"), "module_aware_cem_mpc")
        self.assertEqual(canonical_long_horizon_attack_name("cem-mpc"), "module_aware_cem_mpc")

    def test_project_sequence_respects_constraints(self) -> None:
        actor = DummyActor()
        low = np.full(11, -1.0, dtype=np.float32)
        high = np.full(11, 2.0, dtype=np.float32)
        attacker = build_long_horizon_attacker(
            "module_aware_cem_mpc",
            actor=actor,
            device=torch.device("cpu"),
            obs_low=low,
            obs_high=high,
            seed=7,
            attack_state_scope="local",
            attack_overrides={
                "objective": "deadline",
                "knowledge_level": "K2",
                "horizon": 5,
                "samples": 6,
                "iterations": 1,
                "epsilon": 0.075,
                "temporal_eta": 0.02,
                "total_l1_budget": 0.19,
            },
        )
        key = (0, 3)
        proposal = np.ones((5, 11), dtype=np.float32)
        seq = attacker._project_sequence(key, proposal)
        local = seq[:, [0, 1, 10]]
        self.assertLessEqual(float(np.max(np.abs(seq))), 0.075 + 1e-7)
        self.assertLessEqual(float(np.max(np.abs(local[0]))), 0.02 + 1e-7)
        self.assertLessEqual(float(np.max(np.abs(np.diff(local, axis=0)))), 0.02 + 1e-7)
        self.assertLessEqual(float(np.sum(np.abs(local))), 0.19 + 1e-7)
        self.assertTrue(np.allclose(seq[:, [2, 3, 4, 5, 6, 7, 8, 9]], 0.0))


if __name__ == "__main__":
    unittest.main()
