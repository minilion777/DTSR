from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from evc.experimental_long_horizon_v2 import (
    EXPERIMENTAL_DEADLINE_PGD_V2,
    EXPERIMENTAL_SMALL_DRIFT_Q_V2,
    ExperimentalMomentumSmallDriftQAttacker,
    MomentumSmallDriftConfig,
    StealthDeadlineConfig,
    build_experimental_long_horizon_attacker,
)
from evc.formal_experimental_long_horizon import build_formal_experimental_long_horizon_attacker
from evc.merged_attacks import AttackContext, LOCAL_ATTACK_IDX


class DummyActor(nn.Module):
    def forward(self, x):
        return torch.tanh(1.4 * x[:, :1] - 0.8 * x[:, 1:2] - 0.2)


class DummyCritic(nn.Module):
    def forward(self, obs, action):
        return action - 0.01 * obs[:, :1]


class AlternatingBaseAttacker:
    epsilon = 0.2
    seed = 11
    obs_low = torch.zeros(11)
    obs_high = torch.ones(11)

    def __init__(self) -> None:
        self.calls = 0

    def clone(self):
        return AlternatingBaseAttacker()

    def reset(self) -> None:
        self.calls = 0

    def attack(self, obs_batch):
        obs = np.asarray(obs_batch, dtype=np.float32)
        delta = np.zeros_like(obs, dtype=np.float32)
        patterns = (
            np.asarray((0.10, -0.06, 0.04), dtype=np.float32),
            np.asarray((-0.03, 0.10, -0.08), dtype=np.float32),
        )
        pattern = patterns[self.calls % len(patterns)]
        for offset, idx in enumerate(LOCAL_ATTACK_IDX):
            delta[:, int(idx)] = pattern[offset % pattern.size]
        self.calls += 1
        return obs + delta


def context() -> AttackContext:
    return AttackContext(
        scenario="O",
        time_index=0,
        station=0,
        is_new_arrival=False,
        raw_price=300.0,
        price_threshold=400.0,
        soc_new_threshold=0.5,
        soc_rollout_threshold=0.3,
        even_station_target=1.0,
        odd_station_target=-0.5,
    )


class ExperimentalLongHorizonV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.actor = DummyActor()
        self.critic = DummyCritic()
        self.low = np.zeros(11, dtype=np.float32)
        self.high = np.ones(11, dtype=np.float32)

    def test_deadline_phase_starts_at_zero_for_different_durations(self) -> None:
        attacker = build_experimental_long_horizon_attacker(
            EXPERIMENTAL_DEADLINE_PGD_V2,
            actor=self.actor,
            device=torch.device("cpu"),
            obs_low=self.low,
            obs_high=self.high,
            seed=7,
        )
        for vehicle_id, duration in enumerate((3, 6, 12)):
            obs = np.zeros(11, dtype=np.float32)
            obs[1] = duration / 12.0
            weight, phase, remaining, _ = attacker._schedule_weight((0, vehicle_id), obs)
            self.assertEqual(remaining, duration)
            self.assertAlmostEqual(phase, 0.0)
            self.assertAlmostEqual(weight, 0.0)

    def test_deadline_clean_prefix_returns_zero_delta(self) -> None:
        attacker = build_experimental_long_horizon_attacker(
            EXPERIMENTAL_DEADLINE_PGD_V2,
            actor=self.actor,
            device=torch.device("cpu"),
            obs_low=self.low,
            obs_high=self.high,
            seed=8,
        )
        obs = np.zeros((1, 11), dtype=np.float32)
        obs[0, 0] = 0.4
        obs[0, 1] = 0.5
        adv = attacker.attack_with_metadata(
            obs,
            contexts=[context()],
            vehicle_ids=[5],
            episode_indices=[0],
        )
        self.assertTrue(np.allclose(adv, obs, atol=1e-8))

    def test_small_drift_respects_epsilon_and_slew(self) -> None:
        cfg = MomentumSmallDriftConfig(epsilon=0.04, step_size=0.006, slew_limit=0.007, momentum=0.9)
        attacker = build_experimental_long_horizon_attacker(
            EXPERIMENTAL_SMALL_DRIFT_Q_V2,
            actor=self.actor,
            critic=self.critic,
            device=torch.device("cpu"),
            obs_low=self.low,
            obs_high=self.high,
            seed=9,
            config=cfg,
        )
        obs = np.full((1, 11), 0.5, dtype=np.float32)
        prev_delta = np.zeros(11, dtype=np.float32)
        for _ in range(8):
            adv = attacker.attack_with_metadata(
                obs,
                contexts=[context()],
                vehicle_ids=[3],
                episode_indices=[0],
            )
            delta = adv[0] - obs[0]
            self.assertLessEqual(float(np.max(np.abs(delta))), cfg.epsilon + 1e-7)
            self.assertLessEqual(float(np.max(np.abs(delta - prev_delta))), cfg.slew_limit + 1e-7)
            self.assertTrue(np.allclose(delta[[2, 3, 4, 5, 6, 7, 8, 9]], 0.0))
            prev_delta = delta

    def test_small_drift_keeps_current_q_direction_in_the_loop(self) -> None:
        cfg = MomentumSmallDriftConfig(
            epsilon=0.12,
            step_size=0.024,
            slew_limit=0.020,
            momentum=0.86,
            current_direction_weight=0.35,
        )
        attacker = ExperimentalMomentumSmallDriftQAttacker(AlternatingBaseAttacker(), config=cfg)
        obs = np.full((1, 11), 0.5, dtype=np.float32)
        deltas: list[np.ndarray] = []
        for _ in range(8):
            adv = attacker.attack_with_metadata(
                obs,
                contexts=[context()],
                vehicle_ids=[4],
                episode_indices=[0],
            )
            deltas.append(adv[0] - obs[0])

        cosines: list[float] = []
        for prev_delta, delta in zip(deltas, deltas[1:]):
            local_prev = prev_delta[list(LOCAL_ATTACK_IDX)]
            local_cur = delta[list(LOCAL_ATTACK_IDX)]
            denom = float(np.linalg.norm(local_prev) * np.linalg.norm(local_cur))
            if denom > 1e-10:
                cosines.append(float(np.dot(local_prev, local_cur) / denom))
        self.assertLess(float(np.mean(cosines)), 0.95)

    def test_formal_aliases_route_to_experimental_v2_implementations(self) -> None:
        small = build_formal_experimental_long_horizon_attacker(
            "local_small_drift_q",
            actor=self.actor,
            critic=self.critic,
            device=torch.device("cpu"),
            obs_low=self.low,
            obs_high=self.high,
            seed=12,
        )
        deadline = build_formal_experimental_long_horizon_attacker(
            "local_deadline_drift_pgd",
            actor=self.actor,
            critic=self.critic,
            device=torch.device("cpu"),
            obs_low=self.low,
            obs_high=self.high,
            seed=13,
        )
        self.assertEqual(small.name, EXPERIMENTAL_SMALL_DRIFT_Q_V2)
        self.assertEqual(deadline.name, EXPERIMENTAL_DEADLINE_PGD_V2)
        self.assertEqual(getattr(small, "formal_alias"), "local_small_drift_q")
        self.assertEqual(getattr(deadline, "formal_alias"), "local_deadline_drift_pgd")


if __name__ == "__main__":
    unittest.main()
