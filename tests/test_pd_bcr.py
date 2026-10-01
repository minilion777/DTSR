from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from evc.offline_dae_det_temporal_shield import LOCAL_SHIELD_INDICES
from evc.pd_bcr import PDBCRConfig, PersistentDivergenceBeliefCorrector


def _state(soc: float, remaining_time: float, cost: float = 0.0) -> np.ndarray:
    state = np.zeros((11,), dtype=np.float32)
    state[list(LOCAL_SHIELD_INDICES)] = (soc, remaining_time, cost)
    return state


def _fake_env():
    env = SimpleNamespace(
        t=0,
        horizon=12,
        max_power=60.0,
        slice_hours=0.25,
        battery_capacity=60.0,
        signals=SimpleNamespace(price=np.ones((12,), dtype=np.float32)),
    )
    env._cost_upper_bound = lambda: 100.0
    return env


class PDBCRConstraintGateTests(unittest.TestCase):
    def test_arrival_belief_is_initialized_from_routed_observation(self) -> None:
        processor = PersistentDivergenceBeliefCorrector(PDBCRConfig())
        env = _fake_env()
        routed = _state(0.37, 0.58, 0.13)
        shielded = _state(0.40, 0.55, 0.12)

        selected, diagnostics = processor.process_batch(
            routed_states=[routed],
            shielded_states=[shielded],
            vehicle_ids=[7],
            is_new_arrivals=[1],
            env=env,
        )

        np.testing.assert_allclose(
            diagnostics["belief_states"][0][list(LOCAL_SHIELD_INDICES)],
            routed[list(LOCAL_SHIELD_INDICES)],
        )
        np.testing.assert_allclose(selected[0], shielded)
        self.assertEqual(diagnostics["branches"], ["shield"])

    def test_constraint_gate_uses_fixed_fusion_only_after_persistent_divergence(self) -> None:
        config = PDBCRConfig(
            divergence_threshold=0.01,
            persistence_threshold=0.0,
            shield_residual_max=0.01,
            persistence_window=3,
            fusion_weight=0.7,
        )
        processor = PersistentDivergenceBeliefCorrector(config)
        env = _fake_env()
        shielded_steps = [
            _state(0.20, 0.50),
            _state(0.30, 0.50),
            _state(0.40, 0.50),
        ]
        outputs = []
        diagnostics = []
        for step, shielded in enumerate(shielded_steps):
            env.t = step
            selected, info = processor.process_batch(
                routed_states=[shielded.copy()],
                shielded_states=[shielded],
                vehicle_ids=[1],
                is_new_arrivals=[int(step == 0)],
                env=env,
            )
            processor.update_actions([1], np.asarray([[0.0]], dtype=np.float32), step)
            outputs.append(selected[0])
            diagnostics.append(info)

        np.testing.assert_allclose(outputs[0], shielded_steps[0])
        np.testing.assert_allclose(outputs[1], shielded_steps[1])
        self.assertEqual(diagnostics[2]["branches"], ["belief_fusion"])
        belief_core = diagnostics[2]["belief_states"][0][list(LOCAL_SHIELD_INDICES)]
        expected_core = 0.3 * shielded_steps[2][list(LOCAL_SHIELD_INDICES)] + 0.7 * belief_core
        np.testing.assert_allclose(
            outputs[2][list(LOCAL_SHIELD_INDICES)], expected_core, atol=1e-6
        )

    def test_failed_residual_condition_returns_shield_exactly(self) -> None:
        config = PDBCRConfig(
            divergence_threshold=0.0,
            persistence_threshold=-1.0,
            shield_residual_max=0.001,
            persistence_window=2,
            fusion_weight=0.7,
        )
        processor = PersistentDivergenceBeliefCorrector(config)
        env = _fake_env()
        first = _state(0.20, 0.50)
        processor.process_batch(
            routed_states=[first],
            shielded_states=[first],
            vehicle_ids=[2],
            is_new_arrivals=[1],
            env=env,
        )
        processor.update_actions([2], np.asarray([[0.0]], dtype=np.float32), 0)
        env.t = 1
        routed = _state(0.30, 0.50)
        shielded = _state(0.45, 0.50)

        selected, diagnostics = processor.process_batch(
            routed_states=[routed],
            shielded_states=[shielded],
            vehicle_ids=[2],
            is_new_arrivals=[0],
            env=env,
        )

        self.assertGreater(diagnostics["shield_residual"][0], config.shield_residual_max)
        self.assertEqual(diagnostics["branches"], ["shield"])
        np.testing.assert_array_equal(selected[0], shielded)


if __name__ == "__main__":
    unittest.main()
