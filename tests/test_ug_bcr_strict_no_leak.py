from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from evc.merged_core import ChargingEnv, TRAIN_PROFILE
from evc.ug_bcr import (
    BeliefCoreConfig,
    BeliefCoreEstimator,
    UrgencyGateConfig,
    UrgencyGatedBeliefSelector,
)
SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dtsr_multiday_common import load_ug_bcr_config


TIME_DECAY = 1.0 / 12.0
SIGNALS_PATH = Path(__file__).parents[1] / "data" / "signals_reference.json"


def _state(*, soc: float = 0.0, remaining: float = 0.75, cost: float = 0.2) -> np.ndarray:
    state = np.zeros(11, dtype=np.float32)
    state[0] = soc
    state[1] = remaining
    state[10] = cost
    return state


def _strict_estimator() -> BeliefCoreEstimator:
    return BeliefCoreEstimator(
        BeliefCoreConfig(
            pred_weight=0.75,
            max_pred_weight=0.92,
            use_known_initial_soc=True,
            use_known_initial_cost=True,
            time_initialization="routed_observation",
        )
    )


def _env() -> ChargingEnv:
    return ChargingEnv(signals_path=SIGNALS_PATH)


def _drift_gate() -> UrgencyGatedBeliefSelector:
    return UrgencyGatedBeliefSelector(
        UrgencyGateConfig(
            enabled=True,
            urgency_gain_threshold=0.0,
            soc_drop_threshold=1.0,
            time_drop_threshold=0.001,
            uncertainty_threshold=1.0,
            temporal_residual_threshold=1.0,
            innovation_ema_decay=0.80,
            time_innovation_threshold=0.006,
            soc_innovation_threshold=1.0,
            consecutive_steps=2,
            min_remaining_steps=0.0,
            max_remaining_steps=20.0,
        )
    )


def _select(
    gate: UrgencyGatedBeliefSelector,
    estimator: BeliefCoreEstimator,
    env: ChargingEnv,
    policy: np.ndarray,
    belief: np.ndarray,
    *,
    is_new: int,
) -> tuple[np.ndarray, str]:
    selected, branches = gate.select_batch(
        [policy],
        [belief],
        [7],
        [is_new],
        estimator,
        None,
        env,
        TRAIN_PROFILE,
        {},
        {},
        {},
    )
    return selected[0], branches[0]


def test_repair_batch_interface_cannot_accept_clean_state() -> None:
    parameters = inspect.signature(BeliefCoreEstimator.repair_batch).parameters
    assert list(parameters) == [
        "self",
        "policy_states",
        "vehicle_ids",
        "is_new_arrivals",
        "detector_scores",
        "env",
    ]


def test_runtime_function_has_no_clean_state_name() -> None:
    source_path = Path(__file__).parents[1] / "evc" / "ug_bcr.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    runtime = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run_defense_runtime"
    )
    argument_names = {arg.arg for arg in runtime.args.args}
    loaded_names = {
        node.id
        for node in ast.walk(runtime)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert "clean_states" not in argument_names
    assert "clean_states" not in loaded_names


def test_new_arrival_time_initializes_from_routed_observation() -> None:
    env = _env()
    estimator = _strict_estimator()
    routed = _state(remaining=0.75)
    repaired = estimator.repair_batch([routed], [7], [1], None, env)[0]
    assert repaired[1] == pytest.approx(0.75)
    assert estimator.prev_belief_state_by_vehicle[7][1] == pytest.approx(0.75)


def test_normal_time_propagation_has_zero_innovation_and_no_activation() -> None:
    env = _env()
    estimator = _strict_estimator()
    gate = _drift_gate()
    first = _state(remaining=0.75)
    belief = estimator.repair_batch([first], [7], [1], None, env)[0]
    _, branch = _select(gate, estimator, env, first, belief, is_new=1)
    assert branch == "shield"
    estimator.update_actions([7], np.asarray([[0.0]], dtype=np.float32), 0)

    normal = _state(remaining=0.75 - TIME_DECAY)
    belief = estimator.repair_batch([normal], [7], [0], None, env)[0]
    selected, branch = _select(gate, estimator, env, normal, belief, is_new=0)
    assert estimator.innovation(7)[1] == pytest.approx(0.0, abs=1e-6)
    assert branch == "shield"
    np.testing.assert_allclose(selected, normal, atol=1e-6)


def test_persistent_positive_time_innovation_activates_after_history() -> None:
    env = _env()
    estimator = _strict_estimator()
    gate = _drift_gate()
    attacked = _state(remaining=0.75)

    belief = estimator.repair_batch([attacked], [7], [1], None, env)[0]
    _, branch = _select(gate, estimator, env, attacked, belief, is_new=1)
    assert branch == "shield"
    estimator.update_actions([7], np.asarray([[0.0]], dtype=np.float32), 0)

    belief = estimator.repair_batch([attacked], [7], [0], None, env)[0]
    assert estimator.innovation(7)[1] == pytest.approx(TIME_DECAY, abs=1e-6)
    _, branch = _select(gate, estimator, env, attacked, belief, is_new=0)
    assert branch == "shield"
    estimator.update_actions([7], np.asarray([[0.0]], dtype=np.float32), 1)

    belief = estimator.repair_batch([attacked], [7], [0], None, env)[0]
    selected, branch = _select(gate, estimator, env, attacked, belief, is_new=0)
    assert estimator.innovation(7)[1] > TIME_DECAY
    assert branch == "belief"
    assert selected[1] < attacked[1]


def test_identical_runtime_inputs_produce_identical_actual_and_shadow_outputs() -> None:
    env_actual = _env()
    env_shadow = _env()
    actual_estimator, shadow_estimator = _strict_estimator(), _strict_estimator()
    actual_gate, shadow_gate = _drift_gate(), _drift_gate()
    observations = [_state(remaining=0.75), _state(remaining=0.75), _state(remaining=0.75)]
    actual_outputs: list[np.ndarray] = []
    shadow_outputs: list[np.ndarray] = []

    for step, observation in enumerate(observations):
        is_new = int(step == 0)
        for estimator, gate, env, outputs in (
            (actual_estimator, actual_gate, env_actual, actual_outputs),
            (shadow_estimator, shadow_gate, env_shadow, shadow_outputs),
        ):
            belief = estimator.repair_batch([observation], [7], [is_new], None, env)[0]
            selected, _ = _select(gate, estimator, env, observation, belief, is_new=is_new)
            outputs.append(selected)
            estimator.update_actions([7], np.asarray([[0.0]], dtype=np.float32), step)

    np.testing.assert_allclose(actual_outputs, shadow_outputs, atol=1e-6)


def test_legacy_config_is_rejected(tmp_path: Path) -> None:
    legacy = tmp_path / "ug_bcr_config.json"
    legacy.write_text(
        json.dumps({"belief": {"initial_" + "time_from_system": True}, "urgency_gate": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Legacy UG-BCR"):
        load_ug_bcr_config(legacy)
