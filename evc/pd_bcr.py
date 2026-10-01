from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .merged_core import ChargingEnv, min_max_denormalization, normalize_scalar, to_numpy_1d
from .offline_dae_det_temporal_shield import LOCAL_SHIELD_INDICES
from .dtsr_runtime import rollout_episode_with_dtsr


_TIME_DECAY = 1.0 / 12.0


@dataclass(frozen=True)
class PDBCRConfig:
    """Paper-aligned constraint-gated persistent-divergence belief correction."""

    schema_version: int = 1
    leakage_policy: str = "strict_no_clean_state"
    core_weights: tuple[float, float, float] = (1.0, 1.0, 0.1)
    divergence_threshold: float = 0.008
    persistence_threshold: float = 0.0
    shield_residual_max: float = 0.06
    persistence_window: int = 3
    fusion_weight: float = 0.7
    cosine_epsilon: float = 1e-8
    force_off_on_new_arrival: bool = True
    clip_core_to_unit_interval: bool = True

    def __post_init__(self) -> None:
        if int(self.schema_version) != 1:
            raise ValueError("PD-BCR requires schema_version=1.")
        if str(self.leakage_policy) != "strict_no_clean_state":
            raise ValueError("PD-BCR requires leakage_policy='strict_no_clean_state'.")
        if len(self.core_weights) != len(LOCAL_SHIELD_INDICES):
            raise ValueError("core_weights must match the three guarded variables.")
        if any(not np.isfinite(value) or float(value) <= 0.0 for value in self.core_weights):
            raise ValueError("core_weights must be finite and positive.")
        if not np.isfinite(self.divergence_threshold) or float(self.divergence_threshold) < 0.0:
            raise ValueError("divergence_threshold must be finite and non-negative.")
        if not -1.0 <= float(self.persistence_threshold) <= 1.0:
            raise ValueError("persistence_threshold must be in [-1, 1].")
        if not np.isfinite(self.shield_residual_max) or float(self.shield_residual_max) < 0.0:
            raise ValueError("shield_residual_max must be finite and non-negative.")
        if int(self.persistence_window) < 2:
            raise ValueError("persistence_window must be at least 2.")
        if not 0.0 < float(self.fusion_weight) < 1.0:
            raise ValueError("fusion_weight must be in (0, 1).")
        if not np.isfinite(self.cosine_epsilon) or float(self.cosine_epsilon) <= 0.0:
            raise ValueError("cosine_epsilon must be finite and positive.")


def load_pd_bcr_config(path: str | Path) -> PDBCRConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "core_weights" in payload:
        payload["core_weights"] = tuple(payload["core_weights"])
    return PDBCRConfig(**payload)


def pd_bcr_config_payload(config: PDBCRConfig) -> dict[str, Any]:
    return asdict(config)


class PersistentDivergenceBeliefCorrector:
    """PD-BCR applied strictly after Temporal Shield and before the actor.

    The belief is initialized from the routed observation at arrival and then
    propagated only with past executed actions and known charging dynamics.
    The correction is a fixed-weight fusion activated by the conjunction of
    divergence magnitude, direction persistence, and small Shield residual.
    """

    processor_name = "pd_bcr"
    runtime_pipeline_order = "DAE/DET route -> Temporal Shield -> PD-BCR -> Actor"
    is_pd_bcr = True

    def __init__(self, config: PDBCRConfig | None = None) -> None:
        self.config = config or PDBCRConfig()
        self.belief_core_by_vehicle: dict[int, np.ndarray] = {}
        self.prev_action_by_vehicle: dict[int, np.ndarray] = {}
        self.prev_time_by_vehicle: dict[int, int] = {}
        self.divergence_history_by_vehicle: dict[int, list[np.ndarray]] = {}
        self.total = 0
        self.activation_count = 0
        self.divergence_sum = 0.0
        self.persistence_sum = 0.0
        self.persistence_count = 0
        self.shield_residual_sum = 0.0
        self.fusion_shift_sum = 0.0
        self.magnitude_pass_count = 0
        self.persistence_pass_count = 0
        self.residual_pass_count = 0
        self.last_gate_scores: list[float] = []

    def reset(self) -> None:
        self.__init__(self.config)

    def _propagate_core(self, vehicle_id: int, env: ChargingEnv) -> np.ndarray:
        previous = self.belief_core_by_vehicle[int(vehicle_id)]
        action = to_numpy_1d(
            self.prev_action_by_vehicle.get(int(vehicle_id), np.asarray([0.0], dtype=np.float32))
        )
        action_scalar = float(action[0]) if action.size else 0.0
        soc_step = float(env.max_power * env.slice_hours / env.battery_capacity)
        soc = float(previous[0] + action_scalar * soc_step)
        remaining_time = float(max(previous[1] - _TIME_DECAY, 0.0))
        price_index = int(
            np.clip(self.prev_time_by_vehicle.get(int(vehicle_id), int(env.t) - 1), 0, env.horizon - 1)
        )
        previous_cost = float(
            min_max_denormalization(float(previous[2]), 0.0, env._cost_upper_bound())
        )
        step_cost = float(
            action_scalar * env.max_power * env.slice_hours * float(env.signals.price[price_index])
        )
        cost = float(
            np.clip(
                normalize_scalar(previous_cost + step_cost, 0.0, env._cost_upper_bound()),
                0.0,
                1.0,
            )
        )
        propagated = np.asarray([soc, remaining_time, cost], dtype=np.float32)
        if self.config.clip_core_to_unit_interval:
            propagated = np.clip(propagated, 0.0, 1.0).astype(np.float32)
        return propagated

    def _persistence_score(self, history: Sequence[np.ndarray]) -> float:
        if len(history) < int(self.config.persistence_window):
            return float("nan")
        recent = list(history)[-int(self.config.persistence_window) :]
        similarities: list[float] = []
        epsilon = float(self.config.cosine_epsilon)
        for previous, current in zip(recent[:-1], recent[1:]):
            previous_norm = float(np.linalg.norm(previous, ord=2))
            current_norm = float(np.linalg.norm(current, ord=2))
            similarities.append(
                float(np.dot(current, previous) / (previous_norm * current_norm + epsilon))
            )
        return float(np.mean(similarities))

    def process_batch(
        self,
        *,
        routed_states: Sequence[np.ndarray],
        shielded_states: Sequence[np.ndarray],
        vehicle_ids: Sequence[int],
        is_new_arrivals: Sequence[int],
        env: ChargingEnv,
        detector_scores: Sequence[float] | None = None,
        route_flags: Sequence[bool] | None = None,
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        del detector_scores, route_flags
        if not (
            len(routed_states)
            == len(shielded_states)
            == len(vehicle_ids)
            == len(is_new_arrivals)
        ):
            raise ValueError("PD-BCR batch inputs must have equal lengths.")

        weights = np.asarray(self.config.core_weights, dtype=np.float32)
        selected_states: list[np.ndarray] = []
        belief_states: list[np.ndarray] = []
        branches: list[str] = []
        gate_scores: list[float] = []
        divergence_values: list[float] = []
        persistence_values: list[float] = []
        residual_values: list[float] = []

        for routed_state, shielded_state, vehicle_id, new_flag in zip(
            routed_states, shielded_states, vehicle_ids, is_new_arrivals
        ):
            vid = int(vehicle_id)
            routed = to_numpy_1d(routed_state).astype(np.float32)
            shielded = to_numpy_1d(shielded_state).astype(np.float32)
            routed_core = routed[list(LOCAL_SHIELD_INDICES)].astype(np.float32)
            shielded_core = shielded[list(LOCAL_SHIELD_INDICES)].astype(np.float32)

            if bool(new_flag) or vid not in self.belief_core_by_vehicle:
                # Paper Eq. (17): initialize from the routed core observation.
                belief_core = routed_core.copy()
                self.divergence_history_by_vehicle[vid] = []
            else:
                belief_core = self._propagate_core(vid, env)

            divergence_vector = weights * (shielded_core - belief_core)
            divergence = float(np.linalg.norm(divergence_vector, ord=2))
            shield_residual = float(np.linalg.norm(weights * (routed_core - shielded_core), ord=2))

            history = self.divergence_history_by_vehicle.setdefault(vid, [])
            history.append(divergence_vector.astype(np.float32))
            if len(history) > int(self.config.persistence_window):
                del history[:-int(self.config.persistence_window)]
            persistence = self._persistence_score(history)

            magnitude_pass = divergence >= float(self.config.divergence_threshold)
            persistence_pass = bool(
                np.isfinite(persistence)
                and persistence >= float(self.config.persistence_threshold)
            )
            residual_pass = shield_residual <= float(self.config.shield_residual_max)
            gate = magnitude_pass and persistence_pass and residual_pass
            if bool(self.config.force_off_on_new_arrival) and bool(new_flag):
                gate = False

            selected = shielded.copy()
            if gate:
                fusion_weight = float(self.config.fusion_weight)
                corrected_core = (
                    (1.0 - fusion_weight) * shielded_core
                    + fusion_weight * belief_core
                )
                if self.config.clip_core_to_unit_interval:
                    corrected_core = np.clip(corrected_core, 0.0, 1.0)
                selected[list(LOCAL_SHIELD_INDICES)] = corrected_core.astype(np.float32)
                branch = "belief_fusion"
                self.activation_count += 1
            else:
                # Conjunctive constraint: any failed condition returns Shield unchanged.
                branch = "shield"

            belief_state = shielded.copy()
            belief_state[list(LOCAL_SHIELD_INDICES)] = belief_core.astype(np.float32)
            self.belief_core_by_vehicle[vid] = belief_core.astype(np.float32).copy()
            fusion_shift = float(
                np.linalg.norm(
                    weights
                    * (
                        selected[list(LOCAL_SHIELD_INDICES)]
                        - shielded[list(LOCAL_SHIELD_INDICES)]
                    ),
                    ord=2,
                )
            )

            selected_states.append(selected.astype(np.float32))
            belief_states.append(belief_state.astype(np.float32))
            branches.append(branch)
            gate_scores.append(float(gate))
            divergence_values.append(divergence)
            persistence_values.append(persistence)
            residual_values.append(shield_residual)
            self.total += 1
            self.divergence_sum += divergence
            self.shield_residual_sum += shield_residual
            self.fusion_shift_sum += fusion_shift
            self.magnitude_pass_count += int(magnitude_pass)
            self.persistence_pass_count += int(persistence_pass)
            self.residual_pass_count += int(residual_pass)
            if np.isfinite(persistence):
                self.persistence_sum += float(persistence)
                self.persistence_count += 1

        self.last_gate_scores = gate_scores
        return selected_states, {
            "belief_states": belief_states,
            "branches": branches,
            "gate_scores": gate_scores,
            "divergence": divergence_values,
            "persistence": persistence_values,
            "shield_residual": residual_values,
        }

    def update_actions(self, vehicle_ids: Sequence[int], actions: np.ndarray, current_time: int) -> None:
        for vehicle_id, action in zip(vehicle_ids, np.asarray(actions, dtype=np.float32)):
            vid = int(vehicle_id)
            self.prev_action_by_vehicle[vid] = to_numpy_1d(action).astype(np.float32)
            self.prev_time_by_vehicle[vid] = int(current_time)

    def summary(self) -> dict[str, float | int | str]:
        total = int(self.total)
        return {
            "pd_bcr_total": total,
            "pd_bcr_activation_count": int(self.activation_count),
            "pd_bcr_activation_rate": 0.0 if total == 0 else float(self.activation_count / total),
            "pd_bcr_divergence_mean": 0.0 if total == 0 else float(self.divergence_sum / total),
            "pd_bcr_persistence_mean": (
                0.0 if self.persistence_count == 0 else float(self.persistence_sum / self.persistence_count)
            ),
            "pd_bcr_shield_residual_mean": (
                0.0 if total == 0 else float(self.shield_residual_sum / total)
            ),
            "pd_bcr_fusion_shift_mean": 0.0 if total == 0 else float(self.fusion_shift_sum / total),
            "pd_bcr_magnitude_pass_rate": 0.0 if total == 0 else float(self.magnitude_pass_count / total),
            "pd_bcr_persistence_pass_rate": 0.0 if total == 0 else float(self.persistence_pass_count / total),
            "pd_bcr_residual_pass_rate": 0.0 if total == 0 else float(self.residual_pass_count / total),
            "pd_bcr_divergence_threshold": float(self.config.divergence_threshold),
            "pd_bcr_persistence_threshold": float(self.config.persistence_threshold),
            "pd_bcr_shield_residual_max": float(self.config.shield_residual_max),
            "pd_bcr_persistence_window": int(self.config.persistence_window),
            "pd_bcr_fusion_weight": float(self.config.fusion_weight),
            "pd_bcr_gate_mode": "conjunctive",
        }


def rollout_episode_with_pd_bcr(*args, pd_bcr_config: PDBCRConfig, **kwargs) -> dict:
    processor = PersistentDivergenceBeliefCorrector(pd_bcr_config)
    kwargs["enable_shield"] = True
    kwargs["post_shield_processor"] = processor
    kwargs.setdefault("label", "attack_dae_det_shield_pd_bcr")
    return rollout_episode_with_dtsr(*args, **kwargs)


__all__ = [
    "PDBCRConfig",
    "PersistentDivergenceBeliefCorrector",
    "load_pd_bcr_config",
    "pd_bcr_config_payload",
    "rollout_episode_with_pd_bcr",
]
