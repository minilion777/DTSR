from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .ug_bcr import (
    _TIME_DECAY,
    BeliefCoreConfig,
    BeliefCoreEstimator,
    UGBCRConfig,
    UrgencyGateConfig,
    rollout_episode_with_ug_bcr,
)
from .merged_core import ChargingEnv, RewardProfile, TRAIN_PROFILE, to_numpy_1d
from .offline_dae_det_temporal_shield import LocalTemporalShieldConfig


V3_FEATURE_NAMES: tuple[str, ...] = (
    "deadline_pressure_gain",
    "deadline_pressure_gain_ema",
    "soc_drop",
    "time_drop",
    "cost_drop",
    "abs_soc_drop",
    "abs_time_drop",
    "abs_cost_drop",
    "positive_soc_innovation",
    "positive_time_innovation",
    "abs_cost_innovation",
    "soc_innovation_ema",
    "time_innovation_ema",
    "innovation_ema_norm",
    "belief_uncertainty",
    "policy_soc",
    "policy_time",
    "belief_soc",
    "belief_time",
    "core_disagreement_l1",
    "detector_score",
    "det_route_flag",
    "is_new_arrival",
    "time_index_fraction",
)


@dataclass(frozen=True)
class ContinuousGateConfig:
    """Serializable linear probability gate fitted only on validation scenes."""

    feature_names: tuple[str, ...] = V3_FEATURE_NAMES
    feature_means: tuple[float, ...] = field(default_factory=lambda: tuple(0.0 for _ in V3_FEATURE_NAMES))
    feature_scales: tuple[float, ...] = field(default_factory=lambda: tuple(1.0 for _ in V3_FEATURE_NAMES))
    coefficients: tuple[float, ...] = field(default_factory=lambda: tuple(0.0 for _ in V3_FEATURE_NAMES))
    intercept: float = 0.0
    decision_threshold: float = 0.5
    pressure_ema_decay: float = 0.72
    innovation_ema_decay: float = 0.70
    horizon_steps: int = 344
    force_policy_on_new_arrival: bool = True

    def __post_init__(self) -> None:
        size = len(self.feature_names)
        if tuple(self.feature_names) != V3_FEATURE_NAMES:
            raise ValueError("UG-BCR-v3 feature order does not match the runtime feature contract.")
        if not (len(self.feature_means) == len(self.feature_scales) == len(self.coefficients) == size):
            raise ValueError("UG-BCR-v3 scaler/model vectors must match feature_names.")
        if any((not np.isfinite(value) or float(value) <= 0.0) for value in self.feature_scales):
            raise ValueError("UG-BCR-v3 feature scales must be finite and positive.")
        if not 0.0 < float(self.decision_threshold) < 1.0:
            raise ValueError("UG-BCR-v3 decision_threshold must be in (0, 1).")


@dataclass(frozen=True)
class UGBCRV3Config:
    schema_version: int = 3
    leakage_policy: str = "strict_no_clean_state"
    uses_clean_state: bool = False
    uses_true_remaining_time: bool = False
    base_v2: UGBCRConfig = field(default_factory=UGBCRConfig)
    continuous_gate: ContinuousGateConfig = field(default_factory=ContinuousGateConfig)
    training_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.schema_version) != 3:
            raise ValueError("UG-BCR-v3 requires schema_version=3.")
        if str(self.leakage_policy) != "strict_no_clean_state":
            raise ValueError("UG-BCR-v3 requires strict_no_clean_state.")
        if bool(self.uses_clean_state) or bool(self.uses_true_remaining_time):
            raise ValueError("UG-BCR-v3 runtime cannot use clean state or true remaining time.")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _deadline_pressure(soc: float, normalized_time: float, target_soc: float) -> float:
    remaining_steps = max(float(normalized_time) / _TIME_DECAY, 1.0)
    return float(np.clip(max(0.0, float(target_soc) - float(soc)) / remaining_steps, 0.0, 1.0))


def build_v3_feature_vector(
    *,
    policy_soc: float,
    policy_time: float,
    policy_cost: float,
    belief_soc: float,
    belief_time: float,
    belief_cost: float,
    soc_innovation: float,
    time_innovation: float,
    cost_innovation: float,
    belief_uncertainty: float,
    detector_score: float,
    det_route_flag: float,
    is_new_arrival: float,
    time_index: int,
    target_soc: float,
    previous_pressure_gain_ema: float | None,
    previous_soc_innovation_ema: float,
    previous_time_innovation_ema: float,
    pressure_ema_decay: float,
    innovation_ema_decay: float,
    horizon_steps: int,
) -> tuple[np.ndarray, float, float, float]:
    soc_drop = float(policy_soc - belief_soc)
    time_drop = float(policy_time - belief_time)
    cost_drop = float(policy_cost - belief_cost)
    pressure_gain = _deadline_pressure(belief_soc, belief_time, target_soc) - _deadline_pressure(
        policy_soc, policy_time, target_soc
    )
    pressure_old = pressure_gain if previous_pressure_gain_ema is None else float(previous_pressure_gain_ema)
    pressure_decay = float(np.clip(pressure_ema_decay, 0.0, 1.0))
    pressure_ema = pressure_decay * pressure_old + (1.0 - pressure_decay) * pressure_gain
    positive_soc_innovation = max(float(soc_innovation), 0.0)
    positive_time_innovation = max(float(time_innovation), 0.0)
    innovation_decay = float(np.clip(innovation_ema_decay, 0.0, 1.0))
    soc_ema = innovation_decay * float(previous_soc_innovation_ema) + (1.0 - innovation_decay) * positive_soc_innovation
    time_ema = innovation_decay * float(previous_time_innovation_ema) + (1.0 - innovation_decay) * positive_time_innovation
    features = np.asarray(
        [
            pressure_gain,
            pressure_ema,
            soc_drop,
            time_drop,
            cost_drop,
            abs(soc_drop),
            abs(time_drop),
            abs(cost_drop),
            positive_soc_innovation,
            positive_time_innovation,
            abs(float(cost_innovation)),
            soc_ema,
            time_ema,
            math.hypot(soc_ema, time_ema),
            float(belief_uncertainty),
            float(policy_soc),
            float(policy_time),
            float(belief_soc),
            float(belief_time),
            abs(soc_drop) + abs(time_drop) + abs(cost_drop),
            _safe_float(detector_score),
            float(bool(det_route_flag)),
            float(bool(is_new_arrival)),
            float(np.clip(float(time_index) / max(int(horizon_steps), 1), 0.0, 1.0)),
        ],
        dtype=np.float64,
    )
    return features, float(pressure_ema), float(soc_ema), float(time_ema)


class ContinuousScoreBeliefSelector:
    """Single-score UG-BCR-v3 selector; no multi-condition hard gate is used."""

    def __init__(self, config: ContinuousGateConfig) -> None:
        self.config = config
        self.total = 0
        self.belief_count = 0
        self.shield_count = 0
        self.score_sum = 0.0
        self.score_min = 1.0
        self.score_max = 0.0
        self.last_scores: list[float] = []
        self.pressure_gain_ema_by_vehicle: dict[int, float] = {}
        self.soc_innovation_ema_by_vehicle: dict[int, float] = {}
        self.time_innovation_ema_by_vehicle: dict[int, float] = {}

    def reset(self) -> None:
        self.__init__(self.config)

    def _probability(self, features: np.ndarray) -> float:
        means = np.asarray(self.config.feature_means, dtype=np.float64)
        scales = np.asarray(self.config.feature_scales, dtype=np.float64)
        coefficients = np.asarray(self.config.coefficients, dtype=np.float64)
        standardized = (features - means) / scales
        logit = float(np.dot(standardized, coefficients) + float(self.config.intercept))
        if logit >= 0.0:
            probability = 1.0 / (1.0 + math.exp(-min(logit, 60.0)))
        else:
            exp_value = math.exp(max(logit, -60.0))
            probability = exp_value / (1.0 + exp_value)
        return float(np.clip(probability, 0.0, 1.0))

    def select_batch(
        self,
        policy_states: Sequence[np.ndarray],
        belief_states: Sequence[np.ndarray],
        vehicle_ids: Sequence[int],
        is_new_arrivals: Sequence[int],
        belief_estimator: BeliefCoreEstimator,
        shield_config: LocalTemporalShieldConfig | None,
        env: ChargingEnv,
        reward_profile: RewardProfile,
        prev_policy_obs_by_vehicle: dict[int, np.ndarray],
        prev_action_by_vehicle: dict[int, np.ndarray],
        prev_time_by_vehicle: dict[int, int],
        detector_scores: Sequence[float] | None = None,
        route_flags: Sequence[bool] | None = None,
    ) -> tuple[list[np.ndarray], list[str]]:
        del shield_config, prev_policy_obs_by_vehicle, prev_action_by_vehicle, prev_time_by_vehicle
        detector_values = (
            list(detector_scores)
            if detector_scores is not None
            else [0.0 for _ in policy_states]
        )
        route_values = (
            list(route_flags)
            if route_flags is not None
            else [False for _ in policy_states]
        )
        selected: list[np.ndarray] = []
        branches: list[str] = []
        scores: list[float] = []
        for policy_state, belief_state, vehicle_id, new_flag, detector_score, route_flag in zip(
            policy_states,
            belief_states,
            vehicle_ids,
            is_new_arrivals,
            detector_values,
            route_values,
        ):
            vid = int(vehicle_id)
            policy_vec = to_numpy_1d(policy_state).astype(np.float32)
            belief_vec = to_numpy_1d(belief_state).astype(np.float32)
            innovation = belief_estimator.innovation(vid)
            features, pressure_ema, soc_ema, time_ema = build_v3_feature_vector(
                policy_soc=float(policy_vec[0]),
                policy_time=float(policy_vec[1]),
                policy_cost=float(policy_vec[10]),
                belief_soc=float(belief_vec[0]),
                belief_time=float(belief_vec[1]),
                belief_cost=float(belief_vec[10]),
                soc_innovation=float(innovation[0]),
                time_innovation=float(innovation[1]),
                cost_innovation=float(innovation[2]),
                belief_uncertainty=float(belief_estimator.uncertainty(vid)),
                detector_score=float(detector_score),
                det_route_flag=float(bool(route_flag)),
                is_new_arrival=float(bool(new_flag)),
                time_index=int(env.t),
                target_soc=float(reward_profile.exit_target_min),
                previous_pressure_gain_ema=self.pressure_gain_ema_by_vehicle.get(vid),
                previous_soc_innovation_ema=self.soc_innovation_ema_by_vehicle.get(vid, 0.0),
                previous_time_innovation_ema=self.time_innovation_ema_by_vehicle.get(vid, 0.0),
                pressure_ema_decay=float(self.config.pressure_ema_decay),
                innovation_ema_decay=float(self.config.innovation_ema_decay),
                horizon_steps=int(self.config.horizon_steps),
            )
            self.pressure_gain_ema_by_vehicle[vid] = pressure_ema
            self.soc_innovation_ema_by_vehicle[vid] = soc_ema
            self.time_innovation_ema_by_vehicle[vid] = time_ema
            score = self._probability(features)
            use_belief = score >= float(self.config.decision_threshold)
            if bool(self.config.force_policy_on_new_arrival) and bool(new_flag):
                use_belief = False
            if use_belief:
                selected.append(belief_vec)
                branches.append("belief")
                self.belief_count += 1
            else:
                selected.append(policy_vec)
                branches.append("shield")
                self.shield_count += 1
            scores.append(score)
            self.total += 1
            self.score_sum += score
            self.score_min = min(self.score_min, score)
            self.score_max = max(self.score_max, score)
        self.last_scores = scores
        return selected, branches

    def summary(self) -> dict[str, float | int]:
        total = int(self.total)
        return {
            "urgency_gate_total": total,
            "urgency_gate_belief_count": int(self.belief_count),
            "urgency_gate_shield_count": int(self.shield_count),
            "urgency_gate_belief_rate": 0.0 if total == 0 else float(self.belief_count / total),
            "urgency_gate_shield_rate": 0.0 if total == 0 else float(self.shield_count / total),
            "ug_bcr_v3_score_mean": 0.0 if total == 0 else float(self.score_sum / total),
            "ug_bcr_v3_score_min": 0.0 if total == 0 else float(self.score_min),
            "ug_bcr_v3_score_max": 0.0 if total == 0 else float(self.score_max),
            "ug_bcr_v3_decision_threshold": float(self.config.decision_threshold),
        }


def reconstruct_v3_features_from_audit(
    frame: pd.DataFrame,
    *,
    pressure_ema_decay: float = 0.72,
    innovation_ema_decay: float = 0.70,
    horizon_steps: int = 344,
    target_soc: float = float(TRAIN_PROFILE.exit_target_min),
) -> pd.DataFrame:
    required = {
        "scenario_id",
        "attack_key",
        "time_index",
        "batch_index",
        "vehicle_id",
        "policy_soc",
        "policy_time",
        "policy_cost",
        "belief_soc",
        "belief_time",
        "belief_cost",
        "soc_innovation",
        "time_innovation",
        "cost_innovation",
        "belief_uncertainty",
        "detector_score",
        "det_route_flag",
        "is_new_arrival",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"UG-BCR-v3 audit data is missing columns: {missing}")
    ordered = frame.sort_values(
        ["scenario_id", "attack_key", "time_index", "batch_index"], kind="mergesort"
    ).copy()
    pressure_state: dict[tuple[str, str, int], float] = {}
    soc_state: dict[tuple[str, str, int], float] = {}
    time_state: dict[tuple[str, str, int], float] = {}
    feature_rows: list[np.ndarray] = []
    for row in ordered.itertuples(index=False):
        key = (str(row.scenario_id), str(row.attack_key), int(row.vehicle_id))
        features, pressure_ema, soc_ema, time_ema = build_v3_feature_vector(
            policy_soc=_safe_float(row.policy_soc),
            policy_time=_safe_float(row.policy_time),
            policy_cost=_safe_float(row.policy_cost),
            belief_soc=_safe_float(row.belief_soc),
            belief_time=_safe_float(row.belief_time),
            belief_cost=_safe_float(row.belief_cost),
            soc_innovation=_safe_float(row.soc_innovation),
            time_innovation=_safe_float(row.time_innovation),
            cost_innovation=_safe_float(row.cost_innovation),
            belief_uncertainty=_safe_float(row.belief_uncertainty),
            detector_score=_safe_float(row.detector_score),
            det_route_flag=_safe_float(row.det_route_flag),
            is_new_arrival=_safe_float(row.is_new_arrival),
            time_index=int(row.time_index),
            target_soc=float(target_soc),
            previous_pressure_gain_ema=pressure_state.get(key),
            previous_soc_innovation_ema=soc_state.get(key, 0.0),
            previous_time_innovation_ema=time_state.get(key, 0.0),
            pressure_ema_decay=float(pressure_ema_decay),
            innovation_ema_decay=float(innovation_ema_decay),
            horizon_steps=int(horizon_steps),
        )
        pressure_state[key] = pressure_ema
        soc_state[key] = soc_ema
        time_state[key] = time_ema
        feature_rows.append(features)
    feature_matrix = np.vstack(feature_rows)
    # Several runtime features intentionally reuse audit column names. Assigning
    # them explicitly keeps the original row/index order without duplicate-column
    # joins and normalizes their dtype to the exact runtime representation.
    for column_index, column_name in enumerate(V3_FEATURE_NAMES):
        ordered[column_name] = feature_matrix[:, column_index]
    return ordered


def _v2_config_from_payload(payload: dict[str, Any]) -> UGBCRConfig:
    return UGBCRConfig(
        schema_version=int(payload.get("schema_version", 2)),
        leakage_policy=str(payload.get("leakage_policy", "strict_no_clean_state")),
        time_initialization=str(payload.get("time_initialization", "routed_observation")),
        uses_clean_state=bool(payload.get("uses_clean_state", False)),
        uses_true_remaining_time=bool(payload.get("uses_true_remaining_time", False)),
        belief=BeliefCoreConfig(**dict(payload.get("belief") or {})),
        urgency_gate=UrgencyGateConfig(**dict(payload.get("urgency_gate") or {})),
    )


def load_ug_bcr_v3_config(path: str | Path) -> UGBCRV3Config:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    gate_payload = dict(payload["continuous_gate"])
    for key in ("feature_names", "feature_means", "feature_scales", "coefficients"):
        gate_payload[key] = tuple(gate_payload[key])
    return UGBCRV3Config(
        schema_version=int(payload["schema_version"]),
        leakage_policy=str(payload["leakage_policy"]),
        uses_clean_state=bool(payload["uses_clean_state"]),
        uses_true_remaining_time=bool(payload["uses_true_remaining_time"]),
        base_v2=_v2_config_from_payload(dict(payload["base_v2"])),
        continuous_gate=ContinuousGateConfig(**gate_payload),
        training_metadata=dict(payload.get("training_metadata") or {}),
    )


def ug_bcr_v3_config_payload(config: UGBCRV3Config) -> dict[str, Any]:
    return asdict(config)


def rollout_episode_with_ug_bcr_v3(*args, ug_bcr_v3_config: UGBCRV3Config, **kwargs) -> dict:
    selector = ContinuousScoreBeliefSelector(ug_bcr_v3_config.continuous_gate)
    kwargs["ug_bcr_config"] = ug_bcr_v3_config.base_v2
    kwargs["urgency_gate_override"] = selector
    return rollout_episode_with_ug_bcr(*args, **kwargs)


__all__ = [
    "V3_FEATURE_NAMES",
    "ContinuousGateConfig",
    "UGBCRV3Config",
    "ContinuousScoreBeliefSelector",
    "build_v3_feature_vector",
    "reconstruct_v3_features_from_audit",
    "load_ug_bcr_v3_config",
    "ug_bcr_v3_config_payload",
    "rollout_episode_with_ug_bcr_v3",
]
