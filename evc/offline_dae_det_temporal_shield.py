from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .defense import (
    PosteriorBenefitMLPDetector,
    SequentialDAERuntime,
    posterior_detector_probabilities,
    reconstruction_batch,
)
from .merged_core import (
    Actor,
    ChargingEnv,
    QueueItem,
    RewardProfile,
    TRAIN_PROFILE,
    min_max_denormalization,
    normalize_scalar,
    to_numpy_1d,
)
from .rollout_utils import update_active_vehicle_ids

LOCAL_SHIELD_INDICES = (0, 1, 10)
ALL_SHIELD_INDICES = None
_TIME_DECAY = 1.0 / 12.0


@dataclass
class LocalTemporalShieldConfig:
    state_scope: str = 'local'
    tau_soc: float = 0.02
    tau_time: float = 0.005
    tau_cost: float = 0.02
    calibration_quantile: float = 0.99
    min_tau_soc: float = 0.02
    min_tau_time: float = 0.005
    min_tau_cost: float = 0.02
    max_tau_soc: float = 0.08
    max_tau_time: float = 0.03
    max_tau_cost: float = 0.08
    initial_soc: float = 0.0
    initial_cost_norm: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return {
            'state_scope': str(self.state_scope),
            'tau_soc': float(self.tau_soc),
            'tau_time': float(self.tau_time),
            'tau_cost': float(self.tau_cost),
            'calibration_quantile': float(self.calibration_quantile),
            'min_tau_soc': float(self.min_tau_soc),
            'min_tau_time': float(self.min_tau_time),
            'min_tau_cost': float(self.min_tau_cost),
            'max_tau_soc': float(self.max_tau_soc),
            'max_tau_time': float(self.max_tau_time),
            'max_tau_cost': float(self.max_tau_cost),
            'initial_soc': float(self.initial_soc),
            'initial_cost_norm': float(self.initial_cost_norm),
            'shield_indices': list(LOCAL_SHIELD_INDICES),
        }


@dataclass
class TemporalShieldArtifact:
    config: LocalTemporalShieldConfig
    metadata: dict
    calibration_stats: dict


def _canonical_scope(state_scope: str) -> str:
    token = str(state_scope or 'local').strip().lower()
    if token not in {'local', 'all'}:
        raise ValueError(f'Temporal shield only supports local/all scopes, got: {state_scope}')
    return token


def calibrate_local_temporal_shield(
    arrivals: pd.DataFrame,
    signals_path,
    actor: Actor,
    device: torch.device,
    *,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    calibration_quantile: float = 0.99,
    min_tau_soc: float = 0.02,
    min_tau_time: float = 0.005,
    min_tau_cost: float = 0.02,
    max_tau_soc: float = 0.08,
    max_tau_time: float = 0.03,
    max_tau_cost: float = 0.08,
    state_scope: str = 'local',
) -> tuple[LocalTemporalShieldConfig, dict]:
    state_scope = _canonical_scope(state_scope)
    env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
    env.reset()
    actor = actor.to(device).eval()
    idx = 0
    active: list[QueueItem] = []
    active_vehicle_ids: list[int] = []
    prev_policy_obs_by_vehicle: dict[int, np.ndarray] = {}
    prev_action_by_vehicle: dict[int, np.ndarray] = {}
    prev_time_by_vehicle: dict[int, int] = {}
    residual_soc: list[float] = []
    residual_time: list[float] = []
    residual_cost: list[float] = []
    calibration_samples = 0

    def _compute_actions(policy_states: list[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.as_tensor(np.asarray(policy_states, dtype=np.float32), dtype=torch.float32, device=device)
            return actor(state_t).detach().cpu().numpy()

    while env.t < env.horizon:
        new_states: list[np.ndarray] = []
        new_stations: list[int] = []
        new_vehicle_ids: list[int] = []
        while idx < len(arrivals) and int(arrivals.loc[idx, 'Arrive_time']) == env.t:
            new_states.append(env.build_initial_obs(int(arrivals.loc[idx, 'Duration_of_stay'])))
            new_stations.append(int(arrivals.loc[idx, 'Station']))
            new_vehicle_ids.append(int(idx))
            idx += 1
        if new_states:
            calibration_samples += len(new_states)
            actions = _compute_actions(new_states)
            for obs, action, station, vehicle_id in zip(new_states, actions, new_stations, new_vehicle_ids):
                env.enqueue(obs, action, station)
                prev_policy_obs_by_vehicle[int(vehicle_id)] = to_numpy_1d(obs)
                prev_action_by_vehicle[int(vehicle_id)] = to_numpy_1d(action)
                prev_time_by_vehicle[int(vehicle_id)] = int(env.t)

        if active:
            active_states = [item.obs for item in active]
            calibration_samples += len(active_states)
            for vehicle_id, obs in zip(active_vehicle_ids, active_states):
                prev_state = prev_policy_obs_by_vehicle.get(int(vehicle_id))
                prev_action = prev_action_by_vehicle.get(int(vehicle_id))
                prev_time_index = prev_time_by_vehicle.get(int(vehicle_id))
                if prev_state is None or prev_action is None or prev_time_index is None:
                    continue
                soc_center, time_center, cost_center = _physical_centers_from_prev(
                    prev_state,
                    prev_action,
                    int(prev_time_index),
                    env,
                )
                obs_vec = to_numpy_1d(obs)
                residual_soc.append(abs(float(obs_vec[0]) - soc_center))
                residual_time.append(abs(float(obs_vec[1]) - time_center))
                residual_cost.append(abs(float(obs_vec[10]) - cost_center))
            actions = _compute_actions(active_states)
            for item, action, vehicle_id in zip(active, actions, active_vehicle_ids):
                env.enqueue(item.obs, action, item.station)
                prev_policy_obs_by_vehicle[int(vehicle_id)] = to_numpy_1d(item.obs)
                prev_action_by_vehicle[int(vehicle_id)] = to_numpy_1d(action)
                prev_time_by_vehicle[int(vehicle_id)] = int(env.t)

        step_vehicle_ids = new_vehicle_ids + active_vehicle_ids
        transitions, next_active, _ = env.step()
        active = next_active
        active_vehicle_ids = update_active_vehicle_ids(step_vehicle_ids, transitions)

    def _summarize(values: list[float], minimum: float, maximum: float) -> tuple[float, float, float, float]:
        if not values:
            return float(minimum), 0.0, 0.0, 0.0
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        quantile = float(np.quantile(arr, float(np.clip(calibration_quantile, 0.0, 1.0))))
        tau = float(np.clip(max(float(minimum), quantile), float(minimum), float(maximum)))
        return tau, float(quantile), float(np.mean(arr)), float(np.max(arr))

    tau_soc, q_soc, mean_soc, max_soc = _summarize(residual_soc, float(min_tau_soc), float(max_tau_soc))
    tau_time, q_time, mean_time, max_time = _summarize(residual_time, float(min_tau_time), float(max_tau_time))
    tau_cost, q_cost, mean_cost, max_cost = _summarize(residual_cost, float(min_tau_cost), float(max_tau_cost))
    config = LocalTemporalShieldConfig(
        state_scope=state_scope,
        tau_soc=tau_soc,
        tau_time=tau_time,
        tau_cost=tau_cost,
        calibration_quantile=float(calibration_quantile),
        min_tau_soc=float(min_tau_soc),
        min_tau_time=float(min_tau_time),
        min_tau_cost=float(min_tau_cost),
        max_tau_soc=float(max_tau_soc),
        max_tau_time=float(max_tau_time),
        max_tau_cost=float(max_tau_cost),
    )
    stats = {
        'calibration_samples': calibration_samples,
        'calibration_quantile': float(calibration_quantile),
        'residual_soc_quantile': q_soc,
        'residual_time_quantile': q_time,
        'residual_cost_quantile': q_cost,
        'residual_soc_mean': mean_soc,
        'residual_time_mean': mean_time,
        'residual_cost_mean': mean_cost,
        'residual_soc_max': max_soc,
        'residual_time_max': max_time,
        'residual_cost_max': max_cost,
        'tau_soc': float(config.tau_soc),
        'tau_time': float(config.tau_time),
        'tau_cost': float(config.tau_cost),
    }
    return config, stats


def save_temporal_shield_bundle(
    config: LocalTemporalShieldConfig,
    path: str | Path,
    *,
    metadata: dict | None = None,
    calibration_stats: dict | None = None,
) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_type': 'offline_dae_det_temporal_shield',
            'config': config.to_dict(),
            'metadata': dict(metadata or {}),
            'calibration_stats': dict(calibration_stats or {}),
        },
        target,
    )
    return target


def load_temporal_shield_bundle(path: str | Path) -> TemporalShieldArtifact:
    payload = torch.load(Path(path).expanduser().resolve(), map_location='cpu', weights_only=False)
    if not isinstance(payload, dict) or str(payload.get('model_type', '')) != 'offline_dae_det_temporal_shield':
        raise ValueError(f'Not a temporal shield artifact: {path}')
    config_dict = dict(payload.get('config') or {})
    config = LocalTemporalShieldConfig(
        state_scope=str(config_dict.get('state_scope', 'local')),
        tau_soc=float(config_dict.get('tau_soc', 0.02)),
        tau_time=float(config_dict.get('tau_time', 0.005)),
        tau_cost=float(config_dict.get('tau_cost', 0.02)),
        calibration_quantile=float(config_dict.get('calibration_quantile', 0.99)),
        min_tau_soc=float(config_dict.get('min_tau_soc', 0.02)),
        min_tau_time=float(config_dict.get('min_tau_time', 0.005)),
        min_tau_cost=float(config_dict.get('min_tau_cost', 0.02)),
        max_tau_soc=float(config_dict.get('max_tau_soc', 0.08)),
        max_tau_time=float(config_dict.get('max_tau_time', 0.03)),
        max_tau_cost=float(config_dict.get('max_tau_cost', 0.08)),
        initial_soc=float(config_dict.get('initial_soc', 0.0)),
        initial_cost_norm=float(config_dict.get('initial_cost_norm', 0.2)),
    )
    return TemporalShieldArtifact(
        config=config,
        metadata=dict(payload.get('metadata') or {}),
        calibration_stats=dict(payload.get('calibration_stats') or {}),
    )


def _physical_centers_from_prev(
    prev_state: np.ndarray,
    prev_action: np.ndarray,
    prev_time_index: int,
    env: ChargingEnv,
) -> tuple[float, float, float]:
    prev_vec = to_numpy_1d(prev_state)
    prev_act = to_numpy_1d(prev_action)
    action_scalar = float(prev_act[0]) if prev_act.size else 0.0
    soc_step = float(env.max_power * env.slice_hours / env.battery_capacity)
    soc_center = float(prev_vec[0] + action_scalar * soc_step)
    time_center = float(max(prev_vec[1] - _TIME_DECAY, 0.0))
    price_idx = int(np.clip(int(prev_time_index), 0, env.horizon - 1))
    prev_cost = float(min_max_denormalization(float(prev_vec[10]), 0.0, env._cost_upper_bound()))
    step_cost = float(action_scalar * env.max_power * env.slice_hours * float(env.signals.price[price_idx]))
    cost_center = float(np.clip(normalize_scalar(prev_cost + step_cost, 0.0, env._cost_upper_bound()), 0.0, 1.0))
    return soc_center, time_center, cost_center


def _shield_single_state(
    state: np.ndarray,
    prev_state: np.ndarray | None,
    prev_action: np.ndarray | None,
    prev_time_index: int | None,
    config: LocalTemporalShieldConfig,
    env: ChargingEnv,
    *,
    is_new_arrival: bool,
) -> tuple[np.ndarray, dict[str, bool]]:
    corrected = to_numpy_1d(state).copy()
    if prev_state is None or prev_action is None or prev_time_index is None or is_new_arrival:
        soc_center = float(config.initial_soc)
        time_center = float(corrected[1])
        cost_center = float(config.initial_cost_norm)
    else:
        soc_center, time_center, cost_center = _physical_centers_from_prev(prev_state, prev_action, int(prev_time_index), env)
    corrected[0] = float(np.clip(corrected[0], soc_center - float(config.tau_soc), soc_center + float(config.tau_soc)))
    corrected[1] = float(np.clip(corrected[1], time_center - float(config.tau_time), time_center + float(config.tau_time)))
    corrected[10] = float(np.clip(corrected[10], cost_center - float(config.tau_cost), cost_center + float(config.tau_cost)))
    changes = np.abs(corrected - to_numpy_1d(state))
    flags = {
        'soc': bool(changes[0] > 1e-8),
        'time': bool(changes[1] > 1e-8),
        'cost': bool(changes[10] > 1e-8),
    }
    return corrected.astype(np.float32), flags


def _route_policy_states_core_only(
    attacked_states,
    attacked_flags,
    defender: nn.Module | None,
    detector_model: PosteriorBenefitMLPDetector | None,
    actor: Actor,
    device: torch.device,
    *,
    route_mode: str,
    detector_threshold: float | None,
    detector_feature_mode: str = 'posterior',
    time_indices=None,
    stations=None,
    is_new_arrivals=None,
    prev_obs_refs=None,
    vehicle_ids=None,
    episode_index: int = 0,
    dae_runtime: SequentialDAERuntime | None = None,
):
    """Route policy states with selective core-state DAE repair.

    The DAE may reconstruct the full 11-dimensional observation, but only the
    safety-critical physical coordinates ``LOCAL_SHIELD_INDICES`` (SOC,
    remaining time, cumulative cost) are injected into the policy state.  The
    exogenous and price-window coordinates are kept from the observed state to
    avoid full-state DAE reconstruction bias under low-amplitude all-state drift.
    """
    del attacked_flags, detector_feature_mode
    if route_mode == 'none':
        return [to_numpy_1d(s) for s in attacked_states], [False for _ in attacked_states], np.full((len(attacked_states),), np.nan, dtype=np.float32)
    if route_mode != 'detector':
        raise ValueError(f'Unsupported mainline route_mode: {route_mode!r}')
    if defender is None:
        raise ValueError('repair_mode=core_only requires a defender')
    obs_arr = np.asarray(attacked_states, dtype=np.float32).reshape(-1, 11)
    if obs_arr.shape[0] == 0:
        return [], [], np.zeros((0,), dtype=np.float32)
    if dae_runtime is not None and vehicle_ids is not None:
        recovered_full = dae_runtime.reconstruct_batch(obs_arr, vehicle_ids=vehicle_ids, episode_index=episode_index)
    else:
        recovered_full = reconstruction_batch(defender, obs_arr, device)
    recovered_full = np.asarray(recovered_full, dtype=np.float32).reshape(-1, 11)
    recovered_core = obs_arr.copy()
    recovered_core[:, list(LOCAL_SHIELD_INDICES)] = recovered_full[:, list(LOCAL_SHIELD_INDICES)]

    if route_mode == 'detector':
        if detector_threshold is None:
            raise ValueError('route_mode=detector requires detector_threshold')
        if detector_model is None:
            raise ValueError('route_mode=detector requires detector_model')
        if not isinstance(detector_model, PosteriorBenefitMLPDetector):
            raise ValueError(f'repair_mode=core_only expects PosteriorBenefitMLPDetector, got {type(detector_model)!r}')
        scores = posterior_detector_probabilities(
            detector_model,
            obs_arr,
            recovered_core,
            actor,
            device,
            time_indices=time_indices,
            stations=stations,
            is_new_arrivals=is_new_arrivals,
            prev_obs_inputs=prev_obs_refs,
            include_temporal=bool(getattr(detector_model, 'include_temporal', True)),
        )
        score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        flags = [bool(score >= float(detector_threshold)) for score in score_arr]
        routed = [recovered_core[i].reshape(-1) if flags[i] else obs_arr[i].reshape(-1) for i in range(len(flags))]
        return routed, flags, score_arr
    raise ValueError(f'Unknown route_mode: {route_mode}')



__all__ = [
    "LOCAL_SHIELD_INDICES",
    "ALL_SHIELD_INDICES",
    "LocalTemporalShieldConfig",
    "TemporalShieldArtifact",
    "calibrate_local_temporal_shield",
    "save_temporal_shield_bundle",
    "load_temporal_shield_bundle",
]
