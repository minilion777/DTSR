from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
from .long_horizon_attacks import build_long_horizon_attacker
from .merged_attacks import AttackScope, PGDStateAttacker, attack_batch_by_context, build_state_attacker
from .merged_core import (
    Actor,
    ChargingEnv,
    Critic,
    QueueItem,
    RewardProfile,
    TRAIN_PROFILE,
    min_max_denormalization,
    normalize_scalar,
    to_numpy_1d,
)
from .merged_pipeline import (
    _build_contexts,
    _rollout_label,
    _route_policy_states,
    normalize_result_frame,
    summarize_metrics,
)
from .sequential_adversary import observation_bounds_for_arrivals, update_active_vehicle_ids

LOCAL_SHIELD_INDICES = (0, 1, 10)
ALL_SHIELD_INDICES = None
_TIME_DECAY = 1.0 / 12.0
SHORT_TUNING_ATTACK_ALGORITHMS = ('opposite_pgd', 'q_function')
LONG_TUNING_ATTACKS_BY_SCOPE = {
    'local': ('local_small_drift_q',),
    'all': (),
}


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


def shield_metric_columns(summary: dict[str, Any], *, prefix: str) -> dict[str, float | int]:
    return {
        f'shield_correction_mean_{prefix}': float(summary.get('shield_correction_mean', 0.0)),
        f'shield_correction_max_{prefix}': float(summary.get('shield_correction_max', 0.0)),
    }


def _empty_shield_metric_columns(*, prefix: str) -> dict[str, float]:
    return {
        f'shield_correction_mean_{prefix}': np.nan,
        f'shield_correction_max_{prefix}': np.nan,
    }


def attack_metric_columns(summary: dict[str, Any], *, prefix: str) -> dict[str, float | int]:
    return {
        f'attack_delta_count_{prefix}': int(summary.get('attack_delta_count', 0)),
        f'attack_delta_linf_mean_{prefix}': float(summary.get('attack_delta_linf_mean', 0.0)),
        f'attack_delta_l2_mean_{prefix}': float(summary.get('attack_delta_l2_mean', 0.0)),
        f'attack_delta_local_linf_mean_{prefix}': float(summary.get('attack_delta_local_linf_mean', 0.0)),
        f'attack_delta_local_l2_mean_{prefix}': float(summary.get('attack_delta_local_l2_mean', 0.0)),
        f'attack_delta_env_linf_mean_{prefix}': float(summary.get('attack_delta_env_linf_mean', 0.0)),
        f'attack_delta_env_l2_mean_{prefix}': float(summary.get('attack_delta_env_l2_mean', 0.0)),
        f'attack_delta_price_linf_mean_{prefix}': float(summary.get('attack_delta_price_linf_mean', 0.0)),
        f'attack_delta_price_l2_mean_{prefix}': float(summary.get('attack_delta_price_l2_mean', 0.0)),
        f'attack_delta_linf_max_{prefix}': float(summary.get('attack_delta_linf_max', 0.0)),
        f'attack_delta_l2_max_{prefix}': float(summary.get('attack_delta_l2_max', 0.0)),
        f'attack_delta_local_linf_max_{prefix}': float(summary.get('attack_delta_local_linf_max', 0.0)),
        f'attack_delta_local_l2_max_{prefix}': float(summary.get('attack_delta_local_l2_max', 0.0)),
    }


def _empty_attack_metric_columns(*, prefix: str) -> dict[str, float]:
    return {
        f'attack_delta_count_{prefix}': np.nan,
        f'attack_delta_linf_mean_{prefix}': np.nan,
        f'attack_delta_l2_mean_{prefix}': np.nan,
        f'attack_delta_local_linf_mean_{prefix}': np.nan,
        f'attack_delta_local_l2_mean_{prefix}': np.nan,
        f'attack_delta_env_linf_mean_{prefix}': np.nan,
        f'attack_delta_env_l2_mean_{prefix}': np.nan,
        f'attack_delta_price_linf_mean_{prefix}': np.nan,
        f'attack_delta_price_l2_mean_{prefix}': np.nan,
        f'attack_delta_linf_max_{prefix}': np.nan,
        f'attack_delta_l2_max_{prefix}': np.nan,
        f'attack_delta_local_linf_max_{prefix}': np.nan,
        f'attack_delta_local_l2_max_{prefix}': np.nan,
    }


def _fresh_attacker(attacker):
    if attacker is None:
        return None
    cloned = attacker.clone() if hasattr(attacker, 'clone') else attacker
    if hasattr(cloned, 'reset'):
        cloned.reset()
    return cloned



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
    del detector_feature_mode
    if route_mode == 'none':
        return [to_numpy_1d(s) for s in attacked_states], [False for _ in attacked_states], np.full((len(attacked_states),), np.nan, dtype=np.float32)
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

    if route_mode == 'always_dae':
        flags = [True for _ in range(obs_arr.shape[0])]
        return [recovered_core[i].reshape(-1) for i in range(obs_arr.shape[0])], flags, np.full((obs_arr.shape[0],), np.nan, dtype=np.float32)
    if route_mode == 'oracle':
        flags = [bool(flag) for flag in attacked_flags]
        routed = [recovered_core[i].reshape(-1) if flags[i] else obs_arr[i].reshape(-1) for i in range(len(flags))]
        return routed, flags, np.full((len(flags),), np.nan, dtype=np.float32)
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


def rollout_episode_with_dae_det_temporal_shield(
    arrivals: pd.DataFrame,
    actor: Actor,
    signals_path,
    device: torch.device,
    reward_profile: RewardProfile,
    *,
    attack_enabled: bool = False,
    attack_scenario: str = 'O',
    attacker: PGDStateAttacker | None = None,
    learned_adversary=None,
    defender: nn.Module | None = None,
    detector_model: PosteriorBenefitMLPDetector | None = None,
    detector_threshold: float | None = None,
    shield_config: LocalTemporalShieldConfig | None = None,
    route_mode: str = 'none',
    enable_shield: bool = False,
    epsilon: float = 0.15,
    state_scope: str = 'local',
    obs_low: np.ndarray | None = None,
    obs_high: np.ndarray | None = None,
    exploration_noise: float = 0.0,
    price_threshold: float = 400.0,
    soc_new_threshold: float = 0.5,
    soc_rollout_threshold: float = 0.3,
    even_station_target: float = 1.0,
    odd_station_target: float = -0.5,
    attack_ratio: float = 1.0,
    attack_scope: AttackScope = 'obs',
    label: str | None = None,
    repair_mode: str = 'full',
) -> dict:
    env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
    if attacker is None and learned_adversary is not None:
        if obs_low is None or obs_high is None:
            obs_low, obs_high = observation_bounds_for_arrivals(arrivals, signals_path, reward_profile)
        attacker = build_state_attacker(
            actor,
            device=device,
            algorithm='learned_sequence',
            epsilon=float(epsilon),
            obs_low=obs_low,
            obs_high=obs_high,
            adversary=learned_adversary,
            attack_state_scope=state_scope,
            signals_path=signals_path,
            reward_profile=reward_profile,
        )
    env.reset()
    actor = actor.to(device).eval()
    idx = 0
    active: list[QueueItem] = []
    active_vehicle_ids: list[int] = []
    route_count = 0
    route_total = 0
    attack_obs_count = 0
    correction_values: list[float] = []
    max_corrections: list[float] = []
    soc_clamp = 0
    time_clamp = 0
    cost_clamp = 0
    shield_process_total = 0
    attack_delta_count = 0
    attack_delta_linf_sum = 0.0
    attack_delta_l2_sum = 0.0
    attack_delta_local_linf_sum = 0.0
    attack_delta_local_l2_sum = 0.0
    attack_delta_env_linf_sum = 0.0
    attack_delta_env_l2_sum = 0.0
    attack_delta_price_linf_sum = 0.0
    attack_delta_price_l2_sum = 0.0
    attack_delta_linf_max = 0.0
    attack_delta_l2_max = 0.0
    attack_delta_local_linf_max = 0.0
    attack_delta_local_l2_max = 0.0
    prev_observed_obs_by_vehicle: dict[int, np.ndarray] = {}
    prev_policy_obs_by_vehicle: dict[int, np.ndarray] = {}
    prev_action_by_vehicle: dict[int, np.ndarray] = {}
    prev_time_by_vehicle: dict[int, int] = {}
    dae_runtime = None if defender is None else SequentialDAERuntime(defender, device)
    repair_mode = str(repair_mode or 'full').strip().lower().replace('-', '_')
    if repair_mode not in {'full', 'core_only'}:
        raise ValueError(f'Unknown repair_mode: {repair_mode!r}')

    def _lookup_prev_observed(vehicle_ids: list[int], observed_states: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for vehicle_id, observed_state in zip(vehicle_ids, observed_states):
            observed_vec = to_numpy_1d(observed_state)
            out.append(prev_observed_obs_by_vehicle.get(int(vehicle_id), observed_vec))
        return out

    def _apply_shield_batch(
        policy_states: list[np.ndarray],
        vehicle_ids: list[int],
        is_new_arrivals: list[int],
        current_time: int,
    ) -> list[np.ndarray]:
        nonlocal soc_clamp, time_clamp, cost_clamp, shield_process_total
        del current_time
        if not enable_shield or shield_config is None:
            return [to_numpy_1d(s) for s in policy_states]
        route_states = [to_numpy_1d(s) for s in policy_states]
        out: list[np.ndarray] = []
        for idx, (policy_state, vehicle_id, new_flag) in enumerate(zip(route_states, vehicle_ids, is_new_arrivals)):
            policy_vec = to_numpy_1d(policy_state)
            prev_state = prev_policy_obs_by_vehicle.get(int(vehicle_id))
            prev_action = prev_action_by_vehicle.get(int(vehicle_id))
            prev_time_index = prev_time_by_vehicle.get(int(vehicle_id))
            corrected, flags = _shield_single_state(
                policy_vec,
                prev_state,
                prev_action,
                prev_time_index,
                shield_config,
                env,
                is_new_arrival=bool(new_flag),
            )
            chosen = to_numpy_1d(corrected).astype(np.float32)
            diff = np.abs(chosen - policy_vec)
            guarded_diff = diff[list(LOCAL_SHIELD_INDICES)]
            correction_values.append(float(np.mean(guarded_diff)))
            max_corrections.append(float(np.max(guarded_diff)))
            soc_clamp += int(flags['soc'])
            time_clamp += int(flags['time'])
            cost_clamp += int(flags['cost'])
            shield_process_total += 1
            out.append(chosen)
            prev_policy_obs_by_vehicle[int(vehicle_ids[idx])] = to_numpy_1d(chosen)
        return out

    def _update_prev_observed(vehicle_ids: list[int], observed_states: list[np.ndarray]) -> None:
        for vehicle_id, observed_state in zip(vehicle_ids, observed_states):
            prev_observed_obs_by_vehicle[int(vehicle_id)] = to_numpy_1d(observed_state)

    def _compute_actions(policy_states: list[np.ndarray], *, apply_noise: bool = True) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.as_tensor(np.asarray(policy_states, dtype=np.float32), dtype=torch.float32, device=device)
            actions = actor(state_t).detach().cpu().numpy()
        if apply_noise and exploration_noise > 0.0:
            actions = actions + np.random.normal(0.0, exploration_noise, size=actions.shape)
        return np.clip(actions, -1.0, 1.0)

    def _update_prev_actions(vehicle_ids: list[int], actions: np.ndarray, current_time: int) -> None:
        for vehicle_id, action in zip(vehicle_ids, np.asarray(actions, dtype=np.float32)):
            prev_action_by_vehicle[int(vehicle_id)] = to_numpy_1d(action)
            prev_time_by_vehicle[int(vehicle_id)] = int(current_time)

    def _record_attack_delta_stats(clean_states: list[np.ndarray], attacked_states: list[np.ndarray], attacked_flags: list[bool]) -> None:
        nonlocal attack_delta_count, attack_delta_linf_sum, attack_delta_l2_sum
        nonlocal attack_delta_local_linf_sum, attack_delta_local_l2_sum
        nonlocal attack_delta_env_linf_sum, attack_delta_env_l2_sum
        nonlocal attack_delta_price_linf_sum, attack_delta_price_l2_sum
        nonlocal attack_delta_linf_max, attack_delta_l2_max
        nonlocal attack_delta_local_linf_max, attack_delta_local_l2_max
        guarded_idx = list(LOCAL_SHIELD_INDICES)
        env_idx = [2, 3, 4]
        price_idx = [5, 6, 7, 8, 9]
        for clean_state, attacked_state, attacked_flag in zip(clean_states, attacked_states, attacked_flags):
            if not bool(attacked_flag):
                continue
            clean_vec = to_numpy_1d(clean_state)
            attacked_vec = to_numpy_1d(attacked_state)
            delta = attacked_vec - clean_vec
            local_delta = delta[guarded_idx]
            env_delta = delta[env_idx]
            price_delta = delta[price_idx]
            linf = float(np.max(np.abs(delta)))
            l2 = float(np.linalg.norm(delta, ord=2))
            local_linf = float(np.max(np.abs(local_delta)))
            local_l2 = float(np.linalg.norm(local_delta, ord=2))
            env_linf = float(np.max(np.abs(env_delta)))
            env_l2 = float(np.linalg.norm(env_delta, ord=2))
            price_linf = float(np.max(np.abs(price_delta)))
            price_l2 = float(np.linalg.norm(price_delta, ord=2))
            attack_delta_count += 1
            attack_delta_linf_sum += linf
            attack_delta_l2_sum += l2
            attack_delta_local_linf_sum += local_linf
            attack_delta_local_l2_sum += local_l2
            attack_delta_env_linf_sum += env_linf
            attack_delta_env_l2_sum += env_l2
            attack_delta_price_linf_sum += price_linf
            attack_delta_price_l2_sum += price_l2
            attack_delta_linf_max = max(attack_delta_linf_max, linf)
            attack_delta_l2_max = max(attack_delta_l2_max, l2)
            attack_delta_local_linf_max = max(attack_delta_local_linf_max, local_linf)
            attack_delta_local_l2_max = max(attack_delta_local_l2_max, local_l2)

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
            contexts = _build_contexts(env, new_states, new_stations, attack_scenario, True, price_threshold, soc_new_threshold, soc_rollout_threshold, even_station_target, odd_station_target)
            attacked_states, attacked_flags = attack_batch_by_context(
                attacker if attack_enabled else None,
                new_states,
                contexts,
                attack_ratio=attack_ratio,
                attack_scope=attack_scope,
                vehicle_ids=new_vehicle_ids,
                episode_index=0,
                seed=42 if attacker is None else int(getattr(attacker, 'seed', 42)),
            )
            _record_attack_delta_stats(new_states, attacked_states, attacked_flags)
            observed_states = attacked_states if attack_enabled else [to_numpy_1d(x) for x in new_states]
            prev_refs = _lookup_prev_observed(new_vehicle_ids, observed_states)
            route_fn = _route_policy_states_core_only if repair_mode == 'core_only' else _route_policy_states
            policy_states, route_flags, _ = route_fn(
                observed_states,
                attacked_flags,
                defender,
                detector_model,
                actor,
                device,
                route_mode=route_mode,
                detector_threshold=detector_threshold,
                detector_feature_mode='posterior',
                time_indices=[env.t for _ in new_states],
                stations=new_stations,
                is_new_arrivals=[1 for _ in new_states],
                prev_obs_refs=prev_refs,
                vehicle_ids=new_vehicle_ids,
                episode_index=0,
                dae_runtime=dae_runtime,
            )
            policy_states = _apply_shield_batch(policy_states, new_vehicle_ids, [1 for _ in new_states], int(env.t))
            route_count += int(sum(route_flags))
            route_total += len(route_flags)
            attack_obs_count += int(sum(attacked_flags))
            actions = _compute_actions(policy_states)
            for clean_obs, action, station in zip(new_states, actions, new_stations):
                env.enqueue(clean_obs, action, station)
            _update_prev_actions(new_vehicle_ids, actions, int(env.t))
            _update_prev_observed(new_vehicle_ids, observed_states)

        if active:
            active_states = [item.obs for item in active]
            active_stations = [item.station for item in active]
            contexts = _build_contexts(env, active_states, active_stations, attack_scenario, False, price_threshold, soc_new_threshold, soc_rollout_threshold, even_station_target, odd_station_target)
            attacked_states, attacked_flags = attack_batch_by_context(
                attacker if attack_enabled else None,
                active_states,
                contexts,
                attack_ratio=attack_ratio,
                attack_scope=attack_scope,
                vehicle_ids=active_vehicle_ids,
                episode_index=0,
                seed=42 if attacker is None else int(getattr(attacker, 'seed', 42)),
            )
            _record_attack_delta_stats(active_states, attacked_states, attacked_flags)
            observed_states = attacked_states if attack_enabled else [to_numpy_1d(x) for x in active_states]
            prev_refs = _lookup_prev_observed(active_vehicle_ids, observed_states)
            route_fn = _route_policy_states_core_only if repair_mode == 'core_only' else _route_policy_states
            policy_states, route_flags, _ = route_fn(
                observed_states,
                attacked_flags,
                defender,
                detector_model,
                actor,
                device,
                route_mode=route_mode,
                detector_threshold=detector_threshold,
                detector_feature_mode='posterior',
                time_indices=[env.t for _ in active_states],
                stations=active_stations,
                is_new_arrivals=[0 for _ in active_states],
                prev_obs_refs=prev_refs,
                vehicle_ids=active_vehicle_ids,
                episode_index=0,
                dae_runtime=dae_runtime,
            )
            policy_states = _apply_shield_batch(policy_states, active_vehicle_ids, [0 for _ in active_states], int(env.t))
            route_count += int(sum(route_flags))
            route_total += len(route_flags)
            attack_obs_count += int(sum(attacked_flags))
            actions = _compute_actions(policy_states)
            for item, action in zip(active, actions):
                env.enqueue(item.obs, action, item.station)
            _update_prev_actions(active_vehicle_ids, actions, int(env.t))
            _update_prev_observed(active_vehicle_ids, observed_states)

        step_vehicle_ids = new_vehicle_ids + active_vehicle_ids
        transitions, next_active, _ = env.step()
        active = next_active
        active_vehicle_ids = update_active_vehicle_ids(step_vehicle_ids, transitions)

    if label is not None:
        rollout_label = str(label)
    elif enable_shield:
        rollout_label = 'attack_dae_det_shield' if attack_enabled else 'clean_dae_det_shield'
    else:
        rollout_label = _rollout_label(attack_enabled, route_mode)
    summary = summarize_metrics(env.metrics, rollout_label)
    summary['route_count'] = int(route_count)
    summary['route_total'] = int(route_total)
    summary['route_rate'] = 0.0 if route_total == 0 else float(route_count / route_total)
    summary['attack_obs_count'] = int(attack_obs_count)
    summary['attack_obs_rate'] = 0.0 if route_total == 0 else float(attack_obs_count / route_total)
    summary['attack_ratio_target'] = float(np.clip(attack_ratio, 0.0, 1.0))
    summary['attack_scope'] = str(attack_scope)
    summary['attack_delta_count'] = int(attack_delta_count)
    summary['attack_delta_linf_mean'] = 0.0 if attack_delta_count == 0 else float(attack_delta_linf_sum / attack_delta_count)
    summary['attack_delta_l2_mean'] = 0.0 if attack_delta_count == 0 else float(attack_delta_l2_sum / attack_delta_count)
    summary['attack_delta_local_linf_mean'] = 0.0 if attack_delta_count == 0 else float(attack_delta_local_linf_sum / attack_delta_count)
    summary['attack_delta_local_l2_mean'] = 0.0 if attack_delta_count == 0 else float(attack_delta_local_l2_sum / attack_delta_count)
    summary['attack_delta_env_linf_mean'] = 0.0 if attack_delta_count == 0 else float(attack_delta_env_linf_sum / attack_delta_count)
    summary['attack_delta_env_l2_mean'] = 0.0 if attack_delta_count == 0 else float(attack_delta_env_l2_sum / attack_delta_count)
    summary['attack_delta_price_linf_mean'] = 0.0 if attack_delta_count == 0 else float(attack_delta_price_linf_sum / attack_delta_count)
    summary['attack_delta_price_l2_mean'] = 0.0 if attack_delta_count == 0 else float(attack_delta_price_l2_sum / attack_delta_count)
    summary['attack_delta_linf_max'] = float(attack_delta_linf_max)
    summary['attack_delta_l2_max'] = float(attack_delta_l2_max)
    summary['attack_delta_local_linf_max'] = float(attack_delta_local_linf_max)
    summary['attack_delta_local_l2_max'] = float(attack_delta_local_l2_max)
    summary['repair_mode'] = str(repair_mode)
    summary['shield_correction_mean'] = float(np.mean(correction_values)) if correction_values else 0.0
    summary['shield_correction_max'] = float(np.max(max_corrections)) if max_corrections else 0.0
    summary['shield_soc_clamp_rate'] = 0.0 if shield_process_total == 0 else float(soc_clamp / shield_process_total)
    summary['shield_time_clamp_rate'] = 0.0 if shield_process_total == 0 else float(time_clamp / shield_process_total)
    summary['shield_cost_clamp_rate'] = 0.0 if shield_process_total == 0 else float(cost_clamp / shield_process_total)
    return summary


def collect_adversary_rollouts_for_dae_det_temporal_shield(
    arrivals: pd.DataFrame,
    signals_path,
    actor: Actor,
    adversary,
    adversary_value,
    defender: nn.Module,
    detector_model: PosteriorBenefitMLPDetector,
    detector_threshold: float,
    device: torch.device,
    *,
    shield_config: LocalTemporalShieldConfig | None,
    enable_shield: bool,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    epsilon: float = 0.15,
    state_scope: str = 'local',
    phase_steps: int = 2048,
    start_episode_index: int = 0,
    obs_low: np.ndarray | None = None,
    obs_high: np.ndarray | None = None,
):
    raise NotImplementedError('Learned sequence adversary rollouts are not restored in the GRU-VAE temporal shield line yet.')


def train_learned_sequence_adversary_for_dae_det_temporal_shield(
    arrivals: pd.DataFrame,
    signals_path,
    actor: Actor,
    defender: nn.Module,
    detector_model: PosteriorBenefitMLPDetector,
    detector_threshold: float,
    device: torch.device,
    *,
    shield_config: LocalTemporalShieldConfig | None,
    enable_shield: bool = True,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    epsilon: float = 0.15,
    state_scope: str = 'local',
    iters: int = 200,
    phase_steps: int = 2048,
    ppo_epochs: int = 10,
    num_minibatches: int = 32,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    adv_actor_lr: float = 1e-3,
    adv_critic_lr: float = 1e-5,
    adv_entropy_coeff: float = 1e-4,
    hidden_dim: int = 128,
    seed: int = 42,
    print_every: int = 10,
):
    raise NotImplementedError('Learned sequence adversary training is not restored in the GRU-VAE temporal shield line yet.')


def _safe_recovery(clean_reward: float, attack_reward: float, defended_reward: float) -> float:
    denom = float(clean_reward - attack_reward)
    if abs(denom) <= 1e-8:
        return 0.0
    return float((defended_reward - attack_reward) / denom)


def _validation_attack_ids_for_scope(state_scope: str) -> tuple[str, ...]:
    scope = _canonical_scope(state_scope)
    return tuple(SHORT_TUNING_ATTACK_ALGORITHMS) + tuple(LONG_TUNING_ATTACKS_BY_SCOPE[scope])


def _build_validation_attacker(
    attack_id: str,
    *,
    actor: Actor,
    device: torch.device,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    critic: Critic | None,
    state_scope: str,
    seed: int,
):
    token = str(attack_id).strip().lower()
    if token in SHORT_TUNING_ATTACK_ALGORITHMS:
        return build_state_attacker(
            actor,
            device=device,
            algorithm=token,
            epsilon=0.10,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=critic if token == 'q_function' else None,
            attack_state_scope=state_scope,
        )
    return build_long_horizon_attacker(
        token,
        actor=actor,
        device=device,
        obs_low=obs_low,
        obs_high=obs_high,
        critic=critic,
        seed=seed,
    )


def _run_temporal_shield_clean_baselines(
    arrivals: pd.DataFrame,
    actor: Actor,
    signals_path,
    device: torch.device,
    reward_profile: RewardProfile,
    *,
    defender: nn.Module,
    detector_model: PosteriorBenefitMLPDetector,
    detector_threshold: float,
    state_scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    clean_reward = rollout_episode_with_dae_det_temporal_shield(
        arrivals,
        actor,
        signals_path,
        device,
        reward_profile,
        attack_enabled=False,
        defender=None,
        detector_model=None,
        route_mode='none',
        enable_shield=False,
        state_scope=state_scope,
    )
    clean_dae_det = rollout_episode_with_dae_det_temporal_shield(
        arrivals,
        actor,
        signals_path,
        device,
        reward_profile,
        attack_enabled=False,
        defender=defender,
        detector_model=detector_model,
        detector_threshold=float(detector_threshold),
        route_mode='detector',
        enable_shield=False,
        state_scope=state_scope,
    )
    return clean_reward, clean_dae_det


def _run_temporal_shield_attack_baselines(
    arrivals: pd.DataFrame,
    actor: Actor,
    signals_path,
    device: torch.device,
    reward_profile: RewardProfile,
    *,
    defender: nn.Module,
    detector_model: PosteriorBenefitMLPDetector,
    detector_threshold: float,
    state_scope: str,
    critic: Critic | None,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    seed: int,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for attack_id in _validation_attack_ids_for_scope(state_scope):
        attacker = _build_validation_attacker(
            attack_id,
            actor=actor,
            device=device,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=critic,
            state_scope=state_scope,
            seed=seed,
        )
        attack_none = rollout_episode_with_dae_det_temporal_shield(
            arrivals,
            actor,
            signals_path,
            device,
            reward_profile,
            attack_enabled=True,
            attack_scenario='O',
            attacker=_fresh_attacker(attacker),
            defender=None,
            detector_model=None,
            route_mode='none',
            enable_shield=False,
            state_scope=state_scope,
            attack_ratio=1.0,
            attack_scope='obs',
        )
        attack_dae_det = rollout_episode_with_dae_det_temporal_shield(
            arrivals,
            actor,
            signals_path,
            device,
            reward_profile,
            attack_enabled=True,
            attack_scenario='O',
            attacker=_fresh_attacker(attacker),
            defender=defender,
            detector_model=detector_model,
            detector_threshold=float(detector_threshold),
            route_mode='detector',
            enable_shield=False,
            state_scope=state_scope,
            attack_ratio=1.0,
            attack_scope='obs',
        )
        rows[str(attack_id)] = {
            'attack_id': str(attack_id),
            'attack_reward': float(attack_none['ep_reward']),
            'attack_dae_det_reward': float(attack_dae_det['ep_reward']),
            'route_rate_attack_dae_det': float(attack_dae_det.get('route_rate', 0.0)),
            'attack_dae_det_exit_vio': int(attack_dae_det.get('exit_vio', 0)),
            'attack_dae_det_run_vio': int(attack_dae_det.get('run_vio', 0)),
        }
    return rows


def _evaluate_temporal_shield_candidate(
    arrivals: pd.DataFrame,
    actor: Actor,
    signals_path,
    device: torch.device,
    reward_profile: RewardProfile,
    *,
    defender: nn.Module,
    detector_model: PosteriorBenefitMLPDetector,
    detector_threshold: float,
    state_scope: str,
    shield_config: LocalTemporalShieldConfig,
    clean_reward: dict[str, Any],
    clean_dae_det: dict[str, Any],
    attack_baselines: dict[str, dict[str, Any]],
    critic: Critic | None,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    seed: int,
    tau_soc_scale: float,
    tau_time_scale: float,
    tau_cost_scale: float,
) -> dict[str, Any]:
    clean_shield = rollout_episode_with_dae_det_temporal_shield(
        arrivals,
        actor,
        signals_path,
        device,
        reward_profile,
        attack_enabled=False,
        defender=defender,
        detector_model=detector_model,
        detector_threshold=float(detector_threshold),
        shield_config=shield_config,
        route_mode='detector',
        enable_shield=True,
        state_scope=state_scope,
    )
    attack_rows: list[dict[str, Any]] = []
    recovery_values: list[float] = []
    correction_values: list[float] = []
    for attack_id in _validation_attack_ids_for_scope(state_scope):
        baseline = dict(attack_baselines[str(attack_id)])
        attacker = _build_validation_attacker(
            attack_id,
            actor=actor,
            device=device,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=critic,
            state_scope=state_scope,
            seed=seed,
        )
        attack_shield = rollout_episode_with_dae_det_temporal_shield(
            arrivals,
            actor,
            signals_path,
            device,
            reward_profile,
            attack_enabled=True,
            attack_scenario='O',
            attacker=attacker,
            defender=defender,
            detector_model=detector_model,
            detector_threshold=float(detector_threshold),
            shield_config=shield_config,
            route_mode='detector',
            enable_shield=True,
            state_scope=state_scope,
            attack_ratio=1.0,
            attack_scope='obs',
        )
        recovery = _safe_recovery(float(clean_reward['ep_reward']), float(baseline['attack_reward']), float(attack_shield['ep_reward']))
        correction_mean = float(attack_shield.get('shield_correction_mean', 0.0))
        recovery_values.append(float(recovery))
        correction_values.append(float(correction_mean))
        attack_rows.append(
            {
                'attack_id': str(attack_id),
                'attack_reward': float(baseline['attack_reward']),
                'attack_dae_det_reward': float(baseline['attack_dae_det_reward']),
                'attack_shield_reward': float(attack_shield['ep_reward']),
                'recovery_shield': float(recovery),
                'shield_correction_mean_attack': float(correction_mean),
            }
        )
    clean_drop_shield = float(clean_reward['ep_reward'] - clean_shield['ep_reward'])
    return {
        'state_scope': str(state_scope),
        'tau_soc': float(shield_config.tau_soc),
        'tau_time': float(shield_config.tau_time),
        'tau_cost': float(shield_config.tau_cost),
        'tau_soc_scale': float(tau_soc_scale),
        'tau_time_scale': float(tau_time_scale),
        'tau_cost_scale': float(tau_cost_scale),
        'clean_reward': float(clean_reward['ep_reward']),
        'clean_dae_det_reward': float(clean_dae_det['ep_reward']),
        'clean_shield_reward': float(clean_shield['ep_reward']),
        'clean_drop_dae_det': float(clean_reward['ep_reward'] - clean_dae_det['ep_reward']),
        'clean_drop_shield': float(clean_drop_shield),
        'worst_case_recovery_shield': float(min(recovery_values)) if recovery_values else float('-inf'),
        'mean_recovery_shield': float(np.mean(recovery_values)) if recovery_values else float('-inf'),
        'mean_shield_correction_mean_attack': float(np.mean(correction_values)) if correction_values else 0.0,
        'route_rate_clean': float(clean_shield.get('route_rate', 0.0)),
        'attack_metrics': attack_rows,
    }


def _tuning_candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(row.get('worst_case_recovery_shield', float('-inf'))),
        float(row.get('mean_recovery_shield', float('-inf'))),
        -float(row.get('clean_drop_shield', float('inf'))),
        -float(row.get('mean_shield_correction_mean_attack', float('inf'))),
    )


def _select_attack_tuned_candidate(
    candidate_rows: Sequence[dict[str, Any]],
    *,
    baseline_row: dict[str, Any],
    clean_drop_cap: float,
) -> tuple[dict[str, Any], str, bool]:
    feasible_rows = [dict(row) for row in candidate_rows if float(row.get('clean_drop_shield', float('inf'))) <= float(clean_drop_cap)]
    if not feasible_rows:
        return dict(baseline_row), 'no_feasible_candidate', False
    best_feasible = max(feasible_rows, key=_tuning_candidate_sort_key)
    baseline_worst_case = float(baseline_row.get('worst_case_recovery_shield', float('-inf')))
    best_worst_case = float(best_feasible.get('worst_case_recovery_shield', float('-inf')))
    if best_worst_case <= baseline_worst_case + 1e-8:
        return dict(baseline_row), 'no_acceptable_improvement', False
    return dict(best_feasible), 'improved', True


def tune_temporal_shield_with_attacks(
    arrivals: pd.DataFrame,
    actor: Actor,
    signals_path,
    device: torch.device,
    *,
    defender: nn.Module,
    detector_model: PosteriorBenefitMLPDetector,
    detector_threshold: float,
    base_config: LocalTemporalShieldConfig,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    clean_drop_limit: float = 50.0,
    tau_soc_scales: Sequence[float] = (0.75, 1.0, 1.25),
    tau_time_scales: Sequence[float] = (1.0, 1.5),
    tau_cost_scales: Sequence[float] = (0.75, 1.0, 1.25),
    critic: Critic | None = None,
    seed: int = 42,
) -> tuple[LocalTemporalShieldConfig, dict[str, Any]]:
    state_scope = _canonical_scope(base_config.state_scope)
    clean_reward, clean_dae_det = _run_temporal_shield_clean_baselines(
        arrivals,
        actor,
        signals_path,
        device,
        reward_profile,
        defender=defender,
        detector_model=detector_model,
        detector_threshold=float(detector_threshold),
        state_scope=state_scope,
    )
    obs_low, obs_high = observation_bounds_for_arrivals(arrivals, signals_path, reward_profile)
    attack_baselines = _run_temporal_shield_attack_baselines(
        arrivals,
        actor,
        signals_path,
        device,
        reward_profile,
        defender=defender,
        detector_model=detector_model,
        detector_threshold=float(detector_threshold),
        state_scope=state_scope,
        critic=critic,
        obs_low=obs_low,
        obs_high=obs_high,
        seed=int(seed),
    )
    base_candidate_config = LocalTemporalShieldConfig(
        state_scope=state_scope,
        tau_soc=float(base_config.tau_soc),
        tau_time=float(base_config.tau_time),
        tau_cost=float(base_config.tau_cost),
        calibration_quantile=float(base_config.calibration_quantile),
        min_tau_soc=float(base_config.min_tau_soc),
        min_tau_time=float(base_config.min_tau_time),
        min_tau_cost=float(base_config.min_tau_cost),
        max_tau_soc=float(base_config.max_tau_soc),
        max_tau_time=float(base_config.max_tau_time),
        max_tau_cost=float(base_config.max_tau_cost),
        initial_soc=float(base_config.initial_soc),
        initial_cost_norm=float(base_config.initial_cost_norm),
    )
    baseline_row = _evaluate_temporal_shield_candidate(
        arrivals,
        actor,
        signals_path,
        device,
        reward_profile,
        defender=defender,
        detector_model=detector_model,
        detector_threshold=float(detector_threshold),
        state_scope=state_scope,
        shield_config=base_candidate_config,
        clean_reward=clean_reward,
        clean_dae_det=clean_dae_det,
        attack_baselines=attack_baselines,
        critic=critic,
        obs_low=obs_low,
        obs_high=obs_high,
        seed=int(seed),
        tau_soc_scale=1.0,
        tau_time_scale=1.0,
        tau_cost_scale=1.0,
    )
    clean_drop_cap = float(min(float(clean_drop_limit), float(baseline_row['clean_drop_dae_det']) + 2.0))
    baseline_row['feasible'] = bool(float(baseline_row['clean_drop_shield']) <= clean_drop_cap)
    baseline_row['clean_drop_cap'] = float(clean_drop_cap)
    candidate_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[float, float, float, float]] = set()
    for tau_soc_scale in tau_soc_scales:
        for tau_time_scale in tau_time_scales:
            for tau_cost_scale in tau_cost_scales:
                tau_soc = float(np.clip(float(base_config.tau_soc) * float(tau_soc_scale), float(base_config.min_tau_soc), float(base_config.max_tau_soc)))
                tau_time = float(np.clip(float(base_config.tau_time) * float(tau_time_scale), float(base_config.min_tau_time), float(base_config.max_tau_time)))
                tau_cost = float(np.clip(float(base_config.tau_cost) * float(tau_cost_scale), float(base_config.min_tau_cost), float(base_config.max_tau_cost)))
                candidate_key = (
                    round(tau_soc, 10),
                    round(tau_time, 10),
                    round(tau_cost, 10),
                )
                if candidate_key in seen_keys:
                    continue
                seen_keys.add(candidate_key)
                candidate_config = LocalTemporalShieldConfig(
                    state_scope=state_scope,
                    tau_soc=float(tau_soc),
                    tau_time=float(tau_time),
                    tau_cost=float(tau_cost),
                    calibration_quantile=float(base_config.calibration_quantile),
                    min_tau_soc=float(base_config.min_tau_soc),
                    min_tau_time=float(base_config.min_tau_time),
                    min_tau_cost=float(base_config.min_tau_cost),
                    max_tau_soc=float(base_config.max_tau_soc),
                    max_tau_time=float(base_config.max_tau_time),
                    max_tau_cost=float(base_config.max_tau_cost),
                    initial_soc=float(base_config.initial_soc),
                    initial_cost_norm=float(base_config.initial_cost_norm),
                )
                row = _evaluate_temporal_shield_candidate(
                    arrivals,
                    actor,
                    signals_path,
                    device,
                    reward_profile,
                    defender=defender,
                    detector_model=detector_model,
                    detector_threshold=float(detector_threshold),
                    state_scope=state_scope,
                    shield_config=candidate_config,
                    clean_reward=clean_reward,
                    clean_dae_det=clean_dae_det,
                    attack_baselines=attack_baselines,
                    critic=critic,
                    obs_low=obs_low,
                    obs_high=obs_high,
                    seed=int(seed),
                    tau_soc_scale=float(tau_soc_scale),
                    tau_time_scale=float(tau_time_scale),
                    tau_cost_scale=float(tau_cost_scale),
                )
                row['feasible'] = bool(float(row['clean_drop_shield']) <= clean_drop_cap)
                row['clean_drop_cap'] = float(clean_drop_cap)
                candidate_rows.append(row)
    selected_row, selection_status, improvement_flag = _select_attack_tuned_candidate(
        candidate_rows,
        baseline_row=baseline_row,
        clean_drop_cap=clean_drop_cap,
    )
    selected_config = LocalTemporalShieldConfig(
        state_scope=state_scope,
        tau_soc=float(selected_row['tau_soc']),
        tau_time=float(selected_row['tau_time']),
        tau_cost=float(selected_row['tau_cost']),
        calibration_quantile=float(base_config.calibration_quantile),
        min_tau_soc=float(base_config.min_tau_soc),
        min_tau_time=float(base_config.min_tau_time),
        min_tau_cost=float(base_config.min_tau_cost),
        max_tau_soc=float(base_config.max_tau_soc),
        max_tau_time=float(base_config.max_tau_time),
        max_tau_cost=float(base_config.max_tau_cost),
        initial_soc=float(base_config.initial_soc),
        initial_cost_norm=float(base_config.initial_cost_norm),
    )
    summary = {
        'state_scope': state_scope,
        'short_attack_algorithms': list(SHORT_TUNING_ATTACK_ALGORITHMS),
        'long_attack_names': list(LONG_TUNING_ATTACKS_BY_SCOPE[state_scope]),
        'attack_ids': list(_validation_attack_ids_for_scope(state_scope)),
        'tau_soc_scales': [float(v) for v in tau_soc_scales],
        'tau_time_scales': [float(v) for v in tau_time_scales],
        'tau_cost_scales': [float(v) for v in tau_cost_scales],
        'clean_drop_limit': float(clean_drop_limit),
        'effective_clean_drop_cap': float(clean_drop_cap),
        'baseline_row': dict(baseline_row),
        'selected_row': dict(selected_row),
        'candidate_rows': [dict(row) for row in candidate_rows],
        'selection_status': str(selection_status),
        'improvement_flag': bool(improvement_flag),
    }
    return selected_config, summary


def eval_temporal_shield_suite(
    arrivals: pd.DataFrame,
    signals_path,
    actor: Actor,
    defender: nn.Module,
    detector_model: PosteriorBenefitMLPDetector,
    detector_threshold: float,
    shield_config: LocalTemporalShieldConfig,
    device: torch.device,
    *,
    reward_profile: RewardProfile = TRAIN_PROFILE,
    eval_algorithms: Sequence[str] = ('opposite_pgd', 'q_function', 'learned_sequence'),
    state_scope: str = 'local',
    epsilon_q_pgd: float = 0.1,
    epsilon_learned: float = 0.15,
    attack_scenario: str = 'O',
    attack_scope: AttackScope = 'obs',
    attack_ratio: float = 1.0,
    alpha: float | None = None,
    iters: int | None = None,
    critic: Critic | None = None,
    learned_adv_iters: int = 200,
    learned_adv_phase_steps: int = 2048,
    learned_adv_ppo_epochs: int = 10,
    learned_adv_num_minibatches: int = 32,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    adv_actor_lr: float = 1e-3,
    adv_critic_lr: float = 1e-5,
    adv_entropy_coeff: float = 1e-4,
    adversary_hidden_dim: int = 128,
    seed: int = 42,
    print_every: int = 10,
) -> pd.DataFrame:
    del learned_adv_iters, learned_adv_phase_steps, learned_adv_ppo_epochs, learned_adv_num_minibatches
    del gamma, gae_lambda, clip_eps, adv_actor_lr, adv_critic_lr, adv_entropy_coeff, adversary_hidden_dim, print_every
    state_scope = _canonical_scope(state_scope)
    obs_low, obs_high = observation_bounds_for_arrivals(arrivals, signals_path, reward_profile)
    clean_reward = rollout_episode_with_dae_det_temporal_shield(
        arrivals,
        actor,
        signals_path,
        device,
        reward_profile,
        attack_enabled=False,
        defender=None,
        detector_model=None,
        route_mode='none',
        enable_shield=False,
    )
    clean_dae_det = rollout_episode_with_dae_det_temporal_shield(
        arrivals,
        actor,
        signals_path,
        device,
        reward_profile,
        attack_enabled=False,
        defender=defender,
        detector_model=detector_model,
        detector_threshold=float(detector_threshold),
        route_mode='detector',
        enable_shield=False,
    )
    clean_shield = rollout_episode_with_dae_det_temporal_shield(
        arrivals,
        actor,
        signals_path,
        device,
        reward_profile,
        attack_enabled=False,
        defender=defender,
        detector_model=detector_model,
        detector_threshold=float(detector_threshold),
        shield_config=shield_config,
        route_mode='detector',
        enable_shield=True,
        state_scope=state_scope,
    )
    clean_metric_columns = shield_metric_columns(clean_shield, prefix='clean')
    clean_attack_metric_columns = attack_metric_columns(clean_shield, prefix='clean')
    rows: list[dict[str, Any]] = []
    for algorithm in eval_algorithms:
        algo = str(algorithm).strip().lower()
        if algo == 'learned_sequence':
            row = {
                'algorithm': algo,
                'epsilon': float(epsilon_learned),
                'state_scope': state_scope,
                'clean_reward': float(clean_reward['ep_reward']),
                'clean_dae_det_reward': float(clean_dae_det['ep_reward']),
                'clean_shield_reward': float(clean_shield['ep_reward']),
                'attack_reward': np.nan,
                'attack_dae_det_reward': np.nan,
                'attack_dae_det_shield_reward': np.nan,
                'recovery_dae_det': np.nan,
                'recovery_shield': np.nan,
                'clean_drop_dae_det': float(clean_reward['ep_reward'] - clean_dae_det['ep_reward']),
                'clean_drop_shield': float(clean_reward['ep_reward'] - clean_shield['ep_reward']),
                'route_rate_clean': float(clean_shield.get('route_rate', 0.0)),
                'route_rate_attack_dae_det': np.nan,
                'route_rate_attack_shield': np.nan,
                'shield_soc_clamp_rate_attack': np.nan,
                'shield_time_clamp_rate_attack': np.nan,
                'shield_cost_clamp_rate_attack': np.nan,
                'attack_exit_vio': np.nan,
                'attack_dae_det_exit_vio': np.nan,
                'attack_shield_exit_vio': np.nan,
                'attack_run_vio': np.nan,
                'attack_dae_det_run_vio': np.nan,
                'attack_shield_run_vio': np.nan,
                'tau_soc': float(shield_config.tau_soc),
                'tau_time': float(shield_config.tau_time),
                'tau_cost': float(shield_config.tau_cost),
                'unsupported_reason': 'learned_sequence adversary restoration is pending in the GRU-VAE shield line',
            }
            row.update(clean_metric_columns)
            row.update(clean_attack_metric_columns)
            row.update(_empty_shield_metric_columns(prefix='attack'))
            row.update(_empty_attack_metric_columns(prefix='attack'))
            row.update(_empty_attack_metric_columns(prefix='attack_dae_det'))
            row.update(_empty_attack_metric_columns(prefix='attack_shield'))
            rows.append(row)
            continue
        attack_epsilon = float(epsilon_q_pgd)
        attacker = build_state_attacker(
            actor,
            device=device,
            algorithm=algo,
            epsilon=attack_epsilon,
            alpha=alpha,
            iters=iters,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=critic if algo == 'q_function' else None,
            attack_state_scope=state_scope,
            signals_path=signals_path,
            reward_profile=reward_profile,
        )
        attack_none_attacker = attacker.clone() if hasattr(attacker, 'clone') else attacker
        attack_dae_det_attacker = attacker.clone() if hasattr(attacker, 'clone') else attacker
        attack_shield_attacker = attacker.clone() if hasattr(attacker, 'clone') else attacker
        for attacker_copy in (attack_none_attacker, attack_dae_det_attacker, attack_shield_attacker):
            if hasattr(attacker_copy, 'reset'):
                attacker_copy.reset()
        attack_none = rollout_episode_with_dae_det_temporal_shield(
            arrivals,
            actor,
            signals_path,
            device,
            reward_profile,
            attack_enabled=True,
            attack_scenario=attack_scenario,
            attacker=attack_none_attacker,
            defender=None,
            detector_model=None,
            route_mode='none',
            enable_shield=False,
            state_scope=state_scope,
            attack_ratio=attack_ratio,
            attack_scope=attack_scope,
        )
        attack_dae_det = rollout_episode_with_dae_det_temporal_shield(
            arrivals,
            actor,
            signals_path,
            device,
            reward_profile,
            attack_enabled=True,
            attack_scenario=attack_scenario,
            attacker=attack_dae_det_attacker,
            defender=defender,
            detector_model=detector_model,
            detector_threshold=float(detector_threshold),
            route_mode='detector',
            enable_shield=False,
            state_scope=state_scope,
            attack_ratio=attack_ratio,
            attack_scope=attack_scope,
        )
        attack_shield = rollout_episode_with_dae_det_temporal_shield(
            arrivals,
            actor,
            signals_path,
            device,
            reward_profile,
            attack_enabled=True,
            attack_scenario=attack_scenario,
            attacker=attack_shield_attacker,
            defender=defender,
            detector_model=detector_model,
            detector_threshold=float(detector_threshold),
            shield_config=shield_config,
            route_mode='detector',
            enable_shield=True,
            epsilon=attack_epsilon,
            state_scope=state_scope,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_ratio=attack_ratio,
            attack_scope=attack_scope,
        )
        row = {
            'algorithm': algo,
            'epsilon': attack_epsilon,
            'state_scope': state_scope,
            'clean_reward': float(clean_reward['ep_reward']),
            'clean_dae_det_reward': float(clean_dae_det['ep_reward']),
            'clean_shield_reward': float(clean_shield['ep_reward']),
            'attack_reward': float(attack_none['ep_reward']),
            'attack_dae_det_reward': float(attack_dae_det['ep_reward']),
            'attack_dae_det_shield_reward': float(attack_shield['ep_reward']),
            'recovery_dae_det': _safe_recovery(float(clean_reward['ep_reward']), float(attack_none['ep_reward']), float(attack_dae_det['ep_reward'])),
            'recovery_shield': _safe_recovery(float(clean_reward['ep_reward']), float(attack_none['ep_reward']), float(attack_shield['ep_reward'])),
            'clean_drop_dae_det': float(clean_reward['ep_reward'] - clean_dae_det['ep_reward']),
            'clean_drop_shield': float(clean_reward['ep_reward'] - clean_shield['ep_reward']),
            'route_rate_clean': float(clean_shield.get('route_rate', 0.0)),
            'route_rate_attack_dae_det': float(attack_dae_det.get('route_rate', 0.0)),
            'route_rate_attack_shield': float(attack_shield.get('route_rate', 0.0)),
            'shield_soc_clamp_rate_attack': float(attack_shield.get('shield_soc_clamp_rate', 0.0)),
            'shield_time_clamp_rate_attack': float(attack_shield.get('shield_time_clamp_rate', 0.0)),
            'shield_cost_clamp_rate_attack': float(attack_shield.get('shield_cost_clamp_rate', 0.0)),
            'attack_exit_vio': int(attack_none.get('exit_vio', 0)),
            'attack_dae_det_exit_vio': int(attack_dae_det.get('exit_vio', 0)),
            'attack_shield_exit_vio': int(attack_shield.get('exit_vio', 0)),
            'attack_run_vio': int(attack_none.get('run_vio', 0)),
            'attack_dae_det_run_vio': int(attack_dae_det.get('run_vio', 0)),
            'attack_shield_run_vio': int(attack_shield.get('run_vio', 0)),
            'tau_soc': float(shield_config.tau_soc),
            'tau_time': float(shield_config.tau_time),
            'tau_cost': float(shield_config.tau_cost),
        }
        row.update(clean_metric_columns)
        row.update(clean_attack_metric_columns)
        row.update(attack_metric_columns(attack_none, prefix='attack'))
        row.update(attack_metric_columns(attack_dae_det, prefix='attack_dae_det'))
        row.update(attack_metric_columns(attack_shield, prefix='attack_shield'))
        row.update(shield_metric_columns(attack_shield, prefix='attack'))
        rows.append(row)
    return normalize_result_frame(pd.DataFrame(rows), rename_keys=False, digits=6)


__all__ = [
    'LOCAL_SHIELD_INDICES',
    'ALL_SHIELD_INDICES',
    'LocalTemporalShieldConfig',
    'TemporalShieldArtifact',
    'calibrate_local_temporal_shield',
    'save_temporal_shield_bundle',
    'load_temporal_shield_bundle',
    'shield_metric_columns',
    'attack_metric_columns',
    'rollout_episode_with_dae_det_temporal_shield',
    'tune_temporal_shield_with_attacks',
    'collect_adversary_rollouts_for_dae_det_temporal_shield',
    'train_learned_sequence_adversary_for_dae_det_temporal_shield',
    'eval_temporal_shield_suite',
]
