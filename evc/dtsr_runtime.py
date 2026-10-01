from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .defense import PosteriorBenefitMLPDetector, SequentialDAERuntime
from .attacks import AttackScope, PGDStateAttacker, attack_batch_by_context
from .merged_core import (
    Actor,
    ChargingEnv,
    QueueItem,
    RewardProfile,
    to_numpy_1d,
)
from .merged_pipeline import _build_contexts, _rollout_label, _route_policy_states, summarize_metrics
from .offline_dae_det_temporal_shield import (
    LOCAL_SHIELD_INDICES,
    LocalTemporalShieldConfig,
    _route_policy_states_core_only,
    _shield_single_state,
)
from .rollout_utils import update_active_vehicle_ids

def rollout_episode_with_dtsr(
    arrivals: pd.DataFrame,
    actor: Actor,
    signals_path,
    device: torch.device,
    reward_profile: RewardProfile,
    *,
    attack_enabled: bool = False,
    attack_scenario: str = 'O',
    attacker: PGDStateAttacker | None = None,
    defender: nn.Module | None = None,
    detector_model: PosteriorBenefitMLPDetector | None = None,
    detector_threshold: float | None = None,
    shield_config: LocalTemporalShieldConfig | None = None,
    route_mode: str = 'none',
    enable_shield: bool = False,
    post_shield_processor: Any | None = None,
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
    """Evaluate the DTSR routing, Shield, and optional post-Shield corrector."""
    if post_shield_processor is not None and (not enable_shield or shield_config is None):
        raise ValueError('post_shield_processor requires an enabled Temporal Shield.')
    env = ChargingEnv(signals_path=signals_path, reward_profile=reward_profile)
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

    def _apply_shield_batch(policy_states: list[np.ndarray], vehicle_ids: list[int], is_new_arrivals: list[int]) -> list[np.ndarray]:
        nonlocal soc_clamp, time_clamp, cost_clamp, shield_process_total
        if not enable_shield or shield_config is None:
            return [to_numpy_1d(s) for s in policy_states]
        route_states = [to_numpy_1d(s) for s in policy_states]
        out: list[np.ndarray] = []
        for idx_local, (policy_state, vehicle_id, new_flag) in enumerate(zip(route_states, vehicle_ids, is_new_arrivals)):
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
            prev_policy_obs_by_vehicle[int(vehicle_ids[idx_local])] = to_numpy_1d(chosen)
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
        if post_shield_processor is not None:
            post_shield_processor.update_actions(vehicle_ids, actions, current_time)

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

    def run_defense_runtime(
        observed_states: Sequence[np.ndarray],
        attacked_flags: Sequence[bool],
        stations: Sequence[int],
        vehicle_ids: Sequence[int],
        is_new_flags: Sequence[int],
    ) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
        """Run the defense with no clean or pre-attack state in its interface."""
        nonlocal route_count, route_total, attack_obs_count
        observed_states = [to_numpy_1d(x).astype(np.float32) for x in observed_states]
        stations = [int(x) for x in stations]
        vehicle_ids = [int(x) for x in vehicle_ids]
        is_new_flags = [int(x) for x in is_new_flags]
        attacked_flags = [bool(x) for x in attacked_flags]
        prev_refs = _lookup_prev_observed(vehicle_ids, observed_states)
        route_fn = _route_policy_states_core_only if repair_mode == 'core_only' else _route_policy_states
        policy_states, route_flags, det_scores = route_fn(
            observed_states,
            attacked_flags,
            defender,
            detector_model,
            actor,
            device,
            route_mode=route_mode,
            detector_threshold=detector_threshold,
            detector_feature_mode='posterior',
            time_indices=[env.t for _ in observed_states],
            stations=stations,
            is_new_arrivals=is_new_flags,
            prev_obs_refs=prev_refs,
            vehicle_ids=vehicle_ids,
            episode_index=0,
            dae_runtime=dae_runtime,
        )
        routed_states = [to_numpy_1d(s).astype(np.float32) for s in policy_states]
        shielded_states = _apply_shield_batch(routed_states, vehicle_ids, is_new_flags)
        processor_diagnostics: dict[str, Any] = {}
        if post_shield_processor is not None:
            policy_states, processor_diagnostics = post_shield_processor.process_batch(
                routed_states=routed_states,
                shielded_states=shielded_states,
                vehicle_ids=vehicle_ids,
                is_new_arrivals=is_new_flags,
                env=env,
                detector_scores=det_scores,
                route_flags=route_flags,
            )
        else:
            policy_states = shielded_states
        route_count += int(sum(route_flags))
        route_total += len(route_flags)
        attack_obs_count += int(sum(attacked_flags))
        actions = _compute_actions(policy_states)
        _update_prev_actions(vehicle_ids, actions, int(env.t))
        _update_prev_observed(vehicle_ids, observed_states)
        diagnostics = {
            'observed_states': observed_states,
            'routed_states': routed_states,
            'shielded_states': shielded_states,
            'selected_states': policy_states,
            'route_flags': route_flags,
            'det_scores': det_scores,
            'processor': processor_diagnostics,
        }
        return policy_states, actions, diagnostics

    def _process_batch(clean_states, stations, vehicle_ids, is_new_flags):
        contexts = _build_contexts(
            env,
            clean_states,
            stations,
            attack_scenario,
            bool(is_new_flags[0]) if is_new_flags else False,
            price_threshold,
            soc_new_threshold,
            soc_rollout_threshold,
            even_station_target,
            odd_station_target,
        )
        attacked_states, attacked_flags = attack_batch_by_context(
            attacker if attack_enabled else None,
            clean_states,
            contexts,
            attack_ratio=attack_ratio,
            attack_scope=attack_scope,
            vehicle_ids=vehicle_ids,
            episode_index=0,
            seed=42 if attacker is None else int(getattr(attacker, 'seed', 42)),
        )
        _record_attack_delta_stats(clean_states, attacked_states, attacked_flags)
        observed_states = attacked_states if attack_enabled else [to_numpy_1d(x) for x in clean_states]
        policy_states, actions, diagnostics = run_defense_runtime(
            observed_states,
            attacked_flags,
            stations,
            vehicle_ids,
            is_new_flags,
        )
        return policy_states, actions

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
            policy_states, actions = _process_batch(new_states, new_stations, new_vehicle_ids, [1 for _ in new_states])
            for clean_obs, action, station in zip(new_states, actions, new_stations):
                env.enqueue(clean_obs, action, station)

        if active:
            active_states = [item.obs for item in active]
            active_stations = [item.station for item in active]
            policy_states, actions = _process_batch(active_states, active_stations, active_vehicle_ids, [0 for _ in active_states])
            for item, action in zip(active, actions):
                env.enqueue(item.obs, action, item.station)

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
    if post_shield_processor is not None:
        summary.update(post_shield_processor.summary())
        summary['post_shield_processor_name'] = str(
            getattr(post_shield_processor, 'processor_name', type(post_shield_processor).__name__)
        )
        summary['runtime_pipeline_order'] = str(
            getattr(
                post_shield_processor,
                'runtime_pipeline_order',
                'DAE/DET route -> Temporal Shield -> Corrector -> Actor',
            )
        )
    summary['pd_bcr_enabled'] = bool(
        post_shield_processor is not None
        and getattr(post_shield_processor, 'is_pd_bcr', False)
    )
    return summary






__all__ = [
    'rollout_episode_with_dtsr',
]

