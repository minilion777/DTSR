from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .defense import PosteriorBenefitMLPDetector, SequentialDAERuntime
from .merged_attacks import AttackScope, PGDStateAttacker, attack_batch_by_context, build_state_attacker
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
from .merged_pipeline import _build_contexts, _rollout_label, _route_policy_states, summarize_metrics
from .offline_dae_det_temporal_shield import (
    LOCAL_SHIELD_INDICES,
    LocalTemporalShieldConfig,
    _route_policy_states_core_only,
    _shield_single_state,
)
from .sequential_adversary import observation_bounds_for_arrivals, update_active_vehicle_ids

_TIME_DECAY = 1.0 / 12.0

@dataclass
class BeliefCoreConfig:
    """No-leakage belief-guided core repair configuration.

    The belief state uses only defended/attacked observations, past executed actions, and known
    charging dynamics. It never reads clean SOC, true remaining time, or future violation labels.
    """

    enabled: bool = True
    pred_weight: float = 0.55
    obs_weight: float = 0.28
    detector_gain: float = 0.25
    max_pred_weight: float = 0.82
    soc_margin: float = 0.010
    time_margin: float = 0.000
    cost_margin: float = 0.015
    disagreement_gain: float = 0.6
    uncertainty_decay: float = 0.80
    use_known_initial_soc: bool = True
    use_known_initial_cost: bool = True
    time_initialization: str = 'routed_observation'

@dataclass
class UrgencyGateConfig:
    """No-leakage urgency-gated belief activation.

    This gate does not try to classify the attack name. It keeps Temporal Shield
    as the default branch and enables belief repair only when the belief estimate
    implies a more urgent departure-SOC requirement than the DAE/shield input,
    while remaining sufficiently confident and not showing strong one-step
    temporal inconsistency.
    """

    enabled: bool = False
    target_soc_margin: float = 0.000
    urgency_gain_threshold: float = 0.010
    soc_drop_threshold: float = 0.025
    time_drop_threshold: float = 0.013
    uncertainty_threshold: float = 0.065
    temporal_residual_threshold: float = 0.022
    ema_decay: float = 0.80
    innovation_ema_decay: float = 0.80
    time_innovation_threshold: float = 0.006
    soc_innovation_threshold: float = 0.010
    consecutive_steps: int = 2
    min_remaining_steps: float = 1.0
    max_remaining_steps: float = 18.0



@dataclass
class UGBCRConfig:
    """Configuration for Urgency-Gated Belief-Guided Core Repair.

    This final module intentionally keeps only the useful belief estimator and
    urgency gate. It does not include the earlier adaptive attack-type router or
    action recovery shield, which were not selected as final defenses.
    """

    schema_version: int = 2
    leakage_policy: str = 'strict_no_clean_state'
    time_initialization: str = 'routed_observation'
    uses_clean_state: bool = False
    uses_true_remaining_time: bool = False
    belief: BeliefCoreConfig = field(default_factory=BeliefCoreConfig)
    urgency_gate: UrgencyGateConfig = field(default_factory=UrgencyGateConfig)

    def __post_init__(self) -> None:
        if int(self.schema_version) < 2:
            raise ValueError('UG-BCR-v2 requires schema_version >= 2.')
        if str(self.leakage_policy) != 'strict_no_clean_state':
            raise ValueError('UG-BCR-v2 requires leakage_policy="strict_no_clean_state".')
        if bool(self.uses_clean_state) or bool(self.uses_true_remaining_time):
            raise ValueError('UG-BCR-v2 cannot use clean state or true remaining time.')
        if str(self.time_initialization) != 'routed_observation':
            raise ValueError('UG-BCR-v2 remaining time must initialize from routed_observation.')
        if str(self.belief.time_initialization) != 'routed_observation':
            raise ValueError('BeliefCoreConfig.time_initialization must be routed_observation.')


def load_ug_bcr_config(path: str | Path) -> UGBCRConfig:
    """Load and validate a strict no-leakage UG-BCR-v2 JSON artifact."""

    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'UG-BCR configuration must be a JSON object: {path}')
    belief_payload = dict(payload.get('belief') or {})
    gate_payload = dict(payload.get('urgency_gate') or {})
    return UGBCRConfig(
        schema_version=int(payload.get('schema_version', 0)),
        leakage_policy=str(payload.get('leakage_policy', '')),
        time_initialization=str(payload.get('time_initialization', '')),
        uses_clean_state=bool(payload.get('uses_clean_state', False)),
        uses_true_remaining_time=bool(payload.get('uses_true_remaining_time', False)),
        belief=BeliefCoreConfig(**belief_payload),
        urgency_gate=UrgencyGateConfig(**gate_payload),
    )


class UrgencyGatedBeliefSelector:
    """Selective belief activation based on deadline urgency.

    Default output is the DAE/Temporal-Shield input state. The belief candidate is
    used only when it makes the departure deadline more urgent (lower SOC and/or
    shorter remaining time), the belief uncertainty is not high, and the current
    one-step physical residual is not strong. This avoids using belief as a
    universal smoother for Q-drift attacks.
    """

    def __init__(self, config: UrgencyGateConfig | None = None) -> None:
        self.config = config or UrgencyGateConfig()
        self.total = 0
        self.belief_count = 0
        self.shield_count = 0
        self.urgency_policy_sum = 0.0
        self.urgency_belief_sum = 0.0
        self.urgency_gain_sum = 0.0
        self.soc_drop_sum = 0.0
        self.time_drop_sum = 0.0
        self.temporal_residual_sum = 0.0
        self.uncertainty_sum = 0.0
        self.gain_ema_by_vehicle: dict[int, float] = {}
        self.time_innovation_ema_by_vehicle: dict[int, float] = {}
        self.soc_innovation_ema_by_vehicle: dict[int, float] = {}
        self.streak_by_vehicle: dict[int, int] = {}

    def reset(self) -> None:
        self.__init__(self.config)

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    def _urgency(self, state: np.ndarray, env: ChargingEnv, reward_profile: RewardProfile) -> float:
        vec = to_numpy_1d(state)
        soc = float(vec[0])
        rem_steps = max(float(vec[1]) / _TIME_DECAY, 1.0)
        soc_step = max(float(env.max_power * env.slice_hours / env.battery_capacity), 1e-6)
        target = float(reward_profile.exit_target_min + self.config.target_soc_margin)
        need = max(0.0, target - soc)
        required_fraction = need / max(rem_steps * soc_step, 1e-6)
        return self._clip01(required_fraction)

    def _temporal_residual(
        self,
        policy_state: np.ndarray,
        vehicle_id: int,
        is_new_arrival: int,
        shield_config: LocalTemporalShieldConfig | None,
        env: ChargingEnv,
        prev_policy_obs_by_vehicle: dict[int, np.ndarray],
        prev_action_by_vehicle: dict[int, np.ndarray],
        prev_time_by_vehicle: dict[int, int],
    ) -> float:
        if shield_config is None:
            return 0.0
        vid = int(vehicle_id)
        corrected, _ = _shield_single_state(
            to_numpy_1d(policy_state),
            prev_policy_obs_by_vehicle.get(vid),
            prev_action_by_vehicle.get(vid),
            prev_time_by_vehicle.get(vid),
            shield_config,
            env,
            is_new_arrival=bool(is_new_arrival),
        )
        diff = np.abs(to_numpy_1d(corrected)[list(LOCAL_SHIELD_INDICES)] - to_numpy_1d(policy_state)[list(LOCAL_SHIELD_INDICES)])
        return float(np.max(diff))

    def select_batch(
        self,
        policy_states: Sequence[np.ndarray],
        belief_states: Sequence[np.ndarray],
        vehicle_ids: Sequence[int],
        is_new_arrivals: Sequence[int],
        belief_estimator: 'BeliefCoreEstimator',
        shield_config: LocalTemporalShieldConfig | None,
        env: ChargingEnv,
        reward_profile: RewardProfile,
        prev_policy_obs_by_vehicle: dict[int, np.ndarray],
        prev_action_by_vehicle: dict[int, np.ndarray],
        prev_time_by_vehicle: dict[int, int],
        detector_scores: Sequence[float] | None = None,
        route_flags: Sequence[bool] | None = None,
    ) -> tuple[list[np.ndarray], list[str]]:
        selected: list[np.ndarray] = []
        branches: list[str] = []
        for policy_state, belief_state, vehicle_id, new_flag in zip(policy_states, belief_states, vehicle_ids, is_new_arrivals):
            vid = int(vehicle_id)
            policy_vec = to_numpy_1d(policy_state).astype(np.float32)
            belief_vec = to_numpy_1d(belief_state).astype(np.float32)
            urgency_policy = self._urgency(policy_vec, env, reward_profile)
            urgency_belief = self._urgency(belief_vec, env, reward_profile)
            raw_gain = float(urgency_belief - urgency_policy)
            old_gain = float(self.gain_ema_by_vehicle.get(vid, raw_gain))
            gain_ema = float(self.config.ema_decay * old_gain + (1.0 - self.config.ema_decay) * raw_gain)
            self.gain_ema_by_vehicle[vid] = gain_ema
            soc_drop = float(policy_vec[0] - belief_vec[0])
            time_drop = float(policy_vec[1] - belief_vec[1])
            rem_steps = float(max(policy_vec[1] / _TIME_DECAY, 0.0))
            uncertainty = float(belief_estimator.uncertainty(vid))
            innovation = belief_estimator.innovation(vid)
            positive_soc_innovation = max(float(innovation[0]), 0.0)
            positive_time_innovation = max(float(innovation[1]), 0.0)
            innovation_decay = float(np.clip(self.config.innovation_ema_decay, 0.0, 1.0))
            old_soc_ema = float(self.soc_innovation_ema_by_vehicle.get(vid, 0.0))
            old_time_ema = float(self.time_innovation_ema_by_vehicle.get(vid, 0.0))
            soc_ema = innovation_decay * old_soc_ema + (1.0 - innovation_decay) * positive_soc_innovation
            time_ema = innovation_decay * old_time_ema + (1.0 - innovation_decay) * positive_time_innovation
            self.soc_innovation_ema_by_vehicle[vid] = float(soc_ema)
            self.time_innovation_ema_by_vehicle[vid] = float(time_ema)
            persistent_drift = (
                time_ema > float(self.config.time_innovation_threshold)
                or soc_ema > float(self.config.soc_innovation_threshold)
            )
            temporal = self._temporal_residual(
                policy_vec,
                vid,
                int(new_flag),
                shield_config,
                env,
                prev_policy_obs_by_vehicle,
                prev_action_by_vehicle,
                prev_time_by_vehicle,
            )
            candidate = (
                persistent_drift
                and gain_ema > float(self.config.urgency_gain_threshold)
                and (soc_drop > float(self.config.soc_drop_threshold) or time_drop > float(self.config.time_drop_threshold))
                and uncertainty < float(self.config.uncertainty_threshold)
                and temporal < float(self.config.temporal_residual_threshold)
                and rem_steps >= float(self.config.min_remaining_steps)
                and rem_steps <= float(self.config.max_remaining_steps)
            )
            if candidate:
                self.streak_by_vehicle[vid] = int(self.streak_by_vehicle.get(vid, 0)) + 1
            else:
                self.streak_by_vehicle[vid] = 0
            # A new arrival has no trustworthy temporal history. Its first action
            # remains on the Denoise/DET -> Temporal Shield branch.
            if bool(new_flag):
                self.streak_by_vehicle[vid] = 0
                use_belief = False
            else:
                use_belief = self.streak_by_vehicle[vid] >= int(max(1, self.config.consecutive_steps))
            if use_belief:
                selected.append(belief_vec)
                branches.append('belief')
                self.belief_count += 1
            else:
                selected.append(policy_vec)
                branches.append('shield')
                self.shield_count += 1
            self.total += 1
            self.urgency_policy_sum += float(urgency_policy)
            self.urgency_belief_sum += float(urgency_belief)
            self.urgency_gain_sum += float(raw_gain)
            self.soc_drop_sum += float(soc_drop)
            self.time_drop_sum += float(time_drop)
            self.temporal_residual_sum += float(temporal)
            self.uncertainty_sum += float(uncertainty)
        return selected, branches

    def summary(self) -> dict[str, float | int]:
        n = int(self.total)
        return {
            'urgency_gate_total': n,
            'urgency_gate_belief_count': int(self.belief_count),
            'urgency_gate_shield_count': int(self.shield_count),
            'urgency_gate_belief_rate': 0.0 if n == 0 else float(self.belief_count / n),
            'urgency_gate_shield_rate': 0.0 if n == 0 else float(self.shield_count / n),
            'urgency_policy_mean': 0.0 if n == 0 else float(self.urgency_policy_sum / n),
            'urgency_belief_mean': 0.0 if n == 0 else float(self.urgency_belief_sum / n),
            'urgency_gain_mean': 0.0 if n == 0 else float(self.urgency_gain_sum / n),
            'urgency_soc_drop_mean': 0.0 if n == 0 else float(self.soc_drop_sum / n),
            'urgency_time_drop_mean': 0.0 if n == 0 else float(self.time_drop_sum / n),
            'urgency_temporal_residual_mean': 0.0 if n == 0 else float(self.temporal_residual_sum / n),
            'urgency_uncertainty_mean': 0.0 if n == 0 else float(self.uncertainty_sum / n),
            'urgency_time_innovation_ema_mean': 0.0 if not self.time_innovation_ema_by_vehicle else float(np.mean(list(self.time_innovation_ema_by_vehicle.values()))),
            'urgency_soc_innovation_ema_mean': 0.0 if not self.soc_innovation_ema_by_vehicle else float(np.mean(list(self.soc_innovation_ema_by_vehicle.values()))),
        }

class BeliefCoreEstimator:
    """Lightweight history-based core-state belief estimator.

    It is intentionally deterministic and no-leakage: estimates are propagated from previous
    defended belief and the previous executed action using the known EV charging dynamics, then
    fused with the current routed observation/DAE estimate. SOC and remaining time are fused
    conservatively (using the lower of observation and prediction-biased estimate) because the
    stealthy deadline attack tends to overstate both.
    """

    def __init__(self, config: BeliefCoreConfig | None = None) -> None:
        self.config = config or BeliefCoreConfig()
        self.prev_belief_state_by_vehicle: dict[int, np.ndarray] = {}
        self.prev_action_by_vehicle: dict[int, np.ndarray] = {}
        self.prev_time_by_vehicle: dict[int, int] = {}
        self.uncertainty_by_vehicle: dict[int, float] = {}
        self.core_innovation_by_vehicle: dict[int, np.ndarray] = {}
        self.fusion_count = 0
        self.mean_disagreement_sum = 0.0
        self.max_disagreement = 0.0
        self.uncertainty_sum = 0.0

    def reset(self) -> None:
        self.prev_belief_state_by_vehicle.clear()
        self.prev_action_by_vehicle.clear()
        self.prev_time_by_vehicle.clear()
        self.uncertainty_by_vehicle.clear()
        self.core_innovation_by_vehicle.clear()
        self.fusion_count = 0
        self.mean_disagreement_sum = 0.0
        self.max_disagreement = 0.0
        self.uncertainty_sum = 0.0

    def _physical_centers(self, prev_state: np.ndarray, prev_action: np.ndarray, prev_time_index: int, env: ChargingEnv) -> tuple[float, float, float]:
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

    def repair_batch(
        self,
        policy_states: Sequence[np.ndarray],
        vehicle_ids: Sequence[int],
        is_new_arrivals: Sequence[int],
        detector_scores: np.ndarray | None,
        env: ChargingEnv,
    ) -> list[np.ndarray]:
        if not self.config.enabled:
            return [to_numpy_1d(s).astype(np.float32) for s in policy_states]
        out: list[np.ndarray] = []
        scores = np.zeros((len(policy_states),), dtype=np.float32) if detector_scores is None else np.asarray(detector_scores, dtype=np.float32).reshape(-1)
        if scores.size != len(policy_states):
            scores = np.resize(scores, len(policy_states)).astype(np.float32)
        for state, vehicle_id, new_flag, score in zip(policy_states, vehicle_ids, is_new_arrivals, scores):
            vec = to_numpy_1d(state).astype(np.float32).copy()
            obs_core = vec[list(LOCAL_SHIELD_INDICES)].astype(np.float32)
            vid = int(vehicle_id)
            if bool(new_flag) or vid not in self.prev_belief_state_by_vehicle:
                pred_core = obs_core.copy()
                if self.config.use_known_initial_soc:
                    pred_core[0] = float(env.initial_soc)
                # Strict no-leak: remaining time is initialized only from the
                # current observation after Denoise/DET routing.
                pred_core[1] = float(obs_core[1])
                if self.config.use_known_initial_cost:
                    pred_core[2] = float(env.initial_cost_norm)
            else:
                pred_core = np.asarray(
                    self._physical_centers(
                        self.prev_belief_state_by_vehicle[vid],
                        self.prev_action_by_vehicle.get(vid, np.asarray([0.0], dtype=np.float32)),
                        self.prev_time_by_vehicle.get(vid, int(env.t) - 1),
                        env,
                    ),
                    dtype=np.float32,
                )
            innovation = (obs_core - pred_core).astype(np.float32)
            self.core_innovation_by_vehicle[vid] = innovation
            disagreement = float(np.max(np.abs(innovation)))
            old_unc = float(self.uncertainty_by_vehicle.get(vid, 0.0))
            unc = float(self.config.uncertainty_decay * old_unc + (1.0 - self.config.uncertainty_decay) * disagreement)
            self.uncertainty_by_vehicle[vid] = unc
            w_pred = float(self.config.pred_weight + self.config.detector_gain * max(0.0, min(1.0, float(score))))
            w_pred += float(self.config.disagreement_gain * min(0.08, disagreement))
            w_pred = float(np.clip(w_pred, 0.0, self.config.max_pred_weight))
            fused = w_pred * pred_core + (1.0 - w_pred) * obs_core
            # Conservative one-sided fusion for the two deadline-critical dimensions.
            fused[0] = min(float(fused[0]), float(obs_core[0]), float(pred_core[0] + self.config.soc_margin))
            fused[1] = min(float(fused[1]), float(obs_core[1]), float(pred_core[1] + self.config.time_margin))
            fused[2] = float(np.clip(fused[2], float(pred_core[2] - self.config.cost_margin), float(pred_core[2] + self.config.cost_margin)))
            vec[list(LOCAL_SHIELD_INDICES)] = fused.astype(np.float32)
            self.prev_belief_state_by_vehicle[vid] = vec.copy()
            self.fusion_count += 1
            self.mean_disagreement_sum += disagreement
            self.max_disagreement = max(self.max_disagreement, disagreement)
            self.uncertainty_sum += unc
            out.append(vec.astype(np.float32))
        return out

    def update_actions(self, vehicle_ids: Sequence[int], actions: np.ndarray, current_time: int) -> None:
        for vehicle_id, action in zip(vehicle_ids, np.asarray(actions, dtype=np.float32)):
            vid = int(vehicle_id)
            self.prev_action_by_vehicle[vid] = to_numpy_1d(action)
            self.prev_time_by_vehicle[vid] = int(current_time)

    def uncertainty(self, vehicle_id: int) -> float:
        return float(self.uncertainty_by_vehicle.get(int(vehicle_id), 0.0))

    def innovation(self, vehicle_id: int) -> np.ndarray:
        return self.core_innovation_by_vehicle.get(
            int(vehicle_id),
            np.zeros(3, dtype=np.float32),
        ).copy()

    def summary(self) -> dict[str, float | int]:
        n = int(self.fusion_count)
        return {
            'belief_fusion_count': n,
            'belief_disagreement_mean': 0.0 if n == 0 else float(self.mean_disagreement_sum / n),
            'belief_disagreement_max': float(self.max_disagreement),
            'belief_uncertainty_mean': 0.0 if n == 0 else float(self.uncertainty_sum / n),
        }

def rollout_episode_with_ug_bcr(
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
    enable_belief: bool = False,
    enable_urgency_gate: bool = False,
    ug_bcr_config: UGBCRConfig | None = None,
    urgency_gate_override: Any | None = None,
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
    audit_records: list[dict[str, Any]] | None = None,
    audit_context: dict[str, Any] | None = None,
) -> dict:
    """Evaluate DAE/Detector/Temporal Shield with optional UG-BCR belief repair."""
    config = ug_bcr_config or UGBCRConfig()
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
    belief = BeliefCoreEstimator(config.belief)
    urgency_gate = urgency_gate_override or UrgencyGatedBeliefSelector(config.urgency_gate)
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
        belief.update_actions(vehicle_ids, actions, current_time)

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
        if enable_urgency_gate:
            # Default to the shield branch; activate belief only when it makes
            # the departure-SOC requirement more urgent and reliable.
            pre_belief_policy_states = [to_numpy_1d(s).astype(np.float32) for s in policy_states]
            belief_states = belief.repair_batch(policy_states, vehicle_ids, is_new_flags, det_scores, env)
            policy_states, _branches = urgency_gate.select_batch(
                policy_states,
                belief_states,
                vehicle_ids,
                is_new_flags,
                belief,
                shield_config,
                env,
                reward_profile,
                prev_policy_obs_by_vehicle,
                prev_action_by_vehicle,
                prev_time_by_vehicle,
                detector_scores=det_scores,
                route_flags=route_flags,
            )
        elif enable_belief:
            policy_states = belief.repair_batch(policy_states, vehicle_ids, is_new_flags, det_scores, env)
        policy_states = _apply_shield_batch(policy_states, vehicle_ids, is_new_flags)
        route_count += int(sum(route_flags))
        route_total += len(route_flags)
        attack_obs_count += int(sum(attacked_flags))
        actions = _compute_actions(policy_states)
        _update_prev_actions(vehicle_ids, actions, int(env.t))
        _update_prev_observed(vehicle_ids, observed_states)
        diagnostics = {
            'observed_states': observed_states,
            'pre_belief_policy_states': pre_belief_policy_states if enable_urgency_gate else policy_states,
            'belief_states': belief_states if enable_urgency_gate else policy_states,
            'branches': _branches if enable_urgency_gate else ['shield' for _ in policy_states],
            'gate_scores': list(getattr(urgency_gate, 'last_scores', [float('nan') for _ in policy_states]))
            if enable_urgency_gate else [float('nan') for _ in policy_states],
            'route_flags': route_flags,
            'det_scores': det_scores,
            'selected_states': policy_states,
        }
        return policy_states, actions, diagnostics

    def _record_runtime_audit(
        clean_states: Sequence[np.ndarray],
        attacked_flags: Sequence[bool],
        stations: Sequence[int],
        vehicle_ids: Sequence[int],
        is_new_flags: Sequence[int],
        diagnostics: dict[str, Any],
    ) -> None:
        if audit_records is None or not enable_urgency_gate:
            return
        meta = dict(audit_context or {})
        guarded_idx = list(LOCAL_SHIELD_INDICES)
        for local_idx, values in enumerate(zip(
            clean_states,
            diagnostics['observed_states'],
            diagnostics['pre_belief_policy_states'],
            diagnostics['belief_states'],
            diagnostics['selected_states'],
            vehicle_ids,
            stations,
            is_new_flags,
            attacked_flags,
            diagnostics['route_flags'],
            diagnostics['det_scores'],
            diagnostics['branches'],
            diagnostics['gate_scores'],
        )):
            (clean_state, observed_state, policy_state, belief_state, selected_state,
             vehicle_id, station, new_flag, attacked_flag, route_flag, det_score, branch, gate_score) = values
            clean_vec = to_numpy_1d(clean_state).astype(np.float32)
            observed_vec = to_numpy_1d(observed_state).astype(np.float32)
            policy_vec = to_numpy_1d(policy_state).astype(np.float32)
            belief_vec = to_numpy_1d(belief_state).astype(np.float32)
            selected_vec = to_numpy_1d(selected_state).astype(np.float32)
            policy_err = np.abs(policy_vec[guarded_idx] - clean_vec[guarded_idx])
            belief_err = np.abs(belief_vec[guarded_idx] - clean_vec[guarded_idx])
            selected_err = np.abs(selected_vec[guarded_idx] - clean_vec[guarded_idx])
            observed_err = np.abs(observed_vec[guarded_idx] - clean_vec[guarded_idx])
            innovation = belief.innovation(int(vehicle_id))
            audit_records.append({
                **meta,
                'time_index': int(env.t),
                'batch_index': int(local_idx),
                'vehicle_id': int(vehicle_id),
                'station': int(station),
                'is_new_arrival': int(new_flag),
                'attacked_flag': int(bool(attacked_flag)),
                'det_route_flag': int(bool(route_flag)),
                'detector_score': float(det_score) if np.isfinite(float(det_score)) else float('nan'),
                'ug_branch': str(branch),
                'ug_belief_selected': int(str(branch) == 'belief'),
                'ug_gate_score': float(gate_score) if np.isfinite(float(gate_score)) else float('nan'),
                'clean_soc': float(clean_vec[0]),
                'clean_time': float(clean_vec[1]),
                'clean_cost': float(clean_vec[10]),
                'observed_soc': float(observed_vec[0]),
                'observed_time': float(observed_vec[1]),
                'observed_cost': float(observed_vec[10]),
                'policy_soc': float(policy_vec[0]),
                'policy_time': float(policy_vec[1]),
                'policy_cost': float(policy_vec[10]),
                'belief_soc': float(belief_vec[0]),
                'belief_time': float(belief_vec[1]),
                'belief_cost': float(belief_vec[10]),
                'selected_soc': float(selected_vec[0]),
                'selected_time': float(selected_vec[1]),
                'selected_cost': float(selected_vec[10]),
                'soc_innovation': float(innovation[0]),
                'time_innovation': float(innovation[1]),
                'cost_innovation': float(innovation[2]),
                'observed_core_linf_error': float(np.max(observed_err)),
                'observed_core_l1_error': float(np.sum(observed_err)),
                'policy_core_linf_error': float(np.max(policy_err)),
                'policy_core_l1_error': float(np.sum(policy_err)),
                'belief_core_linf_error': float(np.max(belief_err)),
                'belief_core_l1_error': float(np.sum(belief_err)),
                'selected_core_linf_error': float(np.max(selected_err)),
                'selected_core_l1_error': float(np.sum(selected_err)),
                'belief_l1_improvement': float(np.sum(policy_err) - np.sum(belief_err)),
                'belief_linf_improvement': float(np.max(policy_err) - np.max(belief_err)),
                'belief_uncertainty': float(belief.uncertainty(int(vehicle_id))),
            })

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
        _record_runtime_audit(clean_states, attacked_flags, stations, vehicle_ids, is_new_flags, diagnostics)
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
    summary.update(belief.summary())
    summary.update(urgency_gate.summary())
    summary['ug_bcr_belief_enabled'] = bool(enable_belief)
    summary['ug_bcr_urgency_gate_enabled'] = bool(enable_urgency_gate)
    return summary






__all__ = [
    'BeliefCoreConfig',
    'UrgencyGateConfig',
    'UGBCRConfig',
    'load_ug_bcr_config',
    'UrgencyGatedBeliefSelector',
    'BeliefCoreEstimator',
    'rollout_episode_with_ug_bcr',
]
