from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .merged_attacks import AttackContext, GLOBAL_ATTACK_IDX, LOCAL_ATTACK_IDX, PGDStateAttacker, attack_indices_for_state_scope, build_state_attacker, canonical_attack_state_scope
from .merged_core import to_numpy_1d


LONG_HORIZON_ATTACK_NAMES = (
    'local_small_drift_q',
    'local_deadline_drift_pgd',
    'full_pipeline_adaptive_deadline',
    'module_aware_cem_mpc',
    'temporal_shift_attack',
)
OPTIONAL_EXTENDED_LONG_HORIZON_ATTACK_NAMES = (
)


@dataclass(frozen=True)
class LongHorizonAttackSpec:
    name: str
    state_scope: str
    base_algorithm: str
    description: str


ATTACK_SPECS: dict[str, LongHorizonAttackSpec] = {
    'local_small_drift_q': LongHorizonAttackSpec(
        name='local_small_drift_q',
        state_scope='local',
        base_algorithm='q_function',
        description='Local small-drift Q attack that accumulates shallow perturbations across time.',
    ),
    'local_deadline_drift_pgd': LongHorizonAttackSpec(
        name='local_deadline_drift_pgd',
        state_scope='local',
        base_algorithm='opposite_pgd',
        description='Local deadline-coupled PGD drift that keeps each step light and gradually amplifies under-charge pressure near departure.',
    ),
    'full_pipeline_adaptive_deadline': LongHorizonAttackSpec(
        name='full_pipeline_adaptive_deadline',
        state_scope='local',
        base_algorithm='opposite_pgd',
        description='White-box adaptive deadline attack that selects each perturbation by explicitly evaluating candidates through the full DAE+Detector+UG-BCR+Temporal-Shield defense pipeline before the actor.',
    ),
    'module_aware_cem_mpc': LongHorizonAttackSpec(
        name='module_aware_cem_mpc',
        state_scope='local_or_global',
        base_algorithm='cem_mpc',
        description='Module-aware CEM-MPC long-horizon adaptive attack for Experiment 4 knowledge ladder and objective studies.',
    ),
    'temporal_shift_attack': LongHorizonAttackSpec(
        name='temporal_shift_attack',
        state_scope='all',
        base_algorithm='q_function',
        description='All-state temporal phase-shift attack over remaining time, SOC/cost, and price-window features.',
    ),
    'local_temporal_shift_attack': LongHorizonAttackSpec(
        name='local_temporal_shift_attack',
        state_scope='local',
        base_algorithm='q_function',
        description='Local-only temporal phase-shift attack over SOC, remaining time, and cumulative cost.',
    ),
}


def canonical_long_horizon_attack_name(value: str | None) -> str:
    token = str(value or '').strip().lower().replace('-', '_')
    aliases = {
        # DA-Deadline aliases are intentionally mapped to FP-Adaptive-Deadline in this release.
        'lt2': 'local_small_drift_q',
        'lt_local_small_drift_q': 'local_small_drift_q',
        'local_small_drift_q': 'local_small_drift_q',
        'small_drift_q': 'local_small_drift_q',
        'local_deadline_drift_pgd': 'local_deadline_drift_pgd',
        'deadline_drift_pgd': 'local_deadline_drift_pgd',
        'defense_aware_deadline_drift_pgd': 'full_pipeline_adaptive_deadline',
        'da_deadline': 'full_pipeline_adaptive_deadline',
        'adaptive_deadline': 'full_pipeline_adaptive_deadline',
        'defense_aware_deadline': 'full_pipeline_adaptive_deadline',
        'defense_aware_deadline_drift_pgd_v2': 'full_pipeline_adaptive_deadline',
        'da_deadline_v2': 'full_pipeline_adaptive_deadline',
        'adaptive_deadline_v2': 'full_pipeline_adaptive_deadline',
        'damage_constrained_deadline': 'full_pipeline_adaptive_deadline',
        'defense_aware_deadline_drift_pgd_v2_calibrated': 'full_pipeline_adaptive_deadline',
        'da_deadline_v2_calibrated': 'full_pipeline_adaptive_deadline',
        'adaptive_deadline_v2_calibrated': 'full_pipeline_adaptive_deadline',
        'full_pipeline_adaptive_deadline': 'full_pipeline_adaptive_deadline',
        'adaptive_deadline_full_pipeline': 'full_pipeline_adaptive_deadline',
        'fp_adaptive_deadline': 'full_pipeline_adaptive_deadline',
        'whitebox_adaptive_deadline': 'full_pipeline_adaptive_deadline',
        'true_adaptive_deadline': 'full_pipeline_adaptive_deadline',
        'module_aware_cem_mpc': 'module_aware_cem_mpc',
        'module_aware_mpc': 'module_aware_cem_mpc',
        'cem_mpc': 'module_aware_cem_mpc',
        'cem_mpc_adaptive': 'module_aware_cem_mpc',
        'adaptive_knowledge_ladder': 'module_aware_cem_mpc',
        'exp4_module_aware': 'module_aware_cem_mpc',
        'temporal_shift_attack': 'temporal_shift_attack',
        'temporal_shift': 'temporal_shift_attack',
        'time_price_shift_attack': 'temporal_shift_attack',
        'local_temporal_shift_attack': 'local_temporal_shift_attack',
        'temporal_shift_local': 'local_temporal_shift_attack',
        'local_temporal_shift': 'local_temporal_shift_attack',
        'local_time_shift_attack': 'local_temporal_shift_attack',
    }
    if token not in aliases:
        raise ValueError(f'Unsupported long-horizon attack: {value!r}')
    return aliases[token]


class StatefulLongHorizonAttacker:
    def __init__(
        self,
        base_attacker: PGDStateAttacker,
        *,
        name: str,
        attack_state_scope: str,
        epsilon: float | None = None,
        passive_decay: float = 0.9,
    ) -> None:
        self.base_attacker = base_attacker
        self.name = str(name)
        self.algorithm = self.name
        self.seed = int(getattr(base_attacker, 'seed', 42))
        self.attack_state_scope = canonical_attack_state_scope(attack_state_scope)
        self.attack_indices = tuple(int(v) for v in attack_indices_for_state_scope(self.attack_state_scope))
        self.epsilon = float(base_attacker.epsilon if epsilon is None else epsilon)
        self.passive_decay = float(np.clip(passive_decay, 0.0, 1.0))
        self.obs_low = None if getattr(base_attacker, 'obs_low', None) is None else base_attacker.obs_low.detach().cpu().numpy().reshape(-1)
        self.obs_high = None if getattr(base_attacker, 'obs_high', None) is None else base_attacker.obs_high.detach().cpu().numpy().reshape(-1)
        self.prev_delta_by_key: dict[tuple[int, int], np.ndarray] = {}
        self.prev_prev_delta_by_key: dict[tuple[int, int], np.ndarray] = {}
        self.prev_adv_obs_by_key: dict[tuple[int, int], np.ndarray] = {}
        self.step_count_by_key: defaultdict[tuple[int, int], int] = defaultdict(int)

    def reset(self) -> None:
        if hasattr(self.base_attacker, 'reset'):
            self.base_attacker.reset()
        self.prev_delta_by_key.clear()
        self.prev_prev_delta_by_key.clear()
        self.prev_adv_obs_by_key.clear()
        self.step_count_by_key.clear()

    def clone(self):
        raise NotImplementedError

    def observe_batch(
        self,
        obs_batch: np.ndarray,
        *,
        contexts: list[AttackContext] | None = None,
        vehicle_ids: list[int] | None = None,
        episode_indices: list[int] | None = None,
    ) -> None:
        del contexts
        obs_arr = np.asarray(obs_batch, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        vehicle_id_list = [int(v) for v in vehicle_ids] if vehicle_ids is not None else [int(i) for i in range(int(obs_arr.shape[0]))]
        episode_id_list = [int(v) for v in episode_indices] if episode_indices is not None else [0 for _ in vehicle_id_list]
        for row_idx, (vehicle_id, episode_id) in enumerate(zip(vehicle_id_list, episode_id_list)):
            key = (int(episode_id), int(vehicle_id))
            obs_vec = to_numpy_1d(obs_arr[row_idx])
            prev_delta = self.prev_delta_by_key.get(key)
            if prev_delta is not None:
                self.prev_prev_delta_by_key[key] = self._mask_delta(prev_delta)
                self.prev_delta_by_key[key] = self._mask_delta(prev_delta * self.passive_decay)
            self.prev_adv_obs_by_key[key] = obs_vec.copy()
            self.step_count_by_key[key] += 1

    def attack_with_metadata(
        self,
        obs_batch: np.ndarray,
        *,
        contexts: list[AttackContext],
        vehicle_ids: list[int],
        episode_indices: list[int],
    ) -> np.ndarray:
        obs_arr = np.asarray(obs_batch, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        keys = [(int(episode_id), int(vehicle_id)) for vehicle_id, episode_id in zip(vehicle_ids, episode_indices)]
        actor = getattr(self.base_attacker, 'actor', None)
        if hasattr(actor, 'prepare_attack_keys'):
            actor.prepare_attack_keys(keys)
        base_adv = self._base_attack(obs_arr, contexts, keys=keys)
        out_rows: list[np.ndarray] = []
        for row_idx, (key, context) in enumerate(zip(keys, contexts)):
            if hasattr(actor, 'prepare_attack_keys'):
                actor.prepare_attack_keys([key])
            current_obs = to_numpy_1d(obs_arr[row_idx])
            base_delta = self._mask_delta(to_numpy_1d(base_adv[row_idx]) - current_obs)
            shaped_delta = self._shape_delta(key, current_obs, base_delta, context)
            adv_obs = self._project_obs(current_obs, current_obs + shaped_delta)
            final_delta = self._mask_delta(adv_obs - current_obs)
            prev_delta = self.prev_delta_by_key.get(key)
            if prev_delta is not None:
                self.prev_prev_delta_by_key[key] = self._mask_delta(prev_delta)
            self.prev_delta_by_key[key] = final_delta.copy()
            self.prev_adv_obs_by_key[key] = adv_obs.copy()
            self.step_count_by_key[key] += 1
            out_rows.append(adv_obs.astype(np.float32))
        return np.asarray(out_rows, dtype=np.float32)

    def _base_attack(
        self,
        obs_arr: np.ndarray,
        contexts: list[AttackContext],
        *,
        keys: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        del contexts, keys
        return np.asarray(self.base_attacker.attack(obs_arr), dtype=np.float32)

    def _shape_delta(
        self,
        key: tuple[int, int],
        obs: np.ndarray,
        base_delta: np.ndarray,
        context: AttackContext,
    ) -> np.ndarray:
        del key, obs, context
        return self._bounded_delta(base_delta)

    def _mask_delta(self, delta: np.ndarray) -> np.ndarray:
        delta_vec = to_numpy_1d(delta)
        out = np.zeros_like(delta_vec, dtype=np.float32)
        out[list(self.attack_indices)] = delta_vec[list(self.attack_indices)]
        return out

    def _bounded_delta(self, delta: np.ndarray, *, epsilon: float | None = None) -> np.ndarray:
        max_eps = float(self.epsilon if epsilon is None else epsilon)
        bounded = np.clip(self._mask_delta(delta), -max_eps, max_eps)
        return bounded.astype(np.float32)

    def _project_obs(self, original: np.ndarray, proposal: np.ndarray) -> np.ndarray:
        original_vec = to_numpy_1d(original)
        proposal_vec = to_numpy_1d(proposal)
        delta = self._bounded_delta(proposal_vec - original_vec)
        candidate = original_vec + delta
        if self.obs_low is None or self.obs_high is None:
            return np.clip(candidate, 0.0, 1.0).astype(np.float32)
        return np.clip(candidate, self.obs_low, self.obs_high).astype(np.float32)

    def _prev_delta(self, key: tuple[int, int]) -> np.ndarray:
        prev = self.prev_delta_by_key.get(key)
        if prev is None:
            return np.zeros((11,), dtype=np.float32)
        return self._mask_delta(prev)

    def _prev_prev_delta(self, key: tuple[int, int]) -> np.ndarray:
        prev = self.prev_prev_delta_by_key.get(key)
        if prev is None:
            return np.zeros((11,), dtype=np.float32)
        return self._mask_delta(prev)

    def _step_count(self, key: tuple[int, int]) -> int:
        return int(self.step_count_by_key.get(key, 0))

    def _deadline_phase(self, obs: np.ndarray) -> float:
        obs_vec = to_numpy_1d(obs)
        t_re = float(obs_vec[1]) if obs_vec.size > 1 else 0.0
        return float(np.clip(1.0 - t_re, 0.0, 1.0))

    def _undercharge_bias(
        self,
        obs: np.ndarray,
        context: AttackContext,
        *,
        local_scale: float,
        time_scale: float,
        cost_scale: float,
        env_scale: float,
        price_scale: float,
        local_clip: float | None = None,
    ) -> np.ndarray:
        obs_vec = to_numpy_1d(obs)
        phase = self._deadline_phase(obs_vec)
        soc = float(obs_vec[0]) if obs_vec.size > 0 else 0.0
        low_soc = float(np.clip(0.55 - soc, 0.0, 0.55) / 0.55)
        phase = float(max(phase, 0.55 * low_soc))
        urgency = float(0.35 + 0.65 * phase)
        bias = np.zeros_like(obs_vec, dtype=np.float32)
        bias[0] = float(local_scale * urgency)
        bias[1] = float(time_scale * urgency)
        bias[10] = float(cost_scale * urgency)
        env_curve = np.asarray((0.55, 0.25, 0.85), dtype=np.float32)
        env_sign = 1.0 if float(context.raw_price) >= float(context.price_threshold) else -0.35
        bias[self.attack_indices_for_env()] = env_scale * env_curve * env_sign * urgency
        price_curve = np.asarray((0.90, 0.45, 0.00, -0.55, -1.00), dtype=np.float32)
        bias[self.attack_indices_for_price()] = price_scale * price_curve * urgency
        if local_clip is not None:
            local_idx = self.attack_indices_for_local()
            bias[local_idx] = np.clip(bias[local_idx], -float(local_clip), float(local_clip))
        return bias.astype(np.float32)

    def attack_indices_for_local(self) -> np.ndarray:
        return np.asarray(LOCAL_ATTACK_IDX, dtype=np.int64)

    def attack_indices_for_env(self) -> np.ndarray:
        return np.asarray(GLOBAL_ATTACK_IDX[:3], dtype=np.int64)

    def attack_indices_for_price(self) -> np.ndarray:
        return np.asarray((5, 6, 7, 8, 9), dtype=np.int64)


class LongHorizonSmallDriftQAttacker(StatefulLongHorizonAttacker):
    def __init__(
        self,
        base_attacker: PGDStateAttacker,
        *,
        epsilon: float = 0.055,
        drift_decay: float = 0.86,
        step_scale: float = 1.02,
        passive_decay: float = 0.92,
    ) -> None:
        super().__init__(
            base_attacker,
            name='local_small_drift_q',
            attack_state_scope='local',
            epsilon=epsilon,
            passive_decay=passive_decay,
        )
        self.drift_decay = float(np.clip(drift_decay, 0.0, 0.99))
        self.step_scale = float(max(step_scale, 1e-3))

    def clone(self):
        return LongHorizonSmallDriftQAttacker(
            self.base_attacker.clone(),
            epsilon=self.epsilon,
            drift_decay=self.drift_decay,
            step_scale=self.step_scale,
            passive_decay=self.passive_decay,
        )

    def _shape_delta(
        self,
        key: tuple[int, int],
        obs: np.ndarray,
        base_delta: np.ndarray,
        context: AttackContext,
    ) -> np.ndarray:
        del obs, context
        prev_delta = self._prev_delta(key)
        ramp = float(min(1.0, 0.45 + 0.12 * self._step_count(key)))
        drift = self.drift_decay * prev_delta + self.step_scale * base_delta
        return self._bounded_delta(drift * ramp)


class LongHorizonLocalDeadlineDriftPGDAttacker(StatefulLongHorizonAttacker):
    _local_curve = np.asarray((0.75, 1.00, 0.65), dtype=np.float32)

    def __init__(
        self,
        base_attacker: PGDStateAttacker,
        *,
        epsilon: float = 0.055,
        drift_decay: float = 0.95,
        step_scale: float = 1.04,
        passive_decay: float = 0.97,
        deadline_gain: float = 1.35,
        late_phase_budget_scale: float = 1.0,
        terminal_phase_budget_scale: float = 1.0,
        late_push_start: float = 0.62,
        late_dim_weights: tuple[float, float, float] = (0.75, 1.00, 0.65),
        mid_phase_start: float = 0.48,
        no_rebound_start: float = 0.75,
        no_rebound_hold_ratio: float = 0.94,
    ) -> None:
        super().__init__(
            base_attacker,
            name='local_deadline_drift_pgd',
            attack_state_scope='local',
            epsilon=epsilon,
            passive_decay=passive_decay,
        )
        self.drift_decay = float(np.clip(drift_decay, 0.0, 0.995))
        self.step_scale = float(max(step_scale, 1e-3))
        self.deadline_gain = float(max(deadline_gain, 0.0))
        self.late_phase_budget_scale = float(max(late_phase_budget_scale, 1.0))
        self.terminal_phase_budget_scale = float(max(terminal_phase_budget_scale, 1.0))
        self.late_push_start = float(np.clip(late_push_start, 0.0, 0.98))
        self.mid_phase_start = float(np.clip(mid_phase_start, 0.0, 0.95))
        self.no_rebound_start = float(np.clip(no_rebound_start, 0.0, 0.99))
        self.no_rebound_hold_ratio = float(np.clip(no_rebound_hold_ratio, 0.0, 1.2))
        late_weights = np.asarray(late_dim_weights, dtype=np.float32).reshape(-1)
        if late_weights.size != 3:
            raise ValueError('late_dim_weights must contain exactly three values for SOC / time / cost.')
        self.late_dim_weights = late_weights.astype(np.float32)
        self.prev_target_action_by_key: dict[tuple[int, int], float] = {}
        self.prev_realized_action_by_key: dict[tuple[int, int], float] = {}

    def reset(self) -> None:
        super().reset()
        self.prev_target_action_by_key.clear()
        self.prev_realized_action_by_key.clear()

    def clone(self):
        return LongHorizonLocalDeadlineDriftPGDAttacker(
            self.base_attacker.clone(),
            epsilon=self.epsilon,
            drift_decay=self.drift_decay,
            step_scale=self.step_scale,
            passive_decay=self.passive_decay,
            deadline_gain=self.deadline_gain,
            late_phase_budget_scale=self.late_phase_budget_scale,
            terminal_phase_budget_scale=self.terminal_phase_budget_scale,
            late_push_start=self.late_push_start,
            late_dim_weights=tuple(float(v) for v in self.late_dim_weights.tolist()),
            mid_phase_start=self.mid_phase_start,
            no_rebound_start=self.no_rebound_start,
            no_rebound_hold_ratio=self.no_rebound_hold_ratio,
        )

    def _target_action(self, key: tuple[int, int], obs: np.ndarray, context: AttackContext) -> float:
        phase = self._deadline_phase(obs)
        phase_mid = float(np.clip((phase - self.mid_phase_start) / max(1e-6, 1.0 - self.mid_phase_start), 0.0, 1.0))
        phase_late = float(np.clip((phase - self.late_push_start) / max(1e-6, 1.0 - self.late_push_start), 0.0, 1.0))
        phase_terminal = float(np.clip((phase - self.no_rebound_start) / max(1e-6, 1.0 - self.no_rebound_start), 0.0, 1.0))
        soc = float(to_numpy_1d(obs)[0])
        low_soc = float(np.clip((0.62 - soc) / 0.62, 0.0, 1.0))
        cheap_bonus = 0.22 if float(context.raw_price) < float(context.price_threshold) else 0.06
        target = -(0.18 + 0.26 * phase_mid + 0.34 * phase_late + 0.20 * low_soc + cheap_bonus)
        if phase >= self.no_rebound_start:
            terminal_floor = -(0.42 + 0.20 * phase_terminal + 0.16 * low_soc + 0.10 * cheap_bonus)
            target = min(target, terminal_floor)
        target = float(np.clip(target, -1.0, -0.18))
        prev_target = self.prev_target_action_by_key.get(key)
        if prev_target is not None and phase >= self.mid_phase_start:
            target = min(target, float(prev_target))
        if prev_target is not None and phase >= self.no_rebound_start:
            target = min(target, float(prev_target) - (0.04 + 0.05 * phase_terminal))
        target = float(np.clip(target, -1.0, -0.18))
        self.prev_target_action_by_key[key] = target
        return target

    def _actor_action_value(self, obs: np.ndarray) -> float:
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        device = getattr(self.base_attacker, 'device', torch.device('cpu'))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=device).reshape(1, -1)
            action = self.base_attacker.actor(obs_t).reshape(-1)
        return float(action.detach().cpu().numpy()[0])

    def _terminal_action_ceiling(self, key: tuple[int, int], current_action: float, phase_terminal: float) -> float:
        required_action = float(current_action - (0.03 + 0.06 * phase_terminal))
        prev_realized = self.prev_realized_action_by_key.get(key)
        if prev_realized is not None:
            required_action = min(required_action, float(prev_realized) - (0.025 + 0.04 * phase_terminal))
        return required_action

    def _record_realized_action(self, key: tuple[int, int], action: float) -> None:
        self.prev_realized_action_by_key[key] = float(action)

    def _base_attack(
        self,
        obs_arr: np.ndarray,
        contexts: list[AttackContext],
        *,
        keys: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        if keys is None:
            raise ValueError('local_deadline_drift_pgd requires per-sample keys for sustained target tracking.')
        targets: list[float] = []
        for key, obs, context in zip(keys, obs_arr, contexts):
            targets.append(self._target_action(key, obs, context))
        target_actions = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
        return np.asarray(self.base_attacker.attack(obs_arr, target_actions=target_actions), dtype=np.float32)

    def _shape_delta(
        self,
        key: tuple[int, int],
        obs: np.ndarray,
        base_delta: np.ndarray,
        context: AttackContext,
    ) -> np.ndarray:
        prev_delta = self._prev_delta(key)
        step = self._step_count(key)
        phase = self._deadline_phase(obs)
        phase_mid = float(np.clip((phase - self.mid_phase_start) / max(1e-6, 1.0 - self.mid_phase_start), 0.0, 1.0))
        phase_late = float(np.clip((phase - self.late_push_start) / max(1e-6, 1.0 - self.late_push_start), 0.0, 1.0))
        phase_terminal = float(np.clip((phase - self.no_rebound_start) / max(1e-6, 1.0 - self.no_rebound_start), 0.0, 1.0))
        ramp = float(min(1.0, 0.42 + 0.050 * step + 0.16 * phase + 0.08 * phase_late))
        drift = self.drift_decay * prev_delta + self.step_scale * base_delta
        weighted = np.zeros_like(drift, dtype=np.float32)
        local_idx = self.attack_indices_for_local()
        local_weights = np.asarray(
            (
                1.02 + 0.18 * phase,
                1.02 + 0.18 * phase,
                1.02 + 0.18 * phase,
            ),
            dtype=np.float32,
        )
        weighted[local_idx] = local_weights * drift[local_idx]
        bias = self._undercharge_bias(
            obs,
            context,
            local_scale=(0.010 + 0.007 * phase) * self.deadline_gain,
            time_scale=(0.013 + 0.009 * phase) * self.deadline_gain,
            cost_scale=(0.008 + 0.005 * phase) * self.deadline_gain,
            env_scale=0.0,
            price_scale=0.0,
            local_clip=0.022,
        )
        weighted += 0.78 * bias
        if phase >= self.late_push_start:
            weighted[local_idx] += 0.0032 * phase_late * self.deadline_gain * self.late_phase_budget_scale * self.late_dim_weights
        if phase >= self.no_rebound_start:
            floor_values = np.maximum(prev_delta[local_idx] * self.no_rebound_hold_ratio, 0.0)
            weighted[local_idx] = np.maximum(weighted[local_idx], floor_values)
            weighted[local_idx] += 0.0024 * phase_terminal * self.deadline_gain * np.asarray((0.10, 1.15, 0.85), dtype=np.float32)
        effective_epsilon = float(
            self.epsilon * (
                1.0
                + 0.04 * phase_mid
                + (self.late_phase_budget_scale - 1.0) * phase_late
                + 0.12 * phase_terminal
                + (self.terminal_phase_budget_scale - 1.0) * phase_terminal
            )
        )
        candidate_delta = self._bounded_delta(weighted * ramp, epsilon=effective_epsilon)
        if phase < self.no_rebound_start:
            if phase >= self.mid_phase_start:
                self._record_realized_action(key, self._actor_action_value(obs + candidate_delta))
            return candidate_delta

        base_only_delta = self._bounded_delta((1.35 + 0.20 * phase_terminal) * base_delta + 0.60 * prev_delta, epsilon=effective_epsilon)
        stronger_base_delta = self._bounded_delta((1.60 + 0.30 * phase_terminal) * base_delta + 0.80 * prev_delta, epsilon=effective_epsilon)
        terminal_floor = np.maximum(
            prev_delta[local_idx] * self.no_rebound_hold_ratio,
            np.asarray(
                (
                    0.014 + 0.008 * phase_terminal,
                    0.040 + 0.010 * phase_terminal,
                    0.028 + 0.008 * phase_terminal,
                ),
                dtype=np.float32,
            ),
        )
        terminal_templates = (
            np.asarray((0.16, 1.10, 0.84), dtype=np.float32),
            np.asarray((0.24, 1.34, 0.98), dtype=np.float32),
            np.asarray((0.34, 1.58, 1.10), dtype=np.float32),
        )

        current_action = self._actor_action_value(obs)
        chosen_delta = candidate_delta
        chosen_action = self._actor_action_value(obs + chosen_delta)

        for alt_delta in (base_only_delta, stronger_base_delta):
            alt_action = self._actor_action_value(obs + alt_delta)
            if alt_action < chosen_action:
                chosen_delta = alt_delta
                chosen_action = alt_action

        required_action = self._terminal_action_ceiling(key, current_action, phase_terminal)
        for refine_idx, template in enumerate(terminal_templates):
            if chosen_action <= required_action:
                break
            corrective = chosen_delta.copy()
            corrective[local_idx] = np.maximum(corrective[local_idx], terminal_floor)
            corrective[local_idx] += (
                0.0022
                * (1.0 + phase_terminal + 0.35 * refine_idx)
                * self.deadline_gain
                * template
            )
            corrective = self._bounded_delta(corrective, epsilon=effective_epsilon)
            corrective_action = self._actor_action_value(obs + corrective)
            if corrective_action < chosen_action:
                chosen_delta = corrective
                chosen_action = corrective_action

        if chosen_action > required_action:
            saturating = chosen_delta.copy()
            saturating[local_idx] = np.maximum(
                saturating[local_idx],
                np.asarray(
                    (
                        min(effective_epsilon, 0.024 + 0.010 * phase_terminal),
                        min(effective_epsilon, 0.050 + 0.004 * phase_terminal),
                        min(effective_epsilon, 0.038 + 0.008 * phase_terminal),
                    ),
                    dtype=np.float32,
                ),
            )
            saturating = self._bounded_delta(saturating, epsilon=effective_epsilon)
            saturating_action = self._actor_action_value(obs + saturating)
            if saturating_action < chosen_action:
                chosen_delta = saturating
                chosen_action = saturating_action

        self._record_realized_action(key, chosen_action)
        return chosen_delta

class LongHorizonTemporalShiftAttacker(StatefulLongHorizonAttacker):
    _price_indices = np.asarray((5, 6, 7, 8, 9), dtype=np.int64)

    def __init__(
        self,
        base_attacker: PGDStateAttacker,
        *,
        name: str = 'temporal_shift_attack',
        attack_state_scope: str = 'all',
        epsilon: float = 0.055,
        passive_decay: float = 0.96,
        drift_decay: float = 0.88,
    ) -> None:
        super().__init__(
            base_attacker,
            name=name,
            attack_state_scope=attack_state_scope,
            epsilon=epsilon,
            passive_decay=passive_decay,
        )
        self.drift_decay = float(np.clip(drift_decay, 0.0, 0.995))

    def clone(self):
        return LongHorizonTemporalShiftAttacker(
            self.base_attacker.clone(),
            name=self.name,
            attack_state_scope=self.attack_state_scope,
            epsilon=self.epsilon,
            passive_decay=self.passive_decay,
            drift_decay=self.drift_decay,
        )

    def _actor_action_value(self, obs: np.ndarray) -> float:
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        device = getattr(self.base_attacker, 'device', torch.device('cpu'))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=device).reshape(1, -1)
            action = self.base_attacker.actor(obs_t).reshape(-1)
        return float(action.detach().cpu().numpy()[0])

    def _q_value_for_delta(self, obs: np.ndarray, delta: np.ndarray) -> float:
        critic = getattr(self.base_attacker, 'critic', None)
        actor = getattr(self.base_attacker, 'actor', None)
        if critic is None or actor is None:
            return float('nan')
        device = getattr(self.base_attacker, 'device', torch.device('cpu'))
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        adv_vec = self._project_obs(obs_vec, obs_vec + delta)
        with torch.no_grad():
            clean_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=device).reshape(1, -1)
            adv_t = torch.as_tensor(adv_vec, dtype=torch.float32, device=device).reshape(1, -1)
            action_t = actor(adv_t)
            critic_obs = adv_t if bool(getattr(critic, "uses_state_value", False)) else clean_t
            q_t = critic(critic_obs, action_t).reshape(-1)
        return float(q_t.detach().cpu().numpy()[0])

    def _candidate_set(self, key: tuple[int, int], obs: np.ndarray, base_delta: np.ndarray, context: AttackContext) -> list[np.ndarray]:
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        prev = self._prev_delta(key)
        phase = self._deadline_phase(obs_vec)
        low_soc = float(np.clip((0.65 - float(obs_vec[0])) / 0.65, 0.0, 1.0))
        ramp = float(min(1.0, 0.40 + 0.28 * phase + 0.16 * low_soc + 0.04 * self._step_count(key)))
        template = np.zeros_like(obs_vec, dtype=np.float32)
        template[0] = 0.014 + 0.014 * phase + 0.010 * low_soc
        template[1] = 0.026 + 0.020 * phase + 0.010 * low_soc
        template[10] = 0.010 + 0.010 * phase

        prices = obs_vec[self._price_indices]
        left = np.roll(prices, -1) - prices
        right = np.roll(prices, 1) - prices
        rising = 1.0 if float(context.raw_price) >= float(context.price_threshold) else -0.65
        price_left = np.zeros_like(obs_vec, dtype=np.float32)
        price_right = np.zeros_like(obs_vec, dtype=np.float32)
        price_left[self._price_indices] = 0.45 * np.clip(left, -0.035, 0.035)
        price_right[self._price_indices] = 0.45 * np.clip(right, -0.035, 0.035)

        base = self.drift_decay * prev + ramp * base_delta
        time_template = self.drift_decay * prev + ramp * template
        price_template = time_template.copy()
        price_template[self._price_indices] += rising * price_left[self._price_indices]
        alt_price_template = time_template.copy()
        alt_price_template[self._price_indices] += -rising * price_right[self._price_indices]
        mixed = 0.45 * base + 0.55 * price_template
        return [base, time_template, price_template, alt_price_template, mixed]

    def _shape_delta(self, key: tuple[int, int], obs: np.ndarray, base_delta: np.ndarray, context: AttackContext) -> np.ndarray:
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        candidates = [self._bounded_delta(cand, epsilon=self.epsilon) for cand in self._candidate_set(key, obs_vec, base_delta, context)]
        best = candidates[0]
        best_score = float('inf')
        for cand in candidates:
            q = self._q_value_for_delta(obs_vec, cand)
            score = q if np.isfinite(q) else self._actor_action_value(obs_vec + cand)
            if score < best_score:
                best_score = float(score)
                best = cand
        return self._bounded_delta(best, epsilon=self.epsilon)


class DefenseAwareDeadlineDriftPGDAttacker(LongHorizonLocalDeadlineDriftPGDAttacker):
    """Defense-aware deadline drift using a differentiable/surrogate stealth objective.

    The attacker is intentionally conservative on each step.  It selects among
    PGD-derived and template core perturbations that (i) suppress the actor's
    charging action, (ii) keep SOC / remaining-time / cost perturbations smooth
    over time, and (iii) reduce an urgency-gate surrogate by making the observed
    SOC/time pair look less deadline-critical.  The true environment state is
    not changed; only the observation passed into the defense pipeline is
    perturbed.
    """

    def __init__(
        self,
        base_attacker: PGDStateAttacker,
        *,
        epsilon: float = 0.050,
        drift_decay: float = 0.985,
        step_scale: float = 0.72,
        passive_decay: float = 0.985,
        deadline_gain: float = 1.05,
        stealth_weight: float = 1.35,
        action_weight: float = 1.0,
        urgency_weight: float = 0.40,
        smooth_step_clip: float = 0.012,
        shield_soc_coeff: float = 0.40,
        shield_time_coeff: float = 1.00,
        shield_cost_coeff: float = 0.35,
    ) -> None:
        super().__init__(
            base_attacker,
            epsilon=epsilon,
            drift_decay=drift_decay,
            step_scale=step_scale,
            passive_decay=passive_decay,
            deadline_gain=deadline_gain,
            late_phase_budget_scale=1.0,
            terminal_phase_budget_scale=1.0,
            late_push_start=0.55,
            mid_phase_start=0.42,
            no_rebound_start=0.78,
            no_rebound_hold_ratio=0.98,
        )
        self.name = 'defense_aware_deadline_drift_pgd'
        self.algorithm = self.name
        self.stealth_weight = float(stealth_weight)
        self.action_weight = float(action_weight)
        self.urgency_weight = float(urgency_weight)
        self.smooth_step_clip = float(max(smooth_step_clip, 1e-6))
        self.shield_coeff = np.asarray((shield_soc_coeff, shield_time_coeff, shield_cost_coeff), dtype=np.float32)

    def clone(self):
        return DefenseAwareDeadlineDriftPGDAttacker(
            self.base_attacker.clone(),
            epsilon=self.epsilon,
            drift_decay=self.drift_decay,
            step_scale=self.step_scale,
            passive_decay=self.passive_decay,
            deadline_gain=self.deadline_gain,
            stealth_weight=self.stealth_weight,
            action_weight=self.action_weight,
            urgency_weight=self.urgency_weight,
            smooth_step_clip=self.smooth_step_clip,
            shield_soc_coeff=float(self.shield_coeff[0]),
            shield_time_coeff=float(self.shield_coeff[1]),
            shield_cost_coeff=float(self.shield_coeff[2]),
        )

    def _temporal_surrogate_residual(self, key: tuple[int, int], obs: np.ndarray, delta: np.ndarray) -> float:
        """Approximate the residual a Temporal Shield would see.

        It uses the previous adversarial observation and the actor action on it
        as a differentiable/surrogate reference.  This does not access hidden
        simulator state and is therefore consistent with an observation attacker.
        """
        prev_adv = self.prev_adv_obs_by_key.get(key)
        if prev_adv is None:
            # New arrivals are allowed more flexibility; only penalize abrupt deltas.
            prev_delta = self._prev_delta(key)[list(LOCAL_ATTACK_IDX)]
            cur_delta = delta[list(LOCAL_ATTACK_IDX)]
            return float(np.mean(np.abs(cur_delta - prev_delta)))
        prev_adv = to_numpy_1d(prev_adv).astype(np.float32)
        prev_action = self._actor_action_value(prev_adv)
        # ChargingEnv default: max_power=..., slice_hours=1/12, battery_capacity=...
        # The exact env object is not available inside this attacker, so use the
        # observed project scale (about 0.025 SOC per full-action step) as a
        # conservative local surrogate.  This is intentionally a soft penalty;
        # the true Shield is still evaluated in the rollout.
        soc_step = 0.025
        pred = prev_adv.copy()
        pred[0] = float(np.clip(prev_adv[0] + soc_step * max(prev_action, 0.0), 0.0, 1.1))
        pred[1] = float(max(prev_adv[1] - 1.0 / 12.0, 0.0))
        # Cost is monotone and noisy; smoothness is a better surrogate than an exact model.
        pred[10] = float(prev_adv[10])
        adv = to_numpy_1d(obs).astype(np.float32) + to_numpy_1d(delta).astype(np.float32)
        local_res = np.abs(adv[list(LOCAL_ATTACK_IDX)] - pred[list(LOCAL_ATTACK_IDX)])
        return float(np.mean(self.shield_coeff * local_res))

    def _urgency_surrogate(self, obs: np.ndarray, delta: np.ndarray) -> float:
        adv = to_numpy_1d(obs).astype(np.float32) + to_numpy_1d(delta).astype(np.float32)
        soc = float(adv[0])
        rt = float(max(adv[1], 1.0 / 12.0))
        urgency = max(0.0, 0.9 - soc) / rt
        # Soft gate around the deadline-risk threshold; lower is stealthier.
        return float(1.0 / (1.0 + np.exp(-4.0 * (urgency - 2.16))))

    def _candidate_score(self, key: tuple[int, int], obs: np.ndarray, delta: np.ndarray) -> float:
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        adv_vec = self._project_obs(obs_vec, obs_vec + self._bounded_delta(delta))
        clean_action = self._actor_action_value(obs_vec)
        adv_action = self._actor_action_value(adv_vec)
        action_damage = float(clean_action - adv_action)  # higher means more charging suppression
        residual = self._temporal_surrogate_residual(key, obs_vec, adv_vec - obs_vec)
        urgency = self._urgency_surrogate(obs_vec, adv_vec - obs_vec)
        norm_penalty = float(np.linalg.norm((adv_vec - obs_vec)[list(LOCAL_ATTACK_IDX)], ord=2))
        return float(self.action_weight * action_damage - self.stealth_weight * residual - self.urgency_weight * urgency - 0.04 * norm_penalty)

    def _shape_delta(
        self,
        key: tuple[int, int],
        obs: np.ndarray,
        base_delta: np.ndarray,
        context: AttackContext,
    ) -> np.ndarray:
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        prev_delta = self._prev_delta(key)
        step = self._step_count(key)
        phase = self._deadline_phase(obs_vec)
        local_idx = self.attack_indices_for_local()

        # Smooth defense-aware bias: make SOC and remaining time look slightly
        # larger to suppress deadline urgency, while avoiding abrupt residuals.
        low_soc = float(np.clip((0.62 - float(obs_vec[0])) / 0.62, 0.0, 1.0))
        ramp = float(min(1.0, 0.30 + 0.10 * step + 0.30 * phase + 0.18 * low_soc))
        stealth_bias = np.zeros_like(obs_vec, dtype=np.float32)
        stealth_bias[0] = (0.010 + 0.010 * phase + 0.010 * low_soc) * self.deadline_gain
        stealth_bias[1] = (0.016 + 0.015 * phase + 0.010 * low_soc) * self.deadline_gain
        stealth_bias[10] = (0.008 + 0.010 * phase) * self.deadline_gain

        # Candidate 1: PGD damage with strong smoothness.
        cand_pgd = self.drift_decay * prev_delta + self.step_scale * base_delta
        cand_pgd[local_idx] = np.clip(cand_pgd[local_idx], prev_delta[local_idx] - self.smooth_step_clip, prev_delta[local_idx] + self.smooth_step_clip)

        # Candidate 2: smooth stealth template.
        cand_template = self.drift_decay * prev_delta + ramp * stealth_bias
        cand_template[local_idx] = np.clip(cand_template[local_idx], prev_delta[local_idx] - self.smooth_step_clip, prev_delta[local_idx] + self.smooth_step_clip)

        # Candidate 3: balanced PGD+template.
        cand_mix = 0.55 * cand_template + 0.45 * cand_pgd
        cand_mix[local_idx] = np.clip(cand_mix[local_idx], prev_delta[local_idx] - self.smooth_step_clip, prev_delta[local_idx] + self.smooth_step_clip)

        # Candidate 4: terminal stealth push, still bounded/smooth.
        cand_terminal = cand_template.copy()
        if phase > 0.62:
            extra = np.zeros_like(obs_vec, dtype=np.float32)
            extra[0] = 0.004 + 0.006 * phase
            extra[1] = 0.006 + 0.008 * phase
            extra[10] = 0.004 + 0.004 * phase
            cand_terminal = cand_terminal + extra
            cand_terminal[local_idx] = np.clip(cand_terminal[local_idx], prev_delta[local_idx] - self.smooth_step_clip, prev_delta[local_idx] + self.smooth_step_clip)

        candidates = [cand_pgd, cand_template, cand_mix, cand_terminal]
        best_delta = None
        best_score = -float('inf')
        for cand in candidates:
            cand = self._bounded_delta(cand)
            score = self._candidate_score(key, obs_vec, cand)
            if score > best_score:
                best_score = score
                best_delta = cand
        if best_delta is None:
            best_delta = self._bounded_delta(cand_mix)
        # Track the realized post-defense surrogate action on the adversarial obs.
        self._record_realized_action(key, self._actor_action_value(obs_vec + best_delta))
        return self._bounded_delta(best_delta)


class DamageFirstDefenseAwareDeadlineDriftPGDAttacker(LongHorizonLocalDeadlineDriftPGDAttacker):
    """Damage-first, stealth-second defense-aware deadline drift.

    This version deliberately includes the standard deadline-drift candidate in
    the candidate set so attack-only strength cannot collapse simply because
    stealth penalties are large.  It first ranks candidates by charging-action
    suppression, keeps candidates within a damage tolerance of the best one,
    and only then selects the smoother / lower-urgency / lower-temporal-residual
    candidate.  This implements the evaluation principle used in the paper
    notes: match raw attack strength first, then test defense bypass.
    """

    def __init__(
        self,
        base_attacker: PGDStateAttacker,
        *,
        epsilon: float = 0.075,
        drift_decay: float = 0.965,
        step_scale: float = 1.18,
        passive_decay: float = 0.98,
        deadline_gain: float = 1.70,
        damage_tolerance: float = 0.82,
        stealth_weight: float = 0.18,
        urgency_weight: float = 0.08,
        smooth_step_clip: float = 0.045,
    ) -> None:
        super().__init__(
            base_attacker,
            epsilon=epsilon,
            drift_decay=drift_decay,
            step_scale=step_scale,
            passive_decay=passive_decay,
            deadline_gain=deadline_gain,
            late_phase_budget_scale=1.22,
            terminal_phase_budget_scale=1.30,
            late_push_start=0.50,
            mid_phase_start=0.38,
            no_rebound_start=0.70,
            no_rebound_hold_ratio=0.99,
        )
        self.name = 'defense_aware_deadline_drift_pgd_v2'
        self.algorithm = self.name
        self.damage_tolerance = float(np.clip(damage_tolerance, 0.0, 1.0))
        self.stealth_weight = float(max(0.0, stealth_weight))
        self.urgency_weight = float(max(0.0, urgency_weight))
        self.smooth_step_clip = float(max(1e-6, smooth_step_clip))

    def clone(self):
        return DamageFirstDefenseAwareDeadlineDriftPGDAttacker(
            self.base_attacker.clone(),
            epsilon=self.epsilon,
            drift_decay=self.drift_decay,
            step_scale=self.step_scale,
            passive_decay=self.passive_decay,
            deadline_gain=self.deadline_gain,
            damage_tolerance=self.damage_tolerance,
            stealth_weight=self.stealth_weight,
            urgency_weight=self.urgency_weight,
            smooth_step_clip=self.smooth_step_clip,
        )

    def _temporal_residual_surrogate(self, key: tuple[int, int], obs: np.ndarray, delta: np.ndarray) -> float:
        prev_adv = self.prev_adv_obs_by_key.get(key)
        local_idx = self.attack_indices_for_local()
        if prev_adv is None:
            return float(np.mean(np.abs(delta[local_idx] - self._prev_delta(key)[local_idx])))
        prev_adv = to_numpy_1d(prev_adv).astype(np.float32)
        prev_action = self._actor_action_value(prev_adv)
        pred = prev_adv.copy()
        pred[0] = float(np.clip(prev_adv[0] + 0.025 * max(prev_action, 0.0), 0.0, 1.1))
        pred[1] = float(max(prev_adv[1] - 1.0 / 12.0, 0.0))
        pred[10] = float(prev_adv[10])
        adv = to_numpy_1d(obs).astype(np.float32) + to_numpy_1d(delta).astype(np.float32)
        weights = np.asarray((0.30, 1.00, 0.35), dtype=np.float32)
        return float(np.mean(weights * np.abs(adv[local_idx] - pred[local_idx])))

    def _urgency_gate_surrogate(self, obs: np.ndarray, delta: np.ndarray) -> float:
        adv = to_numpy_1d(obs).astype(np.float32) + to_numpy_1d(delta).astype(np.float32)
        soc = float(adv[0])
        rt = float(max(adv[1], 1.0 / 12.0))
        urgency = max(0.0, 0.9 - soc) / rt
        return float(1.0 / (1.0 + np.exp(-4.0 * (urgency - 2.16))))

    def _candidate_stats(self, key: tuple[int, int], obs: np.ndarray, delta: np.ndarray) -> dict[str, float]:
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        delta = self._bounded_delta(delta)
        adv = self._project_obs(obs_vec, obs_vec + delta)
        delta = self._mask_delta(adv - obs_vec)
        clean_action = self._actor_action_value(obs_vec)
        adv_action = self._actor_action_value(adv)
        damage = float(max(0.0, clean_action - adv_action))
        residual = self._temporal_residual_surrogate(key, obs_vec, delta)
        urgency = self._urgency_gate_surrogate(obs_vec, delta)
        norm = float(np.linalg.norm(delta[list(LOCAL_ATTACK_IDX)], ord=2))
        stealth_score = -self.stealth_weight * residual - self.urgency_weight * urgency - 0.02 * norm
        return dict(damage=damage, residual=residual, urgency=urgency, norm=norm, stealth_score=stealth_score, adv_action=float(adv_action))

    def _shape_delta(self, key: tuple[int, int], obs: np.ndarray, base_delta: np.ndarray, context: AttackContext) -> np.ndarray:
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        prev_delta = self._prev_delta(key)
        step = self._step_count(key)
        phase = self._deadline_phase(obs_vec)
        local_idx = self.attack_indices_for_local()

        # Candidate 0: the parent strong deadline drift. This is the damage anchor.
        parent_candidate = super()._shape_delta(key, obs_vec, base_delta, context)

        # Damage-first templates: positive SOC/time/cost perturbations suppress urgency.
        low_soc = float(np.clip((0.65 - float(obs_vec[0])) / 0.65, 0.0, 1.0))
        ramp = float(min(1.0, 0.42 + 0.10 * step + 0.22 * phase + 0.12 * low_soc))
        base = self.drift_decay * prev_delta + self.step_scale * base_delta
        base[local_idx] = np.clip(base[local_idx], prev_delta[local_idx] - self.smooth_step_clip, prev_delta[local_idx] + self.smooth_step_clip)
        bias = np.zeros_like(obs_vec, dtype=np.float32)
        bias[0] = (0.020 + 0.018 * phase + 0.012 * low_soc) * self.deadline_gain
        bias[1] = (0.040 + 0.020 * phase + 0.015 * low_soc) * self.deadline_gain
        bias[10] = (0.026 + 0.014 * phase) * self.deadline_gain
        smooth_template = self.drift_decay * prev_delta + ramp * bias
        smooth_template[local_idx] = np.clip(smooth_template[local_idx], prev_delta[local_idx] - self.smooth_step_clip, prev_delta[local_idx] + self.smooth_step_clip)

        mixed = 0.50 * parent_candidate + 0.50 * smooth_template
        hard = parent_candidate.copy()
        # Late-session hard push keeps the attack damaging under defense.
        if phase >= 0.55:
            hard[local_idx] += np.asarray((0.010 + 0.012 * phase, 0.024 + 0.018 * phase, 0.016 + 0.014 * phase), dtype=np.float32) * self.deadline_gain
        saturating = parent_candidate.copy()
        if phase >= 0.70:
            saturating[local_idx] = np.maximum(
                saturating[local_idx],
                np.asarray((0.030, 0.060, 0.046), dtype=np.float32),
            )

        candidates = [parent_candidate, base, smooth_template, mixed, hard, saturating]
        scored = []
        for cand in candidates:
            cand = self._bounded_delta(cand)
            st = self._candidate_stats(key, obs_vec, cand)
            scored.append((cand, st))
        best_damage = max(st['damage'] for _, st in scored)
        if best_damage <= 1e-6:
            # If the actor-action surrogate is flat, choose the parent damage anchor.
            chosen = parent_candidate
            chosen_action = self._actor_action_value(obs_vec + self._bounded_delta(chosen))
        else:
            strong = [(cand, st) for cand, st in scored if st['damage'] >= self.damage_tolerance * best_damage]
            # Among similarly damaging candidates, choose the stealthiest one.
            cand, st = max(strong, key=lambda x: x[1]['stealth_score'])
            chosen = cand
            chosen_action = st['adv_action']
        self._record_realized_action(key, chosen_action)
        return self._bounded_delta(chosen)


class FullPipelineAdaptiveDeadlineAttacker(DamageFirstDefenseAwareDeadlineDriftPGDAttacker):
    """White-box adaptive deadline attack against the full selected defense.

    Candidate perturbations are not ranked only by a raw actor or hand-written
    surrogate.  For each candidate, the attacker explicitly runs a shadow copy
    of the selected defense stack

        DAE -> posterior detector routing -> UG-BCR urgency gate -> Temporal Shield -> Actor

    and selects the candidate that most suppresses the post-defense actor action
    while mildly penalizing detector routing, belief-branch activation, temporal
    corrections, and perturbation norm.  The shadow defense uses only observation
    history, previous defended actions, current time/station context, and known
    model parameters; it does not inspect future labels or true future outcomes.
    """

    def __init__(
        self,
        base_attacker: PGDStateAttacker,
        *,
        epsilon: float = 0.075,
        drift_decay: float = 0.965,
        step_scale: float = 1.18,
        passive_decay: float = 0.98,
        deadline_gain: float = 1.70,
        damage_tolerance: float = 0.70,
        stealth_weight: float = 0.08,
        urgency_weight: float = 0.04,
        smooth_step_clip: float = 0.045,
        post_action_weight: float = 1.00,
        route_penalty: float = 0.035,
        belief_penalty: float = 0.055,
        temporal_penalty: float = 0.080,
        norm_penalty: float = 0.018,
    ) -> None:
        super().__init__(
            base_attacker,
            epsilon=epsilon,
            drift_decay=drift_decay,
            step_scale=step_scale,
            passive_decay=passive_decay,
            deadline_gain=deadline_gain,
            damage_tolerance=damage_tolerance,
            stealth_weight=stealth_weight,
            urgency_weight=urgency_weight,
            smooth_step_clip=smooth_step_clip,
        )
        self.name = 'full_pipeline_adaptive_deadline'
        self.algorithm = self.name
        self.post_action_weight = float(post_action_weight)
        self.route_penalty = float(route_penalty)
        self.belief_penalty = float(belief_penalty)
        self.temporal_penalty = float(temporal_penalty)
        self.norm_penalty = float(norm_penalty)
        self._target_cfg: dict[str, Any] | None = None
        self._target_ready = False
        self._shadow_prev_observed: dict[int, np.ndarray] = {}
        self._shadow_prev_policy: dict[int, np.ndarray] = {}
        self._shadow_prev_action: dict[int, np.ndarray] = {}
        self._shadow_prev_time: dict[int, int] = {}
        self._shadow_dae_runtime = None
        self._shadow_belief = None
        self._shadow_gate = None
        self._shadow_env = None
        self._candidate_log: list[dict[str, float]] = []

    def configure_target_defense(
        self,
        *,
        defender,
        detector_model,
        detector_threshold: float,
        shield_config,
        ug_bcr_config,
        reward_profile,
        signals_path,
        device,
        actor=None,
        repair_mode: str = 'core_only',
    ) -> None:
        self._target_cfg = dict(
            defender=defender,
            detector_model=detector_model,
            detector_threshold=float(detector_threshold),
            shield_config=shield_config,
            ug_bcr_config=ug_bcr_config,
            reward_profile=reward_profile,
            signals_path=signals_path,
            device=device,
            actor=self.base_attacker.actor if actor is None else actor,
            repair_mode=str(repair_mode or 'core_only').strip().lower().replace('-', '_'),
        )
        self._target_ready = True
        self._reset_shadow_pipeline()

    def _reset_shadow_pipeline(self) -> None:
        self._shadow_prev_observed = {}
        self._shadow_prev_policy = {}
        self._shadow_prev_action = {}
        self._shadow_prev_time = {}
        self._candidate_log = []
        if not self._target_ready or self._target_cfg is None:
            self._shadow_dae_runtime = None
            self._shadow_belief = None
            self._shadow_gate = None
            self._shadow_env = None
            return
        from .defense import SequentialDAERuntime
        from .merged_core import ChargingEnv
        from .ug_bcr import BeliefCoreEstimator, UrgencyGatedBeliefSelector

        cfg = self._target_cfg
        self._shadow_dae_runtime = SequentialDAERuntime(cfg['defender'], cfg['device'])
        self._shadow_belief = BeliefCoreEstimator(cfg['ug_bcr_config'].belief)
        self._shadow_gate = UrgencyGatedBeliefSelector(cfg['ug_bcr_config'].urgency_gate)
        self._shadow_env = ChargingEnv(signals_path=cfg['signals_path'], reward_profile=cfg['reward_profile'])
        self._shadow_env.reset()

    def reset(self) -> None:
        super().reset()
        self._reset_shadow_pipeline()

    def clone(self):
        cloned = FullPipelineAdaptiveDeadlineAttacker(
            self.base_attacker.clone(),
            epsilon=self.epsilon,
            drift_decay=self.drift_decay,
            step_scale=self.step_scale,
            passive_decay=self.passive_decay,
            deadline_gain=self.deadline_gain,
            damage_tolerance=self.damage_tolerance,
            stealth_weight=self.stealth_weight,
            urgency_weight=self.urgency_weight,
            smooth_step_clip=self.smooth_step_clip,
            post_action_weight=self.post_action_weight,
            route_penalty=self.route_penalty,
            belief_penalty=self.belief_penalty,
            temporal_penalty=self.temporal_penalty,
            norm_penalty=self.norm_penalty,
        )
        if self._target_cfg is not None:
            cloned.configure_target_defense(**self._target_cfg)
        return cloned

    def _clone_dae_runtime(self):
        if self._shadow_dae_runtime is None:
            return None
        from .defense import SequentialDAERuntime
        rt = SequentialDAERuntime(self._shadow_dae_runtime.model, self._shadow_dae_runtime.device)
        rt.buffers = defaultdict(lambda: deque(maxlen=rt.seq_len))
        for key, buf in self._shadow_dae_runtime.buffers.items():
            rt.buffers[key] = deque([np.asarray(x, dtype=np.float32).copy() for x in buf], maxlen=rt.seq_len)
        return rt

    def _shadow_ready(self) -> bool:
        return bool(self._target_ready and self._target_cfg is not None and self._shadow_env is not None)

    def _actor_action_on_state(self, state: np.ndarray) -> float:
        cfg = self._target_cfg or {}
        actor = cfg.get('actor', self.base_attacker.actor)
        device = cfg.get('device', getattr(self.base_attacker, 'device', torch.device('cpu')))
        with torch.no_grad():
            st = torch.as_tensor(to_numpy_1d(state).astype(np.float32), dtype=torch.float32, device=device).reshape(1, -1)
            act = actor(st).reshape(-1)
        return float(act.detach().cpu().numpy()[0])

    def _target_pipeline_eval(
        self,
        *,
        key: tuple[int, int],
        obs: np.ndarray,
        delta: np.ndarray,
        context: AttackContext,
        commit: bool = False,
    ) -> dict[str, Any]:
        if not self._shadow_ready():
            adv = self._project_obs(obs, obs + delta)
            return dict(action=self._actor_action_on_state(adv), route_flag=0.0, belief_flag=0.0, temporal=0.0, final_state=adv)

        from .offline_dae_det_temporal_shield import LOCAL_SHIELD_INDICES, _route_policy_states_core_only, _shield_single_state
        from .merged_pipeline import _route_policy_states

        cfg = self._target_cfg
        assert cfg is not None
        episode_id, vehicle_id = int(key[0]), int(key[1])
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        adv = self._project_obs(obs_vec, obs_vec + self._bounded_delta(delta)).astype(np.float32)
        # Use cloned state for candidate scoring; use live shadow state only when committing the chosen perturbation.
        dae_runtime = self._shadow_dae_runtime if commit else self._clone_dae_runtime()
        belief = self._shadow_belief if commit else deepcopy(self._shadow_belief)
        gate = self._shadow_gate if commit else deepcopy(self._shadow_gate)
        prev_observed = self._shadow_prev_observed if commit else {k: v.copy() for k, v in self._shadow_prev_observed.items()}
        prev_policy = self._shadow_prev_policy if commit else {k: v.copy() for k, v in self._shadow_prev_policy.items()}
        prev_action = self._shadow_prev_action if commit else {k: v.copy() for k, v in self._shadow_prev_action.items()}
        prev_time = self._shadow_prev_time if commit else dict(self._shadow_prev_time)
        env = self._shadow_env
        env.t = int(context.time_index)
        is_new = int(bool(context.is_new_arrival))
        route_fn = _route_policy_states_core_only if cfg.get('repair_mode', 'core_only') == 'core_only' else _route_policy_states
        prev_ref = prev_observed.get(vehicle_id, adv)
        policy_states, route_flags, det_scores = route_fn(
            [adv],
            [True],
            cfg['defender'],
            cfg['detector_model'],
            cfg['actor'],
            cfg['device'],
            route_mode='detector',
            detector_threshold=float(cfg['detector_threshold']),
            detector_feature_mode='posterior',
            time_indices=[int(context.time_index)],
            stations=[int(context.station)],
            is_new_arrivals=[is_new],
            prev_obs_refs=[prev_ref],
            vehicle_ids=[vehicle_id],
            episode_index=episode_id,
            dae_runtime=dae_runtime,
        )
        belief_states = belief.repair_batch(policy_states, [vehicle_id], [is_new], det_scores, env)
        selected_states, branches = gate.select_batch(
            policy_states,
            belief_states,
            [vehicle_id],
            [is_new],
            belief,
            cfg['shield_config'],
            env,
            cfg['reward_profile'],
            prev_policy,
            prev_action,
            prev_time,
        )
        policy_vec = to_numpy_1d(selected_states[0]).astype(np.float32)
        corrected, flags = _shield_single_state(
            policy_vec,
            prev_policy.get(vehicle_id),
            prev_action.get(vehicle_id),
            prev_time.get(vehicle_id),
            cfg['shield_config'],
            env,
            is_new_arrival=bool(is_new),
        )
        final_state = to_numpy_1d(corrected).astype(np.float32)
        action = self._actor_action_on_state(final_state)
        temporal = float(np.max(np.abs(final_state[list(LOCAL_SHIELD_INDICES)] - policy_vec[list(LOCAL_SHIELD_INDICES)])))
        if commit:
            prev_policy[vehicle_id] = final_state.copy()
            prev_observed[vehicle_id] = adv.copy()
            prev_action[vehicle_id] = np.asarray([action], dtype=np.float32)
            prev_time[vehicle_id] = int(context.time_index)
            if hasattr(belief, 'update_actions'):
                belief.update_actions([vehicle_id], np.asarray([[action]], dtype=np.float32), int(context.time_index))
        return dict(
            action=float(action),
            route_flag=float(bool(route_flags[0])),
            belief_flag=float(branches[0] == 'belief'),
            temporal=float(temporal),
            final_state=final_state,
            det_score=float(np.asarray(det_scores).reshape(-1)[0]) if np.asarray(det_scores).size else float('nan'),
            shield_soc=float(flags.get('soc', 0)),
            shield_time=float(flags.get('time', 0)),
            shield_cost=float(flags.get('cost', 0)),
        )

    def _pipeline_candidate_score(self, key: tuple[int, int], obs: np.ndarray, delta: np.ndarray, context: AttackContext) -> tuple[float, dict[str, Any]]:
        delta = self._bounded_delta(delta)
        info = self._target_pipeline_eval(key=key, obs=obs, delta=delta, context=context, commit=False)
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        phase = self._deadline_phase(obs_vec)
        soc_need = float(np.clip(0.90 - float(obs_vec[0]), 0.0, 1.0))
        risk = float(phase * soc_need)
        norm = float(np.linalg.norm(delta[list(LOCAL_ATTACK_IDX)], ord=2))
        # Lower post-defense action means less charging / more discharge.  The small
        # penalties keep the search from relying only on crude detector-triggering jumps.
        score = (
            self.post_action_weight * (-float(info['action']))
            + 0.20 * risk
            - self.route_penalty * float(info.get('route_flag', 0.0))
            - self.belief_penalty * float(info.get('belief_flag', 0.0))
            - self.temporal_penalty * float(info.get('temporal', 0.0))
            - self.norm_penalty * norm
        )
        info['score'] = float(score)
        info['norm'] = norm
        return float(score), info

    def _full_pipeline_candidate_set(self, key: tuple[int, int], obs: np.ndarray, base_delta: np.ndarray, context: AttackContext) -> list[np.ndarray]:
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        prev_delta = self._prev_delta(key)
        step = self._step_count(key)
        phase = self._deadline_phase(obs_vec)
        local_idx = self.attack_indices_for_local()
        parent_candidate = super()._shape_delta(key, obs_vec, base_delta, context)
        # super()._shape_delta records a realized action; this class commits its own
        # chosen target-pipeline action below, so the parent update is harmless but not used.
        low_soc = float(np.clip((0.68 - float(obs_vec[0])) / 0.68, 0.0, 1.0))
        ramp = float(min(1.0, 0.38 + 0.11 * step + 0.24 * phase + 0.15 * low_soc))
        base = self.drift_decay * prev_delta + self.step_scale * base_delta
        base[local_idx] = np.clip(base[local_idx], prev_delta[local_idx] - self.smooth_step_clip, prev_delta[local_idx] + self.smooth_step_clip)
        bias = np.zeros_like(obs_vec, dtype=np.float32)
        bias[0] = (0.020 + 0.020 * phase + 0.012 * low_soc) * self.deadline_gain
        bias[1] = (0.042 + 0.024 * phase + 0.014 * low_soc) * self.deadline_gain
        bias[10] = (0.026 + 0.016 * phase) * self.deadline_gain
        smooth_template = self.drift_decay * prev_delta + ramp * bias
        smooth_template[local_idx] = np.clip(smooth_template[local_idx], prev_delta[local_idx] - self.smooth_step_clip, prev_delta[local_idx] + self.smooth_step_clip)
        mixed = 0.55 * parent_candidate + 0.45 * smooth_template
        anti_belief = smooth_template.copy()
        # Try to make policy urgency and belief urgency closer: do not overstate SOC as
        # aggressively as time/cost, which often reduces UG-BCR belief switching while still
        # suppressing late charging.
        anti_belief[0] = 0.65 * anti_belief[0] + 0.35 * prev_delta[0]
        hard = parent_candidate.copy()
        if phase >= 0.50:
            hard[local_idx] += np.asarray((0.010 + 0.010 * phase, 0.028 + 0.020 * phase, 0.018 + 0.014 * phase), dtype=np.float32) * self.deadline_gain
        saturating = parent_candidate.copy()
        if phase >= 0.68:
            saturating[local_idx] = np.maximum(saturating[local_idx], np.asarray((0.032, 0.062, 0.048), dtype=np.float32))
        # Keep the search compact enough for full 5-seed evaluation: each
        # candidate is still evaluated through the full target defense pipeline.
        return [parent_candidate, smooth_template, anti_belief, hard if phase < 0.68 else saturating]

    def _shape_delta(self, key: tuple[int, int], obs: np.ndarray, base_delta: np.ndarray, context: AttackContext) -> np.ndarray:
        obs_vec = to_numpy_1d(obs).astype(np.float32)
        if not self._shadow_ready():
            return super()._shape_delta(key, obs_vec, base_delta, context)
        candidates = [self._bounded_delta(c) for c in self._full_pipeline_candidate_set(key, obs_vec, base_delta, context)]
        best_delta = candidates[0]
        best_score = -float('inf')
        best_info: dict[str, Any] = {}
        for cand in candidates:
            score, info = self._pipeline_candidate_score(key, obs_vec, cand, context)
            if score > best_score:
                best_score = score
                best_delta = cand
                best_info = info
        # Commit the chosen candidate to the internal target-defense shadow state.
        commit_info = self._target_pipeline_eval(key=key, obs=obs_vec, delta=best_delta, context=context, commit=True)
        chosen_action = float(commit_info.get('action', best_info.get('action', self._actor_action_value(obs_vec + best_delta))))
        self._record_realized_action(key, chosen_action)
        self._candidate_log.append(
            dict(
                score=float(best_score),
                action=float(chosen_action),
                route=float(commit_info.get('route_flag', best_info.get('route_flag', 0.0))),
                belief=float(commit_info.get('belief_flag', best_info.get('belief_flag', 0.0))),
                temporal=float(commit_info.get('temporal', best_info.get('temporal', 0.0))),
            )
        )
        return self._bounded_delta(best_delta)


def build_long_horizon_attacker(
    name: str,
    *,
    actor,
    device,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    critic=None,
    seed: int = 42,
    attack_state_scope: str | None = None,
    attack_overrides: Mapping[str, Any] | None = None,
):
    canonical_name = canonical_long_horizon_attack_name(name)
    overrides = dict(attack_overrides or {})
    local_deadline_override_keys = {'base_epsilon', 'base_alpha', 'base_iters', 'epsilon'}
    if canonical_name == 'module_aware_cem_mpc':
        from .module_aware_attacks import CEMMPCConfig, ModuleAwareCEMMPCAttacker

        outer_override_keys = {'epsilon', 'passive_decay', 'base_epsilon', 'base_alpha', 'base_iters'}
        cfg = CEMMPCConfig.from_overrides({k: v for k, v in overrides.items() if k not in outer_override_keys})
        state_scope = attack_state_scope
        if state_scope is None:
            state_scope = 'global' if cfg.objective == 'economic' else 'local'
        base_attacker = build_state_attacker(
            actor,
            device=device,
            algorithm='opposite_pgd',
            epsilon=float(overrides.get('base_epsilon', min(0.025, float(overrides.get('epsilon', 0.075))))),
            alpha=float(overrides.get('base_alpha', 0.006)),
            iters=int(overrides.get('base_iters', 1)),
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_state_scope=state_scope,
        )
        return ModuleAwareCEMMPCAttacker(
            base_attacker,
            epsilon=float(overrides.get('epsilon', 0.075)),
            passive_decay=float(overrides.get('passive_decay', 0.98)),
            attack_state_scope=state_scope,
            config=cfg,
        )
    unknown_override_keys = sorted(set(overrides) - local_deadline_override_keys)
    if unknown_override_keys:
        raise ValueError(f'Unsupported long-horizon attack override keys: {unknown_override_keys}')
    if overrides and canonical_name != 'local_deadline_drift_pgd':
        raise ValueError(
            'attack_overrides are currently supported only for local_deadline_drift_pgd; '
            f'got {canonical_name!r}.'
        )
    if canonical_name == 'local_small_drift_q':
        if critic is None:
            raise ValueError('local_small_drift_q requires a critic.')
        base_attacker = build_state_attacker(
            actor,
            device=device,
            algorithm='q_function',
            epsilon=0.03,
            alpha=0.010,
            iters=5,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=critic,
            attack_state_scope='local',
        )
        return LongHorizonSmallDriftQAttacker(
            base_attacker,
            epsilon=0.055,
            drift_decay=0.86,
            step_scale=1.02,
            passive_decay=0.92,
        )
    if canonical_name in {'temporal_shift_attack', 'local_temporal_shift_attack'}:
        state_scope = 'local' if canonical_name == 'local_temporal_shift_attack' else 'all'
        base_attacker = build_state_attacker(
            actor,
            device=device,
            algorithm='q_function' if critic is not None else 'opposite_pgd',
            epsilon=0.026,
            alpha=0.006,
            iters=5,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=critic,
            attack_state_scope=state_scope,
        )
        return LongHorizonTemporalShiftAttacker(
            base_attacker,
            name=canonical_name,
            attack_state_scope=state_scope,
            epsilon=0.055,
            passive_decay=0.96,
            drift_decay=0.88,
        )
    if canonical_name == 'local_deadline_drift_pgd':
        base_epsilon = float(overrides.get('base_epsilon', 0.028))
        base_alpha = float(overrides.get('base_alpha', 0.008))
        base_iters = int(overrides.get('base_iters', 5))
        outer_epsilon = float(overrides.get('epsilon', 0.055))
        if not np.isfinite(base_epsilon) or base_epsilon <= 0.0:
            raise ValueError('local_deadline_drift_pgd base_epsilon must be finite and positive.')
        if not np.isfinite(base_alpha) or base_alpha <= 0.0:
            raise ValueError('local_deadline_drift_pgd base_alpha must be finite and positive.')
        if base_iters <= 0:
            raise ValueError('local_deadline_drift_pgd base_iters must be positive.')
        if not np.isfinite(outer_epsilon) or outer_epsilon <= 0.0:
            raise ValueError('local_deadline_drift_pgd epsilon must be finite and positive.')
        base_attacker = build_state_attacker(
            actor,
            device=device,
            algorithm='opposite_pgd',
            epsilon=base_epsilon,
            alpha=base_alpha,
            iters=base_iters,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_state_scope='local',
        )
        return LongHorizonLocalDeadlineDriftPGDAttacker(
            base_attacker,
            epsilon=outer_epsilon,
            drift_decay=0.95,
            step_scale=1.04,
            passive_decay=0.97,
            deadline_gain=1.35,
        )
    if canonical_name == 'defense_aware_deadline_drift_pgd':
        base_attacker = build_state_attacker(
            actor,
            device=device,
            algorithm='opposite_pgd',
            epsilon=0.028,
            alpha=0.008,
            iters=6,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_state_scope='local',
        )
        return DefenseAwareDeadlineDriftPGDAttacker(
            base_attacker,
            epsilon=0.055,
            drift_decay=0.975,
            step_scale=1.00,
            passive_decay=0.985,
            deadline_gain=1.25,
            stealth_weight=0.80,
            action_weight=1.20,
            urgency_weight=0.25,
            smooth_step_clip=0.018,
        )
    if canonical_name == 'defense_aware_deadline_drift_pgd_v2':
        base_attacker = build_state_attacker(
            actor,
            device=device,
            algorithm='opposite_pgd',
            epsilon=0.036,
            alpha=0.010,
            iters=7,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_state_scope='local',
        )
        return DamageFirstDefenseAwareDeadlineDriftPGDAttacker(
            base_attacker,
            epsilon=0.075,
            drift_decay=0.965,
            step_scale=1.18,
            passive_decay=0.98,
            deadline_gain=1.70,
            damage_tolerance=0.82,
            stealth_weight=0.18,
            urgency_weight=0.08,
            smooth_step_clip=0.045,
        )
    if canonical_name == 'defense_aware_deadline_drift_pgd_v2_calibrated':
        base_attacker = build_state_attacker(
            actor,
            device=device,
            algorithm='opposite_pgd',
            epsilon=0.028,
            alpha=0.008,
            iters=6,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_state_scope='local',
        )
        return DamageFirstDefenseAwareDeadlineDriftPGDAttacker(
            base_attacker,
            epsilon=0.055,
            drift_decay=0.965,
            step_scale=1.08,
            passive_decay=0.98,
            deadline_gain=1.35,
            damage_tolerance=0.70,
            stealth_weight=0.35,
            urgency_weight=0.18,
            smooth_step_clip=0.030,
        )
    if canonical_name == 'full_pipeline_adaptive_deadline':
        base_attacker = build_state_attacker(
            actor,
            device=device,
            algorithm='opposite_pgd',
            epsilon=0.036,
            alpha=0.010,
            iters=7,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_state_scope='local',
        )
        return FullPipelineAdaptiveDeadlineAttacker(
            base_attacker,
            epsilon=0.075,
            drift_decay=0.965,
            step_scale=1.18,
            passive_decay=0.98,
            deadline_gain=1.70,
            damage_tolerance=0.70,
            stealth_weight=0.08,
            urgency_weight=0.04,
            smooth_step_clip=0.045,
            post_action_weight=1.00,
            route_penalty=0.035,
            belief_penalty=0.055,
            temporal_penalty=0.080,
            norm_penalty=0.018,
        )
    raise ValueError(f'Unsupported long-horizon attack: {name!r}')


def describe_long_horizon_attacks() -> list[dict[str, Any]]:
    return [
        {
            'name': spec.name,
            'state_scope': spec.state_scope,
            'base_algorithm': spec.base_algorithm,
            'description': spec.description,
        }
        for spec in ATTACK_SPECS.values()
    ]


__all__ = [
    'ATTACK_SPECS',
    'LONG_HORIZON_ATTACK_NAMES',
    'OPTIONAL_EXTENDED_LONG_HORIZON_ATTACK_NAMES',
    'LongHorizonAttackSpec',
    'LongHorizonSmallDriftQAttacker',
    'LongHorizonLocalDeadlineDriftPGDAttacker',
    'LongHorizonTemporalShiftAttacker',
    'DefenseAwareDeadlineDriftPGDAttacker',
    'DamageFirstDefenseAwareDeadlineDriftPGDAttacker',
    'FullPipelineAdaptiveDeadlineAttacker',
    'build_long_horizon_attacker',
    'canonical_long_horizon_attack_name',
    'describe_long_horizon_attacks',
]
