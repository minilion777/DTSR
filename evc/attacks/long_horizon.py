from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .short_horizon import (
    AttackContext,
    GLOBAL_ATTACK_IDX,
    LOCAL_ATTACK_IDX,
    PGDStateAttacker,
    attack_indices_for_state_scope,
    build_state_attacker,
    canonical_attack_state_scope,
)
from ..merged_core import to_numpy_1d


@dataclass(frozen=True)
class LongHorizonAttackSpec:
    name: str
    state_scope: str
    base_algorithm: str
    description: str


ATTACK_SPECS: dict[str, LongHorizonAttackSpec] = {
    "local_small_drift_q": LongHorizonAttackSpec(
        name="local_small_drift_q",
        state_scope="local",
        base_algorithm="q_function",
        description="Local small-drift Q attack accumulated across time.",
    ),
    "local_deadline_drift_pgd": LongHorizonAttackSpec(
        name="local_deadline_drift_pgd",
        state_scope="local",
        base_algorithm="opposite_pgd",
        description="Local deadline-coupled PGD drift accumulated across time.",
    ),
}
LONG_HORIZON_ATTACK_NAMES = tuple(ATTACK_SPECS)


def canonical_long_horizon_attack_name(value: str | None) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "lt2": "local_small_drift_q",
        "lt_local_small_drift_q": "local_small_drift_q",
        "local_small_drift_q": "local_small_drift_q",
        "lt1": "local_deadline_drift_pgd",
        "lt_local_deadline_drift_pgd": "local_deadline_drift_pgd",
        "local_deadline_drift_pgd": "local_deadline_drift_pgd",
    }
    if token not in aliases:
        raise ValueError(f"Unsupported long-horizon attack: {value!r}")
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
    if attack_state_scope not in (None, "local"):
        raise ValueError("Retained long-horizon attacks use local state scope.")

    overrides = dict(attack_overrides or {})
    allowed_override_keys = {"base_epsilon", "base_alpha", "base_iters", "epsilon"}
    unknown_override_keys = sorted(set(overrides) - allowed_override_keys)
    if unknown_override_keys:
        raise ValueError(f"Unsupported long-horizon attack override keys: {unknown_override_keys}")
    if overrides and canonical_name != "local_deadline_drift_pgd":
        raise ValueError("attack_overrides are supported only for local_deadline_drift_pgd.")

    if canonical_name == "local_small_drift_q":
        if critic is None:
            raise ValueError("local_small_drift_q requires a critic.")
        base_attacker = build_state_attacker(
            actor,
            device=device,
            algorithm="q_function",
            epsilon=0.03,
            alpha=0.010,
            iters=5,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=critic,
            attack_state_scope="local",
        )
        return LongHorizonSmallDriftQAttacker(
            base_attacker,
            epsilon=0.055,
            drift_decay=0.86,
            step_scale=1.02,
            passive_decay=0.92,
        )

    base_epsilon = float(overrides.get("base_epsilon", 0.028))
    base_alpha = float(overrides.get("base_alpha", 0.008))
    base_iters = int(overrides.get("base_iters", 5))
    outer_epsilon = float(overrides.get("epsilon", 0.055))
    if not np.isfinite(base_epsilon) or base_epsilon <= 0.0:
        raise ValueError("base_epsilon must be finite and positive.")
    if not np.isfinite(base_alpha) or base_alpha <= 0.0:
        raise ValueError("base_alpha must be finite and positive.")
    if base_iters <= 0:
        raise ValueError("base_iters must be positive.")
    if not np.isfinite(outer_epsilon) or outer_epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive.")

    base_attacker = build_state_attacker(
        actor,
        device=device,
        algorithm="opposite_pgd",
        epsilon=base_epsilon,
        alpha=base_alpha,
        iters=base_iters,
        seed=seed,
        obs_low=obs_low,
        obs_high=obs_high,
        attack_state_scope="local",
    )
    return LongHorizonLocalDeadlineDriftPGDAttacker(
        base_attacker,
        epsilon=outer_epsilon,
        drift_decay=0.95,
        step_scale=1.04,
        passive_decay=0.97,
        deadline_gain=1.35,
    )


def describe_long_horizon_attacks() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "state_scope": spec.state_scope,
            "base_algorithm": spec.base_algorithm,
            "description": spec.description,
        }
        for spec in ATTACK_SPECS.values()
    ]


__all__ = [
    "ATTACK_SPECS",
    "LONG_HORIZON_ATTACK_NAMES",
    "LongHorizonAttackSpec",
    "LongHorizonSmallDriftQAttacker",
    "LongHorizonLocalDeadlineDriftPGDAttacker",
    "build_long_horizon_attacker",
    "canonical_long_horizon_attack_name",
    "describe_long_horizon_attacks",
]
