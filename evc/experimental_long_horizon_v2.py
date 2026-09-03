from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .long_horizon_attacks import StatefulLongHorizonAttacker
from .merged_attacks import AttackContext, LOCAL_ATTACK_IDX, build_state_attacker
from .merged_core import to_numpy_1d


EXPERIMENTAL_SMALL_DRIFT_Q_V2 = "experimental_small_drift_q_v2"
EXPERIMENTAL_DEADLINE_PGD_V2 = "experimental_deadline_pgd_v2"


def _smoothstep01(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return float(x * x * (3.0 - 2.0 * x))


@dataclass(frozen=True)
class MomentumSmallDriftConfig:
    epsilon: float = 0.150
    step_size: float = 0.039
    slew_limit: float = 0.030
    momentum: float = 0.86
    current_direction_weight: float = 0.24
    drift_decay: float = 0.89
    base_delta_gain: float = 2.55
    action_pressure_weight: float = 2.50
    initial_ramp: float = 0.55
    ramp_per_step: float = 0.10
    passive_decay: float = 0.96

    def validate(self) -> None:
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if self.slew_limit <= 0.0:
            raise ValueError("slew_limit must be positive")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if not 0.0 <= self.current_direction_weight <= 1.0:
            raise ValueError("current_direction_weight must be in [0, 1]")
        if not 0.0 <= self.drift_decay < 1.0:
            raise ValueError("drift_decay must be in [0, 1)")
        if self.base_delta_gain < 0.0:
            raise ValueError("base_delta_gain must be non-negative")
        if self.action_pressure_weight < 0.0:
            raise ValueError("action_pressure_weight must be non-negative")


class ExperimentalMomentumSmallDriftQAttacker(StatefulLongHorizonAttacker):
    """Experimental cumulative Q attack with explicit temporal momentum.

    This class is deliberately separate from the production long-horizon
    attackers.  The pointwise Q attack supplies a damage direction, while an
    EMA and a per-step slew limit make the realized perturbation small and
    directionally persistent.
    """

    def __init__(self, base_attacker, *, config: MomentumSmallDriftConfig | None = None) -> None:
        self.config = config or MomentumSmallDriftConfig()
        self.config.validate()
        super().__init__(
            base_attacker,
            name=EXPERIMENTAL_SMALL_DRIFT_Q_V2,
            attack_state_scope="local",
            epsilon=float(self.config.epsilon),
            passive_decay=float(self.config.passive_decay),
        )
        self.momentum_by_key: dict[tuple[int, int], np.ndarray] = {}

    def reset(self) -> None:
        super().reset()
        self.momentum_by_key.clear()

    def clone(self):
        return ExperimentalMomentumSmallDriftQAttacker(self.base_attacker.clone(), config=self.config)

    def _action_pressure_direction(self, obs: np.ndarray) -> np.ndarray:
        actor = getattr(self.base_attacker, "actor", None)
        if actor is None or float(self.config.action_pressure_weight) <= 0.0:
            return np.zeros_like(to_numpy_1d(obs), dtype=np.float32)
        device = getattr(self.base_attacker, "device", torch.device("cpu"))
        try:
            with torch.enable_grad():
                obs_t = (
                    torch.as_tensor(to_numpy_1d(obs), dtype=torch.float32, device=device)
                    .reshape(1, -1)
                    .detach()
                    .clone()
                    .requires_grad_(True)
                )
                action = actor(obs_t).reshape(-1)[0]
                grad = torch.autograd.grad(action, obs_t, retain_graph=False, create_graph=False)[0]
        except RuntimeError:
            return np.zeros_like(to_numpy_1d(obs), dtype=np.float32)
        direction = -grad.detach().cpu().numpy().reshape(-1).astype(np.float32)
        direction = self._mask_delta(direction)
        scale = float(np.max(np.abs(direction[list(LOCAL_ATTACK_IDX)])))
        if scale <= 1e-9:
            return np.zeros_like(direction, dtype=np.float32)
        return (direction / scale).astype(np.float32)

    def _shape_delta(
        self,
        key: tuple[int, int],
        obs: np.ndarray,
        base_delta: np.ndarray,
        context: AttackContext,
    ) -> np.ndarray:
        del context
        prev = self._prev_delta(key)
        direction = self._mask_delta(base_delta)
        scale = float(np.max(np.abs(direction[list(LOCAL_ATTACK_IDX)])))
        if scale > 1e-9:
            direction = direction / scale
        else:
            direction = np.zeros_like(direction, dtype=np.float32)

        old_momentum = self.momentum_by_key.get(key)
        if old_momentum is None:
            old_momentum = np.zeros_like(direction, dtype=np.float32)
        beta = float(self.config.momentum)
        momentum = self._mask_delta(beta * old_momentum + (1.0 - beta) * direction)
        momentum = np.clip(momentum, -1.0, 1.0).astype(np.float32)
        self.momentum_by_key[key] = momentum.copy()

        current_weight = float(self.config.current_direction_weight)
        pressure = self._action_pressure_direction(obs)
        mixed_direction = (
            (1.0 - current_weight) * momentum
            + current_weight * direction
            + float(self.config.action_pressure_weight) * pressure
        )
        mixed_direction = np.clip(self._mask_delta(mixed_direction), -1.0, 1.0)
        mixed_scale = float(np.max(np.abs(mixed_direction[list(LOCAL_ATTACK_IDX)])))
        if mixed_scale > 1e-9:
            mixed_direction = mixed_direction / max(mixed_scale, 0.60)

        step_count = self._step_count(key)
        ramp = float(
            min(
                1.0,
                float(self.config.initial_ramp) + float(self.config.ramp_per_step) * step_count,
            )
        )
        direction_step = float(self.config.step_size) * ramp * mixed_direction
        drift_component = float(self.config.drift_decay) * prev + float(self.config.base_delta_gain) * base_delta
        proposal = drift_component + direction_step
        slew = float(self.config.slew_limit)
        proposal = np.clip(proposal, prev - slew, prev + slew)
        return self._bounded_delta(proposal, epsilon=float(self.config.epsilon))


@dataclass(frozen=True)
class StealthDeadlineConfig:
    epsilon: float = 0.085
    base_epsilon: float = 0.028
    base_alpha: float = 0.008
    base_iters: int = 5
    minimum_onset_phase: float = 0.52
    attack_window_fraction: float = 0.45
    min_attack_steps: int = 3
    max_attack_steps: int = 6
    full_strength_fraction: float = 0.72
    slew_limit_start: float = 0.004
    slew_limit_end: float = 0.020
    action_shift_start: float = 0.04
    action_shift_end: float = 0.70
    safety_soc_min: float = 0.17
    safety_soc_max: float = 1.0
    max_power: float = 0.07
    slice_hours: float = 0.25
    battery_capacity: float = 0.04992
    passive_decay: float = 0.98
    damage_target_early: float = -0.20
    damage_target_late: float = -0.78

    def validate(self) -> None:
        if self.epsilon <= 0.0 or self.base_epsilon <= 0.0:
            raise ValueError("epsilon budgets must be positive")
        if self.base_alpha <= 0.0 or self.base_iters <= 0:
            raise ValueError("base PGD alpha/iters must be positive")
        if not 0.0 <= self.minimum_onset_phase < 1.0:
            raise ValueError("minimum_onset_phase must be in [0, 1)")
        if not 0.0 < self.attack_window_fraction <= 1.0:
            raise ValueError("attack_window_fraction must be in (0, 1]")
        if self.min_attack_steps <= 0 or self.max_attack_steps < self.min_attack_steps:
            raise ValueError("invalid attack step window")
        if not 0.0 < self.full_strength_fraction <= 1.0:
            raise ValueError("full_strength_fraction must be in (0, 1]")
        if self.safety_soc_min >= self.safety_soc_max:
            raise ValueError("invalid SOC safety interval")


class ExperimentalStealthDeadlinePGDAttacker(StatefulLongHorizonAttacker):
    """Deadline attack with a clean prefix and non-terminal SOC constraints.

    The attack phase is normalized by each vehicle's own stay duration.  The
    perturbation is exactly zero before an adaptive late window, then ramps
    smoothly.  Candidate perturbations are rejected when their actor action
    would move the true next SOC outside the running safety interval.
    """

    def __init__(self, base_attacker, *, config: StealthDeadlineConfig | None = None) -> None:
        self.config = config or StealthDeadlineConfig()
        self.config.validate()
        super().__init__(
            base_attacker,
            name=EXPERIMENTAL_DEADLINE_PGD_V2,
            attack_state_scope="local",
            epsilon=float(self.config.epsilon),
            passive_decay=float(self.config.passive_decay),
        )
        self.initial_steps_by_key: dict[tuple[int, int], int] = {}
        self.start_index_by_key: dict[tuple[int, int], int] = {}
        self.phase_log: list[dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self.initial_steps_by_key.clear()
        self.start_index_by_key.clear()
        self.phase_log.clear()

    def clone(self):
        return ExperimentalStealthDeadlinePGDAttacker(self.base_attacker.clone(), config=self.config)

    @staticmethod
    def _remaining_steps(obs: np.ndarray) -> int:
        t_re = float(to_numpy_1d(obs)[1])
        return max(int(round(12.0 * t_re)), 1)

    def _ensure_timing(self, key: tuple[int, int], obs: np.ndarray) -> tuple[int, int, int, float]:
        remaining_steps = self._remaining_steps(obs)
        initial_steps = self.initial_steps_by_key.get(key)
        if initial_steps is None:
            initial_steps = remaining_steps
            self.initial_steps_by_key[key] = int(initial_steps)
            desired_window = int(round(float(self.config.attack_window_fraction) * initial_steps))
            desired_window = int(np.clip(desired_window, self.config.min_attack_steps, self.config.max_attack_steps))
            minimum_start = int(np.ceil(float(self.config.minimum_onset_phase) * max(initial_steps - 1, 0)))
            window_start = max(initial_steps - desired_window, 0)
            self.start_index_by_key[key] = int(max(minimum_start, window_start))

        elapsed_index = int(max(initial_steps - remaining_steps, 0))
        denom = max(initial_steps - 1, 1)
        phase = float(np.clip(elapsed_index / denom, 0.0, 1.0))
        return int(initial_steps), int(remaining_steps), elapsed_index, phase

    def _schedule_weight(self, key: tuple[int, int], obs: np.ndarray) -> tuple[float, float, int, int]:
        initial_steps, remaining_steps, elapsed_index, phase = self._ensure_timing(key, obs)
        start_index = int(self.start_index_by_key[key])
        if elapsed_index < start_index:
            return 0.0, phase, remaining_steps, start_index
        span = max((initial_steps - 1) - start_index, 1)
        local_progress = float(np.clip((elapsed_index - start_index) / span, 0.0, 1.0))
        full_at = float(self.config.full_strength_fraction)
        weight = _smoothstep01(local_progress / max(full_at, 1e-6))
        return weight, phase, remaining_steps, start_index

    def _actor_action(self, obs: np.ndarray) -> float:
        actor = self.base_attacker.actor
        device = getattr(self.base_attacker, "device", torch.device("cpu"))
        with torch.no_grad():
            obs_t = torch.as_tensor(to_numpy_1d(obs), dtype=torch.float32, device=device).reshape(1, -1)
            action = actor(obs_t).reshape(-1)
        return float(action.detach().cpu().numpy()[0])

    def _damage_target(self, phase: float, context: AttackContext) -> float:
        price_bonus = 0.08 if float(context.raw_price) < float(context.price_threshold) else 0.0
        target = (
            float(self.config.damage_target_early)
            + phase * (float(self.config.damage_target_late) - float(self.config.damage_target_early))
            - price_bonus
        )
        return float(np.clip(target, -1.0, 1.0))

    def _base_attack(
        self,
        obs_arr: np.ndarray,
        contexts: list[AttackContext],
        *,
        keys: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        if keys is None:
            raise ValueError("experimental deadline attacker requires vehicle keys")
        targets: list[float] = []
        for key, obs, context in zip(keys, obs_arr, contexts):
            weight, phase, _, _ = self._schedule_weight(key, obs)
            clean_action = self._actor_action(obs)
            damage_target = self._damage_target(phase, context)
            targets.append((1.0 - weight) * clean_action + weight * damage_target)
        return np.asarray(
            self.base_attacker.attack(obs_arr, target_actions=np.asarray(targets, dtype=np.float32).reshape(-1, 1)),
            dtype=np.float32,
        )

    def _candidate_is_safe(
        self,
        obs: np.ndarray,
        delta: np.ndarray,
        *,
        remaining_steps: int,
        clean_action: float,
        action_shift_limit: float,
    ) -> tuple[bool, float, float]:
        adv_obs = self._project_obs(obs, to_numpy_1d(obs) + delta)
        action = self._actor_action(adv_obs)
        action_shift = abs(action - clean_action)
        if action_shift > float(action_shift_limit) + 1e-8:
            return False, action, float("nan")
        soc = float(to_numpy_1d(obs)[0])
        soc_gain = float(self.config.max_power * self.config.slice_hours / self.config.battery_capacity)
        next_soc = soc + soc_gain * action
        is_terminal_decision = int(remaining_steps) <= 1
        if not is_terminal_decision:
            safe = float(self.config.safety_soc_min) <= next_soc <= float(self.config.safety_soc_max)
            return bool(safe), action, next_soc
        return True, action, next_soc

    def _shape_delta(
        self,
        key: tuple[int, int],
        obs: np.ndarray,
        base_delta: np.ndarray,
        context: AttackContext,
    ) -> np.ndarray:
        weight, phase, remaining_steps, start_index = self._schedule_weight(key, obs)
        if weight <= 1e-12:
            chosen = np.zeros_like(to_numpy_1d(obs), dtype=np.float32)
            self.phase_log.append(
                {
                    "key": key,
                    "phase": phase,
                    "weight": 0.0,
                    "remaining_steps": remaining_steps,
                    "start_index": start_index,
                    "action_shift": 0.0,
                }
            )
            return chosen

        prev = self._prev_delta(key)
        budget = max(float(self.config.epsilon) * weight, 1e-6)
        slew = (
            float(self.config.slew_limit_start)
            + weight * (float(self.config.slew_limit_end) - float(self.config.slew_limit_start))
        )
        raw = 0.90 * prev + self._mask_delta(base_delta)
        raw = np.clip(raw, prev - slew, prev + slew)
        raw = self._bounded_delta(raw, epsilon=budget)

        clean_action = self._actor_action(obs)
        damage_target = self._damage_target(phase, context)
        shift_limit = (
            float(self.config.action_shift_start)
            + weight * (float(self.config.action_shift_end) - float(self.config.action_shift_start))
        )
        candidates = [
            raw,
            self._bounded_delta(0.75 * raw + 0.25 * prev, epsilon=budget),
            self._bounded_delta(0.50 * raw + 0.50 * prev, epsilon=budget),
            self._bounded_delta(prev, epsilon=budget),
            np.zeros_like(raw, dtype=np.float32),
        ]
        best = np.zeros_like(raw, dtype=np.float32)
        best_score = float("inf")
        best_action = clean_action
        for candidate in candidates:
            safe, action, _ = self._candidate_is_safe(
                obs,
                candidate,
                remaining_steps=remaining_steps,
                clean_action=clean_action,
                action_shift_limit=shift_limit,
            )
            if not safe:
                continue
            smooth_penalty = float(np.max(np.abs(candidate - prev)))
            norm_penalty = float(np.max(np.abs(candidate)))
            score = abs(action - damage_target) + 0.18 * smooth_penalty + 0.04 * norm_penalty
            if score < best_score:
                best_score = float(score)
                best = candidate
                best_action = float(action)

        self.phase_log.append(
            {
                "key": key,
                "phase": phase,
                "weight": weight,
                "remaining_steps": remaining_steps,
                "start_index": start_index,
                "action_shift": abs(best_action - clean_action),
            }
        )
        return self._bounded_delta(best, epsilon=budget)


def build_experimental_long_horizon_attacker(
    name: str,
    *,
    actor,
    device: torch.device,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    critic=None,
    seed: int = 42,
    config: MomentumSmallDriftConfig | StealthDeadlineConfig | None = None,
):
    token = str(name).strip().lower().replace("-", "_")
    if token == EXPERIMENTAL_SMALL_DRIFT_Q_V2:
        if critic is None:
            raise ValueError("experimental small-drift Q requires a critic")
        cfg = config if isinstance(config, MomentumSmallDriftConfig) else MomentumSmallDriftConfig()
        base = build_state_attacker(
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
        return ExperimentalMomentumSmallDriftQAttacker(base, config=cfg)
    if token == EXPERIMENTAL_DEADLINE_PGD_V2:
        cfg = config if isinstance(config, StealthDeadlineConfig) else StealthDeadlineConfig()
        base = build_state_attacker(
            actor,
            device=device,
            algorithm="opposite_pgd",
            epsilon=float(cfg.base_epsilon),
            alpha=float(cfg.base_alpha),
            iters=int(cfg.base_iters),
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            attack_state_scope="local",
        )
        return ExperimentalStealthDeadlinePGDAttacker(base, config=cfg)
    raise ValueError(f"unknown experimental long-horizon attack: {name!r}")


__all__ = [
    "EXPERIMENTAL_SMALL_DRIFT_Q_V2",
    "EXPERIMENTAL_DEADLINE_PGD_V2",
    "MomentumSmallDriftConfig",
    "StealthDeadlineConfig",
    "ExperimentalMomentumSmallDriftQAttacker",
    "ExperimentalStealthDeadlinePGDAttacker",
    "build_experimental_long_horizon_attacker",
]
