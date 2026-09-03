from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

from .experimental_long_horizon_v2 import (
    EXPERIMENTAL_DEADLINE_PGD_V2,
    EXPERIMENTAL_SMALL_DRIFT_Q_V2,
    MomentumSmallDriftConfig,
    StealthDeadlineConfig,
    build_experimental_long_horizon_attacker,
)


FORMAL_EXPERIMENTAL_LONG_HORIZON_ALIASES: dict[str, str] = {
    "local_small_drift_q": EXPERIMENTAL_SMALL_DRIFT_Q_V2,
    "local_deadline_drift_pgd": EXPERIMENTAL_DEADLINE_PGD_V2,
}


def canonical_formal_experimental_long_horizon_name(name: str | None) -> str | None:
    token = str(name or "").strip().lower().replace("-", "_")
    return FORMAL_EXPERIMENTAL_LONG_HORIZON_ALIASES.get(token)


def uses_formal_experimental_long_horizon(name: str | None) -> bool:
    return canonical_formal_experimental_long_horizon_name(name) is not None


def _positive_scale(value: float | None) -> float:
    scale = 1.0 if value is None else float(value)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"strength_scale must be finite and positive, got {value!r}")
    return scale


def build_formal_experimental_long_horizon_attacker(
    name: str,
    *,
    actor,
    device: torch.device,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    critic=None,
    seed: int = 42,
    strength_scale: float | None = 1.0,
    attack_state_scope: str = "local",
    attack_overrides: dict[str, Any] | None = None,
):
    """Build the v2 long-horizon attacks for formal paper reruns.

    This adapter deliberately leaves ``evc.long_horizon_attacks`` unchanged.
    The paper-facing keys keep their old names, while selected evaluation
    scripts can opt into the v2 implementations through this function.
    """

    if str(attack_state_scope).strip().lower() != "local":
        raise ValueError("formal experimental long-horizon attacks are local-scope only")
    experimental_name = canonical_formal_experimental_long_horizon_name(name)
    if experimental_name is None:
        raise ValueError(f"not a formal experimental long-horizon attack: {name!r}")

    scale = _positive_scale(strength_scale)
    overrides = dict(attack_overrides or {})

    if experimental_name == EXPERIMENTAL_SMALL_DRIFT_Q_V2:
        cfg = MomentumSmallDriftConfig()
        cfg = replace(
            cfg,
            epsilon=float(overrides.get("epsilon", cfg.epsilon * scale)),
            step_size=float(overrides.get("step_size", cfg.step_size * scale)),
            slew_limit=float(overrides.get("slew_limit", cfg.slew_limit * scale)),
            momentum=float(overrides.get("momentum", cfg.momentum)),
            current_direction_weight=float(
                overrides.get("current_direction_weight", cfg.current_direction_weight)
            ),
            drift_decay=float(overrides.get("drift_decay", cfg.drift_decay)),
            base_delta_gain=float(overrides.get("base_delta_gain", cfg.base_delta_gain)),
            action_pressure_weight=float(
                overrides.get("action_pressure_weight", cfg.action_pressure_weight)
            ),
            initial_ramp=float(overrides.get("initial_ramp", cfg.initial_ramp)),
            ramp_per_step=float(overrides.get("ramp_per_step", cfg.ramp_per_step)),
            passive_decay=float(overrides.get("passive_decay", cfg.passive_decay)),
        )
        attacker = build_experimental_long_horizon_attacker(
            experimental_name,
            actor=actor,
            critic=critic,
            device=device,
            obs_low=obs_low,
            obs_high=obs_high,
            seed=int(seed),
            config=cfg,
        )
        attacker.base_attacker.epsilon = float(overrides.get("base_epsilon", 0.030 * scale))
        attacker.base_attacker.alpha = float(overrides.get("base_alpha", 0.010 * scale))
        attacker.base_attacker.iters = int(overrides.get("base_iters", 5))
        attacker.formal_alias = "local_small_drift_q"
        attacker.formal_experimental_name = experimental_name
        return attacker

    cfg = StealthDeadlineConfig()
    cfg = replace(
        cfg,
        epsilon=float(overrides.get("epsilon", cfg.epsilon * scale)),
        base_epsilon=float(overrides.get("base_epsilon", cfg.base_epsilon * scale)),
        base_alpha=float(overrides.get("base_alpha", cfg.base_alpha * scale)),
        base_iters=int(overrides.get("base_iters", cfg.base_iters)),
        slew_limit_start=float(overrides.get("slew_limit_start", cfg.slew_limit_start * scale)),
        slew_limit_end=float(overrides.get("slew_limit_end", cfg.slew_limit_end * scale)),
        action_shift_start=float(overrides.get("action_shift_start", cfg.action_shift_start * scale)),
        action_shift_end=float(overrides.get("action_shift_end", cfg.action_shift_end)),
        minimum_onset_phase=float(
            overrides.get("minimum_onset_phase", cfg.minimum_onset_phase)
        ),
        attack_window_fraction=float(
            overrides.get("attack_window_fraction", cfg.attack_window_fraction)
        ),
        min_attack_steps=int(overrides.get("min_attack_steps", cfg.min_attack_steps)),
        max_attack_steps=int(overrides.get("max_attack_steps", cfg.max_attack_steps)),
        full_strength_fraction=float(
            overrides.get("full_strength_fraction", cfg.full_strength_fraction)
        ),
        safety_soc_min=float(overrides.get("safety_soc_min", cfg.safety_soc_min)),
        safety_soc_max=float(overrides.get("safety_soc_max", cfg.safety_soc_max)),
        passive_decay=float(overrides.get("passive_decay", cfg.passive_decay)),
        damage_target_early=float(
            overrides.get("damage_target_early", cfg.damage_target_early)
        ),
        damage_target_late=float(
            overrides.get("damage_target_late", cfg.damage_target_late)
        ),
    )
    attacker = build_experimental_long_horizon_attacker(
        experimental_name,
        actor=actor,
        critic=critic,
        device=device,
        obs_low=obs_low,
        obs_high=obs_high,
        seed=int(seed),
        config=cfg,
    )
    attacker.formal_alias = "local_deadline_drift_pgd"
    attacker.formal_experimental_name = experimental_name
    return attacker


__all__ = [
    "FORMAL_EXPERIMENTAL_LONG_HORIZON_ALIASES",
    "canonical_formal_experimental_long_horizon_name",
    "uses_formal_experimental_long_horizon",
    "build_formal_experimental_long_horizon_attacker",
]
