"""Unified short- and long-horizon attacks used by the paper experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .short_horizon import (
    ALL_ATTACK_IDX,
    GLOBAL_ATTACK_IDX,
    LOCAL_ATTACK_IDX,
    AttackAlgorithm,
    AttackContext,
    AttackScenario,
    AttackScope,
    PGDStateAttacker,
    attack_batch_by_context,
    attack_indices_for_state_scope,
    build_state_attacker,
    canonical_attack_state_scope,
    scenario_target,
)
from .long_horizon import (
    ATTACK_SPECS as LONG_HORIZON_ATTACK_SPECS,
    LONG_HORIZON_ATTACK_NAMES,
    LongHorizonAttackSpec,
    LongHorizonLocalDeadlineDriftPGDAttacker,
    LongHorizonSmallDriftQAttacker,
    build_long_horizon_attacker,
    canonical_long_horizon_attack_name,
    describe_long_horizon_attacks,
)


@dataclass(frozen=True)
class ShortHorizonAttackSpec:
    name: str
    algorithm: AttackAlgorithm
    scenario: AttackScenario
    state_scope: str = "all"


SHORT_HORIZON_ATTACK_SPECS: dict[str, ShortHorizonAttackSpec] = {
    "opposite_pgd": ShortHorizonAttackSpec("opposite_pgd", "opposite_pgd", "O"),
    "opposite_fgsm": ShortHorizonAttackSpec("opposite_fgsm", "opposite_fgsm", "O"),
    "q_function": ShortHorizonAttackSpec("q_function", "q_function", "O"),
    "electhacker_C": ShortHorizonAttackSpec("electhacker_C", "electhacker", "C"),
    "electhacker_F": ShortHorizonAttackSpec("electhacker_F", "electhacker", "F"),
    "electhacker_O": ShortHorizonAttackSpec("electhacker_O", "electhacker", "O"),
}
SHORT_HORIZON_ATTACK_NAMES = tuple(SHORT_HORIZON_ATTACK_SPECS)
ALL_PAPER_ATTACK_NAMES = SHORT_HORIZON_ATTACK_NAMES + LONG_HORIZON_ATTACK_NAMES


def build_short_horizon_attacker(
    name: str,
    *,
    actor: torch.nn.Module,
    device: torch.device,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    critic: torch.nn.Module | None = None,
    seed: int = 42,
) -> PGDStateAttacker:
    try:
        spec = SHORT_HORIZON_ATTACK_SPECS[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported short-horizon attack: {name!r}") from exc
    if spec.algorithm == "q_function" and critic is None:
        raise ValueError("q_function requires a critic.")
    return build_state_attacker(
        actor,
        device=device,
        algorithm=spec.algorithm,
        seed=seed,
        obs_low=obs_low,
        obs_high=obs_high,
        critic=critic if spec.algorithm == "q_function" else None,
        attack_state_scope=spec.state_scope,
    )


def describe_attacks() -> list[dict[str, Any]]:
    short = [
        {
            "name": spec.name,
            "horizon": "short",
            "algorithm": spec.algorithm,
            "scenario": spec.scenario,
            "state_scope": spec.state_scope,
        }
        for spec in SHORT_HORIZON_ATTACK_SPECS.values()
    ]
    long = [dict(item, horizon="long") for item in describe_long_horizon_attacks()]
    return short + long


__all__ = [
    "ALL_ATTACK_IDX",
    "ALL_PAPER_ATTACK_NAMES",
    "GLOBAL_ATTACK_IDX",
    "LOCAL_ATTACK_IDX",
    "LONG_HORIZON_ATTACK_NAMES",
    "LONG_HORIZON_ATTACK_SPECS",
    "SHORT_HORIZON_ATTACK_NAMES",
    "SHORT_HORIZON_ATTACK_SPECS",
    "AttackAlgorithm",
    "AttackContext",
    "AttackScenario",
    "AttackScope",
    "LongHorizonAttackSpec",
    "LongHorizonLocalDeadlineDriftPGDAttacker",
    "LongHorizonSmallDriftQAttacker",
    "PGDStateAttacker",
    "ShortHorizonAttackSpec",
    "attack_batch_by_context",
    "attack_indices_for_state_scope",
    "build_long_horizon_attacker",
    "build_short_horizon_attacker",
    "build_state_attacker",
    "canonical_attack_state_scope",
    "canonical_long_horizon_attack_name",
    "describe_attacks",
    "scenario_target",
]
