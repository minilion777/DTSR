from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .formal_experimental_long_horizon import (
    build_formal_experimental_long_horizon_attacker,
)
from .merged_attacks import PGDStateAttacker
from .native_backbone_attacks import build_native_attacker
from .offpolicy_backbones import load_evaluation_backbone


SUPPORTED_BACKBONES = ("ddpg", "td3", "sac", "ppo")
NATIVE_ATTACK_KEYS = (
    "opposite_pgd",
    "q_function",
    "local_small_drift_q",
    "local_deadline_drift_pgd",
)
SHORT_ATTACK_KEYS = ("opposite_pgd", "q_function")
LONG_ATTACK_KEYS = ("local_small_drift_q", "local_deadline_drift_pgd")
SHORT_EPSILON = 0.10
LONG_EPSILON = 0.055


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_artifact_layout(
    package_root: str | Path,
    algorithm: str,
    seed: int,
) -> dict[str, Path]:
    name = str(algorithm).strip().lower()
    if name not in {"td3", "sac", "ppo"}:
        raise ValueError("Native artifact layout is only for TD3/SAC/PPO retraining.")
    root = Path(package_root)
    base = root / "artifacts" / f"native_{name}_seed{int(seed)}"
    return {
        "base": base,
        "clean": base / "clean",
        "dae": base / "dae",
        "det": base / "det",
        "shield": base / "shield",
        "ug_bcr": base / "ug_bcr",
        "dtsr_results": root / "results" / f"native_{name}_dtsr_seed{int(seed)}",
    }


def default_native_bundle_path(
    package_root: str | Path,
    algorithm: str,
    seed: int,
) -> Path:
    name = str(algorithm).strip().lower()
    if int(seed) != 42:
        raise ValueError(
            "No implicit native bundle is defined outside seed=42; pass --bundle-path."
        )
    filenames = {
        "td3": "bundle_selected_ep50.pt",
        "sac": "bundle_selected_ep40.pt",
        "ppo": "bundle_selected_ep15.pt",
    }
    if name not in filenames:
        raise ValueError(f"No implicit native bundle for {algorithm!r}.")
    return (
        Path(package_root)
        / "models"
        / f"independent_{name}_seed{int(seed)}"
        / filenames[name]
    )


@dataclass(frozen=True)
class FrozenBackbone:
    algorithm: str
    bundle_path: Path
    bundle_sha256: str
    actor: nn.Module
    critic: nn.Module
    payload: dict[str, Any]
    policy_action_mode: str

    def provenance(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "bundle_path": str(self.bundle_path.resolve()),
            "bundle_sha256": self.bundle_sha256,
            "policy_action_mode": self.policy_action_mode,
            "checkpoint_episode": (self.payload.get("metadata") or {}).get(
                "checkpoint_episode"
            ),
        }


@dataclass(frozen=True)
class FrozenAttackPlan:
    algorithm: str
    config_path: Path | None
    config_sha256: str | None
    calibration_split: str
    calibration_scenario_ids: tuple[str, ...]
    selected: dict[str, dict[str, Any]]

    def profile(self, attack_key: str) -> dict[str, Any] | None:
        key = str(attack_key)
        if key not in NATIVE_ATTACK_KEYS:
            raise ValueError(f"Unsupported DTSR attack: {key!r}")
        if self.algorithm == "ddpg":
            return None
        return dict(self.selected[key]["profile"])

    def profile_id(self, attack_key: str) -> str:
        if self.algorithm == "ddpg":
            return "canonical_ddpg"
        return str(self.selected[str(attack_key)]["profile_id"])

    def provenance(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "config_path": None
            if self.config_path is None
            else str(self.config_path.resolve()),
            "config_sha256": self.config_sha256,
            "calibration_split": self.calibration_split,
            "calibration_scenario_ids": list(self.calibration_scenario_ids),
            "profile_ids": {
                key: self.profile_id(key) for key in NATIVE_ATTACK_KEYS
            },
        }


def _assert_deterministic_actor(actor: nn.Module, algorithm: str) -> None:
    try:
        parameter = next(actor.parameters())
        device = parameter.device
    except StopIteration:
        device = torch.device("cpu")
    obs_dim = int(getattr(actor, "obs_dim", 11))
    probe = torch.linspace(0.0, 1.0, steps=2 * obs_dim, device=device).reshape(2, obs_dim)
    with torch.no_grad():
        first = actor(probe)
        second = actor(probe)
    if isinstance(first, (tuple, list)) or isinstance(second, (tuple, list)):
        raise TypeError(
            f"{algorithm} DTSR policy must return deterministic action tensors, not tuples."
        )
    if first.shape != second.shape or not torch.equal(first, second):
        raise RuntimeError(f"{algorithm} DTSR policy is not deterministic during evaluation.")
    if not bool(torch.isfinite(first).all()):
        raise RuntimeError(f"{algorithm} DTSR policy produced non-finite actions.")


def load_frozen_backbone(
    algorithm: str,
    bundle_path: str | Path,
    device: torch.device,
) -> FrozenBackbone:
    name = str(algorithm).strip().lower()
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported DTSR backbone: {algorithm!r}")
    path = Path(bundle_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actor, critic, payload = load_evaluation_backbone(name, path, device)
    actor.eval()
    critic.eval()
    for module in (actor, critic):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    _assert_deterministic_actor(actor, name)
    return FrozenBackbone(
        algorithm=name,
        bundle_path=path,
        bundle_sha256=sha256_file(path),
        actor=actor,
        critic=critic,
        payload=payload,
        policy_action_mode="tanh_mean" if name in {"sac", "ppo"} else "deterministic_actor",
    )


def _effective_profile_epsilon(attack_key: str, profile: dict[str, Any]) -> float:
    if str(profile.get("kind", "pointwise")) == "pointwise":
        return float(profile["epsilon"])
    overrides = dict(profile.get("attack_overrides") or {})
    if "epsilon" not in overrides:
        raise ValueError(f"Long-horizon profile {attack_key!r} lacks an epsilon override.")
    return float(overrides["epsilon"])


def load_frozen_attack_plan(
    algorithm: str,
    config_path: str | Path | None,
) -> FrozenAttackPlan:
    name = str(algorithm).strip().lower()
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(f"Unsupported DTSR backbone: {algorithm!r}")
    if name == "ddpg":
        return FrozenAttackPlan(
            algorithm=name,
            config_path=None,
            config_sha256=None,
            calibration_split="canonical",
            calibration_scenario_ids=(),
            selected={},
        )
    if config_path is None:
        raise ValueError(f"{name} requires --native-config.")
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = str(payload.get("calibration_split", "")).strip().lower()
    if split != "val":
        raise ValueError(
            f"Native attack configuration must be calibrated on val, got {split!r}."
        )
    scenario_ids = tuple(
        str(x)
        for x in (
            payload.get("calibration_scenario_ids")
            or payload.get("validation_scenario_ids")
            or payload.get("scenario_ids")
            or ()
        )
    )
    if not scenario_ids or any(not scenario_id.startswith("val_") for scenario_id in scenario_ids):
        raise ValueError("Native attack calibration provenance must contain only val scenarios.")
    selected_all = dict(payload.get("selected") or {})
    if name not in selected_all:
        raise ValueError(f"Native attack configuration has no selection for {name!r}.")
    selected = dict(selected_all[name])
    missing = [key for key in NATIVE_ATTACK_KEYS if key not in selected]
    if missing:
        raise ValueError(f"Native attack configuration is missing profiles: {missing}")
    for key in NATIVE_ATTACK_KEYS:
        record = dict(selected[key])
        profile = dict(record.get("profile") or {})
        if not record.get("profile_id") or not profile:
            raise ValueError(f"Invalid frozen profile record for {name}/{key}.")
        expected = SHORT_EPSILON if key in SHORT_ATTACK_KEYS else LONG_EPSILON
        observed = _effective_profile_epsilon(key, profile)
        if not np.isclose(observed, expected, atol=1e-9, rtol=0.0):
            raise ValueError(
                f"Frozen {name}/{key} epsilon mismatch: expected={expected}, observed={observed}"
            )
    return FrozenAttackPlan(
        algorithm=name,
        config_path=path,
        config_sha256=sha256_file(path),
        calibration_split=split,
        calibration_scenario_ids=scenario_ids,
        selected=selected,
    )


def build_frozen_attacker(
    attack_key: str,
    *,
    backbone: FrozenBackbone,
    attack_plan: FrozenAttackPlan,
    device: torch.device,
    obs_low: np.ndarray,
    obs_high: np.ndarray,
    seed: int,
):
    key = str(attack_key)
    if backbone.algorithm != attack_plan.algorithm:
        raise ValueError(
            f"Backbone/attack-plan mismatch: {backbone.algorithm} != {attack_plan.algorithm}"
        )
    if key not in NATIVE_ATTACK_KEYS:
        raise ValueError(f"Unsupported DTSR attack: {key!r}")
    if backbone.algorithm != "ddpg":
        return build_native_attacker(
            key,
            attack_plan.profile(key) or {},
            actor=backbone.actor,
            critic=backbone.critic,
            device=device,
            obs_low=obs_low,
            obs_high=obs_high,
            seed=int(seed),
        )
    if key in SHORT_ATTACK_KEYS:
        return PGDStateAttacker(
            backbone.actor,
            device=device,
            algorithm=key,
            epsilon=SHORT_EPSILON,
            alpha=0.01,
            iters=10,
            seed=int(seed),
            obs_low=obs_low,
            obs_high=obs_high,
            critic=backbone.critic if key == "q_function" else None,
            attack_state_scope="all",
        )
    return build_formal_experimental_long_horizon_attacker(
        key,
        actor=backbone.actor,
        critic=backbone.critic,
        device=device,
        obs_low=obs_low,
        obs_high=obs_high,
        seed=int(seed),
        attack_state_scope="local",
    )


def validate_dataset_backbone(
    metadata: dict[str, Any] | None,
    backbone: FrozenBackbone,
    *,
    split: str,
) -> None:
    payload = dict(metadata or {})
    recorded = dict(payload.get("backbone") or {})
    if not recorded:
        if backbone.algorithm == "ddpg":
            return
        raise ValueError(
            f"{split} clean dataset has no backbone provenance; regenerate it for "
            f"{backbone.algorithm}."
        )
    if str(recorded.get("algorithm", "")).lower() != backbone.algorithm:
        raise ValueError(
            f"{split} clean dataset backbone mismatch: "
            f"{recorded.get('algorithm')!r} != {backbone.algorithm!r}"
        )
    recorded_sha = str(recorded.get("bundle_sha256", ""))
    if recorded_sha != backbone.bundle_sha256:
        raise ValueError(
            f"{split} clean dataset bundle hash mismatch; regenerate clean trajectories."
        )
    recorded_split = str(payload.get("split", split)).lower()
    if recorded_split != str(split).lower():
        raise ValueError(
            f"Clean dataset split mismatch: recorded={recorded_split!r}, expected={split!r}"
        )


def validate_attack_plan_provenance(
    metadata: dict[str, Any] | None,
    attack_plan: FrozenAttackPlan,
) -> None:
    if attack_plan.algorithm == "ddpg":
        return
    recorded = dict((metadata or {}).get("attack_plan") or {})
    if not recorded:
        raise ValueError("Artifact has no frozen native-attack provenance.")
    if str(recorded.get("algorithm", "")).lower() != attack_plan.algorithm:
        raise ValueError("Artifact native-attack algorithm does not match the backbone.")
    if str(recorded.get("config_sha256", "")) != str(attack_plan.config_sha256):
        raise ValueError("Artifact native-attack config hash mismatch.")
    recorded_profiles = dict(recorded.get("profile_ids") or {})
    expected_profiles = dict(attack_plan.provenance()["profile_ids"])
    if recorded_profiles != expected_profiles:
        raise ValueError("Artifact native-attack profile IDs do not match the frozen plan.")


__all__ = [
    "SUPPORTED_BACKBONES",
    "NATIVE_ATTACK_KEYS",
    "SHORT_ATTACK_KEYS",
    "LONG_ATTACK_KEYS",
    "SHORT_EPSILON",
    "LONG_EPSILON",
    "FrozenBackbone",
    "FrozenAttackPlan",
    "sha256_file",
    "native_artifact_layout",
    "default_native_bundle_path",
    "load_frozen_backbone",
    "load_frozen_attack_plan",
    "build_frozen_attacker",
    "validate_dataset_backbone",
    "validate_attack_plan_provenance",
]
