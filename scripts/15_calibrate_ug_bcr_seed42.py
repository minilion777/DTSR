from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from _common import (
    PACKAGE_ROOT,
    deterministic_subset,
    load_manifest,
    load_scenario,
    resolve_device,
    write_json,
)
from dtsr_multiday_common import (
    EP100_ACTOR_PATH,
    EP100_BUNDLE_PATH,
    REPAIR_MODE,
    safe_recovery,
    set_all_seeds,
    ug_bcr_config_payload,
)

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.defense import load_dae, load_detector
from evc.merged_core import ChargingEnv, TRAIN_PROFILE
from evc.native_dtsr import (
    SUPPORTED_BACKBONES,
    build_frozen_attacker,
    default_native_bundle_path,
    load_frozen_attack_plan,
    load_frozen_backbone,
    native_artifact_layout,
    validate_attack_plan_provenance,
    validate_dataset_backbone,
)
from evc.offline_dae_det_temporal_shield import load_temporal_shield_bundle
from evc.ug_bcr import BeliefCoreConfig, UGBCRConfig, UrgencyGateConfig, rollout_episode_with_ug_bcr


ATTACK_SPECS: list[dict[str, Any]] = [
    {
        "key": "local_deadline_drift_pgd",
        "display": "deadline_pgd",
        "algorithm": "local_deadline_drift_pgd",
        "scope": "local",
        "weight": 0.50,
    },
    {
        "key": "local_small_drift_q",
        "display": "small_drift_q",
        "algorithm": "local_small_drift_q",
        "scope": "local",
        "weight": 0.20,
    },
    {"key": "opposite_pgd", "display": "PGD", "algorithm": "opposite_pgd", "scope": "all", "weight": 0.15},
    {"key": "q_function", "display": "Q", "algorithm": "q_function", "scope": "all", "weight": 0.15},
]


def configure_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), 0.0 if arr.size <= 1 else float(np.std(arr, ddof=1))


def fresh_attacker(attacker):
    clone = attacker.clone() if hasattr(attacker, "clone") else attacker
    if hasattr(clone, "reset"):
        clone.reset()
    return clone


def build_attacker(spec: dict[str, Any], backbone, attack_plan, device, arrivals, signal_path, seed: int):
    env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
    low, high = env.observation_bounds(max_duration_of_stay=max(12, int(arrivals["Duration_of_stay"].max())))
    return build_frozen_attacker(
        str(spec["algorithm"]),
        backbone=backbone,
        attack_plan=attack_plan,
        device=device,
        obs_low=low,
        obs_high=high,
        seed=int(seed),
    )


def rollout(
    arrivals,
    actor,
    signal_path,
    device,
    *,
    attacker=None,
    dae=None,
    detector_model=None,
    detector_threshold=None,
    shield_config=None,
    ug_config=None,
    enable_shield: bool = False,
    enable_ug: bool = False,
    state_scope: str = "local",
    label: str,
):
    return rollout_episode_with_ug_bcr(
        arrivals,
        actor,
        signal_path,
        device,
        TRAIN_PROFILE,
        attack_enabled=attacker is not None,
        attack_scenario="O",
        attacker=attacker,
        defender=dae,
        detector_model=detector_model,
        detector_threshold=detector_threshold,
        shield_config=shield_config,
        route_mode="none" if dae is None else "detector",
        enable_shield=bool(enable_shield),
        enable_belief=bool(enable_ug),
        enable_urgency_gate=bool(enable_ug),
        ug_bcr_config=ug_config,
        state_scope=state_scope,
        attack_scope="obs",
        label=label,
        repair_mode=REPAIR_MODE,
    )


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def candidate_configs() -> list[tuple[str, UGBCRConfig]]:
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        (
            "baseline_hysteresis",
            {"pred_weight": 0.60, "detector_gain": 0.20, "max_pred_weight": 0.82, "uncertainty_decay": 0.80},
            {"urgency_gain_threshold": 0.008, "soc_drop_threshold": 0.020, "time_drop_threshold": 0.010, "uncertainty_threshold": 0.060, "temporal_residual_threshold": 0.020, "ema_decay": 0.80, "consecutive_steps": 2},
        ),
        (
            "deadline_sensitive",
            {"pred_weight": 0.65, "detector_gain": 0.20, "max_pred_weight": 0.86, "uncertainty_decay": 0.82},
            {"urgency_gain_threshold": 0.006, "soc_drop_threshold": 0.014, "time_drop_threshold": 0.006, "uncertainty_threshold": 0.075, "temporal_residual_threshold": 0.028, "ema_decay": 0.78, "consecutive_steps": 2},
        ),
        (
            "time_sensitive",
            {"pred_weight": 0.60, "detector_gain": 0.15, "max_pred_weight": 0.84, "uncertainty_decay": 0.80},
            {"urgency_gain_threshold": 0.007, "soc_drop_threshold": 0.022, "time_drop_threshold": 0.005, "uncertainty_threshold": 0.070, "temporal_residual_threshold": 0.026, "ema_decay": 0.75, "consecutive_steps": 2},
        ),
        (
            "soc_sensitive",
            {"pred_weight": 0.62, "detector_gain": 0.20, "max_pred_weight": 0.84, "uncertainty_decay": 0.82},
            {"urgency_gain_threshold": 0.007, "soc_drop_threshold": 0.012, "time_drop_threshold": 0.011, "uncertainty_threshold": 0.068, "temporal_residual_threshold": 0.024, "ema_decay": 0.80, "consecutive_steps": 2},
        ),
        (
            "conservative_guard",
            {"pred_weight": 0.52, "detector_gain": 0.15, "max_pred_weight": 0.76, "uncertainty_decay": 0.86},
            {"urgency_gain_threshold": 0.012, "soc_drop_threshold": 0.026, "time_drop_threshold": 0.013, "uncertainty_threshold": 0.052, "temporal_residual_threshold": 0.017, "ema_decay": 0.84, "consecutive_steps": 2},
        ),
        (
            "loose_temporal",
            {"pred_weight": 0.62, "detector_gain": 0.20, "max_pred_weight": 0.86, "uncertainty_decay": 0.80},
            {"urgency_gain_threshold": 0.006, "soc_drop_threshold": 0.018, "time_drop_threshold": 0.008, "uncertainty_threshold": 0.080, "temporal_residual_threshold": 0.032, "ema_decay": 0.78, "consecutive_steps": 2},
        ),
        (
            "low_uncertainty",
            {"pred_weight": 0.58, "detector_gain": 0.18, "max_pred_weight": 0.80, "uncertainty_decay": 0.82},
            {"urgency_gain_threshold": 0.008, "soc_drop_threshold": 0.018, "time_drop_threshold": 0.008, "uncertainty_threshold": 0.050, "temporal_residual_threshold": 0.020, "ema_decay": 0.82, "consecutive_steps": 2},
        ),
        (
            "high_pred_weight",
            {"pred_weight": 0.70, "detector_gain": 0.18, "max_pred_weight": 0.88, "uncertainty_decay": 0.84},
            {"urgency_gain_threshold": 0.007, "soc_drop_threshold": 0.016, "time_drop_threshold": 0.007, "uncertainty_threshold": 0.070, "temporal_residual_threshold": 0.024, "ema_decay": 0.78, "consecutive_steps": 2},
        ),
        (
            "deadline_fast_gate",
            {"pred_weight": 0.74, "detector_gain": 0.20, "max_pred_weight": 0.92, "uncertainty_decay": 0.80},
            {"urgency_gain_threshold": 0.003, "soc_drop_threshold": 0.008, "time_drop_threshold": 0.003, "uncertainty_threshold": 0.090, "temporal_residual_threshold": 0.040, "ema_decay": 0.70, "consecutive_steps": 1},
        ),
        (
            "deadline_time_ultra",
            {"pred_weight": 0.70, "detector_gain": 0.25, "max_pred_weight": 0.90, "uncertainty_decay": 0.78},
            {"urgency_gain_threshold": 0.004, "soc_drop_threshold": 0.018, "time_drop_threshold": 0.002, "uncertainty_threshold": 0.085, "temporal_residual_threshold": 0.035, "ema_decay": 0.72, "consecutive_steps": 1},
        ),
        (
            "deadline_balanced_active",
            {"pred_weight": 0.68, "detector_gain": 0.22, "max_pred_weight": 0.90, "uncertainty_decay": 0.80},
            {"urgency_gain_threshold": 0.004, "soc_drop_threshold": 0.010, "time_drop_threshold": 0.004, "uncertainty_threshold": 0.080, "temporal_residual_threshold": 0.030, "ema_decay": 0.74, "consecutive_steps": 1},
        ),
        (
            "deadline_high_conf",
            {"pred_weight": 0.72, "detector_gain": 0.18, "max_pred_weight": 0.88, "uncertainty_decay": 0.86},
            {"urgency_gain_threshold": 0.005, "soc_drop_threshold": 0.010, "time_drop_threshold": 0.003, "uncertainty_threshold": 0.065, "temporal_residual_threshold": 0.025, "ema_decay": 0.78, "consecutive_steps": 2},
        ),
    ]
    out: list[tuple[str, UGBCRConfig]] = []
    pred_candidates = (0.55, 0.65, 0.75)
    max_pred_candidates = (0.82, 0.88, 0.92)
    innovation_candidates = ((0.70, 0.004), (0.80, 0.008), (0.90, 0.012))
    urgency_candidates = (0.004, 0.008, 0.012)
    uncertainty_candidates = (0.05, 0.07, 0.09)
    consecutive_candidates = (2, 3, 4)
    for candidate_index, (name, belief_overrides, gate_overrides) in enumerate(rows):
        innovation_decay, time_innovation_threshold = innovation_candidates[candidate_index % len(innovation_candidates)]
        belief = BeliefCoreConfig(
            enabled=True,
            pred_weight=min(pred_candidates, key=lambda value: abs(value - float(belief_overrides["pred_weight"]))),
            detector_gain=float(belief_overrides["detector_gain"]),
            max_pred_weight=min(max_pred_candidates, key=lambda value: abs(value - float(belief_overrides["max_pred_weight"]))),
            disagreement_gain=0.60,
            uncertainty_decay=float(belief_overrides["uncertainty_decay"]),
            soc_margin=0.010,
            time_margin=0.000,
            cost_margin=0.015,
            use_known_initial_soc=True,
            use_known_initial_cost=True,
            time_initialization="routed_observation",
        )
        gate = UrgencyGateConfig(
            enabled=True,
            target_soc_margin=0.000,
            urgency_gain_threshold=min(urgency_candidates, key=lambda value: abs(value - float(gate_overrides["urgency_gain_threshold"]))),
            soc_drop_threshold=float(gate_overrides["soc_drop_threshold"]),
            time_drop_threshold=float(gate_overrides["time_drop_threshold"]),
            uncertainty_threshold=min(uncertainty_candidates, key=lambda value: abs(value - float(gate_overrides["uncertainty_threshold"]))),
            temporal_residual_threshold=float(gate_overrides["temporal_residual_threshold"]),
            ema_decay=float(gate_overrides["ema_decay"]),
            innovation_ema_decay=float(innovation_decay),
            time_innovation_threshold=float(time_innovation_threshold),
            soc_innovation_threshold=float(gate_overrides.get("soc_innovation_threshold", 0.010)),
            consecutive_steps=min(consecutive_candidates, key=lambda value: abs(value - int(gate_overrides["consecutive_steps"]))),
            min_remaining_steps=1.0,
            max_remaining_steps=18.0,
        )
        out.append((name, UGBCRConfig(belief=belief, urgency_gate=gate)))

    # Targeted strict-no-leak gate-relaxation study.  These candidates keep the
    # belief propagation fixed at the published v2 setting and only relax the
    # selector thresholds.  They are opt-in through --candidate-names so the
    # original calibration grid and published artifact remain reproducible.
    def fixed_v2_belief() -> BeliefCoreConfig:
        return BeliefCoreConfig(
            enabled=True,
            pred_weight=0.65,
            obs_weight=0.28,
            detector_gain=0.25,
            max_pred_weight=0.88,
            soc_margin=0.010,
            time_margin=0.000,
            cost_margin=0.015,
            disagreement_gain=0.60,
            uncertainty_decay=0.78,
            use_known_initial_soc=True,
            use_known_initial_cost=True,
            time_initialization="routed_observation",
        )

    def gate_candidate(**overrides: Any) -> UrgencyGateConfig:
        payload: dict[str, Any] = {
            "enabled": True,
            "target_soc_margin": 0.000,
            "urgency_gain_threshold": 0.004,
            "soc_drop_threshold": 0.018,
            "time_drop_threshold": 0.002,
            "uncertainty_threshold": 0.090,
            "temporal_residual_threshold": 0.035,
            "ema_decay": 0.72,
            "innovation_ema_decay": 0.70,
            "time_innovation_threshold": 0.004,
            "soc_innovation_threshold": 0.010,
            "consecutive_steps": 2,
            "min_remaining_steps": 1.0,
            "max_remaining_steps": 18.0,
        }
        payload.update(overrides)
        return UrgencyGateConfig(**payload)

    targeted_gate_candidates: list[tuple[str, dict[str, Any]]] = [
        ("strict_v2_control", {}),
        ("relax_residual_050", {"temporal_residual_threshold": 0.050}),
        (
            "relax_drift_003",
            {
                "urgency_gain_threshold": 0.003,
                "time_drop_threshold": 0.0015,
                "time_innovation_threshold": 0.003,
                "temporal_residual_threshold": 0.050,
            },
        ),
        (
            "relax_balanced_060",
            {
                "urgency_gain_threshold": 0.003,
                "soc_drop_threshold": 0.014,
                "time_drop_threshold": 0.0015,
                "uncertainty_threshold": 0.100,
                "temporal_residual_threshold": 0.060,
                "time_innovation_threshold": 0.0025,
            },
        ),
        (
            "relax_uncertainty_110",
            {
                "urgency_gain_threshold": 0.003,
                "soc_drop_threshold": 0.014,
                "time_drop_threshold": 0.001,
                "uncertainty_threshold": 0.110,
                "temporal_residual_threshold": 0.060,
                "time_innovation_threshold": 0.0025,
            },
        ),
        (
            "relax_fast_confirm",
            {
                "urgency_gain_threshold": 0.003,
                "soc_drop_threshold": 0.014,
                "time_drop_threshold": 0.001,
                "uncertainty_threshold": 0.100,
                "temporal_residual_threshold": 0.060,
                "time_innovation_threshold": 0.0025,
                "consecutive_steps": 1,
            },
        ),
    ]
    out.extend(
        (
            name,
            UGBCRConfig(belief=fixed_v2_belief(), urgency_gate=gate_candidate(**gate_overrides)),
        )
        for name, gate_overrides in targeted_gate_candidates
    )
    return out


def build_cache(
    *,
    phase: str,
    frame: pd.DataFrame,
    actor,
    backbone,
    attack_plan,
    device,
    dae,
    detector_model,
    detector_threshold: float,
    shield_config,
    seed: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    cache: list[dict[str, Any]] = []
    print(f"[UG-BCR] {phase} baseline scenes={len(frame)} attacks={[a['key'] for a in ATTACK_SPECS]}")
    for scene_pos, row in frame.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(row)
        clean_raw = rollout(arrivals, actor, signal_path, device, label=f"{phase}__clean_raw")
        clean_shield = rollout(
            arrivals,
            actor,
            signal_path,
            device,
            dae=dae,
            detector_model=detector_model,
            detector_threshold=detector_threshold,
            shield_config=shield_config,
            enable_shield=True,
            label=f"{phase}__clean_shield",
        )
        attack_baselines: dict[str, dict[str, Any]] = {}
        for attack_idx, spec in enumerate(ATTACK_SPECS):
            attack_seed = int(seed + attack_idx * 100_000 + int(scene_pos) + 1)
            attacker = build_attacker(spec, backbone, attack_plan, device, arrivals, signal_path, attack_seed)
            raw = rollout(
                arrivals,
                actor,
                signal_path,
                device,
                attacker=fresh_attacker(attacker),
                state_scope=str(spec["scope"]),
                label=f"{phase}__{spec['key']}__attack",
            )
            shield = rollout(
                arrivals,
                actor,
                signal_path,
                device,
                attacker=fresh_attacker(attacker),
                dae=dae,
                detector_model=detector_model,
                detector_threshold=detector_threshold,
                shield_config=shield_config,
                enable_shield=True,
                state_scope=str(spec["scope"]),
                label=f"{phase}__{spec['key']}__shield",
            )
            attack_baselines[str(spec["key"])] = {"seed": attack_seed, "raw": raw, "shield": shield}
            append_csv(
                output_dir / "ug_bcr_baselines_live.csv",
                {
                    "phase": phase,
                    "scenario_id": scenario_id,
                    "attack_key": spec["key"],
                    "clean_raw_reward": float(clean_raw["ep_reward"]),
                    "attack_reward": float(raw["ep_reward"]),
                    "shield_reward": float(shield["ep_reward"]),
                    "shield_recovery_pct": safe_recovery(float(clean_raw["ep_reward"]), float(raw["ep_reward"]), float(shield["ep_reward"])) * 100.0,
                    "shield_belief_rate": float(shield.get("urgency_gate_belief_rate", 0.0)),
                },
            )
        cache.append(
            {
                "phase": phase,
                "scene_pos": int(scene_pos),
                "scenario_id": scenario_id,
                "arrivals": arrivals,
                "signal_path": signal_path,
                "clean_raw": clean_raw,
                "clean_shield": clean_shield,
                "attack_baselines": attack_baselines,
            }
        )
        print(
            f"[UG-BCR] {phase} baseline {len(cache):02d}/{len(frame)} {scenario_id} "
            f"clean_raw={float(clean_raw['ep_reward']):.2f} clean_shield={float(clean_shield['ep_reward']):.2f}"
        )
    return cache


def evaluate_candidates(
    *,
    phase: str,
    cache: list[dict[str, Any]],
    candidates: list[tuple[str, UGBCRConfig]],
    actor,
    backbone,
    attack_plan,
    device,
    dae,
    detector_model,
    detector_threshold: float,
    shield_config,
    output_dir: Path,
    clean_drop_cap: float,
    clean_drop_cap_pct: float,
    clean_activation_cap: float,
    clean_vs_shield_drop_cap: float,
    main_attack_drop_cap: float,
    deadline_improvement_floor: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    aggregate_path = output_dir / "ug_bcr_candidate_summary_live.csv"
    best_path = output_dir / "ug_bcr_best_live.json"
    if phase == "stage1" and aggregate_path.exists():
        aggregate_path.unlink()
    for candidate_index, (candidate_name, ug_config) in enumerate(candidates, start=1):
        clean_drops: list[float] = []
        clean_drop_pcts: list[float] = []
        clean_belief_rates: list[float] = []
        clean_vs_shield_deltas: list[float] = []
        clean_exit_vios: list[float] = []
        clean_run_vios: list[float] = []
        belief_rates: list[float] = []
        uncertainty_values: list[float] = []
        recovery_by_attack: dict[str, list[float]] = {str(spec["key"]): [] for spec in ATTACK_SPECS}
        shield_recovery_by_attack: dict[str, list[float]] = {str(spec["key"]): [] for spec in ATTACK_SPECS}
        print(f"[UG-BCR] {phase} candidate {candidate_index:02d}/{len(candidates)} {candidate_name}")
        for scene_idx, cached in enumerate(cache, start=1):
            clean_ug = rollout(
                cached["arrivals"],
                actor,
                cached["signal_path"],
                device,
                dae=dae,
                detector_model=detector_model,
                detector_threshold=detector_threshold,
                shield_config=shield_config,
                ug_config=ug_config,
                enable_shield=True,
                enable_ug=True,
                label=f"{phase}__{candidate_name}__clean_ug_bcr",
            )
            clean_raw_reward = float(cached["clean_raw"]["ep_reward"])
            clean_shield_reward = float(cached["clean_shield"]["ep_reward"])
            clean_drops.append(clean_raw_reward - float(clean_ug["ep_reward"]))
            clean_drop_pcts.append(
                100.0 * (clean_raw_reward - float(clean_ug["ep_reward"])) / max(abs(clean_raw_reward), 1e-8)
            )
            clean_vs_shield_deltas.append(float(clean_ug["ep_reward"]) - clean_shield_reward)
            clean_exit_vios.append(float(clean_ug.get("exit_vio", 0.0)))
            clean_run_vios.append(float(clean_ug.get("run_vio", 0.0)))
            clean_belief_rate = float(clean_ug.get("urgency_gate_belief_rate", 0.0))
            clean_belief_rates.append(clean_belief_rate)
            belief_rates.append(clean_belief_rate)
            uncertainty_values.append(float(clean_ug.get("urgency_uncertainty_mean", 0.0)))
            for spec in ATTACK_SPECS:
                attack_key = str(spec["key"])
                baseline = cached["attack_baselines"][attack_key]
                attacker = build_attacker(
                    spec,
                    backbone,
                    attack_plan,
                    device,
                    cached["arrivals"],
                    cached["signal_path"],
                    int(baseline["seed"]),
                )
                result = rollout(
                    cached["arrivals"],
                    actor,
                    cached["signal_path"],
                    device,
                    attacker=fresh_attacker(attacker),
                    dae=dae,
                    detector_model=detector_model,
                    detector_threshold=detector_threshold,
                    shield_config=shield_config,
                    ug_config=ug_config,
                    enable_shield=True,
                    enable_ug=True,
                    state_scope=str(spec["scope"]),
                    label=f"{phase}__{candidate_name}__{attack_key}__ug_bcr",
                )
                raw_reward = float(baseline["raw"]["ep_reward"])
                shield_reward = float(baseline["shield"]["ep_reward"])
                rec = safe_recovery(clean_raw_reward, raw_reward, float(result["ep_reward"])) * 100.0
                shield_rec = safe_recovery(clean_raw_reward, raw_reward, shield_reward) * 100.0
                recovery_by_attack[attack_key].append(rec)
                shield_recovery_by_attack[attack_key].append(shield_rec)
                belief_rates.append(float(result.get("urgency_gate_belief_rate", 0.0)))
                uncertainty_values.append(float(result.get("urgency_uncertainty_mean", 0.0)))
                append_csv(
                    output_dir / "ug_bcr_rollouts_live.csv",
                    {
                        "phase": phase,
                        "candidate_index": candidate_index,
                        "candidate_name": candidate_name,
                        "scenario_id": cached["scenario_id"],
                        "attack_key": attack_key,
                        "clean_reward": clean_raw_reward,
                        "attack_reward": raw_reward,
                        "shield_reward": shield_reward,
                        "ug_bcr_reward": float(result["ep_reward"]),
                        "shield_recovery_pct": shield_rec,
                        "ug_bcr_recovery_pct": rec,
                        "belief_rate": float(result.get("urgency_gate_belief_rate", 0.0)),
                        "uncertainty_mean": float(result.get("urgency_uncertainty_mean", 0.0)),
                    },
                )
            print(
                f"[UG-BCR] {phase} candidate {candidate_index:02d}/{len(candidates)} "
                f"scene {scene_idx:02d}/{len(cache)} {cached['scenario_id']} "
                f"clean_drop={clean_drops[-1]:.2f} belief={belief_rates[-1]:.3f}"
            )

        row: dict[str, Any] = {
            "phase": phase,
            "candidate_index": int(candidate_index),
            "candidate_name": candidate_name,
            "clean_drop_mean": mean_std(clean_drops)[0],
            "clean_drop_std": mean_std(clean_drops)[1],
            "clean_drop_mean_pct": mean_std(clean_drop_pcts)[0],
            "clean_activation_mean": mean_std(clean_belief_rates)[0],
            "clean_vs_shield_delta_mean": mean_std(clean_vs_shield_deltas)[0],
            "clean_exit_vio_mean": mean_std(clean_exit_vios)[0],
            "clean_run_vio_mean": mean_std(clean_run_vios)[0],
            "belief_rate_mean": mean_std(belief_rates)[0],
            "belief_rate_std": mean_std(belief_rates)[1],
            "uncertainty_mean": mean_std(uncertainty_values)[0],
        }
        score = 0.0
        for spec in ATTACK_SPECS:
            attack_key = str(spec["key"])
            rec_mean, rec_std = mean_std(recovery_by_attack[attack_key])
            shield_mean, shield_std = mean_std(shield_recovery_by_attack[attack_key])
            row[f"{attack_key}_recovery_mean_pct"] = rec_mean
            row[f"{attack_key}_recovery_std_pct"] = rec_std
            row[f"{attack_key}_shield_recovery_mean_pct"] = shield_mean
            row[f"{attack_key}_shield_recovery_std_pct"] = shield_std
            row[f"{attack_key}_delta_vs_shield_pct"] = rec_mean - shield_mean
            score += float(spec["weight"]) * (rec_mean - rec_std)
        row["weighted_lower_bound_score"] = float(score)
        row["worst_target_recovery_pct"] = float(min(
            row["local_small_drift_q_recovery_mean_pct"],
            row["local_deadline_drift_pgd_recovery_mean_pct"],
        ))
        row["feasible_clean"] = bool(
            row["clean_drop_mean"] <= clean_drop_cap
            and row["clean_drop_mean_pct"] <= clean_drop_cap_pct
            and row["clean_activation_mean"] <= clean_activation_cap
            and row["clean_vs_shield_delta_mean"] >= -float(clean_vs_shield_drop_cap)
            and row["clean_exit_vio_mean"] <= 4.0
            and row["clean_run_vio_mean"] <= 1.5
        )
        row["feasible_main_attack"] = bool(
            row["opposite_pgd_delta_vs_shield_pct"] >= -float(main_attack_drop_cap)
            and row["q_function_delta_vs_shield_pct"] >= -float(main_attack_drop_cap)
        )
        row["feasible_deadline"] = bool(row["local_deadline_drift_pgd_delta_vs_shield_pct"] >= float(deadline_improvement_floor))
        row["feasible"] = bool(row["feasible_clean"] and row["feasible_main_attack"] and row["feasible_deadline"])
        row["selection_score"] = (
            (1000.0 if row["feasible"] else 0.0)
            + float(row["worst_target_recovery_pct"])
            + 0.10 * float(row["weighted_lower_bound_score"])
            - 0.02 * max(0.0, float(row["clean_drop_mean"]))
        )
        rows.append(row)
        append_csv(aggregate_path, row)

        best_frame = pd.DataFrame(rows).sort_values(
            ["selection_score", "weighted_lower_bound_score", "clean_drop_mean"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        write_json(best_path, best_frame.iloc[0].to_dict())
        print(
            f"[UG-BCR] {phase} candidate {candidate_index:02d}/{len(candidates)} {candidate_name} "
            f"score={row['weighted_lower_bound_score']:.2f} "
            f"clean_drop={row['clean_drop_mean']:.2f}+/-{row['clean_drop_std']:.2f} "
            f"deadline={row['local_deadline_drift_pgd_recovery_mean_pct']:.1f}+/-{row['local_deadline_drift_pgd_recovery_std_pct']:.1f} "
            f"(d={row['local_deadline_drift_pgd_delta_vs_shield_pct']:+.1f}) "
            f"small={row['local_small_drift_q_recovery_mean_pct']:.1f}+/-{row['local_small_drift_q_recovery_std_pct']:.1f} "
            f"PGD={row['opposite_pgd_recovery_mean_pct']:.1f}+/-{row['opposite_pgd_recovery_std_pct']:.1f} "
            f"Q={row['q_function_recovery_mean_pct']:.1f}+/-{row['q_function_recovery_std_pct']:.1f} "
            f"belief={row['belief_rate_mean']:.3f} feasible={int(row['feasible'])}"
        )
    return pd.DataFrame(rows)


def main() -> None:
    configure_line_buffering()
    parser = argparse.ArgumentParser(description="Calibrate final UG-BCR using frozen full-state DAE, DeT, and Temporal Shield artifacts.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", choices=SUPPORTED_BACKBONES, default="ddpg")
    parser.add_argument("--actor-path", type=Path, default=EP100_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=EP100_BUNDLE_PATH)
    parser.add_argument(
        "--native-config",
        type=Path,
        default=PACKAGE_ROOT / "results" / "native_attack_calibration_seed42" / "native_attack_config.json",
    )
    parser.add_argument("--dae-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday")
    parser.add_argument("--detector-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "det_seed42_fullstate")
    parser.add_argument("--shield-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "shield_seed42_fullstate")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "ug_bcr_seed42_fullstate")
    parser.add_argument("--dtsr-dir", type=Path, default=PACKAGE_ROOT / "results" / "dtsr_retrain_seed42")
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--stage1-scenes", type=int, default=4)
    parser.add_argument("--stage2-scenes", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--clean-drop-cap", type=float, default=5.0)
    parser.add_argument("--clean-drop-cap-pct", type=float, default=2.0)
    parser.add_argument("--clean-activation-cap", type=float, default=0.02)
    parser.add_argument("--clean-vs-shield-drop-cap", type=float, default=1.0)
    parser.add_argument("--main-attack-drop-cap", type=float, default=2.0)
    parser.add_argument("--deadline-improvement-floor", type=float, default=0.0)
    parser.add_argument(
        "--candidate-names",
        type=str,
        default="",
        help="Optional comma-separated candidate names; empty evaluates the complete grid.",
    )
    parser.add_argument(
        "--publish-to-dtsr-dir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy the selected config/manifest into --dtsr-dir. Disable for non-destructive experiments.",
    )
    args = parser.parse_args()

    if args.algorithm != "ddpg":
        layout = native_artifact_layout(PACKAGE_ROOT, args.algorithm, args.seed)
        if args.bundle_path == EP100_BUNDLE_PATH:
            args.bundle_path = default_native_bundle_path(PACKAGE_ROOT, args.algorithm, args.seed)
        if args.dae_artifact_dir == PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday":
            args.dae_artifact_dir = layout["dae"]
        if args.detector_artifact_dir == PACKAGE_ROOT / "artifacts" / "det_seed42_fullstate":
            args.detector_artifact_dir = layout["det"]
        if args.shield_artifact_dir == PACKAGE_ROOT / "artifacts" / "shield_seed42_fullstate":
            args.shield_artifact_dir = layout["shield"]
        if args.output_dir == PACKAGE_ROOT / "artifacts" / "ug_bcr_seed42_fullstate":
            args.output_dir = layout["ug_bcr"]
        if args.dtsr_dir == PACKAGE_ROOT / "results" / "dtsr_retrain_seed42":
            args.dtsr_dir = layout["dtsr_results"]

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.dtsr_dir.mkdir(parents=True, exist_ok=True)
    for live_name in (
        "ug_bcr_baselines_live.csv",
        "ug_bcr_rollouts_live.csv",
        "ug_bcr_candidate_summary_live.csv",
        "ug_bcr_best_live.json",
        "ug_bcr_stage1_summary.csv",
        "ug_bcr_stage2_summary.csv",
        "ug_bcr_manifest.json",
        "ug_bcr_config.json",
    ):
        (args.output_dir / live_name).unlink(missing_ok=True)
    set_all_seeds(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)
    print(f"[UG-BCR] device={device}, seed={args.seed}, repair_mode={REPAIR_MODE}")
    print(f"[UG-BCR] stage1_scenes={args.stage1_scenes}, stage2_scenes={args.stage2_scenes}, top_k={args.top_k}")

    backbone = load_frozen_backbone(args.algorithm, args.bundle_path, device)
    attack_plan = load_frozen_attack_plan(args.algorithm, args.native_config)
    actor = backbone.actor

    dae_path = args.dae_artifact_dir / "dtsr_dae.pt"
    detector_path = args.detector_artifact_dir / "dtsr_detector.pt"
    shield_path = args.shield_artifact_dir / "dtsr_temporal_shield.pt"
    dae = load_dae(dae_path, device).eval()
    detector_artifact = load_detector(detector_path, device)
    detector_model = detector_artifact.model.eval()
    detector_threshold = float(detector_artifact.threshold)
    shield_artifact = load_temporal_shield_bundle(shield_path)
    shield_config = shield_artifact.config
    artifact_manifests = (
        ("DAE", args.dae_artifact_dir / "dae_manifest.json"),
        ("DET", args.detector_artifact_dir / "det_manifest.json"),
        ("Shield", args.shield_artifact_dir / "shield_manifest.json"),
    )
    for artifact_name, manifest_path in artifact_manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        validate_dataset_backbone(manifest, backbone, split="train")
        validate_attack_plan_provenance(manifest, attack_plan)
        print(f"[UG-BCR] verified {artifact_name} provenance for {backbone.algorithm}")
    print(f"[UG-BCR] DAE={dae_path}")
    print(f"[UG-BCR] Detector={detector_path}, threshold={detector_threshold:.6f}")
    print(f"[UG-BCR] Shield={shield_path}, tau=({shield_config.tau_soc:.4f},{shield_config.tau_time:.4f},{shield_config.tau_cost:.4f})")

    manifest = load_manifest(args.split).sort_values("Scenario_ID", kind="mergesort").reset_index(drop=True)
    stage1_frame = deterministic_subset(manifest, args.stage1_scenes, args.seed + 101)
    stage2_frame = deterministic_subset(manifest, args.stage2_scenes, args.seed + 202)

    all_candidates = candidate_configs()
    requested_candidate_names = [name.strip() for name in str(args.candidate_names).split(",") if name.strip()]
    if requested_candidate_names:
        candidate_lookup = dict(all_candidates)
        unknown_candidate_names = [name for name in requested_candidate_names if name not in candidate_lookup]
        if unknown_candidate_names:
            raise ValueError(f"Unknown UG-BCR candidate names: {unknown_candidate_names}")
        all_candidates = [(name, candidate_lookup[name]) for name in requested_candidate_names]
    write_json(
        args.output_dir / "ug_bcr_candidate_grid.json",
        {
            name: ug_bcr_config_payload(config)
            for name, config in all_candidates
        },
    )

    stage1_cache = build_cache(
        phase="stage1",
        frame=stage1_frame,
        actor=actor,
        backbone=backbone,
        attack_plan=attack_plan,
        device=device,
        dae=dae,
        detector_model=detector_model,
        detector_threshold=detector_threshold,
        shield_config=shield_config,
        seed=args.seed + 1_000,
        output_dir=args.output_dir,
    )
    stage1_summary = evaluate_candidates(
        phase="stage1",
        cache=stage1_cache,
        candidates=all_candidates,
        actor=actor,
        backbone=backbone,
        attack_plan=attack_plan,
        device=device,
        dae=dae,
        detector_model=detector_model,
        detector_threshold=detector_threshold,
        shield_config=shield_config,
        output_dir=args.output_dir,
        clean_drop_cap=float(args.clean_drop_cap),
        clean_drop_cap_pct=float(args.clean_drop_cap_pct),
        clean_activation_cap=float(args.clean_activation_cap),
        clean_vs_shield_drop_cap=float(args.clean_vs_shield_drop_cap),
        main_attack_drop_cap=float(args.main_attack_drop_cap),
        deadline_improvement_floor=float(args.deadline_improvement_floor),
    )
    stage1_summary.to_csv(args.output_dir / "ug_bcr_stage1_summary.csv", index=False, encoding="utf-8-sig")
    top_names = (
        stage1_summary.sort_values(["selection_score", "weighted_lower_bound_score"], ascending=[False, False], kind="mergesort")
        .head(max(1, int(args.top_k)))["candidate_name"]
        .astype(str)
        .tolist()
    )
    selected_stage2_candidates = [(name, config) for name, config in all_candidates if name in set(top_names)]
    print(f"[UG-BCR] stage2 candidates={top_names}")

    stage2_cache = build_cache(
        phase="stage2",
        frame=stage2_frame,
        actor=actor,
        backbone=backbone,
        attack_plan=attack_plan,
        device=device,
        dae=dae,
        detector_model=detector_model,
        detector_threshold=detector_threshold,
        shield_config=shield_config,
        seed=args.seed + 2_000,
        output_dir=args.output_dir,
    )
    stage2_summary = evaluate_candidates(
        phase="stage2",
        cache=stage2_cache,
        candidates=selected_stage2_candidates,
        actor=actor,
        backbone=backbone,
        attack_plan=attack_plan,
        device=device,
        dae=dae,
        detector_model=detector_model,
        detector_threshold=detector_threshold,
        shield_config=shield_config,
        output_dir=args.output_dir,
        clean_drop_cap=float(args.clean_drop_cap),
        clean_drop_cap_pct=float(args.clean_drop_cap_pct),
        clean_activation_cap=float(args.clean_activation_cap),
        clean_vs_shield_drop_cap=float(args.clean_vs_shield_drop_cap),
        main_attack_drop_cap=float(args.main_attack_drop_cap),
        deadline_improvement_floor=float(args.deadline_improvement_floor),
    )
    stage2_summary.to_csv(args.output_dir / "ug_bcr_stage2_summary.csv", index=False, encoding="utf-8-sig")
    feasible_stage2 = stage2_summary.loc[stage2_summary["feasible"].astype(bool)].copy()
    if feasible_stage2.empty:
        raise RuntimeError(
            "No UG-BCR-v2 candidate satisfies the strict clean-drop/clean-activation constraints; "
            "do not publish or reuse a non-feasible configuration."
        )
    best_row = (
        feasible_stage2.sort_values(["selection_score", "weighted_lower_bound_score", "clean_drop_mean"], ascending=[False, False, True], kind="mergesort")
        .iloc[0]
        .to_dict()
    )
    best_name = str(best_row["candidate_name"])
    best_config = dict(all_candidates)[best_name]
    config_payload = ug_bcr_config_payload(best_config)
    config_path = args.output_dir / "ug_bcr_config.json"
    write_json(config_path, config_payload)
    published_config_path = None
    if bool(args.publish_to_dtsr_dir):
        published_config_path = args.dtsr_dir / "ug_bcr_config.json"
        shutil.copyfile(config_path, published_config_path)

    manifest_payload = {
        "status": "calibrated",
        "seed": int(args.seed),
        "algorithm": backbone.algorithm,
        "backbone": backbone.provenance(),
        "attack_plan": attack_plan.provenance(),
        "repair_mode": REPAIR_MODE,
        "runtime_pipeline_order": "DAE/DET route -> UG-BCR belief+urgency gate -> Temporal Shield -> Actor",
        "dae": str(dae_path),
        "detector": str(detector_path),
        "detector_threshold": float(detector_threshold),
        "temporal_shield": str(shield_path),
        "ug_bcr_config": str(config_path),
        "published_to_dtsr_dir": bool(args.publish_to_dtsr_dir),
        "dtsr_runtime_config_copy": None if published_config_path is None else str(published_config_path),
        "stage1_scenes": int(len(stage1_cache)),
        "stage2_scenes": int(len(stage2_cache)),
        "candidate_count": int(len(all_candidates)),
        "stage2_candidates": top_names,
        "selected_candidate": best_name,
        "selected_row": best_row,
        "selection_rule": "feasible first, then maximize min(SmallDrift recovery, Deadline recovery), with clean-drop and clean-activation guardrails",
        "attack_weights": {str(spec["key"]): float(spec["weight"]) for spec in ATTACK_SPECS},
        "constraints": {
            "clean_drop_cap": float(args.clean_drop_cap),
            "clean_drop_cap_pct": float(args.clean_drop_cap_pct),
            "clean_activation_cap": float(args.clean_activation_cap),
            "clean_vs_shield_drop_cap": float(args.clean_vs_shield_drop_cap),
            "main_attack_drop_cap": float(args.main_attack_drop_cap),
            "deadline_improvement_floor": float(args.deadline_improvement_floor),
        },
        "outputs": {
            "candidate_grid": str(args.output_dir / "ug_bcr_candidate_grid.json"),
            "baselines_live": str(args.output_dir / "ug_bcr_baselines_live.csv"),
            "rollouts_live": str(args.output_dir / "ug_bcr_rollouts_live.csv"),
            "candidate_summary_live": str(args.output_dir / "ug_bcr_candidate_summary_live.csv"),
            "stage1_summary": str(args.output_dir / "ug_bcr_stage1_summary.csv"),
            "stage2_summary": str(args.output_dir / "ug_bcr_stage2_summary.csv"),
            "best_live": str(args.output_dir / "ug_bcr_best_live.json"),
        },
        "elapsed_minutes": float((time.perf_counter() - started) / 60.0),
    }
    write_json(args.output_dir / "ug_bcr_manifest.json", manifest_payload)
    if bool(args.publish_to_dtsr_dir):
        write_json(args.dtsr_dir / "ug_bcr_manifest.json", manifest_payload)
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))
    print("[UG-BCR] Done.")


if __name__ == "__main__":
    main()
