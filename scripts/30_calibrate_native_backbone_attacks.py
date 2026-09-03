from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from _common import PACKAGE_ROOT, load_manifest, load_scenario, resolve_device, write_json
from dtsr_multiday_common import REPAIR_MODE, set_all_seeds, to_scalar_summary
import _strength_eval_common as attack_common

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.merged_core import ChargingEnv, TRAIN_PROFILE
from evc.native_backbone_attacks import build_native_attacker
from evc.offpolicy_backbones import load_evaluation_backbone
from evc.ug_bcr import rollout_episode_with_ug_bcr


DEFAULT_BUNDLES = {
    "ddpg": PACKAGE_ROOT
    / "models"
    / "multiday_ddpg_baseline_bundle"
    / "bundle_multiday_best.pt",
    "td3": PACKAGE_ROOT
    / "models"
    / "independent_td3_seed42"
    / "bundle_selected_ep50.pt",
    "sac": PACKAGE_ROOT
    / "models"
    / "independent_sac_seed42"
    / "bundle_selected_ep40.pt",
    "ppo": PACKAGE_ROOT
    / "models"
    / "independent_ppo_seed42"
    / "bundle_selected_ep15.pt",
}
ATTACK_KEYS = (
    "opposite_pgd",
    "q_function",
    "local_small_drift_q",
    "local_deadline_drift_pgd",
)
LONG_HORIZON_ATTACK_KEYS = {
    "local_small_drift_q",
    "local_deadline_drift_pgd",
}


def epsilon_for_attack(attack_key: str) -> float:
    if attack_key == "clean":
        return 0.0
    return 0.055 if attack_key in LONG_HORIZON_ATTACK_KEYS else 0.10


def candidate_profiles(attack_key: str) -> dict[str, dict[str, Any]]:
    if attack_key == "opposite_pgd":
        return {
            "op10_r1": {
                "kind": "pointwise",
                "epsilon": 0.10,
                "alpha": 0.010,
                "iters": 10,
                "restarts": 1,
            },
            "op20_r2": {
                "kind": "pointwise",
                "epsilon": 0.10,
                "alpha": 0.010,
                "iters": 20,
                "restarts": 2,
            },
            "op40_r2": {
                "kind": "pointwise",
                "epsilon": 0.10,
                "alpha": 0.005,
                "iters": 40,
                "restarts": 2,
            },
            "op60_r3": {
                "kind": "pointwise",
                "epsilon": 0.10,
                "alpha": 0.003,
                "iters": 60,
                "restarts": 3,
            },
        }
    if attack_key == "q_function":
        base = {
            "kind": "pointwise",
            "epsilon": 0.10,
            "alpha": 0.005,
            "iters": 40,
            "restarts": 2,
        }
        return {
            "qmin10_r1": {
                **base,
                "alpha": 0.010,
                "iters": 10,
                "restarts": 1,
                "q_mode": "min",
            },
            "q1_10_r1": {
                **base,
                "alpha": 0.010,
                "iters": 10,
                "restarts": 1,
                "q_mode": "q1",
            },
            "qmin20_r2": {
                **base,
                "alpha": 0.010,
                "iters": 20,
                "q_mode": "min",
            },
            "qmin40_r2": {**base, "q_mode": "min"},
            "qmean40_r2": {**base, "q_mode": "mean"},
            "q1_40_r2": {**base, "q_mode": "q1"},
            "q2_40_r2": {**base, "q_mode": "q2"},
        }
    if attack_key == "local_small_drift_q":
        common = {
            "kind": "long_horizon",
            "attack_overrides": {
                "epsilon": 0.055,
                "base_iters": 10,
            },
        }
        return {
            "small_balanced": {
                **common,
                "attack_overrides": {
                    **common["attack_overrides"],
                    "step_size": 0.039,
                    "slew_limit": 0.030,
                    "base_epsilon": 0.030,
                    "base_alpha": 0.010,
                    "momentum": 0.86,
                    "current_direction_weight": 0.24,
                    "drift_decay": 0.89,
                    "base_delta_gain": 2.55,
                    "action_pressure_weight": 2.50,
                },
            },
            "small_fast_q": {
                **common,
                "attack_overrides": {
                    **common["attack_overrides"],
                    "step_size": 0.055,
                    "slew_limit": 0.045,
                    "base_epsilon": 0.055,
                    "base_alpha": 0.008,
                    "base_iters": 15,
                    "momentum": 0.70,
                    "current_direction_weight": 0.40,
                    "drift_decay": 0.92,
                    "base_delta_gain": 3.50,
                    "action_pressure_weight": 4.00,
                    "initial_ramp": 0.75,
                    "ramp_per_step": 0.15,
                    "passive_decay": 0.98,
                },
            },
            "small_actor_pressure": {
                **common,
                "attack_overrides": {
                    **common["attack_overrides"],
                    "step_size": 0.055,
                    "slew_limit": 0.055,
                    "base_epsilon": 0.055,
                    "base_alpha": 0.005,
                    "base_iters": 20,
                    "momentum": 0.55,
                    "current_direction_weight": 0.25,
                    "drift_decay": 0.95,
                    "base_delta_gain": 2.00,
                    "action_pressure_weight": 8.00,
                    "initial_ramp": 1.00,
                    "ramp_per_step": 0.00,
                    "passive_decay": 1.00,
                },
            },
            "small_q_persistent": {
                **common,
                "attack_overrides": {
                    **common["attack_overrides"],
                    "step_size": 0.055,
                    "slew_limit": 0.055,
                    "base_epsilon": 0.055,
                    "base_alpha": 0.004,
                    "base_iters": 30,
                    "momentum": 0.20,
                    "current_direction_weight": 0.80,
                    "drift_decay": 0.99,
                    "base_delta_gain": 5.00,
                    "action_pressure_weight": 0.00,
                    "initial_ramp": 1.00,
                    "ramp_per_step": 0.00,
                    "passive_decay": 1.00,
                },
            },
        }
    if attack_key == "local_deadline_drift_pgd":
        common = {
            "kind": "long_horizon",
            "attack_overrides": {"epsilon": 0.055},
        }
        return {
            "deadline_moderate": {
                **common,
                "attack_overrides": {
                    **common["attack_overrides"],
                    "base_epsilon": 0.040,
                    "base_alpha": 0.005,
                    "base_iters": 20,
                    "minimum_onset_phase": 0.45,
                    "attack_window_fraction": 0.55,
                    "min_attack_steps": 3,
                    "max_attack_steps": 8,
                    "full_strength_fraction": 0.60,
                    "slew_limit_start": 0.006,
                    "slew_limit_end": 0.030,
                    "action_shift_start": 0.05,
                    "action_shift_end": 1.00,
                    "damage_target_early": -0.30,
                    "damage_target_late": -0.90,
                },
            },
            "deadline_long_window": {
                **common,
                "attack_overrides": {
                    **common["attack_overrides"],
                    "base_epsilon": 0.055,
                    "base_alpha": 0.005,
                    "base_iters": 25,
                    "minimum_onset_phase": 0.35,
                    "attack_window_fraction": 0.75,
                    "min_attack_steps": 4,
                    "max_attack_steps": 10,
                    "full_strength_fraction": 0.50,
                    "slew_limit_start": 0.010,
                    "slew_limit_end": 0.055,
                    "action_shift_start": 0.10,
                    "action_shift_end": 1.50,
                    "damage_target_early": -0.35,
                    "damage_target_late": -1.00,
                },
            },
            "deadline_high_pressure": {
                **common,
                "attack_overrides": {
                    **common["attack_overrides"],
                    "base_epsilon": 0.055,
                    "base_alpha": 0.004,
                    "base_iters": 35,
                    "minimum_onset_phase": 0.30,
                    "attack_window_fraction": 0.80,
                    "min_attack_steps": 5,
                    "max_attack_steps": 11,
                    "full_strength_fraction": 0.45,
                    "slew_limit_start": 0.015,
                    "slew_limit_end": 0.055,
                    "action_shift_start": 0.15,
                    "action_shift_end": 1.80,
                    "damage_target_early": -0.45,
                    "damage_target_late": -1.00,
                },
            },
        }
    raise ValueError(attack_key)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rollout(
    arrivals,
    signal_path,
    actor,
    attacker,
    attack_key: str,
    device,
) -> dict[str, Any]:
    summary = rollout_episode_with_ug_bcr(
        arrivals,
        actor,
        signal_path,
        device,
        TRAIN_PROFILE,
        attack_enabled=attack_key != "clean",
        attack_scenario="O",
        attacker=attacker,
        epsilon=epsilon_for_attack(attack_key),
        state_scope=(
            "local"
            if attack_key in LONG_HORIZON_ATTACK_KEYS
            else "all"
        ),
        route_mode="none",
        enable_shield=False,
        enable_belief=False,
        enable_urgency_gate=False,
        attack_ratio=1.0,
        attack_scope="obs",
        repair_mode=REPAIR_MODE,
        label=f"native_calibration__{attack_key}",
    )
    return to_scalar_summary(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "native_attack_calibration_seed42",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "calibration_rollouts.jsonl"
    if not args.resume and jsonl_path.exists():
        jsonl_path.unlink()
    device = resolve_device(args.device)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    backbones = {
        name: load_evaluation_backbone(name, path, device)[:2]
        for name, path in DEFAULT_BUNDLES.items()
    }
    manifest = load_manifest("val").sort_values(
        "Scenario_ID", kind="mergesort"
    ).iloc[: args.scenes].reset_index(drop=True)
    attack_specs = {spec["key"]: spec for spec in attack_common.ATTACK_SPECS}

    rows = load_jsonl(jsonl_path) if args.resume else []
    completed = {
        (row["algorithm"], row["scenario_id"], row["attack_key"], row["profile_id"])
        for row in rows
    }
    started = time.perf_counter()
    for episode_index, (_, scenario) in enumerate(manifest.iterrows(), start=1):
        arrivals, signal_path, scenario_id = load_scenario(scenario)
        max_duration = max(12, int(arrivals["Duration_of_stay"].max()))
        env = ChargingEnv(signal_path, TRAIN_PROFILE)
        low, high = env.observation_bounds(max_duration_of_stay=max_duration)
        for algorithm, (actor, critic) in backbones.items():
            clean_key = (algorithm, scenario_id, "clean", "clean")
            if clean_key not in completed:
                set_all_seeds(args.seed + episode_index)
                result = {
                    "algorithm": algorithm,
                    "scenario_id": scenario_id,
                    "attack_key": "clean",
                    "profile_id": "clean",
                    **rollout(arrivals, signal_path, actor, None, "clean", device),
                }
                append_jsonl(jsonl_path, result)
                rows.append(result)
                completed.add(clean_key)
        for attack_index, attack_key in enumerate(ATTACK_KEYS, start=1):
            attack_seed = args.seed + attack_index * 100_000 + episode_index
            ddpg_key = ("ddpg", scenario_id, attack_key, "canonical_ddpg")
            if ddpg_key not in completed:
                actor, critic = backbones["ddpg"]
                set_all_seeds(attack_seed)
                spec = attack_specs[attack_key]
                attacker = attack_common.build_attacker_for_rollout(
                    attack_spec=spec,
                    actor=actor,
                    critic=critic,
                    device=device,
                    arrivals=arrivals,
                    signal_path=signal_path,
                    attack_seed=attack_seed,
                    epsilon=epsilon_for_attack(attack_key),
                    dae=None,
                    detector_model=None,
                    detector_threshold=0.0,
                    shield_config=None,
                    ug_bcr_config=None,
                    formal_long_outer_epsilon=(
                        epsilon_for_attack(attack_key)
                        if attack_key in LONG_HORIZON_ATTACK_KEYS
                        else None
                    ),
                )
                result = {
                    "algorithm": "ddpg",
                    "scenario_id": scenario_id,
                    "attack_key": attack_key,
                    "profile_id": "canonical_ddpg",
                    **rollout(arrivals, signal_path, actor, attacker, attack_key, device),
                }
                append_jsonl(jsonl_path, result)
                rows.append(result)
                completed.add(ddpg_key)
            for algorithm in ("td3", "sac", "ppo"):
                actor, critic = backbones[algorithm]
                for profile_id, profile in candidate_profiles(attack_key).items():
                    key = (algorithm, scenario_id, attack_key, profile_id)
                    if key in completed:
                        continue
                    set_all_seeds(attack_seed)
                    attacker = build_native_attacker(
                        attack_key,
                        profile,
                        actor=actor,
                        critic=critic,
                        device=device,
                        obs_low=low,
                        obs_high=high,
                        seed=attack_seed,
                    )
                    result = {
                        "algorithm": algorithm,
                        "scenario_id": scenario_id,
                        "attack_key": attack_key,
                        "profile_id": profile_id,
                        **rollout(
                            arrivals, signal_path, actor, attacker, attack_key, device
                        ),
                    }
                    append_jsonl(jsonl_path, result)
                    rows.append(result)
                    completed.add(key)
                    print(
                        f"[native-cal] {scenario_id} {algorithm} {attack_key} "
                        f"{profile_id} reward={result['ep_reward']:.3f}",
                        flush=True,
                    )

    frame = pd.DataFrame(rows)
    clean = frame[frame["attack_key"] == "clean"][
        ["algorithm", "scenario_id", "ep_reward"]
    ].rename(columns={"ep_reward": "clean_reward"})
    attacks = frame[frame["attack_key"] != "clean"].merge(
        clean, on=["algorithm", "scenario_id"], how="inner"
    )
    attacks["reward_degradation"] = attacks["clean_reward"] - attacks["ep_reward"]
    attacks["normalized_degradation"] = (
        attacks["reward_degradation"] / attacks["clean_reward"].abs().clip(lower=1e-9)
    )
    summary = (
        attacks.groupby(["algorithm", "attack_key", "profile_id"], as_index=False)
        .agg(
            scenario_count=("scenario_id", "count"),
            reward_degradation_mean=("reward_degradation", "mean"),
            normalized_degradation_mean=("normalized_degradation", "mean"),
            attack_reward_mean=("ep_reward", "mean"),
            exit_vio_mean=("exit_vio", "mean"),
            run_vio_mean=("run_vio", "mean"),
            linf_max=("attack_delta_linf_max", "max"),
        )
    )
    ddpg_targets = (
        summary[summary["algorithm"] == "ddpg"]
        .set_index("attack_key")["normalized_degradation_mean"]
        .to_dict()
    )
    selected: dict[str, dict[str, Any]] = {"td3": {}, "sac": {}, "ppo": {}}
    for algorithm in ("td3", "sac", "ppo"):
        for attack_key in ATTACK_KEYS:
            candidates = summary[
                (summary["algorithm"] == algorithm)
                & (summary["attack_key"] == attack_key)
            ].copy()
            target = float(ddpg_targets[attack_key])
            candidates["absolute_gap_to_ddpg"] = (
                candidates["normalized_degradation_mean"] - target
            ).abs()
            winner = candidates.sort_values(
                ["absolute_gap_to_ddpg", "normalized_degradation_mean"],
                ascending=[True, False],
            ).iloc[0]
            profile_id = str(winner["profile_id"])
            selected[algorithm][attack_key] = {
                "profile_id": profile_id,
                "profile": candidate_profiles(attack_key)[profile_id],
                "validation_normalized_degradation": float(
                    winner["normalized_degradation_mean"]
                ),
                "ddpg_target_normalized_degradation": target,
                "relative_gap": float(
                    winner["normalized_degradation_mean"] / target - 1.0
                ),
            }

    output = {
        "calibration_split": "val",
        "scenario_count": int(len(manifest)),
        "scenario_ids": manifest["Scenario_ID"].astype(str).tolist(),
        "seed": int(args.seed),
        "selection_objective": "minimum absolute normalized-degradation gap to canonical DDPG",
        "fixed_budgets": {"short_epsilon": 0.10, "long_epsilon": 0.055},
        "selected": selected,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    frame.to_csv(args.output_dir / "calibration_rollouts.csv", index=False)
    summary.to_csv(args.output_dir / "calibration_summary.csv", index=False)
    write_json(args.output_dir / "native_attack_config.json", output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
