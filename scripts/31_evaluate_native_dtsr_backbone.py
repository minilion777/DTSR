from __future__ import annotations

import argparse
import json
import math
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
from dtsr_multiday_common import REPAIR_MODE, safe_recovery, set_all_seeds, to_scalar_summary

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.defense import load_dae, load_detector
from evc.merged_core import ChargingEnv, TRAIN_PROFILE
from evc.native_dtsr import (
    LONG_ATTACK_KEYS,
    LONG_EPSILON,
    NATIVE_ATTACK_KEYS,
    SHORT_EPSILON,
    build_frozen_attacker,
    default_native_bundle_path,
    load_frozen_attack_plan,
    load_frozen_backbone,
    native_artifact_layout,
    sha256_file,
    validate_attack_plan_provenance,
    validate_dataset_backbone,
)
from evc.offline_dae_det_temporal_shield import load_temporal_shield_bundle
from evc.ug_bcr import load_ug_bcr_config, rollout_episode_with_ug_bcr


NATIVE_BACKBONES = ("td3", "sac", "ppo")
STAGES = ("attack", "dae_det", "shield", "ug_bcr")
ATTACK_SCOPE = {
    "opposite_pgd": "all",
    "q_function": "all",
    "local_small_drift_q": "local",
    "local_deadline_drift_pgd": "local",
}

# Canonical positions in scripts/_strength_eval_common.py::ATTACK_SPECS.
# Keep these offsets stable so the raw and defended rollouts use identical
# stochastic starts even when this evaluator runs only the native four attacks.
ATTACK_SEED_MULTIPLIER = {
    "clean": 1,
    "opposite_pgd": 2,
    "q_function": 3,
    "local_small_drift_q": 8,
    "local_deadline_drift_pgd": 9,
}


def configure_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_manifest_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def assert_artifact_provenance(
    artifact_name: str,
    manifest_path: Path,
    backbone,
    attack_plan,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{artifact_name} manifest not found: {manifest_path}")
    manifest = load_manifest_json(manifest_path)
    validate_dataset_backbone(manifest, backbone, split="train")
    validate_attack_plan_provenance(manifest, attack_plan)
    return manifest


def build_attacker(attack_key, backbone, attack_plan, device, arrivals, signal_path, seed):
    env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
    low, high = env.observation_bounds(
        max_duration_of_stay=max(12, int(arrivals["Duration_of_stay"].max()))
    )
    return build_frozen_attacker(
        attack_key,
        backbone=backbone,
        attack_plan=attack_plan,
        device=device,
        obs_low=low,
        obs_high=high,
        seed=int(seed),
    )


def rollout_stage(
    stage: str,
    arrivals,
    signal_path,
    actor,
    device,
    *,
    attacker,
    attack_scope: str,
    dae,
    detector_model,
    detector_threshold: float,
    shield_config,
    ug_config,
    label: str,
) -> dict[str, Any]:
    use_dae = stage != "attack"
    use_shield = stage in {"shield", "ug_bcr"}
    use_ug = stage == "ug_bcr"
    result = rollout_episode_with_ug_bcr(
        arrivals,
        actor,
        signal_path,
        device,
        TRAIN_PROFILE,
        attack_enabled=attacker is not None,
        attack_scenario="O",
        attacker=attacker,
        defender=dae if use_dae else None,
        detector_model=detector_model if use_dae else None,
        detector_threshold=detector_threshold if use_dae else None,
        shield_config=shield_config if use_shield else None,
        route_mode="detector" if use_dae else "none",
        enable_shield=use_shield,
        enable_belief=use_ug,
        enable_urgency_gate=use_ug,
        ug_bcr_config=ug_config if use_ug else None,
        state_scope=attack_scope,
        attack_scope="obs",
        label=label,
        repair_mode=REPAIR_MODE,
    )
    return to_scalar_summary(result)


def summarize(rows: pd.DataFrame, attack_keys: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_rows: list[dict[str, Any]] = []
    for (attack_key, stage), group in rows.groupby(["attack_key", "stage"], sort=False):
        condition_rows.append(
            {
                "attack_key": attack_key,
                "stage": stage,
                "scenario_count": int(group["scenario_id"].nunique()),
                "reward_mean": float(group["ep_reward"].mean()),
                "reward_std": float(group["ep_reward"].std(ddof=1)),
                "exit_vio_mean": float(group["exit_vio"].mean()),
                "run_vio_mean": float(group["run_vio"].mean()),
                "route_rate_mean": float(group.get("route_rate", pd.Series(0.0, index=group.index)).mean()),
                "shield_correction_mean": float(
                    group.get("shield_correction_mean", pd.Series(0.0, index=group.index)).mean()
                ),
                "ug_bcr_belief_rate_mean": float(
                    group.get("urgency_gate_belief_rate", pd.Series(0.0, index=group.index)).mean()
                ),
            }
        )

    recovery_rows: list[dict[str, Any]] = []
    clean_raw = rows[(rows["attack_key"] == "clean") & (rows["stage"] == "attack")].set_index("scenario_id")
    for attack_key in attack_keys:
        attacked = rows[(rows["attack_key"] == attack_key) & (rows["stage"] == "attack")].set_index("scenario_id")
        for stage in STAGES[1:]:
            defended = rows[(rows["attack_key"] == attack_key) & (rows["stage"] == stage)].set_index("scenario_id")
            clean_defended = rows[(rows["attack_key"] == "clean") & (rows["stage"] == stage)].set_index("scenario_id")
            common = clean_raw.index.intersection(attacked.index).intersection(defended.index).intersection(clean_defended.index)
            recovery_values = np.asarray(
                [
                    safe_recovery(
                        float(clean_raw.at[scene, "ep_reward"]),
                        float(attacked.at[scene, "ep_reward"]),
                        float(defended.at[scene, "ep_reward"]),
                    )
                    for scene in common
                ],
                dtype=np.float64,
            )
            valid = np.isfinite(recovery_values)
            recovery_rows.append(
                {
                    "attack_key": attack_key,
                    "stage": stage,
                    "scenario_count": int(len(common)),
                    "valid_recovery_count": int(valid.sum()),
                    "attack_degradation_mean": float(
                        (clean_raw.loc[common, "ep_reward"] - attacked.loc[common, "ep_reward"]).mean()
                    ),
                    "defense_reward_gain_mean": float(
                        (defended.loc[common, "ep_reward"] - attacked.loc[common, "ep_reward"]).mean()
                    ),
                    "recovery_mean": float(recovery_values[valid].mean()) if valid.any() else float("nan"),
                    "recovery_std": float(recovery_values[valid].std(ddof=1)) if valid.sum() > 1 else float("nan"),
                    "clean_reward_delta_mean": float(
                        (clean_defended.loc[common, "ep_reward"] - clean_raw.loc[common, "ep_reward"]).mean()
                    ),
                    "exit_vio_reduction_mean": float(
                        (attacked.loc[common, "exit_vio"] - defended.loc[common, "exit_vio"]).mean()
                    ),
                    "run_vio_reduction_mean": float(
                        (attacked.loc[common, "run_vio"] - defended.loc[common, "run_vio"]).mean()
                    ),
                }
            )
    return pd.DataFrame(condition_rows), pd.DataFrame(recovery_rows)


def main() -> None:
    configure_line_buffering()
    parser = argparse.ArgumentParser(
        description="Evaluate backbone-native TD3/SAC/PPO DTSR artifacts on an untouched test split."
    )
    parser.add_argument("--algorithm", choices=NATIVE_BACKBONES, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bundle-path", type=Path)
    parser.add_argument(
        "--native-config",
        type=Path,
        default=PACKAGE_ROOT / "results" / "native_attack_calibration_seed42" / "native_attack_config.json",
    )
    parser.add_argument("--dae-artifact-dir", type=Path)
    parser.add_argument("--detector-artifact-dir", type=Path)
    parser.add_argument("--shield-artifact-dir", type=Path)
    parser.add_argument("--ug-bcr-artifact-dir", type=Path)
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument(
        "--selection-mode",
        choices=("seeded", "first"),
        default="seeded",
        help=(
            "Scenario selection within the sorted test manifest. Use 'first' to "
            "match the frozen test_day_0001..N attack-strength evaluation."
        ),
    )
    parser.add_argument("--attack-keys", default=",".join(NATIVE_ATTACK_KEYS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    layout = native_artifact_layout(PACKAGE_ROOT, args.algorithm, args.seed)
    args.bundle_path = args.bundle_path or default_native_bundle_path(PACKAGE_ROOT, args.algorithm, args.seed)
    args.dae_artifact_dir = args.dae_artifact_dir or layout["dae"]
    args.detector_artifact_dir = args.detector_artifact_dir or layout["det"]
    args.shield_artifact_dir = args.shield_artifact_dir or layout["shield"]
    args.ug_bcr_artifact_dir = args.ug_bcr_artifact_dir or layout["ug_bcr"]
    args.output_dir = args.output_dir or (layout["dtsr_results"] / "test_evaluation")
    attack_keys = tuple(key.strip() for key in args.attack_keys.split(",") if key.strip())
    unknown = sorted(set(attack_keys) - set(NATIVE_ATTACK_KEYS))
    if unknown:
        raise ValueError(f"Unknown native attack keys: {unknown}")

    set_all_seeds(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)
    backbone = load_frozen_backbone(args.algorithm, args.bundle_path, device)
    attack_plan = load_frozen_attack_plan(args.algorithm, args.native_config)

    dae_path = args.dae_artifact_dir / "dtsr_dae.pt"
    detector_path = args.detector_artifact_dir / "dtsr_detector.pt"
    shield_path = args.shield_artifact_dir / "dtsr_temporal_shield.pt"
    ug_path = args.ug_bcr_artifact_dir / "ug_bcr_config.json"
    dae = load_dae(dae_path, device).eval()
    detector_artifact = load_detector(detector_path, device)
    detector_model = detector_artifact.model.eval()
    shield_config = load_temporal_shield_bundle(shield_path).config
    ug_config = load_ug_bcr_config(ug_path)

    manifests = {
        "dae": assert_artifact_provenance("DAE", args.dae_artifact_dir / "dae_manifest.json", backbone, attack_plan),
        "detector": assert_artifact_provenance("DeT", args.detector_artifact_dir / "det_manifest.json", backbone, attack_plan),
        "shield": assert_artifact_provenance("Shield", args.shield_artifact_dir / "shield_manifest.json", backbone, attack_plan),
        "ug_bcr": assert_artifact_provenance("UG-BCR", args.ug_bcr_artifact_dir / "ug_bcr_manifest.json", backbone, attack_plan),
    }
    if not bool(manifests["ug_bcr"].get("selected_row", {}).get("feasible", False)):
        raise ValueError("UG-BCR artifact was not selected as a feasible calibration candidate.")

    frame = load_manifest(args.split).sort_values("Scenario_ID", kind="mergesort").reset_index(drop=True)
    if args.selection_mode == "first":
        if args.scenes <= 0 or args.scenes > len(frame):
            raise ValueError(
                f"Invalid --scenes={args.scenes}; available test scenes={len(frame)}"
            )
        frame = frame.iloc[: args.scenes].copy().reset_index(drop=True)
    else:
        frame = deterministic_subset(frame, args.scenes, args.seed + 31_000)
    scenario_ids = [str(value) for value in frame["Scenario_ID"].tolist()]
    overlap = sorted(set(scenario_ids).intersection(attack_plan.calibration_scenario_ids))
    if overlap:
        raise ValueError(f"Test evaluation overlaps attack calibration scenarios: {overlap}")

    artifact_paths = {
        "bundle": Path(args.bundle_path),
        "native_config": Path(args.native_config),
        "dae": dae_path,
        "detector": detector_path,
        "shield": shield_path,
        "ug_bcr": ug_path,
    }
    run_config = {
        "schema_version": 1,
        "algorithm": args.algorithm,
        "seed": int(args.seed),
        "split": args.split,
        "selection_mode": args.selection_mode,
        "attack_seed_rule": "seed + 100000 * canonical_full_attack_multiplier + one_based_scene_index",
        "scenario_ids": scenario_ids,
        "attack_keys": list(attack_keys),
        "stages": list(STAGES),
        "repair_mode": REPAIR_MODE,
        "backbone": backbone.provenance(),
        "attack_plan": attack_plan.provenance(),
        "artifact_sha256": {name: sha256_file(path) for name, path in artifact_paths.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    rows_path = args.output_dir / "rollouts.jsonl"
    if rows_path.exists() and args.resume:
        if not config_path.exists() or load_manifest_json(config_path) != run_config:
            raise ValueError("Existing resumable evaluation has a different run configuration.")
    else:
        rows_path.unlink(missing_ok=True)
        write_json(config_path, run_config)

    rows = load_jsonl(rows_path)
    completed = {(row["scenario_id"], row["attack_key"], row["stage"]) for row in rows}
    expected = len(frame) * (1 + len(attack_keys)) * len(STAGES)
    started = time.perf_counter()
    print(f"[Native DTSR Eval] algorithm={args.algorithm} device={device} rows={len(rows)}/{expected}")
    for scene_index, row in frame.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(row)
        for attack_key in ("clean",) + attack_keys:
            scope = ATTACK_SCOPE.get(attack_key, "local")
            # Match scripts/29_compare_attack_strength_backbones.py exactly.
            attack_seed = int(
                args.seed
                + 100_000 * ATTACK_SEED_MULTIPLIER[attack_key]
                + scene_index
                + 1
            )
            for stage in STAGES:
                key = (scenario_id, attack_key, stage)
                if key in completed:
                    continue
                attacker = None
                if attack_key != "clean":
                    attacker = build_attacker(
                        attack_key,
                        backbone,
                        attack_plan,
                        device,
                        arrivals,
                        signal_path,
                        attack_seed,
                    )
                result = rollout_stage(
                    stage,
                    arrivals,
                    signal_path,
                    backbone.actor,
                    device,
                    attacker=attacker,
                    attack_scope=scope,
                    dae=dae,
                    detector_model=detector_model,
                    detector_threshold=float(detector_artifact.threshold),
                    shield_config=shield_config,
                    ug_config=ug_config,
                    label=f"{scenario_id}__{attack_key}__{stage}",
                )
                output_row = {
                    "algorithm": args.algorithm,
                    "scenario_id": scenario_id,
                    "attack_key": attack_key,
                    "stage": stage,
                    "attack_seed": attack_seed,
                    **result,
                }
                if any(isinstance(value, float) and not math.isfinite(value) for value in output_row.values()):
                    raise FloatingPointError(f"Non-finite rollout output for {key}")
                append_jsonl(rows_path, output_row)
                rows.append(output_row)
                completed.add(key)
                print(
                    f"[Native DTSR Eval] {len(rows):03d}/{expected} {scenario_id} "
                    f"{attack_key}/{stage} reward={float(result['ep_reward']):.2f}"
                )

    result_frame = pd.DataFrame(rows).sort_values(
        ["scenario_id", "attack_key", "stage"], kind="mergesort"
    )
    condition_summary, recovery_summary = summarize(result_frame, attack_keys)
    result_frame.to_csv(args.output_dir / "rollouts.csv", index=False, encoding="utf-8-sig")
    condition_summary.to_csv(args.output_dir / "condition_summary.csv", index=False, encoding="utf-8-sig")
    recovery_summary.to_csv(args.output_dir / "recovery_summary.csv", index=False, encoding="utf-8-sig")

    attack_rows = result_frame[result_frame["attack_key"] != "clean"]
    observed_budget = float(attack_rows["attack_delta_linf_max"].max()) if len(attack_rows) else 0.0
    observed_budget_by_attack = {
        attack_key: float(
            attack_rows.loc[attack_rows["attack_key"] == attack_key, "attack_delta_linf_max"].max()
        )
        for attack_key in attack_keys
    }
    budget_checks = {
        attack_key: observed_budget_by_attack[attack_key]
        <= (LONG_EPSILON if attack_key in LONG_ATTACK_KEYS else SHORT_EPSILON) + 1e-6
        for attack_key in attack_keys
    }
    checks = {
        "complete": int(len(result_frame)) == int(expected),
        "finite_rewards": bool(np.isfinite(result_frame["ep_reward"].to_numpy(dtype=float)).all()),
        "test_only": args.split == "test" and all(scene.startswith("test_") for scene in scenario_ids),
        "no_calibration_overlap": not overlap,
        "all_attacks_effective": bool(
            (recovery_summary.groupby("attack_key")["attack_degradation_mean"].first() > 0.0).all()
        ),
        "linf_budget_respected": all(budget_checks.values()),
    }
    summary_payload = {
        "status": "complete" if all(checks.values()) else "completed_with_failed_checks",
        "algorithm": args.algorithm,
        "scenario_count": int(len(frame)),
        "row_count": int(len(result_frame)),
        "checks": checks,
        "observed_attack_linf_max": observed_budget,
        "observed_attack_linf_max_by_attack": observed_budget_by_attack,
        "linf_budget_check_by_attack": budget_checks,
        "elapsed_minutes": float((time.perf_counter() - started) / 60.0),
        "outputs": {
            "rollouts": str(args.output_dir / "rollouts.csv"),
            "conditions": str(args.output_dir / "condition_summary.csv"),
            "recovery": str(args.output_dir / "recovery_summary.csv"),
        },
    }
    write_json(args.output_dir / "summary.json", summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
