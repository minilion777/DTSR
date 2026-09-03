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

from _common import PACKAGE_ROOT, load_manifest, load_scenario, resolve_device, write_json
from dtsr_multiday_common import REPAIR_MODE, safe_recovery, set_all_seeds, to_scalar_summary
import _strength_eval_common as attack_common

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.defense import load_dae, load_detector
from evc.merged_core import ChargingEnv, TRAIN_PROFILE
from evc.native_dtsr import (
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


ATTACK_KEYS = (
    "opposite_fgsm",
    "electhacker_c",
    "electhacker_f",
    "electhacker_o",
)
ATTACK_SCENARIO = {
    "opposite_fgsm": "O",
    "electhacker_c": "C",
    "electhacker_f": "F",
    "electhacker_o": "O",
}
SHORT_EPSILON = 0.10
T95_DF19 = 2.093024054408263
DEFAULT_RAW_DIR = PACKAGE_ROOT / "results" / "attack_strength_remaining4_backbones_seed42"
DEFAULT_NATIVE_CONFIG = (
    PACKAGE_ROOT / "results" / "native_attack_calibration_seed42" / "native_attack_config.json"
)
DEFAULT_PRICE_THRESHOLD = (
    PACKAGE_ROOT
    / "results"
    / "attack120_short_horizon"
    / "ehc_threshold_fix"
    / "electhacker_c_price_threshold.json"
)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifact(
    name: str,
    manifest_path: Path,
    backbone,
    attack_plan,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{name} manifest missing: {manifest_path}")
    manifest = load_json(manifest_path)
    validate_dataset_backbone(manifest, backbone, split="train")
    validate_attack_plan_provenance(manifest, attack_plan)
    return manifest


def build_eval_attacker(
    attack_key: str,
    *,
    actor,
    critic,
    device: torch.device,
    arrivals: pd.DataFrame,
    signal_path: Path,
    seed: int,
):
    spec_lookup = {str(spec["key"]): spec for spec in attack_common.ATTACK_SPECS}
    spec = spec_lookup[str(attack_key)]
    env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
    low, high = env.observation_bounds(
        max_duration_of_stay=max(12, int(arrivals["Duration_of_stay"].max()))
    )
    return attack_common.build_short_attacker(
        algorithm=str(spec["algorithm"]),
        actor=actor,
        critic=critic,
        device=device,
        low=low,
        high=high,
        seed=int(seed),
        epsilon=SHORT_EPSILON,
    )


def run_full_dtsr(
    arrivals: pd.DataFrame,
    signal_path: Path,
    actor,
    device: torch.device,
    *,
    attacker,
    attack_key: str,
    dae,
    detector_model,
    detector_threshold: float,
    shield_config,
    ug_config,
    price_threshold: float,
    label: str,
) -> dict[str, Any]:
    summary = rollout_episode_with_ug_bcr(
        arrivals,
        actor,
        signal_path,
        device,
        TRAIN_PROFILE,
        attack_enabled=attacker is not None,
        attack_scenario=ATTACK_SCENARIO.get(attack_key, "O"),
        attacker=attacker,
        defender=dae,
        detector_model=detector_model,
        detector_threshold=float(detector_threshold),
        shield_config=shield_config,
        route_mode="detector",
        enable_shield=True,
        enable_belief=True,
        enable_urgency_gate=True,
        ug_bcr_config=ug_config,
        epsilon=SHORT_EPSILON,
        state_scope="all",
        price_threshold=float(price_threshold),
        attack_ratio=1.0,
        attack_scope="obs",
        label=label,
        repair_mode=REPAIR_MODE,
    )
    return to_scalar_summary(summary)


def summarize(
    algorithm: str,
    raw: pd.DataFrame,
    defended: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_algorithm = raw[raw["algorithm"] == algorithm].copy()
    raw_clean = raw_algorithm[raw_algorithm["attack_key"] == "clean"].set_index("scenario_id")
    defended_clean = defended[defended["attack_key"] == "clean"].set_index("scenario_id")
    paired_rows: list[dict[str, Any]] = []
    for attack_key in ATTACK_KEYS:
        raw_attack = raw_algorithm[raw_algorithm["attack_key"] == attack_key].set_index("scenario_id")
        defended_attack = defended[defended["attack_key"] == attack_key].set_index("scenario_id")
        common = (
            raw_clean.index.intersection(raw_attack.index)
            .intersection(defended_clean.index)
            .intersection(defended_attack.index)
        )
        for scenario_id in common:
            clean_raw_reward = float(raw_clean.at[scenario_id, "ep_reward"])
            attack_raw_reward = float(raw_attack.at[scenario_id, "ep_reward"])
            clean_dtsr_reward = float(defended_clean.at[scenario_id, "ep_reward"])
            attack_dtsr_reward = float(defended_attack.at[scenario_id, "ep_reward"])
            paired_rows.append(
                {
                    "algorithm": algorithm,
                    "scenario_id": scenario_id,
                    "attack_key": attack_key,
                    "clean_raw_reward": clean_raw_reward,
                    "attack_raw_reward": attack_raw_reward,
                    "clean_dtsr_reward": clean_dtsr_reward,
                    "attack_dtsr_reward": attack_dtsr_reward,
                    "attack_degradation": clean_raw_reward - attack_raw_reward,
                    "defense_reward_gain": attack_dtsr_reward - attack_raw_reward,
                    "recovery": safe_recovery(
                        clean_raw_reward,
                        attack_raw_reward,
                        attack_dtsr_reward,
                    ),
                    "clean_reward_delta": clean_dtsr_reward - clean_raw_reward,
                    "exit_vio_reduction": float(raw_attack.at[scenario_id, "exit_vio"])
                    - float(defended_attack.at[scenario_id, "exit_vio"]),
                    "run_vio_reduction": float(raw_attack.at[scenario_id, "run_vio"])
                    - float(defended_attack.at[scenario_id, "run_vio"]),
                    "raw_attack_linf_max": float(raw_attack.at[scenario_id, "attack_delta_linf_max"]),
                    "dtsr_attack_linf_max": float(defended_attack.at[scenario_id, "attack_delta_linf_max"]),
                }
            )
    paired = pd.DataFrame(paired_rows)
    summary_rows: list[dict[str, Any]] = []
    for attack_key, group in paired.groupby("attack_key", sort=False):
        recovery = group["recovery"].to_numpy(dtype=np.float64)
        recovery = recovery[np.isfinite(recovery)]
        mean = float(np.mean(recovery))
        std = float(np.std(recovery, ddof=1)) if recovery.size > 1 else 0.0
        ci_half = T95_DF19 * std / math.sqrt(max(int(recovery.size), 1))
        summary_rows.append(
            {
                "algorithm": algorithm,
                "attack_key": attack_key,
                "scenario_count": int(group["scenario_id"].nunique()),
                "valid_recovery_count": int(recovery.size),
                "attack_degradation_mean": float(group["attack_degradation"].mean()),
                "defense_reward_gain_mean": float(group["defense_reward_gain"].mean()),
                "recovery_mean": mean,
                "recovery_std": std,
                "recovery_ci95_low": mean - ci_half,
                "recovery_ci95_high": mean + ci_half,
                "clean_reward_delta_mean": float(group["clean_reward_delta"].mean()),
                "exit_vio_reduction_mean": float(group["exit_vio_reduction"].mean()),
                "run_vio_reduction_mean": float(group["run_vio_reduction"].mean()),
                "raw_attack_linf_max": float(group["raw_attack_linf_max"].max()),
                "dtsr_attack_linf_max": float(group["dtsr_attack_linf_max"].max()),
            }
        )
    return paired, pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen native TD3/SAC/PPO DTSR on FGSM and ElectHacker-C/F/O."
    )
    parser.add_argument("--algorithm", choices=("td3", "sac", "ppo"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--bundle-path", type=Path)
    parser.add_argument("--native-config", type=Path, default=DEFAULT_NATIVE_CONFIG)
    parser.add_argument("--raw-results-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--price-threshold-file", type=Path, default=DEFAULT_PRICE_THRESHOLD)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.scenes <= 0:
        raise ValueError("--scenes must be positive")
    layout = native_artifact_layout(PACKAGE_ROOT, args.algorithm, args.seed)
    args.bundle_path = args.bundle_path or default_native_bundle_path(
        PACKAGE_ROOT, args.algorithm, args.seed
    )
    args.output_dir = args.output_dir or (
        layout["dtsr_results"] / "remaining4_test_evaluation"
    )
    price_threshold = attack_common.load_price_threshold_from_path(
        args.price_threshold_file
    )
    raw_config = load_json(args.raw_results_dir / "run_config.json")
    raw = pd.read_csv(args.raw_results_dir / "rollouts.csv")
    expected_scenarios = [f"test_day_{index:04d}" for index in range(1, args.scenes + 1)]
    if raw_config.get("scenario_ids", [])[: args.scenes] != expected_scenarios:
        raise ValueError("Raw attack source does not use the expected test scenarios.")
    if not math.isclose(float(raw_config["short_epsilon"]), SHORT_EPSILON, abs_tol=1e-12):
        raise ValueError("Raw attack source epsilon mismatch.")
    if not math.isclose(
        float(raw_config["electhacker_c_price_threshold"]),
        float(price_threshold),
        abs_tol=1e-12,
    ):
        raise ValueError("ElectHacker-C threshold mismatch with raw source.")

    set_all_seeds(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)
    backbone = load_frozen_backbone(args.algorithm, args.bundle_path, device)
    if raw_config["bundles"][args.algorithm]["sha256"] != backbone.bundle_sha256:
        raise ValueError("Raw attack source backbone hash mismatch.")
    attack_plan = load_frozen_attack_plan(args.algorithm, args.native_config)

    dae_path = layout["dae"] / "dtsr_dae.pt"
    detector_path = layout["det"] / "dtsr_detector.pt"
    shield_path = layout["shield"] / "dtsr_temporal_shield.pt"
    ug_path = layout["ug_bcr"] / "ug_bcr_config.json"
    manifests = {
        "dae": validate_artifact("DAE", layout["dae"] / "dae_manifest.json", backbone, attack_plan),
        "detector": validate_artifact("Detector", layout["det"] / "det_manifest.json", backbone, attack_plan),
        "shield": validate_artifact("Shield", layout["shield"] / "shield_manifest.json", backbone, attack_plan),
        "ug_bcr": validate_artifact("UG-BCR", layout["ug_bcr"] / "ug_bcr_manifest.json", backbone, attack_plan),
    }
    if not bool(manifests["ug_bcr"].get("selected_row", {}).get("feasible", False)):
        raise ValueError("UG-BCR artifact is not a feasible frozen candidate.")

    dae = load_dae(dae_path, device).eval()
    detector_artifact = load_detector(detector_path, device)
    detector_model = detector_artifact.model.eval()
    shield_config = load_temporal_shield_bundle(shield_path).config
    ug_config = load_ug_bcr_config(ug_path)

    manifest = load_manifest("test").sort_values("Scenario_ID", kind="mergesort")
    manifest = manifest.iloc[: args.scenes].reset_index(drop=True)
    actual_scenarios = manifest["Scenario_ID"].astype(str).tolist()
    if actual_scenarios != expected_scenarios:
        raise ValueError("Test manifest scenario ordering changed.")

    raw_algorithm = raw[
        (raw["algorithm"] == args.algorithm)
        & (raw["scenario_id"].isin(expected_scenarios))
        & (raw["attack_key"].isin(("clean",) + ATTACK_KEYS))
    ].copy()
    if len(raw_algorithm) != args.scenes * (1 + len(ATTACK_KEYS)):
        raise ValueError("Raw attack source is incomplete for this algorithm.")
    raw_lookup = raw_algorithm.set_index(["scenario_id", "attack_key"])

    run_config = {
        "experiment": "remaining4_frozen_native_dtsr",
        "algorithm": args.algorithm,
        "seed": int(args.seed),
        "scenario_ids": expected_scenarios,
        "attack_keys": list(ATTACK_KEYS),
        "epsilon": SHORT_EPSILON,
        "price_threshold": float(price_threshold),
        "raw_source": {
            "path": str(args.raw_results_dir.resolve()),
            "run_config_sha256": sha256_file(args.raw_results_dir / "run_config.json"),
            "rollouts_sha256": sha256_file(args.raw_results_dir / "rollouts.csv"),
        },
        "backbone": backbone.provenance(),
        "attack_plan": attack_plan.provenance(),
        "defense_artifact_sha256": {
            "dae": sha256_file(dae_path),
            "detector": sha256_file(detector_path),
            "shield": sha256_file(shield_path),
            "ug_bcr": sha256_file(ug_path),
        },
        "defense_stage": "full_dtsr_ug_bcr",
        "repair_mode": REPAIR_MODE,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    rows_path = args.output_dir / "rollouts.jsonl"
    if rows_path.exists() and args.resume:
        if not config_path.exists() or load_json(config_path) != run_config:
            raise ValueError("Existing resumable run has a different configuration.")
    else:
        rows_path.unlink(missing_ok=True)
        write_json(config_path, run_config)

    rows = load_jsonl(rows_path)
    completed = {(row["scenario_id"], row["attack_key"]) for row in rows}
    expected = args.scenes * (1 + len(ATTACK_KEYS))
    started = time.perf_counter()
    print(
        f"[remaining4-dtsr] algorithm={args.algorithm} device={device} "
        f"rows={len(rows)}/{expected}",
        flush=True,
    )
    for scene_index, scenario in manifest.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(scenario)
        for attack_key in ("clean",) + ATTACK_KEYS:
            key = (scenario_id, attack_key)
            if key in completed:
                continue
            raw_row = raw_lookup.loc[key]
            attack_seed = int(raw_row["attack_seed"])
            set_all_seeds(attack_seed)
            attacker = None
            if attack_key != "clean":
                attacker = build_eval_attacker(
                    attack_key,
                    actor=backbone.actor,
                    critic=backbone.critic,
                    device=device,
                    arrivals=arrivals,
                    signal_path=signal_path,
                    seed=attack_seed,
                )
            rollout_started = time.perf_counter()
            result = run_full_dtsr(
                arrivals,
                signal_path,
                backbone.actor,
                device,
                attacker=attacker,
                attack_key=attack_key,
                dae=dae,
                detector_model=detector_model,
                detector_threshold=float(detector_artifact.threshold),
                shield_config=shield_config,
                ug_config=ug_config,
                price_threshold=price_threshold,
                label=f"{args.algorithm}__{scenario_id}__{attack_key}__full_dtsr",
            )
            output = {
                "algorithm": args.algorithm,
                "scenario_id": scenario_id,
                "attack_key": attack_key,
                "attack_seed": attack_seed,
                "epsilon": 0.0 if attack_key == "clean" else SHORT_EPSILON,
                "runtime_seconds": float(time.perf_counter() - rollout_started),
                **result,
            }
            scalar_values = [
                value
                for value in output.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if not all(math.isfinite(float(value)) for value in scalar_values):
                raise FloatingPointError(f"Non-finite result for {key}")
            done_count = int(output.get("done_count", output.get("done_cnt", -1)))
            if done_count != 344:
                raise RuntimeError(f"Incomplete rollout for {key}: done_count={done_count}")
            append_jsonl(rows_path, output)
            rows.append(output)
            completed.add(key)
            print(
                f"[remaining4-dtsr] {len(rows):03d}/{expected} {scenario_id} "
                f"{attack_key} reward={float(output['ep_reward']):.3f} "
                f"linf={float(output['attack_delta_linf_max']):.5f}",
                flush=True,
            )

    defended = pd.DataFrame(rows).sort_values(
        ["scenario_id", "attack_key"], kind="mergesort"
    )
    paired, summary = summarize(args.algorithm, raw_algorithm, defended)
    defended.to_csv(args.output_dir / "rollouts.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(args.output_dir / "paired_recovery.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "recovery_summary.csv", index=False, encoding="utf-8-sig")
    attack_rows = defended[defended["attack_key"] != "clean"]
    checks = {
        "complete": len(defended) == expected,
        "expected_rollouts": expected,
        "completed_rollouts": int(len(defended)),
        "all_finite": bool(np.isfinite(defended["ep_reward"].to_numpy(float)).all()),
        "all_raw_attacks_effective": bool((summary["attack_degradation_mean"] > 0.0).all()),
        "all_recovery_finite": bool(np.isfinite(summary["recovery_mean"].to_numpy(float)).all()),
        "linf_budget_respected": bool(
            (attack_rows["attack_delta_linf_max"].astype(float) <= SHORT_EPSILON + 1e-6).all()
        ),
        "elapsed_seconds_this_invocation": float(time.perf_counter() - started),
    }
    write_json(args.output_dir / "checks.json", checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
