from __future__ import annotations

import argparse
import hashlib
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
import _strength_eval_common as table_eval

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.defense import load_dae, load_detector
from evc.merged_core import TRAIN_PROFILE, load_actor_from_path
from evc.offline_dae_det_temporal_shield import load_temporal_shield_bundle
from evc.offpolicy_backbones import load_backbone_bundle
from evc.ug_bcr_v3 import load_ug_bcr_v3_config, rollout_episode_with_ug_bcr_v3


DEFAULT_DDPG_ACTOR = (
    PACKAGE_ROOT / "models" / "multiday_ddpg_baseline_bundle" / "actor_multiday_best.pt"
)
DEFAULT_DAE = PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday" / "dtsr_dae.pt"
DEFAULT_DETECTOR = (
    PACKAGE_ROOT / "artifacts" / "det_seed42_fullstate" / "dtsr_detector.pt"
)
DEFAULT_SHIELD = (
    PACKAGE_ROOT
    / "artifacts"
    / "shield_seed42_fullstate_newlong_v2"
    / "dtsr_temporal_shield.pt"
)
DEFAULT_UG_BCR_V3 = (
    PACKAGE_ROOT
    / "artifacts"
    / "ug_bcr_v3_seed42_newlong_v2"
    / "ug_bcr_v3_config.json"
)
DEFAULT_ATTACK_KEYS = (
    "opposite_pgd",
    "q_function",
    "local_small_drift_q",
    "local_deadline_drift_pgd",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def condition_specs(attack_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    specs = [
        {"attack_key": "clean", "stage_key": "attack"},
        {"attack_key": "clean", "stage_key": "ug_bcr"},
    ]
    for attack_key in attack_keys:
        specs.append({"attack_key": attack_key, "stage_key": "attack"})
        specs.append({"attack_key": attack_key, "stage_key": "ug_bcr"})
    return specs


def attack_epsilon(attack_key: str, short_epsilon: float, long_epsilon: float) -> float:
    if attack_key in {"local_small_drift_q", "local_deadline_drift_pgd"}:
        return float(long_epsilon)
    return float(short_epsilon)


def actor_divergence(
    target_actor: torch.nn.Module,
    ddpg_actor: torch.nn.Module,
    device: torch.device,
    clean_npz: Path,
) -> dict[str, float | int]:
    payload = np.load(clean_npz, allow_pickle=True)
    candidates = [
        "states",
        "observations",
        "obs",
        "clean_inputs",
        "inputs",
    ]
    key = next((name for name in candidates if name in payload.files), None)
    if key is None:
        for name in payload.files:
            values = np.asarray(payload[name])
            if values.ndim >= 2 and values.shape[-1] == 11:
                key = name
                break
    if key is None:
        raise ValueError(f"No observation array found in {clean_npz}; keys={payload.files}")
    observations = np.asarray(payload[key], dtype=np.float32).reshape(-1, 11)
    if len(observations) > 10000:
        observations = observations[:10000]
    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=device)
    with torch.no_grad():
        target_action = target_actor(obs_t).reshape(-1).cpu().numpy()
        ddpg_action = ddpg_actor(obs_t).reshape(-1).cpu().numpy()
    delta = target_action - ddpg_action
    return {
        "sample_count": int(len(observations)),
        "action_mse_vs_ddpg": float(np.mean(delta**2)),
        "action_mae_vs_ddpg": float(np.mean(np.abs(delta))),
        "action_max_abs_vs_ddpg": float(np.max(np.abs(delta))),
        "target_action_std": float(np.std(target_action)),
        "ddpg_action_std": float(np.std(ddpg_action)),
    }


def summarize_results(rows: pd.DataFrame, attack_keys: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_rows = []
    for (attack_key, stage_key), group in rows.groupby(["attack_key", "stage_key"], sort=False):
        condition_rows.append(
            {
                "attack_key": attack_key,
                "condition": (
                    "Clean"
                    if attack_key == "clean" and stage_key == "attack"
                    else "Clean + Strict zero-shot DTSR"
                    if attack_key == "clean"
                    else "Attack"
                    if stage_key == "attack"
                    else "Attack + Strict zero-shot DTSR"
                ),
                "stage_key": stage_key,
                "scenario_count": int(len(group)),
                "reward_mean": float(group["ep_reward"].mean()),
                "reward_std": float(group["ep_reward"].std(ddof=1)),
                "exit_vio_mean": float(group["exit_vio"].mean()),
                "run_vio_mean": float(group["run_vio"].mean()),
                "route_rate_mean": float(group["route_rate"].mean()),
                "shield_correction_mean": float(group["shield_correction_mean"].mean()),
            }
        )
    condition_summary = pd.DataFrame(condition_rows)

    recovery_rows = []
    clean = rows[
        (rows["attack_key"] == "clean") & (rows["stage_key"] == "attack")
    ].set_index("scenario_id")
    clean_defended = rows[
        (rows["attack_key"] == "clean") & (rows["stage_key"] == "ug_bcr")
    ].set_index("scenario_id")
    for attack_key in attack_keys:
        attack = rows[
            (rows["attack_key"] == attack_key) & (rows["stage_key"] == "attack")
        ].set_index("scenario_id")
        defended = rows[
            (rows["attack_key"] == attack_key) & (rows["stage_key"] == "ug_bcr")
        ].set_index("scenario_id")
        common = clean.index.intersection(attack.index).intersection(defended.index)
        per_scene = []
        for scenario_id in common:
            recovery = safe_recovery(
                clean.at[scenario_id, "ep_reward"],
                attack.at[scenario_id, "ep_reward"],
                defended.at[scenario_id, "ep_reward"],
            )
            per_scene.append(recovery)
        recovery_values = np.asarray(per_scene, dtype=np.float64)
        valid = np.isfinite(recovery_values)
        clean_common = clean.loc[common]
        attack_common = attack.loc[common]
        defended_common = defended.loc[common]
        clean_def_common = clean_defended.loc[common]
        recovery_rows.append(
            {
                "attack_key": attack_key,
                "scenario_count": int(len(common)),
                "valid_recovery_count": int(np.sum(valid)),
                "attack_degradation_mean": float(
                    (clean_common["ep_reward"] - attack_common["ep_reward"]).mean()
                ),
                "defense_reward_gain_mean": float(
                    (defended_common["ep_reward"] - attack_common["ep_reward"]).mean()
                ),
                "recovery_mean": (
                    float(np.mean(recovery_values[valid])) if bool(np.any(valid)) else float("nan")
                ),
                "recovery_std": (
                    float(np.std(recovery_values[valid], ddof=1))
                    if int(np.sum(valid)) > 1
                    else float("nan")
                ),
                "exit_vio_reduction_mean": float(
                    (attack_common["exit_vio"] - defended_common["exit_vio"]).mean()
                ),
                "run_vio_reduction_mean": float(
                    (attack_common["run_vio"] - defended_common["run_vio"]).mean()
                ),
                "clean_dtsr_reward_delta_mean": float(
                    (clean_def_common["ep_reward"] - clean_common["ep_reward"]).mean()
                ),
                "clean_dtsr_exit_vio_delta_mean": float(
                    (clean_def_common["exit_vio"] - clean_common["exit_vio"]).mean()
                ),
                "clean_dtsr_run_vio_delta_mean": float(
                    (clean_def_common["run_vio"] - clean_common["run_vio"]).mean()
                ),
            }
        )
    return condition_summary, pd.DataFrame(recovery_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate strict DDPG-to-TD3/SAC zero-shot DTSR transfer."
    )
    parser.add_argument("--backbone-bundle", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--attack-keys", default=",".join(DEFAULT_ATTACK_KEYS))
    parser.add_argument("--short-epsilon", type=float, default=0.1)
    parser.add_argument("--long-epsilon", type=float, default=0.055)
    parser.add_argument("--ddpg-actor", type=Path, default=DEFAULT_DDPG_ACTOR)
    parser.add_argument("--dae", type=Path, default=DEFAULT_DAE)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--shield", type=Path, default=DEFAULT_SHIELD)
    parser.add_argument("--ug-bcr-v3", type=Path, default=DEFAULT_UG_BCR_V3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    attack_keys = tuple(key.strip() for key in args.attack_keys.split(",") if key.strip())
    attack_lookup = {spec["key"]: spec for spec in table_eval.ATTACK_SPECS}
    unknown = sorted(set(attack_keys) - set(attack_lookup))
    if unknown:
        raise ValueError(f"Unknown attack keys: {unknown}")
    if not attack_keys:
        raise ValueError("At least one attack key is required.")
    for path in (
        args.backbone_bundle,
        args.ddpg_actor,
        args.dae,
        args.detector,
        args.shield,
        args.ug_bcr_v3,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_path = args.output_dir / "rollouts.jsonl"
    if not args.resume and intermediate_path.exists():
        intermediate_path.unlink()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)
    set_all_seeds(args.seed)
    actor, critic, backbone_payload = load_backbone_bundle(args.backbone_bundle, device)
    algorithm = str(backbone_payload["algorithm"])
    ddpg_actor = load_actor_from_path(args.ddpg_actor, device)
    divergence = actor_divergence(
        actor,
        ddpg_actor,
        device,
        PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday" / "clean" / "clean_val.npz",
    )
    dae = load_dae(args.dae, device).eval()
    detector_artifact = load_detector(args.detector, device)
    detector_model = detector_artifact.model
    detector_threshold = float(detector_artifact.threshold)
    shield_config = load_temporal_shield_bundle(args.shield).config
    ug_bcr_v3_config = load_ug_bcr_v3_config(args.ug_bcr_v3)
    ug_bcr_config = ug_bcr_v3_config.base_v2

    manifest = load_manifest(args.split).sort_values(
        "Scenario_ID", kind="mergesort"
    ).reset_index(drop=True)
    if len(manifest) < args.scenes:
        raise RuntimeError(f"Requested {args.scenes} scenes, found {len(manifest)}.")
    manifest = manifest.iloc[: args.scenes].copy().reset_index(drop=True)
    specs = condition_specs(attack_keys)
    expected_keys = {
        (str(row["Scenario_ID"]), spec["attack_key"], spec["stage_key"])
        for _, row in manifest.iterrows()
        for spec in specs
    }
    existing = load_jsonl(intermediate_path) if args.resume else []
    existing = [
        row
        for row in existing
        if (row["scenario_id"], row["attack_key"], row["stage_key"]) in expected_keys
    ]
    completed = {
        (row["scenario_id"], row["attack_key"], row["stage_key"]) for row in existing
    }
    if len(completed) != len(existing):
        raise RuntimeError("Duplicate rollout keys in resume file.")

    artifacts = {
        "ddpg_actor": args.ddpg_actor,
        "dae": args.dae,
        "detector": args.detector,
        "shield": args.shield,
        "ug_bcr_v3": args.ug_bcr_v3,
    }
    run_config = {
        "experiment": "strict_zero_shot_ddpg_dtsr_to_cross_backbone",
        "algorithm": algorithm,
        "seed": int(args.seed),
        "split": args.split,
        "scenario_count": int(len(manifest)),
        "attack_keys": list(attack_keys),
        "short_epsilon": float(args.short_epsilon),
        "long_epsilon": float(args.long_epsilon),
        "attack_seed_rule": "seed + attack_index*100000 + episode_index",
        "backbone_bundle": str(args.backbone_bundle),
        "backbone_metadata": backbone_payload.get("metadata"),
        "actor_divergence_vs_ddpg": divergence,
        "strict_zero_shot": True,
        "target_backbone_data_used_for_defense": False,
        "frozen_artifacts": {
            key: {"path": str(path), "sha256": sha256_file(path)}
            for key, path in artifacts.items()
        },
        "detector_threshold": detector_threshold,
    }
    write_json(args.output_dir / "run_config.json", run_config)

    stage_lookup = {spec["key"]: spec for spec in table_eval.STAGE_SPECS}
    attack_offsets = {
        attack_key: (index + 1) * 100_000 for index, attack_key in enumerate(attack_keys)
    }
    started = time.perf_counter()
    for episode_index, (_, scenario_row) in enumerate(manifest.iterrows(), start=1):
        arrivals, signal_path, scenario_id = load_scenario(scenario_row)
        for condition in specs:
            key = (scenario_id, condition["attack_key"], condition["stage_key"])
            if key in completed:
                continue
            attack_key = condition["attack_key"]
            stage_key = condition["stage_key"]
            attack_seed = int(
                args.seed + attack_offsets.get(attack_key, 0) + episode_index
            )
            set_all_seeds(attack_seed)
            attack_spec = attack_lookup[attack_key]
            epsilon = (
                0.0
                if attack_key == "clean"
                else attack_epsilon(attack_key, args.short_epsilon, args.long_epsilon)
            )
            attacker = table_eval.build_attacker_for_rollout(
                attack_spec=attack_spec,
                actor=actor,
                critic=critic,
                device=device,
                arrivals=arrivals,
                signal_path=signal_path,
                attack_seed=attack_seed,
                epsilon=epsilon,
                dae=dae,
                detector_model=detector_model,
                detector_threshold=detector_threshold,
                shield_config=shield_config,
                ug_bcr_config=ug_bcr_config,
                formal_long_outer_epsilon=(
                    args.long_epsilon
                    if attack_key in {"local_small_drift_q", "local_deadline_drift_pgd"}
                    else None
                ),
            )
            stage_spec = stage_lookup[stage_key]
            stage_kwargs = table_eval.stage_kwargs(
                stage_spec,
                dae=dae,
                detector_model=detector_model,
                detector_threshold=detector_threshold,
                shield_config=shield_config,
                ug_bcr_config=ug_bcr_config,
            )
            rollout_function = table_eval.rollout_episode_with_ug_bcr
            if stage_key == "ug_bcr":
                rollout_function = rollout_episode_with_ug_bcr_v3
                stage_kwargs.pop("ug_bcr_config", None)
                stage_kwargs["ug_bcr_v3_config"] = ug_bcr_v3_config
            rollout_started = time.perf_counter()
            summary = rollout_function(
                arrivals,
                actor,
                signal_path,
                device,
                TRAIN_PROFILE,
                attack_enabled=attack_key != "clean",
                attack_scenario=str(attack_spec["scenario"]),
                attacker=attacker,
                epsilon=epsilon,
                state_scope=str(attack_spec.get("scope", "all")),
                attack_ratio=1.0,
                attack_scope="obs",
                label=f"{algorithm}__{attack_key}__{stage_key}",
                repair_mode=REPAIR_MODE,
                **stage_kwargs,
            )
            result = {
                "algorithm": algorithm,
                "scenario_id": scenario_id,
                "episode_index": int(episode_index),
                "seed": int(args.seed),
                "attack_seed": int(attack_seed),
                "attack_key": attack_key,
                "stage_key": stage_key,
                "strict_zero_shot": True,
                "runtime_seconds": float(time.perf_counter() - rollout_started),
                **to_scalar_summary(summary),
            }
            append_jsonl(intermediate_path, result)
            existing.append(result)
            completed.add(key)
            print(
                f"[{algorithm}] {len(completed):03d}/{len(expected_keys)} "
                f"{scenario_id} {attack_key}/{stage_key} "
                f"reward={result['ep_reward']:.3f} exit={result['exit_vio']} "
                f"run={result['run_vio']}",
                flush=True,
            )

    rows = pd.DataFrame(existing)
    rows = rows[
        rows.apply(
            lambda row: (row["scenario_id"], row["attack_key"], row["stage_key"])
            in expected_keys,
            axis=1,
        )
    ].copy()
    rows.to_csv(args.output_dir / "rollouts.csv", index=False, encoding="utf-8-sig")
    condition_summary, recovery_summary = summarize_results(rows, attack_keys)
    condition_summary.to_csv(
        args.output_dir / "condition_summary.csv", index=False, encoding="utf-8-sig"
    )
    recovery_summary.to_csv(
        args.output_dir / "recovery_summary.csv", index=False, encoding="utf-8-sig"
    )
    checks = []
    for row in recovery_summary.to_dict(orient="records"):
        checks.append(
            {
                "attack_key": row["attack_key"],
                "attack_effective": bool(row["attack_degradation_mean"] > 0.0),
                "defense_reward_improves": bool(row["defense_reward_gain_mean"] > 0.0),
                "defense_exit_violations_non_increasing": bool(
                    row["exit_vio_reduction_mean"] >= 0.0
                ),
                "defense_running_violations_non_increasing": bool(
                    row["run_vio_reduction_mean"] >= 0.0
                ),
                "positive_mean_recovery": bool(
                    math.isfinite(row["recovery_mean"]) and row["recovery_mean"] > 0.0
                ),
            }
        )
    clean_row = condition_summary[
        (condition_summary["attack_key"] == "clean")
        & (condition_summary["stage_key"] == "attack")
    ].iloc[0]
    clean_defended_row = condition_summary[
        (condition_summary["attack_key"] == "clean")
        & (condition_summary["stage_key"] == "ug_bcr")
    ].iloc[0]
    clean_check = {
        "reward_delta": float(
            clean_defended_row["reward_mean"] - clean_row["reward_mean"]
        ),
        "exit_vio_delta": float(
            clean_defended_row["exit_vio_mean"] - clean_row["exit_vio_mean"]
        ),
        "run_vio_delta": float(
            clean_defended_row["run_vio_mean"] - clean_row["run_vio_mean"]
        ),
        "reward_non_decreasing": bool(
            clean_defended_row["reward_mean"] >= clean_row["reward_mean"]
        ),
        "exit_violations_non_increasing": bool(
            clean_defended_row["exit_vio_mean"] <= clean_row["exit_vio_mean"]
        ),
        "running_violations_non_increasing": bool(
            clean_defended_row["run_vio_mean"] <= clean_row["run_vio_mean"]
        ),
    }
    clean_safe = all(
        clean_check[key]
        for key in (
            "reward_non_decreasing",
            "exit_violations_non_increasing",
            "running_violations_non_increasing",
        )
    )
    attack_recovery_success = all(
        row["attack_effective"]
        and row["defense_reward_improves"]
        and row["positive_mean_recovery"]
        for row in checks
    )
    final_status = {
        "algorithm": algorithm,
        "completed_rollouts": int(len(rows)),
        "expected_rollouts": int(len(expected_keys)),
        "complete": bool(len(rows) == len(expected_keys)),
        "elapsed_seconds_this_invocation": float(time.perf_counter() - started),
        "actor_divergence_vs_ddpg": divergence,
        "clean_safety_check": clean_check,
        "attack_recovery_success": bool(attack_recovery_success),
        "strict_zero_shot_full_expectation_met": bool(
            attack_recovery_success and clean_safe
        ),
        "checks": checks,
    }
    write_json(args.output_dir / "expectation_checks.json", final_status)
    print(json.dumps(final_status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
