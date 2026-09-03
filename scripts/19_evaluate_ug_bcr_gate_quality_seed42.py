from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

table_eval = importlib.import_module("_strength_eval_common")
from evc.ug_bcr_v3 import load_ug_bcr_v3_config, rollout_episode_with_ug_bcr_v3

DEFAULT_ATTACK_KEYS = ("local_small_drift_q", "local_deadline_drift_pgd")
GATE_QUALITY_PAPER_COLUMNS = (
    "Attack",
    "Type",
    "Oracle positive/%",
    "UG activation/%",
    "UG precision/%",
    "UG recall/%",
    "UG F1/%",
    "Selected improve/%",
    "Improve capture/%",
)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_std(values: pd.Series | np.ndarray | list[float]) -> float:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    return float(np.std(data, ddof=1)) if data.size > 1 else 0.0


def safe_rate(num: float, den: float) -> float:
    return 0.0 if float(den) <= 0.0 else float(num) / float(den)


def format_mean_std(mean: float, std: float, digits: int = 1) -> str:
    if not np.isfinite(mean):
        return "-"
    return f"{float(mean):.{digits}f}+/-{float(std):.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def build_gate_quality_paper_frame(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        capture = float(row["improvement_capture_pct"])
        rows.append(
            {
                "Attack": row["attack_display_name"],
                "Type": row["attack_family"],
                "Oracle positive/%": format_mean_std(
                    float(row["oracle_positive_rate_pct"]),
                    float(row["oracle_positive_scene_std_pct"]),
                    1,
                ),
                "UG activation/%": format_mean_std(
                    float(row["ug_activation_rate_pct"]),
                    float(row["ug_activation_scene_std_pct"]),
                    1,
                ),
                "UG precision/%": format_mean_std(
                    float(row["ug_precision_pct"]),
                    float(row["ug_precision_scene_std_pct"]),
                    1,
                ),
                "UG recall/%": format_mean_std(
                    float(row["ug_recall_pct"]),
                    float(row["ug_recall_scene_std_pct"]),
                    1,
                ),
                "UG F1/%": format_mean_std(
                    float(row["ug_f1_pct"]),
                    float(row["ug_f1_scene_std_pct"]),
                    1,
                ),
                "Selected improve/%": f"{float(row['selected_improve_rate_pct']):.1f}",
                "Improve capture/%": f"{capture:.1f}" if np.isfinite(capture) else "-",
            }
        )
    return pd.DataFrame(rows, columns=list(GATE_QUALITY_PAPER_COLUMNS))


def summarize_gate_quality(
    audit_frame: pd.DataFrame,
    *,
    core_error_threshold: float,
    improvement_margin: float,
    output_dir: Path,
    latest_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = audit_frame.copy()
    frame["oracle_positive"] = (
        (frame["policy_core_l1_error"].astype(float) >= float(core_error_threshold))
        & (
            frame["belief_core_l1_error"].astype(float) + float(improvement_margin)
            < frame["policy_core_l1_error"].astype(float)
        )
    )
    frame["gate_positive"] = frame["ug_belief_selected"].astype(int) > 0
    frame["selected_improved"] = (
        frame["gate_positive"]
        & (frame["selected_core_l1_error"].astype(float) + float(improvement_margin) < frame["policy_core_l1_error"].astype(float))
    )
    frame["selected_harmed"] = (
        frame["gate_positive"]
        & (frame["selected_core_l1_error"].astype(float) > frame["policy_core_l1_error"].astype(float) + float(improvement_margin))
    )
    frame["positive_improvement"] = np.maximum(frame["belief_l1_improvement"].astype(float), 0.0)

    raw_path = output_dir / "tables" / "ug_bcr_gate_quality_raw.csv"
    table_eval.atomic_csv(frame, raw_path)

    rows: list[dict[str, Any]] = []
    group_cols = ["attack_key", "attack_display_name", "attack_family"]
    for group_key, subset in frame.groupby(group_cols, sort=False):
        attack_key, display, family = group_key
        gate = subset["gate_positive"].to_numpy(dtype=bool)
        positive = subset["oracle_positive"].to_numpy(dtype=bool)
        improved = subset["selected_improved"].to_numpy(dtype=bool)
        harmed = subset["selected_harmed"].to_numpy(dtype=bool)
        tp = int(np.sum(gate & positive))
        fp = int(np.sum(gate & ~positive))
        fn = int(np.sum(~gate & positive))
        tn = int(np.sum(~gate & ~positive))
        precision = safe_rate(tp, tp + fp)
        recall = safe_rate(tp, tp + fn)
        f1 = 0.0 if precision + recall <= 0.0 else float(2.0 * precision * recall / (precision + recall))
        oracle_improvement = float(np.sum(subset.loc[positive, "positive_improvement"].astype(float)))
        captured_improvement = float(np.sum(subset.loc[gate & positive, "positive_improvement"].astype(float)))
        capture = float(captured_improvement / oracle_improvement) if oracle_improvement > 1e-12 else float("nan")
        scenario_values = []
        for _, scene_subset in subset.groupby("scenario_id", sort=False):
            scene_gate = scene_subset["gate_positive"].to_numpy(dtype=bool)
            scene_positive = scene_subset["oracle_positive"].to_numpy(dtype=bool)
            scene_tp = int(np.sum(scene_gate & scene_positive))
            scene_fp = int(np.sum(scene_gate & ~scene_positive))
            scene_fn = int(np.sum(~scene_gate & scene_positive))
            scene_precision = safe_rate(scene_tp, scene_tp + scene_fp)
            scene_recall = safe_rate(scene_tp, scene_tp + scene_fn)
            scene_f1 = (
                0.0
                if scene_precision + scene_recall <= 0.0
                else float(2.0 * scene_precision * scene_recall / (scene_precision + scene_recall))
            )
            scenario_values.append(
                {
                    "precision": scene_precision * 100.0,
                    "recall": scene_recall * 100.0,
                    "f1": scene_f1 * 100.0,
                    "activation": float(np.mean(scene_gate) * 100.0),
                    "positive": float(np.mean(scene_positive) * 100.0),
                }
            )
        scene_frame = pd.DataFrame(scenario_values)
        row = {
            "attack_key": str(attack_key),
            "attack_display_name": str(display),
            "attack_family": str(family),
            "scenario_count": int(subset["scenario_id"].nunique()),
            "sample_count": int(len(subset)),
            "attacked_sample_rate_pct": float(np.mean(subset["attacked_flag"].astype(int)) * 100.0),
            "oracle_positive_rate_pct": float(np.mean(positive) * 100.0),
            "ug_activation_rate_pct": float(np.mean(gate) * 100.0),
            "ug_precision_pct": precision * 100.0,
            "ug_recall_pct": recall * 100.0,
            "ug_f1_pct": f1 * 100.0,
            "false_activation_rate_pct": safe_rate(fp, fp + tn) * 100.0,
            "miss_rate_pct": safe_rate(fn, tp + fn) * 100.0,
            "selected_improve_rate_pct": safe_rate(int(np.sum(improved)), int(np.sum(gate))) * 100.0,
            "selected_harm_rate_pct": safe_rate(int(np.sum(harmed)), int(np.sum(gate))) * 100.0,
            "improvement_capture_pct": capture * 100.0 if np.isfinite(capture) else float("nan"),
            "policy_core_l1_error_mean": float(np.mean(subset["policy_core_l1_error"].astype(float))),
            "belief_core_l1_error_mean": float(np.mean(subset["belief_core_l1_error"].astype(float))),
            "selected_core_l1_error_mean": float(np.mean(subset["selected_core_l1_error"].astype(float))),
            "belief_l1_improvement_mean": float(np.mean(subset["belief_l1_improvement"].astype(float))),
            "ug_precision_scene_std_pct": sample_std(scene_frame["precision"]) if not scene_frame.empty else 0.0,
            "ug_recall_scene_std_pct": sample_std(scene_frame["recall"]) if not scene_frame.empty else 0.0,
            "ug_f1_scene_std_pct": sample_std(scene_frame["f1"]) if not scene_frame.empty else 0.0,
            "ug_activation_scene_std_pct": sample_std(scene_frame["activation"]) if not scene_frame.empty else 0.0,
            "oracle_positive_scene_std_pct": sample_std(scene_frame["positive"]) if not scene_frame.empty else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        family_order = {"clean": 0, "short": 1, "task": 2, "long": 3, "adaptive": 4}
        summary["_order"] = summary["attack_family"].map(family_order).fillna(99)
        summary = summary.sort_values(["_order", "attack_key"], kind="mergesort").drop(columns=["_order"])

    paper = build_gate_quality_paper_frame(summary)

    tables_dir = output_dir / "tables"
    table_eval.atomic_csv(summary, tables_dir / "ug_bcr_gate_quality_summary.csv")
    table_eval.atomic_csv(paper, tables_dir / "ug_bcr_gate_quality_paper.csv")
    note = (
        f"Oracle positive means the belief candidate reduces core L1 error by more than {improvement_margin:g} "
        f"while the DAE/DET policy core error is at least {core_error_threshold:g}. "
        "Metrics are computed on runtime UG-BCR audit records before Temporal Shield."
    )
    md = "# UG-BCR Gate Quality\n\n" + markdown_table(paper) + "\n" + note + "\n"
    atomic_text(tables_dir / "ug_bcr_gate_quality_paper.md", md)
    if latest_dir is not None:
        latest_dir.mkdir(parents=True, exist_ok=True)
        table_eval.atomic_csv(paper, latest_dir / "table5_ug_bcr_gate_quality_latest.csv")
        atomic_text(latest_dir / "table5_ug_bcr_gate_quality_latest.md", md)
    return summary, paper


def attack_family(attack_key: str, algorithm: str | None) -> str:
    if attack_key == "clean":
        return "clean"
    if algorithm in {"opposite_pgd", "opposite_fgsm", "q_function"}:
        return "short"
    if algorithm == "electhacker":
        return "task"
    if algorithm in {"local_small_drift_q", "local_deadline_drift_pgd"}:
        return "long"
    if algorithm == "full_pipeline_adaptive_deadline":
        return "adaptive"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UG-BCR urgency-gate quality on runtime state records.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor-path", type=Path, default=table_eval.EP100_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=table_eval.EP100_BUNDLE_PATH)
    parser.add_argument("--dtsr-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dtsr_seed42_multiday")
    parser.add_argument("--dae-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday")
    parser.add_argument("--detector-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "det_seed42_fullstate")
    parser.add_argument("--shield-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "shield_seed42_fullstate")
    parser.add_argument(
        "--ug-bcr-config-path",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / "ug_bcr_seed42_fullstate" / "ug_bcr_config.json",
    )
    parser.add_argument(
        "--ug-bcr-v3-config-path",
        type=Path,
        default=None,
        help="Optional independent v3 config. When set, its frozen base_v2 belief estimator is used.",
    )
    parser.add_argument(
        "--price-threshold-file",
        type=Path,
        default=PACKAGE_ROOT
        / "results"
        / "attack120_short_horizon"
        / "ehc_threshold_fix"
        / "electhacker_c_price_threshold.json",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument(
        "--scene-start",
        type=int,
        default=1,
        help="One-based first scenario after sorting by Scenario_ID; default preserves the original first-N behavior.",
    )
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument(
        "--attack-seed-base",
        type=int,
        default=None,
        help="Independent rollout/attacker seed base. Defaults to --seed for backward compatibility.",
    )
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--attack-keys", default=",".join(DEFAULT_ATTACK_KEYS))
    parser.add_argument("--include-clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mode",
        choices=["gate", "attack-only"],
        default="gate",
        help="gate runs the frozen UG-BCR pipeline and audits its selector; attack-only records paired undefended baselines.",
    )
    parser.add_argument(
        "--formal-long-strength-mapping",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the experiment-17 strength mapping to the two formal long-horizon attacks.",
    )
    parser.add_argument("--core-error-threshold", type=float, default=0.010)
    parser.add_argument("--improvement-margin", type=float, default=0.003)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "ug_bcr_gate_quality_20scenes_seed42",
    )
    parser.add_argument("--latest-dir", type=Path, default=PACKAGE_ROOT / "results" / "paper_tables_latest")
    parser.add_argument("--max-rollouts", type=int, default=0, help="Debug only; 0 runs the complete experiment.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if int(args.seed) != 42:
        raise ValueError("UG-BCR gate-quality experiment is fixed to seed=42 artifacts.")
    if int(args.scenes) <= 0:
        raise ValueError("--scenes must be positive.")
    if int(args.scene_start) <= 0:
        raise ValueError("--scene-start must be positive and one-based.")
    if not np.isfinite(float(args.epsilon)) or float(args.epsilon) <= 0.0:
        raise ValueError("--epsilon must be finite and positive.")

    attack_keys = tuple(table_eval.parse_key_list(args.attack_keys))
    attack_lookup = {spec["key"]: spec for spec in table_eval.ATTACK_SPECS}
    unknown = sorted(set(attack_keys) - set(attack_lookup))
    if unknown:
        raise ValueError(f"Unknown attack keys: {unknown}")
    selected_specs = [attack_lookup[key] for key in attack_keys]
    if args.include_clean:
        selected_specs = [attack_lookup["clean"]] + selected_specs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_result = args.output_dir / "final_status.json"
    if existing_result.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite completed result: {existing_result}")
    if args.overwrite:
        for path in [
            args.output_dir / "tables" / "ug_bcr_gate_quality_raw.csv",
            args.output_dir / "tables" / "ug_bcr_gate_quality_summary.csv",
            args.output_dir / "tables" / "ug_bcr_gate_quality_paper.csv",
            args.output_dir / "tables" / "ug_bcr_gate_quality_paper.md",
            args.output_dir / "tables" / "ug_bcr_gate_rollout_summary.csv",
            args.output_dir / "final_status.json",
            args.output_dir / "run_config.json",
            args.output_dir / "scenario_order.csv",
        ]:
            if path.exists():
                path.unlink()

    table_eval.set_all_seeds(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = table_eval.resolve_device(args.device)
    actor = table_eval.load_actor_from_path(args.actor_path, device).eval()
    bundle_payload = table_eval.load_actor_critic_bundle(args.bundle_path, device)
    if not table_eval.actor_matches_bundle(actor, bundle_payload):
        raise RuntimeError("Selected actor does not match the actor in the baseline bundle.")
    critic_state = bundle_payload.get("critic_state_dict")
    if critic_state is None:
        raise RuntimeError("The selected ep100 bundle has no critic_state_dict.")
    critic = table_eval.Critic().to(device)
    critic.load_state_dict(critic_state)
    actor.eval()
    critic.eval()
    for module in (actor, critic):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    dae_path = table_eval.existing_artifact_path(args.dae_artifact_dir, args.dtsr_dir, "dtsr_dae.pt")
    detector_path = table_eval.existing_artifact_path(args.detector_artifact_dir, args.dtsr_dir, "dtsr_detector.pt")
    shield_path = table_eval.existing_artifact_path(args.shield_artifact_dir, args.dtsr_dir, "dtsr_temporal_shield.pt")
    dae = table_eval.load_dae(dae_path, device).eval()
    detector_artifact = table_eval.load_detector(detector_path, device)
    detector_model = detector_artifact.model
    detector_threshold = float(detector_artifact.threshold)
    shield_config = table_eval.load_temporal_shield_bundle(shield_path).config
    ug_bcr_v3_config = None
    if args.ug_bcr_v3_config_path is not None:
        if not args.ug_bcr_v3_config_path.exists():
            raise FileNotFoundError(f"Missing UG-BCR v3 config: {args.ug_bcr_v3_config_path}")
        ug_bcr_v3_config = load_ug_bcr_v3_config(args.ug_bcr_v3_config_path)
        ug_bcr_config = ug_bcr_v3_config.base_v2
    else:
        if not args.ug_bcr_config_path.exists():
            raise FileNotFoundError(f"Missing UG-BCR config: {args.ug_bcr_config_path}")
        ug_bcr_config = table_eval.load_ug_bcr_config(args.ug_bcr_config_path)
    price_threshold = (
        table_eval.load_price_threshold_from_path(args.price_threshold_file)
        if args.price_threshold_file.exists()
        else table_eval.load_price_threshold(args.dtsr_dir)
    )

    manifest = table_eval.load_manifest(args.split).sort_values("Scenario_ID", kind="mergesort").reset_index(drop=True)
    scene_offset = int(args.scene_start) - 1
    scene_stop = scene_offset + int(args.scenes)
    if len(manifest) < scene_stop:
        raise RuntimeError(
            f"Requested scenarios {args.scene_start}..{scene_stop}, found only {len(manifest)} in split {args.split}."
        )
    manifest = manifest.iloc[scene_offset:scene_stop].copy().reset_index(drop=True)
    table_eval.atomic_csv(manifest, args.output_dir / "scenario_order.csv")
    attack_seed_base = int(args.seed if args.attack_seed_base is None else args.attack_seed_base)
    v3_config_hash = None if args.ug_bcr_v3_config_path is None else sha256_file(args.ug_bcr_v3_config_path)

    run_config = {
        "seed": int(args.seed),
        "attack_seed_base": attack_seed_base,
        "attack_seed_rule": "attack_seed_base + attack_index*100000 + local_episode_index(1..N)",
        "split": str(args.split),
        "scene_start": int(args.scene_start),
        "scenario_count": int(len(manifest)),
        "first_scenario_id": str(manifest.iloc[0]["Scenario_ID"]),
        "last_scenario_id": str(manifest.iloc[-1]["Scenario_ID"]),
        "attack_keys": [str(spec["key"]) for spec in selected_specs],
        "mode": str(args.mode),
        "epsilon": float(args.epsilon),
        "core_error_threshold": float(args.core_error_threshold),
        "improvement_margin": float(args.improvement_margin),
        "audit_point": "after DAE/DET route and UG-BCR gate, before Temporal Shield",
        "oracle_positive": "belief core L1 error + margin < policy core L1 error and policy core L1 error >= threshold",
        "actor_path": str(args.actor_path),
        "dae_artifact": str(dae_path),
        "detector_artifact": str(detector_path),
        "shield_artifact": str(shield_path),
        "ug_bcr_config": str(args.ug_bcr_config_path),
        "ug_bcr_v3_config": None if args.ug_bcr_v3_config_path is None else str(args.ug_bcr_v3_config_path),
        "ug_bcr_v3_config_sha256": v3_config_hash,
        "ug_bcr_version": 3 if ug_bcr_v3_config is not None else 2,
        "formal_long_strength_mapping_applied": bool(args.formal_long_strength_mapping),
        "formal_long_strength_mapping": {
            "local_deadline_drift_pgd": "outer=epsilon; inner=0.028*(epsilon/0.055); alpha=0.008*(epsilon/0.055); iters=5",
            "local_small_drift_q": "formal-v2 nominal attacker scaled by epsilon/0.055; inner iters=5",
        },
    }
    table_eval.write_json(args.output_dir / "run_config.json", run_config)

    attack_seed_offsets = {
        str(spec["key"]): index * 100_000 for index, spec in enumerate(selected_specs)
    }
    audit_records: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    expected_rollouts = int(len(manifest) * len(selected_specs))
    completed = 0
    started = time.perf_counter()

    for episode_index, (_, scenario_row) in enumerate(manifest.iterrows(), start=1):
        arrivals, signal_path, scenario_id = table_eval.load_scenario(scenario_row)
        for attack_spec in selected_specs:
            attack_key = str(attack_spec["key"])
            algorithm = attack_spec["algorithm"]
            attack_seed = int(attack_seed_base + attack_seed_offsets[attack_key] + episode_index)
            table_eval.set_all_seeds(attack_seed)
            attacker = table_eval.build_attacker_for_rollout(
                attack_spec=attack_spec,
                actor=actor,
                critic=critic,
                device=device,
                arrivals=arrivals,
                signal_path=signal_path,
                attack_seed=attack_seed,
                epsilon=float(args.epsilon),
                dae=dae,
                detector_model=detector_model,
                detector_threshold=detector_threshold,
                shield_config=shield_config,
                ug_bcr_config=ug_bcr_config,
                formal_long_outer_epsilon=(
                    float(args.epsilon)
                    if bool(args.formal_long_strength_mapping)
                    and algorithm in {"local_small_drift_q", "local_deadline_drift_pgd"}
                    else None
                ),
            )
            base_attacker = None if attacker is None else getattr(attacker, "base_attacker", None)
            context = {
                "scenario_id": scenario_id,
                "episode_index": int(episode_index),
                "attack_key": attack_key,
                "attack_display_name": str(attack_spec["display"]),
                "attack_family": attack_family(attack_key, None if algorithm is None else str(algorithm)),
                "algorithm": None if algorithm is None else str(algorithm),
                "attack_seed": int(attack_seed),
                "epsilon": float(args.epsilon),
                "configured_outer_epsilon": (
                    None if attacker is None else float(getattr(attacker, "epsilon", args.epsilon))
                ),
                "configured_inner_epsilon": (
                    None if base_attacker is None else float(getattr(base_attacker, "epsilon", float("nan")))
                ),
                "configured_alpha": (
                    None if base_attacker is None else float(getattr(base_attacker, "alpha", float("nan")))
                ),
                "configured_iters": (
                    None if base_attacker is None else int(getattr(base_attacker, "iters", 0))
                ),
            }
            rollout_start = time.perf_counter()
            if args.mode == "attack-only":
                rollout_function = table_eval.rollout_episode_with_ug_bcr
                defense_kwargs = {
                    "defender": None,
                    "detector_model": None,
                    "detector_threshold": None,
                    "shield_config": None,
                    "route_mode": "none",
                    "enable_shield": False,
                    "enable_belief": False,
                    "enable_urgency_gate": False,
                    "ug_bcr_config": None,
                }
            else:
                rollout_function = (
                    rollout_episode_with_ug_bcr_v3
                    if ug_bcr_v3_config is not None
                    else table_eval.rollout_episode_with_ug_bcr
                )
                defense_kwargs = {
                    "defender": dae,
                    "detector_model": detector_model,
                    "detector_threshold": detector_threshold,
                    "shield_config": shield_config,
                    "route_mode": "detector",
                    "enable_shield": True,
                    "enable_belief": True,
                    "enable_urgency_gate": True,
                    **(
                        {"ug_bcr_v3_config": ug_bcr_v3_config}
                        if ug_bcr_v3_config is not None
                        else {"ug_bcr_config": ug_bcr_config}
                    ),
                }
            summary = rollout_function(
                arrivals,
                actor,
                signal_path,
                device,
                table_eval.TRAIN_PROFILE,
                attack_enabled=attack_key != "clean",
                attack_scenario=str(attack_spec["scenario"]),
                attacker=attacker,
                epsilon=float(args.epsilon),
                state_scope=str(attack_spec["scope"]),
                price_threshold=float(price_threshold),
                attack_ratio=1.0,
                attack_scope="obs",
                label=f"{attack_key}__ug_bcr_gate_quality",
                repair_mode=table_eval.REPAIR_MODE,
                audit_records=audit_records if args.mode == "gate" else None,
                audit_context=context if args.mode == "gate" else None,
                **defense_kwargs,
            )
            runtime_seconds = float(time.perf_counter() - rollout_start)
            scalar = table_eval.to_scalar_summary(summary)
            rollout_row = {
                **context,
                "runtime_seconds": runtime_seconds,
                **scalar,
            }
            rollout_rows.append(rollout_row)
            completed += 1
            print(
                f"[{completed:03d}/{expected_rollouts}] scene={episode_index:02d} "
                f"{attack_spec['display']} reward={float(rollout_row['ep_reward']):.2f} "
                f"route={float(rollout_row.get('route_rate', 0.0)) * 100.0:.1f}% "
                f"belief={float(rollout_row.get('urgency_gate_belief_rate', 0.0)) * 100.0:.1f}% "
                f"run/exit={int(rollout_row.get('run_vio', 0))}/{int(rollout_row.get('exit_vio', 0))} "
                f"time={runtime_seconds:.2f}s",
                flush=True,
            )
            if args.max_rollouts > 0 and completed >= int(args.max_rollouts):
                print(f"Debug stop after {completed} rollouts.", flush=True)
                return
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rollout_frame = pd.DataFrame(rollout_rows)
    table_eval.atomic_csv(rollout_frame, args.output_dir / "tables" / "ug_bcr_gate_rollout_summary.csv")
    elapsed = float(time.perf_counter() - started)
    if args.mode == "attack-only":
        table_eval.write_json(
            args.output_dir / "final_status.json",
            {
                "completed_rollouts": int(completed),
                "expected_rollouts": int(expected_rollouts),
                "elapsed_seconds": elapsed,
                "mode": "attack-only",
                "rollout_summary": str(args.output_dir / "tables" / "ug_bcr_gate_rollout_summary.csv"),
            },
        )
        print(f"Completed {completed}/{expected_rollouts} attack-only rollouts in {elapsed / 60.0:.1f} min.", flush=True)
        print(f"Saved: {args.output_dir / 'tables' / 'ug_bcr_gate_rollout_summary.csv'}", flush=True)
        return
    audit_frame = pd.DataFrame(audit_records)
    if audit_frame.empty:
        raise RuntimeError("No UG-BCR audit records were collected.")
    summary, paper = summarize_gate_quality(
        audit_frame,
        core_error_threshold=float(args.core_error_threshold),
        improvement_margin=float(args.improvement_margin),
        output_dir=args.output_dir,
        latest_dir=args.latest_dir,
    )
    table_eval.write_json(
        args.output_dir / "final_status.json",
        {
            "completed_rollouts": int(completed),
            "expected_rollouts": int(expected_rollouts),
            "audit_records": int(len(audit_frame)),
            "elapsed_seconds": elapsed,
            "summary_table": str(args.output_dir / "tables" / "ug_bcr_gate_quality_summary.csv"),
            "paper_table": str(args.output_dir / "tables" / "ug_bcr_gate_quality_paper.csv"),
            "latest_table": str(args.latest_dir / "table5_ug_bcr_gate_quality_latest.csv"),
            "summary_rows": int(len(summary)),
            "paper_rows": int(len(paper)),
        },
    )
    print(f"Completed {completed}/{expected_rollouts} rollouts in {elapsed / 60.0:.1f} min.", flush=True)
    print(f"Saved: {args.output_dir / 'tables' / 'ug_bcr_gate_quality_paper.md'}", flush=True)
    print(f"Saved: {args.latest_dir / 'table5_ug_bcr_gate_quality_latest.md'}", flush=True)


if __name__ == "__main__":
    main()
