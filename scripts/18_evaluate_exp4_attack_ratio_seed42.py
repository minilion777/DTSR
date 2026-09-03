from __future__ import annotations

import argparse
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

from cli import posterior_detector_dataset_from_unified_pair
from evc.defense import posterior_detector_probabilities
from evc.merged_pipeline import build_pair_dataset_from_clean_trajectories, collect_clean_trajectories

table_eval = importlib.import_module("_strength_eval_common")


DEFAULT_ATTACK_KEYS = ("opposite_pgd", "q_function", "opposite_fgsm")
DEFAULT_RATIOS = (0.0, 0.25, 0.50, 0.75, 1.00)
STAGE_KEYS = ("attack", "denoise", "denoise_det", "shield", "ug_bcr")
SHORT_ALGORITHMS = {"opposite_pgd", "q_function", "opposite_fgsm"}
STAGE_LABELS = {
    "attack": "Raw attack",
    "denoise": "Denoise",
    "denoise_det": "Denoise+DET",
    "shield": "+Shield",
    "ug_bcr": "+UG-BCR",
}
DET_ROUTING_SUMMARY_METRICS = (
    "attack_sample_rate_pct",
    "benefit_positive_rate_pct",
    "route_rate_pct",
    "beneficial_routing_precision_pct",
    "beneficial_routing_recall_pct",
    "benefit_routing_f1_pct",
    "harmful_route_rate_pct",
    "nonbeneficial_rejection_rate_pct",
    "net_benefit_capture_rate_pct",
)
DET_ROUTING_PAPER_METRICS = (
    ("benefit_positive_rate_pct", "Benefit positive/%"),
    ("beneficial_routing_precision_pct", "BR Precision/%"),
    ("beneficial_routing_recall_pct", "BR Recall/%"),
    ("benefit_routing_f1_pct", "BR F1/%"),
    ("net_benefit_capture_rate_pct", "NBCR/%"),
)
DET_ROUTING_PAPER_COLUMNS = (
    "Attack",
    "rho",
    *(label for _, label in DET_ROUTING_PAPER_METRICS),
)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in str(raw).split(",") if token.strip())
    if not values:
        raise ValueError("At least one ratio value is required.")
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Every ratio must be finite and in [0, 1].")
    if len(set(values)) != len(values):
        raise ValueError("Ratio list must not contain duplicates.")
    return values


def ratio_key(value: float | None) -> str:
    return "clean" if value is None else f"{float(value):.4f}"


def format_mean_std(mean: float, std: float, digits: int = 1) -> str:
    if not np.isfinite(mean):
        return "-"
    return f"{float(mean):.{digits}f}±{float(std):.{digits}f}"


def sample_std(values: pd.Series | np.ndarray | list[float]) -> float:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    return float(np.std(data, ddof=1)) if data.size > 1 else 0.0


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


def condition_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row["scenario_id"]),
        str(row["attack_key"]),
        str(row["ratio_key"]),
        str(row["stage_key"]),
    )


def build_conditions(attack_keys: tuple[str, ...], ratios: tuple[float, ...]) -> list[dict[str, Any]]:
    attack_lookup = {spec["key"]: spec for spec in table_eval.ATTACK_SPECS}
    stage_lookup = {spec["key"]: spec for spec in table_eval.STAGE_SPECS}
    conditions: list[dict[str, Any]] = []
    for stage_key in STAGE_KEYS:
        conditions.append(
            {
                "attack_base_key": "clean",
                "attack_key": "clean",
                "algorithm": None,
                "attack_display_name": "Clean",
                "attack_scenario": "O",
                "attack_state_scope": "all",
                "attack_ratio": None,
                "ratio_key": "clean",
                "stage_key": stage_key,
                "stage_display_name": stage_lookup[stage_key]["display"],
                "seen_in_dtsr_training": False,
            }
        )
    for attack_key in attack_keys:
        attack_spec = attack_lookup[attack_key]
        if attack_spec["algorithm"] not in SHORT_ALGORITHMS:
            raise ValueError(f"Experiment 4 accepts only short-horizon attacks; got {attack_key}.")
        for ratio in ratios:
            for stage_key in STAGE_KEYS:
                conditions.append(
                    {
                        "attack_base_key": attack_key,
                        "attack_key": f"{attack_key}_rho_{ratio_key(ratio)}",
                        "algorithm": attack_spec["algorithm"],
                        "attack_display_name": f"{attack_spec['display']} (rho={float(ratio):.2f})",
                        "attack_scenario": attack_spec["scenario"],
                        "attack_state_scope": attack_spec["scope"],
                        "attack_ratio": float(ratio),
                        "ratio_key": ratio_key(ratio),
                        "stage_key": stage_key,
                        "stage_display_name": stage_lookup[stage_key]["display"],
                        "seen_in_dtsr_training": bool(attack_spec["seen"]),
                    }
                )
    return conditions


def summarize_outputs(long_df: pd.DataFrame, output_dir: Path, latest_dir: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_raw = long_df[
        (long_df["attack_base_key"] == "clean") & (long_df["stage_key"] == "attack")
    ][["scenario_id", "ep_reward"]].rename(columns={"ep_reward": "clean_raw_reward"})
    if clean_raw["scenario_id"].nunique() != long_df["scenario_id"].nunique():
        raise RuntimeError("Missing clean raw baseline for at least one scenario.")

    raw_rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    attack_order = (
        long_df[long_df["attack_base_key"] != "clean"][
            ["attack_base_key", "attack_display_base", "attack_ratio", "ratio_key"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    for _, meta in attack_order.iterrows():
        attack_data = long_df[
            (long_df["attack_base_key"] == meta["attack_base_key"])
            & (long_df["ratio_key"] == meta["ratio_key"])
        ]
        raw_attack = attack_data[attack_data["stage_key"] == "attack"][
            ["scenario_id", "ep_reward"]
        ].rename(columns={"ep_reward": "attack_reward"})
        full = attack_data[attack_data["stage_key"] == "ug_bcr"][
            ["scenario_id", "ep_reward", "route_rate"]
        ].rename(columns={"ep_reward": "defended_reward"})
        paired = clean_raw.merge(raw_attack, on="scenario_id", how="inner").merge(
            full, on="scenario_id", how="inner"
        )
        denominator = paired["clean_raw_reward"].to_numpy(dtype=float) - paired["attack_reward"].to_numpy(dtype=float)
        numerator = paired["defended_reward"].to_numpy(dtype=float) - paired["attack_reward"].to_numpy(dtype=float)
        recovery = np.full(denominator.shape, np.nan, dtype=np.float64)
        valid = np.abs(denominator) > 1e-8
        recovery[valid] = numerator[valid] / denominator[valid] * 100.0
        route_values = paired["route_rate"].to_numpy(dtype=float) * 100.0
        clean_drop = paired["clean_raw_reward"].to_numpy(dtype=float) - paired["defended_reward"].to_numpy(dtype=float)

        raw_row: dict[str, Any] = {
            "attack_base_key": str(meta["attack_base_key"]),
            "attack": str(meta["attack_display_base"]),
            "attack_ratio": float(meta["attack_ratio"]),
            "ratio_key": str(meta["ratio_key"]),
            "scenario_count": int(len(paired)),
            "recovery_mean_pct": float(np.nanmean(recovery)) if np.isfinite(recovery).any() else float("nan"),
            "recovery_std_pct": sample_std(recovery),
            "route_rate_mean_pct": float(np.mean(route_values)),
            "route_rate_std_pct": sample_std(route_values),
            "clean_drop_mean": float(np.mean(clean_drop)),
            "clean_drop_std": sample_std(clean_drop),
        }
        paper_row: dict[str, Any] = {
            "Attack type": str(meta["attack_display_base"]),
            "rho": f"{float(meta['attack_ratio']):.2f}",
        }
        for stage_key in STAGE_KEYS:
            subset = attack_data[attack_data["stage_key"] == stage_key]
            rewards = subset["ep_reward"].astype(float).to_numpy()
            raw_row[f"{stage_key}_reward_mean"] = float(np.mean(rewards))
            raw_row[f"{stage_key}_reward_std"] = sample_std(rewards)
            raw_row[f"{stage_key}_scenario_count"] = int(len(rewards))
            paper_row[STAGE_LABELS[stage_key]] = format_mean_std(
                raw_row[f"{stage_key}_reward_mean"], raw_row[f"{stage_key}_reward_std"], 1
            )
        paper_row["Recovery/%"] = format_mean_std(raw_row["recovery_mean_pct"], raw_row["recovery_std_pct"], 1)
        paper_row["Route/%"] = format_mean_std(raw_row["route_rate_mean_pct"], raw_row["route_rate_std_pct"], 1)
        paper_row["Clean loss"] = format_mean_std(raw_row["clean_drop_mean"], raw_row["clean_drop_std"], 1)
        raw_rows.append(raw_row)
        paper_rows.append(paper_row)

    tables_dir = output_dir / "tables"
    raw_frame = pd.DataFrame(raw_rows)
    paper_frame = pd.DataFrame(paper_rows)[
        [
            "Attack type",
            "rho",
            "Raw attack",
            "Denoise",
            "Denoise+DET",
            "+Shield",
            "+UG-BCR",
            "Recovery/%",
            "Route/%",
            "Clean loss",
        ]
    ]
    table_eval.atomic_csv(long_df, tables_dir / "exp4_attack_ratio_rollouts.csv")
    table_eval.atomic_csv(raw_frame, tables_dir / "exp4_attack_ratio_summary_raw.csv")
    table_eval.atomic_csv(paper_frame, tables_dir / "exp4_attack_ratio_paper.csv")
    note = (
        "Experiment 4 uses 20 fixed test scenes, seed=42 artifacts, epsilon=0.10, "
        "and varies the short-horizon attack coverage ratio rho. Values are mean±sample std."
    )
    md = "# Experiment 4 Attack-Ratio Robustness\n\n" + markdown_table(paper_frame) + "\n" + note + "\n"
    atomic_text(tables_dir / "exp4_attack_ratio_paper.md", md)
    if latest_dir is not None:
        latest_dir.mkdir(parents=True, exist_ok=True)
        table_eval.atomic_csv(paper_frame, latest_dir / "table4_attack_ratio_latest.csv")
        atomic_text(latest_dir / "table4_attack_ratio_latest.md", md)
    return raw_frame, paper_frame


def safe_rate(num: float, den: float) -> float:
    return 0.0 if float(den) <= 0.0 else float(num) / float(den)


def summarize_vector(values: pd.Series | np.ndarray | list[float]) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(data)), sample_std(data)


def create_det_routing_quality(
    *,
    manifest: pd.DataFrame,
    attack_keys: tuple[str, ...],
    ratios: tuple[float, ...],
    actor,
    critic,
    device: torch.device,
    dae,
    detector_model,
    detector_threshold: float,
    price_threshold: float,
    seed: int,
    epsilon: float,
    attack_seed_offsets: dict[str, int],
    output_dir: Path,
    latest_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attack_lookup = {spec["key"]: spec for spec in table_eval.ATTACK_SPECS}
    rows: list[dict[str, Any]] = []
    audit_ratios = tuple(float(value) for value in ratios if float(value) > 0.0)
    if not audit_ratios:
        empty = pd.DataFrame()
        return empty, empty, empty
    audit_total = int(len(manifest) * len(attack_keys) * len(audit_ratios))
    audit_done = 0
    print(
        f"[det-audit start] scenarios={len(manifest)} attacks={len(attack_keys)} "
        f"ratios={len(audit_ratios)} total={audit_total}",
        flush=True,
    )

    for episode_index, (_, scenario_row) in enumerate(manifest.iterrows(), start=1):
        arrivals, signal_path, scenario_id = table_eval.load_scenario(scenario_row)
        env = table_eval.ChargingEnv(signals_path=signal_path, reward_profile=table_eval.TRAIN_PROFILE)
        max_duration = max(12, int(arrivals["Duration_of_stay"].max()))
        low, high = env.observation_bounds(max_duration_of_stay=max_duration)
        clean_bundle = collect_clean_trajectories(
            arrivals,
            actor,
            signal_path,
            device,
            reward_profile=table_eval.TRAIN_PROFILE,
            episodes=1,
        )
        for attack_key in attack_keys:
            attack_spec = attack_lookup[attack_key]
            attack_seed = int(seed + attack_seed_offsets[attack_key] + episode_index)
            for ratio in audit_ratios:
                attacker = table_eval.build_short_attacker(
                    algorithm=str(attack_spec["algorithm"]),
                    actor=actor,
                    critic=critic,
                    device=device,
                    low=low,
                    high=high,
                    seed=attack_seed,
                    epsilon=float(epsilon),
                )
                pair_bundle = build_pair_dataset_from_clean_trajectories(
                    clean_bundle,
                    attacker,
                    str(attack_spec["scenario"]),
                    price_threshold=float(price_threshold),
                    attack_ratio=float(ratio),
                    attack_scope="obs",
                )
                detector_dataset = posterior_detector_dataset_from_unified_pair(
                    pair_bundle,
                    actor,
                    dae,
                    device,
                    profile_tag=f"exp4_ratio_{attack_key}_{ratio_key(ratio)}",
                    train_attack_tags=[attack_key],
                    benefit_margin=0.0,
                    benefit_action_weight=1.0,
                    benefit_state_weight=1.0,
                    posterior_label_mode="benefit",
                    use_benefit_sample_weights=False,
                    state_scope=str(attack_spec["scope"]),
                    repair_mode=table_eval.REPAIR_MODE,
                )
                # The DET training helper appends a clean-identity half for training balance.
                # For this experiment's routing audit, evaluate only the mixed observation stream.
                source_count = int((detector_dataset.metadata or {}).get("source_samples", pair_bundle.clean_inputs.shape[0]))
                obs_inputs = detector_dataset.obs_inputs[:source_count]
                rec_inputs = detector_dataset.rec_inputs[:source_count]
                prev_obs_inputs = None if detector_dataset.prev_obs_inputs is None else detector_dataset.prev_obs_inputs[:source_count]
                time_indices = None if detector_dataset.time_indices is None else detector_dataset.time_indices[:source_count]
                stations = None if detector_dataset.stations is None else detector_dataset.stations[:source_count]
                is_new_arrivals = None if detector_dataset.is_new_arrivals is None else detector_dataset.is_new_arrivals[:source_count]
                attack_mask = (
                    np.zeros((source_count,), dtype=bool)
                    if detector_dataset.attack_mask is None
                    else np.asarray(detector_dataset.attack_mask[:source_count], dtype=np.int64).reshape(-1) > 0
                )
                benefits = np.asarray(detector_dataset.benefit_scores[:source_count], dtype=np.float64).reshape(-1)
                labels = benefits > 0.0
                probabilities = posterior_detector_probabilities(
                    detector_model,
                    obs_inputs,
                    rec_inputs,
                    actor,
                    device,
                    time_indices=time_indices,
                    stations=stations,
                    is_new_arrivals=is_new_arrivals,
                    prev_obs_inputs=prev_obs_inputs,
                    include_temporal=bool(getattr(detector_model, "include_temporal", True)),
                )
                routed = np.asarray(probabilities, dtype=np.float64).reshape(-1) >= float(detector_threshold)
                tp = int(np.sum(routed & labels))
                fp = int(np.sum(routed & ~labels))
                fn = int(np.sum(~routed & labels))
                tn = int(np.sum(~routed & ~labels))
                positive_count = int(np.sum(labels))
                negative_count = int(np.sum(~labels))
                precision = safe_rate(tp, tp + fp)
                recall = safe_rate(tp, tp + fn)
                f1 = 0.0 if precision + recall <= 0.0 else float(2.0 * precision * recall / (precision + recall))
                harmful_route_rate = safe_rate(fp, fp + tn)
                nonbeneficial_rejection = safe_rate(tn, tn + fp)
                positive_benefit = np.maximum(benefits, 0.0)
                nbcr_den = float(np.sum(positive_benefit))
                nbcr = float(np.sum(benefits[routed]) / nbcr_den) if nbcr_den > 1e-12 else float("nan")
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "episode_index": int(episode_index),
                        "attack_base_key": attack_key,
                        "attack": str(attack_spec["display"]),
                        "attack_ratio": float(ratio),
                        "rho": f"{float(ratio):.2f}",
                        "sample_count": int(source_count),
                        "attack_sample_count": int(np.sum(attack_mask)),
                        "attack_sample_rate_pct": float(np.mean(attack_mask) * 100.0),
                        "benefit_positive_count": positive_count,
                        "benefit_positive_rate_pct": float(np.mean(labels) * 100.0),
                        "benefit_negative_count": negative_count,
                        "route_count": int(np.sum(routed)),
                        "route_rate_pct": float(np.mean(routed) * 100.0),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tn": tn,
                        "beneficial_routing_precision_pct": precision * 100.0,
                        "beneficial_routing_recall_pct": recall * 100.0,
                        "benefit_routing_f1_pct": f1 * 100.0,
                        "harmful_route_rate_pct": harmful_route_rate * 100.0,
                        "nonbeneficial_rejection_rate_pct": nonbeneficial_rejection * 100.0,
                        "net_benefit_capture_rate_pct": nbcr * 100.0 if np.isfinite(nbcr) else float("nan"),
                        "benefit_sum_positive": nbcr_den,
                        "benefit_sum_routed": float(np.sum(benefits[routed])),
                        "detector_threshold": float(detector_threshold),
                    }
                )
                audit_done += 1
                print(
                    f"[det-audit {audit_done:04d}/{audit_total}] scene={episode_index:02d} "
                    f"{attack_spec['display']} | rho={float(ratio):.2f} "
                    f"benefit+={float(np.mean(labels) * 100.0):.1f}% "
                    f"route={float(np.mean(routed) * 100.0):.1f}% "
                    f"BR-recall={recall * 100.0:.1f}% "
                    f"NB-reject={nonbeneficial_rejection * 100.0:.1f}% "
                    f"NBCR={nbcr * 100.0:.1f}%",
                    flush=True,
                )

    raw_frame = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    paper_names = dict(DET_ROUTING_PAPER_METRICS)
    for (attack_key, ratio), subset in raw_frame.groupby(["attack_base_key", "attack_ratio"], sort=False):
        attack_name = str(subset["attack"].iloc[0])
        summary_row: dict[str, Any] = {
            "attack_base_key": str(attack_key),
            "attack": attack_name,
            "attack_ratio": float(ratio),
            "rho": f"{float(ratio):.2f}",
            "scenario_count": int(subset["scenario_id"].nunique()),
            "sample_count_mean": float(subset["sample_count"].mean()),
        }
        paper_row: dict[str, Any] = {"Attack": attack_name, "rho": f"{float(ratio):.2f}"}
        for column in DET_ROUTING_SUMMARY_METRICS:
            mean, std = summarize_vector(subset[column])
            summary_row[f"{column}_mean"] = mean
            summary_row[f"{column}_std"] = std
            if column in paper_names:
                paper_row[paper_names[column]] = format_mean_std(mean, std, 1)
        summary_rows.append(summary_row)
        paper_rows.append(paper_row)

    summary_frame = pd.DataFrame(summary_rows)
    paper_frame = pd.DataFrame(paper_rows)[list(DET_ROUTING_PAPER_COLUMNS)]
    tables_dir = output_dir / "tables"
    table_eval.atomic_csv(raw_frame, tables_dir / "exp4_det_routing_quality_raw.csv")
    table_eval.atomic_csv(summary_frame, tables_dir / "exp4_det_routing_quality_summary.csv")
    table_eval.atomic_csv(paper_frame, tables_dir / "exp4_det_routing_quality_paper.csv")
    note = (
        "DET routing quality is computed on the mixed observation stream only. "
        "Oracle positive means the full-state DAE candidate has positive decision-aware benefit."
    )
    md = "# Experiment 4 DET Routing Quality\n\n" + markdown_table(paper_frame) + "\n" + note + "\n"
    atomic_text(tables_dir / "exp4_det_routing_quality_paper.md", md)
    if latest_dir is not None:
        latest_dir.mkdir(parents=True, exist_ok=True)
        table_eval.atomic_csv(paper_frame, latest_dir / "table4_det_routing_quality_latest.csv")
        atomic_text(latest_dir / "table4_det_routing_quality_latest.md", md)
    return raw_frame, summary_frame, paper_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Experiment 4: defense robustness under mixed short-horizon attack ratios."
    )
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
        "--price-threshold-file",
        type=Path,
        default=PACKAGE_ROOT
        / "results"
        / "attack120_short_horizon"
        / "ehc_threshold_fix"
        / "electhacker_c_price_threshold.json",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--ratios", default=",".join(f"{value:.2f}" for value in DEFAULT_RATIOS))
    parser.add_argument("--attack-keys", default=",".join(DEFAULT_ATTACK_KEYS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "exp4_attack_ratio_20scenes_seed42",
    )
    parser.add_argument("--latest-dir", type=Path, default=PACKAGE_ROOT / "results" / "paper_tables_latest")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-routing-quality",
        action="store_true",
        help="Skip the offline DET decision-aware routing quality audit.",
    )
    parser.add_argument("--max-rollouts", type=int, default=0, help="Debug only; 0 runs the complete experiment.")
    args = parser.parse_args()

    if int(args.seed) != 42:
        raise ValueError("Experiment 4 is fixed to seed=42 trained artifacts.")
    if not np.isfinite(float(args.epsilon)) or float(args.epsilon) <= 0.0:
        raise ValueError("--epsilon must be finite and positive.")

    ratios = parse_float_list(args.ratios)
    attack_keys = tuple(table_eval.parse_key_list(args.attack_keys))
    if not attack_keys:
        raise ValueError("At least one attack key is required.")
    conditions = build_conditions(attack_keys, ratios)
    stage_lookup = {spec["key"]: spec for spec in table_eval.STAGE_SPECS}
    attack_lookup = {spec["key"]: spec for spec in table_eval.ATTACK_SPECS}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_path = args.output_dir / "intermediate" / "attack_ratio_rollouts.jsonl"
    if args.overwrite:
        for path in [
            intermediate_path,
            args.output_dir / "tables" / "exp4_attack_ratio_rollouts.csv",
            args.output_dir / "tables" / "exp4_attack_ratio_summary_raw.csv",
            args.output_dir / "tables" / "exp4_attack_ratio_paper.csv",
            args.output_dir / "tables" / "exp4_attack_ratio_paper.md",
            args.output_dir / "tables" / "exp4_det_routing_quality_raw.csv",
            args.output_dir / "tables" / "exp4_det_routing_quality_summary.csv",
            args.output_dir / "tables" / "exp4_det_routing_quality_paper.csv",
            args.output_dir / "tables" / "exp4_det_routing_quality_paper.md",
            args.output_dir / "final_status.json",
            args.output_dir / "run_config.json",
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
    if not args.ug_bcr_config_path.exists():
        raise FileNotFoundError(f"Missing UG-BCR config: {args.ug_bcr_config_path}")
    ug_bcr_config = table_eval.load_ug_bcr_config(args.ug_bcr_config_path)
    price_threshold = (
        table_eval.load_price_threshold_from_path(args.price_threshold_file)
        if args.price_threshold_file.exists()
        else table_eval.load_price_threshold(args.dtsr_dir)
    )

    manifest = table_eval.load_manifest(args.split).sort_values("Scenario_ID", kind="mergesort").reset_index(drop=True)
    if len(manifest) < int(args.scenes):
        raise RuntimeError(f"Requested {args.scenes} scenarios, found only {len(manifest)}.")
    manifest = manifest.iloc[: int(args.scenes)].copy().reset_index(drop=True)

    expected_keys = {
        (
            str(row["Scenario_ID"]),
            str(condition["attack_key"]),
            str(condition["ratio_key"]),
            str(condition["stage_key"]),
        )
        for _, row in manifest.iterrows()
        for condition in conditions
    }
    existing_rows = table_eval.load_jsonl(intermediate_path) if args.resume else []
    existing_rows = [row for row in existing_rows if condition_key(row) in expected_keys]
    completed_keys = {condition_key(row) for row in existing_rows}
    if len(completed_keys) != len(existing_rows):
        raise RuntimeError("Intermediate JSONL contains duplicate rollout keys.")

    run_config = {
        "seed": int(args.seed),
        "split": str(args.split),
        "scenario_count": int(len(manifest)),
        "attack_keys": list(attack_keys),
        "ratios": list(ratios),
        "epsilon": float(args.epsilon),
        "stage_keys": list(STAGE_KEYS),
        "expected_rollouts": int(len(expected_keys)),
        "attack_seed_rule": "seed + attack_index * 100000 + episode_index; shared across ratios and stages",
        "det_routing_quality": "offline mixed-observation stream audit, rho=0 excluded from panel (d)",
        "statistics": f"mean +/- sample std across {len(manifest)} paired scenarios",
        "actor_path": str(args.actor_path),
        "dae_artifact": str(dae_path),
        "detector_artifact": str(detector_path),
        "shield_artifact": str(shield_path),
        "ug_bcr_config": str(args.ug_bcr_config_path),
    }
    table_eval.write_json(args.output_dir / "run_config.json", run_config)

    attack_seed_offsets = {
        "clean": 0,
        **{attack_key: (index + 1) * 100_000 for index, attack_key in enumerate(attack_keys)},
    }
    expected_total = len(expected_keys)
    started = time.perf_counter()
    new_rollouts = 0
    for episode_index, (_, scenario_row) in enumerate(manifest.iterrows(), start=1):
        arrivals, signal_path, scenario_id = table_eval.load_scenario(scenario_row)
        env = table_eval.ChargingEnv(signals_path=signal_path, reward_profile=table_eval.TRAIN_PROFILE)
        max_duration = max(12, int(arrivals["Duration_of_stay"].max()))
        low, high = env.observation_bounds(max_duration_of_stay=max_duration)
        for condition in conditions:
            key = (
                scenario_id,
                str(condition["attack_key"]),
                str(condition["ratio_key"]),
                str(condition["stage_key"]),
            )
            if key in completed_keys:
                continue
            attack_base_key = str(condition["attack_base_key"])
            attack_seed = int(args.seed + attack_seed_offsets[attack_base_key] + episode_index)
            table_eval.set_all_seeds(attack_seed)
            attack_ratio = 0.0 if condition["attack_ratio"] is None else float(condition["attack_ratio"])
            attack_enabled = attack_base_key != "clean" and attack_ratio > 0.0
            attacker = None
            configured: dict[str, Any] = {}
            if attack_enabled:
                attack_spec = attack_lookup[attack_base_key]
                attacker = table_eval.build_short_attacker(
                    algorithm=str(attack_spec["algorithm"]),
                    actor=actor,
                    critic=critic,
                    device=device,
                    low=low,
                    high=high,
                    seed=attack_seed,
                    epsilon=float(args.epsilon),
                )
                configured = {
                    "configured_epsilon": float(args.epsilon),
                    "configured_alpha": float(getattr(attacker, "alpha", np.nan)),
                    "configured_iters": int(getattr(attacker, "iters", 0)),
                }
            stage_spec = stage_lookup[str(condition["stage_key"])]
            kwargs = table_eval.stage_kwargs(
                stage_spec,
                dae=dae,
                detector_model=detector_model,
                detector_threshold=detector_threshold,
                shield_config=shield_config,
                ug_bcr_config=ug_bcr_config,
            )
            rollout_start = time.perf_counter()
            summary = table_eval.rollout_episode_with_ug_bcr(
                arrivals,
                actor,
                signal_path,
                device,
                table_eval.TRAIN_PROFILE,
                attack_enabled=attack_enabled,
                attack_scenario=str(condition["attack_scenario"]),
                attacker=attacker,
                epsilon=float(args.epsilon),
                state_scope=str(condition["attack_state_scope"]),
                price_threshold=float(price_threshold),
                attack_ratio=attack_ratio,
                attack_scope="obs",
                label=f"{condition['attack_key']}__rho_{condition['ratio_key']}__{condition['stage_key']}",
                repair_mode=table_eval.REPAIR_MODE,
                **kwargs,
            )
            runtime_seconds = float(time.perf_counter() - rollout_start)
            scalar = table_eval.to_scalar_summary(summary)
            row = {
                "scenario_id": scenario_id,
                "episode_index": int(episode_index),
                "seed": int(args.seed),
                "attack_seed": int(attack_seed),
                "attack_base_key": attack_base_key,
                "attack_key": str(condition["attack_key"]),
                "attack_display_base": "Clean" if attack_base_key == "clean" else str(attack_lookup[attack_base_key]["display"]),
                "attack_display_name": str(condition["attack_display_name"]),
                "algorithm": None if condition["algorithm"] is None else str(condition["algorithm"]),
                "attack_scenario": str(condition["attack_scenario"]),
                "attack_state_scope": str(condition["attack_state_scope"]),
                "attack_ratio": None if condition["attack_ratio"] is None else attack_ratio,
                "ratio_key": str(condition["ratio_key"]),
                "epsilon": float(args.epsilon),
                "stage_key": str(condition["stage_key"]),
                "stage_display_name": str(condition["stage_display_name"]),
                "seen_in_dtsr_training": bool(condition["seen_in_dtsr_training"]),
                "runtime_seconds": runtime_seconds,
                **configured,
                **scalar,
            }
            if int(row.get("done_cnt", -1)) != 344:
                raise RuntimeError(f"Incomplete rollout for {key}: done_cnt={row.get('done_cnt')}")
            if attack_base_key == "clean" and int(row.get("attack_obs_count", -1)) != 0:
                raise RuntimeError(f"Clean rollout unexpectedly attacked observations for {key}.")
            table_eval.append_jsonl(intermediate_path, row)
            completed_keys.add(key)
            new_rollouts += 1
            print(
                f"[{len(completed_keys):04d}/{expected_total}] scene={episode_index:02d} "
                f"{condition['attack_display_name']} | rho={attack_ratio:.2f} | {condition['stage_display_name']} "
                f"reward={float(row['ep_reward']):.3f} "
                f"route={float(row.get('route_rate', 0.0)) * 100.0:.1f}% "
                f"run/exit={int(row.get('run_vio', 0))}/{int(row.get('exit_vio', 0))} "
                f"time={runtime_seconds:.2f}s",
                flush=True,
            )
            if args.max_rollouts > 0 and new_rollouts >= int(args.max_rollouts):
                print(f"Debug stop after {new_rollouts} new rollouts.", flush=True)
                return
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    completed_rows = [row for row in table_eval.load_jsonl(intermediate_path) if condition_key(row) in expected_keys]
    if len(completed_rows) != expected_total:
        raise RuntimeError(f"Expected {expected_total} completed rollouts, found {len(completed_rows)}.")
    long_df = pd.DataFrame(completed_rows).sort_values(
        ["episode_index", "attack_base_key", "attack_ratio", "stage_key"], kind="mergesort"
    )
    raw_frame, paper_frame = summarize_outputs(long_df, args.output_dir, args.latest_dir)
    det_quality_summary = pd.DataFrame()
    det_quality_paper = pd.DataFrame()
    if not args.skip_routing_quality:
        _, det_quality_summary, det_quality_paper = create_det_routing_quality(
            manifest=manifest,
            attack_keys=attack_keys,
            ratios=ratios,
            actor=actor,
            critic=critic,
            device=device,
            dae=dae,
            detector_model=detector_model,
            detector_threshold=detector_threshold,
            price_threshold=float(price_threshold),
            seed=int(args.seed),
            epsilon=float(args.epsilon),
            attack_seed_offsets=attack_seed_offsets,
            output_dir=args.output_dir,
            latest_dir=args.latest_dir,
        )
    elapsed = float(time.perf_counter() - started)
    table_eval.write_json(
        args.output_dir / "final_status.json",
        {
            "completed_rollouts": int(len(long_df)),
            "expected_rollouts": int(expected_total),
            "elapsed_seconds_this_run": elapsed,
            "paper_table": str(args.output_dir / "tables" / "exp4_attack_ratio_paper.csv"),
            "det_routing_quality_table": None
            if det_quality_paper.empty
            else str(args.output_dir / "tables" / "exp4_det_routing_quality_paper.csv"),
            "latest_table": str(args.latest_dir / "table4_attack_ratio_latest.csv"),
            "latest_det_routing_quality_table": None
            if det_quality_paper.empty
            else str(args.latest_dir / "table4_det_routing_quality_latest.csv"),
        },
    )
    print(f"Completed {len(long_df)}/{expected_total} rollouts in {elapsed / 60.0:.1f} min.", flush=True)
    print(f"Saved: {args.output_dir / 'tables' / 'exp4_attack_ratio_paper.md'}", flush=True)
    if not det_quality_paper.empty:
        print(f"Saved: {args.output_dir / 'tables' / 'exp4_det_routing_quality_paper.md'}", flush=True)


if __name__ == "__main__":
    main()
