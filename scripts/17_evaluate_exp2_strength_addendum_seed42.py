from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
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

from evc.ug_bcr_v3 import load_ug_bcr_v3_config, rollout_episode_with_ug_bcr_v3

table_eval = importlib.import_module("_strength_eval_common")


DEFAULT_SHORT_EPSILONS = (0.100, 0.150, 0.200)
DEFAULT_LONG_EPSILONS = (0.055, 0.040, 0.025)
SHORT_NOMINAL_EPSILON = 0.100
SHORT_ATTACK_ALGORITHMS = {"opposite_pgd", "q_function", "opposite_fgsm", "electhacker"}
LONG_ATTACK_ALGORITHMS = {"local_small_drift_q", "local_deadline_drift_pgd", "full_pipeline_adaptive_deadline"}
STRENGTH_ATTACK_KEYS = (
    "opposite_pgd",
    "q_function",
    "opposite_fgsm",
    "electhacker_c",
    "electhacker_f",
    "electhacker_o",
    "local_small_drift_q",
    "local_deadline_drift_pgd",
    "full_pipeline_adaptive_deadline",
)
LONG_NOMINAL_OUTER_EPSILON = 0.055
LONG_NOMINAL_INNER_EPSILON = 0.028
LONG_NOMINAL_ALPHA = 0.008
LONG_INNER_ITERS = 5
SMALL_DRIFT_NOMINAL_OUTER_EPSILON = 0.055
SMALL_DRIFT_NOMINAL_INNER_EPSILON = 0.030
SMALL_DRIFT_NOMINAL_ALPHA = 0.010
SMALL_DRIFT_INNER_ITERS = 5
ADAPTIVE_NOMINAL_OUTER_EPSILON = 0.075
ADAPTIVE_NOMINAL_INNER_EPSILON = 0.036
ADAPTIVE_NOMINAL_ALPHA = 0.010
ADAPTIVE_INNER_ITERS = 7

STAGE_KEYS = ("attack", "denoise", "denoise_det", "shield", "ug_bcr")


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in str(raw).split(",") if token.strip())
    if not values:
        raise ValueError("At least one epsilon value is required.")
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Every epsilon must be finite and positive.")
    if len(set(values)) != len(values):
        raise ValueError("Epsilon lists must not contain duplicates.")
    return values


def sample_std(values: pd.Series | np.ndarray | list[float]) -> float:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    return float(np.std(data, ddof=1)) if data.size > 1 else 0.0


def format_mean_std(mean: float, std: float, digits: int = 1) -> str:
    if not np.isfinite(mean):
        return "-"
    return f"{float(mean):.{digits}f}±{float(std):.{digits}f}"


def epsilon_key(value: float | None) -> str:
    return "clean" if value is None else f"{float(value):.6f}"


def short_label(epsilon: float) -> str:
    return f"PGD* (ε={float(epsilon):.2f})"


def long_label(epsilon: float) -> str:
    return f"deadline_pgd (ε_long={float(epsilon):.3f})"


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


def condition_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["scenario_id"]),
        str(row["attack_key"]),
        str(row["stage_key"]),
    )


def long_attack_parameters(outer_epsilon: float) -> dict[str, float | int]:
    ratio = float(outer_epsilon) / LONG_NOMINAL_OUTER_EPSILON
    return {
        "epsilon": float(outer_epsilon),
        "base_epsilon": LONG_NOMINAL_INNER_EPSILON * ratio,
        "base_alpha": LONG_NOMINAL_ALPHA * ratio,
        "base_iters": LONG_INNER_ITERS,
    }


def strength_kind(attack_spec: dict[str, Any]) -> str:
    algorithm = attack_spec["algorithm"]
    if algorithm is None:
        return "clean"
    if algorithm in SHORT_ATTACK_ALGORITHMS:
        return "short"
    if algorithm in LONG_ATTACK_ALGORITHMS:
        return "long"
    raise ValueError(f"Unsupported strength attack algorithm: {algorithm}")


def attack_display_with_epsilon(attack_spec: dict[str, Any], epsilon: float, kind: str) -> str:
    if kind == "short":
        return f"{attack_spec['display']} (ε={float(epsilon):.2f})"
    return f"{attack_spec['display']} (ε_long={float(epsilon):.3f})"


def build_conditions(
    short_epsilons: tuple[float, ...],
    long_epsilons: tuple[float, ...],
    attack_keys: tuple[str, ...] = STRENGTH_ATTACK_KEYS,
) -> list[dict[str, Any]]:
    stage_lookup = {spec["key"]: spec for spec in table_eval.STAGE_SPECS}
    attack_lookup = {spec["key"]: spec for spec in table_eval.ATTACK_SPECS}
    selected_attack_specs = [attack_lookup[key] for key in attack_keys]
    conditions: list[dict[str, Any]] = []
    for stage_key in STAGE_KEYS:
        conditions.append(
            {
                "attack_family": "clean",
                "attack_base_key": "clean",
                "attack_key": "clean",
                "algorithm": None,
                "attack_scenario": "O",
                "attack_display_name": "Clean",
                "epsilon": None,
                "epsilon_key": "clean",
                "stage_key": stage_key,
                "stage_display_name": stage_lookup[stage_key]["display"],
                "attack_state_scope": "all",
                "seen_in_dtsr_training": False,
            }
        )
    for attack_spec in selected_attack_specs:
        kind = strength_kind(attack_spec)
        epsilons = short_epsilons if kind == "short" else long_epsilons
        epsilon_prefix = "eps" if kind == "short" else "epslong"
        for epsilon in epsilons:
            for stage_key in STAGE_KEYS:
                conditions.append(
                    {
                        "attack_family": kind,
                        "attack_base_key": str(attack_spec["key"]),
                        "attack_key": f"{attack_spec['key']}_{epsilon_prefix}_{epsilon_key(epsilon)}",
                        "algorithm": attack_spec["algorithm"],
                        "attack_scenario": attack_spec["scenario"],
                        "attack_display_name": attack_display_with_epsilon(attack_spec, epsilon, kind),
                        "epsilon": float(epsilon),
                        "epsilon_key": epsilon_key(epsilon),
                        "stage_key": stage_key,
                        "stage_display_name": stage_lookup[stage_key]["display"],
                        "attack_state_scope": attack_spec["scope"],
                        "seen_in_dtsr_training": bool(attack_spec["seen"])
                        and math.isclose(float(epsilon), SHORT_NOMINAL_EPSILON),
                    }
                )
    return conditions


def build_attacker(
    condition: dict[str, Any],
    *,
    actor,
    critic,
    device: torch.device,
    low: np.ndarray,
    high: np.ndarray,
    attack_seed: int,
    dae,
    detector_model,
    detector_threshold: float,
    shield_config,
    ug_bcr_config,
    signal_path: Path,
):
    algorithm = condition["algorithm"]
    epsilon = condition["epsilon"]
    if algorithm is None:
        return None, {}
    if algorithm in SHORT_ATTACK_ALGORITHMS:
        epsilon = float(epsilon)
        if algorithm == "opposite_fgsm":
            alpha = epsilon
            iters = 1
        elif algorithm == "electhacker":
            alpha = epsilon / 20.0
            iters = 100
        else:
            alpha = epsilon / 10.0
            iters = 10
        return (
            table_eval.PGDStateAttacker(
                actor,
                device=device,
                algorithm=str(algorithm),
                epsilon=epsilon,
                alpha=alpha,
                iters=iters,
                seed=int(attack_seed),
                obs_low=low,
                obs_high=high,
                critic=critic if algorithm == "q_function" else None,
                attack_state_scope="all",
            ),
            {
                "configured_outer_epsilon": epsilon,
                "configured_inner_epsilon": epsilon,
                "configured_alpha": alpha,
                "configured_iters": iters,
            },
        )
    if algorithm == "local_deadline_drift_pgd":
        parameters = long_attack_parameters(float(epsilon))
        return (
            table_eval.build_formal_experimental_long_horizon_attacker(
                str(algorithm),
                actor=actor,
                device=device,
                obs_low=low,
                obs_high=high,
                critic=critic,
                seed=int(attack_seed),
                attack_state_scope="local",
                attack_overrides=parameters,
            ),
            {
                "configured_outer_epsilon": float(parameters["epsilon"]),
                "configured_inner_epsilon": float(parameters["base_epsilon"]),
                "configured_alpha": float(parameters["base_alpha"]),
                "configured_iters": int(parameters["base_iters"]),
            },
        )
    if algorithm == "local_small_drift_q":
        epsilon = float(epsilon)
        ratio = epsilon / SMALL_DRIFT_NOMINAL_OUTER_EPSILON
        attacker = table_eval.build_formal_experimental_long_horizon_attacker(
            str(algorithm),
            actor=actor,
            device=device,
            obs_low=low,
            obs_high=high,
            critic=critic,
            seed=int(attack_seed),
            attack_state_scope="local",
            strength_scale=ratio,
        )
        attacker.base_attacker.iters = SMALL_DRIFT_INNER_ITERS
        return (
            attacker,
            {
                "configured_outer_epsilon": float(attacker.epsilon),
                "configured_inner_epsilon": float(attacker.base_attacker.epsilon),
                "configured_alpha": float(attacker.base_attacker.alpha),
                "configured_iters": int(attacker.base_attacker.iters),
            },
        )
    if algorithm == "full_pipeline_adaptive_deadline":
        epsilon = float(epsilon)
        ratio = epsilon / ADAPTIVE_NOMINAL_OUTER_EPSILON
        attacker = table_eval.build_long_horizon_attacker(
            str(algorithm),
            actor=actor,
            device=device,
            obs_low=low,
            obs_high=high,
            critic=critic,
            seed=int(attack_seed),
            attack_state_scope="local",
        )
        attacker.epsilon = epsilon
        attacker.base_attacker.epsilon = ADAPTIVE_NOMINAL_INNER_EPSILON * ratio
        attacker.base_attacker.alpha = ADAPTIVE_NOMINAL_ALPHA * ratio
        attacker.base_attacker.iters = ADAPTIVE_INNER_ITERS
        if hasattr(attacker, "smooth_step_clip"):
            attacker.smooth_step_clip = float(getattr(attacker, "smooth_step_clip")) * ratio
        if not hasattr(attacker, "configure_target_defense"):
            raise RuntimeError("Adaptive attacker does not expose configure_target_defense().")
        attacker.configure_target_defense(
            defender=dae,
            detector_model=detector_model,
            detector_threshold=float(detector_threshold),
            shield_config=shield_config,
            ug_bcr_config=ug_bcr_config,
            reward_profile=table_eval.TRAIN_PROFILE,
            signals_path=signal_path,
            device=device,
            actor=actor,
            repair_mode=table_eval.REPAIR_MODE,
        )
        if not bool(getattr(attacker, "_target_ready", False)):
            raise RuntimeError("FullPipelineAdaptiveDeadlineAttacker target defense is not ready.")
        return (
            attacker,
            {
                "configured_outer_epsilon": epsilon,
                "configured_inner_epsilon": float(attacker.base_attacker.epsilon),
                "configured_alpha": float(attacker.base_attacker.alpha),
                "configured_iters": int(attacker.base_attacker.iters),
            },
        )
    raise ValueError(f"Unsupported attack algorithm: {algorithm}")


def summarize_table(
    long_df: pd.DataFrame,
    output_dir: Path,
    latest_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_raw = long_df[
        (long_df["attack_family"] == "clean") & (long_df["stage_key"] == "attack")
    ][["scenario_id", "ep_reward"]].rename(columns={"ep_reward": "clean_raw_reward"})
    if clean_raw["scenario_id"].nunique() != long_df["scenario_id"].nunique():
        raise RuntimeError("Missing clean raw baseline for at least one scenario.")

    rows_raw: list[dict[str, Any]] = []
    rows_paper: list[dict[str, Any]] = []
    rows_table4: list[dict[str, Any]] = []
    stage_lookup = {spec["key"]: spec for spec in table_eval.STAGE_SPECS}
    attack_order = (
        long_df[long_df["attack_family"] != "clean"][
            ["attack_family", "attack_base_key", "epsilon_key", "attack_key", "attack_display_name", "epsilon"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    for _, attack_meta in attack_order.iterrows():
        attack_data = long_df[long_df["attack_key"] == attack_meta["attack_key"]]
        raw_attack = attack_data[attack_data["stage_key"] == "attack"][
            ["scenario_id", "ep_reward"]
        ].rename(columns={"ep_reward": "attack_reward"})
        full = attack_data[attack_data["stage_key"] == "ug_bcr"][
            ["scenario_id", "ep_reward", "route_rate"]
        ].rename(columns={"ep_reward": "defended_reward"})
        paired = clean_raw.merge(raw_attack, on="scenario_id", how="inner").merge(
            full, on="scenario_id", how="inner"
        )
        if paired.empty:
            raise RuntimeError(f"Missing paired recovery rows for {attack_meta['attack_display_name']}.")
        denominator = paired["clean_raw_reward"].to_numpy(dtype=float) - paired["attack_reward"].to_numpy(dtype=float)
        numerator = paired["defended_reward"].to_numpy(dtype=float) - paired["attack_reward"].to_numpy(dtype=float)
        valid = np.abs(denominator) > 1e-8
        recovery = np.full(denominator.shape, np.nan, dtype=np.float64)
        recovery[valid] = numerator[valid] / denominator[valid] * 100.0
        recovery_mean = float(np.nanmean(recovery))
        recovery_std = sample_std(recovery)
        route_values = paired["route_rate"].to_numpy(dtype=float) * 100.0
        route_mean = float(np.mean(route_values))
        route_std = sample_std(route_values)
        clean_drop = paired["clean_raw_reward"].to_numpy(dtype=float) - paired["defended_reward"].to_numpy(dtype=float)

        raw_row: dict[str, Any] = {
            "attack_family": attack_meta["attack_family"],
            "attack_base_key": attack_meta["attack_base_key"],
            "attack_key": attack_meta["attack_key"],
            "scenario": attack_meta["attack_display_name"],
            "epsilon": float(attack_meta["epsilon"]),
            "scenario_count": int(len(paired)),
            "recovery_mean_pct": recovery_mean,
            "recovery_std_pct": recovery_std,
            "route_rate_mean_pct": route_mean,
            "route_rate_std_pct": route_std,
            "clean_drop_mean": float(np.mean(clean_drop)),
            "clean_drop_std": sample_std(clean_drop),
        }
        paper_row: dict[str, Any] = {"场景": attack_meta["attack_display_name"]}
        for stage_key in STAGE_KEYS:
            subset = attack_data[attack_data["stage_key"] == stage_key]
            rewards = subset["ep_reward"].astype(float).to_numpy()
            reward_mean = float(np.mean(rewards))
            reward_std = sample_std(rewards)
            raw_row[f"{stage_key}_reward_mean"] = reward_mean
            raw_row[f"{stage_key}_reward_std"] = reward_std
            raw_row[f"{stage_key}_scenario_count"] = int(len(rewards))
            paper_row[str(stage_lookup[stage_key]["display"])] = format_mean_std(reward_mean, reward_std, 1)
        paper_row["恢复率/%"] = format_mean_std(recovery_mean, recovery_std, 1)
        paper_row["路由率/%"] = format_mean_std(route_mean, route_std, 1)
        rows_raw.append(raw_row)
        rows_paper.append(paper_row)
        rows_table4.append(
            {
                "attack_base_key": str(attack_meta["attack_base_key"]),
                "epsilon": float(attack_meta["epsilon"]),
                "Attack": paper_row["Attack"],
                "Denoise": paper_row["Denoise"],
                "Denoise + DET": paper_row["Denoise+DET"],
                "+Temporal Shield": paper_row["+Shield"],
                "+UG-BCR": paper_row["+UG-BCR"],
                "完整防御恢复率/%": paper_row["恢复率/%"],
            }
        )

    raw_frame = pd.DataFrame(rows_raw)
    paper_frame = pd.DataFrame(rows_paper)[
        ["场景", "Attack", "Denoise", "Denoise+DET", "+Shield", "+UG-BCR", "恢复率/%", "路由率/%"]
    ]

    clean_data = long_df[long_df["attack_family"] == "clean"]
    clean_row: dict[str, Any] = {"攻击方法": "Clean", "扰动强度 ε": "—"}
    table4_stage_labels = {
        "attack": "Attack",
        "denoise": "Denoise",
        "denoise_det": "Denoise + DET",
        "shield": "+Temporal Shield",
        "ug_bcr": "+UG-BCR",
    }
    for stage_key, label in table4_stage_labels.items():
        values = clean_data[clean_data["stage_key"] == stage_key]["ep_reward"].astype(float).to_numpy()
        clean_row[label] = format_mean_std(float(np.mean(values)), sample_std(values), 1)
    clean_row["完整防御恢复率/%"] = "—"

    table4_attack_labels = {
        "opposite_pgd": "PGD",
        "q_function": "Q-function",
        "opposite_fgsm": "FGSM",
        "electhacker_c": "EH-C",
        "electhacker_f": "EH-F",
        "electhacker_o": "EH-O",
        "local_small_drift_q": "Small-drift Q",
        "local_deadline_drift_pgd": "Deadline-PGD",
    }
    table4_attack_order = {key: index for index, key in enumerate(table4_attack_labels)}
    ordered_table4_rows = sorted(
        rows_table4,
        key=lambda row: (table4_attack_order[str(row["attack_base_key"])], float(row["epsilon"])),
    )
    formal_rows = [clean_row]
    for row in ordered_table4_rows:
        attack_key = str(row.pop("attack_base_key"))
        epsilon = float(row.pop("epsilon"))
        formal_rows.append(
            {
                "攻击方法": table4_attack_labels[attack_key],
                "扰动强度 ε": f"{epsilon:.3f}" if attack_key.startswith("local_") else f"{epsilon:.2f}",
                **row,
            }
        )
    table4_frame = pd.DataFrame(formal_rows)[
        [
            "攻击方法",
            "扰动强度 ε",
            "Attack",
            "Denoise",
            "Denoise + DET",
            "+Temporal Shield",
            "+UG-BCR",
            "完整防御恢复率/%",
        ]
    ]

    tables_dir = output_dir / "tables"
    table_eval.atomic_csv(long_df, tables_dir / "exp2_strength_addendum_rollouts.csv")
    table_eval.atomic_csv(raw_frame, tables_dir / "exp2_strength_addendum_raw.csv")
    table_eval.atomic_csv(paper_frame, tables_dir / "exp2_strength_addendum_paper.csv")
    table_eval.atomic_csv(table4_frame, tables_dir / "table4_multistage_strength_v3.csv")
    note = (
        "20 fixed test scenes, seed=42 artifacts. Short attack is Opposite-PGD; "
        "long attack is local deadline PGD. Values are mean±sample std across scenes."
    )
    md = "# Table 2 Strength Addendum\n\n" + markdown_table(paper_frame) + "\n" + note + "\n"
    atomic_text(tables_dir / "exp2_strength_addendum_paper.md", md)
    table4_md = "# Table 4 Multi-stage Defense and Strength Robustness (UG-BCR v3)\n\n" + markdown_table(table4_frame) + "\n" + note + "\n"
    atomic_text(tables_dir / "table4_multistage_strength_v3.md", table4_md)

    if latest_dir is not None:
        latest_dir.mkdir(parents=True, exist_ok=True)
        table_eval.atomic_csv(paper_frame, latest_dir / "table2_strength_addendum_latest.csv")
        atomic_text(latest_dir / "table2_strength_addendum_latest.md", md)
        table_eval.atomic_csv(table4_frame, latest_dir / "table4_multistage_strength_v3_latest.csv")
        atomic_text(latest_dir / "table4_multistage_strength_v3_latest.md", table4_md)
    return raw_frame, paper_frame, table4_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate compact attack-strength rows for the Table 2 ablation addendum."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor-path", type=Path, default=table_eval.EP100_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=table_eval.EP100_BUNDLE_PATH)
    parser.add_argument("--dtsr-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dtsr_seed42_multiday")
    parser.add_argument("--dae-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday")
    parser.add_argument(
        "--detector-artifact-dir",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / "det_seed42_fullstate",
    )
    parser.add_argument(
        "--shield-artifact-dir",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / "shield_seed42_fullstate",
    )
    parser.add_argument(
        "--ug-bcr-config-path",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / "ug_bcr_seed42_fullstate" / "ug_bcr_config.json",
    )
    parser.add_argument(
        "--ug-bcr-v3-config-path",
        type=Path,
        default=None,
        help="Optional UG-BCR-v3 continuous-gate config used for the final ug_bcr stage.",
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
    parser.add_argument("--short-epsilons", default=",".join(f"{value:.3f}" for value in DEFAULT_SHORT_EPSILONS))
    parser.add_argument("--long-epsilons", default=",".join(f"{value:.3f}" for value in DEFAULT_LONG_EPSILONS))
    parser.add_argument(
        "--strength-attack-keys",
        default=",".join(STRENGTH_ATTACK_KEYS),
        help="Comma-separated non-clean attack keys to include in the strength addendum.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "exp2_strength_allattacks_20scenes_seed42",
    )
    parser.add_argument(
        "--latest-table-dir",
        type=Path,
        default=PACKAGE_ROOT / "results" / "paper_tables_latest",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--refresh-stages",
        default="",
        help="Comma-separated stages to recompute while retaining unaffected cached rows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rollouts", type=int, default=0, help="Debug only; 0 runs the complete table.")
    args = parser.parse_args()

    if int(args.seed) != 42:
        raise ValueError("This addendum is fixed to the seed=42 trained artifacts.")
    if int(args.scenes) <= 0:
        raise ValueError("--scenes must be positive.")

    short_epsilons = parse_float_list(args.short_epsilons)
    long_epsilons = parse_float_list(args.long_epsilons)
    selected_strength_attack_keys = tuple(table_eval.parse_key_list(args.strength_attack_keys))
    unknown_strength_attack_keys = sorted(set(selected_strength_attack_keys) - set(STRENGTH_ATTACK_KEYS))
    if unknown_strength_attack_keys:
        raise ValueError(f"Unknown strength attack keys: {unknown_strength_attack_keys}")
    if "clean" in selected_strength_attack_keys:
        raise ValueError("Clean is included automatically; do not pass it in --strength-attack-keys.")
    if not selected_strength_attack_keys:
        raise ValueError("At least one non-clean strength attack key is required.")
    conditions = build_conditions(short_epsilons, long_epsilons, selected_strength_attack_keys)
    stage_lookup = {spec["key"]: spec for spec in table_eval.STAGE_SPECS}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_path = args.output_dir / "intermediate" / "strength_addendum_rollouts.jsonl"
    if args.overwrite:
        for path in [
            intermediate_path,
            args.output_dir / "tables" / "exp2_strength_addendum_rollouts.csv",
            args.output_dir / "tables" / "exp2_strength_addendum_raw.csv",
            args.output_dir / "tables" / "exp2_strength_addendum_paper.csv",
            args.output_dir / "tables" / "exp2_strength_addendum_paper.md",
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
    checkpoint_episode = int((bundle_payload.get("metadata") or {}).get("checkpoint_episode", -1))
    if checkpoint_episode != 100:
        raise RuntimeError(f"Expected ep100 DDPG, got checkpoint_episode={checkpoint_episode}.")
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
    detector_path = table_eval.existing_artifact_path(
        args.detector_artifact_dir, args.dtsr_dir, "dtsr_detector.pt"
    )
    shield_path = table_eval.existing_artifact_path(
        args.shield_artifact_dir, args.dtsr_dir, "dtsr_temporal_shield.pt"
    )
    dae = table_eval.load_dae(dae_path, device).eval()
    detector_artifact = table_eval.load_detector(detector_path, device)
    detector_model = detector_artifact.model
    detector_threshold = float(detector_artifact.threshold)
    shield_config = table_eval.load_temporal_shield_bundle(shield_path).config
    if not args.ug_bcr_config_path.exists():
        raise FileNotFoundError(f"Missing UG-BCR config: {args.ug_bcr_config_path}")
    ug_bcr_config = table_eval.load_ug_bcr_config(args.ug_bcr_config_path)
    ug_bcr_v3_config = None
    if args.ug_bcr_v3_config_path is not None:
        if not args.ug_bcr_v3_config_path.exists():
            raise FileNotFoundError(f"Missing UG-BCR-v3 config: {args.ug_bcr_v3_config_path}")
        ug_bcr_v3_config = load_ug_bcr_v3_config(args.ug_bcr_v3_config_path)
        ug_bcr_config = ug_bcr_v3_config.base_v2
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
            str(condition["stage_key"]),
        )
        for _, row in manifest.iterrows()
        for condition in conditions
    }
    existing_rows = table_eval.load_jsonl(intermediate_path) if args.resume else []
    existing_rows = [row for row in existing_rows if condition_key(row) in expected_keys]
    refresh_stages = set(table_eval.parse_key_list(args.refresh_stages))
    unknown_refresh_stages = sorted(refresh_stages - set(STAGE_KEYS))
    if unknown_refresh_stages:
        raise ValueError(f"Unknown --refresh-stages values: {unknown_refresh_stages}")
    if refresh_stages:
        existing_rows = [row for row in existing_rows if str(row["stage_key"]) not in refresh_stages]
        atomic_text(
            intermediate_path,
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n" for row in existing_rows),
        )
    completed_keys = {condition_key(row) for row in existing_rows}
    if len(completed_keys) != len(existing_rows):
        raise RuntimeError("Intermediate JSONL contains duplicate rollout keys.")

    run_config = {
        "seed": int(args.seed),
        "split": str(args.split),
        "scenario_count": int(len(manifest)),
        "strength_attack_keys": list(selected_strength_attack_keys),
        "short_epsilons": list(short_epsilons),
        "short_alpha_rule": "PGD/Q alpha = epsilon / 10, FGSM alpha = epsilon, ElectHacker alpha = epsilon / 20",
        "long_outer_epsilons": list(long_epsilons),
        "long_nominal_outer_epsilon": LONG_NOMINAL_OUTER_EPSILON,
        "long_nominal_inner_epsilon": LONG_NOMINAL_INNER_EPSILON,
        "long_nominal_alpha": LONG_NOMINAL_ALPHA,
        "long_scaling_rule": "outer epsilon and inner base attack epsilon/alpha scale together; temporal dynamics stay fixed",
        "attack_seed_rule": "seed + attack_base_key_offset + episode_index; shared across strengths and stages",
        "stage_keys": list(STAGE_KEYS),
        "refreshed_stage_keys": sorted(refresh_stages),
        "expected_rollouts": int(len(expected_keys)),
        "statistics": f"mean +/- sample std across {len(manifest)} paired scenarios",
        "actor_path": str(args.actor_path),
        "dae_artifact": str(dae_path),
        "detector_artifact": str(detector_path),
        "shield_artifact": str(shield_path),
        "ug_bcr_config": str(args.ug_bcr_config_path),
        "ug_bcr_v3_config": None if args.ug_bcr_v3_config_path is None else str(args.ug_bcr_v3_config_path),
        "ug_bcr_v3_config_sha256": (
            None if args.ug_bcr_v3_config_path is None else sha256_file(args.ug_bcr_v3_config_path)
        ),
        "ug_bcr_version": 3 if ug_bcr_v3_config is not None else 2,
    }
    table_eval.write_json(args.output_dir / "run_config.json", run_config)

    expected_total = len(expected_keys)
    attack_seed_offsets = {
        "clean": 0,
        **{attack_key: (index + 1) * 100_000 for index, attack_key in enumerate(selected_strength_attack_keys)},
    }
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
                str(condition["stage_key"]),
            )
            if key in completed_keys:
                continue
            attack_seed = int(args.seed + attack_seed_offsets[str(condition["attack_base_key"])] + episode_index)
            table_eval.set_all_seeds(attack_seed)
            attacker, configured = build_attacker(
                condition,
                actor=actor,
                critic=critic,
                device=device,
                low=low,
                high=high,
                attack_seed=attack_seed,
                dae=dae,
                detector_model=detector_model,
                detector_threshold=detector_threshold,
                shield_config=shield_config,
                ug_bcr_config=ug_bcr_config,
                signal_path=signal_path,
            )
            stage_spec = stage_lookup[str(condition["stage_key"])]
            kwargs = table_eval.stage_kwargs(
                stage_spec,
                dae=dae,
                detector_model=detector_model,
                detector_threshold=detector_threshold,
                shield_config=shield_config,
                ug_bcr_config=ug_bcr_config,
            )
            rollout_function = table_eval.rollout_episode_with_ug_bcr
            if ug_bcr_v3_config is not None and str(condition["stage_key"]) == "ug_bcr":
                rollout_function = rollout_episode_with_ug_bcr_v3
                kwargs.pop("ug_bcr_config", None)
                kwargs["ug_bcr_v3_config"] = ug_bcr_v3_config
            rollout_start = time.perf_counter()
            epsilon = 0.0 if condition["epsilon"] is None else float(condition["epsilon"])
            summary = rollout_function(
                arrivals,
                actor,
                signal_path,
                device,
                table_eval.TRAIN_PROFILE,
                attack_enabled=condition["attack_family"] != "clean",
                attack_scenario=str(condition["attack_scenario"]),
                attacker=attacker,
                epsilon=epsilon,
                state_scope=str(condition["attack_state_scope"]),
                price_threshold=float(price_threshold),
                attack_ratio=1.0,
                attack_scope="obs",
                label=f"{condition['attack_key']}__{condition['stage_key']}",
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
                "attack_family": str(condition["attack_family"]),
                "attack_base_key": str(condition["attack_base_key"]),
                "attack_key": str(condition["attack_key"]),
                "algorithm": None if condition["algorithm"] is None else str(condition["algorithm"]),
                "attack_scenario": str(condition["attack_scenario"]),
                "attack_display_name": str(condition["attack_display_name"]),
                "epsilon": None if condition["epsilon"] is None else float(condition["epsilon"]),
                "epsilon_key": str(condition["epsilon_key"]),
                "attack_state_scope": str(condition["attack_state_scope"]),
                "seen_in_dtsr_training": bool(condition["seen_in_dtsr_training"]),
                "stage_key": str(condition["stage_key"]),
                "stage_display_name": str(condition["stage_display_name"]),
                "runtime_seconds": runtime_seconds,
                **configured,
                **scalar,
            }
            if int(row.get("done_cnt", -1)) != 344:
                raise RuntimeError(f"Incomplete rollout for {key}: done_cnt={row.get('done_cnt')}")
            table_eval.append_jsonl(intermediate_path, row)
            completed_keys.add(key)
            new_rollouts += 1
            print(
                f"[{len(completed_keys):04d}/{expected_total}] scene={episode_index:02d} "
                f"{condition['attack_display_name']} | {condition['stage_display_name']} "
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
        ["episode_index", "attack_base_key", "epsilon_key", "stage_key"], kind="mergesort"
    )
    raw_frame, paper_frame, table4_frame = summarize_table(long_df, args.output_dir, args.latest_table_dir)
    elapsed = float(time.perf_counter() - started)
    table_eval.write_json(
        args.output_dir / "final_status.json",
        {
            "completed_rollouts": int(len(long_df)),
            "expected_rollouts": int(expected_total),
            "elapsed_seconds_this_run": elapsed,
            "paper_table": str(args.output_dir / "tables" / "exp2_strength_addendum_paper.csv"),
            "formal_table4": str(args.output_dir / "tables" / "table4_multistage_strength_v3.csv"),
            "latest_table": str(args.latest_table_dir / "table2_strength_addendum_latest.csv"),
            "raw_rows": int(len(raw_frame)),
            "paper_rows": int(len(paper_frame)),
            "formal_table4_rows": int(len(table4_frame)),
        },
    )
    print(f"Completed {len(long_df)}/{expected_total} rollouts in {elapsed / 60.0:.1f} min.", flush=True)
    print(f"Saved: {args.output_dir / 'tables' / 'exp2_strength_addendum_paper.md'}", flush=True)
    print(f"Saved: {args.latest_table_dir / 'table2_strength_addendum_latest.md'}", flush=True)


if __name__ == "__main__":
    main()
