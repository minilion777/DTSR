from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from _common import (
    PACKAGE_ROOT,
    actor_matches_bundle,
    load_manifest,
    load_scenario,
    resolve_device,
    write_json,
)
from dtsr_multiday_common import (
    ABLATION_ADDITION_ORDER,
    EP100_ACTOR_PATH,
    EP100_BUNDLE_PATH,
    REPAIR_MODE,
    RUNTIME_PIPELINE_ORDER,
    load_ug_bcr_config,
    safe_recovery,
    set_all_seeds,
    to_scalar_summary,
)

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.defense import load_dae, load_detector
from evc.formal_experimental_long_horizon import (
    build_formal_experimental_long_horizon_attacker,
    uses_formal_experimental_long_horizon,
)
from evc.long_horizon_attacks import build_long_horizon_attacker
from evc.merged_attacks import PGDStateAttacker
from evc.merged_core import (
    ChargingEnv,
    Critic,
    TRAIN_PROFILE,
    load_actor_critic_bundle,
    load_actor_from_path,
)
from evc.offline_dae_det_temporal_shield import load_temporal_shield_bundle
from evc.ug_bcr import rollout_episode_with_ug_bcr


ATTACK_SPECS: list[dict[str, Any]] = [
    {
        "key": "clean",
        "display": "Clean",
        "algorithm": None,
        "scenario": "O",
        "scope": "all",
        "seen": False,
    },
    {
        "key": "opposite_pgd",
        "display": "PGD*",
        "algorithm": "opposite_pgd",
        "scenario": "O",
        "scope": "all",
        "seen": True,
    },
    {
        "key": "q_function",
        "display": "Q-function*",
        "algorithm": "q_function",
        "scenario": "O",
        "scope": "all",
        "seen": True,
    },
    {
        "key": "opposite_fgsm",
        "display": "FGSM",
        "algorithm": "opposite_fgsm",
        "scenario": "O",
        "scope": "all",
        "seen": False,
    },
    {
        "key": "electhacker_c",
        "display": "EH-C",
        "algorithm": "electhacker",
        "scenario": "C",
        "scope": "all",
        "seen": False,
    },
    {
        "key": "electhacker_f",
        "display": "EH-F",
        "algorithm": "electhacker",
        "scenario": "F",
        "scope": "all",
        "seen": False,
    },
    {
        "key": "electhacker_o",
        "display": "EH-O",
        "algorithm": "electhacker",
        "scenario": "O",
        "scope": "all",
        "seen": False,
    },
    {
        "key": "local_small_drift_q",
        "display": "small_drift_q",
        "algorithm": "local_small_drift_q",
        "scenario": "O",
        "scope": "local",
        "seen": False,
    },
    {
        "key": "local_deadline_drift_pgd",
        "display": "deadline_pgd",
        "algorithm": "local_deadline_drift_pgd",
        "scenario": "O",
        "scope": "local",
        "seen": False,
    },
    {
        "key": "full_pipeline_adaptive_deadline",
        "display": "FP-Adaptive-Deadline",
        "algorithm": "full_pipeline_adaptive_deadline",
        "scenario": "O",
        "scope": "local",
        "seen": False,
    },
]

STAGE_SPECS: list[dict[str, Any]] = [
    {
        "key": "attack",
        "display": "Attack",
        "route_mode": "none",
        "use_dae": False,
        "use_detector": False,
        "enable_shield": False,
        "enable_belief": False,
        "enable_urgency_gate": False,
    },
    {
        "key": "denoise",
        "display": "Denoise",
        "route_mode": "always_dae",
        "use_dae": True,
        "use_detector": False,
        "enable_shield": False,
        "enable_belief": False,
        "enable_urgency_gate": False,
    },
    {
        "key": "denoise_det",
        "display": "Denoise+DET",
        "route_mode": "detector",
        "use_dae": True,
        "use_detector": True,
        "enable_shield": False,
        "enable_belief": False,
        "enable_urgency_gate": False,
    },
    {
        "key": "shield",
        "display": "+Shield",
        "route_mode": "detector",
        "use_dae": True,
        "use_detector": True,
        "enable_shield": True,
        "enable_belief": False,
        "enable_urgency_gate": False,
    },
    {
        "key": "ug_bcr",
        "display": "+UG-BCR",
        "route_mode": "detector",
        "use_dae": True,
        "use_detector": True,
        "enable_shield": True,
        "enable_belief": True,
        "enable_urgency_gate": True,
    },
]

TABLE3_ATTACK_KEYS = [
    "opposite_pgd",
    "q_function",
    "electhacker_f",
    "local_small_drift_q",
    "local_deadline_drift_pgd",
    "full_pipeline_adaptive_deadline",
]
TABLE3_STAGE_KEYS = ["attack", "denoise_det", "shield", "ug_bcr"]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def sample_std(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def format_mean_std(mean: float, std: float, digits: int = 1) -> str:
    return f"{float(mean):.{digits}f}±{float(std):.{digits}f}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(c) for c in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in frame.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def latex_escape(value: str) -> str:
    mapping = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(mapping.get(char, char) for char in str(value))


def latex_table(frame: pd.DataFrame, caption: str, label: str) -> str:
    align = "l" + "c" * (len(frame.columns) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        r"\hline",
        " & ".join(latex_escape(c) for c in frame.columns) + r" \\",
        r"\hline",
    ]
    for _, row in frame.iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c in frame.columns) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_boot: int = 5000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_price_threshold(dtsr_dir: Path) -> float:
    path = dtsr_dir / "electhacker_c_price_threshold.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_split") != "val":
        raise RuntimeError("ElectHacker-C threshold must be calibrated on validation split.")
    return float(payload["price_threshold"])


def load_price_threshold_from_path(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_split") not in {None, "val"}:
        raise RuntimeError("ElectHacker-C threshold must be calibrated on validation split.")
    if "price_threshold" in payload:
        return float(payload["price_threshold"])
    if "new_threshold" in payload:
        return float(payload["new_threshold"])
    raise KeyError(f"No price_threshold/new_threshold in {path}")


def existing_artifact_path(primary_dir: Path | None, fallback_dir: Path, filename: str) -> Path:
    candidates = []
    if primary_dir is not None:
        candidates.append(Path(primary_dir) / filename)
    candidates.append(Path(fallback_dir) / filename)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {filename}; checked: {[str(path) for path in candidates]}")


def parse_key_list(raw: str) -> list[str]:
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def build_short_attacker(
    *,
    algorithm: str,
    actor,
    critic,
    device: torch.device,
    low: np.ndarray,
    high: np.ndarray,
    seed: int,
    epsilon: float,
) -> PGDStateAttacker:
    if algorithm == "opposite_fgsm":
        alpha = float(epsilon)
        iters = 1
    elif algorithm == "electhacker":
        alpha = 0.005
        iters = 100
    else:
        alpha = 0.01
        iters = 10
    return PGDStateAttacker(
        actor,
        device=device,
        algorithm=algorithm,
        epsilon=float(epsilon),
        alpha=float(alpha),
        iters=int(iters),
        seed=int(seed),
        obs_low=low,
        obs_high=high,
        critic=critic if algorithm == "q_function" else None,
        attack_state_scope="all",
    )


def build_attacker_for_rollout(
    *,
    attack_spec: dict[str, Any],
    actor,
    critic,
    device: torch.device,
    arrivals: pd.DataFrame,
    signal_path: Path,
    attack_seed: int,
    epsilon: float,
    dae,
    detector_model,
    detector_threshold: float,
    shield_config,
    ug_bcr_config,
    formal_long_outer_epsilon: float | None = None,
):
    algorithm = attack_spec["algorithm"]
    if algorithm is None:
        return None
    env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
    max_duration = max(12, int(arrivals["Duration_of_stay"].max()))
    low, high = env.observation_bounds(max_duration_of_stay=max_duration)
    if algorithm in {"opposite_pgd", "opposite_fgsm", "q_function", "electhacker"}:
        return build_short_attacker(
            algorithm=algorithm,
            actor=actor,
            critic=critic,
            device=device,
            low=low,
            high=high,
            seed=attack_seed,
            epsilon=epsilon,
        )

    if uses_formal_experimental_long_horizon(str(algorithm)):
        formal_kwargs: dict[str, Any] = {}
        if formal_long_outer_epsilon is not None:
            outer_epsilon = float(formal_long_outer_epsilon)
            if not np.isfinite(outer_epsilon) or outer_epsilon <= 0.0:
                raise ValueError("formal_long_outer_epsilon must be finite and positive.")
            strength_ratio = outer_epsilon / 0.055
            if algorithm == "local_deadline_drift_pgd":
                formal_kwargs["attack_overrides"] = {
                    "epsilon": outer_epsilon,
                    "base_epsilon": 0.028 * strength_ratio,
                    "base_alpha": 0.008 * strength_ratio,
                    "base_iters": 5,
                }
            elif algorithm == "local_small_drift_q":
                formal_kwargs["attack_overrides"] = {
                    "epsilon": outer_epsilon,
                    "step_size": 0.039 * strength_ratio,
                    "slew_limit": min(0.030 * strength_ratio, outer_epsilon),
                    "base_epsilon": min(0.030 * strength_ratio, outer_epsilon),
                    "base_alpha": min(0.010 * strength_ratio, outer_epsilon),
                    "base_iters": 5,
                }
        attacker = build_formal_experimental_long_horizon_attacker(
            str(algorithm),
            actor=actor,
            device=device,
            obs_low=low,
            obs_high=high,
            critic=critic,
            seed=attack_seed,
            attack_state_scope=str(attack_spec.get("scope", "local")),
            **formal_kwargs,
        )
        if formal_long_outer_epsilon is not None and algorithm == "local_small_drift_q":
            attacker.base_attacker.iters = 5
    else:
        attacker = build_long_horizon_attacker(
            algorithm,
            actor=actor,
            device=device,
            obs_low=low,
            obs_high=high,
            critic=critic,
            seed=attack_seed,
        )
    if algorithm == "full_pipeline_adaptive_deadline":
        if not hasattr(attacker, "configure_target_defense"):
            raise RuntimeError("Adaptive attacker does not expose configure_target_defense().")
        attacker.configure_target_defense(
            defender=dae,
            detector_model=detector_model,
            detector_threshold=float(detector_threshold),
            shield_config=shield_config,
            ug_bcr_config=ug_bcr_config,
            reward_profile=TRAIN_PROFILE,
            signals_path=signal_path,
            device=device,
            actor=actor,
            repair_mode=REPAIR_MODE,
        )
        if not bool(getattr(attacker, "_target_ready", False)):
            raise RuntimeError("FullPipelineAdaptiveDeadlineAttacker target defense is not ready.")
    return attacker


def stage_kwargs(
    stage_spec: dict[str, Any],
    *,
    dae,
    detector_model,
    detector_threshold: float,
    shield_config,
    ug_bcr_config,
) -> dict[str, Any]:
    return {
        "defender": dae if stage_spec["use_dae"] else None,
        "detector_model": detector_model if stage_spec["use_detector"] else None,
        "detector_threshold": float(detector_threshold) if stage_spec["use_detector"] else None,
        "shield_config": shield_config if stage_spec["enable_shield"] else None,
        "route_mode": stage_spec["route_mode"],
        "enable_shield": bool(stage_spec["enable_shield"]),
        "enable_belief": bool(stage_spec["enable_belief"]),
        "enable_urgency_gate": bool(stage_spec["enable_urgency_gate"]),
        "ug_bcr_config": ug_bcr_config if stage_spec["enable_belief"] else None,
    }


def adaptive_diagnostics(attacker) -> dict[str, Any]:
    if attacker is None:
        return {}
    log = getattr(attacker, "_candidate_log", None)
    if not log:
        return {}
    frame = pd.DataFrame(log)
    result: dict[str, Any] = {"adaptive_candidate_count": int(len(frame))}
    for column, output_name in [
        ("score", "adaptive_mean_selected_score"),
        ("route", "adaptive_route_rate"),
        ("belief", "adaptive_belief_rate"),
        ("temporal", "adaptive_temporal_mean"),
        ("action", "adaptive_post_defense_action_mean"),
    ]:
        if column in frame.columns:
            result[output_name] = float(frame[column].mean())
    return result


def create_summary(long_df: pd.DataFrame, output_dir: Path, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for attack_spec in ATTACK_SPECS:
        for stage_spec in STAGE_SPECS:
            subset = long_df[
                (long_df["attack_key"] == attack_spec["key"])
                & (long_df["stage_key"] == stage_spec["key"])
            ].sort_values("episode_index")
            if subset.empty:
                continue
            row = {
                "attack_key": attack_spec["key"],
                "attack_display_name": attack_spec["display"],
                "seen_in_dtsr_training": bool(attack_spec["seen"]),
                "stage_key": stage_spec["key"],
                "stage_display_name": stage_spec["display"],
                "scenario_count": int(len(subset)),
            }
            for source, target in [
                ("ep_reward", "reward"),
                ("run_vio", "run_vio"),
                ("exit_vio", "exit_vio"),
                ("mean_fin_soc", "mean_fin_soc"),
                ("route_rate", "route_rate"),
                ("shield_correction_mean", "shield_correction"),
                ("urgency_gate_belief_rate", "ug_bcr_belief_rate"),
            ]:
                values = subset[source].astype(float).to_numpy() if source in subset.columns else np.zeros(len(subset), dtype=float)
                row[f"{target}_mean"] = float(np.mean(values))
                row[f"{target}_std"] = sample_std(values)
            rows.append(row)
    frame = pd.DataFrame(rows)
    atomic_csv(frame, output_dir / "tables" / "table_ablation_summary_raw.csv")
    return frame


def create_recovery_interval_summary(long_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    clean_raw = long_df[
        (long_df["attack_key"] == "clean") & (long_df["stage_key"] == "attack")
    ][["scenario_id", "ep_reward"]].rename(columns={"ep_reward": "clean_raw_reward"})
    rows: list[dict[str, Any]] = []
    for attack_key in sorted(key for key in long_df["attack_key"].dropna().unique() if key != "clean"):
        attack_data = long_df[long_df["attack_key"] == attack_key]
        raw_attack = attack_data[attack_data["stage_key"] == "attack"][
            ["scenario_id", "ep_reward"]
        ].rename(columns={"ep_reward": "attack_reward"})
        for stage_key in sorted(key for key in attack_data["stage_key"].dropna().unique() if key != "attack"):
            defended = attack_data[attack_data["stage_key"] == stage_key][
                ["scenario_id", "attack_display_name", "seen_in_dtsr_training", "stage_display_name", "ep_reward"]
            ].rename(columns={"ep_reward": "defended_reward"})
            paired = clean_raw.merge(raw_attack, on="scenario_id", how="inner").merge(defended, on="scenario_id", how="inner")
            if paired.empty:
                continue
            denominator = paired["clean_raw_reward"].to_numpy(dtype=float) - paired["attack_reward"].to_numpy(dtype=float)
            numerator = paired["defended_reward"].to_numpy(dtype=float) - paired["attack_reward"].to_numpy(dtype=float)
            valid = np.abs(denominator) > 1e-8
            recovery = np.full(denominator.shape, np.nan, dtype=np.float64)
            recovery[valid] = numerator[valid] / denominator[valid] * 100.0
            recovery = recovery[np.isfinite(recovery)]
            if recovery.size == 0:
                continue
            mean = float(np.mean(recovery))
            std = sample_std(recovery)
            rows.append(
                {
                    "attack_key": attack_key,
                    "attack_display_name": str(defended["attack_display_name"].iloc[0]),
                    "seen_in_dtsr_training": bool(defended["seen_in_dtsr_training"].iloc[0]),
                    "stage_key": stage_key,
                    "stage_display_name": str(defended["stage_display_name"].iloc[0]),
                    "scenario_count": int(recovery.size),
                    "recovery_mean_pct": mean,
                    "recovery_std_pct": std,
                    "recovery_mean_pm_std": f"{mean:.1f} +/- {std:.1f}%",
                    "recovery_min_pct": float(np.min(recovery)),
                    "recovery_max_pct": float(np.max(recovery)),
                }
            )
    frame = pd.DataFrame(rows)
    atomic_csv(frame, output_dir / "tables" / "recovery_interval_summary.csv")
    return frame


def create_table2(long_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_raw = long_df[
        (long_df["attack_key"] == "clean") & (long_df["stage_key"] == "attack")
    ][["scenario_id", "ep_reward"]].rename(columns={"ep_reward": "clean_raw_reward"})
    rows_raw: list[dict[str, Any]] = []
    rows_paper: list[dict[str, Any]] = []

    for attack_spec in ATTACK_SPECS:
        attack_key = attack_spec["key"]
        attack_data = long_df[long_df["attack_key"] == attack_key]
        raw_row: dict[str, Any] = {
            "attack_key": attack_key,
            "scene": attack_spec["display"],
            "seen_in_dtsr_training": bool(attack_spec["seen"]),
        }
        paper_row: dict[str, Any] = {"场景": attack_spec["display"]}
        for stage_spec in STAGE_SPECS:
            stage_key = stage_spec["key"]
            subset = attack_data[attack_data["stage_key"] == stage_key]
            rewards = subset["ep_reward"].astype(float).to_numpy()
            reward_mean = float(np.mean(rewards))
            reward_std = sample_std(rewards)
            raw_row[f"{stage_key}_reward_mean"] = reward_mean
            raw_row[f"{stage_key}_reward_std"] = reward_std
            raw_row[f"{stage_key}_scenario_count"] = int(len(rewards))
            paper_row[stage_spec["display"]] = format_mean_std(reward_mean, reward_std, 1)

        full = attack_data[attack_data["stage_key"] == "ug_bcr"][
            ["scenario_id", "ep_reward", "route_rate"]
        ].rename(columns={"ep_reward": "defended_reward"})
        raw_attack = attack_data[attack_data["stage_key"] == "attack"][
            ["scenario_id", "ep_reward"]
        ].rename(columns={"ep_reward": "attack_reward"})
        paired = clean_raw.merge(raw_attack, on="scenario_id", how="inner").merge(full, on="scenario_id", how="inner")
        if attack_key == "clean":
            recovery = np.asarray([], dtype=np.float64)
            recovery_mean = float("nan")
            recovery_std = float("nan")
            valid_recovery_count = 0
            paper_row["恢复率/%"] = "—"
        else:
            denominator = paired["clean_raw_reward"].to_numpy(dtype=float) - paired["attack_reward"].to_numpy(dtype=float)
            numerator = paired["defended_reward"].to_numpy(dtype=float) - paired["attack_reward"].to_numpy(dtype=float)
            valid = np.abs(denominator) > 1e-8
            recovery = np.full(denominator.shape, np.nan, dtype=np.float64)
            recovery[valid] = numerator[valid] / denominator[valid] * 100.0
            recovery_mean = float(np.nanmean(recovery))
            recovery_std = sample_std(recovery)
            valid_recovery_count = int(np.sum(np.isfinite(recovery)))
            paper_row["恢复率/%"] = format_mean_std(recovery_mean, recovery_std, 1)

        route_rates = paired["route_rate"].to_numpy(dtype=float) * 100.0
        route_mean = float(np.mean(route_rates))
        route_std = sample_std(route_rates)
        paper_row["路由率/%"] = format_mean_std(route_mean, route_std, 1)

        raw_row.update(
            {
                "recovery_mean_pct": recovery_mean,
                "recovery_std_pct": recovery_std,
                "recovery_valid_scenario_count": valid_recovery_count,
                "route_rate_mean_pct": route_mean,
                "route_rate_std_pct": route_std,
            }
        )
        rows_raw.append(raw_row)
        rows_paper.append(paper_row)

    raw_frame = pd.DataFrame(rows_raw)
    paper_frame = pd.DataFrame(rows_paper)[
        ["场景", "Attack", "Denoise", "Denoise+DET", "+Shield", "+UG-BCR", "恢复率/%", "路由率/%"]
    ]
    tables_dir = output_dir / "tables"
    atomic_csv(raw_frame, tables_dir / "table2_reward_ablation_raw.csv")
    atomic_csv(paper_frame, tables_dir / "table2_reward_ablation_paper.csv")
    note_cn = "注：表中结果均为单一训练种子（seed=42）下120个固定测试场景的均值±标准差；*表示DTSR离线训练样本中包含该攻击。"
    note_en = "Results are reported as mean ± standard deviation over 120 fixed test episodes using a single training seed (seed=42). * indicates attacks included in the offline DTSR training set."
    (tables_dir / "table2_reward_ablation.md").write_text(markdown_table(paper_frame) + "\n" + note_cn + "\n\n" + note_en + "\n", encoding="utf-8")
    (tables_dir / "table2_reward_ablation.tex").write_text(
        latex_table(paper_frame, "Multi-stage offline defense reward ablation.", "tab:dtsr_reward_ablation") + "% " + note_en + "\n",
        encoding="utf-8",
    )
    return raw_frame, paper_frame


def create_table3(long_df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_raw: list[dict[str, Any]] = []
    rows_paper: list[dict[str, Any]] = []
    attack_lookup = {row["key"]: row for row in ATTACK_SPECS}
    stage_lookup = {row["key"]: row for row in STAGE_SPECS}
    for attack_key in TABLE3_ATTACK_KEYS:
        attack_spec = attack_lookup[attack_key]
        raw_row: dict[str, Any] = {
            "attack_key": attack_key,
            "scene": attack_spec["display"],
        }
        paper_row: dict[str, Any] = {"场景": attack_spec["display"]}
        for stage_key in TABLE3_STAGE_KEYS:
            subset = long_df[
                (long_df["attack_key"] == attack_key) & (long_df["stage_key"] == stage_key)
            ]
            run_values = subset["run_vio"].astype(float).to_numpy()
            exit_values = subset["exit_vio"].astype(float).to_numpy()
            run_mean, run_std = float(np.mean(run_values)), sample_std(run_values)
            exit_mean, exit_std = float(np.mean(exit_values)), sample_std(exit_values)
            raw_row[f"{stage_key}_run_vio_mean"] = run_mean
            raw_row[f"{stage_key}_run_vio_std"] = run_std
            raw_row[f"{stage_key}_exit_vio_mean"] = exit_mean
            raw_row[f"{stage_key}_exit_vio_std"] = exit_std
            raw_row[f"{stage_key}_scenario_count"] = int(len(subset))
            paper_row[stage_lookup[stage_key]["display"]] = (
                f"{run_mean:.1f}±{run_std:.1f} / {exit_mean:.1f}±{exit_std:.1f}"
            )
        rows_raw.append(raw_row)
        rows_paper.append(paper_row)

    raw_frame = pd.DataFrame(rows_raw)
    paper_frame = pd.DataFrame(rows_paper)[
        ["场景", "Attack", "Denoise+DET", "+Shield", "+UG-BCR"]
    ]
    tables_dir = output_dir / "tables"
    atomic_csv(raw_frame, tables_dir / "table3_violation_ablation_raw.csv")
    atomic_csv(paper_frame, tables_dir / "table3_violation_ablation_paper.csv")
    note_cn = "注：表中结果均为单一训练种子（seed=42）下120个固定测试场景的均值±标准差；*表示DTSR离线训练样本中包含该攻击。每个单元格按“运行期违规/离站违规”报告。"
    note_en = "Results are reported as mean ± standard deviation over 120 fixed test episodes using a single training seed (seed=42). * indicates attacks included in the offline DTSR training set. Each violation entry is reported as running violations / exit violations."
    (tables_dir / "table3_violation_ablation.md").write_text(markdown_table(paper_frame) + "\n" + note_cn + "\n\n" + note_en + "\n", encoding="utf-8")
    (tables_dir / "table3_violation_ablation.tex").write_text(
        latex_table(paper_frame, "Running/exit violation ablation under representative attacks.", "tab:dtsr_violation_ablation") + "% " + note_en + "\n",
        encoding="utf-8",
    )
    return raw_frame, paper_frame


def create_stage_recovery_statistics(long_df: pd.DataFrame, output_dir: Path, seed: int) -> tuple[pd.DataFrame, list[str]]:
    transitions = [
        ("attack", "denoise"),
        ("denoise", "denoise_det"),
        ("denoise_det", "shield"),
        ("shield", "ug_bcr"),
    ]
    rows: list[dict[str, Any]] = []
    flags: list[str] = []
    for attack_index, attack_spec in enumerate(ATTACK_SPECS):
        attack_data = long_df[long_df["attack_key"] == attack_spec["key"]]
        if attack_data.empty:
            continue
        for transition_index, (before_key, after_key) in enumerate(transitions):
            before = attack_data[attack_data["stage_key"] == before_key][["scenario_id", "ep_reward"]].rename(columns={"ep_reward": "before"})
            after = attack_data[attack_data["stage_key"] == after_key][["scenario_id", "ep_reward"]].rename(columns={"ep_reward": "after"})
            paired = before.merge(after, on="scenario_id", how="inner")
            if paired.empty:
                continue
            delta = paired["after"].to_numpy(dtype=float) - paired["before"].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean_ci(delta, seed=seed + attack_index * 100 + transition_index)
            rows.append(
                {
                    "attack_key": attack_spec["key"],
                    "attack_display_name": attack_spec["display"],
                    "before_stage": before_key,
                    "after_stage": after_key,
                    "scenario_count": int(len(delta)),
                    "mean_delta_reward": float(np.mean(delta)),
                    "median_delta_reward": float(np.median(delta)),
                    "improved_scenario_count": int(np.sum(delta > 0.0)),
                    "degraded_scenario_count": int(np.sum(delta < 0.0)),
                    "unchanged_scenario_count": int(np.sum(delta == 0.0)),
                    "bootstrap_95ci_lower": ci_low,
                    "bootstrap_95ci_upper": ci_high,
                }
            )

        denoise_pairs = attack_data.pivot(index="scenario_id", columns="stage_key", values="ep_reward")
        if {"attack", "denoise"}.issubset(denoise_pairs.columns):
            failure_fraction = float(np.mean(denoise_pairs["denoise"] < denoise_pairs["attack"]))
            if failure_fraction > 0.50:
                flags.append(f"DENOISE_GENERAL_FAILURE:{attack_spec['key']}:{failure_fraction:.4f}")

    stats = pd.DataFrame(rows)
    atomic_csv(stats, output_dir / "tables" / "dtsr_stage_recovery_statistics.csv")

    if not stats.empty:
        stats_lookup = stats.set_index(["attack_key", "before_stage", "after_stage"])
        det_rows = []
        for key in ("opposite_pgd", "q_function"):
            idx = (key, "denoise", "denoise_det")
            if idx in stats_lookup.index:
                det_rows.append(float(stats_lookup.loc[idx, "mean_delta_reward"]))
        if len(det_rows) == 2 and all(value <= 0.0 for value in det_rows):
            flags.append("DET_NO_BENEFIT")

    shield_subset = long_df[long_df["stage_key"] == "shield"]
    if not shield_subset.empty:
        if float(shield_subset["shield_correction_mean"].abs().max()) <= 1e-12 and all(
            float(shield_subset[column].abs().max()) <= 1e-12
            for column in ["shield_soc_clamp_rate", "shield_time_clamp_rate", "shield_cost_clamp_rate"]
            if column in shield_subset.columns
        ):
            flags.append("SHIELD_INACTIVE")

    full_subset = long_df[long_df["stage_key"] == "ug_bcr"]
    if not full_subset.empty and "urgency_gate_belief_rate" in full_subset.columns:
        by_attack = full_subset.groupby("attack_key")["urgency_gate_belief_rate"].mean()
        if bool((by_attack < 0.001).all()):
            flags.append("UG_BCR_NEVER_ACTIVATES")
        if bool((by_attack > 0.999).all()):
            flags.append("UG_BCR_ALWAYS_ACTIVATES")

    clean = long_df[long_df["attack_key"] == "clean"].pivot(index="scenario_id", columns="stage_key", values="ep_reward")
    if {"attack", "ug_bcr"}.issubset(clean.columns):
        raw_mean = float(clean["attack"].mean())
        full_mean = float(clean["ug_bcr"].mean())
        drop = max(0.0, raw_mean - full_mean)
        if drop > 0.03 * max(abs(raw_mean), 1e-8):
            flags.append("CLEAN_PRESERVATION_FAILURE")

    flags = list(dict.fromkeys(flags))
    return stats, flags


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed seed=42 DTSR on 120 test scenes and build paper Tables 2 and 3."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor-path", type=Path, default=EP100_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=EP100_BUNDLE_PATH)
    parser.add_argument("--dtsr-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dtsr_seed42_multiday")
    parser.add_argument("--dae-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday")
    parser.add_argument("--detector-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "det_seed42_fullstate")
    parser.add_argument("--shield-artifact-dir", type=Path, default=None)
    parser.add_argument("--ug-bcr-config-path", type=Path, default=None)
    parser.add_argument("--price-threshold-file", type=Path, default=PACKAGE_ROOT / "results" / "attack120_short_horizon" / "ehc_threshold_fix" / "electhacker_c_price_threshold.json")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--scenes", type=int, default=120)
    parser.add_argument(
        "--stage-keys",
        default=",".join(spec["key"] for spec in STAGE_SPECS),
        help="Comma-separated stage keys from: attack,denoise,denoise_det,shield,ug_bcr.",
    )
    parser.add_argument(
        "--attack-keys",
        default=",".join(spec["key"] for spec in ATTACK_SPECS),
        help="Comma-separated attack keys. Clean is always included.",
    )
    parser.add_argument("--strict-final-120", action="store_true", help="Require complete 120-scenario final table mode.")
    parser.add_argument("--sanity-only", action="store_true", help="Run fixed validation sanity evaluation without generating final paper tables.")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--attack-ratio", type=float, default=1.0)
    parser.add_argument("--attack-scope", choices=["obs", "vehicle", "window"], default="obs")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "results" / "dtsr_retrain_seed42")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rollouts", type=int, default=0, help="Debug only: stop after N new rollouts; 0 means all.")
    args = parser.parse_args()

    if int(args.seed) != 42:
        raise ValueError("Final table experiment is fixed to the single DTSR training seed seed=42.")
    if int(args.scenes) <= 0:
        raise ValueError("--scenes must be positive.")
    if args.strict_final_120 and args.split == "test" and not args.sanity_only and int(args.scenes) != 120:
        raise ValueError("Final Table 2/Table 3 experiment must use all 120 fixed test scenarios.")
    if args.split == "val" and int(args.scenes) > 60:
        raise ValueError("Validation split contains only 60 scenarios.")
    if not math.isclose(float(args.epsilon), 0.1, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Final table experiment is locked to epsilon=0.1 for the short-horizon attacks.")

    requested_attack_keys = parse_key_list(args.attack_keys)
    if "clean" not in requested_attack_keys:
        requested_attack_keys.insert(0, "clean")
    requested_attack_keys = list(dict.fromkeys(requested_attack_keys))
    attack_lookup = {spec["key"]: spec for spec in ATTACK_SPECS}
    unknown_attack_keys = [key for key in requested_attack_keys if key not in attack_lookup]
    if unknown_attack_keys:
        raise ValueError(f"Unknown attack keys: {unknown_attack_keys}")
    selected_attack_specs = [attack_lookup[key] for key in requested_attack_keys]
    stage_lookup = {spec["key"]: spec for spec in STAGE_SPECS}
    requested_stage_keys = parse_key_list(args.stage_keys)
    unknown_stage_keys = [key for key in requested_stage_keys if key not in stage_lookup]
    if unknown_stage_keys:
        raise ValueError(f"Unknown stage keys: {unknown_stage_keys}")
    selected_stage_specs = [stage_lookup[key] for key in requested_stage_keys]
    if not selected_stage_specs:
        raise ValueError("At least one stage key must be selected.")
    if args.strict_final_120 and not args.sanity_only and args.split == "test" and len(selected_attack_specs) != len(ATTACK_SPECS):
        raise ValueError("Final table mode requires the complete 10-row attack set.")
    if args.strict_final_120 and [spec["key"] for spec in selected_stage_specs] != [spec["key"] for spec in STAGE_SPECS]:
        raise ValueError("Final table mode requires the complete five-stage ablation.")

    if args.overwrite and args.output_dir.exists():
        import shutil
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        intermediate_path = args.output_dir / "intermediate" / "table_rollouts.jsonl"
        if intermediate_path.exists():
            intermediate_path.unlink()

    set_all_seeds(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)

    actor = load_actor_from_path(args.actor_path, device).eval()
    bundle_payload = load_actor_critic_bundle(args.bundle_path, device)
    if not actor_matches_bundle(actor, bundle_payload):
        raise RuntimeError("Selected actor does not match bundle actor_state_dict.")
    checkpoint_episode = int((bundle_payload.get("metadata") or {}).get("checkpoint_episode", -1))
    if checkpoint_episode != 100:
        raise RuntimeError(f"Expected ep100 DDPG, got checkpoint_episode={checkpoint_episode}.")
    critic_state = bundle_payload.get("critic_state_dict")
    if critic_state is None:
        raise RuntimeError("The selected ep100 bundle has no critic_state_dict.")
    critic = Critic().to(device)
    critic.load_state_dict(critic_state)
    actor.eval()
    critic.eval()
    for module in (actor, critic):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    selected_stage_keys = {spec["key"] for spec in selected_stage_specs}
    selected_attack_keys = {spec["key"] for spec in selected_attack_specs}
    adaptive_selected = "full_pipeline_adaptive_deadline" in selected_attack_keys
    need_dae = any(bool(spec["use_dae"]) for spec in selected_stage_specs) or adaptive_selected
    need_detector = any(bool(spec["use_detector"]) for spec in selected_stage_specs) or adaptive_selected
    need_shield = any(bool(spec["enable_shield"]) for spec in selected_stage_specs) or adaptive_selected
    need_ug_bcr = any(bool(spec["enable_belief"]) or bool(spec["enable_urgency_gate"]) for spec in selected_stage_specs) or adaptive_selected

    dae = None
    dae_path = None
    if need_dae:
        dae_path = existing_artifact_path(args.dae_artifact_dir, args.dtsr_dir, "dtsr_dae.pt")
        dae = load_dae(dae_path, device)
        dae.eval()

    detector_model = None
    detector_threshold = float("nan")
    detector_path = None
    if need_detector:
        detector_path = existing_artifact_path(args.detector_artifact_dir, args.dtsr_dir, "dtsr_detector.pt")
        detector_artifact = load_detector(detector_path, device)
        detector_model = detector_artifact.model
        detector_threshold = float(detector_artifact.threshold)

    shield_config = None
    shield_path = None
    if need_shield:
        shield_dir = args.shield_artifact_dir if args.shield_artifact_dir is not None else args.dtsr_dir
        shield_path = existing_artifact_path(shield_dir, args.dtsr_dir, "dtsr_temporal_shield.pt")
        shield_artifact = load_temporal_shield_bundle(shield_path)
        shield_config = shield_artifact.config

    ug_bcr_config = None
    ug_bcr_path = None
    if need_ug_bcr:
        ug_bcr_path = args.ug_bcr_config_path if args.ug_bcr_config_path is not None else args.dtsr_dir / "ug_bcr_config.json"
        if not ug_bcr_path.exists():
            raise FileNotFoundError(f"Missing UG-BCR config: {ug_bcr_path}")
        ug_bcr_config = load_ug_bcr_config(ug_bcr_path)

    if args.price_threshold_file.exists():
        price_threshold = load_price_threshold_from_path(args.price_threshold_file)
    else:
        price_threshold = load_price_threshold(args.dtsr_dir)

    dtsr_manifest_path = args.dtsr_dir / "dtsr_manifest.json"
    if dtsr_manifest_path.exists():
        dtsr_manifest = json.loads(dtsr_manifest_path.read_text(encoding="utf-8"))
        if int(dtsr_manifest.get("seed", 42)) != 42:
            raise RuntimeError("DTSR manifest seed is not 42.")
        if str(dtsr_manifest.get("repair_mode", REPAIR_MODE)) != REPAIR_MODE:
            raise RuntimeError("DTSR manifest repair_mode does not match runtime.")
        if str(dtsr_manifest.get("runtime_pipeline_order", RUNTIME_PIPELINE_ORDER)) != RUNTIME_PIPELINE_ORDER:
            raise RuntimeError("DTSR manifest runtime pipeline order does not match audited original logic.")
    else:
        dtsr_manifest = {}

    test_manifest = load_manifest(args.split).sort_values("Scenario_ID", kind="mergesort").reset_index(drop=True)
    if len(test_manifest) < int(args.scenes):
        raise RuntimeError(f"Requested {args.scenes} {args.split} scenarios, found only {len(test_manifest)}.")
    test_manifest = test_manifest.iloc[: int(args.scenes)].copy().reset_index(drop=True)
    if args.split == "test" and int(args.scenes) == 120:
        expected_ids = [f"test_day_{index:04d}" for index in range(1, 121)]
        actual_ids = test_manifest["Scenario_ID"].astype(str).tolist()
        if actual_ids != expected_ids:
            raise RuntimeError("Test scenario order is not test_day_0001 ... test_day_0120.")

    scenario_order = test_manifest[["Scenario_ID", "Vehicle_File", "Signal_File", "Context_File"]].copy()
    scenario_order.insert(0, "episode_index", np.arange(1, len(test_manifest) + 1, dtype=int))
    scenario_order["base_seed"] = int(args.seed)
    atomic_csv(scenario_order, args.output_dir / "scenario_order.csv")

    run_config = {
        "seed": int(args.seed),
        "device": str(device),
        "actor_path": str(args.actor_path),
        "bundle_path": str(args.bundle_path),
        "ddpg_checkpoint_episode": checkpoint_episode,
        "dtsr_dir": str(args.dtsr_dir),
        "dae_artifact": None if dae_path is None else str(dae_path),
        "detector_artifact": None if detector_path is None else str(detector_path),
        "detector_threshold": None if not np.isfinite(detector_threshold) else float(detector_threshold),
        "shield_artifact": None if shield_path is None else str(shield_path),
        "ug_bcr_config": None if ug_bcr_path is None else str(ug_bcr_path),
        "repair_mode": REPAIR_MODE,
        "runtime_pipeline_order": RUNTIME_PIPELINE_ORDER,
        "ablation_addition_order": ABLATION_ADDITION_ORDER,
        "split": str(args.split),
        "scenario_count": int(len(test_manifest)),
        "attack_count": len(selected_attack_specs),
        "attack_keys": [spec["key"] for spec in selected_attack_specs],
        "stage_count": len(selected_stage_specs),
        "stage_keys": [spec["key"] for spec in selected_stage_specs],
        "expected_rollouts": len(test_manifest) * len(selected_attack_specs) * len(selected_stage_specs),
        "sanity_only": bool(args.sanity_only),
        "strict_final_120": bool(args.strict_final_120),
        "epsilon_short_attacks": float(args.epsilon),
        "attack_ratio": float(args.attack_ratio),
        "attack_scope": str(args.attack_scope),
        "electhacker_c_price_threshold": price_threshold,
        "electhacker_c_threshold_source": "validation median frozen during DTSR training",
        "adaptive_attack_target": "final full DTSR for every ablation column",
        "statistics": "single seed=42; mean ± sample std across 120 fixed test episodes",
    }
    write_json(args.output_dir / "run_config.json", run_config)

    environment_info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_json(args.output_dir / "environment_info.json", environment_info)

    intermediate_path = args.output_dir / "intermediate" / "table_rollouts.jsonl"
    expected_key_set = {
        (str(row["Scenario_ID"]), str(attack_spec["key"]), str(stage_spec["key"]))
        for _, row in test_manifest.iterrows()
        for attack_spec in selected_attack_specs
        for stage_spec in selected_stage_specs
    }
    existing_rows_all = load_jsonl(intermediate_path) if args.resume else []
    existing_rows = [
        row for row in existing_rows_all
        if (str(row["scenario_id"]), str(row["attack_key"]), str(row["stage_key"])) in expected_key_set
    ]
    completed_keys = {
        (str(row["scenario_id"]), str(row["attack_key"]), str(row["stage_key"]))
        for row in existing_rows
    }
    if len(completed_keys) != len(existing_rows):
        raise RuntimeError("Intermediate JSONL contains duplicate rollout keys.")

    expected_total = len(test_manifest) * len(selected_attack_specs) * len(selected_stage_specs)
    new_rollouts = 0
    started = time.perf_counter()

    for episode_index, (_, scenario_row) in enumerate(test_manifest.iterrows(), start=1):
        arrivals, signal_path, scenario_id = load_scenario(scenario_row)
        for attack_index, attack_spec in enumerate(selected_attack_specs):
            attack_seed = int(args.seed + attack_index * 100_000 + episode_index)
            for stage_index, stage_spec in enumerate(selected_stage_specs):
                key = (scenario_id, attack_spec["key"], stage_spec["key"])
                if key in completed_keys:
                    continue
                set_all_seeds(attack_seed)
                attacker = build_attacker_for_rollout(
                    attack_spec=attack_spec,
                    actor=actor,
                    critic=critic,
                    device=device,
                    arrivals=arrivals,
                    signal_path=signal_path,
                    attack_seed=attack_seed,
                    epsilon=args.epsilon,
                    dae=dae,
                    detector_model=detector_model,
                    detector_threshold=detector_threshold,
                    shield_config=shield_config,
                    ug_bcr_config=ug_bcr_config,
                )
                kwargs = stage_kwargs(
                    stage_spec,
                    dae=dae,
                    detector_model=detector_model,
                    detector_threshold=detector_threshold,
                    shield_config=shield_config,
                    ug_bcr_config=ug_bcr_config,
                )
                rollout_start = time.perf_counter()
                summary = rollout_episode_with_ug_bcr(
                    arrivals,
                    actor,
                    signal_path,
                    device,
                    TRAIN_PROFILE,
                    attack_enabled=attack_spec["key"] != "clean",
                    attack_scenario=attack_spec["scenario"],
                    attacker=attacker,
                    epsilon=float(args.epsilon),
                    state_scope=attack_spec["scope"],
                    price_threshold=float(price_threshold),
                    attack_ratio=float(args.attack_ratio),
                    attack_scope=str(args.attack_scope),
                    label=f"{attack_spec['key']}__{stage_spec['key']}",
                    repair_mode=REPAIR_MODE,
                    **kwargs,
                )
                runtime_seconds = float(time.perf_counter() - rollout_start)
                scalar = to_scalar_summary(summary)
                row = {
                    "scenario_id": scenario_id,
                    "episode_index": int(episode_index),
                    "seed": int(args.seed),
                    "attack_seed": int(attack_seed),
                    "attack_key": attack_spec["key"],
                    "attack_display_name": attack_spec["display"],
                    "attack_scenario": attack_spec["scenario"],
                    "attack_state_scope": attack_spec["scope"],
                    "seen_in_dtsr_training": bool(attack_spec["seen"]),
                    "stage_index": int(stage_index),
                    "stage_key": stage_spec["key"],
                    "stage_display_name": stage_spec["display"],
                    "runtime_seconds": runtime_seconds,
                    **scalar,
                    **adaptive_diagnostics(attacker),
                }
                if int(row.get("done_cnt", -1)) != 344:
                    raise RuntimeError(f"Incomplete rollout for {key}: done_cnt={row.get('done_cnt')}")
                if attack_spec["key"] == "clean" and int(row.get("attack_obs_count", -1)) != 0:
                    raise RuntimeError(f"Clean rollout unexpectedly attacked observations for {key}.")
                append_jsonl(intermediate_path, row)
                completed_keys.add(key)
                new_rollouts += 1
                completed = len(completed_keys)
                print(
                    f"[{completed:04d}/{expected_total}] ep={episode_index:03d} "
                    f"{attack_spec['display']} | {stage_spec['display']} "
                    f"reward={float(row['ep_reward']):.3f} "
                    f"run/exit={int(row.get('run_vio', 0))}/{int(row.get('exit_vio', 0))} "
                    f"time={runtime_seconds:.2f}s"
                )
                if args.max_rollouts > 0 and new_rollouts >= int(args.max_rollouts):
                    print(f"Debug stop after {new_rollouts} new rollouts.")
                    return
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    rows = [
        row for row in load_jsonl(intermediate_path)
        if (str(row["scenario_id"]), str(row["attack_key"]), str(row["stage_key"])) in expected_key_set
    ]
    if len(rows) != expected_total:
        raise RuntimeError(f"Expected {expected_total} completed rollouts, found {len(rows)}.")
    long_df = pd.DataFrame(rows)
    unique_count = long_df[["scenario_id", "attack_key", "stage_key"]].drop_duplicates().shape[0]
    if unique_count != expected_total:
        raise RuntimeError(f"Expected {expected_total} unique rollout keys, got {unique_count}.")
    long_df = long_df.sort_values(["episode_index", "attack_key", "stage_index"], kind="mergesort").reset_index(drop=True)
    atomic_csv(long_df, args.output_dir / "tables" / "table_ablation_episode_metrics_long.csv")

    summary_df = create_summary(long_df, args.output_dir, args.seed)
    recovery_interval_df = create_recovery_interval_summary(long_df, args.output_dir)
    stage_stats, anomaly_flags = create_stage_recovery_statistics(long_df, args.output_dir, args.seed)
    full_stage_keys = [spec["key"] for spec in STAGE_SPECS]
    selected_stage_keys_ordered = [spec["key"] for spec in selected_stage_specs]
    has_complete_final_tables = (
        selected_stage_keys_ordered == full_stage_keys
        and [spec["key"] for spec in selected_attack_specs] == [spec["key"] for spec in ATTACK_SPECS]
    )
    if args.sanity_only or not has_complete_final_tables:
        sanity_path = args.output_dir / "validation_sanity_summary.csv"
        atomic_csv(summary_df, sanity_path)
        write_json(
            args.output_dir / "validation_sanity_flags.json",
            {
                "flags": anomaly_flags,
                "partial_stage_eval": not has_complete_final_tables,
                "selected_stage_keys": selected_stage_keys_ordered,
                "selected_attack_keys": [spec["key"] for spec in selected_attack_specs],
            },
        )
        report_lines = [
            "# DTSR Configurable Evaluation Report",
            "",
            f"- Completed rollouts: {len(long_df)} / {expected_total}",
            f"- Split/scenes: {args.split} / {len(test_manifest)}",
            f"- Stage keys: {', '.join(selected_stage_keys_ordered)}",
            f"- Attack keys: {', '.join(spec['key'] for spec in selected_attack_specs)}",
            f"- DAE artifact: {dae_path}",
            f"- Detector artifact: {detector_path}",
            f"- Detector threshold: {detector_threshold if np.isfinite(detector_threshold) else 'n/a'}",
            f"- Anomaly flags: {anomaly_flags if anomaly_flags else 'None'}",
            "",
            "## Stage Summary",
            "",
            markdown_table(summary_df),
        ]
        if not stage_stats.empty:
            report_lines.extend(["", "## Paired Stage Gains", "", markdown_table(stage_stats)])
        if not recovery_interval_df.empty:
            report_lines.extend(["", "## Recovery Mean +/- Std", "", markdown_table(recovery_interval_df)])
        (args.output_dir / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")
        print(summary_df.to_string(index=False))
        print(f"Validation sanity completed: {len(long_df)} rollouts")
        print(f"Flags: {anomaly_flags if anomaly_flags else 'None'}")
        print(f"Saved: {sanity_path}")
        return

    table2_raw, table2_paper = create_table2(long_df, summary_df, args.output_dir)
    table3_raw, table3_paper = create_table3(long_df, args.output_dir)

    adaptive_full = long_df[
        (long_df["attack_key"] == "full_pipeline_adaptive_deadline")
        & (long_df["stage_key"] == "ug_bcr")
    ]
    adaptive_summary = {
        "scenario_count": int(len(adaptive_full)),
        "adaptive_candidate_count_mean": float(adaptive_full.get("adaptive_candidate_count", pd.Series(dtype=float)).mean()) if "adaptive_candidate_count" in adaptive_full.columns else None,
        "adaptive_mean_selected_score": float(adaptive_full.get("adaptive_mean_selected_score", pd.Series(dtype=float)).mean()) if "adaptive_mean_selected_score" in adaptive_full.columns else None,
        "adaptive_route_rate": float(adaptive_full.get("adaptive_route_rate", pd.Series(dtype=float)).mean()) if "adaptive_route_rate" in adaptive_full.columns else None,
        "adaptive_belief_rate": float(adaptive_full.get("adaptive_belief_rate", pd.Series(dtype=float)).mean()) if "adaptive_belief_rate" in adaptive_full.columns else None,
    }
    write_json(args.output_dir / "tables" / "adaptive_attack_diagnostics.json", adaptive_summary)
    write_json(args.output_dir / "tables" / "anomaly_flags.json", {"flags": anomaly_flags})

    report_lines = [
        "# DTSR seed=42 multiday retraining and 120-scenario ablation report",
        "",
        "## Experiment identity",
        "",
        f"- Frozen DDPG actor: `{args.actor_path}`",
        f"- Frozen DDPG bundle: `{args.bundle_path}`",
        f"- DDPG checkpoint episode: {checkpoint_episode}",
        f"- DTSR training seed: {args.seed}",
        f"- Runtime order preserved from original code: `{RUNTIME_PIPELINE_ORDER}`",
        f"- Ablation addition order: `{ABLATION_ADDITION_ORDER}`",
        f"- Repair mode: `{REPAIR_MODE}`",
        f"- ElectHacker-C threshold: {price_threshold:.8f}, frozen from validation prices",
        "",
        "## Statistical meaning of ±",
        "",
        "All mean ± standard deviation values are computed across the same 120 fixed test episodes under a single DTSR training seed (seed=42). Sample standard deviation uses ddof=1. They are not five-seed standard deviations.",
        "",
        "## Table 2",
        "",
        markdown_table(table2_paper),
        "## Table 3",
        "",
        markdown_table(table3_paper),
        "## Integrity and anomaly flags",
        "",
        f"- Completed rollouts: {len(long_df)} / {expected_total}",
        f"- Anomaly flags: {anomaly_flags if anomaly_flags else 'None'}",
        "",
        "## Stage gain audit",
        "",
        f"Detailed paired stage deltas are saved to `{args.output_dir / 'tables' / 'dtsr_stage_recovery_statistics.csv'}`.",
        "",
        "## Adaptive attack audit",
        "",
        json.dumps(adaptive_summary, ensure_ascii=False, indent=2),
        "",
        "## Rule",
        "",
        "No test result is used for hyperparameter selection. If an anomaly flag requires a parameter change, calibration must return to validation split and the complete 120-scenario test must be rerun from a frozen configuration.",
    ]
    (args.output_dir / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\nTable 2:\n")
    print(table2_paper.to_string(index=False))
    print("\nTable 3:\n")
    print(table3_paper.to_string(index=False))
    print(f"\nCompleted {len(long_df)} rollouts in {(time.perf_counter() - started) / 60.0:.2f} min.")
    print(f"Anomaly flags: {anomaly_flags if anomaly_flags else 'None'}")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
