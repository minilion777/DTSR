from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from _common import PACKAGE_ROOT, actor_matches_bundle, load_manifest, load_scenario, resolve_device, write_json
from dtsr_multiday_common import EP100_ACTOR_PATH, EP100_BUNDLE_PATH, REPAIR_MODE, RUNTIME_PIPELINE_ORDER, set_all_seeds, to_scalar_summary

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.defense import load_dae, load_detector
from evc.exp4_visualization import create_exp4_figures
from evc.long_horizon_attacks import build_long_horizon_attacker
from evc.merged_core import ChargingEnv, Critic, TRAIN_PROFILE, load_actor_critic_bundle, load_actor_from_path
from evc.offline_dae_det_temporal_shield import load_temporal_shield_bundle
from evc.ug_bcr import rollout_episode_with_ug_bcr
from evc.ug_bcr_v3 import load_ug_bcr_v3_config, rollout_episode_with_ug_bcr_v3


EXPECTED_PIPELINE_ORDER = "DAE/DET route -> UG-BCR belief+urgency gate -> Temporal Shield -> Actor"


STAGE_ATTACK = {
    "key": "attack",
    "display": "Attack-only",
    "route_mode": "none",
    "use_dae": False,
    "use_detector": False,
    "enable_shield": False,
    "enable_belief": False,
    "enable_urgency_gate": False,
}

STAGE_FULL_DTSR = {
    "key": "full_dtsr",
    "display": "Full DTSR",
    "route_mode": "detector",
    "use_dae": True,
    "use_detector": True,
    "enable_shield": True,
    "enable_belief": True,
    "enable_urgency_gate": True,
}

STAGES = (STAGE_ATTACK, STAGE_FULL_DTSR)

EXP4_CONDITIONS = [
    {"condition_key": "4A_deadline_K0", "section": "4A", "target": "Deadline Denial", "objective": "deadline", "knowledge": "K0", "scope": "local"},
    {"condition_key": "4A_deadline_K1", "section": "4A", "target": "Deadline Denial", "objective": "deadline", "knowledge": "K1", "scope": "local"},
    {"condition_key": "4A_deadline_K2", "section": "4A", "target": "Deadline Denial", "objective": "deadline", "knowledge": "K2", "scope": "local"},
    {"condition_key": "4A_deadline_K3", "section": "4A", "target": "Deadline Denial", "objective": "deadline", "knowledge": "K3", "scope": "local"},
    {"condition_key": "4A_deadline_K4", "section": "4A", "target": "Deadline Denial", "objective": "deadline", "knowledge": "K4", "scope": "local"},
    {"condition_key": "4B_economic_K4", "section": "4B", "target": "Economic Shift", "objective": "economic", "knowledge": "K4", "scope": "global"},
]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(path)


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    headers = [cell(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines) + "\n"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def existing_artifact_path(primary_dir: Path | None, fallback_dir: Path, filename: str) -> Path:
    candidates = []
    if primary_dir is not None:
        candidates.append(Path(primary_dir) / filename)
    candidates.append(Path(fallback_dir) / filename)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {filename}; checked: {[str(path) for path in candidates]}")


def load_price_threshold_from_path(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_split") not in {None, "val"}:
        raise RuntimeError("Price threshold must be calibrated on validation split.")
    if "price_threshold" in payload:
        return float(payload["price_threshold"])
    if "new_threshold" in payload:
        return float(payload["new_threshold"])
    raise KeyError(f"No price_threshold/new_threshold in {path}")


def load_price_threshold(dtsr_dir: Path) -> float:
    return load_price_threshold_from_path(dtsr_dir / "electhacker_c_price_threshold.json")


def stage_kwargs(stage_spec: dict[str, Any], *, dae, detector_model, detector_threshold: float, shield_config, ug_bcr_config) -> dict[str, Any]:
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
    out: dict[str, Any] = {"adaptive_candidate_count": int(len(frame))}
    for source, target in [
        ("score", "adaptive_mean_selected_score"),
        ("route", "adaptive_route_rate"),
        ("belief", "adaptive_belief_rate"),
        ("temporal", "adaptive_temporal_mean"),
        ("action", "adaptive_post_defense_action_mean"),
        ("budget_used", "adaptive_budget_used_mean"),
        ("budget_remaining", "adaptive_budget_remaining_mean"),
        ("predicted_return", "adaptive_predicted_return_mean"),
        ("predicted_cost", "adaptive_predicted_cost_mean"),
        ("predicted_final_soc", "adaptive_predicted_final_soc_mean"),
    ]:
        if source in frame.columns:
            values = pd.to_numeric(frame[source], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            out[target] = float(np.mean(values)) if values.size else float("nan")
    return out


def build_exp4_attacker(
    *,
    condition: dict[str, Any],
    actor,
    critic,
    device: torch.device,
    arrivals: pd.DataFrame,
    signal_path: Path,
    attack_seed: int,
    dae,
    detector_model,
    detector_threshold: float,
    shield_config,
    ug_bcr_config,
    ug_bcr_v3_config,
    args,
):
    env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
    max_duration = max(12, int(arrivals["Duration_of_stay"].max()))
    low, high = env.observation_bounds(max_duration_of_stay=max_duration)
    overrides = {
        "objective": condition["objective"],
        "knowledge_level": condition["knowledge"],
        "horizon": int(args.horizon),
        "samples": int(args.samples),
        "iterations": int(args.iterations),
        "elite_frac": float(args.elite_frac),
        "epsilon": float(args.epsilon),
        "temporal_eta": float(args.temporal_eta),
        "total_l1_budget": float(args.total_l1_budget),
        "discount": float(args.discount),
    }
    attacker = build_long_horizon_attacker(
        "module_aware_cem_mpc",
        actor=actor,
        device=device,
        obs_low=low,
        obs_high=high,
        critic=critic,
        seed=attack_seed,
        attack_state_scope=condition["scope"],
        attack_overrides=overrides,
    )
    attacker.configure_target_defense(
        defender=dae,
        detector_model=detector_model,
        detector_threshold=float(detector_threshold),
        shield_config=shield_config,
        ug_bcr_config=ug_bcr_config,
        ug_bcr_v3_config=ug_bcr_v3_config,
        reward_profile=TRAIN_PROFILE,
        signals_path=signal_path,
        device=device,
        actor=actor,
        repair_mode=REPAIR_MODE,
    )
    if not bool(getattr(attacker, "_target_ready", False)):
        raise RuntimeError("Module-aware CEM-MPC attacker target defense is not ready.")
    return attacker


def rollout_once(
    *,
    arrivals: pd.DataFrame,
    actor,
    critic,
    device: torch.device,
    signal_path: Path,
    condition: dict[str, Any],
    stage_spec: dict[str, Any],
    attack_seed: int,
    dae,
    detector_model,
    detector_threshold: float,
    shield_config,
    ug_bcr_config,
    ug_bcr_v3_config,
    price_threshold: float,
    args,
) -> tuple[dict[str, Any], float]:
    set_all_seeds(attack_seed)
    attacker = None
    if condition["condition_key"] != "clean":
        attacker = build_exp4_attacker(
            condition=condition,
            actor=actor,
            critic=critic,
            device=device,
            arrivals=arrivals,
            signal_path=signal_path,
            attack_seed=attack_seed,
            dae=dae,
            detector_model=detector_model,
            detector_threshold=detector_threshold,
            shield_config=shield_config,
            ug_bcr_config=ug_bcr_config,
            ug_bcr_v3_config=ug_bcr_v3_config,
            args=args,
        )
    kwargs = stage_kwargs(
        stage_spec,
        dae=dae,
        detector_model=detector_model,
        detector_threshold=detector_threshold,
        shield_config=shield_config,
        ug_bcr_config=ug_bcr_config,
    )
    rollout_function = rollout_episode_with_ug_bcr
    if ug_bcr_v3_config is not None and bool(stage_spec["enable_belief"]):
        rollout_function = rollout_episode_with_ug_bcr_v3
        kwargs.pop("ug_bcr_config", None)
        kwargs["ug_bcr_v3_config"] = ug_bcr_v3_config
    start = time.perf_counter()
    summary = rollout_function(
        arrivals,
        actor,
        signal_path,
        device,
        TRAIN_PROFILE,
        attack_enabled=condition["condition_key"] != "clean",
        attack_scenario="O",
        attacker=attacker,
        epsilon=float(args.epsilon),
        state_scope=condition.get("scope", "all"),
        price_threshold=float(price_threshold),
        attack_ratio=float(args.attack_ratio),
        attack_scope=str(args.attack_scope),
        label=f"{condition['condition_key']}__{stage_spec['key']}",
        repair_mode=REPAIR_MODE,
        **kwargs,
    )
    elapsed = float(time.perf_counter() - start)
    row = {**to_scalar_summary(summary), **adaptive_diagnostics(attacker)}
    return row, elapsed


def clean_condition() -> dict[str, Any]:
    return {"condition_key": "clean", "section": "baseline", "target": "Clean", "objective": "clean", "knowledge": "clean", "scope": "all"}


def select_restart_per_condition(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Select restarts within the same condition only.

    Knowledge-level rows remain single-K summaries.  This intentionally does
    not perform cumulative best-of aggregation across K0..K4.
    """
    attack_df = raw_df[raw_df["condition_key"] != "clean"].copy()
    clean_df = raw_df[raw_df["condition_key"] == "clean"].copy()
    if attack_df.empty:
        return attack_df
    clean_pivot = clean_df.pivot_table(index="scenario_id", columns="stage_key", values=["ep_reward", "ep_r1", "exit_vio", "done_cnt"], aggfunc="first")
    selected_keys: list[tuple[str, str, int]] = []
    for (scenario_id, condition_key), group in attack_df.groupby(["scenario_id", "condition_key"], sort=False):
        pivot = group.pivot_table(index="restart", columns="stage_key", values=["ep_reward", "ep_r1", "exit_vio", "done_cnt", "mean_fin_soc"], aggfunc="first")
        target = str(group["objective"].iloc[0])
        rows: list[dict[str, Any]] = []
        for restart in pivot.index:
            item = {"restart": int(restart)}
            for metric in ["ep_reward", "ep_r1", "exit_vio", "done_cnt", "mean_fin_soc"]:
                for stage in ["attack", "full_dtsr"]:
                    item[f"{metric}_{stage}"] = float(pivot.loc[restart, (metric, stage)]) if (metric, stage) in pivot.columns else float("nan")
            rows.append(item)
        rank = pd.DataFrame(rows)
        if rank.empty:
            continue
        if target == "economic":
            clean_exit = float(clean_pivot.loc[scenario_id, ("exit_vio", "full_dtsr")]) if (scenario_id in clean_pivot.index and ("exit_vio", "full_dtsr") in clean_pivot.columns) else 0.0
            rank["soc_valid"] = rank["exit_vio_full_dtsr"].astype(float) <= clean_exit
            valid = rank[rank["soc_valid"]].copy()
            if valid.empty:
                valid = rank.sort_values(["exit_vio_full_dtsr", "ep_r1_full_dtsr"], ascending=[True, False]).head(1)
            else:
                valid = valid.sort_values(["ep_r1_full_dtsr", "ep_reward_full_dtsr"], ascending=[False, True]).head(1)
            chosen = int(valid["restart"].iloc[0])
        else:
            rank = rank.sort_values(["exit_vio_full_dtsr", "ep_reward_full_dtsr"], ascending=[False, True])
            chosen = int(rank["restart"].iloc[0])
        selected_keys.append((str(scenario_id), str(condition_key), chosen))
    key_frame = pd.DataFrame(selected_keys, columns=["scenario_id", "condition_key", "restart"])
    return attack_df.merge(key_frame, on=["scenario_id", "condition_key", "restart"], how="inner")


def create_tables(raw_df: pd.DataFrame, output_dir: Path, seed: int) -> None:
    if raw_df.empty:
        return
    selected = select_restart_per_condition(raw_df)
    atomic_csv(raw_df.sort_values(["episode_index", "condition_key", "restart", "stage_key"]), output_dir / "tables" / "exp4_all_restarts.csv")
    atomic_csv(selected.sort_values(["episode_index", "condition_key", "stage_key"]), output_dir / "tables" / "exp4_selected_restarts.csv")
    clean = raw_df[raw_df["condition_key"] == "clean"].copy()
    clean_attack = clean[clean["stage_key"] == "attack"][["scenario_id", "ep_reward", "ep_r1", "exit_vio", "done_cnt"]].rename(
        columns={"ep_reward": "clean_attack_reward", "ep_r1": "clean_attack_cost", "exit_vio": "clean_attack_exit_vio", "done_cnt": "clean_done_cnt"}
    )
    clean_full = clean[clean["stage_key"] == "full_dtsr"][["scenario_id", "ep_reward", "ep_r1", "exit_vio", "done_cnt"]].rename(
        columns={"ep_reward": "clean_full_reward", "ep_r1": "clean_full_cost", "exit_vio": "clean_full_exit_vio", "done_cnt": "clean_full_done_cnt"}
    )
    clean_base = clean_attack.merge(clean_full, on="scenario_id", how="outer")
    rows: list[dict[str, Any]] = []
    for condition_key, group in selected.groupby("condition_key", sort=False):
        attack_rows = group[group["stage_key"] == "attack"].copy()
        full_rows = group[group["stage_key"] == "full_dtsr"].copy()
        merged = attack_rows[["scenario_id", "ep_reward", "ep_r1", "exit_vio", "done_cnt", "target", "knowledge", "objective"]].rename(
            columns={"ep_reward": "attack_reward", "ep_r1": "attack_cost", "exit_vio": "attack_exit_vio", "done_cnt": "attack_done_cnt"}
        ).merge(
            full_rows[["scenario_id", "ep_reward", "ep_r1", "exit_vio", "done_cnt", "mean_fin_soc", "route_rate", "shield_correction_mean", "urgency_gate_belief_rate"]].rename(
                columns={"ep_reward": "full_reward", "ep_r1": "full_cost", "exit_vio": "full_exit_vio", "done_cnt": "full_done_cnt"}
            ),
            on="scenario_id",
            how="inner",
        ).merge(clean_base, on="scenario_id", how="left")
        if merged.empty:
            continue
        recovery = []
        for _, item in merged.iterrows():
            denom = float(item["clean_attack_reward"]) - float(item["attack_reward"])
            recovery.append(float("nan") if abs(denom) <= 1e-8 else 100.0 * (float(item["full_reward"]) - float(item["attack_reward"])) / denom)
        objective = str(merged["objective"].iloc[0])
        if objective == "economic":
            harm_values = merged["full_cost"].astype(float).to_numpy() - merged["clean_full_cost"].astype(float).to_numpy()
            clean_cost = np.maximum(np.abs(merged["clean_full_cost"].astype(float).to_numpy()), 1e-8)
            harm_pct = 100.0 * harm_values / clean_cost
            harm_text = f"{np.nanmean(harm_values):.4f} cost ({np.nanmean(harm_pct):.1f}%)"
        else:
            done = np.maximum(merged["full_done_cnt"].astype(float).to_numpy(), 1.0)
            harm_values = 100.0 * merged["full_exit_vio"].astype(float).to_numpy() / done
            harm_text = f"{np.nanmean(harm_values):.1f}% exit violation"
        rows.append(
            {
                "Target": str(merged["target"].iloc[0]),
                "Knowledge": str(merged["knowledge"].iloc[0]),
                "Attack-only Return": float(np.nanmean(merged["attack_reward"].astype(float).to_numpy())),
                "Full DTSR Return": float(np.nanmean(merged["full_reward"].astype(float).to_numpy())),
                "Recovery/%": float(np.nanmean(np.asarray(recovery, dtype=float))),
                "Target Harm": harm_text,
                "Scenario Count": int(merged["scenario_id"].nunique()),
            }
        )
    main = pd.DataFrame(rows)
    order = {condition["knowledge"]: idx for idx, condition in enumerate(EXP4_CONDITIONS)}
    if not main.empty:
        main["_order"] = main["Knowledge"].map(order).fillna(99)
        main = main.sort_values(["Target", "_order"]).drop(columns=["_order"])
    atomic_csv(main, output_dir / "tables" / "table_exp4_main.csv")
    mechanism_rows = []
    for condition_key, group in selected[selected["stage_key"] == "full_dtsr"].groupby("condition_key", sort=False):
        mechanism_rows.append(
            {
                "condition_key": condition_key,
                "Target": str(group["target"].iloc[0]),
                "Knowledge": str(group["knowledge"].iloc[0]),
                "route_rate_mean": float(np.nanmean(group["route_rate"].astype(float).to_numpy())),
                "ug_bcr_belief_rate_mean": float(np.nanmean(group.get("urgency_gate_belief_rate", pd.Series(dtype=float)).astype(float).to_numpy())),
                "shield_correction_mean": float(np.nanmean(group.get("shield_correction_mean", pd.Series(dtype=float)).astype(float).to_numpy())),
                "adaptive_budget_used_mean": float(np.nanmean(group.get("adaptive_budget_used_mean", pd.Series(dtype=float)).astype(float).to_numpy())),
                "adaptive_post_defense_action_mean": float(np.nanmean(group.get("adaptive_post_defense_action_mean", pd.Series(dtype=float)).astype(float).to_numpy())),
            }
        )
    atomic_csv(pd.DataFrame(mechanism_rows), output_dir / "tables" / "table_exp4_mechanisms.csv")
    if not main.empty:
        (output_dir / "tables" / "table_exp4_main.md").write_text(markdown_table(main), encoding="utf-8")
    report = {
        "seed": int(seed),
        "raw_rows": int(len(raw_df)),
        "selected_rows": int(len(selected)),
        "conditions": [condition["condition_key"] for condition in EXP4_CONDITIONS],
        "knowledge_summary_mode": "single_knowledge_level",
        "cumulative_best_of_across_knowledge": False,
    }
    write_json(output_dir / "summary" / "exp4_table_manifest.json", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 4: module-aware adaptive long-horizon CEM-MPC attacks.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor-path", type=Path, default=EP100_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=EP100_BUNDLE_PATH)
    parser.add_argument("--dtsr-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dtsr_seed42_multiday")
    parser.add_argument("--dae-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday")
    parser.add_argument("--detector-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "det_seed42_fullstate")
    parser.add_argument("--shield-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "shield_seed42_fullstate")
    parser.add_argument("--ug-bcr-config-path", type=Path, default=PACKAGE_ROOT / "artifacts" / "ug_bcr_seed42_fullstate" / "ug_bcr_config.json")
    parser.add_argument(
        "--ug-bcr-v3-config-path",
        type=Path,
        default=None,
        help="Optional UG-BCR-v3 config used by both Full DTSR and the K3/K4 attacker shadow pipeline.",
    )
    parser.add_argument("--price-threshold-file", type=Path, default=PACKAGE_ROOT / "results" / "attack120_short_horizon" / "ehc_threshold_fix" / "electhacker_c_price_threshold.json")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--elite-frac", type=float, default=0.125)
    parser.add_argument("--epsilon", type=float, default=0.075)
    parser.add_argument("--temporal-eta", type=float, default=0.045)
    parser.add_argument("--total-l1-budget", type=float, default=1.20)
    parser.add_argument("--discount", type=float, default=0.97)
    parser.add_argument("--attack-ratio", type=float, default=1.0)
    parser.add_argument("--attack-scope", choices=["obs", "vehicle", "window"], default="obs")
    parser.add_argument("--conditions", default=",".join(condition["condition_key"] for condition in EXP4_CONDITIONS))
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "results" / "exp4_module_aware_long_horizon_seed42")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Load artifacts and write config without running rollouts.")
    parser.add_argument("--max-rollouts", type=int, default=0, help="Debug only: stop after N new rollouts; 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if RUNTIME_PIPELINE_ORDER != EXPECTED_PIPELINE_ORDER:
        raise RuntimeError(
            f"Pipeline contract mismatch: expected {EXPECTED_PIPELINE_ORDER!r}, got {RUNTIME_PIPELINE_ORDER!r}."
        )
    if int(args.seed) != 42:
        raise ValueError("Experiment 4 is locked to the single DTSR training seed seed=42.")
    if int(args.scenes) <= 0 or int(args.restarts) <= 0:
        raise ValueError("--scenes and --restarts must be positive.")
    if args.split == "val" and int(args.scenes) > 60:
        raise ValueError("Validation split contains only 60 scenarios.")
    if not math.isclose(float(args.attack_ratio), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Experiment 4 uses fixed attack_ratio=1.0 for budget-controlled comparison.")
    condition_lookup = {condition["condition_key"]: condition for condition in EXP4_CONDITIONS}
    requested_conditions = [token.strip() for token in str(args.conditions).split(",") if token.strip()]
    unknown = [key for key in requested_conditions if key not in condition_lookup]
    if unknown:
        raise ValueError(f"Unknown Experiment 4 conditions: {unknown}")
    selected_conditions = [condition_lookup[key] for key in requested_conditions]
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    dae_path = existing_artifact_path(args.dae_artifact_dir, args.dtsr_dir, "dtsr_dae.pt")
    dae = load_dae(dae_path, device).eval()
    detector_path = existing_artifact_path(args.detector_artifact_dir, args.dtsr_dir, "dtsr_detector.pt")
    detector_artifact = load_detector(detector_path, device)
    detector_model = detector_artifact.model
    detector_threshold = float(detector_artifact.threshold)
    shield_dir = args.shield_artifact_dir if args.shield_artifact_dir is not None else args.dtsr_dir
    shield_path = existing_artifact_path(shield_dir, args.dtsr_dir, "dtsr_temporal_shield.pt")
    shield_config = load_temporal_shield_bundle(shield_path).config
    ug_bcr_path = args.ug_bcr_config_path if args.ug_bcr_config_path is not None else args.dtsr_dir / "ug_bcr_config.json"
    if not ug_bcr_path.exists():
        raise FileNotFoundError(f"Missing UG-BCR config: {ug_bcr_path}")
    from dtsr_multiday_common import load_ug_bcr_config

    ug_bcr_config = load_ug_bcr_config(ug_bcr_path)
    ug_bcr_v3_config = None
    if args.ug_bcr_v3_config_path is not None:
        if not args.ug_bcr_v3_config_path.exists():
            raise FileNotFoundError(f"Missing UG-BCR-v3 config: {args.ug_bcr_v3_config_path}")
        ug_bcr_v3_config = load_ug_bcr_v3_config(args.ug_bcr_v3_config_path)
        ug_bcr_config = ug_bcr_v3_config.base_v2
    price_threshold = load_price_threshold_from_path(args.price_threshold_file) if args.price_threshold_file.exists() else load_price_threshold(args.dtsr_dir)

    manifest = load_manifest(args.split).sort_values("Scenario_ID", kind="mergesort").reset_index(drop=True)
    if len(manifest) < int(args.scenes):
        raise RuntimeError(f"Requested {args.scenes} {args.split} scenarios, found only {len(manifest)}.")
    manifest = manifest.iloc[: int(args.scenes)].copy().reset_index(drop=True)
    scenario_order = manifest[["Scenario_ID", "Vehicle_File", "Signal_File", "Context_File"]].copy()
    scenario_order.insert(0, "episode_index", np.arange(1, len(manifest) + 1, dtype=int))
    atomic_csv(scenario_order, args.output_dir / "scenario_order.csv")

    run_config = {
        "seed": int(args.seed),
        "device": str(device),
        "split": str(args.split),
        "scenes": int(args.scenes),
        "restarts": int(args.restarts),
        "conditions": [condition["condition_key"] for condition in selected_conditions],
        "stages": [stage["key"] for stage in STAGES],
        "attacker": "module_aware_cem_mpc",
        "horizon": int(args.horizon),
        "samples": int(args.samples),
        "iterations": int(args.iterations),
        "elite_frac": float(args.elite_frac),
        "epsilon": float(args.epsilon),
        "temporal_eta": float(args.temporal_eta),
        "total_l1_budget": float(args.total_l1_budget),
        "discount": float(args.discount),
        "attack_ratio": float(args.attack_ratio),
        "repair_mode": REPAIR_MODE,
        "runtime_pipeline_order": RUNTIME_PIPELINE_ORDER,
        "dae_artifact": str(dae_path),
        "detector_artifact": str(detector_path),
        "detector_threshold": float(detector_threshold),
        "shield_artifact": str(shield_path),
        "ug_bcr_config": str(ug_bcr_path),
        "ug_bcr_v3_config": None if args.ug_bcr_v3_config_path is None else str(args.ug_bcr_v3_config_path),
        "ug_bcr_v3_config_sha256": (
            None if args.ug_bcr_v3_config_path is None else sha256_file(args.ug_bcr_v3_config_path)
        ),
        "ug_bcr_version": 3 if ug_bcr_v3_config is not None else 2,
        "attacker_shadow_pipeline_order": EXPECTED_PIPELINE_ORDER,
        "price_threshold": float(price_threshold),
        "knowledge_summary_mode": "single_knowledge_level",
        "cumulative_best_of_across_knowledge": False,
        "restart_selection": "Within each condition only: Deadline uses max Full-DTSR exit violations then min return; Economic uses max Full-DTSR cost subject to clean-level exit violations.",
    }
    write_json(args.output_dir / "run_config.json", run_config)
    write_json(
        args.output_dir / "environment_info.json",
        {"python": sys.version, "torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "device": str(device)},
    )
    if args.preflight:
        print(f"Preflight OK. Wrote {args.output_dir / 'run_config.json'}")
        return

    raw_path = args.output_dir / "intermediate" / "exp4_rollouts.jsonl"
    existing_rows = load_jsonl(raw_path) if args.resume else []
    completed = {
        (str(row["scenario_id"]), str(row["condition_key"]), int(row.get("restart", 0)), str(row["stage_key"]))
        for row in existing_rows
    }
    clean = clean_condition()
    expected_jobs = []
    for episode_index, (_, scenario_row) in enumerate(manifest.iterrows(), start=1):
        scenario_id = str(scenario_row["Scenario_ID"])
        for stage in STAGES:
            expected_jobs.append((episode_index, scenario_row, clean, 0, stage, scenario_id))
        for condition in selected_conditions:
            for restart in range(int(args.restarts)):
                for stage in STAGES:
                    expected_jobs.append((episode_index, scenario_row, condition, restart, stage, scenario_id))
    new_rollouts = 0
    total = len(expected_jobs)
    for job_index, (episode_index, scenario_row, condition, restart, stage, scenario_id) in enumerate(expected_jobs, start=1):
        key = (scenario_id, condition["condition_key"], int(restart), stage["key"])
        if key in completed:
            continue
        arrivals, signal_path, _ = load_scenario(scenario_row)
        attack_seed = int(args.seed + episode_index * 10_000 + restart * 100 + sum(ord(ch) for ch in condition["condition_key"]))
        row, runtime_seconds = rollout_once(
            arrivals=arrivals,
            actor=actor,
            critic=critic,
            device=device,
            signal_path=signal_path,
            condition=condition,
            stage_spec=stage,
            attack_seed=attack_seed,
            dae=dae,
            detector_model=detector_model,
            detector_threshold=detector_threshold,
            shield_config=shield_config,
            ug_bcr_config=ug_bcr_config,
            ug_bcr_v3_config=ug_bcr_v3_config,
            price_threshold=price_threshold,
            args=args,
        )
        full_row = {
            "scenario_id": scenario_id,
            "episode_index": int(episode_index),
            "seed": int(args.seed),
            "attack_seed": int(attack_seed),
            "restart": int(restart),
            "condition_key": condition["condition_key"],
            "section": condition["section"],
            "target": condition["target"],
            "objective": condition["objective"],
            "knowledge": condition["knowledge"],
            "attack_state_scope": condition["scope"],
            "stage_key": stage["key"],
            "stage_display_name": stage["display"],
            "runtime_seconds": float(runtime_seconds),
            **row,
        }
        if int(full_row.get("done_cnt", -1)) != 344:
            raise RuntimeError(f"Incomplete rollout for {key}: done_cnt={full_row.get('done_cnt')}")
        append_jsonl(raw_path, full_row)
        completed.add(key)
        new_rollouts += 1
        print(
            f"[{job_index:04d}/{total}] ep={episode_index:03d} {condition['condition_key']} "
            f"r={restart} {stage['key']} reward={float(full_row['ep_reward']):.3f} "
            f"exit={int(full_row.get('exit_vio', 0))} time={runtime_seconds:.2f}s"
        )
        if args.max_rollouts > 0 and new_rollouts >= int(args.max_rollouts):
            print(f"Debug stop after {new_rollouts} new rollouts.")
            break
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    raw_rows = load_jsonl(raw_path)
    raw_df = pd.DataFrame(raw_rows)
    create_tables(raw_df, args.output_dir, int(args.seed))
    create_exp4_figures(raw_path, args.output_dir / "figures", seed=int(args.seed))
    print(f"Experiment 4 tables written under {args.output_dir / 'tables'}")
    print(f"Experiment 4 figures written under {args.output_dir / 'figures'}")


if __name__ == "__main__":
    main()
