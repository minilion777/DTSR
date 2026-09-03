from __future__ import annotations

import argparse
import importlib.util
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

from _common import (  # noqa: E402
    actor_matches_bundle,
    load_manifest,
    load_scenario,
    resolve_device,
    write_json,
)
from dtsr_multiday_common import (  # noqa: E402
    EP100_ACTOR_PATH,
    EP100_BUNDLE_PATH,
    REPAIR_MODE,
    RUNTIME_PIPELINE_ORDER,
    load_ug_bcr_config,
    set_all_seeds,
    to_scalar_summary,
)
from evc.defense import load_dae, load_detector  # noqa: E402
from evc.merged_attacks import attack_batch_by_context  # noqa: E402
from evc.merged_core import (  # noqa: E402
    ChargingEnv,
    Critic,
    TRAIN_PROFILE,
    load_actor_critic_bundle,
    load_actor_from_path,
    to_numpy_1d,
)
from evc.merged_pipeline import _build_contexts, summarize_metrics  # noqa: E402
from evc.offline_dae_det_temporal_shield import LOCAL_SHIELD_INDICES, load_temporal_shield_bundle  # noqa: E402
from evc.sequential_adversary import update_active_vehicle_ids  # noqa: E402


def _load_table_module():
    path = SCRIPT_DIR / "_strength_eval_common.py"
    spec = importlib.util.spec_from_file_location("dtsr_table_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load table evaluation module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TABLE_EVAL = _load_table_module()
ATTACK_SPECS: list[dict[str, Any]] = TABLE_EVAL.ATTACK_SPECS
parse_key_list = TABLE_EVAL.parse_key_list
build_attacker_for_rollout = TABLE_EVAL.build_attacker_for_rollout
existing_artifact_path = TABLE_EVAL.existing_artifact_path
load_price_threshold = TABLE_EVAL.load_price_threshold
load_price_threshold_from_path = TABLE_EVAL.load_price_threshold_from_path
format_mean_std = TABLE_EVAL.format_mean_std
markdown_table = TABLE_EVAL.markdown_table

ATTACK_TYPE_BY_KEY = {
    "clean": "Clean",
    "opposite_pgd": "short",
    "q_function": "short",
    "opposite_fgsm": "short",
    "electhacker_c": "task",
    "electhacker_f": "task",
    "electhacker_o": "task",
    "local_small_drift_q": "long",
    "local_deadline_drift_pgd": "long",
    "full_pipeline_adaptive_deadline": "adaptive-long",
}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return out


def sample_std(values: pd.Series | np.ndarray | list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def direction_features(
    delta: np.ndarray,
    prev_delta: np.ndarray,
    *,
    dimensions: list[int],
    eps: float = 1e-8,
) -> tuple[float, float, float]:
    cur = np.asarray(delta, dtype=np.float32).reshape(-1)[dimensions]
    prev = np.asarray(prev_delta, dtype=np.float32).reshape(-1)[dimensions]
    adjacent_linf = float(np.max(np.abs(cur - prev)))
    active = (np.abs(cur) > eps) & (np.abs(prev) > eps)
    if bool(np.any(active)):
        sign_ratio = float(np.mean(np.sign(cur[active]) == np.sign(prev[active])))
    else:
        sign_ratio = float("nan")
    denom = float(np.linalg.norm(cur) * np.linalg.norm(prev))
    cosine = float(np.dot(cur, prev) / denom) if denom > eps else float("nan")
    return adjacent_linf, sign_ratio, cosine


def rollout_attack_temporal_profile(
    arrivals: pd.DataFrame,
    actor,
    signal_path: Path,
    device: torch.device,
    *,
    attack_enabled: bool,
    attack_scenario: str,
    attacker,
    state_scope: str,
    price_threshold: float,
    attack_ratio: float,
    attack_scope: str,
) -> dict[str, Any]:
    env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
    env.reset()
    actor = actor.to(device).eval()
    idx = 0
    active = []
    active_vehicle_ids: list[int] = []
    attack_obs_count = 0
    route_total = 0
    delta_linf: list[float] = []
    delta_l2: list[float] = []
    local_linf: list[float] = []
    local_l2: list[float] = []
    adjacent_linf: list[float] = []
    local_adjacent_linf: list[float] = []
    sign_ratios: list[float] = []
    local_sign_ratios: list[float] = []
    cosine_values: list[float] = []
    local_cosine_values: list[float] = []
    same_direction_pairs = 0
    comparable_pairs = 0
    prev_delta_by_vehicle: dict[int, np.ndarray] = {}
    dims_all = list(range(11))
    dims_local = list(LOCAL_SHIELD_INDICES)
    direction_dims = dims_local if str(state_scope) == "local" else dims_all

    def compute_actions(states: list[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.as_tensor(np.asarray(states, dtype=np.float32), dtype=torch.float32, device=device)
            return actor(state_t).detach().cpu().numpy()

    def record(clean_states: list[np.ndarray], observed_states: list[np.ndarray], flags: list[bool], vehicle_ids: list[int]) -> None:
        nonlocal attack_obs_count, route_total, same_direction_pairs, comparable_pairs
        route_total += len(flags)
        for clean_state, observed_state, flag, vehicle_id in zip(clean_states, observed_states, flags, vehicle_ids):
            if not bool(flag):
                continue
            attack_obs_count += 1
            clean_vec = to_numpy_1d(clean_state)
            observed_vec = to_numpy_1d(observed_state)
            delta = observed_vec - clean_vec
            local_delta = delta[dims_local]
            delta_linf.append(float(np.max(np.abs(delta))))
            delta_l2.append(float(np.linalg.norm(delta, ord=2)))
            local_linf.append(float(np.max(np.abs(local_delta))))
            local_l2.append(float(np.linalg.norm(local_delta, ord=2)))
            prev = prev_delta_by_vehicle.get(int(vehicle_id))
            if prev is not None:
                adj, sign_ratio, cosine = direction_features(delta, prev, dimensions=direction_dims)
                local_adj, local_sign, local_cos = direction_features(delta, prev, dimensions=dims_local)
                adjacent_linf.append(adj)
                local_adjacent_linf.append(local_adj)
                if np.isfinite(sign_ratio):
                    sign_ratios.append(sign_ratio)
                if np.isfinite(local_sign):
                    local_sign_ratios.append(local_sign)
                if np.isfinite(cosine):
                    cosine_values.append(cosine)
                    comparable_pairs += 1
                    same_direction_pairs += int(cosine > 0.0)
                if np.isfinite(local_cos):
                    local_cosine_values.append(local_cos)
            prev_delta_by_vehicle[int(vehicle_id)] = delta.astype(np.float32)

    while env.t < env.horizon:
        new_states: list[np.ndarray] = []
        new_stations: list[int] = []
        new_vehicle_ids: list[int] = []
        while idx < len(arrivals) and int(arrivals.loc[idx, "Arrive_time"]) == env.t:
            new_states.append(env.build_initial_obs(int(arrivals.loc[idx, "Duration_of_stay"])))
            new_stations.append(int(arrivals.loc[idx, "Station"]))
            new_vehicle_ids.append(int(idx))
            idx += 1

        if new_states:
            contexts = _build_contexts(env, new_states, new_stations, attack_scenario, True, price_threshold, 0.5, 0.3, 1.0, -0.5)
            attacked_states, flags = attack_batch_by_context(
                attacker if attack_enabled else None,
                new_states,
                contexts,
                attack_ratio=attack_ratio,
                attack_scope=attack_scope,
                vehicle_ids=new_vehicle_ids,
                episode_index=0,
                seed=42 if attacker is None else int(getattr(attacker, "seed", 42)),
            )
            observed = attacked_states if attack_enabled else [to_numpy_1d(x) for x in new_states]
            record(new_states, observed, flags, new_vehicle_ids)
            actions = compute_actions(observed)
            for clean_obs, action, station in zip(new_states, actions, new_stations):
                env.enqueue(clean_obs, action, station)

        if active:
            active_states = [item.obs for item in active]
            active_stations = [item.station for item in active]
            contexts = _build_contexts(env, active_states, active_stations, attack_scenario, False, price_threshold, 0.5, 0.3, 1.0, -0.5)
            attacked_states, flags = attack_batch_by_context(
                attacker if attack_enabled else None,
                active_states,
                contexts,
                attack_ratio=attack_ratio,
                attack_scope=attack_scope,
                vehicle_ids=active_vehicle_ids,
                episode_index=0,
                seed=42 if attacker is None else int(getattr(attacker, "seed", 42)),
            )
            observed = attacked_states if attack_enabled else [to_numpy_1d(x) for x in active_states]
            record(active_states, observed, flags, active_vehicle_ids)
            actions = compute_actions(observed)
            for item, action in zip(active, actions):
                env.enqueue(item.obs, action, item.station)

        step_vehicle_ids = new_vehicle_ids + active_vehicle_ids
        transitions, next_active, _ = env.step()
        active = next_active
        active_vehicle_ids = update_active_vehicle_ids(step_vehicle_ids, transitions)

    summary = summarize_metrics(env.metrics, "attack_temporal_profile" if attack_enabled else "clean_temporal_profile")
    summary["route_count"] = 0
    summary["route_total"] = int(route_total)
    summary["route_rate"] = 0.0
    summary["attack_obs_count"] = int(attack_obs_count)
    summary["attack_obs_rate"] = 0.0 if route_total <= 0 else float(attack_obs_count / route_total)
    summary["attack_delta_count"] = int(len(delta_linf))
    summary["attack_delta_linf_mean"] = mean_or_nan(delta_linf)
    summary["attack_delta_l2_mean"] = mean_or_nan(delta_l2)
    summary["attack_delta_local_linf_mean"] = mean_or_nan(local_linf)
    summary["attack_delta_local_l2_mean"] = mean_or_nan(local_l2)
    summary["temporal_pair_count"] = int(len(adjacent_linf))
    summary["temporal_adjacent_linf_mean"] = mean_or_nan(adjacent_linf)
    summary["temporal_local_adjacent_linf_mean"] = mean_or_nan(local_adjacent_linf)
    summary["temporal_direction_ratio_mean"] = mean_or_nan(sign_ratios)
    summary["temporal_local_direction_ratio_mean"] = mean_or_nan(local_sign_ratios)
    summary["temporal_cosine_mean"] = mean_or_nan(cosine_values)
    summary["temporal_local_cosine_mean"] = mean_or_nan(local_cosine_values)
    summary["temporal_same_direction_pair_rate"] = (
        float(same_direction_pairs / comparable_pairs) if comparable_pairs > 0 else float("nan")
    )
    return summary


def create_summary(long_df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, str]] = []
    for attack_key, subset in long_df.groupby("attack_key", sort=False):
        display = str(subset["attack_display_name"].iloc[0])
        attack_type = str(subset["attack_temporal_type"].iloc[0])
        row: dict[str, Any] = {
            "attack_key": attack_key,
            "attack_display_name": display,
            "attack_temporal_type": attack_type,
            "scenario_count": int(len(subset)),
        }
        for col in [
            "ep_reward",
            "run_vio",
            "exit_vio",
            "attack_delta_linf_mean",
            "attack_delta_local_linf_mean",
            "temporal_adjacent_linf_mean",
            "temporal_local_adjacent_linf_mean",
            "temporal_direction_ratio_mean",
            "temporal_local_direction_ratio_mean",
            "temporal_same_direction_pair_rate",
            "attack_obs_rate",
        ]:
            vals = pd.to_numeric(subset[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean(skipna=True))
            row[f"{col}_std"] = sample_std(vals.to_numpy(dtype=np.float64))
        row["temporal_pair_count_sum"] = int(pd.to_numeric(subset["temporal_pair_count"], errors="coerce").fillna(0).sum())
        rows.append(row)

        paper_rows.append(
            {
                "场景": display,
                "类型": attack_type,
                "平均幅度": format_mean_std(row["attack_delta_linf_mean_mean"], row["attack_delta_linf_mean_std"], 3),
                "局部幅度": format_mean_std(row["attack_delta_local_linf_mean_mean"], row["attack_delta_local_linf_mean_std"], 3),
                "相邻变化": format_mean_std(row["temporal_adjacent_linf_mean_mean"], row["temporal_adjacent_linf_mean_std"], 3),
                "局部相邻变化": format_mean_std(row["temporal_local_adjacent_linf_mean_mean"], row["temporal_local_adjacent_linf_mean_std"], 3),
                "方向比": format_mean_std(row["temporal_direction_ratio_mean_mean"], row["temporal_direction_ratio_mean_std"], 3),
                "Reward": format_mean_std(row["ep_reward_mean"], row["ep_reward_std"], 1),
                "N_run": format_mean_std(row["run_vio_mean"], row["run_vio_std"], 1),
                "N_exit": format_mean_std(row["exit_vio_mean"], row["exit_vio_std"], 1),
            }
        )
    summary = pd.DataFrame(rows)
    paper = pd.DataFrame(paper_rows)
    atomic_csv(summary, output_dir / "tables" / "attack_temporal_feature_summary_raw.csv")
    atomic_csv(paper, output_dir / "tables" / "attack_temporal_feature_table_paper.csv")
    return summary, paper


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile attack temporal features for the seed=42 full-state DTSR experiments.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor-path", type=Path, default=EP100_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=EP100_BUNDLE_PATH)
    parser.add_argument("--dtsr-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dtsr_seed42_multiday")
    parser.add_argument("--dae-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday")
    parser.add_argument("--detector-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "det_seed42_fullstate")
    parser.add_argument("--shield-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "shield_seed42_fullstate")
    parser.add_argument("--ug-bcr-config-path", type=Path, default=PACKAGE_ROOT / "artifacts" / "ug_bcr_seed42_fullstate" / "ug_bcr_config.json")
    parser.add_argument("--price-threshold-file", type=Path, default=PACKAGE_ROOT / "results" / "attack120_short_horizon" / "ehc_threshold_fix" / "electhacker_c_price_threshold.json")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--scenes", type=int, default=20)
    parser.add_argument("--attack-keys", default=",".join(spec["key"] for spec in ATTACK_SPECS))
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--attack-ratio", type=float, default=1.0)
    parser.add_argument("--attack-scope", choices=["obs", "vehicle", "window"], default="obs")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "results" / "attack_temporal_profile_20scenes_seed42")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rollouts", type=int, default=0)
    args = parser.parse_args()

    if int(args.seed) != 42:
        raise ValueError("This profiling run is fixed to seed=42.")
    if int(args.scenes) <= 0:
        raise ValueError("--scenes must be positive.")
    if not math.isclose(float(args.epsilon), 0.1, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Profile table is locked to epsilon=0.1 for short-horizon attacks.")

    requested_attack_keys = parse_key_list(args.attack_keys)
    if "clean" not in requested_attack_keys:
        requested_attack_keys.insert(0, "clean")
    requested_attack_keys = list(dict.fromkeys(requested_attack_keys))
    attack_lookup = {spec["key"]: spec for spec in ATTACK_SPECS}
    unknown = [key for key in requested_attack_keys if key not in attack_lookup]
    if unknown:
        raise ValueError(f"Unknown attack keys: {unknown}")
    selected_attack_specs = [attack_lookup[key] for key in requested_attack_keys]

    if args.overwrite and args.output_dir.exists():
        import shutil

        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)
    actor = load_actor_from_path(args.actor_path, device).eval()
    bundle_payload = load_actor_critic_bundle(args.bundle_path, device)
    if not actor_matches_bundle(actor, bundle_payload):
        raise RuntimeError("Selected actor does not match bundle actor_state_dict.")
    critic_state = bundle_payload.get("critic_state_dict")
    if critic_state is None:
        raise RuntimeError("The selected bundle has no critic_state_dict.")
    critic = Critic().to(device)
    critic.load_state_dict(critic_state)
    actor.eval()
    critic.eval()
    for module in (actor, critic):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    adaptive_selected = any(spec["key"] == "full_pipeline_adaptive_deadline" for spec in selected_attack_specs)
    dae = load_dae(existing_artifact_path(args.dae_artifact_dir, args.dtsr_dir, "dtsr_dae.pt"), device) if adaptive_selected else None
    detector_model = None
    detector_threshold = float("nan")
    shield_config = None
    ug_bcr_config = None
    if adaptive_selected:
        detector_artifact = load_detector(existing_artifact_path(args.detector_artifact_dir, args.dtsr_dir, "dtsr_detector.pt"), device)
        detector_model = detector_artifact.model
        detector_threshold = float(detector_artifact.threshold)
        shield_config = load_temporal_shield_bundle(existing_artifact_path(args.shield_artifact_dir, args.dtsr_dir, "dtsr_temporal_shield.pt")).config
        if not args.ug_bcr_config_path.exists():
            raise FileNotFoundError(f"Missing UG-BCR config: {args.ug_bcr_config_path}")
        ug_bcr_config = load_ug_bcr_config(args.ug_bcr_config_path)

    if args.price_threshold_file.exists():
        price_threshold = load_price_threshold_from_path(args.price_threshold_file)
    else:
        price_threshold = load_price_threshold(args.dtsr_dir)

    manifest = load_manifest(args.split).sort_values("Scenario_ID", kind="mergesort").reset_index(drop=True)
    if len(manifest) < int(args.scenes):
        raise RuntimeError(f"Requested {args.scenes} scenes, found {len(manifest)}.")
    manifest = manifest.iloc[: int(args.scenes)].copy().reset_index(drop=True)

    scenario_order = manifest[["Scenario_ID", "Vehicle_File", "Signal_File", "Context_File"]].copy()
    scenario_order.insert(0, "episode_index", np.arange(1, len(manifest) + 1, dtype=int))
    atomic_csv(scenario_order, args.output_dir / "scenario_order.csv")
    write_json(
        args.output_dir / "run_config.json",
        {
            "seed": int(args.seed),
            "split": str(args.split),
            "scenes": int(args.scenes),
            "attack_keys": [spec["key"] for spec in selected_attack_specs],
            "epsilon": float(args.epsilon),
            "attack_ratio": float(args.attack_ratio),
            "attack_scope": str(args.attack_scope),
            "price_threshold": float(price_threshold),
            "repair_mode": REPAIR_MODE,
            "runtime_pipeline_order": RUNTIME_PIPELINE_ORDER,
            "temporal_feature_definition": {
                "mean_magnitude": "mean per attacked observation L-infinity norm of adv_obs-clean_obs",
                "adjacent_change": "mean per vehicle consecutive L-infinity norm of delta_t-delta_{t-1}",
                "direction_ratio": "mean sign consistency over comparable dimensions for consecutive deltas",
            },
        },
    )

    intermediate_path = args.output_dir / "intermediate" / "attack_temporal_rollouts.jsonl"
    expected_keys = {
        (str(row["Scenario_ID"]), str(spec["key"]))
        for _, row in manifest.iterrows()
        for spec in selected_attack_specs
    }
    existing_rows = load_jsonl(intermediate_path) if args.resume else []
    existing_rows = [row for row in existing_rows if (str(row["scenario_id"]), str(row["attack_key"])) in expected_keys]
    completed = {(str(row["scenario_id"]), str(row["attack_key"])) for row in existing_rows}
    if len(completed) != len(existing_rows):
        raise RuntimeError("Duplicate rollout keys in profile JSONL.")

    new_rollouts = 0
    expected_total = len(expected_keys)
    started = time.perf_counter()
    for episode_index, (_, scenario_row) in enumerate(manifest.iterrows(), start=1):
        arrivals, signal_path, scenario_id = load_scenario(scenario_row)
        for attack_index, attack_spec in enumerate(selected_attack_specs):
            key = (scenario_id, attack_spec["key"])
            if key in completed:
                continue
            attack_seed = int(args.seed + attack_index * 100_000 + episode_index)
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
            t0 = time.perf_counter()
            summary = rollout_attack_temporal_profile(
                arrivals,
                actor,
                Path(signal_path),
                device,
                attack_enabled=attack_spec["key"] != "clean",
                attack_scenario=attack_spec["scenario"],
                attacker=attacker,
                state_scope=attack_spec["scope"],
                price_threshold=float(price_threshold),
                attack_ratio=float(args.attack_ratio),
                attack_scope=str(args.attack_scope),
            )
            row = {
                "scenario_id": scenario_id,
                "episode_index": int(episode_index),
                "seed": int(args.seed),
                "attack_seed": int(attack_seed),
                "attack_key": attack_spec["key"],
                "attack_display_name": attack_spec["display"],
                "attack_temporal_type": ATTACK_TYPE_BY_KEY.get(attack_spec["key"], "other"),
                "attack_scenario": attack_spec["scenario"],
                "attack_state_scope": attack_spec["scope"],
                "seen_in_dtsr_training": bool(attack_spec["seen"]),
                "runtime_seconds": float(time.perf_counter() - t0),
                **to_scalar_summary(summary),
            }
            if int(row.get("done_cnt", -1)) != 344:
                raise RuntimeError(f"Incomplete rollout for {key}: done_cnt={row.get('done_cnt')}")
            append_jsonl(intermediate_path, row)
            completed.add(key)
            new_rollouts += 1
            print(
                f"[{len(completed):04d}/{expected_total}] ep={episode_index:03d} "
                f"{attack_spec['display']} reward={float(row['ep_reward']):.2f} "
                f"mag={float(row.get('attack_delta_linf_mean', float('nan'))):.4f} "
                f"adj={float(row.get('temporal_adjacent_linf_mean', float('nan'))):.4f} "
                f"dir={float(row.get('temporal_direction_ratio_mean', float('nan'))):.4f} "
                f"time={float(row['runtime_seconds']):.1f}s",
                flush=True,
            )
            if args.max_rollouts and new_rollouts >= int(args.max_rollouts):
                return

    rows = load_jsonl(intermediate_path)
    long_df = pd.DataFrame(rows)
    atomic_csv(long_df, args.output_dir / "tables" / "attack_temporal_feature_episode_metrics.csv")
    summary, paper = create_summary(long_df, args.output_dir)
    report_lines = [
        "# Attack Temporal Feature Profile",
        "",
        f"- Completed rollouts: {len(long_df)} / {expected_total}",
        f"- Split/scenes: {args.split} / {args.scenes}",
        f"- Runtime seconds: {time.perf_counter() - started:.1f}",
        "",
        "## Paper Table",
        "",
        markdown_table(paper),
        "",
        "## Raw Summary",
        "",
        markdown_table(summary),
    ]
    (args.output_dir / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(paper.to_string(index=False), flush=True)
    print(f"Saved: {args.output_dir / 'tables' / 'attack_temporal_feature_table_paper.csv'}", flush=True)


if __name__ == "__main__":
    main()
