from __future__ import annotations

import argparse
import json
import sys
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
from dtsr_multiday_common import EP100_ACTOR_PATH, EP100_BUNDLE_PATH, REPAIR_MODE, safe_recovery, set_all_seeds

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.defense import load_dae, load_detector
from evc.merged_core import ChargingEnv, TRAIN_PROFILE
from evc.native_dtsr import (
    SUPPORTED_BACKBONES,
    build_frozen_attacker,
    default_native_bundle_path,
    load_frozen_attack_plan,
    load_frozen_backbone,
    native_artifact_layout,
    validate_attack_plan_provenance,
    validate_dataset_backbone,
)
from evc.offline_dae_det_temporal_shield import (
    LocalTemporalShieldConfig,
    calibrate_local_temporal_shield,
    save_temporal_shield_bundle,
)
from evc.ug_bcr import rollout_episode_with_ug_bcr


ATTACK_SPECS: list[dict[str, Any]] = [
    {"key": "local_deadline_drift_pgd", "display": "deadline_pgd", "algorithm": "local_deadline_drift_pgd", "scope": "local", "weight": 0.45},
    {"key": "local_small_drift_q", "display": "small_drift_q", "algorithm": "local_small_drift_q", "scope": "local", "weight": 0.35},
    {"key": "opposite_pgd", "display": "PGD*", "algorithm": "opposite_pgd", "scope": "all", "weight": 0.10},
    {"key": "q_function", "display": "Q-function*", "algorithm": "q_function", "scope": "all", "weight": 0.10},
]


def configure_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)


def parse_float_list(raw: str) -> list[float]:
    return [float(token.strip()) for token in str(raw).split(",") if token.strip()]


def fresh_attacker(attacker):
    clone = attacker.clone() if hasattr(attacker, "clone") else attacker
    if hasattr(clone, "reset"):
        clone.reset()
    return clone


def build_attacker(spec: dict[str, Any], backbone, attack_plan, device, arrivals, signal_path, seed: int):
    env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
    low, high = env.observation_bounds(max_duration_of_stay=max(12, int(arrivals["Duration_of_stay"].max())))
    return build_frozen_attacker(
        str(spec["algorithm"]),
        backbone=backbone,
        attack_plan=attack_plan,
        device=device,
        obs_low=low,
        obs_high=high,
        seed=int(seed),
    )


def rollout(
    arrivals,
    actor,
    signal_path,
    device,
    *,
    attacker=None,
    dae=None,
    detector_model=None,
    detector_threshold=None,
    shield_config=None,
    enable_shield: bool = False,
    state_scope: str = "local",
    label: str,
):
    return rollout_episode_with_ug_bcr(
        arrivals,
        actor,
        signal_path,
        device,
        TRAIN_PROFILE,
        attack_enabled=attacker is not None,
        attack_scenario="O",
        attacker=attacker,
        defender=dae,
        detector_model=detector_model,
        detector_threshold=detector_threshold,
        shield_config=shield_config,
        route_mode="none" if dae is None else "detector",
        enable_shield=bool(enable_shield),
        enable_belief=False,
        enable_urgency_gate=False,
        ug_bcr_config=None,
        state_scope=state_scope,
        attack_scope="obs",
        label=label,
        repair_mode=REPAIR_MODE,
    )


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), 0.0 if arr.size <= 1 else float(np.std(arr, ddof=1))


def main() -> None:
    configure_line_buffering()
    parser = argparse.ArgumentParser(description="Calibrate focused Temporal Shield using existing full-state DAE+DeT artifacts.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", choices=SUPPORTED_BACKBONES, default="ddpg")
    parser.add_argument("--actor-path", type=Path, default=EP100_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=EP100_BUNDLE_PATH)
    parser.add_argument(
        "--native-config",
        type=Path,
        default=PACKAGE_ROOT / "results" / "native_attack_calibration_seed42" / "native_attack_config.json",
    )
    parser.add_argument("--dae-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday")
    parser.add_argument("--detector-artifact-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "det_seed42_fullstate")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "shield_seed42_fullstate")
    parser.add_argument("--clean-calibration-scenes", type=int, default=20)
    parser.add_argument("--tuning-scenes", type=int, default=5)
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--state-scope", choices=["local"], default="local")
    parser.add_argument("--calibration-quantile", type=float, default=0.99)
    parser.add_argument("--aggregation-quantile", type=float, default=0.85)
    parser.add_argument("--min-tau-soc", type=float, default=0.010)
    parser.add_argument("--max-tau-soc", type=float, default=0.060)
    parser.add_argument("--min-tau-time", type=float, default=0.0025)
    parser.add_argument("--max-tau-time", type=float, default=0.025)
    parser.add_argument("--min-tau-cost", type=float, default=0.010)
    parser.add_argument("--max-tau-cost", type=float, default=0.060)
    parser.add_argument("--tau-soc-scales", default="0.50,0.75,1.00")
    parser.add_argument("--tau-time-scales", default="0.50,0.75,1.00")
    parser.add_argument("--tau-cost-scales", default="0.75,1.00,1.25")
    parser.add_argument("--clean-drop-cap", type=float, default=15.0)
    parser.add_argument("--clean-run-vio-cap", type=float, default=1.5)
    parser.add_argument("--clean-exit-vio-cap", type=float, default=4.0)
    parser.add_argument("--main-attack-recovery-drop-cap", type=float, default=2.0)
    args = parser.parse_args()

    if args.algorithm != "ddpg":
        layout = native_artifact_layout(PACKAGE_ROOT, args.algorithm, args.seed)
        if args.bundle_path == EP100_BUNDLE_PATH:
            args.bundle_path = default_native_bundle_path(PACKAGE_ROOT, args.algorithm, args.seed)
        if args.dae_artifact_dir == PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday":
            args.dae_artifact_dir = layout["dae"]
        if args.detector_artifact_dir == PACKAGE_ROOT / "artifacts" / "det_seed42_fullstate":
            args.detector_artifact_dir = layout["det"]
        if args.output_dir == PACKAGE_ROOT / "artifacts" / "shield_seed42_fullstate":
            args.output_dir = layout["shield"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = resolve_device(args.device)
    print(f"[SHIELD] device={device}, seed={args.seed}")

    backbone = load_frozen_backbone(args.algorithm, args.bundle_path, device)
    attack_plan = load_frozen_attack_plan(args.algorithm, args.native_config)
    actor = backbone.actor

    dae_path = args.dae_artifact_dir / "dtsr_dae.pt"
    detector_path = args.detector_artifact_dir / "dtsr_detector.pt"
    dae = load_dae(dae_path, device).eval()
    detector_artifact = load_detector(detector_path, device)
    detector_model = detector_artifact.model.eval()
    detector_threshold = float(detector_artifact.threshold)
    dae_manifest_path = args.dae_artifact_dir / "dae_manifest.json"
    detector_manifest_path = args.detector_artifact_dir / "det_manifest.json"
    dae_manifest = json.loads(dae_manifest_path.read_text(encoding="utf-8")) if dae_manifest_path.exists() else {}
    detector_manifest = json.loads(detector_manifest_path.read_text(encoding="utf-8")) if detector_manifest_path.exists() else {}
    for artifact_name, artifact_manifest in (("DAE", dae_manifest), ("DET", detector_manifest)):
        validate_dataset_backbone(artifact_manifest, backbone, split="train")
        validate_attack_plan_provenance(artifact_manifest, attack_plan)
        print(f"[SHIELD] verified {artifact_name} provenance for {backbone.algorithm}")
    print(f"[SHIELD] DAE={dae_path}")
    print(f"[SHIELD] Detector={detector_path}, threshold={detector_threshold:.6f}")

    manifest = load_manifest(args.split).sort_values("Scenario_ID", kind="mergesort").reset_index(drop=True)
    clean_frame = deterministic_subset(manifest, args.clean_calibration_scenes, args.seed)
    calibration_rows: list[dict[str, Any]] = []
    print(f"[SHIELD] Clean residual calibration scenes={len(clean_frame)}")
    for idx, row in clean_frame.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(row)
        _, stats = calibrate_local_temporal_shield(
            arrivals,
            signal_path,
            actor,
            device,
            reward_profile=TRAIN_PROFILE,
            calibration_quantile=float(args.calibration_quantile),
            min_tau_soc=float(args.min_tau_soc),
            min_tau_time=float(args.min_tau_time),
            min_tau_cost=float(args.min_tau_cost),
            max_tau_soc=float(args.max_tau_soc),
            max_tau_time=float(args.max_tau_time),
            max_tau_cost=float(args.max_tau_cost),
            state_scope=args.state_scope,
        )
        stats["scenario_id"] = scenario_id
        calibration_rows.append(stats)
        print(
            f"[SHIELD] calib {idx + 1:02d}/{len(clean_frame)} {scenario_id} "
            f"q_soc={stats['residual_soc_quantile']:.6f} "
            f"q_time={stats['residual_time_quantile']:.6f} "
            f"q_cost={stats['residual_cost_quantile']:.6f}"
        )
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_df.to_csv(args.output_dir / "temporal_shield_clean_calibration.csv", index=False, encoding="utf-8-sig")

    tau_soc = float(np.clip(np.quantile(calibration_df["residual_soc_quantile"], args.aggregation_quantile), args.min_tau_soc, args.max_tau_soc))
    tau_time = float(np.clip(np.quantile(calibration_df["residual_time_quantile"], args.aggregation_quantile), args.min_tau_time, args.max_tau_time))
    tau_cost = float(np.clip(np.quantile(calibration_df["residual_cost_quantile"], args.aggregation_quantile), args.min_tau_cost, args.max_tau_cost))
    base_config = LocalTemporalShieldConfig(
        state_scope=args.state_scope,
        tau_soc=tau_soc,
        tau_time=tau_time,
        tau_cost=tau_cost,
        calibration_quantile=float(args.calibration_quantile),
        min_tau_soc=float(args.min_tau_soc),
        min_tau_time=float(args.min_tau_time),
        min_tau_cost=float(args.min_tau_cost),
        max_tau_soc=float(args.max_tau_soc),
        max_tau_time=float(args.max_tau_time),
        max_tau_cost=float(args.max_tau_cost),
    )
    print(f"[SHIELD] base tau: soc={tau_soc:.6f}, time={tau_time:.6f}, cost={tau_cost:.6f}")

    tuning_frame = deterministic_subset(manifest, args.tuning_scenes, args.seed + 17)
    scenario_cache: list[dict[str, Any]] = []
    print(f"[SHIELD] Tuning scenes={len(tuning_frame)}, attacks={[a['key'] for a in ATTACK_SPECS]}")
    for scene_idx, row in tuning_frame.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(row)
        clean_raw = rollout(arrivals, actor, signal_path, device, state_scope="local", label="clean_raw")
        clean_det = rollout(
            arrivals,
            actor,
            signal_path,
            device,
            dae=dae,
            detector_model=detector_model,
            detector_threshold=detector_threshold,
            state_scope="local",
            label="clean_dae_det",
        )
        attack_baselines: dict[str, dict[str, Any]] = {}
        for attack_idx, spec in enumerate(ATTACK_SPECS):
            attack_seed = int(args.seed + attack_idx * 100_000 + scene_idx + 1)
            attacker = build_attacker(spec, backbone, attack_plan, device, arrivals, signal_path, attack_seed)
            raw = rollout(
                arrivals,
                actor,
                signal_path,
                device,
                attacker=fresh_attacker(attacker),
                state_scope=str(spec["scope"]),
                label=f"{spec['key']}__attack",
            )
            det = rollout(
                arrivals,
                actor,
                signal_path,
                device,
                attacker=fresh_attacker(attacker),
                dae=dae,
                detector_model=detector_model,
                detector_threshold=detector_threshold,
                state_scope=str(spec["scope"]),
                label=f"{spec['key']}__dae_det",
            )
            attack_baselines[str(spec["key"])] = {
                "raw": raw,
                "det": det,
                "seed": attack_seed,
            }
        scenario_cache.append(
            {
                "scenario_id": scenario_id,
                "arrivals": arrivals,
                "signal_path": signal_path,
                "clean_raw": clean_raw,
                "clean_det": clean_det,
                "attack_baselines": attack_baselines,
                "scene_idx": scene_idx,
            }
        )
        print(
            f"[SHIELD] baseline {len(scenario_cache):02d}/{len(tuning_frame)} {scenario_id} "
            f"clean_raw={float(clean_raw['ep_reward']):.2f} clean_det={float(clean_det['ep_reward']):.2f}"
        )

    tau_soc_scales = parse_float_list(args.tau_soc_scales)
    tau_time_scales = parse_float_list(args.tau_time_scales)
    tau_cost_scales = parse_float_list(args.tau_cost_scales)
    candidates: list[LocalTemporalShieldConfig] = []
    seen: set[tuple[float, float, float]] = set()
    for soc_scale in tau_soc_scales:
        for time_scale in tau_time_scales:
            for cost_scale in tau_cost_scales:
                soc = float(np.clip(tau_soc * soc_scale, args.min_tau_soc, args.max_tau_soc))
                time_tau = float(np.clip(tau_time * time_scale, args.min_tau_time, args.max_tau_time))
                cost = float(np.clip(tau_cost * cost_scale, args.min_tau_cost, args.max_tau_cost))
                key = (round(soc, 10), round(time_tau, 10), round(cost, 10))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    LocalTemporalShieldConfig(
                        state_scope=args.state_scope,
                        tau_soc=soc,
                        tau_time=time_tau,
                        tau_cost=cost,
                        calibration_quantile=float(args.calibration_quantile),
                        min_tau_soc=float(args.min_tau_soc),
                        min_tau_time=float(args.min_tau_time),
                        min_tau_cost=float(args.min_tau_cost),
                        max_tau_soc=float(args.max_tau_soc),
                        max_tau_time=float(args.max_tau_time),
                        max_tau_cost=float(args.max_tau_cost),
                    )
                )
    print(f"[SHIELD] candidate count={len(candidates)}")

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_config: LocalTemporalShieldConfig | None = None
    best_key: tuple[float, ...] | None = None
    baseline_det_recovery: dict[str, list[float]] = {str(spec["key"]): [] for spec in ATTACK_SPECS}
    for cached in scenario_cache:
        for spec in ATTACK_SPECS:
            key = str(spec["key"])
            raw = cached["attack_baselines"][key]["raw"]
            det = cached["attack_baselines"][key]["det"]
            baseline_det_recovery[key].append(
                safe_recovery(float(cached["clean_raw"]["ep_reward"]), float(raw["ep_reward"]), float(det["ep_reward"])) * 100.0
            )
    baseline_det_mean = {key: mean_std(values)[0] for key, values in baseline_det_recovery.items()}

    for cand_idx, config in enumerate(candidates, start=1):
        clean_drops: list[float] = []
        clean_run_vios: list[float] = []
        clean_exit_vios: list[float] = []
        recovery_by_attack: dict[str, list[float]] = {str(spec["key"]): [] for spec in ATTACK_SPECS}
        route_by_attack: dict[str, list[float]] = {str(spec["key"]): [] for spec in ATTACK_SPECS}
        correction_values: list[float] = []
        for cached in scenario_cache:
            clean_shield = rollout(
                cached["arrivals"],
                actor,
                cached["signal_path"],
                device,
                dae=dae,
                detector_model=detector_model,
                detector_threshold=detector_threshold,
                shield_config=config,
                enable_shield=True,
                state_scope="local",
                label="clean_shield",
            )
            clean_drops.append(float(cached["clean_raw"]["ep_reward"]) - float(clean_shield["ep_reward"]))
            clean_run_vios.append(float(clean_shield.get("run_vio", 0.0)))
            clean_exit_vios.append(float(clean_shield.get("exit_vio", 0.0)))
            correction_values.append(float(clean_shield.get("shield_correction_mean", 0.0)))
            for spec in ATTACK_SPECS:
                attack_key = str(spec["key"])
                attack_seed = int(cached["attack_baselines"][attack_key]["seed"])
                attacker = build_attacker(spec, backbone, attack_plan, device, cached["arrivals"], cached["signal_path"], attack_seed)
                shield = rollout(
                    cached["arrivals"],
                    actor,
                    cached["signal_path"],
                    device,
                    attacker=fresh_attacker(attacker),
                    dae=dae,
                    detector_model=detector_model,
                    detector_threshold=detector_threshold,
                    shield_config=config,
                    enable_shield=True,
                    state_scope=str(spec["scope"]),
                    label=f"{attack_key}__shield",
                )
                raw = cached["attack_baselines"][attack_key]["raw"]
                recovery_by_attack[attack_key].append(
                    safe_recovery(float(cached["clean_raw"]["ep_reward"]), float(raw["ep_reward"]), float(shield["ep_reward"])) * 100.0
                )
                route_by_attack[attack_key].append(float(shield.get("route_rate", 0.0)))
                correction_values.append(float(shield.get("shield_correction_mean", 0.0)))

        row: dict[str, Any] = {
            "candidate_index": cand_idx,
            "tau_soc": float(config.tau_soc),
            "tau_time": float(config.tau_time),
            "tau_cost": float(config.tau_cost),
            "clean_drop_mean": mean_std(clean_drops)[0],
            "clean_drop_std": mean_std(clean_drops)[1],
            "clean_run_vio_mean": mean_std(clean_run_vios)[0],
            "clean_exit_vio_mean": mean_std(clean_exit_vios)[0],
            "shield_correction_mean": mean_std(correction_values)[0],
        }
        score = 0.0
        for spec in ATTACK_SPECS:
            attack_key = str(spec["key"])
            rec_mean, rec_std = mean_std(recovery_by_attack[attack_key])
            route_mean, _ = mean_std(route_by_attack[attack_key])
            row[f"{attack_key}_recovery_mean_pct"] = rec_mean
            row[f"{attack_key}_recovery_std_pct"] = rec_std
            row[f"{attack_key}_route_rate_mean"] = route_mean
            score += float(spec["weight"]) * (rec_mean - 0.5 * rec_std)
        row["weighted_conservative_recovery_score"] = float(score)
        row["feasible_clean"] = bool(
            row["clean_drop_mean"] <= float(args.clean_drop_cap)
            and row["clean_run_vio_mean"] <= float(args.clean_run_vio_cap)
            and row["clean_exit_vio_mean"] <= float(args.clean_exit_vio_cap)
        )
        row["feasible_main_attack"] = bool(
            row["opposite_pgd_recovery_mean_pct"] >= baseline_det_mean["opposite_pgd"] - float(args.main_attack_recovery_drop_cap)
            and row["q_function_recovery_mean_pct"] >= baseline_det_mean["q_function"] - float(args.main_attack_recovery_drop_cap)
        )
        row["feasible"] = bool(row["feasible_clean"] and row["feasible_main_attack"])
        rows.append(row)
        selection_key = (
            1.0 if row["feasible"] else 0.0,
            float(row["weighted_conservative_recovery_score"]),
            -float(row["clean_drop_mean"]),
            -float(row["shield_correction_mean"]),
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_row = dict(row)
            best_config = config
        print(
            f"[SHIELD] candidate {cand_idx:02d}/{len(candidates)} "
            f"tau=({config.tau_soc:.4f},{config.tau_time:.4f},{config.tau_cost:.4f}) "
            f"score={row['weighted_conservative_recovery_score']:.2f} "
            f"clean_drop={row['clean_drop_mean']:.2f} "
            f"small={row['local_small_drift_q_recovery_mean_pct']:.1f}% "
            f"deadline={row['local_deadline_drift_pgd_recovery_mean_pct']:.1f}% "
            f"feasible={int(row['feasible'])}"
        )

    if best_config is None or best_row is None:
        raise RuntimeError("No temporal shield candidate was evaluated.")
    tuning_df = pd.DataFrame(rows)
    tuning_df.to_csv(args.output_dir / "temporal_shield_tuning.csv", index=False, encoding="utf-8-sig")
    shield_summary = {
        "seed": int(args.seed),
        "algorithm": backbone.algorithm,
        "backbone": backbone.provenance(),
        "attack_plan": attack_plan.provenance(),
        "state_scope": args.state_scope,
        "clean_calibration_scenes": int(len(clean_frame)),
        "tuning_scenes": int(len(tuning_frame)),
        "calibration_quantile": float(args.calibration_quantile),
        "aggregation_quantile": float(args.aggregation_quantile),
        "base_tau_soc": float(tau_soc),
        "base_tau_time": float(tau_time),
        "base_tau_cost": float(tau_cost),
        "selected_row": best_row,
        "baseline_det_recovery_mean_pct": baseline_det_mean,
        "selection_rule": "feasible first, then weighted mean(recovery - 0.5*std), then lower clean drop/correction",
        "attack_weights": {str(spec["key"]): float(spec["weight"]) for spec in ATTACK_SPECS},
    }
    shield_path = save_temporal_shield_bundle(
        best_config,
        args.output_dir / "dtsr_temporal_shield.pt",
        metadata={
            "algorithm": backbone.algorithm,
            "backbone": backbone.provenance(),
            "attack_plan": attack_plan.provenance(),
            "policy": str(backbone.bundle_path.resolve()),
            "seed": int(args.seed),
            "dae_artifact": str(dae_path),
            "detector_artifact": str(detector_path),
            "detector_threshold": float(detector_threshold),
            "repair_mode": REPAIR_MODE,
            "state_scope": args.state_scope,
        },
        calibration_stats=shield_summary,
    )
    write_json(args.output_dir / "temporal_shield_summary.json", shield_summary)
    manifest_payload = {
        "status": "calibrated",
        "seed": int(args.seed),
        "algorithm": backbone.algorithm,
        "backbone": backbone.provenance(),
        "attack_plan": attack_plan.provenance(),
        "temporal_shield": str(shield_path),
        "dae": str(dae_path),
        "detector": str(detector_path),
        "detector_threshold": float(detector_threshold),
        "selected_tau_soc": float(best_config.tau_soc),
        "selected_tau_time": float(best_config.tau_time),
        "selected_tau_cost": float(best_config.tau_cost),
        "selected_row": best_row,
        "outputs": {
            "clean_calibration": str(args.output_dir / "temporal_shield_clean_calibration.csv"),
            "tuning": str(args.output_dir / "temporal_shield_tuning.csv"),
            "summary": str(args.output_dir / "temporal_shield_summary.json"),
        },
    }
    write_json(args.output_dir / "shield_manifest.json", manifest_payload)
    print(json.dumps(manifest_payload, ensure_ascii=False, indent=2))
    print("[SHIELD] Done.")


if __name__ == "__main__":
    main()
