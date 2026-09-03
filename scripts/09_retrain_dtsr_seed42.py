from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import PACKAGE_ROOT, actor_matches_bundle, deterministic_subset, load_manifest, load_scenario, resolve_device, write_json
from dtsr_multiday_common import (
    ABLATION_ADDITION_ORDER,
    EP100_ACTOR_PATH,
    EP100_BUNDLE_PATH,
    REPAIR_MODE,
    RUNTIME_PIPELINE_ORDER,
    runtime_dae_validation_metrics,
    safe_recovery,
    set_all_seeds,
    ug_bcr_config_payload,
    union_observation_bounds,
    validation_price_median,
)

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from cli import merge_pair_bundles_for_unified, posterior_detector_dataset_from_unified_pair
from evc.defense import (
    posterior_detector_probabilities,
    save_dae,
    save_dae_history,
    save_detector,
    save_detector_history,
)
from evc.long_horizon_attacks import build_long_horizon_attacker
from evc.merged_attacks import PGDStateAttacker
from evc.merged_core import ChargingEnv, Critic, TRAIN_PROFILE, load_actor_critic_bundle, load_actor_from_path
from evc.merged_pipeline import (
    PairDatasetBundle,
    build_pair_dataset_from_clean_trajectories,
    load_clean_trajectory_dataset,
    save_pair_dataset,
    train_dae_from_bundle,
    train_detector_from_bundle,
)
from evc.offline_dae_det_temporal_shield import (
    LocalTemporalShieldConfig,
    calibrate_local_temporal_shield,
    save_temporal_shield_bundle,
)
from evc.ug_bcr import BeliefCoreConfig, UGBCRConfig, UrgencyGateConfig, rollout_episode_with_ug_bcr


def configure_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)


UG_CANDIDATES = [
    ("baseline", 0.010, 0.025, 0.013, 0.065, 0.022),
    ("sensitive", 0.006, 0.020, 0.010, 0.075, 0.028),
    ("conservative", 0.014, 0.030, 0.016, 0.055, 0.018),
    ("soc_sensitive", 0.008, 0.015, 0.013, 0.065, 0.022),
    ("time_sensitive", 0.010, 0.025, 0.008, 0.065, 0.022),
    ("low_uncertainty", 0.010, 0.025, 0.013, 0.050, 0.022),
    ("high_uncertainty", 0.010, 0.025, 0.013, 0.080, 0.022),
    ("tight_temporal", 0.010, 0.025, 0.013, 0.065, 0.015),
    ("loose_temporal", 0.010, 0.025, 0.013, 0.065, 0.030),
]


def build_offline_attacker(actor, critic, device, algorithm: str, epsilon: float, alpha: float, iters: int, state_scope: str, seed: int, split: str):
    low, high = union_observation_bounds(split)
    return PGDStateAttacker(
        actor,
        device=device,
        algorithm=algorithm,
        epsilon=epsilon,
        alpha=alpha,
        iters=iters,
        seed=seed,
        obs_low=low,
        obs_high=high,
        critic=critic if algorithm == "q_function" else None,
        attack_state_scope=state_scope,
    )


def best_threshold(labels, probabilities) -> dict:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    candidates = np.linspace(0.05, 0.95, 181)
    best = None
    for threshold in candidates:
        pred = probabilities >= threshold
        tp = int(np.sum((labels == 1) & pred))
        tn = int(np.sum((labels == 0) & (~pred)))
        fp = int(np.sum((labels == 0) & pred))
        fn = int(np.sum((labels == 1) & (~pred)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        fpr = fp / max(fp + tn, 1)
        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_positive_rate": float(fpr),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
        feasible = fpr <= 0.10
        key = (1 if feasible else 0, f1, recall, precision, -fpr, -float(threshold))
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    return best[1]


def subset_pair_bundle_episodes(bundle: PairDatasetBundle, max_episodes: int) -> PairDatasetBundle:
    if int(max_episodes) <= 0 or bundle.episode_indices is None:
        return bundle
    episode_indices = np.asarray(bundle.episode_indices, dtype=np.int64).reshape(-1)
    unique_episodes = list(dict.fromkeys(int(x) for x in episode_indices.tolist()))
    source_count = max(int((bundle.metadata or {}).get("source_bundle_count", 1)), 1)
    episode_blocks = np.array_split(np.asarray(unique_episodes, dtype=np.int64), source_count)
    selected = {
        int(ep)
        for block in episode_blocks
        for ep in block[: int(max_episodes)].tolist()
    }
    if len(selected) >= len(unique_episodes):
        return bundle
    mask = np.asarray([int(x) in selected for x in episode_indices], dtype=bool)

    def take(values):
        return None if values is None else np.asarray(values)[mask]

    metadata = dict(bundle.metadata or {})
    metadata.update({
        "validator_episode_limit": int(max_episodes),
        "validator_episode_limit_per_source_bundle": int(max_episodes),
        "validator_source_bundle_count": int(source_count),
        "validator_selected_episodes": sorted(selected),
        "validator_samples": int(mask.sum()),
    })
    return PairDatasetBundle(
        adv_inputs=take(bundle.adv_inputs),
        clean_inputs=take(bundle.clean_inputs),
        metadata=metadata,
        clean_anchor_inputs=take(bundle.clean_anchor_inputs),
        time_indices=take(bundle.time_indices),
        stations=take(bundle.stations),
        is_new_arrivals=take(bundle.is_new_arrivals),
        vehicle_ids=take(bundle.vehicle_ids),
        episode_indices=take(bundle.episode_indices),
        attack_mask=take(bundle.attack_mask),
    )


def calibrate_multiday_shield(actor, device, scenario_count: int, seed: int, state_scope: str):
    manifest = deterministic_subset(load_manifest("val"), scenario_count, seed)
    rows = []
    for _, row in manifest.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(row)
        _, stats = calibrate_local_temporal_shield(
            arrivals,
            signal_path,
            actor,
            device,
            reward_profile=TRAIN_PROFILE,
            calibration_quantile=0.99,
            state_scope=state_scope,
        )
        stats["scenario_id"] = scenario_id
        rows.append(stats)
    frame = pd.DataFrame(rows)
    tau_soc = float(np.clip(np.quantile(frame["residual_soc_quantile"], 0.90), 0.02, 0.08))
    tau_time = float(np.clip(np.quantile(frame["residual_time_quantile"], 0.90), 0.005, 0.03))
    tau_cost = float(np.clip(np.quantile(frame["residual_cost_quantile"], 0.90), 0.02, 0.08))
    config = LocalTemporalShieldConfig(
        state_scope=state_scope,
        tau_soc=tau_soc,
        tau_time=tau_time,
        tau_cost=tau_cost,
        calibration_quantile=0.99,
    )
    summary = {
        "scenario_count": len(frame),
        "seed": int(seed),
        "tau_soc": tau_soc,
        "tau_time": tau_time,
        "tau_cost": tau_cost,
        "aggregation": "90th percentile of per-day 0.99 residual quantiles",
    }
    return config, summary, frame


def fresh_attacker(attacker):
    clone = attacker.clone() if hasattr(attacker, "clone") else attacker
    if hasattr(clone, "reset"):
        clone.reset()
    return clone


def build_calibration_attacker(name: str, actor, critic, device, arrivals, signal_path, seed: int):
    env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
    low, high = env.observation_bounds(max_duration_of_stay=max(12, int(arrivals["Duration_of_stay"].max())))
    if name in {"opposite_pgd", "q_function"}:
        return PGDStateAttacker(
            actor,
            device=device,
            algorithm=name,
            epsilon=0.1,
            alpha=0.01,
            iters=10,
            seed=seed,
            obs_low=low,
            obs_high=high,
            critic=critic if name == "q_function" else None,
            attack_state_scope="all",
        )
    return build_long_horizon_attacker(
        name,
        actor=actor,
        device=device,
        obs_low=low,
        obs_high=high,
        critic=critic,
        seed=seed,
    )


def make_ug_config(candidate) -> UGBCRConfig:
    name, urgency_gain, soc_drop, time_drop, uncertainty, temporal = candidate
    del name
    return UGBCRConfig(
        belief=BeliefCoreConfig(enabled=True),
        urgency_gate=UrgencyGateConfig(
            enabled=True,
            urgency_gain_threshold=float(urgency_gain),
            soc_drop_threshold=float(soc_drop),
            time_drop_threshold=float(time_drop),
            uncertainty_threshold=float(uncertainty),
            temporal_residual_threshold=float(temporal),
        ),
    )


def calibrate_ug_bcr(
    *,
    actor,
    critic,
    dae,
    detector_model,
    detector_threshold: float,
    shield_config,
    device,
    seed: int,
    scene_count: int,
    output_dir: Path,
    candidate_limit: int = 0,
) -> tuple[UGBCRConfig, dict, pd.DataFrame]:
    attacks = ["opposite_pgd", "q_function", "local_small_drift_q", "local_deadline_drift_pgd"]
    candidates = UG_CANDIDATES if int(candidate_limit) <= 0 else UG_CANDIDATES[: int(candidate_limit)]
    if not candidates:
        raise ValueError("UG-BCR calibration candidate set is empty.")
    manifest = deterministic_subset(load_manifest("val"), scene_count, seed)
    rows: list[dict] = []
    clean_raw_cache: dict[str, dict] = {}

    for _, scenario_row in manifest.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(scenario_row)
        clean_raw_cache[scenario_id] = rollout_episode_with_ug_bcr(
            arrivals,
            actor,
            signal_path,
            device,
            TRAIN_PROFILE,
            attack_enabled=False,
            route_mode="none",
            enable_shield=False,
            enable_belief=False,
            enable_urgency_gate=False,
            label="clean_raw",
            repair_mode=REPAIR_MODE,
        )

    for candidate_index, candidate in enumerate(candidates):
        candidate_name = candidate[0]
        ug_config = make_ug_config(candidate)
        for _, scenario_row in manifest.iterrows():
            arrivals, signal_path, scenario_id = load_scenario(scenario_row)
            clean_raw = clean_raw_cache[scenario_id]
            clean_full = rollout_episode_with_ug_bcr(
                arrivals,
                actor,
                signal_path,
                device,
                TRAIN_PROFILE,
                attack_enabled=False,
                defender=dae,
                detector_model=detector_model,
                detector_threshold=detector_threshold,
                shield_config=shield_config,
                route_mode="detector",
                enable_shield=True,
                enable_belief=True,
                enable_urgency_gate=True,
                ug_bcr_config=ug_config,
                label="clean_full",
                repair_mode=REPAIR_MODE,
            )
            clean_drop_ratio = max(0.0, float(clean_raw["ep_reward"]) - float(clean_full["ep_reward"])) / max(abs(float(clean_raw["ep_reward"])), 1e-8)
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "candidate_name": candidate_name,
                    "scenario_id": scenario_id,
                    "attack": "clean",
                    "clean_drop_ratio": clean_drop_ratio,
                    "recovery": np.nan,
                    "exit_vio": int(clean_full.get("exit_vio", 0)),
                    "run_vio": int(clean_full.get("run_vio", 0)),
                    "belief_rate": float(clean_full.get("urgency_gate_belief_rate", 0.0)),
                    "reward": float(clean_full["ep_reward"]),
                }
            )

            for attack_index, attack_name in enumerate(attacks):
                attack_seed = int(seed + attack_index * 100_000 + int(scenario_id.split("_")[-1]))
                attacker = build_calibration_attacker(attack_name, actor, critic, device, arrivals, signal_path, attack_seed)
                attack_only = rollout_episode_with_ug_bcr(
                    arrivals,
                    actor,
                    signal_path,
                    device,
                    TRAIN_PROFILE,
                    attack_enabled=True,
                    attack_scenario="O",
                    attacker=fresh_attacker(attacker),
                    route_mode="none",
                    enable_shield=False,
                    enable_belief=False,
                    enable_urgency_gate=False,
                    state_scope="all" if attack_name in {"opposite_pgd", "q_function"} else "local",
                    attack_scope="obs",
                    label="attack_only",
                    repair_mode=REPAIR_MODE,
                )
                full = rollout_episode_with_ug_bcr(
                    arrivals,
                    actor,
                    signal_path,
                    device,
                    TRAIN_PROFILE,
                    attack_enabled=True,
                    attack_scenario="O",
                    attacker=fresh_attacker(attacker),
                    defender=dae,
                    detector_model=detector_model,
                    detector_threshold=detector_threshold,
                    shield_config=shield_config,
                    route_mode="detector",
                    enable_shield=True,
                    enable_belief=True,
                    enable_urgency_gate=True,
                    ug_bcr_config=ug_config,
                    state_scope="all" if attack_name in {"opposite_pgd", "q_function"} else "local",
                    attack_scope="obs",
                    label="full",
                    repair_mode=REPAIR_MODE,
                )
                rows.append(
                    {
                        "candidate_index": candidate_index,
                        "candidate_name": candidate_name,
                        "scenario_id": scenario_id,
                        "attack": attack_name,
                        "clean_drop_ratio": clean_drop_ratio,
                        "recovery": safe_recovery(float(clean_raw["ep_reward"]), float(attack_only["ep_reward"]), float(full["ep_reward"])),
                        "exit_vio": int(full.get("exit_vio", 0)),
                        "run_vio": int(full.get("run_vio", 0)),
                        "belief_rate": float(full.get("urgency_gate_belief_rate", 0.0)),
                        "reward": float(full["ep_reward"]),
                    }
                )
        print(f"[UG-BCR calibration] finished candidate {candidate_index}: {candidate_name}")

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "ug_bcr_calibration.csv", index=False, encoding="utf-8-sig")
    summary_rows: list[dict] = []
    for candidate_index, candidate in enumerate(candidates):
        candidate_name = candidate[0]
        sub = frame[frame["candidate_index"] == candidate_index]
        clean_sub = sub[sub["attack"] == "clean"]
        attack_sub = sub[sub["attack"] != "clean"]
        summary_rows.append(
            {
                "candidate_index": candidate_index,
                "candidate_name": candidate_name,
                "mean_clean_drop_ratio": float(clean_sub["clean_drop_ratio"].mean()),
                "mean_recovery": float(attack_sub["recovery"].mean()),
                "mean_exit_vio": float(attack_sub["exit_vio"].mean()),
                "mean_run_vio": float(attack_sub["run_vio"].mean()),
                "mean_belief_rate": float(attack_sub["belief_rate"].mean()),
                "feasible_clean": bool(float(clean_sub["clean_drop_ratio"].mean()) <= 0.02),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / "ug_bcr_candidate_summary.csv", index=False, encoding="utf-8-sig")
    feasible = summary_frame[summary_frame["feasible_clean"]].copy()
    pool = feasible if not feasible.empty else summary_frame.copy()
    pool = pool.sort_values(
        ["mean_recovery", "mean_exit_vio", "mean_run_vio", "mean_belief_rate", "candidate_index"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    )
    selected_row = pool.iloc[0]
    selected_index = int(selected_row["candidate_index"])
    selected_config = make_ug_config(candidates[selected_index])
    selection = {
        "selected_candidate_index": selected_index,
        "selected_candidate_name": str(selected_row["candidate_name"]),
        "selection_rule": "validation-only: clean_drop<=2%, then recovery desc, exit/run/belief asc",
        "selected_summary": {key: (bool(value) if isinstance(value, (np.bool_, bool)) else float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer, int)) else value) for key, value in selected_row.to_dict().items()},
        "scene_count": int(scene_count),
        "attacks": attacks,
        "seed": int(seed),
        "runtime_pipeline_order": RUNTIME_PIPELINE_ORDER,
    }
    write_json(output_dir / "ug_bcr_selection.json", selection)
    return selected_config, selection, frame


def main() -> None:
    configure_line_buffering()
    parser = argparse.ArgumentParser(description="Retrain multiday DTSR with seed=42 and validation-only model/calibration selection.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actor-path", type=Path, default=EP100_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=EP100_BUNDLE_PATH)
    parser.add_argument("--clean-train", type=Path, default=PACKAGE_ROOT / "artifacts" / "dtsr_seed42_multiday" / "clean" / "clean_train.npz")
    parser.add_argument("--clean-val", type=Path, default=PACKAGE_ROOT / "artifacts" / "dtsr_seed42_multiday" / "clean" / "clean_val.npz")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dtsr_seed42_multiday")
    parser.add_argument("--state-scope", choices=["local", "all"], default="all")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--pgd-iters", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--dae-epochs", type=int, default=50)
    parser.add_argument("--detector-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--dae-val-every", type=int, default=5)
    parser.add_argument("--dae-validator-scenes", type=int, default=20)
    parser.add_argument("--shield-val-scenes", type=int, default=30)
    parser.add_argument("--ug-calibration-scenes", type=int, default=8)
    parser.add_argument("--ug-candidate-limit", type=int, default=0, help="Debug only: limit UG-BCR candidates; 0 means all 9.")
    parser.add_argument("--save-pair-datasets", action="store_true")
    args = parser.parse_args()

    if int(args.seed) != 42:
        raise ValueError("This final paper run is intentionally fixed to a single training seed: seed=42.")
    set_all_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    actor = load_actor_from_path(args.actor_path, device)
    bundle_payload = load_actor_critic_bundle(args.bundle_path, device)
    if bundle_payload.get("critic_state_dict") is None:
        raise RuntimeError("Selected ep100 bundle does not contain critic weights.")
    if not actor_matches_bundle(actor, bundle_payload):
        raise RuntimeError("Actor and bundle actor_state_dict do not match.")
    checkpoint_episode = int((bundle_payload.get("metadata") or {}).get("checkpoint_episode", -1))
    if checkpoint_episode != 100:
        raise RuntimeError(f"Expected ep100 DDPG, got checkpoint_episode={checkpoint_episode}.")
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()
    critic = Critic().to(device)
    critic.load_state_dict(bundle_payload["critic_state_dict"])
    for p in critic.parameters():
        p.requires_grad_(False)
    critic.eval()

    clean_train = load_clean_trajectory_dataset(args.clean_train)
    clean_val = load_clean_trajectory_dataset(args.clean_val)
    if clean_train.clean_inputs.shape[1] != 11 or clean_val.clean_inputs.shape[1] != 11:
        raise ValueError("DTSR expects 11-dimensional observations.")

    attacks = ["opposite_pgd", "q_function"]
    train_pairs = []
    val_pairs = []
    for attack_index, algorithm in enumerate(attacks):
        train_attacker = build_offline_attacker(
            actor, critic, device, algorithm, args.epsilon, 0.01, args.pgd_iters, args.state_scope, args.seed + attack_index * 1000, "train"
        )
        val_attacker = build_offline_attacker(
            actor, critic, device, algorithm, args.epsilon, 0.01, args.pgd_iters, args.state_scope, args.seed + attack_index * 1000, "val"
        )
        train_pair = build_pair_dataset_from_clean_trajectories(
            clean_train,
            train_attacker,
            "O",
            attack_ratio=1.0,
            attack_scope="obs",
            chunk_size=args.chunk_size,
        )
        val_pair = build_pair_dataset_from_clean_trajectories(
            clean_val,
            val_attacker,
            "O",
            attack_ratio=1.0,
            attack_scope="obs",
            chunk_size=args.chunk_size,
        )
        train_pair.metadata.update({"algorithm": algorithm, "state_scope": args.state_scope, "bounds_split": "train"})
        val_pair.metadata.update({"algorithm": algorithm, "state_scope": args.state_scope, "bounds_split": "val"})
        train_pairs.append(train_pair)
        val_pairs.append(val_pair)

    unified_train = merge_pair_bundles_for_unified(train_pairs, attack_tags=attacks)
    unified_val = merge_pair_bundles_for_unified(val_pairs, attack_tags=attacks)
    if args.save_pair_datasets:
        save_pair_dataset(unified_train, args.output_dir / "pair_train.npz")
        save_pair_dataset(unified_val, args.output_dir / "pair_val.npz")

    validator_bundle = subset_pair_bundle_episodes(unified_val, args.dae_validator_scenes)
    validator_episodes = (
        len(np.unique(validator_bundle.episode_indices))
        if validator_bundle.episode_indices is not None
        else "all"
    )
    print(f"  DAE validator: {validator_bundle.clean_inputs.shape[0]} samples from {validator_episodes} episodes")

    def dae_validator(model):
        return runtime_dae_validation_metrics(
            model,
            clean_inputs=validator_bundle.clean_inputs,
            adv_inputs=validator_bundle.adv_inputs,
            actor=actor,
            device=device,
            episode_indices=validator_bundle.episode_indices,
            vehicle_ids=validator_bundle.vehicle_ids,
            attack_mask=validator_bundle.attack_mask,
            batch_size=2048,
            clean_penalty_weight=0.25,
        )

    dae, dae_result = train_dae_from_bundle(
        unified_train,
        actor,
        device,
        epochs=args.dae_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_state=1.0,
        lambda_identity=1.0,
        validator=dae_validator,
        val_every=args.dae_val_every,
        select_by="dae_checkpoint_score",
        log_every=1,
        progress_dir=args.output_dir,
        progress_prefix="dae",
        seq_len=args.seq_len,
        hidden_dim=128,
        latent_dim=64,
        decoder_hidden_dim=128,
        beta_kl=1e-3,
        lambda_robust=0.2,
        include_clean_sequences=True,
        state_scope=args.state_scope,
    )
    dae_path = save_dae(
        dae,
        args.output_dir / "dtsr_dae.pt",
        metadata={
            "policy": str(args.actor_path),
            "policy_checkpoint_episode": checkpoint_episode,
            "seed": int(args.seed),
            "train_attacks": attacks,
            "epsilon": args.epsilon,
            "state_scope": args.state_scope,
            "repair_mode": REPAIR_MODE,
            "checkpoint_selection": "validation full-state policy candidate action recovery with clean identity penalty",
            "best_epoch": int(dae_result.best_epoch),
            "best_metric_name": str(dae_result.best_metric_name),
            "best_metric_value": float(dae_result.best_metric_value),
        },
    )
    save_dae_history(dae_result, args.output_dir / "dae_history.csv")
    pd.DataFrame(dae_result.validator_rows).to_csv(args.output_dir / "dae_validation_history.csv", index=False, encoding="utf-8-sig")
    write_json(
        args.output_dir / "dae_best_epoch.json",
        {
            "best_epoch": int(dae_result.best_epoch),
            "best_metric_name": str(dae_result.best_metric_name),
            "best_metric_value": float(dae_result.best_metric_value),
        },
    )

    detector_train = posterior_detector_dataset_from_unified_pair(
        unified_train,
        actor,
        dae,
        device,
        profile_tag="multiday_dtsr_seed42",
        train_attack_tags=attacks,
        benefit_margin=0.0,
        benefit_action_weight=1.0,
        benefit_state_weight=1.0,
        posterior_label_mode="benefit",
        use_benefit_sample_weights=True,
        state_scope=args.state_scope,
        repair_mode=REPAIR_MODE,
    )
    detector_val = posterior_detector_dataset_from_unified_pair(
        unified_val,
        actor,
        dae,
        device,
        profile_tag="multiday_dtsr_seed42_val",
        train_attack_tags=attacks,
        benefit_margin=0.0,
        benefit_action_weight=1.0,
        benefit_state_weight=1.0,
        posterior_label_mode="benefit",
        use_benefit_sample_weights=False,
        state_scope=args.state_scope,
        repair_mode=REPAIR_MODE,
    )
    detector, detector_result = train_detector_from_bundle(
        detector_train,
        actor,
        dae,
        device,
        epochs=args.detector_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=128,
        dropout=0.1,
        val_ratio=0.2,
        detector_temporal=True,
        detector_feature_mode="posterior",
        seed=args.seed,
        state_scope=args.state_scope,
        progress_dir=args.output_dir,
        progress_prefix="detector",
    )
    val_probabilities = posterior_detector_probabilities(
        detector,
        detector_val.obs_inputs,
        detector_val.rec_inputs,
        actor,
        device,
        time_indices=detector_val.time_indices,
        stations=detector_val.stations,
        is_new_arrivals=detector_val.is_new_arrivals,
        prev_obs_inputs=detector_val.prev_obs_inputs,
        include_temporal=True,
    )
    threshold_report = best_threshold(detector_val.labels, val_probabilities)
    detector_path = save_detector(
        detector,
        args.output_dir / "dtsr_detector.pt",
        threshold=threshold_report["threshold"],
        metadata={
            "policy": str(args.actor_path),
            "policy_checkpoint_episode": checkpoint_episode,
            "seed": int(args.seed),
            "train_attacks": attacks,
            "epsilon": args.epsilon,
            "state_scope": args.state_scope,
            "repair_mode": REPAIR_MODE,
            "posterior_candidate_state": "full_reconstruction",
            "best_epoch": int(detector_result.best_epoch),
            "best_metric_name": str(detector_result.best_metric_name),
            "best_metric_value": float(detector_result.best_metric_value),
            "threshold_selection": "external validation benefit-F1 with FPR<=0.10 preference",
        },
        history={"threshold_report": threshold_report},
    )
    save_detector_history(detector_result, args.output_dir / "detector_history.csv")
    write_json(args.output_dir / "detector_threshold_report.json", threshold_report)

    shield_config, shield_summary, shield_rows = calibrate_multiday_shield(actor, device, args.shield_val_scenes, args.seed, args.state_scope)
    shield_path = save_temporal_shield_bundle(
        shield_config,
        args.output_dir / "dtsr_temporal_shield.pt",
        metadata={
            "policy": str(args.actor_path),
            "policy_checkpoint_episode": checkpoint_episode,
            "seed": int(args.seed),
            "data": "multi-day validation scenarios",
            "state_scope": args.state_scope,
            "runtime_pipeline_order": RUNTIME_PIPELINE_ORDER,
        },
        calibration_stats=shield_summary,
    )
    shield_rows.to_csv(args.output_dir / "temporal_shield_calibration.csv", index=False, encoding="utf-8-sig")
    write_json(args.output_dir / "temporal_shield_summary.json", shield_summary)

    price_threshold, validation_price_count = validation_price_median()
    write_json(
        args.output_dir / "electhacker_c_price_threshold.json",
        {
            "source_split": "val",
            "method": "global median of raw validation prices",
            "price_threshold": price_threshold,
            "validation_price_count": validation_price_count,
        },
    )

    ug_config, ug_selection, _ = calibrate_ug_bcr(
        actor=actor,
        critic=critic,
        dae=dae,
        detector_model=detector,
        detector_threshold=float(threshold_report["threshold"]),
        shield_config=shield_config,
        device=device,
        seed=args.seed,
        scene_count=args.ug_calibration_scenes,
        output_dir=args.output_dir,
        candidate_limit=args.ug_candidate_limit,
    )
    ug_path = args.output_dir / "ug_bcr_config.json"
    write_json(ug_path, ug_bcr_config_payload(ug_config))

    manifest = {
        "status": "trained",
        "seed": int(args.seed),
        "frozen_ddpg_actor": str(args.actor_path),
        "frozen_ddpg_bundle": str(args.bundle_path),
        "ddpg_checkpoint_episode": checkpoint_episode,
        "dae": str(dae_path),
        "dae_max_epochs": int(args.dae_epochs),
        "dae_best_epoch": int(dae_result.best_epoch),
        "dae_best_metric_name": str(dae_result.best_metric_name),
        "dae_best_metric_value": float(dae_result.best_metric_value),
        "detector": str(detector_path),
        "detector_max_epochs": int(args.detector_epochs),
        "detector_best_epoch": int(detector_result.best_epoch),
        "detector_best_metric_name": str(detector_result.best_metric_name),
        "detector_best_metric_value": float(detector_result.best_metric_value),
        "detector_threshold": threshold_report,
        "temporal_shield": str(shield_path),
        "temporal_shield_summary": shield_summary,
        "ug_bcr_config": str(ug_path),
        "ug_bcr_selection": ug_selection,
        "electhacker_c_price_threshold": price_threshold,
        "train_attacks": attacks,
        "epsilon": args.epsilon,
        "state_scope": args.state_scope,
        "repair_mode": REPAIR_MODE,
        "runtime_pipeline_order": RUNTIME_PIPELINE_ORDER,
        "ablation_addition_order": ABLATION_ADDITION_ORDER,
        "clean_train_samples": int(clean_train.clean_inputs.shape[0]),
        "clean_val_samples": int(clean_val.clean_inputs.shape[0]),
        "posterior_label_runtime_alignment_fix": True,
        "multiday_attack_bounds_fix": "split-level union bounds",
    }
    write_json(args.output_dir / "dtsr_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
