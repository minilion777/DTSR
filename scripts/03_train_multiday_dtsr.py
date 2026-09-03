from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import (
    DEFAULT_ACTOR_PATH,
    DEFAULT_BUNDLE_PATH,
    actor_matches_bundle,
    PACKAGE_ROOT,
    deterministic_subset,
    load_manifest,
    load_scenario,
    resolve_device,
    write_json,
)

sys.path.insert(0, str(PACKAGE_ROOT))
# These two DTSR dataset builders used to be re-exported by the large CLI
# entry point.  Importing them from their implementation module keeps this
# four-module release independent of the removed CLI and its unrelated
# baselines.
from evc.dtsr_datasets import (
    merge_pair_bundles_for_unified,
    posterior_detector_dataset_from_unified_pair,
)
from evc.defense import (
    posterior_detector_probabilities,
    save_dae,
    save_dae_history,
    save_detector,
    save_detector_history,
)
from evc.merged_attacks import PGDStateAttacker
from evc.merged_core import (
    ChargingEnv,
    Critic,
    TRAIN_PROFILE,
    load_actor_critic_bundle,
    load_actor_from_path,
)
from evc.merged_pipeline import (
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
from evc.ug_bcr import BeliefCoreConfig, UGBCRConfig, UrgencyGateConfig


def configure_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)


def build_attacker(actor, critic, device, algorithm, epsilon, alpha, iters, state_scope):
    sample_row = load_manifest("train").iloc[0]
    arrivals, signal_path, _ = load_scenario(sample_row)
    env = ChargingEnv(signal_path, TRAIN_PROFILE)
    low, high = env.observation_bounds(max_duration_of_stay=12)
    return PGDStateAttacker(
        actor,
        device=device,
        algorithm=algorithm,
        epsilon=epsilon,
        alpha=alpha,
        iters=iters,
        seed=42,
        obs_low=low,
        obs_high=high,
        critic=critic if algorithm == "q_function" else None,
        attack_state_scope=state_scope,
    )


def best_threshold(labels, probabilities):
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
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        }
        feasible = fpr <= 0.10
        key = (1 if feasible else 0, f1, recall, -fpr)
        if best is None or key > best[0]:
            best = (key, row)
    return best[1]


def calibrate_multiday_shield(actor, device, scenario_count, seed, state_scope):
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
        "tau_soc": tau_soc,
        "tau_time": tau_time,
        "tau_cost": tau_cost,
        "aggregation": "90th percentile of per-day 0.99 residual quantiles",
    }
    return config, summary, frame


def main():
    configure_line_buffering()
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--actor-path", type=Path, default=DEFAULT_ACTOR_PATH)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--clean-train", type=Path, default=PACKAGE_ROOT / "artifacts" / "clean" / "clean_train.npz")
    parser.add_argument("--clean-val", type=Path, default=PACKAGE_ROOT / "artifacts" / "clean" / "clean_val.npz")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "runs" / "dtsr")
    parser.add_argument("--state-scope", choices=["local", "all"], default="all")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--pgd-iters", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--dae-epochs", type=int, default=50)
    parser.add_argument("--detector-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--shield-val-scenes", type=int, default=30)
    parser.add_argument("--save-pair-datasets", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    actor = load_actor_from_path(args.actor_path, device)
    bundle_payload = load_actor_critic_bundle(args.bundle_path, device)
    if bundle_payload.get("critic_state_dict") is None:
        raise RuntimeError("Selected bundle does not contain critic weights.")
    if not actor_matches_bundle(actor, bundle_payload):
        raise RuntimeError(
            "--actor-path and --bundle-path do not contain the same actor weights. "
            "DTSR training uses Q-function attacks, so a matching critic bundle is required."
        )
    critic = Critic().to(device)
    critic.load_state_dict(bundle_payload["critic_state_dict"])
    critic.eval()

    clean_train = load_clean_trajectory_dataset(args.clean_train)
    clean_val = load_clean_trajectory_dataset(args.clean_val)

    attacks = [
        ("opposite_pgd", None),
        ("q_function", critic),
    ]
    train_pairs = []
    val_pairs = []
    attack_tags = []
    for algorithm, _ in attacks:
        attacker = build_attacker(
            actor, critic, device, algorithm,
            args.epsilon, 0.01, args.pgd_iters, args.state_scope,
        )
        train_pair = build_pair_dataset_from_clean_trajectories(
            clean_train,
            attacker,
            "O",
            attack_ratio=1.0,
            attack_scope="obs",
            chunk_size=args.chunk_size,
        )
        attacker.reset()
        val_pair = build_pair_dataset_from_clean_trajectories(
            clean_val,
            attacker,
            "O",
            attack_ratio=1.0,
            attack_scope="obs",
            chunk_size=args.chunk_size,
        )
        train_pair.metadata.update({"algorithm": algorithm, "state_scope": args.state_scope})
        val_pair.metadata.update({"algorithm": algorithm, "state_scope": args.state_scope})
        train_pairs.append(train_pair)
        val_pairs.append(val_pair)
        attack_tags.append(algorithm)

    unified_train = merge_pair_bundles_for_unified(train_pairs, attack_tags=attack_tags)
    unified_val = merge_pair_bundles_for_unified(val_pairs, attack_tags=attack_tags)
    if args.save_pair_datasets:
        save_pair_dataset(unified_train, args.output_dir / "pair_train.npz")
        save_pair_dataset(unified_val, args.output_dir / "pair_val.npz")

    dae, dae_result = train_dae_from_bundle(
        unified_train,
        actor,
        device,
        epochs=args.dae_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_state=1.0,
        lambda_identity=1.0,
        seq_len=args.seq_len,
        hidden_dim=128,
        latent_dim=64,
        decoder_hidden_dim=128,
        beta_kl=1e-3,
        lambda_robust=0.2,
        include_clean_sequences=True,
        state_scope=args.state_scope,
        log_every=1,
        progress_dir=args.output_dir,
        progress_prefix="dae",
    )
    dae_path = save_dae(
        dae,
        args.output_dir / "dtsr_dae.pt",
        metadata={
            "policy": str(args.actor_path),
            "train_attacks": attack_tags,
            "epsilon": args.epsilon,
            "state_scope": args.state_scope,
            "data": "multi-day semi-synthetic paired scenarios",
        },
    )
    save_dae_history(dae_result, args.output_dir / "dae_history.csv")

    detector_train = posterior_detector_dataset_from_unified_pair(
        unified_train,
        actor,
        dae,
        device,
        profile_tag="multiday_dtsr",
        train_attack_tags=attack_tags,
        benefit_margin=0.0,
        benefit_action_weight=1.0,
        benefit_state_weight=1.0,
        posterior_label_mode="benefit",
        use_benefit_sample_weights=True,
        state_scope=args.state_scope,
    )
    detector_val = posterior_detector_dataset_from_unified_pair(
        unified_val,
        actor,
        dae,
        device,
        profile_tag="multiday_dtsr_val",
        train_attack_tags=attack_tags,
        benefit_margin=0.0,
        benefit_action_weight=1.0,
        benefit_state_weight=1.0,
        posterior_label_mode="benefit",
        use_benefit_sample_weights=False,
        state_scope=args.state_scope,
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
        detector_feature_mode="sequence",
        seed=42,
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
            "train_attacks": attack_tags,
            "epsilon": args.epsilon,
            "state_scope": args.state_scope,
            "threshold_selection": "validation benefit-F1 with FPR preference",
        },
        history={"threshold_report": threshold_report},
    )
    save_detector_history(detector_result, args.output_dir / "detector_history.csv")
    write_json(args.output_dir / "detector_threshold_report.json", threshold_report)

    shield_config, shield_summary, shield_rows = calibrate_multiday_shield(
        actor,
        device,
        args.shield_val_scenes,
        20260711,
        args.state_scope,
    )
    shield_path = save_temporal_shield_bundle(
        shield_config,
        args.output_dir / "dtsr_temporal_shield.pt",
        metadata={
            "policy": str(args.actor_path),
            "data": "multi-day validation scenarios",
            "state_scope": args.state_scope,
        },
        calibration_stats=shield_summary,
    )
    shield_rows.to_csv(args.output_dir / "temporal_shield_calibration.csv", index=False)

    ug_bcr = UGBCRConfig(
        belief=BeliefCoreConfig(enabled=True),
        urgency_gate=UrgencyGateConfig(
            enabled=True,
            urgency_gain_threshold=0.010,
            soc_drop_threshold=0.025,
            time_drop_threshold=0.013,
            uncertainty_threshold=0.065,
            temporal_residual_threshold=0.022,
        ),
    )
    ug_bcr_path = args.output_dir / "ug_bcr_config.json"
    write_json(ug_bcr_path, asdict(ug_bcr))

    manifest = {
        "status": "trained",
        "frozen_ddpg_actor": str(args.actor_path),
        "frozen_ddpg_bundle": str(args.bundle_path),
        "dae": str(dae_path),
        "detector": str(detector_path),
        "detector_threshold": threshold_report,
        "temporal_shield": str(shield_path),
        "ug_bcr_config": str(ug_bcr_path),
        "train_attacks": attack_tags,
        "epsilon": args.epsilon,
        "state_scope": args.state_scope,
        "clean_train_samples": int(clean_train.clean_inputs.shape[0]),
        "clean_val_samples": int(clean_val.clean_inputs.shape[0]),
    }
    write_json(args.output_dir / "dtsr_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
