from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import PACKAGE_ROOT, resolve_device, write_json
from dtsr_multiday_common import EP100_ACTOR_PATH, EP100_BUNDLE_PATH, REPAIR_MODE, set_all_seeds

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.dtsr_datasets import posterior_detector_dataset_from_unified_pair
from evc.defense import (
    load_dae,
    posterior_detector_probabilities,
    save_detector,
    save_detector_history,
)
from evc.merged_pipeline import PairDatasetBundle, load_pair_dataset, train_detector_from_bundle
from evc.native_dtsr import (
    SUPPORTED_BACKBONES,
    default_native_bundle_path,
    load_frozen_attack_plan,
    load_frozen_backbone,
    native_artifact_layout,
    validate_attack_plan_provenance,
    validate_dataset_backbone,
)


def configure_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)


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


def _slice_optional(array, mask):
    if array is None:
        return None
    return np.asarray(array)[mask]


def subset_pair_bundle_by_episode(bundle: PairDatasetBundle, max_episodes_per_source: int) -> PairDatasetBundle:
    if max_episodes_per_source is None or int(max_episodes_per_source) <= 0:
        return bundle
    if bundle.episode_indices is None:
        print("[DET TRAIN] episode_indices missing; cannot episode-subset this pair bundle.")
        return bundle

    episode_indices = np.asarray(bundle.episode_indices, dtype=np.int64).reshape(-1)
    unique_episodes = np.array(list(dict.fromkeys(episode_indices.tolist())), dtype=np.int64)
    source_count = int((bundle.metadata or {}).get("source_bundle_count", 1) or 1)
    source_count = max(source_count, 1)
    blocks = np.array_split(unique_episodes, source_count)
    selected: set[int] = set()
    for block in blocks:
        selected.update(int(ep) for ep in block[: int(max_episodes_per_source)])
    mask = np.isin(episode_indices, np.fromiter(selected, dtype=np.int64))
    metadata = dict(bundle.metadata or {})
    metadata["original_samples_before_episode_subset"] = int(bundle.clean_inputs.shape[0])
    metadata["original_unique_episodes_before_episode_subset"] = int(unique_episodes.size)
    metadata["episode_subset_max_per_source"] = int(max_episodes_per_source)
    metadata["episode_subset_selected_episodes"] = int(len(selected))
    metadata["samples"] = int(mask.sum())
    if bundle.attack_mask is not None:
        metadata["attacked_samples"] = int(np.asarray(bundle.attack_mask, dtype=np.int64)[mask].sum())
    return PairDatasetBundle(
        adv_inputs=np.asarray(bundle.adv_inputs)[mask],
        clean_inputs=np.asarray(bundle.clean_inputs)[mask],
        metadata=metadata,
        clean_anchor_inputs=_slice_optional(bundle.clean_anchor_inputs, mask),
        time_indices=_slice_optional(bundle.time_indices, mask),
        stations=_slice_optional(bundle.stations, mask),
        is_new_arrivals=_slice_optional(bundle.is_new_arrivals, mask),
        vehicle_ids=_slice_optional(bundle.vehicle_ids, mask),
        episode_indices=episode_indices[mask],
        attack_mask=_slice_optional(bundle.attack_mask, mask),
    )


def main() -> None:
    configure_line_buffering()
    parser = argparse.ArgumentParser(description="Train detector only, reusing full-state DAE and cached pair datasets.")
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
    parser.add_argument("--pair-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday" / "pairs")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "det_seed42_fullstate")
    parser.add_argument("--detector-epochs", type=int, default=30)
    parser.add_argument("--detector-val-every", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--train-scenes-per-attack", type=int, default=100)
    parser.add_argument("--val-scenes-per-attack", type=int, default=20)
    parser.add_argument("--state-scope", choices=["local", "all"], default="all")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--benefit-margin", type=float, default=0.0)
    parser.add_argument("--benefit-action-weight", type=float, default=1.0)
    parser.add_argument("--benefit-state-weight", type=float, default=1.0)
    parser.add_argument("--use-benefit-sample-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-detector-datasets", action="store_true")
    args = parser.parse_args()

    if args.algorithm != "ddpg":
        layout = native_artifact_layout(PACKAGE_ROOT, args.algorithm, args.seed)
        if args.bundle_path == EP100_BUNDLE_PATH:
            args.bundle_path = default_native_bundle_path(PACKAGE_ROOT, args.algorithm, args.seed)
        legacy_dae = PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday"
        legacy_det = PACKAGE_ROOT / "artifacts" / "det_seed42_fullstate"
        if args.dae_artifact_dir == legacy_dae:
            args.dae_artifact_dir = layout["dae"]
        if args.pair_dir == legacy_dae / "pairs":
            args.pair_dir = layout["dae"] / "pairs"
        if args.output_dir == legacy_det:
            args.output_dir = layout["det"]

    if int(args.seed) != 42:
        raise ValueError("Detector-only training is fixed to seed=42 for the paper run.")
    if REPAIR_MODE != "full":
        raise ValueError(f"Detector-only full-state run expects REPAIR_MODE='full', got {REPAIR_MODE!r}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.seed)
    device = resolve_device(args.device)
    print(f"[DET TRAIN] device={device}, seed={args.seed}")

    backbone = load_frozen_backbone(args.algorithm, args.bundle_path, device)
    attack_plan = load_frozen_attack_plan(args.algorithm, args.native_config)
    actor = backbone.actor
    checkpoint_episode = int((backbone.payload.get("metadata") or {}).get("checkpoint_episode", -1))
    print(f"[DET TRAIN] algorithm={backbone.algorithm}, policy_action_mode={backbone.policy_action_mode}")

    dae_path = args.dae_artifact_dir / "dtsr_dae.pt"
    dae = load_dae(dae_path, device).eval()
    dae_manifest_path = args.dae_artifact_dir / "dae_manifest.json"
    dae_manifest = json.loads(dae_manifest_path.read_text(encoding="utf-8")) if dae_manifest_path.exists() else {}
    validate_dataset_backbone(dae_manifest, backbone, split="train")
    validate_attack_plan_provenance(dae_manifest, attack_plan)
    print(f"[DET TRAIN] DAE={dae_path}, best_epoch={dae_manifest.get('best_epoch')}")

    pair_train_path = args.pair_dir / "pair_train_unified.npz"
    pair_val_path = args.pair_dir / "pair_val_unified.npz"
    print(f"[DET TRAIN] Loading pairs: {pair_train_path} / {pair_val_path}")
    unified_train = load_pair_dataset(pair_train_path)
    unified_val = load_pair_dataset(pair_val_path)
    validate_dataset_backbone(unified_train.metadata, backbone, split="train")
    validate_dataset_backbone(unified_val.metadata, backbone, split="val")
    validate_attack_plan_provenance(unified_train.metadata, attack_plan)
    validate_attack_plan_provenance(unified_val.metadata, attack_plan)
    attacks = list((unified_train.metadata or {}).get("train_attacks") or ["opposite_pgd", "q_function"])
    print(f"[DET TRAIN] pair train samples={unified_train.clean_inputs.shape[0]}, val samples={unified_val.clean_inputs.shape[0]}")
    unified_train = subset_pair_bundle_by_episode(unified_train, args.train_scenes_per_attack)
    unified_val = subset_pair_bundle_by_episode(unified_val, args.val_scenes_per_attack)
    print(
        "[DET TRAIN] pair subset samples="
        f"train={unified_train.clean_inputs.shape[0]} "
        f"(scenes/source={args.train_scenes_per_attack}), "
        f"val={unified_val.clean_inputs.shape[0]} "
        f"(scenes/source={args.val_scenes_per_attack})"
    )

    print("[DET TRAIN] Building posterior detector train dataset ...")
    detector_train = posterior_detector_dataset_from_unified_pair(
        unified_train,
        actor,
        dae,
        device,
        profile_tag="det_only_seed42_train",
        train_attack_tags=attacks,
        benefit_margin=args.benefit_margin,
        benefit_action_weight=args.benefit_action_weight,
        benefit_state_weight=args.benefit_state_weight,
        posterior_label_mode="benefit",
        use_benefit_sample_weights=bool(args.use_benefit_sample_weights),
        state_scope=args.state_scope,
        repair_mode=REPAIR_MODE,
    )
    print("[DET TRAIN] Building posterior detector validation dataset ...")
    detector_val = posterior_detector_dataset_from_unified_pair(
        unified_val,
        actor,
        dae,
        device,
        profile_tag="det_only_seed42_val",
        train_attack_tags=attacks,
        benefit_margin=args.benefit_margin,
        benefit_action_weight=args.benefit_action_weight,
        benefit_state_weight=args.benefit_state_weight,
        posterior_label_mode="benefit",
        use_benefit_sample_weights=False,
        state_scope=args.state_scope,
        repair_mode=REPAIR_MODE,
    )
    print(
        "[DET TRAIN] detector train samples="
        f"{detector_train.obs_inputs.shape[0]}, positive_rate={detector_train.metadata.get('label_positive_rate'):.4f}"
    )
    print(
        "[DET TRAIN] detector val samples="
        f"{detector_val.obs_inputs.shape[0]}, positive_rate={detector_val.metadata.get('label_positive_rate'):.4f}"
    )
    if args.save_detector_datasets:
        np.savez_compressed(args.output_dir / "detector_train_dataset.npz", metadata=detector_train.metadata)
        np.savez_compressed(args.output_dir / "detector_val_dataset.npz", metadata=detector_val.metadata)

    detector, detector_result = train_detector_from_bundle(
        detector_train,
        actor,
        dae,
        device,
        epochs=args.detector_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        val_ratio=args.val_ratio,
        detector_temporal=True,
        detector_feature_mode="posterior",
        seed=args.seed,
        state_scope=args.state_scope,
        progress_dir=args.output_dir,
        progress_prefix="detector",
        val_every=args.detector_val_every,
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
            "algorithm": backbone.algorithm,
            "backbone": backbone.provenance(),
            "attack_plan": attack_plan.provenance(),
            "policy": str(backbone.bundle_path.resolve()),
            "policy_checkpoint_episode": checkpoint_episode,
            "seed": int(args.seed),
            "train_attacks": attacks,
            "epsilon": float(args.epsilon),
            "state_scope": args.state_scope,
            "repair_mode": REPAIR_MODE,
            "posterior_candidate_state": "full_reconstruction",
            "dae_artifact": str(dae_path),
            "dae_best_epoch": dae_manifest.get("best_epoch"),
            "best_epoch": int(detector_result.best_epoch),
            "best_metric_name": str(detector_result.best_metric_name),
            "best_metric_value": float(detector_result.best_metric_value),
            "val_every": int(args.detector_val_every),
            "threshold_selection": "external validation benefit-F1 with FPR<=0.10 preference",
            "train_scenes_per_attack": int(args.train_scenes_per_attack),
            "val_scenes_per_attack": int(args.val_scenes_per_attack),
        },
        history={"threshold_report": threshold_report},
    )
    save_detector_history(detector_result, args.output_dir / "detector_history.csv")
    write_json(args.output_dir / "detector_threshold_report.json", threshold_report)

    manifest = {
        "status": "trained",
        "seed": int(args.seed),
        "algorithm": backbone.algorithm,
        "backbone": backbone.provenance(),
        "attack_plan": attack_plan.provenance(),
        "policy_checkpoint_episode": checkpoint_episode,
        "dae": str(dae_path),
        "dae_best_epoch": dae_manifest.get("best_epoch"),
        "detector": str(detector_path),
        "detector_max_epochs": int(args.detector_epochs),
        "detector_val_every": int(args.detector_val_every),
        "detector_best_epoch": int(detector_result.best_epoch),
        "detector_best_metric_name": str(detector_result.best_metric_name),
        "detector_best_metric_value": float(detector_result.best_metric_value),
        "detector_threshold": threshold_report,
        "train_pair_samples": int(unified_train.clean_inputs.shape[0]),
        "val_pair_samples": int(unified_val.clean_inputs.shape[0]),
        "train_scenes_per_attack": int(args.train_scenes_per_attack),
        "val_scenes_per_attack": int(args.val_scenes_per_attack),
        "detector_train_samples": int(detector_train.obs_inputs.shape[0]),
        "detector_val_samples": int(detector_val.obs_inputs.shape[0]),
        "detector_train_positive_rate": float(detector_train.metadata.get("label_positive_rate", 0.0)),
        "detector_val_positive_rate": float(detector_val.metadata.get("label_positive_rate", 0.0)),
        "train_attacks": attacks,
        "state_scope": args.state_scope,
        "repair_mode": REPAIR_MODE,
        "posterior_label_mode": "benefit",
        "benefit_margin": float(args.benefit_margin),
        "benefit_action_weight": float(args.benefit_action_weight),
        "benefit_state_weight": float(args.benefit_state_weight),
        "use_benefit_sample_weights": bool(args.use_benefit_sample_weights),
    }
    write_json(args.output_dir / "det_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("[DET TRAIN] Done.")


if __name__ == "__main__":
    main()
