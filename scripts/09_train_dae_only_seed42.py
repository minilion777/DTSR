"""DAE-only training script: PGD* + Q-function* pairs → unified GRU-VAE → best checkpoint.

Follows Codex checklist Phase 1 Steps 8-18.
Freezes ep100 DDPG, uses train/val bounds per split, selects by full-state DAE recovery score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from _common import (
    PACKAGE_ROOT,
    resolve_device,
    write_json,
)

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.dtsr_datasets import merge_pair_bundles_for_unified
from dtsr_multiday_common import (
    EP100_ACTOR_PATH,
    EP100_BUNDLE_PATH,
    REPAIR_MODE,
    runtime_dae_validation_metrics,
    set_all_seeds,
    union_observation_bounds,
)
from evc.defense import (
    DenoisingAutoencoder,
    save_dae,
    save_dae_history,
)
from evc.merged_pipeline import (
    CleanTrajectoryBundle,
    PairDatasetBundle,
    build_pair_dataset_from_clean_trajectories,
    load_clean_trajectory_dataset,
    load_pair_dataset,
    save_pair_dataset,
    train_dae_from_bundle,
)
from evc.native_dtsr import (
    SUPPORTED_BACKBONES,
    FrozenAttackPlan,
    FrozenBackbone,
    build_frozen_attacker,
    default_native_bundle_path,
    load_frozen_attack_plan,
    load_frozen_backbone,
    native_artifact_layout,
    validate_dataset_backbone,
)


# ── Fixed attack parameters (checklist §9) ──────────────────────────
def configure_line_buffering() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)


# ── Fixed DAE hyperparameters (checklist §12, §14) ──────────────────
DAE_ARCH = dict(seq_len=8, hidden_dim=128, latent_dim=64, decoder_hidden_dim=128, num_layers=1)
DAE_TRAIN = dict(
    max_epochs=50, batch_size=512, lr=1e-3,
    lambda_state=1.0, lambda_identity=1.0, beta_kl=1e-3, lambda_robust=0.2,
    include_clean_sequences=True, state_scope="all",
)


def build_offline_attacker(
    backbone: FrozenBackbone,
    attack_plan: FrozenAttackPlan,
    device,
    attack_key: str,
    seed: int,
    split: str,
):
    """Create the frozen backbone-native attacker using split-only bounds."""
    low, high = union_observation_bounds(split)
    return build_frozen_attacker(
        attack_key,
        backbone=backbone,
        attack_plan=attack_plan,
        device=device,
        obs_low=low,
        obs_high=high,
        seed=int(seed),
    )


def build_pair(clean_bundle: CleanTrajectoryBundle, attacker) -> PairDatasetBundle:
    """Offline attack on clean trajectories → aligned pair dataset."""
    return build_pair_dataset_from_clean_trajectories(
        clean_bundle,
        attacker,
        attack_scenario="O",  # irrelevant for PGD/Q attacks (target not used)
        attack_ratio=1.0,
        attack_scope="obs",
    )


def make_dae_validator(actor, device, val_bundle: PairDatasetBundle):
    """Closure that returns full-state validation metrics for current DAE state."""
    clean = val_bundle.clean_inputs
    adv = val_bundle.adv_inputs
    ep_idx = val_bundle.episode_indices
    veh_ids = val_bundle.vehicle_ids
    mask = getattr(val_bundle, "attack_mask", None)

    def validator(model: torch.nn.Module) -> dict:
        return runtime_dae_validation_metrics(
            model,
            clean_inputs=clean,
            adv_inputs=adv,
            actor=actor,
            device=device,
            episode_indices=ep_idx,
            vehicle_ids=veh_ids,
            attack_mask=mask,
        )
    return validator


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


def save_manifest(
    path: Path,
    *,
    best_epoch: int,
    best_score: float,
    best_result: dict,
    max_epochs: int,
    seed: int,
    backbone: FrozenBackbone,
    attack_plan: FrozenAttackPlan,
) -> None:
    write_json(path, {
        "status": "trained",
        "seed": int(seed),
        "algorithm": backbone.algorithm,
        "backbone": backbone.provenance(),
        "attack_plan": attack_plan.provenance(),
        "train_attacks": ["opposite_pgd", "q_function"],
        "epsilon": 0.1,
        "state_scope": "all",
        "repair_mode": REPAIR_MODE,
        "repaired_state_dimensions": "all_11",
        **DAE_ARCH,
        "max_epochs": max_epochs,
        "best_epoch": best_epoch,
        "selection_metric": "dae_checkpoint_score",
        "train_bounds_source": "train union",
        "val_bounds_source": "val union",
        "test_used_for_training": False,
        "best_dae_checkpoint_score": best_score,
        "best_dae_action_attack_mse": best_result.get("dae_action_attack_mse"),
        "best_dae_action_recovered_mse": best_result.get("dae_action_recovered_mse"),
        "best_dae_action_mse_reduction_pct": best_result.get("dae_action_mse_reduction_pct"),
        "best_dae_clean_identity_action_mse": best_result.get("dae_clean_identity_action_mse"),
    })


def main():
    configure_line_buffering()
    parser = argparse.ArgumentParser(description="DAE-only training (PGD* + Q*)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algorithm", choices=SUPPORTED_BACKBONES, default="ddpg")
    parser.add_argument("--actor-path", default=str(EP100_ACTOR_PATH), help="Deprecated; the bundle is authoritative.")
    parser.add_argument("--bundle-path", type=Path, default=EP100_BUNDLE_PATH)
    parser.add_argument(
        "--native-config",
        type=Path,
        default=PACKAGE_ROOT / "results" / "native_attack_calibration_seed42" / "native_attack_config.json",
    )
    parser.add_argument("--clean-train", type=Path, default=PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday" / "clean" / "clean_train.npz")
    parser.add_argument("--clean-val", type=Path, default=PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday" / "clean" / "clean_val.npz")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday")
    parser.add_argument("--save-pair-datasets", action="store_true", default=True)
    parser.add_argument("--skip-pair-generation", action="store_true", default=False,
                        help="Skip pair generation if unified pairs already exist on disk")
    parser.add_argument("--dae-epochs", type=int, default=DAE_TRAIN["max_epochs"],
                        help=f"Max DAE training epochs (default: {DAE_TRAIN['max_epochs']})")
    parser.add_argument("--dae-val-every", type=int, default=5,
                        help="Run DAE checkpoint validation every N epochs.")
    parser.add_argument("--dae-validator-scenes", type=int, default=20,
                        help="Use the first N validation episodes per attack source for DAE checkpoint selection; <=0 uses all.")
    args = parser.parse_args()

    if args.algorithm != "ddpg":
        layout = native_artifact_layout(PACKAGE_ROOT, args.algorithm, args.seed)
        if args.bundle_path == EP100_BUNDLE_PATH:
            args.bundle_path = default_native_bundle_path(PACKAGE_ROOT, args.algorithm, args.seed)
        legacy_dae = PACKAGE_ROOT / "artifacts" / "dae_seed42_multiday"
        if args.output_dir == legacy_dae:
            args.output_dir = layout["dae"]
        legacy_clean = legacy_dae / "clean"
        if args.clean_train == legacy_clean / "clean_train.npz":
            args.clean_train = layout["clean"] / "clean_train.npz"
        if args.clean_val == legacy_clean / "clean_val.npz":
            args.clean_val = layout["clean"] / "clean_val.npz"

    device = resolve_device(args.device)
    seed = args.seed
    set_all_seeds(seed)

    output_dir = Path(args.output_dir)
    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DAE TRAIN] device={device}, seed={seed}")

    # ── 1. Load frozen DDPG ───────────────────────────────────────
    print(f"[DAE TRAIN] Loading frozen {args.algorithm.upper()} backbone ...")
    backbone = load_frozen_backbone(args.algorithm, args.bundle_path, device)
    attack_plan = load_frozen_attack_plan(args.algorithm, args.native_config)
    actor = backbone.actor
    print(f"[DAE TRAIN] policy_action_mode={backbone.policy_action_mode}")

    # ── 2. Load clean datasets ────────────────────────────────────
    print("[DAE TRAIN] Loading clean datasets ...")
    clean_train = load_clean_trajectory_dataset(args.clean_train)
    clean_val   = load_clean_trajectory_dataset(args.clean_val)
    validate_dataset_backbone(clean_train.metadata, backbone, split="train")
    validate_dataset_backbone(clean_val.metadata, backbone, split="val")
    print(f"  train: {clean_train.clean_inputs.shape[0]} samples")
    print(f"  val:   {clean_val.clean_inputs.shape[0]} samples")

    # ── 3-5. Generate or load attack pairs ─────────────────────────
    unified_train_path = pairs_dir / "pair_train_unified.npz"
    unified_val_path   = pairs_dir / "pair_val_unified.npz"

    if args.skip_pair_generation and unified_train_path.exists() and unified_val_path.exists():
        print("[DAE TRAIN] Skipping pair generation — loading existing unified pairs ...")
        unified_train = load_pair_dataset(str(unified_train_path))
        unified_val   = load_pair_dataset(str(unified_val_path))
    else:
        print("[DAE TRAIN] Building attackers ...")
        pgd_train_att = build_offline_attacker(backbone, attack_plan, device, "opposite_pgd", seed + 100_000, "train")
        q_train_att   = build_offline_attacker(backbone, attack_plan, device, "q_function", seed + 200_000, "train")
        pgd_val_att   = build_offline_attacker(backbone, attack_plan, device, "opposite_pgd", seed + 1_100_000, "val")
        q_val_att     = build_offline_attacker(backbone, attack_plan, device, "q_function", seed + 1_200_000, "val")

        print("[DAE TRAIN] Generating PGD train pairs ...")
        pgd_train = build_pair(clean_train, pgd_train_att)
        print(f"  PGD train: {pgd_train.clean_inputs.shape[0]} samples")
        print("[DAE TRAIN] Generating Q-function train pairs ...")
        q_train = build_pair(clean_train, q_train_att)
        print(f"  Q train:   {q_train.clean_inputs.shape[0]} samples")
        print("[DAE TRAIN] Generating PGD val pairs ...")
        pgd_val = build_pair(clean_val, pgd_val_att)
        print(f"  PGD val:   {pgd_val.clean_inputs.shape[0]} samples")
        print("[DAE TRAIN] Generating Q-function val pairs ...")
        q_val = build_pair(clean_val, q_val_att)
        print(f"  Q val:     {q_val.clean_inputs.shape[0]} samples")

        print("[DAE TRAIN] Merging unified train / val ...")
        unified_train = merge_pair_bundles_for_unified(
            [pgd_train, q_train],
            attack_tags=["opposite_pgd", "q_function"],
        )
        unified_val = merge_pair_bundles_for_unified(
            [pgd_val, q_val],
            attack_tags=["opposite_pgd", "q_function"],
        )
        unified_train.metadata.update(
            {
                "split": "train",
                "backbone": backbone.provenance(),
                "attack_plan": attack_plan.provenance(),
            }
        )
        unified_val.metadata.update(
            {
                "split": "val",
                "backbone": backbone.provenance(),
                "attack_plan": attack_plan.provenance(),
            }
        )
        if args.save_pair_datasets:
            save_pair_dataset(pgd_train, str(pairs_dir / "pair_train_pgd.npz"))
            save_pair_dataset(q_train,   str(pairs_dir / "pair_train_q.npz"))
            save_pair_dataset(pgd_val,   str(pairs_dir / "pair_val_pgd.npz"))
            save_pair_dataset(q_val,     str(pairs_dir / "pair_val_q.npz"))
            save_pair_dataset(unified_train, str(unified_train_path))
            save_pair_dataset(unified_val,   str(unified_val_path))
            print("[DAE TRAIN] Pair datasets saved.")

    validate_dataset_backbone(unified_train.metadata, backbone, split="train")
    validate_dataset_backbone(unified_val.metadata, backbone, split="val")
    print(f"  unified train: {unified_train.clean_inputs.shape[0]} samples")
    print(f"  unified val:   {unified_val.clean_inputs.shape[0]} samples")

    # ── 6. Train DAE ──────────────────────────────────────────────
    print("[DAE TRAIN] Training DAE ...")
    validator_bundle = subset_pair_bundle_episodes(unified_val, args.dae_validator_scenes)
    validator_episodes = (
        len(np.unique(validator_bundle.episode_indices))
        if validator_bundle.episode_indices is not None
        else "all"
    )
    print(f"  DAE validator: {validator_bundle.clean_inputs.shape[0]} samples from {validator_episodes} episodes")
    validator = make_dae_validator(actor, device, validator_bundle)

    dae, result = train_dae_from_bundle(
        unified_train,
        actor,
        device=device,
        epochs=args.dae_epochs,
        batch_size=DAE_TRAIN["batch_size"],
        lr=DAE_TRAIN["lr"],
        lambda_state=DAE_TRAIN["lambda_state"],
        lambda_identity=DAE_TRAIN["lambda_identity"],
        validator=validator,
        val_every=args.dae_val_every,
        select_by="dae_checkpoint_score",
        log_every=1,
        progress_dir=output_dir,
        progress_prefix="dae",
        **DAE_ARCH,
        beta_kl=DAE_TRAIN["beta_kl"],
        lambda_robust=DAE_TRAIN["lambda_robust"],
        include_clean_sequences=DAE_TRAIN["include_clean_sequences"],
        state_scope=DAE_TRAIN["state_scope"],
    )

    print(f"[DAE TRAIN] Best epoch: {result.best_epoch}, score: {result.best_metric_value:.6f}")

    # ── 7. Save artifacts ─────────────────────────────────────────
    dae_path = output_dir / "dtsr_dae.pt"
    save_dae(dae, str(dae_path))
    # Also save best epoch backup
    best_path = output_dir / f"dtsr_dae_epoch{result.best_epoch}.pt"
    save_dae(dae, str(best_path))
    print(f"[DAE TRAIN] Saved DAE to {dae_path}")

    # history CSVs
    hist_path = save_dae_history(result, str(output_dir / "dae_history.csv"))
    print(f"[DAE TRAIN] dae_history.csv saved to {hist_path}")

    val_rows = result.validator_rows
    if val_rows:
        import pandas as pd
        val_df = pd.DataFrame(val_rows)
        val_df.to_csv(output_dir / "dae_validation_history.csv", index=False)
        print(f"[DAE TRAIN] dae_validation_history.csv: {len(val_df)} rows")

    # best epoch JSON
    best_result = {}
    best_val_idx = result.best_epoch // max(int(args.dae_val_every), 1) - 1
    if val_rows and 0 <= best_val_idx < len(val_rows):
        best_result = val_rows[best_val_idx]
    write_json(output_dir / "dae_best_epoch.json", {
        "max_epochs": args.dae_epochs,
        "best_epoch": result.best_epoch,
        "selection_metric": "dae_checkpoint_score",
        "validation_every_epochs": int(args.dae_val_every),
        "validator_episode_limit": int(args.dae_validator_scenes),
        "best_dae_checkpoint_score": result.best_metric_value,
        "best_dae_action_attack_mse": best_result.get("dae_action_attack_mse"),
        "best_dae_action_recovered_mse": best_result.get("dae_action_recovered_mse"),
        "best_dae_action_mse_reduction_pct": best_result.get("dae_action_mse_reduction_pct"),
        "best_dae_clean_identity_action_mse": best_result.get("dae_clean_identity_action_mse"),
    })

    # manifest
    save_manifest(
        output_dir / "dae_manifest.json",
        best_epoch=result.best_epoch,
        best_score=result.best_metric_value,
        best_result=best_result,
        max_epochs=args.dae_epochs,
        seed=seed,
        backbone=backbone,
        attack_plan=attack_plan,
    )

    print("[DAE TRAIN] Done.")
    return dae, result


if __name__ == "__main__":
    main()
