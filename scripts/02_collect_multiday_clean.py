from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from _common import (
    PACKAGE_ROOT,
    deterministic_subset,
    load_manifest,
    load_scenario,
    resolve_device,
    write_json,
)

sys.path.insert(0, str(PACKAGE_ROOT))
from dtsr_multiday_common import EP100_ACTOR_PATH, EP100_BUNDLE_PATH, set_all_seeds
from evc.merged_core import TRAIN_PROFILE
from evc.merged_pipeline import (
    CleanTrajectoryBundle,
    collect_clean_trajectories,
    save_clean_trajectory_dataset,
)
from evc.native_dtsr import (
    SUPPORTED_BACKBONES,
    default_native_bundle_path,
    load_frozen_backbone,
    native_artifact_layout,
)


def collect_split(split, scenario_count, max_samples, seed, backbone, device):
    actor = backbone.actor
    manifest = deterministic_subset(load_manifest(split), scenario_count, seed)
    clean_inputs = []
    raw_prices = []
    time_indices = []
    stations = []
    is_new_arrivals = []
    vehicle_ids = []
    episode_indices = []
    total = 0
    scenario_ids = []

    for episode_id, row in manifest.iterrows():
        arrivals, signal_path, scenario_id = load_scenario(row)
        remaining = None if max_samples is None else max(int(max_samples) - total, 0)
        if remaining == 0:
            break
        bundle = collect_clean_trajectories(
            arrivals,
            actor,
            signal_path,
            device,
            reward_profile=TRAIN_PROFILE,
            episodes=1,
            max_samples=remaining,
        )
        count = int(bundle.clean_inputs.shape[0])
        scenario_ids.append(str(scenario_id))
        clean_inputs.append(bundle.clean_inputs)
        raw_prices.append(bundle.raw_prices)
        time_indices.append(bundle.time_indices)
        stations.append(bundle.stations)
        is_new_arrivals.append(bundle.is_new_arrivals)
        vehicle_ids.append(bundle.vehicle_ids)
        episode_indices.append(np.full((count,), episode_id, dtype=np.int64))
        total += count
        print(f"[{split}] {episode_id + 1}/{len(manifest)} {scenario_id}: total_samples={total}")

    if not clean_inputs:
        raise RuntimeError(f"No clean trajectories were collected for split={split}")

    return CleanTrajectoryBundle(
        clean_inputs=np.concatenate(clean_inputs, axis=0).astype(np.float32),
        metadata={
            "samples": total,
            "split": split,
            "scenario_count": len(clean_inputs),
            "collection_mode": "multiday_clean_rollout",
            "policy": str(backbone.bundle_path.resolve()),
            "backbone": backbone.provenance(),
            "reward_profile": TRAIN_PROFILE.name,
            "selection_seed": int(seed),
            "scenario_ids": scenario_ids,
        },
        raw_prices=np.concatenate(raw_prices, axis=0).astype(np.float32),
        time_indices=np.concatenate(time_indices, axis=0).astype(np.int64),
        stations=np.concatenate(stations, axis=0).astype(np.int64),
        is_new_arrivals=np.concatenate(is_new_arrivals, axis=0).astype(np.int64),
        vehicle_ids=np.concatenate(vehicle_ids, axis=0).astype(np.int64),
        episode_indices=np.concatenate(episode_indices, axis=0).astype(np.int64),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--algorithm", choices=SUPPORTED_BACKBONES, default="ddpg")
    parser.add_argument("--bundle-path", type=Path, default=EP100_BUNDLE_PATH)
    parser.add_argument(
        "--actor-path",
        type=Path,
        default=EP100_ACTOR_PATH,
        help="Deprecated compatibility option; the frozen bundle is authoritative.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-scenes", type=int, default=200)
    parser.add_argument("--val-scenes", type=int, default=60)
    parser.add_argument("--train-max-samples", type=int, default=300000)
    parser.add_argument("--val-max-samples", type=int, default=120000)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "artifacts" / "clean")
    args = parser.parse_args()

    legacy_output = PACKAGE_ROOT / "artifacts" / "clean"
    if args.algorithm != "ddpg" and args.output_dir == legacy_output:
        args.output_dir = native_artifact_layout(PACKAGE_ROOT, args.algorithm, args.seed)["clean"]
    if args.algorithm != "ddpg" and args.bundle_path == EP100_BUNDLE_PATH:
        args.bundle_path = default_native_bundle_path(PACKAGE_ROOT, args.algorithm, args.seed)

    set_all_seeds(args.seed)
    device = resolve_device(args.device)
    backbone = load_frozen_backbone(args.algorithm, args.bundle_path, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_bundle = collect_split(
        "train", args.train_scenes, args.train_max_samples, args.seed, backbone, device
    )
    val_bundle = collect_split(
        "val", args.val_scenes, args.val_max_samples, args.seed + 1, backbone, device
    )

    train_path = save_clean_trajectory_dataset(train_bundle, args.output_dir / "clean_train.npz")
    val_path = save_clean_trajectory_dataset(val_bundle, args.output_dir / "clean_val.npz")
    write_json(
        args.output_dir / "clean_dataset_audit.json",
        {
            "seed": int(args.seed),
            "backbone": backbone.provenance(),
            "obs_dim": int(train_bundle.clean_inputs.shape[1]),
            "train_sample_count": int(train_bundle.clean_inputs.shape[0]),
            "val_sample_count": int(val_bundle.clean_inputs.shape[0]),
            "train_scenario_count": int(train_bundle.metadata["scenario_count"]),
            "val_scenario_count": int(val_bundle.metadata["scenario_count"]),
            "train_scenario_ids": list(train_bundle.metadata.get("scenario_ids") or []),
            "val_scenario_ids": list(val_bundle.metadata.get("scenario_ids") or []),
        },
    )
    print(f"Saved train Dnormal: {train_path}")
    print(f"Saved validation Dnormal: {val_path}")


if __name__ == "__main__":
    main()
