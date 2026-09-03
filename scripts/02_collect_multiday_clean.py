from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from _common import PACKAGE_ROOT, deterministic_subset, load_manifest, load_scenario, resolve_device, write_json

sys.path.insert(0, str(PACKAGE_ROOT))
from evc.merged_core import TRAIN_PROFILE, load_actor_from_path, set_seed
from evc.merged_pipeline import CleanTrajectoryBundle, collect_clean_trajectories, save_clean_trajectory_dataset


def collect_split(split: str, scenes: int, max_samples: int, seed: int, actor, device) -> CleanTrajectoryBundle:
    pieces = []
    scenario_ids = []
    total = 0
    for episode_index, row in deterministic_subset(load_manifest(split), scenes, seed).iterrows():
        remaining = max(int(max_samples) - total, 0)
        if remaining == 0:
            break
        arrivals, signal_path, scenario_id = load_scenario(row)
        piece = collect_clean_trajectories(arrivals, actor, signal_path, device, TRAIN_PROFILE, episodes=1, max_samples=remaining)
        piece.episode_indices = np.full((len(piece.clean_inputs),), int(episode_index), dtype=np.int64)
        pieces.append(piece)
        scenario_ids.append(scenario_id)
        total += len(piece.clean_inputs)
    if not pieces:
        raise RuntimeError(f"No {split} trajectories were collected.")
    return CleanTrajectoryBundle(
        clean_inputs=np.concatenate([item.clean_inputs for item in pieces]),
        metadata={"split": split, "scenario_count": len(pieces), "scenario_ids": scenario_ids, "samples": total},
        raw_prices=np.concatenate([item.raw_prices for item in pieces]),
        time_indices=np.concatenate([item.time_indices for item in pieces]),
        stations=np.concatenate([item.stations for item in pieces]),
        is_new_arrivals=np.concatenate([item.is_new_arrivals for item in pieces]),
        vehicle_ids=np.concatenate([item.vehicle_ids for item in pieces]),
        episode_indices=np.concatenate([item.episode_indices for item in pieces]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect clean DDPG trajectories from paired scenarios.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--actor-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-scenes", type=int, default=500)
    parser.add_argument("--val-scenes", type=int, default=60)
    parser.add_argument("--train-max-samples", type=int, default=300000)
    parser.add_argument("--val-max-samples", type=int, default=120000)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "runs" / "clean")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    actor = load_actor_from_path(args.actor_path, device).eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = collect_split("train", args.train_scenes, args.train_max_samples, args.seed, actor, device)
    val = collect_split("val", args.val_scenes, args.val_max_samples, args.seed + 1, actor, device)
    save_clean_trajectory_dataset(train, args.output_dir / "clean_train.npz")
    save_clean_trajectory_dataset(val, args.output_dir / "clean_val.npz")
    write_json(args.output_dir / "clean_manifest.json", {"train_samples": len(train.clean_inputs), "val_samples": len(val.clean_inputs)})


if __name__ == "__main__":
    main()
