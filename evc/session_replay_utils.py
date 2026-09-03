from __future__ import annotations

from typing import Sequence

import numpy as np

from .merged_pipeline import PairDatasetBundle


def _episode_offset(bundle: PairDatasetBundle) -> int:
    episodes = np.asarray(
        bundle.episode_indices if bundle.episode_indices is not None else np.zeros((0,), dtype=np.int64),
        dtype=np.int64,
    ).reshape(-1)
    return int(np.max(episodes)) + 1 if episodes.size else 1


def merge_pair_bundles_preserve_sessions(
    bundles: Sequence[PairDatasetBundle],
    *,
    attack_tags: Sequence[str] | None = None,
) -> PairDatasetBundle:
    valid = [bundle for bundle in bundles if int(np.asarray(bundle.clean_inputs).reshape(-1, 11).shape[0]) > 0]
    if not valid:
        empty = np.zeros((0, 11), dtype=np.float32)
        return PairDatasetBundle(empty, empty.copy(), {'samples': 0, 'attacked_samples': 0})

    adv_inputs: list[np.ndarray] = []
    clean_inputs: list[np.ndarray] = []
    time_indices: list[np.ndarray] = []
    stations: list[np.ndarray] = []
    is_new_arrivals: list[np.ndarray] = []
    vehicle_ids: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    attack_masks: list[np.ndarray] = []
    offset = 0

    for bundle in valid:
        adv = np.asarray(bundle.adv_inputs, dtype=np.float32).reshape(-1, 11)
        clean = np.asarray(bundle.clean_inputs, dtype=np.float32).reshape(-1, 11)
        count = int(clean.shape[0])
        if adv.shape != clean.shape:
            raise ValueError('pair bundle adv_inputs and clean_inputs must align.')
        adv_inputs.append(adv)
        clean_inputs.append(clean)
        time_indices.append(
            np.asarray(bundle.time_indices, dtype=np.int64).reshape(-1)
            if bundle.time_indices is not None
            else np.arange(count, dtype=np.int64)
        )
        stations.append(
            np.asarray(bundle.stations, dtype=np.int64).reshape(-1)
            if bundle.stations is not None
            else np.zeros((count,), dtype=np.int64)
        )
        is_new_arrivals.append(
            np.asarray(bundle.is_new_arrivals, dtype=np.int64).reshape(-1)
            if bundle.is_new_arrivals is not None
            else np.zeros((count,), dtype=np.int64)
        )
        vehicle_ids.append(
            np.asarray(bundle.vehicle_ids, dtype=np.int64).reshape(-1)
            if bundle.vehicle_ids is not None
            else np.arange(count, dtype=np.int64)
        )
        raw_episode = (
            np.asarray(bundle.episode_indices, dtype=np.int64).reshape(-1)
            if bundle.episode_indices is not None
            else np.zeros((count,), dtype=np.int64)
        )
        episode_indices.append(raw_episode + int(offset))
        attack_masks.append(
            np.asarray(bundle.attack_mask, dtype=np.int64).reshape(-1)
            if bundle.attack_mask is not None
            else (np.max(np.abs(adv - clean), axis=1) > 1e-8).astype(np.int64)
        )
        offset += max(_episode_offset(bundle), 1)

    merged_adv = np.concatenate(adv_inputs, axis=0)
    merged_clean = np.concatenate(clean_inputs, axis=0)
    merged_mask = np.concatenate(attack_masks, axis=0)
    return PairDatasetBundle(
        adv_inputs=merged_adv,
        clean_inputs=merged_clean,
        metadata={
            'collection_mode': 'offline_session_dae_adaptive_replay_pool',
            'train_attacks': list(attack_tags or []),
            'samples': int(merged_clean.shape[0]),
            'attacked_samples': int(merged_mask.sum()),
            'source_bundle_count': int(len(valid)),
        },
        time_indices=np.concatenate(time_indices, axis=0),
        stations=np.concatenate(stations, axis=0),
        is_new_arrivals=np.concatenate(is_new_arrivals, axis=0),
        vehicle_ids=np.concatenate(vehicle_ids, axis=0),
        episode_indices=np.concatenate(episode_indices, axis=0),
        attack_mask=merged_mask,
    )
