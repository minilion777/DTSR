from __future__ import annotations

import numpy as np
import torch

from .defense import (
    build_previous_step_inputs,
    canonical_state_scope,
    dae_reconstruction_with_history,
    defended_indices_for_scope,
    weighted_state_error_np,
)
from .merged_pipeline import DetectorDatasetBundle, PairDatasetBundle
from .offline_dae_det_temporal_shield import LOCAL_SHIELD_INDICES


def _episode_offset_for_bundle(bundle: PairDatasetBundle) -> int:
    episodes = np.asarray(
        getattr(bundle, "episode_indices", np.zeros((0,), dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1)
    return int(np.max(episodes)) + 1 if episodes.size else 1


def merge_pair_bundles_for_unified(
    bundles: list[PairDatasetBundle],
    *,
    attack_tags: list[str],
) -> PairDatasetBundle:
    if not bundles:
        raise ValueError("Unified DAE training requires at least one pair bundle.")
    if len(bundles) != len(attack_tags):
        raise ValueError("Each pair bundle requires one attack tag.")
    source_scopes = [
        str((bundle.metadata or {}).get("state_scope", "local")) for bundle in bundles
    ]
    state_scope = canonical_state_scope(source_scopes[0])
    if any(canonical_state_scope(scope) != state_scope for scope in source_scopes):
        raise ValueError(
            f"Unified pair merge requires one state_scope, got {source_scopes!r}."
        )
    adv_inputs: list[np.ndarray] = []
    clean_inputs: list[np.ndarray] = []
    time_indices: list[np.ndarray] = []
    stations: list[np.ndarray] = []
    is_new_arrivals: list[np.ndarray] = []
    vehicle_ids: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    attack_masks: list[np.ndarray] = []
    episode_offset = 0
    for bundle in bundles:
        adv = np.asarray(bundle.adv_inputs, dtype=np.float32).reshape(-1, 11)
        clean = np.asarray(bundle.clean_inputs, dtype=np.float32).reshape(-1, 11)
        count = int(clean.shape[0])
        if adv.shape != clean.shape:
            raise ValueError("Unified pair merge requires aligned adv_inputs and clean_inputs.")
        adv_inputs.append(adv)
        clean_inputs.append(clean)
        time_indices.append(
            np.asarray(bundle.time_indices, dtype=np.int64).reshape(-1)
            if bundle.time_indices is not None
            else np.zeros((count,), dtype=np.int64)
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
        episode_indices.append(raw_episode + int(episode_offset))
        attack_masks.append(
            np.asarray(bundle.attack_mask, dtype=np.int64).reshape(-1)
            if bundle.attack_mask is not None
            else (np.max(np.abs(adv - clean), axis=1) > 1e-8).astype(np.int64)
        )
        episode_offset += max(_episode_offset_for_bundle(bundle), 1)
    merged_metadata = {
        "collection_mode": "unified_offline_attack_from_dnormal",
        "train_attacks": list(attack_tags),
        "samples": int(sum(arr.shape[0] for arr in clean_inputs)),
        "attacked_samples": int(sum(int(mask.sum()) for mask in attack_masks)),
        "source_bundle_count": int(len(bundles)),
        "policy_input_mode": "clean",
        "attack_trigger_mode": f"candidate_all_{state_scope}_obs",
        "state_scope": state_scope,
        "attack_state_scope": state_scope,
        "defense_state_scope": state_scope,
        "attack_state_indices": list(defended_indices_for_scope(state_scope)),
    }
    return PairDatasetBundle(
        adv_inputs=np.concatenate(adv_inputs, axis=0),
        clean_inputs=np.concatenate(clean_inputs, axis=0),
        metadata=merged_metadata,
        time_indices=np.concatenate(time_indices, axis=0),
        stations=np.concatenate(stations, axis=0),
        is_new_arrivals=np.concatenate(is_new_arrivals, axis=0),
        vehicle_ids=np.concatenate(vehicle_ids, axis=0),
        episode_indices=np.concatenate(episode_indices, axis=0),
        attack_mask=np.concatenate(attack_masks, axis=0),
    )


def canonical_posterior_label_mode(value: str | None) -> str:
    token = str(value or "benefit").strip().lower().replace("-", "_")
    if token in {"benefit", "repair_benefit", "utility", "gain"}:
        return "benefit"
    if token in {"attack", "attack_clean", "attacked", "attack_label"}:
        return "attack"
    raise ValueError(f"Unsupported posterior label mode: {value}")


def posterior_detector_dataset_from_unified_pair(
    bundle: PairDatasetBundle,
    actor,
    defender,
    device: torch.device,
    *,
    profile_tag: str,
    train_attack_tags: list[str],
    benefit_margin: float = 0.0,
    benefit_action_weight: float = 1.0,
    benefit_state_weight: float = 1.0,
    posterior_label_mode: str = "benefit",
    use_benefit_sample_weights: bool = True,
    state_scope: str = "local",
    repair_mode: str = "full",
) -> DetectorDatasetBundle:
    if defender is None:
        raise ValueError("posterior detector dataset requires a trained DAE defender.")
    label_mode = canonical_posterior_label_mode(posterior_label_mode)
    state_scope = canonical_state_scope(state_scope)
    repair_mode = str(repair_mode or "full").strip().lower().replace("-", "_")
    if repair_mode not in {"full", "core_only"}:
        raise ValueError(f"Unsupported posterior repair_mode: {repair_mode!r}")
    state_indices = tuple(
        int(value)
        for value in (
            LOCAL_SHIELD_INDICES
            if repair_mode == "core_only"
            else defended_indices_for_scope(state_scope)
        )
    )
    clean_inputs = np.asarray(bundle.clean_inputs, dtype=np.float32).reshape(-1, 11)
    adv_inputs = np.asarray(bundle.adv_inputs, dtype=np.float32).reshape(-1, 11)
    attack_mask = (
        (np.max(np.abs(adv_inputs - clean_inputs), axis=1) > 1e-8).astype(np.int64)
        if bundle.attack_mask is None
        else np.asarray(bundle.attack_mask, dtype=np.int64).reshape(-1)
    )
    count = clean_inputs.shape[0]
    time_indices = (
        np.asarray(bundle.time_indices, dtype=np.int64).reshape(-1)
        if bundle.time_indices is not None
        else np.zeros((count,), dtype=np.int64)
    )
    stations = (
        np.asarray(bundle.stations, dtype=np.int64).reshape(-1)
        if bundle.stations is not None
        else np.zeros((count,), dtype=np.int64)
    )
    is_new_arrivals = (
        np.asarray(bundle.is_new_arrivals, dtype=np.int64).reshape(-1)
        if bundle.is_new_arrivals is not None
        else np.zeros((count,), dtype=np.int64)
    )
    vehicle_ids = (
        np.asarray(bundle.vehicle_ids, dtype=np.int64).reshape(-1)
        if bundle.vehicle_ids is not None
        else np.arange(count, dtype=np.int64)
    )
    episode_indices = (
        np.asarray(bundle.episode_indices, dtype=np.int64).reshape(-1)
        if bundle.episode_indices is not None
        else np.zeros((count,), dtype=np.int64)
    )

    adv_prev = build_previous_step_inputs(
        adv_inputs, episode_indices=episode_indices, vehicle_ids=vehicle_ids
    )
    clean_prev = build_previous_step_inputs(
        clean_inputs, episode_indices=episode_indices, vehicle_ids=vehicle_ids
    )
    adv_rec_full = dae_reconstruction_with_history(
        defender,
        adv_inputs,
        device,
        episode_indices=episode_indices,
        vehicle_ids=vehicle_ids,
    )
    clean_rec_full = dae_reconstruction_with_history(
        defender,
        clean_inputs,
        device,
        episode_indices=episode_indices,
        vehicle_ids=vehicle_ids,
    )
    if repair_mode == "core_only":
        core_idx = list(LOCAL_SHIELD_INDICES)
        adv_rec = adv_inputs.copy()
        clean_rec = clean_inputs.copy()
        adv_rec[:, core_idx] = adv_rec_full[:, core_idx]
        clean_rec[:, core_idx] = clean_rec_full[:, core_idx]
    else:
        adv_rec = adv_rec_full
        clean_rec = clean_rec_full

    actor = actor.to(device).eval()
    with torch.no_grad():
        clean_act = actor(torch.as_tensor(clean_inputs, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
        adv_act = actor(torch.as_tensor(adv_inputs, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
        adv_rec_act = actor(torch.as_tensor(adv_rec, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
        clean_rec_act = actor(torch.as_tensor(clean_rec, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)

    action_weight = max(float(benefit_action_weight), 0.0)
    state_weight = max(float(benefit_state_weight), 0.0)
    margin = max(float(benefit_margin), 0.0)
    adv_benefit = (
        state_weight
        * (
            weighted_state_error_np(adv_inputs, clean_inputs, state_indices=state_indices)
            - weighted_state_error_np(adv_rec, clean_inputs, state_indices=state_indices)
        )
        + action_weight * (((clean_act - adv_act) ** 2) - ((clean_act - adv_rec_act) ** 2))
    ).astype(np.float32)
    clean_benefit = (
        state_weight
        * (0.0 - weighted_state_error_np(clean_rec, clean_inputs, state_indices=state_indices))
        + action_weight * (0.0 - ((clean_act - clean_rec_act) ** 2))
    ).astype(np.float32)

    obs_inputs = np.concatenate([adv_inputs, clean_inputs], axis=0)
    rec_inputs = np.concatenate([adv_rec, clean_rec], axis=0)
    clean_refs = np.concatenate([clean_inputs, clean_inputs], axis=0)
    prev_obs_inputs = np.concatenate([adv_prev, clean_prev], axis=0)
    benefit_scores = np.concatenate([adv_benefit, clean_benefit], axis=0).astype(np.float32)
    attack_mask_full = np.concatenate(
        [attack_mask.astype(np.int64), np.zeros((count,), dtype=np.int64)], axis=0
    )
    time_indices_full = np.concatenate([time_indices, time_indices], axis=0)
    stations_full = np.concatenate([stations, stations], axis=0)
    is_new_arrivals_full = np.concatenate([is_new_arrivals, is_new_arrivals], axis=0)
    vehicle_ids_full = np.concatenate([vehicle_ids, vehicle_ids], axis=0)
    episode_indices_full = np.concatenate([episode_indices, episode_indices], axis=0)
    labels = (
        (benefit_scores > margin).astype(np.int64)
        if label_mode == "benefit"
        else attack_mask_full.astype(np.int64)
    )
    keep_mask = (
        np.ones((labels.shape[0],), dtype=bool)
        if label_mode == "attack" or margin <= 0.0
        else np.abs(benefit_scores) > margin
    )
    if not bool(np.any(keep_mask)):
        raise ValueError("Posterior detector dataset is empty after applying benefit margin.")
    obs_inputs = obs_inputs[keep_mask]
    rec_inputs = rec_inputs[keep_mask]
    clean_refs = clean_refs[keep_mask]
    prev_obs_inputs = prev_obs_inputs[keep_mask]
    labels = labels[keep_mask]
    benefit_scores = benefit_scores[keep_mask]
    attack_mask_full = attack_mask_full[keep_mask]
    time_indices_full = time_indices_full[keep_mask]
    stations_full = stations_full[keep_mask]
    is_new_arrivals_full = is_new_arrivals_full[keep_mask]
    vehicle_ids_full = vehicle_ids_full[keep_mask]
    episode_indices_full = episode_indices_full[keep_mask]
    sample_weights = None
    if label_mode == "benefit" and bool(use_benefit_sample_weights):
        scale = max(
            float(np.quantile(np.abs(benefit_scores), 0.75)) if benefit_scores.size else 0.0,
            1e-6,
        )
        sample_weights = np.clip(np.abs(benefit_scores) / scale, 0.25, 4.0).astype(np.float32)
        sample_weights /= max(float(np.mean(sample_weights)), 1e-6)

    metadata = {
        "collection_mode": f"unified_posterior_detector_{label_mode}_label",
        "detector_mode": "posterior",
        "posterior_label_mode": label_mode,
        "profile_tag": str(profile_tag),
        "train_attacks": list(train_attack_tags),
        "samples": int(obs_inputs.shape[0]),
        "source_samples": int(count),
        "attacked_samples": int(np.sum(attack_mask_full)),
        "clean_identity_samples": int(np.sum(attack_mask_full == 0)),
        "ambiguous_dropped_samples": int(np.sum(~keep_mask)),
        "positive_samples": int(np.sum(labels == 1)),
        "negative_samples": int(np.sum(labels == 0)),
        "benefit_margin": float(margin),
        "benefit_action_weight": float(action_weight),
        "benefit_state_weight": float(state_weight),
        "use_benefit_sample_weights": bool(label_mode == "benefit" and use_benefit_sample_weights),
        "benefit_score_mean": float(np.mean(benefit_scores)) if benefit_scores.size else 0.0,
        "benefit_score_std": float(np.std(benefit_scores)) if benefit_scores.size else 0.0,
        "label_positive_rate": float(np.mean(labels == 1)) if labels.size else 0.0,
        "benefit_positive_rate": float(np.mean(benefit_scores > margin)) if benefit_scores.size else 0.0,
        "policy_input_mode": "posterior_accept_recovered",
        "attack_trigger_mode": f"candidate_all_{state_scope}_obs",
        "state_scope": state_scope,
        "attack_state_scope": state_scope,
        "defense_state_scope": state_scope,
        "repair_mode": repair_mode,
        "posterior_candidate_state": "core_only_injected" if repair_mode == "core_only" else "full_reconstruction",
        "attack_state_indices": list(state_indices),
    }
    return DetectorDatasetBundle(
        clean_inputs=clean_refs,
        adv_inputs=obs_inputs,
        metadata=metadata,
        time_indices=time_indices_full,
        stations=stations_full,
        is_new_arrivals=is_new_arrivals_full,
        vehicle_ids=vehicle_ids_full,
        episode_indices=episode_indices_full,
        attack_mask=attack_mask_full,
        clean_refs=clean_refs,
        obs_inputs=obs_inputs,
        rec_inputs=rec_inputs,
        labels=labels,
        benefit_scores=benefit_scores,
        prev_obs_inputs=prev_obs_inputs,
        sample_weights=sample_weights,
    )


__all__ = [
    "merge_pair_bundles_for_unified",
    "posterior_detector_dataset_from_unified_pair",
]
