from __future__ import annotations

import numpy as np
import torch

from evc.defense import DenoisingAutoencoder
from evc.dtsr_datasets import (
    merge_pair_bundles_for_unified,
    posterior_detector_dataset_from_unified_pair,
)
from evc.merged_pipeline import PairDatasetBundle
from evc.offpolicy_backbones import GaussianActor


def _pair(offset: float, episode: int) -> PairDatasetBundle:
    clean = np.full((4, 11), 0.5, dtype=np.float32)
    adv = clean.copy()
    adv[:, :3] = np.clip(adv[:, :3] + offset, 0.0, 1.0)
    return PairDatasetBundle(
        adv_inputs=adv,
        clean_inputs=clean,
        metadata={"state_scope": "all"},
        time_indices=np.arange(4, dtype=np.int64),
        stations=np.zeros(4, dtype=np.int64),
        is_new_arrivals=np.zeros(4, dtype=np.int64),
        vehicle_ids=np.arange(4, dtype=np.int64),
        episode_indices=np.full(4, episode, dtype=np.int64),
        attack_mask=np.ones(4, dtype=np.int64),
    )


def test_merge_pair_bundles_offsets_episodes() -> None:
    merged = merge_pair_bundles_for_unified(
        [_pair(0.05, 0), _pair(-0.05, 0)],
        attack_tags=["opposite_pgd", "q_function"],
    )
    assert merged.clean_inputs.shape == (8, 11)
    assert merged.metadata["source_bundle_count"] == 2
    assert set(np.unique(merged.episode_indices).tolist()) == {0, 1}
    assert int(merged.attack_mask.sum()) == 8


def test_posterior_dataset_supports_sac_deterministic_actor() -> None:
    pair = _pair(0.05, 0)
    dae = DenoisingAutoencoder(
        hidden_dim=16,
        latent_dim=8,
        decoder_hidden_dim=16,
        seq_len=2,
    ).eval()
    actor = GaussianActor(hidden_dim=16).eval()
    dataset = posterior_detector_dataset_from_unified_pair(
        pair,
        actor,
        dae,
        torch.device("cpu"),
        profile_tag="sac_smoke",
        train_attack_tags=["opposite_pgd"],
        posterior_label_mode="benefit",
        state_scope="all",
        repair_mode="full",
    )
    assert dataset.obs_inputs.shape == (8, 11)
    assert dataset.rec_inputs.shape == (8, 11)
    assert dataset.labels.shape == (8,)
    assert np.isfinite(dataset.benefit_scores).all()
    assert dataset.metadata["detector_mode"] == "posterior"
