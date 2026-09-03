from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .defense import (
    DAETrainResult,
    DenoisingAutoencoder,
    PosteriorBenefitMLPDetector,
    SequenceDenoiseDataset,
    _action_mse,
    _kl_to_conditional_prior,
    canonical_state_scope,
    conditioning_indices_for_scope,
    dae_reconstruction_with_history,
    defended_indices_for_scope,
    grouped_state_loss,
    posterior_detector_probabilities_tensor,
)
from .merged_attacks import AttackContext, PGDStateAttacker
from .merged_core import Actor
from .merged_pipeline import CleanTrajectoryBundle, PairDatasetBundle, build_pair_dataset_from_clean_trajectories
from .session_replay_utils import merge_pair_bundles_preserve_sessions


@dataclass
class BucketReplayStats:
    bucket_session_counts: dict[str, int]
    batch_quotas: dict[str, int]
    steps_per_epoch: int


def _empty_pair_bundle(tag: str = 'empty') -> PairDatasetBundle:
    empty = np.zeros((0, 11), dtype=np.float32)
    return PairDatasetBundle(empty, empty.copy(), {'collection_mode': tag, 'samples': 0, 'attacked_samples': 0})


class _SourceKindOverrideDataset(Dataset):
    def __init__(self, base: SequenceDenoiseDataset, source_kind: int) -> None:
        self.base = base
        self.source_kind = int(source_kind)
        self.seq_len = int(getattr(base, 'seq_len', 1))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        row = dict(self.base[int(idx)])
        row['source_kind'] = torch.as_tensor(self.source_kind, dtype=torch.long)
        return row


def _session_count(bundle: PairDatasetBundle | CleanTrajectoryBundle) -> int:
    values = np.asarray(bundle.clean_inputs, dtype=np.float32).reshape(-1, 11)
    if values.shape[0] == 0:
        return 0
    episodes = (
        np.asarray(bundle.episode_indices, dtype=np.int64).reshape(-1)
        if bundle.episode_indices is not None
        else np.zeros((values.shape[0],), dtype=np.int64)
    )
    vehicles = (
        np.asarray(bundle.vehicle_ids, dtype=np.int64).reshape(-1)
        if bundle.vehicle_ids is not None
        else np.arange(values.shape[0], dtype=np.int64)
    )
    return int(len({(int(ep), int(vid)) for ep, vid in zip(episodes.tolist(), vehicles.tolist())}))


def _build_bucket_datasets(
    *,
    clean_bundle: CleanTrajectoryBundle,
    pair_bundles: Mapping[str, PairDatasetBundle],
    seq_len: int | None,
    local_indices: Sequence[int],
    global_indices: Sequence[int],
) -> dict[str, Dataset]:
    datasets: dict[str, Dataset] = {
        'clean': _SourceKindOverrideDataset(
            SequenceDenoiseDataset(
                np.asarray(clean_bundle.clean_inputs, dtype=np.float32).reshape(-1, 11),
                np.asarray(clean_bundle.clean_inputs, dtype=np.float32).reshape(-1, 11),
                episode_indices=clean_bundle.episode_indices,
                vehicle_ids=clean_bundle.vehicle_ids,
                seq_len=int(seq_len or 8),
                include_clean_sequences=False,
                local_indices=local_indices,
                global_indices=global_indices,
            ),
            source_kind=1,
        )
    }
    for name, bundle in pair_bundles.items():
        datasets[name] = SequenceDenoiseDataset(
            np.asarray(bundle.adv_inputs, dtype=np.float32).reshape(-1, 11),
            np.asarray(bundle.clean_inputs, dtype=np.float32).reshape(-1, 11),
            episode_indices=bundle.episode_indices,
            vehicle_ids=bundle.vehicle_ids,
            seq_len=int(seq_len or 8),
            include_clean_sequences=False,
            local_indices=local_indices,
            global_indices=global_indices,
        )
    common_seq_len = max(int(getattr(dataset, 'seq_len', 1)) for dataset in datasets.values())
    if any(int(getattr(dataset, 'seq_len', 1)) != common_seq_len for dataset in datasets.values()):
        return _build_bucket_datasets(
            clean_bundle=clean_bundle,
            pair_bundles=pair_bundles,
            seq_len=common_seq_len,
            local_indices=local_indices,
            global_indices=global_indices,
        )
    return datasets


def _batch_quotas(
    batch_size: int,
    ratios: Mapping[str, float],
    datasets: Mapping[str, Dataset],
) -> dict[str, int]:
    available = {name: int(len(dataset)) > 0 and float(ratios.get(name, 0.0)) > 0.0 for name, dataset in datasets.items()}
    weight_sum = sum(float(ratios.get(name, 0.0)) for name, ok in available.items() if ok)
    if weight_sum <= 0.0:
        raise ValueError('Bucketed replay has no available data sources.')
    raw = {
        name: float(batch_size) * float(ratios.get(name, 0.0)) / weight_sum if ok else 0.0
        for name, ok in available.items()
    }
    quotas = {name: int(np.floor(value)) for name, value in raw.items()}
    for name, ok in available.items():
        if ok and quotas[name] <= 0:
            quotas[name] = 1
    while sum(quotas.values()) > int(batch_size):
        candidates = [name for name, value in quotas.items() if value > 1]
        if not candidates:
            break
        name = min(candidates, key=lambda key: raw[key] - np.floor(raw[key]))
        quotas[name] -= 1
    while sum(quotas.values()) < int(batch_size):
        candidates = [name for name, ok in available.items() if ok]
        name = max(candidates, key=lambda key: raw[key] - quotas[key])
        quotas[name] += 1
    return quotas


def _cycle_next(name: str, loaders: Mapping[str, DataLoader], loader_iters: dict[str, object]) -> dict[str, torch.Tensor]:
    try:
        return next(loader_iters[name])  # type: ignore[arg-type]
    except StopIteration:
        loader_iters[name] = iter(loaders[name])
        return next(loader_iters[name])  # type: ignore[arg-type]


def _concat_batches(batches: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = ('x_local', 'x_global', 'length', 'target_full', 'source_kind')
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys}


def train_gru_vae_dae_bucketed(
    *,
    clean_bundle: CleanTrajectoryBundle,
    pair_bundles: Mapping[str, PairDatasetBundle],
    bucket_ratios: Mapping[str, float],
    actor: Actor,
    device: torch.device,
    state_scope: str = 'all',
    seq_len: int | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_steps_per_epoch: int | None = 256,
    hidden_dim: int = 128,
    latent_dim: int = 64,
    num_layers: int = 1,
    decoder_hidden_dim: int = 128,
    beta_kl: float = 1e-4,
    lambda_recon: float = 1.0,
    lambda_identity: float = 2.0,
    lambda_robust: float = 0.1,
    local_recon_weight: float = 1.0,
    global_recon_weight: float = 0.0,
    init_model: DenoisingAutoencoder | None = None,
    log_every: int = 5,
) -> tuple[DenoisingAutoencoder, DAETrainResult, BucketReplayStats]:
    scope = canonical_state_scope(state_scope)
    defended = defended_indices_for_scope(scope)
    conditioning = conditioning_indices_for_scope(scope)
    datasets = _build_bucket_datasets(
        clean_bundle=clean_bundle,
        pair_bundles=pair_bundles,
        seq_len=seq_len,
        local_indices=defended,
        global_indices=conditioning,
    )
    quotas = _batch_quotas(int(batch_size), bucket_ratios, datasets)
    loaders = {
        name: DataLoader(dataset, batch_size=int(quotas[name]), shuffle=True)
        for name, dataset in datasets.items()
        if int(quotas.get(name, 0)) > 0 and int(len(dataset)) > 0
    }
    if not loaders:
        raise ValueError('Bucketed replay could not create any non-empty loaders.')
    loader_iters: dict[str, object] = {name: iter(loader) for name, loader in loaders.items()}
    auto_steps = max(
        int(np.ceil(len(datasets[name]) / max(int(quotas.get(name, 1)), 1)))
        for name in loaders
    )
    steps_per_epoch = int(min(auto_steps, int(max_steps_per_epoch))) if max_steps_per_epoch and int(max_steps_per_epoch) > 0 else int(auto_steps)
    seq_len_final = max(int(getattr(dataset, 'seq_len', 1)) for dataset in datasets.values())
    if init_model is None:
        model = DenoisingAutoencoder(
            input_dim=11,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            seq_len=seq_len_final,
            decoder_hidden_dim=decoder_hidden_dim,
            local_indices=defended,
            global_indices=conditioning,
        ).to(device)
    else:
        model = init_model.to(device)
        if tuple(int(v) for v in getattr(model, 'local_indices', ())) != tuple(int(v) for v in defended):
            raise ValueError('init_model local_indices do not match state_scope.')
        if tuple(int(v) for v in getattr(model, 'global_indices', ())) != tuple(int(v) for v in conditioning):
            raise ValueError('init_model global_indices do not match state_scope.')
        model.seq_len = int(max(int(getattr(model, 'seq_len', seq_len_final)), int(seq_len_final)))
    for param in model.parameters():
        param.requires_grad_(True)
    actor = actor.to(device).eval()
    for param in actor.parameters():
        param.requires_grad_(False)
    optimizer = optim.Adam(model.parameters(), lr=float(lr))
    loss_history: list[float] = []
    recon_loss_history: list[float] = []
    kl_loss_history: list[float] = []
    robust_loss_history: list[float] = []
    clean_adv_action_mse: list[float] = []
    clean_recovered_action_mse: list[float] = []
    clean_adv_state_mse: list[float] = []
    clean_recovered_state_mse: list[float] = []
    validator_rows: list[dict] = []
    is_best_history: list[bool] = []
    best_loss = float('inf')
    best_state = None
    best_epoch = -1
    metric_bundle = merge_pair_bundles_preserve_sessions(list(pair_bundles.values()), attack_tags=list(pair_bundles.keys()))
    metric_adv_inputs = np.asarray(metric_bundle.adv_inputs, dtype=np.float32).reshape(-1, 11)
    metric_clean_inputs = np.asarray(metric_bundle.clean_inputs, dtype=np.float32).reshape(-1, 11)
    metric_episode_indices = metric_bundle.episode_indices
    metric_vehicle_ids = metric_bundle.vehicle_ids
    idx_t = torch.as_tensor(list(defended), dtype=torch.long, device=device)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_rec = 0.0
        total_kl = 0.0
        total_rob = 0.0
        steps = 0
        for _ in range(int(steps_per_epoch)):
            source_batches = [_cycle_next(name, loaders, loader_iters) for name in loaders]
            batch = _concat_batches(source_batches)
            x_local = batch['x_local'].to(device)
            x_global = batch['x_global'].to(device)
            lengths = batch['length'].to(device)
            target_full = batch['target_full'].to(device)
            source_kind = batch['source_kind'].to(device)
            recon, stats = model(x_local, x_global, lengths, return_stats=True, sample_latent=True)
            zero = target_full.new_tensor(0.0)
            adv_rows = source_kind == 0
            clean_rows = source_kind != 0
            attack_rec_loss = (
                grouped_state_loss(
                    recon[adv_rows],
                    target_full[adv_rows],
                    local_weight=local_recon_weight,
                    global_weight=global_recon_weight,
                    local_indices=defended,
                    global_indices=conditioning,
                )
                if bool(torch.any(adv_rows).item())
                else zero
            )
            identity_rec_loss = (
                grouped_state_loss(
                    recon[clean_rows],
                    target_full[clean_rows],
                    local_weight=local_recon_weight,
                    global_weight=global_recon_weight,
                    local_indices=defended,
                    global_indices=conditioning,
                )
                if bool(torch.any(clean_rows).item())
                else zero
            )
            rec_loss = float(lambda_recon) * attack_rec_loss + float(lambda_identity) * identity_rec_loss
            kl_loss = _kl_to_conditional_prior(
                stats['mu_post'],
                stats['logvar_post'],
                stats['mu_prior'],
                stats['logvar_prior'],
            )
            robust_loss = zero
            if float(lambda_robust) > 0.0:
                clean_action = actor(target_full).detach()
                recovered_action = actor(recon)
                robust_loss = torch.mean((recovered_action - clean_action) ** 2)
            loss = rec_loss + float(beta_kl) * kl_loss + float(lambda_robust) * robust_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_rec += float(rec_loss.detach().cpu())
            total_kl += float(kl_loss.detach().cpu())
            total_rob += float(robust_loss.detach().cpu())
            steps += 1
        loss_history.append(total_loss / max(steps, 1))
        recon_loss_history.append(total_rec / max(steps, 1))
        kl_loss_history.append(total_kl / max(steps, 1))
        robust_loss_history.append(total_rob / max(steps, 1))

        model.eval()
        with torch.no_grad():
            if metric_clean_inputs.shape[0] > 0:
                clean_t = torch.as_tensor(metric_clean_inputs, dtype=torch.float32, device=device)
                adv_t = torch.as_tensor(metric_adv_inputs, dtype=torch.float32, device=device)
                rec_np = dae_reconstruction_with_history(
                    model,
                    metric_adv_inputs,
                    device,
                    episode_indices=metric_episode_indices,
                    vehicle_ids=metric_vehicle_ids,
                    seq_len=seq_len_final,
                )
                rec_t = torch.as_tensor(rec_np, dtype=torch.float32, device=device)
                clean_adv_state_mse.append(float(torch.mean((adv_t.index_select(1, idx_t) - clean_t.index_select(1, idx_t)) ** 2).detach().cpu()))
                clean_recovered_state_mse.append(float(torch.mean((rec_t.index_select(1, idx_t) - clean_t.index_select(1, idx_t)) ** 2).detach().cpu()))
                clean_adv_action_mse.append(_action_mse(actor, clean_t, adv_t))
                clean_recovered_action_mse.append(_action_mse(actor, clean_t, rec_t))
            else:
                clean_adv_state_mse.append(float('nan'))
                clean_recovered_state_mse.append(float('nan'))
                clean_adv_action_mse.append(float('nan'))
                clean_recovered_action_mse.append(float('nan'))

        current_best = bool(loss_history[-1] < best_loss)
        if current_best:
            best_loss = float(loss_history[-1])
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        is_best_history.append(current_best)
        if epoch == 1 or epoch % max(int(log_every), 1) == 0 or epoch == int(epochs):
            print(
                f"[session-paper-dae] epoch={epoch:03d}/{epochs} "
                f"loss={loss_history[-1]:.6f} rec={recon_loss_history[-1]:.6f} "
                f"kl={kl_loss_history[-1]:.6f} robust={robust_loss_history[-1]:.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.metadata = {
        'algorithm': 'gru_vae_dae_bucketed',
        'state_scope': scope,
        'seq_len': int(seq_len_final),
        'hidden_dim': int(getattr(model, 'hidden_dim', hidden_dim)),
        'latent_dim': int(getattr(model, 'latent_dim', latent_dim)),
        'decoder_hidden_dim': int(getattr(model, 'decoder_hidden_dim', decoder_hidden_dim)),
        'beta_kl': float(beta_kl),
        'lambda_recon': float(lambda_recon),
        'lambda_identity': float(lambda_identity),
        'lambda_robust': float(lambda_robust),
        'bucket_ratios': {str(name): float(value) for name, value in bucket_ratios.items()},
        'bucket_quotas': {str(name): int(value) for name, value in quotas.items()},
    }
    stats = BucketReplayStats(
        bucket_session_counts={
            'clean': int(_session_count(clean_bundle)),
            **{str(name): int(_session_count(bundle)) for name, bundle in pair_bundles.items()},
        },
        batch_quotas={str(name): int(value) for name, value in quotas.items()},
        steps_per_epoch=int(steps_per_epoch),
    )
    return model, DAETrainResult(
        loss_history=loss_history,
        recon_loss_history=recon_loss_history,
        kl_loss_history=kl_loss_history,
        robust_loss_history=robust_loss_history,
        clean_adv_action_mse=clean_adv_action_mse,
        clean_recovered_action_mse=clean_recovered_action_mse,
        clean_adv_state_mse=clean_adv_state_mse,
        clean_recovered_state_mse=clean_recovered_state_mse,
        validator_rows=validator_rows,
        is_best_history=is_best_history,
        best_epoch=best_epoch,
        best_metric_name='loss',
        best_metric_value=float(best_loss),
    ), stats


class PaperAdaptiveStateAttacker(PGDStateAttacker):
    def __init__(
        self,
        actor: torch.nn.Module,
        *,
        device: torch.device,
        base_algorithm: str,
        defender: nn.Module,
        detector_model: PosteriorBenefitMLPDetector,
        epsilon: float | None = None,
        alpha: float | None = None,
        iters: int | None = None,
        seed: int = 42,
        obs_low: np.ndarray | torch.Tensor | None = None,
        obs_high: np.ndarray | torch.Tensor | None = None,
        critic: torch.nn.Module | None = None,
        attack_state_scope: str = 'local',
        detector_bypass_weight: float = 0.25,
        adaptive_mode: str = 'joint_route',
    ) -> None:
        super().__init__(
            actor,
            device=device,
            algorithm=base_algorithm,
            epsilon=epsilon,
            alpha=alpha,
            iters=iters,
            seed=seed,
            obs_low=obs_low,
            obs_high=obs_high,
            critic=critic,
            attack_state_scope=attack_state_scope,
        )
        if not isinstance(detector_model, PosteriorBenefitMLPDetector):
            raise ValueError('PaperAdaptiveStateAttacker requires PosteriorBenefitMLPDetector.')
        self.base_algorithm = str(self.algorithm)
        self.algorithm = f'adaptive_{self.base_algorithm}'
        self.defender = defender.to(device).eval()
        self.detector_model = detector_model.to(device).eval()
        self.detector_bypass_weight = float(detector_bypass_weight)
        self.scope = canonical_state_scope(attack_state_scope)
        self.adaptive_mode = str(adaptive_mode).strip().lower()
        if self.adaptive_mode not in {'joint_route', 'repairer_only'}:
            raise ValueError(f'Unsupported adaptive_mode: {adaptive_mode!r}')
        self.seq_len_runtime = max(int(getattr(self.defender, 'seq_len', 1)), 1)
        self.history_buffers: defaultdict[tuple[int, int], deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=max(self.seq_len_runtime - 1, 1))
        )
        for module in (self.actor, self.critic, self.defender, self.detector_model):
            if module is None:
                continue
            module.eval()

    def reset(self) -> None:
        super().reset()
        self.history_buffers.clear()

    def clone(self) -> 'PaperAdaptiveStateAttacker':
        cloned = PaperAdaptiveStateAttacker(
            self.actor,
            device=self.device,
            base_algorithm=self.base_algorithm,
            defender=self.defender,
            detector_model=self.detector_model,
            epsilon=self.epsilon,
            alpha=self.alpha,
            iters=self.iters,
            seed=self.seed,
            obs_low=None if self.obs_low is None else self.obs_low.detach().cpu().numpy().reshape(-1),
            obs_high=None if self.obs_high is None else self.obs_high.detach().cpu().numpy().reshape(-1),
            critic=self.critic,
            attack_state_scope=self.attack_state_scope,
            detector_bypass_weight=self.detector_bypass_weight,
            adaptive_mode=self.adaptive_mode,
        )
        cloned.reset()
        return cloned

    def _normalize_episode_ids(self, episode_indices: Sequence[int] | np.ndarray | None, batch_size: int) -> list[int]:
        if episode_indices is None:
            return [0 for _ in range(batch_size)]
        values = [int(v) for v in np.asarray(episode_indices, dtype=np.int64).reshape(-1)]
        if len(values) != batch_size:
            raise ValueError('episode_indices length must match batch size.')
        return values

    def _normalize_vehicle_ids(self, vehicle_ids: Sequence[int] | np.ndarray | None, batch_size: int) -> list[int]:
        if vehicle_ids is None:
            return [int(i) for i in range(batch_size)]
        values = [int(v) for v in np.asarray(vehicle_ids, dtype=np.int64).reshape(-1)]
        if len(values) != batch_size:
            raise ValueError('vehicle_ids length must match batch size.')
        return values

    def _context_arrays(self, contexts: Sequence[AttackContext]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray([int(ctx.time_index) for ctx in contexts], dtype=np.int64),
            np.asarray([int(ctx.station) for ctx in contexts], dtype=np.int64),
            np.asarray([1 if bool(ctx.is_new_arrival) else 0 for ctx in contexts], dtype=np.int64),
        )

    def _history_views(
        self,
        clean_obs: torch.Tensor,
        *,
        vehicle_ids: Sequence[int],
        episode_indices: Sequence[int],
    ) -> tuple[list[list[np.ndarray]], torch.Tensor]:
        histories: list[list[np.ndarray]] = []
        prev_refs: list[torch.Tensor] = []
        for row_idx, (episode_id, vehicle_id) in enumerate(zip(episode_indices, vehicle_ids)):
            hist = list(self.history_buffers[(int(episode_id), int(vehicle_id))])
            histories.append([np.asarray(item, dtype=np.float32).reshape(-1).copy() for item in hist])
            if hist:
                prev_refs.append(torch.as_tensor(hist[-1], dtype=torch.float32, device=self.device))
            else:
                prev_refs.append(clean_obs[row_idx].detach())
        return histories, torch.stack(prev_refs, dim=0).reshape(-1, clean_obs.shape[-1])

    def _sequence_tensor(self, current_obs: torch.Tensor, histories: Sequence[Sequence[np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
        seqs: list[torch.Tensor] = []
        lengths: list[int] = []
        for row_idx, history in enumerate(histories):
            if history:
                hist_t = torch.as_tensor(np.asarray(history, dtype=np.float32), dtype=torch.float32, device=self.device).reshape(-1, current_obs.shape[-1])
                full = torch.cat([hist_t, current_obs[row_idx : row_idx + 1]], dim=0)
            else:
                full = current_obs[row_idx : row_idx + 1]
            keep = full[-self.seq_len_runtime :]
            padded = current_obs.new_zeros((self.seq_len_runtime, current_obs.shape[-1]))
            padded[: keep.shape[0]] = keep
            seqs.append(padded)
            lengths.append(int(keep.shape[0]))
        return torch.stack(seqs, dim=0), torch.as_tensor(lengths, dtype=torch.long, device=self.device)

    def _soft_route_state(
        self,
        current_obs: torch.Tensor,
        *,
        histories: Sequence[Sequence[np.ndarray]],
        prev_obs_inputs: torch.Tensor,
        time_indices: np.ndarray,
        stations: np.ndarray,
        is_new_arrivals: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_t, len_t = self._sequence_tensor(current_obs, histories)
        reconstructed = self.defender(seq_t, None, len_t)
        repair_prob = posterior_detector_probabilities_tensor(
            self.detector_model,
            current_obs,
            reconstructed,
            self.actor,
            self.device,
            time_indices=time_indices,
            stations=stations,
            is_new_arrivals=is_new_arrivals,
            prev_obs_inputs=prev_obs_inputs,
            include_temporal=bool(getattr(self.detector_model, 'include_temporal', True)),
        )
        routed = repair_prob.view(-1, 1) * reconstructed + (1.0 - repair_prob.view(-1, 1)) * current_obs
        return routed, repair_prob, reconstructed

    def _adaptive_policy_state(
        self,
        current_obs: torch.Tensor,
        *,
        histories: Sequence[Sequence[np.ndarray]],
        prev_obs_inputs: torch.Tensor,
        time_indices: np.ndarray,
        stations: np.ndarray,
        is_new_arrivals: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.adaptive_mode == 'repairer_only':
            seq_t, len_t = self._sequence_tensor(current_obs, histories)
            reconstructed = self.defender(seq_t, None, len_t)
            repair_prob = torch.ones((current_obs.shape[0],), dtype=current_obs.dtype, device=current_obs.device)
            return reconstructed, repair_prob, reconstructed
        return self._soft_route_state(
            current_obs,
            histories=histories,
            prev_obs_inputs=prev_obs_inputs,
            time_indices=time_indices,
            stations=stations,
            is_new_arrivals=is_new_arrivals,
        )

    def _append_history_rows(
        self,
        observed_obs: np.ndarray,
        *,
        vehicle_ids: Sequence[int],
        episode_indices: Sequence[int],
    ) -> None:
        observed_arr = np.asarray(observed_obs, dtype=np.float32).reshape(-1, 11)
        for row_idx, (episode_id, vehicle_id) in enumerate(zip(episode_indices, vehicle_ids)):
            self.history_buffers[(int(episode_id), int(vehicle_id))].append(observed_arr[row_idx].reshape(-1).copy())

    def observe_batch(
        self,
        obs_batch: np.ndarray,
        *,
        contexts: Sequence[AttackContext] | None = None,
        vehicle_ids: Sequence[int] | np.ndarray | None = None,
        episode_indices: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        del contexts
        batch = np.asarray(obs_batch, dtype=np.float32).reshape(-1, 11)
        vehicle_id_list = self._normalize_vehicle_ids(vehicle_ids, int(batch.shape[0]))
        episode_id_list = self._normalize_episode_ids(episode_indices, int(batch.shape[0]))
        self._append_history_rows(batch, vehicle_ids=vehicle_id_list, episode_indices=episode_id_list)

    def attack_with_metadata(
        self,
        obs_batch: np.ndarray,
        *,
        contexts: Sequence[AttackContext],
        vehicle_ids: Sequence[int] | np.ndarray | None = None,
        episode_indices: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        clean_obs = torch.as_tensor(np.asarray(obs_batch, dtype=np.float32).reshape(-1, 11), dtype=torch.float32, device=self.device)
        batch_size = int(clean_obs.shape[0])
        if batch_size == 0:
            return np.zeros((0, 11), dtype=np.float32)
        vehicle_id_list = self._normalize_vehicle_ids(vehicle_ids, batch_size)
        episode_id_list = self._normalize_episode_ids(episode_indices, batch_size)
        histories, prev_obs_inputs = self._history_views(clean_obs, vehicle_ids=vehicle_id_list, episode_indices=episode_id_list)
        time_indices, stations, is_new_arrivals = self._context_arrays(contexts)
        with torch.no_grad():
            with torch.backends.cudnn.flags(enabled=False):
                clean_policy_state, _, _ = self._adaptive_policy_state(
                    clean_obs,
                    histories=histories,
                    prev_obs_inputs=prev_obs_inputs,
                    time_indices=time_indices,
                    stations=stations,
                    is_new_arrivals=is_new_arrivals,
                )
                clean_actions = self._actor_mean_action(clean_policy_state).detach()
        original = clean_obs.detach().clone()
        image = self._random_start(original)
        for _ in range(self.iters):
            image.requires_grad_(True)
            with torch.backends.cudnn.flags(enabled=False):
                policy_state, repair_prob, _ = self._adaptive_policy_state(
                    image,
                    histories=histories,
                    prev_obs_inputs=prev_obs_inputs,
                    time_indices=time_indices,
                    stations=stations,
                    is_new_arrivals=is_new_arrivals,
                )
            if self.base_algorithm == 'q_function':
                if self.critic is None:
                    raise RuntimeError('adaptive_q_function requires critic.')
                objective = -self.critic(policy_state, self._actor_mean_action(policy_state)).mean()
            else:
                objective = torch.mean((self._actor_mean_action(policy_state) - clean_actions) ** 2)
            if self.adaptive_mode == 'joint_route' and self.detector_bypass_weight > 0.0:
                objective = objective + self.detector_bypass_weight * (1.0 - repair_prob).mean()
            grad = torch.autograd.grad(objective, image, retain_graph=False, create_graph=False)[0]
            mask = self._local_attack_mask(original)
            adv = image + self.alpha * grad.sign() * mask
            eta = torch.clamp(adv - original, min=-self.epsilon, max=self.epsilon) * mask
            image = self._project_obs(original, original + eta).detach()
        adv_np = image.detach().cpu().numpy().astype(np.float32)
        self._append_history_rows(adv_np, vehicle_ids=vehicle_id_list, episode_indices=episode_id_list)
        return adv_np


def collect_paper_adaptive_pair_dataset(
    clean_bundle: CleanTrajectoryBundle,
    attacker: PaperAdaptiveStateAttacker,
    *,
    attack_scenario: str = 'O',
    attack_ratio: float = 1.0,
    attack_scope: str = 'obs',
    collection_mode: str,
    state_scope: str,
) -> PairDatasetBundle:
    attacker.reset()
    bundle = build_pair_dataset_from_clean_trajectories(
        clean_bundle,
        attacker,
        attack_scenario,
        attack_ratio=attack_ratio,
        attack_scope=attack_scope,
    )
    bundle.metadata.update(
        {
            'collection_mode': str(collection_mode),
            'paper_adaptive': True,
            'adaptive_base_algorithm': str(attacker.base_algorithm),
            'state_scope': canonical_state_scope(state_scope),
            'attack_state_scope': canonical_state_scope(state_scope),
            'defense_state_scope': canonical_state_scope(state_scope),
            'attack_state_indices': list(defended_indices_for_scope(state_scope)),
            'policy_input_mode': 'posterior_soft_route',
            'detector_bypass_weight': float(attacker.detector_bypass_weight),
        }
    )
    return bundle
