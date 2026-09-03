from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from .merged_core import ensure_dir, normalize_result_frame

DEFAULT_STATE_WEIGHTS = torch.tensor(
    [4.0, 3.0, 1.5, 1.5, 1.5, 1.2, 1.2, 1.2, 1.2, 1.2, 2.0],
    dtype=torch.float32,
)


@dataclass
class DAETrainResult:
    loss_history: list[float]
    recon_loss_history: list[float]
    kl_loss_history: list[float]
    robust_loss_history: list[float]
    clean_adv_action_mse: list[float]
    clean_recovered_action_mse: list[float]
    clean_adv_state_mse: list[float]
    clean_recovered_state_mse: list[float]
    validator_rows: list[dict[str, Any]]
    is_best_history: list[bool]
    best_epoch: int
    best_metric_name: str
    best_metric_value: float


@dataclass
class DetectorTrainResult:
    train_loss_history: list[float]
    val_loss_history: list[float]
    val_accuracy_history: list[float]
    val_precision_history: list[float]
    val_recall_history: list[float]
    val_f1_history: list[float]
    is_best_history: list[bool]
    best_epoch: int
    best_metric_name: str
    best_metric_value: float


@dataclass
class DetectorArtifact:
    model: nn.Module
    threshold: float
    metadata: dict[str, Any]


STATE_LOCAL_IDX = (0, 1, 10)
STATE_GLOBAL_IDX = (2, 3, 4, 5, 6, 7, 8, 9)
STATE_ALL_IDX = tuple(range(11))
LOCAL_ONLY_MAINLINE = True
DETECTOR_SELECTION_MIN_CLEAN_ACCURACY = 0.95


def _json_progress_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_progress_value(value.item())
    if isinstance(value, torch.Tensor):
        return _json_progress_value(value.detach().cpu().tolist())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_progress_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_progress_value(v) for v in value]
    return value


def _write_progress_frame(path: Path | None, rows: Sequence[dict[str, Any]]) -> None:
    if path is None:
        return
    ensure_dir(path.parent)
    normalize_result_frame(pd.DataFrame(list(rows)), rename_keys=False).to_csv(
        path,
        index=False,
        float_format='%.6f',
        encoding='utf-8-sig',
    )


def _write_progress_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    ensure_dir(path.parent)
    path.write_text(json.dumps(_json_progress_value(payload), ensure_ascii=False, indent=2), encoding='utf-8')


def _clip_reconstruction_to_model_bounds(model: nn.Module, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    low = getattr(model, 'obs_low', None)
    high = getattr(model, 'obs_high', None)
    if low is not None and high is not None:
        low_np = low.detach().cpu().numpy().reshape(-1).astype(np.float32)
        high_np = high.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if low_np.shape[0] == arr.shape[-1] and high_np.shape[0] == arr.shape[-1]:
            return np.clip(arr, low_np.reshape((1,) * (arr.ndim - 1) + (-1,)), high_np.reshape((1,) * (arr.ndim - 1) + (-1,)))
    return np.clip(arr, 0.0, 1.0)


def canonical_state_scope(value: str | None) -> str:
    token = str(value or 'local').strip().lower().replace('-', '_')
    aliases = {
        'local_only': 'local',
        'local': 'local',
        'global_only': 'global',
        'global': 'global',
        'all_state': 'all',
        'all_states': 'all',
        'full': 'all',
        'local_global': 'all',
        'local_and_global': 'all',
        'all': 'all',
    }
    if token not in aliases:
        raise ValueError(f'Unsupported state scope: {value!r}')
    return aliases[token]


def defended_indices_for_scope(scope: str | None) -> tuple[int, ...]:
    token = canonical_state_scope(scope)
    if token == 'local':
        return STATE_LOCAL_IDX
    if token == 'global':
        return STATE_GLOBAL_IDX
    return STATE_ALL_IDX


def conditioning_indices_for_scope(scope: str | None) -> tuple[int, ...]:
    token = canonical_state_scope(scope)
    if token == 'local':
        return STATE_GLOBAL_IDX
    if token == 'global':
        return STATE_LOCAL_IDX
    return ()


def build_previous_step_inputs(inputs: np.ndarray, *, episode_indices: np.ndarray | None, vehicle_ids: np.ndarray | None) -> np.ndarray:
    inputs = np.asarray(inputs, dtype=np.float32).reshape(-1, 11)
    total = int(inputs.shape[0])
    if total == 0:
        return np.zeros((0, 11), dtype=np.float32)
    if episode_indices is None:
        episode_indices = np.zeros((total,), dtype=np.int64)
    if vehicle_ids is None:
        vehicle_ids = np.arange(total, dtype=np.int64)
    episode_indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    vehicle_ids = np.asarray(vehicle_ids, dtype=np.int64).reshape(-1)
    if episode_indices.shape[0] != total or vehicle_ids.shape[0] != total:
        raise ValueError('episode_indices/vehicle_ids and inputs must have identical batch size.')
    prev = np.zeros_like(inputs, dtype=np.float32)
    last_by_key: dict[tuple[int, int], np.ndarray] = {}
    for idx in range(total):
        key = (int(episode_indices[idx]), int(vehicle_ids[idx]))
        current = inputs[idx].astype(np.float32, copy=False)
        prev[idx] = last_by_key.get(key, current)
        last_by_key[key] = current
    return prev


def weighted_state_error_np(
    inputs: np.ndarray,
    clean_refs: np.ndarray,
    *,
    state_weights: np.ndarray | None = None,
    state_indices: Sequence[int] | None = None,
) -> np.ndarray:
    inputs_arr = np.asarray(inputs, dtype=np.float32).reshape(-1, 11)
    clean_arr = np.asarray(clean_refs, dtype=np.float32).reshape(-1, 11)
    if inputs_arr.shape != clean_arr.shape:
        raise ValueError('inputs and clean_refs must have identical shape.')
    weights = DEFAULT_STATE_WEIGHTS.detach().cpu().numpy().astype(np.float32) if state_weights is None else np.asarray(state_weights, dtype=np.float32).reshape(-1)
    if weights.shape[0] != inputs_arr.shape[1]:
        raise ValueError('state_weights and inputs must have identical feature dimension.')
    if state_indices is not None:
        idx = np.asarray([int(v) for v in state_indices], dtype=np.int64).reshape(-1)
        if idx.size == 0:
            return np.zeros((inputs_arr.shape[0],), dtype=np.float32)
        inputs_arr = inputs_arr[:, idx]
        clean_arr = clean_arr[:, idx]
        weights = weights[idx]
    return np.mean(((inputs_arr - clean_arr) ** 2) * weights.reshape(1, -1), axis=1).astype(np.float32)


def _binary_operating_metrics_np(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs_arr = np.asarray(probs, dtype=np.float32).reshape(-1)
    if labels_arr.shape[0] != probs_arr.shape[0]:
        raise ValueError('labels and probs must have identical batch size.')
    pred = probs_arr >= float(threshold)
    label_pos = labels_arr > 0
    tp = float(np.sum(np.logical_and(pred, label_pos)))
    tn = float(np.sum(np.logical_and(~pred, ~label_pos)))
    fp = float(np.sum(np.logical_and(pred, ~label_pos)))
    fn = float(np.sum(np.logical_and(~pred, label_pos)))
    total = max(tp + tn + fp + fn, 1.0)
    precision = 0.0 if (tp + fp) <= 0.0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) <= 0.0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) <= 0.0 else 2.0 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / total
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'false_negative_rate': 0.0 if (tp + fn) <= 0.0 else float(fn / (tp + fn)),
        'false_positive_rate': 0.0 if (fp + tn) <= 0.0 else float(fp / (fp + tn)),
    }


def _grouped_sample_split_indices(
    total: int,
    *,
    val_ratio: float,
    rng: np.random.Generator,
    episode_indices: np.ndarray | None = None,
    vehicle_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    total = int(total)
    if total <= 1:
        indices = np.arange(total, dtype=np.int64)
        return indices, indices.copy()
    if episode_indices is None or vehicle_ids is None:
        indices = np.arange(total, dtype=np.int64)
        rng.shuffle(indices)
        split = max(1, int(round(total * (1.0 - float(val_ratio)))))
        split = min(max(split, 1), total - 1)
        return indices[:split], indices[split:]
    episode_arr = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    vehicle_arr = np.asarray(vehicle_ids, dtype=np.int64).reshape(-1)
    if episode_arr.shape[0] != total or vehicle_arr.shape[0] != total:
        raise ValueError('episode_indices/vehicle_ids and inputs must have identical batch size.')
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, key in enumerate(zip(episode_arr.tolist(), vehicle_arr.tolist())):
        groups[(int(key[0]), int(key[1]))].append(int(idx))
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    target_train = max(1, int(round(total * (1.0 - float(val_ratio)))))
    train_idx: list[int] = []
    val_idx: list[int] = []
    running = 0
    for pos, key in enumerate(group_keys):
        members = groups[key]
        remaining = len(group_keys) - pos - 1
        if running < target_train or remaining == 0:
            train_idx.extend(members)
            running += len(members)
        else:
            val_idx.extend(members)
    if not val_idx:
        val_idx = train_idx[-max(1, min(len(train_idx) // 5, len(train_idx) - 1)) :]
        train_idx = train_idx[: len(train_idx) - len(val_idx)]
    if not train_idx:
        train_idx = val_idx[:1]
        val_idx = val_idx[1:] or train_idx.copy()
    train_arr = np.asarray(train_idx, dtype=np.int64)
    val_arr = np.asarray(val_idx, dtype=np.int64)
    rng.shuffle(train_arr)
    rng.shuffle(val_arr)
    return train_arr, val_arr


def _split_state_tensor(seq: torch.Tensor, *, local_indices: Sequence[int] = STATE_LOCAL_IDX, global_indices: Sequence[int] = STATE_GLOBAL_IDX) -> tuple[torch.Tensor, torch.Tensor]:
    local_idx = torch.as_tensor(list(local_indices), dtype=torch.long, device=seq.device)
    local = seq.index_select(-1, local_idx) if len(local_indices) > 0 else seq.new_zeros((*seq.shape[:-1], 0))
    global_idx = torch.as_tensor(list(global_indices), dtype=torch.long, device=seq.device)
    global_part = seq.index_select(-1, global_idx) if len(global_indices) > 0 else seq.new_zeros((*seq.shape[:-1], 0))
    return local, global_part



def _split_state_array(arr: np.ndarray, *, local_indices: Sequence[int] = STATE_LOCAL_IDX, global_indices: Sequence[int] = STATE_GLOBAL_IDX) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float32)
    local = arr[..., list(local_indices)] if len(local_indices) > 0 else np.zeros((*arr.shape[:-1], 0), dtype=np.float32)
    global_part = arr[..., list(global_indices)] if len(global_indices) > 0 else np.zeros((*arr.shape[:-1], 0), dtype=np.float32)
    return local, global_part



def _merge_state_tensor(local_part: torch.Tensor, global_part: torch.Tensor, *, input_dim: int = 11, local_indices: Sequence[int] = STATE_LOCAL_IDX, global_indices: Sequence[int] = STATE_GLOBAL_IDX) -> torch.Tensor:
    out = torch.zeros((*local_part.shape[:-1], int(input_dim)), dtype=local_part.dtype, device=local_part.device)
    if len(local_indices) > 0:
        out[..., list(local_indices)] = local_part
    if len(global_indices) > 0:
        out[..., list(global_indices)] = global_part
    return out



def _gather_last_step(seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    lengths = lengths.to(seq.device, dtype=torch.long).reshape(-1)
    pos = torch.clamp(lengths - 1, min=0)
    batch_idx = torch.arange(seq.shape[0], device=seq.device)
    return seq[batch_idx, pos, :]


def _ensure_sequence_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x.view(1, 1, -1)
    if x.ndim == 2:
        return x.unsqueeze(1)
    return x


class _DualStreamSequenceMixin:
    def _encode_stream(self, encoder: nn.GRU, seq: torch.Tensor, lengths: torch.Tensor, hidden_size: int) -> torch.Tensor:
        batch_size = int(seq.shape[0])
        lengths = lengths.to(seq.device, dtype=torch.long).reshape(-1)
        hidden = torch.zeros((batch_size, hidden_size), dtype=seq.dtype, device=seq.device)
        valid_mask = lengths > 0
        if bool(valid_mask.any()):
            valid_seq = seq[valid_mask]
            valid_lengths = lengths[valid_mask].detach().cpu()
            packed = nn.utils.rnn.pack_padded_sequence(valid_seq, valid_lengths, batch_first=True, enforce_sorted=False)
            _, h_n = encoder(packed)
            hidden[valid_mask] = h_n[-1]
        return hidden

    def _prepare_inputs(self, x_local: torch.Tensor, x_global: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if x_global is None:
            seq_full = _ensure_sequence_tensor(x_local).float()
            local_seq, global_seq = _split_state_tensor(seq_full, local_indices=self.local_indices, global_indices=self.global_indices)
            return seq_full, local_seq.float(), global_seq.float()
        local_seq = _ensure_sequence_tensor(x_local).float()
        global_seq = _ensure_sequence_tensor(x_global).float()
        seq_full = _merge_state_tensor(local_seq, global_seq, input_dim=self.input_dim, local_indices=self.local_indices, global_indices=self.global_indices)
        return seq_full, local_seq, global_seq


class DenoisingAutoencoder(nn.Module, _DualStreamSequenceMixin):
    """v9 dual-stream GRU-VAE denoiser.

    Sequence axis remains vehicle-major, but state dimensions are split into:
    - local vehicle dynamics: [soc, remaining_time, cum_cost]
    - global context: [pv, wt, load, future_prices]
    """

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 1,
        seq_len: int = 8,
        decoder_hidden_dim: int = 128,
        local_indices: Sequence[int] = STATE_LOCAL_IDX,
        global_indices: Sequence[int] = STATE_GLOBAL_IDX,
        local_only_output: bool = True,
        global_passthrough: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.seq_len = int(seq_len)
        self.decoder_hidden_dim = int(decoder_hidden_dim)
        self.local_indices = tuple(int(v) for v in local_indices)
        self.global_indices = tuple(int(v) for v in global_indices)
        self.local_only_output = bool(local_only_output)
        self.global_passthrough = bool(global_passthrough)
        self.local_dim = len(self.local_indices)
        self.global_dim = len(self.global_indices)
        if self.local_dim <= 0:
            raise ValueError('DenoisingAutoencoder requires at least one defended state index.')
        self.local_hidden_dim = self.hidden_dim if self.global_dim <= 0 else max(self.hidden_dim // 2, 8)
        self.global_hidden_dim = max(self.hidden_dim - self.local_hidden_dim, 8) if self.global_dim > 0 else 0
        self.fused_hidden_dim = self.local_hidden_dim + self.global_hidden_dim

        self.local_encoder = nn.GRU(
            input_size=self.local_dim,
            hidden_size=self.local_hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.global_encoder = None
        if self.global_dim > 0:
            self.global_encoder = nn.GRU(
                input_size=self.global_dim,
                hidden_size=self.global_hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
        self.posterior_mu = nn.Linear(self.fused_hidden_dim, self.latent_dim)
        self.posterior_logvar = nn.Linear(self.fused_hidden_dim, self.latent_dim)
        self.prior_mu = nn.Linear(self.fused_hidden_dim, self.latent_dim)
        self.prior_logvar = nn.Linear(self.fused_hidden_dim, self.latent_dim)
        self.decoder_trunk = nn.Sequential(
            nn.Linear(self.latent_dim + self.fused_hidden_dim, self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.ReLU(),
        )
        self.local_head = nn.Sequential(nn.Linear(self.decoder_hidden_dim, self.local_dim), nn.Sigmoid())

    def get_config(self) -> dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'num_layers': self.num_layers,
            'seq_len': self.seq_len,
            'decoder_hidden_dim': self.decoder_hidden_dim,
            'local_indices': list(self.local_indices),
            'global_indices': list(self.global_indices),
            'local_only_output': self.local_only_output,
            'global_passthrough': self.global_passthrough,
        }

    def _encode_dual(self, local_seq: torch.Tensor, global_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h_local = self._encode_stream(self.local_encoder, local_seq, lengths, self.local_hidden_dim)
        if self.global_encoder is None or self.global_dim <= 0:
            return h_local
        h_global = self._encode_stream(self.global_encoder, global_seq, lengths, self.global_hidden_dim)
        return torch.cat([h_local, h_global], dim=1)

    def forward(
        self,
        x_local: torch.Tensor,
        x_global: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        *,
        return_stats: bool = False,
        sample_latent: bool | None = None,
    ):
        seq_full, local_seq, global_seq = self._prepare_inputs(x_local, x_global)
        batch_size, seq_steps, _ = seq_full.shape
        if lengths is None:
            lengths = torch.full((batch_size,), seq_steps, dtype=torch.long, device=seq_full.device)
        else:
            lengths = lengths.to(seq_full.device, dtype=torch.long).reshape(-1)
        h_post = self._encode_dual(local_seq, global_seq, lengths)
        prev_lengths = torch.clamp(lengths - 1, min=0)
        h_prev = self._encode_dual(local_seq, global_seq, prev_lengths)
        mu_post = self.posterior_mu(h_post)
        logvar_post = torch.clamp(self.posterior_logvar(h_post), min=-10.0, max=10.0)
        mu_prior = self.prior_mu(h_prev)
        logvar_prior = torch.clamp(self.prior_logvar(h_prev), min=-10.0, max=10.0)
        if sample_latent is None:
            sample_latent = bool(self.training)
        if sample_latent:
            std = torch.exp(0.5 * logvar_post)
            z_t = mu_post + torch.randn_like(std) * std
        else:
            z_t = mu_post
        trunk = self.decoder_trunk(torch.cat([z_t, h_prev], dim=1))
        recon_local = self.local_head(trunk)
        current_global = _gather_last_step(global_seq, lengths)
        recon = _merge_state_tensor(recon_local, current_global, input_dim=self.input_dim, local_indices=self.local_indices, global_indices=self.global_indices)
        if return_stats:
            return recon, {
                'mu_post': mu_post,
                'logvar_post': logvar_post,
                'mu_prior': mu_prior,
                'logvar_prior': logvar_prior,
                'lengths': lengths,
            }
        return recon


class DetectorGRUVAE(nn.Module, _DualStreamSequenceMixin):
    """v9 dual-stream sequence anomaly detector using grouped reconstruction error."""

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 1,
        seq_len: int = 8,
        local_indices: Sequence[int] = STATE_LOCAL_IDX,
        global_indices: Sequence[int] = STATE_GLOBAL_IDX,
        grouped_score_alpha: float = 0.5,
        local_only_score: bool = True,
        global_passthrough: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.num_layers = int(num_layers)
        self.seq_len = int(seq_len)
        self.local_indices = tuple(int(v) for v in local_indices)
        self.global_indices = tuple(int(v) for v in global_indices)
        self.local_only_score = bool(local_only_score)
        self.global_passthrough = bool(global_passthrough)
        self.local_dim = len(self.local_indices)
        self.global_dim = len(self.global_indices)
        if self.local_dim <= 0:
            raise ValueError('DetectorGRUVAE requires at least one scored state index.')
        self.local_hidden_dim = self.hidden_dim if self.global_dim <= 0 else max(self.hidden_dim // 2, 8)
        self.global_hidden_dim = max(self.hidden_dim - self.local_hidden_dim, 8) if self.global_dim > 0 else 0
        self.fused_hidden_dim = self.local_hidden_dim + self.global_hidden_dim
        self.grouped_score_alpha = float(grouped_score_alpha)

        self.local_encoder = nn.GRU(
            input_size=self.local_dim,
            hidden_size=self.local_hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.global_encoder = None
        if self.global_dim > 0:
            self.global_encoder = nn.GRU(
                input_size=self.global_dim,
                hidden_size=self.global_hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
        self.posterior_mu = nn.Linear(self.fused_hidden_dim, self.latent_dim)
        self.posterior_logvar = nn.Linear(self.fused_hidden_dim, self.latent_dim)
        self.prior_mu = nn.Linear(self.fused_hidden_dim, self.latent_dim)
        self.prior_logvar = nn.Linear(self.fused_hidden_dim, self.latent_dim)
        self.decoder_init = nn.Linear(self.fused_hidden_dim + self.latent_dim, self.fused_hidden_dim)
        self.decoder_gru = nn.GRU(
            input_size=self.latent_dim,
            hidden_size=self.fused_hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.local_head = nn.Sequential(
            nn.Linear(self.fused_hidden_dim, self.local_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.local_hidden_dim, self.local_dim),
            nn.Sigmoid(),
        )

    def get_config(self) -> dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'num_layers': self.num_layers,
            'seq_len': self.seq_len,
            'local_indices': list(self.local_indices),
            'global_indices': list(self.global_indices),
            'grouped_score_alpha': self.grouped_score_alpha,
            'local_only_score': self.local_only_score,
            'global_passthrough': self.global_passthrough,
        }

    def _encode_dual(self, local_seq: torch.Tensor, global_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h_local = self._encode_stream(self.local_encoder, local_seq, lengths, self.local_hidden_dim)
        if self.global_encoder is None or self.global_dim <= 0:
            return h_local
        h_global = self._encode_stream(self.global_encoder, global_seq, lengths, self.global_hidden_dim)
        return torch.cat([h_local, h_global], dim=1)

    def forward(
        self,
        x_local: torch.Tensor,
        x_global: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
        *,
        return_stats: bool = False,
        sample_latent: bool | None = None,
    ):
        seq_full, local_seq, global_seq = self._prepare_inputs(x_local, x_global)
        batch_size, seq_steps, _ = seq_full.shape
        if lengths is None:
            lengths = torch.full((batch_size,), seq_steps, dtype=torch.long, device=seq_full.device)
        else:
            lengths = lengths.to(seq_full.device, dtype=torch.long).reshape(-1)
        h_post = self._encode_dual(local_seq, global_seq, lengths)
        prev_lengths = torch.clamp(lengths - 1, min=0)
        h_prev = self._encode_dual(local_seq, global_seq, prev_lengths)
        mu_post = self.posterior_mu(h_post)
        logvar_post = torch.clamp(self.posterior_logvar(h_post), min=-10.0, max=10.0)
        mu_prior = self.prior_mu(h_prev)
        logvar_prior = torch.clamp(self.prior_logvar(h_prev), min=-10.0, max=10.0)
        if sample_latent is None:
            sample_latent = bool(self.training)
        if sample_latent:
            std = torch.exp(0.5 * logvar_post)
            z_t = mu_post + torch.randn_like(std) * std
        else:
            z_t = mu_post
        z_seq = z_t.unsqueeze(1).expand(batch_size, seq_steps, self.latent_dim)
        init_hidden = torch.tanh(self.decoder_init(torch.cat([h_prev, z_t], dim=1))).unsqueeze(0).expand(self.num_layers, batch_size, self.fused_hidden_dim).contiguous()
        dec_out, _ = self.decoder_gru(z_seq, init_hidden)
        recon_local = self.local_head(dec_out)
        recon_seq = _merge_state_tensor(recon_local, global_seq, input_dim=self.input_dim, local_indices=self.local_indices, global_indices=self.global_indices)
        if return_stats:
            return recon_seq, {
                'mu_post': mu_post,
                'logvar_post': logvar_post,
                'mu_prior': mu_prior,
                'logvar_prior': logvar_prior,
                'lengths': lengths,
            }
        return recon_seq


class PosteriorBenefitMLPDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1, include_temporal: bool = True) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.include_temporal = bool(include_temporal)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)

    def get_config(self) -> dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'include_temporal': self.include_temporal,
        }


def _posterior_context_feature_tensor(
    batch_size: int,
    device: torch.device,
    *,
    time_indices: np.ndarray | Sequence[int] | None = None,
    stations: np.ndarray | Sequence[int] | None = None,
    is_new_arrivals: np.ndarray | Sequence[int] | None = None,
) -> torch.Tensor:
    count = int(batch_size)
    if count <= 0:
        return torch.zeros((0, 3), dtype=torch.float32, device=device)
    def _feature_1d(values: np.ndarray | Sequence[int] | torch.Tensor | None) -> torch.Tensor:
        if values is None:
            return torch.zeros((count,), dtype=torch.float32, device=device)
        if isinstance(values, torch.Tensor):
            out = values.to(device=device, dtype=torch.float32).reshape(-1)
        else:
            out = torch.as_tensor(np.asarray(values, dtype=np.float32).reshape(-1), dtype=torch.float32, device=device)
        if out.shape[0] != count:
            raise ValueError('Context feature arrays must align with batch size.')
        return out

    time_t = _feature_1d(time_indices)
    station_t = _feature_1d(stations)
    arrival_t = _feature_1d(is_new_arrivals)
    time_scale = torch.clamp(torch.max(torch.abs(time_t)).detach(), min=1.0)
    station_scale = torch.clamp(torch.max(torch.abs(station_t)).detach(), min=1.0)
    return torch.stack([time_t / time_scale, station_t / station_scale, arrival_t], dim=1)


def build_posterior_detector_features_tensor(
    obs_inputs: np.ndarray | Sequence[np.ndarray] | torch.Tensor,
    rec_inputs: np.ndarray | Sequence[np.ndarray] | torch.Tensor,
    actor: nn.Module,
    device: torch.device,
    *,
    time_indices: np.ndarray | Sequence[int] | torch.Tensor | None = None,
    stations: np.ndarray | Sequence[int] | torch.Tensor | None = None,
    is_new_arrivals: np.ndarray | Sequence[int] | torch.Tensor | None = None,
    prev_obs_inputs: np.ndarray | Sequence[np.ndarray] | torch.Tensor | None = None,
    include_temporal: bool = True,
) -> torch.Tensor:
    if isinstance(obs_inputs, torch.Tensor):
        obs_t = obs_inputs.to(device=device, dtype=torch.float32)
    else:
        obs_t = torch.as_tensor(np.asarray(obs_inputs, dtype=np.float32).reshape(-1, 11), dtype=torch.float32, device=device)
    if isinstance(rec_inputs, torch.Tensor):
        rec_t = rec_inputs.to(device=device, dtype=torch.float32)
    else:
        rec_t = torch.as_tensor(np.asarray(rec_inputs, dtype=np.float32).reshape(-1, 11), dtype=torch.float32, device=device)
    obs_t = obs_t.reshape(-1, 11)
    rec_t = rec_t.reshape(-1, 11)
    if obs_t.shape != rec_t.shape:
        raise ValueError('obs_inputs and rec_inputs must have identical shape.')
    actor = actor.to(device).eval()
    residual_t = torch.abs(rec_t - obs_t)
    act_obs = actor(obs_t).reshape(-1, 1)
    act_rec = actor(rec_t).reshape(-1, 1)
    act_delta = torch.abs(act_rec - act_obs)
    context_t = _posterior_context_feature_tensor(
        obs_t.shape[0],
        device,
        time_indices=time_indices,
        stations=stations,
        is_new_arrivals=is_new_arrivals,
    )
    features = [obs_t, rec_t, residual_t, act_obs, act_rec, act_delta, context_t]
    if include_temporal:
        if prev_obs_inputs is None:
            prev_t = obs_t
        elif isinstance(prev_obs_inputs, torch.Tensor):
            prev_t = prev_obs_inputs.to(device=device, dtype=torch.float32).reshape(-1, 11)
        else:
            prev_t = torch.as_tensor(np.asarray(prev_obs_inputs, dtype=np.float32).reshape(-1, 11), dtype=torch.float32, device=device)
        if prev_t.shape != obs_t.shape:
            raise ValueError('prev_obs_inputs and obs_inputs must have identical shape.')
        prev_act = actor(prev_t).reshape(-1, 1)
        obs_delta = obs_t - prev_t
        rec_delta = rec_t - prev_t
        act_obs_delta = act_obs - prev_act
        act_rec_delta = act_rec - prev_act
        features.extend([prev_t, obs_delta, rec_delta, prev_act, act_obs_delta, act_rec_delta])
    return torch.cat(features, dim=1)


@torch.no_grad()
def build_posterior_detector_features(
    obs_inputs: np.ndarray | Sequence[np.ndarray],
    rec_inputs: np.ndarray | Sequence[np.ndarray],
    actor: nn.Module,
    device: torch.device,
    *,
    time_indices: np.ndarray | Sequence[int] | None = None,
    stations: np.ndarray | Sequence[int] | None = None,
    is_new_arrivals: np.ndarray | Sequence[int] | None = None,
    prev_obs_inputs: np.ndarray | Sequence[np.ndarray] | None = None,
    include_temporal: bool = True,
) -> torch.Tensor:
    return build_posterior_detector_features_tensor(
        obs_inputs,
        rec_inputs,
        actor,
        device,
        time_indices=time_indices,
        stations=stations,
        is_new_arrivals=is_new_arrivals,
        prev_obs_inputs=prev_obs_inputs,
        include_temporal=include_temporal,
    )


def posterior_detector_probabilities_tensor(
    detector_model: PosteriorBenefitMLPDetector,
    obs_inputs: np.ndarray | Sequence[np.ndarray] | torch.Tensor,
    rec_inputs: np.ndarray | Sequence[np.ndarray] | torch.Tensor,
    actor: nn.Module,
    device: torch.device,
    *,
    time_indices: np.ndarray | Sequence[int] | torch.Tensor | None = None,
    stations: np.ndarray | Sequence[int] | torch.Tensor | None = None,
    is_new_arrivals: np.ndarray | Sequence[int] | torch.Tensor | None = None,
    prev_obs_inputs: np.ndarray | Sequence[np.ndarray] | torch.Tensor | None = None,
    include_temporal: bool | None = None,
) -> torch.Tensor:
    if include_temporal is None:
        include_temporal = bool(getattr(detector_model, 'include_temporal', True))
    features = build_posterior_detector_features_tensor(
        obs_inputs,
        rec_inputs,
        actor,
        device,
        time_indices=time_indices,
        stations=stations,
        is_new_arrivals=is_new_arrivals,
        prev_obs_inputs=prev_obs_inputs,
        include_temporal=bool(include_temporal),
    )
    detector_model = detector_model.to(device).eval()
    return torch.sigmoid(detector_model(features)).reshape(-1)


@torch.no_grad()
def posterior_detector_probabilities(
    detector_model: PosteriorBenefitMLPDetector,
    obs_inputs: np.ndarray | Sequence[np.ndarray],
    rec_inputs: np.ndarray | Sequence[np.ndarray],
    actor: nn.Module,
    device: torch.device,
    *,
    time_indices: np.ndarray | Sequence[int] | None = None,
    stations: np.ndarray | Sequence[int] | None = None,
    is_new_arrivals: np.ndarray | Sequence[int] | None = None,
    prev_obs_inputs: np.ndarray | Sequence[np.ndarray] | None = None,
    include_temporal: bool | None = None,
) -> np.ndarray:
    return (
        posterior_detector_probabilities_tensor(
            detector_model,
            obs_inputs,
            rec_inputs,
            actor,
            device,
            time_indices=time_indices,
            stations=stations,
            is_new_arrivals=is_new_arrivals,
            prev_obs_inputs=prev_obs_inputs,
            include_temporal=include_temporal,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        .reshape(-1)
    )


def train_posterior_detector(
    obs_inputs: np.ndarray,
    rec_inputs: np.ndarray,
    labels: np.ndarray,
    actor: nn.Module,
    device: torch.device,
    *,
    time_indices: np.ndarray | None = None,
    stations: np.ndarray | None = None,
    is_new_arrivals: np.ndarray | None = None,
    episode_indices: np.ndarray | None = None,
    vehicle_ids: np.ndarray | None = None,
    prev_obs_inputs: np.ndarray | None = None,
    include_temporal: bool = True,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    val_ratio: float = 0.2,
    seed: int = 42,
    sample_weights: np.ndarray | None = None,
    progress_dir: str | Path | None = None,
    progress_prefix: str = 'detector',
    val_every: int = 1,
) -> tuple[PosteriorBenefitMLPDetector, DetectorTrainResult]:
    obs_arr = np.asarray(obs_inputs, dtype=np.float32).reshape(-1, 11)
    rec_arr = np.asarray(rec_inputs, dtype=np.float32).reshape(-1, 11)
    labels_arr = np.asarray(labels, dtype=np.float32).reshape(-1)
    if obs_arr.shape != rec_arr.shape:
        raise ValueError('obs_inputs and rec_inputs must have identical shape.')
    if obs_arr.shape[0] == 0 or labels_arr.shape[0] != obs_arr.shape[0]:
        raise ValueError('Posterior detector requires non-empty aligned inputs and labels.')
    if prev_obs_inputs is None and include_temporal:
        prev_obs_inputs = build_previous_step_inputs(obs_arr, episode_indices=episode_indices, vehicle_ids=vehicle_ids)
    features = build_posterior_detector_features(
        obs_arr,
        rec_arr,
        actor,
        device,
        time_indices=time_indices,
        stations=stations,
        is_new_arrivals=is_new_arrivals,
        prev_obs_inputs=prev_obs_inputs,
        include_temporal=include_temporal,
    ).detach()
    weight_arr = None if sample_weights is None else np.asarray(sample_weights, dtype=np.float32).reshape(-1)
    if weight_arr is not None and weight_arr.shape[0] != labels_arr.shape[0]:
        raise ValueError('sample_weights and labels must have identical batch size.')
    rng = np.random.default_rng(seed)
    train_idx, val_idx = _grouped_sample_split_indices(
        int(labels_arr.shape[0]),
        val_ratio=val_ratio,
        rng=rng,
        episode_indices=episode_indices,
        vehicle_ids=vehicle_ids,
    )
    model = PosteriorBenefitMLPDetector(
        input_dim=int(features.shape[1]),
        hidden_dim=hidden_dim,
        dropout=dropout,
        include_temporal=include_temporal,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(lr))
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    label_t = torch.as_tensor(labels_arr, dtype=torch.float32, device=device)
    weight_t = None if weight_arr is None else torch.as_tensor(weight_arr, dtype=torch.float32, device=device)
    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_accuracy_history: list[float] = []
    val_precision_history: list[float] = []
    val_recall_history: list[float] = []
    val_f1_history: list[float] = []
    is_best_history: list[bool] = []
    best_epoch = -1
    best_metric_name = 'val_benefit_f1'
    best_metric_value = -float('inf')
    best_metric_key = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    progress_root = None if progress_dir is None else Path(progress_dir)
    progress_name = str(progress_prefix).strip() or 'detector'
    progress_rows: list[dict[str, Any]] = []
    history_live_path = None if progress_root is None else progress_root / f'{progress_name}_history_live.csv'
    best_live_path = None if progress_root is None else progress_root / f'{progress_name}_best_live.json'
    validate_every = max(int(val_every), 1)
    for epoch in range(int(epochs)):
        model.train()
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        batch_losses: list[float] = []
        for start in range(0, len(shuffled), int(batch_size)):
            batch_idx_np = shuffled[start : start + int(batch_size)]
            batch_idx = torch.as_tensor(batch_idx_np, dtype=torch.long, device=device)
            logits = model(features[batch_idx])
            loss_vec = criterion(logits, label_t[batch_idx])
            if weight_t is not None:
                batch_weights = weight_t[batch_idx]
                loss = torch.sum(loss_vec * batch_weights) / torch.clamp(batch_weights.sum(), min=1e-6)
            else:
                loss = loss_vec.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        train_loss_history.append(float(np.mean(batch_losses) if batch_losses else 0.0))

        model.eval()
        current_best = False
        should_validate = (epoch == 0) or ((epoch + 1) % validate_every == 0) or (epoch == int(epochs) - 1)
        val_loss = float('nan')
        metrics = {
            'accuracy': float('nan'),
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
        }
        if should_validate:
            with torch.no_grad():
                val_logits = model(features[torch.as_tensor(val_idx, dtype=torch.long, device=device)])
                val_loss_vec = criterion(val_logits, label_t[torch.as_tensor(val_idx, dtype=torch.long, device=device)])
                if weight_t is not None:
                    val_weights = weight_t[torch.as_tensor(val_idx, dtype=torch.long, device=device)]
                    val_loss = float((torch.sum(val_loss_vec * val_weights) / torch.clamp(val_weights.sum(), min=1e-6)).detach().cpu())
                else:
                    val_loss = float(val_loss_vec.mean().detach().cpu())
                val_probs = torch.sigmoid(val_logits).detach().cpu().numpy().astype(np.float32).reshape(-1)
            metrics = _binary_operating_metrics_np(labels_arr[val_idx], val_probs, threshold=0.5)
            metric_key = (
                float(metrics['f1']),
                float(metrics['recall']),
                float(metrics['precision']),
                float(metrics['accuracy']),
                -float(val_loss),
            )
            if best_metric_key is None or metric_key > best_metric_key:
                best_metric_key = metric_key
                best_metric_value = float(metrics['f1'])
                best_epoch = epoch + 1
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                current_best = True
        val_loss_history.append(val_loss)
        val_accuracy_history.append(float(metrics['accuracy']))
        val_precision_history.append(float(metrics['precision']))
        val_recall_history.append(float(metrics['recall']))
        val_f1_history.append(float(metrics['f1']))
        is_best_history.append(current_best)
        progress_row = {
            'epoch': int(epoch + 1),
            'max_epochs': int(epochs),
            'train_loss': train_loss_history[-1],
            'val_loss': val_loss,
            'val_accuracy': float(metrics['accuracy']),
            'val_precision': float(metrics['precision']),
            'val_recall': float(metrics['recall']),
            'val_f1': float(metrics['f1']),
            'is_best': bool(current_best),
            'best_epoch': int(best_epoch),
            'best_metric_name': str(best_metric_name),
            'best_metric_value': float(best_metric_value),
        }
        progress_rows.append(progress_row)
        _write_progress_frame(history_live_path, progress_rows)
        _write_progress_json(
            best_live_path,
            {
                'epoch': int(epoch + 1),
                'max_epochs': int(epochs),
                'best_epoch': int(best_epoch),
                'best_metric_name': str(best_metric_name),
                'best_metric_value': float(best_metric_value),
                'latest_epoch_is_best': bool(current_best),
                'latest_epoch_validated': bool(should_validate),
                'val_every': int(validate_every),
            },
        )
        if should_validate:
            print(
                f'[DET-posterior] epoch={epoch + 1:03d}/{epochs} '
                f'train_loss={train_loss_history[-1]:.6f} val_loss={val_loss:.6f} '
                f'f1={float(metrics["f1"]):.6f} recall={float(metrics["recall"]):.6f} '
                f'precision={float(metrics["precision"]):.6f} best_epoch={best_epoch} '
                f'best_{best_metric_name}={best_metric_value:.6f} is_best={int(current_best)}',
                flush=True,
            )
        else:
            print(
                f'[DET-posterior] epoch={epoch + 1:03d}/{epochs} '
                f'train_loss={train_loss_history[-1]:.6f} val=skip '
                f'best_epoch={best_epoch} best_{best_metric_name}={best_metric_value:.6f}',
                flush=True,
            )
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model, DetectorTrainResult(
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        val_accuracy_history=val_accuracy_history,
        val_precision_history=val_precision_history,
        val_recall_history=val_recall_history,
        val_f1_history=val_f1_history,
        is_best_history=is_best_history,
        best_epoch=best_epoch,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
    )


class SequenceDenoiseDataset(Dataset):
    def __init__(
        self,
        adv_inputs: np.ndarray,
        clean_inputs: np.ndarray,
        *,
        episode_indices: np.ndarray | None,
        vehicle_ids: np.ndarray | None,
        seq_len: int,
        include_clean_sequences: bool = True,
        local_indices: Sequence[int] = STATE_LOCAL_IDX,
        global_indices: Sequence[int] = STATE_GLOBAL_IDX,
    ) -> None:
        self.adv_inputs = np.asarray(adv_inputs, dtype=np.float32).reshape(-1, 11)
        self.clean_inputs = np.asarray(clean_inputs, dtype=np.float32).reshape(-1, 11)
        if self.adv_inputs.shape != self.clean_inputs.shape:
            raise ValueError('adv_inputs and clean_inputs must align.')
        total = int(self.clean_inputs.shape[0])
        if episode_indices is None:
            episode_indices = np.zeros((total,), dtype=np.int64)
        if vehicle_ids is None:
            vehicle_ids = np.arange(total, dtype=np.int64)
        self.episode_indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
        self.vehicle_ids = np.asarray(vehicle_ids, dtype=np.int64).reshape(-1)
        self.seq_len = int(seq_len)
        self.include_clean_sequences = bool(include_clean_sequences)
        self.local_indices = tuple(int(v) for v in local_indices)
        self.global_indices = tuple(int(v) for v in global_indices)
        self._build_groups()
        self.samples: list[tuple[int, int]] = [(i, 0) for i in range(total)]
        if self.include_clean_sequences:
            self.samples.extend((i, 1) for i in range(total))

    def _build_groups(self) -> None:
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, pair in enumerate(zip(self.episode_indices.tolist(), self.vehicle_ids.tolist())):
            groups[(int(pair[0]), int(pair[1]))].append(int(idx))
        self.group_positions: dict[int, int] = {}
        self.group_members: dict[int, list[int]] = {}
        for indices in groups.values():
            for pos, idx in enumerate(indices):
                self.group_positions[idx] = pos
                self.group_members[idx] = indices

    def __len__(self) -> int:
        return len(self.samples)

    def _window_indices(self, idx: int) -> tuple[list[int], int]:
        members = self.group_members[int(idx)]
        pos = self.group_positions[int(idx)]
        start = max(0, pos - self.seq_len + 1)
        indices = members[start : pos + 1]
        return indices, len(indices)

    def __getitem__(self, row: int):
        idx, source_kind = self.samples[int(row)]
        indices, length = self._window_indices(idx)
        seq_full = np.zeros((self.seq_len, self.clean_inputs.shape[1]), dtype=np.float32)
        source = self.clean_inputs if int(source_kind) == 1 else self.adv_inputs
        seq_full[:length] = source[indices]
        x_local, x_global = _split_state_array(seq_full, local_indices=self.local_indices, global_indices=self.global_indices)
        target_full = self.clean_inputs[idx]
        target_local, target_global = _split_state_array(target_full.reshape(1, -1), local_indices=self.local_indices, global_indices=self.global_indices)
        return {
            'x_local': torch.as_tensor(x_local, dtype=torch.float32),
            'x_global': torch.as_tensor(x_global, dtype=torch.float32),
            'length': torch.as_tensor(length, dtype=torch.long),
            'target_full': torch.as_tensor(target_full, dtype=torch.float32),
            'target_local': torch.as_tensor(target_local.reshape(-1), dtype=torch.float32),
            'target_global': torch.as_tensor(target_global.reshape(-1), dtype=torch.float32),
            'source_kind': torch.as_tensor(int(source_kind), dtype=torch.long),
        }


class DetectorSequenceDataset(Dataset):
    def __init__(
        self,
        clean_inputs: np.ndarray,
        *,
        episode_indices: np.ndarray | None,
        vehicle_ids: np.ndarray | None,
        seq_len: int,
        local_indices: Sequence[int] = STATE_LOCAL_IDX,
        global_indices: Sequence[int] = STATE_GLOBAL_IDX,
    ) -> None:
        self.clean_inputs = np.asarray(clean_inputs, dtype=np.float32).reshape(-1, 11)
        total = int(self.clean_inputs.shape[0])
        if episode_indices is None:
            episode_indices = np.zeros((total,), dtype=np.int64)
        if vehicle_ids is None:
            vehicle_ids = np.arange(total, dtype=np.int64)
        self.episode_indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
        self.vehicle_ids = np.asarray(vehicle_ids, dtype=np.int64).reshape(-1)
        self.seq_len = int(seq_len)
        self.local_indices = tuple(int(v) for v in local_indices)
        self.global_indices = tuple(int(v) for v in global_indices)
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, pair in enumerate(zip(self.episode_indices.tolist(), self.vehicle_ids.tolist())):
            groups[(int(pair[0]), int(pair[1]))].append(int(idx))
        self.samples: list[list[int]] = []
        for indices in groups.values():
            for pos in range(len(indices)):
                start = max(0, pos - self.seq_len + 1)
                self.samples.append(indices[start : pos + 1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        indices = self.samples[int(idx)]
        length = len(indices)
        seq_full = np.zeros((self.seq_len, self.clean_inputs.shape[1]), dtype=np.float32)
        seq_full[:length] = self.clean_inputs[indices]
        x_local, x_global = _split_state_array(seq_full, local_indices=self.local_indices, global_indices=self.global_indices)
        return {
            'x_local': torch.as_tensor(x_local, dtype=torch.float32),
            'x_global': torch.as_tensor(x_global, dtype=torch.float32),
            'length': torch.as_tensor(length, dtype=torch.long),
            'target_seq': torch.as_tensor(seq_full, dtype=torch.float32),
        }



def _build_history_windows_numpy(inputs: np.ndarray, *, episode_indices: np.ndarray | None, vehicle_ids: np.ndarray | None, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.asarray(inputs, dtype=np.float32).reshape(-1, 11)
    total = int(inputs.shape[0])
    if episode_indices is None:
        episode_indices = np.zeros((total,), dtype=np.int64)
    if vehicle_ids is None:
        vehicle_ids = np.arange(total, dtype=np.int64)
    episode_indices = np.asarray(episode_indices, dtype=np.int64).reshape(-1)
    vehicle_ids = np.asarray(vehicle_ids, dtype=np.int64).reshape(-1)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, pair in enumerate(zip(episode_indices.tolist(), vehicle_ids.tolist())):
        groups[(int(pair[0]), int(pair[1]))].append(int(idx))
    group_positions: dict[int, int] = {}
    group_members: dict[int, list[int]] = {}
    for indices in groups.values():
        for pos, idx in enumerate(indices):
            group_positions[idx] = pos
            group_members[idx] = indices
    seqs = np.zeros((total, int(seq_len), inputs.shape[1]), dtype=np.float32)
    lengths = np.zeros((total,), dtype=np.int64)
    for idx in range(total):
        members = group_members[int(idx)]
        pos = group_positions[int(idx)]
        start = max(0, pos - int(seq_len) + 1)
        indices = members[start : pos + 1]
        seqs[idx, : len(indices)] = inputs[indices]
        lengths[idx] = len(indices)
    return seqs, lengths


def _masked_reconstruction_loss(recon_seq: torch.Tensor, target_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    batch, steps, _ = recon_seq.shape
    mask = torch.arange(steps, device=recon_seq.device).unsqueeze(0) < lengths.unsqueeze(1)
    diff = (recon_seq - target_seq) ** 2
    diff = diff.mean(dim=2)
    diff = diff * mask.float()
    denom = torch.clamp(mask.float().sum(), min=1.0)
    return diff.sum() / denom


def _kl_to_conditional_prior(mu_post: torch.Tensor, logvar_post: torch.Tensor, mu_prior: torch.Tensor, logvar_prior: torch.Tensor) -> torch.Tensor:
    var_post = torch.exp(logvar_post)
    var_prior = torch.exp(logvar_prior)
    kl = 0.5 * (
        (logvar_prior - logvar_post)
        + (var_post + (mu_post - mu_prior) ** 2) / torch.clamp(var_prior, min=1e-8)
        - 1.0
    )
    return kl.sum(dim=1).mean()


def weighted_state_loss(pred: torch.Tensor, target: torch.Tensor, state_weights: torch.Tensor | None = None) -> torch.Tensor:
    weights = DEFAULT_STATE_WEIGHTS if state_weights is None else state_weights
    weights = weights.to(pred.device, dtype=pred.dtype).view(1, -1)
    return torch.mean(((pred - target) ** 2) * weights)


def grouped_state_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    local_weight: float = 1.0,
    global_weight: float = 0.5,
    state_weights: torch.Tensor | None = None,
    local_indices: Sequence[int] = STATE_LOCAL_IDX,
    global_indices: Sequence[int] = STATE_GLOBAL_IDX,
) -> torch.Tensor:
    weights = DEFAULT_STATE_WEIGHTS if state_weights is None else state_weights
    weights = weights.to(pred.device, dtype=pred.dtype)
    local_idx = torch.as_tensor([int(v) for v in local_indices], dtype=torch.long, device=pred.device)
    global_idx = torch.as_tensor([int(v) for v in global_indices], dtype=torch.long, device=pred.device)
    if local_idx.numel() > 0:
        pred_local = pred.index_select(1, local_idx)
        tgt_local = target.index_select(1, local_idx)
        local_weights = weights.index_select(0, local_idx).view(1, -1)
        local_loss = torch.mean(((pred_local - tgt_local) ** 2) * local_weights)
    else:
        local_loss = pred.new_tensor(0.0)
    if global_idx.numel() > 0:
        pred_global = pred.index_select(1, global_idx)
        tgt_global = target.index_select(1, global_idx)
        global_weights = weights.index_select(0, global_idx).view(1, -1)
        global_loss = torch.mean(((pred_global - tgt_global) ** 2) * global_weights)
    else:
        global_loss = pred.new_tensor(0.0)
    return float(local_weight) * local_loss + float(global_weight) * global_loss


def _action_mse(actor: nn.Module, clean: torch.Tensor, other: torch.Tensor) -> float:
    with torch.no_grad():
        clean_act = actor(clean).detach().cpu().numpy().reshape(-1)
        other_act = actor(other).detach().cpu().numpy().reshape(-1)
    return float(np.mean((clean_act - other_act) ** 2)) if clean_act.size else 0.0


def train_dae(
    bundle,
    actor: nn.Module | None,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    log_every: int = 1,
    seq_len: int = 8,
    hidden_dim: int = 128,
    latent_dim: int = 64,
    num_layers: int = 1,
    decoder_hidden_dim: int = 128,
    beta_kl: float = 1e-3,
    lambda_recon: float = 1.0,
    lambda_identity: float = 1.0,
    lambda_robust: float = 0.0,
    include_clean_sequences: bool = True,
    validator: Callable[[nn.Module], dict] | None = None,
    val_every: int = 1,
    select_by: str = 'reward_recovery',
    local_recon_weight: float = 1.0,
    global_recon_weight: float = 0.5,
    state_scope: str = 'local',
    progress_dir: str | Path | None = None,
    progress_prefix: str = 'dae',
) -> tuple[DenoisingAutoencoder, DAETrainResult]:
    adv_inputs = np.asarray(bundle.adv_inputs, dtype=np.float32).reshape(-1, 11)
    clean_inputs = np.asarray(bundle.clean_inputs, dtype=np.float32).reshape(-1, 11)
    episode_indices = getattr(bundle, 'episode_indices', None)
    vehicle_ids = getattr(bundle, 'vehicle_ids', None)
    scope = canonical_state_scope(state_scope)
    defended_indices = defended_indices_for_scope(scope)
    conditioning_indices = conditioning_indices_for_scope(scope)

    model = DenoisingAutoencoder(
        input_dim=11,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        seq_len=seq_len,
        decoder_hidden_dim=decoder_hidden_dim,
        local_indices=defended_indices,
        global_indices=conditioning_indices,
    ).to(device)
    dataset = SequenceDenoiseDataset(
        adv_inputs,
        clean_inputs,
        episode_indices=episode_indices,
        vehicle_ids=vehicle_ids,
        seq_len=seq_len,
        include_clean_sequences=include_clean_sequences,
        local_indices=defended_indices,
        global_indices=conditioning_indices,
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=float(lr))
    if actor is not None:
        actor = actor.to(device).eval()
        for p in actor.parameters():
            p.requires_grad_(False)

    loss_history: list[float] = []
    recon_loss_history: list[float] = []
    kl_loss_history: list[float] = []
    robust_loss_history: list[float] = []
    clean_adv_action_mse: list[float] = []
    clean_recovered_action_mse: list[float] = []
    clean_adv_state_mse: list[float] = []
    clean_recovered_state_mse: list[float] = []
    validator_rows: list[dict[str, Any]] = []
    is_best_history: list[bool] = []

    best_epoch = -1
    best_metric_name = 'loss'
    best_metric_value = float('inf')
    best_state_dict = None
    progress_root = None if progress_dir is None else Path(progress_dir)
    progress_name = str(progress_prefix).strip() or 'dae'
    progress_rows: list[dict[str, Any]] = []
    history_live_path = None if progress_root is None else progress_root / f'{progress_name}_history_live.csv'
    validation_live_path = None if progress_root is None else progress_root / f'{progress_name}_validation_history_live.csv'
    best_live_path = None if progress_root is None else progress_root / f'{progress_name}_best_live.json'

    for epoch in range(int(epochs)):
        model.train()
        total_loss = total_rec = total_kl = total_rob = 0.0
        steps = 0
        for batch in loader:
            x_local = batch['x_local'].to(device)
            x_global = batch['x_global'].to(device)
            lengths = batch['length'].to(device)
            target_full = batch['target_full'].to(device)
            source_kind = batch['source_kind'].to(device)
            recon, stats = model(x_local, x_global, lengths, return_stats=True, sample_latent=True)
            zero_loss = torch.tensor(0.0, device=device)
            attack_mask = source_kind == 0
            clean_mask = source_kind == 1
            attack_rec_loss = (
                grouped_state_loss(
                    recon[attack_mask],
                    target_full[attack_mask],
                    local_weight=local_recon_weight,
                    global_weight=0.0,
                    local_indices=defended_indices,
                    global_indices=conditioning_indices,
                )
                if bool(torch.any(attack_mask).item())
                else zero_loss
            )
            identity_rec_loss = (
                grouped_state_loss(
                    recon[clean_mask],
                    target_full[clean_mask],
                    local_weight=local_recon_weight,
                    global_weight=0.0,
                    local_indices=defended_indices,
                    global_indices=conditioning_indices,
                )
                if bool(torch.any(clean_mask).item())
                else zero_loss
            )
            rec_loss = float(lambda_recon) * attack_rec_loss + float(lambda_identity) * identity_rec_loss
            kl_loss = _kl_to_conditional_prior(stats['mu_post'], stats['logvar_post'], stats['mu_prior'], stats['logvar_prior'])
            robust_loss = torch.tensor(0.0, device=device)
            if actor is not None and float(lambda_robust) > 0.0:
                with torch.no_grad():
                    clean_action = actor(target_full)
                recovered_action = actor(recon)
                robust_loss = torch.mean((recovered_action - clean_action) ** 2)
            loss = rec_loss + float(beta_kl) * kl_loss + float(lambda_robust) * robust_loss
            optimizer.zero_grad()
            loss.backward()
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
            clean_t = torch.as_tensor(clean_inputs, dtype=torch.float32, device=device)
            adv_t = torch.as_tensor(adv_inputs, dtype=torch.float32, device=device)
            rec_np = dae_reconstruction_with_history(model, adv_inputs, device, episode_indices=episode_indices, vehicle_ids=vehicle_ids, seq_len=seq_len)
            rec_t = torch.as_tensor(rec_np, dtype=torch.float32, device=device)
            idx_t = torch.as_tensor(list(defended_indices), dtype=torch.long, device=device)
            clean_adv_state_mse.append(float(torch.mean((adv_t.index_select(1, idx_t) - clean_t.index_select(1, idx_t)) ** 2).detach().cpu()))
            clean_recovered_state_mse.append(float(torch.mean((rec_t.index_select(1, idx_t) - clean_t.index_select(1, idx_t)) ** 2).detach().cpu()))
            if actor is not None:
                clean_adv_action_mse.append(_action_mse(actor, clean_t, adv_t))
                clean_recovered_action_mse.append(_action_mse(actor, clean_t, rec_t))
            else:
                clean_adv_action_mse.append(float('nan'))
                clean_recovered_action_mse.append(float('nan'))

        current_best = False
        compare_better = False
        metric_value = float('nan')
        if validator is not None and int(val_every) > 0 and ((epoch + 1) % int(val_every) == 0):
            validator_row = dict(validator(model))
            validator_row['epoch'] = int(epoch + 1)
            validator_rows.append(validator_row)
            metric_value = float(validator_row.get(select_by, loss_history[-1]))
            best_metric_name = str(select_by)
            metric_token = str(select_by).strip().lower()
            maximize_metric = any(token in metric_token for token in ('reward', 'recovery', 'score', 'ratio', 'f1', 'accuracy', 'reduction'))
            if maximize_metric:
                compare_better = metric_value > best_metric_value
                if best_epoch < 0:
                    compare_better = True
            else:
                compare_better = metric_value < best_metric_value
                if best_epoch < 0:
                    compare_better = True
        elif validator is not None and int(val_every) > 0:
            compare_better = False
        else:
            metric_value = loss_history[-1]
            compare_better = (metric_value < best_metric_value) or (best_epoch < 0)

        if compare_better:
            best_metric_value = float(metric_value)
            best_epoch = int(epoch + 1)
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            current_best = True
        is_best_history.append(current_best)
        best_metric_out = np.nan if best_epoch < 0 else float(best_metric_value)
        progress_row = {
            'epoch': int(epoch + 1),
            'max_epochs': int(epochs),
            'loss': loss_history[-1],
            'recon_loss': recon_loss_history[-1],
            'kl_loss': kl_loss_history[-1],
            'robust_loss': robust_loss_history[-1],
            'clean_adv_action_mse': clean_adv_action_mse[-1] if clean_adv_action_mse else np.nan,
            'clean_recovered_action_mse': clean_recovered_action_mse[-1] if clean_recovered_action_mse else np.nan,
            'clean_adv_state_mse': clean_adv_state_mse[-1] if clean_adv_state_mse else np.nan,
            'clean_recovered_state_mse': clean_recovered_state_mse[-1] if clean_recovered_state_mse else np.nan,
            'is_best': bool(current_best),
            'best_epoch': int(best_epoch),
            'best_metric_name': str(best_metric_name),
            'best_metric_value': best_metric_out,
        }
        if validator_rows and int(validator_rows[-1].get('epoch', -1)) == int(epoch + 1):
            progress_row.update({k: v for k, v in validator_rows[-1].items() if k != 'epoch'})
            _write_progress_frame(validation_live_path, validator_rows)
        progress_rows.append(progress_row)
        _write_progress_frame(history_live_path, progress_rows)
        _write_progress_json(
            best_live_path,
            {
                'epoch': int(epoch + 1),
                'max_epochs': int(epochs),
                'best_epoch': int(best_epoch),
                'best_metric_name': str(best_metric_name),
                'best_metric_value': best_metric_out,
                'latest_epoch_is_best': bool(current_best),
                'latest_validated_epoch': int(validator_rows[-1]['epoch']) if validator_rows else None,
            },
        )
        if epoch == 0 or (epoch + 1) % int(max(log_every, 1)) == 0 or epoch == epochs - 1:
            best_metric_text = 'n/a' if best_epoch < 0 else f'{best_metric_value:.6f}'
            print(
                f'[DAE-v9] epoch={epoch + 1:03d}/{epochs} '
                f'loss={loss_history[-1]:.6f} rec={recon_loss_history[-1]:.6f} '
                f'kl={kl_loss_history[-1]:.6f} robust={robust_loss_history[-1]:.6f} '
                f'best_epoch={best_epoch} best_{best_metric_name}={best_metric_text} '
                f'is_best={int(current_best)}',
                flush=True,
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
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
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
    )


def train_sequence_detector(
    clean_inputs: np.ndarray,
    device: torch.device,
    *,
    episode_indices: np.ndarray | None,
    vehicle_ids: np.ndarray | None,
    eval_clean_inputs: np.ndarray | None = None,
    eval_adv_inputs: np.ndarray | None = None,
    eval_episode_indices: np.ndarray | None = None,
    eval_vehicle_ids: np.ndarray | None = None,
    seq_len: int = 8,
    hidden_dim: int = 128,
    latent_dim: int = 64,
    num_layers: int = 1,
    beta_kl: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    seed: int = 42,
    state_scope: str = 'local',
    progress_dir: str | Path | None = None,
    progress_prefix: str = 'detector',
) -> tuple[DetectorGRUVAE, DetectorTrainResult]:
    scope = canonical_state_scope(state_scope)
    defended_indices = defended_indices_for_scope(scope)
    conditioning_indices = conditioning_indices_for_scope(scope)
    dataset = DetectorSequenceDataset(
        clean_inputs,
        episode_indices=episode_indices,
        vehicle_ids=vehicle_ids,
        seq_len=seq_len,
        local_indices=defended_indices,
        global_indices=conditioning_indices,
    )
    total = len(dataset)
    if total == 0:
        raise ValueError('Detector training requires non-empty Dnormal sequences.')
    rng = np.random.default_rng(seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    split = max(1, int(round(total * (1.0 - float(val_ratio)))))
    if split >= total:
        split = max(total - 1, 1)
    train_loader = DataLoader(Subset(dataset, indices[:split].tolist()), batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(Subset(dataset, indices[split:].tolist()), batch_size=int(batch_size), shuffle=False)

    model = DetectorGRUVAE(
        input_dim=11,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        seq_len=seq_len,
        local_indices=defended_indices,
        global_indices=conditioning_indices,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(lr))

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    val_accuracy_history: list[float] = []
    val_precision_history: list[float] = []
    val_recall_history: list[float] = []
    val_f1_history: list[float] = []
    is_best_history: list[bool] = []
    best_epoch = -1
    best_metric_name = 'val_loss'
    best_metric_value = float('inf')
    best_metric_key = None
    best_state_dict = None
    progress_root = None if progress_dir is None else Path(progress_dir)
    progress_name = str(progress_prefix).strip() or 'detector'
    progress_rows: list[dict[str, Any]] = []
    history_live_path = None if progress_root is None else progress_root / f'{progress_name}_history_live.csv'
    best_live_path = None if progress_root is None else progress_root / f'{progress_name}_best_live.json'

    for epoch in range(int(epochs)):
        model.train()
        batch_losses = []
        for batch in train_loader:
            x_local = batch['x_local'].to(device)
            x_global = batch['x_global'].to(device)
            lengths = batch['length'].to(device)
            target_seq = batch['target_seq'].to(device)
            recon_seq, stats = model(x_local, x_global, lengths, return_stats=True, sample_latent=True)
            rec_loss = _masked_reconstruction_loss(recon_seq, target_seq, lengths)
            kl_loss = _kl_to_conditional_prior(stats['mu_post'], stats['logvar_post'], stats['mu_prior'], stats['logvar_prior'])
            loss = rec_loss + float(beta_kl) * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        train_loss_history.append(float(np.mean(batch_losses) if batch_losses else 0.0))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x_local = batch['x_local'].to(device)
                x_global = batch['x_global'].to(device)
                lengths = batch['length'].to(device)
                target_seq = batch['target_seq'].to(device)
                recon_seq, stats = model(x_local, x_global, lengths, return_stats=True, sample_latent=False)
                rec_loss = _masked_reconstruction_loss(recon_seq, target_seq, lengths)
                kl_loss = _kl_to_conditional_prior(stats['mu_post'], stats['logvar_post'], stats['mu_prior'], stats['logvar_prior'])
                val_losses.append(float((rec_loss + float(beta_kl) * kl_loss).detach().cpu()))
        val_loss = float(np.mean(val_losses) if val_losses else train_loss_history[-1])
        val_loss_history.append(val_loss)
        current_best = False

        if eval_clean_inputs is not None and eval_adv_inputs is not None:
            metrics, _, _ = evaluate_detector_metrics(
                model,
                eval_clean_inputs,
                eval_adv_inputs,
                device,
                episode_indices=eval_episode_indices,
                vehicle_ids=eval_vehicle_ids,
                threshold=None,
                grid_size=31,
            )
            val_accuracy_history.append(float(metrics['clean_accuracy']))
            val_precision_history.append(float(metrics['attack_precision']))
            val_recall_history.append(float(metrics['attack_recall']))
            val_f1_history.append(float(metrics['attack_f1']))
            clean_ok = float(metrics['clean_accuracy']) >= float(DETECTOR_SELECTION_MIN_CLEAN_ACCURACY)
            metric_key = (
                1.0 if clean_ok else 0.0,
                float(metrics['attack_f1']),
                float(metrics['attack_recall']),
                float(metrics['attack_precision']),
                float(metrics['clean_accuracy']),
                -float(metrics['false_negative_rate']),
                -float(val_loss),
            )
            if best_metric_key is None or metric_key > best_metric_key:
                best_metric_key = metric_key
                best_metric_name = 'val_attack_f1_under_clean_floor'
                best_metric_value = float(metrics['attack_f1'])
                best_epoch = int(epoch + 1)
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                current_best = True
        else:
            val_accuracy_history.append(float('nan'))
            val_precision_history.append(float('nan'))
            val_recall_history.append(float('nan'))
            val_f1_history.append(float('nan'))
            current_best = val_loss < best_metric_value or best_epoch < 0
            if current_best:
                best_metric_value = val_loss
                best_epoch = int(epoch + 1)
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        is_best_history.append(current_best)
        progress_row = {
            'epoch': int(epoch + 1),
            'max_epochs': int(epochs),
            'train_loss': train_loss_history[-1],
            'val_loss': val_loss,
            'val_accuracy': val_accuracy_history[-1],
            'val_precision': val_precision_history[-1],
            'val_recall': val_recall_history[-1],
            'val_f1': val_f1_history[-1],
            'is_best': bool(current_best),
            'best_epoch': int(best_epoch),
            'best_metric_name': str(best_metric_name),
            'best_metric_value': float(best_metric_value),
        }
        progress_rows.append(progress_row)
        _write_progress_frame(history_live_path, progress_rows)
        _write_progress_json(
            best_live_path,
            {
                'epoch': int(epoch + 1),
                'max_epochs': int(epochs),
                'best_epoch': int(best_epoch),
                'best_metric_name': str(best_metric_name),
                'best_metric_value': float(best_metric_value),
                'latest_epoch_is_best': bool(current_best),
            },
        )
        print(
            f'[DET-sequence] epoch={epoch + 1:03d}/{epochs} '
            f'train_loss={train_loss_history[-1]:.6f} val_loss={val_loss:.6f} '
            f'f1={val_f1_history[-1]:.6f} recall={val_recall_history[-1]:.6f} '
            f'precision={val_precision_history[-1]:.6f} best_epoch={best_epoch} '
            f'best_{best_metric_name}={best_metric_value:.6f} is_best={int(current_best)}',
            flush=True,
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model, DetectorTrainResult(
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        val_accuracy_history=val_accuracy_history,
        val_precision_history=val_precision_history,
        val_recall_history=val_recall_history,
        val_f1_history=val_f1_history,
        is_best_history=is_best_history,
        best_epoch=best_epoch,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
    )


def _detector_last_step_local_score(
    seq_t: torch.Tensor,
    recon_seq: torch.Tensor,
    lengths: torch.Tensor,
    *,
    local_indices: Sequence[int] = STATE_LOCAL_IDX,
    global_indices: Sequence[int] = STATE_GLOBAL_IDX,
) -> torch.Tensor:
    pos = torch.clamp(lengths - 1, min=0).long()
    batch_idx = torch.arange(seq_t.shape[0], device=seq_t.device)
    obs_t = seq_t[batch_idx, pos, :]
    recon_t = recon_seq[batch_idx, pos, :]
    obs_local, _ = _split_state_tensor(obs_t, local_indices=local_indices, global_indices=global_indices)
    rec_local, _ = _split_state_tensor(recon_t, local_indices=local_indices, global_indices=global_indices)
    return torch.max(torch.abs(obs_local - rec_local), dim=1).values


@torch.no_grad()
def detector_anomaly_scores(model: DetectorGRUVAE, obs_inputs: np.ndarray, device: torch.device, *, episode_indices: np.ndarray | None, vehicle_ids: np.ndarray | None, batch_size: int = 1024, seq_len: int | None = None) -> np.ndarray:
    inputs = np.asarray(obs_inputs, dtype=np.float32).reshape(-1, 11)
    if inputs.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    seq_len = int(seq_len or getattr(model, 'seq_len', 1))
    seqs, lengths = _build_history_windows_numpy(inputs, episode_indices=episode_indices, vehicle_ids=vehicle_ids, seq_len=seq_len)
    model = model.to(device).eval()
    out: list[np.ndarray] = []
    for start in range(0, seqs.shape[0], int(batch_size)):
        end = min(seqs.shape[0], start + int(batch_size))
        seq_t = torch.as_tensor(seqs[start:end], dtype=torch.float32, device=device)
        len_t = torch.as_tensor(lengths[start:end], dtype=torch.long, device=device)
        recon_seq = model(seq_t, None, len_t, sample_latent=False)
        scores = _detector_last_step_local_score(
            seq_t,
            recon_seq,
            len_t,
            local_indices=getattr(model, 'local_indices', STATE_LOCAL_IDX),
            global_indices=getattr(model, 'global_indices', STATE_GLOBAL_IDX),
        ).detach().cpu().numpy().astype(np.float32)
        out.append(scores)
    return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.float32)


def select_canomaly_from_scores(clean_scores: np.ndarray, attacked_scores: np.ndarray, *, grid_size: int = 31, min_clean_accuracy: float | None = DETECTOR_SELECTION_MIN_CLEAN_ACCURACY) -> tuple[float, list[dict[str, Any]], int]:
    clean_scores = np.asarray(clean_scores, dtype=np.float32).reshape(-1)
    attacked_scores = np.asarray(attacked_scores, dtype=np.float32).reshape(-1)
    all_scores = np.concatenate([clean_scores, attacked_scores], axis=0) if attacked_scores.size else clean_scores.copy()
    if all_scores.size == 0:
        return float('inf'), [], -1
    thresholds = np.unique(np.quantile(all_scores, np.linspace(0.0, 1.0, max(int(grid_size), 5)))).tolist()
    thresholds.append(float(np.max(all_scores) + 1e-6))
    rows: list[dict[str, Any]] = []
    best_idx = -1
    best_key = None
    clean_floor = None if min_clean_accuracy is None else float(np.clip(min_clean_accuracy, 0.0, 1.0))
    for idx, threshold in enumerate(thresholds):
        clean_pred = clean_scores > float(threshold)
        attack_pred = attacked_scores > float(threshold)
        tp = float(np.sum(attack_pred))
        fn = float(attack_pred.size - np.sum(attack_pred))
        fp = float(np.sum(clean_pred))
        tn = float(clean_pred.size - np.sum(clean_pred))
        precision = 0.0 if (tp + fp) <= 0.0 else tp / (tp + fp)
        recall = 0.0 if attacked_scores.size == 0 else tp / max(float(attacked_scores.size), 1e-6)
        f1 = 0.0 if (precision + recall) <= 0.0 else 2.0 * precision * recall / max(precision + recall, 1e-6)
        clean_accuracy = 0.0 if (tn + fp) <= 0.0 else tn / (tn + fp)
        fnr = 0.0 if (tp + fn) <= 0.0 else fn / (tp + fn)
        clean_ok = clean_floor is None or clean_accuracy >= clean_floor
        row = {
            'candidate_rank': int(idx),
            'threshold': float(threshold),
            'clean_accuracy': float(clean_accuracy),
            'false_positive_rate': float(1.0 - clean_accuracy),
            'attack_precision': float(precision),
            'attack_recall': float(recall),
            'attack_f1': float(f1),
            'false_negative_rate': float(fnr),
            'clean_accuracy_floor': None if clean_floor is None else float(clean_floor),
            'clean_constraint_ok': bool(clean_ok),
        }
        rows.append(row)
        if clean_ok:
            key = (1.0, float(f1), float(recall), float(clean_accuracy), -float(fnr), -float(threshold))
        else:
            key = (0.0, float(clean_accuracy), float(f1), float(recall), -float(fnr), -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best_idx = idx
    return float(rows[best_idx]['threshold']), rows, int(best_idx)


@torch.no_grad()
def dae_reconstruction_with_history(model: nn.Module, obs_inputs: np.ndarray, device: torch.device, *, episode_indices: np.ndarray | None, vehicle_ids: np.ndarray | None, batch_size: int = 1024, seq_len: int | None = None) -> np.ndarray:
    inputs = np.asarray(obs_inputs, dtype=np.float32).reshape(-1, 11)
    if inputs.shape[0] == 0:
        return np.zeros((0, 11), dtype=np.float32)
    seq_len = int(seq_len or getattr(model, 'seq_len', 1))
    seqs, lengths = _build_history_windows_numpy(inputs, episode_indices=episode_indices, vehicle_ids=vehicle_ids, seq_len=seq_len)
    model = model.to(device).eval()
    out: list[np.ndarray] = []
    for start in range(0, seqs.shape[0], int(batch_size)):
        end = min(seqs.shape[0], start + int(batch_size))
        seq_t = torch.as_tensor(seqs[start:end], dtype=torch.float32, device=device)
        len_t = torch.as_tensor(lengths[start:end], dtype=torch.long, device=device)
        recon = model(seq_t, None, len_t, sample_latent=False).detach().cpu().numpy().astype(np.float32)
        out.append(recon)
    result = np.concatenate(out, axis=0) if out else np.zeros((0, 11), dtype=np.float32)
    return _clip_reconstruction_to_model_bounds(model, result)


@torch.no_grad()
def reconstruction_batch(model: nn.Module, obs_inputs: np.ndarray | list[np.ndarray], device: torch.device) -> np.ndarray:
    inputs = np.asarray(obs_inputs, dtype=np.float32).reshape(-1, 11)
    if inputs.shape[0] == 0:
        return np.zeros((0, 11), dtype=np.float32)
    model = model.to(device).eval()
    obs_t = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    recon = model(obs_t.unsqueeze(1), None, sample_latent=False).detach().cpu().numpy().astype(np.float32)
    return _clip_reconstruction_to_model_bounds(model, recon)


class SequentialDAERuntime:
    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.seq_len = int(getattr(model, 'seq_len', 1))
        self.buffers: defaultdict[tuple[int, int], deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.seq_len))

    def reset(self) -> None:
        self.buffers.clear()

    @torch.no_grad()
    def reconstruct_batch(self, obs_batch: Sequence[np.ndarray] | np.ndarray, *, vehicle_ids: Sequence[int] | np.ndarray, episode_index: int = 0) -> np.ndarray:
        batch = np.asarray(obs_batch, dtype=np.float32).reshape(-1, 11)
        vehicle_ids = np.asarray(vehicle_ids, dtype=np.int64).reshape(-1)
        seqs = np.zeros((batch.shape[0], self.seq_len, batch.shape[1]), dtype=np.float32)
        lengths = np.zeros((batch.shape[0],), dtype=np.int64)
        keys: list[tuple[int, int]] = []
        for i in range(batch.shape[0]):
            key = (int(episode_index), int(vehicle_ids[i]))
            keys.append(key)
            hist = list(self.buffers[key])
            full = hist + [batch[i]]
            keep = full[-self.seq_len :]
            seqs[i, : len(keep)] = np.asarray(keep, dtype=np.float32)
            lengths[i] = len(keep)
        seq_t = torch.as_tensor(seqs, dtype=torch.float32, device=self.device)
        len_t = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        recon = self.model(seq_t, None, len_t, sample_latent=False).detach().cpu().numpy().astype(np.float32)
        for key, obs in zip(keys, batch):
            self.buffers[key].append(np.asarray(obs, dtype=np.float32).reshape(-1))
        return _clip_reconstruction_to_model_bounds(self.model, recon)


class SequentialDetectorRuntime:
    def __init__(self, model: DetectorGRUVAE, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.seq_len = int(getattr(model, 'seq_len', 1))
        self.buffers: defaultdict[tuple[int, int], deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=self.seq_len))

    def reset(self) -> None:
        self.buffers.clear()

    @torch.no_grad()
    def score_batch(self, obs_batch: Sequence[np.ndarray] | np.ndarray, *, vehicle_ids: Sequence[int] | np.ndarray, episode_index: int = 0, threshold: float | None = None) -> tuple[np.ndarray, list[bool]]:
        batch = np.asarray(obs_batch, dtype=np.float32).reshape(-1, 11)
        vehicle_ids = np.asarray(vehicle_ids, dtype=np.int64).reshape(-1)
        seqs = np.zeros((batch.shape[0], self.seq_len, batch.shape[1]), dtype=np.float32)
        lengths = np.zeros((batch.shape[0],), dtype=np.int64)
        keys: list[tuple[int, int]] = []
        for i in range(batch.shape[0]):
            key = (int(episode_index), int(vehicle_ids[i]))
            keys.append(key)
            hist = list(self.buffers[key])
            full = hist + [batch[i]]
            keep = full[-self.seq_len :]
            seqs[i, : len(keep)] = np.asarray(keep, dtype=np.float32)
            lengths[i] = len(keep)
        seq_t = torch.as_tensor(seqs, dtype=torch.float32, device=self.device)
        len_t = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        recon_seq = self.model(seq_t, None, len_t, sample_latent=False)
        scores = _detector_last_step_local_score(
            seq_t,
            recon_seq,
            len_t,
            local_indices=getattr(self.model, 'local_indices', STATE_LOCAL_IDX),
            global_indices=getattr(self.model, 'global_indices', STATE_GLOBAL_IDX),
        ).detach().cpu().numpy().astype(np.float32)
        for key, obs in zip(keys, batch):
            self.buffers[key].append(np.asarray(obs, dtype=np.float32).reshape(-1))
        flags = [bool(score > float(threshold)) for score in scores] if threshold is not None else [False for _ in scores]
        return scores, flags


def _subset_mask_from_attack(clean_inputs: np.ndarray, adv_inputs: np.ndarray, attack_mask: np.ndarray | None = None) -> np.ndarray:
    if attack_mask is not None:
        mask = np.asarray(attack_mask, dtype=np.int64).reshape(-1) > 0
        if bool(mask.any()):
            return mask
    clean_inputs = np.asarray(clean_inputs, dtype=np.float32).reshape(-1, 11)
    adv_inputs = np.asarray(adv_inputs, dtype=np.float32).reshape(-1, 11)
    return np.max(np.abs(clean_inputs - adv_inputs), axis=1) > 1e-8


@torch.no_grad()
def evaluate_dae_metrics(
    clean_inputs: np.ndarray,
    adv_inputs: np.ndarray,
    actor: nn.Module,
    defender: nn.Module,
    device: torch.device,
    *,
    episode_indices: np.ndarray | None = None,
    vehicle_ids: np.ndarray | None = None,
    attack_mask: np.ndarray | None = None,
    batch_size: int = 1024,
    state_scope: str | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    clean_inputs = np.asarray(clean_inputs, dtype=np.float32).reshape(-1, 11)
    adv_inputs = np.asarray(adv_inputs, dtype=np.float32).reshape(-1, 11)
    if state_scope is None:
        state_indices = tuple(int(v) for v in getattr(defender, 'local_indices', STATE_LOCAL_IDX))
        scope_name = 'model'
    else:
        scope_name = canonical_state_scope(state_scope)
        state_indices = defended_indices_for_scope(scope_name)
    subset_mask = _subset_mask_from_attack(clean_inputs, adv_inputs, attack_mask=attack_mask)
    recovered = dae_reconstruction_with_history(defender, adv_inputs, device, episode_indices=episode_indices, vehicle_ids=vehicle_ids, batch_size=batch_size)
    actor = actor.to(device).eval()
    clean_t = torch.as_tensor(clean_inputs, dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(adv_inputs, dtype=torch.float32, device=device)
    rec_t = torch.as_tensor(recovered, dtype=torch.float32, device=device)
    clean_act = actor(clean_t).detach().cpu().numpy().reshape(-1)
    adv_act = actor(adv_t).detach().cpu().numpy().reshape(-1)
    rec_act = actor(rec_t).detach().cpu().numpy().reshape(-1)
    attack_state_err = (adv_inputs[:, list(state_indices)] - clean_inputs[:, list(state_indices)])[subset_mask]
    rec_state_err = (recovered[:, list(state_indices)] - clean_inputs[:, list(state_indices)])[subset_mask]
    attack_action_err = (adv_act - clean_act)[subset_mask]
    rec_action_err = (rec_act - clean_act)[subset_mask]
    mae = lambda arr: 0.0 if np.asarray(arr).size == 0 else float(np.mean(np.abs(arr)))
    rmse = lambda arr: 0.0 if np.asarray(arr).size == 0 else float(np.sqrt(np.mean(np.asarray(arr) ** 2)))
    state_attack_mae = mae(attack_state_err)
    state_recovered_mae = mae(rec_state_err)
    action_attack_mse = 0.0 if np.asarray(attack_action_err).size == 0 else float(np.mean(np.asarray(attack_action_err) ** 2))
    action_recovered_mse = 0.0 if np.asarray(rec_action_err).size == 0 else float(np.mean(np.asarray(rec_action_err) ** 2))
    metrics = {
        'sample_count': int(clean_inputs.shape[0]),
        'attacked_sample_count': int(np.sum(subset_mask)),
        'state_scope': scope_name,
        'state_dims': ','.join(str(v) for v in state_indices),
        'state_attack_mae': float(state_attack_mae),
        'state_recovered_mae': float(state_recovered_mae),
        'state_attack_rmse': rmse(attack_state_err),
        'state_recovered_rmse': rmse(rec_state_err),
        'state_mae_reduction': float(state_attack_mae - state_recovered_mae),
        'state_mae_reduction_pct': 0.0 if state_attack_mae <= 1e-12 else float((state_attack_mae - state_recovered_mae) / state_attack_mae),
        'action_attack_mae': mae(attack_action_err),
        'action_recovered_mae': mae(rec_action_err),
        'action_attack_mse': float(action_attack_mse),
        'action_recovered_mse': float(action_recovered_mse),
        'action_mse_reduction': float(action_attack_mse - action_recovered_mse),
        'action_mse_reduction_pct': 0.0 if action_attack_mse <= 1e-12 else float((action_attack_mse - action_recovered_mse) / action_attack_mse),
    }
    attack_dim_mae = np.zeros((clean_inputs.shape[1],), dtype=np.float32)
    rec_dim_mae = np.zeros((clean_inputs.shape[1],), dtype=np.float32)
    if np.asarray(attack_state_err).size:
        attack_dim_mae[list(state_indices)] = np.mean(np.abs(attack_state_err), axis=0)
        rec_dim_mae[list(state_indices)] = np.mean(np.abs(rec_state_err), axis=0)
    per_dim = pd.DataFrame({
        'state_dim': list(range(clean_inputs.shape[1])),
        'attack_mae': attack_dim_mae,
        'recovered_mae': rec_dim_mae,
    })
    per_dim['mae_reduction'] = per_dim['attack_mae'] - per_dim['recovered_mae']
    per_dim['mae_reduction_pct'] = np.where(np.abs(per_dim['attack_mae']) <= 1e-12, 0.0, per_dim['mae_reduction'] / np.clip(per_dim['attack_mae'], 1e-12, None))
    return metrics, per_dim


@torch.no_grad()
def evaluate_detector_metrics(detector_model: DetectorGRUVAE, clean_inputs: np.ndarray, adv_inputs: np.ndarray, device: torch.device, *, episode_indices: np.ndarray | None, vehicle_ids: np.ndarray | None, threshold: float | None = None, grid_size: int = 31, attack_mask: np.ndarray | None = None) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    clean_inputs = np.asarray(clean_inputs, dtype=np.float32).reshape(-1, 11)
    adv_inputs = np.asarray(adv_inputs, dtype=np.float32).reshape(-1, 11)
    clean_scores = detector_anomaly_scores(detector_model, clean_inputs, device, episode_indices=episode_indices, vehicle_ids=vehicle_ids, seq_len=int(getattr(detector_model, 'seq_len', 1)))
    adv_scores_all = detector_anomaly_scores(detector_model, adv_inputs, device, episode_indices=episode_indices, vehicle_ids=vehicle_ids, seq_len=int(getattr(detector_model, 'seq_len', 1)))
    subset_mask = _subset_mask_from_attack(clean_inputs, adv_inputs, attack_mask=attack_mask)
    adv_scores = adv_scores_all[subset_mask]
    if threshold is None:
        threshold, history_rows, best_idx = select_canomaly_from_scores(clean_scores, adv_scores, grid_size=grid_size)
    else:
        threshold = float(threshold)
        _, history_rows, best_idx = select_canomaly_from_scores(clean_scores, adv_scores, grid_size=grid_size)
    clean_pred = clean_scores > float(threshold)
    adv_pred = adv_scores > float(threshold)
    tp = int(np.sum(adv_pred))
    fn = int(adv_pred.size - tp)
    fp = int(np.sum(clean_pred))
    tn = int(clean_pred.size - fp)
    precision = 0.0 if (tp + fp) == 0 else float(tp / float(tp + fp))
    recall = 0.0 if (tp + fn) == 0 else float(tp / float(tp + fn))
    f1 = 0.0 if (precision + recall) == 0 else float((2.0 * precision * recall) / (precision + recall))
    clean_accuracy = 0.0 if (tn + fp) == 0 else float(tn / float(tn + fp))
    false_negative_rate = 0.0 if (tp + fn) == 0 else float(fn / float(tp + fn))
    false_positive_rate = 0.0 if (fp + tn) == 0 else float(fp / float(fp + tn))
    metrics = {
        'clean_sample_count': int(clean_scores.size),
        'attacked_sample_count': int(adv_scores.size),
        'threshold': float(threshold),
        'clean_accuracy': float(clean_accuracy),
        'false_positive_rate': float(false_positive_rate),
        'attack_precision': float(precision),
        'attack_recall': float(recall),
        'attack_f1': float(f1),
        'false_negative_rate': float(false_negative_rate),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'clean_score_mean': 0.0 if clean_scores.size == 0 else float(np.mean(clean_scores)),
        'attack_score_mean': 0.0 if adv_scores.size == 0 else float(np.mean(adv_scores)),
    }
    history_df = pd.DataFrame(history_rows)
    if not history_df.empty:
        history_df['is_selected'] = False
        if 0 <= int(best_idx) < len(history_df):
            history_df.loc[int(best_idx), 'is_selected'] = True
    confusion_df = pd.DataFrame([
        {'subset': 'clean', 'actual': 0, 'predicted': 0, 'count': int(tn)},
        {'subset': 'clean', 'actual': 0, 'predicted': 1, 'count': int(fp)},
        {'subset': 'attack', 'actual': 1, 'predicted': 0, 'count': int(fn)},
        {'subset': 'attack', 'actual': 1, 'predicted': 1, 'count': int(tp)},
    ])
    return metrics, history_df, confusion_df


def save_dae(model: nn.Module, path: str | Path, *, metadata: Optional[dict[str, Any]] = None) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    meta = dict(getattr(model, 'metadata', {}) or {})
    if metadata:
        meta.update(metadata)
    payload = {
        'model_type': 'gru_vae_dae',
        'model_config': model.get_config(),
        'state_dict': model.state_dict(),
        'metadata': meta,
    }
    torch.save(payload, path)
    return path


def load_dae(path: str | Path, device: torch.device) -> nn.Module:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get('model_type') != 'gru_vae_dae':
        raise ValueError(f'Unsupported DAE artifact type in {path}: {payload.get("model_type")!r}')
    model = DenoisingAutoencoder(**dict(payload.get('model_config') or {}))
    model.load_state_dict(payload['state_dict'])
    model.metadata = dict(payload.get('metadata') or {})
    model.to(device).eval()
    return model


def save_detector(model: nn.Module, path: str | Path, *, threshold: float, metadata: Optional[dict[str, Any]] = None, history: Optional[dict[str, Any]] = None) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if isinstance(model, DetectorGRUVAE):
        model_type = 'gru_vae_detector'
    elif isinstance(model, PosteriorBenefitMLPDetector):
        model_type = 'posterior_mlp_detector'
    else:
        raise ValueError(f'Unsupported detector model type for save_detector: {type(model)!r}')
    payload = {
        'model_type': model_type,
        'model_config': model.get_config(),
        'threshold': float(threshold),
        'state_dict': model.state_dict(),
        'metadata': metadata or {},
        'history': history or {},
    }
    torch.save(payload, path)
    return path


def load_detector(path: str | Path, device: torch.device) -> DetectorArtifact:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model_type = payload.get('model_type')
    if model_type == 'gru_vae_detector':
        model = DetectorGRUVAE(**dict(payload.get('model_config') or {}))
    elif model_type == 'posterior_mlp_detector':
        model = PosteriorBenefitMLPDetector(**dict(payload.get('model_config') or {}))
    else:
        raise ValueError(f'Unsupported detector artifact type in {path}: {model_type!r}')
    model.load_state_dict(payload['state_dict'])
    model.to(device).eval()
    return DetectorArtifact(model=model, threshold=float(payload.get('threshold', 0.0)), metadata=dict(payload.get('metadata') or {}))


def save_dae_history(result: DAETrainResult, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    rows = []
    validator_by_epoch = {
        int(row.get('epoch')): {k: v for k, v in row.items() if k != 'epoch'}
        for row in (result.validator_rows or [])
        if row.get('epoch') is not None
    }
    n = len(result.loss_history)
    for i in range(n):
        row = {
            'epoch': i + 1,
            'loss': result.loss_history[i],
            'recon_loss': result.recon_loss_history[i],
            'kl_loss': result.kl_loss_history[i],
            'robust_loss': result.robust_loss_history[i],
            'clean_adv_action_mse': result.clean_adv_action_mse[i] if i < len(result.clean_adv_action_mse) else np.nan,
            'clean_recovered_action_mse': result.clean_recovered_action_mse[i] if i < len(result.clean_recovered_action_mse) else np.nan,
            'clean_adv_state_mse': result.clean_adv_state_mse[i] if i < len(result.clean_adv_state_mse) else np.nan,
            'clean_recovered_state_mse': result.clean_recovered_state_mse[i] if i < len(result.clean_recovered_state_mse) else np.nan,
            'is_best': bool(result.is_best_history[i]) if i < len(result.is_best_history) else False,
        }
        row.update(validator_by_epoch.get(i + 1, {}))
        rows.append(row)
    normalize_result_frame(pd.DataFrame(rows), rename_keys=False).to_csv(path, index=False, float_format='%.6f')
    return path


def save_detector_history(result: DetectorTrainResult, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    rows = []
    n = len(result.train_loss_history)
    for i in range(n):
        rows.append({
            'epoch': i + 1,
            'train_loss': result.train_loss_history[i],
            'val_loss': result.val_loss_history[i] if i < len(result.val_loss_history) else np.nan,
            'val_accuracy': result.val_accuracy_history[i] if i < len(result.val_accuracy_history) else np.nan,
            'val_precision': result.val_precision_history[i] if i < len(result.val_precision_history) else np.nan,
            'val_recall': result.val_recall_history[i] if i < len(result.val_recall_history) else np.nan,
            'val_f1': result.val_f1_history[i] if i < len(result.val_f1_history) else np.nan,
            'is_best': bool(result.is_best_history[i]) if i < len(result.is_best_history) else False,
        })
    normalize_result_frame(pd.DataFrame(rows), rename_keys=False).to_csv(path, index=False, float_format='%.6f')
    return path
