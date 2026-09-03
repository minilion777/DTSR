from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _common import PACKAGE_ROOT, load_manifest, load_scenario

# Some cached NPZ files were written with NumPy 2.x, whose pickled metadata
# references numpy._core.  The training environment may run NumPy 1.x, where the
# same modules live under numpy.core.  Register aliases before any allow_pickle
# loads so cached datasets remain readable without regenerating them.
if "numpy._core" not in sys.modules:
    import numpy.core as _np_core

    sys.modules.setdefault("numpy._core", _np_core)
    try:
        import numpy.core.multiarray as _np_multiarray

        sys.modules.setdefault("numpy._core.multiarray", _np_multiarray)
    except Exception:
        pass
    try:
        import numpy.core.numeric as _np_numeric

        sys.modules.setdefault("numpy._core.numeric", _np_numeric)
    except Exception:
        pass

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from evc.defense import dae_reconstruction_with_history
from evc.merged_core import ChargingEnv, TRAIN_PROFILE, to_numpy_1d
from evc.offline_dae_det_temporal_shield import LOCAL_SHIELD_INDICES
from evc.ug_bcr import BeliefCoreConfig, UGBCRConfig, UrgencyGateConfig

EP100_ACTOR_PATH = PACKAGE_ROOT / "models" / "multiday_ddpg_baseline_bundle" / "actor_multiday_best.pt"
EP100_BUNDLE_PATH = PACKAGE_ROOT / "models" / "multiday_ddpg_baseline_bundle" / "bundle_multiday_best.pt"
RUNTIME_PIPELINE_ORDER = "DAE/DET route -> UG-BCR belief+urgency gate -> Temporal Shield -> Actor"
ABLATION_ADDITION_ORDER = "Attack -> Denoise -> Denoise+DET -> +Shield -> +UG-BCR"
REPAIR_MODE = "full"


def set_all_seeds(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def union_observation_bounds(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Union of valid per-scenario environment bounds for one split.

    The old multiday trainer used the first train day only.  That can clip a
    clean state from another day before the adversarial L-infinity projection.
    A split-level union keeps the original physical bound logic while making it
    valid for heterogeneous multiday signals.
    """
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for _, row in load_manifest(split).iterrows():
        arrivals, signal_path, _ = load_scenario(row)
        env = ChargingEnv(signals_path=signal_path, reward_profile=TRAIN_PROFILE)
        max_duration = max(float(arrivals["Duration_of_stay"].max()), 12.0)
        low, high = env.observation_bounds(max_duration_of_stay=max_duration)
        lows.append(np.asarray(low, dtype=np.float32))
        highs.append(np.asarray(high, dtype=np.float32))
    if not lows:
        raise RuntimeError(f"No scenarios found for split={split!r}")
    low = np.min(np.stack(lows, axis=0), axis=0).astype(np.float32)
    high = np.max(np.stack(highs, axis=0), axis=0).astype(np.float32)
    if np.any(low > high):
        raise AssertionError("Invalid union observation bounds.")
    return low, high


def dae_validation_metrics(
    model: torch.nn.Module,
    *,
    clean_inputs: np.ndarray,
    adv_inputs: np.ndarray,
    actor: torch.nn.Module,
    device: torch.device,
    episode_indices: np.ndarray,
    vehicle_ids: np.ndarray,
    attack_mask: np.ndarray | None,
    batch_size: int = 2048,
    clean_penalty_weight: float = 0.25,
    repair_mode: str | None = None,
) -> dict[str, float]:
    repair_mode = str(repair_mode or REPAIR_MODE).strip().lower().replace("-", "_")
    if repair_mode != "full":
        raise ValueError(f"DAE-only validation is full-state only, got repair_mode={repair_mode!r}")

    clean = np.asarray(clean_inputs, dtype=np.float32).reshape(-1, 11)
    adv = np.asarray(adv_inputs, dtype=np.float32).reshape(-1, 11)
    if clean.shape != adv.shape:
        raise ValueError("clean_inputs and adv_inputs must align")
    mask = (
        np.max(np.abs(adv - clean), axis=1) > 1e-8
        if attack_mask is None
        else np.asarray(attack_mask, dtype=np.int64).reshape(-1) > 0
    )
    rec_adv_full = dae_reconstruction_with_history(
        model,
        adv,
        device,
        episode_indices=episode_indices,
        vehicle_ids=vehicle_ids,
        batch_size=batch_size,
    )
    rec_clean_full = dae_reconstruction_with_history(
        model,
        clean,
        device,
        episode_indices=episode_indices,
        vehicle_ids=vehicle_ids,
        batch_size=batch_size,
    )
    rec_adv = rec_adv_full
    rec_clean = rec_clean_full
    guarded_idx = list(range(11))

    actor = freeze_module(actor.to(device))
    with torch.no_grad():
        clean_action = actor(torch.as_tensor(clean, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
        adv_action = actor(torch.as_tensor(adv, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
        rec_adv_action = actor(torch.as_tensor(rec_adv, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)
        rec_clean_action = actor(torch.as_tensor(rec_clean, dtype=torch.float32, device=device)).detach().cpu().numpy().reshape(-1)

    attack_action_error = (adv_action - clean_action)[mask]
    recovered_action_error = (rec_adv_action - clean_action)[mask]
    attack_mse = 0.0 if attack_action_error.size == 0 else float(np.mean(attack_action_error ** 2))
    recovered_mse = 0.0 if recovered_action_error.size == 0 else float(np.mean(recovered_action_error ** 2))
    reduction = 0.0 if attack_mse <= 1e-12 else float((attack_mse - recovered_mse) / attack_mse)
    clean_identity_mse = float(np.mean((rec_clean_action - clean_action) ** 2)) if clean_action.size else 0.0
    clean_identity_ratio = clean_identity_mse / max(attack_mse, 1e-12)
    checkpoint_score = reduction - float(clean_penalty_weight) * clean_identity_ratio

    attack_state_mae = 0.0 if not bool(np.any(mask)) else float(np.mean(np.abs(adv[mask][:, guarded_idx] - clean[mask][:, guarded_idx])))
    recovered_state_mae = 0.0 if not bool(np.any(mask)) else float(np.mean(np.abs(rec_adv[mask][:, guarded_idx] - clean[mask][:, guarded_idx])))

    prefix = "full"
    result = {
        "repair_mode": repair_mode,
        "dae_checkpoint_score": float(checkpoint_score),
        "dae_action_attack_mse": float(attack_mse),
        "dae_action_recovered_mse": float(recovered_mse),
        "dae_action_mse_reduction_pct": float(reduction),
        "dae_clean_identity_action_mse": float(clean_identity_mse),
        "dae_clean_identity_ratio": float(clean_identity_ratio),
        "dae_state_attack_mae": float(attack_state_mae),
        "dae_state_recovered_mae": float(recovered_state_mae),
        "attacked_sample_count": int(np.sum(mask)),
    }
    result.update({
        f"{prefix}_checkpoint_score": float(checkpoint_score),
        f"{prefix}_action_attack_mse": float(attack_mse),
        f"{prefix}_action_recovered_mse": float(recovered_mse),
        f"{prefix}_action_mse_reduction_pct": float(reduction),
        f"{prefix}_clean_identity_action_mse": float(clean_identity_mse),
        f"{prefix}_clean_identity_ratio": float(clean_identity_ratio),
        f"{prefix}_state_attack_mae": float(attack_state_mae),
        f"{prefix}_state_recovered_mae": float(recovered_state_mae),
    })
    return result

def runtime_dae_validation_metrics(model: torch.nn.Module, **kwargs) -> dict[str, float]:
    return dae_validation_metrics(model, **kwargs, repair_mode=REPAIR_MODE)


def load_ug_bcr_config(path: Path) -> UGBCRConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 1)) < 2:
        raise ValueError(
            "Legacy UG-BCR configuration may access true remaining time. "
            "Recalibrate UG-BCR-v2 before evaluation."
        )
    if payload.get("leakage_policy") != "strict_no_clean_state":
        raise ValueError("UG-BCR-v2 configuration must declare strict_no_clean_state.")
    if bool(payload.get("uses_clean_state", True)) or bool(payload.get("uses_true_remaining_time", True)):
        raise ValueError("UG-BCR-v2 configuration cannot use clean state or true remaining time.")
    return UGBCRConfig(
        schema_version=int(payload["schema_version"]),
        leakage_policy=str(payload["leakage_policy"]),
        time_initialization=str(payload.get("time_initialization", "")),
        uses_clean_state=bool(payload["uses_clean_state"]),
        uses_true_remaining_time=bool(payload["uses_true_remaining_time"]),
        belief=BeliefCoreConfig(**dict(payload.get("belief") or {})),
        urgency_gate=UrgencyGateConfig(**dict(payload.get("urgency_gate") or {})),
    )


def ug_bcr_config_payload(config: UGBCRConfig) -> dict[str, Any]:
    return asdict(config)


def validation_price_median() -> tuple[float, int]:
    values: list[float] = []
    for _, row in load_manifest("val").iterrows():
        _, signal_path, _ = load_scenario(row)
        payload = json.loads(Path(signal_path).read_text(encoding="utf-8"))
        values.extend(float(v) for v in payload["price"])
    if not values:
        raise RuntimeError("Validation signal set contains no prices.")
    return float(np.median(np.asarray(values, dtype=np.float64))), int(len(values))


def safe_recovery(clean_reward: float, attack_reward: float, defended_reward: float) -> float:
    denominator = float(clean_reward) - float(attack_reward)
    if abs(denominator) <= 1e-8:
        return float("nan")
    return float((float(defended_reward) - float(attack_reward)) / denominator)


def to_scalar_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if not isinstance(value, list)}
